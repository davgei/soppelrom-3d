"""Register Polycam's own .ply export into OUR reconstruction's coordinate frame.

WHY THIS EXISTS
    Polycam's exports look better than our TSDF reconstruction, so the viewers should be able to
    draw them instead. But they are Z-up (the pipeline assumes Y-up/ARKit) and they do not sit in
    the same pose as our reconstruction, so they cannot simply be loaded — they have to be
    registered. In the annotation tool the user draws 3D boxes AGAINST the displayed cloud, so a
    misaligned backdrop silently shifts the ground truth every model trains on. That makes this a
    correctness problem there, not a cosmetic one, and it is why every transform carries a quality
    record and why callers can insist on a passing one (`require_ok`).

THE TWO FRAMES — READ THIS BEFORE USING THE API
    "raw"     — our reconstruction's own frame: CACHE_ROOT/<stem>/cloud.ply and mesh_poisson.ply as
                they sit on disk. annotate3d draws mesh_poisson.ply unrotated, and annotation boxes
                and entrances are stored in this frame.
    "gravity" — raw rotated by Scene.rotation (pipeline.compute_scene). place3d draws this: it shows
                Scene.mesh / Scene.aligned, both already rotated by that matrix.
    The CACHED transform maps Polycam raw -> OUR raw. `aligned_polycam_cloud(stem)` therefore
    returns the raw frame (right for annotate3d), and `gravity_rotation=scene.rotation` composes the
    extra rotation on top (right for place3d). Caching in the raw frame keeps the cache valid even
    if the gravity estimate is ever re-tuned.

HOW THE REGISTRATION WORKS (what actually turned out to matter)
    1. Both clouds are gravity-aligned first, which reduces the search to yaw + translation.
       The up axis of the Polycam cloud is MEASURED, not assumed (like view_ply), but measuring it
       from the single largest plane is not enough: in long narrow rooms the biggest plane is a
       WALL, and picking it left the room lying on its side (48052, 48054 failed exactly that way).
       So several plane normals are kept as up-axis CANDIDATES, both signs, filtered by "a room is
       wider than it is tall", and the score decides.
    2. Yaw + horizontal offset come from an FFT cross-correlation of the two floor-projected
       occupancy maps: exhaustive over translation, 5-degree steps in yaw. Centroid alignment is
       what failed on the long rooms (76858, 82434, 45470) — in a 22 m corridor the two clouds
       cover different stretches, so their centroids are metres apart and ICP cannot recover.
    3. Multi-scale POINT-TO-PLANE ICP, voxels 0.20 -> 0.10 -> 0.04 -> 0.025 m with the maximum
       correspondence distance shrinking alongside. Point-to-plane is what buys the accuracy on
       flat floors and walls; the first attempt's point-to-point ICP stalled at 19 cm RMSE.
    4. Candidates are pruned on the coarse levels and only the best few pay for the fine levels.
    Uniform scale was tested as a diagnostic (TransformationEstimationPointToPoint(True)) and came
    out at 0.991-1.001 on all nine scans: both clouds are metric, there is no scale error, and
    scale is deliberately NOT part of the shipped transform.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from .paths import CACHE_ROOT, PLY_DIR

reg = o3d.pipelines.registration

UP = np.array([0.0, 1.0, 0.0])

# Bump when the algorithm changes so stale cache entries are recomputed instead of trusted.
ALIGN_VERSION = 3

# Quality gate — thresholds measured on the 9-scan test set, reasoning in _gate().
GATE_MEDIAN_M = 0.040
GATE_SHARPNESS = 0.68
GATE_OVERLAP = 0.75

_EVAL_VOXEL = 0.02          # both clouds are thinned to 2 cm before scoring: metrics stay
_STRICT_M = 0.05            # comparable between a 1 M-point and a 2.6 M-point cloud
_COARSE_M = 0.30            # beyond this a Polycam point is treated as "our scan never saw it"

_ICP_STAGES = ((0.20, 0.60, 50), (0.10, 0.25, 40), (0.04, 0.10, 40), (0.025, 0.05, 30))


# --------------------------------------------------------------------------- small geometry helpers

def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / (np.linalg.norm(vector) + 1e-12)


def _rot_to_up(normal: np.ndarray) -> np.ndarray:
    """Rotation taking `normal` onto +Y."""
    normal = _unit(normal)
    axis = np.cross(normal, UP)
    sin_a = float(np.linalg.norm(axis))
    if sin_a < 1e-8:
        if normal @ UP > 0:
            return np.eye(3)
        return o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0.0, 0.0])
    angle = float(np.arccos(np.clip(normal @ UP, -1.0, 1.0)))
    return o3d.geometry.get_rotation_matrix_from_axis_angle((axis / sin_a) * angle)


def _yaw3(cos_a: float, sin_a: float) -> np.ndarray:
    """3x3 rotation about +Y built from a 2D rotation acting on (X, Z)."""
    return np.array([[cos_a, 0.0, -sin_a], [0.0, 1.0, 0.0], [sin_a, 0.0, cos_a]])


def _planes(cloud: o3d.geometry.PointCloud, k: int = 6, thresh: float = 0.05,
            min_frac: float = 0.02, seed: int = 42) -> list[tuple[float, np.ndarray]]:
    """The k largest planes, biggest first, as (inlier fraction, unit normal)."""
    o3d.utility.random.seed(seed)          # RANSAC is randomised; seed it so a cache entry is reproducible
    work = o3d.geometry.PointCloud(cloud)
    total = len(cloud.points)
    found: list[tuple[float, np.ndarray]] = []
    for _ in range(k):
        if len(work.points) < max(500, int(min_frac * total)):
            break
        try:
            model, inliers = work.segment_plane(thresh, 3, 300)
        except Exception:
            break
        if len(inliers) >= min_frac * total:
            found.append((len(inliers) / total, _unit(model[:3])))
        work = work.select_by_index(inliers, invert=True)
    return found


def _floor_level(points: np.ndarray) -> float:
    """Height of the floor in a gravity-aligned cloud: the densest 5 cm height bin in the lower
    third. Taking the minimum Y instead would latch onto stray points below the floor."""
    y = points[:, 1]
    low, high = np.percentile(y, [0.5, 99.5])
    if high - low < 0.2:
        return float(low)
    edges = np.arange(low, high + 0.05, 0.05)
    hist, edges = np.histogram(y, bins=edges)
    in_band = edges[:-1] < low + 0.35 * (high - low)
    if not in_band.any():
        return float(low)
    index = int(np.argmax(np.where(in_band, hist, -1)))
    return float(edges[index] + 0.025)


def _wall_slice(points: np.ndarray, floor: float) -> np.ndarray:
    """XZ projection of the 0.25-2.2 m band above the floor — the room outline. Vertical structure
    is far more distinctive for correlation than the floor itself (every room has a flat floor)."""
    band = points[(points[:, 1] > floor + 0.25) & (points[:, 1] < floor + 2.2)]
    return (band if len(band) >= 500 else points)[:, [0, 2]]


def _up_candidates(cloud: o3d.geometry.PointCloud) -> list[np.ndarray]:
    """Plausible gravity-align rotations for a cloud of unknown convention.

    Several plane normals are offered, not just the largest: the largest plane is a wall in long
    narrow rooms. Both signs of each are kept because the floor normal may point either way. The
    only candidates thrown away are the absurd ones — a waste-room is never much taller than it is
    wide. That slack is deliberate: a strict "wider than tall" test discarded the CORRECT axis of
    46582, a 4.4 m tall room only 3.7 m across, and left it registered against a wall."""
    points = np.asarray(cloud.points)
    axes: list[np.ndarray] = []
    for _, normal in _planes(cloud):
        if not any(abs(normal @ axis) > 0.94 for axis in axes):
            axes.append(normal)
    shortest = np.zeros(3)
    shortest[int(np.argmin(np.ptp(points, axis=0)))] = 1.0     # rooms are wider than they are tall
    if not any(abs(shortest @ axis) > 0.94 for axis in axes):
        axes.append(shortest)

    keep: list[np.ndarray] = []
    for axis in axes:
        for sign in (1.0, -1.0):
            rotation = _rot_to_up(sign * axis)
            low, high = np.percentile(points @ rotation.T, [1.0, 99.0], axis=0)
            extent = high - low
            if extent[1] > 1.6 * min(extent[0], extent[2]) or extent[1] > 9.0:
                continue
            keep.append(rotation)
    return keep or [np.eye(3)]


# --------------------------------------------------------------------------- global 2D search

def _occupancy(xz: np.ndarray, origin: np.ndarray, size: int, cell: float) -> np.ndarray:
    cells = np.floor((xz - origin) / cell).astype(np.int64)
    inside = ((cells[:, 0] >= 0) & (cells[:, 0] < size) & (cells[:, 1] >= 0) & (cells[:, 1] < size))
    grid = np.zeros((size, size), dtype=np.float32)
    cells = cells[inside]
    np.add.at(grid, (cells[:, 1], cells[:, 0]), 1.0)
    return (grid >= 2.0).astype(np.float32)     # >=2 drops isolated flyers, keeps real surfaces


def _fft_yaw_search(source_xz: np.ndarray, target_xz: np.ndarray, cell: float = 0.12,
                    yaw_step: float = 5.0, top: int = 6) -> list[tuple[float, float, np.ndarray]]:
    """Best (score, yaw_deg, [dx, dz]) pairs from correlating the two occupancy maps.

    Exhaustive in translation for every yaw, which is the point: aligning centroids assumes the two
    clouds cover the same stretch of room, and in a 20 m corridor they do not."""
    span = max(float(np.ptp(source_xz, axis=0).max()), float(np.ptp(target_xz, axis=0).max()))
    size = int(np.clip(2 ** np.ceil(np.log2(max(2.2 * span / cell, 64.0))), 64, 1024))

    target_origin = target_xz.mean(axis=0) - (size * cell) / 2.0
    target_fft = np.fft.rfft2(_occupancy(target_xz, target_origin, size, cell))
    source_origin = np.full(2, -(size * cell) / 2.0)
    pivot = source_xz.mean(axis=0)

    hits: list[tuple[float, float, np.ndarray]] = []
    for degrees in np.arange(0.0, 360.0, yaw_step):
        angle = np.radians(degrees)
        rot2 = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        grid = _occupancy((source_xz - pivot) @ rot2.T, source_origin, size, cell)
        occupied = float(grid.sum())
        if occupied < 10.0:
            continue
        # irfft2(F_t * conj(F_s))[k] = sum_x s[x] t[x + k]: a source cell x lands on target cell x+k
        corr = np.fft.irfft2(target_fft * np.conj(np.fft.rfft2(grid)), (size, size))
        flat = int(np.argmax(corr))
        row, col = divmod(flat, size)
        score = float(corr[row, col]) / occupied          # fraction of source cells that found a match
        d_col = col - size if col > size // 2 else col     # the correlation wraps around
        d_row = row - size if row > size // 2 else row
        offset = np.array([d_col, d_row], dtype=float) * cell + (target_origin - source_origin)
        hits.append((score, float(degrees), offset))
    hits.sort(key=lambda hit: -hit[0])
    return hits[:top]


# --------------------------------------------------------------------------- ICP

def _prepped(cloud: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    down = cloud.voxel_down_sample(voxel)
    # point-to-plane needs target normals; estimating them on both sides costs nothing extra
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.0, max_nn=30))
    return down


def _icp(pairs: dict[float, tuple], transform: np.ndarray, stages) -> np.ndarray:
    for voxel, max_corr, iterations in stages:
        source, target = pairs[voxel]
        transform = reg.registration_icp(
            source, target, max_corr, transform,
            reg.TransformationEstimationPointToPlane(),
            reg.ICPConvergenceCriteria(max_iteration=iterations),
        ).transformation
    return np.asarray(transform, dtype=float)


# --------------------------------------------------------------------------- quality record

@dataclass
class AlignQuality:
    """How much the transform can be trusted.

    fitness    fraction of Polycam points with one of ours within 5 cm
    rmse       RMS distance over exactly those points. KEPT FOR REFERENCE, NOT GATED ON: because it
               averages only the points that were already within 5 cm it barely moves even for a
               completely wrong pose (measured 1.8-2.8 cm for a deliberate 180-degree flip). Anyone
               reading "RMSE 1.7 cm" as proof of alignment is being misled, which is why the gate
               uses the three below instead.
    overlap    fraction within 30 cm — how much of the Polycam cloud our scan covers AT ALL
    sharpness  fitness / overlap: of the co-visible part, how much agrees to 5 cm. The real
               alignment measure, because fitness alone punishes a scan for coverage differences
               (our TSDF is truncated at 5 m depth, Polycam's export is not).
    residual_median / residual_p90
               nearest-neighbour distance over the co-visible part, in metres — the honest accuracy
               numbers. Measured 1.2-1.9 cm median on all nine test scans. The p90 tail (5-17 cm)
               sits almost entirely on non-planar clutter where the two reconstructions genuinely
               disagree: restricted to the big floor and wall planes the median is 1.6-1.8 cm and
               p90 3-6 cm, so the POSE is good and the tail is a surface-detail difference.
    ok         passes the gate -> safe to annotate against
    """
    fitness: float
    rmse: float
    overlap: float
    sharpness: float
    residual_median: float
    residual_p90: float
    ok: bool
    reason: str
    n_points: int
    seconds: float
    version: int = ALIGN_VERSION

    @property
    def summary(self) -> str:
        state = "godkjent" if self.ok else "AVVIST"
        return (f"{state}: median avvik {self.residual_median * 100:.1f} cm, "
                f"overlapp {self.overlap:.2f}, skarphet {self.sharpness:.2f}")


def _gate(quality: dict) -> tuple[bool, str]:
    """Three conditions, each catching a different way the registration can be untrustworthy.

    Calibrated against both the nine correct results and deliberately corrupted poses (the true
    transform perturbed by a shift, a small yaw, and a 180-degree flip).

    median <= 4 cm    — a loose sanity floor, NOT the discriminator. All nine correct scans land at
                        1.2-1.9 cm, so this only trips if something is badly wrong. It is loose on
                        purpose: in a long corridor of parallel walls even a flipped pose keeps most
                        points near SOME wall (80623 flipped still measured 4.6 cm), so no threshold
                        on a distance average can carry the decision by itself.
    sharpness >= 0.68 — of the part both clouds see, at least two thirds agrees to 5 cm. This is the
                        primary discriminator. Correct poses score 0.72-0.90; the ceiling is set by
                        Polycam being ~3x sparser and by real reconstruction differences, so 1.0 is
                        not attainable. Grossly wrong poses score 0.21-0.66, and 1 degree of
                        injected yaw error already drops a good scan to 0.66 — the boundary sits
                        just inside "1 degree of yaw is wrong", which is deliberately strict given
                        that annotation boxes are drawn against this cloud.
    overlap >= 0.75   — sharpness is meaningless when the clouds barely touch: a small patch locked
                        onto one wall can look sharp while the rest floats. Correct poses reach
                        0.93-1.00; wrong poses 0.24-0.59.
    The two are complementary, and both are needed: a 180-degree flip of 80623 is caught by overlap
    (0.38) while its sharpness stays 0.54, and a flip of 46582 is caught by sharpness (0.48) while
    its overlap stays 0.81.
    """
    if quality["residual_median"] > GATE_MEDIAN_M:
        return False, (f"median avvik {quality['residual_median'] * 100:.1f} cm "
                       f"> {GATE_MEDIAN_M * 100:.0f} cm")
    if quality["overlap"] < GATE_OVERLAP:
        return False, f"for lite overlapp ({quality['overlap']:.2f} < {GATE_OVERLAP:.2f})"
    if quality["sharpness"] < GATE_SHARPNESS:
        return False, f"for lav skarphet ({quality['sharpness']:.2f} < {GATE_SHARPNESS:.2f})"
    return True, "innenfor kravene"


def _evaluate(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
              transform: np.ndarray) -> dict:
    strict = reg.evaluate_registration(source, target, _STRICT_M, transform)
    coarse = reg.evaluate_registration(source, target, _COARSE_M, transform)
    fitness, overlap = float(strict.fitness), float(coarse.fitness)

    moved = o3d.geometry.PointCloud(source).transform(transform)
    distance = np.asarray(moved.compute_point_cloud_distance(target))
    covisible = distance[distance < _COARSE_M]
    if len(covisible) > 100:
        median, p90 = (float(v) for v in np.percentile(covisible, [50, 90]))
    else:
        median, p90 = float("inf"), float("inf")
    return {
        "fitness": fitness,
        "rmse": float(strict.inlier_rmse),
        "overlap": overlap,
        "sharpness": fitness / max(overlap, 1e-9),
        "residual_median": median,
        "residual_p90": p90,
    }


# --------------------------------------------------------------------------- paths and cache

def find_ply(stem: str) -> Path | None:
    """The Polycam export for a scan, by exact stem or an unambiguous prefix (a scan number)."""
    exact = PLY_DIR / f"{stem}.ply"
    if exact.exists():
        return exact
    hits = [path for path in sorted(PLY_DIR.glob("*.ply")) if path.stem.startswith(stem)]
    return hits[0] if len(hits) == 1 else None


def our_cloud_path(stem: str) -> Path:
    return CACHE_ROOT / stem / "cloud.ply"


def cache_path(stem: str) -> Path:
    """Where the transform is stored. Next to the rest of the scan's derived data, so it is thrown
    away together with the reconstruction it was registered against."""
    return CACHE_ROOT / stem / "polycam_align.json"


def _read_cache(stem: str) -> tuple[np.ndarray, AlignQuality] | None:
    path = cache_path(stem)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("version", 0)) != ALIGN_VERSION:
            return None                                    # algorithm changed -> recompute
        transform = np.asarray(data["transform"], dtype=float).reshape(4, 4)
        quality = AlignQuality(**data["quality"])
    except Exception:
        return None                                        # corrupt cache is not worth crashing over
    return transform, quality


def _write_cache(stem: str, transform: np.ndarray, quality: AlignQuality) -> None:
    path = cache_path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": ALIGN_VERSION,
        "scan": stem,
        "frame": "polycam_raw -> our_raw (cloud.ply / mesh_poisson.ply as stored)",
        "transform": np.asarray(transform, dtype=float).tolist(),
        "quality": asdict(quality),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- the registration

def compute_transform(stem: str, verbose: bool = False) -> tuple[np.ndarray, AlignQuality] | None:
    """Register the Polycam export onto our reconstruction. Returns (4x4 into OUR RAW frame,
    quality) or None when either cloud is missing. Takes tens of seconds — use align_transform()
    so the answer is cached."""
    ply = find_ply(stem)
    ours_path = our_cloud_path(stem)
    if ply is None or not ours_path.exists():
        return None
    started = time.time()

    polycam = o3d.io.read_point_cloud(str(ply))
    ours = o3d.io.read_point_cloud(str(ours_path))
    if not len(polycam.points) or not len(ours.points):
        return None

    polycam_lo = polycam.voxel_down_sample(0.06)
    ours_lo = ours.voxel_down_sample(0.06)

    # our own gravity frame is scaffolding only: the returned transform goes back to the raw frame,
    # so it does not matter whether this matches backbone's estimate exactly
    horizontal = [normal for _, normal in _planes(ours_lo) if abs(normal @ UP) > 0.85]
    to_gravity = _rot_to_up(horizontal[0] if horizontal else UP)
    target_g = np.asarray(ours_lo.points) @ to_gravity.T
    target_floor = _floor_level(target_g)
    target_xz = _wall_slice(target_g, target_floor)

    pairs = {voxel: (_prepped(polycam, voxel), _prepped(ours, voxel)) for voxel, _, _ in _ICP_STAGES}
    source_eval = polycam.voxel_down_sample(_EVAL_VOXEL)
    target_eval = ours.voxel_down_sample(_EVAL_VOXEL)

    candidates: list[tuple[float, np.ndarray]] = []
    for up_rotation in _up_candidates(polycam_lo):
        source_g = np.asarray(polycam_lo.points) @ up_rotation.T
        source_floor = _floor_level(source_g)
        source_xz = _wall_slice(source_g, source_floor)
        pivot = np.array([source_xz[:, 0].mean(), 0.0, source_xz[:, 1].mean()])
        for score, degrees, offset in _fft_yaw_search(source_xz, target_xz):
            angle = np.radians(degrees)
            yaw = _yaw3(float(np.cos(angle)), float(np.sin(angle)))
            rotation = yaw @ up_rotation
            translation = -yaw @ pivot + np.array([offset[0], target_floor - source_floor, offset[1]])
            transform = np.eye(4)
            transform[:3, :3] = to_gravity.T @ rotation      # ... and back out of our gravity frame
            transform[:3, 3] = to_gravity.T @ translation
            candidates.append((score, transform))
    candidates.sort(key=lambda item: -item[0])
    candidates = candidates[:10]

    # Prune on the two coarse levels — a wrong pose is already obvious there — then let only the
    # best few pay for the expensive fine levels.
    coarse, fine = _ICP_STAGES[:2], _ICP_STAGES[2:]
    ranked = []
    for _, transform in candidates:
        rough = _icp(pairs, transform, coarse)
        ranked.append((float(reg.evaluate_registration(source_eval, target_eval, 0.10, rough).fitness),
                       rough))
    ranked.sort(key=lambda item: -item[0])

    best: tuple[float, np.ndarray] | None = None
    for _, rough in ranked[:4]:
        refined = _icp(pairs, rough, fine)
        fitness = float(reg.evaluate_registration(source_eval, target_eval, _STRICT_M,
                                                  refined).fitness)
        if best is None or fitness > best[0]:
            best = (fitness, refined)
    assert best is not None
    transform = best[1]

    scores = _evaluate(source_eval, target_eval, transform)
    ok, reason = _gate(scores)
    quality = AlignQuality(
        fitness=round(scores["fitness"], 4),
        rmse=round(scores["rmse"], 5),
        overlap=round(scores["overlap"], 4),
        sharpness=round(scores["sharpness"], 4),
        residual_median=round(scores["residual_median"], 5),
        residual_p90=round(scores["residual_p90"], 5),
        ok=ok,
        reason=reason,
        n_points=len(polycam.points),
        seconds=round(time.time() - started, 1),
    )
    if verbose:
        print(f"{stem}: {quality.summary}  ({quality.seconds:.0f} s)")
    return transform, quality


def cached_transform(stem: str) -> tuple[np.ndarray, AlignQuality] | None:
    """The registration ONLY if it is already cached and current — never computes one.

    The interactive viewers use this to decide what to draw: computing a transform takes 8-45 s,
    and doing that while a scan opens would freeze the GUI (and make arrow-key scan switching
    unusable). None here means "not registered yet", which is a different situation from
    "registered and rejected" and the viewers say so differently."""
    return _read_cache(stem)


def align_transform(stem: str, force: bool = False,
                    verbose: bool = False) -> tuple[np.ndarray, AlignQuality] | None:
    """Cached version of compute_transform. `force=True` recomputes and overwrites the cache."""
    if not force:
        cached = _read_cache(stem)
        if cached is not None:
            return cached
    result = compute_transform(stem, verbose=verbose)
    if result is None:
        return None
    _write_cache(stem, *result)
    return result


# --------------------------------------------------------------------------- viewer API

def aligned_polycam_cloud(stem: str, gravity_rotation: np.ndarray | None = None,
                          require_ok: bool = False, force: bool = False,
                          ) -> tuple[o3d.geometry.PointCloud, AlignQuality] | None:
    """The Polycam cloud placed in the frame the viewers draw in, plus its quality record.

    gravity_rotation
        None            -> OUR RAW frame: mesh_poisson.ply / cloud.ply as stored on disk, which is
                           what annotate3d draws and what annotation boxes live in.
        Scene.rotation  -> the gravity-aligned frame that place3d draws (it shows Scene.mesh and
                           Scene.aligned, both already rotated by that matrix).
    require_ok
        True -> return None unless the transform passes the quality gate. The annotation tool should
        set this: drawing boxes against a cloud that is 20 cm off silently corrupts the labels.

    Returns None when there is no Polycam export for the scan, no reconstruction to register
    against, or (with require_ok) no trustworthy transform.
    """
    result = align_transform(stem, force=force)
    if result is None:
        return None
    transform, quality = result
    if require_ok and not quality.ok:
        return None
    path = find_ply(stem)
    if path is None:
        return None
    cloud = o3d.io.read_point_cloud(str(path))
    cloud.transform(transform)
    if gravity_rotation is not None:
        cloud.rotate(np.asarray(gravity_rotation, dtype=float), center=(0.0, 0.0, 0.0))
    return cloud, quality


def has_polycam(stem: str) -> bool:
    return find_ply(stem) is not None


def available_stems() -> list[str]:
    """Scans that have both a Polycam export and a reconstruction to register it against."""
    return [path.stem for path in sorted(PLY_DIR.glob("*.ply")) if our_cloud_path(path.stem).exists()]


# --------------------------------------------------------------------------- CLI

def _overlay_snapshot(stem: str, out_path: Path, width: int = 1700, height: int = 1100) -> bool:
    """Our cloud in blue, the aligned Polycam cloud in orange, seen straight down with the ceiling
    cropped away. Numbers do not prove alignment — and neither does a plain 3D overlay, where the
    ceiling hides the whole room and whichever cloud is nearer hides the other. Cropping to the
    0.15-1.8 m band above the floor puts both clouds' walls and bins side by side in one image:
    where they coincide the colours interleave, where they do not you see two separate outlines."""
    result = aligned_polycam_cloud(stem)
    if result is None:
        print(f"{stem}: ingen .ply eller ingen rekonstruksjon")
        return False
    polycam, quality = result
    ours = o3d.io.read_point_cloud(str(our_cloud_path(stem)))

    # a top-down view is only meaningful in a gravity-aligned frame
    horizontal = [normal for _, normal in _planes(ours.voxel_down_sample(0.06))
                  if abs(normal @ UP) > 0.85]
    to_gravity = _rot_to_up(horizontal[0] if horizontal else UP)
    ours.rotate(to_gravity, center=(0.0, 0.0, 0.0))
    polycam.rotate(to_gravity, center=(0.0, 0.0, 0.0))
    floor = _floor_level(np.asarray(ours.voxel_down_sample(0.06).points))

    def band(cloud: o3d.geometry.PointCloud, colour: list[float]) -> o3d.geometry.PointCloud:
        points = np.asarray(cloud.points)
        keep = (points[:, 1] > floor + 0.15) & (points[:, 1] < floor + 1.8)
        cut = cloud.select_by_index(np.flatnonzero(keep)).voxel_down_sample(0.02)
        cut.paint_uniform_color(colour)
        return cut

    ours_band = band(ours, [0.30, 0.55, 0.98])
    poly_band = band(polycam, [1.00, 0.55, 0.08])
    if not len(ours_band.points) or not len(poly_band.points):
        print(f"{stem}: for få punkter i høydebåndet til å lage overlay")
        return False

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(visible=False, width=width, height=height)
    viewer.add_geometry(ours_band)
    viewer.add_geometry(poly_band)
    option = viewer.get_render_option()
    option.point_size = 2.0
    option.background_color = np.array([0.05, 0.05, 0.07])
    control = viewer.get_view_control()
    control.set_front([0.0, 1.0, 0.0])          # straight down
    control.set_up([0.0, 0.0, -1.0])
    control.set_zoom(0.62)
    for _ in range(8):
        viewer.poll_events()
        viewer.update_renderer()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    viewer.capture_screen_image(str(out_path), do_render=True)
    viewer.destroy_window()
    print(f"{stem}: {quality.summary} -> {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Registrer Polycams .ply mot vaar rekonstruksjon.")
    parser.add_argument("--scan", action="append", default=None, help="skann-stamme (kan gjentas)")
    parser.add_argument("--all", action="store_true", help="alle skann som har bade .ply og cache")
    parser.add_argument("--force", action="store_true", help="regn ut paa nytt selv om cache finnes")
    parser.add_argument("--overlay-dir", default=None, help="skriv overlay-PNG hit")
    args = parser.parse_args()

    stems = args.scan or (available_stems() if args.all else [])
    if not stems:
        raise SystemExit("bruk --scan <stamme> eller --all")

    rows = []
    for stem in stems:
        result = align_transform(stem, force=args.force)
        if result is None:
            print(f"{stem:32} ingen .ply / ingen rekonstruksjon")
            continue
        _, quality = result
        rows.append((stem, quality))
        flag = "OK " if quality.ok else "AVV"
        print(f"{flag} {stem:32} treff {quality.fitness:.3f}  median {quality.residual_median * 100:5.2f} cm  "
              f"p90 {quality.residual_p90 * 100:5.2f} cm  overlapp {quality.overlap:.3f}  "
              f"skarphet {quality.sharpness:.3f}  {quality.seconds:5.1f} s  {quality.reason}")
        if args.overlay_dir:
            _overlay_snapshot(stem, Path(args.overlay_dir) / f"overlay_{stem}.png")
    if rows:
        good = sum(1 for _, q in rows if q.ok)
        print(f"\n{good}/{len(rows)} skann innenfor kravene "
              f"(median <= {GATE_MEDIAN_M * 100:.0f} cm, overlapp >= {GATE_OVERLAP:.2f}, "
              f"skarphet >= {GATE_SHARPNESS:.2f})")


if __name__ == "__main__":
    main()
