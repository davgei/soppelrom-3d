"""Learned door/entrance finder, so entrances are located automatically (no manual clicking).

Doors are *openings* — an absence of wall where the floor leaks out and people walk through — so a
point-cloud net has nothing to look at. Instead we work at the grid level: sample candidate points
all along the room's floor perimeter, describe each with a few cheap features (how much wall is
right here, whether open space leaks outward, whether the scanner walked through, camera traffic),
and let a small classifier trained on your clicked entrances decide which perimeter stretches are
real doors. Proposing along the whole perimeter (not only pre-detected gaps) is what lets the model
actually cover every door — "little wall here" becomes a feature, not a hard filter.

find_doors() uses the trained model when models/doors_latest.pt exists, and otherwise falls back to
the hand-written heuristic (placement.detect_entrances) so the pipeline always works.

Two guarantees matter more than the score, because an entrance is what a push-path is routed to and
a room with NO entrance yields zero placement suggestions — a lost door costs the whole analysis:

  1. find_doors NEVER returns an empty list while any candidate exists. If nothing clears
     KEEP_PROB we return the single highest-scoring candidate anyway and mark the result as a
     guess (LAST_DETECTION["confident"] is False, and we log a Norwegian warning).
  2. Every returned point sits INSIDE THE ROOM FOOTPRINT (footprint.mask), and candidates far from
     it are dropped before the winner is picked. Read that guarantee narrowly: the footprint mask is
     NOT the same as "floor the scanner actually saw". backbone._footprint closes the grid with a
     ~0.65 m kernel and then fills holes, so the mask deliberately covers floor that is occluded or
     unscanned. Measured over the 139 scans, a median 13% of mask cells have no scanned geometry near
     the floor plane, and of the 139 delivered entrances only 56 stand on a cell with real floor
     evidence while 55 stand on a cell that was never scanned (before this change: 29 and 75). So
     this reduces "floating" entrances but does not eliminate them, and a true test would need the
     point cloud, which find_doors does not receive.

The classifier now decides alone. An earlier hard gate ("only candidates the scanner physically
walked over can be doors", walked_frac > 0.02) was measured to COST ~9 points of scene-level
accuracy on the 85 scans with a clicked entrance: the top-1 automatic entrance landed within 2 m of a
clicked door in 24% of scenes with the gate and 33% without, held out (mean of five scene-level
4-fold splits), median error 5.5 m -> 4.3 m; on the training set 27% -> 38%. The scanner very often
films a doorway from a few metres away instead of walking through it, so the gate threw away the
correct candidate. walked_frac remains a FEATURE, so the model can still weigh it — softly, and
only where it helps.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label

log = logging.getLogger(__name__)

WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "doors_latest.pt"

# Feature order — keep in sync with train_doors. All are cheap local grid/geometry signals.
FEATURE_NAMES = ("wall_frac", "outside_open_frac", "walked_frac", "camera_count", "camera_dist_m")
CANDIDATE_SPACING_M = 0.30   # sample a candidate roughly every this far along the boundary
WINDOW_M = 0.45              # local neighbourhood used to describe a candidate
KEEP_PROB = 0.40             # find_doors keeps candidates scoring at least this (favours recall)
MERGE_RADIUS_M = 0.8         # kept candidates closer than this are one door
# One GUESS only. A room can genuinely have two entrances, but the model is right about the top one
# in 42% of scenes (within 3 m, held out) — so a second guess is more often a wrong door than a real
# one, and a wrong door drags the push-path through the wrong wall and produces placements that look
# plausible but are not. Measured cost of dropping to 1 on eight auto-detected scans: +1 placement
# candidate (+2.3%), i.e. none. This caps AUTO-DETECTION only: clicked entrances bypass find_doors
# entirely (pipeline.compute_scene), so you can still mark as many real doors as a room has.
MAX_DOORS = 1

# "on the floor, and not floating" — an entrance is only useful if a bin can actually be rolled
# from it onto the room floor, so a candidate must have real, observed floor right next to it.
FLOOR_SUPPORT_RADIUS_M = 0.50  # neighbourhood the candidate must find floor in
FLOOR_SUPPORT_FRAC = 0.20      # at least this share of that neighbourhood must be observed floor
FLOOR_REACH_M = 0.35           # and the nearest observed floor cell must be at most this far away
FLOOR_MAX_AIRBORNE_M = 0.60    # a cell whose LOWEST geometry is higher than this is a wall top /
                               # ledge / ceiling hit, not somewhere you can stand

_cache: dict = {}

# What the last find_doors call actually did, so a caller/GUI can tell a confident detection from a
# last-resort guess without changing find_doors' return type (list[(x, z)], as before).
LAST_DETECTION: dict = {
    "points": [],            # [(x, z), ...] as returned
    "probs": [],             # model probability per returned point (NaN = heuristic fallback)
    "confident": False,      # True only if at least one candidate cleared keep_prob
    "source": "none",        # "model" | "model-guess" | "heuristic"
    "n_candidates": 0,
    "n_floor_rejected": 0,   # candidates dropped as void/floating
}


def reset_last_detection() -> None:
    """Clear LAST_DETECTION.

    MUST be called by any caller that decides an entrance WITHOUT going through find_doors — in
    pipeline.compute_scene that is the clicked-entrance path and the is_enclosed path. Verified:
    processing an auto-detected scan and then a clicked one leaves the clicked room reporting the
    previous room's source/confidence, so a loop over scans would attribute one room's confidence to
    another. LAST_DETECTION is module state and this function cannot fix that on its own.
    (No consumer reads LAST_DETECTION yet, so this is latent rather than live.)"""
    LAST_DETECTION.update(points=[], probs=[], confident=False, source="none",
                          n_candidates=0, n_floor_rejected=0)


def _to_cells(points_xz: np.ndarray, origin: np.ndarray, cell: float, shape: tuple[int, int]):
    cols = np.floor((points_xz[:, 0] - origin[0]) / cell).astype(int)
    rows = np.floor((points_xz[:, 1] - origin[1]) / cell).astype(int)
    inside = (cols >= 0) & (cols < shape[1]) & (rows >= 0) & (rows < shape[0])
    return rows[inside], cols[inside]


def candidate_openings(fs, footprint, wall_mask: np.ndarray | None, camera_xz) -> list[dict]:
    """Candidate door locations sampled along the floor perimeter, each with features.

    Returns a list of {'center_xz': (x, z), 'features': np.ndarray}. Candidates are spaced along
    the whole boundary so every real door has one nearby; the classifier separates doors from
    plain walls using the features."""
    cell, origin = fs.cell, fs.origin
    rows, cols = fs.free.shape
    floor_region = footprint.mask.astype(bool)
    wall = wall_mask if wall_mask is not None else np.zeros((rows, cols), dtype=bool)

    wall_near = binary_dilation(wall, iterations=max(1, int(0.25 / cell)))
    # cells just outside the room that are NOT blocked by wall = open space a door leads into
    outside = binary_dilation(floor_region, iterations=max(1, int(0.5 / cell))) & ~floor_region
    outside_open = outside & ~wall_near

    camera_xz = np.asarray(camera_xz) if camera_xz is not None else np.empty((0, 2))
    walked = np.zeros((rows, cols), dtype=bool)
    if len(camera_xz):
        r_idx, c_idx = _to_cells(camera_xz, origin, cell, (rows, cols))
        walked[r_idx, c_idx] = True
    walked_d = binary_dilation(walked, iterations=max(1, int(0.6 / cell))) if walked.any() else walked

    # seed candidates along the floor perimeter AND on the open cells just outside it, so doors
    # sit near a candidate even when the clicked point is a little outside the main floor region
    perimeter = floor_region & ~binary_erosion(floor_region, iterations=1)
    seeds = perimeter | outside_open
    per_cells = np.argwhere(seeds)
    if not len(per_cells):
        return []

    # greedy spacing so we get roughly one candidate per CANDIDATE_SPACING_M of boundary
    sep = max(1, int(CANDIDATE_SPACING_M / cell))
    taken = np.zeros((rows, cols), dtype=bool)
    picked: list[tuple[int, int]] = []
    for r, c in per_cells[np.lexsort((per_cells[:, 1], per_cells[:, 0]))]:
        if taken[r, c]:
            continue
        picked.append((int(r), int(c)))
        taken[max(0, r - sep):r + sep + 1, max(0, c - sep):c + sep + 1] = True

    win = max(1, int(WINDOW_M / cell))
    candidates: list[dict] = []
    for r, c in picked:
        r0, r1 = max(0, r - win), min(rows, r + win + 1)
        c0, c1 = max(0, c - win), min(cols, c + win + 1)
        wall_frac = float(wall[r0:r1, c0:c1].mean())
        outside_open_frac = float(outside_open[r0:r1, c0:c1].mean())
        walked_frac = float(walked_d[r0:r1, c0:c1].mean()) if walked_d.any() else 0.0
        cx = float(origin[0] + (c + 0.5) * cell)
        cz = float(origin[1] + (r + 0.5) * cell)
        if len(camera_xz):
            dist = np.hypot(camera_xz[:, 0] - cx, camera_xz[:, 1] - cz)
            camera_dist = float(dist.min())
            camera_count = float((dist < 1.0).sum())
        else:
            camera_dist, camera_count = 5.0, 0.0
        features = np.array(
            [wall_frac, outside_open_frac, walked_frac, min(camera_count, 50.0), min(camera_dist, 5.0)],
            dtype=np.float32,
        )
        candidates.append({"center_xz": (cx, cz), "features": features})
    return candidates


MIN_DOOR_M = 0.55  # a real doorway is at least this wide; narrower gaps are scan holes


def largest_opening_m(fs, footprint, wall_mask: np.ndarray | None) -> float:
    """Width of the widest gap in the wall ring around the floor (0 if fully walled in). A gap is
    floor-boundary that is NOT backed by wall — i.e. somewhere you could walk out."""
    cell = fs.cell
    rows, cols = fs.free.shape
    floor = footprint.mask.astype(bool)
    wall = wall_mask if wall_mask is not None else np.zeros((rows, cols), dtype=bool)
    outer = binary_dilation(floor, iterations=max(1, int(0.4 / cell))) & ~floor
    wall_near = binary_dilation(wall, iterations=max(1, int(0.35 / cell)))
    opening = outer & ~wall_near
    labels, n = label(opening)
    widest = 0.0
    for i in range(1, n + 1):
        cells = np.argwhere(labels == i)
        extent = (cells.max(axis=0) - cells.min(axis=0) + 1) * cell
        widest = max(widest, float(max(extent)))
    return widest


def is_enclosed(fs, footprint, wall_mask: np.ndarray | None, min_door_m: float = MIN_DOOR_M) -> bool:
    """True if the room has no wall-ring gap wide enough to enter/exit — i.e. it was scanned with
    the door closed, so it is a sealed box with only small scan holes. Such rooms have no valid
    entrance or placement and should be skipped."""
    return largest_opening_m(fs, footprint, wall_mask) < min_door_m


def _merge_scored(points: list[tuple[tuple[float, float], float]], radius: float):
    """Greedy-merge scored points (((x, z), prob)) closer than `radius`; each cluster becomes its
    centroid with the best score. Returns clusters sorted by score, strongest first."""
    merged: list[tuple[tuple[float, float], float]] = []
    used = [False] * len(points)
    for i, (p, pr) in enumerate(points):
        if used[i]:
            continue
        cluster = [(p, pr)]
        used[i] = True
        for j in range(i + 1, len(points)):
            q = points[j][0]
            if not used[j] and np.hypot(q[0] - p[0], q[1] - p[1]) < radius:
                cluster.append(points[j])
                used[j] = True
        xs = np.array([c[0][0] for c in cluster])
        zs = np.array([c[0][1] for c in cluster])
        merged.append(((float(xs.mean()), float(zs.mean())), max(pr2 for _, pr2 in cluster)))
    merged.sort(key=lambda m: -m[1])
    return merged


def on_observed_floor(fs, footprint, centers_xz) -> np.ndarray:
    """Boolean per point: does it stand on OBSERVED FLOOR rather than float in a void?

    Three local conditions, all measured against the footprint mask (the flat floor actually seen by
    the scanner), because that mask is the only thing that proves a place is walkable:
      * support — at least FLOOR_SUPPORT_FRAC of the FLOOR_SUPPORT_RADIUS_M box around the point is
        observed floor, so the point is beside a real floor and not in the middle of an unscanned
        hole that merely happens to touch the room boundary;
      * reach — the nearest observed floor cell is at most FLOOR_REACH_M away, so a bin can be
        wheeled from the entrance onto the floor without crossing unknown space;
      * not airborne — if a per-cell "lowest geometry above the floor plane" grid is available, the
        point's own cell must not consist purely of geometry high up in the air (a wall top, a
        shelf, a mezzanine edge). FreeSpaceResult does not carry that grid today, so this condition
        is skipped unless a caller supplies fs.low_above_floor; the snap below still keeps the
        RETURNED point on the floor in the meantime.
    """
    cell, origin = fs.cell, fs.origin
    floor = np.asarray(footprint.mask, dtype=bool)
    rows, cols = floor.shape
    pts = np.asarray(centers_xz, dtype=float).reshape(-1, 2)
    if not len(pts) or not floor.any():
        return np.zeros(len(pts), dtype=bool)

    dist_floor = distance_transform_edt(~floor) * cell
    # summed-area table so the local floor fraction is O(1) per candidate
    integral = np.pad(np.cumsum(np.cumsum(floor.astype(np.int32), axis=0), axis=1), ((1, 0), (1, 0)))
    k = max(1, int(round(FLOOR_SUPPORT_RADIUS_M / cell)))
    low = getattr(fs, "low_above_floor", None)

    cols_i = np.floor((pts[:, 0] - origin[0]) / cell).astype(int)
    rows_i = np.floor((pts[:, 1] - origin[1]) / cell).astype(int)
    ok = np.zeros(len(pts), dtype=bool)
    for i, (r, c) in enumerate(zip(rows_i, cols_i)):
        if not (0 <= r < rows and 0 <= c < cols):
            continue  # outside the grid entirely: cannot be on any observed floor
        r0, r1 = max(0, r - k), min(rows, r + k + 1)
        c0, c1 = max(0, c - k), min(cols, c + k + 1)
        area = max((r1 - r0) * (c1 - c0), 1)
        support = (integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]) / area
        airborne = low is not None and np.isfinite(low[r, c]) and float(low[r, c]) > FLOOR_MAX_AIRBORNE_M
        ok[i] = (support >= FLOOR_SUPPORT_FRAC) and (dist_floor[r, c] <= FLOOR_REACH_M) and not airborne
    return ok


def snap_to_floor(fs, footprint, point: tuple[float, float]) -> tuple[float, float]:
    """Move a point to the centre of the nearest footprint-mask cell (no-op if it is already on one).
    Merging several candidates into a cluster centroid can land the centroid just outside the room, so
    this pulls it back in.

    NB: this only restores membership of footprint.mask; it does NOT re-check the support/airborne
    conditions in on_observed_floor, so a snapped point can still end up on a mask cell that was
    never scanned (measured: 3 of 139). Nor is the mask proof of observed floor — see the module
    docstring. Do not read this as a guarantee that the entrance is on real floor."""
    cell, origin = fs.cell, fs.origin
    floor = np.asarray(footprint.mask, dtype=bool)
    rows, cols = floor.shape
    c = int(np.floor((point[0] - origin[0]) / cell))
    r = int(np.floor((point[1] - origin[1]) / cell))
    if 0 <= r < rows and 0 <= c < cols and floor[r, c]:
        return (float(point[0]), float(point[1]))
    cells = np.argwhere(floor)
    if not len(cells):
        return (float(point[0]), float(point[1]))
    xs = origin[0] + (cells[:, 1] + 0.5) * cell
    zs = origin[1] + (cells[:, 0] + 0.5) * cell
    j = int(np.argmin(np.hypot(xs - point[0], zs - point[1])))
    return (float(xs[j]), float(zs[j]))


class DoorNet(nn.Module):
    """Tiny MLP: standardized perimeter features -> one logit (P(door))."""

    def __init__(self, n_features: int = len(FEATURE_NAMES), hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DoorClassifier:
    def __init__(self, weights: Path, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # weights_only=False: our own checkpoint stores numpy mean/std, which PyTorch 2.6's default
        # (weights_only=True) refuses to unpickle. Safe here since we produced the file ourselves.
        checkpoint = torch.load(weights, map_location=self.device, weights_only=False)
        self.mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["std"], dtype=np.float32)
        self.net = DoorNet(n_features=len(self.mean)).to(self.device)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()

    def score(self, features_list: list[np.ndarray]) -> list[float]:
        if not features_list:
            return []
        x = (np.stack(features_list) - self.mean) / self.std
        with torch.no_grad():
            probs = torch.sigmoid(self.net(torch.from_numpy(x).float().to(self.device)))
        return probs.cpu().numpy().tolist()


def load_door_model(weights: str | Path | None = None, device: str | None = None) -> DoorClassifier | None:
    path = Path(weights) if weights else WEIGHTS_PATH
    if not path.exists():
        return None
    return DoorClassifier(path, device)


def _cached_model() -> DoorClassifier | None:
    """Load once and reuse; reload only when the weights file changes (e.g. after a retrain)."""
    if not WEIGHTS_PATH.exists():
        return None
    mtime = WEIGHTS_PATH.stat().st_mtime
    if _cache.get("mtime") != mtime:
        _cache["model"] = load_door_model()
        _cache["mtime"] = mtime
    return _cache.get("model")


def find_doors_scored(fs, footprint, wall_mask, camera_xz,
                      keep_prob: float = KEEP_PROB) -> tuple[list[tuple[tuple[float, float], float]], bool]:
    """Automatic entrance detection with the confidence kept: ([( (x, z), prob ), ...], confident).

    `confident` is False when nothing cleared `keep_prob` and the best candidate was returned as a
    last-resort guess (prob is NaN for the geometric fallback, which has no model score). Callers
    that just want the points use find_doors(); this one exists so the GUI/report can flag a guess.
    """
    from . import placement  # local import to avoid an import cycle

    # Reset FIRST: LAST_DETECTION is module state, so leaving the previous scan's verdict in place
    # made a caller that loops over scans attribute one room's confidence to another (and the clicked
    # and enclosed paths in pipeline.compute_scene never reach this function at all).
    LAST_DETECTION.update(points=[], probs=[], confident=False, source="none",
                          n_candidates=0, n_floor_rejected=0)

    model = _cached_model()
    candidates = candidate_openings(fs, footprint, wall_mask, camera_xz) if model is not None else []

    if model is None or not candidates:
        # no model, or a footprint so ragged that not one candidate could be seeded
        points = placement.detect_entrances(fs, footprint, wall_mask, camera_xz)
        points = [snap_to_floor(fs, footprint, p) for p in points]
        LAST_DETECTION.update(points=points, probs=[float("nan")] * len(points), confident=False,
                              source="heuristic", n_candidates=len(candidates), n_floor_rejected=0)
        if not points:
            log.warning("fant ingen inngang i det hele tatt – rommet får ingen forslag til "
                        "plassering. Klikk inngangen manuelt med src.set_entrance.")
        return [(p, float("nan")) for p in points], False

    probs = np.asarray(model.score([c["features"] for c in candidates]), dtype=float)
    centers = [c["center_xz"] for c in candidates]

    # Floor constraint FIRST, so a floating candidate can never win the fallback below. If it would
    # reject everything (a scan with no usable floor at all) we keep the whole pool rather than
    # return nothing — no entrance is the worst outcome there is.
    floor_ok = on_observed_floor(fs, footprint, centers)
    n_rejected = int((~floor_ok).sum())
    pool = np.flatnonzero(floor_ok) if floor_ok.any() else np.arange(len(candidates))

    keep = pool[probs[pool] >= keep_prob]
    confident = bool(len(keep))
    if not confident:
        # The user's rule: always hand back the single highest-confidence point. A rough door beats
        # no door, because no door means no analysis for the whole room.
        keep = pool[[int(np.argmax(probs[pool]))]]

    # keep only the strongest one or two doors after merging — not every perimeter blip
    merged = _merge_scored([(centers[i], float(probs[i])) for i in keep], MERGE_RADIUS_M)[:MAX_DOORS]
    doors_out = [(snap_to_floor(fs, footprint, xz), p) for xz, p in merged]

    LAST_DETECTION.update(points=[xz for xz, _ in doors_out], probs=[p for _, p in doors_out],
                          confident=confident, source="model" if confident else "model-guess",
                          n_candidates=len(candidates), n_floor_rejected=n_rejected)
    best = doors_out[0][1] if doors_out else float("nan")
    if confident:
        log.info("inngang funnet automatisk (p=%.2f, %d kandidat(er), %d forkastet som flyvende)",
                 best, len(candidates), n_rejected)
    else:
        log.warning("usikker inngang: ingen kandidat over terskelen (%.2f) – bruker beste gjetning "
                    "(p=%.2f). Sjekk resultatet, og klikk inngangen med src.set_entrance hvis "
                    "plasseringen ser feil ut.", keep_prob, best)
    return doors_out, confident


def find_doors(fs, footprint, wall_mask, camera_xz, keep_prob: float = KEEP_PROB) -> list[tuple[float, float]]:
    """Automatic entrance detection. Uses the trained model if present, else the heuristic.

    Returns [(x, z), ...], at most MAX_DOORS, strongest first — and never empty while a single
    candidate exists (see find_doors_scored / LAST_DETECTION for whether it was a real detection or
    a last-resort guess)."""
    doors_out, _ = find_doors_scored(fs, footprint, wall_mask, camera_xz, keep_prob)
    return [xz for xz, _ in doors_out]
