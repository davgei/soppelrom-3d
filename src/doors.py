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
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_dilation, binary_erosion

WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "doors_latest.pt"

# Feature order — keep in sync with train_doors. All are cheap local grid/geometry signals.
FEATURE_NAMES = ("wall_frac", "outside_open_frac", "walked_frac", "camera_count", "camera_dist_m")
CANDIDATE_SPACING_M = 0.30   # sample a candidate roughly every this far along the boundary
WINDOW_M = 0.45              # local neighbourhood used to describe a candidate
KEEP_PROB = 0.40             # find_doors keeps candidates scoring at least this (favours recall)
MERGE_RADIUS_M = 0.8         # kept candidates closer than this are one door

_cache: dict = {}


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


def _merge_points(points: list[tuple[float, float]], radius: float) -> list[tuple[float, float]]:
    """Greedy-merge points closer than `radius` into their centroid (one door per cluster)."""
    merged: list[tuple[float, float]] = []
    used = [False] * len(points)
    for i, p in enumerate(points):
        if used[i]:
            continue
        cluster = [p]
        used[i] = True
        for j in range(i + 1, len(points)):
            if not used[j] and np.hypot(points[j][0] - p[0], points[j][1] - p[1]) < radius:
                cluster.append(points[j])
                used[j] = True
        arr = np.array(cluster)
        merged.append((float(arr[:, 0].mean()), float(arr[:, 1].mean())))
    return merged


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
        checkpoint = torch.load(weights, map_location=self.device)
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


def find_doors(fs, footprint, wall_mask, camera_xz, keep_prob: float = KEEP_PROB) -> list[tuple[float, float]]:
    """Automatic entrance detection. Uses the trained model if present, else the heuristic."""
    from . import placement  # local import to avoid an import cycle

    model = _cached_model()
    if model is None:
        return placement.detect_entrances(fs, footprint, wall_mask, camera_xz)

    candidates = candidate_openings(fs, footprint, wall_mask, camera_xz)
    if not candidates:
        return placement.detect_entrances(fs, footprint, wall_mask, camera_xz)

    probs = model.score([c["features"] for c in candidates])
    # A real door is where the scanner actually walked in/out (feature index 2 = walked_frac).
    # Gating on that on top of the learned score cuts false doors on walls no one approached.
    kept = [c["center_xz"] for c, p in zip(candidates, probs)
            if p >= keep_prob and c["features"][2] > 0.02]
    if not kept:  # a room has at least one way in — keep the single best walked-through candidate
        walked = [(c, p) for c, p in zip(candidates, probs) if c["features"][2] > 0.02] \
            or list(zip(candidates, probs))
        kept = [max(walked, key=lambda cp: cp[1])[0]["center_xz"]]
    return _merge_points(kept, MERGE_RADIUS_M)
