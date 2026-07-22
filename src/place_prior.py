"""Learned placement prior: score a candidate spot by how much it looks like where YOU actually
put bins.

The hard rules (no overlap, off the push-path, reachable, don't block existing) are enforced by the
packer in placement.py. This model adds a SOFT, data-driven preference on top: trained on your
annotated bins (positive = a real bin stands here) vs random free-floor spots (negative), it learns
your habits — how close to a wall, how close to other bins, how much clearance, how far from the
door — and returns P(a bin belongs here). The packer uses it to RANK where to place new bins so
they match your real layout. If no model is trained, the packer falls back to its geometric ranking.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt

WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "place_prior.pt"

FEATURE_NAMES = ("dist_wall_m", "clearance_m", "dist_other_bin_m", "dist_entrance_m")
_cache: dict = {}


def scene_feature_maps(fs, wall_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell distance-to-wall and clearance (distance to nearest obstacle) maps for one scene."""
    cell = fs.cell
    wall = wall_mask if wall_mask is not None else np.zeros(fs.free.shape, dtype=bool)
    dist_wall = distance_transform_edt(~wall) * cell if wall.any() else np.full(fs.free.shape, 5.0)
    clearance = distance_transform_edt(fs.free) * cell
    return dist_wall, clearance


def features_at(
    xz: tuple[float, float],
    fs,
    dist_wall: np.ndarray,
    clearance: np.ndarray,
    other_centers: list[tuple[float, float]],
    entrances: list[tuple[float, float]],
) -> np.ndarray:
    """Feature vector describing one candidate spot (must match training and inference exactly)."""
    cell, origin = fs.cell, fs.origin
    rows, cols = fs.free.shape
    col = int(np.clip((xz[0] - origin[0]) / cell, 0, cols - 1))
    row = int(np.clip((xz[1] - origin[1]) / cell, 0, rows - 1))
    dw = float(dist_wall[row, col])
    cl = float(clearance[row, col])
    dob = (float(np.min([np.hypot(xz[0] - ox, xz[1] - oz) for ox, oz in other_centers]))
           if other_centers else 5.0)
    den = (float(np.min([np.hypot(xz[0] - ex, xz[1] - ez) for ex, ez in entrances]))
           if entrances else 8.0)
    return np.array([min(dw, 3.0), min(cl, 3.0), min(dob, 5.0), min(den, 8.0)], dtype=np.float32)


class PlacePriorNet(nn.Module):
    """Tiny MLP: standardized placement features -> one logit (P(a bin belongs here))."""

    def __init__(self, n_features: int = len(FEATURE_NAMES), hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PlacePrior:
    def __init__(self, weights: Path, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weights, map_location=self.device, weights_only=False)  # our own file
        self.mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["std"], dtype=np.float32)
        self.net = PlacePriorNet(n_features=len(self.mean)).to(self.device)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()

    def score(self, feats: list[np.ndarray]) -> list[float]:
        if not feats:
            return []
        x = (np.stack(feats) - self.mean) / self.std
        with torch.no_grad():
            probs = torch.sigmoid(self.net(torch.from_numpy(x).float().to(self.device)))
        return probs.cpu().numpy().tolist()

    def score_spots(self, spots_xz, fs, wall_mask, other_centers, entrances) -> list[float]:
        dist_wall, clearance = scene_feature_maps(fs, wall_mask)
        feats = [features_at(xz, fs, dist_wall, clearance, other_centers, entrances) for xz in spots_xz]
        return self.score(feats)


def load_place_prior(weights: str | Path | None = None, device: str | None = None) -> PlacePrior | None:
    path = Path(weights) if weights else WEIGHTS_PATH
    if not path.exists():
        return None
    return PlacePrior(path, device)


def cached_prior() -> PlacePrior | None:
    """Load once, reload only when the weights file changes (e.g. after a retrain)."""
    if not WEIGHTS_PATH.exists():
        return None
    mtime = WEIGHTS_PATH.stat().st_mtime
    if _cache.get("mtime") != mtime:
        _cache["prior"] = load_place_prior()
        _cache["mtime"] = mtime
    return _cache.get("prior")
