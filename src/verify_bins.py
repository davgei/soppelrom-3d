"""PointNet++ verification head: does a 3D proposal really look like a bin?

The YOLO -> back-projection pipeline over-proposes: walls, floor clutter and thin slivers
survive the hand-set size gate in binfit and reach the annotation tool as false boxes. This
small PointNet++ classifier looks at the actual 3D point distribution *inside* a proposal box
and returns P(bin). prepare_scan uses it to drop confident non-bins before they are saved.

Design choices tuned for our tiny dataset (few annotated scenes):
  * Points are cropped from the reconstructed cloud and kept in METRES (XZ centered on the
    crop centroid, Y = height above the floor). Absolute size and "stands on the floor" are the
    signals that separate a bin from a sliver, so we deliberately do NOT scale them away.
  * The box footprint (long, short, height) is fed to the classifier head as extra features.
  * The net is intentionally small (2 set-abstraction layers) to avoid overfitting.

Pure PyTorch (no compiled CUDA ops), so it runs on the GPU when available and on CPU otherwise.
If models/verifier_latest.pt is absent, load_verifier() returns None and the pipeline behaves
exactly as before.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .annotations import BinBox

WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "verifier_latest.pt"

N_POINTS = 512          # points sampled per proposal before the net
REF_SCALE = 1.0         # metres; keep metric scale so absolute size stays a signal
CROP_MARGIN_M = 0.10    # grow the box a little so the whole bin is captured
MIN_CROP_POINTS = 20    # fewer points than this inside the box -> treat as a non-bin
EXTRA_DIMS = 3          # box (long, short, height) fed to the classifier head

# prepare_scan thresholds (conservative: a dropped real bin costs more than a false proposal)
DROP_BELOW = 0.30       # drop a kept proposal when P(bin) is below this (lower = keep more bins)
REVIEW_BELOW = 0.60     # keep, but force human review (never auto-approve) below this


# --------------------------------------------------------------------------------------
# Preprocessing — shared by training and inference so both see identical inputs
# --------------------------------------------------------------------------------------

def crop_box(points: np.ndarray, center, extent, axes, margin: float = CROP_MARGIN_M) -> np.ndarray:
    """Return the subset of `points` (N,3) inside the oriented box, grown by `margin`."""
    ux, uy, uz = axes
    delta = points - np.asarray(center, dtype=float)
    lx = delta @ ux
    ly = delta @ uy
    lz = delta @ uz
    ex, ey, ez = extent
    inside = (
        (np.abs(lx) <= ex / 2 + margin)
        & (np.abs(ly) <= ey / 2 + margin)
        & (np.abs(lz) <= ez / 2 + margin)
    )
    return points[inside]


def normalize_points(points: np.ndarray, floor_height: float | None) -> np.ndarray:
    """Center XZ on the crop centroid, set Y=0 at the floor, keep metric metres."""
    out = points.astype(np.float32).copy()
    centroid = out.mean(axis=0)
    y0 = floor_height if floor_height is not None else float(out[:, 1].min())
    out[:, 0] -= centroid[0]
    out[:, 2] -= centroid[2]
    out[:, 1] -= y0
    out /= REF_SCALE
    return out


def sample_points(points: np.ndarray, n: int = N_POINTS, rng: np.random.Generator | None = None) -> np.ndarray:
    """Resample to exactly `n` points (random subset if more, resample with replacement if fewer)."""
    rng = rng or np.random.default_rng()
    replace = len(points) < n
    idx = rng.choice(len(points), n, replace=replace)
    return points[idx]


def box_extra(box: BinBox) -> np.ndarray:
    """Yaw-invariant footprint features for the classifier head: (long, short, height)/scale."""
    ex, ey, ez = box.extent
    long_side, short_side = max(ex, ez), min(ex, ez)
    return np.array([long_side, short_side, ey], dtype=np.float32) / REF_SCALE


def prepare_box(
    box: BinBox, cloud_points: np.ndarray, floor_height: float | None, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray] | None:
    """Crop -> normalize -> sample. Returns (points (N,3), extra (3,)) or None if too sparse."""
    crop = crop_box(cloud_points, box.center, box.extent, box.local_axes())
    if len(crop) < MIN_CROP_POINTS:
        return None
    points = sample_points(normalize_points(crop, floor_height), N_POINTS, rng)
    return points, box_extra(box)


# --------------------------------------------------------------------------------------
# Minimal PointNet++ (single-scale grouping), implemented in plain PyTorch
# --------------------------------------------------------------------------------------

def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index. points (B,N,C), idx (B,S) or (B,S,K) -> (B,S,C) / (B,S,K,C)."""
    batch = torch.arange(points.shape[0], device=points.device)
    view = [points.shape[0]] + [1] * (idx.dim() - 1)
    batch = batch.view(view).expand_as(idx)
    return points[batch, idx]


def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """FPS indices (B,npoint). Deterministic start at index 0 (inputs are pre-shuffled)."""
    b, n, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.full((b, n), 1e10, device=device)
    farthest = torch.zeros(b, dtype=torch.long, device=device)
    batch = torch.arange(b, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch, farthest, :].view(b, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def _query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Group up to `nsample` neighbours within `radius` of each query point -> (B,S,nsample)."""
    b, n, _ = xyz.shape
    s = new_xyz.shape[1]
    device = xyz.device
    sqrdist = torch.cdist(new_xyz, xyz) ** 2
    group_idx = torch.arange(n, device=device).view(1, 1, n).repeat(b, s, 1)
    group_idx[sqrdist > radius ** 2] = n
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0:1].expand(-1, -1, nsample)
    empty = group_idx == n
    group_idx[empty] = group_first[empty]  # pad missing neighbours with the nearest one
    return group_idx


def _sample_and_group(
    npoint: int, radius: float, nsample: int, xyz: torch.Tensor, features: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    fps_idx = _farthest_point_sample(xyz, npoint)
    new_xyz = _index_points(xyz, fps_idx)
    idx = _query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = _index_points(xyz, idx) - new_xyz.unsqueeze(2)  # local coordinates
    if features is not None:
        grouped = torch.cat([grouped_xyz, _index_points(features, idx)], dim=-1)
    else:
        grouped = grouped_xyz
    return new_xyz, grouped  # (B,npoint,3), (B,npoint,nsample,3+C)


class _SetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: list[int]) -> None:
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers: list[nn.Module] = []
        last = in_channel + 3
        for out in mlp:
            layers += [nn.Conv2d(last, out, 1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor | None):
        new_xyz, grouped = _sample_and_group(self.npoint, self.radius, self.nsample, xyz, features)
        grouped = grouped.permute(0, 3, 2, 1)          # (B, C+3, nsample, npoint)
        grouped = self.mlp(grouped)
        new_features = torch.max(grouped, dim=2)[0]     # (B, mlp[-1], npoint)
        return new_xyz, new_features.permute(0, 2, 1)   # (B, npoint, mlp[-1])


class _GlobalAbstraction(nn.Module):
    def __init__(self, in_channel: int, mlp: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_channel + 3
        for out in mlp:
            layers += [nn.Conv1d(last, out, 1), nn.BatchNorm1d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xyz, features], dim=-1).permute(0, 2, 1)  # (B, 3+C, N)
        return torch.max(self.mlp(x), dim=-1)[0]                 # (B, mlp[-1])


class BinVerifierNet(nn.Module):
    """Compact PointNet++ SSG -> single logit (P(bin))."""

    def __init__(self, extra_dims: int = EXTRA_DIMS) -> None:
        super().__init__()
        self.sa1 = _SetAbstraction(npoint=128, radius=0.2, nsample=32, in_channel=0, mlp=[32, 32, 64])
        self.sa2 = _SetAbstraction(npoint=32, radius=0.4, nsample=32, in_channel=64, mlp=[64, 64, 128])
        self.global_sa = _GlobalAbstraction(in_channel=128, mlp=[128, 256])
        self.fc1 = nn.Linear(256 + extra_dims, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, xyz: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        l1_xyz, l1_feat = self.sa1(xyz, None)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        g = self.global_sa(l2_xyz, l2_feat)
        z = torch.relu(self.bn1(self.fc1(torch.cat([g, extra], dim=1))))
        return self.fc2(self.drop(z)).squeeze(-1)  # (B,)


# --------------------------------------------------------------------------------------
# Inference wrapper
# --------------------------------------------------------------------------------------

class BinVerifier:
    def __init__(self, weights: Path, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weights, map_location=self.device, weights_only=False)  # our own file
        self.net = BinVerifierNet(extra_dims=checkpoint.get("extra_dims", EXTRA_DIMS)).to(self.device)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()

    def score_boxes(
        self, boxes: list[BinBox], cloud_points: np.ndarray, floor_height: float | None
    ) -> list[float]:
        """P(bin) for each box. Too-sparse crops score 0.0 (not a bin) without touching the net."""
        rng = np.random.default_rng(0)  # deterministic sampling at inference
        prepared: list[tuple[np.ndarray, np.ndarray]] = []
        valid: list[int] = []
        for i, box in enumerate(boxes):
            item = prepare_box(box, cloud_points, floor_height, rng)
            if item is not None:
                prepared.append(item)
                valid.append(i)

        scores = [0.0] * len(boxes)
        if not prepared:
            return scores
        xyz = torch.from_numpy(np.stack([p for p, _ in prepared])).float().to(self.device)
        extra = torch.from_numpy(np.stack([e for _, e in prepared])).float().to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.net(xyz, extra)).cpu().numpy()
        for i, prob in zip(valid, probs):
            scores[i] = float(prob)
        return scores

    def score_box(self, box: BinBox, cloud_points: np.ndarray, floor_height: float | None) -> float:
        return self.score_boxes([box], cloud_points, floor_height)[0]


def load_verifier(weights: str | Path | None = None, device: str | None = None) -> BinVerifier | None:
    """Return a verifier if trained weights exist, else None (pipeline runs unchanged)."""
    path = Path(weights) if weights else WEIGHTS_PATH
    if not path.exists():
        return None
    return BinVerifier(path, device)
