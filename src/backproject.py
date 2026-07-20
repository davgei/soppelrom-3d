"""Back-project 2D detections to 3D and merge them across frames into bin instances.

Each 2D box is lifted to a 3D point set via the frame's depth map and pose. Detections of the
same physical bin from different frames land in the same place in world space, so clustering
their centroids (DBSCAN) groups them into instances. Requiring detections from several frames
filters out single-frame flukes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.signal import find_peaks

from .detection import Detection2D
from .scan_io import Keyframe, ScanArchive

SPLIT_MAX_SINGLE = 1.6      # a cluster longer than this may be several bins in a row
SPLIT_MIN_SEPARATION = 0.5  # density peaks (bins) must be at least this far apart (m)
MIN_SEGMENT_POINTS = 30

_ARKIT_FLIP = np.diag([1.0, -1.0, -1.0])


@dataclass
class BinInstance:
    center: np.ndarray          # (x, y, z) world, metres
    size: np.ndarray            # (length, height, width) metres — footprint length x width, height up
    yaw_deg: float              # rotation of the footprint around the up axis
    rect: tuple                 # cv2.minAreaRect in (X, Z)
    n_views: int
    mean_confidence: float
    labels: dict[str, int] = field(default_factory=dict)
    points: np.ndarray | None = None


def _box_points_world(
    archive: ScanArchive,
    keyframe: Keyframe,
    xyxy: np.ndarray,
    shrink: float = 0.15,
    min_confidence: int = 54,
    depth_band_m: float = 0.6,
) -> np.ndarray:
    depth = archive.depth_m(keyframe.timestamp)
    confidence = archive.confidence(keyframe.timestamp)
    depth_h, depth_w = depth.shape
    scale_x = depth_w / keyframe.rgb_width
    scale_y = depth_h / keyframe.rgb_height

    x1, y1, x2, y2 = xyxy
    box_w, box_h = x2 - x1, y2 - y1
    x1 += shrink * box_w
    x2 -= shrink * box_w
    y1 += shrink * box_h
    y2 -= shrink * box_h

    u1 = max(int(x1 * scale_x), 0)
    u2 = min(int(np.ceil(x2 * scale_x)), depth_w)
    v1 = max(int(y1 * scale_y), 0)
    v2 = min(int(np.ceil(y2 * scale_y)), depth_h)
    if u2 <= u1 or v2 <= v1:
        return np.empty((0, 3))

    us, vs = np.meshgrid(np.arange(u1, u2), np.arange(v1, v2))
    z = depth[v1:v2, u1:u2]
    conf = confidence[v1:v2, u1:u2]
    valid = (conf >= min_confidence) & (z > 0.1)
    if not valid.any():
        return np.empty((0, 3))

    z_valid = z[valid]
    z_med = float(np.median(z_valid))
    keep = valid & (np.abs(z - z_med) < depth_band_m)

    u = us[keep]
    v = vs[keep]
    zk = z[keep]
    fx = keyframe.intrinsics[0, 0] * scale_x
    fy = keyframe.intrinsics[1, 1] * scale_y
    cx = keyframe.intrinsics[0, 2] * scale_x
    cy = keyframe.intrinsics[1, 2] * scale_y
    x = (u - cx) * zk / fx
    y = (v - cy) * zk / fy
    points_camera = np.stack([x, y, zk], axis=1) @ _ARKIT_FLIP.T

    rotation = keyframe.pose_cam_to_world[:3, :3]
    translation = keyframe.pose_cam_to_world[:3, 3]
    return points_camera @ rotation.T + translation


def _cluster_centroids(centroids: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Simple DBSCAN over a handful of detection centroids (avoids a sklearn dependency)."""
    n = len(centroids)
    labels = np.full(n, -1)
    visited = np.zeros(n, dtype=bool)
    cluster = 0
    distance = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)
    for i in range(n):
        if visited[i]:
            continue
        neighbors = np.where(distance[i] < eps)[0]
        if len(neighbors) < min_samples:
            continue
        queue = list(neighbors)
        visited[i] = True
        labels[i] = cluster
        while queue:
            j = queue.pop()
            if labels[j] == -1:
                labels[j] = cluster
            if visited[j]:
                continue
            visited[j] = True
            j_neighbors = np.where(distance[j] < eps)[0]
            if len(j_neighbors) >= min_samples:
                queue.extend(j_neighbors)
        cluster += 1
    return labels


def _split_masks(points_xz: np.ndarray, bin_width: float = 0.05) -> list[np.ndarray] | None:
    """Split a merged cluster of adjacent bins by finding density valleys (the gaps between
    bins) along the cluster's long axis. Returns per-segment boolean masks, or None if the
    cluster is single-bin sized or shows no clear multi-bin structure."""
    rect = cv2.minAreaRect(points_xz.astype(np.float32))
    box = cv2.boxPoints(rect)
    edge_a, edge_b = box[1] - box[0], box[2] - box[1]
    axis = edge_a if np.linalg.norm(edge_a) >= np.linalg.norm(edge_b) else edge_b
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return None
    projection = points_xz @ (axis / norm)

    low, high = float(projection.min()), float(projection.max())
    if high - low <= SPLIT_MAX_SINGLE:
        return None

    n_bins = max(4, int((high - low) / bin_width))
    hist, edges = np.histogram(projection, bins=n_bins, range=(low, high))
    smooth = np.convolve(hist, np.ones(3) / 3, mode="same")
    distance = max(1, int(SPLIT_MIN_SEPARATION / ((high - low) / n_bins)))
    peaks, _ = find_peaks(smooth, distance=distance, prominence=max(smooth.max() * 0.25, 1.0))
    if len(peaks) <= 1:
        return None

    cuts = [float(edges[p1 + int(np.argmin(smooth[p1:p2 + 1]))]) for p1, p2 in zip(peaks[:-1], peaks[1:])]
    bounds = [-np.inf, *cuts, np.inf]
    return [(projection >= a) & (projection < b) for a, b in zip(bounds[:-1], bounds[1:])]


def _instance_from_points(
    points: np.ndarray,
    floor_height: float | None,
    n_views: int,
    mean_confidence: float,
    label_counts: dict[str, int],
) -> BinInstance:
    rect = cv2.minAreaRect(points[:, [0, 2]].astype(np.float32))
    (cx, cz), (side_a, side_b), angle = rect
    length, width = (side_a, side_b) if side_a >= side_b else (side_b, side_a)
    y_min = floor_height if floor_height is not None else float(points[:, 1].min())
    y_max = float(np.percentile(points[:, 1], 98))
    return BinInstance(
        center=np.array([cx, (y_min + y_max) / 2, cz]),
        size=np.array([length, y_max - y_min, width]),
        yaw_deg=float(angle),
        rect=rect,
        n_views=n_views,
        mean_confidence=mean_confidence,
        labels=label_counts,
        points=points,
    )


def merge_detections(
    archive: ScanArchive,
    per_frame: dict[int, list[Detection2D]],
    floor_height: float | None = None,
    eps: float = 0.4,
    min_views: int = 2,
    max_points_per_detection: int = 400,
) -> list[BinInstance]:
    centroids: list[np.ndarray] = []
    point_sets: list[np.ndarray] = []
    confidences: list[float] = []
    labels_2d: list[str] = []

    for timestamp, detections in per_frame.items():
        keyframe = archive.keyframe(timestamp)
        for det in detections:
            points = _box_points_world(archive, keyframe, det.xyxy)
            if len(points) < 30:
                continue
            if len(points) > max_points_per_detection:
                idx = np.random.default_rng(0).choice(len(points), max_points_per_detection, replace=False)
                points = points[idx]
            centroids.append(points.mean(axis=0))
            point_sets.append(points)
            confidences.append(det.confidence)
            labels_2d.append(det.label)

    if not centroids:
        return []

    cluster_labels = _cluster_centroids(np.array(centroids), eps, min_samples=min_views)
    instances: list[BinInstance] = []
    for cluster_id in range(cluster_labels.max() + 1 if len(cluster_labels) else 0):
        member = np.where(cluster_labels == cluster_id)[0]
        if len(member) < min_views:
            continue
        points = np.concatenate([point_sets[i] for i in member])

        if floor_height is not None:
            points = points[points[:, 1] > floor_height + 0.05]
            if len(points) < 30:
                continue

        lo, hi = np.percentile(points, [2, 98], axis=0)
        trimmed = points[np.all((points >= lo) & (points <= hi), axis=1)]
        if len(trimmed) < 30:
            trimmed = points

        label_counts: dict[str, int] = {}
        for i in member:
            label_counts[labels_2d[i]] = label_counts.get(labels_2d[i], 0) + 1
        mean_confidence = float(np.mean([confidences[i] for i in member]))

        masks = _split_masks(trimmed[:, [0, 2]])
        groups = masks if masks is not None else [np.ones(len(trimmed), dtype=bool)]
        for group in groups:
            segment = trimmed[group]
            if len(segment) < MIN_SEGMENT_POINTS:
                continue
            instances.append(
                _instance_from_points(
                    segment, floor_height, len(member), mean_confidence, label_counts
                )
            )

    instances.sort(key=lambda b: -b.n_views)
    return instances
