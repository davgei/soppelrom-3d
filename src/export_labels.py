"""Export approved 3D box annotations as 2D YOLO training labels.

Each approved 3D box is projected into every posed RGB frame of its scan. A depth-map
occlusion test drops frames where the box is hidden or barely visible, so one 3D
annotation yields tens of clean 2D labels for free. Scans are deduplicated by content
hash and split into train/val BY SCAN so frames of the same room never leak across splits.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import time
from pathlib import Path

import cv2
import numpy as np

from .annotations import BIN_TYPES, STATUS_APPROVED, BinBox, load_annotations
from .paths import ANNOTATION_DIR, DATASET_DIR, PROJECT_ROOT, RAW_DIR
from .scan_io import Keyframe, ScanArchive

_FLIP = np.diag([1.0, -1.0, -1.0])
CLASS_NAMES = list(BIN_TYPES)


def _robust_rmtree(path: Path, attempts: int = 6) -> None:
    """Delete a tree even when Windows/OneDrive briefly locks files: clear the read-only bit and
    retry a few times with a short pause (sync/AV usually release the handle). Best-effort — if
    something stays locked we warn and let the export overwrite the current files in place, rather
    than crashing the whole training run."""
    def _onexc(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except Exception:
            pass

    for _ in range(attempts):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onexc=_onexc)          # Python 3.12+
        except TypeError:
            shutil.rmtree(path, onerror=lambda f, p, e: _onexc(f, p, e))  # older Python
        except Exception:
            pass
        if not path.exists():
            return
        time.sleep(0.7)
    if path.exists():
        print(f"warning: could not fully clear {path} (locked by OneDrive/another app?) — "
              "exporting over the existing files", flush=True)


def _zip_hash(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            md5.update(chunk)
    return md5.hexdigest()


def project_box(
    box: BinBox, keyframe: Keyframe, depth: np.ndarray
) -> tuple[float, float, float, float] | None:
    rotation = keyframe.pose_cam_to_world[:3, :3]
    translation = keyframe.pose_cam_to_world[:3, 3]
    corners_camera = (box.corners() - translation) @ rotation @ _FLIP
    z = corners_camera[:, 2]
    if z.min() < 0.15:
        return None

    fx = keyframe.intrinsics[0, 0]
    fy = keyframe.intrinsics[1, 1]
    cx = keyframe.intrinsics[0, 2]
    cy = keyframe.intrinsics[1, 2]
    u = fx * corners_camera[:, 0] / z + cx
    v = fy * corners_camera[:, 1] / z + cy

    x1, y1, x2, y2 = float(u.min()), float(v.min()), float(u.max()), float(v.max())
    full_area = (x2 - x1) * (y2 - y1)
    x1c, y1c = max(x1, 0.0), max(y1, 0.0)
    x2c, y2c = min(x2, float(keyframe.rgb_width)), min(y2, float(keyframe.rgb_height))
    if x2c - x1c < 12 or y2c - y1c < 12:
        return None
    if (x2c - x1c) * (y2c - y1c) < 0.25 * full_area:
        return None

    depth_h, depth_w = depth.shape
    scale_x = depth_w / keyframe.rgb_width
    scale_y = depth_h / keyframe.rgb_height
    du1 = max(int(x1c * scale_x), 0)
    du2 = min(int(np.ceil(x2c * scale_x)), depth_w)
    dv1 = max(int(y1c * scale_y), 0)
    dv2 = min(int(np.ceil(y2c * scale_y)), depth_h)
    region = depth[dv1:dv2, du1:du2]
    valid = region > 0.1
    if valid.sum() >= 10:
        in_range = valid & (region >= z.min() - 0.4) & (region <= z.max() + 0.4)
        if in_range.sum() / valid.sum() < 0.2:
            return None

    return x1c, y1c, x2c, y2c


def export_scan(
    zip_path: Path,
    boxes: list[BinBox],
    split: str,
    stride: int,
) -> tuple[int, int]:
    images_dir = DATASET_DIR / "images" / split
    labels_dir = DATASET_DIR / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    archive = ScanArchive(zip_path)
    n_labels = 0
    n_frames = 0
    for index, timestamp in enumerate(archive.timestamps):
        if index % stride:
            continue
        keyframe = archive.keyframe(timestamp)
        depth = archive.depth_m(timestamp)
        lines: list[str] = []
        for box in boxes:
            bbox = project_box(box, keyframe, depth)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            w, h = keyframe.rgb_width, keyframe.rgb_height
            class_id = CLASS_NAMES.index(box.bin_type) if box.bin_type in CLASS_NAMES else 3
            lines.append(
                f"{class_id} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )
        stem = f"{zip_path.stem}_{timestamp}"
        rgb = archive.rgb(timestamp)
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_frames += 1
        n_labels += len(lines)
    archive.close()
    return n_frames, n_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Export approved 3D boxes as a YOLO 2D dataset.")
    parser.add_argument("--val-scan", default=None, help="scan stem to use as validation (default: first)")
    parser.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    parser.add_argument("--clean", action="store_true", help="delete the existing dataset first")
    args = parser.parse_args()

    if args.clean and DATASET_DIR.exists():
        _robust_rmtree(DATASET_DIR)

    annotated: list[tuple[Path, list[BinBox]]] = []
    seen_hashes: set[str] = set()
    for annotation_path in sorted(ANNOTATION_DIR.glob("*.json")):
        zip_path = RAW_DIR / f"{annotation_path.stem}.zip"
        if not zip_path.exists():
            print(f"skipping {annotation_path.stem}: zip not found in data/raw")
            continue
        content_hash = _zip_hash(zip_path)
        if content_hash in seen_hashes:
            print(f"skipping {annotation_path.stem}: duplicate scan content")
            continue
        seen_hashes.add(content_hash)
        _, boxes = load_annotations(annotation_path)
        approved = [b for b in boxes if b.status == STATUS_APPROVED]
        if approved:
            annotated.append((zip_path, approved))

    if len(annotated) < 2:
        raise SystemExit("need at least 2 annotated scans (1 train + 1 val)")

    val_stem = args.val_scan or annotated[0][0].stem
    print(f"{len(annotated)} unique annotated scans, val scan: {val_stem}\n")

    totals = {"train": [0, 0], "val": [0, 0]}
    for zip_path, boxes in annotated:
        split = "val" if zip_path.stem == val_stem else "train"
        n_frames, n_labels = export_scan(zip_path, boxes, split, args.stride)
        totals[split][0] += n_frames
        totals[split][1] += n_labels
        print(f"{zip_path.stem} -> {split}: {n_frames} frames, {n_labels} labels")

    yaml_path = DATASET_DIR / "dataset.yaml"
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"path: {DATASET_DIR.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8",
    )
    print(f"\ntrain: {totals['train'][0]} frames / {totals['train'][1]} labels")
    print(f"val:   {totals['val'][0]} frames / {totals['val'][1]} labels")
    print(f"dataset -> {yaml_path}")


if __name__ == "__main__":
    main()
