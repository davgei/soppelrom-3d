"""Train the PointNet++ bin verifier from the annotated scans and publish verifier_latest.pt.

Dataset (all crops taken from the reconstructed cloud, same as inference):
  positives      – the human 3D annotations (real bins)
  hard negatives – cached pipeline proposals that do NOT match any annotation
                   (exactly the false positives we want to suppress)
  random negatives – random oriented boxes on the floor away from any bin, biased toward
                   slivers/clutter, so the net learns absolute size and "bin-like" shape

Because there are few scenes, we split by SCENE (not by crop) so the validation score reflects
generalization to unseen rooms, and augment positives heavily (yaw, mirror, jitter, mild scale).

    .venv\\Scripts\\python.exe -m src.train_verifier            # train + hold out some scenes
    .venv\\Scripts\\python.exe -m src.train_verifier --val-frac 0   # use every scene, save final
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader, Dataset

from . import verify_bins
from .annotations import BinBox, load_annotations
from .reconstruct import ReconstructionConfig, reconstruct
from .scan_io import ScanArchive
from .verify_bins import EXTRA_DIMS, N_POINTS, BinVerifierNet, box_extra, crop_box, normalize_points

from .paths import ANNOTATION_DIR, CACHE_ROOT, PROJECT_ROOT, RAW_DIR

GT_MATCH_DIST = 0.5      # a proposal within this XZ distance of an annotation is the SAME bin
NEG_CLEAR_DIST = 0.75    # random negatives must be at least this far (XZ) from every bin
RANDOM_NEG_PER_SCENE = 8


def _load_cloud(stem: str) -> np.ndarray | None:
    cache = CACHE_ROOT / stem / "cloud.ply"
    if cache.exists():
        return np.asarray(o3d.io.read_point_cloud(str(cache)).points)
    zip_path = RAW_DIR / f"{stem}.zip"
    if zip_path.exists():
        cloud = reconstruct(ScanArchive(zip_path), ReconstructionConfig(min_confidence=255, max_depth_m=5.0))
        return np.asarray(cloud.points)
    return None


def _gt_centers(boxes: list[BinBox]) -> np.ndarray:
    if not boxes:
        return np.empty((0, 2))
    return np.array([[b.center[0], b.center[2]] for b in boxes])


def _matches_any(box: BinBox, centers: np.ndarray, thresh: float) -> bool:
    if not len(centers):
        return False
    c = np.array([box.center[0], box.center[2]])
    return float(np.min(np.linalg.norm(centers - c, axis=1))) < thresh


def _random_negative_boxes(
    cloud_points: np.ndarray, floor_height: float | None, gt_centers: np.ndarray,
    count: int, rng: np.random.Generator,
) -> list[BinBox]:
    xz_min = cloud_points[:, [0, 2]].min(axis=0)
    xz_max = cloud_points[:, [0, 2]].max(axis=0)
    y0 = floor_height if floor_height is not None else float(cloud_points[:, 1].min())
    boxes: list[BinBox] = []
    attempts = 0
    while len(boxes) < count and attempts < count * 25:
        attempts += 1
        cx = float(rng.uniform(xz_min[0], xz_max[0]))
        cz = float(rng.uniform(xz_min[1], xz_max[1]))
        if _matches_any(BinBox(center=[cx, y0, cz], extent=[0, 0, 0], yaw_deg=0.0), gt_centers, NEG_CLEAR_DIST):
            continue
        kind = int(rng.integers(0, 3))
        if kind == 0:      # wall/edge sliver
            ex, ez, ey = rng.uniform(0.05, 0.30), rng.uniform(0.4, 1.6), rng.uniform(0.4, 2.0)
        elif kind == 1:    # low floor clutter / debris
            ex, ez, ey = rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.6)
        else:              # bin-sized box but in an empty spot
            ex, ez, ey = rng.uniform(0.4, 1.6), rng.uniform(0.4, 1.0), rng.uniform(0.6, 1.6)
        box = BinBox(center=[cx, y0 + ey / 2, cz], extent=[float(ex), float(ey), float(ez)],
                     yaw_deg=float(rng.uniform(0, 360)))
        if len(crop_box(cloud_points, box.center, box.extent, box.local_axes())) < verify_bins.MIN_CROP_POINTS:
            continue  # empty air is not an informative negative; we want real geometry
        boxes.append(box)
    return boxes


def _crops_for_scene(stem: str, rng: np.random.Generator) -> list[dict]:
    """Return per-crop samples {points (M,3) normalized, extra (3,), label} for one scene."""
    annotation = ANNOTATION_DIR / f"{stem}.json"
    if not annotation.exists():
        return []
    cloud_points = _load_cloud(stem)
    if cloud_points is None or not len(cloud_points):
        return []
    floor_height, gt_boxes = load_annotations(annotation)
    gt_centers = _gt_centers(gt_boxes)

    def make(box: BinBox, label: int) -> dict | None:
        crop = crop_box(cloud_points, box.center, box.extent, box.local_axes())
        if len(crop) < verify_bins.MIN_CROP_POINTS:
            return None
        return {"points": normalize_points(crop, floor_height), "extra": box_extra(box),
                "label": label, "scene": stem}

    samples: list[dict] = []
    for box in gt_boxes:                                   # positives
        item = make(box, 1)
        if item:
            samples.append(item)

    proposals = CACHE_ROOT / stem / "proposals.json"       # hard negatives
    if proposals.exists():
        _, prop_boxes = load_annotations(proposals)
        for box in prop_boxes:
            if not _matches_any(box, gt_centers, GT_MATCH_DIST):
                item = make(box, 0)
                if item:
                    samples.append(item)

    for box in _random_negative_boxes(cloud_points, floor_height, gt_centers, RANDOM_NEG_PER_SCENE, rng):
        item = make(box, 0)
        if item:
            samples.append(item)
    return samples


class CropDataset(Dataset):
    def __init__(self, samples: list[dict], augment: bool, seed: int = 0) -> None:
        self.samples = samples
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, points: np.ndarray, extra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = points.copy()
        theta = self.rng.uniform(0, 2 * np.pi)              # yaw about the vertical axis
        cos, sin = np.cos(theta), np.sin(theta)
        x, z = points[:, 0].copy(), points[:, 2].copy()
        points[:, 0] = cos * x - sin * z
        points[:, 2] = sin * x + cos * z
        if self.rng.random() < 0.5:                          # mirror
            points[:, 0] *= -1
        scale = float(self.rng.uniform(0.95, 1.05))          # mild scale (size stays a signal)
        points *= scale
        points += self.rng.normal(0, 0.01, points.shape).astype(np.float32)  # jitter
        return points, extra * scale

    def __getitem__(self, index: int):
        sample = self.samples[index]
        points, extra = sample["points"], sample["extra"]
        if self.augment:
            points, extra = self._augment(points, extra)
        replace = len(points) < N_POINTS
        idx = self.rng.choice(len(points), N_POINTS, replace=replace)
        return (
            torch.from_numpy(points[idx]).float(),
            torch.from_numpy(np.asarray(extra, dtype=np.float32)),
            torch.tensor(float(sample["label"])),
        )


def _evaluate(net: BinVerifierNet, loader: DataLoader, device: str) -> dict:
    net.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for xyz, extra, label in loader:
            pred = (torch.sigmoid(net(xyz.to(device), extra.to(device))) >= 0.5).cpu()
            label = label.bool()
            pred = pred.bool()
            tp += int((pred & label).sum())
            fp += int((pred & ~label).sum())
            tn += int((~pred & ~label).sum())
            fn += int((~pred & label).sum())
    total = max(tp + fp + tn + fn, 1)
    return {
        "acc": (tp + tn) / total,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "neg_rejected": tn / max(tn + fp, 1),  # share of non-bins correctly dropped
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PointNet++ bin verifier.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.25, help="fraction of SCENES held out (0 = train on all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    scenes = sorted(p.stem for p in ANNOTATION_DIR.glob("*.json"))
    if not scenes:
        raise SystemExit(f"no annotations in {ANNOTATION_DIR} — annotate some scans first")

    rng.shuffle(scenes)
    n_val = int(len(scenes) * args.val_frac)
    val_scenes = set(scenes[:n_val])
    train_scenes = [s for s in scenes if s not in val_scenes]
    print(f"scenes: {len(scenes)} total -> {len(train_scenes)} train, {len(val_scenes)} val")

    train_samples, val_samples = [], []
    for stem in scenes:
        crops = _crops_for_scene(stem, rng)
        (val_samples if stem in val_scenes else train_samples).extend(crops)

    n_pos = sum(s["label"] == 1 for s in train_samples)
    n_neg = len(train_samples) - n_pos
    print(f"train crops: {len(train_samples)} ({n_pos} bins, {n_neg} non-bins);  val crops: {len(val_samples)}")
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("need both bins and non-bins in the training set — check annotations/proposals")

    train_loader = DataLoader(CropDataset(train_samples, augment=True, seed=args.seed),
                              batch_size=args.batch, shuffle=True, drop_last=len(train_samples) >= args.batch)
    val_loader = (
        DataLoader(CropDataset(val_samples, augment=False, seed=0), batch_size=args.batch)
        if val_samples else None
    )

    device = args.device
    net = BinVerifierNet(extra_dims=EXTRA_DIMS).to(device)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_metric, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        net.train()
        epoch_loss = 0.0
        for xyz, extra, label in train_loader:
            optimizer.zero_grad()
            loss = criterion(net(xyz.to(device), extra.to(device)), label.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * len(label)
        scheduler.step()
        epoch_loss /= max(len(train_samples), 1)

        if val_loader is not None:
            metrics = _evaluate(net, val_loader, device)
            # balance keeping real bins (recall) with dropping noise (neg_rejected)
            selection = metrics["recall"] + metrics["neg_rejected"]
            if epoch % 5 == 0 or epoch == args.epochs:
                print(f"epoch {epoch:3d}  loss {epoch_loss:.3f}  "
                      f"val acc {metrics['acc']:.2f}  recall {metrics['recall']:.2f}  "
                      f"neg-rejected {metrics['neg_rejected']:.2f}  (fp {metrics['fp']}, fn {metrics['fn']})")
            if selection > best_metric:
                best_metric = selection
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            if epoch % 5 == 0 or epoch == args.epochs:
                print(f"epoch {epoch:3d}  loss {epoch_loss:.3f}")
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    verify_bins.WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "extra_dims": EXTRA_DIMS, "n_points": N_POINTS},
               verify_bins.WEIGHTS_PATH)
    print(f"\nsaved verifier -> {verify_bins.WEIGHTS_PATH} (used automatically by prepare_scan)")
    if val_loader is not None:
        final = _evaluate(net, val_loader, device)
        print(f"final val: recall {final['recall']:.2f} (real bins kept), "
              f"neg-rejected {final['neg_rejected']:.2f} (noise dropped), acc {final['acc']:.2f}")


if __name__ == "__main__":
    main()
