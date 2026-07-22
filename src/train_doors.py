"""Train the door/entrance detector from your clicked entrances and publish doors_latest.pt.

For every scan where you have clicked at least one entrance, we rebuild the same geometry the
pipeline uses (footprint, free space, wall mask, camera path), generate candidate openings, and
label each candidate positive if it lands near a clicked entrance — negative otherwise. A small
MLP then learns which openings are real doors. Split BY SCENE so the score reflects new rooms.

    .venv\\Scripts\\python.exe -m src.train_doors            # hold out some scenes for validation
    .venv\\Scripts\\python.exe -m src.train_doors --val-frac 0   # train on all, save final model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from . import backbone, doors, freespace, placement, set_entrance
from .doors import FEATURE_NAMES, DoorNet, candidate_openings
from .loader import load_point_cloud
from .pipeline import CACHE_ROOT, RAW_DIR, load_existing_bins
from .reconstruct import ReconstructionConfig
from .scan_io import ScanArchive

MATCH_DIST = 1.0  # a candidate within this of a clicked entrance counts as that door (m)


def _scene_context(stem: str):
    """Rebuild (fs, footprint, wall_mask, camera_xz, rotation) exactly as the pipeline does."""
    zip_path = RAW_DIR / f"{stem}.zip"
    cache_cloud = CACHE_ROOT / stem / "cloud.ply"
    if not cache_cloud.exists() or not zip_path.exists():
        return None
    pcd, archive, _ = load_point_cloud(
        zip_path, str(cache_cloud), ReconstructionConfig(min_confidence=255, max_depth_m=5.0)
    )
    try:
        geometry, aligned = backbone.analyze(pcd)
    except Exception:
        archive.close()
        return None
    footprint = geometry.footprint
    rotation = geometry.rotation if geometry.rotation is not None else np.eye(3)
    fs = freespace.compute_free_space(aligned, geometry.floor_height_m, footprint)
    existing = load_existing_bins(stem, rotation)
    poisson = CACHE_ROOT / stem / "mesh_poisson.ply"
    if poisson.exists():
        wall_points = np.asarray(o3d.io.read_triangle_mesh(str(poisson)).vertices) @ rotation.T
    else:
        wall_points = np.asarray(aligned.points)
    wall_mask = placement.build_wall_mask(fs, wall_points, geometry.floor_height_m, existing)
    camera_world = np.array([archive.keyframe(ts).pose_cam_to_world[:3, 3] for ts in archive.timestamps])
    camera_xz = (camera_world @ rotation.T)[:, [0, 2]]
    archive.close()
    return fs, footprint, wall_mask, camera_xz, rotation


def _scene_samples(stem: str) -> tuple[list[dict], int, int]:
    """Return (samples, n_clicked_doors, n_doors_with_a_matching_candidate) for one scene."""
    clicked = set_entrance.load_entrances(stem)
    if not clicked:
        return [], 0, 0
    context = _scene_context(stem)
    if context is None:
        return [], 0, 0
    fs, footprint, wall_mask, camera_xz, rotation = context
    clicked3d = np.array([[x, 0.0, z] for x, z in clicked]) @ rotation.T
    doors_xz = clicked3d[:, [0, 2]]

    candidates = candidate_openings(fs, footprint, wall_mask, camera_xz)
    if not candidates:
        return [], len(doors_xz), 0
    cand_xz = np.array([c["center_xz"] for c in candidates])

    # Clean labels: mark only the SINGLE nearest candidate to each clicked door as positive, so
    # wall points beside a doorway stay negative instead of contaminating the door class.
    labels = np.zeros(len(candidates), dtype=int)
    door_hit = 0
    for dx, dz in doors_xz:
        dists = np.hypot(cand_xz[:, 0] - dx, cand_xz[:, 1] - dz)
        nearest = int(np.argmin(dists))
        if dists[nearest] < MATCH_DIST:
            labels[nearest] = 1
            door_hit += 1
    samples = [{"features": c["features"], "label": int(labels[i]), "scene": stem}
               for i, c in enumerate(candidates)]
    return samples, len(doors_xz), door_hit


def _evaluate(net: DoorNet, x: torch.Tensor, y: torch.Tensor) -> dict:
    net.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(net(x)) >= 0.5)
    y = y.bool()
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    tn = int((~pred & ~y).sum())
    fn = int((~pred & y).sum())
    total = max(tp + fp + tn + fn, 1)
    return {
        "acc": (tp + tn) / total,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "neg_rejected": tn / max(tn + fp, 1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the door/entrance detector.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--val-frac", type=float, default=0.25, help="fraction of SCENES held out (0 = all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    scenes = sorted(p.stem for p in set_entrance.ENTRANCE_DIR.glob("*.json"))
    scenes = [s for s in scenes if set_entrance.load_entrances(s)]
    if not scenes:
        raise SystemExit(f"no clicked entrances in {set_entrance.ENTRANCE_DIR} — click some doors first")

    print(f"building features for {len(scenes)} scene(s) with clicked doors ...")
    per_scene: dict[str, list[dict]] = {}
    doors_total = doors_covered = 0
    for stem in scenes:
        samples, n_doors, n_hit = _scene_samples(stem)
        doors_total += n_doors
        doors_covered += n_hit
        if samples:
            per_scene[stem] = samples
    print(f"clicked doors: {doors_total}, of which {doors_covered} have a matching candidate "
          f"(candidate-recall ceiling {100 * doors_covered / max(doors_total, 1):.0f}%)")

    usable = [s for s in scenes if s in per_scene]
    rng.shuffle(usable)
    n_val = int(len(usable) * args.val_frac)
    val_scenes = set(usable[:n_val])
    train_samples = [s for stem in usable if stem not in val_scenes for s in per_scene[stem]]
    val_samples = [s for stem in usable if stem in val_scenes for s in per_scene[stem]]

    n_pos = sum(s["label"] for s in train_samples)
    n_neg = len(train_samples) - n_pos
    print(f"train openings: {len(train_samples)} ({n_pos} doors, {n_neg} non-doors); val: {len(val_samples)}")
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("need both doors and non-doors in the training set")

    train_x = np.stack([s["features"] for s in train_samples])
    train_y = np.array([s["label"] for s in train_samples], dtype=np.float32)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-6

    device = args.device
    xt = torch.from_numpy((train_x - mean) / std).float().to(device)
    yt = torch.from_numpy(train_y).to(device)
    if val_samples:
        val_x = np.stack([s["features"] for s in val_samples])
        val_y = np.array([s["label"] for s in val_samples], dtype=np.float32)
        xv = torch.from_numpy((val_x - mean) / std).float().to(device)
        yv = torch.from_numpy(val_y).to(device)

    net = DoorNet(n_features=len(FEATURE_NAMES)).to(device)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-3)

    best_metric, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        net.train()
        optimizer.zero_grad()
        loss = criterion(net(xt), yt)
        loss.backward()
        optimizer.step()
        if val_samples:
            metrics = _evaluate(net, xv, yv)
            selection = metrics["recall"] + metrics["neg_rejected"]
            if selection > best_metric:
                best_metric = selection
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if epoch % 50 == 0 or epoch == args.epochs:
                print(f"epoch {epoch:3d}  loss {float(loss):.3f}  val recall {metrics['recall']:.2f} "
                      f"neg-rejected {metrics['neg_rejected']:.2f}  (fp {metrics['fp']}, fn {metrics['fn']})")
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if epoch % 50 == 0 or epoch == args.epochs:
                print(f"epoch {epoch:3d}  loss {float(loss):.3f}")

    doors.WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mean": mean, "std": std, "features": list(FEATURE_NAMES)},
               doors.WEIGHTS_PATH)
    print(f"\nsaved door detector -> {doors.WEIGHTS_PATH} (used automatically by the pipeline)")
    if val_samples:
        final = _evaluate(net, xv, yv)
        print(f"final val: recall {final['recall']:.2f} (doors found), "
              f"neg-rejected {final['neg_rejected']:.2f} (false openings dropped), acc {final['acc']:.2f}")


if __name__ == "__main__":
    main()
