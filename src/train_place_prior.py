"""Train the placement prior from YOUR annotated bins and publish place_prior.pt.

Positives: every annotated bin (its placement features) — where a bin actually stands. Negatives:
random free-floor spots away from any bin — where you did NOT put one. A small MLP learns the
difference, so the packer can rank new-bin spots to look like your real layout. Split BY SCENE so
the score reflects unseen rooms.

    .venv\\Scripts\\python.exe -m src.train_place_prior            # hold out some scenes
    .venv\\Scripts\\python.exe -m src.train_place_prior --val-frac 0   # train on all, final model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from . import place_prior
from .paths import ANNOTATION_DIR
from .pipeline import compute_scene
from .place_prior import FEATURE_NAMES, PlacePriorNet, features_at, scene_feature_maps

NEG_PER_SCENE = 12
NEG_CLEAR_M = 0.8   # a random negative must be at least this far from every bin


def _scene_samples(stem: str, rng: np.random.Generator) -> list[dict]:
    try:
        scene = compute_scene(stem, "4-hjuls container")
    except Exception:
        return []
    bins = [(float(b[0]), float(b[1])) for b in scene.existing]
    if not bins:
        return []
    fs = scene.fs
    dist_wall, clearance = scene_feature_maps(fs, scene.wall_mask)
    samples: list[dict] = []

    for i, (bx, bz) in enumerate(bins):  # positives (leave-one-out for the "other bins" feature)
        others = bins[:i] + bins[i + 1:]
        feat = features_at((bx, bz), fs, dist_wall, clearance, others, scene.entrances)
        samples.append({"features": feat, "label": 1, "scene": stem})

    rows, cols = np.where(fs.free)  # negatives: random free floor away from any bin
    if len(rows):
        order = rng.permutation(len(rows))
        added = 0
        for k in order:
            wx = fs.origin[0] + (cols[k] + 0.5) * fs.cell
            wz = fs.origin[1] + (rows[k] + 0.5) * fs.cell
            if min(np.hypot(wx - bx, wz - bz) for bx, bz in bins) < NEG_CLEAR_M:
                continue
            feat = features_at((wx, wz), fs, dist_wall, clearance, bins, scene.entrances)
            samples.append({"features": feat, "label": 0, "scene": stem})
            added += 1
            if added >= NEG_PER_SCENE:
                break
    return samples


def _evaluate(net: PlacePriorNet, x: torch.Tensor, y: torch.Tensor) -> dict:
    net.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(net(x)) >= 0.5)
    y = y.bool()
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    tn = int((~pred & ~y).sum()); fn = int((~pred & y).sum())
    total = max(tp + fp + tn + fn, 1)
    return {"acc": (tp + tn) / total, "recall": tp / max(tp + fn, 1),
            "neg_rejected": tn / max(tn + fp, 1), "fp": fp, "fn": fn}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the placement prior from annotated bins.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--val-frac", type=float, default=0.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    scenes = sorted(p.stem for p in ANNOTATION_DIR.glob("*.json"))
    if not scenes:
        raise SystemExit(f"no annotations in {ANNOTATION_DIR}")
    print(f"building placement features for {len(scenes)} annotated scene(s) …")
    per_scene: dict[str, list[dict]] = {}
    for stem in scenes:
        s = _scene_samples(stem, rng)
        if any(x["label"] == 1 for x in s):
            per_scene[stem] = s

    usable = list(per_scene)
    rng.shuffle(usable)
    n_val = int(len(usable) * args.val_frac)
    val_scenes = set(usable[:n_val])
    train = [x for stem in usable if stem not in val_scenes for x in per_scene[stem]]
    val = [x for stem in usable if stem in val_scenes for x in per_scene[stem]]

    n_pos = sum(x["label"] for x in train)
    n_neg = len(train) - n_pos
    print(f"scenes with bins: {len(usable)}  |  train {len(train)} ({n_pos} bins, {n_neg} non-bins)  |  val {len(val)}")
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("need both bins and non-bins to train")

    train_x = np.stack([x["features"] for x in train])
    train_y = np.array([x["label"] for x in train], dtype=np.float32)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-6

    device = args.device
    xt = torch.from_numpy((train_x - mean) / std).float().to(device)
    yt = torch.from_numpy(train_y).to(device)
    if val:
        vx = torch.from_numpy((np.stack([x["features"] for x in val]) - mean) / std).float().to(device)
        vy = torch.from_numpy(np.array([x["label"] for x in val], dtype=np.float32)).to(device)

    net = PlacePriorNet(n_features=len(FEATURE_NAMES)).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=device))
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-3)

    best_metric, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        net.train()
        optimizer.zero_grad()
        loss = criterion(net(xt), yt)
        loss.backward()
        optimizer.step()
        if val:
            m = _evaluate(net, vx, vy)
            selection = m["recall"] + m["neg_rejected"]
            if selection > best_metric:
                best_metric = selection
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if epoch % 50 == 0 or epoch == args.epochs:
                print(f"epoch {epoch:3d}  loss {float(loss):.3f}  val recall {m['recall']:.2f} "
                      f"neg-rejected {m['neg_rejected']:.2f}  (fp {m['fp']}, fn {m['fn']})")
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if epoch % 50 == 0 or epoch == args.epochs:
                print(f"epoch {epoch:3d}  loss {float(loss):.3f}")

    place_prior.WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mean": mean.tolist(), "std": std.tolist(),
                "features": list(FEATURE_NAMES)}, place_prior.WEIGHTS_PATH)
    print(f"\nsaved placement prior -> {place_prior.WEIGHTS_PATH} (used automatically by the packer)")
    if val:
        f = _evaluate(net, vx, vy)
        print(f"final val: recall {f['recall']:.2f} (real bin spots kept), "
              f"neg-rejected {f['neg_rejected']:.2f} (random spots rejected)")


if __name__ == "__main__":
    main()
