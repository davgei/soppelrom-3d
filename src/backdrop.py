"""Which cloud the two interactive 3D viewers draw BEHIND the boxes.

WHY THIS IS A SHARED MODULE
    place3d and annotate3d both want Polycam's own export as the backdrop, both must fall back to
    our own reconstruction when it cannot be trusted, and both must say on screen which one they
    are showing. Keeping the decision — and the Norwegian wording for it — in one place stops the
    two viewers from drifting apart. It also keeps the headless preview renderer honest by
    construction: pipeline.analyze_and_render never calls this module, so all 139 preview sheets
    keep coming from our own reconstruction whether or not a .ply exists.

TWO RULES THAT ARE NOT NEGOTIABLE
    1. THE GATE DECIDES. ply_align.AlignQuality.ok is required before a Polycam cloud is drawn.
       In annotate3d the user draws ground-truth boxes against the backdrop, so a misaligned cloud
       silently shifts the labels every model trains on. In place3d the same error would only be
       cosmetic, but a viewer that shows an unregistered cloud while its sibling refuses to is
       worse than useless for judging the registration — so both demand a passing gate, and both
       name the reason in the panel when it fails instead of showing something wrong quietly.
    2. CACHE ONLY. A registration costs 8-45 seconds. Running one while a scan opens would freeze
       the GUI, and switching scans with the arrow keys would become unusable. Nothing here ever
       computes a transform: an unregistered scan is reported as such, together with the command
       that fixes it.

THE FRAME (the part that is easy to get wrong)
    The Polycam cloud is scenery. It is moved INTO the frame the viewer already draws in and is
    never read back from — no floor height, no raycast, no box, no push-path is derived from it.
    place3d draws the gravity-aligned frame  -> pass gravity_rotation=scene.rotation.
    annotate3d draws mesh_poisson.ply as stored, and its boxes and entrances live in that frame
                                             -> pass gravity_rotation=None (the default).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from . import ply_align

# ~800k points: 2.5 px reads as a continuous surface without swallowing the box wireframes on top.
POINT_SIZE = 2.5

OURS_LABEL = "egen rekonstruksjon"

# The "dollhouse" crop. Our Poisson mesh is one-sided, so the viewers can cull its backfaces and
# look straight down into the room; a point cloud has no facing to cull, and its ceiling would hide
# the floor overlay, the boxes and the push-path from any camera above the room — which is where
# both viewers start. Keeping a fixed band above the floor is the point-cloud equivalent.
# A relative rule ("shave the top off") was tried first and does not work: a Polycam export reaches
# well past the room (80623 spans 4.9 m in height because the scan continues outside the door), so
# the top of the cloud is nowhere near the room's ceiling. Everything these viewers care about —
# bins are at most 1.3 m, a door 2.1 m — lives inside 2 m of the floor, so measure from the FLOOR.
_HEADROOM_M = 2.0
_MIN_KEPT_FRACTION = 0.20  # a crop that throws away almost everything is wrong: keep the full cloud


@dataclass
class Backdrop:
    """The Polycam option for one scan, plus the wording for the on-screen indicator.

    cloud       the aligned Polycam cloud, in the caller's frame, gate passed. None means "draw our
                own reconstruction" — for any reason, from "no export exists" to "gate rejected it".
    dollhouse   the same cloud cropped to _HEADROOM_M above the floor, so a camera above the room
                sees into it. Display only — the points are dropped, nothing is moved.
    quality     the registration's quality record when there is one, even when it was rejected, so
                the viewer can show the user what was wrong.
    """
    cloud: o3d.geometry.PointCloud | None
    dollhouse: o3d.geometry.PointCloud | None
    quality: ply_align.AlignQuality | None
    label: str
    reason: str

    @property
    def available(self) -> bool:
        """True only for a Polycam cloud that is registered AND passed the gate."""
        return self.cloud is not None


def _dollhouse(cloud: o3d.geometry.PointCloud,
               floor_height: float | None) -> o3d.geometry.PointCloud:
    points = np.asarray(cloud.points)
    if len(points) < 100:
        return cloud
    y = points[:, 1]
    # the pipeline's floor when the caller knows it; otherwise the cloud's own low percentile, which
    # is robust to the stray points that always sit below the floor
    floor = float(floor_height) if floor_height is not None else float(np.percentile(y, 1.0))
    keep = np.flatnonzero(y < floor + _HEADROOM_M)
    if len(keep) < _MIN_KEPT_FRACTION * len(points):
        return cloud
    return cloud.select_by_index(keep)


def load(stem: str, gravity_rotation: np.ndarray | None = None,
         floor_height: float | None = None) -> Backdrop:
    """Decide what to draw for one scan. Never computes a registration (see CACHE ONLY above).

    gravity_rotation  Scene.rotation for place3d, None for annotate3d — see THE FRAME above.
    floor_height      only used to bound the ceiling crop; the cloud itself is never measured.
    """
    if not ply_align.has_polycam(stem):
        return Backdrop(None, None, None, OURS_LABEL, "ingen Polycam-eksport for dette skannet")

    # cache read (~5 ms) purely to decide, so a missing or stale entry never triggers a 45 s ICP
    cached = ply_align.cached_transform(stem)
    if cached is None:
        return Backdrop(None, None, None, OURS_LABEL,
                        "Polycam-eksporten er ikke registrert ennå — kjør\n"
                        f"python -m src.ply_align --scan {stem}")
    _, quality = cached
    if not quality.ok:
        return Backdrop(None, None, quality, OURS_LABEL, f"Polycam-sky AVVIST: {quality.reason}")

    result = ply_align.aligned_polycam_cloud(stem, gravity_rotation=gravity_rotation,
                                             require_ok=True)
    if result is None:      # the .ply or the cache vanished between the two calls
        return Backdrop(None, None, quality, OURS_LABEL, "klarte ikke å laste Polycam-skyen")
    cloud, quality = result
    return Backdrop(cloud, _dollhouse(cloud, floor_height), quality,
                    polycam_label(quality), f"godkjent — {quality.reason}")


def polycam_label(quality: ply_align.AlignQuality) -> str:
    """The short indicator. The median residual is in it on purpose: it is the number that tells
    the user how far the thing they are drawing against can be from our own geometry."""
    return f"Polycam-sky, avvik {quality.residual_median * 100:.1f} cm"


def status_text(backdrop: Backdrop, showing_polycam: bool) -> str:
    """Two lines for the viewer panel: which cloud is on screen, and how good / why not."""
    if showing_polycam and backdrop.available:
        quality = backdrop.quality
        return (f"Bakgrunn: {backdrop.label}\n"
                f"p90 {quality.residual_p90 * 100:.0f} cm · overlapp {quality.overlap:.2f} · "
                f"skarphet {quality.sharpness:.2f}")
    if backdrop.available:
        return f"Bakgrunn: {OURS_LABEL}\n(Polycam-sky finnes, men er slått av)"
    return f"Bakgrunn: {OURS_LABEL}\n({backdrop.reason})"


def material(point_size: float = POINT_SIZE) -> "object":
    """Unlit material for the Polycam cloud: the export carries per-point colour, and lighting a
    point cloud only darkens it. Imported lazily so this module stays safe to import headlessly."""
    from open3d.visualization import rendering

    record = rendering.MaterialRecord()
    record.shader = "defaultUnlit"
    record.point_size = float(point_size)
    return record


def main() -> None:
    """Print the decision for one or more scans — the same one the viewers make, without a GUI."""
    import argparse

    parser = argparse.ArgumentParser(description="Hvilken 3D-bakgrunn viserne velger for et skann.")
    parser.add_argument("--scan", action="append", default=None, help="skann-stamme (kan gjentas)")
    parser.add_argument("--all", action="store_true", help="alle skann med .ply og rekonstruksjon")
    args = parser.parse_args()

    stems = args.scan or (ply_align.available_stems() if args.all else [])
    if not stems:
        raise SystemExit("bruk --scan <stamme> eller --all")
    for stem in stems:
        choice = load(stem)
        points = len(choice.cloud.points) if choice.available else 0
        print(f"{stem:32} {'POLYCAM' if choice.available else 'EGEN   '} "
              f"{points:>9} punkter  {choice.reason}")


if __name__ == "__main__":
    main()
