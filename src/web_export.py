"""Per-scan plan data for the browser view.

Why this exists as an export step rather than something the web server computes: compute_scene() reads
the point cloud, rebuilds the free-space grid and re-runs placement -- seconds per scan, and it needs
open3d. The browser only ever needs the RESULT, so the numbers are written once alongside the preview
PNGs (by analyze_and_render, the same pass that draws them) and the server just reads two small files.

Two files per scan:

  plan.json   geometry in METRES: footprint rect, existing bins, our proposals, entrances, the push
              path, plus the grid's origin/cell so pixel coordinates and world coordinates convert.
              ~5 KB.
  masks.png   the free-space grids packed as BIT FLAGS in a single 8-bit grey channel (see MASK_BITS).
              Not one grid per RGB channel and not RGBA: a browser that draws an RGBA PNG onto a
              canvas may premultiply, so the colour of a fully transparent pixel is not guaranteed to
              survive getImageData -- exactly the pixels a mask needs to read back. One opaque channel
              has no such rule. 5 cm cells, so a 20x11 m yard is 400x220 px and a few KB.

              The browser reads it once into an ImageData and does instant point lookups while a bin
              is being dragged, which is how the page can say "this corner is off the scanned floor"
              without a round trip and without a second copy of the placement rules in JavaScript.

The masks are the authority: the fine judgement (does this layout still leave a push corridor?) is a
graph search on the SAME grid, run in Python by web.py on drop. Nothing in the browser decides
whether a placement is legal.

One measured trap for whoever validates against these grids: DO NOT test a bin by its centre cell.
On Frydenlundgata 4B not one of the five existing bins has an occupied centre, and three of the five
have an unobserved one -- because a depth scan of a closed container registers its SIDES and never
sees through its middle, so the middle cell holds no points above ground. Their footprints are
29-43% occupied, which is the honest signal. Test the footprint, and expect a bin's own footprint to
be partly unobserved even when the bin is unquestionably there.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .annotations import BIN_TYPES

PLAN_NAME = "plan.json"
MASKS_NAME = "masks.png"
MESH_NAME = "room.glb"

# plan.json format version. The browser refuses a plan it does not understand rather than drawing a
# room with, say, the length and width silently swapped.
# 2: added "mesh" (the .glb for the 3D view) and "floor_y".
PLAN_VERSION = 2

# Triangles to keep in room.glb. The cache meshes are far heavier than a browser needs -- median 487k
# triangles across the 322 scans, up to 936k, which is ~12 MB of raw buffers for a 9 x 10 m room.
# 120k measures 2.25 MB and, just as usefully, lands under 65 536 vertices for every scan sampled, so
# the index buffer is 16-bit and is halved. Decimation is the slow part of the export at 1.3-4.1 s.
MESH_TARGET_TRIANGLES = 120_000

# Bit per grid in masks.png. Shipped in plan.json too, so the browser reads the bit positions from
# the data instead of hard-coding numbers that could drift from this table.
MASK_BITS: dict[str, int] = {
    "floor_observed": 1,    # the scanner saw this cell's ground at all
    "occupied": 2,          # something stands on it
    "free": 4,              # floor, seen, and clear
    "accessible": 8,        # free floor connected to where the scanner actually walked
    "corridor": 16,         # the push path: entrance -> around the existing bins
}


def corners_from_yaw(cx: float, cz: float, length: float, width: float,
                     yaw_deg: float) -> list[list[float]]:
    """Four (X, Z) corners for a bin, using the SAME convention as annotations.BinBox.

    BinBox.local_axes puts length along ux = (cos yaw, sin yaw) and width along uz = (-sin yaw,
    cos yaw). Any other choice here would draw every bin rotated in the browser while the desktop app
    drew it correctly, which reads as a bad scan rather than a bug.
    """
    yaw = math.radians(yaw_deg)
    cos, sin = math.cos(yaw), math.sin(yaw)
    half_l, half_w = length / 2.0, width / 2.0
    out = []
    for along, across in ((-half_l, -half_w), (half_l, -half_w), (half_l, half_w), (-half_l, half_w)):
        out.append([round(cx + along * cos - across * sin, 3),
                    round(cz + along * sin + across * cos, 3)])
    return out


def _from_cv_rect(rect) -> tuple[float, float, float, float, float]:
    """cv2.minAreaRect-style rect -> (cx, cz, length, width, yaw_deg) in BinBox's convention.

    NOT rect[2]: OpenCV's angle is its own thing (4.5+ returns it in (0, 90]), so reading it as a yaw
    puts bins at the wrong rotation. annotations.BinBox.from_min_area_rect already solved this by
    going through boxPoints and taking the first edge's direction; this repeats that derivation
    instead of inventing a second one.
    """
    points = cv2.boxPoints(rect)
    edge_along = points[1] - points[0]
    edge_across = points[2] - points[1]
    length = float(np.linalg.norm(edge_along))
    width = float(np.linalg.norm(edge_across))
    yaw_deg = math.degrees(math.atan2(float(edge_along[1]), float(edge_along[0])))
    cx, cz = points.mean(axis=0)
    return float(cx), float(cz), length, width, yaw_deg


def _bin_entry(cx: float, cz: float, length: float, width: float, yaw_deg: float,
               kind: str, source: str) -> dict:
    # Normalise so length is always the long side, turning the box 90 degrees to compensate -- the
    # same pairing pipeline.load_existing_bins needs, and for the same reason: swapping the sides
    # without turning moves the footprint. Without this the SAME bin type came out "1.37 x 0.78" in
    # one room and "0.78 x 1.37" in another, purely because cv2 picked a different first edge.
    if width > length:
        length, width, yaw_deg = width, length, yaw_deg + 90.0
    yaw_deg = ((yaw_deg + 180.0) % 360.0) - 180.0     # keep it in (-180, 180]
    return {
        "center": [round(float(cx), 3), round(float(cz), 3)],
        "length_m": round(float(length), 3),
        "width_m": round(float(width), 3),
        "yaw_deg": round(float(yaw_deg), 2),
        "type": kind,
        "source": source,          # "existing" (annotated / detected) or "proposed" (ours)
        "corners": corners_from_yaw(cx, cz, length, width, yaw_deg),
    }


def _path_polyline(result) -> list[list[float]]:
    """The push corridor's skeleton as world-space points.

    result.route is a boolean grid, not an ordered path, so this returns the cells it marks rather
    than pretending to know the order. The browser draws them as dots along the route, which is what
    the preview PNGs do too.
    """
    route = getattr(result, "route", None)
    if route is None or not np.any(route):
        return []
    rows, cols = np.nonzero(route)
    origin, cell = result.origin, result.cell
    # Half a cell so a point sits in the MIDDLE of the cell it came from, not on its corner.
    xs = origin[0] + (cols + 0.5) * cell
    zs = origin[1] + (rows + 0.5) * cell
    return [[round(float(x), 3), round(float(z), 3)] for x, z in zip(xs, zs)]


def _passage_width() -> float:
    """Short side of the biggest real bin -- how wide the push corridor has to be.

    Mirrors pipeline.compute_scene: the corridor is sized for the largest bin that will ever be
    wheeled down it, so a spot is not accepted just because a small dunk happens to fit.
    """
    real = ("4-hjuls container", "2-hjuls dunk")
    biggest = max(real, key=lambda name: BIN_TYPES[name][0] * BIN_TYPES[name][2])
    return min(BIN_TYPES[biggest][0], BIN_TYPES[biggest][2])


def _view_bounds(grid: dict, groups: list[list[list[float]]], margin: float = 0.45) -> dict:
    """World-space box the browser should frame, covering the grid AND everything drawn on it.

    Sizing the canvas to the grid alone clips the room outline: the grid spans the observed floor
    POINTS, while the room rect is their rotated min-area rect, whose axis-aligned bounding box is
    larger. On Frydenlundgata 4B the grid is 9.15 x 9.75 m and the rect's corners run from -1.3 to
    10.7 m along one axis -- two metres of outline off-canvas. (render.annotated_topdown has the same
    union for the same reason.)
    """
    cell, (ox, oz) = grid["cell"], grid["origin"]
    xs = [ox, ox + grid["cols"] * cell]
    zs = [oz, oz + grid["rows"] * cell]
    for points in groups:
        for x, z in points:
            xs.append(x)
            zs.append(z)
    return {
        "min": [round(min(xs) - margin, 3), round(min(zs) - margin, 3)],
        "max": [round(max(xs) + margin, 3), round(max(zs) + margin, 3)],
    }


def build_plan(scene) -> dict:
    """Everything the browser needs about one scan, in metres."""
    result = scene.result
    footprint = scene.footprint
    grid_shape = scene.fs.free.shape          # (rows = Z, cols = X)

    # pipeline drops the bin type on the way into placement (footprints are all it needs), so the
    # types are fetched back from the same source in the same order -- otherwise the browser labels
    # every existing bin identically and you cannot tell a molok from a dunk in the list.
    from . import pipeline
    types = pipeline.existing_bin_types(scene.stem)
    existing = [
        _bin_entry(cx, cz, length, width, yaw,
                   types[index] if index < len(types) else "ukjent", "existing")
        for index, (cx, cz, length, width, yaw) in enumerate(result.existing_bins)
    ]
    proposed = []
    for candidate in result.candidates:
        cx, cz, length, width, yaw = _from_cv_rect(candidate.rect)
        proposed.append(_bin_entry(cx, cz, length, width, yaw,
                                   candidate.bin_type or result.bin_type, "proposed"))
    bins = existing + proposed
    grid = {
        "cell": round(float(result.cell), 4),
        "origin": [round(float(result.origin[0]), 4), round(float(result.origin[1]), 4)],
        "cols": int(grid_shape[1]),
        "rows": int(grid_shape[0]),
    }
    room_corners = [[round(float(x), 3), round(float(z), 3)]
                    for x, z in cv2.boxPoints(footprint.rect)]
    entrances = [[round(float(x), 3), round(float(z), 3)] for x, z in result.entrances]
    return {
        "version": PLAN_VERSION,
        "scan": scene.stem,
        "address": scene.address,
        "grid": grid,
        "mask_bits": MASK_BITS,
        "view": _view_bounds(grid, [room_corners, entrances, *(b["corners"] for b in bins)]),
        # The corridor must fit the LARGEST bin's short side, not the placed one's -- same number
        # pipeline.compute_scene passes to find_placements, exported so the server's re-check cannot
        # quietly use a different width than the run that produced these proposals.
        "passage_width_m": round(float(_passage_width()), 3),
        "room": {
            "length_m": round(float(footprint.length_m), 2),
            "width_m": round(float(footprint.width_m), 2),
            "area_m2": round(float(footprint.area_m2), 1),
            "angle_deg": round(float(footprint.angle_deg), 2),
            "corners": room_corners,
            "indoor": bool(scene.geometry.is_indoor),
            "free_area_m2": round(float(scene.fs.free_area_m2), 1),
            "floor_height_m": round(float(scene.floor_height), 3),
            "enclosed": bool(scene.enclosed),
        },
        # every type the browser is allowed to place, so the size table cannot drift from annotations.py
        "bin_types": {name: {"length_m": spec[0], "height_m": spec[1], "width_m": spec[2]}
                      for name, spec in BIN_TYPES.items()},
        "bins": bins,
        "entrances": entrances,
        "entrance_source": "innesperret" if scene.enclosed else ("klikket" if scene.clicked else "auto"),
        "path": _path_polyline(result),
        # Height of the floor plane in the SAME frame as room.glb, so the 3D view can lay the floor
        # overlay and stand the bins on it. Kept at the top level next to the geometry it applies to,
        # not only inside "room", because every 3D drawing call needs it.
        "floor_y": round(float(scene.floor_height), 4),
    }


def build_masks(scene) -> Image.Image:
    """The grids packed as bit flags in one 8-bit channel -- see MASK_BITS and the module docstring."""
    fs = scene.fs
    result = scene.result
    packed = np.zeros(fs.free.shape, dtype=np.uint8)
    grids = {
        "floor_observed": fs.floor_observed,
        "occupied": fs.occupied,
        "free": fs.free,
        "accessible": getattr(result, "accessible", None),
        "corridor": getattr(result, "reachable", None),
    }
    for name, grid in grids.items():
        if grid is None:
            continue     # e.g. a sealed room has no corridor at all
        packed |= (np.asarray(grid).astype(bool).astype(np.uint8) * MASK_BITS[name])
    return Image.fromarray(packed, mode="L")


def write_mesh(scene, out_dir: Path) -> dict | None:
    """Write room.glb -- the scan mesh the 3D view draws. None when the scan has no usable mesh.

    scene.mesh is already rotated into the gravity-aligned frame by compute_scene, which is the frame
    plan.json's metres are in, so the .glb needs no transform of its own and a bin at (x, z) in the
    plan stands at (x, z) in the 3D view.
    """
    if getattr(scene, "mesh", None) is None or not scene.mesh.has_triangles():
        return None
    from . import glb
    return glb.write_open3d_mesh(out_dir / MESH_NAME, scene.mesh, MESH_TARGET_TRIANGLES)


def write(scene, out_dir: Path) -> dict:
    """Write plan.json + masks.png + room.glb for one computed scene. Returns the plan dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # The mesh is written BEFORE plan.json so the plan can record what actually landed on disk -- if
    # the export failed, the plan says mesh: null and the browser opens the 2D plan instead of asking
    # for a .glb that is not there.
    plan = build_plan(scene)
    plan["mesh"] = write_mesh(scene, out_dir)
    (out_dir / PLAN_NAME).write_text(json.dumps(plan, indent=1), encoding="utf-8")
    # optimize=True on a 3-colour image is nearly free and roughly halves it
    build_masks(scene).save(out_dir / MASKS_NAME, optimize=True)
    return plan


def main() -> None:
    """Write the browser's data for scans that do not have it yet.

    analyze_and_render produces it as a side effect of "Generer bilder", but only for scans generated
    since this export existed -- 136 of the 141 rendered scans had none, and the browser listed them as
    unopenable while the desktop app showed them fine. Re-running "Generer bilder" would redo the four
    offscreen renders too, and those are the expensive part: the analysis is 1.4 s per scan and the mesh
    another 1.3-4.1 s, so the whole set of 322 is about half an hour rather than a day.

    Writes ONLY plan.json, masks.png and room.glb. Existing PNGs and stats.json are left exactly as they
    are, so running this can never lose a rendered sheet.
    """
    import argparse
    import time

    from . import pipeline

    parser = argparse.ArgumentParser(
        description="Skriv plandata (plan.json + masks.png) slik at rommet kan åpnes i nettleseren")
    parser.add_argument("stems", nargs="*",
                        help="skann-ID-er; tomt = alle forberedte skann som mangler plandata")
    parser.add_argument("--force", action="store_true",
                        help="skriv på nytt selv om plandataene finnes")
    parser.add_argument("--previewed-only", action="store_true",
                        help="bare skann som alt har bilder (stats.json)")
    parser.add_argument("--bin-type", default="4-hjuls container",
                        help="kassetypen forslagene dimensjoneres etter")
    parser.add_argument("--limit", type=int, help="stopp etter N skann")
    args = parser.parse_args()

    stems = args.stems or pipeline.list_scans()
    todo = []
    for stem in stems:
        if not pipeline.is_prepared(stem):
            continue          # no point cloud yet: prepare_scan has to run first
        out = pipeline.preview_dir(stem)
        if args.previewed_only and not (out / "stats.json").exists():
            continue
        if (out / PLAN_NAME).exists() and not args.force:
            continue
        todo.append(stem)
    if args.limit:
        todo = todo[:args.limit]

    if not todo:
        print("[plandata] ingenting å gjøre — alle valgte skann har plandata "
              "(bruk --force for å skrive dem på nytt)")
        return

    # ~1.4 s for the analysis plus 1.3-4.1 s to decimate and write the mesh; 5 s per scan is the
    # measured middle. Printed up front because the whole set is half an hour and that should not be
    # a surprise halfway through.
    print(f"[plandata] {len(todo)} skann, ca. {len(todo) * 5 / 60:.0f} min")
    failed: list[tuple[str, str]] = []
    started = time.perf_counter()
    for index, stem in enumerate(todo, 1):
        try:
            scene = pipeline.compute_scene(stem, args.bin_type)
            plan = write(scene, pipeline.preview_dir(stem))
        except KeyboardInterrupt:
            # Stopping is normal for a minutes-long job: the scans already written keep their plan
            # data, and a re-run picks up exactly where this left off (they are skipped as done).
            print(f"\n[plandata] avbrutt etter {index - 1} av {len(todo)}")
            break
        except Exception as error:      # noqa: BLE001 - one bad scan must not stop the batch
            failed.append((stem, f"{type(error).__name__}: {error}"))
            print(f"[{index}/{len(todo)}] {stem}: FEIL — {type(error).__name__}: {error}")
            continue
        mesh = plan.get("mesh")
        print(f"[{index}/{len(todo)}] {plan.get('address') or stem} — "
              f"{len(plan['bins'])} kasser, {plan['room']['area_m2']} m², "
              + (f"{mesh['bytes'] / 1024**2:.1f} MB 3D" if mesh else "ingen 3D-modell"))

    print(f"[plandata] ferdig på {(time.perf_counter() - started) / 60:.1f} min")
    if failed:
        print(f"[plandata] {len(failed)} skann feilet:")
        for stem, message in failed:
            print(f"  {stem}: {message}")


if __name__ == "__main__":
    main()
