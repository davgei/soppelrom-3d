"""Interactive 3D placement viewer.

Browse scans and SEE, on the room mesh: the push-path a large bin can be wheeled along to a door
(blue floor), the free bin space around it (green), suggested new bins (green boxes), existing bins
(red boxes) and entrances (magenta). Read-only sibling of the annotation tool — same drag-to-orbit
controls and Prev/Next scan navigation.

BACKDROP: when Polycam's own export for the scan is registered and passes the quality gate it is
drawn INSTEAD of our reconstruction (it looks considerably better), toggled with B. Everything
else — bins, push-path, entrances, camera — is still computed from and positioned by the pipeline's
own geometry, so the swap is purely what you look at. See src/backdrop.py for the rules; the panel
always names the cloud on screen, and says why when Polycam is not used.

PANEL / SCALING: Open3D's gui has no reflowing layout manager — every size is a number the caller
computes — so every number here is a multiple of window.theme.font_size (an "em", via uitheme.em).
That is what makes the panel and its buttons follow the theme font and the display's DPI instead of
clipping at 150% scaling. The panel itself is a gui.ScrollableVert holding gui.CollapsableVert
sections: content that does not fit gets a scrollbar rather than being cut off, and a section the
user does not need can be folded away. Colours come from src/uitheme.py (which derives them from
src/style.py), so a green box means the same thing here as in the rendered preview PNGs.

    .venv\\Scripts\\python.exe -m src.place3d
"""
from __future__ import annotations

import math
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.ndimage import binary_closing, binary_dilation, label
from scipy.sparse.csgraph import dijkstra

from . import backdrop, pipeline, placement
from . import uitheme as T
from .annotations import BIN_TYPES


def _bin_box_lineset(rect: tuple, y_min: float, y_max: float, color) -> o3d.geometry.LineSet:
    """Wireframe box from a footprint rect (cv2.minAreaRect in X,Z) and a height range."""
    corners_xz = cv2.boxPoints(rect)
    corners = [[x, y_min, z] for x, z in corners_xz] + [[x, y_max, z] for x, z in corners_xz]
    edges = (
        [[i, (i + 1) % 4] for i in range(4)]
        + [[4 + i, 4 + (i + 1) % 4] for i in range(4)]
        + [[i, 4 + i] for i in range(4)]
    )
    lineset = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(np.array(corners, dtype=float)),
        o3d.utility.Vector2iVector(np.array(edges)),
    )
    lineset.paint_uniform_color(list(color))
    return lineset

# Scene semantics — the SAME colours the preview PNGs are drawn with (uitheme derives them from
# style.py). Do not hand-tune them here: that is exactly how "green" came to mean three different
# things in three windows. The proposal ghost used to be blue, which collided with the blue
# push-path it stands on; it is green now, like the proposed bins in every rendered sheet.
FREE_COLOR = T.rgb_of("free_floor")       # green   = free bin space
PATH_COLOR = T.rgb_of("path")             # blue    = push-path a large bin can reach a door through
ROUTE_COLOR = T.rgb_of("path_soft")       # brighter= the near-straight route the corridor is built on
OCC_COLOR = T.rgb_of("occupied_floor")    # red     = occupied floor
PROPOSAL_COLOR = T.rgb_of("new_bin")      # translucent GREEN ghost bin = suggested new bin
ENTRANCE_COLOR = T.rgb_of("entrance")     # magenta sphere = entrance
EXISTING_EDGE_COLOR = T.rgb_of("existing_bin")  # red wireframe around a bin that is already there

ANIM_SPEED_WALK = 1.05    # m/s walking empty-handed
ANIM_WHEEL_SPEED = {      # m/s wheeling a bin: big bins roll faster, a full bin is heavier
    ("stor", "full"): 0.63, ("stor", "tom"): 0.92,
    ("liten", "full"): 0.52, ("liten", "tom"): 0.80,
}
ANIM_EMPTY_SECONDS = 2.8  # duration of the tipping animation at the truck
ANIM_STRIDE_M = 1.0       # metres per full walk cycle (drives the leg swing)
# The worker, his bin and the truck are PROPS, not data: they are painted like the real objects
# (hi-vis orange, grey container / green dunk, green lorry) and deliberately stay outside the
# semantic palette, so no reader mistakes them for a measurement.
FIGURE_COLOR = (1.0, 0.45, 0.05)     # hi-vis orange — the renovation worker
HEAD_COLOR = (0.96, 0.83, 0.66)
BIN_HEIGHT = {"stor": 1.25, "liten": 1.15}  # real Norwegian bin heights (4-hjuls / 2-hjuls)
CLUTTER_COLOR = T.rgb_of("warning")         # amber floor = clutter blocking it (not bins, not walls)
BIN_BODY_COLOR = {"stor": (0.16, 0.17, 0.19), "liten": (0.47, 0.60, 0.38)}  # norsk: grå container, grønn dunk
BIN_TIP_DEG = {"stor": 45.0, "liten": 55.0}  # max tipping angle at the truck hopper
WHEEL_COLOR = (0.06, 0.06, 0.06)
TRASH_COLOR = (0.35, 0.28, 0.18)
TRUCK_BODY_COLOR = (0.15, 0.45, 0.23)   # green renovation truck
TRUCK_CAB_COLOR = (0.92, 0.92, 0.90)
TRUCK_DARK = (0.18, 0.18, 0.20)


def _leg(pts: np.ndarray, mode: str, size: str | None,
         dims: tuple[float, float, float] | None = None, bin_idx: int | None = None) -> dict | None:
    """A movement leg. mode: 'walk' (empty-handed), 'full' (wheeling a full bin to the truck),
    'tom' (wheeling the emptied bin back). dims/bin_idx identify the REAL bin being wheeled, so
    it can be drawn at true size and hidden from its parking spot while carried."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-3:
        return None
    return {"kind": "move", "pts": pts, "cum": np.concatenate([[0.0], np.cumsum(seg)]),
            "total": total, "mode": mode, "size": size, "dims": dims, "bin_idx": bin_idx}


def _walk_legs(scene) -> list[dict]:
    """The full collection round, nearest bin first, looping:
    walk empty-handed to the bin -> wheel the FULL bin to the truck (entrance) -> tipping pause
    (emptying) -> wheel the EMPTY bin back to its spot -> next bin -> finally walk out.

    Routing runs through the push-path corridor with every bin footprint BLOCKED, so the walker
    goes around the other bins instead of through them; the return legs reuse the same path in
    reverse (his own track). Falls back to the permissive corridor, then a straight segment, only
    when blocking the bins seals a target off."""
    if not scene.entrances or not scene.existing:
        return []
    start = np.array(scene.entrances[0], dtype=float)
    order = sorted(range(len(scene.existing)),
                   key=lambda k: (scene.existing[k][0] - start[0]) ** 2 + (scene.existing[k][1] - start[1]) ** 2)
    bins = [scene.existing[k] for k in order]
    sizes = ["stor" if max(b[2], b[3]) >= 1.0 else "liten" for b in bins]
    dims = [(float(b[2]), float(b[3]), BIN_HEIGHT[s]) for b, s in zip(bins, sizes)]
    centers = [np.array([b[0], b[1]], dtype=float) for b in bins]

    fs = scene.fs
    cell, origin = fs.cell, fs.origin
    to_bin: list[np.ndarray | None] = [None] * len(bins)     # entrance -> bin i
    between: list[np.ndarray | None] = [None] * len(bins)    # bin i-1 -> bin i
    bins_mask = placement._boxes_mask(bins, origin, cell, fs.free.shape, grow=0.05)

    def route_over(passable: np.ndarray) -> tuple[list, list]:
        """Shortest paths entrance->bin and bin->bin across `passable`, or Nones where unreachable."""
        outs: list[np.ndarray | None] = [None] * len(bins)
        betweens: list[np.ndarray | None] = [None] * len(bins)
        if passable is None or not passable.any():
            return outs, betweens
        graph, node_id, ys, xs = placement._grid_graph(passable, np.ones(passable.shape))

        def node_at(xz: np.ndarray) -> int:
            col = int(round((xz[0] - origin[0]) / cell))
            row = int(round((xz[1] - origin[1]) / cell))
            near = placement._nearest_true(passable, row, col)
            return int(node_id[near]) if near is not None else -1

        nodes = [node_at(start)] + [node_at(c) for c in centers]
        if any(n < 0 for n in nodes):
            return outs, betweens
        dist, pred = dijkstra(graph, directed=False, indices=nodes, return_predecessors=True)

        def path(k: int, t: int, anchor: np.ndarray | None) -> np.ndarray | None:
            """World polyline from source #k to node t (anchor prepended if given)."""
            if t < 0 or not np.isfinite(dist[k, t]):
                return None
            cells: list[tuple[int, int]] = []
            node, guard = t, 0
            while node >= 0 and guard < ys.size + 5:
                cells.append((int(ys[node]), int(xs[node])))
                node = int(pred[k, node])
                guard += 1
            cells.reverse()
            pts = np.array([[origin[0] + (c + 0.5) * cell, origin[1] + (r + 0.5) * cell]
                            for r, c in cells])
            if anchor is not None:
                pts = np.vstack([anchor[None, :], pts])
            if len(pts) > 4:  # light downsample, keeping the exact endpoints
                pts = np.vstack([pts[0], pts[1:-1:2], pts[-1]])
            return pts

        for i in range(len(bins)):
            outs[i] = path(0, nodes[i + 1], start)
            if i > 0:
                betweens[i] = path(i, nodes[i + 1], None)
        return outs, betweens

    # First choice: the push-path corridor with the bins blocked, so he walks the route a bin is
    # actually wheeled along and goes AROUND the other bins.
    reach = scene.result.reachable
    if reach is not None and reach.any():
        to_bin, between = route_over(reach & ~bins_mask)

    # Fallback: the free floor. The corridor is only as wide as the largest bin, so a bin standing in
    # a nook can sit outside it (measured: 1653 had 0 of 1 bins reachable through the corridor) — and
    # the old fallback was a STRAIGHT LINE from the entrance, which walked through walls and off the
    # floor entirely. Any walkable floor route is better than that.
    if any(p is None for p in to_bin) or any(between[i] is None for i in range(1, len(bins))):
        floor = fs.free & ~bins_mask
        if floor.any():
            floor = binary_closing(floor, iterations=max(1, int(0.15 / cell))) & fs.floor_observed
            alt_to, alt_between = route_over(floor & ~bins_mask)
            for i in range(len(bins)):
                if to_bin[i] is None:
                    to_bin[i] = alt_to[i]
                if i > 0 and between[i] is None:
                    between[i] = alt_between[i]

    # Last routed attempt: ANY observed floor, ignoring clutter. A bin wedged behind junk is still
    # reachable in real life (you shove the junk aside), and walking over clutter is far less wrong
    # than being dropped from the round — 76858 lost all 3 bins, i.e. the whole animation, when this
    # tier was missing. Still never leaves the scanned floor.
    if any(p is None for p in to_bin) or any(between[i] is None for i in range(1, len(bins))):
        rough = np.asarray(fs.floor_observed, dtype=bool) & ~bins_mask
        if rough.any():
            alt_to, alt_between = route_over(rough)
            for i in range(len(bins)):
                if to_bin[i] is None:
                    to_bin[i] = alt_to[i]
                if i > 0 and between[i] is None:
                    between[i] = alt_between[i]

    # Still unreachable = genuinely cut off. Drop that bin from the round instead of teleporting the
    # walker through a wall; a missing bin is honest, a straight line through the building is not.
    keep = [i for i in range(len(bins)) if to_bin[i] is not None]
    if len(keep) != len(bins):
        dropped = [bins[i] for i in range(len(bins)) if i not in keep]
        print(f"[henterunde] {len(dropped)} kasse(r) uten gangbar rute — utelatt fra animasjonen",
              flush=True)
        bins = [bins[i] for i in keep]
        sizes = [sizes[i] for i in keep]
        dims = [dims[i] for i in keep]
        centers = [centers[i] for i in keep]
        to_bin = [to_bin[i] for i in keep]
        between = [between[i] for i in keep]
    if not bins:
        return []
    # A bin can be reachable from the ENTRANCE but not from the PREVIOUS bin (blocked doorway, or the
    # previous bin was dropped). Walking out and in again is correct and always available, so use it —
    # leaving between[i] as None crashed _leg with a 0-d array.
    for i in range(1, len(bins)):
        if between[i] is None:
            between[i] = to_bin[i]

    legs: list[dict] = []

    def add(leg: dict | None) -> None:
        if leg is not None:
            legs.append(leg)

    for i in range(len(bins)):
        add(_leg(to_bin[i] if i == 0 else between[i], "walk", sizes[i]))
        add(_leg(to_bin[i][::-1].copy(), "full", sizes[i], dims[i], order[i]))
        # He must tip the bin TOWARD THE TRUCK, not back the way he came. The truck is anchored just
        # outside the dump spot, so its direction is what the figure and the bin should face; using the
        # arrival heading made him tip into thin air whenever the two differed.
        arrive_dir = (to_bin[i][0] - to_bin[i][1]) if len(to_bin[i]) > 1 else np.array([1.0, 0.0])
        truck_pos, _ = _truck_anchor(scene, to_bin[i][0], arrive_dir)
        dump_dir = np.asarray(truck_pos, dtype=float) - np.asarray(to_bin[i][0], dtype=float)
        if float(np.hypot(*dump_dir)) < 1e-6:
            dump_dir = arrive_dir
        legs.append({"kind": "pause", "total": ANIM_EMPTY_SECONDS, "pos": to_bin[i][0],
                     "dir": dump_dir, "mode": "dump", "size": sizes[i],
                     "dims": dims[i], "bin_idx": order[i]})
        add(_leg(to_bin[i].copy(), "tom", sizes[i], dims[i], order[i]))
    add(_leg(to_bin[-1][::-1].copy(), "walk", None))  # walk back out empty-handed
    return legs


def _point_at(leg: dict, s: float) -> np.ndarray:
    s = float(np.clip(s, 0.0, leg["total"]))
    cum, pts = leg["cum"], leg["pts"]
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = max(0, min(i, len(pts) - 2))
    span = cum[i + 1] - cum[i]
    frac = (s - cum[i]) / span if span > 1e-9 else 0.0
    return pts[i] * (1 - frac) + pts[i + 1] * frac


def _limb(a, b, radius: float) -> o3d.geometry.TriangleMesh:
    """A cylinder between two 3D points (Open3D's cylinder is built along Z; rotate it there)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v = b - a
    length = max(float(np.linalg.norm(v)), 1e-6)
    limb = o3d.geometry.TriangleMesh.create_cylinder(radius, length, resolution=8, split=1)
    z = np.array([0.0, 0.0, 1.0])
    d = v / length
    axis = np.cross(z, d)
    s = float(np.linalg.norm(axis))
    if s > 1e-8:
        angle = math.atan2(s, float(z @ d))
        limb.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis / s * angle), center=(0, 0, 0))
    elif float(z @ d) < 0:
        limb.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([math.pi, 0.0, 0.0]), center=(0, 0, 0))
    limb.translate((a + b) / 2)
    return limb


def _capsule(a, b, radius: float, color) -> o3d.geometry.TriangleMesh:
    """A rounded limb: cylinder with sphere caps — the chunky, soft game-character look."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    part = _limb(a, b, radius)
    for end in (a, b):
        cap = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=8)
        cap.translate(end)
        part += cap
    part.paint_uniform_color(list(color))
    return part


def _figure_mesh(pos: np.ndarray, direction: np.ndarray, floor: float,
                 phase: float, mode: str) -> o3d.geometry.TriangleMesh:
    """A chunky, rounded little worker (Human Fall Flat-style): fat capsule body, big head right
    on the shoulders, stubby swinging arms and legs, and a slight bob while walking. When wheeling
    ('full'/'tom') the hands hold the bin ahead; while dumping ('dump') he stands still."""
    d = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(d))
    d = d / norm if norm > 1e-6 else np.array([1.0, 0.0])
    p = np.array([-d[1], d[0]])
    x, z = float(pos[0]), float(pos[1])
    moving = mode in ("walk", "full", "tom")
    bob = 0.025 * abs(math.sin(phase)) if moving else 0.0
    hip_y = floor + 0.52 + bob
    shoulder_y = floor + 1.08 + bob
    head_y = floor + 1.34 + bob
    stride = {"walk": 0.34, "full": 0.22, "tom": 0.28}.get(mode, 0.0)

    def at(offset2: np.ndarray, y: float) -> np.ndarray:
        return np.array([x + offset2[0], y, z + offset2[1]])

    parts: list[o3d.geometry.TriangleMesh] = []
    for side, leg_phase in ((1.0, 0.0), (-1.0, math.pi)):
        hip = at(p * 0.10 * side, hip_y)
        foot = at(p * 0.10 * side + d * stride * math.sin(phase + leg_phase), floor + 0.06)
        parts.append(_capsule(hip, foot, 0.075, FIGURE_COLOR))
    parts.append(_capsule(at(p * 0.0, hip_y), at(p * 0.0, shoulder_y), 0.17, FIGURE_COLOR))  # body
    for side, arm_phase in ((1.0, math.pi), (-1.0, 0.0)):
        shoulder = at(p * 0.20 * side, shoulder_y)
        if mode in ("full", "tom", "dump"):  # both hands forward on the bin handle
            hand = at(d * 0.40 + p * 0.11 * side, floor + 0.95 + bob)
        else:
            hand = at(p * 0.20 * side + d * 0.26 * math.sin(phase + arm_phase), floor + 0.62 + bob)
        parts.append(_capsule(shoulder, hand, 0.055, FIGURE_COLOR))
    head = o3d.geometry.TriangleMesh.create_sphere(0.155, resolution=12)
    head.translate([x, head_y + 0.10, z])
    head.paint_uniform_color(list(HEAD_COLOR))
    parts.append(head)
    figure = parts[0]
    for part in parts[1:]:
        figure += part
    figure.compute_vertex_normals()
    return figure


def _frame_transform(pos: np.ndarray, direction: np.ndarray, floor: float) -> np.ndarray:
    """4x4 mapping local coords (x = travel direction, y = up, z = side) into the world."""
    d = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(d))
    d = d / norm if norm > 1e-6 else np.array([1.0, 0.0])
    transform = np.eye(4)
    transform[:3, 0] = [d[0], 0.0, d[1]]
    transform[:3, 2] = [-d[1], 0.0, d[0]]
    transform[:3, 3] = [float(pos[0]), floor, float(pos[1])]
    return transform


def _box(x0: float, x1: float, y0: float, y1: float, half_w: float, color) -> o3d.geometry.TriangleMesh:
    box = o3d.geometry.TriangleMesh.create_box(x1 - x0, y1 - y0, 2 * half_w)
    box.translate([x0, y0, -half_w])
    box.paint_uniform_color(list(color))
    return box


def _wheel(x: float, z: float, radius: float, width: float = 0.12) -> o3d.geometry.TriangleMesh:
    wheel = o3d.geometry.TriangleMesh.create_cylinder(radius, width, resolution=12, split=1)
    wheel.translate([x, radius, z])  # cylinder axis is already local Z = the side axis
    wheel.paint_uniform_color(list(WHEEL_COLOR))
    return wheel


def _bin_model_local(length: float, width: float, height: float, cls: str,
                     lid_open: bool, paint: bool = True) -> o3d.geometry.TriangleMesh:
    """A Norwegian wheelie bin in local coords (x = along its length, y = up, z = side), centered
    at the origin: tapered body (green dunk / dark-grey container), hinged lid, black wheels
    (2 rear on the small class, 4 corners on the container). paint=False leaves the mesh
    uncolored so a translucent ghost material can tint the whole bin."""
    body_color = BIN_BODY_COLOR[cls]
    lid_color = tuple(c * 0.72 for c in body_color)
    base_y = 0.10 if cls == "liten" else 0.13
    body_top = height * 0.88
    taper = 0.78

    lb, wb = length * taper / 2, width * taper / 2
    lt, wt = length / 2, width / 2
    lo = [[-lb, base_y, -wb], [lb, base_y, -wb], [lb, base_y, wb], [-lb, base_y, wb]]
    hi = [[-lt, body_top, -wt], [lt, body_top, -wt], [lt, body_top, wt], [-lt, body_top, wt]]
    tris = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    body = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.array(lo + hi)),
                                     o3d.utility.Vector3iVector(np.array(tris)))
    body.paint_uniform_color(list(body_color))

    lid = _box(-lt * 1.06, lt * 1.06, body_top, height, wt * 1.06, lid_color)
    if lid_open:  # hinged at the rear top edge, flipped up and back
        hinge = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, 0.0, math.radians(105)])
        lid.rotate(hinge, center=(-lt, body_top, 0.0))

    parts = body + lid
    if cls == "liten":
        for side in (-1, 1):
            parts += _wheel(-lt + 0.06, side * (wt - 0.04), 0.10)
    else:
        for ux in (-lt + 0.10, lt - 0.10):
            for side in (-1, 1):
                parts += _wheel(ux, side * (wt - 0.05), 0.09)
    if not paint:
        parts.vertex_colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
    return parts


def _wheeled_bin_mesh(pos: np.ndarray, direction: np.ndarray, floor: float,
                      dims: tuple[float, float, float], cls: str,
                      tip_deg: float = 0.0) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """The bin in front of the figure, sized to the REAL bin being collected (dims from the
    annotation). The lid stays closed in transport — both ways — and swings open only while the
    bin is tipped at the truck (tip_deg > 25). Returns (mesh, world mouth position)."""
    length, width, height = dims
    parts = _bin_model_local(length, width, height, cls, lid_open=tip_deg > 25.0)
    lt = length / 2
    tip = np.eye(4)  # tip forward about the front-bottom edge
    theta = math.radians(tip_deg)
    tip[:2, :2] = [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
    tip[:3, 3] = [lt - (lt * math.cos(theta)), lt * math.sin(theta), 0.0]
    center = np.array([float(pos[0]), float(pos[1])]) + _unit2(direction) * (0.42 + length / 2)
    transform = _frame_transform(center, direction, floor) @ tip
    parts.transform(transform)
    parts.compute_vertex_normals()
    mouth = (transform @ np.array([lt * 0.9, height * 0.88, 0.0, 1.0]))[:3]
    return parts, mouth


def _bin_model_world(cx: float, cz: float, length: float, width: float, yaw_deg: float,
                     floor: float, cls: str, height: float | None = None,
                     ghost: bool = False) -> o3d.geometry.TriangleMesh:
    """A parked bin at its real position/size/orientation. ghost=True builds it unpainted for a
    translucent proposal material."""
    h = height if height else BIN_HEIGHT[cls]
    parts = _bin_model_local(length, width, h, cls, lid_open=False, paint=not ghost)
    yaw = math.radians(yaw_deg)
    parts.transform(_frame_transform(np.array([cx, cz]), np.array([math.cos(yaw), math.sin(yaw)]), floor))
    parts.compute_vertex_normals()
    return parts


def _unit2(direction: np.ndarray) -> np.ndarray:
    d = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(d))
    return d / norm if norm > 1e-6 else np.array([1.0, 0.0])


# The truck mesh already starts 1.55 m from its anchor and runs outwards to 5.45 m, so anchoring
# almost ON the dump point puts the hopper right beside it with the body pointing away — which is how
# a lorry actually stands. Marching out past the room boundary first (an earlier attempt) parked it
# absurdly far away; what matters is the DIRECTION being away from the cloud, not the distance.
TRUCK_STANDOFF_M = 0.3


def _truck_anchor(scene, dump_pos, arrive_dir) -> tuple[np.ndarray, np.ndarray]:
    """Where to park the truck: OUTSIDE the scanned room, hopper facing back toward the dump spot.

    The dump spot is the entrance, which sits ON the scanned floor, so parking the truck there put a
    5.5 m lorry inside the point cloud, straddling the bins. A real truck stands out on the street, so
    march outwards from the entrance until clear of the room footprint, then add a little more.
    Direction: the walker's arrival heading when that already points out of the room, otherwise
    straight out from the room centre (the door model is not reliable enough to trust blindly)."""
    footprint = scene.footprint
    mask = np.asarray(footprint.mask, dtype=bool)
    origin, cell = footprint.origin, footprint.cell
    rows, cols = mask.shape
    centre = np.array(footprint.center_xz, dtype=float)
    start = np.array([float(dump_pos[0]), float(dump_pos[1])])

    outward = start - centre
    norm = float(np.hypot(*outward))
    outward = outward / norm if norm > 1e-6 else np.array([1.0, 0.0])
    arrive = np.asarray(arrive_dir, dtype=float).ravel()[:2]
    if float(np.hypot(*arrive)) > 1e-6:
        arrive = arrive / float(np.hypot(*arrive))
        if float(arrive @ outward) > 0.2:     # the walk already leaves the room — keep that heading
            outward = arrive

    return start + outward * TRUCK_STANDOFF_M, outward


def _truck_mesh(pos: np.ndarray, direction_out: np.ndarray, floor: float) -> o3d.geometry.TriangleMesh:
    """A simple renovation truck parked outside the entrance, rear hopper toward the dump spot:
    dark open hopper, green body, white cab, black wheels."""
    parts = _box(1.55, 2.85, 0.30, 1.05, 0.58, TRUCK_DARK)          # rear hopper (the bin tips in here)
    parts += _box(2.85, 4.65, 0.35, 1.80, 0.65, TRUCK_BODY_COLOR)   # cargo body
    parts += _box(4.65, 5.45, 0.32, 1.35, 0.62, TRUCK_CAB_COLOR)    # cab
    for x in (3.20, 4.90):
        for side in (-1, 1):
            parts += _wheel(x, side * 0.62, 0.30, width=0.18)
    parts.transform(_frame_transform(pos, direction_out, floor))
    parts.compute_vertex_normals()
    return parts


def _trash_bits_mesh(tau: float, mouth: np.ndarray, direction: np.ndarray,
                     floor: float) -> o3d.geometry.TriangleMesh | None:
    """Little trash lumps arcing out of the tipped bin into the truck hopper."""
    d = _unit2(direction)
    p = np.array([-d[1], d[0]])
    bits = None
    for j in range(6):
        t = tau - (0.30 + j * 0.05)
        if t <= 0:
            continue
        side = ((j * 0.37) % 1.0 - 0.5) * 0.20
        fx = mouth[[0, 2]] + d * (0.05 + 1.2 * t) + p * side
        y = float(mouth[1]) + 0.03 - 2.6 * t * t
        if y < floor + 0.42:  # swallowed by the hopper
            continue
        bit = o3d.geometry.TriangleMesh.create_sphere(0.035, resolution=6)
        bit.translate([float(fx[0]), y, float(fx[1])])
        bits = bit if bits is None else bits + bit
    if bits is not None:
        bits.compute_vertex_normals()
        bits.paint_uniform_color(list(TRASH_COLOR))
    return bits


# ---------------------------------------------------------------- panel geometry (everything in em)

# The panel asks for a share of the window and is then clamped: never narrower than the longest
# checkbox label needs, never wider than it can use, so a 4K window spends its extra pixels on the
# 3D view. Expressed in em, so the clamp follows the theme font / DPI instead of fighting it.
PANEL_MIN_EM = 20.0
PANEL_MAX_EM = 27.0
PANEL_FRACTION = 0.26     # wanted share of the window width
PANEL_MAX_SHARE = 0.55    # hard cap: a very narrow window must still show some scene


def _panel_width(content_width: int, font_size: int) -> int:
    """Panel width in px for a window `content_width` px wide at theme `font_size` px."""
    low = int(round(PANEL_MIN_EM * font_size))
    high = int(round(PANEL_MAX_EM * font_size))
    width = max(low, min(high, int(round(content_width * PANEL_FRACTION))))
    width = min(width, max(int(content_width * PANEL_MAX_SHARE), 1))
    return max(1, min(width, max(content_width - 1, 1)))


def _split_frame(x: int, y: int, width: int, height: int,
                 font_size: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Split a window rect into (scene, panel) rects as (x, y, w, h).

    Pure arithmetic on purpose: the GUI cannot be clicked from a test, but this can be called with
    any synthetic window size to prove that neither frame ever comes out zero or negative.
    """
    panel_w = _panel_width(width, font_size)
    scene_w = max(width - panel_w, 1)
    height = max(height, 1)
    return (x, y, scene_w, height), (x + scene_w, y, panel_w, height)


# ---------------------------------------------------------------- fonts

# Points, not pixels: Open3D multiplies a font's point size by the window's DPI scaling, so a
# 17-point heading keeps its 1.06x ratio to the 16-point body font on a 100% and a 200% display
# alike. Sizes must be registered before the first window exists (the atlas is built there).
_HEADING_POINTS = 17
_SMALL_POINTS = 13
# Glyphs outside Latin-1 that the panel may use. Open3D only rasterises the ranges it is told about.
_EXTRA_CODE_POINTS = [0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2192, 0x25A0]
_SEMIBOLD_FILES = ("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")


def _semibold_font_path(regular: str) -> str:
    """A heavier face from the same font directory as `regular`, for section headings. Falls back to
    the regular face, so a machine without Segoe UI Semibold just gets a slightly larger heading."""
    directory = Path(regular).parent
    for name in _SEMIBOLD_FILES:
        candidate = directory / name
        if candidate.exists():
            return str(candidate)
    return regular


def _register_fonts() -> dict[str, int]:
    """Segoe UI (so æ/ø/å render) plus a heading and a small size, as {name: font id}.

    Returns {} when no TTF is available or Open3D refuses the file; callers then fall back to
    DEFAULT_FONT_ID and the panel simply has one text size instead of three.
    """
    app = gui.Application.instance
    path = T.o3d_font_path()
    if not path:
        return {}
    ids: dict[str, int] = {}
    try:
        base = gui.FontDescription(path)
        try:
            base.add_typeface_for_code_points(path, _EXTRA_CODE_POINTS)
        except Exception:  # noqa: BLE001 - extra glyphs are a nicety, the base font is not
            pass
        app.set_font(gui.Application.DEFAULT_FONT_ID, base)
        ids["heading"] = app.add_font(
            gui.FontDescription(_semibold_font_path(path), point_size=_HEADING_POINTS))
        ids["small"] = app.add_font(gui.FontDescription(path, point_size=_SMALL_POINTS))
    except Exception:  # noqa: BLE001 - never let a font problem stop the viewer from opening
        return ids
    return ids


class PlacementViewer:
    def __init__(self, bin_type: str = "4-hjuls container") -> None:
        self.scans = [s for s in pipeline.list_scans() if pipeline.is_prepared(s)]
        if not self.scans:
            raise SystemExit("no prepared scans to show — prepare some first")
        self.index = 0
        self.bin_type = bin_type if bin_type in BIN_TYPES else "4-hjuls container"
        # backdrop preference: remembered across scans, but only honoured where the gate passes
        self.use_polycam = True
        self._backdrop: backdrop.Backdrop | None = None
        self._ours_name = "mesh"   # geometry name our own reconstruction was added under

        gui.Application.instance.initialize()
        # Fonts must be registered between initialize() and the first create_window(): Open3D builds
        # its glyph atlas when the window is created and cannot grow it afterwards.
        self._fonts = _register_fonts()
        self.window = gui.Application.instance.create_window("Søppelrom 3D — plassering & skyve-sti", 1500, 950)
        window = self.window

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(window.renderer)
        self.scene.scene.set_background(T.rgba("scene_bg"))  # "nothing scanned here", not black
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.scene.set_on_key(self._on_key)
        self.scene.set_on_mouse(self._on_mouse)  # custom turntable orbit, like the annotation tool
        window.add_child(self.scene)

        self._cor = np.zeros(3)   # centre of rotation (level-horizon turntable)
        self.orbit: dict | None = None
        self.pan: dict | None = None

        # ScrollableVert, not Vert: the old panel laid its controls out past the bottom of a small
        # window and the last checkboxes were simply unreachable. Now they scroll.
        self.panel = gui.ScrollableVert(T.emf(window, 0.3), T.margins(window, 0.6))
        self.panel.background_color = T.gui_color("panel_bg")
        self._build_scan_section()
        self._build_view_section()
        self._build_backdrop_section()
        self._build_round_section()
        self._build_legend_section()
        self._build_help_section()
        window.add_child(self.panel)
        window.set_on_layout(self._on_layout)

        self._anim: dict | None = None
        self._leg_idx = 0
        self._leg_pos = 0.0        # metres travelled along the current leg
        self._anim_time = 0.0      # simulated seconds this round (shown on the clock)
        self._walked = 0.0         # metres walked in total (drives the leg-swing phase)
        self._last_tick: float | None = None
        self._mover_name = "sti_mover"
        self._bin_name = "sti_bin"
        self._trash_name = "sti_trash"
        self._n_exist = 0
        self._n_cands = 0
        self._clutter_m2 = 0.0
        self._cam: np.ndarray | None = None  # smoothed follow-camera state (eye+target)
        self.window.set_on_tick_event(self._tick)
        self._load()

    # ---------- panel widgets (every size in em, so the panel follows font size / DPI) ----------

    def _font(self, step: str = "body") -> int:
        """Font id for 'heading' / 'small'; the theme default for anything else (incl. failure)."""
        return self._fonts.get(step, gui.Application.DEFAULT_FONT_ID)

    def _label(self, text: str, role: str = "text", step: str = "body") -> gui.Label:
        label = gui.Label(text)
        label.text_color = T.gui_color(role)
        label.font_id = self._font(step)
        return label

    def _section(self, title: str, expanded: bool = True) -> gui.CollapsableVert:
        """A labelled, foldable group added to the panel. Folding is the escape valve on a short
        window: the user hides what they are not using instead of hunting for a clipped control."""
        section = gui.CollapsableVert(title, T.emf(self.window, 0.3),
                                      T.margins(self.window, left=0.7, top=0.15, bottom=0.25))
        section.font_id = self._font("heading")
        section.set_is_open(expanded)
        self.panel.add_child(section)
        return section

    def _button(self, text: str, on_clicked, tooltip: str = "", primary: bool = False) -> gui.Button:
        button = gui.Button(text)
        # padding in em (Open3D's own unit for buttons) — the button grows with the font, so a row of
        # buttons keeps the same proportion of the panel width at any DPI
        button.horizontal_padding_em = 0.7
        button.vertical_padding_em = 0.3
        button.background_color = T.gui_color("accent" if primary else "panel_bg_alt")
        button.set_on_clicked(on_clicked)
        if tooltip:
            button.tooltip = tooltip
        return button

    def _checkbox(self, text: str, checked: bool = False, on_checked=None,
                  tooltip: str = "") -> gui.Checkbox:
        box = gui.Checkbox(text)
        box.checked = checked
        if on_checked is not None:
            box.set_on_checked(on_checked)
        if tooltip:
            box.tooltip = tooltip
        return box

    def _swatch(self, role: str):
        """A solid colour chip for the legend, sized in em.

        An ImageWidget is used because Open3D only actually paints background_color for a few widget
        types (a Label's is ignored), and a chip must show the EXACT semantic colour — describing it
        in words is what let the old legend claim "oransje" while the scene drew amber.
        """
        side = T.em(self.window, 0.85)
        try:
            pixels = np.empty((side, side, 3), dtype=np.uint8)
            pixels[:] = np.array(T.rgb255_of(role), dtype=np.uint8)
            return gui.ImageWidget(o3d.geometry.Image(pixels))
        except Exception:  # noqa: BLE001 - fall back to a coloured square glyph
            return self._label("■", role, "small")

    def _legend_row(self, parent: gui.Widget, role: str, text: str) -> None:
        row = gui.Horiz(T.emf(self.window, 0.5))
        row.add_child(self._swatch(role))
        row.add_child(self._label(text, "text_muted", "small"))
        row.add_stretch()
        parent.add_child(row)

    def _wrap(self, text: str, chars: int = 0) -> str:
        """Hard-wrap text: Open3D labels never wrap, they just run out of the panel.

        The default width is what the narrowest panel can show — Segoe UI averages about half an em
        per character, and the panel's inner width is PANEL_MIN_EM minus the margins.
        """
        width = chars or max(16, int((PANEL_MIN_EM - 2.4) * 1.9))
        return "\n".join(textwrap.fill(line, width) if line.strip() else line
                         for line in str(text).splitlines())

    # ---------- panel sections ----------

    def _build_scan_section(self) -> None:
        section = self._section("Skann")
        self.scan_pos_label = self._label("", "text_muted", "small")
        section.add_child(self.scan_pos_label)
        self.scan_label = self._label("", "text", "heading")
        section.add_child(self.scan_label)

        nav = gui.Horiz(T.emf(self.window, 0.4))
        nav.add_child(self._button("< Forrige", lambda: self._step(-1),
                                   "Forrige skann (pil venstre)", primary=True))
        nav.add_child(self._button("Neste >", lambda: self._step(1),
                                   "Neste skann (pil høyre)", primary=True))
        nav.add_stretch()   # buttons keep their size, the leftover width stays empty
        section.add_child(nav)

        self.type_combo = gui.Combobox()
        for name in BIN_TYPES:
            self.type_combo.add_item(name)
        self.type_combo.selected_text = self.bin_type
        self.type_combo.set_on_selection_changed(lambda text, _index: self._set_type(text))
        # label and field on one line, and the field kept at its natural width by the trailing
        # stretch — full-width would push its dropdown arrow to the far edge of the panel
        type_row = gui.Horiz(T.emf(self.window, 0.4))
        type_row.add_child(self._label("Kassetype", "text_muted", "small"))
        type_row.add_child(self.type_combo)
        type_row.add_stretch()
        section.add_child(type_row)

        self.stats_label = self._label("", "text", "body")
        section.add_child(self.stats_label)

    def _build_view_section(self) -> None:
        section = self._section("Visning")
        self.ground_check = self._checkbox(
            "Gulv (ledig/opptatt/rot)", True,
            lambda v: self._set_visible("overlay_ground", v and not self.heat_check.checked),
            "Grønt = ledig, rødt = opptatt, gult = rot")
        section.add_child(self.ground_check)
        self.path_check = self._checkbox(
            "Skyve-sti (blå)", True, lambda v: self._set_visible("overlay_path", v),
            "Korridoren en stor kasse kan trilles langs til inngangen")
        section.add_child(self.path_check)
        self.cand_check = self._checkbox(
            "Forslag (grønne kasser)", True, self._toggle_cands,
            "Gjennomsiktige grønne kasser = foreslåtte nye plasser")
        section.add_child(self.cand_check)
        self.heat_check = self._checkbox(
            "Gåtid-varmekart", False, self._toggle_heat,
            "Gangtid fra inngangen for hver ledige gulvcelle (erstatter gulvlaget)")
        section.add_child(self.heat_check)

    def _build_backdrop_section(self) -> None:
        section = self._section("Bakgrunn")
        self.polycam_check = self._checkbox("Polycam-sky (B)", True, self._set_polycam,
                                            "Bytt mellom Polycam-eksporten og egen rekonstruksjon")
        section.add_child(self.polycam_check)
        # off by default: a point cloud has no backfaces to cull, so an uncropped Polycam cloud
        # hides the floor overlays and the bins from the camera this viewer opens with (above the
        # room). Checking this shows the full height, ceiling and all.
        self.ceiling_check = self._checkbox("Full høyde (m/ himling)", False,
                                            lambda _v: self._apply_backdrop(),
                                            "Uten denne kuttes skyen 2 m over gulvet")
        section.add_child(self.ceiling_check)
        self.backdrop_label = self._label("", "text_muted", "small")
        section.add_child(self.backdrop_label)

    def _build_round_section(self) -> None:
        section = self._section("Henterunde")
        self.anim_check = self._checkbox("Animer henting", True, None,
                                         "Renovatøren triller kassene ut til bilen og tilbake")
        section.add_child(self.anim_check)
        self.pause_check = self._checkbox("Pause renovatør")
        section.add_child(self.pause_check)
        self.follow_check = self._checkbox("Følgekamera", False, None,
                                           "Kameraet henger bak renovatøren")
        section.add_child(self.follow_check)
        self.speed_slider = gui.Slider(gui.Slider.DOUBLE)
        self.speed_slider.set_limits(0.25, 4.0)
        self.speed_slider.double_value = 1.0
        speed_row = gui.Horiz(T.emf(self.window, 0.4))
        speed_row.add_child(self._label("Fart", "text_muted", "small"))
        speed_row.add_child(self.speed_slider)   # takes the leftover width of the row
        section.add_child(speed_row)
        self.clock_label = self._label("Tid 0:00", "text", "body")
        section.add_child(self.clock_label)
        # what he is doing right now ("går", "triller full kasse (stor)", "tømmer!") — its own label
        # so the clock line never has to re-wrap while the animation runs
        self.motion_label = self._label("", "text_muted", "small")
        section.add_child(self.motion_label)

    def _build_legend_section(self) -> None:
        """The colour key. Same roles as the preview PNGs — one colour, one meaning, everywhere.

        Two columns: the whole key then costs four rows instead of seven, which is what lets the rest
        of the panel fit without scrolling on a normal window. The longest label needs ~118 px and the
        narrowest column is (20 em - margins) / 2 ≈ 139 px, so it still fits at the minimum width.
        """
        section = self._section("Tegnforklaring")
        grid = gui.VGrid(2, T.emf(self.window, 0.35))
        for role, text in (("free_floor", "Ledig gulv"),
                           ("occupied_floor", "Opptatt gulv"),
                           ("warning", "Rot på gulvet"),
                           ("path", "Skyve-sti"),
                           ("new_bin", "Forslag: ny kasse"),
                           ("existing_bin", "Eksisterende kasse"),
                           ("entrance", "Inngang")):
            self._legend_row(grid, role, text)
        section.add_child(grid)

    def _build_help_section(self) -> None:
        section = self._section("Taster og mus", expanded=False)
        section.add_child(self._label(
            "Dra = roter\n"
            "Høyreklikk + dra = panorer\n"
            "Scroll = zoom\n"
            "Pil venstre/høyre = bytt skann\n"
            "B = bytt bakgrunn (Polycam / egen)",
            "text_muted", "small"))

    # ---------- layout / navigation ----------

    def _on_layout(self, _ctx) -> None:
        rect = self.window.content_rect
        scene_rect, panel_rect = _split_frame(rect.x, rect.y, rect.width, rect.height,
                                              self.window.theme.font_size)
        self.scene.frame = gui.Rect(*scene_rect)
        self.panel.frame = gui.Rect(*panel_rect)

    def _step(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.scans)
        self._load()

    def _set_type(self, bin_type: str) -> None:
        self.bin_type = bin_type
        self._load()

    def _on_key(self, event: gui.KeyEvent) -> gui.Widget.EventCallbackResult:
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.key == gui.KeyName.LEFT:
            self._step(-1)
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key == gui.KeyName.RIGHT:
            self._step(1)
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key == gui.KeyName.B:
            # setting .checked does not fire the handler, so drive the change ourselves
            self.polycam_check.checked = not self.polycam_check.checked
            self._set_polycam(self.polycam_check.checked)
            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.IGNORED

    # ---------- backdrop: Polycam's own export vs our reconstruction ----------

    def _set_polycam(self, wanted: bool) -> None:
        self.use_polycam = bool(wanted)
        self._apply_backdrop()

    def _apply_backdrop(self) -> None:
        """Show exactly one backdrop and name it in the panel. Pure visibility switching — the
        geometries are added once per scan, so toggling never reloads or re-registers anything."""
        choice = self._backdrop
        polycam = bool(choice is not None and choice.available and self.use_polycam)
        self._set_visible(self._ours_name, not polycam)
        self._set_visible("polycam", polycam and self.ceiling_check.checked)
        self._set_visible("polycam_low", polycam and not self.ceiling_check.checked)
        # backdrop.status_text() owns the wording (and the p90 / overlap / sharpness reading); we
        # only fold its lines to the panel width, since an Open3D label would run off the edge.
        self.backdrop_label.text = (self._wrap(backdrop.status_text(choice, polycam))
                                    if choice is not None else "")
        # amber only when Polycam is IMPOSSIBLE for this scan (no export, or the gate rejected it);
        # a cloud the user simply switched off is not a problem worth colouring
        missing = choice is not None and not choice.available
        self.backdrop_label.text_color = T.gui_color("warning" if missing else "text_muted")
        self.window.set_needs_layout()   # the wrapped text can change line count
        self.window.post_redraw()

    # ---------- camera: level-horizon turntable orbit + pan (mirrors the annotation tool) ----------

    def _camera_basis(self):
        view = np.asarray(self.scene.scene.camera.get_view_matrix(), dtype=float)
        cam_to_world = np.linalg.inv(view)
        return cam_to_world[:3, 0], cam_to_world[:3, 1], cam_to_world[:3, 3]  # right, up, eye

    def _start_orbit(self, event) -> None:
        _, _, eye = self._camera_basis()
        offset = eye - self._cor
        radius = max(float(np.linalg.norm(offset)), 0.1)
        self.orbit = {
            "x": float(event.x), "y": float(event.y), "radius": radius,
            "azimuth": math.atan2(float(offset[0]), float(offset[2])),
            "elevation": math.asin(float(np.clip(offset[1] / radius, -1.0, 1.0))),
        }

    def _apply_orbit(self, event) -> None:
        o = self.orbit
        rate = 0.006
        azimuth = o["azimuth"] - (float(event.x) - o["x"]) * rate
        elevation = float(np.clip(o["elevation"] + (float(event.y) - o["y"]) * rate, -1.53, 1.53))
        offset = o["radius"] * np.array([
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.cos(azimuth),
        ])
        self.scene.look_at(self._cor, self._cor + offset, [0.0, 1.0, 0.0])  # world-up = level horizon
        self.window.post_redraw()

    def _start_pan(self, event) -> None:
        right, up, eye = self._camera_basis()
        self.pan = {"x": float(event.x), "y": float(event.y), "right": right, "up": up,
                    "eye": eye, "cor": self._cor.copy()}

    def _apply_pan(self, event) -> None:
        pan = self.pan
        distance = max(float(np.linalg.norm(pan["cor"] - pan["eye"])), 0.5)
        fov = float(self.scene.scene.camera.get_field_of_view())
        per_px = 2 * distance * math.tan(math.radians(fov) / 2) / max(self.scene.frame.height, 1)
        dx = (float(event.x) - pan["x"]) * per_px
        dy = (float(event.y) - pan["y"]) * per_px
        offset = -dx * pan["right"] + dy * pan["up"]
        self._cor = pan["cor"] + offset
        self.scene.look_at(self._cor, pan["eye"] + offset, pan["up"])
        self.window.post_redraw()

    def _on_mouse(self, event) -> gui.Widget.EventCallbackResult:
        consumed = gui.Widget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.Type.WHEEL:
            return gui.Widget.EventCallbackResult.IGNORED  # let the built-in zoom handle scroll
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.RIGHT):
            self._start_pan(event)
            return consumed
        if self.pan is not None:
            if event.type == gui.MouseEvent.Type.DRAG:
                self._apply_pan(event)
                return consumed
            if event.type == gui.MouseEvent.Type.BUTTON_UP:
                self.pan = None
                return consumed
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            self._start_orbit(event)
            return consumed
        if event.type == gui.MouseEvent.Type.DRAG and self.orbit is not None:
            self._apply_orbit(event)
            return consumed
        if event.type == gui.MouseEvent.Type.BUTTON_UP and self.orbit is not None:
            self.orbit = None
            return consumed
        return gui.Widget.EventCallbackResult.IGNORED

    # ---------- rendering ----------

    @staticmethod
    def _layer_cloud(fs, layers, y: float) -> o3d.geometry.PointCloud | None:
        """Point cloud from (mask, color) layers on the free-space grid, at height y."""
        cell, origin = fs.cell, fs.origin
        rows, cols = fs.free.shape
        yy, xx = np.mgrid[0:rows, 0:cols]
        wx = origin[0] + (xx + 0.5) * cell
        wz = origin[1] + (yy + 0.5) * cell
        pts, cols_ = [], []
        for mask, color in layers:
            if not mask.any():
                continue
            x, z = wx[mask], wz[mask]
            pts.append(np.stack([x, np.full(x.shape, y), z], axis=1))
            cols_.append(np.tile(color, (len(x), 1)))
        if not pts:
            return None
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.concatenate(pts))
        cloud.colors = o3d.utility.Vector3dVector(np.concatenate(cols_))
        return cloud

    def _floor_overlays(self, scene) -> tuple:
        """(ground, path, clutter_m2): ground = green free / red occupied / ORANGE clutter (tall
        stuff on the floor that is neither an annotated bin nor wall — bikes, pallets, junk);
        path = the blue corridor + bright route, drawn slightly higher so it wins when visible."""
        fs = scene.fs
        rows, cols = fs.free.shape
        reach = scene.result.reachable if scene.result.reachable is not None else np.zeros((rows, cols), bool)
        route = scene.result.route if scene.result.route is not None else np.zeros((rows, cols), bool)
        occupied = fs.occupied & fs.floor_observed
        bins_mask = (placement._boxes_mask(scene.existing, fs.origin, fs.cell, fs.free.shape, grow=0.15)
                     if scene.existing else np.zeros((rows, cols), bool))
        wall_near = (binary_dilation(scene.wall_mask, iterations=2)
                     if scene.wall_mask is not None and scene.wall_mask.any() else np.zeros((rows, cols), bool))
        clutter = occupied & ~bins_mask & ~wall_near
        clutter_m2 = float(clutter.sum() * fs.cell * fs.cell)
        ground = self._layer_cloud(fs, [
            (occupied & ~clutter, OCC_COLOR),
            (clutter, CLUTTER_COLOR),
            (fs.free, FREE_COLOR),
        ], scene.floor_height + 0.03)
        path = self._layer_cloud(fs, [
            (reach & ~route, PATH_COLOR),
            (route, ROUTE_COLOR),
        ], scene.floor_height + 0.045)
        return ground, path, clutter_m2

    def _heat_overlay(self, scene) -> o3d.geometry.PointCloud | None:
        """Walking-time heatmap: every free floor cell colored by shortest walking time from the
        entrance (viridis, dark = near, bright = far), so expensive corners stand out."""
        fs = scene.fs
        free = fs.free
        if not scene.entrances or not free.any():
            return None
        cell, origin = fs.cell, fs.origin
        graph, node_id, ys, xs = placement._grid_graph(free, np.ones(free.shape))
        if ys.size == 0:
            return None
        ex, ez = scene.entrances[0]
        labels, n = label(free)  # seed inside the LARGEST free region (the actual floor), not a
        if n > 1:                # stray speck that happens to sit nearest the doorway
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            main = labels == int(sizes.argmax())
        else:
            main = free
        near = placement._nearest_true(main, int(round((ez - origin[1]) / cell)),
                                       int(round((ex - origin[0]) / cell)))
        if near is None:
            return None
        dist = dijkstra(graph, directed=False, indices=int(node_id[near]))
        finite = np.isfinite(dist)
        if not finite.any():
            return None
        seconds = dist[finite] * cell / ANIM_SPEED_WALK
        vmax = max(float(np.percentile(seconds, 97)), 1e-6)
        from matplotlib import colormaps
        colors = colormaps["viridis"](np.clip(seconds / vmax, 0.0, 1.0))[:, :3]
        y = scene.floor_height + 0.035
        pts = np.stack([origin[0] + (xs[finite] + 0.5) * cell,
                        np.full(int(finite.sum()), y),
                        origin[1] + (ys[finite] + 0.5) * cell], axis=1)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud

    def _set_visible(self, name: str, visible: bool) -> None:
        if self.scene.scene.has_geometry(name):
            self.scene.scene.show_geometry(name, bool(visible))
            self.window.post_redraw()

    def _toggle_cands(self, visible: bool) -> None:
        for i in range(self._n_cands):
            if self.scene.scene.has_geometry(f"cand_{i}"):
                self.scene.scene.show_geometry(f"cand_{i}", bool(visible))
        self.window.post_redraw()

    def _toggle_heat(self, on: bool) -> None:
        self._set_visible("overlay_heat", on)
        self._set_visible("overlay_ground", (not on) and self.ground_check.checked)

    def _status(self, text: str, role: str = "text") -> None:
        """Write the panel's status line, wrapped and coloured by severity.

        set_needs_layout() matters: Open3D sizes a label once, at layout time, so a status that grows
        from one line to three would have the extra lines clipped without a fresh layout pass.
        """
        self.stats_label.text_color = T.gui_color(role)
        self.stats_label.text = self._wrap(text)
        self.window.set_needs_layout()

    def _load(self) -> None:
        stem = self.scans[self.index]
        self.scan_pos_label.text = f"Skann {self.index + 1} av {len(self.scans)}"
        self.scan_label.text = self._wrap(stem, chars=24)
        self._status("Beregner … (kan ta et par sekunder)", "text_muted")
        try:
            scene = pipeline.compute_scene(stem, self.bin_type)
        except Exception as error:  # noqa: BLE001 - surface any failure in the panel
            self.scene.scene.clear_geometry()
            self._anim = None  # stop the walker from re-adding itself to the cleared scene
            self._backdrop = None
            self.backdrop_label.text = ""
            self._status(f"Feil: {error}", "danger")
            self.window.post_redraw()
            return
        self._render(scene)

    def _render(self, scene) -> None:
        self.scene.scene.clear_geometry()

        if scene.mesh is not None:
            mesh = scene.mesh
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            self._ours_name = "mesh"
            self.scene.scene.add_geometry("mesh", mesh, material)
            bounds = mesh.get_axis_aligned_bounding_box()
        else:
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self._ours_name = "cloud"
            self.scene.scene.add_geometry("cloud", scene.aligned, material)
            bounds = scene.aligned.get_axis_aligned_bounding_box()

        # Polycam's export as an alternative backdrop, in the gravity-aligned frame this viewer
        # draws (scene.mesh / scene.aligned are already rotated by scene.rotation). The camera
        # below is framed on OUR bounds either way, so switching backdrops never moves the view.
        self._backdrop = backdrop.load(scene.stem, gravity_rotation=scene.rotation,
                                       floor_height=scene.floor_height)
        if self._backdrop.available:
            cloud_material = backdrop.material()
            self.scene.scene.add_geometry("polycam", self._backdrop.cloud, cloud_material)
            self.scene.scene.add_geometry("polycam_low", self._backdrop.dollhouse, cloud_material)
        self._apply_backdrop()

        omat = rendering.MaterialRecord()
        omat.shader = "defaultUnlit"
        omat.point_size = 7.0
        ground, path, self._clutter_m2 = self._floor_overlays(scene)
        if ground is not None:
            self.scene.scene.add_geometry("overlay_ground", ground, omat)
            self.scene.scene.show_geometry("overlay_ground",
                                           self.ground_check.checked and not self.heat_check.checked)
        if path is not None:
            self.scene.scene.add_geometry("overlay_path", path, omat)
            self.scene.scene.show_geometry("overlay_path", self.path_check.checked)
        heat = self._heat_overlay(scene)
        if heat is not None:
            self.scene.scene.add_geometry("overlay_heat", heat, omat)
            self.scene.scene.show_geometry("overlay_heat", self.heat_check.checked)

        floor = scene.floor_height
        bin_mat = rendering.MaterialRecord()
        bin_mat.shader = "defaultLit"
        # A red wireframe around every existing bin. The bin bodies are painted like real Norwegian
        # bins (grey container / green dunk), which is nice to look at but says nothing — the outline
        # is what makes "red = kassen står her allerede" read the same as in the preview sheets.
        edge_mat = rendering.MaterialRecord()
        edge_mat.shader = "unlitLine"
        edge_mat.line_width = 2.0
        self._n_exist = len(scene.existing)
        for i, (bx, bz, bl, bw, byaw) in enumerate(scene.existing):
            cls = "stor" if max(bl, bw) >= 1.0 else "liten"
            model = _bin_model_world(bx, bz, bl, bw, byaw, floor, cls)
            self.scene.scene.add_geometry(f"exist_{i}", model, bin_mat)
            outline = _bin_box_lineset(((bx, bz), (bl, bw), byaw), floor,
                                       floor + BIN_HEIGHT[cls], EXISTING_EDGE_COLOR)
            self.scene.scene.add_geometry(f"exist_edge_{i}", outline, edge_mat)
        ghost_mat = rendering.MaterialRecord()
        ghost_mat.shader = "defaultLitTransparency"
        ghost_mat.base_color = [*PROPOSAL_COLOR, 0.72]   # translucent, so the floor reads through
        self._n_cands = len(scene.result.candidates)
        for i, cand in enumerate(scene.result.candidates):
            (ccx, ccz), (rl, rw), ang = cand.rect
            cls = "stor" if (cand.bin_type == "4-hjuls container" or max(rl, rw) >= 1.0) else "liten"
            cand_height = BIN_TYPES.get(cand.bin_type, (0.0, 0.0, 0.0))[1] or None
            ghost = _bin_model_world(ccx, ccz, rl, rw, ang, floor, cls, height=cand_height, ghost=True)
            self.scene.scene.add_geometry(f"cand_{i}", ghost, ghost_mat)
            self.scene.scene.show_geometry(f"cand_{i}", self.cand_check.checked)

        sphere_mat = rendering.MaterialRecord()
        sphere_mat.shader = "defaultUnlit"
        for i, (ex, ez) in enumerate(scene.entrances):
            sphere = o3d.geometry.TriangleMesh.create_sphere(0.15, resolution=12)
            sphere.translate([ex, floor + 0.15, ez])
            sphere.paint_uniform_color(list(ENTRANCE_COLOR))
            self.scene.scene.add_geometry(f"entrance_{i}", sphere, sphere_mat)

        center = np.asarray(bounds.get_center())
        extent = np.asarray(bounds.get_extent())
        self.scene.setup_camera(60.0, bounds, center)
        self._cor = center.copy()  # orbit around the room centre, level horizon
        eye = center + np.array([0.0, extent[1] * 1.5 + 3.0, extent[2] * 0.8 + 3.0])
        self.scene.look_at(center, eye, [0.0, 1.0, 0.0])

        self._setup_path_animation(scene)
        if self._anim is not None:  # park the renovation truck outside the entrance he dumps at
            pause = next((l for l in self._anim["legs"] if l["kind"] == "pause"), None)
            if pause is not None:
                truck_mat = rendering.MaterialRecord()
                truck_mat.shader = "defaultLit"
                truck_pos, truck_dir = _truck_anchor(scene, pause["pos"], pause["dir"])
                self.scene.scene.add_geometry(
                    "sti_truck", _truck_mesh(truck_pos, truck_dir, self._anim["floor"]), truck_mat)

        if scene.enclosed:
            self._status("OBS: innesperret rom (dør lukket i scan) - hoppet over", "warning")
        else:
            self._status(
                f"{len(scene.result.candidates)} nye plasser · {len(scene.existing)} eksisterende\n"
                f"Ledig gulv {scene.fs.free_area_m2:.1f} m² · rot {self._clutter_m2:.1f} m²"
            )
        self.window.post_redraw()

    # ---------- push-path animation (a stick figure walking the routed push-path) ----------

    def _setup_path_animation(self, scene) -> None:
        self._anim = None
        self._leg_idx = 0
        self._leg_pos = 0.0
        self._anim_time = 0.0
        self._walked = 0.0
        self._last_tick = None
        self._cam = None
        legs = _walk_legs(scene)
        if legs:
            self._anim = {"legs": legs, "floor": float(scene.floor_height)}

    def _tick(self) -> bool:
        if self._anim is None or not self.anim_check.checked:
            cleared = False
            for name in (self._mover_name, self._bin_name, self._trash_name):
                if self.scene.scene.has_geometry(name):
                    self.scene.scene.remove_geometry(name)
                    cleared = True
            self._last_tick = None
            self.motion_label.text = "" if self._anim is not None else "ingen runde å vise"
            if cleared:
                self.window.post_redraw()
            return False

        now = time.monotonic()
        raw_dt = min(now - self._last_tick, 0.1) if self._last_tick is not None else 0.0
        self._last_tick = now
        dt = 0.0 if self.pause_check.checked else raw_dt * float(self.speed_slider.double_value)
        legs = self._anim["legs"]
        leg = legs[self._leg_idx]

        def leg_rate(current: dict) -> float:
            if current["kind"] == "pause":
                return 1.0  # pause legs advance in seconds
            if current["mode"] == "walk":
                return ANIM_SPEED_WALK
            return ANIM_WHEEL_SPEED[(current["size"], current["mode"])]

        advance = leg_rate(leg) * dt
        self._leg_pos += advance
        self._anim_time += dt
        if leg["kind"] == "move":
            self._walked += advance
        while self._leg_pos >= leg["total"]:
            self._leg_pos -= leg["total"]
            self._leg_idx += 1
            if self._leg_idx >= len(legs):  # full round done — loop and restart the clock
                self._leg_idx = 0
                self._anim_time = 0.0
            leg = legs[self._leg_idx]

        floor = self._anim["floor"]
        phase = self._walked / ANIM_STRIDE_M * 2 * math.pi
        trash = None
        if leg["kind"] == "pause":  # standing at the truck, tipping the bin out
            position = np.asarray(leg["pos"], dtype=float)
            direction = np.asarray(leg["dir"], dtype=float)
            tau = self._leg_pos / leg["total"]
            tip_max = BIN_TIP_DEG[leg["size"]]
            if tau < 0.35:
                tip = tip_max * tau / 0.35
            elif tau < 0.70:
                tip = tip_max + 3.0 * math.sin(tau * 26.0)
            else:
                tip = tip_max * (1.0 - (tau - 0.70) / 0.30)
            figure = _figure_mesh(position, direction, floor, phase, "dump")
            bin_mesh, mouth = _wheeled_bin_mesh(position, direction, floor, leg["dims"],
                                                leg["size"], tip_deg=tip)
            if 0.30 < tau < 0.80:
                trash = _trash_bits_mesh(tau, mouth, direction, floor)
            status = "tømmer!"
        else:
            position = _point_at(leg, self._leg_pos)
            ahead = _point_at(leg, self._leg_pos + 0.30)
            behind = _point_at(leg, self._leg_pos - 0.05)
            direction = ahead - behind
            if float(np.linalg.norm(direction)) < 1e-6:
                direction = leg["pts"][-1] - leg["pts"][0]
            figure = _figure_mesh(position, direction, floor, phase, leg["mode"])
            if leg["mode"] == "walk":
                bin_mesh = None
                status = "går"
            else:
                bin_mesh, _ = _wheeled_bin_mesh(position, direction, floor, leg["dims"], leg["size"])
                status = ("triller full kasse" if leg["mode"] == "full"
                          else "triller tom kasse tilbake") + f" ({leg['size']})"

        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        for name, mesh in ((self._mover_name, figure), (self._bin_name, bin_mesh),
                           (self._trash_name, trash)):
            if self.scene.scene.has_geometry(name):
                self.scene.scene.remove_geometry(name)
            if mesh is not None:
                self.scene.scene.add_geometry(name, mesh, material)

        # the bin he is wheeling vanishes from its parking spot and reappears when returned
        carried = leg.get("bin_idx") if (leg["kind"] == "pause" or leg.get("mode") in ("full", "tom")) else None
        for k in range(self._n_exist):
            for name in (f"exist_{k}", f"exist_edge_{k}"):  # the red outline travels with its bin
                if self.scene.scene.has_geometry(name):
                    self.scene.scene.show_geometry(name, k != carried)

        if self.follow_check.checked:  # smoothed chase-cam behind the worker
            d2 = _unit2(direction)
            d3 = np.array([d2[0], 0.0, d2[1]])
            pos3 = np.array([float(position[0]), floor, float(position[1])])
            eye = pos3 - d3 * 3.2 + np.array([0.0, 2.1, 0.0])
            target = pos3 + d3 * 0.8 + np.array([0.0, 0.8, 0.0])
            fresh = np.concatenate([eye, target])
            self._cam = fresh if self._cam is None else 0.85 * self._cam + 0.15 * fresh
            self.scene.look_at(self._cam[3:], self._cam[:3], [0.0, 1.0, 0.0])
            self._cor = self._cam[3:].copy()
        else:
            self._cam = None

        # two single-line labels, never re-wrapped: a label whose line count changes needs a fresh
        # layout, and asking for one every frame would relayout the whole window 60 times a second
        seconds = int(self._anim_time)
        self.clock_label.text = f"Tid {seconds // 60}:{seconds % 60:02d}"
        self.motion_label.text = status
        self.window.post_redraw()
        return True

    def run(self) -> None:
        gui.Application.instance.run()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Interaktiv 3D-visning av plassering + skyve-sti")
    parser.add_argument("--scan", default=None, help="stem/sti til skannet som skal vises først")
    parser.add_argument("--bin-type", default="4-hjuls container", help="kassetype for plassering")
    args = parser.parse_args()

    app = PlacementViewer(bin_type=args.bin_type)
    if args.scan:
        stem = Path(args.scan).stem
        if stem in app.scans:
            app.index = app.scans.index(stem)
            app._load()
    app.run()


if __name__ == "__main__":
    main()
