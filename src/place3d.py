"""Interactive 3D placement viewer.

Browse scans and SEE, on the room mesh: the push-path a large bin can be wheeled along to a door
(blue floor), the free bin space around it (green), suggested new bins (green boxes), existing bins
(red boxes) and entrances (magenta). Read-only sibling of the annotation tool — same drag-to-orbit
controls and Prev/Next scan navigation.

    .venv\\Scripts\\python.exe -m src.place3d
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.ndimage import binary_dilation, label
from scipy.sparse.csgraph import dijkstra

from . import pipeline, placement
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

FREE_COLOR = (0.10, 0.80, 0.10)      # green  = free bin space
PATH_COLOR = (0.20, 0.45, 1.00)      # blue   = push-path a large bin can reach a door through
ROUTE_COLOR = (0.15, 0.75, 1.00)     # bright = the near-straight route the corridor is built on
OCC_COLOR = (0.85, 0.10, 0.10)       # red    = occupied floor
PROPOSAL_COLOR = (0.25, 0.55, 0.95)  # translucent BLUE ghost bin = suggested new bin
ENTRANCE_COLOR = (1.00, 0.10, 1.00)  # magenta sphere = entrance

ANIM_SPEED_WALK = 1.05    # m/s walking empty-handed
ANIM_WHEEL_SPEED = {      # m/s wheeling a bin: big bins roll faster, a full bin is heavier
    ("stor", "full"): 0.63, ("stor", "tom"): 0.92,
    ("liten", "full"): 0.52, ("liten", "tom"): 0.80,
}
ANIM_EMPTY_SECONDS = 2.8  # duration of the tipping animation at the truck
ANIM_STRIDE_M = 1.0       # metres per full walk cycle (drives the leg swing)
FIGURE_COLOR = (1.0, 0.45, 0.05)     # hi-vis orange — the renovation worker
HEAD_COLOR = (0.96, 0.83, 0.66)
BIN_HEIGHT = {"stor": 1.25, "liten": 1.15}  # real Norwegian bin heights (4-hjuls / 2-hjuls)
CLUTTER_COLOR = (1.00, 0.62, 0.12)          # orange floor = clutter blocking it (not bins, not walls)
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
    reach = scene.result.reachable
    to_bin: list[np.ndarray | None] = [None] * len(bins)     # entrance -> bin i
    between: list[np.ndarray | None] = [None] * len(bins)    # bin i-1 -> bin i
    if reach is not None and reach.any():
        cell, origin = fs.cell, fs.origin
        bins_mask = placement._boxes_mask(bins, origin, cell, reach.shape, grow=0.05)
        strict = reach & ~bins_mask
        if strict.any():
            graph, node_id, ys, xs = placement._grid_graph(strict, np.ones(strict.shape))

            def node_at(xz: np.ndarray) -> int:
                col = int(round((xz[0] - origin[0]) / cell))
                row = int(round((xz[1] - origin[1]) / cell))
                near = placement._nearest_true(strict, row, col)
                return int(node_id[near]) if near is not None else -1

            nodes = [node_at(start)] + [node_at(c) for c in centers]
            if all(n >= 0 for n in nodes):
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
                    to_bin[i] = path(0, nodes[i + 1], start)
                    if i > 0:
                        between[i] = path(i, nodes[i + 1], None)

    for i in range(len(bins)):  # straight-line last resort (never for the return: it reuses to_bin)
        if to_bin[i] is None:
            to_bin[i] = np.array([start, centers[i]])
        if i > 0 and between[i] is None:
            between[i] = np.array([centers[i - 1], centers[i]])

    legs: list[dict] = []

    def add(leg: dict | None) -> None:
        if leg is not None:
            legs.append(leg)

    for i in range(len(bins)):
        add(_leg(to_bin[i] if i == 0 else between[i], "walk", sizes[i]))
        add(_leg(to_bin[i][::-1].copy(), "full", sizes[i], dims[i], order[i]))
        arrive_dir = (to_bin[i][0] - to_bin[i][1]) if len(to_bin[i]) > 1 else np.array([1.0, 0.0])
        legs.append({"kind": "pause", "total": ANIM_EMPTY_SECONDS, "pos": to_bin[i][0],
                     "dir": arrive_dir, "mode": "dump", "size": sizes[i],
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


class PlacementViewer:
    def __init__(self, bin_type: str = "4-hjuls container") -> None:
        self.scans = [s for s in pipeline.list_scans() if pipeline.is_prepared(s)]
        if not self.scans:
            raise SystemExit("no prepared scans to show — prepare some first")
        self.index = 0
        self.bin_type = bin_type if bin_type in BIN_TYPES else "4-hjuls container"

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Søppelrom 3D — plassering & skyve-sti", 1500, 950)
        em = self.window.theme.font_size

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.scene.set_on_key(self._on_key)
        self.scene.set_on_mouse(self._on_mouse)  # custom turntable orbit, like the annotation tool
        self.window.add_child(self.scene)

        self._cor = np.zeros(3)   # centre of rotation (level-horizon turntable)
        self.orbit: dict | None = None
        self.pan: dict | None = None

        self.panel = gui.Vert(0.4 * em, gui.Margins(0.6 * em, 0.6 * em, 0.6 * em, 0.6 * em))
        self.scan_label = gui.Label("")
        self.panel.add_child(self.scan_label)

        nav = gui.Horiz(0.4 * em)
        prev_btn = gui.Button("< Forrige")
        prev_btn.set_on_clicked(lambda: self._step(-1))
        next_btn = gui.Button("Neste >")
        next_btn.set_on_clicked(lambda: self._step(1))
        nav.add_child(prev_btn)
        nav.add_child(next_btn)
        self.panel.add_child(nav)

        self.panel.add_child(gui.Label("Kassetype:"))
        self.type_combo = gui.Combobox()
        for name in BIN_TYPES:
            self.type_combo.add_item(name)
        self.type_combo.selected_text = self.bin_type
        self.type_combo.set_on_selection_changed(lambda text, _index: self._set_type(text))
        self.panel.add_child(self.type_combo)

        self.anim_check = gui.Checkbox("Animer henting")
        self.anim_check.checked = True
        self.panel.add_child(self.anim_check)
        self.pause_check = gui.Checkbox("Pause renovatør")
        self.panel.add_child(self.pause_check)
        self.follow_check = gui.Checkbox("Følgekamera")
        self.panel.add_child(self.follow_check)
        self.panel.add_child(gui.Label("Avspillingsfart:"))
        self.speed_slider = gui.Slider(gui.Slider.DOUBLE)
        self.speed_slider.set_limits(0.25, 4.0)
        self.speed_slider.double_value = 1.0
        self.panel.add_child(self.speed_slider)
        self.clock_label = gui.Label("⏱ 0:00")
        self.panel.add_child(self.clock_label)

        self.panel.add_child(gui.Label("Vis:"))
        self.ground_check = gui.Checkbox("Gulv (ledig/opptatt/rot)")
        self.ground_check.checked = True
        self.ground_check.set_on_checked(lambda v: self._set_visible("overlay_ground", v and not self.heat_check.checked))
        self.panel.add_child(self.ground_check)
        self.path_check = gui.Checkbox("Skyve-sti (blå)")
        self.path_check.checked = True
        self.path_check.set_on_checked(lambda v: self._set_visible("overlay_path", v))
        self.panel.add_child(self.path_check)
        self.cand_check = gui.Checkbox("Forslag (blå kasser)")
        self.cand_check.checked = True
        self.cand_check.set_on_checked(self._toggle_cands)
        self.panel.add_child(self.cand_check)
        self.heat_check = gui.Checkbox("Gåtid-varmekart")
        self.heat_check.set_on_checked(self._toggle_heat)
        self.panel.add_child(self.heat_check)

        self.stats_label = gui.Label("")
        self.panel.add_child(self.stats_label)
        self.panel.add_child(gui.Label(
            "\nBlått gulv = skyve-sti\n"
            "Grønt gulv = ledig plass\n"
            "Rødt gulv = opptatt\n"
            "Oransje gulv = rot (ikke kasser)\n"
            "Blå kasse (glass) = forslag\n"
            "Grå/grønn kasse = eksisterende\n"
            "Rosa kule = inngang\n"
        ))
        self.panel.add_child(gui.Label("Dra = roter · scroll = zoom\nPil venstre/høyre = bytt skann"))

        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

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

    # ---------- layout / navigation ----------

    def _on_layout(self, _ctx) -> None:
        rect = self.window.content_rect
        panel_width = 19 * self.window.theme.font_size
        self.scene.frame = gui.Rect(rect.x, rect.y, rect.width - panel_width, rect.height)
        self.panel.frame = gui.Rect(rect.get_right() - panel_width, rect.y, panel_width, rect.height)

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
        return gui.Widget.EventCallbackResult.IGNORED

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

    def _load(self) -> None:
        stem = self.scans[self.index]
        self.scan_label.text = f"Skann {self.index + 1}/{len(self.scans)}:\n{stem}"
        self.stats_label.text = "Beregner … (kan ta et par sekunder)"
        try:
            scene = pipeline.compute_scene(stem, self.bin_type)
        except Exception as error:  # noqa: BLE001 - surface any failure in the panel
            self.scene.scene.clear_geometry()
            self._anim = None  # stop the walker from re-adding itself to the cleared scene
            self.stats_label.text = f"Feil: {error}"
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
            self.scene.scene.add_geometry("mesh", mesh, material)
            bounds = mesh.get_axis_aligned_bounding_box()
        else:
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self.scene.scene.add_geometry("cloud", scene.aligned, material)
            bounds = scene.aligned.get_axis_aligned_bounding_box()

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
        self._n_exist = len(scene.existing)
        for i, (bx, bz, bl, bw, byaw) in enumerate(scene.existing):
            cls = "stor" if max(bl, bw) >= 1.0 else "liten"
            model = _bin_model_world(bx, bz, bl, bw, byaw, floor, cls)
            self.scene.scene.add_geometry(f"exist_{i}", model, bin_mat)
        ghost_mat = rendering.MaterialRecord()
        ghost_mat.shader = "defaultLitTransparency"
        ghost_mat.base_color = [PROPOSAL_COLOR[0], PROPOSAL_COLOR[1], PROPOSAL_COLOR[2], 0.78]
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
                self.scene.scene.add_geometry(
                    "sti_truck", _truck_mesh(pause["pos"], pause["dir"], self._anim["floor"]), truck_mat)

        if scene.enclosed:
            self.stats_label.text = "⚠ INNESPERRET rom (dør lukket i scan) — hoppet over"
        else:
            self.stats_label.text = (
                f"{len(scene.result.candidates)} nye plasser  ·  {len(scene.existing)} eksisterende\n"
                f"ledig gulv {scene.fs.free_area_m2:.1f} m²  ·  rot blokkerer {self._clutter_m2:.1f} m²"
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
            if self.scene.scene.has_geometry(f"exist_{k}"):
                self.scene.scene.show_geometry(f"exist_{k}", k != carried)

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

        seconds = int(self._anim_time)
        self.clock_label.text = f"⏱ {seconds // 60}:{seconds % 60:02d} · {status}"
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
