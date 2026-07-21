"""3D annotation tool: review, create and delete bin boxes on top of the Poisson mesh.

Usage:  .venv\\Scripts\\python.exe -m src.annotate3d

CAD-style interactions:
  - "Tegn boks": click corner A on the floor, click corner B (first edge), then PRESS for
    the depth point, DRAG upward to pull the box out of the floor, RELEASE to finish.
    ESC cancels.
  - Selected box shows handles: drag a yellow bottom corner to resize the footprint
    (opposite corner stays fixed), drag the blue top sphere to change height, drag the
    magenta sphere to rotate, drag on the box body to move it. Click a box to select it.
  - Plain drag on empty space orbits the camera as usual; Ctrl+click teleports the
    selected box to the clicked floor point.

A background worker process keeps up to 5 scans prepared ahead while you annotate.
"""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from .annotations import (
    BIN_TYPES,
    BOX_EDGES,
    STATUS_APPROVED,
    STATUS_PROPOSED,
    BinBox,
    load_annotations,
    save_annotations,
)
from .prepare_scan import ANNOTATION_DIR, CACHE_ROOT, PROJECT_ROOT, RAW_DIR, is_prepared
from .set_entrance import ENTRANCE_DIR, load_entrances

STATUS_COLORS = {
    STATUS_PROPOSED: (1.0, 0.55, 0.05),
    STATUS_APPROVED: (0.1, 0.85, 0.2),
}
SELECTED_COLOR = (0.2, 0.5, 1.0)
PREVIEW_COLOR = (0.1, 0.9, 0.9)
HANDLE_COLORS = {
    "corner": (1.0, 0.85, 0.1),
    "top": (0.2, 0.6, 1.0),
    "rotate": (1.0, 0.2, 0.8),
}
CORNER_SIGNS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
HANDLE_NAMES = [f"handle_corner_{i}" for i in range(4)] + ["handle_top", "handle_rotate"]

MODE_NORMAL = "normal"
MODE_DRAW = "draw"
MODE_ENTRANCE = "entrance"


def _estimate_floor(mesh: o3d.geometry.TriangleMesh) -> float:
    """Fallback floor height from mesh vertices (mode of the lower Y band) when the cached
    floor_height is missing."""
    ys = np.asarray(mesh.vertices)[:, 1]
    if not len(ys):
        return 0.0
    lo, hi = np.percentile(ys, [1, 60])
    band = ys[(ys >= lo) & (ys <= hi)]
    if not len(band):
        return float(np.percentile(ys, 5))
    hist, edges = np.histogram(band, bins=40)
    return float(edges[int(hist.argmax())])


def start_background_worker(max_ready: int = 999) -> subprocess.Popen | None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    log = open(CACHE_ROOT / "worker.log", "a", encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.Popen(
        [sys.executable, "-m", "src.prepare_scan", "--watch", "--max-ready", str(max_ready)],
        cwd=str(PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
    )


def _ray_point_distance(ray: tuple[np.ndarray, np.ndarray], point: np.ndarray) -> tuple[float, float]:
    origin, direction = ray
    offset = point - origin
    along = float(offset @ direction)
    if along < 0:
        return float("inf"), 0.0
    return float(np.linalg.norm(offset - along * direction)), along


def _ray_hits_box(ray: tuple[np.ndarray, np.ndarray], box: BinBox) -> float | None:
    origin, direction = ray
    rotation = box.rotation_matrix()
    local_origin = rotation.T @ (origin - np.asarray(box.center))
    local_direction = rotation.T @ direction
    half = np.asarray(box.extent) / 2
    t_min, t_max = -np.inf, np.inf
    for axis in range(3):
        if abs(local_direction[axis]) < 1e-9:
            if abs(local_origin[axis]) > half[axis]:
                return None
            continue
        t1 = (-half[axis] - local_origin[axis]) / local_direction[axis]
        t2 = (half[axis] - local_origin[axis]) / local_direction[axis]
        t1, t2 = min(t1, t2), max(t1, t2)
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None
    if t_max < 0:
        return None
    return max(t_min, 0.0)


class AnnotationApp:
    def __init__(self) -> None:
        self.scans = sorted(RAW_DIR.glob("*.zip"))
        if not self.scans:
            raise SystemExit(f"no scans found in {RAW_DIR}")
        self.scan_index = 0
        self.boxes: list[BinBox] = []
        self.selected: int | None = None
        self.floor_height: float | None = None
        self.entrances: list[tuple[float, float]] = []
        self._drawn_entrances = 0
        self.mesh_loaded = False
        self.dirty = False
        self._last_poll = 0.0
        self._last_preview = 0.0
        self._drawn_boxes = 0
        self.undo_stack: list[tuple[list[BinBox], int | None]] = []
        self._ctrl_down = False
        self.mode = MODE_NORMAL
        self.draw_stage = 0
        self.draw_a: np.ndarray | None = None
        self.draw_b: np.ndarray | None = None
        self.draw_box: BinBox | None = None
        self.drag: dict | None = None
        self.pan: dict | None = None
        self.orbit: dict | None = None
        self._cor = np.zeros(3)
        self._mesh: o3d.geometry.TriangleMesh | None = None
        self._mesh_material: rendering.MaterialRecord | None = None
        self._tri_normals: np.ndarray | None = None
        self._tri_centers: np.ndarray | None = None
        self._last_cull_eye: np.ndarray | None = None
        self._last_cull_time = 0.0

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Søppelrom 3D-annotering", 1500, 950)
        em = self.window.theme.font_size

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.scene.set_on_mouse(self._on_mouse)
        self.scene.set_on_key(self._on_key)
        self.window.add_child(self.scene)

        self.panel = gui.Vert(0.4 * em, gui.Margins(0.6 * em, 0.6 * em, 0.6 * em, 0.6 * em))

        self.scan_label = gui.Label("")
        self.panel.add_child(self.scan_label)
        nav = gui.Horiz(0.4 * em)
        prev_btn = gui.Button("< Forrige")
        prev_btn.set_on_clicked(lambda: self._switch_scan(-1))
        next_btn = gui.Button("Neste >")
        next_btn.set_on_clicked(lambda: self._switch_scan(1))
        nav.add_child(prev_btn)
        nav.add_child(next_btn)
        self.panel.add_child(nav)

        self.status_label = gui.Label("")
        self.panel.add_child(self.status_label)
        self.panel.add_child(gui.Label("Bokser:"))

        self.box_list = gui.ListView()
        self.box_list.set_on_selection_changed(self._on_list_selection)
        self.panel.add_child(self.box_list)

        act = gui.Horiz(0.4 * em)
        approve_btn = gui.Button("Godkjenn")
        approve_btn.set_on_clicked(self._approve_selected)
        delete_btn = gui.Button("Slett")
        delete_btn.set_on_clicked(self._delete_selected)
        undo_btn = gui.Button("Angre")
        undo_btn.set_on_clicked(self._undo)
        act.add_child(approve_btn)
        act.add_child(delete_btn)
        act.add_child(undo_btn)
        self.panel.add_child(act)

        self.panel.add_child(gui.Label("Kassetype:"))
        self.type_combo = gui.Combobox()
        for name in BIN_TYPES:
            self.type_combo.add_item(name)
        self.panel.add_child(self.type_combo)

        new_row = gui.Horiz(0.4 * em)
        draw_btn = gui.Button("Tegn boks")
        draw_btn.set_on_clicked(self._start_draw)
        quick_btn = gui.Button("Standardboks")
        quick_btn.set_on_clicked(self._new_standard_box)
        retype_btn = gui.Button("Sett type")
        retype_btn.set_on_clicked(self._retype_selected)
        new_row.add_child(draw_btn)
        new_row.add_child(quick_btn)
        new_row.add_child(retype_btn)
        self.panel.add_child(new_row)

        entrance_row = gui.Horiz(0.4 * em)
        entrance_btn = gui.Button("Inngang av/på")
        entrance_btn.set_on_clicked(self._toggle_entrance_mode)
        entrance_clear = gui.Button("Nullstill innganger")
        entrance_clear.set_on_clicked(self._clear_entrances)
        entrance_row.add_child(entrance_btn)
        entrance_row.add_child(entrance_clear)
        self.panel.add_child(entrance_row)

        self.mode_label = gui.Label("")
        self.panel.add_child(self.mode_label)

        self.cull_checkbox = gui.Checkbox("Skjul veggbaksider")
        self.cull_checkbox.checked = True
        self.cull_checkbox.set_on_checked(lambda _checked: self._update_culling(force=True))
        self.panel.add_child(self.cull_checkbox)

        self.panel.add_child(gui.Label("Finjuster valgt (5 cm / 5°):"))
        move_grid = gui.VGrid(4, 0.3 * em)
        for text, fn in [
            ("X-", lambda: self._nudge(dx=-0.05)), ("X+", lambda: self._nudge(dx=0.05)),
            ("Z-", lambda: self._nudge(dz=-0.05)), ("Z+", lambda: self._nudge(dz=0.05)),
            ("Rot-", lambda: self._nudge(dyaw=-5.0)), ("Rot+", lambda: self._nudge(dyaw=5.0)),
            ("H-", lambda: self._nudge(dey=-0.05)), ("H+", lambda: self._nudge(dey=0.05)),
            ("L-", lambda: self._nudge(dex=-0.05)), ("L+", lambda: self._nudge(dex=0.05)),
            ("B-", lambda: self._nudge(dez=-0.05)), ("B+", lambda: self._nudge(dez=0.05)),
        ]:
            button = gui.Button(text)
            button.set_on_clicked(fn)
            move_grid.add_child(button)
        self.panel.add_child(move_grid)

        save_btn = gui.Button("Lagre (auto ved bytte)")
        save_btn.set_on_clicked(self._save)
        self.panel.add_child(save_btn)

        self.help_label = gui.Label(
            "Venstre-dra: orbit. Høyre-dra: pan.\n"
            "Tegn boks (T, starter i topdown):\n"
            "  klikk A, klikk B, trykk+dra opp, slipp.\n"
            "Klikk på boks: velg. Håndtak: gult =\n"
            "  størrelse, blå = høyde, rosa = roter.\n"
            "Tastatur (valgt boks):\n"
            "  Del = slett, G = godkjenn\n"
            "  Piltaster = flytt, Q/E = roter\n"
            "  PgUp/PgDn = høyde, 1-4 = type\n"
            "  Ctrl+Z = angre, Ctrl+S = lagre\n"
            "Ctrl+klikk: flytt valgt hit. ESC: avbryt.\n"
            "Oransje = forslag, grønn = godkjent,\nblå = valgt"
        )
        self.panel.add_child(self.help_label)

        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        if hasattr(self.window, "set_on_tick_event"):
            self.window.set_on_tick_event(self._on_tick)

        self.worker = start_background_worker()
        self._load_scan()

    # ---------- layout ----------

    def _on_layout(self, ctx) -> None:
        rect = self.window.content_rect
        panel_width = 18 * self.window.theme.font_size
        self.scene.frame = gui.Rect(rect.x, rect.y, rect.width - panel_width, rect.height)
        self.panel.frame = gui.Rect(rect.get_right() - panel_width, rect.y, panel_width, rect.height)

    # ---------- scan loading ----------

    def _current_zip(self) -> Path:
        return self.scans[self.scan_index]

    def _annotation_path(self) -> Path:
        return ANNOTATION_DIR / f"{self._current_zip().stem}.json"

    def _cache_dir(self) -> Path:
        return CACHE_ROOT / self._current_zip().stem

    def _load_scan(self) -> None:
        self.scene.scene.clear_geometry()
        self.boxes = []
        self.entrances = []
        self.selected = None
        self.mesh_loaded = False
        self.dirty = False
        self._drawn_boxes = 0
        self._drawn_entrances = 0
        self.undo_stack.clear()
        self._cancel_draw()
        self.drag = None
        self.pan = None
        self.orbit = None
        zip_path = self._current_zip()
        self.scan_label.text = f"Skann {self.scan_index + 1}/{len(self.scans)}:\n{zip_path.stem}"

        if not is_prepared(zip_path):
            self.status_label.text = "Forbereder i bakgrunnen ...\n(lastes automatisk når klar)"
            return

        mesh = o3d.io.read_triangle_mesh(str(self._cache_dir() / "mesh_poisson.ply"))
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        self._mesh = mesh
        self._mesh_material = material
        self._tri_normals = np.asarray(mesh.triangle_normals).copy()
        triangles = np.asarray(mesh.triangles)
        self._tri_centers = np.asarray(mesh.vertices)[triangles].mean(axis=1)
        self._last_cull_eye = None
        self.scene.scene.add_geometry("room_mesh", mesh, material)
        self.mesh_loaded = True

        if self._annotation_path().exists():
            self.floor_height, self.boxes = load_annotations(self._annotation_path())
            source = "lagrede annoteringer"
        else:
            self.floor_height, self.boxes = load_annotations(self._cache_dir() / "proposals.json")
            source = "auto-forslag"
        if self.floor_height is None:  # missing in cache -> estimate from the mesh so boxes sit on the floor
            self.floor_height = _estimate_floor(mesh)
        self.entrances = load_entrances(self._current_zip().stem)
        self.status_label.text = f"{len(self.boxes)} bokser ({source})"

        bounds = mesh.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60.0, bounds, bounds.get_center())
        center = np.asarray(bounds.get_center())
        extent = np.asarray(bounds.get_extent())
        eye = center + np.array([0.0, extent[1] * 1.6 + 2.0, extent[2] * 0.7 + 2.0])
        self._cor = center.copy()
        self.scene.look_at(center, eye, [0.0, 1.0, 0.0])
        self._redraw_boxes()
        self._redraw_entrances()
        self._update_culling(force=True)

    def _switch_scan(self, step: int) -> None:
        self._save()
        self.scan_index = (self.scan_index + step) % len(self.scans)
        self._load_scan()

    def _on_tick(self) -> bool:
        if self.mesh_loaded:
            if time.time() - self._last_cull_time > 0.2:
                self._last_cull_time = time.time()
                return self._update_culling()
            return False
        if time.time() - self._last_poll < 2.0:
            return False
        self._last_poll = time.time()
        if is_prepared(self._current_zip()):
            self._load_scan()
            return True
        return False

    def _update_culling(self, force: bool = False) -> bool:
        """Hide mesh triangles facing away from the camera. Walls are scanned from one side
        only (normals point into the room), so their backsides disappear when the camera is
        outside — a dollhouse view that makes annotating inside rooms much easier."""
        if self._mesh is None or self._tri_normals is None:
            return False
        if not self.cull_checkbox.checked:
            if self._last_cull_eye is not None:
                self._last_cull_eye = None
                self._remove_geometry("room_mesh")
                self.scene.scene.add_geometry("room_mesh", self._mesh, self._mesh_material)
                self.window.post_redraw()
                return True
            return False

        _, _, eye = self._camera_basis()
        if not force and self._last_cull_eye is not None:
            if float(np.linalg.norm(eye - self._last_cull_eye)) < 0.02:
                return False
        self._last_cull_eye = eye

        view_dirs = eye - self._tri_centers
        visible = np.einsum("ij,ij->i", self._tri_normals, view_dirs) > 0.0
        culled = o3d.geometry.TriangleMesh(
            self._mesh.vertices,
            o3d.utility.Vector3iVector(np.asarray(self._mesh.triangles)[visible]),
        )
        culled.vertex_colors = self._mesh.vertex_colors
        culled.vertex_normals = self._mesh.vertex_normals
        self._remove_geometry("room_mesh")
        self.scene.scene.add_geometry("room_mesh", culled, self._mesh_material)
        self.window.post_redraw()
        return True

    # ---------- geometry helpers ----------

    def _floor_y(self) -> float:
        return self.floor_height if self.floor_height is not None else 0.0

    def _mouse_ray(self, event: gui.MouseEvent) -> tuple[np.ndarray, np.ndarray] | None:
        x = event.x - self.scene.frame.x
        y = event.y - self.scene.frame.y
        width = self.scene.frame.width
        height = self.scene.frame.height
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        camera = self.scene.scene.camera
        near = np.asarray(camera.unproject(x, y, 0.05, width, height), dtype=float).reshape(3)
        far = np.asarray(camera.unproject(x, y, 0.95, width, height), dtype=float).reshape(3)
        direction = far - near
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1e-9:
            return None
        return near, direction / norm

    def _ray_floor(self, ray: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
        origin, direction = ray
        if abs(direction[1]) < 1e-6:
            return None
        t = (self._floor_y() - origin[1]) / direction[1]
        if t <= 0:
            return None
        return origin + t * direction

    def _height_from_ray(self, ray: tuple[np.ndarray, np.ndarray], center_xz: np.ndarray) -> float:
        origin, direction = ray
        up = np.array([0.0, 1.0, 0.0])
        base = np.array([center_xz[0], self._floor_y(), center_xz[1]])
        b = float(direction @ up)
        w0 = origin - base
        d0 = float(direction @ w0)
        e0 = float(up @ w0)
        denom = 1.0 - b * b
        if abs(denom) < 1e-6:
            return 1.0
        s_line = (e0 - b * d0) / denom
        return float(np.clip(s_line, 0.2, 3.5))

    def _camera_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        view = np.asarray(self.scene.scene.camera.get_view_matrix(), dtype=float)
        cam_to_world = np.linalg.inv(view)
        right = cam_to_world[:3, 0]
        up = cam_to_world[:3, 1]
        eye = cam_to_world[:3, 3]
        return right, up, eye

    def _start_pan(self, event: gui.MouseEvent) -> None:
        right, up, eye = self._camera_basis()
        self.pan = {
            "x": float(event.x), "y": float(event.y),
            "right": right, "up": up, "eye": eye, "cor": self._cor.copy(),
        }

    def _apply_pan(self, event: gui.MouseEvent) -> None:
        pan = self.pan
        distance = max(float(np.linalg.norm(pan["cor"] - pan["eye"])), 0.5)
        fov = float(self.scene.scene.camera.get_field_of_view())
        per_pixel = 2 * distance * math.tan(math.radians(fov) / 2) / max(self.scene.frame.height, 1)
        dx = (float(event.x) - pan["x"]) * per_pixel
        dy = (float(event.y) - pan["y"]) * per_pixel
        offset = -dx * pan["right"] + dy * pan["up"]
        eye = pan["eye"] + offset
        self._cor = pan["cor"] + offset
        # keep the camera's own up vector: world-up degenerates in the top-down view
        self.scene.look_at(self._cor, eye, pan["up"])
        self.window.post_redraw()

    def _start_orbit(self, event: gui.MouseEvent) -> None:
        _, _, eye = self._camera_basis()
        offset = eye - self._cor
        radius = max(float(np.linalg.norm(offset)), 0.1)
        elevation = math.asin(float(np.clip(offset[1] / radius, -1.0, 1.0)))
        azimuth = math.atan2(float(offset[0]), float(offset[2]))
        self.orbit = {
            "x": float(event.x), "y": float(event.y),
            "radius": radius, "azimuth": azimuth, "elevation": elevation,
        }

    def _apply_orbit(self, event: gui.MouseEvent) -> None:
        orbit = self.orbit
        rate = 0.006  # radians per pixel
        azimuth = orbit["azimuth"] - (float(event.x) - orbit["x"]) * rate
        elevation = float(np.clip(
            orbit["elevation"] + (float(event.y) - orbit["y"]) * rate, -1.53, 1.53
        ))
        offset = orbit["radius"] * np.array([
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.cos(azimuth),
        ])
        # turntable orbit: horizon stays level (world up), no roll
        self.scene.look_at(self._cor, self._cor + offset, [0.0, 1.0, 0.0])
        self.window.post_redraw()

    def _top_down_view(self) -> None:
        bounds = self.scene.scene.bounding_box
        center = np.asarray(bounds.get_center())
        extent = np.asarray(bounds.get_extent())
        height = float(max(extent[0], extent[2])) * 1.1 + 2.0
        self._cor = np.array([center[0], self._floor_y(), center[2]])
        eye = self._cor + np.array([0.0, height, 0.0])
        self.scene.look_at(self._cor, eye, [0.0, 0.0, -1.0])
        self.window.post_redraw()

    def _handle_points(self, box: BinBox) -> dict[str, np.ndarray]:
        ux, _, uz = box.local_axes()
        center = np.asarray(box.center)
        ex, ey, ez = box.extent
        y_bottom = center[1] - ey / 2
        handles: dict[str, np.ndarray] = {}
        for index, (a, d) in enumerate(CORNER_SIGNS):
            corner = center + a * ux * ex / 2 + d * uz * ez / 2
            handles[f"handle_corner_{index}"] = np.array([corner[0], y_bottom + 0.05, corner[2]])
        handles["handle_top"] = center + np.array([0.0, ey / 2, 0.0])
        rotate = center + ux * (ex / 2 + 0.3)
        handles["handle_rotate"] = np.array([rotate[0], y_bottom + 0.05, rotate[2]])
        return handles

    # ---------- drawing ----------

    def _remove_geometry(self, name: str) -> None:
        if self.scene.scene.has_geometry(name):
            self.scene.scene.remove_geometry(name)

    def _redraw_boxes(self) -> None:
        for i in range(self._drawn_boxes):
            for suffix in ("line", "fill"):
                self._remove_geometry(f"box_{i}_{suffix}")
        self._drawn_boxes = len(self.boxes)

        for index, box in enumerate(self.boxes):
            color = SELECTED_COLOR if index == self.selected else STATUS_COLORS.get(box.status, (1, 0, 0))
            corners = box.corners()

            lineset = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(corners),
                o3d.utility.Vector2iVector(np.array(BOX_EDGES)),
            )
            lineset.paint_uniform_color(color)
            line_material = rendering.MaterialRecord()
            line_material.shader = "unlitLine"
            line_material.line_width = 5.0 if index == self.selected else 3.0
            self.scene.scene.add_geometry(f"box_{index}_line", lineset, line_material)

            fill = o3d.geometry.TriangleMesh.create_box(*box.extent)
            fill.translate(-np.asarray(box.extent) / 2)
            fill.rotate(box.rotation_matrix(), center=(0, 0, 0))
            fill.translate(np.asarray(box.center))
            fill.compute_vertex_normals()
            fill_material = rendering.MaterialRecord()
            fill_material.shader = "defaultLitTransparency"
            fill_material.base_color = (*color, 0.25)
            self.scene.scene.add_geometry(f"box_{index}_fill", fill, fill_material)

        self._redraw_handles()
        self._refresh_list()

    def _redraw_handles(self) -> None:
        for name in HANDLE_NAMES:
            self._remove_geometry(name)
        if self.selected is None or self.selected >= len(self.boxes):
            return
        box = self.boxes[self.selected]
        for name, position in self._handle_points(box).items():
            kind = "corner" if "corner" in name else ("top" if "top" in name else "rotate")
            sphere = o3d.geometry.TriangleMesh.create_sphere(0.05, resolution=10)
            sphere.translate(position)
            sphere.paint_uniform_color(HANDLE_COLORS[kind])
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self.scene.scene.add_geometry(name, sphere, material)

    def _refresh_list(self) -> None:
        items = []
        for index, box in enumerate(self.boxes):
            ex, ey, ez = box.extent
            items.append(f"#{index + 1} {box.bin_type} [{box.status}] {ex:.2f}x{ez:.2f}x{ey:.2f} m")
        self.box_list.set_items(items)
        if self.selected is not None and self.selected < len(items):
            self.box_list.selected_index = self.selected
        approved = sum(1 for b in self.boxes if b.status == STATUS_APPROVED)
        self.status_label.text = f"{len(self.boxes)} bokser, {approved} godkjent"

    def _draw_point_markers(self) -> None:
        for index, point in enumerate([self.draw_a, self.draw_b]):
            name = f"preview_pt_{index}"
            self._remove_geometry(name)
            if point is None:
                continue
            marker = o3d.geometry.TriangleMesh.create_sphere(0.05, resolution=10)
            marker.translate([point[0], self._floor_y() + 0.05, point[1]])
            marker.paint_uniform_color(PREVIEW_COLOR)
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self.scene.scene.add_geometry(name, marker, material)

    def _draw_preview(self, cursor_floor: np.ndarray | None, height: float | None = None) -> None:
        self._remove_geometry("preview")
        self._draw_point_markers()
        points: np.ndarray | None = None
        edges: list[tuple[int, int]] | None = None
        floor_y = self._floor_y()
        lift = 0.04  # keep preview lines above the floor mesh so it cannot occlude them

        if self.draw_stage == 1 and self.draw_a is not None and cursor_floor is not None:
            points = np.array([
                [self.draw_a[0], floor_y + lift, self.draw_a[1]],
                [cursor_floor[0], floor_y + lift, cursor_floor[2]],
            ])
            edges = [(0, 1)]
        elif self.draw_stage == 2 and cursor_floor is not None:
            rect = self._rect_from_points(np.array([cursor_floor[0], cursor_floor[2]]))
            if rect is not None:
                center_xz, ex, ez, yaw = rect
                temp = BinBox([center_xz[0], floor_y + lift, center_xz[1]], [ex, 0.01, ez], yaw)
                points = temp.corners()
                edges = BOX_EDGES
        elif self.draw_stage == 3 and self.draw_box is not None:
            box = copy.deepcopy(self.draw_box)
            if height is not None:
                box.extent[1] = height
                box.center[1] = floor_y + height / 2
            points = box.corners()
            edges = BOX_EDGES

        if points is None or edges is None:
            return
        lineset = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector2iVector(np.array(edges)),
        )
        lineset.paint_uniform_color(PREVIEW_COLOR)
        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = 4.0
        self.scene.scene.add_geometry("preview", lineset, material)
        self.window.post_redraw()

    # ---------- undo ----------

    def _push_undo(self) -> None:
        self.undo_stack.append((copy.deepcopy(self.boxes), self.selected))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _undo(self) -> None:
        if not self.undo_stack:
            self.status_label.text = "Ingenting å angre"
            return
        self.boxes, self.selected = self.undo_stack.pop()
        if self.selected is not None and self.selected >= len(self.boxes):
            self.selected = None
        self.dirty = True
        self._redraw_boxes()

    def _camera_floor_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Camera right/forward projected onto the floor plane, for screen-relative arrow moves."""
        right, up, eye = self._camera_basis()
        forward = np.cross(up, right)
        right_xz = np.array([right[0], 0.0, right[2]])
        forward_xz = np.array([forward[0], 0.0, forward[2]])
        if np.linalg.norm(forward_xz) < 1e-3:
            forward_xz = np.array([up[0], 0.0, up[2]])  # top-down: use camera-up as screen-up
        norm_r = np.linalg.norm(right_xz)
        norm_f = np.linalg.norm(forward_xz)
        right_xz = right_xz / norm_r if norm_r > 1e-6 else np.array([1.0, 0.0, 0.0])
        forward_xz = forward_xz / norm_f if norm_f > 1e-6 else np.array([0.0, 0.0, -1.0])
        return right_xz, forward_xz

    def _on_key(self, event: gui.KeyEvent) -> gui.Widget.EventCallbackResult:
        if event.key in (gui.KeyName.LEFT_CONTROL, gui.KeyName.RIGHT_CONTROL):
            self._ctrl_down = event.type == gui.KeyEvent.Type.DOWN
            return gui.Widget.EventCallbackResult.IGNORED
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED

        if event.key == gui.KeyName.ESCAPE:
            if self.mode == MODE_DRAW:
                self._cancel_draw()
                self._redraw_boxes()
            else:
                self.selected = None
                self._redraw_boxes()
            return gui.Widget.EventCallbackResult.CONSUMED
        if self._ctrl_down and event.key == gui.KeyName.Z:
            self._undo()
            return gui.Widget.EventCallbackResult.CONSUMED
        if self._ctrl_down and event.key == gui.KeyName.S:
            self._save()
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key == gui.KeyName.T and self.mode != MODE_DRAW:
            self._start_draw()
            return gui.Widget.EventCallbackResult.CONSUMED

        type_keys = {
            gui.KeyName.ONE: 0, gui.KeyName.TWO: 1,
            gui.KeyName.THREE: 2, gui.KeyName.FOUR: 3,
        }
        if event.key in type_keys and self.selected is not None:
            type_names = list(BIN_TYPES)
            index = type_keys[event.key]
            if index < len(type_names):
                self._push_undo()
                self.boxes[self.selected].bin_type = type_names[index]
                self.dirty = True
                self._redraw_boxes()
            return gui.Widget.EventCallbackResult.CONSUMED

        if self.selected is None:
            return gui.Widget.EventCallbackResult.IGNORED

        if event.key in (gui.KeyName.DELETE, gui.KeyName.BACKSPACE):
            self._delete_selected()
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key == gui.KeyName.G:
            self._approve_selected()
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key == gui.KeyName.Q:
            self._nudge(dyaw=-5.0)
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key == gui.KeyName.E:
            self._nudge(dyaw=5.0)
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key == gui.KeyName.PAGE_UP:
            self._nudge(dey=0.05)
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key == gui.KeyName.PAGE_DOWN:
            self._nudge(dey=-0.05)
            return gui.Widget.EventCallbackResult.CONSUMED

        arrow_moves = {
            gui.KeyName.RIGHT: (1, 0), gui.KeyName.LEFT: (-1, 0),
            gui.KeyName.UP: (0, 1), gui.KeyName.DOWN: (0, -1),
        }
        if event.key in arrow_moves:
            step_right, step_forward = arrow_moves[event.key]
            right_xz, forward_xz = self._camera_floor_axes()
            delta = (right_xz * step_right + forward_xz * step_forward) * 0.05
            self._nudge(dx=float(delta[0]), dz=float(delta[2]))
            return gui.Widget.EventCallbackResult.CONSUMED

        return gui.Widget.EventCallbackResult.IGNORED

    # ---------- box operations ----------

    def _on_list_selection(self, _new_value: str, _double: bool) -> None:
        self.selected = self.box_list.selected_index if self.box_list.selected_index >= 0 else None
        self._redraw_boxes()

    def _approve_selected(self) -> None:
        if self.selected is not None:
            self._push_undo()
            self.boxes[self.selected].status = STATUS_APPROVED
            self.dirty = True
            self._redraw_boxes()

    def _delete_selected(self) -> None:
        if self.selected is not None:
            self._push_undo()
            self.boxes.pop(self.selected)
            self.selected = None
            self.dirty = True
            self._redraw_boxes()

    def _retype_selected(self) -> None:
        if self.selected is not None:
            self._push_undo()
            self.boxes[self.selected].bin_type = self.type_combo.selected_text
            self.dirty = True
            self._redraw_boxes()

    def _new_standard_box(self) -> None:
        self._push_undo()
        bin_type = self.type_combo.selected_text
        ex, ey, ez = BIN_TYPES[bin_type]
        bounds = self.scene.scene.bounding_box
        center = bounds.get_center()
        floor = self._floor_y()
        box = BinBox(
            center=[float(center[0]), floor + ey / 2, float(center[2])],
            extent=[ex, ey, ez],
            yaw_deg=0.0,
            bin_type=bin_type,
            status=STATUS_APPROVED,
            source="manuell",
        )
        self.boxes.append(box)
        self.selected = len(self.boxes) - 1
        self.dirty = True
        self._redraw_boxes()

    def _nudge(
        self,
        dx: float = 0.0,
        dz: float = 0.0,
        dyaw: float = 0.0,
        dex: float = 0.0,
        dey: float = 0.0,
        dez: float = 0.0,
    ) -> None:
        if self.selected is None:
            return
        self._push_undo()
        box = self.boxes[self.selected]
        box.center[0] += dx
        box.center[2] += dz
        box.yaw_deg += dyaw
        box.extent[0] = max(0.1, box.extent[0] + dex)
        box.extent[2] = max(0.1, box.extent[2] + dez)
        new_height = max(0.2, box.extent[1] + dey)
        box.center[1] += (new_height - box.extent[1]) / 2  # keep the bottom on the floor
        box.extent[1] = new_height
        self.dirty = True
        self._redraw_boxes()

    # ---------- draw mode ----------

    def _start_draw(self) -> None:
        if not self.mesh_loaded:
            return
        self.mode = MODE_DRAW
        self.draw_stage = 0
        self.draw_a = None
        self.draw_b = None
        self.draw_box = None
        self._top_down_view()
        self.mode_label.text = "Tegner: klikk hjørne A på gulvet"

    def _cancel_draw(self) -> None:
        self.mode = MODE_NORMAL
        self.draw_stage = 0
        self.draw_a = None
        self.draw_b = None
        self.draw_box = None
        self.mode_label.text = ""
        if self.mesh_loaded:
            self._remove_geometry("preview")
            self._remove_geometry("preview_pt_0")
            self._remove_geometry("preview_pt_1")

    def _rect_from_points(self, c_xz: np.ndarray) -> tuple[np.ndarray, float, float, float] | None:
        if self.draw_a is None or self.draw_b is None:
            return None
        edge = self.draw_b - self.draw_a
        ex = float(np.linalg.norm(edge))
        if ex < 0.05:
            return None
        u = edge / ex
        normal = np.array([-u[1], u[0]])
        w = float((c_xz - self.draw_a) @ normal)
        ez = abs(w)
        if ez < 0.05:
            return None
        center_xz = self.draw_a + edge / 2 + normal * (w / 2)
        yaw = math.degrees(math.atan2(float(u[1]), float(u[0])))
        return center_xz, ex, ez, yaw

    def _mouse_draw(self, event: gui.MouseEvent) -> gui.Widget.EventCallbackResult:
        if event.type == gui.MouseEvent.Type.WHEEL:
            return gui.Widget.EventCallbackResult.IGNORED
        ray = self._mouse_ray(event)
        if ray is None:
            return gui.Widget.EventCallbackResult.CONSUMED
        floor_point = self._ray_floor(ray)

        if event.type == gui.MouseEvent.Type.MOVE:
            if time.time() - self._last_preview > 0.03:
                self._last_preview = time.time()
                self._draw_preview(floor_point)
            return gui.Widget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and floor_point is not None:
            point_xz = np.array([floor_point[0], floor_point[2]])
            if self.draw_stage == 0:
                self.draw_a = point_xz
                self.draw_stage = 1
                self.mode_label.text = "Tegner: klikk hjørne B (første kant)"
            elif self.draw_stage == 1:
                if np.linalg.norm(point_xz - self.draw_a) >= 0.05:
                    self.draw_b = point_xz
                    self.draw_stage = 2
                    self.mode_label.text = "Tegner: trykk for dybde, dra opp, slipp"
            elif self.draw_stage == 2:
                rect = self._rect_from_points(point_xz)
                if rect is not None:
                    center_xz, ex, ez, yaw = rect
                    floor = self._floor_y()
                    self.draw_box = BinBox(
                        center=[float(center_xz[0]), floor + 0.5, float(center_xz[1])],
                        extent=[ex, 1.0, ez],
                        yaw_deg=yaw,
                        bin_type=self.type_combo.selected_text,
                        status=STATUS_APPROVED,
                        source="manuell",
                    )
                    self.draw_stage = 3
                    self.mode_label.text = "Tegner: dra opp for høyde, slipp for å fullføre"
                    self._draw_preview(None, height=1.0)
            return gui.Widget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.DRAG and self.draw_stage == 3 and self.draw_box is not None:
            center_xz = np.array([self.draw_box.center[0], self.draw_box.center[2]])
            height = self._height_from_ray(ray, center_xz)
            self.draw_box.extent[1] = height
            self.draw_box.center[1] = self._floor_y() + height / 2
            if time.time() - self._last_preview > 0.03:
                self._last_preview = time.time()
                self._draw_preview(None, height=height)
            return gui.Widget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.BUTTON_UP and self.draw_stage == 3 and self.draw_box is not None:
            self._push_undo()
            self.boxes.append(self.draw_box)
            self.selected = len(self.boxes) - 1
            self.dirty = True
            self._cancel_draw()
            self._redraw_boxes()
            return gui.Widget.EventCallbackResult.CONSUMED

        return gui.Widget.EventCallbackResult.CONSUMED

    # ---------- normal mode: handles, drag, select ----------

    def _pick_handle(self, ray: tuple[np.ndarray, np.ndarray]) -> str | None:
        if self.selected is None or self.selected >= len(self.boxes):
            return None
        best_name: str | None = None
        best_distance = np.inf
        for name, position in self._handle_points(self.boxes[self.selected]).items():
            distance, along = _ray_point_distance(ray, position)
            threshold = 0.06 + 0.02 * along
            if distance < threshold and distance < best_distance:
                best_distance = distance
                best_name = name
        return best_name

    def _pick_box(self, ray: tuple[np.ndarray, np.ndarray]) -> int | None:
        best_index: int | None = None
        best_t = np.inf
        for index, box in enumerate(self.boxes):
            t = _ray_hits_box(ray, box)
            if t is not None and t < best_t:
                best_t = t
                best_index = index
        return best_index

    def _mouse_normal(self, event: gui.MouseEvent) -> gui.Widget.EventCallbackResult:
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            ray = self._mouse_ray(event)
            if ray is None:
                return gui.Widget.EventCallbackResult.IGNORED

            if event.is_modifier_down(gui.KeyModifier.CTRL):
                if self.selected is not None:
                    floor_point = self._ray_floor(ray)
                    if floor_point is not None:
                        self._push_undo()
                        box = self.boxes[self.selected]
                        box.center[0] = float(floor_point[0])
                        box.center[2] = float(floor_point[2])
                        box.center[1] = self._floor_y() + box.extent[1] / 2
                        self.dirty = True
                        self._redraw_boxes()
                return gui.Widget.EventCallbackResult.CONSUMED

            handle = self._pick_handle(ray)
            if handle is not None:
                self._push_undo()
                box = self.boxes[self.selected]
                if handle.startswith("handle_corner_"):
                    corner_index = int(handle.rsplit("_", 1)[1])
                    a, d = CORNER_SIGNS[corner_index]
                    ux, _, uz = box.local_axes()
                    center = np.asarray(box.center)
                    opposite = center - a * ux * box.extent[0] / 2 - d * uz * box.extent[2] / 2
                    self.drag = {"kind": "corner", "opposite_xz": np.array([opposite[0], opposite[2]])}
                elif handle == "handle_top":
                    self.drag = {"kind": "top"}
                else:
                    self.drag = {"kind": "rotate"}
                return gui.Widget.EventCallbackResult.CONSUMED

            hit = self._pick_box(ray)
            if hit is not None:
                if hit != self.selected:
                    self.selected = hit
                    self._redraw_boxes()
                floor_point = self._ray_floor(ray)
                box = self.boxes[hit]
                if floor_point is not None:
                    offset = np.array([box.center[0] - floor_point[0], box.center[2] - floor_point[2]])
                else:
                    offset = np.zeros(2)
                self._push_undo()
                self.drag = {"kind": "move", "offset_xz": offset}
                return gui.Widget.EventCallbackResult.CONSUMED

            self._start_orbit(event)
            return gui.Widget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.DRAG and self.orbit is not None:
            self._apply_orbit(event)
            return gui.Widget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.BUTTON_UP and self.orbit is not None:
            self.orbit = None
            return gui.Widget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.DRAG and self.drag is not None and self.selected is not None:
            ray = self._mouse_ray(event)
            if ray is None:
                return gui.Widget.EventCallbackResult.CONSUMED
            box = self.boxes[self.selected]
            kind = self.drag["kind"]

            if kind == "move":
                floor_point = self._ray_floor(ray)
                if floor_point is not None:
                    offset = self.drag["offset_xz"]
                    box.center[0] = float(floor_point[0] + offset[0])
                    box.center[2] = float(floor_point[2] + offset[1])
                    box.center[1] = self._floor_y() + box.extent[1] / 2
            elif kind == "corner":
                floor_point = self._ray_floor(ray)
                if floor_point is not None:
                    opposite = self.drag["opposite_xz"]
                    target = np.array([floor_point[0], floor_point[2]])
                    ux, _, uz = box.local_axes()
                    ux2 = np.array([ux[0], ux[2]])
                    uz2 = np.array([uz[0], uz[2]])
                    delta = target - opposite
                    dx = float(delta @ ux2)
                    dz = float(delta @ uz2)
                    box.extent[0] = max(0.15, abs(dx))
                    box.extent[2] = max(0.15, abs(dz))
                    new_center = opposite + ux2 * dx / 2 + uz2 * dz / 2
                    box.center[0] = float(new_center[0])
                    box.center[2] = float(new_center[1])
            elif kind == "top":
                center_xz = np.array([box.center[0], box.center[2]])
                height = self._height_from_ray(ray, center_xz)
                box.extent[1] = height
                box.center[1] = self._floor_y() + height / 2
            elif kind == "rotate":
                floor_point = self._ray_floor(ray)
                if floor_point is not None:
                    vx = float(floor_point[0] - box.center[0])
                    vz = float(floor_point[2] - box.center[2])
                    if abs(vx) + abs(vz) > 1e-6:
                        box.yaw_deg = math.degrees(math.atan2(vz, vx))

            self.dirty = True
            if time.time() - self._last_preview > 0.03:
                self._last_preview = time.time()
                self._redraw_boxes()
            return gui.Widget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.BUTTON_UP and self.drag is not None:
            self.drag = None
            self._redraw_boxes()
            return gui.Widget.EventCallbackResult.CONSUMED

        return gui.Widget.EventCallbackResult.IGNORED

    def _on_mouse(self, event: gui.MouseEvent) -> gui.Widget.EventCallbackResult:
        if not self.mesh_loaded:
            return gui.Widget.EventCallbackResult.IGNORED

        # right button = pan, in every mode
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.RIGHT):
            self._start_pan(event)
            return gui.Widget.EventCallbackResult.CONSUMED
        if self.pan is not None:
            if event.type == gui.MouseEvent.Type.DRAG:
                self._apply_pan(event)
                return gui.Widget.EventCallbackResult.CONSUMED
            if event.type == gui.MouseEvent.Type.BUTTON_UP:
                self.pan = None
                return gui.Widget.EventCallbackResult.CONSUMED

        if self.mode == MODE_ENTRANCE:
            return self._mouse_entrance(event)
        if self.mode == MODE_DRAW:
            return self._mouse_draw(event)
        return self._mouse_normal(event)

    # ---------- entrances (doors) ----------

    def _toggle_entrance_mode(self) -> None:
        if self.mode == MODE_ENTRANCE:
            self.mode = MODE_NORMAL
            self.mode_label.text = ""
        else:
            self._cancel_draw()
            self.mode = MODE_ENTRANCE
            self.mode_label.text = "Inngang-modus: klikk = ny dør, Ctrl+klikk = slett nærmeste"

    def _clear_entrances(self) -> None:
        self.entrances = []
        self.dirty = True
        self._redraw_entrances()

    def _redraw_entrances(self) -> None:
        for i in range(self._drawn_entrances):
            self._remove_geometry(f"entrance_{i}")
        self._drawn_entrances = len(self.entrances)
        for i, (x, z) in enumerate(self.entrances):
            sphere = o3d.geometry.TriangleMesh.create_sphere(0.13, resolution=12)
            sphere.translate([x, self._floor_y() + 0.13, z])
            sphere.paint_uniform_color([1.0, 0.1, 1.0])
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            self.scene.scene.add_geometry(f"entrance_{i}", sphere, material)

    def _mouse_entrance(self, event: gui.MouseEvent) -> gui.Widget.EventCallbackResult:
        if event.type != gui.MouseEvent.Type.BUTTON_DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        ray = self._mouse_ray(event)
        if ray is None:
            return gui.Widget.EventCallbackResult.IGNORED
        floor_point = self._ray_floor(ray)
        if floor_point is None:
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.is_modifier_down(gui.KeyModifier.CTRL) and self.entrances:
            pts = np.array(self.entrances)
            nearest = int(np.argmin(np.hypot(pts[:, 0] - floor_point[0], pts[:, 1] - floor_point[2])))
            del self.entrances[nearest]
        else:
            self.entrances.append((float(floor_point[0]), float(floor_point[2])))
        self.dirty = True
        self._redraw_entrances()
        return gui.Widget.EventCallbackResult.CONSUMED

    # ---------- persistence ----------

    def _save(self) -> None:
        ENTRANCE_DIR.mkdir(parents=True, exist_ok=True)
        (ENTRANCE_DIR / f"{self._current_zip().stem}.json").write_text(
            json.dumps({"entrances_xz": [[x, z] for x, z in self.entrances]}, indent=2),
            encoding="utf-8",
        )
        if not self.boxes and not self.dirty and not self._annotation_path().exists():
            return
        save_annotations(
            self._annotation_path(), self._current_zip().name, self.floor_height, self.boxes
        )
        self.dirty = False
        approved = sum(1 for b in self.boxes if b.status == STATUS_APPROVED)
        self.status_label.text = (
            f"Lagret: {len(self.boxes)} bokser, {approved} godkjent, {len(self.entrances)} inngang"
        )

    def _on_close(self) -> bool:
        self._save()
        if self.worker is not None and self.worker.poll() is None:
            self.worker.terminate()
        return True

    def run(self) -> None:
        gui.Application.instance.run()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Søppelrom 3D-annotering")
    parser.add_argument("--scan", default=None, help="stem eller sti til skannet som skal åpnes først")
    args = parser.parse_args()

    app = AnnotationApp()
    if args.scan:
        target = Path(args.scan).stem
        stems = [s.stem for s in app.scans]
        if target in stems:
            app.scan_index = stems.index(target)
            app._load_scan()
    app.run()


if __name__ == "__main__":
    main()
