"""Interactive 3D placement viewer.

Browse scans and SEE, on the room mesh: the push-path a large bin can be wheeled along to a door
(blue floor), the free bin space around it (green), suggested new bins (green boxes), existing bins
(red boxes) and entrances (magenta). Read-only sibling of the annotation tool — same drag-to-orbit
controls and Prev/Next scan navigation.

    .venv\\Scripts\\python.exe -m src.place3d
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from . import pipeline
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
CAND_COLOR = (0.10, 0.90, 0.10)      # green box = suggested new bin
EXIST_COLOR = (1.00, 0.10, 0.10)     # red box  = existing bin
ENTRANCE_COLOR = (1.00, 0.10, 1.00)  # magenta sphere = entrance


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

        self.stats_label = gui.Label("")
        self.panel.add_child(self.stats_label)
        self.panel.add_child(gui.Label(
            "\nBlått gulv = skyve-sti for stor kasse\n"
            "Grønt gulv = ledig plass\n"
            "Rødt gulv = opptatt\n"
            "Grønn boks = forslag til ny kasse\n"
            "Rød boks = eksisterende kasse\n"
            "Rosa kule = inngang\n"
        ))
        self.panel.add_child(gui.Label("Dra = roter · scroll = zoom\nPil venstre/høyre = bytt skann"))

        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)
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

    def _floor_overlay(self, scene) -> o3d.geometry.PointCloud | None:
        """Colored dots just above the floor: push-path (blue), free (green), occupied (red)."""
        fs = scene.fs
        cell, origin = fs.cell, fs.origin
        rows, cols = fs.free.shape
        reach = scene.result.reachable if scene.result.reachable is not None else np.zeros((rows, cols), bool)
        route = scene.result.route if scene.result.route is not None else np.zeros((rows, cols), bool)
        yy, xx = np.mgrid[0:rows, 0:cols]
        wx = origin[0] + (xx + 0.5) * cell
        wz = origin[1] + (yy + 0.5) * cell
        y = scene.floor_height + 0.03

        layers = [
            (fs.occupied & fs.floor_observed, OCC_COLOR),     # red = occupied floor
            (fs.free & ~reach, FREE_COLOR),                   # green = free but off the path
            (reach & ~route, PATH_COLOR),                     # blue = push-path corridor (subset of free)
            (route, ROUTE_COLOR),                             # bright = the near-straight route line
        ]
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

    def _load(self) -> None:
        stem = self.scans[self.index]
        self.scan_label.text = f"Skann {self.index + 1}/{len(self.scans)}:\n{stem}"
        self.stats_label.text = "Beregner … (kan ta et par sekunder)"
        try:
            scene = pipeline.compute_scene(stem, self.bin_type)
        except Exception as error:  # noqa: BLE001 - surface any failure in the panel
            self.scene.scene.clear_geometry()
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

        overlay = self._floor_overlay(scene)
        if overlay is not None:
            omat = rendering.MaterialRecord()
            omat.shader = "defaultUnlit"
            omat.point_size = 7.0
            self.scene.scene.add_geometry("floor_overlay", overlay, omat)

        line_mat = rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 4.0
        floor = scene.floor_height
        for i, (bx, bz, bl, bw, byaw) in enumerate(scene.existing):
            box = _bin_box_lineset(((bx, bz), (bl, bw), byaw), floor, floor + 1.2, EXIST_COLOR)
            self.scene.scene.add_geometry(f"exist_{i}", box, line_mat)
        for i, cand in enumerate(scene.result.candidates):
            cand_height = BIN_TYPES.get(cand.bin_type, BIN_TYPES[self.bin_type])[1]  # per-bin height
            box = _bin_box_lineset(cand.rect, floor, floor + cand_height, CAND_COLOR)
            self.scene.scene.add_geometry(f"cand_{i}", box, line_mat)

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

        if scene.enclosed:
            self.stats_label.text = "⚠ INNESPERRET rom (dør lukket i scan) — hoppet over"
        else:
            self.stats_label.text = (
                f"{len(scene.result.candidates)} nye plasser  ·  {len(scene.existing)} eksisterende  ·  "
                f"ledig gulv {scene.fs.free_area_m2:.1f} m²"
            )
        self.window.post_redraw()

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
