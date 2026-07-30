"""3D annotation tool: review, create and delete bin boxes on top of the Poisson mesh.

Usage:  .venv\\Scripts\\python.exe -m src.annotate3d

BACKDROP: Polycam's own export is drawn instead of our Poisson mesh when it is registered and
passes src/ply_align.py's quality gate (toggle with B; the panel shows the median deviation). This
is deliberately conservative, because the boxes saved here are the ground truth every model trains
on: a backdrop that is 20 cm out of place would silently shift them. Nothing geometric changes with
the backdrop — the floor height, the click-to-floor raycast, box placement, the fixed-size place
mode and the camera framing all keep using our own mesh and the pipeline's cached floor_height. The
Polycam cloud is scenery only, loaded in OUR RAW frame (the one mesh_poisson.ply, the boxes and the
entrances live in), so it cannot move a box even when it is switched on and off mid-session.

CAD-style interactions:
  - "Tegn boks": click corner A on the floor, click corner B (first edge), then PRESS for
    the depth point, DRAG upward to pull the box out of the floor, RELEASE to finish.
    ESC cancels.
  - Selected box shows handles: drag a yellow bottom corner to resize the footprint
    (opposite corner stays fixed), drag the blue top sphere to change height, drag the
    magenta sphere to rotate, drag on the box body to move it. Click a box to select it.
  - Plain drag on empty space orbits the camera as usual; Ctrl+click teleports the
    selected box to the clicked floor point.
  - "Plasser boks" (P): click a floor point to drop a correctly-sized box for the chosen
    bin type — no drawing needed, since each bin type has a fixed real-world size. Hold R and
    move the mouse to rotate the box before dropping it.
  - Ctrl+C / Ctrl+V copy and paste the selected box (the paste lands beside the original).

A background worker process keeps up to 5 scans prepared ahead while you annotate.

PANEL: every size is a multiple of window.theme.font_size (see src/uitheme.py), so the side panel
and its buttons follow the DPI / font size instead of clipping. The panel is a gui.ScrollableVert
with collapsible sections, because a plain gui.Vert squeezes its children into each other once the
content is taller than the window — that is what made the buttons overlap unless the window was
maximised. Buttons sit in gui.VGrid rows, which split the panel into equal columns, so a long label
can no longer push a button past the panel edge. Colours come from src/uitheme.py (which derives
them from src/style.py) so a colour means the same here as in the rendered previews.
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

from . import backdrop, meshcull
from . import uitheme as T
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

# Box colours come from the shared palette (src/uitheme.py -> src/style.py) so that a colour means
# the same thing here, in place3d.py and in the rendered PNG previews. The MEANINGS are unchanged:
# oransje/gul = forslag, grønn = godkjent, blå = valgt.
STATUS_COLORS = {
    STATUS_PROPOSED: T.rgb_of("warning"),
    STATUS_APPROVED: T.rgb_of("new_bin"),
}
SELECTED_COLOR = T.rgb_of("path")
# All four palette hues are spoken for (grønn/rød/blå/magenta), so the "drawing right now" line
# keeps a cyan of its own. It only exists between two clicks, never next to a saved box.
PREVIEW_COLOR = (0.1, 0.9, 0.9)
HANDLE_COLORS = {
    "corner": T.rgb_of("dimension"),   # gult = størrelse, as the help text says
    "top": T.rgb_of("path"),           # blått = høyde
    "rotate": T.rgb_of("entrance"),    # rosa = roter
}
CORNER_SIGNS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
HANDLE_NAMES = [f"handle_corner_{i}" for i in range(4)] + ["handle_top", "handle_rotate"]

MODE_NORMAL = "normal"
MODE_DRAW = "draw"
MODE_ENTRANCE = "entrance"
MODE_PLACE = "place"

# How far above/below the scan's floor_height a box's base may be set by hand. +-3 m is not generous
# for its own sake: the four scans whose stored floor_height was actually the CEILING were off by
# 2.7-3.7 m, so the range has to cover a box that needs rescuing from the roof. Fine steps come from
# the Y-/Y+ buttons, not from dragging a 6 m slider.
BASE_SLIDER_MIN_M = -3.0
BASE_SLIDER_MAX_M = 3.0

# Name + palette role per mode. The mode decides what the next click does, so it gets a coloured
# chip in the panel instead of hiding in a line of grey text.
MODE_STYLE: dict[str, tuple[str, str]] = {
    MODE_NORMAL: ("Rediger", "text_muted"),
    MODE_DRAW: ("Tegner boks", "dimension"),
    MODE_PLACE: ("Plasserer boks", "path"),
    MODE_ENTRANCE: ("Setter inngang", "entrance"),
}


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
    def __init__(self, width: int = 1500, height: int = 950) -> None:
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
        self.clipboard: BinBox | None = None
        self._hover_xz: tuple[float, float] | None = None
        self._ctrl_down = False
        self._r_down = False
        # Guards the height slider against its own set_on_value_changed: assigning double_value fires
        # the callback, which would write the value back into the box on every redraw.
        self._syncing_base = False
        self.base_slider = None
        self.base_label = None
        self._place_yaw = 0.0                                  # rotation for boxes dropped in place mode
        self._place_anchor_xz: tuple[float, float] | None = None  # frozen centre while rotating with R
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
        # backdrop preference, remembered across scans but only honoured where the gate passes.
        # self._mesh stays OUR mesh whatever is on screen: it is what _estimate_floor and the
        # camera framing read, and the Polycam cloud must never feed either.
        self.use_polycam = True
        self._backdrop: backdrop.Backdrop | None = None
        # The see-through-walls cull lives in meshcull, shared with place3d. It used to be inline here
        # and only here, which is why the placement viewer drew rooms as closed boxes.
        self._culler: meshcull.BackfaceCuller | None = None
        self._last_cull_time = 0.0

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "Søppelrom 3D-annotering", int(width), int(height)
        )

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background(T.rgba("scene_bg"))
        self.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.scene.set_on_mouse(self._on_mouse)
        self.scene.set_on_key(self._on_key)
        self.window.add_child(self.scene)

        self._build_panel()

        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        if hasattr(self.window, "set_on_tick_event"):
            self.window.set_on_tick_event(self._on_tick)

        self.worker = start_background_worker()
        self._load_scan()

    # ---------- panel widgets (every size in em, so the panel follows DPI / font size) ----------

    def _text(self, text: str, role: str = "text") -> gui.Label:
        label = gui.Label(text)
        label.text_color = T.gui_color(role)
        return label

    def _section(self, title: str, *, expanded: bool = True) -> gui.CollapsableVert:
        """A collapsible group. Collapsing is what keeps the long help text from pushing the
        buttons out of sight on a short window."""
        section = gui.CollapsableVert(
            title,
            T.emf(self.window, 0.3),
            T.margins(self.window, left=0.7, top=0.15, right=0.1, bottom=0.4),
        )
        section.set_is_open(expanded)
        return section

    def _button(self, text: str, callback, tooltip: str = "", role: str | None = None) -> gui.Button:
        button = gui.Button(text)
        # padding in em as well: a fixed pixel padding is exactly what clipped the labels at 150%
        button.horizontal_padding_em = 0.4
        button.vertical_padding_em = 0.3
        button.set_on_clicked(callback)
        if tooltip:
            button.tooltip = tooltip
        if role is not None:
            button.background_color = T.gui_color(role)
        return button

    def _grid(self, columns: int) -> gui.VGrid:
        """Buttons go in a VGrid, never a Horiz: the grid splits the panel into equal columns and
        gives every child exactly one column, so a button can no longer overflow the panel edge
        (which is what the old Horiz rows did — they kept their preferred width and overlapped)."""
        return gui.VGrid(columns, T.emf(self.window, 0.3))

    def _gap(self, factor: float = 0.5) -> None:
        """Vertical breathing room between blocks. Colour cannot do this job: only ScrollableVert
        and Button paint background_color in Open3D 0.19 (verified), so a Vert 'card' with its own
        surface colour is not available — spacing and type carry the hierarchy instead."""
        self.panel.add_fixed(T.emf(self.window, factor))

    def _legend_row(self, role: str, text: str) -> gui.Label:
        """Legend entry. The colour itself is the swatch: there is no paintable rectangle widget,
        and a block glyph would risk a missing character in Open3D's built-in font."""
        return self._text(text, role)

    def _build_panel(self) -> None:
        # ScrollableVert: a long panel now scrolls instead of squeezing its children into each
        # other. In a plain Vert, content taller than the window made the rows overlap, which is
        # exactly why the buttons were unreachable unless the window was maximised.
        self.panel = gui.ScrollableVert(T.emf(self.window, 0.35), T.margins(self.window, 0.6))
        self.panel.background_color = T.gui_color("panel_bg")

        # -- header: which scan, and how to move between them ------------------------------
        self._scan_counter = self._text("", "text_muted")
        self.panel.add_child(self._scan_counter)
        self.scan_label = self._text("", "text")
        self.panel.add_child(self.scan_label)
        nav = self._grid(2)
        nav.add_child(self._button("< Forrige", lambda: self._switch_scan(-1),
                                   "Forrige skann (lagrer først)"))
        nav.add_child(self._button("Neste >", lambda: self._switch_scan(1),
                                   "Neste skann (lagrer først)"))
        self.panel.add_child(nav)
        # Skipping past the finished ones is the difference between 322 clicks and 216: stepping one at
        # a time walks through every scan already done, and they are scattered through the list rather
        # than gathered at the front.
        self.panel.add_child(self._button("Neste uannoterte (N)", self._next_unannotated,
                                          "Hopp til neste skann uten lagrede annoteringer"))
        self._todo_label = self._text("", "text_muted")
        self.panel.add_child(self._todo_label)
        self._gap(0.6)

        # -- mode line: what the next click will do -----------------------------------------
        self._mode_name = self._text("", "text_muted")
        self.panel.add_child(self._mode_name)
        self.mode_label = self._text("", "text_muted")
        self.panel.add_child(self.mode_label)
        self._gap(0.3)
        self.status_label = self._text("", "text_muted")
        self.panel.add_child(self.status_label)
        self._gap(0.3)

        # -- boxes ---------------------------------------------------------------------------
        boxes = self._section("Bokser")
        self.box_list = gui.ListView()
        # Bounded height: an unbounded ListView claims a huge preferred height and would push the
        # buttons below the fold again. 8 rows fit even a 700 px tall window; longer lists scroll.
        self.box_list.set_max_visible_items(8)
        self.box_list.set_on_selection_changed(self._on_list_selection)
        boxes.add_child(self.box_list)

        approve_row = self._grid(2)
        approve_row.add_child(self._button("Godkjenn (G)", self._approve_selected,
                                           "Marker valgt boks som godkjent", "accent"))
        approve_row.add_child(self._button("Slett (Del)", self._delete_selected,
                                           "Slett valgt boks"))
        boxes.add_child(approve_row)

        edit_row = self._grid(2)
        edit_row.add_child(self._button("Angre (Ctrl+Z)", self._undo, "Angre siste endring"))
        edit_row.add_child(self._button("Lagre (Ctrl+S)", self._save,
                                        "Lagre nå (skjer også automatisk ved skannbytte)"))
        boxes.add_child(edit_row)

        boxes.add_child(self._text("Kassetype", "text_muted"))
        self.type_combo = gui.Combobox()
        for name in BIN_TYPES:
            self.type_combo.add_item(name)
        # in a Horiz with a stretch the combobox keeps its natural width, so the drop-down arrow
        # stays next to the text instead of being flung out to the panel edge
        combo_row = gui.Horiz(0)
        combo_row.add_child(self.type_combo)
        combo_row.add_stretch()
        boxes.add_child(combo_row)

        new_row = self._grid(2)
        new_row.add_child(self._button("Tegn boks (T)", self._start_draw,
                                       "Tegn en boks fra to gulvpunkter + høyde"))
        new_row.add_child(self._button("Plasser (P)", self._toggle_place,
                                       "Plasser boks av/på: klikk på gulvet for ferdig størrelse"))
        boxes.add_child(new_row)
        type_row = self._grid(2)
        type_row.add_child(self._button("Sett type", self._retype_selected,
                                        "Gi valgt boks kassetypen over (eller 1-4)"))
        type_row.add_child(self._text("", "text_muted"))   # keeps the button one column wide
        boxes.add_child(type_row)
        self.panel.add_child(boxes)

        # -- height of the selected box ------------------------------------------------------
        # Its own section, above the nudge grid, because getting the height right is the thing that
        # kept going wrong. One floor_height per scan cannot describe a sloping yard or a scan whose
        # floor plane was fitted to a ramp, and every failure mode was the same shape: the box was in
        # the right place until something re-seated it. So the height is now set here, by hand, and
        # nothing else moves it.
        height_sec = self._section("Høyde over bakken")
        height_sec.add_child(self._text("0 = gulvet slik analysen fant det.\n"
                                        "Dra hvis gulvet er feil.", "text_muted"))
        self.base_slider = gui.Slider(gui.Slider.DOUBLE)
        self.base_slider.set_limits(BASE_SLIDER_MIN_M, BASE_SLIDER_MAX_M)
        self.base_slider.set_on_value_changed(self._on_base_slider)
        height_sec.add_child(self.base_slider)
        self.base_label = self._text("ingen boks valgt", "text_muted")
        height_sec.add_child(self.base_label)
        base_row = self._grid(3)
        base_row.add_child(self._button("Y- (5 cm)", lambda: self._nudge_base(-0.05),
                                        "Senk boksen 5 cm"))
        base_row.add_child(self._button("Y+ (5 cm)", lambda: self._nudge_base(0.05),
                                        "Hev boksen 5 cm"))
        base_row.add_child(self._button("På gulvet", self._seat_on_floor,
                                        "Sett boksen ned på gulvhøyden analysen fant"))
        height_sec.add_child(base_row)
        self.panel.add_child(height_sec)

        # -- nudge the selected box ----------------------------------------------------------
        nudge = self._section("Finjuster valgt")
        nudge.add_child(self._text("5 cm / 5° per klikk\nH = høyde, L = lengde, B = bredde",
                                   "text_muted"))
        move_grid = self._grid(4)
        for text, fn in [
            ("X-", lambda: self._nudge(dx=-0.05)), ("X+", lambda: self._nudge(dx=0.05)),
            ("Z-", lambda: self._nudge(dz=-0.05)), ("Z+", lambda: self._nudge(dz=0.05)),
            ("Rot-", lambda: self._nudge(dyaw=-5.0)), ("Rot+", lambda: self._nudge(dyaw=5.0)),
            ("H-", lambda: self._nudge(dey=-0.05)), ("H+", lambda: self._nudge(dey=0.05)),
            ("L-", lambda: self._nudge(dex=-0.05)), ("L+", lambda: self._nudge(dex=0.05)),
            ("B-", lambda: self._nudge(dez=-0.05)), ("B+", lambda: self._nudge(dez=0.05)),
        ]:
            move_grid.add_child(self._button(text, fn))
        nudge.add_child(move_grid)
        self.panel.add_child(nudge)

        # -- entrances -----------------------------------------------------------------------
        entrance = self._section("Inngang")
        entrance_row = self._grid(2)
        entrance_row.add_child(self._button("Inngang av/på", self._toggle_entrance_mode,
                                            "Klikk = ny dør, Ctrl+klikk = slett nærmeste"))
        entrance_row.add_child(self._button("Nullstill", self._clear_entrances,
                                            "Fjern alle innganger i dette skannet"))
        entrance.add_child(entrance_row)
        self.panel.add_child(entrance)

        # -- view ----------------------------------------------------------------------------
        view = self._section("Visning")
        self.polycam_check = gui.Checkbox("Polycam-sky (B)")
        self.polycam_check.checked = True
        self.polycam_check.tooltip = ("Vis Polycams egen eksport i stedet for vårt Poisson-mesh. "
                                      "Kun bakgrunn — geometrien er den samme.")
        self.polycam_check.set_on_checked(self._set_polycam)
        view.add_child(self.polycam_check)
        self.cull_checkbox = gui.Checkbox("Skjul veggbaksider / himling")
        self.cull_checkbox.checked = True
        self.cull_checkbox.tooltip = "Dukkehus-visning: se ned i rommet uten tak og bakvegger"
        self.cull_checkbox.set_on_checked(self._on_cull_changed)
        view.add_child(self.cull_checkbox)
        self.backdrop_label = self._text("", "text_muted")
        view.add_child(self.backdrop_label)
        self.panel.add_child(view)

        # -- legend: exactly the meanings the PNG previews use -------------------------------
        legend = self._section("Fargekoder")
        legend.add_child(self._legend_row("warning", "Gul boks = forslag"))
        legend.add_child(self._legend_row("new_bin", "Grønn boks = godkjent"))
        legend.add_child(self._legend_row("path", "Blå boks = valgt"))
        legend.add_child(self._legend_row("entrance", "Rosa kule = inngang (dør)"))
        cyan = self._text("Cyan = tegnes nå")
        cyan.text_color = gui.Color(*PREVIEW_COLOR)
        legend.add_child(cyan)
        legend.add_child(self._legend_row("dimension", "Gult håndtak = størrelse"))
        self.panel.add_child(legend)

        # -- help: collapsed by default so it can never push the buttons off screen ----------
        help_section = self._section("Hjelp og hurtigtaster", expanded=False)
        for heading, lines in [
            ("Mus", ["Venstre-dra: orbit · Høyre-dra: pan",
                     "Klikk på boks: velg den",
                     "Ctrl+klikk: flytt valgt boks hit",
                     "ESC: avbryt / opphev valg"]),
            ("Tegn boks (T)", ["Starter i topdown.",
                               "Klikk A, klikk B, trykk + dra opp, slipp."]),
            ("Plasser boks (P)", ["Klikk på gulvet — ferdig størrelse.",
                                  "Hold R + flytt musa for å rotere.",
                                  "P eller ESC avslutter."]),
            ("Håndtak på valgt boks", ["Gult = størrelse, blått = høyde,",
                                       "rosa = roter. Dra i boksen = flytt."]),
            ("Tastatur (valgt boks)", ["Del = slett · G = godkjenn",
                                       "Piltaster = flytt · Q/E = roter",
                                       "PgUp/PgDn = boksens høyde",
                                       "Home/End = hev/senk boksen",
                                       "1-4 = type",
                                       "Ctrl+C / Ctrl+V = kopier / lim inn",
                                       "Ctrl+Z = angre · Ctrl+S = lagre"]),
            ("Skann", ["N = neste uten annoteringer"]),
            ("Visning", ["B = Polycam-sky eller egen bakgrunn"]),
        ]:
            help_section.add_child(self._text(heading, "text"))
            for line in lines:
                help_section.add_child(self._text(line, "text_muted"))
        self.panel.add_child(help_section)

        self._refresh_mode_banner()

    # ---------- layout ----------

    def _panel_width(self, available_width: int) -> int:
        """Panel width in em, clamped against the window: never wider than 45 % of a narrow window
        (so the 3D view keeps a usable area) and never narrower than 12 em (so nothing clips)."""
        ideal = T.em(self.window, 21)   # wide enough for a full list row at the default font size
        lower = T.em(self.window, 12)
        return max(lower, min(ideal, max(lower, int(available_width * 0.45))))

    def _layout_frames(self, rect) -> tuple[gui.Rect, gui.Rect]:
        """Scene + panel rects for a content rect. Pure, so it can be checked with synthetic
        window sizes without clicking anything."""
        width = max(int(rect.width), 0)
        # Leave the scene at least 1 px: below ~12 em of window width the 12 em panel floor would
        # otherwise claim everything and the SceneWidget would be laid out zero-wide.
        panel_width = min(self._panel_width(width), max(width - 1, 0))
        scene_width = max(width - panel_width, 0)
        scene = gui.Rect(rect.x, rect.y, scene_width, rect.height)
        panel = gui.Rect(rect.x + scene_width, rect.y, panel_width, rect.height)
        return scene, panel

    def _on_layout(self, ctx) -> None:
        scene_rect, panel_rect = self._layout_frames(self.window.content_rect)
        self.scene.frame = scene_rect
        self.panel.frame = panel_rect

    # ---------- panel state ----------

    def _refresh_mode_banner(self) -> None:
        """Mirror self.mode into the coloured mode line. Called from every place that changes the
        mode, so the panel can never claim a mode the app is not in."""
        name, role = MODE_STYLE.get(self.mode, MODE_STYLE[MODE_NORMAL])
        self._mode_name.text = f"MODUS: {name.upper()}"
        self._mode_name.text_color = T.gui_color(role)

    def _status(self, text: str, role: str = "text_muted") -> None:
        self.status_label.text = text
        self.status_label.text_color = T.gui_color(role)

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
        self._backdrop = None
        self.backdrop_label.text = ""
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
        self._scan_counter.text = f"SKANN {self.scan_index + 1} AV {len(self.scans)}"
        self.scan_label.text = zip_path.stem
        self._update_todo_label()

        if not is_prepared(zip_path):
            self._status("Forbereder i bakgrunnen ...\n(lastes automatisk når klar)", "warning")
            return

        mesh = o3d.io.read_triangle_mesh(str(self._cache_dir() / "mesh_poisson.ply"))
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        self._mesh = mesh
        self._mesh_material = material
        self._culler = meshcull.BackfaceCuller(mesh)
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
        else:
            # Sanity-check a STORED floor height against the mesh. Files written before the
            # ceiling-vs-floor fix can carry the CEILING as the floor (measured: 4 scans off by
            # 2.7-3.7 m, e.g. Skjelderups gate 14B stored 1.87 where the floor is -1.17). Every box
            # and entrance is then drawn at ceiling height and nothing can be placed on the ground.
            # The mesh estimate (mode of the lower band) is independent and was within 0.01-0.24 m of
            # the truth on exactly those scans, so a large disagreement means the stored value is
            # wrong, not the mesh.
            estimated = _estimate_floor(mesh)
            if abs(self.floor_height - estimated) > 0.60:
                print(f"[annoter] {self._current_zip().stem}: lagret gulvhøyde "
                      f"{self.floor_height:.2f} m avviker {abs(self.floor_height - estimated):.2f} m "
                      f"fra meshet — bruker {estimated:.2f} m i stedet", flush=True)
                self.floor_height = estimated
                self.dirty = True   # so the corrected height is saved with the annotations
        self.entrances = load_entrances(self._current_zip().stem)
        self._status(f"{len(self.boxes)} bokser ({source})")

        # Polycam's export as the backdrop, but only when the gate passes. gravity_rotation is left
        # at None on purpose: this tool draws mesh_poisson.ply unrotated and stores boxes and
        # entrances in that same raw frame, which is exactly the frame ply_align registers into.
        self._backdrop = backdrop.load(self._current_zip().stem, floor_height=self.floor_height)
        if self._backdrop.available:
            cloud_material = backdrop.material()
            self.scene.scene.add_geometry("polycam", self._backdrop.cloud, cloud_material)
            self.scene.scene.add_geometry("polycam_low", self._backdrop.dollhouse, cloud_material)

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
        self._apply_backdrop()

    def _switch_scan(self, step: int) -> None:
        self._save()
        self.scan_index = (self.scan_index + step) % len(self.scans)
        self._load_scan()

    def _is_annotated(self, index: int) -> bool:
        return (ANNOTATION_DIR / f"{self.scans[index].stem}.json").exists()

    def _next_unannotated(self) -> None:
        """Forward to the next scan with no saved annotations, wrapping once.

        Deliberately checks the file on disk each time instead of caching a to-do list: _save() writes
        one the moment you leave a scan, and prepare_scan --watch is often adding more in the
        background, so a list built at startup would be wrong within minutes.
        """
        self._save()
        total = len(self.scans)
        for offset in range(1, total + 1):
            index = (self.scan_index + offset) % total
            if not self._is_annotated(index):
                if index == self.scan_index:
                    break                     # wrapped all the way round to where we started
                self.scan_index = index
                self._load_scan()
                return
        self._status("Alle skann er annotert", "success")

    def _update_todo_label(self) -> None:
        done = sum(1 for i in range(len(self.scans)) if self._is_annotated(i))
        left = len(self.scans) - done
        self._todo_label.text = (f"{done} annotert · {left} igjen" if left
                                 else f"alle {done} annotert")

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

    # ---------- backdrop: Polycam's own export vs our Poisson mesh ----------

    def _showing_polycam(self) -> bool:
        return bool(self._backdrop is not None and self._backdrop.available and self.use_polycam)

    def _show(self, name: str, visible: bool) -> None:
        if self.scene.scene.has_geometry(name):
            self.scene.scene.show_geometry(name, bool(visible))

    def _set_polycam(self, wanted: bool) -> None:
        self.use_polycam = bool(wanted)
        if not self.use_polycam:
            self._update_culling(force=True)   # our mesh may be stale after being hidden
        self._apply_backdrop()

    def _on_cull_changed(self, _checked: bool) -> None:
        """One checkbox, two mechanisms: backface culling on our one-sided mesh, and the ceiling
        crop on the Polycam cloud (points have no facing to cull)."""
        self._update_culling(force=True)
        self._apply_backdrop()

    def _apply_backdrop(self) -> None:
        """Show exactly one backdrop and name it in the panel. Visibility only — both geometries
        are added once per scan, so toggling reloads nothing and registers nothing."""
        polycam = self._showing_polycam()
        self._show("room_mesh", not polycam)
        self._show("polycam", polycam and not self.cull_checkbox.checked)
        self._show("polycam_low", polycam and self.cull_checkbox.checked)
        self.backdrop_label.text = (backdrop.status_text(self._backdrop, polycam)
                                    if self._backdrop is not None else "")
        self.window.post_redraw()

    def _update_culling(self, force: bool = False) -> bool:
        """Hide mesh triangles facing away from the camera — see meshcull for why and how."""
        if self._culler is None:
            return False
        if self._showing_polycam():
            # our mesh is hidden anyway; rebuilding it on every camera move would cost for nothing
            return False
        if not self.cull_checkbox.checked:
            if self._culler._last_eye is None:
                return False
            self._culler.reset()
            self._remove_geometry("room_mesh")
            self.scene.scene.add_geometry("room_mesh", self._culler.mesh, self._mesh_material)
            self.window.post_redraw()
            return True

        _, _, eye = self._camera_basis()
        culled = self._culler.culled_for(eye, force=force)
        if culled is None:
            return False
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
        # frame on the fixed room mesh, not scene.bounding_box: the latter grows as boxes/handles
        # are added, so re-entering place mode kept zooming further and further out.
        bounds = (self._mesh.get_axis_aligned_bounding_box()
                  if self._mesh is not None else self.scene.scene.bounding_box)
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
            # The fallback is for an unknown status only; take it from the palette too, so even that
            # case cannot introduce a pure red that no legend explains.
            color = (SELECTED_COLOR if index == self.selected
                     else STATUS_COLORS.get(box.status, T.rgb_of("danger")))
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
        self._sync_height_ui()
        approved = sum(1 for b in self.boxes if b.status == STATUS_APPROVED)
        # green only when there is nothing left to approve — the count is the progress indicator
        done = bool(self.boxes) and approved == len(self.boxes)
        self._status(f"{len(self.boxes)} bokser, {approved} godkjent",
                     "success" if done else "text_muted")

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
            self._status("Ingenting å angre", "warning")
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
        if event.key == gui.KeyName.R:
            # hold R in place mode: freeze the box where it is and rotate it by moving the mouse
            self._r_down = event.type == gui.KeyEvent.Type.DOWN
            self._place_anchor_xz = self._hover_xz if (self._r_down and self.mode == MODE_PLACE) else None
            return gui.Widget.EventCallbackResult.IGNORED
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED

        if event.key == gui.KeyName.ESCAPE:
            if self.mode != MODE_NORMAL:
                self._cancel_draw()  # leaves draw/place/entrance mode and clears any preview
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
        if self._ctrl_down and event.key == gui.KeyName.C:
            self._copy_selected()
            return gui.Widget.EventCallbackResult.CONSUMED
        if self._ctrl_down and event.key == gui.KeyName.V:
            self._paste()
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key == gui.KeyName.T and self.mode != MODE_DRAW:
            self._start_draw()
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key == gui.KeyName.P:
            self._toggle_place()
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key == gui.KeyName.B:
            # setting .checked does not fire the handler, so drive the change ourselves
            self.polycam_check.checked = not self.polycam_check.checked
            self._set_polycam(self.polycam_check.checked)
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
        if event.key == gui.KeyName.N:
            self._next_unannotated()
            return gui.Widget.EventCallbackResult.CONSUMED
        # Height of the box off the ground, the thing floor_height kept getting wrong. Home/End rather
        # than another modifier: both hands stay where they are while stepping a box up or down.
        if event.key == gui.KeyName.HOME:
            self._nudge_base(0.05)
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.key == gui.KeyName.END:
            self._nudge_base(-0.05)
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

    def _place_box_at(self, x: float, z: float) -> None:
        """Drop a fixed-size box on the floor — the chosen bin type's real-world dimensions,
        so a 2-/4-wheel bin needs no drawing at all, just a click for where it stands."""
        self._push_undo()
        bin_type = self.type_combo.selected_text
        ex, ey, ez = BIN_TYPES[bin_type]
        box = BinBox(
            center=[float(x), self._floor_y() + ey / 2, float(z)],
            extent=[ex, ey, ez],
            yaw_deg=self._place_yaw,
            bin_type=bin_type,
            status=STATUS_APPROVED,
            source="manuell",
        )
        self.boxes.append(box)
        self.selected = len(self.boxes) - 1
        self.dirty = True
        self._redraw_boxes()

    def _copy_selected(self) -> None:
        if self.selected is not None and self.selected < len(self.boxes):
            self.clipboard = copy.deepcopy(self.boxes[self.selected])
            self._status(f"Kopiert: {self.clipboard.bin_type} (Ctrl+V for å lime inn)", "text")

    def _paste(self) -> None:
        if self.clipboard is None:
            self._status("Utklippstavlen er tom (Ctrl+C for å kopiere valgt boks)", "warning")
            return
        self._push_undo()
        box = copy.deepcopy(self.clipboard)
        if self._hover_xz is not None:               # drop the copy under the mouse pointer
            box.center[0], box.center[2] = self._hover_xz
        else:                                        # no hover yet -> beside the original
            box.center[0] += box.extent[0] / 2 + 0.2
        # The copy keeps the original's height. You copied that box because it was sitting right, and
        # dropping the copy onto floor_height would throw away the one thing you had just fixed.
        box.source = "manuell"
        self.boxes.append(box)
        self.selected = len(self.boxes) - 1
        self.dirty = True
        self._redraw_boxes()

    # ---------- height of the selected box ----------

    def _base_of(self, box: BinBox) -> float:
        """Bottom of the box, relative to the scan's floor_height. 0 means sitting on it."""
        return (box.center[1] - box.extent[1] / 2) - self._floor_y()

    def _set_base(self, box: BinBox, base: float) -> None:
        base = max(BASE_SLIDER_MIN_M, min(BASE_SLIDER_MAX_M, base))
        box.center[1] = self._floor_y() + base + box.extent[1] / 2

    def _sync_height_ui(self) -> None:
        """Point the slider at the selected box. Called from _redraw_boxes, so it stays right no
        matter which path changed the selection or the geometry."""
        if getattr(self, "base_slider", None) is None:
            return                       # panel not built yet (called during construction)
        box = self.boxes[self.selected] if self.selected is not None else None
        self.base_slider.enabled = box is not None
        if box is None:
            self.base_label.text = "ingen boks valgt"
            return
        base = self._base_of(box)
        # Guarded: assigning double_value fires the callback in Open3D, which would write the clamped
        # value straight back into the box and quietly drag it towards the slider's range.
        self._syncing_base = True
        try:
            self.base_slider.double_value = max(BASE_SLIDER_MIN_M, min(BASE_SLIDER_MAX_M, base))
        finally:
            self._syncing_base = False
        self.base_label.text = (f"{base:+.2f} m fra gulvet\n"
                                f"bunn {box.center[1] - box.extent[1] / 2:.2f}  "
                                f"høyde {box.extent[1]:.2f} m")

    def _on_base_slider(self, value: float) -> None:
        if self._syncing_base or self.selected is None:
            return
        self._push_undo()
        self._set_base(self.boxes[self.selected], float(value))
        self.dirty = True
        self._redraw_boxes()

    def _nudge_base(self, dy: float) -> None:
        if self.selected is None:
            self._status("Velg en boks først", "warning")
            return
        self._push_undo()
        box = self.boxes[self.selected]
        self._set_base(box, self._base_of(box) + dy)
        self.dirty = True
        self._redraw_boxes()

    def _seat_on_floor(self) -> None:
        """Put the box back on floor_height — useful when the floor IS right and only this box drifted."""
        if self.selected is None:
            self._status("Velg en boks først", "warning")
            return
        self._push_undo()
        self._set_base(self.boxes[self.selected], 0.0)
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
        self.mode_label.text = "Klikk hjørne A på gulvet"
        self._refresh_mode_banner()

    def _cancel_draw(self) -> None:
        self.mode = MODE_NORMAL
        self.draw_stage = 0
        self.draw_a = None
        self.draw_b = None
        self.draw_box = None
        self.mode_label.text = "Klikk en boks for å velge den"
        self._refresh_mode_banner()
        if self.mesh_loaded:
            self._remove_geometry("preview")
            self._remove_geometry("preview_pt_0")
            self._remove_geometry("preview_pt_1")

    # ---------- place mode (click a floor point for a fixed-size box) ----------

    def _toggle_place(self) -> None:
        """Enter/leave place mode. A toggle (not just ESC) so there is always a visible way out."""
        if self.mode == MODE_PLACE:
            self._cancel_draw()  # leaves place mode and clears the preview
            self._redraw_boxes()
        else:
            self._start_place()

    def _start_place(self) -> None:
        if not self.mesh_loaded:
            return
        self._cancel_draw()
        self.mode = MODE_PLACE
        # behold kameravinkelen brukeren står i — plassering skal ikke tvinge topdown
        self.mode_label.text = (
            f"«{self.type_combo.selected_text}»\n"
            "Klikk på gulvet. Hold R + flytt musa for å rotere.\n"
            "P eller ESC avslutter."
        )
        self._refresh_mode_banner()

    def _place_preview(self, xz: tuple[float, float] | None) -> None:
        self._remove_geometry("preview")
        if xz is None:
            return
        ex, ey, ez = BIN_TYPES[self.type_combo.selected_text]
        temp = BinBox([float(xz[0]), self._floor_y() + ey / 2, float(xz[1])], [ex, ey, ez], self._place_yaw)
        lineset = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(temp.corners()),
            o3d.utility.Vector2iVector(np.array(BOX_EDGES)),
        )
        lineset.paint_uniform_color(PREVIEW_COLOR)
        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = 4.0
        self.scene.scene.add_geometry("preview", lineset, material)
        self.window.post_redraw()

    def _mouse_place(self, event: gui.MouseEvent) -> gui.Widget.EventCallbackResult:
        if event.type == gui.MouseEvent.Type.WHEEL:
            return gui.Widget.EventCallbackResult.IGNORED
        ray = self._mouse_ray(event)
        if ray is None:
            return gui.Widget.EventCallbackResult.CONSUMED
        floor_point = self._ray_floor(ray)
        fp_xz = (float(floor_point[0]), float(floor_point[2])) if floor_point is not None else None

        # while R is held, the box stays put and the mouse spins it (yaw = angle from centre to cursor)
        if self._r_down:
            if self._place_anchor_xz is None and fp_xz is not None:
                self._place_anchor_xz = fp_xz
            if self._place_anchor_xz is not None and fp_xz is not None:
                dx, dz = fp_xz[0] - self._place_anchor_xz[0], fp_xz[1] - self._place_anchor_xz[1]
                if math.hypot(dx, dz) > 1e-6:
                    self._place_yaw = math.degrees(math.atan2(dz, dx))
        rotating = self._r_down and self._place_anchor_xz is not None
        target = self._place_anchor_xz if rotating else fp_xz

        if event.type == gui.MouseEvent.Type.MOVE:
            if time.time() - self._last_preview > 0.03:
                self._last_preview = time.time()
                self._place_preview(target)
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and target is not None:
            self._place_box_at(target[0], target[1])
            self._place_preview(target)  # stay in place mode so several bins can be dropped in a row
            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.CONSUMED

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
                self.mode_label.text = "Klikk hjørne B (første kant)"
            elif self.draw_stage == 1:
                if np.linalg.norm(point_xz - self.draw_a) >= 0.05:
                    self.draw_b = point_xz
                    self.draw_stage = 2
                    self.mode_label.text = "Trykk for dybde, dra opp, slipp"
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
                    self.mode_label.text = "Dra opp for høyde, slipp for å fullføre"
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
                        # height left alone on purpose -- see the "move" branch in _mouse_normal
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
                    # Y is deliberately NOT touched. Re-seating to floor_height on every move is what
                    # threw a correctly-placed box up to a wrong floor the moment you nudged it
                    # sideways; the height is now the operator's to set (see the Høyde slider).
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
                # Grow from the box's OWN base, not from floor_height: dragging the top handle used to
                # silently move the bottom too, undoing whatever height had been set for it.
                base = box.center[1] - box.extent[1] / 2
                box.extent[1] = height
                box.center[1] = base + height / 2
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

        # track the floor point under the cursor so Ctrl+V can paste there
        if event.type == gui.MouseEvent.Type.MOVE:
            ray = self._mouse_ray(event)
            floor_point = self._ray_floor(ray) if ray is not None else None
            if floor_point is not None:
                self._hover_xz = (float(floor_point[0]), float(floor_point[2]))

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
        if self.mode == MODE_PLACE:
            return self._mouse_place(event)
        if self.mode == MODE_DRAW:
            return self._mouse_draw(event)
        return self._mouse_normal(event)

    # ---------- entrances (doors) ----------

    def _toggle_entrance_mode(self) -> None:
        if self.mode == MODE_ENTRANCE:
            self.mode = MODE_NORMAL
            self.mode_label.text = "Klikk en boks for å velge den"
        else:
            self._cancel_draw()
            self.mode = MODE_ENTRANCE
            self.mode_label.text = "Klikk = ny dør\nCtrl+klikk = slett nærmeste"
        self._refresh_mode_banner()

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
            sphere.paint_uniform_color(list(T.rgb_of("entrance")))   # magenta = inngang, as in the previews
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
        # Only persist bin annotations the user actually made or edited. Never write untouched
        # proposals to outputs/annotations/ — that would falsely mark the scan "annotert" and feed
        # un-approved auto-proposals into training. (Entrances above are saved regardless.)
        if not self.dirty and not self._annotation_path().exists():
            return
        save_annotations(
            self._annotation_path(), self._current_zip().name, self.floor_height, self.boxes
        )
        self.dirty = False
        approved = sum(1 for b in self.boxes if b.status == STATUS_APPROVED)
        self._status(
            f"Lagret: {len(self.boxes)} bokser, {approved} godkjent, {len(self.entrances)} inngang",
            "success",
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
