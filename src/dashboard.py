"""Søppelrom 3D — desktop control panel.

Browse ALL scans in a list, see the rendered result images (room + measurements, free floor,
proposed new-bin placement), and launch the 3D viewer / annotation / entrance-picker — without
typing zip names in a terminal.

    .venv\\Scripts\\python.exe -m src.dashboard

LAYOUT / SCALING NOTES (why the code looks the way it does)
    The old version packed everything side by side with fixed pixel paddings, so the only size the
    window worked at was "maximised": the action buttons ran off the right edge and the preview image
    only ever grew. Now:

      * everything is placed with grid() and explicit row/column weights, so exactly one row (the
        preview) and one column (the right-hand side) absorb all extra space;
      * the two toolbars are WrapBar instances — a flow layout that moves buttons onto a second row
        instead of letting them be clipped;
      * every font size and padding comes from uitheme.tk_scale_for(root), recomputed (debounced) on
        <Configure>, so the same window is legible at 1000x700 and on a maximised 4K screen;
      * the fonts are *named* Tk fonts, so rescaling means configuring nine font objects — every
        widget that uses them follows automatically;
      * the preview is re-sampled from the cached PIL image (never re-read from disk) on a debounced
        resize, so dragging the window edge stays smooth.

    All colours come from src/uitheme.py, which derives them from src/style.py — the same palette the
    preview PNGs are drawn with. Green = ny kasse, rød = eksisterende kasse, blå = skyve-sti,
    magenta = inngang, i både bildene og dette vinduet.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from PIL import Image, ImageTk

from . import pipeline, ply_align
from . import uitheme as T
from .annotations import BIN_TYPES
from .set_entrance import ENTRANCE_DIR

VIEWS = [
    ("Rom + mål", "room_topdown.png"),
    ("Ledig gulv", "freespace_over_scene.png"),
    ("Plassering (ny kasse)", "placements.png"),
    ("I dag / forslag", "before_after.png"),
]

# The "3D" column in the scan list. Symbols come from ply_align so they cannot drift apart.
PLY_LEGEND = (
    f"{ply_align.PLY_OK} Polycam-sky      {ply_align.PLY_UNREGISTERED} ikke registrert\n"
    f"{ply_align.PLY_REJECTED} avvist      (tomt = ingen eksport)"
)

# Pillow renamed the resampling constants; keep working on both.
LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

# The stat strip under the preview: (key, caption, colour role for the value).
STAT_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("rom", "Rom", "text"),
    ("areal", "Gulvareal", "text"),
    ("kasser", "Kasser i dag", "existing_bin"),
    ("ledig", "Ledig gulv", "free_floor"),
    ("nye", "Nye plasser", "new_bin"),
    ("inngang", "Inngang", "entrance"),
    ("romtype", "Romtype", "text_muted"),
)


# --------------------------------------------------------------------------- flow layout

class WrapBar(ttk.Frame):
    """A horizontal bar whose children move onto a new row instead of being clipped.

    WHY: ttk has no flow/wrap geometry manager, and the single thing the user complained about was
    buttons being cut off unless the window was maximised. Children are positioned with place(),
    which lets us do the line-breaking ourselves and then set the bar's own height to exactly the
    number of rows used — so the surrounding grid still reserves the right amount of space.

    Because every child is place()d, the bar requests width 0 and can therefore never force the
    window to be wider than the user wants it.
    """

    def __init__(self, master, *, hgap: int = 8, vgap: int = 6, style: str = "TFrame") -> None:
        super().__init__(master, style=style, height=1)
        self._items: list[dict] = []
        self._hgap, self._vgap = max(1, hgap), max(1, vgap)
        self._last_width = -1
        self._height = 1
        self._pending: str | None = None
        self.bind("<Configure>", self._on_configure)

    # ----- building
    def add(self, widget: tk.Misc, *, kind: str = "item") -> tk.Misc:
        """kind: 'item' = a normal control, 'sep' = a vertical divider that is dropped if it would
        end up first or last on a row."""
        self._items.append({"widget": widget, "kind": kind, "visible": True})
        return widget

    def separator(self) -> ttk.Separator:
        sep = ttk.Separator(self, orient="vertical")
        self.add(sep, kind="sep")
        return sep

    def set_visible(self, widget: tk.Misc, visible: bool) -> None:
        """Show/hide one child. The re-flow is deferred to idle so hiding seven chips in a loop
        costs one layout pass, not seven."""
        for item in self._items:
            if item["widget"] is widget and item["visible"] != visible:
                item["visible"] = visible
                if not visible:
                    widget.place_forget()
                self._last_width = -1
                if self._pending is None:
                    self._pending = self.after_idle(self._deferred_reflow)
                return

    def _deferred_reflow(self) -> None:
        self._pending = None
        self.reflow()

    def set_gaps(self, hgap: int, vgap: int) -> None:
        if (max(1, hgap), max(1, vgap)) != (self._hgap, self._vgap):
            self._hgap, self._vgap = max(1, hgap), max(1, vgap)
            self._last_width = -1
            self.reflow()

    # ----- layout
    def _on_configure(self, event: tk.Event) -> None:
        # Re-flowing only on a real width change keeps this off the hot path while dragging: our own
        # place() calls fire <Configure> on the children, not on the bar.
        if abs(event.width - self._last_width) > 2:
            self._last_width = event.width
            self.reflow(event.width)

    def reflow(self, width: int | None = None) -> None:
        items = [i for i in self._items if i["visible"]]
        if not items:
            if self._height != 1:
                self._height = 1
                self.configure(height=1)
            return
        if width is None:
            width = self.winfo_width()
        width = int(width)
        if width <= 1:                      # not mapped yet — flow against what we asked for
            width = max(self.winfo_reqwidth(), 400)

        rows: list[list[dict]] = [[]]
        x = 0
        for item in items:
            widget, kind = item["widget"], item["kind"]
            w = 1 if kind == "sep" else max(widget.winfo_reqwidth(), 1)
            first = not rows[-1]
            if kind == "sep" and first:      # never start a row with a divider
                widget.place_forget()
                continue
            gap = 0 if first else self._hgap
            if not first and x + gap + w > width:
                if kind == "sep":            # the divider is what would have overflowed: drop it
                    widget.place_forget()
                    continue
                rows.append([])
                x, gap, first = 0, 0, True
            rows[-1].append({"widget": widget, "kind": kind, "x": x + gap, "w": w})
            x += gap + w

        for row in rows:                     # a divider left dangling at the end of a row is noise
            while row and row[-1]["kind"] == "sep":
                row.pop()["widget"].place_forget()

        y, total = 0, 0
        for row in rows:
            if not row:
                continue
            height = max(max(cell["widget"].winfo_reqheight(), 1) for cell in row)
            for cell in row:
                widget = cell["widget"]
                if cell["kind"] == "sep":
                    widget.place(x=cell["x"], y=y + 3, width=1, height=max(height - 6, 1))
                else:
                    own = max(widget.winfo_reqheight(), 1)
                    widget.place(x=cell["x"], y=y + (height - own) // 2, width=cell["w"], height=own)
            y += height + self._vgap
            total = y
        total = max(total - self._vgap, 1)
        if total != self._height:
            self._height = total
            self.configure(height=total)


class Dashboard:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Søppelrom 3D — kontrollpanel")
        self.root.geometry("1320x840")
        self.root.configure(bg=T.WINDOW_BG)

        self.view = tk.StringVar(value=VIEWS[0][0])
        self.bin_type = tk.StringVar(value="4-hjuls container")
        self.status = tk.StringVar(value="Klar.")
        self.search = tk.StringVar(value="")
        self.count_text = tk.StringVar(value="")
        self._addr_cache: dict[str, str | None] = {}
        self._ply_reason: dict[str, str] = {}   # per scan: why the 3D column shows what it shows
        self._photo: ImageTk.PhotoImage | None = None
        self._pil: Image.Image | None = None
        self._pil_path: Path | None = None      # what _pil was read from, so we never re-read it
        self._rendered: tuple[str, int, int] | None = None
        self._busy = False
        self._resize_job: str | None = None
        self._rescale_job: str | None = None
        self._pads: list[tuple[tk.Misc, str, str]] = []
        self._chips: dict[str, tuple[ttk.Frame, ttk.Label]] = {}
        self._bars: list[WrapBar] = []

        self.scale = T.tk_scale(1320, 840, self._tk_scaling())
        self._fonts: dict[str, tkfont.Font] = {}
        self._init_fonts()
        self._build_style()
        self._build_layout()
        self._populate()
        self._signature = self._file_signature()

        self.root.update_idletasks()
        # A minimum that is actually derived from the content: the scan list cannot be squeezed below
        # its column widths, so ask it how wide it needs to be instead of guessing a number.
        min_w = max(880, self.sidebar.winfo_reqwidth() + self.scale.px(420))
        self.root.minsize(min_w, 620)
        self._reflow_bars()

        # keyboard shortcuts for the most-used actions (letters shown in the button labels); each is
        # ignored while a text field (the address search) has focus, so typing never triggers them
        self.root.bind("<Left>", self._hotkey(lambda: self._step(-1)))
        self.root.bind("<Right>", self._hotkey(lambda: self._step(1)))
        self.root.bind("<g>", self._hotkey(lambda: self._generate([self._selected()])))
        self.root.bind("<G>", self._hotkey(self._generate_all))
        self.root.bind("<o>", self._hotkey(self._open_3d))
        self.root.bind("<a>", self._hotkey(self._annotate))
        self.root.bind("<f>", self._hotkey(self._prepare))
        self.root.bind("<r>", self._hotkey(self._open_reconstruction))
        self.root.bind("<s>", self._hotkey(self._open_stats))
        self.root.bind("<Control-f>", self._focus_search)
        self.root.bind("<Escape>", self._clear_search)
        # refresh scan statuses the moment the dashboard regains focus (e.g. back from annotating)
        self.root.bind("<FocusIn>", lambda _e: self._refresh_if_changed())
        self.root.bind("<Configure>", self._on_root_configure)

    # ---------- fonts ----------

    def _tk_scaling(self) -> float:
        try:
            return float(self.root.tk.call("tk", "scaling"))
        except Exception:  # noqa: BLE001 - never let styling crash the app
            return 96.0 / 72.0

    # step + weight per named font; ttk styles reference these by name, so a rescale only has to
    # reconfigure the nine Font objects and every widget follows.
    _FONT_SPECS: dict[str, tuple[str, str]] = {
        "display": ("display", "semibold"),
        "h1": ("h1", "semibold"),
        "h2": ("h2", "semibold"),
        "body": ("body", "regular"),
        "bodysemi": ("body", "semibold"),
        "small": ("small", "regular"),
        "smallsemi": ("small", "semibold"),
        "tiny": ("tiny", "regular"),
        "mono": ("body", "mono"),
    }

    def _init_fonts(self) -> None:
        for key, (step, weight) in self._FONT_SPECS.items():
            spec = self.scale.font(step, weight)
            family, size = spec[0], int(spec[1])
            style_hint = spec[2] if len(spec) > 2 else ""
            options = dict(
                family=family,
                size=size,
                weight="bold" if style_hint == "bold" else "normal",
                slant="italic" if style_hint == "italic" else "roman",
            )
            name = f"SR.{key}"
            if key in self._fonts:
                self._fonts[key].configure(**options)
                continue
            try:
                self._fonts[key] = tkfont.Font(root=self.root, name=name, exists=False, **options)
            except tk.TclError:
                # the name survives from an earlier Dashboard in the same interpreter (tests)
                existing = tkfont.nametofont(name)
                existing.configure(**options)
                self._fonts[key] = existing

    def font(self, key: str) -> str:
        """Name of a named font, usable directly as a Tk font spec."""
        return f"SR.{key}"

    # ---------- styling ----------

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            # clam is the only built-in theme whose element colours can actually be overridden;
            # the native "vista" theme ignores background/fieldbackground entirely.
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.ttk_style = style
        s = self.scale
        stripe = T.mix("panel_bg", "panel_bg_alt", 0.6)
        self._stripe = stripe

        style.configure(".", background=T.WINDOW_BG, foreground=T.TEXT,
                        fieldbackground=T.FIELD_BG, font=self.font("body"),
                        borderwidth=0, focuscolor=T.FOCUS_RING)

        # ----- surfaces
        style.configure("TFrame", background=T.WINDOW_BG)
        style.configure("Card.TFrame", background=T.PANEL_BG)
        style.configure("Well.TFrame", background=T.WINDOW_BG)
        style.configure("Chip.TFrame", background=T.PANEL_BG_ALT)
        style.configure("Bar.TFrame", background=T.WINDOW_BG)

        # ----- type hierarchy
        style.configure("TLabel", background=T.WINDOW_BG, foreground=T.TEXT, font=self.font("body"))
        style.configure("Display.TLabel", font=self.font("display"), foreground=T.TEXT)
        style.configure("H1.TLabel", background=T.PANEL_BG, font=self.font("h1"), foreground=T.TEXT)
        style.configure("H2.TLabel", background=T.PANEL_BG, font=self.font("h2"), foreground=T.TEXT)
        style.configure("Sub.TLabel", font=self.font("small"), foreground=T.TEXT_MUTED)
        # text_disabled is genuinely unreadable at 8 pt on this background, so the two hint lines
        # (keyboard reminder, 3D-column legend) sit between muted and disabled instead. 0.35 is the
        # darkest blend that still clears WCAG 4.5:1 on both panel_bg (5.02) and window_bg (5.25).
        hint = T.mix("text_muted", "panel_bg", 0.35)
        style.configure("Hint.TLabel", font=self.font("tiny"), foreground=hint)
        style.configure("Caption.TLabel", background=T.PANEL_BG, font=self.font("small"),
                        foreground=T.TEXT_MUTED)
        style.configure("CardSub.TLabel", background=T.PANEL_BG, font=self.font("small"),
                        foreground=T.TEXT_MUTED)
        style.configure("CardHint.TLabel", background=T.PANEL_BG, font=self.font("tiny"),
                        foreground=hint)
        style.configure("Warn.TLabel", background=T.PANEL_BG, font=self.font("smallsemi"),
                        foreground=T.WARNING)
        style.configure("Well.TLabel", background=T.WINDOW_BG, font=self.font("body"),
                        foreground=T.TEXT_MUTED)
        style.configure("Status.TLabel", background=T.PANEL_BG, font=self.font("small"),
                        foreground=T.TEXT_MUTED)
        # chip caption + value (the value colour is set per chip from STAT_FIELDS)
        style.configure("ChipCap.TLabel", background=T.PANEL_BG_ALT, font=self.font("tiny"),
                        foreground=T.TEXT_MUTED)
        for _key, _caption, role in STAT_FIELDS:
            style.configure(f"ChipVal{role}.TLabel", background=T.PANEL_BG_ALT,
                            font=self.font("bodysemi"), foreground=T.HEX[role])

        # ----- buttons. clam draws a bevel from lightcolor/darkcolor; flattening those and using
        # bordercolor for a 1 px ring is what stops the buttons looking like Windows 95.
        def button_style(name: str, bg: str, fg: str, edge: str, hover: str, active: str,
                         font_key: str = "body") -> None:
            style.configure(name, background=bg, foreground=fg, bordercolor=edge,
                            lightcolor=bg, darkcolor=bg, relief="flat", borderwidth=1,
                            focusthickness=1, focuscolor=T.FOCUS_RING,
                            font=self.font(font_key), padding=(s.pad_m, s.pad_s + 1),
                            anchor="center")
            style.map(name,
                      background=[("disabled", T.PANEL_BG), ("pressed", active), ("active", hover)],
                      foreground=[("disabled", T.TEXT_DISABLED)],
                      bordercolor=[("disabled", T.DIVIDER), ("focus", T.FOCUS_RING),
                                   ("active", T.PANEL_EDGE)],
                      lightcolor=[("pressed", active), ("active", hover)],
                      darkcolor=[("pressed", active), ("active", hover)],
                      relief=[("pressed", "flat"), ("!pressed", "flat")])

        button_style("TButton", T.PANEL_BG_ALT, T.TEXT, T.PANEL_EDGE, T.HOVER_BG, T.ACTIVE_BG)
        button_style("Accent.TButton", T.ACCENT, T.TEXT_ON_ACCENT, T.ACCENT,
                     T.ACCENT_HOVER, T.ACCENT_HOVER, font_key="bodysemi")
        # secondary-but-notable: reads as a button, but the accent lettering keeps "Generer bilder"
        # the only filled primary on the bar
        button_style("Ghost.TButton", T.PANEL_BG, T.ACCENT_HOVER, T.HEX["accent_muted"],
                     T.PANEL_BG_ALT, T.PANEL_BG_ALT, font_key="bodysemi")

        # ----- the view selector: radio buttons drawn as filter chips (Toolbutton layout)
        style.configure("Seg.Toolbutton", background=T.PANEL_BG, foreground=T.TEXT_MUTED,
                        bordercolor=T.DIVIDER, lightcolor=T.PANEL_BG, darkcolor=T.PANEL_BG,
                        # a hair tighter than a real button: the four view chips plus the bin-type
                        # group then still fit on one line at 1000 px wide
                        relief="flat", borderwidth=1, padding=(s.pad_s + 1, s.pad_s),
                        font=self.font("body"), anchor="center", focuscolor=T.FOCUS_RING)
        style.map("Seg.Toolbutton",
                  background=[("selected", T.ACCENT), ("active", T.HOVER_BG)],
                  foreground=[("selected", T.TEXT_ON_ACCENT), ("active", T.TEXT)],
                  bordercolor=[("selected", T.ACCENT), ("active", T.PANEL_EDGE)],
                  lightcolor=[("selected", T.ACCENT), ("active", T.HOVER_BG)],
                  darkcolor=[("selected", T.ACCENT), ("active", T.HOVER_BG)],
                  relief=[("selected", "flat"), ("!selected", "flat")])

        # ----- scan list
        # lightcolor/darkcolor must be neutralised as well: clam draws the tree's field border with
        # its own near-white bevel colours, which shows up as a pale box around the dark list.
        style.configure("Treeview", background=T.PANEL_BG, fieldbackground=T.PANEL_BG,
                        foreground=T.TEXT, rowheight=s.row_height, font=self.font("body"),
                        borderwidth=0, relief="flat", bordercolor=T.PANEL_BG,
                        lightcolor=T.PANEL_BG, darkcolor=T.PANEL_BG)
        style.configure("Treeview.Heading", font=self.font("smallsemi"), background=T.PANEL_BG_ALT,
                        foreground=T.TEXT_MUTED, relief="flat", borderwidth=0,
                        padding=(s.pad_s, s.pad_s - 1))
        style.map("Treeview.Heading",
                  background=[("active", T.HOVER_BG)], relief=[("active", "flat")])
        style.map("Treeview",
                  background=[("selected", T.HEX["selection_bg"])],
                  foreground=[("selected", T.HEX["selection_fg"])])
        try:
            # Drop the dotted focus rectangle clam draws inside the selected row: it reads as a
            # rendering glitch on a dark selection. Guarded because the element names are theme
            # internals — if they ever change we keep the default layout instead of crashing.
            style.layout("Treeview.Item", [
                ("Treeitem.padding", {"sticky": "nswe", "children": [
                    ("Treeitem.indicator", {"side": "left", "sticky": ""}),
                    ("Treeitem.image", {"side": "left", "sticky": ""}),
                    ("Treeitem.text", {"side": "left", "sticky": ""}),
                ]}),
            ])
        except tk.TclError:
            pass

        # ----- search field
        style.configure("Search.TEntry", fieldbackground=T.FIELD_BG, foreground=T.TEXT,
                        insertcolor=T.TEXT, bordercolor=T.PANEL_EDGE, lightcolor=T.FIELD_BG,
                        darkcolor=T.FIELD_BG, borderwidth=1, relief="flat",
                        padding=(s.pad_s, s.pad_s - 1))
        style.map("Search.TEntry",
                  bordercolor=[("focus", T.FOCUS_RING)],
                  lightcolor=[("focus", T.FIELD_BG)],
                  fieldbackground=[("focus", T.FIELD_BG)])

        # ----- combobox. A readonly Combobox ignores a plain configure; without an explicit state
        # map clam draws the field with its near-white default -> white text on a white box.
        style.configure("SR.TCombobox", fieldbackground=T.FIELD_BG, background=T.PANEL_BG_ALT,
                        foreground=T.TEXT, arrowcolor=T.TEXT_MUTED, bordercolor=T.PANEL_EDGE,
                        lightcolor=T.FIELD_BG, darkcolor=T.FIELD_BG, borderwidth=1,
                        relief="flat", padding=(s.pad_s, s.pad_s - 2))
        style.map(
            "SR.TCombobox",
            fieldbackground=[("readonly", T.FIELD_BG), ("disabled", T.PANEL_BG)],
            background=[("readonly", T.PANEL_BG_ALT), ("active", T.HOVER_BG)],
            foreground=[("readonly", T.TEXT), ("disabled", T.TEXT_DISABLED)],
            selectbackground=[("readonly", T.FIELD_BG)],
            selectforeground=[("readonly", T.TEXT)],
            arrowcolor=[("readonly", T.TEXT_MUTED), ("active", T.TEXT)],
            bordercolor=[("focus", T.FOCUS_RING), ("hover", T.PANEL_EDGE)],
        )
        # The drop-down list is a classic Tk Listbox (not ttk), styled via the option database.
        self.root.option_add("*TCombobox*Listbox.background", T.PANEL_BG)
        self.root.option_add("*TCombobox*Listbox.foreground", T.TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", T.HEX["selection_bg"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", T.HEX["selection_fg"])
        self.root.option_add("*TCombobox*Listbox.font", self.font("body"))
        self.root.option_add("*TCombobox*Listbox.borderWidth", 0)

        # ----- scrollbar + busy indicator
        style.configure("SR.Vertical.TScrollbar", background=T.PANEL_BG_ALT, troughcolor=T.PANEL_BG,
                        bordercolor=T.PANEL_BG, arrowcolor=T.TEXT_MUTED, lightcolor=T.PANEL_BG_ALT,
                        darkcolor=T.PANEL_BG_ALT, borderwidth=0, relief="flat",
                        gripcount=0, arrowsize=s.px(11), width=s.px(11))
        style.map("SR.Vertical.TScrollbar",
                  background=[("pressed", T.ACTIVE_BG), ("active", T.HOVER_BG)])
        style.configure("SR.Horizontal.TProgressbar", background=T.ACCENT, troughcolor=T.PANEL_BG,
                        bordercolor=T.PANEL_BG, lightcolor=T.ACCENT, darkcolor=T.ACCENT,
                        borderwidth=0, thickness=s.px(4))
        style.configure("TSeparator", background=T.DIVIDER)

    def _restyle(self) -> None:
        """Re-apply everything that depends on the current TkScale (after a resize / DPI change)."""
        self._init_fonts()          # named fonts: every widget picks the new size up by itself
        self._build_style()         # paddings, row height, scrollbar width live in the styles
        self._apply_pads()
        for bar in self._bars:
            bar.set_gaps(self.scale.pad_s, self.scale.pad_s)
        self.body.columnconfigure(0, minsize=self._sidebar_minsize())
        self._apply_tree_columns()
        self.legend_label.configure(
            wraplength=max(self.scale.sidebar_min - 2 * self.scale.pad_m, 120))
        self.busy_bar.configure(length=self.scale.px(120))
        self._reflow_bars()
        self._rendered = None            # the well changed size with the paddings
        self._render_photo()

    # ---------- layout helpers ----------

    def _grid(self, widget: tk.Misc, *, padx: str = "", pady: str = "", **kw) -> tk.Misc:
        """grid() a widget and remember which TkScale padding steps it used, so a rescale can
        re-apply them. padx/pady name a step ('pad_m') or a pair ('pad_m,pad_s' = leading,trailing).
        """
        widget.grid(**kw, **self._pad_values(padx, pady))
        if padx or pady:
            self._pads.append((widget, padx, pady))
        return widget

    def _pad_values(self, padx: str, pady: str) -> dict:
        out = {}
        for name, spec in (("padx", padx), ("pady", pady)):
            if not spec:
                continue
            parts = [p.strip() for p in spec.split(",")]
            values = tuple(0 if p in ("", "0") else int(getattr(self.scale, p)) for p in parts)
            out[name] = values[0] if len(values) == 1 else values
        return out

    def _apply_pads(self) -> None:
        for widget, padx, pady in self._pads:
            try:
                if widget.winfo_manager() != "grid":   # never re-manage a hidden widget
                    continue
                widget.grid_configure(**self._pad_values(padx, pady))
            except tk.TclError:      # widget destroyed (never happens in normal use)
                pass

    def _card(self, master: tk.Misc, *, padding: int = 0, edge: str = T.DIVIDER
              ) -> tuple[tk.Frame, ttk.Frame]:
        """A panel with a 1 px hairline. The hairline is the outer tk.Frame's background showing
        through a 1 px inset — ttk cannot draw a plain border on a frame."""
        shell = tk.Frame(master, bg=edge, bd=0, highlightthickness=0)
        inner = ttk.Frame(shell, style="Card.TFrame", padding=padding)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        return shell, inner

    def _reflow_bars(self) -> None:
        self.root.update_idletasks()
        for bar in self._bars:
            bar.reflow()

    def _sidebar_minsize(self) -> int:
        """How wide the scan list is allowed to be.

        Fixed-width sidebars look broken on an ultrawide: the addresses stay truncated while a
        thousand pixels of empty canvas sit next to the preview. So the list gets a quarter of the
        window, floored at the scale's minimum and capped, and the preview absorbs everything else.
        The tree's own requested width is still a hard floor (grid takes the larger of the two), so
        this can only ever widen the list, never clip it.
        """
        width = max(self.root.winfo_width(), 1)
        return int(min(max(self.scale.sidebar_min, 0.24 * width), self.scale.px(500)))

    def _apply_tree_columns(self) -> None:
        """Column widths in scaled pixels. Only the address column stretches, so the three narrow
        status columns keep their size when the sidebar changes width."""
        s = self.scale
        # 216 px fits "Frydenlundgata 4B, 0169 Oslo" — the longest shape a real address takes here.
        self.tree.column("#0", width=s.px(216), minwidth=s.px(120), stretch=True)
        self.tree.column("status", width=s.px(82), minwidth=s.px(64), anchor="center", stretch=False)
        self.tree.column("bins", width=s.px(54), minwidth=s.px(44), anchor="center", stretch=False)
        self.tree.column("ply", width=s.px(32), minwidth=s.px(28), anchor="center", stretch=False)

    # ---------- layout ----------

    def _build_layout(self) -> None:
        s = self.scale
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)          # only the body grows
        for row in (0, 2, 3):
            root.rowconfigure(row, weight=0)

        # ---------------- header
        header = ttk.Frame(root, style="TFrame")
        self._grid(header, row=0, column=0, sticky="ew", padx="pad_l", pady="pad_l,pad_s")
        header.columnconfigure(1, weight=1)
        title = ttk.Frame(header, style="TFrame")
        title.grid(row=0, column=0, sticky="w")
        ttk.Label(title, text="Søppelrom 3D", style="Display.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(title, text="velg skann · se resultat · åpne i 3D",
                  style="Sub.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(header, text="←/→ blar   ·   G genererer   ·   O åpner 3D   ·   A annoterer",
                  style="Hint.TLabel", anchor="e").grid(row=0, column=1, rowspan=2, sticky="e")

        # ---------------- body: list | viewer
        body = ttk.Frame(root, style="TFrame")
        self.body = body
        self._grid(body, row=1, column=0, sticky="nsew", padx="pad_l", pady="pad_xs")
        body.columnconfigure(0, weight=0, minsize=s.sidebar_min)   # list keeps a fixed-ish width
        # (re-evaluated against the real window width in _maybe_rescale / _restyle)
        body.columnconfigure(1, weight=1)                          # viewer absorbs the space
        body.rowconfigure(0, weight=1)

        self.sidebar, side = self._card(body, padding=s.pad_m)
        self._grid(self.sidebar, row=0, column=0, sticky="nsew")
        side.columnconfigure(0, weight=1)
        side.rowconfigure(2, weight=1)                             # the tree takes the height

        search_row = ttk.Frame(side, style="Card.TFrame")
        self._grid(search_row, row=0, column=0, sticky="ew", pady="0,pad_s")
        search_row.columnconfigure(0, weight=1)
        ttk.Label(search_row, text="Søk adresse", style="Caption.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Label(search_row, textvariable=self.count_text, style="CardHint.TLabel",
                  anchor="e").grid(row=0, column=1, sticky="e")
        self.search_entry = ttk.Entry(side, textvariable=self.search, style="Search.TEntry",
                                     font=self.font("body"))
        self._grid(self.search_entry, row=1, column=0, sticky="ew", pady="0,pad_s")
        self.search_entry.bind("<KeyRelease>", lambda _e: self._populate())

        tree_box = ttk.Frame(side, style="Card.TFrame")
        self._grid(tree_box, row=2, column=0, sticky="nsew")
        tree_box.columnconfigure(0, weight=1)
        tree_box.rowconfigure(0, weight=1)
        # height is deliberately small: the tree must be able to shrink with the window, and the
        # grid weight above gives it every spare pixel anyway.
        self.tree = ttk.Treeview(tree_box, columns=("status", "bins", "ply"), show="tree headings",
                                 height=8, selectmode="browse", style="Treeview")
        self.tree.heading("#0", text="Adresse")
        self.tree.heading("status", text="Status")
        self.tree.heading("bins", text="Kasser")
        self.tree.heading("ply", text="3D")   # ● = Polycam-sky vises i 3D, ○ = ikke registrert, ✗ = avvist
        self._apply_tree_columns()
        # Treeview colours are per row, not per cell, so status and the zebra stripe have to be
        # combined into one tag per (status, parity) pair.
        for status, colour in (("annotated", T.SUCCESS), ("prepared", T.TEXT), ("raw", T.TEXT_MUTED)):
            self.tree.tag_configure(f"{status}_even", foreground=colour, background=T.PANEL_BG)
            self.tree.tag_configure(f"{status}_odd", foreground=colour, background=self._stripe)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())
        scroll = ttk.Scrollbar(tree_box, orient="vertical", command=self.tree.yview,
                               style="SR.Vertical.TScrollbar")
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)

        # wraplength keeps the legend from widening the whole sidebar: a long single line would
        # become the sidebar's requested width and eat the preview's space.
        self.legend_label = ttk.Label(side, text=PLY_LEGEND, style="CardHint.TLabel",
                                      justify="left", wraplength=max(s.sidebar_min - 2 * s.pad_m, 120))
        self._grid(self.legend_label, row=3, column=0, sticky="w", pady="pad_s,0")

        # ---------------- viewer column
        right = ttk.Frame(body, style="TFrame")
        self._grid(right, row=0, column=1, sticky="nsew", padx="pad_m,0")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)          # the preview card grows

        # view selector + bin type, in a bar that wraps instead of clipping
        toolbar = WrapBar(right, hgap=s.pad_s, vgap=s.pad_s, style="Bar.TFrame")
        self._bars.append(toolbar)
        self._grid(toolbar, row=0, column=0, sticky="ew", pady="0,pad_s")
        for label, _file in VIEWS:
            toolbar.add(ttk.Radiobutton(toolbar, text=label, value=label, variable=self.view,
                                        style="Seg.Toolbutton", command=self._on_view_change))
        toolbar.separator()
        # label + combobox live in one frame so a line break never separates them
        bin_group = ttk.Frame(toolbar, style="Bar.TFrame")
        ttk.Label(bin_group, text="Kassetype", style="Sub.TLabel").pack(side="left",
                                                                        padx=(0, s.pad_s))
        combo = ttk.Combobox(bin_group, textvariable=self.bin_type, values=list(BIN_TYPES),
                             state="readonly", width=18, style="SR.TCombobox")
        try:
            combo.configure(font=self.font("body"))   # -font is not in every Tk 8.6 patch level
        except tk.TclError:
            pass
        combo.pack(side="left")
        toolbar.add(bin_group)
        combo.bind("<<ComboboxSelected>>",
                   lambda _e: self.status.set("Trykk «Generer bilder» for å oppdatere plassering."))

        preview_shell, preview = self._card(right, padding=s.pad_m)
        self._grid(preview_shell, row=1, column=0, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(1, weight=1)

        head = ttk.Frame(preview, style="Card.TFrame")
        self._grid(head, row=0, column=0, sticky="ew", pady="0,pad_s")
        head.columnconfigure(0, weight=1)
        self.title_label = ttk.Label(head, text="Velg et skann", style="H1.TLabel", anchor="w")
        self.title_label.grid(row=0, column=0, sticky="ew")
        self.subtitle_label = ttk.Label(head, text="", style="CardSub.TLabel", anchor="w")
        self.subtitle_label.grid(row=1, column=0, sticky="ew")
        self.warn_label = ttk.Label(head, text="", style="Warn.TLabel", anchor="e", justify="right")
        self.warn_label.grid(row=0, column=1, rowspan=2, sticky="e")

        # The image well is window-coloured on purpose: the preview PNGs are rendered on exactly
        # this background (style.BACKDROP), so the picture melts into the well with no seam.
        well_shell = tk.Frame(preview, bg=T.mix("divider", "panel_edge", 0.45),
                              bd=0, highlightthickness=0)
        self._grid(well_shell, row=1, column=0, sticky="nsew")
        self.image_holder = ttk.Frame(well_shell, style="Well.TFrame")
        self.image_holder.pack(fill="both", expand=True, padx=1, pady=1)
        # Without this, the label's requested size (= the image) would push the whole window bigger
        # every time a larger image is rendered.
        self.image_holder.pack_propagate(False)
        self.image_label = ttk.Label(self.image_holder, style="Well.TLabel", anchor="center",
                                     justify="center", text="Velg et skann til venstre.")
        self.image_label.pack(fill="both", expand=True)
        self.image_holder.bind("<Configure>", self._on_area_configure)

        # stat chips: the old single pipe-separated line was simply cut off on a narrow window
        self.stats_bar = WrapBar(preview, hgap=s.pad_s, vgap=s.pad_s, style="Card.TFrame")
        self._bars.append(self.stats_bar)
        self._grid(self.stats_bar, row=2, column=0, sticky="ew", pady="pad_s,0")
        for key, caption, role in STAT_FIELDS:
            chip = ttk.Frame(self.stats_bar, style="Chip.TFrame", padding=(s.pad_s, s.pad_xs + 2))
            ttk.Label(chip, text=caption, style="ChipCap.TLabel").grid(row=0, column=0, sticky="w")
            value = ttk.Label(chip, text="—", style=f"ChipVal{role}.TLabel")
            value.grid(row=1, column=0, sticky="w")
            self.stats_bar.add(chip)
            self._chips[key] = (chip, value)
        self.empty_stats = ttk.Label(self.stats_bar, text="Ingen analyse ennå — trykk «Generer bilder».",
                                     style="CardSub.TLabel")
        self.stats_bar.add(self.empty_stats)

        # ---------------- action bar (wraps)
        actions = WrapBar(root, hgap=s.pad_s, vgap=s.pad_s, style="Bar.TFrame")
        self._bars.append(actions)
        self._grid(actions, row=2, column=0, sticky="ew", padx="pad_l", pady="pad_m,pad_s")

        def button(text: str, command, style_name: str = "TButton") -> ttk.Button:
            return actions.add(ttk.Button(actions, text=text, command=command, style=style_name,
                                          takefocus=True))

        button("◀ Forrige (←)", lambda: self._step(-1))
        button("Neste (→) ▶", lambda: self._step(1))
        actions.separator()
        button("Generer bilder (G)", lambda: self._generate([self._selected()]), "Accent.TButton")
        button("Generer alle (⇧G)", self._generate_all)
        actions.separator()
        button("Åpne i 3D — plassering (O)", self._open_3d)
        button("3D-rekonstruksjon (R)", self._open_reconstruction)
        button("Annotér (A)", self._annotate)
        button("Forbered: bygg 3D + finn kasser (F)", self._prepare)
        actions.separator()
        button("Statistikk (S)", self._open_stats, "Ghost.TButton")

        # ---------------- status bar
        bar_shell = tk.Frame(root, bg=T.DIVIDER, bd=0, highlightthickness=0)
        bar_shell.grid(row=3, column=0, sticky="ew")
        statusbar = ttk.Frame(bar_shell, style="Card.TFrame")
        statusbar.pack(fill="x", pady=(1, 0))
        statusbar.columnconfigure(0, weight=1)
        self._grid(ttk.Label(statusbar, textvariable=self.status, style="Status.TLabel",
                             anchor="w"), row=0, column=0, sticky="ew",
                   padx="pad_l,pad_m", pady="pad_s")
        # Not registered with _grid(): re-applying grid options to a grid_remove()d widget would
        # silently re-manage it, i.e. the busy bar would pop back up on every rescale.
        self.busy_bar = ttk.Progressbar(statusbar, mode="indeterminate", length=s.px(120),
                                        style="SR.Horizontal.TProgressbar")
        self.busy_bar.grid(row=0, column=1, sticky="e", padx=(0, s.pad_l), pady=s.pad_s)
        self.busy_bar.grid_remove()

    # ---------- resize / rescale ----------

    def _on_root_configure(self, event: tk.Event) -> None:
        if event.widget is not self.root:
            return
        # Debounced: a drag fires dozens of Configure events, and re-deriving the whole type scale
        # for each one would make the drag feel like glue.
        if self._rescale_job is not None:
            self.root.after_cancel(self._rescale_job)
        self._rescale_job = self.root.after(180, self._maybe_rescale)

    def _maybe_rescale(self) -> None:
        self._rescale_job = None
        scale = T.tk_scale(self.root.winfo_width(), self.root.winfo_height(), self._tk_scaling())
        if scale == self.scale:
            # Same type scale, but the sidebar share and the line breaks still depend on the width.
            self.body.columnconfigure(0, minsize=self._sidebar_minsize())
            self._reflow_bars()
            return
        self.scale = scale
        self._restyle()

    def _on_area_configure(self, event: tk.Event) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(110, self._render_photo)

    # ---------- scan list ----------

    def _address(self, stem: str) -> str | None:
        """Cached address for a scan (from its stats.json); None if the scan has no rendered stats."""
        if stem not in self._addr_cache:
            path = pipeline.preview_dir(stem) / "stats.json"
            address = None
            if path.exists():
                try:
                    address = json.loads(path.read_text(encoding="utf-8")).get("address")
                except Exception:  # noqa: BLE001 - a bad stats file just falls back to the scan id
                    address = None
            self._addr_cache[stem] = address
        return self._addr_cache[stem]

    def _populate(self) -> None:
        query = self.search.get().strip().lower()
        self.tree.delete(*self.tree.get_children())
        total = shown = annotated_count = 0
        for stem in pipeline.list_scans():
            total += 1
            address = self._address(stem)
            label = address or stem
            if query and query not in label.lower() and query not in stem.lower():
                continue
            if pipeline.is_annotated(stem):
                status, tag = "✓ annotert", "annotated"
                annotated_count += 1
            elif pipeline.is_prepared(stem):
                status, tag = "klar", "prepared"
            else:
                status, tag = "rå", "raw"
            bins = pipeline.existing_bin_count(stem) if pipeline.is_prepared(stem) else ""
            ply, self._ply_reason[stem] = ply_align.backdrop_status(stem)
            parity = "odd" if shown % 2 else "even"
            self.tree.insert("", "end", iid=stem, text=label, values=(status, bins, ply),
                             tags=(f"{tag}_{parity}",))
            shown += 1
        if query:
            self.count_text.set(f"{shown} av {total} skann")
        else:
            self.count_text.set(f"{total} skann · {annotated_count} annotert")
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])

    def _selected(self) -> str | None:
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _editing(self) -> bool:
        return isinstance(self.root.focus_get(), tk.Entry)

    def _hotkey(self, action):
        def handler(_event=None):
            if not self._editing():
                action()
        return handler

    def _focus_search(self, _event=None) -> str:
        self.search_entry.focus_set()
        self.search_entry.selection_range(0, "end")
        return "break"

    def _clear_search(self, _event=None) -> None:
        if self.search.get():
            self.search.set("")
            self._populate()
        self.tree.focus_set()

    def _step(self, delta: int) -> None:
        children = list(self.tree.get_children())
        stem = self._selected()
        if not children or stem is None:
            return
        idx = (children.index(stem) + delta) % len(children)
        self.tree.selection_set(children[idx])
        self.tree.focus(children[idx])
        self.tree.see(children[idx])

    # ---------- viewing ----------

    def _view_file(self) -> str:
        return dict(VIEWS)[self.view.get()]

    def _on_select(self) -> None:
        self._load_stats()
        self._show_image()
        stem = self._selected()   # explain the "3D" column so the symbol is not cryptic
        if stem and stem in self._ply_reason:
            self.status.set(self._ply_reason[stem])

    def _on_view_change(self) -> None:
        stem = self._selected()
        if stem:                     # the caption under the address names the view being shown
            self.subtitle_label.configure(text=f"{stem}   ·   {self.view.get()}")
        self._show_image()

    def _show_image(self) -> None:
        stem = self._selected()
        if stem is None:
            return
        path = pipeline.preview_dir(stem) / self._view_file()
        if not path.exists():
            self._pil = None
            self._pil_path = None
            self._rendered = None
            self._photo = None
            self.image_label.configure(image="", text="Ingen bilder ennå — trykk «Generer bilder».")
            return
        if path != self._pil_path:
            # Read from disk only when the file actually changes; every resize re-samples this copy.
            self._pil = Image.open(path)
            self._pil.load()
            self._pil_path = path
            self._rendered = None
        self._render_photo()

    def _render_photo(self) -> None:
        self._resize_job = None
        if self._pil is None:
            return
        target_w = max(self.image_holder.winfo_width() - 2 * self.scale.pad_s, 80)
        target_h = max(self.image_holder.winfo_height() - 2 * self.scale.pad_s, 80)
        signature = (str(self._pil_path), target_w, target_h)
        if signature == self._rendered:
            return
        source_w, source_h = self._pil.size
        # min() keeps the aspect ratio; the 2.0 cap stops a small render being blown up to mush on a
        # 4K screen (and stops the memory cost of a pointless huge upscale).
        factor = min(target_w / source_w, target_h / source_h, 2.0)
        size = (max(1, int(source_w * factor)), max(1, int(source_h * factor)))
        image = self._pil if size == self._pil.size else self._pil.resize(size, LANCZOS)
        # master= is explicit on purpose: PhotoImage otherwise binds to tkinter._default_root, which
        # is the wrong interpreter as soon as anything else in the process has made a Tk root.
        self._photo = ImageTk.PhotoImage(image, master=self.root)
        self.image_label.configure(image=self._photo, text="")
        self._rendered = signature

    def _load_stats(self) -> None:
        stem = self._selected()
        stats_path = pipeline.preview_dir(stem) / "stats.json" if stem else None
        if not stats_path or not stats_path.exists():
            self.title_label.configure(text=stem or "Velg et skann")
            self.subtitle_label.configure(text=stem or "")
            self.warn_label.configure(text="")
            self._set_stats_visible(False)
            return
        s = json.loads(stats_path.read_text(encoding="utf-8"))
        inne = "innendørs" if s.get("indoor") else "utendørs/åpent"
        self.title_label.configure(text=s.get("address") or stem)
        self.subtitle_label.configure(text=f"{stem}   ·   {self.view.get()}")
        self.warn_label.configure(
            text="⚠ INNESPERRET (dør lukket i scan) — hoppet over" if s.get("closed_room") else "")
        values = {
            "rom": f"{s['length_m']}×{s['width_m']} m",
            "areal": f"{s['area_m2']} m²",
            "kasser": f"{s['n_existing']}",
            "ledig": f"{s['free_area_m2']} m²",
            "nye": f"{s['n_candidates']} × {s['bin_type']}",
            "inngang": f"{s['n_entrances']} ({s['entrance_source']})",
            "romtype": inne,
        }
        for key, (_chip, label) in self._chips.items():
            label.configure(text=values.get(key, "—"))
        self._set_stats_visible(True)

    def _set_stats_visible(self, visible: bool) -> None:
        for chip, _label in self._chips.values():
            self.stats_bar.set_visible(chip, visible)
        self.stats_bar.set_visible(self.empty_stats, not visible)

    # ---------- generation (background) ----------

    def _set_status(self, message: str) -> None:
        self.root.after(0, lambda: self.status.set(message))

    def _generate_all(self) -> None:
        self._generate([s for s in pipeline.list_scans()])

    def _generate(self, stems: list[str | None]) -> None:
        stems = [s for s in stems if s]
        if not stems or self._busy:
            return
        self._busy = True
        self._set_busy_indicator(True)
        threading.Thread(target=self._generate_worker, args=(stems,), daemon=True).start()

    def _set_busy_indicator(self, busy: bool) -> None:
        """A minutes-long prepare/analyse run needs to look alive, not frozen."""
        try:
            if busy:
                self.busy_bar.grid()
                self.busy_bar.start(18)
            else:
                self.busy_bar.stop()
                self.busy_bar.grid_remove()
        except tk.TclError:
            pass

    def _generate_worker(self, stems: list[str]) -> None:
        for stem in stems:
            try:
                if not pipeline.is_prepared(stem):
                    self._set_status(f"Forbereder {stem}: bygger punktsky + 3D-mesh og kjører "
                                     "kasse-deteksjon → forslag … (kan ta et par minutter)")
                    subprocess.run(
                        [sys.executable, "-m", "src.prepare_scan", "--scan", str(pipeline.RAW_DIR / f"{stem}.zip")],
                        cwd=str(pipeline.PROJECT_ROOT),
                    )
                self._set_status(f"Analyserer {stem} …")
                pipeline.analyze_and_render(stem, self.bin_type.get())
            except Exception as error:  # noqa: BLE001 - surface any failure in the status bar
                self._set_status(f"Feil på {stem}: {error}")
        self.root.after(0, self._generate_done)

    def _generate_done(self) -> None:
        self._busy = False
        self._set_busy_indicator(False)
        self._addr_cache.clear()  # new stats.json files may have fresh addresses
        self._populate_keep_selection()
        self._pil_path = None     # the PNGs on disk were just rewritten — drop the cached copy
        self._on_select()
        self._set_status("Ferdig.")

    def _populate_keep_selection(self) -> None:
        stem = self._selected()
        self._populate()
        if stem and self.tree.exists(stem):
            self.tree.selection_set(stem)
            self.tree.focus(stem)

    # ---------- launching external Open3D windows ----------

    def _launch(self, module: str, *args: str) -> None:
        subprocess.Popen([sys.executable, "-m", module, *args], cwd=str(pipeline.PROJECT_ROOT))

    def _scan_paths(self) -> tuple[str, str] | None:
        stem = self._selected()
        if stem is None:
            return None
        return (
            str(pipeline.RAW_DIR / f"{stem}.zip"),
            str(pipeline.CACHE_ROOT / stem / "cloud.ply"),
        )

    def _open_3d(self) -> None:
        stem = self._selected()
        if stem:
            self._launch("src.place3d", "--scan", stem, "--bin-type", self.bin_type.get())
            self._set_status("Åpner 3D-visning (plassering + skyve-sti) — bla med pil venstre/høyre …")

    def _open_reconstruction(self) -> None:
        stem = self._selected()
        if stem:
            self._launch("src.reconstruct3d", "--scan", stem, "--bin-type", self.bin_type.get())
            self._set_status("Åpner 3D-rekonstruksjon (dukkehus) — gulv, vegger, tak, dører/vinduer, kasser …")

    def _open_stats(self) -> None:
        self._set_status("Genererer statistikk …")

        def work() -> None:
            try:
                from . import stats_report
                path = stats_report.build()
                import webbrowser
                webbrowser.open(Path(path).as_uri())
                self._set_status(f"Åpnet statistikk i nettleseren: {path}")
            except Exception as error:  # noqa: BLE001 - surface any failure in the status bar
                self._set_status(f"Feil ved statistikk: {error}")

        threading.Thread(target=work, daemon=True).start()

    def _annotate(self) -> None:
        stem = self._selected()
        if stem:
            self._launch("src.annotate3d", "--scan", stem)
            self._set_status("Åpner annoteringsverktøyet …")

    def _prepare(self) -> None:
        self._generate([self._selected()])

    # ---------- live refresh: watch annotation/entrance files ----------

    def _file_signature(self) -> dict[str, tuple[float, float]]:
        signature = {}
        for stem in pipeline.list_scans():
            annotation = pipeline.ANNOTATION_DIR / f"{stem}.json"
            entrance = ENTRANCE_DIR / f"{stem}.json"
            signature[stem] = (
                annotation.stat().st_mtime if annotation.exists() else 0.0,
                entrance.stat().st_mtime if entrance.exists() else 0.0,
            )
        return signature

    def _refresh_if_changed(self) -> None:
        try:
            signature = self._file_signature()
            changed = [stem for stem, value in signature.items() if self._signature.get(stem) != value]
            if changed:
                self._signature = signature
                self._addr_cache.clear()
                self._populate_keep_selection()
                selected = self._selected()
                if selected in changed and not self._busy:
                    self._set_status(f"Annotering endret — oppdaterer {selected} …")
                    self._generate([selected])
        except Exception:
            pass

    def _poll(self) -> None:
        self._refresh_if_changed()
        self.root.after(1500, self._poll)

    def run(self) -> None:
        self.root.after(1500, self._poll)
        self.root.mainloop()


def main() -> None:
    Dashboard().run()


if __name__ == "__main__":
    main()
