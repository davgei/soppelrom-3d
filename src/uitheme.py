"""Shared UI palette + scaling for the three GUIs (dashboard.py, place3d.py, annotate3d.py).

WHY THIS MODULE EXISTS
    Every GUI used to carry its own private colour constants (dashboard.py had #0d0d0d/#3987e5,
    place3d.py had (0.10, 0.80, 0.10) …). They drifted apart from each other and from the rendered
    previews, so "green" meant three different greens depending on which window you were looking at.
    Here the colours are derived ONCE from src/style.py — the palette the PNG previews are drawn
    with — and handed out in whatever form the toolkit needs:

        hex strings   "#3fc46e"          -> tkinter / ttk
        0..1 floats   (0.25, 0.77, 0.43) -> open3d.visualization.gui.Color, materials, paint_uniform_color
        0..255 ints                      -> PIL / anything else
        BGR tuples                       -> back into style.py / OpenCV, for round-trip checks

    style.py stores BGR (OpenCV's order). Reversing it in each call site is exactly how red and blue
    get silently swapped, so the reversal happens in _from_style() below and nowhere else, and
    _assert_channel_order() runs at import time to catch it if anyone breaks it.

COLOUR SEMANTICS — identical to style.py, do not repurpose:
    green   = proposed NEW bin / free floor          red     = EXISTING bin / occupied floor
    blue    = push-path to the door                  magenta = entrance ("inngang")
    amber   = measurements / dimensions

    Because those four are spoken for by the *scene*, the UI accent is deliberately NOT one of them:
    it is a deeper, desaturated blue (PATH mixed down into the panel) used only for chrome —
    selection, focus ring, the one primary button. A selected list row can therefore never be
    mistaken for a push-path.

USAGE
    from . import uitheme as T

    # tkinter
    root.configure(bg=T.WINDOW_BG)
    s = T.tk_scale_for(root)                       # recompute on <Configure>
    style.configure("TLabel", background=T.PANEL_BG, foreground=T.TEXT, font=s.font("body"))
    frame.grid(padx=s.pad_m, pady=s.pad_s)

    # Open3D
    panel = gui.Vert(T.em(window, 0.4), T.margins(window, 0.6))
    widget.set_background(T.gui_color("scene_bg"))
    material.base_color = T.rgba("new_bin", 0.78)

Run `.venv\\Scripts\\python.exe -m src.uitheme` to print every role with hex + RGB + contrast.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import style

RGB255 = tuple[int, int, int]
RGBf = tuple[float, float, float]


# ---------------------------------------------------------------- conversion (the ONLY place)

def _from_style(bgr: tuple[int, int, int]) -> RGB255:
    """style.py / OpenCV keeps (B, G, R); every UI toolkit wants (R, G, B). Reverse here only."""
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _hex(rgb: RGB255) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(c))) for c in rgb))


def _parse_hex(value: str) -> RGB255:
    text = value.lstrip("#")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


def _mix(a: RGB255, b: RGB255, t: float) -> RGB255:
    """Linear blend, t=0 -> a, t=1 -> b. Used to derive hover/edge tones from the base palette."""
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(ca + (cb - ca) * t)) for ca, cb in zip(a, b))  # type: ignore[return-value]


_WHITE: RGB255 = (255, 255, 255)
_BLACK: RGB255 = (0, 0, 0)


# ---------------------------------------------------------------- the role table

_PANEL = _from_style(style.PANEL)          # #101418  card surface
_BACKDROP = _from_style(style.BACKDROP)    # #0b0e11  the mat everything sits on
_PATH = _from_style(style.PATH)            # #3d8bfd  push-path blue

# Roles are ordered for the printout: surfaces, then text, then interaction, then scene semantics.
# Derived tones exist so no caller invents a one-off colour for a hover state.
ROLES: dict[str, RGB255] = {
    # -- surfaces -------------------------------------------------------------
    "window_bg":      _BACKDROP,
    "panel_bg":       _PANEL,
    # a card *on* a panel: lifted by 5% white, the same trick style.rounded_rect uses
    "panel_bg_alt":   _mix(_PANEL, _WHITE, 0.05),
    "field_bg":       _mix(_PANEL, _BLACK, 0.35),        # input well: recessed, not raised
    "hover_bg":       _mix(_PANEL, _WHITE, 0.10),
    "active_bg":      _mix(_PANEL, _WHITE, 0.17),
    # style.py draws its hairline as PANEL_EDGE (white) at alpha 0.22 over PANEL; an opaque widget
    # border cannot use alpha, so pre-compose the identical result here.
    "panel_edge":     _mix(_PANEL, _from_style(style.PANEL_EDGE), 0.22),
    "divider":        _mix(_PANEL, _WHITE, 0.11),
    "scene_bg":       _from_style(style.NO_DATA),        # empty 3D space: "nothing here", not black
    "shadow":         _from_style(style.SHADOW),
    "paper":          _from_style(style.PAPER),          # light surface (print/report views)

    # -- text -----------------------------------------------------------------
    "text":           _from_style(style.PANEL_TEXT),
    "text_muted":     _from_style(style.MUTED),
    "text_disabled":  _mix(_from_style(style.MUTED), _PANEL, 0.52),
    "text_on_accent": _WHITE,
    "ink":            _from_style(style.INK),            # text ON "paper"

    # -- interaction ----------------------------------------------------------
    # INDIGO, deliberately NOT derived from a scene colour. The accent used to be a duller push-path
    # blue, which made one colour mean two things: "the route a bin is wheeled along" in the images and
    # "the control you should press" in the UI. Indigo sits clear of all four semantic colours (green
    # new bin, red existing bin, blue path, magenta entrance) and of the amber clutter/dimension tone,
    # so it can never be read as scene meaning.
    "accent":         (0x5F, 0x57, 0xD8),
    "accent_hover":   _mix((0x5F, 0x57, 0xD8), _WHITE, 0.16),
    "accent_muted":   _mix((0x5F, 0x57, 0xD8), _PANEL, 0.55),          # disabled primary button
    "focus_ring":     _mix((0x5F, 0x57, 0xD8), _WHITE, 0.34),
    "selection_bg":   _mix((0x5F, 0x57, 0xD8), _PANEL, 0.18),
    "selection_fg":   _WHITE,

    # -- status ---------------------------------------------------------------
    "success":        _from_style(style.NEW_BIN),        # ferdig / godkjent
    "warning":        _from_style(style.DIMENSION),      # mangler noe
    "danger":         _from_style(style.EXISTING_BIN),   # feil / slett

    # -- scene semantics (must match the rendered previews exactly) -----------
    "new_bin":            _from_style(style.NEW_BIN),
    "new_bin_edge":       _from_style(style.NEW_BIN_EDGE),
    "existing_bin":       _from_style(style.EXISTING_BIN),
    "existing_bin_edge":  _from_style(style.EXISTING_BIN_EDGE),
    "path":               _from_style(style.PATH),
    "path_soft":          _from_style(style.PATH_SOFT),
    "entrance":           _from_style(style.ENTRANCE),
    "free_floor":         _from_style(style.FREE_FLOOR),
    "occupied_floor":     _from_style(style.OCCUPIED_FLOOR),
    "unknown_floor":      _from_style(style.UNKNOWN_FLOOR),
    "room_outline":       _from_style(style.ROOM_OUTLINE),
    "dimension":          _from_style(style.DIMENSION),
    "no_data":            _from_style(style.NO_DATA),
}

HEX: dict[str, str] = {name: _hex(rgb) for name, rgb in ROLES.items()}
RGB: dict[str, RGBf] = {
    name: (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0) for name, rgb in ROLES.items()
}

# Norwegian labels for the UI legends, reusing style.LABELS so the wording cannot drift either.
LABELS: dict[str, str] = dict(style.LABELS)


# ---------------------------------------------------------------- lookups

class _UnknownRole(KeyError):
    pass


def _lookup(role: str) -> RGB255:
    try:
        return ROLES[role]
    except KeyError:
        raise _UnknownRole(
            f"ukjent UI-rolle {role!r}. Gyldige: {', '.join(sorted(ROLES))}"
        ) from None


def hex_of(role: str) -> str:
    """'#rrggbb' for tkinter/ttk."""
    return _hex(_lookup(role))


def rgb_of(role: str) -> RGBf:
    """(r, g, b) as 0..1 floats — Open3D gui.Color, materials, paint_uniform_color."""
    r, g, b = _lookup(role)
    return (r / 255.0, g / 255.0, b / 255.0)


def rgb255_of(role: str) -> RGB255:
    return _lookup(role)


def bgr_of(role: str) -> RGB255:
    """Back to OpenCV order, so a GUI overlay can be drawn with style.py helpers."""
    r, g, b = _lookup(role)
    return (b, g, r)


def rgba(role: str, alpha: float = 1.0) -> list[float]:
    """[r, g, b, a] 0..1 — what Open3D's MaterialRecord.base_color expects."""
    r, g, b = rgb_of(role)
    return [r, g, b, float(alpha)]


def gui_color(role: str, alpha: float = 1.0):
    """open3d.visualization.gui.Color for the role. Imported lazily so tkinter-only code (and the
    __main__ printout) never pays for / requires Open3D."""
    from open3d.visualization import gui  # noqa: PLC0415  (lazy on purpose)

    r, g, b = rgb_of(role)
    return gui.Color(r, g, b, float(alpha))


def mix(role_a: str, role_b: str, t: float) -> str:
    """Hex blend between two roles — for a hover tint of an existing colour, never a new hue."""
    return _hex(_mix(_lookup(role_a), _lookup(role_b), t))


def tint(role: str, t: float) -> str:
    """t > 0 lightens toward white, t < 0 darkens toward black. Hex out."""
    base = _lookup(role)
    return _hex(_mix(base, _WHITE if t >= 0 else _BLACK, abs(t)))


# Flat constants for the colours used on almost every line of GUI code.
WINDOW_BG = HEX["window_bg"]
PANEL_BG = HEX["panel_bg"]
PANEL_BG_ALT = HEX["panel_bg_alt"]
FIELD_BG = HEX["field_bg"]
HOVER_BG = HEX["hover_bg"]
ACTIVE_BG = HEX["active_bg"]
PANEL_EDGE = HEX["panel_edge"]
DIVIDER = HEX["divider"]
TEXT = HEX["text"]
TEXT_MUTED = HEX["text_muted"]
TEXT_DISABLED = HEX["text_disabled"]
TEXT_ON_ACCENT = HEX["text_on_accent"]
ACCENT = HEX["accent"]
ACCENT_HOVER = HEX["accent_hover"]
ACCENT_MUTED = HEX["accent_muted"]
FOCUS_RING = HEX["focus_ring"]
SUCCESS = HEX["success"]
WARNING = HEX["warning"]
DANGER = HEX["danger"]
NEW_BIN = HEX["new_bin"]
EXISTING_BIN = HEX["existing_bin"]
PATH = HEX["path"]
ENTRANCE = HEX["entrance"]


# ---------------------------------------------------------------- contrast (readability guard)

def _rel_luminance(rgb: RGB255) -> float:
    def channel(value: int) -> float:
        c = value / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(role_a: str, role_b: str) -> float:
    """WCAG contrast ratio (1..21) between two roles. Use it before putting text on a surface."""
    la, lb = _rel_luminance(_lookup(role_a)), _rel_luminance(_lookup(role_b))
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# ---------------------------------------------------------------- fonts

# Fallback chains, most-wanted first. Windows ships Segoe UI; the last entry of each chain is a Tk
# built-in alias that always resolves, so a machine without any of the named families still renders.
FONT_UI: tuple[str, ...] = ("Segoe UI", "Segoe UI Variable Text", "Helvetica Neue", "DejaVu Sans",
                            "TkDefaultFont")
FONT_UI_SEMIBOLD: tuple[str, ...] = ("Segoe UI Semibold", "Segoe UI Variable Display", "Segoe UI",
                                     "DejaVu Sans", "TkDefaultFont")
FONT_MONO: tuple[str, ...] = ("Cascadia Mono", "Consolas", "DejaVu Sans Mono", "TkFixedFont")

_family_cache: dict[tuple[str, ...], str] = {}


def resolve_family(chain: tuple[str, ...] = FONT_UI) -> str:
    """First family in `chain` that Tk actually has. Requires a Tk root to exist; without one (or
    without tkinter at all) it returns chain[0], which Tk resolves to its default font anyway."""
    if chain in _family_cache:
        return _family_cache[chain]
    picked, verified = chain[0], False
    try:
        import tkinter  # noqa: PLC0415
        from tkinter import font as tkfont  # noqa: PLC0415

        if tkinter._default_root is not None:  # type: ignore[attr-defined]
            available = {name.lower() for name in tkfont.families()}
            for name in chain:
                if name.lower() in available or name.startswith("Tk"):
                    picked, verified = name, True
                    break
    except Exception:  # noqa: BLE001 - a missing/half-initialised Tk must not break styling
        pass
    if verified:
        # Only cache a real answer: caching the unverified first choice before Tk exists would pin a
        # possibly-missing family (e.g. "Segoe UI Semibold") for the whole session.
        _family_cache[chain] = picked
    return picked


def o3d_font_path() -> str | None:
    """Path to a TTF Open3D can load with gui.FontDescription, so the 3D panels get Segoe UI and can
    show æ/ø/å. None when no candidate exists (then Open3D keeps its built-in font).

    Reads style.py's own font search lists (read-only) so the GUIs and the PNGs use the same file.
    """
    dirs = getattr(style, "_FONT_DIRS", [Path(r"C:\Windows\Fonts")])
    names = getattr(style, "_FONT_FILES", {}).get("regular", ("segoeui.ttf", "DejaVuSans.ttf"))
    for name in names:
        for directory in dirs:
            candidate = Path(directory) / name
            if candidate.exists():
                return str(candidate)
    return None


# ---------------------------------------------------------------- Open3D scaling

def em(window, factor: float = 1.0, minimum: int = 1) -> int:
    """`factor` × the theme's font size, in pixels — the only way Open3D sizes should be written.

    Open3D has no layout manager that reflows, so every width/height/spacing is a number; making
    that number a multiple of the font size is what makes the panel follow DPI and the user's font
    size instead of clipping its buttons at 150% scaling.

    Accepts a gui.Window, a gui.Theme, or a plain int font size (handy in tests).
    """
    return max(int(minimum), int(round(font_size(window) * float(factor))))


def emf(window, factor: float = 1.0) -> float:
    """Float variant, for gui.Margins / gui.Vert spacing which take floats."""
    return font_size(window) * float(factor)


def font_size(window) -> int:
    """Theme font size in px from a gui.Window, a gui.Theme or an int. Defaults to 16 if unknown."""
    if isinstance(window, (int, float)):
        return int(window)
    theme = getattr(window, "theme", window)
    return int(getattr(theme, "font_size", 16) or 16)


def margins(window, all: float | None = None, *, left: float | None = None,
            top: float | None = None, right: float | None = None,
            bottom: float | None = None):
    """gui.Margins in em. `margins(w, 0.6)` = 0.6 em on all sides; per-side values override it."""
    from open3d.visualization import gui  # noqa: PLC0415

    base = 0.0 if all is None else float(all)
    pick = lambda side: emf(window, base if side is None else float(side))  # noqa: E731
    return gui.Margins(pick(left), pick(top), pick(right), pick(bottom))


# ---------------------------------------------------------------- tkinter scaling

# Reference window the type scale was designed at. Bigger windows get slightly bigger type, small
# ones slightly smaller — damped, so text never becomes unreadable on a 1000x700 window.
BASE_WIDTH, BASE_HEIGHT = 1440, 900

_FONT_STEPS = {           # points at factor 1.0
    "display": 22,
    "h1": 17,
    "h2": 13,
    "body": 10,
    "small": 9,
    "tiny": 8,
}
_PAD_STEPS = {            # pixels at factor 1.0 / 96 dpi
    "pad_xs": 2,
    "pad_s": 6,
    "pad_m": 10,
    "pad_l": 16,
    "pad_xl": 24,
}


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


@dataclass(frozen=True)
class TkScale:
    """One resolved set of font sizes and paddings for a given window size / DPI.

    Font sizes are POSITIVE, i.e. points: Tk multiplies them by its own dpi scaling, so DPI must NOT
    be applied to them again here. Paddings are pixels, so those DO get the dpi factor — that
    asymmetry is why both numbers live in this one object.
    """

    factor: float          # window-size factor applied to fonts
    px_factor: float       # window-size × dpi factor applied to pixel measures
    tk_scaling: float      # pixels per point, as reported by Tk
    display: int
    h1: int
    h2: int
    body: int
    small: int
    tiny: int
    pad_xs: int
    pad_s: int
    pad_m: int
    pad_l: int
    pad_xl: int
    row_height: int        # ttk.Treeview rowheight
    thumb_min: int         # smallest sensible preview-image side
    sidebar_min: int       # smallest sensible scan-list width

    def px(self, base_px: float) -> int:
        """Scale a pixel measure that has no named step (icon sizes, canvas paddings …)."""
        return max(1, int(round(float(base_px) * self.px_factor)))

    def points(self, step: str = "body") -> int:
        return int(getattr(self, step))

    def font(self, step: str = "body", weight: str = "regular") -> tuple:
        """A Tk font spec tuple, e.g. ('Segoe UI', 10) or ('Segoe UI', 17, 'bold').

        weight: 'regular' | 'semibold' | 'bold' | 'italic' | 'mono'. 'semibold' uses the Segoe UI
        Semibold family when present and falls back to synthetic bold, because ttk cannot express a
        weight between regular and bold any other way.
        """
        size = self.points(step)
        if weight == "mono":
            return (resolve_family(FONT_MONO), size)
        if weight == "semibold":
            family = resolve_family(FONT_UI_SEMIBOLD)
            if family == FONT_UI_SEMIBOLD[0]:      # the real semibold face exists
                return (family, size)
            return (resolve_family(FONT_UI), size, "bold")
        if weight in ("bold", "italic"):
            return (resolve_family(FONT_UI), size, weight)
        return (resolve_family(FONT_UI), size)


def tk_scale(width: float = BASE_WIDTH, height: float = BASE_HEIGHT,
             tk_scaling: float = 96.0 / 72.0) -> TkScale:
    """Derive fonts + paddings from a window size (px) and Tk's scaling (pixels per point).

    The size factor is damped with a square root: a window at half the reference width gets ~0.85×
    type, not 0.5× — the user resizes to see more rows, not to read smaller text.
    """
    raw = min(float(width) / BASE_WIDTH, float(height) / BASE_HEIGHT)
    factor = _clamp(max(raw, 0.01) ** 0.5, 0.86, 1.45)
    dpi_factor = _clamp(float(tk_scaling) / (96.0 / 72.0), 0.75, 2.5)
    px_factor = factor * dpi_factor
    fonts = {name: max(7, int(round(pt * factor))) for name, pt in _FONT_STEPS.items()}
    pads = {name: max(1, int(round(px * px_factor))) for name, px in _PAD_STEPS.items()}
    return TkScale(
        factor=round(factor, 4),
        px_factor=round(px_factor, 4),
        tk_scaling=round(float(tk_scaling), 4),
        row_height=max(20, int(round(30 * px_factor))),
        thumb_min=max(240, int(round(420 * px_factor))),
        sidebar_min=max(220, int(round(320 * px_factor))),
        **fonts,
        **pads,
    )


def tk_scale_for(widget) -> TkScale:
    """TkScale for a live widget/root: reads its current size and the toolkit's dpi scaling.

    Call this from a <Configure> handler and re-apply the fonts; that is what makes the dashboard
    usable both at 1000x700 and maximised on a 4K screen.
    """
    width = height = 0
    scaling = 96.0 / 72.0
    try:
        widget.update_idletasks()
        width, height = widget.winfo_width(), widget.winfo_height()
        if width <= 1 or height <= 1:                    # not mapped yet -> fall back to the request
            width, height = widget.winfo_reqwidth(), widget.winfo_reqheight()
        scaling = float(widget.tk.call("tk", "scaling"))
    except Exception:  # noqa: BLE001 - never let styling crash the app
        pass
    if width <= 1 or height <= 1:
        width, height = BASE_WIDTH, BASE_HEIGHT
    return tk_scale(width, height, scaling)


# ---------------------------------------------------------------- self-test

def _assert_channel_order() -> None:
    """Catch a BGR/RGB swap the instant it is introduced.

    A swap is invisible in code review (both are 3-tuples) but turns every "grønn ny kasse" into a
    blue one. These four assertions pin the hue of the four semantic colours, and the round trip
    pins the conversion itself.
    """
    green = ROLES["new_bin"]
    red = ROLES["existing_bin"]
    blue = ROLES["path"]
    magenta = ROLES["entrance"]
    problems = []
    if not (green[1] > green[0] and green[1] > green[2]):
        problems.append(f"new_bin skal være grønn, er {_hex(green)}")
    if not (red[0] > red[1] and red[0] > red[2]):
        problems.append(f"existing_bin skal være rød, er {_hex(red)}")
    if not (blue[2] > blue[1] and blue[2] > blue[0]):
        problems.append(f"path skal være blå, er {_hex(blue)}")
    if not (magenta[0] > magenta[1] and magenta[2] > magenta[1]):
        problems.append(f"entrance skal være magenta, er {_hex(magenta)}")
    if bgr_of("new_bin") != tuple(style.NEW_BIN):
        problems.append("bgr_of() går ikke tilbake til style.py-verdien")
    if _parse_hex(HEX["new_bin"]) != ROLES["new_bin"]:
        problems.append("hex-strengen stemmer ikke med RGB-verdien")
    if problems:
        raise RuntimeError("uitheme: fargekanalene er byttet om — " + "; ".join(problems))


_assert_channel_order()


def selftest() -> list[str]:
    """Return a list of human-readable warnings (empty = all good). Contrast is a warning, not an
    error, so a deliberate low-contrast decoration cannot block the app from starting."""
    _assert_channel_order()
    warnings: list[str] = []
    checks = [
        ("text", "panel_bg", 7.0),
        ("text", "window_bg", 7.0),
        ("text_muted", "panel_bg", 4.5),
        ("text_on_accent", "accent", 4.5),
        ("selection_fg", "selection_bg", 4.5),
        ("ink", "paper", 7.0),
        ("success", "panel_bg", 3.0),
        ("danger", "panel_bg", 3.0),
        ("warning", "panel_bg", 3.0),
    ]
    for fg, bg, minimum in checks:
        ratio = contrast(fg, bg)
        if ratio < minimum:
            warnings.append(f"lav kontrast {fg} på {bg}: {ratio:.2f} (< {minimum})")
    return warnings


def _main() -> None:
    print(f"uitheme — {len(ROLES)} roller, avledet fra src/style.py\n")
    print(f"{'rolle':<20} {'hex':<9} {'rgb 0-255':<16} {'rgb 0-1':<24} kontrast mot panel_bg")
    print("-" * 96)
    for name in ROLES:
        r255 = ROLES[name]
        rf = RGB[name]
        floats = f"({rf[0]:.3f}, {rf[1]:.3f}, {rf[2]:.3f})"
        ratio = contrast(name, "panel_bg")
        print(f"{name:<20} {HEX[name]:<9} {str(r255):<16} {floats:<24} {ratio:5.2f}")

    print("\nSkalering (Open3D, em):")
    for fs in (12, 16, 24):
        print(f"  font_size={fs:>3} px ->  em×1={em(fs)}  em×0.4={em(fs, 0.4)}  "
              f"em×18={em(fs, 18)}  (panelbredde)")

    print("\nSkalering (tkinter):")
    header = f"  {'vindu':<12} {'faktor':>6} {'px':>5} {'h1':>4} {'body':>5} {'small':>6} " \
             f"{'pad_m':>6} {'rad':>4} {'sidebar':>8}"
    print(header)
    for w, h, sc in ((1000, 700, 96 / 72), (1440, 900, 96 / 72), (1920, 1080, 96 / 72),
                     (2560, 1440, 96 / 72), (1440, 900, 144 / 72)):
        s = tk_scale(w, h, sc)
        tag = f"{w}x{h}" + ("@2x" if sc > 1.5 else "")
        print(f"  {tag:<12} {s.factor:>6.2f} {s.px_factor:>5.2f} {s.h1:>4} {s.body:>5} "
              f"{s.small:>6} {s.pad_m:>6} {s.row_height:>4} {s.sidebar_min:>8}")

    print(f"\nFonter: ui={FONT_UI[0]!r} kjede={FONT_UI}")
    print(f"        semibold kjede={FONT_UI_SEMIBOLD}")
    print(f"        mono kjede={FONT_MONO}")
    print(f"        Open3D TTF: {o3d_font_path()}")

    problems = selftest()
    print("\nSelvtest: " + ("OK — kanalrekkefølge og kontrast i orden" if not problems
                            else "\n  ".join(["advarsler:"] + problems)))


if __name__ == "__main__":
    _main()
