"""Shared visual language for every rendered preview: palette, typography, plan furniture.

Why this module exists: cv2.putText cannot draw æ/ø/å or ², so labels came out as "gronn" and
"m2" — debug output, not something to show a decision-maker. Every string therefore goes through
PIL (Segoe UI) and is alpha-blitted onto the OpenCV BGR image. Colours, panels, legend, scale bar
and dimension lines live here as well so the previews, the floor plan and the 3D snapshots read as
one product instead of five unrelated screenshots.

Colour semantics (do not repurpose):
    green  = proposed NEW bin / free floor      red     = EXISTING bin / occupied floor
    blue   = push-path to the door              magenta = entrance ("inngang")
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BGR = tuple[int, int, int]


def hex_to_bgr(value: str) -> BGR:
    """'#3fc46e' -> (110, 196, 63) in OpenCV's BGR order."""
    text = value.lstrip("#")
    return (int(text[4:6], 16), int(text[2:4], 16), int(text[0:2], 16))


def shade(color: BGR, factor: float) -> BGR:
    """factor < 1 darkens, > 1 lightens. Used for edges/fills of the same semantic colour."""
    return tuple(int(np.clip(channel * factor, 0, 255)) for channel in color)  # type: ignore[return-value]


# ---------------------------------------------------------------- palette

INK = hex_to_bgr("#0f1214")          # text on light surfaces
PAPER = hex_to_bgr("#f7f6f2")        # light surface (matches floorplan.py)
PANEL = hex_to_bgr("#101418")        # translucent panel base (over photo-like scenes)
PANEL_TEXT = hex_to_bgr("#f5f6f7")
PANEL_EDGE = hex_to_bgr("#ffffff")
MUTED = hex_to_bgr("#bcc3c9")        # secondary text on panels

NEW_BIN = hex_to_bgr("#3fc46e")      # proposed new bin
NEW_BIN_EDGE = hex_to_bgr("#1b7c45")
EXISTING_BIN = hex_to_bgr("#ef5b4c")  # bin already in the room
EXISTING_BIN_EDGE = hex_to_bgr("#8f281d")
PATH = hex_to_bgr("#3d8bfd")         # push-path to the door
PATH_SOFT = hex_to_bgr("#8fb8ff")    # the wider corridor around the route
FREE_FLOOR = hex_to_bgr("#4cc47a")
OCCUPIED_FLOOR = hex_to_bgr("#e35d52")
UNKNOWN_FLOOR = hex_to_bgr("#5a6169")
BACKDROP = hex_to_bgr("#0b0e11")     # the mat a plan is mounted on
NO_DATA = hex_to_bgr("#151a1f")      # never scanned: readable as "nothing here", not pure black
ENTRANCE = hex_to_bgr("#e0219a")     # same magenta as the CAD floor plan
ROOM_OUTLINE = hex_to_bgr("#f2f5f6")
DIMENSION = hex_to_bgr("#ffc94d")    # measurement lines / scale bar accent
SHADOW = hex_to_bgr("#000000")

PALETTE: dict[str, BGR] = {
    "ink": INK,
    "paper": PAPER,
    "panel": PANEL,
    "panel_text": PANEL_TEXT,
    "muted": MUTED,
    "new_bin": NEW_BIN,
    "new_bin_edge": NEW_BIN_EDGE,
    "existing_bin": EXISTING_BIN,
    "existing_bin_edge": EXISTING_BIN_EDGE,
    "path": PATH,
    "path_soft": PATH_SOFT,
    "free_floor": FREE_FLOOR,
    "occupied_floor": OCCUPIED_FLOOR,
    "unknown_floor": UNKNOWN_FLOOR,
    "backdrop": BACKDROP,
    "no_data": NO_DATA,
    "entrance": ENTRANCE,
    "room_outline": ROOM_OUTLINE,
    "dimension": DIMENSION,
}

# Norwegian wording, in one place so every view says the same thing.
LABELS: dict[str, str] = {
    "free": "Ledig gulv",
    "occupied": "Opptatt gulv",
    "unknown": "Ikke skannet",
    "new": "Forslag: ny kasse",
    "existing": "Eksisterende kasse",
    "path": "Skyve-sti til inngang",
    "entrance": "Inngang",
    "room": "Rommets yttermål",
    "today": "I dag",
    "proposal": "Forslag",
    "topdown": "sett ovenfra",
}


# ---------------------------------------------------------------- numbers

def num(value: float, decimals: int = 1) -> str:
    """Norwegian number formatting: decimal comma, thin-space thousands."""
    text = f"{value:,.{decimals}f}".replace(",", "\u2009").replace(".", ",")
    return text


def fmt_m(value: float, decimals: int = 2) -> str:
    return f"{num(value, decimals)} m"


def fmt_m2(value: float, decimals: int = 1) -> str:
    return f"{num(value, decimals)} m²"


# ---------------------------------------------------------------- fonts

_FONT_DIRS = [
    Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/Library/Fonts"),
]
_FONT_FILES = {
    "light": ("segoeuil.ttf", "segoeui.ttf", "DejaVuSans.ttf"),
    "regular": ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"),
    "semibold": ("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"),
    "bold": ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"),
}


@lru_cache(maxsize=256)
def get_font(size: int, weight: str = "regular"):
    """Segoe UI at `size` px. Degrades to any bundled TrueType, then to PIL's bitmap default, so a
    machine without the Windows fonts still renders (just less pretty)."""
    for name in _FONT_FILES.get(weight, _FONT_FILES["regular"]):
        for directory in _FONT_DIRS:
            path = directory / name
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size)
                except OSError:
                    continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _line_height(font) -> int:
    ascent, descent = font.getmetrics()
    return int(ascent + descent)


def text_size(
    text: str,
    size: int = 22,
    weight: str = "regular",
    line_spacing: float = 1.3,
) -> tuple[int, int]:
    """(width, height) in px of `text` (may contain \\n). Height uses font metrics, not the glyph
    bbox, so rows of labels line up whether or not they contain descenders."""
    font = get_font(size, weight)
    lines = str(text).split("\n")
    line_h = _line_height(font)
    gap = int(round(size * (line_spacing - 1.0)))
    width = max((int(round(font.getlength(line))) for line in lines), default=0)
    return width, line_h * len(lines) + gap * max(len(lines) - 1, 0)


# ---------------------------------------------------------------- low-level blending

def _clip_region(image: np.ndarray, x: int, y: int, w: int, h: int):
    """Intersect a (x, y, w, h) placement with the image; returns image-slice and source-slice."""
    height, width = image.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return (slice(y0, y1), slice(x0, x1)), (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))


def blend_mask(image: np.ndarray, x: int, y: int, mask: np.ndarray, color: BGR, alpha: float = 1.0) -> None:
    """Alpha-composite a flat `color` into `image` through an 8-bit `mask` placed at (x, y)."""
    clipped = _clip_region(image, x, y, mask.shape[1], mask.shape[0])
    if clipped is None:
        return
    dst, src = clipped
    weight = (mask[src].astype(np.float32) / 255.0 * float(alpha))[..., None]
    roi = image[dst].astype(np.float32)
    image[dst] = np.clip(roi * (1.0 - weight) + np.asarray(color, np.float32) * weight, 0, 255).astype(np.uint8)


def blit_rgba(image: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    """Composite an RGBA (PIL-order) layer onto a BGR OpenCV image at (x, y)."""
    clipped = _clip_region(image, x, y, rgba.shape[1], rgba.shape[0])
    if clipped is None:
        return
    dst, src = clipped
    patch = rgba[src]
    weight = (patch[..., 3:4].astype(np.float32) / 255.0)
    bgr = patch[..., 2::-1].astype(np.float32)
    roi = image[dst].astype(np.float32)
    image[dst] = np.clip(roi * (1.0 - weight) + bgr * weight, 0, 255).astype(np.uint8)


_SS = 4  # supersample factor: PIL draws the shape 4x too big, the downscale gives the anti-aliasing


def _rounded_mask(w: int, h: int, radius: int, outline: int = 0) -> np.ndarray:
    w, h = max(int(w), 1), max(int(h), 1)
    radius = int(np.clip(radius, 0, min(w, h) // 2))
    big = Image.new("L", (w * _SS, h * _SS), 0)
    draw = ImageDraw.Draw(big)
    shape = (0, 0, w * _SS - 1, h * _SS - 1)
    if outline > 0:
        draw.rounded_rectangle(shape, radius=radius * _SS, outline=255, width=max(outline * _SS, 1))
    else:
        draw.rounded_rectangle(shape, radius=radius * _SS, fill=255)
    return np.asarray(big.resize((w, h), Image.LANCZOS))


def rounded_rect(
    image: np.ndarray,
    xyxy: tuple[int, int, int, int],
    *,
    color: BGR = PANEL,
    alpha: float = 0.74,
    radius: int = 14,
    edge_color: BGR | None = PANEL_EDGE,
    edge_alpha: float = 0.22,
    edge_thickness: int = 2,
) -> None:
    """Anti-aliased translucent panel with an optional hairline edge (in place)."""
    x0, y0, x1, y1 = (int(round(v)) for v in xyxy)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return
    blend_mask(image, x0, y0, _rounded_mask(w, h, radius), color, alpha)
    if edge_color is not None and edge_alpha > 0:
        blend_mask(image, x0, y0, _rounded_mask(w, h, radius, outline=edge_thickness), edge_color, edge_alpha)


# ---------------------------------------------------------------- text

_ANCHOR_X = {"l": 0.0, "c": 0.5, "m": 0.5, "r": 1.0}
_ANCHOR_Y = {"t": 0.0, "c": 0.5, "m": 0.5, "b": 1.0}


def draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    size: int = 22,
    color: BGR = PANEL_TEXT,
    weight: str = "regular",
    anchor: str = "lt",
    align: str = "left",
    line_spacing: float = 1.3,
    halo: int = 2,
    halo_color: BGR | None = None,
    halo_alpha: float = 0.85,
    box: bool = False,
    box_color: BGR = PANEL,
    box_alpha: float = 0.74,
    box_pad: tuple[int, int] = (14, 9),
    box_radius: int = 12,
    box_edge: BGR | None = PANEL_EDGE,
    box_edge_alpha: float = 0.22,
) -> tuple[int, int, int, int]:
    """Draw `text` (supports \\n) through PIL onto a BGR image so æ/ø/å/² render properly.

    `anchor` is two chars, horizontal (l/c/r) then vertical (t/c/b), and refers to the text block —
    or to the pill when `box=True`. The halo is a stroke around the glyphs; it is what keeps small
    labels readable on top of a photo-like scan. Returns the drawn bounds (x0, y0, x1, y1),
    including the pill, so callers can stack blocks.
    """
    text = str(text)
    font = get_font(size, weight)
    lines = text.split("\n")
    line_h = _line_height(font)
    gap = int(round(size * (line_spacing - 1.0)))
    widths = [int(round(font.getlength(line))) for line in lines]
    tw, th = max(widths + [0]), line_h * len(lines) + gap * max(len(lines) - 1, 0)

    pad_x, pad_y = (box_pad if box else (0, 0))
    total_w, total_h = tw + 2 * pad_x, th + 2 * pad_y
    left = int(round(org[0] - _ANCHOR_X.get(anchor[0], 0.0) * total_w))
    top = int(round(org[1] - _ANCHOR_Y.get(anchor[1] if len(anchor) > 1 else "t", 0.0) * total_h))
    if box:
        rounded_rect(
            image, (left, top, left + total_w, top + total_h), color=box_color, alpha=box_alpha,
            radius=box_radius, edge_color=box_edge, edge_alpha=box_edge_alpha,
        )

    if halo_color is None:  # dark halo behind light text, light halo behind dark text
        luma = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        halo_color = SHADOW if luma > 120 else PAPER
    margin = halo + 2
    layer = Image.new("RGBA", (tw + 2 * margin, th + 2 * margin), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    fill = (color[2], color[1], color[0], 255)
    stroke = (halo_color[2], halo_color[1], halo_color[0], int(np.clip(halo_alpha, 0, 1) * 255))
    for index, line in enumerate(lines):
        offset = 0
        if align in ("center", "centre"):
            offset = (tw - widths[index]) // 2
        elif align == "right":
            offset = tw - widths[index]
        draw.text(
            (margin + offset, margin + index * (line_h + gap)), line, font=font, fill=fill,
            anchor="la", stroke_width=halo if halo > 0 else 0, stroke_fill=stroke if halo > 0 else None,
        )
    blit_rgba(image, np.asarray(layer), left + pad_x - margin, top + pad_y - margin)
    return (left, top, left + total_w, top + total_h)


# ---------------------------------------------------------------- shapes

def soft_shadow(
    image: np.ndarray,
    polygons: Sequence[np.ndarray],
    *,
    offset: tuple[int, int] = (0, 5),
    blur: int = 17,
    strength: float = 0.5,
) -> None:
    """Darken a blurred, offset copy of `polygons` so boxes lift off the scan instead of floating
    flat on it. One pass for all polygons (cheap) and only inside their bounding box."""
    polys = [np.asarray(p, np.int32).reshape(-1, 2) for p in polygons if len(p) >= 3]
    if not polys:
        return
    stacked = np.concatenate(polys, axis=0)
    grow = blur * 2 + max(abs(offset[0]), abs(offset[1])) + 4
    height, width = image.shape[:2]
    x0 = int(max(stacked[:, 0].min() - grow, 0))
    y0 = int(max(stacked[:, 1].min() - grow, 0))
    x1 = int(min(stacked[:, 0].max() + grow, width))
    y1 = int(min(stacked[:, 1].max() + grow, height))
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    shifted = [p + np.array([offset[0] - x0, offset[1] - y0], np.int32) for p in polys]
    cv2.fillPoly(mask, shifted, 255, lineType=cv2.LINE_AA)
    kernel = blur if blur % 2 == 1 else blur + 1
    mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    weight = (mask.astype(np.float32) / 255.0 * float(strength))[..., None]
    roi = image[y0:y1, x0:x1].astype(np.float32)
    image[y0:y1, x0:x1] = np.clip(roi * (1.0 - weight), 0, 255).astype(np.uint8)


def draw_polygon(
    image: np.ndarray,
    points: np.ndarray,
    *,
    color: BGR,
    thickness: int = 3,
    fill_alpha: float = 0.0,
    fill_color: BGR | None = None,
    edge_color: BGR | None = None,
    edge_grow: int = 4,
    edge_alpha: float = 0.45,
    closed: bool = True,
) -> None:
    """Anti-aliased outlined polygon: a darker, slightly wider line underneath (edge_*) separates
    the box from whatever it sits on, then the semantic colour on top."""
    pts = np.asarray(points, np.int32).reshape(-1, 1, 2)
    if fill_alpha > 0:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_AA)
        blend_mask(image, 0, 0, mask, fill_color or color, fill_alpha)
    if edge_color is None:
        edge_color = shade(color, 0.45)
    if edge_grow > 0 and edge_alpha > 0:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [pts], closed, 255, thickness + edge_grow, cv2.LINE_AA)
        blend_mask(image, 0, 0, mask, edge_color, edge_alpha)
    cv2.polylines(image, [pts], closed, color, thickness, cv2.LINE_AA)


def marker_dot(
    image: np.ndarray,
    center: tuple[int, int],
    *,
    color: BGR = ENTRANCE,
    radius: int = 11,
    ring: int = 3,
    ring_color: BGR = PANEL_TEXT,
) -> None:
    """A filled dot with a light ring — reads as a placed marker rather than a stray pixel."""
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(image, (cx, cy), radius + ring + 1, shade(color, 0.35), -1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), radius + ring, ring_color, -1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), radius, color, -1, cv2.LINE_AA)


def badge(
    image: np.ndarray,
    center: tuple[int, int],
    text: str,
    *,
    color: BGR = NEW_BIN,
    text_color: BGR = INK,
    size: int = 22,
) -> None:
    """Numbered chip used to label a proposed bin (1, 2, 3 …)."""
    tw, th = text_size(text, size, "bold")
    radius = int(max(tw, th) / 2 + 9)
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(image, (cx, cy), radius + 2, shade(color, 0.4), -1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), radius, color, -1, cv2.LINE_AA)
    draw_text(image, text, (cx, cy), size=size, weight="bold", color=text_color, anchor="cc", halo=0)


# ---------------------------------------------------------------- plan furniture

def corner_origin(
    image: np.ndarray, corner: str, w: int, h: int, margin: int
) -> tuple[int, int]:
    """Top-left pixel of a w×h panel placed in `corner` ("top-left", "bottom-right", "top-centre"…).
    Public so callers can reserve/lay out panel space before anything is drawn."""
    height, width = image.shape[:2]
    corner = corner.replace("_", "-").lower()
    x = width - margin - w if "right" in corner else margin
    y = height - margin - h if "bottom" in corner else margin
    if "center" in corner or "centre" in corner:
        x = (width - w) // 2
    return int(x), int(y)


def legend(
    image: np.ndarray,
    entries: Sequence[tuple[str, BGR] | tuple[str, BGR, str]],
    *,
    corner: str = "bottom-left",
    margin: int = 28,
    title: str | None = None,
    size: int = 21,
    title_size: int = 18,
    swatch: int = 26,
    row_gap: int = 12,
    pad: tuple[int, int] = (18, 16),
    panel_color: BGR = PANEL,
    panel_alpha: float = 0.76,
    text_color: BGR = PANEL_TEXT,
    measure_only: bool = False,
) -> tuple[int, int, int, int]:
    """Translucent rounded panel with one swatch + label per row.

    `entries` are (label, colour) or (label, colour, kind) where kind is "box" (outlined, like a bin
    box), "fill" (solid, like a floor overlay), "line" (a stroke) or "dot" (a marker).
    `measure_only` returns the rect it *would* occupy without drawing — used to keep labels on the
    plan from ending up underneath a panel.
    """
    rows = [(entry[0], entry[1], entry[2] if len(entry) > 2 else "fill") for entry in entries]
    if not rows:
        return (0, 0, 0, 0)
    label_w = max(text_size(label, size)[0] for label, _, _ in rows)
    row_h = max(swatch, text_size("Hg", size)[1])
    content_w = swatch + 14 + label_w
    content_h = len(rows) * row_h + row_gap * (len(rows) - 1)
    title_h = (text_size(title, title_size, "semibold")[1] + 10) if title else 0
    if title:
        content_w = max(content_w, text_size(title, title_size, "semibold")[0])
    w = content_w + 2 * pad[0]
    h = content_h + title_h + 2 * pad[1]
    x, y = corner_origin(image, corner, w, h, margin)
    if measure_only:
        return (x, y, x + w, y + h)
    rounded_rect(image, (x, y, x + w, y + h), color=panel_color, alpha=panel_alpha, radius=16)

    cursor = y + pad[1]
    if title:
        draw_text(image, title.upper(), (x + pad[0], cursor), size=title_size, weight="semibold",
                  color=MUTED, halo=0)
        cursor += title_h
    for label, color, kind in rows:
        sx, sy = x + pad[0], cursor + (row_h - swatch) // 2
        if kind == "box":
            cv2.rectangle(image, (sx, sy + 3), (sx + swatch, sy + swatch - 3), shade(color, 0.45), -1, cv2.LINE_AA)
            cv2.rectangle(image, (sx, sy + 3), (sx + swatch, sy + swatch - 3), color, 3, cv2.LINE_AA)
        elif kind == "line":
            cv2.line(image, (sx, sy + swatch // 2), (sx + swatch, sy + swatch // 2), color, 7, cv2.LINE_AA)
        elif kind == "dot":
            marker_dot(image, (sx + swatch // 2, sy + swatch // 2), color=color, radius=6, ring=2)
        else:
            # hairline around the solid swatch, so a near-black tone (e.g. "not scanned") still reads
            blend_mask(image, sx, sy + 2, _rounded_mask(swatch, swatch - 4, 6), color, 1.0)
            blend_mask(image, sx, sy + 2, _rounded_mask(swatch, swatch - 4, 6, outline=1),
                       PANEL_EDGE, 0.32)
        draw_text(image, label, (sx + swatch + 14, cursor + row_h // 2), size=size, color=text_color,
                  anchor="lc", halo=0)
        cursor += row_h + row_gap
    return (x, y, x + w, y + h)


def title_block(
    image: np.ndarray,
    heading: str,
    lines: Sequence[str] = (),
    *,
    corner: str = "top-left",
    margin: int = 28,
    heading_size: int = 38,
    line_size: int = 24,
    kicker: str | None = None,
    kicker_size: int = 19,
    accent: BGR | None = NEW_BIN,
    pad: tuple[int, int] = (24, 20),
    panel_color: BGR = PANEL,
    panel_alpha: float = 0.78,
    measure_only: bool = False,
) -> tuple[int, int, int, int]:
    """Heading (bold) plus the key numbers, in a panel with a coloured accent bar on the left.

    `kicker` is the small line above the heading (e.g. the view name, "Plassering"); `lines` are the
    key numbers under it, one per row.
    """
    parts: list[tuple[str, int, str, BGR]] = []
    if kicker:
        parts.append((kicker.upper(), kicker_size, "semibold", MUTED))
    parts.append((heading, heading_size, "bold", PANEL_TEXT))
    for line in lines:
        parts.append((line, line_size, "regular", PANEL_TEXT))

    gaps = [0] + [10 if index == 1 and kicker else 8 for index in range(1, len(parts))]
    sizes = [text_size(text, size, weight) for text, size, weight, _ in parts]
    content_w = max(w for w, _ in sizes)
    content_h = sum(h for _, h in sizes) + sum(gaps)
    bar = 6 if accent is not None else 0
    w = content_w + 2 * pad[0] + bar
    h = content_h + 2 * pad[1]
    x, y = corner_origin(image, corner, w, h, margin)
    if measure_only:
        return (x, y, x + w, y + h)
    rounded_rect(image, (x, y, x + w, y + h), color=panel_color, alpha=panel_alpha, radius=18)
    if accent is not None:
        blend_mask(image, x + 10, y + 12, _rounded_mask(bar, h - 24, 3), accent, 1.0)

    cursor = y + pad[1]
    for (text, size, weight, color), (_, line_h), gap in zip(parts, sizes, gaps):
        cursor += gap
        draw_text(image, text, (x + pad[0] + bar, cursor), size=size, weight=weight, color=color, halo=0)
        cursor += line_h
    return (x, y, x + w, y + h)


def header_band(
    image: np.ndarray,
    heading: str,
    lines: Sequence[str] = (),
    *,
    kicker: str | None = None,
    note: str | None = None,
    accent: BGR = NEW_BIN,
    arrow: bool = False,
    ui: float = 1.0,
    background: BGR = BACKDROP,
) -> np.ndarray:
    """Grow the canvas upwards and set a title band there: heading (the address), the key numbers
    under it, an optional small `note` (scan id) on the right and an optional orientation arrow.

    Returns a NEW, taller image. Because the band is added rather than drawn on top, nothing in the
    drawing can ever be covered by the title — which is why every sheet ends with this call. The
    note falls back to its own row when the right-hand side is not genuinely free.
    """
    rows: list[tuple[str, int, str, BGR]] = []
    if kicker:
        rows.append((kicker.upper(), int(18 * ui), "semibold", MUTED))
    rows.append((str(heading), int(37 * ui), "bold", PANEL_TEXT))
    rows.extend((line, int(23 * ui), "regular", MUTED) for line in lines)

    gap = int(8 * ui)
    pad_x, pad_y = int(30 * ui), int(22 * ui)
    arrow_w = arrow_h = 0
    if arrow:
        probe = north_arrow(image, corner="top-right", margin=0, label=None,
                            caption=LABELS["topdown"], length=int(42 * ui), measure_only=True)
        arrow_w, arrow_h = probe[2] - probe[0], probe[3] - probe[1]

    bar = max(4, int(5 * ui))
    text_x = pad_x + bar + int(16 * ui)
    note_size = int(20 * ui)
    inline_note = None
    if note:
        widest = max(text_size(text, size, weight)[0] for text, size, weight, _ in rows)
        note_w = text_size(note, note_size)[0]
        if note_w <= image.shape[1] - text_x - widest - pad_x - arrow_w - int(56 * ui):
            inline_note = note
        else:
            rows.append((note, note_size, "regular", shade(MUTED, 0.8)))

    sizes = [text_size(text, size, weight) for text, size, weight, _ in rows]
    band = sum(h for _, h in sizes) + gap * (len(rows) - 1) + 2 * pad_y
    arrow_margin = 0
    if arrow:
        band = max(band, arrow_h + 2 * int(12 * ui))
        arrow_margin = (band - arrow_h) // 2

    sheet = add_margins(image, top=band, color=background)
    sheet[band - 1:band, :] = shade(PANEL_EDGE, 0.24)
    blend_mask(sheet, pad_x, pad_y, np.full((band - 2 * pad_y, bar), 255, np.uint8), accent, 1.0)

    cursor = pad_y
    for (text, size, weight, color), (_, height) in zip(rows, sizes):
        draw_text(sheet, text, (text_x, cursor), size=size, weight=weight, color=color, halo=0)
        cursor += height + gap
    if arrow:
        # gravity-aligned frames are never georeferenced, so the arrow makes no "N" claim
        north_arrow(sheet, corner="top-right", margin=arrow_margin, label=None,
                    caption=LABELS["topdown"], length=int(42 * ui), panel_alpha=0.0)
    if inline_note:
        right = sheet.shape[1] - (arrow_margin + arrow_w + int(18 * ui) if arrow else pad_x)
        draw_text(sheet, inline_note, (right, band // 2), size=note_size,
                  color=shade(MUTED, 0.8), anchor="rc", halo=0)
    return sheet


_NICE_STEPS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def scale_bar(
    image: np.ndarray,
    px_per_m: float,
    *,
    corner: str = "bottom-right",
    margin: int = 28,
    target_frac: float = 0.16,
    color: BGR = PANEL_TEXT,
    panel_color: BGR = PANEL,
    panel_alpha: float = 0.76,
    size: int = 20,
    measure_only: bool = False,
) -> tuple[int, int, int, int]:
    """Map-style segmented scale bar sized to the image's metres-per-pixel, labelled e.g. "2 m"."""
    if px_per_m <= 0:
        return (0, 0, 0, 0)
    target_px = image.shape[1] * target_frac
    metres = _NICE_STEPS[0]
    for step in _NICE_STEPS:
        if step * px_per_m <= target_px:
            metres = step
    bar_px = int(round(metres * px_per_m))
    label = f"{num(metres, 0 if metres >= 1 else 1)} m"
    bar_h = 12
    label_w, label_h = text_size(label, size, "semibold")
    pad = (16, 12)
    w = max(bar_px, label_w) + 2 * pad[0]
    h = bar_h + 8 + label_h + 2 * pad[1]
    x, y = corner_origin(image, corner, w, h, margin)
    if measure_only:
        return (x, y, x + w, y + h)
    rounded_rect(image, (x, y, x + w, y + h), color=panel_color, alpha=panel_alpha, radius=14)

    bx, by = x + (w - bar_px) // 2, y + pad[1]
    segments = 4 if bar_px >= 120 else 2
    seg = bar_px / segments
    for index in range(segments):
        x0 = int(round(bx + index * seg))
        x1 = int(round(bx + (index + 1) * seg))
        fill = color if index % 2 == 0 else hex_to_bgr("#4a5158")  # alternating segments, map-style
        cv2.rectangle(image, (x0, by), (x1, by + bar_h), fill, -1, cv2.LINE_AA)
    cv2.rectangle(image, (bx, by), (bx + bar_px, by + bar_h), color, 2, cv2.LINE_AA)
    draw_text(image, label, (x + w // 2, by + bar_h + 8), size=size, weight="semibold", color=color,
              anchor="ct", halo=0)
    return (x, y, x + w, y + h)


def north_arrow(
    image: np.ndarray,
    *,
    corner: str = "top-right",
    margin: int = 28,
    label: str | None = "N",
    caption: str | None = None,
    color: BGR = PANEL_TEXT,
    panel_color: BGR = PANEL,
    panel_alpha: float = 0.76,
    length: int = 54,
    measure_only: bool = False,
) -> tuple[int, int, int, int]:
    """Orientation arrow pointing to the top of the image. `label` is the compass letter (pass None
    when the frame is not georeferenced); `caption` adds a small note such as "sett ovenfra"."""
    label_h = text_size(label, 20, "bold")[1] if label else 0
    caption_w, caption_h = text_size(caption, 17)[0:2] if caption else (0, 0)
    pad = (16, 14)
    content_w = max(34, caption_w, text_size(label or "", 20, "bold")[0])
    content_h = label_h + (6 if label else 0) + length + (6 + caption_h if caption else 0)
    w, h = content_w + 2 * pad[0], content_h + 2 * pad[1]
    x, y = corner_origin(image, corner, w, h, margin)
    if measure_only:
        return (x, y, x + w, y + h)
    rounded_rect(image, (x, y, x + w, y + h), color=panel_color, alpha=panel_alpha, radius=14)

    cx = x + w // 2
    cursor = y + pad[1]
    if label:
        draw_text(image, label, (cx, cursor), size=20, weight="bold", color=color, anchor="ct", halo=0)
        cursor += label_h + 6
    tip, tail = (cx, cursor), (cx, cursor + length)
    cv2.line(image, tail, tip, shade(color, 0.35), 7, cv2.LINE_AA)
    cv2.line(image, tail, tip, color, 3, cv2.LINE_AA)
    head = np.array([[cx, cursor - 2], [cx - 9, cursor + 18], [cx + 9, cursor + 18]], np.int32)
    cv2.fillPoly(image, [head], color, cv2.LINE_AA)
    cursor += length
    if caption:
        draw_text(image, caption, (cx, cursor + 6), size=17, color=MUTED, anchor="ct", halo=0)
    return (x, y, x + w, y + h)


def dimension_line(
    image: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    label: str,
    *,
    color: BGR = DIMENSION,
    thickness: int = 3,
    tick: int = 12,
    size: int = 24,
    label_offset: int = 0,
) -> None:
    """Arrowed measurement line with the value in a pill at its midpoint — the thing that makes a
    raster read as a plan. `p0`/`p1` are pixel coordinates; ticks are drawn perpendicular to it."""
    start = np.array(p0, float)
    end = np.array(p1, float)
    direction = end - start
    span = float(np.hypot(*direction))
    if span < 1:
        return
    unit = direction / span
    normal = np.array([-unit[1], unit[0]])
    dark = shade(color, 0.35)
    for width, tone in ((thickness + 4, dark), (thickness, color)):
        cv2.line(image, tuple(start.astype(int)), tuple(end.astype(int)), tone, width, cv2.LINE_AA)
        for point in (start, end):
            a = (point + normal * tick).astype(int)
            b = (point - normal * tick).astype(int)
            cv2.line(image, tuple(a), tuple(b), tone, width, cv2.LINE_AA)
    head = max(8.0, min(18.0, span * 0.08))
    for point, sign in ((start, 1.0), (end, -1.0)):
        tipp = point
        base = point + unit * head * sign
        wing = normal * head * 0.42
        poly = np.array([tipp, base + wing, base - wing], np.int32)
        cv2.fillPoly(image, [poly], color, cv2.LINE_AA)
    mid = (start + end) / 2 + normal * label_offset
    # Keep the value pill fully on the canvas. The caller aims it at the room interior, but on a
    # rotated plan whose outline reaches the edge of the raster the pill still lands half outside and
    # the number is cut off. Clamping the centre is the last line of defence and is a no-op when the
    # pill already fits.
    pad_x, pad_y = 12, 6
    text_w, text_h = text_size(label, size, "semibold")
    half_w, half_h = text_w / 2 + pad_x + 2, text_h / 2 + pad_y + 2
    height, width = image.shape[:2]
    mid[0] = float(np.clip(mid[0], half_w, max(half_w, width - half_w)))
    mid[1] = float(np.clip(mid[1], half_h, max(half_h, height - half_h)))
    draw_text(image, label, tuple(mid.astype(int)), size=size, weight="semibold", color=color,
              anchor="cc", halo=0, box=True, box_color=PANEL, box_alpha=0.8, box_pad=(pad_x, pad_y),
              box_radius=10, box_edge=color, box_edge_alpha=0.5)


# ---------------------------------------------------------------- canvas helpers

def add_margins(
    image: np.ndarray,
    *,
    top: int = 0,
    right: int = 0,
    bottom: int = 0,
    left: int = 0,
    color: BGR = PANEL,
) -> np.ndarray:
    """Grow the canvas (new pixels filled with `color`) to make room for headings/captions."""
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=list(color))


def vignette(image: np.ndarray, strength: float = 0.35, spread: float = 0.55) -> None:
    """Darken the frame edges slightly so panels in the corners keep their contrast (in place)."""
    height, width = image.shape[:2]
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radial = np.sqrt(xs * xs + ys * ys) / np.sqrt(2.0)
    weight = np.clip((radial - spread) / (1.0 - spread), 0.0, 1.0) * float(strength)
    image[:] = np.clip(image.astype(np.float32) * (1.0 - weight[..., None]), 0, 255).astype(np.uint8)


def fit_width(image: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale (never upscale) so wide side-by-side sheets stay a sane file size."""
    if max_width <= 0 or image.shape[1] <= max_width:
        return image
    scale = max_width / image.shape[1]
    return cv2.resize(image, (max_width, max(int(round(image.shape[0] * scale)), 1)), interpolation=cv2.INTER_AREA)
