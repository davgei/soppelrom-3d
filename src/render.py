"""Headless orthographic previews of a point cloud — the images that get shown to colleagues.

Geometry is never computed here: every function only rasterises what backbone/freespace/placement
already decided. All typography, colours and plan furniture (legend, scale bar, dimension lines)
come from src.style so the previews, the CAD floor plan and the 3D snapshots look like one product.

Each preview is laid out as a small sheet: a header band with the address and the key numbers, the
plan itself on a dark mat, and translucent corner panels (legend, scale bar, orientation) placed so
they avoid whatever was drawn on the plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation

from . import style


def _fill_speckle(image: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Fill the 1-pixel gaps between rasterized surface points so a densely-sampled mesh reads as a
    solid surface instead of speckle. Morphological close fills small holes but keeps the outline."""
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))


def _speckle_kernel(count: int, height: int, width: int, factor: float = 4.0) -> int:
    """Kernel for _fill_speckle, from the mean spacing between projected points. A mesh sampled with
    a fixed budget is dense in a small room and sparse in a big yard, so a fixed kernel either leaves
    black speckle or smears detail; deriving it from the actual density keeps both readable."""
    if count <= 0:
        return 3
    spacing = float(np.sqrt(height * width / count))
    return int(np.clip(int(round(spacing * factor)) | 1, 3, 13))


def _rasterize(
    points: np.ndarray,
    colors: np.ndarray,
    axis_u: int,
    axis_v: int,
    axis_depth: int,
    px_per_m: int,
    flip_v: bool,
) -> np.ndarray:
    u = points[:, axis_u]
    v = points[:, axis_v]
    depth = points[:, axis_depth]

    width = max(int((u.max() - u.min()) * px_per_m) + 1, 1)
    height = max(int((v.max() - v.min()) * px_per_m) + 1, 1)
    px = np.clip(((u - u.min()) * px_per_m).astype(int), 0, width - 1)
    py = np.clip(((v - v.min()) * px_per_m).astype(int), 0, height - 1)

    order = np.argsort(depth)  # draw far points first, nearer points overwrite
    image = np.zeros((height, width, 3), np.uint8)
    image[py[order], px[order]] = (colors[order] * 255).astype(np.uint8)
    if flip_v:
        image = image[::-1]
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def ortho_previews(pcd: o3d.geometry.PointCloud, out_dir: str | Path, px_per_m: int = 100) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # ARKit world has Y up: top-down looks along Y (u=X, v=Z); front looks along Z (u=X, v=Y).
    cv2.imwrite(str(out / "topdown_xz.png"), _rasterize(points, colors, 0, 2, 1, px_per_m, flip_v=True))
    cv2.imwrite(str(out / "front_xy.png"), _rasterize(points, colors, 0, 1, 2, px_per_m, flip_v=True))


# --------------------------------------------------------------------------- shared framing

BACKDROP = style.BACKDROP  # the mat the plan is mounted on
NO_DATA = style.NO_DATA    # never scanned / outside the room


@dataclass
class _Frame:
    """A finished top-down raster plus the world -> pixel mapping for drawing on top of it.

    Everything (previews and the before/after sheet) goes through this, so panels rendered from the
    same PlacementResult are guaranteed to share extent, scale and framing.
    """
    image: np.ndarray
    scale: int                  # pixels per grid cell
    cell: float
    origin: np.ndarray          # (X, Z) world coordinate of grid cell (0, 0)
    pad: int                    # border of mat added around the plan
    px_per_m: float             # effective, after integer rounding of `scale`
    marks: list = field(default_factory=list)  # pixel points already drawn on the plan
    ui_scale: float | None = None              # overrides `ui` (before/after pre-compensates it)

    def to_px(self, x: float, z: float) -> tuple[int, int]:
        # +Z points up in the image, so the row axis is mirrored relative to the grid.
        plan_h = self.image.shape[0] - 2 * self.pad
        return (
            int(round(self.pad + (x - self.origin[0]) / self.cell * self.scale)),
            int(round(self.pad + plan_h - 1 - (z - self.origin[1]) / self.cell * self.scale)),
        )

    def poly(self, rect) -> np.ndarray:
        """cv2.minAreaRect-style rect in world X/Z -> pixel polygon."""
        return np.array([self.to_px(x, z) for x, z in cv2.boxPoints(rect)], np.int32)

    @property
    def ui(self) -> float:
        """UI scale factor: keeps panels/labels proportionate on both tiny rooms and big yards."""
        if self.ui_scale is not None:
            return self.ui_scale
        return float(np.clip(min(self.image.shape[:2]) / 950.0, 0.62, 1.45))


def _scene_base(
    aligned_pcd: o3d.geometry.PointCloud,
    shape: tuple[int, int],
    cell: float,
    origin: np.ndarray,
    px_per_m: float,
) -> tuple[np.ndarray, int]:
    """Rasterise the real coloured scene over the analysis grid's extent, at full pixel resolution
    (not one pixel per 5 cm cell) so the floor texture stays sharp. Returns the un-flipped image
    and the integer pixels-per-cell, which is what lets grid masks upscale onto it exactly."""
    rows, cols = shape
    scale = max(int(round(px_per_m * cell)), 1)
    height, width = rows * scale, cols * scale
    points = np.asarray(aligned_pcd.points)
    colors = np.asarray(aligned_pcd.colors)

    px = np.floor((points[:, 0] - origin[0]) / cell * scale).astype(int)
    py = np.floor((points[:, 2] - origin[1]) / cell * scale).astype(int)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py = px[inside], py[inside]
    p, c = points[inside], colors[inside]

    order = np.argsort(p[:, 1])  # draw low first, higher points overwrite (top-down)
    base = np.zeros((height, width, 3), np.uint8)
    base[py[order], px[order]] = (c[order][:, ::-1] * 255).astype(np.uint8)  # RGB -> BGR
    # a sampled mesh leaves pinholes between the projected points; close them so it reads as surface
    base = _fill_speckle(base, _speckle_kernel(len(px), height, width))
    base[np.all(base == 0, axis=2)] = NO_DATA
    return base, scale


def _overlay_grid(
    base: np.ndarray,
    scale: int,
    layers: Sequence[tuple[np.ndarray, style.BGR, float]],
) -> None:
    """Blend boolean grid masks (rows = Z, cols = X) onto the pixel-resolution base, in place."""
    height, width = base.shape[:2]
    for mask, color, alpha in layers:
        if mask is None or not np.any(mask):
            continue
        big = np.repeat(np.repeat(mask.astype(np.uint8) * 255, scale, axis=0), scale, axis=1)
        style.blend_mask(base, 0, 0, big[:height, :width], color, alpha)


def _finish(base: np.ndarray, scale: int, cell: float, origin: np.ndarray, pad: int) -> _Frame:
    """Flip to +Z-up, mount the plan on the backdrop mat and wrap it in a drawable _Frame."""
    image = np.ascontiguousarray(base[::-1])
    if pad:
        image = style.add_margins(image, top=pad, right=pad, bottom=pad, left=pad, color=BACKDROP)
        cv2.rectangle(image, (pad - 1, pad - 1), (image.shape[1] - pad, image.shape[0] - pad),
                      style.shade(style.PANEL_EDGE, 0.22), 1, cv2.LINE_AA)
    return _Frame(image, scale, cell, np.asarray(origin, float), pad, scale / cell)


def _pad_for(shape_px: tuple[int, int]) -> int:
    """A mat wide enough that the plan reads as a mounted drawing rather than a screenshot."""
    return int(np.clip(min(shape_px) * 0.035, 22, 64))


def _plural(count: int, singular: str, plural: str) -> str:
    """Norwegian counting, incl. the zero case: "ingen kasser" reads better than "0 kasser"."""
    if count == 0:
        return f"ingen {plural}"
    return f"{count} {singular if count == 1 else plural}"


def _sentence(text: str) -> str:
    """Capitalise the first letter only — the rest of the phrase is already correctly cased."""
    return text[:1].upper() + text[1:] if text else text


def _new_bins_phrase(count: int) -> str:
    if count == 0:
        return "ingen plass til nye kasser"
    return f"plass til {_plural(count, 'ny kasse', 'nye kasser')}"


# --------------------------------------------------------------------------- sheet layout

def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _inside(rect: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


@dataclass
class _Layout:
    """Where the corner panels go. Measured BEFORE anything is drawn on the plan, so both the panels
    themselves and the labels on the plan can dodge each other."""
    legend_corner: str
    scale_corner: str
    reserved: list[tuple[int, int, int, int]]


def _layout(frame: _Frame, legend_entries: Sequence[tuple], *, corners: Sequence[str] = ()) -> _Layout:
    """Put the scale bar, then the legend, in whichever corner covers the fewest drawn marks."""
    ui = frame.ui
    margin = int(22 * ui)

    def rect_for(corner: str, size: tuple[int, int]) -> tuple[int, int, int, int]:
        x, y = style.corner_origin(frame.image, corner, size[0], size[1], margin)
        return (x, y, x + size[0], y + size[1])

    def covered(rect: tuple[int, int, int, int]) -> int:
        return sum(1 for point in frame.marks if _inside(rect, point))

    probe = style.scale_bar(frame.image, frame.px_per_m, corner="bottom-right", margin=margin,
                            size=int(20 * ui), measure_only=True)
    scale_size = (probe[2] - probe[0], probe[3] - probe[1])
    scale_corner = min(("bottom-right", "bottom-left", "top-right"),
                       key=lambda corner: covered(rect_for(corner, scale_size)))
    scale_rect = rect_for(scale_corner, scale_size)

    probe = _legend(frame, legend_entries, corner="bottom-left", measure_only=True)
    legend_size = (probe[2] - probe[0], probe[3] - probe[1])
    options = list(corners) or ["bottom-left", "top-left", "bottom-right", "top-right"]
    legend_corner = min(options, key=lambda corner: covered(rect_for(corner, legend_size))
                        + 1000 * _overlaps(rect_for(corner, legend_size), scale_rect))
    return _Layout(legend_corner, scale_corner,
                   [scale_rect, rect_for(legend_corner, legend_size)])


def _legend(frame: _Frame, entries, *, corner: str = "bottom-left", measure_only: bool = False):
    ui = frame.ui
    return style.legend(
        frame.image, entries, corner=corner, margin=int(22 * ui), title="Tegnforklaring",
        size=int(21 * ui), title_size=int(17 * ui), swatch=int(26 * ui), row_gap=int(11 * ui),
        pad=(int(17 * ui), int(15 * ui)), measure_only=measure_only,
    )


def _draw_furniture(frame: _Frame, entries, layout: _Layout, *, scale_bar: bool = True) -> None:
    ui = frame.ui
    if scale_bar:
        style.scale_bar(frame.image, frame.px_per_m, corner=layout.scale_corner,
                        margin=int(22 * ui), size=int(20 * ui))
    if entries:
        _legend(frame, entries, corner=layout.legend_corner)


# --------------------------------------------------------------------------- room + measurements

def annotated_topdown(
    aligned_pcd: o3d.geometry.PointCloud,
    footprint,
    out_path: str | Path,
    px_per_m: int = 100,
    *,
    title: str | None = None,
    lines: Sequence[str] | None = None,
    note: str | None = None,
) -> None:
    """Top-down view of the gravity-aligned cloud with the room footprint drawn as a measured plan:
    outlined rectangle, arrowed dimension lines on two adjacent walls, scale bar."""
    points = np.asarray(aligned_pcd.points)
    colors = np.asarray(aligned_pcd.colors)
    u, v, depth = points[:, 0], points[:, 2], points[:, 1]

    # Size the canvas to the cloud AND the footprint rectangle. The outline is a minAreaRect of the
    # floor, so on a rotated or ragged scan a corner can sit OUTSIDE the cloud's own extent — and then
    # the outline, its dimension line and the value pill were drawn off-canvas and cut off. The extra
    # 0.45 m leaves room for the arrowed line and the pill beside it.
    corners = cv2.boxPoints(footprint.rect)
    margin_m = 0.45
    u_min = min(float(u.min()), float(corners[:, 0].min()) - margin_m)
    v_min = min(float(v.min()), float(corners[:, 1].min()) - margin_m)
    u_max = max(float(u.max()), float(corners[:, 0].max()) + margin_m)
    v_max = max(float(v.max()), float(corners[:, 1].max()) + margin_m)
    width = max(int((u_max - u_min) * px_per_m) + 1, 1)
    height = max(int((v_max - v_min) * px_per_m) + 1, 1)
    px = np.clip(((u - u_min) * px_per_m).astype(int), 0, width - 1)
    py = np.clip(((v - v_min) * px_per_m).astype(int), 0, height - 1)

    order = np.argsort(depth)
    base = np.zeros((height, width, 3), np.uint8)
    base[py[order], px[order]] = (colors[order][:, ::-1] * 255).astype(np.uint8)  # RGB -> BGR
    base = _fill_speckle(base, _speckle_kernel(len(px), height, width))
    base[np.all(base == 0, axis=2)] = NO_DATA

    # one "cell" = one metre here, so the same _Frame mapping serves the metre-based views too
    frame = _finish(base, int(px_per_m), 1.0, np.array([u_min, v_min]), _pad_for((height, width)))
    polygon = frame.poly(footprint.rect)
    frame.marks = [tuple(point) for point in polygon]

    entries = [(style.LABELS["room"], style.ROOM_OUTLINE, "line"), ("Måltall", style.DIMENSION, "line")]
    layout = _layout(frame, entries)

    style.soft_shadow(frame.image, [polygon], offset=(0, 4), blur=21, strength=0.45)
    style.draw_polygon(frame.image, polygon, color=style.ROOM_OUTLINE,
                       thickness=max(3, int(3 * frame.ui)), edge_alpha=0.5, edge_grow=5)
    # dimension lines on two adjacent walls, pulled inside the room so they never leave the canvas
    centre = polygon.mean(axis=0)
    label_size = int(23 * frame.ui)
    for a, b in ((0, 1), (1, 2)):
        p0, p1 = polygon[a].astype(float), polygon[b].astype(float)
        inward = centre - (p0 + p1) / 2
        inward /= float(np.hypot(*inward)) or 1.0
        metres = float(np.hypot(*(p1 - p0))) / frame.px_per_m
        label = style.fmt_m(metres)
        # the value sits beside the line, on the room side, so the pill never straddles a wall
        span = p1 - p0
        normal = np.array([-span[1], span[0]]) / (float(np.hypot(*span)) or 1.0)
        toward = 1.0 if float(np.dot(normal, inward)) > 0 else -1.0
        text_w, text_h = style.text_size(label, label_size, "semibold")
        reach = (abs(normal[0]) * (text_w + 24) + abs(normal[1]) * (text_h + 12)) / 2 + 8
        style.dimension_line(
            frame.image, tuple((p0 + inward * 26 * frame.ui).astype(int)),
            tuple((p1 + inward * 26 * frame.ui).astype(int)), label, size=label_size,
            thickness=max(2, int(3 * frame.ui)), tick=int(11 * frame.ui),
            label_offset=toward * reach,
        )
    _draw_furniture(frame, entries, layout)

    detail = list(lines) if lines is not None else [
        f"Yttermål: {style.num(footprint.length_m, 2)} × {style.num(footprint.width_m, 2)} m"
        f" · bruttoareal {style.fmt_m2(footprint.area_m2)}",
    ]
    heading = title or f"{style.num(footprint.length_m, 2)} × {style.num(footprint.width_m, 2)} m"
    sheet = style.header_band(frame.image, heading, detail, kicker="Rom + mål",
                              accent=style.DIMENSION, note=note, arrow=True, ui=frame.ui)
    cv2.imwrite(str(Path(out_path)), sheet)


# --------------------------------------------------------------------------- free floor

_FREESPACE_LEGEND = [
    (style.LABELS["free"], style.FREE_FLOOR, "fill"),
    (style.LABELS["occupied"], style.OCCUPIED_FLOOR, "fill"),
    (style.LABELS["unknown"], NO_DATA, "fill"),
]


def _freespace_lines(result) -> list[str]:
    observed = getattr(result, "observed_floor_area_m2", 0.0) or 0.0
    share = result.free_area_m2 / observed * 100.0 if observed else 0.0
    return [
        f"{style.LABELS['free']}: {style.fmt_m2(result.free_area_m2)} av "
        f"{style.fmt_m2(observed)} skannet gulv ({style.num(share, 0)} %)",
        f"Opptatt av kasser og annet: {style.fmt_m2(result.occupied_on_floor_m2)}",
    ]


def freespace_topdown(result, out_path: str | Path, px_per_m: int = 100, *,
                      title: str | None = None, note: str | None = None) -> None:
    """Top-down free-space map: green = free floor, red = occupied, dark = never scanned."""
    rows, cols = result.free.shape
    scale = max(int(round(px_per_m * result.cell)), 1)
    base = np.zeros((rows, cols, 3), np.uint8)
    base[:] = NO_DATA
    base[result.floor_observed] = style.UNKNOWN_FLOOR
    base[result.occupied] = style.OCCUPIED_FLOOR
    base[result.free] = style.FREE_FLOOR
    base = cv2.resize(base, (cols * scale, rows * scale), interpolation=cv2.INTER_NEAREST)

    frame = _finish(base, scale, result.cell, result.origin, _pad_for(base.shape[:2]))
    layout = _layout(frame, _FREESPACE_LEGEND)
    _draw_furniture(frame, _FREESPACE_LEGEND, layout)
    sheet = style.header_band(frame.image, title or style.fmt_m2(result.free_area_m2),
                              _freespace_lines(result), kicker=style.LABELS["free"],
                              accent=style.FREE_FLOOR, note=note, arrow=True, ui=frame.ui)
    cv2.imwrite(str(Path(out_path)), sheet)


def freespace_over_scene(
    aligned_pcd: o3d.geometry.PointCloud,
    result,
    out_path: str | Path,
    px_per_m: int = 100,
    alpha: float = 0.45,
    *,
    title: str | None = None,
    note: str | None = None,
) -> None:
    """Top-down of the REAL colored scene with translucent green (free) / red (occupied) on top,
    so the computed area can be checked against the actual floor texture."""
    base, scale = _scene_base(aligned_pcd, result.free.shape, result.cell, result.origin, px_per_m)
    _overlay_grid(base, scale, [
        (result.free, style.FREE_FLOOR, alpha),
        (result.occupied, style.OCCUPIED_FLOOR, alpha),
    ])
    frame = _finish(base, scale, result.cell, result.origin, _pad_for(base.shape[:2]))
    entries = _FREESPACE_LEGEND[:2]
    layout = _layout(frame, entries)
    _draw_furniture(frame, entries, layout)
    sheet = style.header_band(frame.image, title or style.fmt_m2(result.free_area_m2),
                              _freespace_lines(result), kicker=style.LABELS["free"],
                              accent=style.FREE_FLOOR, note=note, arrow=True, ui=frame.ui)
    cv2.imwrite(str(Path(out_path)), sheet)


# --------------------------------------------------------------------------- placements

PLACEMENT_LEGEND = [
    (style.LABELS["new"], style.NEW_BIN, "box"),
    (style.LABELS["existing"], style.EXISTING_BIN, "box"),
    (style.LABELS["path"], style.PATH, "line"),
    (style.LABELS["entrance"], style.ENTRANCE, "dot"),
]


def _overlay_push_path(base: np.ndarray, scale: int, result) -> None:
    """blue = the push-path: a near-straight corridor from the entrance to (and around) every
    existing bin, wide enough for the biggest bin. It is kept clear — no new bin sits on it — so the
    bins already in the room can always be wheeled out."""
    reachable = getattr(result, "reachable", None)
    route = getattr(result, "route", None)
    layers = []
    if reachable is not None:
        layers.append((reachable, style.PATH_SOFT, 0.42))
    if route is not None:
        layers.append((binary_dilation(route, iterations=1), style.PATH, 0.8))
    _overlay_grid(base, scale, layers)


def _existing_polys(frame: _Frame, result) -> list[np.ndarray]:
    return [frame.poly(((bx, bz), (bl, bw), byaw)) for bx, bz, bl, bw, byaw in result.existing_bins]


def _draw_boxes(frame: _Frame, polygons: Sequence[np.ndarray], color: style.BGR,
                edge: style.BGR) -> None:
    if not polygons:
        return
    style.soft_shadow(frame.image, polygons, offset=(0, 4), blur=15, strength=0.5)
    for polygon in polygons:
        style.draw_polygon(frame.image, polygon, color=color, thickness=max(2, int(3 * frame.ui)),
                           fill_alpha=0.22, edge_color=edge, edge_alpha=0.55)


def _draw_candidates(frame: _Frame, result, polygons: Sequence[np.ndarray], *,
                     numbered: bool = True) -> None:
    _draw_boxes(frame, polygons, style.NEW_BIN, style.NEW_BIN_EDGE)
    if not numbered:
        return
    for index, cand in enumerate(result.candidates, start=1):
        style.badge(frame.image, frame.to_px(*cand.center_xz), str(index),
                    color=style.NEW_BIN, size=int(21 * frame.ui))


def _draw_entrances(frame: _Frame, result, *, avoid: Sequence[tuple[int, int, int, int]] = ()) -> None:
    """Magenta marker per entrance, with the label on whichever side is free of panels/edges."""
    ui = frame.ui
    label = style.LABELS["entrance"]
    text_w, text_h = style.text_size(label, int(20 * ui), "semibold")
    box_w, box_h = text_w + int(22 * ui), text_h + int(12 * ui)
    offset = int(20 * ui)
    for entrance in result.entrances:
        ex, ey = frame.to_px(*entrance)
        style.marker_dot(frame.image, (ex, ey), color=style.ENTRANCE,
                         radius=int(9 * ui), ring=max(2, int(3 * ui)))
        top, bottom = ey - box_h // 2, ey + box_h // 2
        options = [
            ("lc", (ex + offset, top, ex + offset + box_w, bottom)),
            ("rc", (ex - offset - box_w, top, ex - offset, bottom)),
        ]
        for anchor, rect in options:
            if rect[0] < frame.pad or rect[2] > frame.image.shape[1] - frame.pad:
                continue
            if any(_overlaps(rect, other) for other in avoid):
                continue
            style.draw_text(
                frame.image, label, (ex + offset if anchor == "lc" else ex - offset, ey),
                size=int(20 * ui), weight="semibold", color=style.PANEL_TEXT, anchor=anchor,
                halo=0, box=True, box_color=style.ENTRANCE, box_alpha=0.9,
                box_pad=(int(11 * ui), int(6 * ui)), box_radius=int(9 * ui), box_edge=None,
            )
            break


def _placement_lines(result, extra: Sequence[str] | None) -> list[str]:
    lines = [
        _sentence(f"{_plural(len(result.existing_bins), 'kasse', 'kasser')} i dag → "
                  f"{_new_bins_phrase(len(result.candidates))}"),
    ]
    if not result.entrances:
        lines.append("Ingen inngang funnet — rommet leses som lukket")
    lines.extend(extra or [])
    return lines


def placements_over_scene(
    aligned_pcd: o3d.geometry.PointCloud,
    result,
    out_path: str | Path,
    px_per_m: int = 100,
    *,
    title: str | None = None,
    lines: Sequence[str] | None = None,
    note: str | None = None,
) -> None:
    """Real scene top-down with the push-path (blue), the entrance (magenta), the existing bins
    (red) and GREEN numbered boxes where a new bin fits (open ground all around, off the path)."""
    base, scale = _scene_base(aligned_pcd, result.clearance.shape, result.cell, result.origin, px_per_m)
    _overlay_push_path(base, scale, result)
    frame = _finish(base, scale, result.cell, result.origin, _pad_for(base.shape[:2]))

    existing = _existing_polys(frame, result)
    proposed = [frame.poly(cand.rect) for cand in result.candidates]
    frame.marks = [tuple(point) for polygon in existing + proposed for point in polygon]
    frame.marks += [frame.to_px(*entrance) for entrance in result.entrances]
    layout = _layout(frame, PLACEMENT_LEGEND)

    _draw_boxes(frame, existing, style.EXISTING_BIN, style.EXISTING_BIN_EDGE)
    _draw_candidates(frame, result, proposed)
    _draw_entrances(frame, result, avoid=layout.reserved)
    _draw_furniture(frame, PLACEMENT_LEGEND, layout)

    heading = title or _sentence(_new_bins_phrase(len(result.candidates)))
    sheet = style.header_band(frame.image, heading, _placement_lines(result, lines),
                              kicker=style.LABELS["proposal"], accent=style.NEW_BIN, note=note,
                              arrow=True, ui=frame.ui)
    cv2.imwrite(str(Path(out_path)), sheet)


# --------------------------------------------------------------------------- before / after

def before_after(
    aligned_pcd: o3d.geometry.PointCloud,
    result,
    out_path: str | Path,
    px_per_m: int = 100,
    title: str | None = None,
    *,
    note: str | None = None,
    max_width: int = 3200,
) -> np.ndarray:
    """Side-by-side sheet: "I dag" (only the bins already in the room) next to "Forslag" (the same
    room, same extent and scale, plus the numbered new bins and the push-path).

    Both panels are rasterised from ONE base image and share the _Frame mapping, so the comparison
    cannot drift — identical camera, extent and scale by construction. Returns the sheet as well as
    writing it, so callers can compose it further.
    """
    base, scale = _scene_base(aligned_pcd, result.clearance.shape, result.cell, result.origin, px_per_m)
    pad = _pad_for(base.shape[:2])

    left = _finish(base.copy(), scale, result.cell, result.origin, pad)
    _overlay_push_path(base, scale, result)  # only the proposal shows the path that must stay clear
    right = _finish(base, scale, result.cell, result.origin, pad)

    # the finished sheet is ~2 panels wide and gets downscaled to max_width, so pre-compensate:
    # everything drawn 1/shrink too big lands at the intended size once the sheet is resized
    shrink = min(1.0, max_width / (2.0 * right.image.shape[1] + 12.0))
    left.ui_scale = right.ui_scale = right.ui / shrink

    existing = _existing_polys(left, result)
    proposed = [right.poly(cand.rect) for cand in result.candidates]
    marks = [tuple(point) for polygon in existing + proposed for point in polygon]
    marks += [left.to_px(*entrance) for entrance in result.entrances]
    left.marks = right.marks = marks
    # both panels keep the legend in the same corner, so the eye does not have to re-learn the sheet
    layout = _layout(right, PLACEMENT_LEGEND, corners=("bottom-left", "bottom-right", "top-left"))

    _draw_boxes(left, existing, style.EXISTING_BIN, style.EXISTING_BIN_EDGE)
    _draw_entrances(left, result, avoid=layout.reserved)
    _draw_boxes(right, existing, style.EXISTING_BIN, style.EXISTING_BIN_EDGE)
    _draw_candidates(right, result, proposed)
    _draw_entrances(right, result, avoid=layout.reserved)

    ui = right.ui
    n_new, n_old = len(result.candidates), len(result.existing_bins)
    style.scale_bar(left.image, left.px_per_m, corner=layout.scale_corner, margin=int(22 * ui),
                    size=int(20 * ui))
    _legend(right, PLACEMENT_LEGEND, corner=layout.legend_corner)

    panels = []
    for frame, heading, subtitle, accent in (
        (left, style.LABELS["today"], _sentence(_plural(n_old, "kasse i rommet", "kasser i rommet")),
         style.EXISTING_BIN),
        (right, style.LABELS["proposal"], _sentence(_new_bins_phrase(n_new)),
         style.NEW_BIN if n_new else style.MUTED),
    ):
        panels.append(style.header_band(frame.image, heading, [subtitle], accent=accent, ui=ui))

    gap = max(int(12 * ui), 8)
    divider = np.full((panels[0].shape[0], gap, 3), BACKDROP, np.uint8)
    divider[:, gap // 2 - 1:gap // 2 + 1] = style.shade(style.PANEL_EDGE, 0.3)
    sheet = np.hstack([panels[0], divider, panels[1]])

    caption = _sentence(f"{_plural(n_old, 'kasse', 'kasser')} i dag · {_new_bins_phrase(n_new)}")
    sheet = style.header_band(sheet, title or "Søppelrom", [caption], kicker="Sammenligning",
                              accent=style.NEW_BIN if n_new else style.MUTED, note=note,
                              arrow=True, ui=ui)
    sheet = style.fit_width(sheet, max_width)
    cv2.imwrite(str(Path(out_path)), sheet)
    return sheet


# --------------------------------------------------------------------------- detections

def detections_topdown(
    pcd: o3d.geometry.PointCloud,
    instances,
    out_path: str | Path,
    px_per_m: int = 100,
    *,
    title: str | None = None,
    note: str | None = None,
) -> None:
    """Top-down view with each detected bin drawn as a red footprint rectangle."""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    u, v = points[:, 0], points[:, 2]

    u_min, v_min = u.min(), v.min()
    width = max(int((u.max() - u_min) * px_per_m) + 1, 1)
    height = max(int((v.max() - v_min) * px_per_m) + 1, 1)
    px = np.clip(((u - u_min) * px_per_m).astype(int), 0, width - 1)
    py = np.clip(((v - v_min) * px_per_m).astype(int), 0, height - 1)

    order = np.argsort(points[:, 1])
    base = np.zeros((height, width, 3), np.uint8)
    base[py[order], px[order]] = (colors[order][:, ::-1] * 255).astype(np.uint8)  # RGB -> BGR
    base = _fill_speckle(base, _speckle_kernel(len(px), height, width))
    base[np.all(base == 0, axis=2)] = NO_DATA

    frame = _finish(base, int(px_per_m), 1.0, np.array([u_min, v_min]), _pad_for((height, width)))
    boxes = [frame.poly(inst.rect) for inst in instances]
    frame.marks = [tuple(point) for polygon in boxes for point in polygon]
    entries = [(style.LABELS["existing"], style.EXISTING_BIN, "box")]
    layout = _layout(frame, entries)

    _draw_boxes(frame, boxes, style.EXISTING_BIN, style.EXISTING_BIN_EDGE)
    for index, inst in enumerate(instances, start=1):
        style.badge(frame.image, frame.to_px(inst.center[0], inst.center[2]), str(index),
                    color=style.EXISTING_BIN, text_color=style.PANEL_TEXT, size=int(21 * frame.ui))
    _draw_furniture(frame, entries, layout)

    found = _sentence(_plural(len(instances), "kasse funnet", "kasser funnet"))
    sheet = style.header_band(frame.image, title or found, [] if title is None else [found],
                              kicker="Deteksjon", accent=style.EXISTING_BIN, note=note, arrow=True,
                              ui=frame.ui)
    cv2.imwrite(str(Path(out_path)), sheet)
