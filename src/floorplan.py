"""Clean 2D CAD-style floor plan (PNG + PDF) for one prepared scan.

Reuses pipeline.compute_scene and draws the room rectangle to scale with dimension lines, the
existing bins (solid, coloured by type), the proposed new bins (dashed green, numbered) and the
entrance. A printable one-pager that reads like an architect's plan instead of the technical
point-cloud raster preview.

    .venv\\Scripts\\python.exe -m src.floorplan --scan <stem>
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402

from . import pipeline, style  # noqa: E402
from .annotations import BIN_TYPES, load_annotations  # noqa: E402
from .paths import ANNOTATION_DIR, CACHE_ROOT, PREVIEW_ROOT  # noqa: E402

_BIN_STYLE: dict[str, tuple[str, str]] = {
    "2-hjuls dunk": ("#6fa8c7", "2-hjul"),
    "4-hjuls container": ("#3f6b8a", "4-hjul"),
    "molok": ("#8177c9", "molok"),
    "annet": ("#9aa0a6", "annet"),
}
_PROPOSED_FACE = "#8fdc6b"
_PROPOSED_EDGE = "#2e7d32"
_ENTRANCE = "#e0219a"
_WALL = "#2b2b2b"
_FLOOR = "#f6f5f1"

# Breathing room around the drawing, in metres. An annotated bin can stick out past the measured
# footprint, so the frame is fitted to EVERYTHING drawn instead of to the room rectangle: otherwise
# such a bin is clipped by the axes and crosses the dimension line.
DIM_OFFSET = 0.6        # dimension line, this far outside the drawn geometry
DIM_LABEL_SPACE = 0.7   # room for the measurement text below / left of that line
PLAN_PAD = 0.7          # air on the far sides; the 1 m scale bar lives in the top strip


def _room_frame(footprint) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Rigid transform mapping aligned X/Z metres into a room frame where the long wall is
    horizontal and the room's min corner sits at (0, 0). Returns (R, origin, length, width)."""
    corners = cv2.boxPoints(footprint.rect)
    edge_a = corners[1] - corners[0]
    edge_b = corners[2] - corners[1]
    long_dir = edge_a if np.linalg.norm(edge_a) >= np.linalg.norm(edge_b) else edge_b
    angle = math.atan2(float(long_dir[1]), float(long_dir[0]))
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = corners @ rotation.T
    origin = rotated.min(axis=0)
    extent = rotated.max(axis=0) - origin
    return rotation, origin, float(extent[0]), float(extent[1])


def _to_room(points_xz: np.ndarray, rotation: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return (np.asarray(points_xz, dtype=float) @ rotation.T) - origin


def _bin_corners(cx: float, cz: float, length: float, width: float, yaw_deg: float) -> np.ndarray:
    return cv2.boxPoints(((cx, cz), (length, width), yaw_deg))


def _existing_bins_typed(stem: str, rotation: np.ndarray) -> list[tuple[float, float, float, float, float, str]]:
    """Existing bins in the aligned frame WITH their annotated type (pipeline.load_existing_bins
    drops the type). Mirrors that transform so the plan matches the previews."""
    annotated = ANNOTATION_DIR / f"{stem}.json"
    proposals = CACHE_ROOT / stem / "proposals.json"
    path = annotated if annotated.exists() else (proposals if proposals.exists() else None)
    if path is None:
        return []
    _, boxes = load_annotations(path)
    result = []
    for box in boxes:
        center = rotation @ np.asarray(box.center)
        length = max(box.extent[0], box.extent[2])
        width = min(box.extent[0], box.extent[2])
        result.append((float(center[0]), float(center[2]), float(length), float(width),
                       float(box.yaw_deg), box.bin_type))
    return result


def _dimension(ax, p0: tuple[float, float], p1: tuple[float, float], text: str, vertical: bool) -> None:
    """Architect-style dimension line with double arrow, extension ticks and a centred label."""
    (x0, y0), (x1, y1) = p0, p1
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="<|-|>", color=_WALL, lw=1.1, shrinkA=0, shrinkB=0))
    if vertical:
        ax.text(x0 - 0.18, (y0 + y1) / 2, text, ha="right", va="center", rotation=90,
                fontsize=9.5, color=_WALL)
    else:
        ax.text((x0 + x1) / 2, y0 - 0.18, text, ha="center", va="top", fontsize=9.5, color=_WALL)


def _draw_entrance(ax, point: tuple[float, float], length: float, width: float) -> None:
    """Mark a doorway as a pink opening on the nearest wall plus a short arrow pointing inward."""
    x, y = point
    sides = {
        "bottom": (abs(y - 0.0), np.array([0.0, 1.0])),
        "top": (abs(y - width), np.array([0.0, -1.0])),
        "left": (abs(x - 0.0), np.array([1.0, 0.0])),
        "right": (abs(x - length), np.array([-1.0, 0.0])),
    }
    side, (_, inward) = min(sides.items(), key=lambda kv: kv[1][0])
    if side in ("bottom", "top"):
        wx = float(np.clip(x, 0.45, length - 0.45))
        wy = 0.0 if side == "bottom" else width
        ax.plot([wx - 0.45, wx + 0.45], [wy, wy], color=_ENTRANCE, lw=5, solid_capstyle="butt", zorder=5)
        base = np.array([wx, wy])
    else:
        wy = float(np.clip(y, 0.45, width - 0.45))
        wx = 0.0 if side == "left" else length
        ax.plot([wx, wx], [wy - 0.45, wy + 0.45], color=_ENTRANCE, lw=5, solid_capstyle="butt", zorder=5)
        base = np.array([wx, wy])
    tip = base + inward * 0.6
    ax.annotate("", xy=(tip[0], tip[1]), xytext=(base[0], base[1]),
                arrowprops=dict(arrowstyle="-|>", color=_ENTRANCE, lw=2.0), zorder=5)
    label = base + inward * 0.7
    ax.text(label[0], label[1], "Inngang", color=_ENTRANCE, fontsize=9, fontweight="bold",
            ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))


@dataclass
class PlanInfo:
    """What the plan shows, derivable without drawing it, so a caller can lay out a page (and
    caption the plan) before any axes exists."""
    length_m: float
    width_m: float
    xlim: tuple[float, float]                         # full drawing extent, room-frame metres
    ylim: tuple[float, float]
    existing: list = field(default_factory=list)      # (x, z, length, width, yaw, bin_type)
    present_types: list[str] = field(default_factory=list)
    n_candidates: int = 0

    @property
    def span_x(self) -> float:
        return self.xlim[1] - self.xlim[0]

    @property
    def span_y(self) -> float:
        return self.ylim[1] - self.ylim[0]

    @property
    def dim_x(self) -> float:
        """X of the vertical dimension line (and Y of the horizontal one, via dim_y)."""
        return self.xlim[0] + DIM_LABEL_SPACE

    @property
    def dim_y(self) -> float:
        return self.ylim[0] + DIM_LABEL_SPACE


def plan_info(scene) -> PlanInfo:
    """Measure the plan without drawing it: room size, drawing extent, the typed existing bins and
    which bin types will need a legend entry."""
    rotation, origin, length, width = _room_frame(scene.footprint)
    existing = _existing_bins_typed(scene.stem, scene.rotation)
    present: list[str] = []
    for *_, bin_type in existing:
        if bin_type not in present:
            present.append(bin_type)

    drawn = [np.array([[0.0, 0.0], [length, width]])]
    drawn += [_to_room(_bin_corners(bx, bz, bl, bw, byaw), rotation, origin)
              for bx, bz, bl, bw, byaw, _ in existing]
    drawn += [_to_room(cv2.boxPoints(cand.rect), rotation, origin)
              for cand in scene.result.candidates]
    points = np.vstack(drawn)
    low = points.min(axis=0) - (DIM_OFFSET + DIM_LABEL_SPACE)
    high = points.max(axis=0) + PLAN_PAD
    return PlanInfo(length, width, (float(low[0]), float(high[0])), (float(low[1]), float(high[1])),
                    existing, present, len(scene.result.candidates))


def legend_handles(present_types: list[str]) -> list:
    """Tegnforklaring entries for the plan: the bin types actually drawn, plus proposal + entrance."""
    handles: list = [Patch(facecolor=_BIN_STYLE.get(t, _BIN_STYLE["annet"])[0],
                           edgecolor="#16232e", label=t) for t in present_types]
    handles.append(Patch(facecolor=_PROPOSED_FACE, edgecolor=_PROPOSED_EDGE, label="Forslag (ny kasse)"))
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=_ENTRANCE,
                          markersize=10, label="Inngang"))
    return handles


def draw_plan(ax, scene, *, info: PlanInfo | None = None, scale_bar: bool = True) -> PlanInfo:
    """Draw the CAD plan (walls, existing bins by type, numbered proposals, entrances, dimension
    lines) onto `ax`. Shared by the standalone plan sheet and the A4 report page. Pass `info` to
    reuse a measurement already taken by plan_info()."""
    info = info or plan_info(scene)
    rotation, origin, _, _ = _room_frame(scene.footprint)
    length, width, existing = info.length_m, info.width_m, info.existing
    candidates = scene.result.candidates

    ax.set_aspect("equal")
    ax.set_xlim(*info.xlim)
    ax.set_ylim(*info.ylim)
    ax.axis("off")

    ax.add_patch(Rectangle((0, 0), length, width, facecolor=_FLOOR, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0, 0), length, width, fill=False, edgecolor=_WALL, linewidth=3.2,
                           joinstyle="miter", zorder=2))

    for bx, bz, bl, bw, byaw, bin_type in existing:
        corners = _to_room(_bin_corners(bx, bz, bl, bw, byaw), rotation, origin)
        face, short = _BIN_STYLE.get(bin_type, _BIN_STYLE["annet"])
        ax.add_patch(Polygon(corners, closed=True, facecolor=face, edgecolor="#16232e",
                             linewidth=1.4, zorder=3))
        cx, cy = corners.mean(axis=0)
        ax.text(cx, cy, short, ha="center", va="center", fontsize=7.5, color="white",
                fontweight="bold", zorder=4)

    for index, cand in enumerate(candidates, start=1):
        corners = _to_room(cv2.boxPoints(cand.rect), rotation, origin)
        ax.add_patch(Polygon(corners, closed=True, facecolor=_PROPOSED_FACE, edgecolor=_PROPOSED_EDGE,
                             linewidth=2.0, linestyle=(0, (6, 4)), alpha=0.6, zorder=3))
        cx, cy = corners.mean(axis=0)
        ax.text(cx, cy, str(index), ha="center", va="center", fontsize=11, fontweight="bold",
                color=_PROPOSED_EDGE, zorder=5)

    for entrance in scene.result.entrances:
        point = _to_room(np.array([entrance]), rotation, origin)[0]
        _draw_entrance(ax, (float(point[0]), float(point[1])), length, width)

    # Norwegian decimal comma via style, so the plan and the report sheet never disagree on a number
    _dimension(ax, (0.0, info.dim_y), (length, info.dim_y), style.fmt_m(length), vertical=False)
    _dimension(ax, (info.dim_x, 0.0), (info.dim_x, width), style.fmt_m(width), vertical=True)

    if scale_bar:
        bar_x, bar_y = 0.0, info.ylim[1] - 0.38   # top strip: always clear of the drawing
        ax.plot([bar_x, bar_x + 1.0], [bar_y, bar_y], color=_WALL, lw=3, solid_capstyle="butt")
        ax.text(bar_x + 0.5, bar_y + 0.08, "1 m", ha="center", va="bottom", fontsize=8, color=_WALL)

    return info


def render_floorplan(scene, out_png: Path) -> Path:
    """Draw the floor plan for one computed Scene; writes <out>.png and <out>.pdf. Returns the PNG."""
    footprint = scene.footprint
    measured = plan_info(scene)
    fig_w = 11.0
    fig = plt.figure(figsize=(fig_w, fig_w * measured.span_y / measured.span_x + 1.9), dpi=150)
    grid = fig.add_gridspec(2, 1, height_ratios=[measured.span_y, 1.7], hspace=0.08)
    ax = fig.add_subplot(grid[0])
    bottom = grid[1].subgridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.02)
    info = fig.add_subplot(bottom[0])
    legend_ax = fig.add_subplot(bottom[1])
    for panel in (info, legend_ax):
        panel.axis("off")
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1)

    plan = draw_plan(ax, scene, info=measured)
    length, width, existing, candidates = plan.length_m, plan.width_m, plan.existing, plan.n_candidates

    title = scene.address or scene.stem
    indoor = "Innendørs" if scene.geometry.is_indoor else "Utendørs"
    info.text(0.0, 0.9, title, fontsize=15, fontweight="bold", va="center")
    info.text(0.0, 0.58,
              f"Mål: {style.num(length, 2)} × {style.fmt_m(width)}      "
              f"Areal: {style.fmt_m2(footprint.area_m2)}      {indoor}",
              fontsize=10.5, va="center")
    info.text(0.0, 0.36,
              f"Ledig gulv: {style.fmt_m2(scene.fs.free_area_m2)}      "
              f"Takhøyde: {style.fmt_m(scene.geometry.room_height_m)}",
              fontsize=10.5, va="center")
    info.text(0.0, 0.1,
              f"Eksisterende kasser: {len(existing)}      Foreslåtte nye plasser: {candidates}",
              fontsize=10.5, fontweight="bold", va="center", color=_PROPOSED_EDGE)

    handles = legend_handles(plan.present_types)
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=9.5, handlelength=1.4,
                     labelspacing=0.6, title="Tegnforklaring", title_fontsize=10)

    fig.suptitle("Søppelrom – plantegning", x=0.02, ha="left", fontsize=12, color="#555")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


def build(stem: str, bin_type: str = "4-hjuls container", out_png: Path | None = None) -> Path:
    scene = pipeline.compute_scene(stem, bin_type)
    if out_png is None:
        out_png = PREVIEW_ROOT / stem / "floorplan.png"
    return render_floorplan(scene, out_png)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D plantegning (PNG + PDF) for ett skann")
    parser.add_argument("--scan", required=True, help="skann-stem, f.eks. 71872_20260709T1344_OF")
    parser.add_argument("--bin-type", default="4-hjuls container", choices=list(BIN_TYPES))
    parser.add_argument("--out", default=None, help="valgfri PNG-sti (PDF skrives ved siden av)")
    args = parser.parse_args()
    path = build(args.scan, args.bin_type, Path(args.out) if args.out else None)
    print(f"Skrev {path} og {path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
