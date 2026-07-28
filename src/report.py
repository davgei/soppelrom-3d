"""A4 one-pager per address — the sheet you can hand to whoever owns the property.

Every number on the page comes from pipeline.compute_scene and the annotations it reads; nothing is
invented here, and a quantity the pipeline cannot measure (a ceiling height outdoors, say) is marked
as not measured instead of being filled in. The CAD plan is drawn by floorplan.draw_plan straight
onto an axes of this page, so the report and the standalone plan sheet cannot drift apart.

    .venv\\Scripts\\python.exe -m src.report --scan <stem>
    .venv\\Scripts\\python.exe -m src.report --all
"""
from __future__ import annotations

import argparse
import datetime as dt
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from . import floorplan, pipeline, render, style  # noqa: E402
from .annotations import BIN_TYPES  # noqa: E402
from .paths import PREVIEW_ROOT  # noqa: E402

A4_W, A4_H = 8.27, 11.69          # portrait inches
LEFT, RIGHT = 0.068, 0.932        # print margins as a fraction of the page width


def _hex(color: style.BGR) -> str:
    """style's BGR constants as matplotlib hex, so the page shares one palette with the previews."""
    blue, green, red = color
    return f"#{red:02x}{green:02x}{blue:02x}"


INK = _hex(style.INK)
MUTED = "#5f6b74"
RULE = "#d9d6ce"
PANEL_BG = "#f4f3ef"
ACCENT = _hex(style.NEW_BIN_EDGE)
WARN = _hex(style.EXISTING_BIN_EDGE)
WARN_BG = "#fdeceb"

# Wording that only appears on this sheet. Neutral on purpose: no logo, no signature, no letterhead —
# the page states who generated it and that the numbers are computed, nothing more.
KICKER = "Søppelrom · kartlagt fra 3D-skann"
SUBJECT = "Plass, tilkomst og forslag til nye avfallskasser"
DISCLAIMER = ("Oslo kommune · automatisk generert fra 3D-skann. Alle mål er beregnet ut fra skannet "
              "geometri og må kvalitetssikres på stedet.")
UNKNOWN = "ikke målt"


def _font_stack() -> list[str]:
    """Segoe UI when installed (matches the raster previews), else matplotlib's bundled DejaVu Sans.
    Both cover æ/ø/å and ²; an empty list falls back to whatever sans-serif exists."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return [name for name in ("Segoe UI", "DejaVu Sans") if name in installed]


def _rc() -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": _font_stack() + plt.rcParamsDefault["font.sans-serif"],
        "pdf.fonttype": 42,        # embed TrueType so the PDF text stays selectable/searchable
        "text.color": INK,
        "axes.unicode_minus": False,
    }


# ---------------------------------------------------------------- facts

@dataclass
class Row:
    label: str
    value: str
    kind: str = "main"     # main | sub | good | warn


def _count_by_type(types: list[str]) -> list[tuple[str, int]]:
    """Bin types in the canonical BIN_TYPES order, only those present. An unrecognised or missing
    type keeps a row of its own instead of vanishing from a breakdown whose total is printed above
    it — the sub-rows must always add up to that total."""
    counts = Counter(bin_type or "ukjent type" for bin_type in types)
    order = list(BIN_TYPES) + [name for name in counts if name not in BIN_TYPES]
    return [(name, counts[name]) for name in order if counts[name]]


def _entrance_note(scene) -> str:
    if scene.enclosed:
        return "ingen – rommet leses som lukket"
    if scene.clicked:
        return "satt manuelt"
    return "funnet automatisk"


def _facts(scene, plan: floorplan.PlanInfo) -> list[Row]:
    """The key-numbers block. Only measured quantities: an unmeasurable one says so."""
    geometry, fs = scene.geometry, scene.fs
    rows = [
        Row(style.LABELS["room"], f"{style.num(plan.length_m, 2)} × {style.fmt_m(plan.width_m)}"),
        Row("Gulvareal (yttermål)", style.fmt_m2(scene.footprint.area_m2)),
        Row("Type", "Innendørs rom" if geometry.is_indoor else "Utendørs / åpent"),
    ]
    # room_height_m is a ceiling measurement only indoors; outdoors it is just how high the scan
    # reaches, so it must not be presented as a takhøyde.
    if geometry.is_indoor and geometry.ceiling_height_m is not None:
        rows.append(Row("Takhøyde", style.fmt_m(geometry.room_height_m)))
    else:
        rows.append(Row("Takhøyde", f"{UNKNOWN} (åpent område)"))

    rows.append(Row("Skannet gulv", style.fmt_m2(fs.observed_floor_area_m2)))
    rows.append(Row(style.LABELS["free"], style.fmt_m2(fs.free_area_m2), "good"))
    if fs.observed_floor_area_m2 > 0:
        share = 100.0 * fs.free_area_m2 / fs.observed_floor_area_m2
        rows.append(Row("andel av skannet gulv", f"{style.num(share, 0)} %", "sub"))
    rows.append(Row(style.LABELS["occupied"], style.fmt_m2(fs.occupied_on_floor_m2)))

    rows.append(Row("Kasser i rommet i dag", str(len(plan.existing))))
    for name, count in _count_by_type([bin_[5] for bin_ in plan.existing]):
        rows.append(Row(name, str(count), "sub"))

    candidates = scene.result.candidates
    rows.append(Row("Forslag: nye plasser", str(len(candidates)), "good"))
    for name, count in _count_by_type([cand.bin_type for cand in candidates]):
        rows.append(Row(name, str(count), "sub"))

    rows.append(Row("Innganger", str(len(scene.result.entrances))))
    rows.append(Row("kilde", _entrance_note(scene), "sub"))
    if scene.enclosed:
        rows.append(Row(_wrap("Innesperret – døren var lukket under skanningen. Tilkomst og "
                              "skyve-sti kunne derfor ikke beregnes.", 42), "", "warn"))
    return rows


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width))


# ---------------------------------------------------------------- second visual

@dataclass
class Visual:
    path: Path
    caption: str
    legend_title: str


def _second_visual(scene) -> Visual:
    """The photo-like companion to the plan: the simplified 3D model when that snapshot has already
    been rendered, otherwise the 'Plassering' top-down. Rendered without a title so this page owns
    the wording, and into its own file so the pipeline's placements.png is left alone."""
    out_dir = PREVIEW_ROOT / scene.stem
    snapshot = out_dir / "reconstruction.png"
    if snapshot.exists():
        return Visual(snapshot, "Forenklet 3D-modell av rommet", "3D-modellen")
    path = out_dir / "rapport_plassering.png"
    render.placements_over_scene(scene.scene_vis, scene.result, path)
    return Visual(path, "Skannet rom sett ovenfra · skyve-sti i blått", "bildet av skannet")


# ---------------------------------------------------------------- page furniture

def _caption(fig, x: float, y: float, text: str) -> None:
    fig.text(x, y, text, fontsize=8.5, color=MUTED, va="bottom", fontweight="bold")


def _rule(fig, y: float, x0: float = LEFT, x1: float = RIGHT, color: str = RULE,
          width: float = 1.0, alpha: float = 1.0) -> None:
    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color=color, lw=width,
                          alpha=alpha, solid_capstyle="butt"))


def _panel(fig, x: float, y: float, w: float, h: float) -> None:
    """Light card with a coloured spine — the same 'panel' idea as the raster previews, on paper."""
    fig.add_artist(Rectangle((x, y), w, h, transform=fig.transFigure, facecolor=PANEL_BG,
                             edgecolor=RULE, linewidth=0.8, zorder=0))
    fig.add_artist(Rectangle((x, y), 0.005, h, transform=fig.transFigure, facecolor=ACCENT,
                             edgecolor="none", zorder=1))


PHOTO_COL_W = 0.50      # width of the picture column; the key numbers get the rest of the row
PHOTO_MAX_H = 0.27      # page fraction; keeps a portrait-shaped scan from crowding out the plan
ROW_H = 0.0175          # main row pitch, page fraction
SUB_H = 0.0135
WARN_LINE = 0.0135
PANEL_PAD = 0.013


def _row_height(row: Row) -> float:
    if row.kind == "warn":
        return 0.011 + WARN_LINE * (row.label.count("\n") + 1)
    return SUB_H if row.kind == "sub" else ROW_H


def _facts_height(rows: list[Row]) -> float:
    return sum(_row_height(row) for row in rows) + 2 * PANEL_PAD


def _draw_facts(fig, rows: list[Row], x: float, top: float, w: float) -> None:
    height = _facts_height(rows)
    _panel(fig, x, top - height, w, height)
    left = x + PANEL_PAD + 0.006
    right = x + w - PANEL_PAD
    y = top - PANEL_PAD
    for index, row in enumerate(rows):
        pitch = _row_height(row)
        if row.kind == "warn":
            fig.add_artist(Rectangle((x + 0.008, y - pitch + 0.005), w - 0.016, pitch - 0.007,
                                     transform=fig.transFigure, facecolor=WARN_BG,
                                     edgecolor=WARN, linewidth=0.7, zorder=2))
            # a drawn triangle, not "⚠": the warning-sign glyph is missing from Segoe UI and would
            # print as an empty box on exactly the sheet where the warning matters most
            fig.add_artist(Line2D([left + 0.004], [y - pitch / 2 + 0.001], marker="^",
                                  markersize=8, markerfacecolor=WARN, markeredgecolor=WARN,
                                  linestyle="none", transform=fig.transFigure, zorder=3))
            fig.text(left + 0.017, y - pitch / 2 + 0.001, row.label, fontsize=7.8, color=WARN,
                     va="center", ha="left", fontweight="bold", zorder=3, linespacing=1.4)
            y -= pitch
            continue
        if row.kind == "sub":
            fig.text(left + 0.012, y - pitch / 2, row.label, fontsize=7.8, color=MUTED,
                     va="center", ha="left", zorder=3)
            fig.text(right, y - pitch / 2, row.value, fontsize=7.8, color=MUTED,
                     va="center", ha="right", zorder=3)
        else:
            colour = ACCENT if row.kind == "good" else INK
            weight = "bold" if row.kind == "good" else "normal"
            if index:
                _rule(fig, y, left, right, color=RULE, width=0.6, alpha=0.9)
            fig.text(left, y - pitch / 2, row.label, fontsize=9.0, color=INK, va="center",
                     ha="left", zorder=3)
            fig.text(right, y - pitch / 2, row.value, fontsize=9.0, color=colour, va="center",
                     ha="right", fontweight=weight, zorder=3)
        y -= pitch


_LEGEND_KIND = {
    "box": lambda label, colour: Patch(facecolor=colour, edgecolor="#16232e", label=label),
    "fill": lambda label, colour: Patch(facecolor=colour, edgecolor="none", alpha=0.5, label=label),
    "line": lambda label, colour: Line2D([0], [0], color=colour, lw=4, label=label),
    "dot": lambda label, colour: Line2D([0], [0], marker="o", color="none", markerfacecolor=colour,
                                        markersize=9, label=label),
}


def _scene_handles() -> list:
    """Legend for the raster visual, straight from render.PLACEMENT_LEGEND so the two agree."""
    return [_LEGEND_KIND.get(entry[2] if len(entry) > 2 else "box", _LEGEND_KIND["box"])(
        entry[0], _hex(entry[1])) for entry in render.PLACEMENT_LEGEND]


def _draw_legend(fig, plan: floorplan.PlanInfo, legend_title: str, top: float, height: float) -> None:
    """Two grouped keys side by side: the plan colours bins by TYPE, the raster visual colours them
    by new/existing — spelling that out beats pretending one key covers both."""
    ax = fig.add_axes([LEFT, top - height, RIGHT - LEFT, height])
    ax.axis("off")
    plan_legend = ax.legend(handles=floorplan.legend_handles(plan.present_types),
                            title="Tegnforklaring – plantegning", loc="upper left",
                            bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=8.2,
                            title_fontsize=8.5, handlelength=1.3, labelspacing=0.5, ncols=2,
                            columnspacing=1.4, alignment="left")
    plan_legend.get_title().set_color(MUTED)
    plan_legend.get_title().set_fontweight("bold")
    ax.add_artist(plan_legend)
    scene_legend = ax.legend(handles=_scene_handles(), title=f"Tegnforklaring – {legend_title}",
                             loc="upper left", bbox_to_anchor=(0.52, 1.0), frameon=False,
                             fontsize=8.2, title_fontsize=8.5, handlelength=1.3, labelspacing=0.5,
                             ncols=2, columnspacing=1.4, alignment="left")
    scene_legend.get_title().set_color(MUTED)
    scene_legend.get_title().set_fontweight("bold")


# ---------------------------------------------------------------- the page

def render_page(scene, out_pdf: Path) -> Path:
    """Compose the A4 sheet for one computed Scene; writes <out>.pdf and <out>.png."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    visual = _second_visual(scene)
    photo = plt.imread(str(visual.path))
    info = floorplan.plan_info(scene)

    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(A4_W, A4_H), dpi=170)
        fig.patch.set_facecolor("white")

        # ---- header: address is the headline, the scan id lives in the footer
        heading = scene.address or scene.stem
        fig.text(LEFT, 0.968, KICKER, fontsize=9, color=ACCENT, va="top", fontweight="bold")
        fig.text(LEFT, 0.951, heading, fontsize=19.5, color=INK, va="top", fontweight="bold")
        subtitle = SUBJECT if scene.address else f"{SUBJECT} · adresse ikke lagret i skannet"
        fig.text(LEFT, 0.922, subtitle, fontsize=10.5, color=MUTED, va="top")
        _rule(fig, 0.9035, width=1.0)
        fig.add_artist(Rectangle((LEFT, 0.9025), 0.075, 0.0022, transform=fig.transFigure,
                                 facecolor=ACCENT, edgecolor="none"))

        # ---- measure the blocks with a fixed size first, then let the plan take what is left
        rows = _facts(scene, info)
        facts_h = _facts_height(rows)
        photo_w = PHOTO_COL_W
        photo_h = (photo_w * A4_W) * (photo.shape[0] / photo.shape[1]) / A4_H
        if photo_h > PHOTO_MAX_H:      # a portrait-shaped scan would otherwise push the plan off
            photo_w *= PHOTO_MAX_H / photo_h
            photo_h = PHOTO_MAX_H
        row_h = max(facts_h, photo_h)

        legend_h = 0.074
        legend_top = 0.090 + legend_h
        caption_h = 0.019
        gap = 0.026
        region_top, region_bottom = 0.884, legend_top + 0.024
        room = region_top - region_bottom - (2 * caption_h + gap + row_h)
        if room < 0.20:                # unusually many rows: tighten the gap before the drawing
            gap = 0.016
            room = region_top - region_bottom - (2 * caption_h + gap + row_h)

        plan_w = RIGHT - LEFT
        natural_h = (plan_w * A4_W) * (info.span_y / info.span_x) / A4_H
        plan_h = max(min(natural_h, room), 0.14)
        # A squat room cannot fill the height at full page width, so there is air left over. Spend a
        # little of it under the header and between the blocks; the rest sits above the legend, where
        # slack reads as a normal document margin rather than as a hole.
        slack = max(room - plan_h, 0.0)
        gap += min(slack * 0.25, 0.014)

        # ---- CAD plan, drawn by floorplan so this sheet and the plan sheet cannot drift apart
        top = region_top - min(slack * 0.25, 0.016)
        _caption(fig, LEFT, top - caption_h + 0.004, "PLANTEGNING · SETT OVENFRA · MÅL I METER")
        plan_ax = fig.add_axes([LEFT, top - caption_h - plan_h, plan_w, plan_h])
        plan = floorplan.draw_plan(plan_ax, scene, info=info)
        top -= caption_h + plan_h + gap

        # ---- raster visual + key numbers, side by side. Both keep to the page margins (picture left,
        # numbers right) so a portrait-shaped scan just leaves air in the middle instead of drifting.
        _caption(fig, LEFT, top - caption_h + 0.004, visual.caption.upper())
        photo_ax = fig.add_axes([LEFT, top - caption_h - photo_h, photo_w, photo_h])
        photo_ax.imshow(photo)
        photo_ax.set_xticks([])
        photo_ax.set_yticks([])
        for spine in photo_ax.spines.values():
            spine.set_color(RULE)
            spine.set_linewidth(0.8)

        facts_x = LEFT + PHOTO_COL_W + 0.030
        _caption(fig, facts_x, top - caption_h + 0.004, "NØKKELTALL")
        _draw_facts(fig, rows, facts_x, top - caption_h, RIGHT - facts_x)

        _draw_legend(fig, plan, visual.legend_title, legend_top, legend_h)

        # ---- footer
        _rule(fig, 0.070, width=0.8)
        fig.text(LEFT, 0.055, DISCLAIMER, fontsize=7.6, color=MUTED, va="top")
        stamp = dt.date.today().strftime("%d.%m.%Y")
        fig.text(LEFT, 0.038, f"Skann: {scene.stem}   ·   Generert: {stamp}   ·   "
                              f"Kassetype i beregningen: {scene.bin_type}",
                 fontsize=7.6, color=MUTED, va="top")
        fig.text(RIGHT, 0.038, "Side 1 av 1", fontsize=7.6, color=MUTED, va="top", ha="right")

        fig.savefig(out_pdf, facecolor="white")
        fig.savefig(out_pdf.with_suffix(".png"), facecolor="white")
        plt.close(fig)
    return out_pdf


def build(stem: str, bin_type: str = "4-hjuls container", out_pdf: Path | None = None) -> Path:
    scene = pipeline.compute_scene(stem, bin_type)
    if out_pdf is None:
        out_pdf = PREVIEW_ROOT / stem / "rapport.pdf"
    return render_page(scene, out_pdf)


def main() -> None:
    parser = argparse.ArgumentParser(description="A4 énsides rapport (PDF + PNG) per adresse")
    parser.add_argument("--scan", help="skann-stem, f.eks. 71872_20260709T1344_OF")
    parser.add_argument("--all", action="store_true", help="alle ferdig forberedte skann")
    parser.add_argument("--bin-type", default="4-hjuls container", choices=list(BIN_TYPES))
    parser.add_argument("--out", default=None, help="valgfri PDF-sti (PNG skrives ved siden av)")
    args = parser.parse_args()
    if not args.scan and not args.all:
        parser.error("oppgi --scan <stem> eller --all")

    stems = [args.scan] if args.scan else [s for s in pipeline.list_scans() if pipeline.is_prepared(s)]
    for stem in stems:
        try:
            path = build(stem, args.bin_type, Path(args.out) if args.out else None)
        except Exception as error:  # one unreadable scan must not stop a --all run
            print(f"{stem}: hoppet over ({error})")
            continue
        print(f"Skrev {path} og {path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
