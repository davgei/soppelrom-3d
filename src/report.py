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
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from . import floorplan, pipeline, reconstruct3d, render, style  # noqa: E402
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

# ---------------------------------------------------------------------------
# The two bins this round is about.
#
# READ THIS BEFORE SENDING THE SHEET TO ANYONE. Everything else on these pages is measured from the
# scan; the text below is NOT. It is ordinary practical wording written to be replaced by whatever the
# waste service actually publishes about the new bins -- fractions, bag colours, collection intervals
# and who is responsible for what are policy, and this file has no way to know them. The sheet says so
# in POLICY_NOTE, which must stay on the page for as long as this text is unverified.
# ---------------------------------------------------------------------------
NEW_BINS: tuple[dict[str, str], ...] = (
    {
        "name": "Glass- og metallemballasje",
        "short": "Glass og metall",
        "purpose": "Én felles kasse for emballasje av glass og metall.",
        "yes": "Syltetøyglass, flasker uten pant, hermetikkbokser, lokk og korker av metall, "
               "aluminiumsformer og rene folieboller.",
        "no": "Drikkeglass, keramikk, speil, vindusglass, lyspærer og ildfaste former – de er ikke "
              "emballasje og tåler ikke samme gjenvinning.",
        "hint": "Skyll lett og la det tørke. Emballasjen trenger ikke være ripefri, bare tom.",
    },
    {
        "name": "Matavfall",
        "short": "Matavfall",
        "purpose": "Kasse for matrester og annet organisk kjøkkenavfall.",
        "yes": "Matrester, skrell, kaffegrut og filter, teposer, eggeskall, servietter med matrester.",
        "no": "Flytende væske, store mengder olje, bleier, dyreekskrementer, blomsterjord og "
              "plastposer som ikke er beregnet på matavfall.",
        "hint": "Knyt posen godt. En full kasse som lukker tett gir mindre lukt og færre skadedyr enn "
                "en halvfull som står åpen.",
    },
)

# What the housing cooperative has to do before the bins arrive. Ordered by when it matters: the two
# first are the ones that stop a delivery, the rest make the daily use work.
PREP_STEPS: tuple[tuple[str, str], ...] = (
    ("Rydd selve plassen",
     "Flytt sykler, paller, hageredskap og annet som står der de nye kassene skal stå. Kassene kan "
     "ikke settes ned oppå noe, og en plass som er opptatt på leveringsdagen blir ikke levert."),
    ("Hold trilleveien fri",
     "Ruten fra kassene til inn-/utgangen må være fri hele veien, ikke bare ved kassene. En kasse "
     "trilles på to hjul og trenger sammenhengende, noenlunde jevnt underlag."),
    ("Sjekk døra og terskelen",
     "Måleopp bredden på døråpningen og høyden på terskelen. En høy terskel eller et trinn er den "
     "vanligste grunnen til at en kasse ikke kommer inn i det hele tatt."),
    ("Bestem hvem som har ansvaret",
     "Én kontaktperson for rommet gjør det enklere å melde om full kasse, ødelagt hjul eller lås "
     "som ikke virker."),
    ("Merk kassene tydelig",
     "Sett skilt eller klistremerke på hver kasse med hva som skal i den. Feilsortering er nesten "
     "alltid uvitenhet, ikke vrangvilje."),
    ("Tenk på vinteren",
     "Plassen må kunne måkes og strøs. Kasser som fryser fast eller står bak en brøytekant blir "
     "ikke tømt."),
)

# Below this much free floor the sheet says "there is no room" rather than "nothing was found": a
# 4-hjuls container's footprint alone is 1.07 m2, and it also has to be wheeled in and parked clear of
# the others. 3 m2 of floor scattered around a room is not a place for a bin.
#
# Measured: of the 46 rooms that currently produce no proposal, NONE is under this threshold -- the
# tightest has 4.9 m2 free. So every one of them gets the "nothing found automatically" wording today,
# and that is the honest reading: those rooms are not full, the placement search is not finding what is
# there. Kept anyway, because a genuinely full room must not be told to look again.
NO_ROOM_BELOW_M2 = 3.0

POLICY_NOTE = ("Teksten om hva som skal i hver kasse er generell og må erstattes med "
               "renovasjonstjenestens egen informasjon før arket sendes ut.")
NEXT_STEPS = ("Arket er et forslag, ikke et vedtak. Gi tilbakemelding hvis plasseringen ikke passer – "
              "for eksempel fordi plassen brukes til noe annet, eller fordi tilkomsten er dårligere "
              "enn skannet viser. Målene under er beregnet fra skannet og bør kontrolleres på stedet.")


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


def _line_h(fontsize: float, linespacing: float = 1.42) -> float:
    """Height of one line of text as a fraction of the page.

    Every block on these sheets holds WRAPPED text whose line count depends on the address, the room
    and the wording, so a fixed block height is a collision waiting to happen -- and it happened three
    times on the first draft: the proposal band printed its bin names on top of its own body text, the
    bin panels pushed their closing tip out through the bottom border, and step 1 of the checklist ran
    into the title of step 3. Blocks measure themselves instead.

    A4_H is in inches and matplotlib font sizes are in points, hence the 72.
    """
    return fontsize * linespacing / (A4_H * 72.0)


def _text_h(text: str, fontsize: float, linespacing: float = 1.42) -> float:
    """Height of an already-wrapped block (use _wrap first)."""
    return (text.count("\n") + 1) * _line_h(fontsize, linespacing)


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


def _split_rows(rows: list[Row]) -> tuple[list[Row], list[Row]]:
    """Halve the fact rows for a two-column layout, cutting only before a MAIN row.

    A 'sub' row is a breakdown of the main row above it ("2-hjuls dunk  4" under "Kasser i rommet i
    dag  5"), so a cut between them would print a count whose parts are in the other column and whose
    own total is nowhere near.
    """
    heights = [_row_height(row) for row in rows]
    half = sum(heights) / 2
    running = 0.0
    cut = len(rows)
    for index, (row, height) in enumerate(zip(rows, heights)):
        if running >= half and row.kind != "sub":
            cut = index
            break
        running += height
    return rows[:cut], rows[cut:]


_FACTS_GAP = 0.030


def _facts_columns(rows: list[Row], available: float) -> tuple[list[Row], list[Row]]:
    """The two columns to draw, dropping the breakdown sub-rows if the full set will not fit.

    Sub-rows go first because they are detail: the main row above each group already carries its
    total, so losing "2-hjuls dunk 4 / 4-hjuls container 1" still leaves "Kasser i rommet i dag 5".
    """
    pair = _split_rows(rows)
    if max(_facts_height(pair[0]), _facts_height(pair[1])) > available:
        pair = _split_rows([row for row in rows if row.kind != "sub"])
    return pair


def _facts_columns_height(rows: list[Row], available: float) -> float:
    pair = _facts_columns(rows, available)
    return max(_facts_height(pair[0]), _facts_height(pair[1]) if pair[1] else 0.0)


def _draw_facts_columns(fig, rows: list[Row], top: float, available: float) -> None:
    """The measured table in two columns. One tall column ran off the bottom of the page and printed
    through the footer, twice, while the blocks above it were still being tuned."""
    col_w = (RIGHT - LEFT - _FACTS_GAP) / 2
    left_rows, right_rows = _facts_columns(rows, available)
    _draw_facts(fig, left_rows, LEFT, top, col_w)
    if right_rows:
        _draw_facts(fig, right_rows, LEFT + col_w + _FACTS_GAP, top, col_w)


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

def _proposal_line(scene) -> tuple[str, str]:
    """Headline + supporting line for the proposal band, phrased for what the scan actually found.

    Three genuinely different situations, and one wording for each. A sheet that says "0 nye kasser
    foreslått" next to 130 m2 of free floor tells the reader nothing about why, so the reason has to be
    on the page: a sealed room, no room, or space that the automatic check could not confirm a route
    to. 46 of 322 rooms came out with no proposal, so this is not an edge case.
    """
    n = len(scene.result.candidates)
    free = scene.fs.free_area_m2
    if n >= len(NEW_BINS):
        # The plan draws and numbers EVERY position found, so the text must not claim it shows two --
        # a reader who counts three green boxes against "de to beste" stops trusting the rest of the
        # sheet. The numbering is the link: the new bins are meant for 1 and 2.
        return ("Det er plass til begge de nye kassene",
                f"Skannet viser {style.fmt_m2(free)} ledig gulv og {n} mulige plasseringer, "
                f"nummerert på plantegningen. De to nye kassene er tenkt på plass 1 og 2; "
                f"{'den siste er' if n - len(NEW_BINS) == 1 else 'de øvrige er'} plass til senere.")
    if n > 0:
        return (f"Det er plass til {n} av {len(NEW_BINS)} nye kasser",
                f"Skannet finner {n} plassering{'er' if n > 1 else ''} som en kasse kan trilles til. "
                f"Resten av det ledige gulvet ({style.fmt_m2(free)}) er enten for trangt eller "
                f"uten tilkomst.")
    if scene.enclosed:
        return ("Plassering kunne ikke beregnes",
                "Døren var lukket under skanningen, så tilkomst og trillevei mangler. "
                "Rommet må vurderes på stedet.")
    # Two very different reasons for finding nothing, and telling them apart matters: "det betyr ikke
    # nødvendigvis at det er fullt" printed under 4,9 m2 of free floor reads as the sheet not knowing
    # what it is talking about. A 4-wheel container's footprint is 1.07 m2 and it needs room to be
    # rolled in and stood beside, so a few square metres of scattered floor is genuinely full.
    if free < NO_ROOM_BELOW_M2:
        return ("Det er ikke plass til nye kasser",
                f"Rommet har bare {style.fmt_m2(free)} ledig gulv igjen. En ny kasse trenger plass "
                f"til selve kassa og til å trille den inn, så her må noe annet flyttes eller "
                f"fjernes først.")
    return ("Ingen plassering ble funnet automatisk",
            f"Rommet har {style.fmt_m2(free)} ledig gulv, men den automatiske sjekken fant ingen "
            f"plass med bekreftet trillevei fra inngangen. Det bør vurderes på stedet – det betyr "
            f"ikke nødvendigvis at det er fullt.")


def _band_parts(scene) -> tuple[str, str, str]:
    headline, support = _proposal_line(scene)
    names = "   ·   ".join(f"{index + 1}. {bin_['name']}" for index, bin_ in enumerate(NEW_BINS))
    return headline, _wrap(support, 96), names


def _band_height(scene) -> float:
    _headline, support, _names = _band_parts(scene)
    return (0.014 + _line_h(12.5) + 0.005 + _text_h(support, 9.2, 1.45)
            + 0.008 + _line_h(9.4) + 0.012)


def _draw_new_bins_band(fig, scene, top: float) -> float:
    """The band under the header: what is coming, and how much room the scan says there is.
    Returns the y of its bottom edge."""
    height = _band_height(scene)
    _panel(fig, LEFT, top - height, RIGHT - LEFT, height)
    headline, support, names = _band_parts(scene)
    inner = LEFT + 0.018
    y = top - 0.014
    fig.text(inner, y, headline, fontsize=12.5, color=INK, va="top", fontweight="bold")
    y -= _line_h(12.5) + 0.005
    fig.text(inner, y, support, fontsize=9.2, color=MUTED, va="top", linespacing=1.45)
    y -= _text_h(support, 9.2, 1.45) + 0.008
    fig.text(inner, y, names, fontsize=9.4, color=ACCENT, va="top", fontweight="bold")
    return top - height


def _draw_view_row(fig, paths: list[Path], top: float, height: float) -> float:
    """Up to three small 3D views side by side. Returns the y of the bottom edge.

    Each picture is placed in its own equal column and scaled to FIT that column, never cropped: the
    three renders come out at different aspect ratios (the crop follows the room's silhouette from
    each angle), and stretching them to a common box would distort the room.
    """
    if not paths:
        return top
    gap = 0.012
    columns = len(paths)
    col_w = (RIGHT - LEFT - gap * (columns - 1)) / columns
    for index, path in enumerate(paths):
        image = plt.imread(str(path))
        aspect = image.shape[0] / image.shape[1]
        draw_h = (col_w * A4_W) * aspect / A4_H
        draw_w = col_w
        if draw_h > height:                      # too tall for the row: shrink width to match
            draw_w = col_w * (height / draw_h)
            draw_h = height
        x = LEFT + index * (col_w + gap) + (col_w - draw_w) / 2
        ax = fig.add_axes([x, top - height + (height - draw_h) / 2, draw_w, draw_h])
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(RULE)
            spine.set_linewidth(0.8)
    return top - height


def _kpis(scene) -> list[tuple[str, str]]:
    """The four numbers a board actually asks about, for the strip on page 1. The full table is on
    page 2 -- fifteen rows cannot share an A4 with a plan drawing and three renders."""
    return [
        ("Gulvareal", style.fmt_m2(scene.footprint.area_m2)),
        ("Ledig gulv", style.fmt_m2(scene.fs.free_area_m2)),
        ("Kasser i dag", str(len(scene.result.existing_bins))),
        ("Nye plasser funnet", str(len(scene.result.candidates))),
    ]


def _extra_kpis(scene, plan) -> list[tuple[str, str]]:
    """Numbers for page 2's strip: the ones page 1's four tiles do NOT already show, so the second
    sheet adds information instead of repeating it."""
    geometry = scene.geometry
    if geometry.is_indoor and geometry.ceiling_height_m is not None:
        ceiling = style.fmt_m(geometry.room_height_m)
    else:
        ceiling = "utendørs"
    return [
        ("Rommets yttermål", f"{style.num(plan.length_m, 2)} × {style.fmt_m(plan.width_m)}"),
        ("Takhøyde", ceiling),
        ("Skannet gulv", style.fmt_m2(scene.fs.observed_floor_area_m2)),
        ("Innganger", str(len(scene.result.entrances))),
    ]


def _draw_kpi_strip(fig, items: list[tuple[str, str]], top: float, height: float,
                    value_size: float = 15.5) -> None:
    gap = 0.010
    col_w = (RIGHT - LEFT - gap * (len(items) - 1)) / len(items)
    for index, (label, value) in enumerate(items):
        x = LEFT + index * (col_w + gap)
        fig.add_artist(Rectangle((x, top - height), col_w, height, transform=fig.transFigure,
                                 facecolor=PANEL_BG, edgecolor=RULE, linewidth=0.8, zorder=0))
        # the yttermål value is the longest string here, so it gets a size that still fits its tile
        size = value_size if len(value) <= 9 else value_size - 3.0
        fig.text(x + 0.011, top - 0.013, value, fontsize=size, color=INK, va="top",
                 fontweight="bold")
        fig.text(x + 0.011, top - height + 0.009, label.upper(), fontsize=7.4, color=MUTED,
                 va="bottom", fontweight="bold")


def _draw_kpis(fig, scene, top: float, height: float) -> None:
    items = _kpis(scene)
    gap = 0.010
    col_w = (RIGHT - LEFT - gap * (len(items) - 1)) / len(items)
    for index, (label, value) in enumerate(items):
        x = LEFT + index * (col_w + gap)
        fig.add_artist(Rectangle((x, top - height), col_w, height, transform=fig.transFigure,
                                 facecolor=PANEL_BG, edgecolor=RULE, linewidth=0.8, zorder=0))
        fig.text(x + 0.011, top - 0.013, value, fontsize=15.5, color=INK, va="top",
                 fontweight="bold")
        fig.text(x + 0.011, top - height + 0.009, label.upper(), fontsize=7.4, color=MUTED,
                 va="bottom", fontweight="bold")


def _header(fig, scene, subtitle: str) -> None:
    heading = scene.address or scene.stem
    fig.text(LEFT, 0.968, KICKER, fontsize=9, color=ACCENT, va="top", fontweight="bold")
    fig.text(LEFT, 0.951, heading, fontsize=19.5, color=INK, va="top", fontweight="bold")
    fig.text(LEFT, 0.922, subtitle, fontsize=10.5, color=MUTED, va="top")
    _rule(fig, 0.9035, width=1.0)
    fig.add_artist(Rectangle((LEFT, 0.9025), 0.075, 0.0022, transform=fig.transFigure,
                             facecolor=ACCENT, edgecolor="none"))


def _footer(fig, scene, page: int, pages: int) -> None:
    _rule(fig, 0.070, width=0.8)
    fig.text(LEFT, 0.055, DISCLAIMER, fontsize=7.6, color=MUTED, va="top")
    stamp = dt.date.today().strftime("%d.%m.%Y")
    fig.text(LEFT, 0.038, f"Skann: {scene.stem}   ·   Generert: {stamp}   ·   "
                          f"Kassetype i beregningen: {scene.bin_type}",
             fontsize=7.6, color=MUTED, va="top")
    fig.text(RIGHT, 0.038, f"Side {page} av {pages}", fontsize=7.6, color=MUTED, va="top", ha="right")


def _page_one(scene, info, views: list[Path]):
    """Where the bins go: the proposal, the plan, three angles of the room, the headline numbers."""
    fig = plt.figure(figsize=(A4_W, A4_H), dpi=170)
    fig.patch.set_facecolor("white")
    subtitle = SUBJECT if scene.address else f"{SUBJECT} · adresse ikke lagret i skannet"
    _header(fig, scene, subtitle)

    kpi_h = 0.052
    legend_h = 0.074
    legend_top = 0.090 + legend_h
    caption_h = 0.019
    views_h = 0.135 if views else 0.0
    gap = 0.020

    top = _draw_new_bins_band(fig, scene, 0.884) - gap

    # the plan takes whatever is left once the fixed blocks are reserved
    bottom_reserved = legend_top + 0.020 + kpi_h + gap + (views_h + caption_h + gap if views else 0)
    room = top - bottom_reserved - caption_h
    plan_w = RIGHT - LEFT
    natural_h = (plan_w * A4_W) * (info.span_y / info.span_x) / A4_H
    plan_h = max(min(natural_h, room), 0.13)

    _caption(fig, LEFT, top - caption_h + 0.004, "PLANTEGNING · SETT OVENFRA · MÅL I METER")
    plan_ax = fig.add_axes([LEFT, top - caption_h - plan_h, plan_w, plan_h])
    plan = floorplan.draw_plan(plan_ax, scene, info=info)
    top -= caption_h + plan_h + gap

    if views:
        _caption(fig, LEFT, top - caption_h + 0.004,
                 "ROMMET I 3D · GRØNT = FORESLÅTT PLASS, RØDT = KASSE SOM STÅR DER I DAG")
        top = _draw_view_row(fig, views, top - caption_h, views_h) - gap

    _draw_kpis(fig, scene, top, kpi_h)
    _draw_legend(fig, plan, "3D-modellen", legend_top, legend_h)
    _footer(fig, scene, 1, 2)
    return fig


def _page_two(scene, info):
    """How the bins are used and what has to happen first, plus the full measured table."""
    fig = plt.figure(figsize=(A4_W, A4_H), dpi=170)
    fig.patch.set_facecolor("white")
    _header(fig, scene, "Slik tas de nye kassene i bruk")

    top = 0.884
    caption_h = 0.019
    _caption(fig, LEFT, top - caption_h + 0.004, "DE NYE KASSENE")
    top -= caption_h

    # one block per new bin: purpose, what goes in, what must not, and one practical tip
    for bin_ in NEW_BINS:
        # 104 characters, not 88: the page is 8.27 in wide and the text column is most of it, so the
        # narrower wrap bought nothing and cost two lines per panel -- which is most of why the table
        # further down ran off the bottom of the sheet.
        yes, no = _wrap(bin_["yes"], 104), _wrap(bin_["no"], 104)
        hint = _wrap(bin_["hint"], 108)
        block_h = (0.012 + _line_h(11.5) + 0.003 + _line_h(9.2) + 0.008
                   + _text_h(yes, 9.0) + 0.005 + _text_h(no, 9.0) + 0.006
                   + _text_h(hint, 8.6) + 0.010)
        _panel(fig, LEFT, top - block_h, RIGHT - LEFT, block_h)
        x = LEFT + 0.018
        y = top - 0.012
        fig.text(x, y, bin_["name"], fontsize=11.5, color=INK, va="top", fontweight="bold")
        y -= _line_h(11.5) + 0.003
        fig.text(x, y, bin_["purpose"], fontsize=9.2, color=MUTED, va="top")
        y -= _line_h(9.2) + 0.008
        for prefix, colour, text in (("JA", ACCENT, yes), ("NEI", WARN, no)):
            fig.text(x, y, prefix, fontsize=8.2, color=colour, va="top", fontweight="bold")
            fig.text(x + 0.030, y, text, fontsize=9.0, color=INK, va="top", linespacing=1.42)
            y -= _text_h(text, 9.0) + 0.005
        y -= 0.001
        fig.text(x, y, hint, fontsize=8.6, color=MUTED, va="top", style="italic", linespacing=1.42)
        top -= block_h + 0.011

    # The two notes come BEFORE the checklist: they frame it ("this is a proposal, tell us if it does
    # not fit"), and putting them after meant they had to share the bottom of the page with the table,
    # which they simply printed on top of.
    top -= 0.006
    steps = _wrap(NEXT_STEPS, 112)
    policy = _wrap(POLICY_NOTE, 112)
    fig.text(LEFT, top, steps, fontsize=8.8, color=INK, va="top", linespacing=1.45)
    top -= _text_h(steps, 8.8, 1.45) + 0.005
    fig.text(LEFT, top, policy, fontsize=8.2, color=WARN, va="top", linespacing=1.42)
    top -= _text_h(policy, 8.2, 1.42) + 0.016

    _caption(fig, LEFT, top - caption_h + 0.004, "SLIK FORBEREDER DERE PLASSEN")
    top -= caption_h + 0.004
    col_w = (RIGHT - LEFT - 0.028) / 2
    # Two columns, and the row pitch is the TALLER of the two cells in that row -- one four-line
    # explanation next to a two-line one otherwise ran into the row below it.
    wrapped = [(title, _wrap(text, 52)) for title, text in PREP_STEPS]
    step_h = [_line_h(9.6) + 0.003 + _text_h(text, 8.4, 1.40) + 0.010 for _title, text in wrapped]
    y = top
    for row_start in range(0, len(wrapped), 2):
        row = wrapped[row_start:row_start + 2]
        for column, (title, text) in enumerate(row):
            index = row_start + column
            x = LEFT + column * (col_w + 0.028)
            fig.text(x, y, f"{index + 1}", fontsize=11, color=ACCENT, va="top", fontweight="bold")
            fig.text(x + 0.019, y, title, fontsize=9.6, color=INK, va="top", fontweight="bold")
            fig.text(x + 0.019, y - _line_h(9.6) - 0.003, text, fontsize=8.4, color=MUTED,
                     va="top", linespacing=1.40)
        y -= max(step_h[row_start:row_start + 2])
    top = y - 0.006

    # A four-number strip, NOT the fifteen-row table this started as. That table was the only block on
    # the page that repeated page 1 rather than adding to it, and it was also the one that would not fit
    # -- it collided with the footer, then with the notes, then with the checklist, through four
    # attempts at re-flowing it. These four numbers are the ones page 1 does NOT already carry, and a
    # strip one line tall cannot overflow.
    strip_h = 0.052
    strip_top = 0.096 + strip_h
    _caption(fig, LEFT, strip_top + 0.006, "MER FRA SKANNET")
    _draw_kpi_strip(fig, _extra_kpis(scene, info), strip_top, strip_h)

    _footer(fig, scene, 2, 2)
    return fig


def render_page(scene, out_pdf: Path) -> Path:
    """The two-page proposal sheet for one computed Scene.

    Writes <out>.pdf (both pages, searchable text) plus <out>.png and <out>_side2.png, because the
    dashboard shows PNGs and cannot render a PDF inline. Two pages, not one: printed double-sided it
    is still a single sheet to put in an envelope, and page 1 stands alone as "where the bins go" if
    someone only glances at the first thing they see.
    """
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    info = floorplan.plan_info(scene)
    try:
        views = reconstruct3d.render_report_views(scene, out_pdf.parent)
    except Exception:      # noqa: BLE001 - a sheet without the 3D row is still a usable sheet
        views = []

    with plt.rc_context(_rc()):
        page1 = _page_one(scene, info, views)
        page2 = _page_two(scene, info)
        with PdfPages(out_pdf) as pdf:
            pdf.savefig(page1, facecolor="white")
            pdf.savefig(page2, facecolor="white")
        page1.savefig(out_pdf.with_suffix(".png"), facecolor="white")
        page2.savefig(out_pdf.parent / f"{out_pdf.stem}_side2.png", facecolor="white")
        plt.close(page1)
        plt.close(page2)
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
