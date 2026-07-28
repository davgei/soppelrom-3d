"""Hero numbers: ONE landscape graphic with the totals across every analysed scan.

Why this exists: the per-scan sheets answer "what does THIS room look like", but a presentation also
needs the opening slide that answers "how much have we actually covered". Every figure here is
summed from files the pipeline already wrote (previews/<stem>/stats.json) plus the bin annotations —
nothing is typed in by hand, so the graphic cannot drift away from the analysis it summarises.

Layout is authored in a fixed 1920x1080 design grid and multiplied by `scale`, so the same
composition comes out identically at slide size and at presentation/print size.

    .venv\\Scripts\\python.exe -m src.hero        # writes + prints the PNG paths
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import style
from .annotations import BIN_TYPES, load_annotations
from .paths import ANNOTATION_DIR, CACHE_ROOT, PREVIEW_ROOT

BASE_W, BASE_H = 1920, 1080          # 16:9 design grid; every constant below is in these units
TILE = style.shade(style.PANEL, 1.9)  # a shade lighter than the backdrop: reads as a raised card
TILE_EDGE_ALPHA = 0.13
RULE_ALPHA = 0.12
FOOTNOTE = style.shade(style.MUTED, 0.72)

# One hue (red = existing bin, per the palette) ramped by size, so the split reads as one family
# instead of four unrelated colours.
_TYPE_TINTS = (1.0, 0.82, 0.66, 0.52)

# The grid, in design units. GRID_Y..GRID_Y+GRID_H is the one band that holds every card, so the
# hero card and the 2x2 column always end on the same line.
MARGIN, GUTTER = 84.0, 24.0
GRID_Y, GRID_H = 208.0, 704.0
HERO_W = 800.0
KPI_X = MARGIN + HERO_W + GUTTER
KPI_W = (BASE_W - MARGIN - KPI_X - GUTTER) / 2
KPI_H = (GRID_H - GUTTER) / 2
_KPI_METER = 289.0                   # y of the share meter inside a KPI tile (reserved by all four)


# --------------------------------------------------------------------------- data

@dataclass
class Totals:
    """Everything the sheet prints, all of it summed from real files."""
    scans: int = 0                  # scans with a finished stats.json
    prepared: int = 0               # scans prepared at all (the honest denominator)
    indoor: int = 0
    outdoor: int = 0
    area_m2: float = 0.0
    free_m2: float = 0.0
    candidates: int = 0             # proposed new bin places
    bins: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    verified_rooms: int = 0         # bins drawn/checked by hand
    auto_rooms: int = 0             # bins still only machine-detected
    survey_rooms: int = 0           # sealed room or no entrance found -> needs a site visit
    addresses: int = 0
    postal_codes: int = 0
    updated: datetime | None = None

    @property
    def free_share(self) -> float:
        return self.free_m2 / self.area_m2 if self.area_m2 > 0 else 0.0


def _bin_types(stem: str) -> list[str]:
    """Bin types for one scan, from the same source the pipeline uses for the plans: the hand
    annotation when it exists, otherwise the cached automatic proposals."""
    annotated = ANNOTATION_DIR / f"{stem}.json"
    proposals = CACHE_ROOT / stem / "proposals.json"
    path = annotated if annotated.exists() else (proposals if proposals.exists() else None)
    if path is None:
        return []
    try:
        _, boxes = load_annotations(path)
    except Exception:  # noqa: BLE001 - one malformed file must not kill the whole sheet
        return []
    return [box.bin_type if box.bin_type in BIN_TYPES else "annet" for box in boxes]


def gather() -> Totals:
    totals = Totals(prepared=len(list(CACHE_ROOT.glob("*/done.flag"))))
    stamps: list[float] = []
    postal: set[str] = set()
    addresses: set[str] = set()
    for stats_path in sorted(PREVIEW_ROOT.glob("*/stats.json")):
        stem = stats_path.parent.name
        try:
            st = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        totals.scans += 1
        stamps.append(stats_path.stat().st_mtime)
        if st.get("indoor"):
            totals.indoor += 1
        else:
            totals.outdoor += 1
        totals.area_m2 += float(st.get("area_m2") or 0.0)
        totals.free_m2 += float(st.get("free_area_m2") or 0.0)
        totals.candidates += int(st.get("n_candidates") or 0)
        # a sealed room, or one where no way in was found, cannot be planned from the desk
        if st.get("closed_room") or not int(st.get("n_entrances") or 0):
            totals.survey_rooms += 1
        address = st.get("address")
        if address:
            addresses.add(address)
            match = re.search(r"(\d{4})\s+\D", address)
            if match:
                postal.add(match.group(1))
        types = _bin_types(stem)
        totals.bins += len(types)
        for name in types:
            totals.by_type[name] = totals.by_type.get(name, 0) + 1
        if (ANNOTATION_DIR / f"{stem}.json").exists():
            totals.verified_rooms += 1
        elif types:
            totals.auto_rooms += 1
    totals.addresses = len(addresses)
    totals.postal_codes = len(postal)
    totals.updated = datetime.fromtimestamp(max(stamps)) if stamps else None
    return totals


# --------------------------------------------------------------------------- drawing helpers

class _Sheet:
    """The 16:9 canvas plus the design-unit -> pixel conversion (`u`)."""

    def __init__(self, scale: float):
        self.scale = float(scale)
        self.width, self.height = self.u(BASE_W), self.u(BASE_H)
        self.image = np.empty((self.height, self.width, 3), np.uint8)
        self.image[:] = style.BACKDROP

    def u(self, value: float) -> int:
        return int(round(value * self.scale))


def _text(
    sheet: _Sheet,
    text: str,
    x: float,
    y: float,
    *,
    size: float,
    weight: str = "regular",
    color: style.BGR = style.PANEL_TEXT,
    anchor: str = "lt",
) -> tuple[int, int, int, int]:
    """draw_text in design units, always without a halo (nothing here sits on a photo)."""
    return style.draw_text(sheet.image, text, (sheet.u(x), sheet.u(y)), size=sheet.u(size),
                           weight=weight, color=color, anchor=anchor, halo=0)


def _tile(sheet: _Sheet, x: float, y: float, w: float, h: float) -> None:
    style.rounded_rect(sheet.image, (sheet.u(x), sheet.u(y), sheet.u(x + w), sheet.u(y + h)),
                       color=TILE, alpha=1.0, radius=sheet.u(20), edge_color=style.PANEL_EDGE,
                       edge_alpha=TILE_EDGE_ALPHA, edge_thickness=max(1, sheet.u(1)))


def _bar(sheet: _Sheet, x: float, y: float, w: float, h: float, color: style.BGR,
         alpha: float = 1.0) -> None:
    """Pill-shaped solid bar — used for the accent rule, meters and the split strip."""
    style.rounded_rect(sheet.image, (sheet.u(x), sheet.u(y), sheet.u(x + w), sheet.u(y + h)),
                       color=color, alpha=alpha, radius=sheet.u(h / 2), edge_color=None)


def _rule(sheet: _Sheet, x: float, y: float, w: float) -> None:
    height = max(1, sheet.u(1.5))
    style.blend_mask(sheet.image, sheet.u(x), sheet.u(y), np.full((height, sheet.u(w)), 255, np.uint8),
                     style.PANEL_EDGE, RULE_ALPHA)


def _figure(
    sheet: _Sheet,
    x: float,
    y: float,
    value: str,
    *,
    size: float,
    unit: str | None = None,
    color: style.BGR = style.PANEL_TEXT,
    unit_color: style.BGR | None = None,
) -> int:
    """The big number, with an optional small unit sitting on the SAME baseline.

    PIL anchors a text block by its line box, so aligning the tops would drop the small unit's
    baseline below the digits — the ascent difference is added back explicitly instead.
    """
    bounds = _text(sheet, value, x, y, size=size, weight="bold", color=color)
    if not unit:
        return bounds[2] - bounds[0]
    unit_size = size * 0.32
    ascent = style.get_font(sheet.u(size), "bold").getmetrics()[0]
    unit_ascent = style.get_font(sheet.u(unit_size), "semibold").getmetrics()[0]
    unit_bounds = style.draw_text(
        sheet.image, unit, (bounds[2] + sheet.u(size * 0.10), bounds[1] + ascent - unit_ascent),
        size=sheet.u(unit_size), weight="semibold",
        color=unit_color or style.shade(color, 0.82), halo=0,
    )
    return unit_bounds[2] - bounds[0]


def _kpi(
    sheet: _Sheet,
    box: tuple[float, float, float, float],
    value: str,
    caption: str,
    sub: str,
    *,
    unit: str | None = None,
    color: style.BGR = style.PANEL_TEXT,
) -> tuple[float, float, float]:
    """One secondary tile: figure, caption, supporting line. Every tile reserves the meter row at
    _KPI_METER even when it draws nothing there, so all four figures sit on the same line and the
    padding stays symmetrical. Returns (content x, content width, meter y)."""
    x, y, w, h = box
    _tile(sheet, x, y, w, h)
    pad = 40.0
    _figure(sheet, x + pad, y + 41, value, size=94, unit=unit, color=color)
    _text(sheet, caption, x + pad, y + 192, size=30, weight="semibold")
    _text(sheet, sub, x + pad, y + 238, size=22, color=style.MUTED)
    return x + pad, w - 2 * pad, y + _KPI_METER


# --------------------------------------------------------------------------- the sheet

def _type_rows(totals: Totals) -> list[tuple[str, int, style.BGR]]:
    """Bin types largest first, each with its tint of the existing-bin red."""
    ordered = sorted(totals.by_type.items(), key=lambda item: -item[1])
    rows = []
    for index, (name, count) in enumerate(ordered):
        tint = _TYPE_TINTS[min(index, len(_TYPE_TINTS) - 1)]
        rows.append((name[:1].upper() + name[1:], count, style.shade(style.EXISTING_BIN, tint)))
    return rows


def _header(sheet: _Sheet, totals: Totals) -> None:
    right = BASE_W - MARGIN
    _bar(sheet, MARGIN, 62, 6, 114, style.NEW_BIN)
    _text(sheet, "3D-ANALYSE AV SØPPELROM · OSLO", MARGIN + 28, 62, size=22, weight="semibold",
          color=style.MUTED)
    _text(sheet, "Søppelrom i tall", MARGIN + 28, 96, size=64, weight="bold")
    if totals.updated:
        _text(sheet, f"Datagrunnlag oppdatert {totals.updated:%d.%m.%Y}", right, 74, size=24,
              color=style.MUTED, anchor="rt")
    _text(sheet, f"{totals.addresses} adresser i {totals.postal_codes} postnumre", right, 112,
          size=21, color=FOOTNOTE, anchor="rt")


def _hero_tile(sheet: _Sheet, totals: Totals) -> None:
    """Left card: the two counts the whole project is about — rooms, and the bins found in them."""
    x, y, w, h = MARGIN, GRID_Y, HERO_W, GRID_H
    _tile(sheet, x, y, w, h)
    pad = 48.0
    left = x + pad
    content_w = w - 2 * pad

    _figure(sheet, left, y + 40, style.num(totals.scans, 0), size=188)
    _text(sheet, "søppelrom analysert i 3D", left, y + 284, size=40, weight="semibold")
    _text(sheet, f"{totals.indoor} innendørs · {totals.outdoor} utendørs eller gårdsrom",
          left, y + 345, size=25, color=style.MUTED)

    _rule(sheet, left, y + 420, content_w)

    _figure(sheet, left, y + 452, style.num(totals.bins, 0), size=100)
    _text(sheet, "søppelkasser kartlagt", left, y + 594, size=30, weight="semibold")

    # the type split fills the figure's own line box, so the caption below clears it
    rows = _type_rows(totals)
    row_x, row_h = left + 250, 33.0
    for index, (label, count, color) in enumerate(rows[:4]):
        row_y = y + 453 + index * row_h
        _bar(sheet, row_x, row_y + 10, 18, 11, color)
        _text(sheet, label, row_x + 32, row_y, size=23, color=style.MUTED)
        _text(sheet, style.num(count, 0), left + content_w, row_y, size=23, weight="semibold",
              color=style.PANEL_TEXT, anchor="rt")

    # proportion strip: the same tints, no labels — the rows above already name them
    strip_y, strip_h, gap = y + 650, 14.0, 4.0
    total = max(sum(count for _, count, _ in rows), 1)
    cursor = left
    usable = content_w - gap * max(len(rows) - 1, 0)
    for label, count, color in rows:
        segment = max(usable * count / total, 3.0)
        _bar(sheet, cursor, strip_y, segment, strip_h, color)
        cursor += segment + gap


def _kpi_tiles(sheet: _Sheet, totals: Totals) -> None:
    left_col, right_col = KPI_X, KPI_X + KPI_W + GUTTER
    top_row, bottom_row = GRID_Y, GRID_Y + KPI_H + GUTTER

    # free floor, with a meter for its share of everything measured
    content_x, content_w, meter_y = _kpi(
        sheet, (left_col, top_row, KPI_W, KPI_H), style.num(totals.free_m2, 0),
        "ledig gulv i dag", f"{style.num(totals.free_share * 100, 0)} % av all målt gulvflate",
        unit="m²",
    )
    _bar(sheet, content_x, meter_y, content_w, 10, style.UNKNOWN_FLOOR, alpha=0.5)
    _bar(sheet, content_x, meter_y, max(content_w * totals.free_share, 6.0), 10, style.FREE_FLOOR)

    per_room = totals.candidates / totals.scans if totals.scans else 0.0
    _kpi(sheet, (right_col, top_row, KPI_W, KPI_H), style.num(totals.candidates, 0),
         "nye kasser det er plass til", f"i snitt {style.num(per_room, 1)} per rom",
         color=style.NEW_BIN)

    average = totals.area_m2 / totals.scans if totals.scans else 0.0
    _kpi(sheet, (left_col, bottom_row, KPI_W, KPI_H), style.num(totals.area_m2, 0),
         "gulvflate målt totalt", f"i snitt {style.num(average, 0)} m² per rom", unit="m²")

    # red only when there is actually something to act on
    survey = totals.survey_rooms
    _kpi(sheet, (right_col, bottom_row, KPI_W, KPI_H), style.num(survey, 0),
         "rom som må befares", "innesperret eller uten funnet inngang",
         color=style.EXISTING_BIN if survey else style.PANEL_TEXT)


def _footnote(sheet: _Sheet, totals: Totals) -> None:
    without = max(totals.scans - totals.verified_rooms - totals.auto_rooms, 0)
    coverage = (f"Tallene er summert fra {totals.scans} av {totals.prepared} forberedte skann. "
                f"Kassene er kvalitetssikret manuelt i {totals.verified_rooms} rom; i "
                f"{totals.auto_rooms} rom er de maskinelt gjenkjent og ikke kontrollert ennå")
    coverage += f", og {without} rom har ingen registrerte kasser." if without else "."
    caveat = ("Nye kasseplasser er maskinelle forslag – de må vurderes på befaring og godkjennes "
              "før de tas i bruk. Ledig gulv er skannet, flatt gulv uten kasser eller andre "
              "hindringer.")
    _text(sheet, coverage, MARGIN, 952, size=19, color=FOOTNOTE)
    _text(sheet, caveat, MARGIN, 980, size=19, color=FOOTNOTE)


def render(totals: Totals, scale: float = 2.0) -> np.ndarray:
    sheet = _Sheet(scale)
    style.vignette(sheet.image, strength=0.30, spread=0.35)  # depth on the mat, before any card
    _header(sheet, totals)
    _hero_tile(sheet, totals)
    _kpi_tiles(sheet, totals)
    _footnote(sheet, totals)
    return sheet.image


def build(out_dir: Path | None = None) -> list[Path]:
    """Write the presentation-resolution sheet plus a 1080p copy for slides. Returns both paths."""
    totals = gather()
    out_dir = Path(out_dir) if out_dir else PREVIEW_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    image = render(totals, scale=2.0)          # 3840 x 2160
    primary = out_dir / "_hero.png"
    cv2.imwrite(str(primary), image)
    slide = out_dir / "_hero_1920x1080.png"
    cv2.imwrite(str(slide), style.fit_width(image, BASE_W))
    return [primary, slide]


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
