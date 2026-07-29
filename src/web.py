"""Browser view of the waste rooms: look at a room, move the bins, propose new ones.

Runs locally (python -m src.web) and serves the SAME data the desktop app works from. It exists so
people who will never install open3d can see what a room looks like, try a layout and hand it back.

What it deliberately does NOT do:

  * It never annotates. Annotation is where ground truth comes from and it needs the 3D view; this is
    a plan view for proposing furniture, not for labelling what is already there.
  * It never writes to outputs/annotations or outputs/entrances. A layout from the browser lands in
    outputs/web_proposals/<stem>.json and stays a SUGGESTION until it is approved in the dashboard.
    Nothing here can approve anything.
  * It never recomputes a scene. compute_scene() reads the point cloud and re-runs placement -- too
    slow for a request and it drags open3d in. The browser is served the plan.json + masks.png that
    analyze_and_render already wrote, so a scan must have been generated once before it shows up.

Where the authority sits. The browser does instant checks while you drag (is this corner on scanned
floor, does it overlap another bin) by looking up the mask bitmap it already has. The question that
actually decides whether a layout works -- can every bin still be wheeled in from the entrance --
is a graph search, and it runs HERE, in Python, on the same grid, through the same
placement.route_corridor the proposals themselves came from. There is no second implementation of the
placement rules in JavaScript to drift out of step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_file
from PIL import Image

from . import pipeline, placement, web_export
from .annotations import BIN_TYPES
from .paths import PROJECT_ROOT

# Proposals from the browser. Deliberately a sibling of annotations/ and NOT inside it: the approval
# rule is that only the person running the desktop app promotes anything, so the two must never be
# mistakable for each other by a glob.
PROPOSAL_DIR = PROJECT_ROOT / "outputs" / "web_proposals"

STATIC_DIR = Path(__file__).resolve().parent / "webui"


@dataclass
class Grid:
    """The parts of a FreeSpaceResult that placement.route_corridor actually reads.

    route_corridor touches only cell, origin, free.shape and floor_observed, so the whole corridor
    check runs off masks.png with no point cloud and no open3d import.
    """
    cell: float
    origin: np.ndarray
    free: np.ndarray
    floor_observed: np.ndarray
    occupied: np.ndarray
    accessible: np.ndarray
    corridor: np.ndarray


def _plan_path(stem: str) -> Path:
    return pipeline.preview_dir(stem) / web_export.PLAN_NAME


def _load_plan(stem: str) -> dict:
    path = _plan_path(stem)
    if not path.exists():
        abort(404, f"{stem} har ingen plan ennå — kjør «Generer bilder» for skannet først")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("version") != web_export.PLAN_VERSION:
        abort(409, f"plan.json for {stem} er versjon {plan.get('version')}, "
                   f"koden forventer {web_export.PLAN_VERSION} — generer skannet på nytt")
    return plan


def _load_grid(stem: str, plan: dict) -> Grid:
    path = pipeline.preview_dir(stem) / web_export.MASKS_NAME
    if not path.exists():
        abort(404, f"{stem} mangler {web_export.MASKS_NAME}")
    packed = np.asarray(Image.open(path).convert("L"))
    bits = plan.get("mask_bits") or web_export.MASK_BITS
    grid = plan["grid"]
    return Grid(
        cell=float(grid["cell"]),
        origin=np.asarray(grid["origin"], dtype=float),
        free=(packed & bits["free"]) > 0,
        floor_observed=(packed & bits["floor_observed"]) > 0,
        occupied=(packed & bits["occupied"]) > 0,
        accessible=(packed & bits["accessible"]) > 0,
        corridor=(packed & bits["corridor"]) > 0,
    )


def _as_rect(bin_dict: dict) -> tuple:
    """A submitted bin -> the (centre, size, yaw) tuple placement's mask helpers expect."""
    cx, cz = float(bin_dict["center"][0]), float(bin_dict["center"][1])
    return ((cx, cz), (float(bin_dict["length_m"]), float(bin_dict["width_m"])),
            float(bin_dict["yaw_deg"]))


def _footprint_stats(grid: Grid, bin_dict: dict) -> dict:
    """How much of one bin's footprint sits on scanned floor, and on floor that is otherwise clear."""
    shape = grid.free.shape
    mask = placement._box_mask(_as_rect(bin_dict), grid.origin, grid.cell, shape)
    cells = int(mask.sum())
    if cells == 0:
        return {"cells": 0, "on_scanned_floor": 0.0, "on_free_floor": 0.0, "outside_grid": True}
    return {
        "cells": cells,
        "on_scanned_floor": round(float((mask & grid.floor_observed).sum()) / cells, 3),
        "on_free_floor": round(float((mask & grid.free).sum()) / cells, 3),
        "outside_grid": False,
    }


def _unchanged_existing(plan: dict, bin_dict: dict, tol: float = 0.02) -> bool:
    """Is this submitted bin one of the room's existing bins, still exactly where it was?

    Judged server-side by comparing against the stored plan rather than trusting a "moved" flag from
    the browser, so a client cannot exempt a bin from the rules by mislabelling it.
    """
    if bin_dict.get("source") != "existing":
        return False
    for original in plan.get("bins", []):
        if original.get("source") != "existing":
            continue
        if (abs(original["center"][0] - float(bin_dict["center"][0])) <= tol
                and abs(original["center"][1] - float(bin_dict["center"][1])) <= tol
                and abs(original["length_m"] - float(bin_dict["length_m"])) <= tol
                and abs(original["width_m"] - float(bin_dict["width_m"])) <= tol
                and abs(original["yaw_deg"] - float(bin_dict["yaw_deg"])) <= 1.0):
            return True
    return False


def validate_layout(stem: str, plan: dict, grid: Grid, bins: list[dict],
                    min_on_floor: float = 0.85) -> dict:
    """The authoritative check for a proposed layout.

    Per bin: is its footprint on floor the scanner actually saw, does it overlap another bin, and can
    it still be wheeled in from an entrance. The last one is why this lives in Python -- it is the
    same placement.route_corridor call that produced the automatic proposals, on the same grid, with
    the same passage width, so a layout accepted here is accepted by the same rule the app uses.

    Only bins that can be CHANGED are judged. An existing bin sitting where it has always sat is a
    fact about the room, not a suggestion: it does not have to be wheelable in (it is already in), and
    its footprint reads as 32-67% unscanned because a depth scan never sees through a container.
    Measured on Frydenlundgata 4B, judging them produced three warnings on every one of the five --
    and two of those, the mutual overlaps of 0.09-0.15 m2, are just slop in the hand-drawn boxes.
    They still BLOCK: an unjudged bin is an obstacle like any other, it just is not on trial.

    min_on_floor is a fraction of the footprint, deliberately not "is the centre cell free" -- see the
    trap documented in web_export.
    """
    shape = grid.free.shape
    masks = [placement._box_mask(_as_rect(b), grid.origin, grid.cell, shape) for b in bins]
    judged = [not _unchanged_existing(plan, b) for b in bins]
    verdicts = []
    for index, (bin_dict, mask) in enumerate(zip(bins, masks)):
        stats = _footprint_stats(grid, bin_dict)
        others = np.zeros(shape, dtype=bool)
        for j, other in enumerate(masks):
            if j != index:
                others |= other
        overlap_cells = int((mask & others).sum())
        problems = []
        if judged[index]:
            if stats["outside_grid"]:
                problems.append("utenfor det skannede området")
            elif stats["on_scanned_floor"] < min_on_floor:
                problems.append(f"bare {stats['on_scanned_floor']:.0%} av flaten står på skannet gulv")
            if overlap_cells:
                problems.append(f"overlapper en annen kasse ({overlap_cells * grid.cell**2:.2f} m²)")
        verdicts.append({
            "index": index,
            "type": bin_dict.get("type", ""),
            "source": bin_dict.get("source", "proposed"),
            "judged": judged[index],
            **stats,
            "overlap_m2": round(overlap_cells * grid.cell ** 2, 3),
            "problems": problems,
        })

    # ---- the corridor: can each bin be wheeled in, with the OTHER bins in the way?
    entrances = [(float(x), float(z)) for x, z in plan.get("entrances", [])]
    passage = float(plan.get("passage_width_m") or 0.6)
    rollable_base = grid.free & grid.accessible
    unreachable: list[int] = []
    route_points: list[list[float]] = []
    if not entrances:
        corridor_note = "ingen inngang registrert, så ingenting kan trilles inn"
        unreachable = [i for i in range(len(bins)) if judged[i]]
    else:
        for index, bin_dict in enumerate(bins):
            if not judged[index]:
                continue          # already standing there; nothing to prove about getting it in
            # Every OTHER bin blocks the way; the bin being tested cannot block itself. Testing them
            # one at a time is the point: three bins that each fit alone can still wall each other
            # in, and a single pass with all of them blocking would report that as all three failing.
            blocked = np.zeros(shape, dtype=bool)
            for j, mask in enumerate(masks):
                if j != index:
                    blocked |= mask
            _route, _corridor, region = placement.route_corridor(
                grid, rollable_base & ~blocked, entrances, [tuple(bin_dict["center"])], passage,
            )
            cx, cz = float(bin_dict["center"][0]), float(bin_dict["center"][1])
            col = int(round((cx - grid.origin[0]) / grid.cell))
            row = int(round((cz - grid.origin[1]) / grid.cell))
            near = placement._nearest_true(region, row, col) if region.any() else None
            # "Reachable" must mean the corridor arrives AT the bin, not merely somewhere in the room.
            # Half a metre of slack absorbs the bin's own footprint being cut out of the corridor.
            reach_m = (float(np.hypot(near[0] - row, near[1] - col)) * grid.cell
                       if near is not None else float("inf"))
            if reach_m > 0.5:
                unreachable.append(index)
        n_judged = sum(judged)
        corridor_note = (f"alle {n_judged} nye/flyttede kasser kan trilles inn fra inngangen"
                         if not unreachable
                         else f"{len(unreachable)} av {n_judged} kasser kan ikke trilles inn")

        # One push path for the whole layout, drawn the way the app draws it: entrance, around the
        # bins, to each of them -- so the browser shows the same route the PNGs do.
        all_bins = np.zeros(shape, dtype=bool)
        for mask in masks:
            all_bins |= mask
        route, _corridor, _region = placement.route_corridor(
            grid, rollable_base & ~all_bins, entrances,
            [tuple(b["center"]) for b in bins], passage,
        )
        if route.any():
            rows_, cols_ = np.nonzero(route)
            route_points = [[round(float(grid.origin[0] + (c + 0.5) * grid.cell), 3),
                             round(float(grid.origin[1] + (r + 0.5) * grid.cell), 3)]
                            for r, c in zip(rows_, cols_)]
    for index in unreachable:
        verdicts[index]["problems"].append("kan ikke trilles inn fra inngangen")

    free_after = grid.free.copy()
    for mask in masks:
        free_after &= ~mask
    return {
        "ok": not any(v["problems"] for v in verdicts),
        "bins": verdicts,
        "corridor_note": corridor_note,
        "free_area_m2": round(float(free_after.sum()) * grid.cell ** 2, 1),
        "route": route_points,
    }


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

    @app.get("/")
    def index():
        return send_file(STATIC_DIR / "index.html")

    @app.get("/rom/<stem>")
    def room_page(stem: str):
        return send_file(STATIC_DIR / "plan.html")

    @app.get("/api/scans")
    def api_scans():
        """Every scan with a generated plan, newest work first. Scans that have never been generated
        are listed too, but marked, so it is obvious why they cannot be opened."""
        out = []
        for stem in pipeline.list_scans():
            has_plan = _plan_path(stem).exists()
            out.append({
                "scan": stem,
                "address": pipeline.address_of(stem) or stem,
                "annotated": pipeline.is_annotated(stem),
                "prepared": pipeline.is_prepared(stem),
                "has_plan": has_plan,
                "n_existing": pipeline.existing_bin_count(stem) if pipeline.is_prepared(stem) else 0,
                "has_proposal": (PROPOSAL_DIR / f"{stem}.json").exists(),
            })
        out.sort(key=lambda row: (not row["has_plan"], row["address"].lower()))
        return jsonify({"scans": out, "bin_types": {
            name: {"length_m": spec[0], "height_m": spec[1], "width_m": spec[2]}
            for name, spec in BIN_TYPES.items()}})

    @app.get("/api/plan/<stem>")
    def api_plan(stem: str):
        plan = _load_plan(stem)
        saved = PROPOSAL_DIR / f"{stem}.json"
        if saved.exists():
            try:
                plan["saved_proposal"] = json.loads(saved.read_text(encoding="utf-8"))
            except ValueError:
                pass         # a corrupt draft must not stop the room from opening
        return jsonify(plan)

    @app.get("/api/masks/<stem>")
    def api_masks(stem: str):
        path = pipeline.preview_dir(stem) / web_export.MASKS_NAME
        if not path.exists():
            abort(404)
        return send_file(path, mimetype="image/png")

    @app.post("/api/validate/<stem>")
    def api_validate(stem: str):
        plan = _load_plan(stem)
        bins = (request.get_json(silent=True) or {}).get("bins")
        if not isinstance(bins, list):
            abort(400, "forventet {\"bins\": [...]}")
        return jsonify(validate_layout(stem, plan, _load_grid(stem, plan), bins))

    @app.post("/api/proposal/<stem>")
    def api_proposal(stem: str):
        """Save a layout as a SUGGESTION. Never touches outputs/annotations."""
        plan = _load_plan(stem)
        body = request.get_json(silent=True) or {}
        bins = body.get("bins")
        if not isinstance(bins, list):
            abort(400, "forventet {\"bins\": [...]}")
        grid = _load_grid(stem, plan)
        report = validate_layout(stem, plan, grid, bins)
        PROPOSAL_DIR.mkdir(parents=True, exist_ok=True)
        # Store the VERDICT, not the whole report: the route alone was 311 points, and it plus the
        # per-bin cell counts are recomputed from the layout on every open. These files are human work
        # that travels with the repo, so 15.6 KB of derived data per proposal is not worth carrying.
        summary = {
            "ok": report["ok"],
            "corridor_note": report["corridor_note"],
            "free_area_m2": report["free_area_m2"],
            "problems": {str(b["index"]): b["problems"] for b in report["bins"] if b["problems"]},
        }
        payload = {
            "scan": stem,
            "address": plan.get("address"),
            "by": str(body.get("by") or "").strip()[:80],
            "note": str(body.get("note") or "").strip()[:2000],
            "bins": bins,
            "validation": summary,
            # No timestamp is invented here: the file's own mtime is the truth, and a clock the
            # browser supplied would be the submitter's, not this machine's.
            "status": "forslag",     # never "godkjent" -- only the desktop app may change this
        }
        (PROPOSAL_DIR / f"{stem}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                                                   encoding="utf-8")
        return jsonify({"saved": True, "validation": report})

    return app


def main() -> None:
    import argparse
    import webbrowser

    parser = argparse.ArgumentParser(description="Nettleserversjon av søppelrom-analysen")
    parser.add_argument("--port", type=int, default=5000)
    # Binds to localhost by default on purpose: these are addresses of municipal waste rooms, so
    # reaching the machine from the network has to be a deliberate choice, not the default.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    ready = sum(1 for stem in pipeline.list_scans() if _plan_path(stem).exists())
    total = len(pipeline.list_scans())
    print(f"[web] {ready} av {total} skann har plandata "
          f"({'kjør «Generer bilder» for de andre' if ready < total else 'alle klare'})")
    print(f"[web] http://{args.host}:{args.port}/")
    if not args.no_browser:
        webbrowser.open(f"http://{args.host}:{args.port}/")
    create_app().run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
