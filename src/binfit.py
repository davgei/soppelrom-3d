"""Rule-based fusion scorer: is a 3D candidate a bin, and which type?

Splices measured 3D size (vs known REG priors) with 2D-detection confidence, multi-view
count and (optionally) color. Grounded in our annotated data: a generic size gate removes
~90% of noise without losing real bins, and size fused with appearance makes the final call.
Size priors are known to the centimetre, so this hand-set scorer beats a model trained on the
handful of scans we have; graduate to a learned model only at ~30-60 scans.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .annotations import BIN_TYPES

# Stage-1 generic "could this be a bin at all" gate (rejected 29/32 noise, kept 15/15 real bins)
GATE_MIN_HEIGHT = 0.90
GATE_MIN_WIDTH = 0.55        # min horizontal side
GATE_MAX_LENGTH = 2.10       # max horizontal side
GATE_MAX_ASPECT = 3.2        # length / width
GATE_MIN_VOLUME = 0.30
VETO_MAX_HEIGHT = 1.85

# Size-fit tolerance bands (relative error): full credit within TOL, 0 at 2*TOL
TOL_FOOTPRINT = 0.20
TOL_HEIGHT = 0.30            # height is the noisiest measured dim (floor estimate + top clip)

# Fusion weights (size dominates because priors are exact and measurement is metric)
W_SIZE, W_CONF, W_VIEWS, W_COLOR = 0.45, 0.25, 0.15, 0.15

ACCEPT_SCORE = 0.65
MIN_SIZE_TERM_ACCEPT = 0.45
TYPE_MARGIN = 0.15           # winner must beat runner-up by this to assert the type

# Hard-reject bounds: only CLEAR non-bins are dropped; borderline goes to human review, because
# a false proposal costs one keypress but a dropped real bin must be redrawn from scratch.
REJECT_MIN_WIDTH = 0.20      # thinner = wall/edge sliver
REJECT_MAX_ASPECT = 4.0      # longer+thinner = sliver
REJECT_MIN_HEIGHT = 0.50     # shorter = floor clutter / debris
REJECT_MIN_VOLUME = 0.12     # smaller = point-cloud speck
REJECT_MAX_LENGTH = 2.60     # longer = wall slab (a merged bin PAIR ~2.4 m is still kept for review)
MIN_TYPE_FIT = 0.35         # below this, call it "annet"
MOLOK_SQUARENESS = 0.8       # molok is circular -> footprint must be near-square

SCORE_TYPES = ["2-hjuls dunk", "4-hjuls container", "molok"]


@dataclass
class BinVerdict:
    keep: bool            # passes gate and is at least review-worthy
    auto_accept: bool     # confident enough to pre-approve
    bin_type: str
    score: float
    size_term: float
    conf_term: float
    views_term: float
    color_term: float
    type_fit: float
    type_confident: bool
    reason: str


def _footprint_and_height(size_lhw: np.ndarray | list) -> tuple[float, float, float]:
    length, height, width = float(size_lhw[0]), float(size_lhw[1]), float(size_lhw[2])
    long_side, short_side = max(length, width), min(length, width)
    return long_side, short_side, height


def passes_gate(size_lhw) -> bool:
    long_side, short_side, height = _footprint_and_height(size_lhw)
    volume = size_lhw[0] * size_lhw[1] * size_lhw[2]
    aspect = long_side / max(short_side, 1e-6)
    return (
        height >= GATE_MIN_HEIGHT
        and short_side >= GATE_MIN_WIDTH
        and long_side <= GATE_MAX_LENGTH
        and aspect <= GATE_MAX_ASPECT
        and volume >= GATE_MIN_VOLUME
    )


def _band(rel_error: float, tol: float) -> float:
    return float(np.clip(1.0 - max(0.0, rel_error - tol) / tol, 0.0, 1.0))


def size_fit(size_lhw, prior_lhw) -> float:
    """Orientation-agnostic size match in [0,1]: footprint sides sorted (yaw-invariant),
    height compared separately with a looser band. The weakest dimension governs."""
    m_long, m_short, m_height = _footprint_and_height(size_lhw)
    p_long, p_short, p_height = _footprint_and_height(prior_lhw)
    scores = [
        _band(abs(m_long - p_long) / p_long, TOL_FOOTPRINT),
        _band(abs(m_short - p_short) / p_short, TOL_FOOTPRINT),
        _band(abs(m_height - p_height) / p_height, TOL_HEIGHT),
    ]
    return float(min(scores))


def _type_fits(size_lhw) -> dict[str, float]:
    long_side, short_side, _ = _footprint_and_height(size_lhw)
    squareness = short_side / max(long_side, 1e-6)
    fits: dict[str, float] = {}
    for name in SCORE_TYPES:
        fit = size_fit(size_lhw, BIN_TYPES[name])
        if name == "molok" and squareness < MOLOK_SQUARENESS:
            fit = 0.0  # circular footprint must be near-square
        fits[name] = fit
    return fits


def score_candidate(
    size_lhw,
    mean_confidence: float,
    n_views: int,
    color_term: float | None = None,
) -> BinVerdict:
    fits = _type_fits(size_lhw)
    ordered = sorted(fits.items(), key=lambda kv: kv[1], reverse=True)
    best_type, best_fit = ordered[0]
    second_fit = ordered[1][1] if len(ordered) > 1 else 0.0

    size_term = best_fit
    conf_term = float(np.clip(mean_confidence / 0.5, 0.0, 1.0))
    views_term = float(np.clip((n_views - 1) / 2.0, 0.0, 1.0))

    if color_term is None:
        total = W_SIZE + W_CONF + W_VIEWS
        score = (W_SIZE * size_term + W_CONF * conf_term + W_VIEWS * views_term) / total
        color_out = float("nan")
    else:
        score = (
            W_SIZE * size_term + W_CONF * conf_term
            + W_VIEWS * views_term + W_COLOR * color_term
        )
        color_out = color_term

    type_confident = (best_fit - second_fit) >= TYPE_MARGIN and best_fit >= MIN_TYPE_FIT
    bin_type = best_type if best_fit >= MIN_TYPE_FIT else "annet"

    long_side, short_side, height = _footprint_and_height(size_lhw)
    aspect = long_side / max(short_side, 1e-6)
    volume = float(size_lhw[0] * size_lhw[1] * size_lhw[2])

    if short_side < REJECT_MIN_WIDTH or aspect > REJECT_MAX_ASPECT:
        keep, reason = False, f"sliver ({long_side:.2f}x{short_side:.2f} m, forhold {aspect:.1f})"
    elif height < REJECT_MIN_HEIGHT:
        keep, reason = False, f"for lav ({height:.2f} m)"
    elif height > VETO_MAX_HEIGHT:
        keep, reason = False, f"for hoy ({height:.2f} m) - trolig struktur"
    elif volume < REJECT_MIN_VOLUME:
        keep, reason = False, f"for liten ({volume:.2f} m3)"
    elif long_side > REJECT_MAX_LENGTH:
        keep, reason = False, f"for stor ({long_side:.2f} m) - trolig vegg"
    else:
        keep = True
        reason = "auto-godkjent" if score >= ACCEPT_SCORE else "til gjennomgang"

    auto_accept = (
        keep and passes_gate(size_lhw) and score >= ACCEPT_SCORE
        and size_term >= MIN_SIZE_TERM_ACCEPT and views_term > 0
    )
    return BinVerdict(
        keep=keep,
        auto_accept=auto_accept,
        bin_type=bin_type,
        score=round(score, 3),
        size_term=round(size_term, 3),
        conf_term=round(conf_term, 3),
        views_term=round(views_term, 3),
        color_term=color_out,
        type_fit=round(best_fit, 3),
        type_confident=type_confident,
        reason=reason,
    )
