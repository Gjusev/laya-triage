"""Escalation policy evaluation: coverage/accuracy curve and threshold selection.

Pure logic over evaluated tickets. A record is ``{"confidence": float,
"correct": bool}`` where ``confidence`` is the ticket's gating confidence (the
minimum of its routing decisions' ``answer_confidence`` — the same quantity
``TriagePipeline`` escalates on) and ``correct`` is whether the intent
prediction matched the label.

The curve answers the operational question the plan asks: at which confidence
threshold do we auto-handle tickets, and what accuracy do we buy for the
coverage we keep. Threshold selection: among curve points whose accuracy meets
the target, take maximum coverage (ties -> lower threshold).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class CurvePoint:
    threshold: float
    coverage: float  # fraction of tickets auto-handled (confidence >= threshold)
    accuracy: float  # intent accuracy among auto-handled tickets
    n_handled: int


def coverage_accuracy_curve(
    records: Iterable[dict],
    thresholds: Optional[list[float]] = None,
) -> list[CurvePoint]:
    """Sweep thresholds; report coverage and accuracy at each.

    Default thresholds: 0.0 plus every observed confidence, sorted ascending —
    each such threshold keeps at least one ticket, so accuracy stays defined.
    """
    records = list(records)
    if not records:
        return []
    if thresholds is None:
        thresholds = sorted({0.0, *(r["confidence"] for r in records)})

    n_total = len(records)
    points = []
    for t in thresholds:
        handled = [r for r in records if r["confidence"] >= t]
        if not handled:
            continue
        n_correct = sum(1 for r in handled if r["correct"])
        points.append(
            CurvePoint(
                threshold=t,
                coverage=len(handled) / n_total,
                accuracy=n_correct / len(handled),
                n_handled=len(handled),
            )
        )
    return points


def choose_threshold(curve: list[CurvePoint], target_accuracy: float) -> Optional[CurvePoint]:
    """The operational threshold: max coverage among points meeting the target.

    Returns None when no threshold reaches the target accuracy — the honest
    answer is then "this configuration cannot support that target", not a
    quietly broken threshold.
    """
    eligible = [p for p in curve if p.accuracy >= target_accuracy]
    if not eligible:
        return None
    return max(eligible, key=lambda p: (p.coverage, -p.threshold))


def evaluate_policy(records: Iterable[dict], threshold: float) -> dict:
    """Coverage/accuracy of one fixed threshold (report helper)."""
    records = list(records)
    handled = [r for r in records if r["confidence"] >= threshold]
    n_correct = sum(1 for r in handled if r["correct"])
    return {
        "threshold": threshold,
        "n_total": len(records),
        "n_handled": len(handled),
        "n_escalated": len(records) - len(handled),
        "coverage": len(handled) / len(records) if records else 0.0,
        "accuracy": n_correct / len(handled) if handled else 0.0,
    }
