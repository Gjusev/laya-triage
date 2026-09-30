"""Escalation curve and threshold selection — pure logic, hand-computed expectations.

A record is one evaluated ticket: ``correct`` (intent prediction right) and
``confidence`` (the ticket's gating confidence = min of its routing decisions'
answer_confidence). The curve sweeps thresholds; at each threshold the policy
auto-handles tickets with confidence >= threshold.
"""

import pytest

from laya_triage.escalation import (
    choose_threshold,
    coverage_accuracy_curve,
    evaluate_policy,
)

# 4 tickets: (confidence, correct)
RECORDS = [
    {"confidence": 0.9, "correct": True},
    {"confidence": 0.7, "correct": False},
    {"confidence": 0.5, "correct": True},
    {"confidence": 0.3, "correct": True},
]


def test_curve_points_are_hand_computed():
    curve = coverage_accuracy_curve(RECORDS)

    by_threshold = {round(p.threshold, 2): p for p in curve}

    # threshold 0.0: everything auto-handled, accuracy 3/4
    p = by_threshold[0.0]
    assert (p.coverage, p.accuracy, p.n_handled) == (1.0, 0.75, 4)

    # threshold 0.5: tickets at 0.5, 0.7, 0.9 handled; accuracy 2/3
    p = by_threshold[0.5]
    assert (p.coverage, p.accuracy, p.n_handled) == (0.75, pytest.approx(2 / 3), 3)

    # threshold 0.7: 0.7 and 0.9 handled; accuracy 1/2
    p = by_threshold[0.7]
    assert (p.coverage, p.accuracy, p.n_handled) == (0.5, 0.5, 2)

    # threshold 0.9: only 0.9 handled; accuracy 1/1
    p = by_threshold[0.9]
    assert (p.coverage, p.accuracy, p.n_handled) == (0.25, 1.0, 1)


def test_curve_handles_a_threshold_where_everything_escalates():
    curve = coverage_accuracy_curve([{"confidence": 0.2, "correct": True}])
    # sweeping only the observed confidences (plus 0.0) never empties the handled set
    assert all(p.n_handled >= 1 for p in curve)


def test_choose_threshold_maximizes_coverage_at_target():
    curve = coverage_accuracy_curve(RECORDS)
    # thresholds 0.0 and 0.3 both reach accuracy 0.75 at coverage 1.0;
    # max coverage with ties broken to the lower threshold wins -> 0.0.
    chosen = choose_threshold(curve, target_accuracy=0.75)
    assert chosen.threshold == 0.0
    assert chosen.coverage == 1.0

    # target 1.0: only threshold >= 0.9 reaches it
    chosen = choose_threshold(curve, target_accuracy=1.0)
    assert chosen.threshold == 0.9
    assert chosen.coverage == 0.25


def test_choose_threshold_returns_none_when_target_unreachable():
    curve = coverage_accuracy_curve(
        [{"confidence": 0.9, "correct": False}, {"confidence": 0.8, "correct": False}]
    )
    assert choose_threshold(curve, target_accuracy=0.5) is None


def test_evaluate_policy_reports_coverage_and_accuracy():
    report = evaluate_policy(RECORDS, threshold=0.5)
    assert report["threshold"] == 0.5
    assert report["n_total"] == 4
    assert report["n_escalated"] == 1
    assert report["coverage"] == 0.75
    assert report["accuracy"] == pytest.approx(2 / 3)


def test_empty_records_curve_is_empty():
    assert coverage_accuracy_curve([]) == []
    assert choose_threshold([], target_accuracy=0.5) is None
