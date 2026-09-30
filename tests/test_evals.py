"""Unit tests for the pure logic in the eval package (no downloads, no checkpoints)."""

import json

import pytest

from evals import data
from evals.banking77 import macro_f1
from evals.massive_multilingual import parallel_consistency


# --- data.sample_rows -----------------------------------------------------------


def test_sample_is_deterministic_and_within_bounds():
    rows = [{"i": i} for i in range(100)]
    a = data.sample_rows(rows, limit=10, seed=13)
    b = data.sample_rows(rows, limit=10, seed=13)
    assert a == b
    assert len(a) == 10
    assert all(r in rows for r in a)


def test_sample_returns_everything_when_limit_exceeds_size():
    rows = [{"i": 1}, {"i": 2}]
    assert data.sample_rows(rows, limit=50, seed=1) == rows


# --- banking77.macro_f1 ---------------------------------------------------------


@pytest.fixture
def tiny_vocabulary(monkeypatch):
    """macro_f1 reads its label vocabulary from the schema; shrink it for arithmetic tests."""
    monkeypatch.setattr("laya_triage.schema.INTENT_TO_CLUSTER", {"a": "x", "b": "x"})


def test_macro_f1_mixed_predictions(tiny_vocabulary):
    # a: tp=1 fp=0 fn=1 -> f1 = 2/(2+0+1) = 2/3 ; b: tp=1 fp=1 fn=0 -> f1 = 2/(2+1+0) = 2/3
    records = [
        {"label": "a", "pred": "a"},
        {"label": "a", "pred": "b"},
        {"label": "b", "pred": "b"},
    ]
    assert macro_f1(records, "pred") == pytest.approx(2 / 3)


def test_macro_f1_all_wrong_is_zero(tiny_vocabulary):
    assert macro_f1([{"label": "a", "pred": "b"}, {"label": "b", "pred": "a"}], "pred") == 0.0


def test_macro_f1_unpredicted_label_counts_as_zero(tiny_vocabulary):
    # label b never predicted and never in a record's pred: f1(b)=0 pulls the macro down
    assert macro_f1([{"label": "a", "pred": "a"}], "pred") == 0.5


# --- massive parallel_consistency ------------------------------------------------


def test_parallel_consistency_unanimous_and_partial():
    preds = {
        "es": {1: "cards", 2: "transfers", 3: "fees"},
        "fr": {1: "cards", 2: "transfers", 3: "cards"},
        "de": {1: "cards", 2: "cards", 3: "fees"},
    }
    stats = parallel_consistency(preds)
    assert stats["n_parallel"] == 3
    # unanimous only on id 1 -> 1/3
    assert stats["unanimous_agreement"] == 1 / 3
    # pairwise agreements: 5 of 9 comparisons (id1: all 3 pairs; id2: es-fr; id3: es-de)
    assert stats["mean_pairwise_agreement"] == pytest.approx(5 / 9)


def test_parallel_consistency_ignores_non_shared_ids():
    preds = {"es": {1: "cards"}, "fr": {1: "cards", 99: "fees"}}
    stats = parallel_consistency(preds)
    assert stats["n_parallel"] == 1
    assert stats["unanimous_agreement"] == 1.0


def test_parallel_consistency_requires_two_locales():
    with pytest.raises(ValueError):
        parallel_consistency({"es": {1: "cards"}})


# --- hand_labeled kappa / mae -----------------------------------------------------


def test_cohen_kappa_perfect_agreement_is_one():
    from evals.hand_labeled import cohen_kappa

    assert cohen_kappa([0, 1, 2, 3, 1], [0, 1, 2, 3, 1]) == 1.0
    assert cohen_kappa([0, 1, 2, 3, 1], [0, 1, 2, 3, 1], weighted=True) == 1.0


def test_cohen_kappa_hand_computed_example():
    from evals.hand_labeled import cohen_kappa

    # observed agreement 7/10; marginals a=(5,5), b=(4,6) -> expected 0.5
    # kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4
    a = [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
    b = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    import pytest

    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_weighted_punishes_far_confusions_less_near_than_far():
    from evals.hand_labeled import cohen_kappa

    near = cohen_kappa([0, 1, 2, 3], [1, 2, 3, 2], weighted=True)
    far = cohen_kappa([0, 1, 2, 3], [3, 2, 1, 0], weighted=True)
    # same number of disagreements, but distant ones hurt more
    assert far < near


def test_cohen_kappa_degenerate_distribution_is_one():
    from evals.hand_labeled import cohen_kappa

    # both raters always say the same level -> expected disagreement 0
    assert cohen_kappa([2, 2, 2], [2, 2, 2]) == 1.0


def test_mae_mixed_floats_and_ints():
    from evals.hand_labeled import mae

    assert mae([1.5, 0.0, 2.5], [1, 0, 3]) == (0.5 + 0.0 + 0.5) / 3


# --- analyze.render ---------------------------------------------------------------


def test_render_builds_report_from_artifacts(tmp_path, monkeypatch):
    import evals.analyze as analyze

    results = tmp_path / "results"
    results.mkdir()
    (results / "banking77.json").write_text(
        json.dumps(
            {
                "config": {"n": 2, "seed": 13, "source": "test"},
                "summary": {
                    "direct_accuracy": 0.5, "direct_macro_f1": 0.4,
                    "hierarchical_accuracy": 0.75, "hierarchical_macro_f1": 0.7,
                    "coarse_accuracy": 0.9, "direct_seconds": 1.0, "hierarchical_seconds": 2.0,
                },
                "escalation": {
                    "target_accuracy": 0.75,
                    "chosen_threshold": 0.5, "chosen_coverage": 0.5, "chosen_accuracy": 0.8,
                    "provisional_0.6": {"coverage": 0.4, "accuracy": 0.8, "n_escalated": 1, "n_total": 2},
                    "curve": [
                        {"threshold": 0.0, "coverage": 1.0, "accuracy": 0.5, "n_handled": 2},
                        {"threshold": 0.5, "coverage": 0.5, "accuracy": 0.8, "n_handled": 1},
                    ],
                },
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "phase2.md"
    monkeypatch.setattr(analyze, "RESULTS_DIR", results)
    monkeypatch.setattr(analyze, "REPORT_PATH", report_path)

    report = analyze.render()

    assert "| Intent accuracy | 50.0% | 75.0% | TODO(Phase 3) |" in report
    assert "**0.50** (coverage 50.0%" in report  # chosen threshold line
    assert "TODO(run): `python -m evals.massive_multilingual`" in report
    assert "TODO(run): `python -m evals.hand_labeled`" in report
    assert "TODO(run): `python -m evals.gpt_baseline`" in report
    assert "### Fine-tuned checkpoint" not in report
    assert report_path.exists()


def test_render_fills_fine_tuned_column_from_artifact(tmp_path, monkeypatch):
    import evals.analyze as analyze

    results = tmp_path / "results"
    results.mkdir()
    (results / "banking77.json").write_text(
        json.dumps(
            {
                "config": {"n": 2, "seed": 13, "source": "test"},
                "summary": {
                    "direct_accuracy": 0.5, "direct_macro_f1": 0.4,
                    "hierarchical_accuracy": 0.75, "hierarchical_macro_f1": 0.7,
                    "coarse_accuracy": 0.9, "direct_seconds": 1.0, "hierarchical_seconds": 2.0,
                },
                "escalation": {
                    "target_accuracy": 0.75,
                    "chosen_threshold": 0.5, "chosen_coverage": 0.5, "chosen_accuracy": 0.8,
                    "provisional_0.6": {"coverage": 0.4, "accuracy": 0.8, "n_escalated": 1, "n_total": 2},
                    "curve": [
                        {"threshold": 0.0, "coverage": 1.0, "accuracy": 0.5, "n_handled": 2},
                        {"threshold": 0.5, "coverage": 0.5, "accuracy": 0.8, "n_handled": 1},
                    ],
                },
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    (results / "banking77_finetuned.json").write_text(
        json.dumps(
            {
                "config": {"n": 2, "seed": 13, "source": "test", "fitted_temperatures": [3.8, 1.2, 1.2]},
                "summary": {
                    "fine_tuned_accuracy": 0.9, "fine_tuned_macro_f1": 0.85,
                    "coarse_accuracy": 0.95, "fine_tuned_seconds": 42.0,
                    "signals_en": {"n": 60, "urgency_mae": 0.7, "frustration_mae": 0.69},
                },
                "escalation": {
                    "target_accuracy": 0.75,
                    "chosen_threshold": 0.0, "chosen_coverage": 1.0, "chosen_accuracy": 0.9,
                    "curve": [{"threshold": 0.0, "coverage": 1.0, "accuracy": 0.9, "n_handled": 2}],
                },
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "phase2.md"
    monkeypatch.setattr(analyze, "RESULTS_DIR", results)
    monkeypatch.setattr(analyze, "REPORT_PATH", report_path)

    report = analyze.render()

    assert "| Intent accuracy | 50.0% | 75.0% | 90.0% |" in report
    assert "| Macro-F1 | 0.400 | 0.700 | 0.850 |" in report
    assert "| Seconds (sample) | 1.0 | 2.0 | 42.0 |" in report
    assert "TODO(Phase 3)" not in report
    assert "### Fine-tuned checkpoint (Phase 3)" in report
    assert "[3.8, 1.2, 1.2]" in report
