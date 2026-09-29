"""Mapping invariants for the BANKING77 hierarchy shipped in laya_triage.schema.

The hierarchy is the core design decision of this project: flat choices over many
options collapse (documented in the laya ecosystem: a 12-way choice scored 0.40 on
real data, and issue #102 reports 54.3% -> 60.8% on BANKING77 with a two-stage
shortlist). These tests pin the properties that keep the hierarchy sound.
"""

from laya_triage import schema

# Largest fine choice we allow before the documented choice-collapse risk returns.
MAX_FINE_OPTIONS = 10  # 11+ would land in laya's uncalibrated choice:11+ temperature bucket


def test_mapping_covers_all_77_intents_exactly_once():
    all_intents = [i for intents in schema.CLUSTERS.values() for i in intents]
    assert len(all_intents) == 77
    assert len(set(all_intents)) == 77  # no intent in two clusters


def test_every_fine_choice_stays_under_the_collapse_limit():
    for cluster, intents in schema.CLUSTERS.items():
        assert 2 <= len(intents) <= MAX_FINE_OPTIONS, cluster


def test_eight_departments_cover_every_cluster():
    assert len(set(schema.CLUSTER_DEPARTMENT.values())) == 8
    assert set(schema.CLUSTER_DEPARTMENT) == set(schema.CLUSTERS)


def test_canonical_banking77_label_quirks_are_preserved():
    # Exact HF PolyAI/banking77 spellings; downstream label mappings depend on them.
    assert "Refund_not_showing_up" in schema.INTENT_TO_CLUSTER  # capital R (label 51)
    assert "reverted_card_payment?" in schema.INTENT_TO_CLUSTER  # trailing ? (label 53)


def test_every_cluster_has_a_description_for_the_coarse_choice():
    assert set(schema.CLUSTER_DESCRIPTIONS) == set(schema.CLUSTERS)
    assert all(schema.CLUSTER_DESCRIPTIONS[c].strip() for c in schema.CLUSTERS)


# --- question builders ---------------------------------------------------------


def test_questions_pass_laya_own_validator():
    from laya import Agent  # class import only; no checkpoint is constructed

    for qid, qdef in schema.coarse_questions().items():
        Agent._check_question(qid, qdef)
    for cluster in schema.CLUSTERS:
        for qid, qdef in schema.fine_questions(cluster).items():
            Agent._check_question(qid, qdef)


def test_coarse_questions_are_cluster_choice_plus_signals():
    questions = schema.coarse_questions()
    assert set(questions) == {
        "cluster",
        "urgency",
        "frustration",
        "churn_risk",
        "refund_requested",
    }
    assert set(questions["cluster"]["criteria"]) == set(schema.CLUSTERS)
    assert questions["urgency"]["type"] == "score"
    assert len(questions["urgency"]["criteria"]) == 4  # levels 0-3
    assert questions["frustration"]["type"] == "score"
    assert questions["churn_risk"]["type"] == "noul"
    assert questions["refund_requested"]["type"] == "noul"


def test_fine_questions_scope_the_choice_to_one_cluster():
    questions = schema.fine_questions("fees_charges_and_refunds")
    assert set(questions) == {"intent"}
    assert set(questions["intent"]["criteria"]) == set(
        schema.CLUSTERS["fees_charges_and_refunds"]
    )
    import pytest

    with pytest.raises(KeyError):
        schema.fine_questions("no_such_cluster")
