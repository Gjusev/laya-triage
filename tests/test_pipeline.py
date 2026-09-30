"""TriagePipeline behavior through its public seam: triage() with a fake agent."""

import pytest

from laya_triage.pipeline import TriagePipeline
from laya_triage import schema

from conftest import FakeAgent, make_router


ACCEPTANCE_TEXT = "me cobraron dos veces, reembolsen o cancelo"


def scripted_agent(**overrides) -> FakeAgent:
    base = {
        "cluster": {"type": "choice", "choice": "fees_charges_and_refunds", "answer_confidence": 0.95},
        "intent": {"type": "choice", "choice": "request_refund", "answer_confidence": 0.85},
        "urgency": {"type": "score", "score": 2.5, "answer_confidence": 0.9},
        "frustration": {"type": "score", "score": 2.0, "answer_confidence": 0.8},
        "churn_risk": {"type": "noul", "noul": 0.7, "answer_confidence": 0.75},
        "refund_requested": {"type": "noul", "noul": 0.9, "answer_confidence": 0.95},
    }
    base.update(overrides)
    return FakeAgent(scripted=base)


def test_triage_returns_the_full_picture_in_two_passes():
    agent = scripted_agent()
    result = TriagePipeline(make_router(agent), min_confidence=0.6).triage(ACCEPTANCE_TEXT)

    assert result.cluster == "fees_charges_and_refunds"
    assert result.intent == "request_refund"
    assert result.department == "billing_fees"
    assert result.urgency == 2.5
    assert result.frustration == 2.0
    assert result.churn_risk == 0.7
    assert result.refund_requested == 0.9
    assert not result.escalate
    assert result.confidences["cluster"] == 0.95


def test_stage_one_carries_signals_stage_two_is_only_the_intent_choice():
    agent = scripted_agent()
    TriagePipeline(make_router(agent), min_confidence=0.6).triage("any ticket")

    assert len(agent.calls) == 2
    stage1_qids = set(agent.calls[0][2])
    stage2_qids = set(agent.calls[1][2])
    assert stage1_qids == {"cluster", "urgency", "frustration", "churn_risk", "refund_requested"}
    assert stage2_qids == {"intent"}


def test_stage_two_chooses_only_among_the_winning_cluster_intents():
    agent = scripted_agent(
        cluster={"type": "choice", "choice": "lost_stolen_compromised", "answer_confidence": 0.99},
        intent={"type": "choice", "choice": "compromised_card", "answer_confidence": 0.9},
    )
    result = TriagePipeline(make_router(agent), min_confidence=0.6).triage("my card was stolen")

    assert result.department == "fraud_security"
    offered = set(agent.calls[1][2]["intent"]["criteria"])
    assert offered == set(schema.CLUSTERS["lost_stolen_compromised"])


def test_low_confidence_on_any_decision_escalates_with_reasons():
    agent = scripted_agent(
        intent={"type": "choice", "choice": "request_refund", "answer_confidence": 0.42},
    )
    result = TriagePipeline(make_router(agent), min_confidence=0.6).triage(ACCEPTANCE_TEXT)

    assert result.escalate
    assert len(result.escalation_reasons) == 1
    assert "intent" in result.escalation_reasons[0]
    assert "0.42" in result.escalation_reasons[0]


def test_spanish_routes_to_multilingual_english_to_english():
    result_es = TriagePipeline(make_router(scripted_agent()), min_confidence=0.6).triage(ACCEPTANCE_TEXT)
    result_en = TriagePipeline(make_router(scripted_agent()), min_confidence=0.6).triage(
        "I was charged twice, refund me or I cancel"
    )
    assert result_es.model == "multilingual"
    assert result_en.model == "english"
    assert result_es.routing["model"] == "multilingual"


def test_long_flag_scores_every_window_instead_of_the_first():
    agent = scripted_agent()
    TriagePipeline(make_router(agent), min_confidence=0.6).triage(
        "a very long ticket " * 500, long=True
    )
    methods = [c[0] for c in agent.calls]
    assert methods == ["predict_long", "predict_long"]


def test_threshold_boundary_is_confident_not_escalated():
    # exactly at min_confidence -> not escalated (policy is strictly-below)
    agent = scripted_agent(
        intent={"type": "choice", "choice": "request_refund", "answer_confidence": 0.6},
    )
    result = TriagePipeline(make_router(agent), min_confidence=0.6).triage(ACCEPTANCE_TEXT)
    assert not result.escalate
    assert result.escalation_reasons == []


def test_build_router_defaults_to_multilingual_fallback():
    from laya_triage.pipeline import build_router

    router = build_router()  # pure construction, downloads nothing
    decision = router.route({"message": "me cobraron dos veces, reembolsen o cancelo"})
    assert decision["model"] == "multilingual"  # undecided Latin -> safe checkpoint
    decision_en = router.route({"message": "I was charged twice, refund me or I cancel"})
    assert decision_en["model"] == "english"  # confident English stays English


def test_min_confidence_out_of_range_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        TriagePipeline(make_router(FakeAgent()), min_confidence=1.5)


def test_low_signal_confidence_alone_does_not_escalate():
    # Policy covers routing decisions; a low-confidence urgency score stays
    # informational (measured: score questions top out ~0.8, median 0.54).
    agent = scripted_agent(
        urgency={"type": "score", "score": 2.5, "answer_confidence": 0.4},
    )
    result = TriagePipeline(make_router(agent), min_confidence=0.6).triage(ACCEPTANCE_TEXT)
    assert not result.escalate
    assert result.confidences["urgency"] == 0.4  # still recorded, just not gating


def test_low_cluster_confidence_escalates_alone():
    agent = scripted_agent(
        cluster={"type": "choice", "choice": "fees_charges_and_refunds", "answer_confidence": 0.45},
    )
    result = TriagePipeline(make_router(agent), min_confidence=0.6).triage(ACCEPTANCE_TEXT)
    assert result.escalate
    assert len(result.escalation_reasons) == 1
    assert "cluster" in result.escalation_reasons[0]


def test_both_routing_decisions_low_gives_two_reasons():
    agent = scripted_agent(
        cluster={"type": "choice", "choice": "fees_charges_and_refunds", "answer_confidence": 0.45},
        intent={"type": "choice", "choice": "request_refund", "answer_confidence": 0.30},
    )
    result = TriagePipeline(make_router(agent), min_confidence=0.6).triage(ACCEPTANCE_TEXT)
    assert result.escalate
    assert len(result.escalation_reasons) == 2


def test_lang_hint_overrides_detection():
    # An explicit lang hint routes to the multilingual checkpoint even for English text.
    agent = scripted_agent()
    result = TriagePipeline(make_router(agent), min_confidence=0.6).triage(
        "I was charged twice", lang="es"
    )
    assert result.model == "multilingual"


# --- triage_batch ---------------------------------------------------------------


def test_triage_batch_matches_single_triage_results():
    agent = scripted_agent()
    pipeline = TriagePipeline(make_router(agent), min_confidence=0.6)
    texts = ["ticket one", "ticket two", "ticket three"]

    batched = pipeline.triage_batch(texts)
    assert len(batched) == 3
    for text, result in zip(texts, batched):
        single = pipeline.triage(text)
        assert result == single  # batched path must not change the answer


def test_triage_batch_preserves_order_across_clusters():
    agent = scripted_agent(
        cluster={"type": "choice", "choice": "lost_stolen_compromised", "answer_confidence": 0.99},
        intent={"type": "choice", "choice": "compromised_card", "answer_confidence": 0.9},
    )
    # default fake picks the first criterion for unscripted questions, so an
    # unscripted agent answers every ticket the same; use a fake that varies by
    # text to force different clusters per position
    from conftest import FakeAgent, make_router

    class VaryingAgent(FakeAgent):
        def _answer(self, qid, qdef):
            if qid == "cluster":
                label = "lost_stolen_compromised" if "stolen" in self.current_text else "fees_charges_and_refunds"
                return {"type": "choice", "choice": label, "answer_confidence": 0.9}
            return super()._answer(qid, qdef)

        def predict(self, state, questions, lang=None, **kw):
            self.current_text = state["message"]
            return super().predict(state, questions, lang=lang, **kw)

        def predict_batch(self, states, questions, batch_size=None, lang=None, **kw):
            out = []
            for s in states:
                self.current_text = s["message"]
                out.append(super().predict(s, questions, lang=lang, **kw))
            return out

    agent = VaryingAgent(scripted={
        "intent": {"type": "choice", "choice": "request_refund", "answer_confidence": 0.9},
    })
    pipeline = TriagePipeline(make_router(agent), min_confidence=0.6)
    results = pipeline.triage_batch(["a", "card stolen", "b", "c"])
    assert [r.cluster for r in results] == [
        "fees_charges_and_refunds",
        "lost_stolen_compromised",
        "fees_charges_and_refunds",
        "fees_charges_and_refunds",
    ]


def test_triage_batch_uses_batched_calls_not_per_ticket_predicts():
    agent = scripted_agent()
    # FakeAgent records ("predict_batch", ...) only from predict_batch; the
    # batched pipeline must never call per-ticket predict.
    pipeline = TriagePipeline(make_router(agent), min_confidence=0.6)
    pipeline.triage_batch(["x", "y"])
    methods = {c[0] for c in agent.calls}
    assert "predict" not in methods
    assert "predict_long" not in methods
