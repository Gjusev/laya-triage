"""Coarse-to-fine triage pipeline: one pass for cluster + signals, one for intent.

Escalation policy: every decision comes back with a calibrated
``answer_confidence`` (max probability — the quantity laya's temperature scaling
fits). If any decision falls strictly below ``min_confidence``, the ticket is
flagged for a human with one reason per low-confidence decision. The threshold
itself is an operational choice; Phase 2 picks it from a published
coverage/accuracy curve instead of a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from laya import Router

from laya_triage import schema

# The escalation policy covers routing decisions only. Auxiliary signals are
# score-type questions whose confidence structurally tops out lower (a 4-level
# urgency score rarely exceeds ~0.8; measured median 0.54 over the Phase 1 smoke
# set), so gating on them would escalate nearly every ticket. Phase 2's
# coverage/accuracy curve decides whether signals belong in the policy at all.
ROUTING_DECISIONS = ("cluster", "intent")


@dataclass(frozen=True)
class TriageResult:
    """Everything one triaged ticket produces."""

    text: str
    department: str
    cluster: str
    intent: str
    urgency: float  # 0-3 expected level
    frustration: float  # 0-3 expected level
    churn_risk: float  # P(customer may leave or cancel)
    refund_requested: float  # P(customer asks for money back)
    confidences: dict[str, float]  # question id -> calibrated answer_confidence
    escalate: bool
    escalation_reasons: list[str] = field(default_factory=list)
    model: str = ""  # routed checkpoint name, e.g. "english" / "multilingual"
    routing: dict = field(default_factory=dict)  # laya RouteDecision payload


class TriagePipeline:
    """Two-stage triage over a ``laya.Router``.

    Stage 1 answers the coarse cluster choice and all auxiliary signals in a
    single forward pass; stage 2 answers the intent choice restricted to the
    winning cluster's intents (the documented fix for flat-choice collapse).
    """

    def __init__(self, router, min_confidence: float = 0.6):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self.router = router
        self.min_confidence = min_confidence

    def triage(self, text: str, *, lang: str | None = None, long: bool = False) -> TriageResult:
        """Triage one ticket in any language.

        lang: optional language hint forwarded to the router (e.g. "es").
        long: score every window of a long ticket (``predict_long``) instead of
              letting ``predict`` see only the first one.
        """
        state = {"message": text}

        stage1 = self._predict(state, schema.coarse_questions(), lang=lang, long=long)
        cluster = stage1["answers"]["cluster"]["choice"]
        stage2 = self._predict(state, schema.fine_questions(cluster), lang=lang, long=long)
        return self._assemble(text, stage1, stage2)

    def triage_batch(self, texts, *, lang: str | None = None, batch_size=None) -> list[TriageResult]:
        """Triage many tickets with shared forward passes.

        Answers are identical to triage(); only throughput differs (laya's
        predict_batch groups requests by checkpoint and question schema).
        Long-input scanning (triage's ``long=True``) is not available batched.
        """
        if not texts:
            return []
        stage1 = self.router.predict_batch(
            [{"state": {"message": t}, "questions": schema.coarse_questions(), **({"lang": lang} if lang else {})}
             for t in texts],
            batch_size=batch_size,
        )
        by_cluster = {}
        for i, out in enumerate(stage1):
            by_cluster.setdefault(out["answers"]["cluster"]["choice"], []).append(i)
        stage2_by_index = {}
        for cluster, idxs in by_cluster.items():
            requests = [
                {"state": {"message": texts[i]}, "questions": schema.fine_questions(cluster), **({"lang": lang} if lang else {})}
                for i in idxs
            ]
            for i, out in zip(idxs, self.router.predict_batch(requests, batch_size=batch_size)):
                stage2_by_index[i] = out
        return [self._assemble(texts[i], stage1[i], stage2_by_index[i]) for i in range(len(texts))]

    def _assemble(self, text: str, stage1: dict, stage2: dict) -> TriageResult:
        answers = {**stage1["answers"], **stage2["answers"]}
        cluster = answers["cluster"]["choice"]
        confidences = {
            qid: answers[qid]["answer_confidence"] for qid in ("cluster", "intent", "urgency", "frustration", "churn_risk", "refund_requested")
        }
        reasons = [
            f"{qid} confidence {c:.2f} below {self.min_confidence:.2f}"
            for qid in ROUTING_DECISIONS
            if (c := confidences[qid]) < self.min_confidence
        ]

        return TriageResult(
            text=text,
            department=schema.CLUSTER_DEPARTMENT[cluster],
            cluster=cluster,
            intent=answers["intent"]["choice"],
            urgency=answers["urgency"]["score"],
            frustration=answers["frustration"]["score"],
            churn_risk=answers["churn_risk"]["noul"],
            refund_requested=answers["refund_requested"]["noul"],
            confidences=confidences,
            escalate=bool(reasons),
            escalation_reasons=reasons,
            model=stage1.get("routing", {}).get("model", ""),
            routing=dict(stage1.get("routing", {})),
        )

    def _predict(self, state: dict, questions: dict, *, lang: str | None, long: bool) -> dict:
        if long:
            return self.router.predict_long(state, questions, lang=lang)
        return self.router.predict(state, questions, lang=lang)


def build_router(**router_kwargs) -> Router:
    """Production router: multilingual fallback, detection-confident English stays English.

    laya's language detection leaves short unaccented Latin text undecided, and an
    undecided ticket falls to the Router default. Defaulting to the multilingual
    checkpoint is the safe side of that: the English checkpoint collapses on
    non-English input (0.100 accuracy on Hindi per laya's docs), while English
    detection is a positive decision that still routes to the English checkpoint.
    Construction downloads nothing; the first predict does.
    """
    router_kwargs.setdefault("default", "multilingual")
    return Router(**router_kwargs)
