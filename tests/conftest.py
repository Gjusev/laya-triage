"""Shared deterministic fakes for laya — unit tests never download checkpoints.

The fake is attached to a real ``laya.Router`` via ``Router.attach`` (the seam
laya ships for exactly this), so tests exercise real routing logic (script
detection, language hints) with a scripted agent behind it.
"""

from __future__ import annotations

from laya import Router


class FakeAgent:
    """Deterministic stand-in for a laya Agent.

    ``scripted`` maps question id -> answer dict with just the fields the
    pipeline reads (``choice``/``score``/``noul`` + ``answer_confidence``).
    Unscripted questions get a type-derived default answer at ``confidence``.
    Every call is recorded as ``(method, state, questions)`` for orchestration
    assertions.
    """

    def __init__(self, scripted: dict | None = None, confidence: float = 0.9):
        self.scripted = scripted or {}
        self.confidence = confidence
        self.calls: list[tuple[str, dict, dict]] = []

    def _answer(self, qid: str, qdef: dict) -> dict:
        if qid in self.scripted:
            return dict(self.scripted[qid])
        if qdef["type"] == "choice":
            label = next(iter(qdef["criteria"]))
            return {"type": "choice", "choice": label, "answer_confidence": self.confidence}
        if qdef["type"] == "score":
            return {"type": "score", "score": 1.0, "answer_confidence": self.confidence}
        return {"type": "noul", "noul": 0.2, "answer_confidence": self.confidence}

    def _payload(self, state: dict, questions: dict) -> dict:
        return {
            "model": "fake",
            "answers": {q: self._answer(q, qdef) for q, qdef in questions.items()},
            "usage": {"input_tokens": 1, "output_tokens": 0, "windows": 1},
        }

    def predict(self, state, questions, lang=None, **kwargs):
        self.calls.append(("predict", state, dict(questions)))
        return self._payload(state, questions)

    system_one = predict  # Router.predict calls agent.system_one

    def predict_long(self, state, questions, lang=None, **kwargs):
        self.calls.append(("predict_long", state, dict(questions)))
        return self._payload(state, questions)


def make_router(agent: FakeAgent, **router_kwargs) -> Router:
    """A real Router with the fake attached to every checkpoint laya can route to.

    Uses the same defaults as laya_triage.pipeline.build_router (multilingual
    fallback), so routing assertions match production behavior.
    """
    router_kwargs.setdefault("default", "multilingual")
    router = Router(**router_kwargs)
    router.attach("english", agent)
    router.attach("multilingual", agent)
    return router
