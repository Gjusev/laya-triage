"""MASSIVE multilingual robustness eval: per-language routing behavior and cross-language consistency.

MASSIVE has no banking intents, so there is no ground-truth accuracy to
compute against our hierarchy — the honest metrics are (a) how each language's
confidence and escalation rate behave, and (b) whether the SAME parallel
utterance (MASSIVE ids are parallel across locales) lands on the same coarse
cluster in every language. Consistency measures cross-lingual robustness
directly: a router that understands es/fr/de/hi/ar like English picks the same
cluster for the same utterance.

Usage:
    python -m evals.massive_multilingual [--limit 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from laya_triage import build_router, schema

from evals import data

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "massive.json"
DEFAULT_LIMIT = 200
PROVISIONAL_THRESHOLD = 0.6  # TODO(measure): replace with the chosen operational threshold


def parallel_consistency(predictions_by_locale: dict[str, dict[int, str]]) -> dict:
    """Cross-language agreement over parallel utterance ids.

    ``predictions_by_locale`` maps locale -> {parallel id -> chosen cluster}.
    Metrics over ids present in every locale: unanimous-agreement rate (all
    locales pick the same cluster) and mean pairwise agreement.
    """
    locales = sorted(predictions_by_locale)
    if len(locales) < 2:
        raise ValueError("consistency needs at least two locales")
    shared = set.intersection(*(set(p) for p in predictions_by_locale.values()))
    if not shared:
        raise ValueError("no parallel ids shared across locales")

    unanimous = 0
    pairwise_scores = []
    for utt_id in shared:
        picks = [predictions_by_locale[loc][utt_id] for loc in locales]
        unanimous += len(set(picks)) == 1
        for i in range(len(picks)):
            for j in range(i + 1, len(picks)):
                pairwise_scores.append(picks[i] == picks[j])

    return {
        "n_parallel": len(shared),
        "locales": locales,
        "unanimous_agreement": unanimous / len(shared),
        "mean_pairwise_agreement": sum(pairwise_scores) / len(pairwise_scores),
    }


def run(limit: int = DEFAULT_LIMIT) -> dict:
    router = build_router()
    # cluster-only pass: the consistency metric measures the routing decision,
    # and a single-question head is the lean variant for a 5-language sweep
    questions = {"cluster": schema.coarse_questions()["cluster"]}

    per_locale = {}
    predictions_by_locale = {}
    for locale in data.MASSIVE_LOCALES:
        rows = data.load_massive(locale, limit=limit)
        outputs = router.predict_batch(
            [{"state": {"message": r["utt"]}, "questions": questions} for r in rows]
        )
        confidences = []
        predictions = {}
        for row, out in zip(rows, outputs):
            answer = out["answers"]["cluster"]
            predictions[row["id"]] = answer["choice"]
            confidences.append(answer["answer_confidence"])
        predictions_by_locale[locale] = predictions
        per_locale[locale] = {
            "n": len(rows),
            "mean_cluster_confidence": sum(confidences) / len(confidences),
            "escalation_rate_at_provisional": sum(c < PROVISIONAL_THRESHOLD for c in confidences)
            / len(confidences),
        }
        print(f"  {locale}: mean confidence {per_locale[locale]['mean_cluster_confidence']:.3f}")

    consistency = parallel_consistency(predictions_by_locale)

    result = {
        "config": {
            "limit": limit,
            "split": "test",
            "source": "qanastek/MASSIVE parquet conversion",
            "provisional_threshold": PROVISIONAL_THRESHOLD,
        },
        "per_locale": per_locale,
        "consistency": consistency,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(
        f"  consistency: unanimous {consistency['unanimous_agreement']:.3f} | "
        f"pairwise {consistency['mean_pairwise_agreement']:.3f} over {consistency['n_parallel']} parallel ids"
    )
    print(f"  wrote {RESULTS_PATH}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    run(limit=parser.parse_args().limit)
