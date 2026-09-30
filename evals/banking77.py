"""BANKING77 eval: flat 77-way choice vs the hierarchical pipeline (vs fine-tuned, Phase 3).

Both configurations answer the same sampled test tickets on the same routed
checkpoints; per-ticket records (intent, confidence, correctness) are written
to ``evals/results/banking77.json`` so the escalation curve and every table
are reproducible from the artifact without re-running inference.

Usage:
    python -m evals.banking77 [--limit 500] [--seed 13] [--full]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from laya_triage import TriagePipeline, build_router, schema
from laya_triage.escalation import coverage_accuracy_curve, choose_threshold, evaluate_policy

from evals import data

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "banking77.json"
DEFAULT_LIMIT = 500
DEFAULT_SEED = 13
# Accuracy the operational threshold must guarantee before auto-handling.
TARGET_ACCURACY = 0.75


def direct_questions() -> dict:
    """The flat 77-way baseline: one choice over every intent, mechanical descriptions."""
    return {
        "intent": {
            "type": "choice",
            "instructions": "What does the customer need in `message`?",
            "criteria": {
                i: i.replace("_", " ").rstrip("?") for i in schema.INTENT_TO_CLUSTER
            },
        }
    }


def macro_f1(records: list[dict], key: str) -> float:
    """Macro-F1 over the 77 intents for one configuration's predictions."""
    intents = sorted(schema.INTENT_TO_CLUSTER)
    f1s = []
    for intent in intents:
        tp = sum(1 for r in records if r["label"] == intent and r[key] == intent)
        fp = sum(1 for r in records if r["label"] != intent and r[key] == intent)
        fn = sum(1 for r in records if r["label"] == intent and r[key] != intent)
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return sum(f1s) / len(f1s)


def run(limit: int = DEFAULT_LIMIT, seed: int = DEFAULT_SEED) -> dict:
    rows = data.sample_rows(data.load_banking77("test"), limit=limit, seed=seed)
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    print(f"banking77 eval: {len(rows)} tickets (seed {seed})")

    router = build_router()
    pipeline = TriagePipeline(router, min_confidence=0.6)

    # -- direct 77-way baseline ------------------------------------------------
    t0 = time.perf_counter()
    direct = router.predict_batch(
        [{"state": {"message": t}, "questions": direct_questions()} for t in texts]
    )
    direct_seconds = time.perf_counter() - t0

    # -- hierarchical pipeline -------------------------------------------------
    t0 = time.perf_counter()
    hier = pipeline.triage_batch(texts)
    hier_seconds = time.perf_counter() - t0

    records = []
    for i, (row, d, h) in enumerate(zip(rows, direct, hier)):
        records.append(
            {
                "text": row["text"],
                "label": row["label"],
                "direct_intent": d["answers"]["intent"]["choice"],
                "direct_confidence": d["answers"]["intent"]["answer_confidence"],
                "direct_correct": d["answers"]["intent"]["choice"] == row["label"],
                "hier_cluster": h.cluster,
                "hier_intent": h.intent,
                "hier_cluster_confidence": h.confidences["cluster"],
                "hier_intent_confidence": h.confidences["intent"],
                # the pipeline's gating rule: auto-handled iff both routing decisions pass
                "hier_gating_confidence": min(h.confidences["cluster"], h.confidences["intent"]),
                "hier_correct": h.intent == row["label"],
            }
        )

    direct_acc = sum(r["direct_correct"] for r in records) / len(records)
    hier_acc = sum(r["hier_correct"] for r in records) / len(records)
    coarse_acc = sum(
        schema.INTENT_TO_CLUSTER[r["label"]] == r["hier_cluster"] for r in records
    ) / len(records)

    # escalation curve over the hierarchical gating confidences
    curve_records = [
        {"confidence": r["hier_gating_confidence"], "correct": r["hier_correct"]}
        for r in records
    ]
    curve = coverage_accuracy_curve(curve_records)
    chosen = choose_threshold(curve, target_accuracy=TARGET_ACCURACY)

    result = {
        "config": {
            "limit": limit,
            "seed": seed,
            "split": "test",
            "source": "PolyAI-LDN/task-specific-datasets banking_data",
            "n": len(records),
        },
        "summary": {
            "direct_accuracy": direct_acc,
            "direct_macro_f1": macro_f1(records, "direct_intent"),
            "hierarchical_accuracy": hier_acc,
            "hierarchical_macro_f1": macro_f1(records, "hier_intent"),
            "coarse_accuracy": coarse_acc,
            "fine_tuned_accuracy": None,  # TODO(Phase 3): checkpoint not trained yet
            "direct_seconds": round(direct_seconds, 1),
            "hierarchical_seconds": round(hier_seconds, 1),
        },
        "escalation": {
            "target_accuracy": TARGET_ACCURACY,
            "chosen_threshold": chosen.threshold if chosen else None,
            "chosen_coverage": chosen.coverage if chosen else None,
            "chosen_accuracy": chosen.accuracy if chosen else None,
            "provisional_0.6": evaluate_policy(curve_records, threshold=0.6),
            "curve": [
                {"threshold": p.threshold, "coverage": p.coverage, "accuracy": p.accuracy, "n_handled": p.n_handled}
                for p in curve
            ],
        },
        "records": records,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"  direct 77-way: accuracy {direct_acc:.3f} | macro-F1 {result['summary']['direct_macro_f1']:.3f}")
    print(f"  hierarchical:  accuracy {hier_acc:.3f} | macro-F1 {result['summary']['hierarchical_macro_f1']:.3f} | coarse {coarse_acc:.3f}")
    if chosen:
        print(f"  escalation: threshold {chosen.threshold:.2f} -> coverage {chosen.coverage:.2%} at accuracy {chosen.accuracy:.2%}")
    else:
        print(f"  escalation: target accuracy {TARGET_ACCURACY} unreachable on this sample")
    print(f"  wrote {RESULTS_PATH}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--full", action="store_true", help="run the whole test split (3,080 tickets)")
    args = parser.parse_args()
    if args.full:
        args.limit = 10**9
    run(limit=args.limit, seed=args.seed)
