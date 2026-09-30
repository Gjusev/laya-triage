"""BANKING77 eval of the fine-tuned checkpoint (Phase 3).

Runs the same seed-13 test sample and the same hierarchical two-pass as
``evals.banking77`` against the fine-tuned laya checkpoint produced by
``finetune/laya_triage_banking77.ipynb`` (English-only; the two routing
decisions were trained, the signal heads were not). Also re-checks the
signal heads on the English subset of the hand-labeled set - the plan's
stated regression risk. Writes ``evals/results/banking77_finetuned.json``.

Usage:
    python -m evals.banking77_finetuned --checkpoint <dir> [--limit 200] [--seed 13]
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import laya

from laya_triage import schema
from laya_triage.escalation import choose_threshold, coverage_accuracy_curve

from evals import data
from evals.banking77 import DEFAULT_LIMIT, DEFAULT_SEED, TARGET_ACCURACY, macro_f1

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "banking77_finetuned.json"
HAND_LABELED_PATH = Path(__file__).resolve().parent.parent / "data" / "hand_labeled_200.jsonl"


def hierarchical_eval(agent, rows: list[dict]) -> tuple[list[dict], float]:
    """The notebook's two-pass evaluation: coarse batch, then fine per cluster."""
    texts = [r["text"] for r in rows]
    t0 = time.perf_counter()
    stage1 = agent.predict_batch([{"message": t} for t in texts], schema.coarse_questions())
    clusters = [o["answers"]["cluster"]["choice"] for o in stage1]
    by_cluster = collections.defaultdict(list)
    for i, c in enumerate(clusters):
        by_cluster[c].append(i)
    intents: dict[int, tuple[str, float]] = {}
    for c, idxs in by_cluster.items():
        outs = agent.predict_batch([{"message": texts[i]} for i in idxs], schema.fine_questions(c))
        for i, o in zip(idxs, outs):
            intents[i] = (o["answers"]["intent"]["choice"], o["answers"]["intent"]["answer_confidence"])
    seconds = time.perf_counter() - t0

    records = []
    for i, row in enumerate(rows):
        intent, intent_conf = intents[i]
        records.append(
            {
                "text": row["text"],
                "label": row["label"],
                "pred": intent,
                "cluster": clusters[i],
                "cluster_conf": stage1[i]["answers"]["cluster"]["answer_confidence"],
                "intent_conf": intent_conf,
                "gating_conf": min(stage1[i]["answers"]["cluster"]["answer_confidence"], intent_conf),
                "correct": intent == row["label"],
            }
        )
    return records, seconds


def signal_check(agent) -> dict:
    """Urgency/frustration MAE on the English hand-labeled subset (n=60)."""
    rows = [json.loads(l) for l in HAND_LABELED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    en = [r for r in rows if r["lang"] == "en"]
    outs = agent.predict_batch([{"message": r["text"]} for r in en], schema.coarse_questions())
    u = sum(abs(o["answers"]["urgency"]["score"] - r["urgency"]) for o, r in zip(outs, en)) / len(en)
    f = sum(abs(o["answers"]["frustration"]["score"] - r["frustration"]) for o, r in zip(outs, en)) / len(en)
    return {"n": len(en), "urgency_mae": u, "frustration_mae": f}


def run(checkpoint: str, limit: int = DEFAULT_LIMIT, seed: int = DEFAULT_SEED) -> dict:
    rows = data.sample_rows(data.load_banking77("test"), limit=limit, seed=seed)
    agent = laya.Agent(checkpoint)

    records, seconds = hierarchical_eval(agent, rows)
    acc = sum(r["correct"] for r in records) / len(records)
    coarse = sum(schema.INTENT_TO_CLUSTER[r["label"]] == r["cluster"] for r in records) / len(records)
    curve = coverage_accuracy_curve(
        [{"confidence": r["gating_conf"], "correct": r["correct"]} for r in records]
    )
    chosen = choose_threshold(curve, target_accuracy=TARGET_ACCURACY)

    config_path = Path(checkpoint) / "rl_agent_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    result = {
        "config": {
            "checkpoint": str(checkpoint),
            "limit": limit,
            "seed": seed,
            "split": "test",
            "source": "PolyAI-LDN/task-specific-datasets banking_data",
            "n": len(records),
            "fitted_temperatures": config.get("temperature"),
        },
        "summary": {
            "fine_tuned_accuracy": acc,
            "fine_tuned_macro_f1": macro_f1(
                [{"label": r["label"], "fine_tuned_intent": r["pred"]} for r in records],
                "fine_tuned_intent",
            ),
            "coarse_accuracy": coarse,
            "fine_tuned_seconds": round(seconds, 1),
            "signals_en": signal_check(agent),
        },
        "escalation": {
            "target_accuracy": TARGET_ACCURACY,
            "chosen_threshold": chosen.threshold if chosen else None,
            "chosen_coverage": chosen.coverage if chosen else None,
            "chosen_accuracy": chosen.accuracy if chosen else None,
            "curve": [
                {"threshold": p.threshold, "coverage": p.coverage, "accuracy": p.accuracy, "n_handled": p.n_handled}
                for p in curve
            ],
        },
        "records": records,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"  fine-tuned: accuracy {acc:.3f} | macro-F1 {result['summary']['fine_tuned_macro_f1']:.3f} | coarse {coarse:.3f}")
    print(f"  signals (en, n={result['summary']['signals_en']['n']}): urgency MAE {result['summary']['signals_en']['urgency_mae']:.3f}, frustration MAE {result['summary']['signals_en']['frustration_mae']:.3f}")
    print(f"  wrote {RESULTS_PATH}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to the fine-tuned checkpoint directory")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    run(checkpoint=args.checkpoint, limit=args.limit, seed=args.seed)
