"""Hand-labeled eval: signal MAE, inter-annotation kappa, coarse accuracy per language.

The dataset (``data/hand_labeled_200.jsonl``) carries final urgency/frustration
labels (two blind passes + adjudication, see data/annotation_guide.md) plus the
authored cluster/intent. The model predicts continuous 0-3 scores; MAE is
computed against the integer labels. Coarse accuracy per language uses the
authored cluster as the label.

Usage:
    python -m evals.hand_labeled
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from laya_triage import TriagePipeline, build_router, schema

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "hand_labeled.json"
DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "hand_labeled_200.jsonl"


def cohen_kappa(a: list[int], b: list[int], weighted: bool = False) -> float:
    """(Quadratically weighted) Cohen's kappa between two equal-length label lists.

    Weighted kappa respects the ordinal distance between 0-3 levels; use it
    alongside the strict version because a 0-vs-1 confusion is not a 0-vs-3.
    """
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two non-empty equal-length lists")
    k = max(max(a), max(b)) + 1
    n = len(a)
    observed = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[x][y] += 1
    margin_a = [sum(row) for row in observed]
    margin_b = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def weight(i: int, j: int) -> float:
        if not weighted:
            return 0.0 if i == j else 1.0
        return (i - j) ** 2 / (k - 1) ** 2 if k > 1 else 0.0

    obs_dis = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k))
    exp_dis = sum(
        weight(i, j) * margin_a[i] * margin_b[j] / n for i in range(k) for j in range(k)
    )
    if exp_dis == 0:
        return 1.0  # both raters perfectly consistent with a degenerate distribution
    return 1 - obs_dis / exp_dis


def mae(predictions: list[float], labels: list[int]) -> float:
    """Mean absolute error between continuous predictions and integer labels."""
    if len(predictions) != len(labels) or not labels:
        raise ValueError("mae needs two non-empty equal-length lists")
    return sum(abs(p - l) for p, l in zip(predictions, labels)) / len(labels)


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def run() -> dict:
    rows = load_dataset()
    texts = [r["text"] for r in rows]
    pipeline = TriagePipeline(build_router(), min_confidence=0.6)
    results = pipeline.triage_batch(texts)

    per_lang = defaultdict(lambda: {"n": 0, "urgency_err": 0.0, "frustration_err": 0.0, "cluster_hits": 0})
    for row, result in zip(rows, results):
        stats = per_lang[row["lang"]]
        stats["n"] += 1
        stats["urgency_err"] += abs(result.urgency - row["urgency"])
        stats["frustration_err"] += abs(result.frustration - row["frustration"])
        stats["cluster_hits"] += result.cluster == row["cluster"]

    result = {
        "config": {
            "n": len(rows),
            "languages": sorted(per_lang),
            "dataset": "data/hand_labeled_200.jsonl (two blind AI passes + adjudication, see annotation guide)",
        },
        "signals": {
            "urgency_mae": mae([r.urgency for r in results], [row["urgency"] for row in rows]),
            "frustration_mae": mae([r.frustration for r in results], [row["frustration"] for row in rows]),
            # the promised inter-pass metric: the two blind passes, before adjudication
            "inter_pass_kappa_urgency": cohen_kappa(
                [row["pass1_urgency"] for row in rows], [row["pass2_urgency"] for row in rows]
            ),
            "inter_pass_kappa_urgency_weighted": cohen_kappa(
                [row["pass1_urgency"] for row in rows], [row["pass2_urgency"] for row in rows], weighted=True
            ),
            "inter_pass_kappa_frustration": cohen_kappa(
                [row["pass1_frustration"] for row in rows], [row["pass2_frustration"] for row in rows]
            ),
            "inter_pass_kappa_frustration_weighted": cohen_kappa(
                [row["pass1_frustration"] for row in rows], [row["pass2_frustration"] for row in rows], weighted=True
            ),
            # agreement of the adjudicated final with pass 2 (structurally an
            # upper bound on inter-pass agreement; reported for completeness)
            "final_vs_pass2_kappa_urgency": cohen_kappa(
                [row["urgency"] for row in rows], [row["pass2_urgency"] for row in rows]
            ),
            "final_vs_pass2_kappa_frustration": cohen_kappa(
                [row["frustration"] for row in rows], [row["pass2_frustration"] for row in rows]
            ),
        },
        "records": [
            {
                "id": row["id"],
                "lang": row["lang"],
                "label_cluster": row["cluster"],
                "label_intent": row["intent"],
                "label_urgency": row["urgency"],
                "label_frustration": row["frustration"],
                "predicted_cluster": result.cluster,
                "predicted_intent": result.intent,
                "predicted_urgency": result.urgency,
                "predicted_frustration": result.frustration,
            }
            for row, result in zip(rows, results)
        ],
        "per_language": {
            lang: {
                "n": s["n"],
                "urgency_mae": s["urgency_err"] / s["n"],
                "frustration_mae": s["frustration_err"] / s["n"],
                "coarse_accuracy": s["cluster_hits"] / s["n"],
            }
            for lang, s in per_lang.items()
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"  urgency MAE {result['signals']['urgency_mae']:.3f} | frustration MAE {result['signals']['frustration_mae']:.3f}")
    print(f"  kappa urgency {result['signals']['inter_pass_kappa_urgency']:.3f} (weighted {result['signals']['inter_pass_kappa_urgency_weighted']:.3f})")
    print(f"  kappa frustration {result['signals']['inter_pass_kappa_frustration']:.3f} (weighted {result['signals']['inter_pass_kappa_frustration_weighted']:.3f})")
    print(f"  wrote {RESULTS_PATH}")
    return result


if __name__ == "__main__":
    run()
