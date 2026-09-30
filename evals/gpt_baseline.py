"""GPT-4o-mini baseline on the same BANKING77 sample and the same questions.

Opt-in: requires the ``openai`` package (``uv pip install -e ".[gpt]"``) and
``OPENAI_API_KEY``. Mirrors the two evaluated configurations — flat 77-way
choice and coarse->fine — so accuracy and cost are comparable with the local
laya numbers. Results land in ``evals/results/gpt_baseline.json``.

Usage:
    python -m evals.gpt_baseline [--limit 500] [--seed 13]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from laya_triage import schema

from evals import data
from evals.banking77 import DEFAULT_LIMIT, DEFAULT_SEED, direct_questions

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "gpt_baseline.json"
MODEL = "gpt-4o-mini"
# Pass current pricing explicitly; these defaults are placeholders to override,
# never published measurements. TODO(measure): set from OpenAI's current page when running.
DEFAULT_PRICE_IN = 0.15  # USD per 1M input tokens
DEFAULT_PRICE_OUT = 0.60  # USD per 1M output tokens


def _ask(client, system: str, user: str, options: list[str]) -> tuple[str, object]:
    """One completion constrained to a numbered option list; returns (choice, usage)."""
    listed = "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(options))
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"{user}\n\nAnswer with ONLY the number of the best option (1-{len(options)}):\n{listed}",
            },
        ],
    )
    raw = response.choices[0].message.content.strip()
    try:
        idx = int(raw.split(".")[0].strip()) - 1
        return options[idx], response.usage
    except (ValueError, IndexError):
        return "__parse_error__", response.usage


def run(limit: int = DEFAULT_LIMIT, seed: int = DEFAULT_SEED, price_in: float = DEFAULT_PRICE_IN, price_out: float = DEFAULT_PRICE_OUT) -> dict:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY to run the GPT baseline")
    from openai import OpenAI

    client = OpenAI()
    rows = data.sample_rows(data.load_banking77("test"), limit=limit, seed=seed)
    direct_q = direct_questions()
    direct_options = list(direct_q["intent"]["criteria"])
    system = "You are a bank support ticket intent classifier."

    usage_totals = {"prompt": 0, "completion": 0}
    records = []
    for row in rows:
        # flat 77-way
        choice, usage = _ask(client, system, direct_q["intent"]["instructions"].replace("`message`", "the ticket"), row["text"], direct_options)
        usage_totals["prompt"] += usage.prompt_tokens
        usage_totals["completion"] += usage.completion_tokens
        direct_correct = choice == row["label"]

        # hierarchical: coarse then fine
        cluster, usage = _ask(client, system, schema.coarse_questions()["cluster"]["instructions"].replace("`message`", "the ticket"), row["text"], list(schema.CLUSTERS))
        usage_totals["prompt"] += usage.prompt_tokens
        usage_totals["completion"] += usage.completion_tokens
        intent = "__no_cluster__"
        if cluster in schema.CLUSTERS:
            intent, usage = _ask(client, system, schema.fine_questions(cluster)["intent"]["instructions"].replace("`message`", "the ticket"), row["text"], list(schema.CLUSTERS[cluster]))
            usage_totals["prompt"] += usage.prompt_tokens
            usage_totals["completion"] += usage.completion_tokens

        records.append(
            {
                "label": row["label"],
                "direct_intent": choice,
                "direct_correct": direct_correct,
                "hier_intent": intent,
                "hier_correct": intent == row["label"],
            }
        )

    n = len(records)
    cost_per_ticket = (
        usage_totals["prompt"] * price_in / 1e6 + usage_totals["completion"] * price_out / 1e6
    ) / n
    result = {
        "config": {"model": MODEL, "limit": limit, "seed": seed, "n": n, "price_in_per_m": price_in, "price_out_per_m": price_out},
        "summary": {
            "direct_accuracy": sum(r["direct_correct"] for r in records) / n,
            "hierarchical_accuracy": sum(r["hier_correct"] for r in records) / n,
            "cost_per_1000_tickets_usd": cost_per_ticket * 1000,
            "total_tokens": usage_totals,
        },
        "records": records,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"  direct {result['summary']['direct_accuracy']:.3f} | hier {result['summary']['hierarchical_accuracy']:.3f} | ${result['summary']['cost_per_1000_tickets_usd']:.2f}/1000")
    print(f"  wrote {RESULTS_PATH}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN)
    parser.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT)
    args = parser.parse_args()
    run(limit=args.limit, seed=args.seed, price_in=args.price_in, price_out=args.price_out)
