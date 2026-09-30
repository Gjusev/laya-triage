"""Run every Phase 2 eval that can run here, then render the report.

GPT-4o-mini runs only when OPENAI_API_KEY is set. Each step writes its own
artifact under evals/results/, so steps can be re-run individually.

Usage:
    python -m evals.run_all [--limit 500] [--seed 13]
"""

from __future__ import annotations

import argparse
import os

from evals import analyze, banking77, hand_labeled, massive_multilingual


def main(limit: int, seed: int) -> None:
    print("== banking77 ==")
    banking77.run(limit=limit, seed=seed)
    print("== massive_multilingual ==")
    massive_multilingual.run()
    print("== hand_labeled ==")
    hand_labeled.run()
    if os.environ.get("OPENAI_API_KEY"):
        print("== gpt_baseline ==")
        from evals import gpt_baseline

        gpt_baseline.run(limit=limit, seed=seed)
    else:
        print("== gpt_baseline: skipped (no OPENAI_API_KEY) ==")
    print("== analyze ==")
    analyze.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=banking77.DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=banking77.DEFAULT_SEED)
    args = parser.parse_args()
    main(limit=args.limit, seed=args.seed)
