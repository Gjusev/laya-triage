"""Render the Phase 2 report tables from the results artifacts.

Reads whatever exists in ``evals/results/`` and writes
``docs/results/phase2.md`` (plus stdout). Missing pieces render as
TODO placeholders so the report never invents a number.

Usage:
    python -m evals.analyze
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "results" / "phase2.md"


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def coverage_table(curve: list[dict], deciles=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)) -> list[str]:
    """The curve compressed to one row per coverage decile (closest observed point)."""
    lines = ["| coverage | accuracy @ coverage | threshold |", "|---|---|---|"]
    for d in deciles:
        best = min(curve, key=lambda p: abs(p["coverage"] - d))
        lines.append(f"| {best['coverage']:.1%} | {_pct(best['accuracy'])} | {best['threshold']:.2f} |")
    return lines


def render() -> str:
    b77 = _load("banking77.json")
    massive = _load("massive.json")
    hand = _load("hand_labeled.json")
    gpt = _load("gpt_baseline.json")

    lines = ["# Phase 2 results", ""]
    lines += [
        "All numbers are measurements from the artifacts in `evals/results/`",
        "(reproduce with `python -m evals.run_all`). Fine-tuned and GPT-4o-mini",
        "columns stay TODO until their runs happen.",
        "",
    ]

    # --- BANKING77 -------------------------------------------------------------
    lines += ["## BANKING77 (test split)", ""]
    if b77:
        s = b77["summary"]
        cfg = b77["config"]
        lines += [
            f"Sample: {cfg['n']} tickets, seed {cfg['seed']} ({cfg['source']}).",
            "",
            "| Metric | direct 77-way | hierarchical | fine-tuned |",
            "|---|---|---|---|",
            f"| Intent accuracy | {_pct(s['direct_accuracy'])} | {_pct(s['hierarchical_accuracy'])} | TODO(Phase 3) |",
            f"| Macro-F1 | {_num(s['direct_macro_f1'])} | {_num(s['hierarchical_macro_f1'])} | TODO(Phase 3) |",
            f"| Seconds (sample) | {s['direct_seconds']} | {s['hierarchical_seconds']} | TODO(Phase 3) |",
            "",
            f"Coarse (cluster) accuracy: {_pct(s['coarse_accuracy'])}.",
            "",
        ]
        esc = b77["escalation"]
        lines += [
            "### Escalation curve (hierarchical, gating = min routing confidence)",
            "",
            *coverage_table(esc["curve"]),
            "",
            f"Operational threshold at target accuracy {esc['target_accuracy']:.0%}: "
            + (
                f"**{esc['chosen_threshold']:.2f}** (coverage {_pct(esc['chosen_coverage'])}, accuracy {_pct(esc['chosen_accuracy'])})"
                if esc["chosen_threshold"] is not None
                else "target unreachable on this sample"
            ),
            "",
            f"Provisional 0.6 policy: coverage {_pct(esc['provisional_0.6']['coverage'])}, "
            f"accuracy {_pct(esc['provisional_0.6']['accuracy'])}, "
            f"escalates {esc['provisional_0.6']['n_escalated']}/{esc['provisional_0.6']['n_total']}.",
            "",
        ]
    else:
        lines += ["TODO(run): `python -m evals.banking77`", ""]

    # --- MASSIVE ---------------------------------------------------------------
    lines += ["## Multilingual robustness (MASSIVE, cluster-only pass)", ""]
    if massive:
        cons = massive["consistency"]
        lines += [
            "MASSIVE has no banking intents, so this is behavior, not accuracy:",
            "mean coarse confidence and escalation rate per language, plus whether",
            "the same parallel utterance routes to the same cluster everywhere.",
            "",
            "| Locale | mean cluster confidence | escalation @ provisional |",
            "|---|---|---|",
        ]
        for locale, stats in massive["per_locale"].items():
            lines.append(
                f"| {locale} | {stats['mean_cluster_confidence']:.3f} | {_pct(stats['escalation_rate_at_provisional'])} |"
            )
        lines += [
            "",
            f"Parallel consistency over {cons['n_parallel']} utterances in {len(cons['locales'])} languages: "
            f"unanimous cluster {_pct(cons['unanimous_agreement'])}, mean pairwise {_pct(cons['mean_pairwise_agreement'])}.",
            "",
        ]
    else:
        lines += ["TODO(run): `python -m evals.massive_multilingual`", ""]

    # --- hand-labeled ------------------------------------------------------------
    lines += ["## Hand-labeled 200 (urgency / frustration)", ""]
    if hand:
        sig = hand["signals"]
        lines += [
            "Two blind annotation passes per `data/annotation_guide.md`, disagreements adjudicated.",
            "",
            "| Signal | MAE | kappa (strict) | kappa (weighted) |",
            "|---|---|---|---|",
            f"| urgency | {_num(sig['urgency_mae'])} | {_num(sig['inter_pass_kappa_urgency'])} | {_num(sig['inter_pass_kappa_urgency_weighted'])} |",
            f"| frustration | {_num(sig['frustration_mae'])} | {_num(sig['inter_pass_kappa_frustration'])} | {_num(sig['inter_pass_kappa_frustration_weighted'])} |",
            "",
            "| Language | n | urgency MAE | frustration MAE | coarse accuracy |",
            "|---|---|---|---|---|",
        ]
        for lang, stats in hand["per_language"].items():
            lines.append(
                f"| {lang} | {stats['n']} | {_num(stats['urgency_mae'])} | {_num(stats['frustration_mae'])} | {_pct(stats['coarse_accuracy'])} |"
            )
        lines.append("")
    else:
        lines += ["TODO(run): `python -m evals.hand_labeled`", ""]

    # --- GPT baseline -------------------------------------------------------------
    lines += ["## GPT-4o-mini baseline", ""]
    if gpt:
        s = gpt["summary"]
        lines += [
            f"Model {gpt['config']['model']} on the same sample ({gpt['config']['n']} tickets, seed {gpt['config']['seed']}).",
            "",
            f"- direct accuracy: {_pct(s['direct_accuracy'])}",
            f"- hierarchical accuracy: {_pct(s['hierarchical_accuracy'])}",
            f"- cost per 1,000 tickets: ${s['cost_per_1000_tickets_usd']:.2f}",
            "",
        ]
    else:
        lines += ["TODO(run): `python -m evals.gpt_baseline` (needs OPENAI_API_KEY)", ""]

    lines += ["## Latency and cost (laya)", "", "TODO(measure): p50/p95 latency per decision and $ per 1,000 tickets on the deployment target.", ""]

    report = "\n".join(lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {REPORT_PATH}")
    return report


if __name__ == "__main__":
    render()
