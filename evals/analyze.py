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
        lines.append(f"| {best['coverage']:.1%} | {_pct(best['accuracy'])} | {best['threshold']:.3f} |")
    return lines


def _ft_cells(ft: dict | None) -> tuple[str, str, str]:
    """The fine-tuned column: measured values, or a TODO when no artifact exists."""
    if not ft:
        return "TODO(Phase 3)", "TODO(Phase 3)", "TODO(Phase 3)"
    s = ft["summary"]
    return (
        _pct(s["fine_tuned_accuracy"]),
        _num(s["fine_tuned_macro_f1"]),
        str(s["fine_tuned_seconds"]),
    )


def render() -> str:
    b77 = _load("banking77.json")
    ft = _load("banking77_finetuned.json")
    massive = _load("massive.json")
    hand = _load("hand_labeled.json")
    gpt = _load("gpt_baseline.json")

    lines = ["# Phase 2 results", ""]
    lines += [
        "All numbers are measurements from the artifacts in `evals/results/`",
        "(reproduce with `python -m evals.run_all`). The GPT-4o-mini column",
        "stays TODO until that run happens.",
        "",
    ]

    # --- BANKING77 -------------------------------------------------------------
    lines += ["## BANKING77 (test split)", ""]
    if b77:
        s = b77["summary"]
        cfg = b77["config"]
        acc_c, f1_c, sec_c = _ft_cells(ft)
        lines += [
            f"Sample: {cfg['n']} tickets, seed {cfg['seed']} ({cfg['source']}).",
            "",
            "| Metric | direct 77-way | hierarchical | fine-tuned |",
            "|---|---|---|---|",
            f"| Intent accuracy | {_pct(s['direct_accuracy'])} | {_pct(s['hierarchical_accuracy'])} | {acc_c} |",
            f"| Macro-F1 | {_num(s['direct_macro_f1'])} | {_num(s['hierarchical_macro_f1'])} | {f1_c} |",
            f"| Seconds (sample) | {s['direct_seconds']} | {s['hierarchical_seconds']} | {sec_c} |",
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
        if esc.get("chosen_threshold") is not None:
            chosen_point = min(esc["curve"], key=lambda q: abs(q["threshold"] - esc["chosen_threshold"]))
            k = round(chosen_point["accuracy"] * chosen_point["n_handled"])
            acc_minus_one = (k - 1) / chosen_point["n_handled"]
            lines += [
                f"Margin note: this point auto-handles {chosen_point['n_handled']} tickets with {k} correct; "
                f"one fewer correct ticket puts it at {acc_minus_one:.1%}, below the "
                f"{esc['target_accuracy']:.0%} target. Treat the threshold as approximate at this sample size.",
                "",
            ]
    else:
        lines += ["TODO(run): `python -m evals.banking77`", ""]

    if ft:
        fs = ft["summary"]
        fesc = ft["escalation"]
        sig_base = (
            f" (zero-shot all-language baseline: {_num(hand['signals']['urgency_mae'])} / "
            f"{_num(hand['signals']['frustration_mae'])})"
            if hand
            else ""
        )
        lines += [
            "### Fine-tuned checkpoint (Phase 3)",
            "",
            f"Trained by `finetune/laya_triage_banking77.ipynb` on the BANKING77 train split "
            f"(both routing decisions; the signal heads keep their base weights).",
            "",
            f"- Coarse accuracy {_pct(fs['coarse_accuracy'])}; fitted temperatures (choice, score, noul): "
            f"{ft['config'].get('fitted_temperatures')}.",
            f"- Signal regression check (English hand-labeled subset, n={fs['signals_en']['n']}): "
            f"urgency MAE {_num(fs['signals_en']['urgency_mae'])}, "
            f"frustration MAE {_num(fs['signals_en']['frustration_mae'])}{sig_base}.",
            "- Escalation: "
            + (
                f"the {fesc['target_accuracy']:.0%} accuracy target is met at threshold "
                f"**{fesc['chosen_threshold']:.2f}** with {_pct(fesc['chosen_coverage'])} coverage "
                f"({_pct(fesc['chosen_accuracy'])} accuracy among auto-handled)."
                if fesc.get("chosen_threshold") is not None
                else "the accuracy target is not reachable at any threshold on this sample."
            ),
            "",
        ]

    # --- MASSIVE ---------------------------------------------------------------
    lines += ["## Multilingual robustness (MASSIVE, cluster-only pass)", ""]
    if massive:
        cons = massive["consistency"]
        lines += [
            "MASSIVE has no banking intents, so this is behavior, not accuracy:",
            "mean coarse confidence and escalation rate per language, plus whether",
            "the same parallel utterance routes to the same cluster everywhere.",
            "",
            "| Locale | mean cluster confidence | escalation @ 0.6 | escalation @ 0.84 |",
            "|---|---|---|---|",
        ]
        for locale, stats in massive["per_locale"].items():
            lines.append(
                f"| {locale} | {stats['mean_cluster_confidence']:.3f} | "
                f"{_pct(stats['escalation_rate_at_provisional'])} | {_pct(stats.get('escalation_rate_at_measured'))} |"
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
            "AI-authored, AI-annotated set: two blind annotation passes per",
            "`data/annotation_guide.md`, disagreements adjudicated (see the guide's",
            "provenance section). Inter-pass kappa is computed between the two blind",
            "passes before adjudication.",
            "",
            "| Signal | MAE | inter-pass kappa (strict) | inter-pass kappa (weighted) |",
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
