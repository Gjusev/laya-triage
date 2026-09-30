# Phase 2 results

All numbers are measurements from the artifacts in `evals/results/`
(reproduce with `python -m evals.run_all`). Fine-tuned and GPT-4o-mini
columns stay TODO until their runs happen.

## BANKING77 (test split)

Sample: 200 tickets, seed 13 (PolyAI-LDN/task-specific-datasets banking_data).

| Metric | direct 77-way | hierarchical | fine-tuned |
|---|---|---|---|
| Intent accuracy | 36.5% | 51.0% | TODO(Phase 3) |
| Macro-F1 | 0.284 | 0.443 | TODO(Phase 3) |
| Seconds (sample) | 182.7 | 577.4 | TODO(Phase 3) |

Coarse (cluster) accuracy: 68.0%.

### Escalation curve (hierarchical, gating = min routing confidence)

| coverage | accuracy @ coverage | threshold |
|---|---|---|
| 100.0% | 51.0% | 0.000 |
| 90.0% | 54.4% | 0.441 |
| 80.0% | 60.0% | 0.597 |
| 70.0% | 67.1% | 0.761 |
| 60.0% | 75.8% | 0.851 |
| 50.0% | 78.0% | 0.915 |
| 40.0% | 81.2% | 0.972 |
| 30.0% | 85.0% | 0.996 |
| 20.0% | 90.0% | 0.999 |
| 10.0% | 100.0% | 1.000 |

Operational threshold at target accuracy 75%: **0.84** (coverage 60.5%, accuracy 75.2%)

Provisional 0.6 policy: coverage 79.5%, accuracy 60.4%, escalates 41/200.

Margin note: this point auto-handles 121 tickets with 91 correct; one fewer correct ticket puts it at 74.4%, below the 75% target. Treat the threshold as approximate at this sample size.

## Multilingual robustness (MASSIVE, cluster-only pass)

MASSIVE has no banking intents, so this is behavior, not accuracy:
mean coarse confidence and escalation rate per language, plus whether
the same parallel utterance routes to the same cluster everywhere.

| Locale | mean cluster confidence | escalation @ 0.6 | escalation @ 0.84 |
|---|---|---|---|
| es-ES | 0.563 | 55.0% | 79.0% |
| fr-FR | 0.557 | 56.5% | 75.0% |
| de-DE | 0.543 | 63.5% | 74.5% |
| hi-IN | 0.636 | 45.5% | 68.5% |
| ar-SA | 0.604 | 50.5% | 74.5% |

Parallel consistency over 200 utterances in 5 languages: unanimous cluster 7.0%, mean pairwise 33.5%.

## Hand-labeled 200 (urgency / frustration)

AI-authored, AI-annotated set: two blind annotation passes per
`data/annotation_guide.md`, disagreements adjudicated (see the guide's
provenance section). Inter-pass kappa is computed between the two blind
passes before adjudication.

| Signal | MAE | inter-pass kappa (strict) | inter-pass kappa (weighted) |
|---|---|---|---|
| urgency | 0.813 | 0.781 | 0.908 |
| frustration | 1.071 | 0.792 | 0.893 |

| Language | n | urgency MAE | frustration MAE | coarse accuracy |
|---|---|---|---|---|
| en | 60 | 0.733 | 0.674 | 71.7% |
| es | 60 | 0.883 | 1.297 | 55.0% |
| fr | 25 | 0.823 | 1.222 | 52.0% |
| de | 25 | 0.795 | 1.193 | 36.0% |
| hi | 15 | 0.867 | 1.251 | 53.3% |
| ar | 15 | 0.806 | 1.116 | 20.0% |

## GPT-4o-mini baseline

TODO(run): `python -m evals.gpt_baseline` (needs OPENAI_API_KEY)

## Latency and cost (laya)

TODO(measure): p50/p95 latency per decision and $ per 1,000 tickets on the deployment target.
