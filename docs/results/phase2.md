# Phase 2 results

All numbers are measurements from the artifacts in `evals/results/`
(reproduce with `python -m evals.run_all`). The GPT-4o-mini column
stays TODO until that run happens.

## BANKING77 (test split)

Sample: 200 tickets, seed 13, drawn from the official BANKING77 **test** split (PolyAI-LDN/task-specific-datasets banking_data); the 10,003-ticket train split was used only to fit the fine-tuned checkpoint.

| Metric | direct 77-way | hierarchical | fine-tuned |
|---|---|---|---|
| Intent accuracy | 36.5% | 51.0% | 90.5% |
| Macro-F1 | 0.284 | 0.443 | 0.848 |
| Seconds (sample) | 182.7 | 577.4 | 16.1 |

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

### Fine-tuned checkpoint (Phase 3)

Trained by `finetune/laya_triage_banking77.ipynb` on the BANKING77 train split (both routing decisions; the signal heads keep their base weights).

- Coarse accuracy 96.0%; fitted temperatures (choice, score, noul): [3.825, 1.2, 1.2].
- Signal regression check (English hand-labeled subset, n=60): urgency MAE 0.701, frustration MAE 0.693 (zero-shot all-language baseline: 0.813 / 1.071).
- Escalation: the 75% accuracy target is met at threshold **0.00** with 100.0% coverage (90.5% accuracy among auto-handled).

### Pairwise significance (same tickets, exact McNemar)

All configurations scored the identical sample, so the right test is
paired: among the discordant tickets, is the split of wins one-sided?

| Comparison | base wrong / cand right | base right / cand wrong | exact p |
|---|---:|---:|---|
| flat 77-way vs hierarchical (zero-shot) | 51 | 22 | p = 0.00091 |
| hierarchical zero-shot vs fine-tuned | 81 | 2 | p < 0.00001 |
| flat 77-way vs fine-tuned | 109 | 1 | p < 0.00001 |

With n=200 the independent-confidence-interval view is too coarse for the
+14.5 pp hierarchy-vs-flat gap; the paired test above is the decisive one.

### Full test split (n=3,080, replication)

All three configurations over every ticket of the official test split
(kernel `laya-triage-full-split`): flat 37.4%, hierarchical
51.1%, fine-tuned 89.8% (macro-F1 0.897).
The sample numbers above replicate within a point on every row.

- Paired gap flat vs hierarchical: +13.7 pp (95% CI roughly +11.4 to +16.0); discordant tickets 821-399, p < 0.00001.
- Fine-tuned vs hierarchical: 1225-33 discordant.
- Disagreement anatomy: the 821 hierarchy wins spread over 66 gold intents; 318 (39%) had flat's wrong answer inside the correct cluster (disambiguation), the rest were cross-cluster misroutes the coarse step recovered. The largest single confusion family is top-up variants collapsed by flat into `top_up_by_card_charge`.

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
