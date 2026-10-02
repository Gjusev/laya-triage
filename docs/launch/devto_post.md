# Multilingual ticket triage with a local decision model: hierarchical beats flat, here are the numbers

> Draft dev.to post. Every number below is a measurement from
> `evals/results/` in the [laya-triage](https://github.com/Gjusev/laya-triage)
> repo; reproduce them with `python -m evals.run_all --limit 200`. The
> fine-tuned numbers are from the published run; only the GPT-4o-mini
> baseline is still pending.

Most triage demos show you a classifier. Almost none show you the two things
that decide whether it can run a support queue: what happens when the model
is unsure, and how far you can trust it in a language other than English.

I built [laya-triage](https://github.com/Gjusev/laya-triage) on top of
[laya](https://github.com/NandhaKishorM/laya), an open-source System 1
decision engine that answers typed questions (choices, 0-3 scores,
true-probabilities) in a single forward pass on CPU: no API calls, no
per-ticket cost. One ticket in, one ticket out: department, intent,
urgency, frustration, churn risk, refund request, plus a calibrated
confidence on every decision.

## Flat choices collapse; hierarchies do not

A flat 77-way intent choice over BANKING77 scores **36.5%** in my eval.
The documented failure mode is real: too many options, each option starved
of meaning. The same model, asked to first pick one of 12 coarse clusters
and then one of (at most) 10 intents inside that cluster, scores **51.0%**.
That is a **+14.5 point** gain from structure alone, with zero extra training.

Since both numbers come from n = 200, the honest question is whether that
gap is real at this size. All configurations scored the **same tickets**
(drawn with seed 13 from the official **test** split of 3,080; the
10,003-ticket train split is only what the fine-tune fitted on), so the
comparison is paired: among the 73 tickets the two approaches disagree on,
the hierarchy wins 51 and loses 22 — exact McNemar **p = 0.0009**. The
fine-tuned comparisons are more lopsided still (81–2 and 109–1).

## The escalation curve nobody publishes

Here is the actual trade-off, measured: if you auto-handle only tickets
where the model is confident and escalate the rest to a human, you buy
accuracy with coverage:

| auto-handle | accuracy among auto-handled |
|---|---|
| 100% | 51.0% |
| 80% | 60.0% |
| 60% | 75.8% |
| 40% | 81.2% |

The operational threshold this picks is 0.84: auto-handle ~60% of tickets
at ~75% accuracy, escalate the rest with a reason string attached. A number
you can argue with beats a vibe.

## Multilingual, honestly measured

Non-English banking tickets degrade: coarse accuracy drops from 71.7%
(English) to 36-55% (German/French/Spanish/Hindi) and 20% (Arabic) on a
200-ticket hand-labeled set. That is the single clearest argument for the
fine-tune — which is now run and published (English side): hierarchical
accuracy jumps from 51.0% to **90.5%** (macro-F1 0.848) with the signal
heads untouched, and the multilingual gap is the documented follow-up.
Meanwhile, on out-of-domain input in five languages, the escalation policy
fires in 69-79% of cases per language. No language silently gets nonsense
answers.

## Try it

- Repo with all evals and artifacts: [laya-triage](https://github.com/Gjusev/laya-triage)
- App (single ticket / CSV batch / metrics): [laya-triage on Streamlit](https://laya-triage-8spuhg8fa5qjy8hiteomux.streamlit.app/)
- Fine-tuned BANKING77 checkpoint: [Gjusev/laya-triage-banking77](https://huggingface.co/Gjusev/laya-triage-banking77)

The whole stack (model, evals, escalation curve, hand-labeled dataset,
fine-tune notebook) is Apache 2.0.
