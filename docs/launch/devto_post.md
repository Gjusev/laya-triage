# Multilingual ticket triage with a local decision model: hierarchical beats flat, here are the numbers

> Draft dev.to post. Every number below is a measurement from
> `evals/results/` in the [laya-triage](https://github.com/Gjusev/laya-triage)
> repo — reproduce them with `python -m evals.run_all --limit 200`. Update the
> fine-tuned and GPT-4o-mini numbers after their runs before publishing.
> TODO(publish): fill in the Space link and the checkpoint link.

Most triage demos show you a classifier. Almost none show you the two things
that decide whether it can run a support queue: what happens when the model
is unsure, and how far you can trust it in a language other than English.

I built [laya-triage](https://github.com/Gjusev/laya-triage) on top of
[laya](https://github.com/NandhaKishorM/laya), an open-source System 1
decision engine that answers typed questions (choices, 0-3 scores,
true-probabilities) in a single forward pass on CPU — no API calls, no
per-ticket cost. One ticket in, one ticket out: department, intent,
urgency, frustration, churn risk, refund request — plus a calibrated
confidence on every decision.

## Flat choices collapse; hierarchies do not

A flat 77-way intent choice over BANKING77 scores **36.5%** in my eval
(200-ticket test sample). The documented failure mode is real: too many
options, each option starved of meaning. The same model, asked to first
pick one of 12 coarse clusters and then one of (at most) 10 intents inside
that cluster, scores **51.0%** — a **+14.5 point** gain from structure
alone, zero extra training.

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
at ~75% accuracy, escalate the rest with a reason string attached. That
number is a policy you can argue with — not a vibe.

## Multilingual, honestly measured

Non-English banking tickets degrade: coarse accuracy drops from 71.7%
(English) to 36-55% (German/French/Spanish/Hindi) and 20% (Arabic) on a
200-ticket hand-labeled set. That is the single clearest argument for the
fine-tune (notebook included in the repo). Meanwhile, on out-of-domain
input in five languages, the escalation policy fires in 69-79% of cases
per language — no language silently gets nonsense answers.

## Try it

- Repo with all evals and artifacts: [laya-triage](https://github.com/Gjusev/laya-triage)
- App (single ticket / CSV batch / metrics): TODO(publish): Space link
- Fine-tuned BANKING77 checkpoint: TODO(publish): checkpoint link

The whole stack — model, evals, escalation curve, hand-labeled dataset,
fine-tune notebook — is Apache 2.0.
