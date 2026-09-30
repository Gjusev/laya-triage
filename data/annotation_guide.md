# Annotation guide: urgency and frustration (v1)

This guide governs every label in `hand_labeled_200.jsonl`. Two independent
annotation passes follow it; agreement between the passes is reported as
Cohen's kappa, disagreements are adjudicated against this guide. Annotators
judge **only what the text says** — never invent history ("they must have
written before"), never infer from imagined context.

## Urgency (0–3): how fast the system must react

The customer's need for a *timely* response, evidenced in the text.

| Level | Name | Operational definition |
|---|---|---|
| 0 | none | No time element at all: general questions, curiosity, feature requests, "sometime". |
| 1 | some | Mild time preference ("soon", "these days") or a live inconvenience the customer works around without loss. |
| 2 | high | An explicit deadline, money stuck or blocked *now*, a core action impossible *today*, or "as soon as possible" said plainly. |
| 3 | drop-everything | Active loss in progress (fraud, double charge happening, card compromised), security exposure, or explicit now-markers ("right now", "immediately", "today or I ..."). |

Rules:
- Urgency is not frustration: a perfectly calm ticket can be a 3.
- A threat ("cancelo", "I will leave") without a time marker is **not** urgency by itself; with a deadline ("by Friday", "today") it is at least 2.
- Torn between two levels → pick the lower one.

## Frustration (0–3): the tone of the text

| Level | Name | Operational definition |
|---|---|---|
| 0 | calm | Polite, factual, or positive tone; purely informational. |
| 1 | concerned | Worried or mildly negative wording, still civil ("I'm a bit worried", "esto me preocupa"). |
| 2 | annoyed | Clear complaints: "ridiculous", "again", "third time", sarcasm, pointed criticism of the service. |
| 3 | angry | Strong language: profanity (mild or heavy), shouting (caps/exclamations), threats to leave, legal or regulator threats, insults. |

Rules:
- Judge the words on the page, not the situation's objective severity.
- One caps word for emphasis ("WHY") is 2; a sentence of caps or repeated exclamations is 3.
- Torn between two levels → pick the lower one.

## Language

Tickets are written in natural, native-sounding customer voice for their
language: en, es, fr, de, hi, ar. Do not translate word-for-word; idiomatic
anger and urgency markers differ per language (e.g. Spanish "llevo tres días
esperando" is a 2 urgency; German "sofort" is a 3 marker).

## Provenance

Tickets in this dataset were AI-authored to exercise the taxonomy above, then
labeled in two independent passes by separate AI annotators following this
guide (pass 2 was blind to pass 1). The dataset stores both passes
(`pass1_urgency`/`pass1_frustration`, `pass2_urgency`/`pass2_frustration`)
plus the final adjudicated label (`urgency`/`frustration`); the reported
inter-pass kappa is computed between the two blind passes, before
adjudication. Disagreements were settled by a third pass against this guide
(`adjudicated: true` rows).

Known structural properties, stated so nobody over-reads the numbers:

- **Block structure**: the 60 es tickets intentionally mirror the 60 en
  tickets cluster-for-cluster (same scenario walk), which enables direct
  cross-language comparison of the model's signals. The two blocks are
  paraphrases, not word-for-word translations. The other languages cover
  rotating subsets of the clusters.
- **Thin extreme cells**: levels 3 (and partly 2) are rare by design (the
  target mix mirrors real support queues, where most tickets are calm).
  Per-language MAE at the top of either scale rests on a handful of rows.
- The labels are a training/eval aid for this project, not a human-gold
  benchmark — a human-annotated replacement would strengthen the signal.
  TODO(human): replace or extend with human annotation.
