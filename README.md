# laya-triage

> Multilingual support ticket triage: department, urgency, frustration and churn risk in one forward pass, with confidence-based escalation to humans.

Status: early development. Built on [laya](https://github.com/NandhaKishorM/laya),
the open-source System 1 decision engine (Apache 2.0).

## Why

- Flat classification collapses at scale: a 12-way choice scored 0.40 on real data in the Jev ecosystem. Hierarchical coarse-to-fine routing is the documented fix (54.3% to 60.8% on BANKING77, laya issue #102).
- laya's Router covers 45 of 51 languages automatically, and every answer carries calibrated confidence, so the escalation policy is measurable instead of guessed.

## Design: hierarchical routing (Phase 1)

A single `choice` over the 77 BANKING77 intents is exactly the shape that
collapses (see above), so intent is decided in two stages:

1. **Coarse pass** — one `choice` over 12 clusters **plus all auxiliary signals
   in the same forward pass**: urgency (0–3 score), frustration (0–3 score),
   churn risk (P), refund requested (P).
2. **Fine pass** — one `choice` restricted to the intents of the winning
   cluster (3–10 options, far from the collapse zone).

77 intents → 12 clusters → 8 departments:

| Cluster | Intents | Department |
|---|---|---|
| card_ordering_and_delivery | 6 | cards |
| card_types_and_linking | 9 | cards |
| card_and_pin_malfunctions | 6 | cards |
| card_payment_problems | 6 | payments_cash |
| cash_and_atm | 6 | payments_cash |
| transfers_and_beneficiaries | 9 | transfers |
| top_ups_and_balance_updates | 10 | topups_deposits |
| fees_charges_and_refunds | 8 | billing_fees |
| exchange_rates | 4 | billing_fees |
| identity_and_account_admin | 7 | account_access |
| lost_stolen_compromised | 3 | fraud_security |
| eligibility_and_coverage | 3 | general_service |

BANKING77 is a neobank dataset: it has no loan/credit intents, so a
loans department would be empty by construction and is not modeled.
Intent labels are the exact HuggingFace `PolyAI/banking77` spellings,
including the canonical quirks `Refund_not_showing_up` (label 51) and
`reverted_card_payment?` (label 53).

**Multilingual default.** laya's language detection leaves short unaccented
Latin text (e.g. `me cobraron dos veces, reembolsen o cancelo`) undecided, and
an undecided ticket falls to the Router default. `build_router()` defaults to
the multilingual checkpoint: detection-confident English still routes to the
English checkpoint, everything else gets the checkpoint that does not collapse
off English. Callers who know the language can pass `lang=` to skip detection.

**Escalation policy.** Every answer carries a calibrated `answer_confidence`
(the quantity laya's temperature scaling fits). If either routing decision
(cluster or intent) falls strictly below `min_confidence`, the ticket is
flagged for a human with one reason per low-confidence decision. The policy
deliberately covers routing only: auxiliary signals are score-type questions
whose confidence structurally tops out lower (measured median 0.54 for urgency
over the Phase 1 smoke set), so gating on them would escalate almost every
ticket. Phase 2 replaces the provisional 0.6 threshold with one chosen from a
published coverage/accuracy curve.

**Known limitations (measured on the 20-ticket smoke set, not on BANKING77
itself — Phase 2 measures that properly).** Coarse routing misclassifies a
minority of tickets, sometimes with high confidence (e.g. "please close my
account, I am done with this bank" lands in fees instead of account closure).
The multilingual checkpoint ships an invalid temperature for choices with 11+
options (laya clamps it and warns that confidence from the affected entries is
uncalibrated), so confidence on the 12-option coarse stage in non-English
languages must be treated with care until Phase 2 recalibrates it. On
out-of-domain input (a batch of realistic IT-operations tickets: API outage,
SSO lockout, dark-mode request), 6 of 8 escalate to a human at the provisional
threshold and the 2 auto-handled ones map to the closest banking analog —
the safety property that matters when the hierarchy cannot name the intent.

```python
from laya_triage import TriagePipeline, build_router

pipeline = TriagePipeline(build_router(), min_confidence=0.6)
result = pipeline.triage("me cobraron dos veces, reembolsen o cancelo")
result.department, result.cluster, result.intent   # billing route
result.urgency, result.frustration                  # 0-3 expected levels
result.churn_risk, result.refund_requested          # P(true)
result.escalate, result.escalation_reasons          # human handoff
# long tickets: pipeline.triage(text, long=True) scores every window
```

## Roadmap

- [x] Hierarchical intent routing: coarse clusters to fine intents (BANKING77 mapping)
- [x] Auxiliary signals in the same pass: urgency, frustration, churn risk, refund requested
- [ ] Escalation policy: coverage/accuracy curve over answer confidence, threshold documented
- [ ] Multilingual eval (MASSIVE: es, fr, de, hi, ar) plus a hand-labeled set of 200 tickets published
- [ ] Fine-tuned checkpoint (BANKING77) released on HuggingFace; Streamlit app on HF Spaces

## Development setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or any venv + pip):

```bash
uv venv
uv pip install -e ".[dev]"
pytest                # deterministic unit tests, no checkpoint downloads
pytest -m slow        # opt-in smoke on a real laya checkpoint (downloads on first run)
```

Unit tests mock the laya Agent (`Router.attach` is the seam laya ships for
this) and never download checkpoints; the slow smoke runs 20 Spanish/English
tickets through the real model.

## License

Apache 2.0. See [LICENSE](LICENSE).
