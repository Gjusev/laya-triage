# LinkedIn post draft

> Attach: screenshot of the app's metrics view (escalation curve) or the
> batch view with escalated tickets highlighted. English per repo rules.
> TODO(publish): fill links after the Space and checkpoint are up.

At what confidence do I let a model close a support ticket without a human?

Every triage demo answers "which department?" Almost none answer the
question that actually gates automation: "when should it back off?"

I built a multilingual ticket triage system on laya (an open-source,
local System 1 decision engine, one forward pass on CPU, no API cost) and
published the answer as a curve, not a slogan:

- Flat 77-way intent classification: 36.5% accuracy.
- Same model, hierarchical (12 clusters -> intents): 51.0%.
- Auto-handle only what the model is confident about (threshold 0.84):
  75.2% accuracy on the 60.5% of tickets it keeps. Everything else is
  escalated to a human with the reason attached.

Department, urgency, frustration, churn risk and refund detection all come
out of the same single pass, in any of the 45 languages laya's router covers, with calibrated
confidence per decision.

The repo ships everything: the evals, the escalation curve, a 200-ticket
multilingual labeled dataset (two blind annotation passes, kappa 0.78+),
and a Kaggle notebook to fine-tune the checkpoint.

Links in the first comment. Apache 2.0.
