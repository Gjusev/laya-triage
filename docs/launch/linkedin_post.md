# LinkedIn post draft

> Attach: screenshot of the app's metrics view (escalation curve) or the
> batch view with escalated tickets highlighted. English per repo rules.
> TODO(publish): add the Space link after deployment. Checkpoint is live at
> https://huggingface.co/Gjusev/laya-triage-banking77.

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

And after fine-tuning the two routing decisions on BANKING77 (Kaggle 2xT4,
about 46 minutes): 90.5% accuracy, macro-F1 0.848, with the urgency and
frustration heads untouched. The fine-tuned checkpoint is on Hugging Face
with a model card generated from its own measured run.

Department, urgency, frustration, churn risk and refund detection all come
out of the same single pass, in any of the 45 languages laya's router covers, with calibrated
confidence per decision.

The repo ships everything: the evals, the escalation curve, a 200-ticket
multilingual labeled dataset (two blind annotation passes, kappa 0.78+),
the Kaggle fine-tuning notebook, and the published checkpoint.

Links in the first comment. Apache 2.0.
