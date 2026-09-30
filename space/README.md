---
title: laya-triage
emoji: 🎟️
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.40"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Multilingual ticket triage with confidence-based escalation
---

# laya-triage app

Multilingual support ticket triage on a local System 1 decision model
([laya](https://github.com/NandhaKishorM/laya)): department, urgency,
frustration, churn risk and refund requests in one forward pass, with a
measured confidence-based escalation policy.

Three views: a single ticket, a CSV batch, and the published eval metrics
(including the coverage/accuracy escalation curve).

## Deploying this Space

> **Prerequisite (TODO(push)):** the `laya-triage` install below resolves
> against `origin/main` on GitHub. Push the repo before deploying, or the
> Space build fails at import.

1. Create a Streamlit Space (CPU basic is enough; first inference downloads
   the laya checkpoints and caches them).
2. Copy into the Space repo:
   - `app.py` from the project root,
   - `requirements.txt` from this directory,
   - `evals/results/*.json` into `evals/results/` (metrics view data).
3. Commit; the Space builds itself.

Source, evals, and the measured numbers: [laya-triage on GitHub](https://github.com/Gjusev/laya-triage).
