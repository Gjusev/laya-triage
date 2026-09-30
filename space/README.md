---
title: laya-triage
emoji: 🎟️
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
license: apache-2.0
short_description: Multilingual ticket triage with confidence-based escalation
---

# laya-triage app

Multilingual ticket triage on a local System 1 decision model
([laya](https://github.com/NandhaKishorM/laya)): department, urgency,
frustration, churn risk and refund requests in one forward pass, with a
measured confidence-based escalation policy.

Three views: a single ticket, a CSV batch, and the published eval metrics
(including the coverage/accuracy escalation curve).

## Hosting the app

**Streamlit Community Cloud (free, recommended).** The repo root already
carries a `requirements.txt` deploy manifest and `app.py`, and the metrics
view reads the committed `evals/results/*.json`:

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with the
   GitHub account that owns this repo.
2. New app → repo `Gjusev/laya-triage`, branch `main`, main file `app.py`.
3. First inference downloads the laya checkpoints and caches them; the
   421M model is memory-hungry, so if the free tier's RAM cap is reached
   use a host with more memory instead.

**Hugging Face Space (requires PRO for runtime Spaces on free cpu-basic).**
Builds from `space/Dockerfile` (the same Streamlit app wrapped in Docker).
Copy `app.py`, `space/requirements.txt`, `space/Dockerfile` and
`evals/results/*.json` into the Space repo and commit.

Source, evals, and the measured numbers: [laya-triage on GitHub](https://github.com/Gjusev/laya-triage).
