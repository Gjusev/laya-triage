# laya-triage

> Multilingual support ticket triage: department, urgency, frustration and churn risk in one forward pass, with confidence-based escalation to humans.

Status: early development. Built on [laya](https://github.com/NandhaKishorM/laya),
the open-source System 1 decision engine (Apache 2.0).

## Why

- Flat classification collapses at scale: a 12-way choice scored 0.40 on real data in the Jev ecosystem. Hierarchical coarse-to-fine routing is the documented fix (54.3% to 60.8% on BANKING77, laya issue #102).
- laya's Router covers 45 of 51 languages automatically, and every answer carries calibrated confidence, so the escalation policy is measurable instead of guessed.

## Roadmap

- [ ] Hierarchical intent routing: coarse clusters to fine intents (BANKING77 mapping)
- [ ] Auxiliary signals in the same pass: urgency, frustration, churn risk, refund requested
- [ ] Escalation policy: coverage/accuracy curve over answer confidence, threshold documented
- [ ] Multilingual eval (MASSIVE: es, fr, de, hi, ar) plus a hand-labeled set of 200 tickets published
- [ ] Fine-tuned checkpoint (BANKING77) released on HuggingFace; Streamlit app on HF Spaces

## Development setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or any venv + pip):

```bash
uv venv
uv pip install -e ".[dev]"
pytest
```

## License

Apache 2.0. See [LICENSE](LICENSE).
