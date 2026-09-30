"""Dataset loading for the Phase 2 evals.

BANKING77 comes from the canonical PolyAI repository (the HuggingFace
``PolyAI/banking77`` dataset is script-based and no longer loads with modern
``datasets``): ``banking_data/{train,test,categories}.csv|json`` from
https://github.com/PolyAI-LDN/task-specific-datasets. Files are downloaded
once into ``data/external/`` (gitignored) and asserted against our schema's
77 intents before use.

MASSIVE comes from the parquet conversion of the ``qanastek/MASSIVE`` mirror
(``refs/convert/parquet``), which loads with modern ``datasets`` without
running the original dataset script.
"""

from __future__ import annotations

import csv
import json
import random
import urllib.request
from pathlib import Path

from laya_triage import schema

BANKING77_BASE = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
)
EXTERNAL_DIR = Path(__file__).resolve().parent.parent / "data" / "external"
BANKING77_FILES = ("train.csv", "test.csv", "categories.json")

# MASSIVE locales for the multilingual eval (test split, parquet conversion).
MASSIVE_LOCALES = ("es-ES", "fr-FR", "de-DE", "hi-IN", "ar-SA")
MASSIVE_PARQUET = (
    "hf://datasets/qanastek/MASSIVE@refs/convert/parquet/{locale}/test/0000.parquet"
)


def ensure_banking77() -> Path:
    """Download the canonical BANKING77 files if missing; return their directory."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    for name in BANKING77_FILES:
        target = EXTERNAL_DIR / name
        if not target.exists():
            urllib.request.urlretrieve(f"{BANKING77_BASE}/{name}", target)
    return EXTERNAL_DIR


def load_banking77(split: str = "test") -> list[dict]:
    """Rows as ``{"text": str, "label": str}``; labels asserted against the schema."""
    ensure_banking77()
    categories = json.loads((EXTERNAL_DIR / "categories.json").read_text(encoding="utf-8"))
    if set(categories) != set(schema.INTENT_TO_CLUSTER):
        raise ValueError("BANKING77 categories drifted from laya_triage.schema intents")

    path = EXTERNAL_DIR / f"{split}.csv"
    with open(path, encoding="utf-8", newline="") as fh:
        rows = [{"text": r["text"], "label": r["category"]} for r in csv.DictReader(fh)]
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def sample_rows(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """Deterministic sample; limit >= len(rows) returns everything (no reshuffle)."""
    if limit >= len(rows):
        return list(rows)
    return random.Random(seed).sample(rows, limit)


def load_massive(locale: str, limit: int) -> list[dict]:
    """MASSIVE test-split rows as ``{"id": int, "utt": str}`` for one locale.

    The id is parallel across locales: the same id is the same utterance
    translated, which is what the cross-language consistency metric needs.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files={"test": MASSIVE_PARQUET.format(locale=locale)},
        split="test",
    )
    rows = [{"id": r["id"], "utt": r["utt"]} for r in ds]
    return sample_rows(rows, limit, seed=13)
