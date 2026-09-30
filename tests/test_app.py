"""App helper tests (pure functions; no streamlit runtime, no checkpoints).

Importing app.py under plain pytest executes its streamlit calls in bare
mode (no-ops) without ever calling get_pipeline(), so no model loads.
"""

import io
import json

import app
from laya_triage.pipeline import TriageResult


def make_result(**overrides) -> TriageResult:
    base = dict(
        text="me cobraron dos veces",
        department="payments_cash",
        cluster="card_payment_problems",
        intent="transaction_charged_twice",
        urgency=1.7712,
        frustration=1.9,
        churn_risk=0.87,
        refund_requested=0.91,
        confidences={"cluster": 0.823, "intent": 0.911, "urgency": 0.5, "frustration": 0.6, "churn_risk": 0.9, "refund_requested": 0.9},
        escalate=False,
        escalation_reasons=[],
        model="multilingual",
        routing={},
    )
    base.update(overrides)
    return TriageResult(**base)


def test_result_row_shape_and_rounding():
    row = app.result_row(make_result())
    assert row["department"] == "payments_cash"
    assert row["intent"] == "transaction_charged_twice"
    assert row["urgency"] == 1.77  # rounded for the table
    assert row["escalate"] is False
    assert set(row) == {
        "text", "department", "cluster", "intent", "urgency", "frustration",
        "churn_risk", "refund_requested", "cluster_conf", "intent_conf",
        "escalate", "model",
    }


def test_parse_batch_csv_reads_text_column_and_skips_empty():
    csv_content = "text,other\n\"hello world\",1\n\n\"\",2\n\"second ticket\",3\n"
    texts = app.parse_batch_csv(io.StringIO(csv_content))
    assert texts == ["hello world", "second ticket"]


def test_parse_batch_csv_rejects_missing_text_column():
    import pytest

    with pytest.raises(ValueError, match="text"):
        app.parse_batch_csv(io.StringIO("body\nnope\n"))


def test_load_metrics_reads_whichever_artifacts_exist(tmp_path):
    (tmp_path / "banking77.json").write_text(json.dumps({"summary": {}}), encoding="utf-8")
    metrics = app.load_metrics(tmp_path)
    assert set(metrics) == {"banking77"}
    assert app.load_metrics(tmp_path / "empty") == {}


def test_parse_batch_csv_handles_streamlit_bytes_with_bom():
    import io

    data = b"\xef\xbb\xbftext\n\"via uploaded file\"\n"  # utf-8 BOM + bytes, as streamlit delivers
    assert app.parse_batch_csv(io.BytesIO(data)) == ["via uploaded file"]
