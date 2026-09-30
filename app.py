"""Streamlit app: single ticket, CSV batch, and metrics views.

Run locally:
    streamlit run app.py

Deploy (HuggingFace Space, CPU): copy this file plus space/requirements.txt
and the space/README.md card into a new Streamlit Space, and copy
evals/results/ next to the app so the metrics view has data to render.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from laya_triage import TriagePipeline, build_router

MIN_CONFIDENCE = 0.84  # measured operational threshold (evals/results/banking77.json)
RESULTS_DIR = Path(__file__).parent / "evals" / "results"


@st.cache_resource(show_spinner="loading laya checkpoints (first run downloads them)")
def get_pipeline() -> TriagePipeline:
    return TriagePipeline(build_router(), min_confidence=MIN_CONFIDENCE)


def result_row(result) -> dict:
    """One triaged ticket -> the batch-view dataframe row."""
    return {
        "text": result.text,
        "department": result.department,
        "cluster": result.cluster,
        "intent": result.intent,
        "urgency": round(result.urgency, 2),
        "frustration": round(result.frustration, 2),
        "churn_risk": round(result.churn_risk, 2),
        "refund_requested": round(result.refund_requested, 2),
        "cluster_conf": round(result.confidences["cluster"], 2),
        "intent_conf": round(result.confidences["intent"], 2),
        "escalate": result.escalate,
        "model": result.model,
    }


def parse_batch_csv(file_like) -> list[str]:
    """Ticket texts from an uploaded CSV; requires a `text` column."""
    raw = file_like.read()
    if isinstance(raw, bytes):  # streamlit UploadedFile
        raw = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(raw))
    if "text" not in (reader.fieldnames or []):
        raise ValueError("CSV needs a 'text' column")
    texts = [(row.get("text") or "").strip() for row in reader]
    return [t for t in texts if t]


def load_metrics(results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    """The eval artifacts the metrics view renders ({} when absent)."""
    metrics = {}
    for name in ("banking77", "massive", "hand_labeled"):
        path = results_dir / f"{name}.json"
        if path.exists():
            metrics[name] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


def render_single() -> None:
    st.header("Single ticket")
    text = st.text_area(
        "Ticket text (any language)",
        value="me cobraron dos veces, reembolsen o cancelo",
        height=120,
    )
    lang = st.selectbox("Language hint (optional)", ["auto", "en", "es", "fr", "de", "hi", "ar"])

    if st.button("Triage", type="primary") and text.strip():
        with st.spinner("routing coarse -> fine..."):
            result = get_pipeline().triage(text, lang=None if lang == "auto" else lang)

        if result.escalate:
            st.error("Escalate to a human: " + "; ".join(result.escalation_reasons))
        else:
            st.success("Auto-handled (both routing decisions above threshold)")

        col1, col2 = st.columns(2)
        col1.metric("Department", result.department)
        col1.metric("Cluster", result.cluster)
        col1.metric("Intent", result.intent)
        col2.metric("Urgency (0-3)", f"{result.urgency:.2f}")
        col2.metric("Frustration (0-3)", f"{result.frustration:.2f}")
        col2.metric("Churn / refund", f"{result.churn_risk:.2f} / {result.refund_requested:.2f}")
        st.caption(
            f"routed checkpoint: {result.model or 'n/a'} | "
            f"confidences: cluster {result.confidences['cluster']:.2f}, "
            f"intent {result.confidences['intent']:.2f} | "
            f"threshold {MIN_CONFIDENCE}"
        )


def render_batch() -> None:
    st.header("Batch (CSV)")
    st.caption("Upload a CSV with a `text` column; every row is triaged with batched inference.")
    uploaded = st.file_uploader("tickets.csv", type=["csv"])
    if uploaded is None:
        st.info("Waiting for a CSV.")
        return
    try:
        texts = parse_batch_csv(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        return
    if not texts:
        st.warning("No non-empty rows found.")
        return

    st.info(f"{len(texts)} tickets")
    with st.spinner("triaging..."):
        results = get_pipeline().triage_batch(texts)
    frame = pd.DataFrame([result_row(r) for r in results])

    escalated = int(frame["escalate"].sum())
    st.metric("Escalated to human", f"{escalated}/{len(frame)}")
    st.dataframe(frame, use_container_width=True)
    st.download_button(
        "Download results (CSV)",
        frame.to_csv(index=False).encode("utf-8"),
        file_name="triage_results.csv",
        mime="text/csv",
    )


def render_metrics() -> None:
    st.header("Metrics (published eval artifacts)")
    metrics = load_metrics()
    if not metrics:
        st.info(
            "No eval artifacts found. Copy evals/results/*.json next to the app, "
            "or run `python -m evals.run_all` in the repo."
        )
        return

    if "banking77" in metrics:
        b77 = metrics["banking77"]
        s = b77["summary"]
        st.subheader("BANKING77 (test sample)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Direct 77-way", f"{s['direct_accuracy']:.1%}")
        col2.metric("Hierarchical", f"{s['hierarchical_accuracy']:.1%}")
        col3.metric("Coarse", f"{s['coarse_accuracy']:.1%}")
        esc = b77["escalation"]
        if esc.get("chosen_threshold") is not None:
            st.caption(
                f"operational threshold {esc['chosen_threshold']:.2f}: "
                f"coverage {esc['chosen_coverage']:.1%} at {esc['chosen_accuracy']:.1%} accuracy"
            )
        curve = pd.DataFrame(esc["curve"])
        st.line_chart(curve.set_index("coverage")["accuracy"])

    if "hand_labeled" in metrics:
        sig = metrics["hand_labeled"]["signals"]
        st.subheader("Signals (hand-labeled 200)")
        col1, col2 = st.columns(2)
        col1.metric("Urgency MAE", f"{sig['urgency_mae']:.2f}")
        col2.metric("Frustration MAE", f"{sig['frustration_mae']:.2f}")
        st.caption(
            f"inter-pass kappa: urgency {sig['inter_pass_kappa_urgency']:.2f}, "
            f"frustration {sig['inter_pass_kappa_frustration']:.2f}"
        )

    if "massive" in metrics:
        st.subheader("Multilingual (MASSIVE, out-of-domain behavior)")
        per_locale = pd.DataFrame(metrics["massive"]["per_locale"]).T
        st.dataframe(per_locale, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="laya-triage", page_icon="🎟️", layout="wide")
    st.title("laya-triage")
    st.caption(
        "Multilingual support ticket triage on a local System 1 model: department, "
        "urgency, frustration, churn and refund in one forward pass, with "
        "confidence-based escalation."
    )
    view = st.sidebar.radio("View", ["Single ticket", "Batch (CSV)", "Metrics"])
    if view == "Single ticket":
        render_single()
    elif view == "Batch (CSV)":
        render_batch()
    else:
        render_metrics()


if __name__ == "__main__":
    main()
