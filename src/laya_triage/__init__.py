"""laya-triage: multilingual support ticket triage: department, urgency, frustration and churn risk in one forward pass, with confidence-based escalation to humans."""

from laya_triage.pipeline import TriagePipeline, TriageResult, build_router
from laya_triage import schema

__version__ = "0.1.0"

__all__ = ["TriagePipeline", "TriageResult", "build_router", "schema"]
