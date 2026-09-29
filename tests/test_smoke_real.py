"""Phase 1 smoke on a REAL laya checkpoint — opt-in (``pytest -m slow``).

Downloads the checkpoint on first run and runs CPU inference (two
decisions per ticket; TODO(measure): latency on this machine). Default pytest runs skip this file; the
deterministic unit tests in test_pipeline.py mock the agent instead.
"""

import pytest

from laya_triage import TriagePipeline, build_router
from laya_triage import schema

# Provisional threshold until Phase 2 picks one from a published
# coverage/accuracy curve. TODO(measure): replace with the measured threshold.
PROVISIONAL_MIN_CONFIDENCE = 0.6

# 20 tickets, Spanish and English, spread across clusters and signal levels.
SMOKE_TICKETS = [
    # Spanish
    "me cobraron dos veces, reembolsen o cancelo",
    "no puedo retirar dinero del cajero, me marca error",
    "pedí una tarjeta nueva hace dos semanas y todavía no llega",
    "mi transferencia a otra cuenta lleva tres días pendiente",
    "no puedo verificar mi identidad en la app, me rechaza el documento",
    "me robaron el teléfono con la app del banco abierta",
    "cuanto tarda en reflejarse el dinero después de hacer una recarga en efectivo",
    "me hicieron un cobro extra en el último extracto que no reconozco",
    "el cambio de divisa que me aplicaron es peor que el del mercado",
    "puedo abrir una cuenta si tengo diecisiete años",
    # English
    "I was charged twice for the same payment, refund me or I will cancel my account",
    "my card gets declined every time I try to pay contactless",
    "I forgot my passcode and now I am locked out of my account",
    "how long does a bank transfer to another country usually take?",
    "top up by card failed twice, money left my bank but never arrived",
    "there is a fee on my statement I was never told about, I want it back",
    "my virtual card stopped working when I tried to subscribe online",
    "I want to add my partner as a beneficiary but the app refuses",
    "is there a limit on cash withdrawals abroad per month?",
    "please close my account, I am done with this bank",
]

ACCEPTANCE_TICKET = SMOKE_TICKETS[0]

# Real-world IT-operations tickets (subject + body), used with permission from
# a local triage-console demo batch. Out-of-distribution for the BANKING77
# hierarchy: the interesting behavior is defensible routing where banking
# concepts overlap (billing, locked-out accounts) and honest low-confidence
# escalation for the rest.
OOD_IT_TICKETS = [
    "URGENT: Production API returning 500 errors across all endpoints. "
    "Since 14:30 UTC our main API gateway is returning 500 errors. "
    "Approximately 2,000 customers affected. Revenue impact estimated at "
    "15k/hour. Engineering team has been paged.",
    "Can't export CSV from dashboard. When I click the export button on the "
    "analytics dashboard, nothing happens. Chrome console shows a CORS error. "
    "This has been happening since the last deploy on Friday.",
    "Request: Add dark mode to the mobile app. Several enterprise customers "
    "have asked about dark mode support for the iOS and Android apps. Not "
    "urgent but would be good for the next release.",
    "Data discrepancy in billing invoices. Customer Acme Corp reports their "
    "March invoice shows 340 seats but they only have 280 active users. "
    "Finance team needs to investigate before end of quarter. Customer is "
    "escalating to their VP.",
    "New employee onboarding - access provisioning. Hi, we have 12 new hires "
    "starting Monday. Need accounts provisioned with standard access. Names "
    "and roles attached.",
    "SSO login broken after IdP certificate rotation. After rotating our SAML "
    "certificate this morning, no users can log in via SSO. 500+ employees "
    "locked out. Workaround is direct login but most users don't have "
    "passwords set.",
    "Question about API rate limits for batch processing. We're building an "
    "integration that needs to process ~50k records daily. Current rate limit "
    "is 100 req/min. Can we get an increase or is there a batch endpoint we "
    "should use instead?",
    "Performance regression in search after v4.2 release. Search queries that "
    "used to return in <200ms are now taking 3-5 seconds. Affects the main "
    "product search bar. Customer-facing. Our APM shows the issue started "
    "exactly after the v4.2 deploy.",
]


@pytest.fixture(scope="module")
def pipeline() -> TriagePipeline:
    return TriagePipeline(build_router(), min_confidence=PROVISIONAL_MIN_CONFIDENCE)


@pytest.mark.slow
@pytest.mark.parametrize("text", SMOKE_TICKETS, ids=[t[:30] for t in SMOKE_TICKETS])
def test_smoke_ticket_produces_a_valid_triage(pipeline, text):
    result = pipeline.triage(text)

    assert result.cluster in schema.CLUSTERS
    assert result.intent in schema.CLUSTERS[result.cluster]
    assert result.department == schema.CLUSTER_DEPARTMENT[result.cluster]
    assert 0.0 <= result.urgency <= 3.0
    assert 0.0 <= result.frustration <= 3.0
    assert 0.0 <= result.churn_risk <= 1.0
    assert 0.0 <= result.refund_requested <= 1.0
    assert result.escalate == bool(result.escalation_reasons)
    print(
        f"\n{text[:50]!r}\n  -> {result.department}/{result.cluster}/{result.intent}"
        f" urg={result.urgency:.2f} fru={result.frustration:.2f}"
        f" churn={result.churn_risk:.2f} refund={result.refund_requested:.2f}"
        f" escalate={result.escalate} model={result.model}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("text", OOD_IT_TICKETS, ids=[t[:24] for t in OOD_IT_TICKETS])
def test_smoke_ood_it_ticket_stays_structurally_valid(pipeline, text):
    """OOD tickets must produce a valid triage and escalate-or-route sanely.

    The banking hierarchy cannot name these intents; the contract is that the
    pipeline never crashes, never invents an invalid label, and either routes
    to a defensible area with confidence or flags a human.
    """
    result = pipeline.triage(text)

    assert result.cluster in schema.CLUSTERS
    assert result.intent in schema.CLUSTERS[result.cluster]
    assert 0.0 <= result.urgency <= 3.0
    assert 0.0 <= result.frustration <= 3.0
    assert 0.0 <= result.churn_risk <= 1.0
    assert 0.0 <= result.refund_requested <= 1.0
    print(
        f"\n{text[:44]!r}\n  -> {result.department}/{result.cluster}/{result.intent}"
        f" urg={result.urgency:.2f} cluster_conf={result.confidences['cluster']:.2f}"
        f" intent_conf={result.confidences['intent']:.2f} escalate={result.escalate}"
    )


@pytest.mark.slow
def test_acceptance_criterion_double_charge_refund_threat(pipeline):
    """"me cobraron dos veces, reembolsen o cancelo" -> billing-owning department,
    high urgency, high churn risk, refund requested."""
    r = pipeline.triage(ACCEPTANCE_TICKET)

    assert r.department in {"billing_fees", "payments_cash"}, (
        f"double charge + refund demand landed in {r.cluster}/{r.department}"
    )
    assert r.urgency >= 1.5, f"urgency {r.urgency:.2f}"
    assert r.churn_risk >= 0.5, f"churn_risk {r.churn_risk:.2f}"
    assert r.refund_requested >= 0.5, f"refund_requested {r.refund_requested:.2f}"
    print(
        f"\nacceptance: {r.department}/{r.cluster}/{r.intent}"
        f" urg={r.urgency:.2f} churn={r.churn_risk:.2f} refund={r.refund_requested:.2f}"
    )
