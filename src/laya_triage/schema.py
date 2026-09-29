"""BANKING77 intent hierarchy and laya question builders for ticket triage.

Design decision (documented in the README): a flat choice over the 77 BANKING77
intents collapses — a 12-way choice already scored 0.40 on real data (Jev
ecosystem), and laya issue #102 reports 54.3% -> 60.8% on BANKING77 when the flat
choice is replaced by a two-stage shortlist. We therefore route coarse -> fine:

    77 intents -> 12 coarse clusters (3-10 intents each) -> 8 departments

Every cluster stays at or below 10 intents, far from the collapse zone, and the
coarse stage carries the auxiliary signals (urgency, frustration, churn risk,
refund requested) in the same forward pass.

Intent labels are the exact HuggingFace PolyAI/banking77 spellings, including the
two canonical quirks: ``Refund_not_showing_up`` (capital R, label 51) and
``reverted_card_payment?`` (trailing question mark, label 53).
"""

from __future__ import annotations

# cluster -> department. Eight departments; BANKING77 is a neobank dataset, so
# there are no loan/credit intents (that department would be empty by design).
CLUSTER_DEPARTMENT: dict[str, str] = {
    "card_ordering_and_delivery": "cards",
    "card_types_and_linking": "cards",
    "card_and_pin_malfunctions": "cards",
    "card_payment_problems": "payments_cash",
    "cash_and_atm": "payments_cash",
    "transfers_and_beneficiaries": "transfers",
    "top_ups_and_balance_updates": "topups_deposits",
    "fees_charges_and_refunds": "billing_fees",
    "exchange_rates": "billing_fees",
    "identity_and_account_admin": "account_access",
    "lost_stolen_compromised": "fraud_security",
    "eligibility_and_coverage": "general_service",
}

# cluster -> one-line description used as the option text in the coarse choice.
CLUSTER_DESCRIPTIONS: dict[str, str] = {
    "card_ordering_and_delivery": (
        "ordering, activating or waiting for a new or replacement card"
    ),
    "card_types_and_linking": (
        "card types (virtual, disposable), limits, wallet linking (Apple Pay, Google Pay), supported networks"
    ),
    "card_and_pin_malfunctions": (
        "card or contactless not working, card swallowed by ATM, PIN blocked or change"
    ),
    "card_payment_problems": (
        "card payment declined, pending, charged twice, reverted or not recognised"
    ),
    "cash_and_atm": (
        "ATM or cash withdrawal: cannot withdraw, declined, pending, wrong amount received"
    ),
    "transfers_and_beneficiaries": (
        "transfers failed, pending, cancelled or not received; beneficiary problems"
    ),
    "top_ups_and_balance_updates": (
        "top-ups: failed, pending, not showing yet, limits; balance not updated after adding money"
    ),
    "fees_charges_and_refunds": (
        "unexpected fees or charges, refund requested or missing"
    ),
    "exchange_rates": (
        "exchange rates, exchange charges, currency conversion"
    ),
    "identity_and_account_admin": (
        "identity verification (KYC), source of funds, passcode forgotten, edit personal details, account closure"
    ),
    "lost_stolen_compromised": (
        "card or phone lost or stolen, account or card compromised"
    ),
    "eligibility_and_coverage": (
        "eligibility to open an account: age limit, supported countries and currencies"
    ),
}

# cluster -> exact BANKING77 intent labels, in dataset label order.
CLUSTERS: dict[str, tuple[str, ...]] = {
    "card_ordering_and_delivery": (
        "activate_my_card",
        "card_arrival",
        "card_delivery_estimate",
        "get_physical_card",
        "order_physical_card",
        "getting_spare_card",
    ),
    "card_types_and_linking": (
        "getting_virtual_card",
        "get_disposable_virtual_card",
        "disposable_card_limits",
        "card_linking",
        "visa_or_mastercard",
        "supported_cards_and_currencies",
        "apple_pay_or_google_pay",
        "card_about_to_expire",
        "card_acceptance",
    ),
    "card_and_pin_malfunctions": (
        "card_not_working",
        "contactless_not_working",
        "virtual_card_not_working",
        "card_swallowed",
        "pin_blocked",
        "change_pin",
    ),
    "card_payment_problems": (
        "card_payment_not_recognised",
        "declined_card_payment",
        "pending_card_payment",
        "transaction_charged_twice",
        "reverted_card_payment?",
        "direct_debit_payment_not_recognised",
    ),
    "cash_and_atm": (
        "atm_support",
        "cash_withdrawal_not_recognised",
        "declined_cash_withdrawal",
        "pending_cash_withdrawal",
        "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ),
    "transfers_and_beneficiaries": (
        "cancel_transfer",
        "failed_transfer",
        "declined_transfer",
        "pending_transfer",
        "transfer_not_received_by_recipient",
        "transfer_timing",
        "transfer_into_account",
        "beneficiary_not_allowed",
        "receiving_money",
    ),
    "top_ups_and_balance_updates": (
        "automatic_top_up",
        "pending_top_up",
        "top_up_failed",
        "top_up_reverted",
        "top_up_limits",
        "topping_up_by_card",
        "top_up_by_cash_or_cheque",
        "verify_top_up",
        "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit",
    ),
    "fees_charges_and_refunds": (
        "card_payment_fee_charged",
        "extra_charge_on_statement",
        "transfer_fee_charged",
        "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge",
        "cash_withdrawal_charge",
        "request_refund",
        "Refund_not_showing_up",
    ),
    "exchange_rates": (
        "exchange_rate",
        "exchange_charge",
        "card_payment_wrong_exchange_rate",
        "exchange_via_app",
    ),
    "identity_and_account_admin": (
        "verify_my_identity",
        "why_verify_identity",
        "unable_to_verify_identity",
        "verify_source_of_funds",
        "passcode_forgotten",
        "edit_personal_details",
        "terminate_account",
    ),
    "lost_stolen_compromised": (
        "lost_or_stolen_card",
        "lost_or_stolen_phone",
        "compromised_card",
    ),
    "eligibility_and_coverage": (
        "age_limit",
        "country_support",
        "fiat_currency_support",
    ),
}

DEPARTMENT_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys(CLUSTER_DEPARTMENT.values())
)

INTENT_TO_CLUSTER: dict[str, str] = {
    intent: cluster for cluster, intents in CLUSTERS.items() for intent in intents
}


def _validate() -> None:
    """Guard the partition invariants at import time.

    A future edit that puts an intent in two clusters or leaves a cluster without
    a department should fail loudly here, not silently at prediction time.
    """
    all_intents = [i for intents in CLUSTERS.values() for i in intents]
    if len(all_intents) != len(set(all_intents)):
        raise ValueError("an intent appears in more than one cluster")
    if len(all_intents) != 77:
        raise ValueError(f"expected 77 BANKING77 intents, found {len(all_intents)}")
    if set(CLUSTER_DEPARTMENT) != set(CLUSTERS) or set(CLUSTER_DESCRIPTIONS) != set(CLUSTERS):
        raise ValueError("every cluster needs exactly one department and one description")
    for cluster, intents in CLUSTERS.items():
        if not 2 <= len(intents) <= 10:
            # 11+ options also lands in laya's uncalibrated choice:11+ temperature bucket
            raise ValueError(f"cluster {cluster!r} has {len(intents)} intents (collapse zone)")


_validate()


def _intent_description(intent: str) -> str:
    """Human-readable option text for a BANKING77 label (mechanical transform)."""
    return intent.replace("_", " ").rstrip("?").strip()


def coarse_questions() -> dict:
    """Stage-1 questions: coarse cluster choice plus all auxiliary signals.

    Everything here is answered in a single forward pass; the fine intent choice
    needs the cluster winner, so it runs as stage 2 (see fine_questions).
    Signal wording reuses laya's triage preset where it fits; urgency is a 0-3
    score instead of the preset's boolean is_urgent (per this project's design).
    """
    cluster_q = {
        "type": "choice",
        "instructions": "Which area does the customer's issue in `message` belong to?",
        "criteria": dict(CLUSTER_DESCRIPTIONS),
    }
    urgency_q = {
        "type": "score",
        "instructions": "How urgent is the customer's issue in `message`?",
        "criteria": [
            "no time pressure",
            "some urgency",
            "high urgency",
            "drop-everything urgent",
        ],
    }
    frustration_q = {
        "type": "score",
        "instructions": "How frustrated does the customer sound in `message`?",
        "criteria": [
            "calm and neutral",
            "concerned but civil",
            "clearly annoyed",
            "very angry or using strong language",
        ],
    }
    churn_q = {
        "type": "noul",
        "instructions": "Does `message` suggest the customer may leave for a competitor or cancel?",
    }
    refund_q = {
        "type": "noul",
        "instructions": "Does the customer ask for money back?",
    }
    return {
        "cluster": cluster_q,
        "urgency": urgency_q,
        "frustration": frustration_q,
        "churn_risk": churn_q,
        "refund_requested": refund_q,
    }


def fine_questions(cluster: str) -> dict:
    """Stage-2 questions: intent choice restricted to one cluster's intents."""
    intents = CLUSTERS[cluster]  # KeyError for an unknown cluster is the contract
    return {
        "intent": {
            "type": "choice",
            "instructions": "What exactly does the customer need in `message`?",
            "criteria": {i: _intent_description(i) for i in intents},
        }
    }
