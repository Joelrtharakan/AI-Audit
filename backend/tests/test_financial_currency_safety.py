"""Regression coverage for the currency + epistemic-status hardening pass.

Audit findings before this pass:
  - Currency was already never silently lost (USD stays USD, INR stays
    INR) -- confirmed by direct reproduction, no fix needed.
  - Evidence status was already never upgraded (BELIEF/UNVERIFIED never
    became VERIFIED) -- the extractor already maps every EvidenceStatus
    other than VERIFIED/REPORTED down to "UNVERIFIED" internally.
  - Independent financial observations were already never merged or
    overwritten -- each source claim produces its own FinancialObservation.
  - Multiple currencies already correctly blocked consolidated
    aggregation (status NOT_ASSESSABLE, no fabricated combined amount).

Gap found and fixed: the multi-currency path discarded ALL per-currency
information, returning a blank NOT_ASSESSABLE result with no visibility
into what each currency's own exposure actually was -- and the frontend
renderer's hasFinData gate hid the entire Cost & Financial Exposure
Analysis section whenever status was NOT_ASSESSABLE, so a genuine
multi-currency finding produced ZERO rendered financial section at all
(strictly worse than "misleading": completely invisible).

Fix: added CurrencyExposure/currency_breakdown to the financial model,
computed per-currency using the SAME deterministic calculate_confirmed_
impact already used for the single-currency path (reused, not
reimplemented) scoped to each currency's own observations; extended the
JS renderer's gate and added a compact per-currency display block.

Uses abstract, domain-neutral test fixtures.
"""

from __future__ import annotations

import math

from app.financial.engine import analyze_financial_exposure
from app.financial.models import FinancialEpistemicStatus
from app.models.agent import EvidenceItem, EvidenceStatus


def test_usd_only_currency_preserved():
    finding = "A verified exposure of USD 2,000 was identified."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.currency == "USD"
    assert res.confirmed_impact.verified_gross_exposure == 2000.0


def test_inr_only_unchanged_baseline():
    finding = "A verified exposure of INR 50,000 was identified."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.currency == "INR"
    assert res.confirmed_impact.verified_gross_exposure == 50000.0


def test_usd_never_silently_becomes_inr():
    finding = "A verified exposure of USD 2,000 was identified."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.currency != "INR"


def test_currency_symbol_and_code_parsing():
    for text, expected_currency, expected_amount in [
        ("$2,000 was verified.", "USD", 2000.0),
        ("USD 2,000 was verified.", "USD", 2000.0),
        ("₹50,000 was verified.", "INR", 50000.0),
        ("EUR 3,000 was verified.", "EUR", 3000.0),
    ]:
        ledger = [EvidenceItem(claim=text, status=EvidenceStatus.VERIFIED, source="finance")]
        res = analyze_financial_exposure(text, evidence_ledger=ledger)
        assert res.confirmed_impact.currency == expected_currency, text
        assert res.confirmed_impact.verified_gross_exposure == expected_amount, text


def test_adversarial_multi_currency_case_exact_reproduction():
    """The exact adversarial case: a BELIEF-level USD claim and a
    VERIFIED INR claim must never be combined, never silently converted,
    and BELIEF must never become VERIFIED."""
    ledger = [
        EvidenceItem(claim="A supplier dispute resulted in a verified exposure of USD 2,000.", status=EvidenceStatus.BELIEF, source="C1"),
        EvidenceItem(claim="A separate related expense of INR 50,000 was also verified.", status=EvidenceStatus.VERIFIED, source="C2"),
        EvidenceItem(claim="No authoritative exchange rate was available in the evidence.", status=EvidenceStatus.VERIFIED, source="C3"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)

    # Must never produce a fabricated consolidated/converted figure.
    assert res.status == FinancialEpistemicStatus.NOT_ASSESSABLE
    assert res.confirmed_impact.verified_gross_exposure is None
    assert res.conversion_status == "NOT_AVAILABLE"

    # Both currencies must be independently preserved.
    by_currency = {c.currency: c for c in res.currency_breakdown}
    assert set(by_currency) == {"USD", "INR"}

    inr = by_currency["INR"]
    assert inr.gross_amount == 50000.0
    assert inr.status == "VERIFIED"

    usd = by_currency["USD"]
    # BELIEF-sourced amount must never be shown as VERIFIED.
    assert usd.status != "VERIFIED"
    assert usd.gross_amount is None  # never promoted to a verified gross figure
    assert usd.reported_amount == 2000.0

    # No NaN/Infinity anywhere in the result.
    d = res.model_dump()

    def _check(obj):
        if isinstance(obj, float):
            assert not math.isnan(obj) and not math.isinf(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _check(v)
        elif isinstance(obj, list):
            for v in obj:
                _check(v)

    _check(d)


def test_usd_gross_and_usd_recovery_same_currency_valid():
    finding = "A verified exposure of USD 2,000 was identified, and USD 500 was recovered."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 2000.0
    assert res.confirmed_impact.verified_recovery == 500.0
    assert res.confirmed_impact.confirmed_net_loss == 1500.0


def test_usd_gross_inr_recovery_never_subtracted_across_currencies():
    """Currency mismatch between gross and recovery must never produce a
    net figure -- this is caught by the same multi-currency gate at
    extraction time (the two amounts are simply different currencies
    within the same finding)."""
    ledger = [
        EvidenceItem(claim="A verified exposure of USD 2,000 was identified.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="A recovery of INR 50,000 was verified.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.status == FinancialEpistemicStatus.NOT_ASSESSABLE
    assert res.confirmed_impact.confirmed_net_loss is None


def test_multiple_verified_currencies_all_preserved_independently():
    ledger = [
        EvidenceItem(claim="A verified exposure of USD 2,000 was identified.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="A verified exposure of INR 50,000 was identified.", status=EvidenceStatus.VERIFIED, source="C2"),
        EvidenceItem(claim="A verified exposure of EUR 1,000 was identified.", status=EvidenceStatus.VERIFIED, source="C3"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    by_currency = {c.currency: c for c in res.currency_breakdown}
    assert set(by_currency) == {"USD", "INR", "EUR"}
    assert by_currency["USD"].status == "VERIFIED"
    assert by_currency["INR"].status == "VERIFIED"
    assert by_currency["EUR"].status == "VERIFIED"


def test_duplicate_corroborating_same_currency_observations_not_double_counted():
    finding = "A verified exposure of USD 2,000 was identified."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="Log A"),
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="Log A"),
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 2000.0
