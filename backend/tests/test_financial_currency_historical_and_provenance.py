"""Regression coverage for the two remaining limitations disclosed at the
end of the previous currency-hardening pass:

Limitation 1 -- historical annualized exposure was a single global field,
so a multi-currency finding with a historical figure in one currency
(e.g. "Historical records show verified losses of USD 10,000 per year")
lost that figure entirely once a different-currency current fact
triggered the multi-currency NOT_ASSESSABLE path -- the historical USD
annualization simply vanished from the output.

Limitation 2 -- BELIEF-sourced evidence was already never upgraded to
VERIFIED for calculation purposes (the four-bucket calculation model was
correct), but the rendered/structured output collapsed BELIEF into the
same "REPORTED" label as genuinely REPORTED evidence, losing the
distinction the auditor needs to see.

Fix: reused (not reimplemented) the existing deterministic
calculate_annualized_exposure per currency subset in the multi-currency
branch, and added a provenance-only source_evidence_status field to
FinancialObservation/CurrencyExposure that carries the original
EvidenceStatus through purely for rendering -- the authoritative
four-bucket verification_status/calculation eligibility model is
completely unchanged.

Uses abstract, domain-neutral test fixtures.
"""

from __future__ import annotations

from app.financial.engine import analyze_financial_exposure
from app.models.agent import EvidenceItem, EvidenceStatus


def _breakdown_by_currency(res):
    return {c.currency: c for c in res.currency_breakdown}


def test_historical_usd_annualization_visible_alongside_current_inr():
    """Case A from the task spec: historical USD annualized exposure
    must remain visible even though the overall consolidated status is
    NOT_ASSESSABLE due to the separate current INR fact."""
    ledger = [
        EvidenceItem(claim="Historical records show verified losses of USD 10,000 per year.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="A separate verified expense of INR 50,000 was identified.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    by_curr = _breakdown_by_currency(res)
    assert by_curr["INR"].gross_amount == 50000.0
    assert by_curr["INR"].status == "VERIFIED"
    assert by_curr["USD"].historical_is_assessable is True
    assert by_curr["USD"].historical_annualized_amount == 10000.0
    # Historical USD must never contaminate current INR gross exposure.
    assert by_curr["USD"].gross_amount is None


def test_historical_inr_current_usd_symmetric_case():
    """Case D from the task's test matrix: the mirror of Case A."""
    ledger = [
        EvidenceItem(claim="Historical records show verified losses of INR 150,000 per year.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="A separate verified expense of USD 2,000 was identified.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    by_curr = _breakdown_by_currency(res)
    assert by_curr["USD"].gross_amount == 2000.0
    assert by_curr["INR"].historical_annualized_amount == 150000.0
    assert by_curr["INR"].gross_amount is None


def test_multiple_historical_currencies_all_independently_visible():
    ledger = [
        EvidenceItem(claim="Historical records show verified losses of USD 10,000 per year.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="Historical records show verified losses of EUR 8,000 per year.", status=EvidenceStatus.VERIFIED, source="C2"),
        EvidenceItem(claim="Historical records show verified losses of INR 500,000 per year.", status=EvidenceStatus.VERIFIED, source="C3"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    by_curr = _breakdown_by_currency(res)
    assert by_curr["USD"].historical_annualized_amount == 10000.0
    assert by_curr["EUR"].historical_annualized_amount == 8000.0
    assert by_curr["INR"].historical_annualized_amount == 500000.0
    # No numeric cross-currency aggregation anywhere.
    assert res.status.value == "NOT_ASSESSABLE"


def test_same_currency_current_and_historical_stay_semantically_distinct():
    """Single-currency case: current exposure and historical annualized
    exposure must not be summed, even when both are established (this is
    the pre-existing single-currency path -- confirmed unchanged)."""
    ledger = [
        EvidenceItem(claim="A verified rework cost of INR 20,000 was incurred for the current nonconformity.", status=EvidenceStatus.VERIFIED, source="QA log"),
        EvidenceItem(claim="Historically, similar incidents occurred 4 times over the past year, each verified at INR 15,000.", status=EvidenceStatus.VERIFIED, source="CAPA database"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 20000.0
    assert res.annualized_exposure.annualized_amount == 60000.0
    assert res.confirmed_impact.verified_gross_exposure != 80000.0


def test_belief_evidence_status_preserved_through_financial_observation():
    from app.financial.extractor import extract_financial_observations

    obs, *_ = extract_financial_observations(
        "x",
        evidence_ledger=[
            EvidenceItem(claim="A supplier dispute resulted in a USD 2,000 exposure.", status=EvidenceStatus.BELIEF, source="C1"),
        ],
    )
    assert len(obs) == 1
    assert obs[0].source_evidence_status == "BELIEF"
    # Calculation-eligibility status is unaffected by this change.
    assert obs[0].verification_status == "UNVERIFIED"


def test_belief_rendered_as_belief_not_reported_in_currency_breakdown():
    """Case B / Case (13) from the task spec: BELIEF must be visibly
    distinct from REPORTED in the structured output."""
    ledger = [
        EvidenceItem(claim="A supplier dispute resulted in a USD 2,000 exposure.", status=EvidenceStatus.BELIEF, source="C1"),
        EvidenceItem(claim="A separate INR 50,000 expense was verified.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    by_curr = _breakdown_by_currency(res)
    assert by_curr["USD"].source_evidence_status == "BELIEF"
    assert by_curr["INR"].source_evidence_status == "VERIFIED"


def test_reported_status_preserved_distinctly():
    ledger = [EvidenceItem(claim="The finding reports an exposure of USD 2,000.", status=EvidenceStatus.REPORTED, source="C1")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.source_evidence_ids  # sanity: extraction occurred
    from app.financial.extractor import extract_financial_observations
    obs, *_ = extract_financial_observations("x", evidence_ledger=ledger)
    assert obs[0].source_evidence_status == "REPORTED"


def test_verified_status_preserved_distinctly():
    from app.financial.extractor import extract_financial_observations
    ledger = [EvidenceItem(claim="Records verify an exposure of USD 2,000.", status=EvidenceStatus.VERIFIED, source="C1")]
    obs, *_ = extract_financial_observations("x", evidence_ledger=ledger)
    assert obs[0].source_evidence_status == "VERIFIED"


def test_belief_never_becomes_verified_in_currency_breakdown_status():
    ledger = [
        EvidenceItem(claim="A supplier dispute resulted in a USD 2,000 exposure.", status=EvidenceStatus.BELIEF, source="C1"),
        EvidenceItem(claim="A separate INR 50,000 expense was verified.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    by_curr = _breakdown_by_currency(res)
    assert by_curr["USD"].status != "VERIFIED"
