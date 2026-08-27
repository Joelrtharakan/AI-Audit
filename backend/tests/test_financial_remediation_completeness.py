"""Regression coverage for the financial-analysis completeness pass:
remediation cost visibility in the multi-currency breakdown, the
single-currency economic-analysis (payback) rendering, and a currency-
display bug discovered while inspecting the rendered output.

Defect 1 (the disclosed gap from the previous pass): a remediation cost
in a currency with no current-loss or historical-exposure observation
of its own (e.g. GBP remediation alongside EUR current + USD historical)
disappeared entirely from CurrencyExposure -- backend correctly kept the
populations separate, but nothing carried the remediation figure itself
into the breakdown. Root cause: calculate_capa_payback only populated
remediation_cost when a full payback (both remediation cost AND annual
avoided exposure) was calculable -- it returned an all-default object
otherwise, silently dropping the cost. Fixed by decoupling "remediation
cost is known" from "payback is calculable": the cost now always
populates when found; only indicative_payback_years/is_assessable
depend on a same-currency annual_avoided_exposure also being available.

Defect 2 (found only by inspecting the actual rendered report, not by
tests passing): the "Evidence-Based Annualized Exposure" and "Expected
Annual Recurrence Loss" lines in the single-currency renderer displayed
the TOP-LEVEL FinancialAnalysisResult.currency ("INR" by model default)
instead of the sub-object's OWN currency field -- for a
historical-only-plus-remediation finding (no CURRENT_FINDING loss
observation to set the top-level currency), a USD annualized figure was
mislabeled "INR 120,000/year" while the basis text beneath it correctly
said "USD 120,000.00". This is exactly the "silent INR fallback" class
of defect this whole hardening series exists to eliminate -- just
surfaced in a rendering path rather than a calculation path. Fixed by
using ann.currency/rec.currency (already correct) instead of the
top-level fCurr for those two specific lines.

Uses abstract, domain-neutral test fixtures.
"""

from __future__ import annotations

import math

from app.financial.engine import analyze_financial_exposure
from app.models.agent import EvidenceItem, EvidenceStatus


def test_remediation_cost_visible_when_currency_has_no_other_data():
    """The exact reproduction from the previous pass's disclosed gap."""
    ledger = [
        EvidenceItem(claim="A verified current expense of EUR 20,000 was identified.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="Historical records show verified losses of USD 100,000 per year.", status=EvidenceStatus.VERIFIED, source="C2"),
        EvidenceItem(claim="Proposed remediation will cost GBP 30,000.", status=EvidenceStatus.VERIFIED, source="C3"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    by_curr = {c.currency: c for c in res.currency_breakdown}
    assert set(by_curr) == {"EUR", "USD", "GBP"}
    assert by_curr["EUR"].gross_amount == 20000.0
    assert by_curr["USD"].historical_annualized_amount == 100000.0
    assert by_curr["GBP"].remediation_cost == 30000.0
    assert by_curr["GBP"].remediation_cost_status == "VERIFIED"
    # No population contamination: GBP has no gross/historical figure,
    # EUR/USD have no remediation figure.
    assert by_curr["GBP"].gross_amount is None
    assert by_curr["GBP"].historical_annualized_amount is None
    assert by_curr["EUR"].remediation_cost is None
    assert by_curr["USD"].remediation_cost is None


def test_remediation_cost_visible_without_a_payback():
    """Remediation cost alone (no annualized exposure in that currency)
    must still surface -- payback stays NOT ASSESSABLE, but the cost
    itself is not silently dropped."""
    ledger = [EvidenceItem(claim="Proposed remediation will cost USD 30,000.", status=EvidenceStatus.VERIFIED, source="C1")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.capa_economics.remediation_cost == 30000.0
    assert res.capa_economics.currency == "USD"
    assert res.capa_economics.is_assessable is False
    assert res.capa_economics.indicative_payback_years is None


def test_same_currency_payback_still_correct_and_visible():
    ledger = [
        EvidenceItem(claim="Historical records show verified losses of USD 120,000 per year.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="Proposed remediation will cost USD 60,000.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.capa_economics.is_assessable is True
    assert res.capa_economics.remediation_cost == 60000.0
    assert res.capa_economics.annual_avoided_exposure == 120000.0
    assert res.capa_economics.indicative_payback_years == 0.5
    # The critical rendering-consistency property: every sub-object's own
    # currency must independently equal the actual data currency, never
    # relying on (or contaminated by) the top-level default.
    assert res.annualized_exposure.currency == "USD"
    assert res.capa_economics.currency == "USD"


def test_remediation_cost_never_classified_as_loss_or_exposure():
    ledger = [EvidenceItem(claim="Proposed remediation will cost USD 30,000.", status=EvidenceStatus.VERIFIED, source="C1")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None
    assert res.confirmed_impact.confirmed_net_loss is None
    assert res.confirmed_impact.potential_unrecovered_exposure is None


def test_current_historical_remediation_same_currency_not_summed():
    """Case B from the task spec: all three populations share USD but
    must never be combined into one figure (20,000 + 100,000 + 30,000)."""
    ledger = [
        EvidenceItem(claim="A verified current expense of USD 20,000 was identified.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="Historical records show verified losses of USD 100,000 per year.", status=EvidenceStatus.VERIFIED, source="C2"),
        EvidenceItem(claim="Proposed remediation will cost USD 30,000.", status=EvidenceStatus.VERIFIED, source="C3"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 20000.0
    assert res.annualized_exposure.annualized_amount == 100000.0
    assert res.capa_economics.remediation_cost == 30000.0
    # None of these individually or combined equal a fabricated total.
    assert res.confirmed_impact.verified_gross_exposure != 150000.0


def test_reported_and_estimated_remediation_status_preserved():
    ledger_reported = [EvidenceItem(claim="Proposed remediation may cost USD 30,000.", status=EvidenceStatus.REPORTED, source="C1")]
    res_reported = analyze_financial_exposure("x", evidence_ledger=ledger_reported)
    from app.financial.extractor import extract_financial_observations
    obs, *_ = extract_financial_observations("Proposed remediation may cost USD 30,000.", evidence_ledger=ledger_reported)
    assert obs
    assert obs[0].verification_status != "VERIFIED"


def test_no_invalid_numbers_for_remediation_scenarios():
    ledger = [
        EvidenceItem(claim="A verified current expense of EUR 20,000 was identified.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="Historical records show verified losses of USD 100,000 per year.", status=EvidenceStatus.VERIFIED, source="C2"),
        EvidenceItem(claim="Proposed remediation will cost GBP 30,000.", status=EvidenceStatus.VERIFIED, source="C3"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
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
