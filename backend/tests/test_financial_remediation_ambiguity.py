"""Regression coverage for multiple-remediation-cost safety, added during
the financial + semantic contamination hardening pass.

Defect reproduced: calculate_capa_payback picked only the FIRST
REMEDIATION_COST/PREVENTION_COST observation it encountered, silently
discarding any other differing remediation estimate -- "Estimated
remediation cost is USD 20,000" + "Another verified remediation
estimate is USD 30,000" produced remediation_cost=20,000 with no trace
that a conflicting 30,000 figure also existed.

Fix: calculate_capa_payback now distinguishes three cases:
  A. Exactly one remediation observation (or several that all
     corroborate the SAME value) -- use it, unchanged from before.
  B. An explicit "total/combined/overall ... cost" statement among
     several differing observations -- use that total exclusively
     (never additionally summing the components it already covers).
  C. Multiple DIFFERING remediation amounts with no explicit total
     marker -- never summed, never picked arbitrarily; flagged as
     remediation_status="REQUIRES_RECONCILIATION" with the distinct
     conflicting amounts preserved for the auditor.

This applies uniformly whether the multiple amounts are framed as
"alternatives" (Option A / Option B) or as ambiguous/conflicting
estimates -- the evidence provides no reliable structural signal to
tell these apart, so both must fail closed to reconciliation-required
rather than guessing.

Uses abstract, domain-neutral test fixtures.
"""

from __future__ import annotations

from app.financial.engine import analyze_financial_exposure
from app.models.agent import EvidenceItem, EvidenceStatus


def test_single_remediation_cost_unaffected():
    ledger = [EvidenceItem(claim="Proposed remediation will cost USD 60,000.", status=EvidenceStatus.VERIFIED, source="C1")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.capa_economics.remediation_cost == 60000.0
    assert res.capa_economics.remediation_status == "NOT_APPLICABLE"
    assert res.capa_economics.conflicting_remediation_amounts == []


def test_explicit_total_selected_over_components():
    """Case A: an explicit 'total remediation program cost' statement is
    used exclusively -- not additionally summed with the components it
    already covers."""
    ledger = [
        EvidenceItem(claim="Implementation cost is USD 20,000.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="Training cost is USD 5,000.", status=EvidenceStatus.VERIFIED, source="C2"),
        EvidenceItem(claim="The total remediation program cost is USD 28,000.", status=EvidenceStatus.VERIFIED, source="C3"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.capa_economics.remediation_cost == 28000.0
    assert res.capa_economics.remediation_cost != 53000.0  # 20000 + 5000 + 28000, never fabricated


def test_alternative_remediation_options_not_summed_or_arbitrarily_chosen():
    """Case B: explicit alternatives must never be added together, and
    the first must never be silently preferred."""
    ledger = [
        EvidenceItem(claim="Option A remediation cost is USD 20,000.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="Option B remediation cost is USD 35,000.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.capa_economics.remediation_cost is None
    assert res.capa_economics.remediation_status == "REQUIRES_RECONCILIATION"
    assert res.capa_economics.conflicting_remediation_amounts == [20000.0, 35000.0]


def test_ambiguous_conflicting_estimates_flagged_not_chosen():
    """Case C: the exact reproduction -- two differing verified
    remediation estimates with no total marker must never silently
    resolve to the first one encountered."""
    ledger = [
        EvidenceItem(claim="Estimated remediation cost is USD 20,000.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="Another verified remediation estimate is USD 30,000.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.capa_economics.remediation_cost is None
    assert res.capa_economics.remediation_cost != 20000.0
    assert res.capa_economics.remediation_status == "REQUIRES_RECONCILIATION"
    assert res.capa_economics.conflicting_remediation_amounts == [20000.0, 30000.0]


def test_identical_corroborating_remediation_amounts_still_resolve():
    """Contrast case: two sources stating the IDENTICAL remediation cost
    corroborate one fact, not a conflict -- must still resolve normally."""
    ledger = [
        EvidenceItem(claim="Proposed remediation will cost USD 60,000.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="Proposed remediation will cost USD 60,000.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.capa_economics.remediation_cost == 60000.0
    assert res.capa_economics.remediation_status == "NOT_APPLICABLE"
