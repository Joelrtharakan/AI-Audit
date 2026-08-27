"""Regression coverage for two defect classes found while auditing
conflict detection and average-cost handling in the financial engine.

Bug 1 -- narrow conflict detection: the existing historical-recurrence
conflict check only detected disagreement on EVENT COUNT (holding cost
and period fixed). Two verified claims agreeing on event count and
period but disagreeing on cost-per-event, or agreeing on event count
and cost but disagreeing on period, were silently accepted and even
combined into a single (wrong) annualized figure instead of triggering
FINANCIAL_CONFLICT_REQUIRES_RECONCILIATION. Generalized to a pairwise
check: any two of {cost, event count, period} agreeing while the third
disagrees is a conflict, symmetric across all three fields.

Bug 2 -- average cost fabricating a single-event exposure: a statement
like "The average cost per incident was verified at INR 15,000" (no
explicit event count anywhere) was extracted as a per-event amount and,
via the calculator's count-or-1 fallback, treated as if it were the
gross exposure of exactly one verified event -- but an "average" is by
definition computed over 2+ instances, so this fabricates a population
size (1) the evidence never states. Fixed by downgrading such a fact's
verification status so it cannot surface as a VERIFIED gross exposure
without an explicit count. A stated average WITH an explicit count
still calculates normally.

Uses abstract, domain-neutral test fixtures.
"""

from __future__ import annotations

from app.financial.engine import analyze_financial_exposure
from app.financial.models import FinancialEpistemicStatus
from app.models.agent import EvidenceItem, EvidenceStatus


def test_conflicting_cost_per_event_requires_reconciliation():
    """Same event count, same period, DIFFERENT cost/event -- a conflict."""
    ledger = [
        EvidenceItem(claim="Historically, 8 incidents occurred over the past year, each verified at INR 15,000.", status=EvidenceStatus.VERIFIED, source="Log A"),
        EvidenceItem(claim="Historically, 8 incidents occurred over the past year, each verified at INR 20,000.", status=EvidenceStatus.VERIFIED, source="Log B"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.status == FinancialEpistemicStatus.FINANCIAL_CONFLICT_REQUIRES_RECONCILIATION


def test_conflicting_observation_period_requires_reconciliation():
    """Same event count, same cost/event, DIFFERENT period -- a conflict.
    Must never silently combine into a single wrong annualized figure."""
    ledger = [
        EvidenceItem(claim="Historically, 8 incidents occurred over the past 12 months, each verified at INR 15,000.", status=EvidenceStatus.VERIFIED, source="Log A"),
        EvidenceItem(claim="Historically, 8 incidents occurred over the past 6 months, each verified at INR 15,000.", status=EvidenceStatus.VERIFIED, source="Log B"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.status == FinancialEpistemicStatus.FINANCIAL_CONFLICT_REQUIRES_RECONCILIATION


def test_fully_consistent_historical_claims_still_annualize_correctly():
    """Contrast case: two sources agreeing on ALL THREE fields corroborate
    (not conflict) -- the pairwise conflict check must not false-positive
    on genuine agreement."""
    ledger = [
        EvidenceItem(claim="Historically, 8 incidents occurred over the past year, each verified at INR 15,000.", status=EvidenceStatus.VERIFIED, source="Log A"),
        EvidenceItem(claim="Historically, 8 incidents occurred over the past year, each verified at INR 15,000.", status=EvidenceStatus.VERIFIED, source="Log B"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.status == FinancialEpistemicStatus.ANNUALIZED_EXPOSURE
    assert res.annualized_exposure.annualized_amount == 120000.0


def test_average_cost_without_count_never_fabricates_single_event_exposure():
    finding = "The average cost per incident was verified at INR 15,000."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None


def test_average_cost_with_explicit_count_calculates_normally():
    finding = "The average cost per incident was verified at INR 15,000 across 8 verified incidents."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 120000.0


def test_non_average_per_event_amount_without_count_still_treated_as_single_event():
    """Contrast case: without the word 'average', a per-event amount with
    no stated count is still legitimately treated as describing the one
    event the finding is about -- this rule is scoped narrowly to
    'average' phrasing, not to per-event amounts in general."""
    finding = "Rework cost was verified at INR 15,000 per event."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 15000.0
