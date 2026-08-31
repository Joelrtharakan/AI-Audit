"""Pass 29 §8-§16: expected remediation expenditure must NEVER be classified as
incurred financial loss. The canonical LLM's `pricing_information` text is the
authority on which monetary values are remediation-cost inputs; the financial
extractor honours that instead of defaulting a verified amount to DIRECT_LOSS.
"""

from __future__ import annotations

from app.financial.engine import analyze_financial_exposure
from app.financial.compatibility import derive_cost_impact_from_financial_analysis
from app.financial.models import FinancialAmountType
from app.models.agent import EvidenceItem, EvidenceStatus


def _ev(claim):
    return EvidenceItem(claim=claim, status=EvidenceStatus.VERIFIED, source="t")


GUARD_FIND = (
    "During inspection of the production area, two machines were found with damaged "
    "safety guards. Engineering determined that both guards require replacement. Each "
    "replacement guard costs Rs.18,000 and installation requires 6 technician-hours per "
    "machine at an internal labor rate of Rs.1,200 per hour."
)
GUARD_EV = [
    _ev("Two machines were found with damaged safety guards."),
    _ev("Each replacement guard costs Rs.18,000; installation 6 technician-hours per machine "
        "at an internal labor rate of Rs.1,200 per hour."),
]
# What the canonical LLM's pricing_information carries for the guard finding.
GUARD_PRICING_CTX = [
    "Rs.18,000 per guard and Rs.1,200 per technician-hour",
    "cost of replacement guards and installation labor",
]


def test_case14_guard_pricing_is_not_direct_loss():
    r = analyze_financial_exposure(GUARD_FIND, evidence_ledger=GUARD_EV,
                                   remediation_cost_context=GUARD_PRICING_CTX)
    assert r.confirmed_impact.financial_factor in ("NOT_ESTABLISHED", None)
    assert r.confirmed_impact.verified_gross_exposure is None
    ci = derive_cost_impact_from_financial_analysis(r)
    assert ci is None or ci.financial_amount is None or ci.financial_amount.factor != "DIRECT_LOSS"


def test_case14_without_context_the_bug_would_reproduce():
    """Guardrail: proves the fix is doing the work (not that the bug never existed)."""
    r = analyze_financial_exposure(GUARD_FIND, evidence_ledger=GUARD_EV)
    assert r.confirmed_impact.financial_factor == "DIRECT_LOSS"  # the Pass-28 bug, no context


def test_case15_actual_scrap_loss_is_still_incurred_loss():
    find = ("Units were scrapped due to the deviation and financial records confirm an "
            "incurred loss of Rs.50,000.")
    ev = [_ev("Financial records confirm an incurred loss of Rs.50,000 from scrapped units.")]
    # No remediation pricing context -- this is a genuine incurred loss.
    r = analyze_financial_exposure(find, evidence_ledger=ev)
    assert r.confirmed_impact.verified_gross_exposure == 50000.0
    assert r.confirmed_impact.financial_factor in ("DIRECT_LOSS", "SCRAP_COST")


def test_case16_both_loss_and_remediation_kept_separate():
    find = ("100 units were scrapped (financial records confirm a Rs.50,000 loss). Separately, "
            "a machine guard requires replacement: each replacement guard costs Rs.18,000 and "
            "installation requires 6 technician-hours at Rs.1,200 per hour.")
    ev = [_ev("100 units scrapped; financial records confirm a Rs.50,000 loss."),
          _ev("Each replacement guard costs Rs.18,000; installation 6 technician-hours at Rs.1,200/hour.")]
    r = analyze_financial_exposure(
        find, evidence_ledger=ev,
        remediation_cost_context=["Rs.18,000 per guard and Rs.1,200 per technician-hour"],
    )
    # The incurred loss survives; the remediation-pricing amounts do not inflate it.
    assert r.confirmed_impact.verified_gross_exposure == 50000.0
    _types = {o.amount_type for o in
              __import__("app.financial.extractor", fromlist=["extract_financial_observations"])
              .extract_financial_observations(find, evidence_ledger=ev,
                  remediation_cost_context=["Rs.18,000 per guard and Rs.1,200 per technician-hour"])[0]}
    assert FinancialAmountType.REMEDIATION_COST in _types
    assert FinancialAmountType.DIRECT_LOSS in _types
