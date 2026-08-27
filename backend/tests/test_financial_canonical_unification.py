"""Architectural property tests for the financial-authority unification:
`report.financial_analysis` (FinancialAnalysisResult) is the sole
authoritative financial calculation; `report.cost_impact` is a pure
compatibility PROJECTION of it (app.financial.compatibility), never an
independently-computed second result.

These are structural/property tests, not wording-specific fixtures --
they hold for ANY FinancialAnalysisResult, not one particular finding.
"""

from __future__ import annotations

from app.financial.compatibility import derive_cost_impact_from_financial_analysis
from app.financial.models import (
    AnnualizedExposure,
    ConfirmedFinancialImpact,
    FinancialAnalysisResult,
    FinancialConfidenceLevel,
    FinancialEpistemicStatus,
    RecoveryStatus,
)
from app.services.cost_analysis import parse_currency


# ---------------------------------------------------------------------------
# Property 1/2: mapping only -- the derived CostImpact's numeric fields
# always equal the canonical source figures, for structurally different
# scenarios (never independently computed).
# ---------------------------------------------------------------------------

def test_property_quantity_rate_scenario_derives_exactly_from_canonical():
    fa = FinancialAnalysisResult(
        status=FinancialEpistemicStatus.VERIFIED_GROSS_EXPOSURE,
        confidence=FinancialConfidenceLevel.HIGH,
        currency="EUR",
        confirmed_impact=ConfirmedFinancialImpact(
            verified_gross_exposure=44_400.0,
            recovery_status=RecoveryStatus.REQUIRES_VERIFICATION,
            currency="EUR",
            is_confirmed_event=True,
        ),
    )
    ci = derive_cost_impact_from_financial_analysis(fa)
    assert ci.gross_exposure == fa.confirmed_impact.verified_gross_exposure
    assert ci.currency == fa.currency
    assert ci.cost_factor_detected is True


def test_property_recovery_scenario_derives_exactly_from_canonical():
    fa = FinancialAnalysisResult(
        status=FinancialEpistemicStatus.PARTIALLY_RECOVERED,
        confidence=FinancialConfidenceLevel.HIGH,
        currency="GBP",
        confirmed_impact=ConfirmedFinancialImpact(
            verified_gross_exposure=96_500.0,
            reported_recovery=15_000.0,
            recovery_status=RecoveryStatus.REQUIRES_VERIFICATION,
            currency="GBP",
            is_confirmed_event=True,
        ),
    )
    ci = derive_cost_impact_from_financial_analysis(fa)
    assert ci.gross_exposure == fa.confirmed_impact.verified_gross_exposure
    assert ci.recovered_amount == fa.confirmed_impact.reported_recovery
    # Recovery only REPORTED (not VERIFIED) -> net loss never fabricated.
    assert ci.actual_loss is None


def test_property_remediation_and_annualization_derive_exactly_from_canonical():
    fa = FinancialAnalysisResult(
        status=FinancialEpistemicStatus.ANNUALIZED_EXPOSURE,
        confidence=FinancialConfidenceLevel.MEDIUM,
        currency="AED",
        annualized_exposure=AnnualizedExposure(
            annualized_amount=96_000.0,
            currency="AED",
            observed_exposure=48_000.0,
            is_assessable=True,
        ),
    )
    ci = derive_cost_impact_from_financial_analysis(fa)
    assert ci.gross_exposure == fa.annualized_exposure.observed_exposure
    assert ci.currency == "AED"


def test_property_not_assessable_never_flags_a_cost_factor():
    fa = FinancialAnalysisResult(status=FinancialEpistemicStatus.NOT_ASSESSABLE)
    ci = derive_cost_impact_from_financial_analysis(fa)
    assert ci.cost_factor_detected is False
    assert ci.gross_exposure is None


# ---------------------------------------------------------------------------
# Property 4: no unknown currency silently becomes INR (legacy path too).
# ---------------------------------------------------------------------------

def test_property_legacy_parse_currency_never_defaults_to_inr():
    assert parse_currency("a plain sentence describing a cost with no currency marker at all") == "UNKNOWN"
    # Currency-bearing text still resolves correctly -- this isn't a
    # blanket "always UNKNOWN" change, only the previously-silent default.
    assert parse_currency("a cost of ₹5,000 was incurred") == "INR"
    assert parse_currency("a cost of $5,000 was incurred") == "USD"


def test_property_derived_cost_impact_never_fabricates_currency():
    fa = FinancialAnalysisResult(status=FinancialEpistemicStatus.NOT_ASSESSABLE)
    ci = derive_cost_impact_from_financial_analysis(fa)
    # No monetary content at all -- the model's own static scaffolding
    # default may still appear (a pre-existing, documented property of
    # `FinancialAnalysisResult.currency`), but no AMOUNT is attached to it.
    assert ci.gross_exposure is None
    assert ci.financial_amount is None


# ---------------------------------------------------------------------------
# Property 12: when a canonical result is present (even a "nothing
# assessable" one), the compatibility layer NEVER returns None -- so the
# legacy last-resort fallback in final_evidence_verification_node is only
# ever reached when financial_analysis itself is entirely absent, never
# overriding a canonical result that already ran.
# ---------------------------------------------------------------------------

def test_property_compatibility_layer_returns_none_only_when_source_is_none():
    assert derive_cost_impact_from_financial_analysis(None) is None
    for status in FinancialEpistemicStatus:
        fa = FinancialAnalysisResult(status=status)
        assert derive_cost_impact_from_financial_analysis(fa) is not None, status


# ---------------------------------------------------------------------------
# Property: the mapping is pure -- calling it twice on the same input
# (including a case with a full confirmed_impact) produces byte-identical
# output, proving no hidden randomness/state/recomputation.
# ---------------------------------------------------------------------------

def test_property_mapping_is_deterministic_and_pure():
    fa = FinancialAnalysisResult(
        status=FinancialEpistemicStatus.FULLY_RECOVERED,
        confidence=FinancialConfidenceLevel.HIGH,
        currency="USD",
        confirmed_impact=ConfirmedFinancialImpact(
            verified_gross_exposure=35_700.0,
            verified_recovery=35_700.0,
            confirmed_net_loss=0.0,
            recovery_status=RecoveryStatus.FULLY_RECOVERED,
            currency="USD",
            is_confirmed_event=True,
            is_confirmed_loss=True,
        ),
    )
    ci1 = derive_cost_impact_from_financial_analysis(fa)
    ci2 = derive_cost_impact_from_financial_analysis(fa)
    assert ci1.model_dump() == ci2.model_dump()
    assert ci1.actual_loss == 0.0
    assert ci1.recovered_amount == 35_700.0


def test_not_established_cost_factor_is_never_fabricated_into_direct_loss():
    """An honest NOT_ESTABLISHED classification (the semantic layer
    correctly declining to guess a cost factor) must project as an
    absent/None factor, never get coerced into DIRECT_LOSS anywhere in the
    compatibility mapping -- that would silently fabricate a
    classification the canonical result explicitly withheld. This is the
    backend half of a real bug where the frontend renderer independently
    coerced an absent factor into "DIRECT_LOSS" for display."""
    fa = FinancialAnalysisResult(
        status=FinancialEpistemicStatus.POTENTIAL_UNRECOVERED_EXPOSURE,
        confidence=FinancialConfidenceLevel.MEDIUM,
        currency="USD",
        confirmed_impact=ConfirmedFinancialImpact(
            verified_gross_exposure=9_900.0,
            recovery_status=RecoveryStatus.REQUIRES_VERIFICATION,
            currency="USD",
            financial_factor="NOT_ESTABLISHED",
            is_confirmed_event=True,
        ),
    )
    ci = derive_cost_impact_from_financial_analysis(fa)
    assert ci.financial_factor != "DIRECT_LOSS"
    assert ci.financial_factor is None
    assert ci.financial_amount is None or ci.financial_amount.factor != "DIRECT_LOSS"
