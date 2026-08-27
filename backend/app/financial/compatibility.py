"""Compatibility projection: `CostImpact`/`FinancialAmount` from the
canonical `FinancialAnalysisResult`.

This module performs MAPPING ONLY. It never extracts financial meaning
from raw text and never computes an amount that isn't already present on
the canonical result -- `report.cost_impact`/`report.financial_amount`
exist for older consumers (a handful of `invariants.py` checks and the
frontend renderer's fallback branch) that predate `FinancialAnalysisResult`
and expect the legacy shape; this module lets them keep working without
maintaining a second, independently-computed financial interpretation.

`derive_cost_impact_from_financial_analysis` is the ONLY place a
`CostImpact` should be built once a `FinancialAnalysisResult` exists. The
legacy `app.services.cost_analysis.analyze_cost_and_financial_impact` is
retained solely as a last-resort fallback for the (rare) case a canonical
result is entirely unavailable -- see `final_evidence_verification.py`.
"""

from __future__ import annotations

from app.financial.models import FinancialAnalysisResult, FinancialEpistemicStatus, RecoveryStatus
from app.models.agent import CostImpact, FinancialAmount
from app.services.cost_analysis import format_currency_amount

# FinancialEpistemicStatus -> the legacy CostEvidenceStatus vocabulary
# (mapping only -- both are closed vocabularies already defined elsewhere;
# this table introduces no new classification logic).
_STATUS_TO_FINANCIAL_STATUS = {
    FinancialEpistemicStatus.NOT_ASSESSABLE: "REQUIRES_ASSESSMENT",
    FinancialEpistemicStatus.NO_FINANCIAL_IMPACT_IDENTIFIED: "UNKNOWN",
    FinancialEpistemicStatus.FINANCIAL_IMPACT_REQUIRES_ASSESSMENT: "REQUIRES_ASSESSMENT",
    FinancialEpistemicStatus.REQUIRES_VERIFICATION: "REPORTED",
    FinancialEpistemicStatus.VERIFIED_EXPOSURE: "VERIFIED",
    FinancialEpistemicStatus.VERIFIED_GROSS_EXPOSURE: "VERIFIED",
    FinancialEpistemicStatus.REPORTED_EXPOSURE: "REPORTED",
    FinancialEpistemicStatus.PARTIALLY_RECOVERED: "RECOVERABLE",
    FinancialEpistemicStatus.FULLY_RECOVERED: "RECOVERED",
    FinancialEpistemicStatus.CONFIRMED_NET_LOSS: "VERIFIED_LOSS",
    FinancialEpistemicStatus.POTENTIAL_UNRECOVERED_EXPOSURE: "POTENTIAL_EXPOSURE",
    FinancialEpistemicStatus.POTENTIAL_EXPOSURE: "POTENTIAL_EXPOSURE",
    FinancialEpistemicStatus.ANNUALIZED_EXPOSURE: "ESTIMATED",
    FinancialEpistemicStatus.EXPECTED_ANNUAL_EXPOSURE: "ESTIMATED",
    FinancialEpistemicStatus.FINANCIAL_CONFLICT_REQUIRES_RECONCILIATION: "REQUIRES_ASSESSMENT",
    FinancialEpistemicStatus.REQUIRES_RECONCILIATION: "REQUIRES_ASSESSMENT",
    FinancialEpistemicStatus.COST_FACTOR_IDENTIFIED_NOT_QUANTIFIABLE: "REQUIRES_ASSESSMENT",
    FinancialEpistemicStatus.FINANCIAL_SEMANTIC_UNAVAILABLE: "REQUIRES_ASSESSMENT",
    FinancialEpistemicStatus.FINANCIAL_SEMANTIC_INCOMPLETE: "REQUIRES_ASSESSMENT",
}

# RecoveryStatus -> the legacy `recoverability_status` vocabulary.
_RECOVERY_STATUS_TO_RECOVERABILITY_STATUS = {
    RecoveryStatus.FULLY_RECOVERED: "RECOVERED",
    RecoveryStatus.PARTIALLY_RECOVERED: "PARTIALLY_RECOVERED",
    RecoveryStatus.VERIFIED_RECOVERED: "RECOVERED",
    RecoveryStatus.VERIFIED_ZERO_RECOVERY: "IRRECOVERABLE",
    RecoveryStatus.REQUIRES_VERIFICATION: "REQUIRES_VERIFICATION",
    RecoveryStatus.NOT_APPLICABLE: "UNKNOWN",
}

_CONFIDENCE_MAP = {"HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW", "NOT_ASSESSABLE": "LOW"}

# FinancialAmountType (the canonical engine's domain-neutral classification,
# read off confirmed_impact.financial_factor as a plain string) -> the
# legacy CostFactorType vocabulary, where a clean one-to-one translation
# exists. General vocabulary mapping only -- no finding-specific values.
# Types with no clean legacy equivalent (e.g. DIRECT_LOSS, which is
# deliberately more general than any single legacy factor) are left
# untranslated rather than guessed.
_AMOUNT_TYPE_TO_LEGACY_FACTOR = {
    "REWORK_COST": "REWORK",
    "SCRAP_COST": "SCRAP",
    "DOWNTIME_COST": "DOWNTIME",
    "CUSTOMER_COMPENSATION": "CUSTOMER COMPENSATION",
    "PENALTY": "PENALTY",
    "DUPLICATE_PAYMENT": "DUPLICATE PAYMENT",
    "OVERPAYMENT": "OVERPAYMENT",
    "REMEDIATION_COST": "REMEDIATION",
    "PREVENTION_COST": "PREVENTION",
}


def derive_cost_impact_from_financial_analysis(
    financial_analysis: FinancialAnalysisResult | None,
) -> CostImpact | None:
    """Project the canonical `FinancialAnalysisResult` onto the legacy
    `CostImpact` shape. Pure field mapping -- no arithmetic, no text
    interpretation. Returns `None` only when there is nothing to project
    (`financial_analysis` itself is `None`), in which case the caller is
    responsible for its own last-resort fallback; this function never
    fabricates a result.
    """
    if financial_analysis is None:
        return None

    confirmed = financial_analysis.confirmed_impact
    _not_detected_statuses = {
        FinancialEpistemicStatus.NOT_ASSESSABLE,
        FinancialEpistemicStatus.NO_FINANCIAL_IMPACT_IDENTIFIED,
        FinancialEpistemicStatus.FINANCIAL_SEMANTIC_UNAVAILABLE,
        FinancialEpistemicStatus.FINANCIAL_SEMANTIC_INCOMPLETE,
    }
    cost_factor_detected = financial_analysis.status not in _not_detected_statuses

    financial_status = _STATUS_TO_FINANCIAL_STATUS.get(financial_analysis.status, "UNKNOWN")
    recoverability_status = _RECOVERY_STATUS_TO_RECOVERABILITY_STATUS.get(confirmed.recovery_status, "UNKNOWN")
    confidence = _CONFIDENCE_MAP.get(financial_analysis.confidence.value, "LOW")

    # Legacy `CostImpact` carries several overlapping names for what is
    # often the same underlying figure (gross_exposure/potential_exposure,
    # outstanding_amount/net_exposure/unrecovered_amount) -- a property of
    # the legacy model itself (see app.services.cost_analysis), not
    # something invented here. Mapped consistently from the same two
    # canonical source values (gross/reported exposure, verified/reported
    # recovery) rather than guessed per field.
    gross_exposure = confirmed.verified_gross_exposure if confirmed.verified_gross_exposure is not None else confirmed.reported_financial_exposure
    # The legacy model has no separate "this is a historical/annualized
    # rate, not a current-finding amount" concept -- when confirmed_impact
    # carries nothing at all but the observation was still verified as an
    # annualizable historical rate, the underlying observed amount is the
    # only monetary fact available, so it is surfaced here too rather than
    # left as a silent None the legacy consumer would misread as "nothing
    # detected".
    if gross_exposure is None and financial_analysis.annualized_exposure.is_assessable:
        gross_exposure = financial_analysis.annualized_exposure.observed_exposure
    reported_exposure = confirmed.reported_financial_exposure
    actual_loss = confirmed.confirmed_net_loss
    recovered_amount = confirmed.verified_recovery if confirmed.verified_recovery is not None else confirmed.reported_recovery
    # "Outstanding"/"net"/"unrecovered" all mean the same thing in legacy
    # usage: gross minus whatever recovery is known, defaulting recovery to
    # zero (not unknown) when gross is established but nothing about
    # recovery was ever stated -- matching the legacy analyzer's own
    # "nothing recovered is assumed fully outstanding" convention.
    if gross_exposure is not None:
        outstanding_amount = gross_exposure - (recovered_amount or 0.0)
    else:
        outstanding_amount = confirmed.potential_unrecovered_exposure
    unrecovered_amount = outstanding_amount

    display_amount = gross_exposure if gross_exposure is not None else reported_exposure
    _display_factor = _AMOUNT_TYPE_TO_LEGACY_FACTOR.get(confirmed.financial_factor, confirmed.financial_factor)
    financial_amount = (
        FinancialAmount(
            amount=display_amount,
            currency=financial_analysis.currency,
            factor=_display_factor if _display_factor != "NOT_ESTABLISHED" else None,
            source_claim_ids=list(confirmed.source_evidence_ids),
            support_status="VERIFIED" if gross_exposure is not None else ("REPORTED" if reported_exposure is not None else "UNKNOWN"),
            confidence=confidence,
        )
        if display_amount is not None
        else None
    )

    narrative = financial_analysis.narrative or financial_analysis.assessment_reason or None

    legacy_factor = _AMOUNT_TYPE_TO_LEGACY_FACTOR.get(confirmed.financial_factor, confirmed.financial_factor)
    if legacy_factor == "NOT_ESTABLISHED":
        legacy_factor = None

    # `potential_exposure`/`net_exposure` are the legacy model's own
    # near-duplicate names for gross exposure and outstanding exposure
    # respectively (see the comment above on the overlapping legacy
    # vocabulary) -- mapped from the same two canonical values, falling
    # back to the new model's own (differently-scoped) unverified-exposure
    # concept only when no gross figure is established at all.
    potential_exposure = gross_exposure if gross_exposure is not None else (
        financial_analysis.potential_exposure.upper_bound if financial_analysis.potential_exposure.is_present else None
    )
    net_exposure = outstanding_amount
    potential_cost_exposure = (
        format_currency_amount(potential_exposure, financial_analysis.currency)
        if potential_exposure is not None
        else None
    )

    return CostImpact(
        cost_factor_detected=cost_factor_detected,
        financial_factor=legacy_factor,
        financial_status=financial_status,
        currency=financial_analysis.currency,
        financial_amount=financial_amount,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        actual_loss=actual_loss,
        actual_loss_status="VERIFIED" if actual_loss is not None else "NOT_ESTABLISHED",
        potential_exposure=potential_exposure,
        potential_cost_exposure=potential_cost_exposure,
        outstanding_amount=outstanding_amount,
        recoverable_amount=unrecovered_amount,
        recovered_amount=recovered_amount,
        unrecovered_amount=unrecovered_amount,
        recoverability_status=recoverability_status,
        amount_confidence=confidence,
        classification_confidence=confidence,
        recovery_confidence=confidence,
        actual_loss_confidence=confidence,
        verified_cost=gross_exposure,
        reported_cost=reported_exposure,
        calculation_basis=confirmed.basis or None,
        assumptions=list(financial_analysis.assumptions),
        evidence_required=list(financial_analysis.uncertainty.evidence_needed_to_resolve),
        evidence_ids=list(confirmed.source_evidence_ids),
        confidence=confidence,
        narrative=narrative,
    )
