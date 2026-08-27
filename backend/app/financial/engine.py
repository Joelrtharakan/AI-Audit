"""Top-level orchestrator for Cost & Financial Exposure Analysis.

Coordinates:
  - Fact extraction & range parsing
  - Currency isolation & conflict handling
  - Deterministic calculations
  - Single authoritative assessment state derivation
  - Multi-dimensional confidence derivation
"""

from __future__ import annotations

import logging
from app.financial.calculator import (
    calculate_annualized_exposure,
    calculate_capa_payback,
    calculate_confirmed_impact,
    calculate_cost_of_quality,
    calculate_potential_exposure,
    calculate_recurrence_exposure,
    calculate_scenarios,
    derive_dimensional_confidence,
)
from app.financial.extractor import extract_financial_observations
from app.financial.models import (
    CurrencyExposure,
    DimensionalConfidence,
    FinancialAmountType,
    FinancialAnalysisResult,
    FinancialConfidenceLevel,
    FinancialEpistemicStatus,
    FinancialUncertainty,
    RecoveryStatus,
)
from app.models.agent import EvidenceClaim, EvidenceItem

logger = logging.getLogger(__name__)


def _build_result_from_observations(
    observations: list,
    observation_period_months: float | None = None,
    annual_event_frequency: float | None = None,
    frequency_range: tuple[float, float] | None = None,
) -> FinancialAnalysisResult:
    """Run the authoritative deterministic calculation + assessment-state
    derivation over an already-validated, single-population,
    single-currency `FinancialObservation` list.

    This is the SOLE numeric-authority path for BOTH the regex-extraction
    engine (`analyze_financial_exposure` below) and the LLM semantic
    engine (`app.financial.semantic_engine`) -- neither path performs its
    own arithmetic; both materialize observations and call this function,
    so a number displayed to the auditor always came from the same,
    already-tested calculator regardless of which upstream interpretation
    path produced the observations.
    """
    # Deterministic calculations
    confirmed = calculate_confirmed_impact(observations)
    potential = calculate_potential_exposure(observations)
    annualized = calculate_annualized_exposure(observations, observation_period_months=observation_period_months)
    recurrence = calculate_recurrence_exposure(
        observations,
        historical_frequency_per_year=annual_event_frequency,
        frequency_range=frequency_range,
    )
    scenarios = calculate_scenarios(confirmed, potential, annualized, recurrence)
    coq = calculate_cost_of_quality(observations)
    capa_econ = calculate_capa_payback(
        observations,
        annual_avoided_exposure=recurrence.expected_annual_exposure or annualized.annualized_amount,
    )

    dim_conf = derive_dimensional_confidence(
        confirmed, potential, annualized, recurrence, observations
    )

    # Authoritative Assessment State derivation
    if confirmed.recovery_status == RecoveryStatus.FULLY_RECOVERED:
        status = FinancialEpistemicStatus.FULLY_RECOVERED
    elif confirmed.recovery_status == RecoveryStatus.PARTIALLY_RECOVERED:
        status = FinancialEpistemicStatus.PARTIALLY_RECOVERED
    elif confirmed.is_confirmed_loss and confirmed.confirmed_net_loss is not None:
        status = FinancialEpistemicStatus.CONFIRMED_NET_LOSS
    elif confirmed.potential_unrecovered_exposure is not None:
        status = FinancialEpistemicStatus.POTENTIAL_UNRECOVERED_EXPOSURE
    elif confirmed.is_confirmed_event and confirmed.verified_gross_exposure is not None:
        status = FinancialEpistemicStatus.VERIFIED_EXPOSURE
    elif confirmed.has_reported_exposure:
        status = FinancialEpistemicStatus.REQUIRES_VERIFICATION
    elif recurrence.is_assessable:
        status = FinancialEpistemicStatus.EXPECTED_ANNUAL_EXPOSURE
    elif annualized.is_assessable:
        status = FinancialEpistemicStatus.ANNUALIZED_EXPOSURE
    elif potential.is_present:
        status = FinancialEpistemicStatus.POTENTIAL_EXPOSURE
    else:
        status = FinancialEpistemicStatus.NOT_ASSESSABLE

    # Explicit assumptions
    assumptions = []
    if annualized.is_assessable:
        assumptions.append(f"Annualization assumes the observed rate over {annualized.observation_period_months:g} month(s) remains constant.")
    if recurrence.is_assessable and recurrence.historical_events_per_year:
        assumptions.append(f"Recurrence forecasting assumes the historical event frequency ({recurrence.historical_events_per_year:g}/year) continues.")
    if confirmed.is_confirmed_loss and confirmed.verified_recovery is not None:
        assumptions.append(f"Net loss reflects verified gross exposure minus verified recovery ({confirmed.currency} {confirmed.verified_recovery:,.2f}).")
    elif confirmed.potential_unrecovered_exposure is not None:
        assumptions.append("Recovery is unverified; net loss is not established and potential unrecovered exposure is up to the verified gross amount.")
    if confirmed.has_reported_exposure:
        assumptions.append("Reported financial amounts and event counts have not been independently verified with authoritative accounting records.")
    if confirmed.potential_additional_events:
        assumptions.append(f"The {confirmed.potential_additional_events} potentially affected deliveries remain unverified and are excluded from verified exposure and recurrence calculations.")

    # Uncertainty & required evidence
    uncertain_factors = []
    needed_evidence = []
    if not confirmed.is_confirmed_loss:
        if confirmed.verified_gross_exposure is not None:
            uncertain_factors.append("Recovery amount is unverified (requires verification).")
            needed_evidence.append("Recovery / payment reversal records or supplier credit notes.")
        elif confirmed.has_reported_exposure:
            uncertain_factors.append("Reported monetary exposure and event counts are unverified.")
            needed_evidence.append("Transaction-level financial records and supplier nonconformity verification.")
            if confirmed.potential_additional_events:
                needed_evidence.append(f"Verification of the {confirmed.potential_additional_events} potentially affected deliveries.")
        else:
            uncertain_factors.append("Actual direct loss has not been independently verified with records.")
            needed_evidence.append("Official financial reconciliation / accounting transaction logs.")
    if not annualized.is_assessable:
        uncertain_factors.append("Observation period or verified event population is unavailable for annualization.")
        needed_evidence.append("Verified timeframe or date range of the observed nonconformities.")
    if not recurrence.is_assessable:
        uncertain_factors.append("Historical event frequency is unavailable for annual loss forecasting.")
        needed_evidence.append("Historical audit logs or CAPA database incident counts for the past 12-24 months.")

    return FinancialAnalysisResult(
        status=status,
        confidence=dim_conf.overall,
        dimensional_confidence=dim_conf,
        # `observations` here is already single-currency-validated (see this
        # function's own docstring) -- the observations' own currency is the
        # authoritative source, never a static default, since sub-analyses
        # like `confirmed` can be entirely empty (e.g. a historical-only
        # finding) while still carrying their model's own default currency.
        currency=observations[0].currency,
        confirmed_impact=confirmed,
        potential_exposure=potential,
        annualized_exposure=annualized,
        recurrence_analysis=recurrence,
        scenario_analysis=scenarios,
        cost_of_quality=coq,
        capa_economics=capa_econ,
        assumptions=assumptions,
        uncertainty=FinancialUncertainty(
            unresolved_factors=uncertain_factors,
            evidence_needed_to_resolve=needed_evidence,
        ),
        assessment_reason="Financial assessment completed based on available structured evidence.",
    )


def analyze_financial_exposure(
    finding_text: str,
    evidence_ledger: list[EvidenceItem] | None = None,
    evidence_claims: list[EvidenceClaim] | None = None,
    recurrence_data: dict | None = None,
    observation_period_months: float | None = None,
    annual_event_frequency: float | None = None,
    frequency_range: tuple[float, float] | None = None,
) -> FinancialAnalysisResult:
    """Analyze financial exposure and cost-of-recurrence strictly from evidence."""
    observations, has_conflict, currency_conflicts = extract_financial_observations(
        finding_text,
        evidence_ledger=evidence_ledger,
        evidence_claims=evidence_claims,
    )

    if not observations:
        if currency_conflicts:
            return FinancialAnalysisResult(
                status=FinancialEpistemicStatus.NOT_ASSESSABLE,
                confidence=FinancialConfidenceLevel.NOT_ASSESSABLE,
                conversion_status="CONFLICT",
                dimensional_confidence=DimensionalConfidence(
                    rationale="An explicit currency code conflicts with a currency symbol in the same amount; the amount is excluded from calculation."
                ),
                assessment_reason=(
                    f"Currency conflict detected ({'; '.join(currency_conflicts)}): an explicit currency "
                    "code and a currency symbol resolve to different currencies for the same amount. "
                    "No calculation, conversion, or exposure figure is produced from a conflicting amount."
                ),
                uncertainty=FinancialUncertainty(
                    unresolved_factors=["Currency identity"],
                    evidence_needed_to_resolve=["Clarification of the intended currency for the conflicting amount."],
                ),
            )
        return FinancialAnalysisResult(
            status=FinancialEpistemicStatus.NOT_ASSESSABLE,
            confidence=FinancialConfidenceLevel.NOT_ASSESSABLE,
            dimensional_confidence=DimensionalConfidence(
                rationale="No verified financial losses, costs, or exposure amounts were detected in the evidence."
            ),
            assessment_reason="No verified financial evidence supports quantification.",
            uncertainty=FinancialUncertainty(
                unresolved_factors=["Direct monetary loss", "Remediation cost", "Recurrence frequency"],
                evidence_needed_to_resolve=["Invoices, transaction logs, scrap reports, or rework timesheets"],
            ),
        )

    # Multi-currency check: if conflicting currencies without conversion rate, mark NOT_ASSESSABLE
    # for the CONSOLIDATED figure, but never discard the underlying facts --
    # each currency's own exposure is computed independently (reusing the
    # same deterministic calculate_confirmed_impact used for the
    # single-currency path, just scoped to that currency's observations)
    # and preserved in currency_breakdown. No conversion or arithmetic
    # ever crosses a currency boundary.
    currencies = {o.currency for o in observations if o.currency}
    if len(currencies) > 1:
        breakdown: list[CurrencyExposure] = []
        for curr in sorted(currencies):
            curr_obs = [o for o in observations if o.currency == curr]
            curr_impact = calculate_confirmed_impact(curr_obs)
            # Historical annualization computed independently within this
            # currency's own observations -- reuses the same deterministic
            # calculate_annualized_exposure used by the single-currency
            # path; never combined with another currency's figure or with
            # this currency's own current-finding gross_amount (different
            # populations, kept separate exactly as in the single-currency
            # case).
            curr_annualized = calculate_annualized_exposure(curr_obs)
            # Remediation cost: a population distinct from current
            # exposure/historical annualization/recovery (never included
            # in calculate_confirmed_impact's loss aggregation). Reuses
            # the existing calculate_capa_payback, scoped to this
            # currency's own observations and its own annualized figure
            # -- so a same-currency payback can surface here, while a
            # cross-currency pairing (remediation cost in a currency with
            # no annualized exposure of its own) naturally yields
            # is_assessable=False without any extra currency check,
            # since annual_avoided_exposure would be None/foreign.
            curr_capa = calculate_capa_payback(curr_obs, annual_avoided_exposure=curr_annualized.annualized_amount)
            _remediation_status = "NOT_ASSESSABLE"
            for _o in curr_obs:
                if _o.amount_type in (FinancialAmountType.REMEDIATION_COST, FinancialAmountType.PREVENTION_COST) and _o.amount is not None:
                    _remediation_status = _o.verification_status
                    break
            if curr_impact.verified_gross_exposure is not None:
                status = "VERIFIED"
            elif curr_impact.reported_financial_exposure is not None:
                status = "REPORTED" if curr_impact.has_reported_exposure else "UNVERIFIED"
            else:
                status = "NOT_ASSESSABLE"
            # Original evidence-ledger status, preserved for rendering
            # only, when every observation contributing an amount in this
            # currency shares one -- never used to influence `status`
            # above, which remains the sole authoritative calculation
            # bucket.
            _orig_statuses = {o.source_evidence_status for o in curr_obs if o.source_evidence_status}
            _orig_status = next(iter(_orig_statuses)) if len(_orig_statuses) == 1 else None
            breakdown.append(CurrencyExposure(
                currency=curr,
                gross_amount=curr_impact.verified_gross_exposure,
                reported_amount=curr_impact.reported_financial_exposure,
                status=status,
                source_evidence_status=_orig_status,
                financial_factor=curr_impact.financial_factor,
                source_evidence_ids=curr_impact.source_evidence_ids,
                historical_annualized_amount=curr_annualized.annualized_amount,
                historical_observation_period_months=curr_annualized.observation_period_months,
                historical_is_assessable=curr_annualized.is_assessable,
                remediation_cost=curr_capa.remediation_cost,
                remediation_cost_status=_remediation_status,
                indicative_payback_years=curr_capa.indicative_payback_years if curr_capa.is_assessable else None,
            ))
        return FinancialAnalysisResult(
            status=FinancialEpistemicStatus.NOT_ASSESSABLE,
            confidence=FinancialConfidenceLevel.NOT_ASSESSABLE,
            currency_breakdown=breakdown,
            conversion_status="NOT_AVAILABLE",
            dimensional_confidence=DimensionalConfidence(
                rationale="Multiple currencies detected without an authoritative exchange rate."
            ),
            assessment_reason=f"Multiple currencies ({', '.join(sorted(currencies))}) present without authoritative conversion basis.",
            uncertainty=FinancialUncertainty(
                unresolved_factors=["Currency conversion basis"],
                evidence_needed_to_resolve=["Official corporate exchange rate for multi-currency reconciliation."],
            ),
        )

    # Contradictory evidence check
    if has_conflict:
        return FinancialAnalysisResult(
            status=FinancialEpistemicStatus.FINANCIAL_CONFLICT_REQUIRES_RECONCILIATION,
            confidence=FinancialConfidenceLevel.LOW,
            dimensional_confidence=DimensionalConfidence(
                overall=FinancialConfidenceLevel.LOW,
                rationale="Authoritative evidence sources state conflicting financial amounts for the same event.",
            ),
            assessment_reason="Contradictory financial amounts identified in evidence. Reconciliation required before quantification.",
            uncertainty=FinancialUncertainty(
                unresolved_factors=["Discrepant transaction / loss figures"],
                evidence_needed_to_resolve=["Audited accounting ledger or final financial reconciliation report."],
            ),
        )

    return _build_result_from_observations(
        observations,
        observation_period_months=observation_period_months,
        annual_event_frequency=annual_event_frequency,
        frequency_range=frequency_range,
    )
