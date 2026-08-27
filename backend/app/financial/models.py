"""Data models for Evidence-Grounded Financial Exposure & Cost-of-Recurrence Analysis.

Strictly enforces:
  - Epistemic separation (VERIFIED GROSS EXPOSURE vs REPORTED FINANCIAL EXPOSURE vs CONFIRMED NET LOSS vs POTENTIAL UNRECOVERED EXPOSURE vs NOT_ESTABLISHED)
  - Provenance isolation: REPORTED/UNVERIFIED amounts and event counts never upgrade to VERIFIED.
  - Potential unverified events (e.g. "7 additional deliveries") tracked separately in `potential_additional_events` and excluded from annualization/recurrence.
  - Recovery safety: Never infer recovery = 0 from absence; never calculate net loss unless BOTH gross and recovery are verified.
  - NaN/Infinity firewall across all numeric fields.
  - Multi-dimensional financial confidence (Transaction, Amount, Recovery, Net Loss, Recurrence, Projection, Overall).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

from app.financial.semantic_models import CalculationTrace


def _check_finite_number(v: float | None) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    return None


class FinancialAmountType(str, Enum):
    DIRECT_LOSS = "DIRECT_LOSS"
    OVERPAYMENT = "OVERPAYMENT"
    DUPLICATE_PAYMENT = "DUPLICATE_PAYMENT"
    RECOVERY = "RECOVERY"
    NET_LOSS = "NET_LOSS"
    UNIT_COST = "UNIT_COST"
    REWORK_COST = "REWORK_COST"
    SCRAP_COST = "SCRAP_COST"
    DOWNTIME_COST = "DOWNTIME_COST"
    CUSTOMER_COMPENSATION = "CUSTOMER_COMPENSATION"
    PENALTY = "PENALTY"
    REMEDIATION_COST = "REMEDIATION_COST"
    POTENTIAL_EXPOSURE = "POTENTIAL_EXPOSURE"
    REVENUE_IMPACT = "REVENUE_IMPACT"
    COST_AVOIDANCE = "COST_AVOIDANCE"
    PREVENTION_COST = "PREVENTION_COST"
    OTHER = "OTHER"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


class FinancialEpistemicStatus(str, Enum):
    NO_FINANCIAL_IMPACT_IDENTIFIED = "NO_FINANCIAL_IMPACT_IDENTIFIED"
    FINANCIAL_IMPACT_REQUIRES_ASSESSMENT = "FINANCIAL_IMPACT_REQUIRES_ASSESSMENT"
    REQUIRES_VERIFICATION = "REQUIRES_VERIFICATION"
    VERIFIED_EXPOSURE = "VERIFIED_EXPOSURE"
    VERIFIED_GROSS_EXPOSURE = "VERIFIED_GROSS_EXPOSURE"
    REPORTED_EXPOSURE = "REPORTED_EXPOSURE"
    PARTIALLY_RECOVERED = "PARTIALLY_RECOVERED"
    FULLY_RECOVERED = "FULLY_RECOVERED"
    CONFIRMED_NET_LOSS = "CONFIRMED_NET_LOSS"
    POTENTIAL_UNRECOVERED_EXPOSURE = "POTENTIAL_UNRECOVERED_EXPOSURE"
    POTENTIAL_EXPOSURE = "POTENTIAL_EXPOSURE"
    ANNUALIZED_EXPOSURE = "ANNUALIZED_EXPOSURE"
    EXPECTED_ANNUAL_EXPOSURE = "EXPECTED_ANNUAL_EXPOSURE"
    FINANCIAL_CONFLICT_REQUIRES_RECONCILIATION = "FINANCIAL_CONFLICT_REQUIRES_RECONCILIATION"
    REQUIRES_RECONCILIATION = "REQUIRES_RECONCILIATION"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"
    # The evidence establishes what KIND of financial exposure exists (a
    # grounded cost factor, e.g. REWORK_COST) but not enough monetary
    # information (a rate, a total, a unit price) to calculate an amount.
    # Distinct from NOT_ASSESSABLE, which also covers "no factor could be
    # identified at all" -- this status specifically preserves the
    # positive information that WAS established.
    COST_FACTOR_IDENTIFIED_NOT_QUANTIFIABLE = "COST_FACTOR_IDENTIFIED_NOT_QUANTIFIABLE"
    # The LLM semantic interpretation stage could not produce a usable,
    # structurally valid interpretation (provider unavailable / invalid
    # output / incomplete). The system reports this honestly rather than
    # silently substituting a keyword/regex financial interpretation. No
    # monetary figure is produced in this state.
    FINANCIAL_SEMANTIC_UNAVAILABLE = "FINANCIAL_SEMANTIC_UNAVAILABLE"
    FINANCIAL_SEMANTIC_INCOMPLETE = "FINANCIAL_SEMANTIC_INCOMPLETE"


class FinancialConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


class RecoveryStatus(str, Enum):
    FULLY_RECOVERED = "FULLY_RECOVERED"
    PARTIALLY_RECOVERED = "PARTIALLY_RECOVERED"
    VERIFIED_RECOVERED = "VERIFIED_RECOVERED"
    VERIFIED_ZERO_RECOVERY = "VERIFIED_ZERO_RECOVERY"
    REQUIRES_VERIFICATION = "REQUIRES_VERIFICATION"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DimensionalConfidence(BaseModel):
    """Multi-dimensional confidence scores grounded strictly in evidence states."""
    transaction_confidence: FinancialConfidenceLevel = FinancialConfidenceLevel.NOT_ASSESSABLE
    amount_confidence: FinancialConfidenceLevel = FinancialConfidenceLevel.NOT_ASSESSABLE
    recovery_confidence: FinancialConfidenceLevel = FinancialConfidenceLevel.NOT_ASSESSABLE
    net_loss_confidence: FinancialConfidenceLevel = FinancialConfidenceLevel.NOT_ASSESSABLE
    recurrence_confidence: FinancialConfidenceLevel = FinancialConfidenceLevel.NOT_ASSESSABLE
    projection_confidence: FinancialConfidenceLevel = FinancialConfidenceLevel.NOT_ASSESSABLE
    overall: FinancialConfidenceLevel = FinancialConfidenceLevel.NOT_ASSESSABLE
    rationale: str = ""


class FinancialObservation(BaseModel):
    """A structured financial fact directly extracted from evidence."""
    observation_id: str
    amount: float | None = None
    amount_min: float | None = None
    amount_max: float | None = None
    currency: str = "INR"
    amount_type: FinancialAmountType = FinancialAmountType.POTENTIAL_EXPOSURE
    unit_amount: float | None = None
    quantity: float | None = None
    event_count: int | None = None
    potential_event_count: int | None = None
    event_id: str | None = None
    event_date: str | None = None
    observation_period_months: float | None = None
    source_evidence_ids: list[str] = Field(default_factory=list)
    source_claim_ids: list[str] = Field(default_factory=list)
    verification_status: Literal["VERIFIED", "REPORTED", "UNVERIFIED", "CONTRADICTED"] = "UNVERIFIED"
    source_reference: str | None = None
    recovery_amount: float | None = None
    recovery_status: RecoveryStatus = RecoveryStatus.REQUIRES_VERIFICATION
    affected_population: int | None = None
    notes: str | None = None
    financial_population: Literal["CURRENT_FINDING", "HISTORICAL", "OTHER"] = "CURRENT_FINDING"
    rate_unit_class: str | None = None
    quantity_unit_class: str | None = None
    # Provenance-only: the ORIGINAL EvidenceStatus string (e.g. "BELIEF",
    # "REPORTED", "VERIFIED", "UNVERIFIED", "INFERRED", "MIXED", ...) as
    # supplied by the evidence ledger, preserved purely for rendering
    # (e.g. distinguishing BELIEF from REPORTED in prose) -- NEVER used
    # for calculation eligibility or aggregation, which remain governed
    # exclusively by `verification_status`'s existing four-bucket model.
    source_evidence_status: str | None = None
    # True only when the statement explicitly marks this amount as a
    # TOTAL/combined/overall figure (e.g. "total remediation program
    # cost") -- the sole signal calculate_capa_payback trusts to prefer
    # one remediation observation over several others, rather than
    # guessing which of multiple stated amounts is authoritative.
    is_aggregate_total: bool = False

    @field_validator("amount", "amount_min", "amount_max", "unit_amount", "quantity", "recovery_amount", mode="before")
    @classmethod
    def validate_finite(cls, v: Any) -> float | None:
        return _check_finite_number(v)


class ConfirmedFinancialImpact(BaseModel):
    """Direct, evidence-confirmed exposure and loss breakdown."""
    verified_gross_exposure: float | None = None
    reported_financial_exposure: float | None = None
    reported_unit_exposure: float | None = None
    reported_event_count: int | None = None
    verified_event_count: int | None = None
    potential_additional_events: int | None = None
    potential_additional_exposure: float | None = None
    verified_recovery: float | None = None
    # A recovery amount stated in evidence whose status is REPORTED/
    # UNVERIFIED rather than VERIFIED -- never fabricated as VERIFIED and
    # never used in confirmed_net_loss (which remains reserved for the
    # both-sides-VERIFIED case), but preserved and surfaced rather than
    # silently discarded merely because it isn't yet VERIFIED.
    reported_recovery: float | None = None
    confirmed_net_loss: float | None = None
    potential_unrecovered_exposure: float | None = None
    recovery_status: RecoveryStatus = RecoveryStatus.REQUIRES_VERIFICATION
    financial_factor: str = "NOT_ESTABLISHED"
    currency: str = "INR"
    observed_events_count: int = 0
    calculation_formula: str = ""
    basis: str = ""
    source_evidence_ids: list[str] = Field(default_factory=list)
    is_confirmed_event: bool = False
    is_confirmed_loss: bool = False
    has_reported_exposure: bool = False
    # Cost-factor identification and monetary quantification are separate
    # dimensions -- a finding can clearly establish WHAT KIND of financial
    # exposure exists (financial_factor above) while providing no monetary
    # rate/amount to calculate HOW MUCH it is. QUANTIFIABLE means a
    # calculation was actually executed; NOT_QUANTIFIABLE means a factor
    # was identified but the evidence lacks the monetary input needed to
    # calculate an amount (quantification_blocker explains what's
    # missing); NOT_ASSESSABLE (the pre-existing default) covers every
    # other case, including "no financial factor could be established at
    # all". Never conflate NOT_QUANTIFIABLE with NOT_ASSESSABLE: the
    # former is real, grounded information for the auditor.
    quantification_status: Literal[
        "QUANTIFIED", "PARTIALLY_QUANTIFIED", "UNQUANTIFIED", "NOT_ASSESSABLE",
        "QUANTIFIABLE", "NOT_QUANTIFIABLE",  # legacy spellings, normalized below
    ] = "NOT_ASSESSABLE"
    quantification_blocker: str = ""

    @field_validator("quantification_status", mode="before")
    @classmethod
    def _canon_quant_status(cls, v: Any) -> Any:
        return {"QUANTIFIABLE": "QUANTIFIED", "NOT_QUANTIFIABLE": "UNQUANTIFIED"}.get(v, v)

    @field_validator("verified_gross_exposure", "reported_financial_exposure", "reported_unit_exposure", "potential_additional_exposure", "verified_recovery", "reported_recovery", "confirmed_net_loss", "potential_unrecovered_exposure", mode="before")
    @classmethod
    def validate_finite(cls, v: Any) -> float | None:
        return _check_finite_number(v)


class PotentialFinancialExposure(BaseModel):
    """Potential exposure where amounts, population, or rates are unverified."""
    lower_bound: float | None = None
    upper_bound: float | None = None
    currency: str = "INR"
    unit_exposure_range: str | None = None
    unverified_event_count: int | None = None
    basis: str = ""
    source_evidence_ids: list[str] = Field(default_factory=list)
    is_present: bool = False

    @field_validator("lower_bound", "upper_bound", mode="before")
    @classmethod
    def validate_finite(cls, v: Any) -> float | None:
        return _check_finite_number(v)


class AnnualizedExposure(BaseModel):
    """Annualized observed exposure based on a verified observation window."""
    annualized_amount: float | None = None
    annualized_range_min: float | None = None
    annualized_range_max: float | None = None
    currency: str = "INR"
    observation_period_months: float | None = None
    observed_exposure: float | None = None
    observed_event_rate_per_year: float | None = None
    calculation_formula: str = ""
    basis: str = ""
    projection_type: str = "ANNUALIZED_OBSERVED_EXPOSURE"
    qualification: str = (
        "Annualized from verified historical exposure over the observed period. "
        "This is an extrapolation of the observed rate and is not a confirmed future loss."
    )
    is_assessable: bool = False
    reason_if_not_assessable: str = ""

    @field_validator("annualized_amount", "annualized_range_min", "annualized_range_max", "observation_period_months", "observed_exposure", "observed_event_rate_per_year", mode="before")
    @classmethod
    def validate_finite(cls, v: Any) -> float | None:
        return _check_finite_number(v)


class RecurrenceAnalysis(BaseModel):
    """Expected annual loss derived strictly from evidence-backed recurrence frequency."""
    historical_events_per_year: float | None = None
    historical_events_range_min: float | None = None
    historical_events_range_max: float | None = None
    average_loss_per_event: float | None = None
    expected_annual_exposure: float | None = None
    expected_annual_range_min: float | None = None
    expected_annual_range_max: float | None = None
    currency: str = "INR"
    calculation_formula: str = ""
    basis: str = ""
    confidence: FinancialConfidenceLevel = FinancialConfidenceLevel.NOT_ASSESSABLE
    is_assessable: bool = False
    reason_if_not_assessable: str = ""
    distinction_note: str = (
        "Expected annual loss represents historical recurrence rate forecasting, "
        "distinct from observed rate annualization."
    )

    @field_validator("historical_events_per_year", "historical_events_range_min", "historical_events_range_max", "average_loss_per_event", "expected_annual_exposure", "expected_annual_range_min", "expected_annual_range_max", mode="before")
    @classmethod
    def validate_finite(cls, v: Any) -> float | None:
        return _check_finite_number(v)


class ScenarioEstimate(BaseModel):
    name: Literal["CONSERVATIVE", "EXPECTED", "HIGH"]
    amount: float | None = None
    currency: str = "INR"
    basis: str = ""

    @field_validator("amount", mode="before")
    @classmethod
    def validate_finite(cls, v: Any) -> float | None:
        return _check_finite_number(v)


class FinancialScenarioAnalysis(BaseModel):
    conservative: ScenarioEstimate | None = None
    expected: ScenarioEstimate | None = None
    high: ScenarioEstimate | None = None
    is_assessable: bool = False


class CostOfQualityBreakdown(BaseModel):
    internal_failure_cost: float | None = None
    external_failure_cost: float | None = None
    appraisal_cost: float | None = None
    prevention_cost: float | None = None
    currency: str = "INR"
    classified_components: list[str] = Field(default_factory=list)

    @field_validator("internal_failure_cost", "external_failure_cost", "appraisal_cost", "prevention_cost", mode="before")
    @classmethod
    def validate_finite(cls, v: Any) -> float | None:
        return _check_finite_number(v)


class CapaEconomicAnalysis(BaseModel):
    remediation_cost: float | None = None
    annual_avoided_exposure: float | None = None
    indicative_payback_years: float | None = None
    currency: str = "INR"
    qualification: str = "Indicative evidence-based economic comparison, not guaranteed ROI."
    is_assessable: bool = False
    # Set only when multiple, differing remediation-cost observations
    # exist with no explicit "total" marker establishing they are
    # additive components of one program -- never silently summed, never
    # arbitrarily picking the first. remediation_cost stays None in this
    # state; the conflicting values are preserved for the auditor.
    remediation_status: str = "NOT_APPLICABLE"
    conflicting_remediation_amounts: list[float] = Field(default_factory=list)
    # The verification_status ("VERIFIED"/"REPORTED"/"UNVERIFIED") of the
    # underlying remediation-cost observation(s) actually used -- distinct
    # from `remediation_status` above (which flags a reconciliation
    # conflict). remediation_cost itself is calculation-eligible and
    # surfaced regardless of evidence strength (a REPORTED/BELIEF-sourced
    # estimate is still a real estimate worth showing), but this field
    # lets the caller/renderer avoid presenting it as VERIFIED when it is
    # not.
    remediation_cost_status: str = "NOT_ASSESSABLE"

    @field_validator("remediation_cost", "annual_avoided_exposure", "indicative_payback_years", mode="before")
    @classmethod
    def validate_finite(cls, v: Any) -> float | None:
        return _check_finite_number(v)


class FinancialUncertainty(BaseModel):
    unresolved_factors: list[str] = Field(default_factory=list)
    evidence_needed_to_resolve: list[str] = Field(default_factory=list)


class CurrencyExposure(BaseModel):
    """Per-currency financial exposure breakdown -- populated when the
    evidence contains more than one currency and no authoritative
    conversion basis exists to consolidate them. Each entry is computed
    independently, within its own currency, using the same deterministic
    calculation rules as the single-currency path; currencies are never
    combined."""
    currency: str
    gross_amount: float | None = None
    reported_amount: float | None = None
    status: str = "NOT_ASSESSABLE"
    # Original evidence-ledger status (e.g. "BELIEF"), when every
    # contributing observation for this currency shares one -- purely for
    # rendering; `status` above remains the authoritative calculation
    # bucket. None when observations disagree or no single value applies.
    source_evidence_status: str | None = None
    financial_factor: str = "NOT_ESTABLISHED"
    source_evidence_ids: list[str] = Field(default_factory=list)
    # Historical annualization, computed independently per currency by
    # reusing the existing deterministic annualization calculation --
    # never combined numerically with another currency's figure or with
    # this currency's own current-finding gross_amount above (different
    # populations).
    historical_annualized_amount: float | None = None
    historical_observation_period_months: float | None = None
    historical_is_assessable: bool = False
    # Remediation cost -- a DISTINCT financial population from current
    # exposure/historical annualization/recovery (see FinancialAmountType
    # .REMEDIATION_COST / .PREVENTION_COST, already excluded from loss
    # aggregation in calculate_confirmed_impact). Computed per currency by
    # reusing the existing calculate_capa_payback, so a currency whose
    # ONLY fact is a remediation cost (no current/historical exposure in
    # that currency) remains visible rather than disappearing entirely.
    remediation_cost: float | None = None
    remediation_cost_status: str = "NOT_ASSESSABLE"
    # Same-currency-only indicative payback -- calculate_capa_payback
    # already requires remediation_cost and annual_avoided_exposure to
    # come from the SAME currency subset here, so a value only appears
    # when both operands are genuinely denominated in this currency.
    indicative_payback_years: float | None = None

    @field_validator("gross_amount", "reported_amount", "historical_annualized_amount", "historical_observation_period_months", "remediation_cost", "indicative_payback_years", mode="before")
    @classmethod
    def validate_finite(cls, v: Any) -> float | None:
        return _check_finite_number(v)


class FinancialAnalysisResult(BaseModel):
    """Authoritative consolidated result for Cost & Financial Exposure Analysis."""
    status: FinancialEpistemicStatus = FinancialEpistemicStatus.NOT_ASSESSABLE
    confidence: FinancialConfidenceLevel = FinancialConfidenceLevel.NOT_ASSESSABLE
    dimensional_confidence: DimensionalConfidence = Field(default_factory=DimensionalConfidence)
    currency: str = "INR"
    
    confirmed_impact: ConfirmedFinancialImpact = Field(default_factory=ConfirmedFinancialImpact)
    currency_breakdown: list[CurrencyExposure] = Field(default_factory=list)
    conversion_status: str = "NOT_APPLICABLE"
    potential_exposure: PotentialFinancialExposure = Field(default_factory=PotentialFinancialExposure)
    annualized_exposure: AnnualizedExposure = Field(default_factory=AnnualizedExposure)
    recurrence_analysis: RecurrenceAnalysis = Field(default_factory=RecurrenceAnalysis)
    scenario_analysis: FinancialScenarioAnalysis = Field(default_factory=FinancialScenarioAnalysis)
    cost_of_quality: CostOfQualityBreakdown = Field(default_factory=CostOfQualityBreakdown)
    capa_economics: CapaEconomicAnalysis = Field(default_factory=CapaEconomicAnalysis)
    
    assumptions: list[str] = Field(default_factory=list)
    uncertainty: FinancialUncertainty = Field(default_factory=FinancialUncertainty)

    # Provenance of THIS result -- which interpretation path actually
    # produced it. Never influences the calculation itself; exists purely
    # so an auditor (or an invariant) can see whether a number came from
    # the LLM-semantic path, the deterministic regex-extraction path, or
    # (last resort, should be rare) the legacy compatibility fallback.
    reasoning_source: Literal[
        "LLM_SEMANTIC", "DETERMINISTIC_REGEX", "DETERMINISTIC_FALLBACK_LEGACY", "NONE"
    ] = "NONE"

    # Honest status of the LLM semantic stage when reasoning_source is
    # LLM_SEMANTIC (see app.financial.semantic_models.FinancialSemanticStatus):
    # "OK" | "LLM_UNAVAILABLE" | "LLM_INVALID" | "LLM_INCOMPLETE" | "NO_EVIDENCE".
    # A non-OK value means NO monetary figure was produced -- the renderer
    # surfaces this instead of showing zeros.
    financial_semantic_status: str = "OK"
    # Whether a financial mechanism exists at all, independent of whether an
    # amount could be calculated: "NONE" | "POTENTIAL" | "MATERIAL" | "CONFIRMED".
    financial_relevance: str = "NONE"
    # Auditable per-calculation trace (LLM-proposed vs deterministically
    # executed result). Never an authoritative numeric source.
    calculation_traces: list[CalculationTrace] = Field(default_factory=list)

    important_qualification: str = (
        "Projected amounts represent evidence-based estimates under the stated assumptions "
        "and must not be interpreted as confirmed future losses."
    )
    assessment_reason: str = ""
    narrative: str = ""
