"""Structured schema for the LLM semantic-understanding layer of REMEDIATION
COST ESTIMATION.

Deliberately SEPARATE from `app.financial.semantic_models` (financial exposure
== money already lost/exposed) and from `app.remediation.models`
(`RemediationCostResult`, the deterministically-assembled canonical result).

The LLM's job here is to understand what remediation a finding implies, what
implementation activities that entails, and what cost drivers are relevant --
and to PROPOSE calculations. It never computes a final number: every arithmetic
result is re-derived deterministically in `app.remediation.calculator` after
`app.remediation.validator` has independently checked structure and provenance.

Nothing in this module performs a calculation or a keyword lookup. It only
describes what the LLM is allowed to say. There are NO domain-specific fields,
enums, or vocabularies -- `remediation_type` and `cost_category` are free
descriptive text.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# How the LLM stage terminated -- an honest, provider-independent status the
# engine maps to an auditor-safe result. Mirrors
# `app.financial.semantic_models.FinancialSemanticStatus`.
RemediationSemanticStatus = Literal[
    "OK", "LLM_UNAVAILABLE", "LLM_INVALID", "LLM_INCOMPLETE", "NO_EVIDENCE"
]

Confidence = Literal["HIGH", "MEDIUM", "LOW"]

# Per-value provenance vocabulary (spec section 5). VERIFIED/REPORTED mean an
# evidence item actually stated the value; ESTIMATED is a defensible inference;
# ASSUMED is an explicit placeholder the LLM chose (and must justify);
# NOT_ESTABLISHED means no basis at all. The validator re-checks every
# VERIFIED/REPORTED claim against real evidence and downgrades unsupported ones.
CostBasisStr = Literal["VERIFIED", "REPORTED", "ESTIMATED", "ASSUMED", "NOT_ESTABLISHED"]
# EVIDENCED  -- an evidence item states the quantity outright.
# DERIVED    -- the LLM combined two or more explicit evidenced values with a
#               transparent relationship it must state (e.g. "2 machines x 6
#               h/machine = 12 h"). This is the LLM's reasoning, never code's:
#               deterministic code only executes the arithmetic the LLM supplies
#               in the linked calculation plan (spec Pass 32 §5).
# ASSUMED / NOT_ESTABLISHED -- placeholder / no basis; never yields a number.
QuantityBasisStr = Literal["EVIDENCED", "DERIVED", "ASSUMED", "NOT_ESTABLISHED"]

# The semantic ROLE of this component's monetary figure relative to the
# remediation plan (spec sections 4, 6, 7, 19). The deterministic calculator
# uses this -- and this alone -- to decide whether amounts are summed, bracketed
# as a range, or reconciled against each other. It is NOT a keyword signal:
#   PER_*         a per-unit / per-hour / per-event RATE (multiply by quantity)
#   COMPONENT     one additive line item of the implementation (summed)
#   SUBTOTAL      a partial roll-up of sibling COMPONENTs (reconciled, not re-added)
#   TOTAL         a stated COMPLETE implementation total (the authoritative figure)
#   ALTERNATIVE   one of several mutually-exclusive options (bracketed, never summed)
AmountType = Literal[
    "PER_QUANTITY", "PER_HOUR", "PER_UNIT", "PER_EVENT", "PER_IMPLEMENTATION",
    "TOTAL", "SUBTOTAL", "COMPONENT", "ALTERNATIVE",
]

Recurrence = Literal["ONE_TIME", "RECURRING"]

DerivedFrom = Literal[
    "FINDING", "EVIDENCE", "ROOT_CAUSE_HYPOTHESIS", "RECOMMENDED_CAPA", "IMPACT", "CONTEXT",
]

# The semantic role the cost LLM assigns to an activity it identified itself
# (fallback path, when no canonical `SemRemediationAction.disposition` exists).
# Mirrors the canonical dispositions. Empty = unclassified (legacy behaviour).
ActivityDisposition = Literal[
    "", "IMMEDIATE_CORRECTION", "CONTAINMENT", "CORRECTIVE_ACTION",
    "CONDITIONAL_SYSTEMIC", "EFFECTIVENESS_CHECK", "INVESTIGATION",
]

CalcOperation = Literal["MULTIPLY", "SUM", "SUBTRACT", "DIVIDE"]

Produces = Literal["LOW", "MOST_LIKELY", "HIGH", "COMPONENT_AMOUNT"]

OverallStatus = Literal["EVIDENCE_BACKED", "ASSUMPTION_BASED", "NOT_ASSESSABLE"]

Estimability = Literal[
    "ESTIMABLE", "BOUNDED_ONLY", "SINGLE_VERIFIED_COST", "NOT_ASSESSABLE",
]

# Concise machine reasons the LLM may give for NOT_ASSESSABLE; the renderer maps
# these to professional user-facing text and NEVER shows an internal diagnostic.
NotAssessableReason = Literal[
    "IMPLEMENTATION_SCOPE_UNKNOWN", "QUANTITY_UNKNOWN", "PRICING_BASIS_UNAVAILABLE",
    "REMEDIATION_NOT_DEFINED", "CONFLICTING_EVIDENCE", "INSUFFICIENT_EVIDENCE", "",
]


def _finite(v: Any) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv) or math.isinf(fv):
        return None
    return fv


class RemediationStrategy(BaseModel):
    """The LLM's semantic reading of what the finding requires -- derived from
    MEANING, never keyword matching. `remediation_type` is free descriptive
    text, never an enum and never an operation licence."""

    condition_identified: str | None = None
    deficient_requirement: str | None = None
    established_basis: str | None = None      # what the evidence actually establishes
    hypothetical_basis: str | None = None     # what remains contingent / unproven
    remediation_summary: str | None = None
    remediation_type: str | None = None
    alternative_strategies: list[str] = Field(default_factory=list)
    interpretation_confidence: Confidence = "LOW"


class ImplementationActivity(BaseModel):
    activity_id: str
    description: str
    rationale: str = ""
    derived_from: DerivedFrom = "FINDING"
    disposition: ActivityDisposition = ""
    depends_on_root_cause: bool = False
    source_reference_ids: list[str] = Field(default_factory=list)
    is_hypothetical: bool = False


class RemediationCostComponent(BaseModel):
    """One semantic cost driver. `cost_category` is free descriptive text -- the
    prompt lists example categories but forces none. A component is only valid
    when the remediation context supports it (spec section 4)."""

    component_id: str
    description: str
    activity_ids: list[str] = Field(default_factory=list)
    cost_category: str = "other"
    # What the monetary value semantically IS (spec Pass 34 §21 / Pass 35 §1).
    # The LLM MUST classify every monetary input. The default is the
    # fail-closed NOT_ESTABLISHED: an omitted classification NEVER silently
    # becomes a remediation cost -- the validator strips the number for that
    # one component (keeping it as an unpriced driver) unless a VERIFIED /
    # REPORTED cited evidence basis independently anchors it. Deterministic
    # code never recovers the missing meaning from text.
    # REMEDIATION_COST / UNIT_RATE / QUOTED_PRICE / BUDGET / ESTIMATE are
    # priceable remediation inputs; OBSERVED_FINANCIAL_LOSS /
    # HISTORICAL_EXPENDITURE describe the finding, not the fix.
    value_kind: Literal[
        "REMEDIATION_COST", "UNIT_RATE", "QUOTED_PRICE", "BUDGET", "ESTIMATE",
        "OBSERVED_FINANCIAL_LOSS", "HISTORICAL_EXPENDITURE", "OTHER", "NOT_ESTABLISHED",
    ] = "NOT_ESTABLISHED"
    quantity: float | None = None
    quantity_unit: str | None = None
    quantity_basis: QuantityBasisStr = "NOT_ESTABLISHED"
    # Required when quantity_basis == "DERIVED": the LLM's transparent, one-line
    # statement of how the quantity was obtained from explicit evidenced values
    # ("2 machines x 6 h/machine = 12 h"). Deterministic code never fills this
    # and never derives a quantity itself (spec Pass 32 §5/§6).
    quantity_derivation: str = ""
    # Optional link to the calculation_proposals entry whose operands establish
    # a DERIVED quantity -- lets the arithmetic executor re-check the product.
    derived_from_calculation_id: str | None = None
    unit_cost: float | None = None
    unit_cost_low: float | None = None       # optional LLM-expressed range for this driver
    unit_cost_high: float | None = None
    unit_cost_basis: CostBasisStr = "NOT_ESTABLISHED"
    currency: str | None = None
    amount_type: AmountType = "COMPONENT"
    # Groups mutually-exclusive ALTERNATIVE components (options for the same
    # decision). Components sharing a group are bracketed as scenarios, never
    # summed. Ungrouped ALTERNATIVE components are all treated as one group.
    alternative_group: str | None = None
    # Marks the option the evidence identifies as the recommended / expected
    # choice within its alternative_group -- used for the most-likely figure.
    is_primary_option: bool = False
    recurrence: Recurrence = "ONE_TIME"
    recurring_period: str | None = None
    source_reference_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    interpretation_confidence: Confidence = "LOW"
    rationale: str = ""

    @field_validator("quantity", "unit_cost", "unit_cost_low", "unit_cost_high", mode="before")
    @classmethod
    def _finite_value(cls, v: Any) -> float | None:
        return _finite(v)


class CalcOperand(BaseModel):
    """One explicit input to a calculation plan. The LLM supplies the numeric
    VALUE and its meaning; deterministic arithmetic only combines the values it
    is given (spec Pass 32 §3/§4/§29/§30). `source_component_id` optionally ties
    the operand back to a cost component; `evidence_refs` cite where the value
    is stated."""

    label: str = ""                       # "machine count", "hours per machine", "hourly rate"
    value: float | None = None
    unit: str | None = None               # "machine", "hour", "currency/hour"
    source_component_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("value", mode="before")
    @classmethod
    def _finite_value(cls, v: Any) -> float | None:
        return _finite(v)


class RemediationCalculationProposal(BaseModel):
    """The LLM's explicit calculation plan for one figure. The LLM decides the
    operands, the operation, the frequency and what the result represents; the
    deterministic executor in `app.remediation.calculator` evaluates the
    arithmetic exactly as specified (spec Pass 32 §3-§5, §29-§31).

    `proposed_result_*` and the executed value are retained for the audit trail
    only -- the authoritative estimate numbers still come from role-based
    (`amount_type`) component assembly, never from this proposal."""

    calculation_id: str
    operation: CalcOperation
    # Legacy / simple form: reference cost components and let the executor read
    # their point amounts. Still fully supported.
    component_ids: list[str] = Field(default_factory=list)
    # Rich form (Pass 32): explicit operands with values + provenance. When
    # present these are what the executor evaluates.
    operands: list[CalcOperand] = Field(default_factory=list)
    produces: Produces = "COMPONENT_AMOUNT"
    # Which component this plan prices (spec Pass 35 §15). Optional -- legacy
    # plans reference components only through `component_ids`.
    target_component_id: str | None = None
    # The frequency of the RESULT this plan produces. NO semantic default
    # (spec Pass 35 §6): a rich self-describing plan (one carrying explicit
    # `operands`) with `frequency` absent is structurally incomplete and is
    # rejected -- the LLM must state ONE_TIME or RECURRING. MUST match the
    # target component's `recurrence`. A "per month/week/quarter/year" rate is
    # RECURRING with `recurring_period` set -- the result is the PERIODIC
    # amount, never multiplied out to a lifetime (spec Pass 33 §2/§6).
    frequency: Recurrence | None = None
    recurring_period: str | None = None
    # An explicit time horizon over which a RECURRING cost is to be totalled.
    # ONLY when the evidence or the auditor's request establishes it. Never
    # inferred (spec Pass 33 §3/§7/§9). `horizon` is the count, `horizon_unit`
    # must equal `recurring_period`.
    horizon: float | None = None
    horizon_unit: str | None = None
    horizon_basis: Literal["EXPLICIT", "DERIVED", "UNKNOWN", "NOT_APPLICABLE"] = "NOT_APPLICABLE"
    currency: str | None = None
    result_represents: str = ""           # "one-time sensor-installation labour cost"
    proposed_result_value: float | None = None
    proposed_result_currency: str | None = None
    rationale: str = ""
    reason: str = ""

    @field_validator("proposed_result_value", mode="before")
    @classmethod
    def _finite_value(cls, v: Any) -> float | None:
        return _finite(v)


class AuditorInputRequired(BaseModel):
    """One concrete, activity-specific piece of evidence the auditor must
    supply before an established remediation activity can be priced. The cost
    LLM produces these -- deterministic code never invents them and never
    invents the value they would carry."""

    remediation_activity: str
    current_pricing_evidence: str = ""      # what IS already available for this activity
    missing_input: str = ""                 # the specific missing item
    why_required: str = ""
    acceptable_evidence: str = ""           # e.g. "supplier quotation OR internal rate + effort estimate"
    enables_estimate_type: str = ""         # EXACT_ESTIMATE | RANGE_ESTIMATE | PARTIAL_ESTIMATE


class RemediationInterpretation(BaseModel):
    """Top-level structured output of the LLM remediation-cost stage."""

    strategy: RemediationStrategy = Field(default_factory=RemediationStrategy)
    activities: list[ImplementationActivity] = Field(default_factory=list)
    cost_components: list[RemediationCostComponent] = Field(default_factory=list)
    calculation_proposals: list[RemediationCalculationProposal] = Field(default_factory=list)
    overall_status: OverallStatus = "NOT_ASSESSABLE"
    estimability: Estimability = "NOT_ASSESSABLE"
    not_assessable_reason: NotAssessableReason = ""
    range_assumptions: list[str] = Field(default_factory=list)
    uncertainty_reasons: list[str] = Field(default_factory=list)
    evidence_improves_estimate: list[str] = Field(default_factory=list)
    auditor_inputs_required: list[AuditorInputRequired] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        for canonical, aliases in {
            "cost_components": ("components", "cost_drivers", "drivers"),
            "calculation_proposals": ("calculations", "proposed_calculations"),
            "activities": ("implementation_activities",),
        }.items():
            if canonical not in d:
                for a in aliases:
                    if a in d:
                        d[canonical] = d.pop(a)
                        break
        return d


class RemediationRejectedItem(BaseModel):
    """A component or calculation proposal that failed deterministic validation
    -- preserved (never silently dropped) so a reviewer can see WHY a number was
    not produced. `detail` is auditor-facing but neutral; no LLM/parser jargon."""

    item_id: str
    kind: Literal["COMPONENT", "ACTIVITY", "CALCULATION"] = "COMPONENT"
    reason_code: Literal[
        "UNKNOWN_REFERENCE", "MISSING_PROVENANCE", "UNSUPPORTED_VERIFIED_CLAIM",
        "PRICING_NOT_SUPPORTED", "INCOMPATIBLE_CURRENCY", "MISSING_QUANTITY",
        "MISSING_UNIT_COST", "INVALID_NUMBER", "AMBIGUOUS_OPERANDS",
        "CONFLICTING_COMPONENTS", "DOUBLE_COUNT", "UNKNOWN_COMPONENT",
        "UNSUPPORTED_OPERATION", "OBSERVED_VALUE_NOT_REMEDIATION",
    ]
    detail: str = ""


class RemediationCalculationTrace(BaseModel):
    """Auditable record of ONE calculation: what the LLM proposed vs what the
    deterministic executor independently computed. Never consumed by the
    calculator or the renderer's authoritative numeric fields."""

    calculation_id: str
    operation: str = ""
    component_ids: list[str] = Field(default_factory=list)
    operands: list[CalcOperand] = Field(default_factory=list)
    produces: str = "COMPONENT_AMOUNT"
    frequency: str = "ONE_TIME"
    recurring_period: str | None = None
    horizon: float | None = None
    horizon_unit: str | None = None
    horizon_basis: str = "NOT_APPLICABLE"
    result_represents: str = ""
    currency: str | None = None
    formula: str = ""
    llm_proposed_result: float | None = None
    executor_result: float | None = None
    disagreement: str | None = None


class RemediationValidationOutcome(BaseModel):
    """Result of running `app.remediation.validator.validate_and_plan`."""

    accepted_calculation_ids: list[str] = Field(default_factory=list)
    rejected: list[RemediationRejectedItem] = Field(default_factory=list)
    llm_disagreements: list[str] = Field(default_factory=list)
    traces: list[RemediationCalculationTrace] = Field(default_factory=list)
    semantic_status: RemediationSemanticStatus = "OK"
    dropped_component_ids: list[str] = Field(default_factory=list)
