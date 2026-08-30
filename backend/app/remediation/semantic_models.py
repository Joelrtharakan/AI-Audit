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
QuantityBasisStr = Literal["EVIDENCED", "ASSUMED", "NOT_ESTABLISHED"]

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

CalcOperation = Literal["MULTIPLY", "SUM", "SUBTRACT"]

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
    quantity: float | None = None
    quantity_unit: str | None = None
    quantity_basis: QuantityBasisStr = "NOT_ESTABLISHED"
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


class RemediationCalculationProposal(BaseModel):
    """A proposed calculation. `proposed_result_*` is retained ONLY for audit /
    disagreement logging -- never authoritative. The real number is
    re-executed deterministically in `app.remediation.calculator`."""

    calculation_id: str
    operation: CalcOperation
    component_ids: list[str] = Field(default_factory=list)
    produces: Produces = "COMPONENT_AMOUNT"
    proposed_result_value: float | None = None
    proposed_result_currency: str | None = None
    reason: str = ""

    @field_validator("proposed_result_value", mode="before")
    @classmethod
    def _finite_value(cls, v: Any) -> float | None:
        return _finite(v)


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
    produces: str = "COMPONENT_AMOUNT"
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
