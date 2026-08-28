"""Canonical result model for Remediation Cost Estimation.

This is the ONE object every downstream consumer (the investigation report, the
API, the renderer) uses. It is assembled deterministically in
`app.remediation.engine` from validated components + executed arithmetic -- the
LLM never writes to it directly.

Remediation cost ("money expected to be spent to fix/prevent the finding") is
kept strictly separate from financial exposure ("money already lost/exposed",
`app.financial.models.FinancialAnalysisResult`).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.remediation.semantic_models import (
    RemediationCalculationTrace,
    RemediationRejectedItem,
)


def _finite(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    return None


class CostBasis(str, Enum):
    VERIFIED = "VERIFIED"
    REPORTED = "REPORTED"
    ESTIMATED = "ESTIMATED"
    ASSUMED = "ASSUMED"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


class RemediationEstimateStatus(str, Enum):
    EVIDENCE_BACKED = "EVIDENCE_BACKED"
    ASSUMPTION_BASED = "ASSUMPTION_BASED"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


class RemediationConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOT_ASSESSABLE = "NOT_ASSESSABLE"


class RemediationCostComponentResult(BaseModel):
    """One cost driver, after deterministic validation + arithmetic."""

    component_id: str
    description: str
    cost_category: str = "other"
    quantity: float | None = None
    quantity_unit: str | None = None
    quantity_basis: CostBasis = CostBasis.NOT_ESTABLISHED
    unit_cost: float | None = None
    unit_cost_basis: CostBasis = CostBasis.NOT_ESTABLISHED
    currency: str | None = None
    calculated_amount: float | None = None      # deterministic: quantity x unit_cost, or the flat total
    calculated_amount_low: float | None = None  # when the component carried a unit-cost range
    calculated_amount_high: float | None = None
    calculation_formula: str = ""
    recurrence: Literal["ONE_TIME", "RECURRING"] = "ONE_TIME"
    recurring_period: str | None = None
    confidence: RemediationConfidence = RemediationConfidence.LOW
    source_reference_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    rationale: str = ""
    is_derived: bool = False

    @field_validator(
        "quantity", "unit_cost", "calculated_amount",
        "calculated_amount_low", "calculated_amount_high", mode="before",
    )
    @classmethod
    def _finite_value(cls, v: Any) -> float | None:
        return _finite(v)


class RemediationCostResult(BaseModel):
    """Authoritative consolidated Remediation Cost Estimate."""

    status: RemediationEstimateStatus = RemediationEstimateStatus.NOT_ASSESSABLE

    remediation_strategy: str = ""
    remediation_rationale: str = ""
    established_basis: str = ""
    hypothetical_basis: str = ""
    alternative_strategies: list[str] = Field(default_factory=list)
    implementation_activities: list[str] = Field(default_factory=list)

    cost_components: list[RemediationCostComponentResult] = Field(default_factory=list)
    currency: str | None = None

    one_time_cost: float | None = None
    recurring_cost: float | None = None
    recurring_period: str | None = None

    # Spec section 14: when some implementation activities are priced and others
    # cannot be, the priced portion is still calculated and reported -- the
    # result is NOT forced to NOT_ASSESSABLE. These name the activities/drivers
    # that carry no defensible price; `is_partial_estimate` is True whenever at
    # least one priced and at least one unpriced component coexist.
    unpriced_activities: list[str] = Field(default_factory=list)
    is_partial_estimate: bool = False

    low_estimate: float | None = None
    most_likely_estimate: float | None = None
    high_estimate: float | None = None
    estimate_classification: CostBasis = CostBasis.NOT_ESTABLISHED

    confidence: RemediationConfidence = RemediationConfidence.NOT_ASSESSABLE
    assumptions: list[str] = Field(default_factory=list)
    range_assumptions: list[str] = Field(default_factory=list)
    uncertainty_reasons: list[str] = Field(default_factory=list)
    evidence_basis: list[str] = Field(default_factory=list)
    estimation_method: str = ""
    evidence_improves_estimate: list[str] = Field(default_factory=list)
    review_required: bool = True

    # Professional, user-facing explanation used when status == NOT_ASSESSABLE.
    # NEVER an internal diagnostic (spec sections 6, 14, 15).
    not_assessable_reason: str = ""

    important_qualification: str = (
        "This estimate represents expected implementation cost and should not be "
        "interpreted as incurred financial loss."
    )

    # ---- provenance / audit -- never a numeric authority ----
    reasoning_source: Literal["LLM_SEMANTIC", "NONE"] = "NONE"
    # OK | LLM_UNAVAILABLE | LLM_INVALID | LLM_INCOMPLETE | NO_EVIDENCE -- for
    # logs / invariants only; the renderer never shows this.
    remediation_semantic_status: str = "OK"
    calculation_traces: list[RemediationCalculationTrace] = Field(default_factory=list)
    rejected_items: list[RemediationRejectedItem] = Field(default_factory=list)

    @field_validator(
        "one_time_cost", "recurring_cost", "low_estimate",
        "most_likely_estimate", "high_estimate", mode="before",
    )
    @classmethod
    def _finite_value(cls, v: Any) -> float | None:
        return _finite(v)
