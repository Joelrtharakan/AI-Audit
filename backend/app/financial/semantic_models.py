"""Structured schema for the LLM semantic-understanding layer of financial
evidence interpretation.

This is deliberately a SEPARATE model from `app.financial.models.
FinancialObservation` (the regex-extractor's output shape): the LLM's job is
to understand meaning and propose relationships/calculations, never to
compute a number itself. `relationship_validator.py` is the only place a
`SemanticClaim`/`SemanticRelationship` pair is converted into a
`FinancialObservation` -- and only after passing every deterministic safety
check -- so the existing, already-tested `calculator.py` functions remain
the sole source of arithmetic, for both the regex-extraction path and this
semantic path.

Nothing in this module performs a calculation. It only describes what the
LLM is allowed to say.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FactType = Literal[
    "QUANTITY", "RATE", "AMOUNT", "RECOVERY", "REMEDIATION_COST",
    "PREVENTION_COST", "OBSERVATION_PERIOD", "PERCENTAGE", "OTHER",
]

Population = Literal[
    "CURRENT_FINDING", "HISTORICAL", "RECOVERY", "REMEDIATION",
    "PREVENTION", "OTHER",
]

# The same four-bucket evidence-eligibility vocabulary used everywhere else
# in the financial pipeline (app.financial.models.FinancialObservation.
# verification_status) -- the LLM's interpretation confidence is a SEPARATE
# dimension (see SemanticClaim.interpretation_confidence) and must never be
# allowed to change this value away from what the source evidence actually
# stated.
EvidenceStatusStr = Literal["VERIFIED", "REPORTED", "UNVERIFIED", "CONTRADICTED"]

# How the LLM semantic stage terminated -- an honest, provider-independent
# status the downstream renderer surfaces instead of the pipeline silently
# substituting a keyword/regex interpretation (architecture spec: "the
# system must fail honestly"). OK means an interpretation was produced;
# NO_EVIDENCE means there was no evidence ledger to interpret (the one case
# where deferring to the deterministic text engine is legitimate).
FinancialSemanticStatus = Literal[
    "OK", "LLM_UNAVAILABLE", "LLM_INVALID", "LLM_INCOMPLETE", "NO_EVIDENCE"
]

# Whether the finding carries a financial mechanism at all -- a SEPARATE
# axis from whether an amount can be calculated (QuantificationAssessment).
# "18 additional rework hours" is MATERIAL relevance / UNQUANTIFIED.
FinancialRelevance = Literal["NONE", "POTENTIAL", "MATERIAL", "CONFIRMED"]

CalculationOperation = Literal["MULTIPLY", "SUBTRACT", "DIVIDE", "ANNUALIZE", "SUM"]

# The cost-factor taxonomy the LLM selects from (mirrors app.financial.
# models.FinancialAmountType's cost-NATURE values -- deliberately excludes
# NET_LOSS/UNIT_COST/POTENTIAL_EXPOSURE, which are calculator-derived or
# verification-status concepts, not something the LLM classifies a claim
# as). The LLM picks the best-SUPPORTED factor with a rationale and the
# claims that support it; relationship_validator.py independently checks
# that support before trusting the selection, never taking it on faith.
CostFactorLiteral = Literal[
    "DIRECT_LOSS", "OVERPAYMENT", "DUPLICATE_PAYMENT", "REWORK_COST",
    "SCRAP_COST", "DOWNTIME_COST", "CUSTOMER_COMPENSATION", "PENALTY",
    "REMEDIATION_COST", "PREVENTION_COST", "REVENUE_IMPACT",
    "COST_AVOIDANCE", "OTHER", "NOT_ESTABLISHED",
]


class SemanticFindingSummary(BaseModel):
    deviation: str | None = None
    affected_object: str | None = None
    process: str | None = None
    requirement: str | None = None
    affected_period: str | None = None
    interpretation_confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"


class SemanticClaim(BaseModel):
    """One atomic fact the LLM extracted from a single evidence item.

    `evidence_status` MUST be copied from the source evidence item's own
    status (never invented, never upgraded) -- enforced again, independent
    of what the LLM claims, by relationship_validator.py before any claim
    is allowed to participate in a calculation.
    """

    claim_id: str
    source_evidence_ids: list[str] = Field(default_factory=list)
    fact_type: FactType
    value: float | None = None
    unit: str | None = None
    currency: str | None = None
    population: Population = "OTHER"
    temporal_scope: str | None = None
    evidence_status: EvidenceStatusStr = "UNVERIFIED"
    explicit: bool = True
    notes: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _finite_value(cls, v: Any) -> float | None:
        if v is None:
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        import math
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv


class SemanticRelationship(BaseModel):
    """A relationship the LLM believes holds between two claims. This is a
    PROPOSAL -- relationship_validator.py independently checks that the
    relationship genuinely connects the cited claims and that population,
    currency, unit-class, and evidence-status are compatible before it is
    ever allowed to combine two claims into a calculation.

    `relationship_type` is FREE DESCRIPTIVE TEXT (e.g. "per-unit rate",
    "each-event amount", "recovery against gross", "competing estimate of
    the same loss") -- it is semantic metadata for the auditor, NOT an
    operation licence. The validator never maps a relationship_type to a
    permitted operation. The only structural signal it acts on is
    `is_conflict`: when true, the two claims are competing values for what
    should be one fact and must never be combined into a calculation.
    """

    relationship_id: str
    relationship_type: str = "related"
    relationship_description: str = ""
    semantic_basis: str = ""
    is_conflict: bool = False
    source_claim: str
    target_claim: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
    evidence_basis: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_type_alias(cls, data: Any) -> Any:
        """Defensive back-compat: a `type` key (the former closed-enum
        field name) is accepted as `relationship_type`, and a label that
        names the relationship a conflict sets `is_conflict`. The primary
        place this is handled is `provider_normalization.normalize_to_canonical`;
        this keeps the model correct if something reaches it unnormalized."""
        if not isinstance(data, dict):
            return data
        if "relationship_type" not in data and "type" in data:
            data = {**data, "relationship_type": data.get("type")}
        label = data.get("relationship_type")
        if isinstance(label, str) and "is_conflict" not in data:
            lowered = label.lower()
            if any(t in lowered for t in ("conflict", "competing", "contradict")):
                data = {**data, "is_conflict": True}
        return data


class CalculationProposal(BaseModel):
    """A proposed calculation. `proposed_result` is retained ONLY for audit
    /disagreement logging -- it is NEVER treated as authoritative. The
    authoritative number always comes from independently re-executing
    `operation` over `inputs` in relationship_validator.py /
    semantic_engine.py using the existing deterministic calculator."""

    calculation_id: str
    operation: CalculationOperation
    inputs: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    proposed_result_value: float | None = None
    proposed_result_currency: str | None = None
    reason: str = ""


class CostFactorAssessment(BaseModel):
    """The LLM's semantic classification of what KIND of financial impact
    the finding represents -- selected from the fixed taxonomy above, not
    a free-text guess and not a deterministic keyword lookup. Must be
    grounded in specific claims (`supporting_claim_ids`); an ungrounded or
    LOW-confidence selection is treated as NOT_ESTABLISHED by the
    validator rather than trusted at face value."""

    selected_factor: CostFactorLiteral = "NOT_ESTABLISHED"
    supporting_claim_ids: list[str] = Field(default_factory=list)
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
    rationale: str = ""


class QuantificationAssessment(BaseModel):
    """Whether the evidence contains enough monetary information to
    calculate an amount for the identified cost factor -- a SEPARATE
    dimension from cost-factor identification itself. A finding can
    clearly establish WHAT KIND of financial exposure exists (e.g.
    REWORK_COST, from labor/activity described in the evidence) while
    providing no rate, unit price, or total to calculate HOW MUCH it is.
    The LLM must never fabricate a monetary value merely to make this
    QUANTIFIABLE; the honest, correct output for a bare activity
    description with no monetary figure is NOT_QUANTIFIABLE."""

    status: Literal[
        "QUANTIFIED", "PARTIALLY_QUANTIFIED", "UNQUANTIFIED", "NOT_ASSESSABLE",
        # Back-compat spellings normalized on ingest (see provider_normalization).
        "QUANTIFIABLE", "NOT_QUANTIFIABLE",
    ] = "NOT_ASSESSABLE"
    blocker: str = ""
    missing_inputs: list[str] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def _canonical_status(cls, v: Any) -> Any:
        return {"QUANTIFIABLE": "QUANTIFIED", "NOT_QUANTIFIABLE": "UNQUANTIFIED"}.get(v, v)


class SemanticFindingInterpretation(BaseModel):
    """Top-level structured output of the LLM semantic-understanding stage."""

    finding: SemanticFindingSummary = Field(default_factory=SemanticFindingSummary)
    claims: list[SemanticClaim] = Field(default_factory=list)
    relationships: list[SemanticRelationship] = Field(default_factory=list)
    calculation_proposals: list[CalculationProposal] = Field(default_factory=list)
    cost_factor: CostFactorAssessment = Field(default_factory=CostFactorAssessment)
    quantification: QuantificationAssessment = Field(default_factory=QuantificationAssessment)
    # Whether a financial mechanism exists at all -- independent of whether
    # it can be quantified. The validator forces the materialized cost
    # factor to None ONLY when the LLM explicitly says "NONE"; a provider
    # that omits the field (None) leaves the rest of the pipeline to judge.
    financial_relevance: FinancialRelevance | None = None


class RejectedCalculation(BaseModel):
    """A calculation proposal that failed deterministic validation --
    preserved (never silently dropped) so the auditor can see WHY the LLM's
    proposal was not executed."""

    calculation_id: str
    reason_code: Literal[
        "UNKNOWN_CLAIM", "MISSING_PROVENANCE", "EVIDENCE_STATUS_INELIGIBLE",
        "POPULATION_MISMATCH", "INCOMPATIBLE_UNITS", "INCOMPATIBLE_CURRENCY",
        "TEMPORAL_MISMATCH", "UNSUPPORTED_RELATIONSHIP", "CONFLICTING_CLAIMS",
        "AMBIGUOUS_RELATIONSHIP", "INVALID_NUMBER", "UNSUPPORTED_OPERATION",
    ]
    detail: str = ""


class CalculationTrace(BaseModel):
    """Auditable record of ONE accepted calculation: what the LLM proposed,
    what the deterministic executor independently computed, and any
    disagreement between them. Purely for transparency -- never consumed by
    the calculator or the renderer's authoritative numeric fields."""

    calculation_id: str
    cost_factor: str = "NOT_ESTABLISHED"
    observation_id: str | None = None
    input_claim_ids: list[str] = Field(default_factory=list)
    semantic_roles: dict[str, str] = Field(default_factory=dict)
    relationship_ids: list[str] = Field(default_factory=list)
    operation: str = ""
    currency: str | None = None
    evidence_status: str = "UNVERIFIED"
    formula: str = ""
    llm_proposed_result: float | None = None
    executor_result: float | None = None
    disagreement: str | None = None
    status: str = ""


class SemanticValidationOutcome(BaseModel):
    """Result of running relationship_validator.py over a
    SemanticFindingInterpretation: what was accepted (as FinancialObservation
    -compatible facts, materialized separately) and what was rejected."""

    accepted_calculation_ids: list[str] = Field(default_factory=list)
    rejected: list[RejectedCalculation] = Field(default_factory=list)
    llm_disagreements: list[str] = Field(default_factory=list)
    traces: list[CalculationTrace] = Field(default_factory=list)
    # How the LLM semantic stage terminated. Set by semantic_engine.py;
    # "OK" whenever validate_and_materialize ran at all.
    semantic_status: FinancialSemanticStatus = "OK"
    # The cost factor actually used for materialization, after validating
    # the LLM's CostFactorAssessment (grounded support + sufficient
    # confidence) -- None when the LLM's selection was ungrounded/
    # unsupported/not provided and the deterministic default was used
    # instead, so an auditor can see whether the factor came from LLM
    # semantic reasoning or a fallback.
    validated_cost_factor: str | None = None
