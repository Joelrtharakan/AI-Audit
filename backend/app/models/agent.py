"""Pydantic models for the agentic LQMS Corrective Action Investigation pipeline.

These models give the investigation report and CA draft a strongly-typed, auditor-facing
schema. Every claim carries an evidence classification status so the auditor can see
exactly what is verified vs. reported vs. inferred.

Key design rules enforced here (not just in prompts):
  - InvestigationReport.human_review_required is always True.
  - CADraft contains ONLY the five AI-controlled fields. No more, no less.
  - FiveWhyStep.status must be one of four explicit evidence-based values.
  - AgentFinalState is a closed enum -- the agent cannot invent new outcomes.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Literal

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Evidence classification
# ---------------------------------------------------------------------------


class EvidenceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    REPORTED = "REPORTED"
    INFERRED = "INFERRED"
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"


class CanonicalFindingState(BaseModel):
    """The single canonical intermediate representation extracted from raw finding text (Section 1).

    All downstream nodes (RCA, 5-Why, Investigation, CAPA, Impact) MUST consume this state
    rather than independently re-interpreting the raw finding.
    """
    raw_finding: str
    observed_deviation: str
    # The condition asserted about the affected object (e.g. "not completed",
    # "missing", "incomplete") — kept separate from affected_objects so the
    # subject and the deviation asserted about it are never conflated into
    # one opaque string.
    deviation_condition: str = "UNKNOWN"
    affected_objects: list[str] = []
    affected_people: list[str] = []
    affected_departments: list[str] = []
    affected_equipment: list[str] = []
    affected_records: list[str] = []
    affected_period: str = "UNKNOWN"
    # The person/role identified as responsible/involved, if the finding names one
    # (e.g. "the responsible technician") — distinct from affected_people (who was
    # impacted) since an actor may be neither affected nor a department.
    actor: str | None = None
    # Layer 2 of the causal chain (Layer 1 is observed_deviation): the
    # action-level explanation of HOW the deviation happened, when the
    # finding/evidence directly states one (e.g. "the check was missed",
    # distinct from the observation "the log was incomplete"). Null if no
    # claim in the finding states a mechanism at this level of specificity.
    immediate_mechanism: str | None = None
    # VERIFIED (from a stated fact) | REPORTED (from an attributed
    # statement) | UNKNOWN (no mechanism-level claim found).
    immediate_mechanism_status: str = "UNKNOWN"
    process: str = "UNKNOWN"
    procedure: str = "UNKNOWN"
    requirement: str = "UNKNOWN"
    facts: list[str] = []
    reported_statements: list[str] = []
    inferred_statements: list[str] = []
    unknowns: list[str] = []
    contradictions: list[str] = []
    prompt_injection_detected: bool = False



class EvidenceItem(BaseModel):
    """A single piece of evidence in the evidence ledger.

    Every important claim produced by the agent must be traceable to one or
    more EvidenceItems. The agent cannot produce a conclusion without a
    corresponding ledger entry.
    """

    claim: str
    source: str  # e.g. "finding_text", "training_records_tool", "auditor_statement"
    source_reference: str | None = None  # e.g. tool call result identifier
    status: EvidenceStatus
    relevance: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    notes: str | None = None


# ---------------------------------------------------------------------------
# 5-Why analysis
# ---------------------------------------------------------------------------


class FiveWhyStep(BaseModel):
    question: str
    answer: str | None = None
    status: Literal["VERIFIED", "SUPPORTED", "REPORTED", "REPORTED_STATEMENT", "REPORTED_UNVERIFIED", "INFERRED", "UNKNOWN", "REQUIRES_EVIDENCE", "NOT_ESTABLISHED"]



class FiveWhyAnalysis(BaseModel):
    steps: list[FiveWhyStep] = []
    is_complete: bool = False
    status_note: str = ""  # e.g. "INCOMPLETE — additional evidence required"


# ---------------------------------------------------------------------------
# Root cause
# ---------------------------------------------------------------------------


class RootCauseStatus(str, Enum):
    VERIFIED = "VERIFIED"
    SUPPORTED = "SUPPORTED"
    STATED_UNVERIFIED = "STATED_UNVERIFIED"
    INFERRED = "INFERRED"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"
    CONTRADICTED = "CONTRADICTED"


class CandidateHypothesis(BaseModel):
    id: str  # e.g. "H1"
    name: str  # e.g. "TRAINING_ASSIGNMENT"
    statement: str
    status: Literal["POSSIBLE", "SUPPORTED", "REFUTED", "UNVERIFIED"] = "POSSIBLE"
    evidence_needed: str
    discrimination_evidence: str | None = None  # what evidence would distinguish THIS hypothesis from the others
    relevance_rank: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"  # Section 7 evidence relevance ranking
    resolves_investigation: str | None = None
    # Why this hypothesis is plausible for THIS finding specifically — distinct
    # from `statement` (the claim itself) so plausibility reasoning isn't lost.
    rationale: str | None = None
    # Facts from the evidence ledger that argue against this hypothesis, if any.
    evidence_against: str | None = None
    # What specific result would CONFIRM this hypothesis.
    confirms_if: str | None = None
    # What specific result would REFUTE this hypothesis.
    refutes_if: str | None = None



class RootCauseAnalysis(BaseModel):
    status: RootCauseStatus
    category: str | None = None  # 6M taxonomy value
    statement: str | None = None
    leading_hypothesis: str | None = None
    candidate_hypotheses: list[CandidateHypothesis] = []
    supporting_evidence: list[str] = []
    contradicting_evidence: list[str] = []
    missing_evidence: list[str] = []
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    narrative: str = ""
    evidence_status: EvidenceStatus = EvidenceStatus.UNKNOWN
    verification_needed: str | None = None


# ---------------------------------------------------------------------------
# Contributing factors
# ---------------------------------------------------------------------------


class ContributingFactor(BaseModel):
    description: str
    evidence_status: EvidenceStatus = EvidenceStatus.INFERRED
    status: Literal["POSSIBLE_UNCONFIRMED", "VERIFIED", "REJECTED"] = "POSSIBLE_UNCONFIRMED"
    # Why this factor is plausible for this finding, and what would confirm it —
    # optional so existing callers that only set `description` still validate.
    rationale: str | None = None
    evidence_required: str | None = None
    # Populated only when status=="REJECTED": what contradicts it.
    evidence_against: str | None = None


# ---------------------------------------------------------------------------
# Investigation plan
# ---------------------------------------------------------------------------


class InvestigationQuestion(BaseModel):
    """A structured investigation question with explicit purpose and discriminating evidence."""

    question: str
    purpose: str  # which hypothesis / unknown this resolves
    evidence: str  # specific document/record type that would answer this


class InvestigationPlan(BaseModel):
    areas: list[str] = []
    questions: list[InvestigationQuestion] = []
    evidence_to_collect: list[str] = []


# ---------------------------------------------------------------------------
# CAPA analysis
# ---------------------------------------------------------------------------


class CapaStatus(str, Enum):
    CAPA_RECOMMENDED = "CAPA_RECOMMENDED"
    CAPA_DRAFT_POSSIBLE = "CAPA_DRAFT_POSSIBLE"
    INVESTIGATION_REQUIRED = "INVESTIGATION_REQUIRED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NO_CAPA_RECOMMENDATION_YET = "NO_CAPA_RECOMMENDATION_YET"


class ConditionalCapaAction(BaseModel):
    """A conditional CAPA branch: what to do IF a specific cause is confirmed."""

    if_cause_confirmed: str  # the condition, e.g. "If training was never assigned by the training matrix"
    recommended_action: str  # specific corrective action for that branch
    # Optional structured fields (populated where the synthesis provides
    # them) distinguishing what KIND of action this is, so a systemic action
    # is never mistaken for an already-confirmed corrective action.
    action_type: Literal["IMMEDIATE_CORRECTION", "CONTAINMENT", "CORRECTIVE_ACTION", "SYSTEMIC_ACTION"] | None = None
    verification_method: str | None = None  # how effectiveness of this action would be verified


class CapaAnalysis(BaseModel):
    status: CapaStatus
    potential_areas: list[str] = []
    recommended_investigation: list[str] = []
    conditional_actions: list[ConditionalCapaAction] = []  # evidence-gated action branches


# ---------------------------------------------------------------------------
# Impact assessment
# ---------------------------------------------------------------------------


class ImpactStatus(str, Enum):
    IMPACT_VERIFIED = "IMPACT_VERIFIED"
    IMPACT_NOT_IDENTIFIED = "IMPACT_NOT_IDENTIFIED"
    IMPACT_POSSIBLE = "IMPACT_POSSIBLE"
    IMPACT_REQUIRES_ASSESSMENT = "IMPACT_REQUIRES_ASSESSMENT"


class ImpactAssessment(BaseModel):
    status: ImpactStatus
    areas: list[str] = []
    narrative: str | None = None
    # Structured fields from the new spec — populated when the model provides them
    affected_object: str | None = None   # the actual process/record/output impacted
    affected_people: str | None = None   # who specifically was affected
    affected_period: str | None = None   # stated period or "requires confirmation"
    process_at_risk: str | None = None   # the process whose output is in question
    relevant_change: str | None = None   # what specifically changed (Rule 18: never assumed)
    potential_effect: str | None = None  # plausible downstream consequence
    evidence_needed: str | None = None   # what would bound the scope
    # Per-field EXPLICIT/INFERRED/UNKNOWN classification (keyed by field name:
    # "affected_object", "affected_period", "process_at_risk",
    # "relevant_change", "potential_effect"), computed deterministically by
    # app.agent.analytical_validator.compute_impact_field_basis — never
    # asserted by the LLM itself, since a model can't be trusted to grade
    # its own certainty.
    field_basis: dict[str, str] = {}



# ---------------------------------------------------------------------------
# Evidence gaps
# ---------------------------------------------------------------------------


class EvidenceGap(BaseModel):
    claim: str
    missing: str
    evidence_status: EvidenceStatus = EvidenceStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Full investigation report (spec section 18)
# ---------------------------------------------------------------------------


class InvestigationReport(BaseModel):
    observation_quality: Literal["SUFFICIENT", "INSUFFICIENT", "CONFLICTING"]
    observation_confidence: Literal["LOW", "MEDIUM", "HIGH"] = "HIGH"
    root_cause_confidence: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    overall_confidence: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"  # backwards compatibility
    investigation_required: Literal["YES", "NO", "LIMITED"]
    root_cause: RootCauseAnalysis
    contributing_factors: list[ContributingFactor] = []
    investigation: InvestigationPlan
    five_why: FiveWhyAnalysis
    capa: CapaAnalysis
    impact_assessment: ImpactAssessment
    evidence_gaps: list[EvidenceGap] = []
    evidence: list[EvidenceItem] = []
    human_review_required: bool = True  # always True -- enforced here, not just prompted
    # "LLM" = normal causal synthesis ran; "DEGRADED" = the LLM call failed
    # and a deterministic, evidence-conservative fallback produced this
    # report instead. Must never be silently indistinguishable from a
    # normal result.
    analysis_mode: Literal["LLM", "DEGRADED"] = "LLM"
    # Provider-router metadata (infrastructure only, no analytical meaning):
    # which provider actually answered, whether Groq -> OpenRouter -> Gemini
    # failover was needed, and the full attempt order.
    provider_used: str | None = None
    fallback_used: bool = False
    provider_attempts: list[str] = []

    @field_validator("human_review_required")
    @classmethod
    def must_require_human_review(cls, v: bool) -> bool:
        if not v:
            raise ValueError("human_review_required must always be True")
        return v


# ---------------------------------------------------------------------------
# CA Draft (spec section 19) -- EXACTLY 5 fields, never more
# ---------------------------------------------------------------------------


class CADraft(BaseModel):
    """The five AI-controlled CA fields.

    This model is the ONLY thing the agent is allowed to write. It cannot
    contain any other field. The permissions layer enforces this in code;
    the schema here provides a second layer of protection.
    """

    model_config = {"extra": "forbid"}

    immediate_action: str
    root_cause: str
    root_cause_category: str
    preventive_action: str
    impact_analysis: str


# ---------------------------------------------------------------------------
# Agent final state
# ---------------------------------------------------------------------------


class AgentFinalState(str, Enum):
    READY_FOR_HUMAN_REVIEW = "READY_FOR_HUMAN_REVIEW"
    INVESTIGATION_REQUIRED = "INVESTIGATION_REQUIRED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    REQUIRES_HUMAN_INPUT = "REQUIRES_HUMAN_INPUT"
    TOOL_FAILURE = "TOOL_FAILURE"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"


# ---------------------------------------------------------------------------
# Agent trace
# ---------------------------------------------------------------------------


class AgentTraceStep(BaseModel):
    icon: Literal["✓", "⚠", "✗"]
    message: str
    timestamp: str = ""

    @classmethod
    def ok(cls, message: str) -> "AgentTraceStep":
        return cls(icon="✓", message=message, timestamp=dt.datetime.now(dt.timezone.utc).isoformat())

    @classmethod
    def warn(cls, message: str) -> "AgentTraceStep":
        return cls(icon="⚠", message=message, timestamp=dt.datetime.now(dt.timezone.utc).isoformat())

    @classmethod
    def error(cls, message: str) -> "AgentTraceStep":
        return cls(icon="✗", message=message, timestamp=dt.datetime.now(dt.timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Request / response for /api/v1/investigate
# ---------------------------------------------------------------------------


class InvestigateRequest(BaseModel):
    """Mirrors the finding/CA screen fields available to the agent (read-only inputs)."""

    # Core finding
    finding_text: str
    ca_number: str = ""
    audit_number: str = ""
    audit_date: str = ""
    clause_number: str = ""
    audit_question: str = ""
    departments: list[str] = []
    nature_of_nc: str = ""
    auditors: list[str] = []
    auditees: list[str] = []
    audit_criteria: str = ""
    area_audited: str = ""
    finding_type: str = ""
    severity: str = ""
    likelihood: str = ""
    risk_result: str = ""

    @field_validator("finding_text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("finding_text must not be blank")
        return v.strip()


class AiMetadata(BaseModel):
    model: str
    prompt_version: str
    generated_at: str
    suggestion_id: str


class InvestigateResponse(BaseModel):
    final_state: AgentFinalState
    report: InvestigationReport | None = None
    ca_draft: CADraft | None = None
    trace: list[AgentTraceStep] = []
    ai_metadata: AiMetadata
