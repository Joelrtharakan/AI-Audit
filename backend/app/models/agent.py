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
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Evidence classification
# ---------------------------------------------------------------------------


class EvidenceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    REPORTED = "REPORTED"
    MIXED = "MIXED"
    INFERRED = "INFERRED"
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTED = "CONTRADICTED"
    REFUTED = "REFUTED"
    UNKNOWN = "UNKNOWN"
    UNRESOLVED = "UNRESOLVED"


class PropositionType(str, Enum):
    OBSERVATION = "OBSERVATION"
    EVENT = "EVENT"
    REPORTED_EVENT = "REPORTED_EVENT"
    REPORTED_MECHANISM = "REPORTED_MECHANISM"
    IMMEDIATE_MECHANISM = "IMMEDIATE_MECHANISM"
    CONTRIBUTING_CAUSE = "CONTRIBUTING_CAUSE"
    ROOT_CAUSE = "ROOT_CAUSE"
    SYSTEMIC_CAUSE = "SYSTEMIC_CAUSE"
    CONTROL_FAILURE = "CONTROL_FAILURE"
    DOCUMENT_REFERENCE = "DOCUMENT_REFERENCE"
    DOCUMENT_CONTENT = "DOCUMENT_CONTENT"
    EVIDENCE_STATE = "EVIDENCE_STATE"
    INVESTIGATION_QUESTION = "INVESTIGATION_QUESTION"
    CAPA_STATE = "CAPA_STATE"
    PREVIOUS_CAPA_STATE = "PREVIOUS_CAPA_STATE"
    CONFLICTED_PROPOSITION = "CONFLICTED_PROPOSITION"
    IMPACT = "IMPACT"
    REQUIREMENT = "REQUIREMENT"


class CausalLevel(str, Enum):
    L0_OBSERVATION = "L0_OBSERVATION"
    L1_EVENT = "L1_EVENT"
    L2_IMMEDIATE_MECHANISM = "L2_IMMEDIATE_MECHANISM"
    L3_CONTRIBUTING_CAUSE = "L3_CONTRIBUTING_CAUSE"
    L4_ROOT_CAUSE = "L4_ROOT_CAUSE"
    L5_SYSTEMIC_CAUSE = "L5_SYSTEMIC_CAUSE"
    L2_REPORTED_MECHANISM = "L2_REPORTED_MECHANISM"  # backwards compatibility
    L3_IMMEDIATE_MECHANISM = "L3_IMMEDIATE_MECHANISM"  # backwards compatibility
    EVIDENCE_STATE = "EVIDENCE_STATE"


class SupportLevel(str, Enum):
    UNSUPPORTED = "UNSUPPORTED"
    PLAUSIBLE = "PLAUSIBLE"
    POSSIBLE = "POSSIBLE"
    SUPPORTED = "SUPPORTED"
    STRONGLY_SUPPORTED = "STRONGLY_SUPPORTED"
    ESTABLISHED = "ESTABLISHED"
    VERIFIED = "VERIFIED"
    REPORTED = "REPORTED"
    UNRESOLVED = "UNRESOLVED"
    UNKNOWN = "UNKNOWN"
    CONTRADICTED = "CONTRADICTED"
    REJECTED = "REJECTED"


class InvestigationMode(str, Enum):
    NORMAL = "NORMAL"
    CONFLICT = "CONFLICT"
    LOW_SPECIFICITY = "LOW_SPECIFICITY"
    DOCUMENT_UNAVAILABLE = "DOCUMENT_UNAVAILABLE"
    RECORD_UNAVAILABLE = "RECORD_UNAVAILABLE"
    REPORTED_MECHANISM = "REPORTED_MECHANISM"
    TEMPORAL_DEVIATION = "TEMPORAL_DEVIATION"
    COMBINED = "COMBINED"


class CausalEdgeType(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    REQUIRES = "REQUIRES"
    DEPENDS_ON = "DEPENDS_ON"
    TEMPORALLY_PRECEDES = "TEMPORALLY_PRECEDES"
    ENABLES = "ENABLES"
    PREVENTS = "PREVENTS"
    RESULTS_IN = "RESULTS_IN"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    EXPLAINS = "EXPLAINS"
    UNRESOLVED = "UNRESOLVED"
    UNRESOLVED_RELATIONSHIP = "UNRESOLVED_RELATIONSHIP"


class CausalRelationship(BaseModel):
    """An explicit directed relationship edge between propositions or evidence in the causal graph."""
    source_id: str
    target_id: str
    edge_type: CausalEdgeType
    evidence_ids: list[str] = []
    notes: str | None = None


class EvidenceCompleteness(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    EVIDENCE_UNAVAILABLE = "EVIDENCE_UNAVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    CONFLICTED = "CONFLICTED"


class EvidenceSourceCategory(str, Enum):
    AUDIT_FINDING = "AUDIT_FINDING"
    OBJECTIVE_RECORD = "OBJECTIVE_RECORD"
    REPORTED_STATEMENT = "REPORTED_STATEMENT"
    SYSTEM_RECORD = "SYSTEM_RECORD"
    ATTACHMENT = "ATTACHMENT"
    EVIDENCE_AVAILABILITY = "EVIDENCE_AVAILABILITY"
    INDEPENDENT_VERIFICATION = "INDEPENDENT_VERIFICATION"
    UNKNOWN_SOURCE = "UNKNOWN_SOURCE"


class ClaimAttribution(str, Enum):
    """Who produced the claim — tracks provenance so a supervisor's assertion
    is never silently equated with an auditor's direct observation."""
    AUDITOR_OBSERVED = "AUDITOR_OBSERVED"
    AUDIT_FINDING = "AUDIT_FINDING"
    PERSON_REPORTED = "PERSON_REPORTED"
    SUPERVISOR_REPORTED = "SUPERVISOR_REPORTED"
    DOCUMENTARY_EVIDENCE = "DOCUMENTARY_EVIDENCE"
    SYSTEM_EVIDENCE = "SYSTEM_EVIDENCE"
    ATTACHMENT = "ATTACHMENT"
    EVIDENCE_AVAILABILITY = "EVIDENCE_AVAILABILITY"
    AI_INFERENCE = "AI_INFERENCE"
    UNKNOWN = "UNKNOWN"


class EpistemicSource(str, Enum):
    AUDIT_OBSERVATION = "AUDIT_OBSERVATION"
    OBJECTIVE_RECORD = "OBJECTIVE_RECORD"
    SYSTEM_RECORD = "SYSTEM_RECORD"
    REPORTED_STATEMENT = "REPORTED_STATEMENT"
    USER_PROVIDED_EVIDENCE = "USER_PROVIDED_EVIDENCE"
    DERIVED = "DERIVED"
    INFERRED = "INFERRED"
    UNKNOWN_SOURCE = "UNKNOWN_SOURCE"


class Proposition(BaseModel):
    """A canonical proposition in the causal graph."""
    id: str  # e.g. "P1"
    statement: str
    type: PropositionType = PropositionType.OBSERVATION
    causal_level: CausalLevel = CausalLevel.L0_OBSERVATION
    support_level: SupportLevel = SupportLevel.UNKNOWN
    source_type: EpistemicSource = EpistemicSource.AUDIT_OBSERVATION
    evidence_ids: list[str] = []
    supporting_evidence_ids: list[str] = []
    contradicting_evidence_ids: list[str] = []
    conflict_ids: list[str] = []
    status: str = "UNKNOWN"
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    temporal_context: str | None = None
    speaker: str | None = None
    document_available: bool = True
    document_content_verified: bool = False
    statement_status: str | None = None
    underlying_event_status: str | None = None


class EvidenceClaim(BaseModel):
    """Claim-level evidence with full provenance."""
    claim_id: str                    # e.g. "C1"
    text: str                        # The actual claim text
    subject: str | None = None       # What entity the claim is about
    predicate: str | None = None     # What is asserted about the subject
    source: str                      # Where this claim came from
    status: EvidenceStatus           # VERIFIED/REPORTED/INFERRED/UNKNOWN/CONTRADICTED
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    evidence_reference: str | None = None
    attribution: ClaimAttribution = ClaimAttribution.UNKNOWN
    polarity: str | None = None      # "positive"/"negative"/None
    speaker: str | None = None
    sentence_group: int | None = None
    source_type: str = "AUDIT_OBSERVATION"
    availability: Literal["AVAILABLE", "UNAVAILABLE", "NOT_APPLICABLE"] = "AVAILABLE"
    document_available: bool = True
    document_content_verified: bool = False
    statement_status: str | None = None
    underlying_event_status: str | None = None

    @property
    def provenance(self) -> str:
        if self.status == EvidenceStatus.INFERRED:
            return "INFERRED"
        if self.status == EvidenceStatus.VERIFIED:
            return "VERIFIED"
        return "REPORTED"

    @property
    def verification(self) -> str:
        if self.status == EvidenceStatus.VERIFIED:
            return "VERIFIED"
        if self.status == EvidenceStatus.CONTRADICTED:
            return "CONTRADICTED"
        if self.status == EvidenceStatus.UNKNOWN:
            return "NOT_ASSESSABLE"
        return "UNVERIFIED"


class EvidenceConflict(BaseModel):
    """A detected conflict between two or more claims about the same proposition."""
    conflict_id: str               # e.g. "CONF1"
    conflict_type: Literal[
        "DELIVERY_VS_RECEIPT", "RECORD_VS_STATEMENT", "COMPLETION_VS_MISSING_RECORD",
        "SYSTEM_RECORD_VS_HUMAN_REPORT", "SYSTEM_VS_HUMAN_REPORT", "DOCUMENT_VS_STATEMENT", "TIMESTAMP_VS_EVENT",
        "CONFLICTING_REPORTS", "CONTRADICTED_BY_EVIDENCE", "INCONSISTENT_RECORDS", "OTHER"
    ] = "CONFLICTING_REPORTS"
    proposition_type: Literal[
        "DELIVERY_VS_RECEIPT", "COMPLETION_VS_MISSING_RECORD", "RECORD_VS_STATEMENT",
        "TIMESTAMP_VS_REPORTED_EVENT", "SYSTEM_RECORD_VS_HUMAN_REPORT", "SYSTEM_STATE_VS_HUMAN_REPORT", "CONFLICTING_REPORTS",
    ] = "CONFLICTING_REPORTS"
    status: Literal["UNRESOLVED", "RESOLVED_FOR", "RESOLVED_AGAINST"] = "UNRESOLVED"
    claims: list[str] = []              # claim_ids involved
    proposition_a_id: str | None = None
    proposition_b_id: str | None = None
    proposition: str = ""               # The proposition they disagree about
    resolution_required: bool = True
    resolution_evidence: str | None = None
    resolution_note: str | None = None
    severity: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"


class CanonicalFindingState(BaseModel):
    """The single canonical intermediate representation extracted from raw finding text (Section 1).

    All downstream nodes (RCA, 5-Why, Investigation, CAPA, Impact) MUST consume this state
    rather than independently re-interpreting the raw finding.
    """
    raw_finding: str
    observed_deviation: str
    # Canonical finding semantics (Requirement 1 & 11)
    finding_subject: str = "UNKNOWN"
    affected_object: str = "UNKNOWN"
    affected_process: str = "UNKNOWN"
    affected_activity: str = "UNKNOWN"
    deviation: str = "UNKNOWN"
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
    time_period: str = "UNKNOWN"
    # The person/role identified as responsible/involved, if the finding names one
    # (e.g. "the responsible technician") — distinct from affected_people (who was
    # impacted) since an actor may be neither affected nor a department.
    actor: str | None = None
    actors: list[str] = []
    entities: list[str] = []
    # Layer 2 of the causal chain (Layer 1 is observed_deviation): the
    # action-level explanation of HOW the deviation happened, when the
    # finding/evidence directly states one (e.g. "the check was missed",
    # distinct from the observation "the log was incomplete"). Null if no
    # claim in the finding states a mechanism at this level of specificity.
    immediate_mechanism: str | None = None
    reported_mechanism: str | None = None
    verified_mechanism: str | None = None
    # VERIFIED (from a stated fact) | REPORTED (from an attributed
    # statement) | UNKNOWN (no mechanism-level claim found).
    immediate_mechanism_status: str = "UNKNOWN"
    mechanism_status: str = "UNKNOWN"
    mechanism_polarity: str | None = None
    process: str = "UNKNOWN"
    procedure: str = "UNKNOWN"
    requirement: str = "UNKNOWN"
    facts: list[str] = []
    verified_observations: list[str] = []
    reported_statements: list[str] = []
    inferred_statements: list[str] = []
    unknowns: list[str] = []
    contradictions: list[str] = []
    prompt_injection_detected: bool = False
    # Claim-level decomposition with full provenance (Phase 1)
    evidence_claims: list[EvidenceClaim] = []
    # Detected conflicts between claims about the same proposition
    evidence_conflicts: list[EvidenceConflict] = []
    propositions: list[Proposition] = []
    investigation_mode: InvestigationMode = InvestigationMode.NORMAL
    evidence_completeness: EvidenceCompleteness = EvidenceCompleteness.COMPLETE
    # Documents the finding references/cites/attaches but which were not
    # available for inspection -- kept separate from evidence_claims so a
    # referenced document's type/name can never be mistaken for an
    # established claim about its contents (Referenced-Evidence Boundary).
    referenced_documents: list[ReferencedDocumentInfo] = []

    # ------------------------------------------------------------------
    # Temporal / recurrence reasoning (app.agent.recurrence_guard). A
    # recurring finding requires a materially different RCA strategy: the
    # question isn't just "why did this happen" but "why didn't the
    # previous corrective action prevent it happening again" -- and
    # completion of that previous action is NEVER treated as proof it was
    # effective.
    # ------------------------------------------------------------------
    recurrence_signal: bool = False
    previous_capa_referenced: bool = False
    previous_capa_status: str | None = None  # "COMPLETED" | None
    previous_capa_effectiveness: str = "NOT_VERIFIED"  # "EFFECTIVE" | "NOT_VERIFIED"
    recurrence_rationale: str | None = None



class ReferencedDocumentInfo(BaseModel):
    """A document/record the finding mentions, cites, or attaches but which
    was not actually available for inspection (Referenced-Evidence Boundary).

    REFERENCED EVIDENCE != INSPECTED EVIDENCE: a document being named is
    evidence that it was named, never evidence of what it contains. This is
    kept as separate metadata precisely so its identity/type (e.g.
    "calibration report") can be preserved for the record without ever being
    treated as a claim about the finding's actual subject, deviation, or
    cause -- see semantic_subject.detect_referenced_unavailable_documents,
    the single structural (provenance-based, not vocabulary-based) producer
    of this field.
    """
    document_type: str                      # raw noun phrase, e.g. "calibration report"
    reference_status: Literal["REFERENCED_UNAVAILABLE", "REFERENCED_AVAILABLE"] = "REFERENCED_UNAVAILABLE"
    content_status: Literal["UNKNOWN", "VERIFIED"] = "UNKNOWN"
    raw_span: str | None = None             # the sentence/clause this was detected in


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
    # MIXED: the answer combines clauses with DIFFERENT evidence provenance
    # CONFLICTING / CONFLICTING_REPORTS: conflicting reported statements on a proposition
    status: Literal["VERIFIED", "SUPPORTED", "REPORTED", "REPORTED_STATEMENT", "REPORTED_UNVERIFIED", "MIXED", "CONFLICTING", "CONFLICTING_REPORTS", "INFERRED", "UNKNOWN", "REQUIRES_EVIDENCE", "NOT_ESTABLISHED"]



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
    ESTABLISHED = "ESTABLISHED"
    STATED_UNVERIFIED = "STATED_UNVERIFIED"
    INFERRED = "INFERRED"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"
    CONTRADICTED = "CONTRADICTED"
    CONFLICTED = "CONFLICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class CandidateHypothesis(BaseModel):
    id: str  # e.g. "H1"
    name: str  # e.g. "TRAINING_ASSIGNMENT"
    statement: str
    status: Literal["POSSIBLE", "SUPPORTED", "REFUTED", "UNRESOLVED", "UNVERIFIED"] = "POSSIBLE"
    evidence_needed: str
    discrimination_evidence: str | None = None  # what evidence would distinguish THIS hypothesis from the others
    relevance_rank: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"  # Section 7 evidence relevance ranking
    resolves_investigation: str | None = None
    # Why this hypothesis is plausible for THIS finding specifically — distinct
    # from `statement` (the claim itself) so plausibility reasoning isn't lost.
    rationale: str | None = None
    # Supporting evidence list
    supporting_evidence: list[str] = []
    # Facts from the evidence ledger that argue against this hypothesis, if any.
    contradicting_evidence: list[str] = []
    evidence_against: str | None = None
    # What specific result would CONFIRM this hypothesis.
    confirms_if: str | None = None
    # What specific result would REFUTE this hypothesis.
    refutes_if: str | None = None
    # Provenance tracking: IDs of claims (C1, C2, ...) or propositions supporting/contradicting
    supporting_claim_ids: list[str] = []
    contradicting_claim_ids: list[str] = []
    causal_level: CausalLevel = CausalLevel.L3_IMMEDIATE_MECHANISM
    # Evidence strength categorization
    evidence_strength: Literal["NONE", "REPORTED", "CORROBORATED", "VERIFIED", "CONFLICTING"] = "NONE"
    # Deterministic per-hypothesis confidence grade (never asserted by the
    # LLM — set by app.agent.analytical_validator.hypothesis_confidence so
    # every hypothesis in a report carries a grade, not just the leading one.
    confidence: Literal["HIGH", "MEDIUM", "LOW"] | None = None



class RootCauseAnalysis(BaseModel):
    status: RootCauseStatus
    category: str | None = None  # 6M taxonomy value
    statement: str | None = None
    leading_hypothesis: str | None = None
    # WHY leading_hypothesis is what it is -- set by
    # app.agent.analytical_validator.leading_hypothesis_status so a report
    # can distinguish "no hypotheses were generated" from "hypotheses exist
    # but are genuinely tied" instead of collapsing both into a blank field.
    leading_hypothesis_status: Literal["SELECTED", "TIED", "NONE"] = "NONE"
    candidate_hypotheses: list[CandidateHypothesis] = []
    supporting_evidence: list[str] = []
    contradicting_evidence: list[str] = []
    missing_evidence: list[str] = []
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    narrative: str = ""
    evidence_status: EvidenceStatus = EvidenceStatus.UNKNOWN
    verification_needed: str | None = None
    # WHY the root cause is/isn't established — the analytical reasoning
    # chain, distinct from the narrative (which describes the finding
    # situation) and from the leading hypothesis (which is the best guess).
    root_cause_basis: str | None = None
    # What evidence would establish the root cause — populated whether or
    # not the root cause is currently established, because even an
    # established root cause should say what confirmed it.
    evidence_required: list[str] = []
    # WHY the leading hypothesis is the leading hypothesis — the specific
    # reasoning that ranked it above the others.
    leading_hypothesis_rationale: str | None = None


# ---------------------------------------------------------------------------
# Core synthesis LLM contracts
# ---------------------------------------------------------------------------


class CoreSynthesisHypothesisLLM(BaseModel):
    id: str = "H1"
    name: str = "HYPOTHESIS"
    statement: str = ""
    supporting_claim_ids: list[str] = []
    contradicting_claim_ids: list[str] = []
    target_proposition_id: str | None = None
    status: str = "POSSIBLE"
    evidence_needed: str | None = None
    confirms_if: str | None = None
    refutes_if: str | None = None
    discrimination_evidence: str | None = None
    relevance_rank: str = "HIGH"
    rationale: str | None = None
    evidence_against: str | None = None


class CoreSynthesisFiveWhyStepLLM(BaseModel):
    level: int | None = 1
    question: str = ""
    answer: str = ""
    status: str = "UNKNOWN"
    evidence_reference: str | None = None


class CoreSynthesisFiveWhyLLM(BaseModel):
    steps: list[CoreSynthesisFiveWhyStepLLM] = []
    is_complete: bool = False
    status_note: str | None = None


class CoreSynthesisRootCauseLLM(BaseModel):
    status: str = "NOT_ESTABLISHED"
    category: str = "TO_BE_CONFIRMED"
    statement: str | None = None
    leading_hypothesis: str | None = None
    root_cause_basis: str | None = None
    evidence_required: list[str] = []
    candidate_hypotheses: list[CoreSynthesisHypothesisLLM] = []
    narrative: str = "The available evidence establishes the observed condition but does not establish why it occurred."
    risk_of_recurrence: str | None = "NOT_ASSESSABLE"
    leading_hypothesis_rationale: str | None = None


class CoreSynthesisOutput(BaseModel):
    root_cause: CoreSynthesisRootCauseLLM = Field(default_factory=CoreSynthesisRootCauseLLM)
    five_why: CoreSynthesisFiveWhyLLM = Field(default_factory=CoreSynthesisFiveWhyLLM)
    contributing_factors: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Contributing factors
# ---------------------------------------------------------------------------


class ContributingFactor(BaseModel):
    description: str
    evidence_status: EvidenceStatus = EvidenceStatus.INFERRED
    status: Literal["ESTABLISHED", "POTENTIAL", "POSSIBLE_UNCONFIRMED", "VERIFIED", "REJECTED", "UNKNOWN"] = "POTENTIAL"
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
    """A structured investigation question represented as a conditional decision tree node.

    Each node specifies its target proposition, objective, required evidence, possible outcomes,
    priority (P1-P6), activation condition, prerequisites (depends_on), branching next steps,
    and activation status.
    """
    id: str | None = None
    question_id: str | None = None
    question: str
    purpose: str = ""  # which proposition / unknown this resolves (synced with objective)
    objective: str = ""  # investigative objective / target
    evidence: str = ""  # specific document/record type that would answer this (synced with evidence_required)
    evidence_required: str | None = None
    target_type: Literal["HYPOTHESIS", "CONFLICT", "DOCUMENT", "OBSERVATION", "PROPOSITION", "OTHER"] = "PROPOSITION"
    target_id: str | None = None
    target_proposition_id: str | None = None
    question_type: str = "PROPOSITION_VERIFICATION"
    resolves: str | None = None
    decision_rule: str | None = None
    hypothesis_tested: str | None = None  # optional link to candidate hypothesis, never authoritative identity
    confirms_if: str | None = None
    supports_if: str | None = None
    refutes_if: str | None = None
    priority: str = "HIGH"  # P1-P6 or CRITICAL/HIGH/MEDIUM/LOW
    blocking: bool = True
    depends_on: str | list[str] | None = None
    activation_condition: str | None = None
    next_question_if_true: str | None = None
    next_question_if_false: str | None = None
    status: str = "ACTIVE"  # ACTIVE, CONDITIONAL, INACTIVE, RESOLVED
    resolution_rule: dict[str, str] = {}  # e.g. {"supports": "...", "weakens": "...", "remains_unresolved": "..."}
    presupposes_cause: bool = False
    presupposes_outcome: bool = False
    possible_outcomes: list[str] = []

    def model_post_init(self, __context: Any) -> None:
        if not self.question_id and self.id:
            self.question_id = self.id
        elif not self.id and self.question_id:
            self.id = self.question_id
        if not self.objective and self.purpose:
            self.objective = self.purpose
        elif not self.purpose and self.objective:
            self.purpose = self.objective
        if not self.evidence_required and self.evidence:
            self.evidence_required = self.evidence
        elif not self.evidence and self.evidence_required:
            self.evidence = self.evidence_required
        if self.activation_condition and self.status == "ACTIVE":
            self.status = "CONDITIONAL"


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
    action_type: Literal["IMMEDIATE_CORRECTION", "CONTAINMENT", "CORRECTIVE_ACTION", "SYSTEMIC_ACTION"] | None = None
    verification_method: str | None = None  # how effectiveness of this action would be verified
    evidence_needed: str | None = None
    effectiveness_owner: str | None = None
    effectiveness_review_period: str | None = None
    root_cause_hypothesis_id: str | None = None


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
    field_basis: dict[str, str] = {}
    impact_observed: str | None = None   # what is objectively verified
    impact_inferred: str | None = None   # what is logically inferred from evidence
    impact_unknown: str | None = None    # what remains unknown



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
    investigation_mode: InvestigationMode = InvestigationMode.NORMAL
    evidence_completeness: EvidenceCompleteness = EvidenceCompleteness.COMPLETE
    root_cause: RootCauseAnalysis
    contributing_factors: list[ContributingFactor] = []
    investigation: InvestigationPlan
    five_why: FiveWhyAnalysis
    capa: CapaAnalysis
    impact_assessment: ImpactAssessment
    evidence_gaps: list[EvidenceGap] = []
    evidence: list[EvidenceItem] = []
    propositions: list[Proposition] = []
    evidence_claims: list[EvidenceClaim] = []
    evidence_conflicts: list[EvidenceConflict] = []
    referenced_documents: list[ReferencedDocumentInfo] = []
    human_review_required: bool = True  # always True -- enforced here, not just prompted
    analysis_mode: Literal["LLM", "DETERMINISTIC", "DEGRADED"] = "LLM"
    analysis_engine: Literal["LLM", "DETERMINISTIC"] = "LLM"
    provider_used: str | None = None
    fallback_used: bool = False
    provider_attempts: list[str] = []
    critic_status: str | None = None

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
    llm_provider: str = ""
    copilot_github_token: str = ""

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
