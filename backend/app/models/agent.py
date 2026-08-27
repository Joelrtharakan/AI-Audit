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

from app.financial.models import FinancialAnalysisResult


# ---------------------------------------------------------------------------
# Evidence classification
# ---------------------------------------------------------------------------


class EvidenceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    REPORTED = "REPORTED"
    MIXED = "MIXED"
    INFERRED = "INFERRED"
    # An entity's epistemic STANCE about the world ("the security team
    # believes the two events are related") -- strictly weaker than
    # REPORTED. REPORTED means a person asserted something they may have
    # directly observed; BELIEF means a person asserted a mental state
    # ABOUT something. Because every causal-strength filter in the codebase
    # tests `== VERIFIED` or `== REPORTED` explicitly, BELIEF is invisible
    # to all of them by construction and can therefore never support a
    # SUPPORTED/ESTABLISHED hypothesis.
    BELIEF = "BELIEF"
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


class SemanticUncertainty(str, Enum):
    REQUIREMENT_UNCERTAIN = "REQUIREMENT_UNCERTAIN"
    OBSERVATION_UNCERTAIN = "OBSERVATION_UNCERTAIN"
    SCOPE_UNCERTAIN = "SCOPE_UNCERTAIN"
    TIME_UNCERTAIN = "TIME_UNCERTAIN"
    RESPONSIBILITY_UNCERTAIN = "RESPONSIBILITY_UNCERTAIN"
    AUTHORIZATION_UNCERTAIN = "AUTHORIZATION_UNCERTAIN"
    CONTROL_EXECUTION_UNCERTAIN = "CONTROL_EXECUTION_UNCERTAIN"
    DOCUMENTATION_UNCERTAIN = "DOCUMENTATION_UNCERTAIN"
    EVENT_SEQUENCE_UNCERTAIN = "EVENT_SEQUENCE_UNCERTAIN"
    MECHANISM_UNCERTAIN = "MECHANISM_UNCERTAIN"
    CAUSAL_LINK_UNCERTAIN = "CAUSAL_LINK_UNCERTAIN"
    IMPACT_UNCERTAIN = "IMPACT_UNCERTAIN"
    RECURRENCE_UNCERTAIN = "RECURRENCE_UNCERTAIN"
    RECOVERY_UNCERTAIN = "RECOVERY_UNCERTAIN"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    NO_MATERIAL_UNCERTAINTY = "NO_MATERIAL_UNCERTAINTY"


class CausalReadiness(str, Enum):
    NOT_READY = "NOT_READY"
    READY_FOR_HYPOTHESIS = "READY_FOR_HYPOTHESIS"
    READY_FOR_CAUSAL_VERIFICATION = "READY_FOR_CAUSAL_VERIFICATION"
    ESTABLISHED = "ESTABLISHED"



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
    NON_ACTIONABLE = "NON_ACTIONABLE"


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


# ---------------------------------------------------------------------------
# Real runtime CausalGraph (Phase 3 hardening).
#
# `CausalRelationship` above is retained but was never populated by the
# production pipeline (confirmed by source audit: read in 4 places, written
# in zero). This is the actual populated runtime causal graph — built by
# `app.agent.causal_graph.build_causal_graph` from already-epistemically-
# disciplined `CandidateHypothesis` data (never from raw LLM prose, never
# from normative semantic-graph edges), and attached to AgentState as
# `causal_graph` by `final_evidence_verification_node`.
# ---------------------------------------------------------------------------


class CausalGraphNodeType(str, Enum):
    OBSERVED_DEVIATION = "OBSERVED_DEVIATION"
    IMMEDIATE_MECHANISM = "IMMEDIATE_MECHANISM"
    CONTRIBUTING_FACTOR = "CONTRIBUTING_FACTOR"
    UNDERLYING_CAUSE = "UNDERLYING_CAUSE"
    SYSTEMIC_ROOT_CAUSE = "SYSTEMIC_ROOT_CAUSE"
    UNKNOWN = "UNKNOWN"
    UNRESOLVED = "UNRESOLVED"


class CausalGraphEdgeStatus(str, Enum):
    VERIFIED = "VERIFIED"
    REPORTED = "REPORTED"
    POSSIBLE = "POSSIBLE"
    UNKNOWN = "UNKNOWN"
    DISPUTED = "DISPUTED"
    REJECTED = "REJECTED"


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
    # An opinion/belief/suspicion/assumption held by a person or team --
    # distinct from PERSON_REPORTED (an account of something observed).
    PERSON_BELIEF = "PERSON_BELIEF"
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


class CausalGraphNode(BaseModel):
    """A single node on the real runtime causal ladder."""
    node_id: str
    node_type: CausalGraphNodeType
    label: str
    causal_level: CausalLevel = CausalLevel.EVIDENCE_STATE
    epistemic_status: EvidenceStatus = EvidenceStatus.UNKNOWN
    proposition_ids: list[str] = []
    evidence_ids: list[str] = []
    semantic_node_ids: list[str] = []
    provenance: EpistemicSource = EpistemicSource.AUDIT_OBSERVATION
    confidence: Literal["HIGH", "MEDIUM", "LOW"] | None = None
    source_hypothesis_id: str | None = None
    # Phase 11 Step 6: a short, controlled-vocabulary concept reference
    # (from CandidateHypothesis.name, e.g. "TRAINING_NOT_COMPLETED"),
    # distinct from `label` (the full hypothesis statement). `label` may
    # legitimately carry hedged/modal prose ("may not have been
    # completed") since it's the LLM's own descriptive sentence; `label` is
    # fine to render directly for a VERIFIED edge (an established fact),
    # but embedding it in a hedged rendering for a non-VERIFIED edge leaks
    # that modal language back out as if the renderer had asserted it.
    # `concept_ref` exists so the renderer has a genuinely neutral noun
    # phrase to name the candidate without repeating its prose claim.
    concept_ref: str | None = None
    # Phase 12: structured comparison/measurement context, populated ONLY
    # from CanonicalFindingState's own typed comparison_*/measurement
    # fields (never re-derived from raw text) — carried here so
    # render_causal_transition_answer can reuse the SAME comparison-
    # subtype-keyed mechanism-category logic build_causal_boundary_answer
    # already uses, instead of losing quantitative precision when
    # rendering a graph-derived transition for a comparison-type finding.
    # Flat scalar fields (not a nested SemanticMeasurement reference) to
    # match build_causal_boundary_answer's existing signature exactly and
    # avoid forward-reference ordering with a model defined later in this
    # file. Unset (None) for the overwhelming majority of nodes — only the
    # OBSERVED_DEVIATION node of a comparison-type finding populates these.
    comparison_type: str | None = None
    comparison_left: str | None = None
    comparison_left_qualifier: str | None = None
    comparison_right: str | None = None
    comparison_subtype: str | None = None
    measurement_value: float | None = None
    measurement_unit: str | None = None
    measurement_qualifier: str | None = None
    measurement_evidence_status: str | None = None


class CausalGraphEdge(BaseModel):
    """A single licensed causal edge. Only ever created by
    `build_causal_graph` when an explicit licensing condition (Section 6:
    verified objective evidence, an explicitly verified mechanism, a
    verified causal proposition, or an explicit reported causal mechanism)
    is met — never from co-occurrence, temporal order, or a normative
    relation (VIOLATES/REQUIRES/GOVERNS/SUBJECT_TO/SATISFIES/CONFORMS_TO)."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    relation_type: CausalEdgeType = CausalEdgeType.RESULTS_IN
    epistemic_status: EvidenceStatus = EvidenceStatus.UNKNOWN
    evidence_ids: list[str] = []
    proposition_ids: list[str] = []
    provenance: EpistemicSource = EpistemicSource.AUDIT_OBSERVATION
    confidence: Literal["HIGH", "MEDIUM", "LOW"] | None = None
    causal_level_transition: str | None = None  # e.g. "OBSERVED_DEVIATION->IMMEDIATE_MECHANISM"
    status: CausalGraphEdgeStatus = CausalGraphEdgeStatus.UNKNOWN
    notes: str | None = None
    # Phase 6 Section 4: explicit derivation marker. The edge's EXISTENCE is
    # always licensed by evidence_strength/status (never by evidence
    # overlap — see build_causal_graph's `_edge_status_for`); overlap is
    # used ONLY to decide the edge's TOPOLOGY (which node it is sourced
    # from) once existence is already licensed. This field makes that
    # distinction machine-readable rather than only documented in a
    # docstring: "DIRECT" = sourced from the observed deviation;
    # "EVIDENCE_CORRELATED" = re-parented onto a shallower causal node
    # because of shared supporting evidence (multi-hop chaining).
    # Phase 7: "EXPLICIT" = sourced from an asserted `deepens_hypothesis_id`
    # reference (Section 5's structured causal candidate) — ranked above
    # EVIDENCE_CORRELATED (inferred from shared evidence) when both would
    # apply, and both ranked above the default DIRECT (sourced straight
    # from the observed deviation, no deeper chaining asserted or inferred).
    derivation: Literal["DIRECT", "EVIDENCE_CORRELATED", "EXPLICIT"] = "DIRECT"


class CausalGraph(BaseModel):
    """The real runtime causal graph. May legitimately contain a single
    OBSERVED_DEVIATION node and zero edges — that is a valid, complete
    graph state (evidence boundary reached immediately), not an error."""
    nodes: list[CausalGraphNode] = []
    edges: list[CausalGraphEdge] = []


class CausalPath(BaseModel):
    """Phase 9 Step 6: a real directed traversal from the observed deviation
    to one hypothesis's terminal node — built ONLY by
    app.agent.causal_graph.build_causal_paths as a pure derivation from
    already-licensed CausalGraph edges. Never constructed from prose, never
    hand-populated to satisfy a schema. Competing hypotheses (Step 5) each
    get their OWN CausalPath — independent hypotheses are never merged into
    one path merely because they originate from the same deviation node."""
    path_id: str
    hypothesis_id: str | None = None
    ordered_node_ids: list[str] = []
    ordered_edge_ids: list[str] = []
    starting_level: str
    terminal_level: str
    epistemic_status: str
    supporting_evidence_ids: list[str] = []
    contradicting_evidence_ids: list[str] = []


class RCACausalNodeRef(BaseModel):
    """A pure structural reference into CausalGraph — never free text
    beyond the node's own already-licensed label (Phase 6 Section 11:
    "Every causal statement must reference causal_node_id ...", "Do not
    generate a plausible textual root cause")."""
    causal_node_id: str
    label: str
    causal_level: str
    epistemic_status: str
    evidence_ids: list[str] = []
    proposition_ids: list[str] = []


class RCACausalEdgeRef(BaseModel):
    causal_edge_id: str
    source_node_id: str
    target_node_id: str
    status: str
    evidence_ids: list[str] = []


class ImpactGraphRef(BaseModel):
    """A structural impact reference — never a generated harm narrative.
    `basis` records WHY this counts as observed/potential/unknown: derived
    from the referenced node/edge's own epistemic_status, not asserted."""
    description: str
    causal_node_id: str | None = None
    causal_edge_id: str | None = None
    evidence_ids: list[str] = []
    basis: Literal["OBSERVED", "POTENTIAL", "UNKNOWN"]


class CausalGraphImpactProjection(BaseModel):
    """Phase 6 Section 14: `build_impact_from_graph` output.

    Reframes OBSERVED/POTENTIAL/UNKNOWN impact in terms the causal graph
    can answer without inventing a downstream-harm detector (which would
    require domain-specific "impact" vocabulary this architecture must not
    hardcode): OBSERVED = the verified deviation itself; POTENTIAL = causal
    nodes whose mechanism is POSSIBLE/REPORTED (an unconfirmed downstream
    effect IS a potential impact, structurally); UNKNOWN = unresolved
    edges, each requiring targeted investigation before impact can be
    assessed. Severity of the finding's wording plays no role."""
    observed: list[ImpactGraphRef] = []
    potential: list[ImpactGraphRef] = []
    unknown_investigation_required: list[ImpactGraphRef] = []


class CausalGraphRCAProjection(BaseModel):
    """Phase 6 Section 11: `build_rca_from_causal_graph` output. A pure
    STRUCTURAL PROJECTION of CausalGraph — contains no field that was not
    computed by reading the graph. If the graph has no established root
    cause, `root_cause_status` reflects NOT_ESTABLISHED; this model
    structurally cannot contain a plausible-sounding root cause the graph
    doesn't support, because every populated field is a list of graph
    node/edge references, not a text-generation slot."""
    observed_deviation: RCACausalNodeRef | None = None
    immediate_mechanisms: list[RCACausalNodeRef] = []
    contributing_factors: list[RCACausalNodeRef] = []
    underlying_causes: list[RCACausalNodeRef] = []
    systemic_root_causes: list[RCACausalNodeRef] = []
    competing_hypotheses: list[RCACausalNodeRef] = []
    unresolved_edges: list[RCACausalEdgeRef] = []
    evidence_boundary_reached: bool = False
    root_cause_status: Literal["ESTABLISHED", "NOT_ESTABLISHED"] = "NOT_ESTABLISHED"


class SemanticNodeType(str, Enum):
    ENTITY = "ENTITY"
    EVENT = "EVENT"
    STATE = "STATE"
    PROCESS = "PROCESS"
    CONTROL = "CONTROL"
    REQUIREMENT = "REQUIREMENT"
    RECORD = "RECORD"
    ACTOR = "ACTOR"
    ROLE = "ROLE"           # Organizational/functional role (distinct from the actor who holds it)
    ATTRIBUTE = "ATTRIBUTE"
    EVIDENCE = "EVIDENCE"   # An evidence artifact cited in a claim (distinct from RECORD which is the finding's target)
    OBSERVATION = "OBSERVATION"
    MEASUREMENT = "MEASUREMENT"

    # Semantic node type priority ordering for type-upgrade decisions.
    # Higher index = more specific; ENTITY is the least-specific fallback.
    @classmethod
    def specificity_rank(cls, node_type: "SemanticNodeType") -> int:
        _RANK = {
            cls.ENTITY: 0,
            cls.OBSERVATION: 1,
            cls.STATE: 2,
            cls.EVENT: 3,
            cls.MEASUREMENT: 4,
            cls.EVIDENCE: 5,
            cls.ATTRIBUTE: 6,
            cls.RECORD: 7,
            cls.ROLE: 8,
            cls.ACTOR: 9,
            cls.PROCESS: 10,
            cls.CONTROL: 11,
            cls.REQUIREMENT: 12,
        }
        return _RANK.get(node_type, 0)


class SemanticRelationType(str, Enum):
    # Semantic structural
    GOVERNS = "GOVERNS"
    TRANSMITTED_TO = "TRANSMITTED_TO"
    RECEIVED_BY = "RECEIVED_BY"
    ACCESSED_BY = "ACCESSED_BY"
    ACKNOWLEDGED_BY = "ACKNOWLEDGED_BY"
    EXECUTED_BY = "EXECUTED_BY"
    RECORDED_IN = "RECORDED_IN"
    APPLIES_TO = "APPLIES_TO"
    MONITORS = "MONITORS"
    VERIFIES = "VERIFIES"
    AUTHORIZES = "AUTHORIZES"
    DEVIATES_FROM = "DEVIATES_FROM"
    PRECEDES = "PRECEDES"
    RELATES_TO = "RELATES_TO"
    HAS_ATTRIBUTE = "HAS_ATTRIBUTE"

    # Normative / Compliance semantics
    SATISFIES = "SATISFIES"
    VIOLATES = "VIOLATES"
    REQUIRES = "REQUIRES"
    REQUIRES_ATTRIBUTE = "REQUIRES_ATTRIBUTE"
    LACKS_REQUIRED_ATTRIBUTE = "LACKS_REQUIRED_ATTRIBUTE"
    NOT_PERFORMED_AS_REQUIRED = "NOT_PERFORMED_AS_REQUIRED"
    NOT_DEMONSTRATED = "NOT_DEMONSTRATED"
    INCONSISTENT_WITH = "INCONSISTENT_WITH"
    WITHIN_REQUIREMENT = "WITHIN_REQUIREMENT"
    OUTSIDE_REQUIREMENT = "OUTSIDE_REQUIREMENT"
    SUBJECT_TO = "SUBJECT_TO"         # Entity is subject to a requirement/control
    CONFORMS_TO = "CONFORMS_TO"       # Entity conforms to a requirement (verified positive)

    # Set of relation types that are NORMATIVE — never causal.
    # Used by INV-ROLE-004 to detect normative-to-causal conflation.
    @classmethod
    def normative_relation_types(cls) -> frozenset["SemanticRelationType"]:
        return frozenset({
            cls.SATISFIES, cls.VIOLATES, cls.REQUIRES, cls.REQUIRES_ATTRIBUTE,
            cls.LACKS_REQUIRED_ATTRIBUTE, cls.NOT_PERFORMED_AS_REQUIRED,
            cls.NOT_DEMONSTRATED, cls.INCONSISTENT_WITH, cls.WITHIN_REQUIREMENT,
            cls.OUTSIDE_REQUIREMENT, cls.SUBJECT_TO, cls.CONFORMS_TO, cls.GOVERNS,
        })


class SemanticNode(BaseModel):
    """An atomic entity, event, state, requirement, or record node in the Semantic Graph."""
    id: str  # e.g. "N_SYS", "N_RECP", "N_NOTIF"
    label: str
    node_type: SemanticNodeType = SemanticNodeType.ENTITY
    epistemic_status: EvidenceStatus = EvidenceStatus.UNKNOWN
    provenance: EpistemicSource = EpistemicSource.AUDIT_OBSERVATION
    source_claim_ids: list[str] = []
    temporal_scope: str | None = None
    quantitative_scope: str | None = None


class SemanticEdge(BaseModel):
    """An explicit semantic relation edge connecting two semantic nodes."""
    id: str  # e.g. "E_DELIVERY"
    source_id: str
    target_id: str
    relation_type: SemanticRelationType = SemanticRelationType.RELATES_TO
    epistemic_status: EvidenceStatus = EvidenceStatus.UNKNOWN
    provenance: EpistemicSource = EpistemicSource.AUDIT_OBSERVATION
    source_claim_ids: list[str] = []
    notes: str | None = None


class SemanticGraph(BaseModel):
    """The explicit structural Semantic Graph answering: 'What entities, events, states, and relations are present?'"""
    nodes: list[SemanticNode] = []
    edges: list[SemanticEdge] = []


class OutputQualityDimension(BaseModel):
    """A single scored dimension in the structural quality assessment (Section 21)."""
    name: str
    score: int           # 0..max_score
    max_score: int
    passed: bool
    reason: str | None = None


class OutputQualityScore(BaseModel):
    """Composite structural quality score for the agent output (Section 21).

    Aggregates 10 independent structural dimensions. This is deliberately
    separate from LLM output fluency — a fluent, well-worded output that
    violates structural invariants will still receive a low score.

    Thresholds:
      score >= 80  → PASS
      60 <= score < 80 → WARNING (investigation report may be emitted with warnings)
      score < 60   → FAIL  (fail-closed gate fires; structured validation error returned)
    """
    dimensions: list[OutputQualityDimension] = []
    total_score: int = 0
    max_possible: int = 100
    grade: str = "UNKNOWN"          # PASS | WARNING | FAIL
    blocker_violations: list[str] = []
    critical_violations: list[str] = []
    computed_at: str | None = None


class SemanticTraceabilityEntry(BaseModel):
    """Traceability mapping linking an output field concept to its source proposition/semantic relation."""
    field_name: str  # e.g. "root_cause.statement", "impact_assessment.affected_object"
    concept: str
    source_proposition_ids: list[str] = []
    source_relation_ids: list[str] = []
    epistemic_status: str = "VERIFIED"
    provenance: str = "AUDIT_OBSERVATION"
    derivation_type: str | None = None  # None | "STRUCTURAL_INFERENCE" | "CAUSAL_PROGRESSION"
    confidence: str = "HIGH"


class SemanticTraceabilityMatrix(BaseModel):
    """Full machine-validated traceability matrix for all final report fields."""
    entries: list[SemanticTraceabilityEntry] = []
    is_valid: bool = True
    untraced_concepts: list[str] = []


class Proposition(BaseModel):
    """A canonical proposition in the causal graph with strict dimension atomicity."""
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
    participants: list[str] = []
    event_type: str | None = None
    state: str | None = None
    temporal_context: str | None = None
    temporal_scope: str | None = None
    quantitative_scope: str | None = None
    speaker: str | None = None
    document_available: bool = True
    document_content_verified: bool = False
    statement_status: str | None = None
    underlying_event_status: str | None = None
    derived_concept: bool = False
    derived_from: list[str] = []
    derivation_type: str | None = None


class HypothesisRelation(BaseModel):
    """Phase 23 Part B: a claim's relation to ONE specific hypothesis --
    explicitly separate from the claim's own epistemic weight
    (EvidenceClaim.status/epistemic_class, which describes the evidence's
    directness/verification level, never its bearing on any particular
    hypothesis). Root cause of the Phase 22 live-model finding: asking the
    LLM for a single field that had to answer both "what kind of evidence
    is this" AND "does it support/contradict this hypothesis" caused it to
    default to the easier, always-safe descriptive answer (OBSERVED)
    instead of committing to the harder relational judgment. Separating
    the two questions into two fields removes that overload."""
    hypothesis_id: str
    relation: Literal["SUPPORTING", "CONTRADICTING", "NEUTRAL", "INSUFFICIENT"]
    reason: str | None = None
    # Phase 24 Part D: outcome of the deterministic relation-validation
    # firewall (app.agent.evidence_interpreter.validate_relation). `relation`
    # above is always the FINAL, validated value; this field preserves what
    # the firewall actually did so the decision is auditable (Part Q: "For
    # each case record ... validation decision").
    validation_decision: Literal["ACCEPT", "DOWNGRADE_TO_NEUTRAL", "DOWNGRADE_TO_INSUFFICIENT", "REJECT"] = "ACCEPT"
    # Phase 25 Rule 6: closes the Phase 24 temporal-safety gap. Set by the
    # interpreter ONLY when this hypothesis is itself a causal claim (one
    # event caused/led to another) -- distinguishes "this relation is
    # grounded in temporal sequence alone" from "independent, non-temporal
    # evidence grounds it" (e.g. an explicit causal statement, a mechanism
    # description, or a directly-verified consequence). NOT_APPLICABLE for
    # every non-causal hypothesis relation. The deterministic firewall
    # (validate_relation) downgrades TEMPORAL_ONLY relations to NEUTRAL --
    # "A happened before B" alone can never SUPPORT/CONTRADICT "A caused B".
    causal_basis: Literal["INDEPENDENT_EVIDENCE", "TEMPORAL_ONLY", "NOT_APPLICABLE"] = "NOT_APPLICABLE"


class EvidenceProposition(str, Enum):
    """Phase 24 Part B/C: structural classification of WHAT KIND of
    proposition a claim makes -- orthogonal to epistemic status (VERIFIED/
    REPORTED, evidentiary weight) and to hypothesis relevance (SUPPORTING/
    CONTRADICTING, the relational judgment). Exists specifically to let the
    deterministic layer distinguish "a search found nothing" (which can
    never by itself prove non-occurrence) from "an authoritative record
    affirmatively states X did not occur" (which can) -- without any
    phrase/keyword list, by requiring the interpreter to name which of
    these four structural shapes the claim actually has."""
    POSITIVE_ASSERTION = "POSITIVE_ASSERTION"    # asserts something occurred/exists/is true
    NEGATIVE_ASSERTION = "NEGATIVE_ASSERTION"    # asserts something did NOT occur/exist/is false
    ABSENCE_OF_EVIDENCE = "ABSENCE_OF_EVIDENCE"  # a search/record contains nothing on the topic -- does NOT itself prove non-occurrence
    EVIDENCE_OF_ABSENCE = "EVIDENCE_OF_ABSENCE"  # an authoritative/exhaustive record affirmatively establishes non-occurrence


class QuantitativeAssertion(BaseModel):
    """Phase 24 Part F: a claim's numeric comparison, structured so the
    deterministic layer can independently verify the arithmetic rather than
    trusting the LLM's stated conclusion. Populated ONLY when the evidence
    text itself states both values and a comparison -- never computed or
    inferred by the LLM (enforced by the prompt, checked by
    app.agent.evidence_interpreter.verify_quantitative)."""
    left: float
    operator: Literal["GT", "LT", "EQ", "NE", "GE", "LE"]
    right: float
    unit: str | None = None


class TemporalRelation(str, Enum):
    """Phase 24 Part G: structural temporal relation between two events a
    claim references, when the evidence states one. Never used to infer
    causality -- temporal precedence alone is not causal proof."""
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    DURING = "DURING"
    OVERLAPS = "OVERLAPS"
    OUTSIDE_PERIOD = "OUTSIDE_PERIOD"
    UNKNOWN = "UNKNOWN"


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
    # ---- Orthogonal axes added for generalized evidence semantics ----
    # MODALITY is the grammatical MOOD of the proposition and is deliberately
    # a separate axis from `status` (evidentiary weight): a counterfactual
    # could itself be corroborated as "yes, this hypothetical was genuinely
    # raised" while remaining non-actual in content. Overloading
    # EvidenceStatus would have destroyed that distinction and forced every
    # existing status check to be rewritten.
    modality: Literal["ACTUAL", "CONDITIONAL", "COUNTERFACTUAL"] = "ACTUAL"
    modality_marker: str | None = None
    # The epistemic stance sub-type when status == BELIEF.
    epistemic_stance: str | None = None   # BELIEF|DOUBT|SUSPICION|ASSUMPTION|OPINION
    stance_holder: str | None = None
    # Phase 21: adaptive-loop provenance/relevance fields. Extends (never
    # duplicates) this canonical claim model with the identity/linkage the
    # EvidenceItem -> EvidenceInterpreter -> EvidenceClaim path needs. All
    # optional and unset for every pre-Phase-21 EvidenceClaim (finding-text
    # extraction via app.agent.claim_extractor) -- only claims produced by
    # app.agent.evidence_interpreter populate them.
    evidence_ids: list[str] = []
    hypothesis_ids: list[str] = []
    # The claim's relationship as interpreted from the evidence: whether it
    # is a direct OBSERVED/REPORTED account, or a judgment of its relevance
    # to a target hypothesis (SUPPORTING/CONTRADICTING), or UNKNOWN when the
    # evidence does not speak to the target at all. Orthogonal to `status`
    # (evidentiary weight) -- see app.agent.evidence_interpreter for how the
    # two axes combine without letting this field alone upgrade `status`.
    # Phase 21 vocabulary preserved for backward compatibility (existing
    # callers/tests that read this field or construct claims with it
    # directly). Phase 23: the interpreter no longer asks the LLM for this
    # value -- it derives it DETERMINISTICALLY from `status` (see
    # app.agent.evidence_interpreter._epistemic_class_for_status). Per-
    # hypothesis relevance now lives exclusively in `hypothesis_relations`.
    epistemic_class: Literal["SUPPORTING", "CONTRADICTING", "OBSERVED", "REPORTED", "UNKNOWN"] | None = None
    # Phase 23 Part B/D: explicit, structured relation of this claim to
    # each hypothesis it was interpreted against -- may legitimately be
    # SUPPORTING for one hypothesis and CONTRADICTING (or absent) for
    # another (Part K: one evidence item, multiple independent relations).
    hypothesis_relations: list[HypothesisRelation] = []
    temporal_context: str | None = None
    extraction_status: Literal["EXTRACTED", "REJECTED", "UNRESOLVED"] = "EXTRACTED"
    # Phase 24: structural safety fields. All optional/None for every
    # pre-Phase-24 claim (finding-text extraction via claim_extractor,
    # Phase 21-23 interpreter output) -- only claims produced by the
    # Phase-24 interpreter populate them.
    proposition_type: EvidenceProposition | None = None
    quantitative: QuantitativeAssertion | None = None
    temporal_relation: TemporalRelation | None = None

    @property
    def provenance(self) -> str:
        if self.status == EvidenceStatus.INFERRED:
            return "INFERRED"
        if self.status == EvidenceStatus.BELIEF:
            return "BELIEF"
        if self.modality != "ACTUAL":
            return self.modality
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


class SemanticMeasurement(BaseModel):
    """A typed, semantically-scoped numeric measurement extracted from a
    finding (Section 8) -- e.g. the 4.2% discrepancy between a recorded and
    calculated yield. Kept as its own typed object with an explicit `role`
    so downstream nodes can never silently reinterpret an OBSERVED_DISCREPANCY
    as a financial amount, a probability, or a confidence score."""
    value: float
    unit: str | None = None  # e.g. "%"
    qualifier: str | None = None  # e.g. "approximately"
    role: Literal["OBSERVED_DISCREPANCY"] = "OBSERVED_DISCREPANCY"
    evidence_status: Literal["VERIFIED", "REPORTED", "UNKNOWN"] = "UNKNOWN"
    source_claim_id: str | None = None


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
    relevant_change: str | None = None
    semantic_type: str | None = None
    affected_objects: list[str] = []
    affected_people: list[str] = []
    affected_departments: list[str] = []
    affected_equipment: list[str] = []
    affected_records: list[str] = []
    affected_period: str = "UNKNOWN"
    finding_detected_period: str | None = None
    transaction_period: str | None = None
    control_at_risk: str | None = None
    financial_amount: FinancialAmount | None = None
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
    # Canonical comparison/relational-finding event (Section 2/7): populated
    # only when the finding is a comparison ("X did not match Y", "X
    # exceeded Y", ...). Downstream nodes (5-Why, impact rendering) MUST
    # read these typed fields instead of re-deriving comparison semantics
    # from raw text.
    comparison_type: str | None = None  # MISMATCH/EXCEEDED/BELOW/INCONSISTENT/RECONCILIATION_FAILURE/MISSING/DUPLICATE
    comparison_left: str | None = None
    comparison_left_qualifier: str | None = None  # e.g. "recorded", "measured"
    comparison_right: str | None = None
    comparison_basis: str | None = None
    # Comparison SUBTYPE/investigation-framework selector (Section 4/5) --
    # e.g. PARAMETER_MISMATCH vs CALCULATION_MISMATCH -- and the reference
    # value's own type (e.g. "APPROVED_PARAMETER").
    comparison_subtype: str | None = None
    comparison_reference_type: str | None = None
    comparison_batch_id: str | None = None
    measurement: SemanticMeasurement | None = None
    # Missing-record/missing-documentation semantic fields (Section 2):
    # ACTIVITY_OR_CONTROL, REQUIRED_EVIDENCE context, and any DOWNSTREAM_EVENT
    # kept distinct so 5-Why/impact/investigation never conflate "documentation
    # is missing" with "the activity did not happen".
    missing_record_activity: str | None = None
    missing_record_context: str | None = None
    downstream_action_text: str | None = None
    downstream_action_present: bool = False
    occurrence_population: str | None = None
    attributed_source: str | None = None
    attributed_proposition: str | None = None
    transition_type: str | None = None
    control_justification_missing: bool = False
    requirement_status: str = "UNKNOWN"
    observed_entity: str | None = None
    affected_entity: str | None = None
    control_entity: str | None = None
    actor_entity: str | None = None
    location_entity: str | None = None
    primary_uncertainty: str = "NO_MATERIAL_UNCERTAINTY"
    secondary_uncertainties: list[str] = []
    blocked_reasoning_steps: list[str] = []
    causal_readiness: str = "NOT_READY"
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
    semantic_graph: SemanticGraph = Field(default_factory=SemanticGraph)

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
    cost_impact: CostImpact | None = None

    # ------------------------------------------------------------------
    # Security / input-integrity telemetry (prompt-injection hardening).
    # A SEPARATE dimension from evidence status -- never overloaded onto
    # EvidenceStatus. Populated by understand_finding_node from
    # app.services.instruction_detector.classify_instruction(); observable
    # via the API for administrators without exposing prompt internals.
    # ------------------------------------------------------------------
    input_integrity_status: Literal[
        "NORMAL", "QUOTED_INSTRUCTION", "INSTRUCTION_LIKE",
        "PROMPT_INJECTION_SUSPECTED", "MALICIOUS_INSTRUCTION",
    ] = "NORMAL"
    security_flags: list[str] = []
    instruction_like_claim_count: int = 0
    excluded_claim_texts: list[str] = []

    # ------------------------------------------------------------------
    # Semantic actionability gate
    # ------------------------------------------------------------------
    is_actionable: bool = True
    actionability_reason: str | None = None
    # ---- Entity-fidelity (Defect 3) ----
    # How confident the deterministic resolver is that finding_subject /
    # affected_object name the REAL entity in the finding. When RESOLVED is
    # not achievable the system must say so rather than emit a plausible
    # generic placeholder ("process compliance") that reads as a real
    # entity. PARTIAL = a genuine noun-phrase fragment was recovered but may
    # be imprecise; UNRESOLVED = nothing usable could be isolated.
    entity_resolution: Literal["RESOLVED", "PARTIAL", "UNRESOLVED"] = "RESOLVED"
    entity_resolution_note: str | None = None
    subject_unresolved: bool = False



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
    # Grammatical mood of the claim -- orthogonal to `status` (see
    # EvidenceClaim.modality). A non-ACTUAL claim is preserved in the ledger
    # (the audit brief forbids discarding the proposition) but can never be
    # VERIFIED.
    modality: Literal["ACTUAL", "CONDITIONAL", "COUNTERFACTUAL"] = "ACTUAL"
    epistemic_stance: str | None = None
    stance_holder: str | None = None
    # Phase 19: adaptive-loop provenance fields. Extends (rather than
    # duplicates, per the audit-first requirement) the existing canonical
    # evidence model with the identity/linkage EvidenceRequest ->
    # EvidenceProvider -> EvidenceItem needs. All optional and unset for
    # every pre-Phase-19 EvidenceItem (finding-text claims, tool results,
    # etc.) -- only evidence acquired through the new
    # app.agent.nodes.evidence_acquisition path populates them.
    evidence_id: str | None = None
    request_id: str | None = None
    question_id: str | None = None
    iteration_id: int | None = None
    collection_timestamp: str | None = None
    # The evidence's relationship to the SPECIFIC hypothesis/claim it was
    # requested to resolve -- orthogonal to `status` (EvidenceStatus is
    # about the evidence's own verification level; this is about what it
    # says relative to the target). SUPPORTING/CONTRADICTING/CONFLICTING/
    # INSUFFICIENT/UNAVAILABLE/UNRESOLVED per Phase 19 Section 3.
    hypothesis_relevance: Literal[
        "SUPPORTING", "CONTRADICTING", "CONFLICTING", "INSUFFICIENT", "UNAVAILABLE", "UNRESOLVED",
    ] | None = None
    supporting_claim_ids: list[str] = []
    contradicting_claim_ids: list[str] = []
    related_node_ids: list[str] = []
    related_edge_ids: list[str] = []


class EvidenceRequest(BaseModel):
    """Phase 19 Section 1/11: a structured request for evidence to resolve
    ONE Stage-B investigation target. Built deterministically from an
    InvestigationQuestion -- never from raw finding text -- and consumed by
    an EvidenceProvider implementation, which is responsible only for
    retrieval, never for causal judgment (Section 1: "Do not make the
    provider responsible for deciding causality")."""
    request_id: str
    question_id: str | None = None
    target_node_id: str | None = None
    target_edge_id: str | None = None
    hypothesis_ids: list[str] = []
    evidence_types: list[str] = []
    required_artifacts: list[str] = []
    objective: str = ""
    iteration_id: int = 0
    graph_version: int = 0
    # Phase 22 Part B/C/D: request lifecycle. `status` starts REQUESTED and
    # is updated deterministically by app.agent.nodes.evidence_acquisition
    # once evidence comes back (FULFILLED when it resolves the target,
    # UNRESOLVED when it doesn't -- never CANCELLED/SUPERSEDED by this
    # engine except when a genuine refinement request supersedes it, see
    # `parent_request_id`). Never wording-based -- see
    # `evidence_acquisition.request_identity` for the deterministic
    # structured identity these states key off of.
    status: Literal[
        "REQUESTED", "FULFILLED", "PARTIALLY_FULFILLED", "SUPERSEDED", "UNRESOLVED", "CANCELLED",
    ] = "REQUESTED"
    # Set ONLY when this request is a genuine refinement of an earlier
    # request that came back insufficient (same target, narrower/different
    # evidence requirement) -- the parent is marked SUPERSEDED, never
    # deleted, so the chain "why was R2 created" stays inspectable.
    parent_request_id: str | None = None


class HypothesisStatusChange(BaseModel):
    """Phase 19 Section 8: one entry in a hypothesis's status-transition
    history. Never overwritten -- app.agent.nodes.evidence_acquisition
    APPENDS one of these per transition, so "why did this hypothesis
    become supported" always has a real, inspectable provenance chain
    instead of only the current status."""
    hypothesis_id: str
    previous_status: str
    new_status: str
    changed_by_evidence_ids: list[str] = []
    changed_by_claim_ids: list[str] = []
    graph_version: int = 0
    iteration_id: int = 0
    reason: str = ""


# ---------------------------------------------------------------------------
# 5-Why analysis
# ---------------------------------------------------------------------------


class FiveWhyStep(BaseModel):
    question: str
    answer: str | None = None
    # MIXED: the answer combines clauses with DIFFERENT evidence provenance
    # CONFLICTING / CONFLICTING_REPORTS: conflicting reported statements on a proposition
    status: Literal["VERIFIED", "SUPPORTED", "REPORTED", "REPORTED_STATEMENT", "REPORTED_UNVERIFIED", "MIXED", "CONFLICTING", "CONFLICTING_REPORTS", "INFERRED", "UNKNOWN", "REQUIRES_EVIDENCE", "NOT_ESTABLISHED"]
    # Phase 3: causal-graph grounding fields. Populated ONLY when this step
    # corresponds to a real edge in AgentState["causal_graph"] (either
    # produced directly by graph traversal, or attached retroactively when
    # a prose-generated step is matched to a real edge) — never fabricated
    # to satisfy a schema requirement. A step with these fields unset is
    # honestly reporting that no causal-graph grounding exists for it.
    step_id: str | None = None
    source_node_id: str | None = None
    target_node_id: str | None = None
    causal_edge_id: str | None = None
    evidence_ids: list[str] = []
    proposition_ids: list[str] = []
    causal_level: str | None = None
    provenance: str | None = None
    # Phase 11 Step 5/8: explicit marker for a terminal evidence-boundary
    # step (no further causal edge is licensed from this node) versus a
    # real causal transition. Distinguishes "the chain legitimately
    # continues no further" from "this step IS a causal transition" —
    # never both true at once for the same step.
    boundary_status: Literal["TRANSITION", "EVIDENCE_BOUNDARY"] | None = None


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
    NOT_APPLICABLE = "NOT_APPLICABLE"


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
    # INDICATIVE: a common-factor/pattern-derived signal (e.g. "these three
    # departments all use the same shared system") that creates a candidate
    # hypothesis worth investigating but does NOT itself evidence the
    # hypothesis's mechanism -- distinct from REPORTED (someone's account of
    # HOW something happened) and never sufficient for promotion to
    # SUPPORTED/ESTABLISHED (only VERIFIED is).
    evidence_strength: Literal["NONE", "REPORTED", "CORROBORATED", "VERIFIED", "CONFLICTING", "INDICATIVE"] = "NONE"
    # Deterministic per-hypothesis confidence grade (never asserted by the
    # LLM — set by app.agent.analytical_validator.hypothesis_confidence so
    # every hypothesis in a report carries a grade, not just the leading one.
    confidence: Literal["HIGH", "MEDIUM", "LOW"] | None = None
    source_type: str = "INFERRED_INVESTIGATION_HYPOTHESIS"
    causal_role: Literal["PRIMARY_CAUSE", "CONTRIBUTING_CAUSE", "DETECTION_FAILURE", "SYSTEMIC_CAUSE", "IMPACT_FACTOR"] = "PRIMARY_CAUSE"
    # Phase 6 Section 3: graph-native back-reference. Populated ONLY by
    # app.agent.causal_graph.build_causal_graph, AFTER it has decided (via
    # the pre-existing, unchanged eligibility engine) whether this
    # hypothesis is licensed to exist as a causal graph node/edge at all —
    # never set by the LLM, never set by this model's own validators. A
    # hypothesis with these fields unset honestly reports it has no current
    # graph representation (e.g. REFUTED candidates are intentionally never
    # stamped — see Section 8, "not every hypothesis... is licensed to
    # exist as a graph node").
    causal_node_id: str | None = None
    causal_edge_id: str | None = None
    causal_edge_source_node_id: str | None = None
    # Phase 7 Section 5: EXPLICIT sequential causal chaining. When the LLM
    # (or the deterministic generator) asserts that THIS hypothesis is a
    # deeper causal explanation of another hypothesis already in the same
    # candidate_hypotheses list, it cites that hypothesis's `id` here. This
    # is validated (must reference a real id in the same response; a
    # dangling reference is dropped, never trusted blindly) and, when the
    # referenced hypothesis is itself graph-licensed, takes PRIORITY over
    # the Phase 4 evidence-overlap heuristic when build_causal_graph decides
    # this edge's source node — a real asserted transition, not an inferred
    # one from incidental shared evidence.
    deepens_hypothesis_id: str | None = None
    # Phase 9 Step 5/6: stamped by app.agent.causal_graph.build_causal_paths
    # with the id of this hypothesis's own independent CausalPath — never
    # shared across competing hypotheses, since each is evaluated on its
    # own supporting/contradicting evidence.
    causal_path_id: str | None = None
    # Phase 20 Part B/C: set to True ONLY by the single authoritative
    # evidence->hypothesis evaluator (app.agent.nodes.evidence_acquisition.
    # reconcile_hypothesis_from_evidence, called from the Phase 19 adaptive
    # loop). Every OTHER status-writing site in the codebase (the pre-
    # existing eligibility/promotion re-derivation loops in
    # final_evidence_verification_node) must skip a hypothesis with this
    # flag set rather than silently overwrite the authoritative decision --
    # this is what makes "exactly one epistemic authority" a real,
    # checkable runtime property (INV-INVEST-028) instead of a design
    # intention. False for every hypothesis whose status has only ever been
    # set by core_synthesis (initial LLM/deterministic synthesis) -- that
    # is a proposal, not yet an authoritative evidence-grounded decision,
    # and remains open to the existing eligibility/promotion logic exactly
    # as before Phase 20.
    status_locked: bool = False


class CausalSufficiencyAssessment(BaseModel):
    observation_sufficiency: Literal["SUFFICIENT", "INSUFFICIENT", "CONFLICTING"] = "SUFFICIENT"
    mechanism_sufficiency: Literal["ESTABLISHED", "SUPPORTED", "POSSIBLE", "NOT_ESTABLISHED", "UNKNOWN"] = "UNKNOWN"
    root_cause_sufficiency: Literal["ESTABLISHED", "SUPPORTED", "POSSIBLE", "NOT_ESTABLISHED", "UNKNOWN"] = "NOT_ESTABLISHED"
    systemic_sufficiency: Literal["ESTABLISHED", "SUPPORTED", "POSSIBLE", "NOT_ESTABLISHED", "UNKNOWN"] = "UNKNOWN"
    impact_sufficiency: Literal["OBSERVED", "POTENTIAL", "CONFIRMED", "UNKNOWN"] = "POTENTIAL"
    financial_sufficiency: Literal["CALCULATED", "ESTIMATED", "NOT_QUANTIFIED", "NOT_APPLICABLE"] = "NOT_APPLICABLE"


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
    causal_sufficiency: CausalSufficiencyAssessment | None = None
    risk_of_recurrence: Literal["LOW", "MEDIUM", "HIGH", "NOT_ASSESSABLE"] = "NOT_ASSESSABLE"
    risk_of_recurrence_rationale: str | None = None
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
    question_type: Literal["ROOT_CAUSE", "DETECTION_CONTROL", "FINANCIAL_IMPACT", "SYSTEMIC", "UNSPECIFIED"] = "ROOT_CAUSE"
    category: str | None = None
    decision_effect: str | None = None
    target_hypothesis_ids: list[str] = []
    hypotheses_tested: list[str] = []
    uncertainty_resolved: str | None = None
    next_step_if_true: str | None = None
    next_step_if_false: str | None = None
    decision_branches: list[str] = []
    discrimination_criterion: str | None = None
    # Phase 4: real graph-grounded fields. Populated ONLY when this question
    # was generated by app.agent.causal_graph.build_causal_uncertainty_graph
    # + the graph-grounded question deriver in plan_investigation_fallback —
    # never fabricated to satisfy a schema requirement. A question without
    # these fields set is honestly reporting it came from the prose/template
    # path, not the causal-graph path.
    target_node_id: str | None = None
    target_edge_id: str | None = None
    source_node_id: str | None = None
    causal_level: str | None = None
    unresolved_relation: str | None = None
    information_gain_rank: int | None = None
    # Phase 14 Section 5: deterministic ordinal information-gain classification
    # for a graph-grounded question, computed ONLY from the number of
    # proposition_ids structurally backing its target causal-uncertainty edge
    # (app.agent.causal_graph.rank_uncertainty_nodes_by_information_gain's
    # same signal) -- never from lexical similarity, question length, or a
    # fabricated probability. Unset for non-graph-grounded questions, same
    # honesty contract as target_node_id/target_edge_id above.
    information_gain_band: Literal["HIGH", "MEDIUM", "LOW"] | None = None
    information_gain_reason: str | None = None
    # Phase 17: which planning stage produced this question -- STAGE_A
    # (pre-synthesis, semantic-graph-only) or STAGE_B (post-synthesis,
    # reasons over CandidateHypothesis + licensed CausalGraph). Unset for
    # legacy-template questions, honestly reporting they came from neither.
    planner_stage: Literal["STAGE_A", "STAGE_B"] | None = None

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
        if not self.target_hypothesis_ids and self.hypothesis_tested:
            self.target_hypothesis_ids = [self.hypothesis_tested]
        if self.activation_condition and self.status == "ACTIVE":
            self.status = "CONDITIONAL"


class InvestigationPlan(BaseModel):
    areas: list[str] = []
    questions: list[InvestigationQuestion] = []
    evidence_to_collect: list[str] = []
    root_cause_questions: list[InvestigationQuestion] = []
    detection_control_questions: list[InvestigationQuestion] = []
    financial_questions: list[InvestigationQuestion] = []
    systemic_questions: list[InvestigationQuestion] = []
    # Phase 15 Section 9: makes the one case where this codebase already
    # legitimately produces zero questions (a non-actionable finding --
    # plan_investigation_node's fast path in app/agent/nodes/
    # investigation_planner.py) an explicit, machine-readable state instead
    # of an empty list a caller has to interpret. Every OTHER path through
    # build_deterministic_investigation_plan still always returns at least
    # one question (Section 8's "plan.questions is NEVER empty" guarantee,
    # unchanged this phase -- see the Phase 15 report for why a broader
    # NO_ACTIONABLE_UNCERTAINTY rollout was not attempted) — this field
    # honestly reports QUESTIONS_GENERATED for every one of those, not a
    # claim that a general graph-driven "nothing left to investigate"
    # judgment exists yet.
    status: Literal["QUESTIONS_GENERATED", "NO_ACTIONABLE_UNCERTAINTY"] = "QUESTIONS_GENERATED"
    # Phase 16 Section 35: real observability over which planning path
    # actually produced this plan's CONTENT. Populated by
    # app.agent.nodes.graph_investigation_planner.plan_investigation_graph_first
    # and consumed by app.agent.nodes.investigation_planner.plan_investigation_node
    # — never a decorative label. See the Phase 16 report for the honest
    # scope of what "GRAPH_FIRST" vs "LEGACY_FALLBACK" actually means here:
    # question CONTENT for an actionable finding still comes from the
    # deterministic template planner (a real, disclosed, unmigrated
    # dependency: CandidateHypothesis/CausalGraph do not exist yet at
    # investigation-planning time in the current LangGraph node ordering),
    # while NO_ACTIONABLE_UNCERTAINTY and TARGET_UNRESOLVED are genuine
    # graph-first structural judgments that own the plan outright.
    planner_mode: Literal[
        "NO_ACTIONABLE_UNCERTAINTY", "TARGET_UNRESOLVED", "GRAPH_FIRST", "LEGACY_FALLBACK",
    ] | None = None
    fallback_reason: str | None = None
    graph_targets_considered: int = 0
    graph_targets_selected: int = 0
    # Phase 17: which planning stage produced this PLAN's content (as
    # opposed to planner_mode, which describes the authoritative state).
    # STAGE_A = app.agent.nodes.graph_investigation_planner (pre-synthesis).
    # STAGE_B = app.agent.nodes.causal_investigation_planner (post-
    # synthesis, reasons over real CandidateHypothesis + CausalGraph).
    # None = legacy template planner.
    planner_stage: Literal["STAGE_A", "STAGE_B"] | None = None


class InvestigationIterationRecord(BaseModel):
    """Phase 18 Section 2: one structured, serializable record of a single
    Stage-B invocation, appended to AgentState["investigation_history"] by
    app.agent.nodes.causal_investigation_planner. Never natural language --
    every field is a real ID or structural signal, so the loop (and any
    caller) can answer "what changed since the last iteration" from state
    alone, not by re-reading generated prose."""
    iteration_id: int
    planner_stage: Literal["STAGE_A", "STAGE_B"] = "STAGE_B"
    investigation_question_ids: list[str] = []
    target_node_ids: list[str] = []
    target_edge_ids: list[str] = []
    hypothesis_ids: list[str] = []
    evidence_ids: list[str] = []
    causal_graph_version: int
    unresolved_targets_before: list[str] = []
    unresolved_targets_after: list[str] = []
    newly_resolved_targets: list[str] = []
    newly_created_uncertainties: list[str] = []
    planner_decision: Literal[
        "SAME_TARGET", "NEW_TARGET", "TARGET_RESOLVED", "HYPOTHESIS_ELIMINATED",
        "NEW_UNCERTAINTY", "NO_ACTIONABLE_UNCERTAINTY",
    ]
    termination_reason: Literal[
        "ROOT_CAUSE_VERIFIED", "CAUSAL_CHAIN_RESOLVED", "NO_ACTIONABLE_UNCERTAINTY",
        "EVIDENCE_BOUNDARY_REACHED", "EVIDENCE_EXHAUSTED", "MAX_ITERATIONS", "FAIL_CLOSED",
    ] | None = None


# ---------------------------------------------------------------------------
# CAPA analysis
# ---------------------------------------------------------------------------


class CapaStatus(str, Enum):
    CAPA_RECOMMENDED = "CAPA_RECOMMENDED"
    CAPA_DRAFT_POSSIBLE = "CAPA_DRAFT_POSSIBLE"
    INVESTIGATION_REQUIRED = "INVESTIGATION_REQUIRED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NO_CAPA_RECOMMENDATION_YET = "NO_CAPA_RECOMMENDATION_YET"
    NOT_APPLICABLE = "NOT_APPLICABLE"


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
    # Phase 22 Part H: traceability to the claims that grounded the
    # hypothesis this branch is conditional on, when available (mirrors
    # CandidateHypothesis.supporting_claim_ids -- never a second, competing
    # vocabulary).
    supporting_claim_ids: list[str] = []


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
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ImpactAssessment(BaseModel):
    status: ImpactStatus
    areas: list[str] = []
    narrative: str | None = None
    # Structured fields from the new spec — populated when the model provides them
    affected_object: str | None = None   # the actual process/record/output impacted
    affected_people: str | None = None   # who specifically was affected
    affected_period: str | None = None   # stated period or "requires confirmation"
    finding_detected_period: str | None = None
    transaction_period: str | None = None
    process_at_risk: str | None = None   # the process whose output is in question
    control_at_risk: str | None = None   # the control/prevention mechanism
    relevant_change: str | None = None   # what specifically changed (Rule 18: never assumed)
    potential_effect: str | None = None  # plausible downstream consequence
    evidence_needed: str | None = None   # what would bound the scope
    field_basis: dict[str, str] = {}
    impact_observed: str | None = None   # what is objectively verified
    impact_inferred: str | None = None   # what is logically inferred from evidence
    impact_unknown: str | None = None    # what remains unknown
    financial_amount: FinancialAmount | None = None


# ---------------------------------------------------------------------------
# Cost & Financial Impact (Sections 26-42)
# ---------------------------------------------------------------------------


class CostEvidenceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    REPORTED = "REPORTED"
    ESTIMATED = "ESTIMATED"
    INFERRED = "INFERRED"
    REQUIRES_ASSESSMENT = "REQUIRES_ASSESSMENT"
    UNKNOWN = "UNKNOWN"


class CostFactorType(str, Enum):
    DUPLICATE_PAYMENT = "DUPLICATE PAYMENT"
    OVERPAYMENT = "OVERPAYMENT"
    UNAUTHORIZED_PAYMENT = "UNAUTHORIZED PAYMENT"
    TRANSACTION_AMOUNT = "TRANSACTION AMOUNT"
    PAYMENT_AMOUNT = "PAYMENT AMOUNT"
    PURCHASE_VALUE = "PURCHASE VALUE"
    REWORK = "REWORK"
    SCRAP = "SCRAP"
    DOWNTIME = "DOWNTIME"
    REPLACEMENT = "REPLACEMENT"
    OVERTIME = "OVERTIME"
    LABOR = "LABOR"
    MATERIAL = "MATERIAL"
    MATERIAL_LOSS = "MATERIAL LOSS"
    PENALTY = "PENALTY"
    FINE = "FINE"
    REFUND = "REFUND"
    REVENUE_LOSS = "REVENUE LOSS"
    RECOVERY_COST = "RECOVERY COST"
    INVESTIGATION_COST = "INVESTIGATION COST"
    OPERATIONAL_LOSS = "OPERATIONAL LOSS"
    OTHER = "OTHER"


class FinancialAmount(BaseModel):
    amount: float | None = None
    formatted: str | None = None  # e.g. "₹125,000"
    currency: str | None = "INR"  # e.g. "INR", "USD", "EUR"
    factor: str | None = None  # e.g. "DUPLICATE_PAYMENT"
    source_claim_ids: list[str] = []
    support_status: Literal["VERIFIED", "REPORTED", "ESTIMATED", "UNVERIFIED", "UNKNOWN"] = "VERIFIED"
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"


class CostComponent(BaseModel):
    name: str
    amount: float | None = None
    currency: str | None = None
    category: str | None = None
    basis: str | None = None
    provenance: str = "UNKNOWN"  # VERIFIED, REPORTED, ESTIMATED, INFERRED, UNKNOWN


class CostImpact(BaseModel):
    cost_factor_detected: bool = False
    cost_factor_type: str | None = None
    financial_factor: str | None = None  # e.g. "DUPLICATE PAYMENT", "REWORK"
    financial_status: str | None = None  # VERIFIED_TRANSACTION, VERIFIED_LOSS, POTENTIAL_EXPOSURE, RECOVERED, RECOVERABLE, VERIFIED, REPORTED, ESTIMATED, REQUIRES_ASSESSMENT, UNKNOWN
    currency: str | None = None
    financial_amount: FinancialAmount | None = None
    transaction_amount: float | None = None
    gross_exposure: float | None = None
    outstanding_amount: float | None = None
    net_exposure: float | None = None
    actual_loss: float | None = None
    actual_loss_status: str | None = None  # "NOT_ESTABLISHED", "VERIFIED", "UNKNOWN", "ESTABLISHED"
    potential_exposure: float | None = None
    potential_cost_exposure: str | None = None
    recoverable_amount: float | None = None
    recovered_amount: float | None = None
    unrecovered_amount: float | None = None
    recoverability: str | None = None  # "UNKNOWN", "RECOVERABLE", "RECOVERED", "IRRECOVERABLE"
    recoverability_status: str | None = None  # "UNKNOWN", "REQUIRES_VERIFICATION", "RECOVERED", "PARTIALLY_RECOVERED", "IRRECOVERABLE"
    amount_confidence: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"] = "HIGH"
    classification_confidence: Literal["LOW", "MEDIUM", "HIGH"] = "HIGH"
    recovery_confidence: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"] = "HIGH"
    actual_loss_confidence: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"] = "HIGH"
    verified_cost: float | None = None
    reported_cost: float | None = None
    estimated_cost: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    cost_components: list[CostComponent] = []
    cost_drivers: list[str] = []
    calculation_basis: str | None = None
    assumptions: list[str] = []
    missing_cost_inputs: list[str] = []
    evidence_required: list[str] = []
    evidence_ids: list[str] = []
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    narrative: str | None = None


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
    mechanism_confidence: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    root_cause_confidence: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    impact_confidence: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
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
    cost_impact: CostImpact | None = None
    financial_amount: FinancialAmount | None = None
    financial_analysis: FinancialAnalysisResult | None = None
    evidence_gaps: list[EvidenceGap] = []
    evidence: list[EvidenceItem] = []
    propositions: list[Proposition] = []
    evidence_claims: list[EvidenceClaim] = []
    evidence_conflicts: list[EvidenceConflict] = []
    referenced_documents: list[ReferencedDocumentInfo] = []
    semantic_graph: SemanticGraph = Field(default_factory=SemanticGraph)
    semantic_traceability: SemanticTraceabilityMatrix = Field(default_factory=SemanticTraceabilityMatrix)
    causal_graph: CausalGraph | None = None
    # Phase 6 Section 11: additive structural RCA projection. Populated
    # alongside (never replacing) `root_cause` — a pure read of
    # `causal_graph`, containing no independently-generated causal text.
    rca_projection: CausalGraphRCAProjection | None = None
    # Phase 6 Section 14: additive structural impact projection. Populated
    # alongside (never replacing) `impact_assessment`.
    impact_graph_projection: CausalGraphImpactProjection | None = None
    # Phase 9 Step 6: real derived causal-path objects, one per hypothesis
    # with a licensed edge chain back to the deviation.
    causal_paths: list[CausalPath] = []
    human_review_required: bool = True  # always True -- enforced here, not just prompted
    analysis_mode: Literal["LLM", "DETERMINISTIC", "DEGRADED"] = "LLM"
    analysis_engine: Literal["LLM", "DETERMINISTIC"] = "LLM"
    provider_used: str | None = None
    fallback_used: bool = False
    provider_attempts: list[str] = []
    critic_status: str | None = None
    # Shadow-mode comparison (canonical semantic finding model pass):
    # points where the existing deterministic pipeline's own
    # interpretation disagreed with the LLM canonical semantic
    # interpretation. Purely observational -- the deterministic values
    # used elsewhere in this report remain authoritative; nothing here
    # ever overrides them. Empty whenever canonical semantic
    # interpretation was not attempted or was unavailable.
    semantic_pipeline_disagreements: list[dict] = Field(default_factory=list)

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
