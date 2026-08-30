"""Canonical semantic finding context: the schema for the ONE LLM
interpretation every downstream module (financial engine, investigation
planner, Five-Why, risk/impact) should reason from, instead of each module
independently re-deriving its own interpretation of the raw finding text.

Extends the financial semantic layer built in the previous pass
(`app.financial.semantic_models.SemanticFindingInterpretation`) rather than
duplicating it -- the financial claims/relationships/calculation proposals
ARE part of this canonical context, not a separate system.

This module defines DATA ONLY. It performs no interpretation, no
validation, and no arithmetic:
  - Interpretation: `app.services.canonical_finding_interpreter`
  - Validation:      `app.services.canonical_context_validator`
  - Arithmetic:       unchanged -- `app.financial.calculator` via
                       `app.financial.relationship_validator`
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.financial.semantic_models import EvidenceStatusStr, SemanticFindingInterpretation

# What kind of thing a piece of extracted meaning IS -- the single most
# important distinction this schema exists to make explicit, so a STATE
# word ("active", "valid", "in force") can never be mistaken for the
# ENTITY it describes, and a CONSEQUENCE/FINANCIAL_METRIC can never be
# mistaken for a CAUSE.
SemanticKind = Literal[
    "ENTITY", "STATE", "EVENT", "CONSEQUENCE", "FINANCIAL_METRIC",
    "HISTORICAL_CONTEXT", "REMEDIATION", "RECOVERY", "CAUSE", "HYPOTHESIS",
]


class CanonicalEntity(BaseModel):
    """One real-world thing the finding concerns (a contract, a line, a
    system, a role) with its STATE tracked as a separate field -- never
    folded into the entity name, and never itself treated as an entity."""

    entity_id: str
    name: str
    kind: SemanticKind = "ENTITY"
    state: str | None = None
    source_evidence_ids: list[str] = Field(default_factory=list)


class CausalClaim(BaseModel):
    """A statement that MAY assert a causal relationship. `is_causal`
    defaults to False -- it may only be True when the evidence text itself
    explicitly asserts causation (e.g. "X caused Y"), never merely because
    two facts co-occur or because one is a financial consequence of the
    other. A financial/historical/recovery/remediation claim is never, by
    itself, evidence of causation."""

    claim_id: str
    statement: str
    is_causal: bool = False
    cause_ref: str | None = None  # entity_id or claim_id
    effect_ref: str | None = None  # entity_id or claim_id
    source_evidence_ids: list[str] = Field(default_factory=list)
    evidence_status: EvidenceStatusStr = "UNVERIFIED"


MissingRecordStatus = Literal[
    "RECORD_EXISTS", "RECORD_INCOMPLETE", "RECORD_MISSING", "RECORD_UNAVAILABLE",
    "ACTIVITY_NOT_RECORDED", "ACTIVITY_NOT_PERFORMED", "UNKNOWN",
]
ComparisonDirection = Literal["ABOVE", "BELOW", "MISMATCH", "UNKNOWN"]
FindingEpistemicStatus = Literal["VERIFIED", "REPORTED", "BELIEF", "INFERRED", "UNKNOWN"]


class SemComparison(BaseModel):
    """A quantitative/relational comparison the finding states. Direction is
    taken ONLY from the finding's own wording -- a bare "differed" is
    MISMATCH, never guessed ABOVE/BELOW. magnitude/unit are null when the
    finding does not state them (never invented)."""

    left: str | None = None
    right: str | None = None
    reference: str | None = None      # the baseline/standard being compared against
    direction: ComparisonDirection = "UNKNOWN"
    magnitude: float | None = None
    unit: str | None = None           # "%", "units", "degrees", ...


class SemRecurrence(BaseModel):
    """An explicitly-stated repeated occurrence. Preservation only -- a bare
    count NEVER implies a root cause or a recurrence-risk classification."""

    count: int | None = None
    event: str | None = None          # "failures", "incidents", ...
    period: str | None = None         # "a six-month period", "the last quarter", ...


class EvidenceBoundary(BaseModel):
    """An explicit statement of what the evidence does NOT establish --
    the LLM is expected to name these, not merely omit unsupported
    content silently, so the auditor sees exactly where reasoning must
    stop."""

    description: str
    related_claim_ids: list[str] = Field(default_factory=list)


# --- LLM-owned investigative / remediation reasoning ----------------------
# The SAME single canonical interpretation now also proposes the substance a
# separate deterministic "second brain" used to manufacture: possible
# explanations, what is unknown and why it matters, and what remediation the
# observed condition actually calls for. The LLM PROPOSES; the deterministic
# validator (`canonical_context_validator`) constrains; downstream nodes
# CONSUME rather than re-deriving. Every field defaults empty so that when
# the LLM-primary flag is OFF (or the model omits a section) the whole
# structure is inert and downstream behavior is byte-identical.

RootCauseStatus = Literal["ESTABLISHED", "NOT_ESTABLISHED", "STATED_UNVERIFIED", "CONTRADICTED"]
HypothesisEpistemic = Literal["POSSIBLE", "SUPPORTED", "REFUTED", "UNKNOWN"]

# Whether the EVIDENCE has established an actual obligation to remediate -- the
# distinction the remediation LLM currently collapses ("a finding exists, so a
# remediation programme must exist"). Determined from what is established, not
# from the mere existence of the finding.
RemediationObligation = Literal[
    # Evidence establishes a nonconformity / deficient condition that must be
    # corrected (the cause may or may not yet be known).
    "ESTABLISHED_CORRECTIVE_OBLIGATION",
    # An unresolved discrepancy between values/records/assessments: reconcile
    # and establish comparability BEFORE any corrective action is considered.
    "RECONCILIATION_REQUIRED",
    # The cause / scope must be established by investigation before a
    # corrective action can be identified.
    "INVESTIGATION_REQUIRED",
    # A bounded correction of the known condition is justified now; nothing
    # systemic is indicated by the current evidence.
    "IMMEDIATE_CORRECTION_ONLY",
    # The condition is understood and no systemic corrective/preventive action
    # is currently justified (e.g. the difference is legitimate once explained).
    "NO_SYSTEMIC_REMEDIATION_JUSTIFIED",
    "NOT_DETERMINED",
]


class SemHypothesis(BaseModel):
    """One POSSIBLE explanation the finding/evidence supports considering.
    `epistemic` may only be SUPPORTED when a VERIFIED causal claim backs it
    -- the validator forces it back to POSSIBLE otherwise. A hypothesis the
    finding itself enumerates carries `from_finding_text = True` and must
    never be ranked above its siblings."""

    hypothesis_id: str
    statement: str
    epistemic: HypothesisEpistemic = "POSSIBLE"
    from_finding_text: bool = False
    rationale: str | None = None
    discriminating_evidence: str | None = None  # what evidence would separate this from the others
    source_evidence_ids: list[str] = Field(default_factory=list)


class SemInvestigationStep(BaseModel):
    """A single investigation priority, framed as the four things that make a
    question worth asking (spec §7): what is unknown, why it matters, what
    evidence resolves it, what decision that evidence enables. Manufactured
    generically for every finding = exactly what this replaces."""

    unknown: str
    why_it_matters: str | None = None
    evidence_that_would_resolve: str | None = None
    decision_enabled: str | None = None
    related_hypothesis_ids: list[str] = Field(default_factory=list)
    priority: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"


class SemRemediationAction(BaseModel):
    """A remediation activity the LLM reasons follows from the observed
    condition. `disposition`:
      INVESTIGATION         -- reconcile / verify / establish the cause or
                               scope. NOT remediation of a defect; it is the
                               work that determines whether one exists.
      IMMEDIATE_CORRECTION  -- objectively justified by the KNOWN condition
                               NOW, independent of root cause.
      CONTAINMENT           -- limit ongoing exposure now.
      CORRECTIVE_ACTION     -- addresses a CONFIRMED cause. Only valid when
                               root_cause_status == ESTABLISHED; the validator
                               downgrades it to CONDITIONAL_SYSTEMIC otherwise.
      CONDITIONAL_SYSTEMIC  -- corrective/preventive action that depends on
                               root-cause confirmation; stays conditional
                               until the cause is established.
      EFFECTIVENESS_CHECK   -- verify the correction worked.
    The validator forces any concrete systemic prescription to
    CONDITIONAL_SYSTEMIC while root cause is NOT_ESTABLISHED. No amounts,
    rates, quantities, or currencies here -- `pricing_evidence_needed`
    names what would make the activity priceable. `addresses_condition` says
    which established condition / information gap / confirmed cause this
    activity is connected to (spec §7) -- an activity with no defensible
    connection must not be proposed."""

    action_id: str
    activity: str
    disposition: Literal[
        "INVESTIGATION", "IMMEDIATE_CORRECTION", "CONTAINMENT",
        "CORRECTIVE_ACTION", "CONDITIONAL_SYSTEMIC", "EFFECTIVENESS_CHECK",
    ] = "INVESTIGATION"
    addresses_condition: str | None = None
    justification: str | None = None
    depends_on_root_cause: bool = False
    pricing_evidence_needed: str | None = None
    scope_evidence_needed: str | None = None


class SemPricingItem(BaseModel):
    """What pricing evidence a remediation activity actually needs -- the
    LLM determines the KIND from the activity's meaning (internal labour +
    effort, supplier/parts/service quotation, implementation estimate,
    training scope, validation effort, contractor quotation, ...). It never
    supplies a number. `observed_value_*` capture a monetary value that
    appears IN THE FINDING and must NOT be treated as a remediation cost."""

    action_id: str | None = None
    pricing_basis: str | None = None       # e.g. "supplier quotation", "internal labour rate + evidenced effort"
    rationale: str | None = None
    evidence_available: bool = False       # is that pricing evidence actually present?
    observed_value_in_finding: str | None = None   # a monetary value the finding states (describes the finding, not the fix)
    observed_value_is_remediation_cost: bool = False


class CanonicalFindingContext(BaseModel):
    """The canonical semantic interpretation of one finding + its evidence
    ledger. `financial` reuses the existing, unchanged financial semantic
    schema/pipeline from the previous pass -- this context ADDS the
    cross-cutting (non-financial-specific) understanding that financial
    analysis alone never needed: what the actual deviation is, what
    entities/states exist, what is and is not causal, and whether a
    previous CAPA is actually referenced."""

    primary_deviation: str | None = None
    primary_deviation_claim_id: str | None = None
    primary_deviation_confidence: Literal["HIGH", "MEDIUM", "LOW", "NOT_ESTABLISHED"] = "NOT_ESTABLISHED"

    # --- LLM-PRIMARY semantic fields (merged into canonical_finding_state) ---
    # The substantive affected object/process/control/activity -- NEVER a
    # causal mechanism, hypothesis, evidence source, requirement text, or
    # reported belief. Null when no substantive subject exists.
    finding_subject: str | None = None
    subject_kind: SemanticKind | None = None
    # The evidence-source clause ("maintenance records show that ...") kept as
    # provenance, never the subject.
    evidence_source: str | None = None
    # A reported action/state the finding attributes to a source, with its
    # epistemic status ("temporary repairs were performed" -> REPORTED).
    reported_observation: str | None = None
    observed_condition: str | None = None
    epistemic_status: FindingEpistemicStatus | None = None

    comparison: SemComparison | None = None
    recurrence: SemRecurrence | None = None

    # Causal mechanisms the FINDING TEXT explicitly enumerates. Never erased,
    # never ranked without evidence -- each becomes a POSSIBLE hypothesis.
    stated_causal_alternatives: list[str] = Field(default_factory=list)
    causal_alternatives_unresolved: bool = False

    missing_record_status: MissingRecordStatus | None = None
    # True when the finding leaves it open whether the underlying activity
    # occurred (missing record != non-performance).
    activity_performance_ambiguity: bool = False

    affected_period: str | None = None
    scope: str | None = None

    entities: list[CanonicalEntity] = Field(default_factory=list)
    causal_claims: list[CausalClaim] = Field(default_factory=list)

    # Explicit boolean, never inferred from recurrence/historical/repeated
    # wording alone -- see canonical_context_validator.py, which
    # independently cross-checks this against the existing deterministic
    # recurrence_guard.detect_recurrence().has_previous_capa_reference
    # signal and forces it False unless BOTH agree.
    explicit_previous_capa_reference: bool = False
    previous_capa_evidence_ids: list[str] = Field(default_factory=list)

    evidence_boundaries: list[EvidenceBoundary] = Field(default_factory=list)
    unresolved_ambiguities: list[str] = Field(default_factory=list)

    # --- LLM-owned investigative / remediation reasoning (spec §6-§10) ---
    # Root-cause epistemic verdict. NOT_ESTABLISHED unless a VERIFIED causal
    # claim backs a single mechanism -- the validator enforces this and
    # clears `leading_hypothesis_id` whenever it is not ESTABLISHED or the
    # finding's own alternatives are unresolved.
    root_cause_status: RootCauseStatus = "NOT_ESTABLISHED"
    leading_hypothesis_id: str | None = None
    candidate_hypotheses: list[SemHypothesis] = Field(default_factory=list)

    # Whether the evidence has established an OBLIGATION to remediate, and why.
    # `NOT_DETERMINED` (default) leaves the deterministic/legacy behavior
    # unchanged. The remediation cost stage consumes this: a
    # RECONCILIATION_REQUIRED / INVESTIGATION_REQUIRED /
    # NO_SYSTEMIC_REMEDIATION_JUSTIFIED obligation means NO systemic CAPA is to
    # be manufactured or priced.
    remediation_obligation: RemediationObligation = "NOT_DETERMINED"
    remediation_obligation_rationale: str | None = None

    # What is genuinely unknown, and the investigation priorities that follow
    # from those gaps -- the LLM decides how many, from the finding, not a
    # fixed count.
    information_gaps: list[str] = Field(default_factory=list)
    investigation_plan: list[SemInvestigationStep] = Field(default_factory=list)

    # THREE DISTINCT LAYERS (spec §14): investigation (what must be
    # established) -> remediation (what must actually be corrected) ->
    # pricing (what would price that correction). `investigation_activities`
    # holds reconcile / compare / verify / determine work whose PURPOSE is to
    # find out what happened -- it is NOT remediation. `remediation_activities`
    # holds ONLY genuine correction / containment / corrective / conditional-
    # systemic / effectiveness-check work and MAY legitimately be empty
    # (spec §9). The validator partitions on the LLM's declared
    # `disposition` -- it never re-classifies by verb. `immediate_actions` /
    # `conditional_actions` are convenience projections of
    # `remediation_activities[*].disposition`.
    investigation_activities: list[SemRemediationAction] = Field(default_factory=list)
    remediation_activities: list[SemRemediationAction] = Field(default_factory=list)
    immediate_actions: list[str] = Field(default_factory=list)
    conditional_actions: list[str] = Field(default_factory=list)
    # Downstream of remediation reasoning (spec §11): every item must map to a
    # genuine `remediation_activities` entry -- the validator drops any that
    # point at investigation work or nothing.
    pricing_information: list[SemPricingItem] = Field(default_factory=list)

    # The unchanged financial semantic layer from the previous pass.
    financial: SemanticFindingInterpretation = Field(default_factory=SemanticFindingInterpretation)


class SemanticDisagreement(BaseModel):
    """One point of disagreement between the existing deterministic
    pipeline's own interpretation and the canonical LLM interpretation --
    recorded for shadow-mode comparison, per the "deterministic result
    stays authoritative until sufficient live validation" requirement.
    Never used to alter any authoritative output by itself."""

    field: str
    deterministic_value: str | None
    canonical_value: str | None
    disagreement_type: Literal[
        "AFFECTED_OBJECT_MISMATCH", "DEVIATION_MISMATCH",
        "PREVIOUS_CAPA_MISMATCH", "POPULATION_MISMATCH", "OTHER",
    ]
    evidence_ids: list[str] = Field(default_factory=list)
    downstream_consequence: str = ""
