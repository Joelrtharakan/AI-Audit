"""Models for the LLM-only /api/v1/analyze-finding pipeline.

These models make "we don't know yet" a first-class, structured outcome:
root_cause.status can be NOT_ESTABLISHED, capa.status can be
INVESTIGATION_REQUIRED, and so on. Nothing here is allowed to fabricate
organizational facts -- see app/prompts/finding_analysis_system_prompt.txt and
app/services/grounding_validator.py, which mechanically checks that claim.
"""

from enum import Enum

from pydantic import BaseModel


class ObservationQualityStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RootCauseStatus(str, Enum):
    # Corroborating evidence was presented in the observation itself (e.g. a specific
    # record/log/document referenced as confirming the cause) -- not just an assertion.
    ESTABLISHED = "ESTABLISHED"
    # The observation itself asserts a specific, plausible cause (e.g. a direct quote or
    # admission from staff naming a specific SOP/training gap) but nothing corroborates
    # it -- no record, log, or document was cited. Surfaced as a strong hypothesis the
    # auditor should verify, not equated with "no cause evident at all".
    SELF_REPORTED = "SELF_REPORTED"
    # The observation gives no causal signal at all.
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


class CapaStatus(str, Enum):
    AI_SUGGESTED = "AI_SUGGESTED"
    INVESTIGATION_REQUIRED = "INVESTIGATION_REQUIRED"


class ImpactStatus(str, Enum):
    ASSESSED = "ASSESSED"
    REQUIRES_ASSESSMENT = "REQUIRES_ASSESSMENT"


class RiskOfRecurrence(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class ObservationQualityResult(BaseModel):
    status: ObservationQualityStatus
    missing_information: list[str] = []


class AttributedStatement(BaseModel):
    """A claim the extraction step found attributed to a specific speaker in the
    observation text (e.g. "staff confirmed...") -- as opposed to a stated_fact, which
    is something the observation asserts directly with no named speaker."""

    speaker: str
    claim: str


class ExtractionResult(BaseModel):
    """Step 1 output: what the observation actually says, nothing more. No causal
    judgment, no inference -- see app/prompts/extraction_prompt.txt. Also the ground
    truth the grounding_validator checks Step 3's output against."""

    stated_facts: list[str] = []
    attributed_statements: list[AttributedStatement] = []
    referenced_records: list[str] = []
    # Specific named systems/software/equipment/documents mentioned in the text (e.g.
    # "LIMS", "SOP-LAB-001") -- kept separate from referenced_records because a system
    # can be *named* without being cited as corroborating evidence.
    named_systems_or_documents: list[str] = []
    timeframe: str | None = None
    asset_or_location: str | None = None
    # True only if the observation explicitly describes a release, shipment, or
    # customer/patient-facing event. Gates whether generation is allowed to reach for
    # recall/quarantine/notify-customer language -- see grounding_validator.py 3.2.1.
    external_impact_stated: bool = False


class CausationClassification(BaseModel):
    """Step 2 output: reasons ONLY over the Step 1 extraction (not raw finding_text) to
    decide root_cause_status. See app/prompts/causation_classification_fewshot.txt for
    the worked examples this is calibrated against."""

    root_cause_status: RootCauseStatus
    category: str | None = None
    reasoning: str = ""


class GroundingViolation(BaseModel):
    """A single claim in the generation output that couldn't be traced back to the
    finding text or extraction -- see grounding_validator.py."""

    field: str
    claimed_entity: str
    note: str = ""


class GroundingReport(BaseModel):
    """Populated on every response, whether or not violations occurred -- this is what
    the golden-eval harness and log aggregation read to tell whether the guardrail is a
    rare backstop (good) or firing constantly (the generation prompt needs work)."""

    checked_entities: int = 0
    hard_violations: list[GroundingViolation] = []
    soft_violations: list[GroundingViolation] = []
    status_downgraded: bool = False
    consistency_checked: bool = False
    consistency_mismatch: bool = False


class AnalysisSummary(BaseModel):
    observation_quality: ObservationQualityStatus
    confidence: ConfidenceLevel
    requires_investigation: bool


class RootCause(BaseModel):
    status: RootCauseStatus
    # Populated for ESTABLISHED and SELF_REPORTED (one of the fixed 6M taxonomy
    # values); always null for NOT_ESTABLISHED.
    category: str | None = None
    category_other_text: str | None = None
    statement: str | None = None
    # True whenever status != ESTABLISHED -- this is a candidate the auditor must
    # confirm, not a finding of fact. Redundant with status but makes frontend/report
    # logic that only cares about "is this confirmed or not" simpler and harder to get
    # wrong than re-deriving it from the status string each time.
    is_hypothesis: bool = False
    # Descriptive text on what the auditor needs to verify -- populated only for
    # SELF_REPORTED (e.g. "Verify via training attendance records"); null otherwise.
    # NOT_ESTABLISHED has nothing specific to verify (investigate from scratch instead),
    # ESTABLISHED needs no verification (already corroborated).
    verification_needed: str | None = None
    # One-line reasoning from the Step 2 classifier explaining the status decision --
    # surfaced for auditor transparency and used by the golden-eval harness to explain
    # mismatches.
    classification_reasoning: str | None = None
    # ESTABLISHED only.
    risk_of_recurrence: RiskOfRecurrence | None = None
    # ESTABLISHED only. A role, never a named individual -- enforced in code, see
    # grounding_validator.py 3.2.2.
    recommended_capa_owner: str | None = None


class Investigation(BaseModel):
    areas: list[str] = []
    questions: list[str] = []
    evidence_required: list[str] = []


class CapaSuggestion(BaseModel):
    status: CapaStatus
    # Populated only when status == AI_SUGGESTED (i.e. root cause is ESTABLISHED).
    corrective_actions_immediate: list[str] = []
    preventive_actions_recurrence: list[str] = []
    # Populated only when status == INVESTIGATION_REQUIRED.
    recommended_investigation: list[str] = []
    potential_corrective_action_areas: list[str] = []


class ImpactAnalysis(BaseModel):
    status: ImpactStatus
    areas: list[str] = []
    narrative: str | None = None


class AiMetadata(BaseModel):
    model: str
    prompt_version: str
    generated_at: str
    suggestion_id: str


class AnalyzeFindingResponse(BaseModel):
    analysis: AnalysisSummary
    extraction: ExtractionResult
    root_cause: RootCause
    # Conditions that allowed the cause to occur -- never action items. Populated for
    # ESTABLISHED and SELF_REPORTED (labeled as hypotheses for the latter in the
    # frontend); empty list is a valid, expected output, especially for NOT_ESTABLISHED.
    contributing_factors: list[str] = []
    five_why: list[str] = []
    investigation: Investigation
    capa: CapaSuggestion
    impact_analysis: ImpactAnalysis
    ai_limitations: list[str] = []
    grounding_report: GroundingReport = GroundingReport()
    ai_metadata: AiMetadata
