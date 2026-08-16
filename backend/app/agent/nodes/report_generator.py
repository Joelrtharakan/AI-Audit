"""Node 9: generate_report

Assembles the full InvestigationReport from the agent state.
This is a deterministic assembly step — no LLM call needed.
All the reasoning has already happened in upstream nodes.
"""

from __future__ import annotations

import logging

from app.agent.grounding_guard import build_source_text, filter_list_field, ungrounded_entities
from app.agent.state import AgentState
from app.models.agent import (
    AgentFinalState,
    AgentTraceStep,
    CapaStatus,
    EvidenceStatus,
    ImpactStatus,
    InvestigationPlan,
    InvestigationReport,
    RootCauseAnalysis,
)
from app.models.analysis import ObservationQualityStatus

logger = logging.getLogger(__name__)


def _final_grounding_sweep(state: AgentState, root_cause, five_why, capa, impact, trace: list) -> None:
    """Defense-in-depth: every field checked here should already be clean from
    the per-node guards in rca.py/impact.py/capa.py. This sweep exists so that
    if any code path is ever added that bypasses those guards, a residual
    violation is caught and logged here rather than silently reaching the
    auditor. Mutates root_cause/five_why/capa/impact in place."""
    evidence_ledger = state.get("evidence_ledger", [])
    # Base source text deliberately excludes root_cause.narrative itself --
    # checking a field against text that includes that same field would make
    # every check trivially self-grounded and catch nothing.
    base_source_text = build_source_text(state["request"].finding_text, evidence_ledger)

    if root_cause.narrative and ungrounded_entities(root_cause.narrative, base_source_text):
        trace.append(AgentTraceStep.warn(
            "FINAL SWEEP: root cause narrative referenced an ungrounded entity — replaced"
        ))
        logger.error("Final grounding sweep caught a violation that should have been caught upstream (root_cause.narrative)")
        root_cause.narrative = "Root cause analysis could not be validated for this finding. Manual investigation is required."
        root_cause.status = "NOT_ESTABLISHED"  # type: ignore[assignment]
        root_cause.statement = None

    root_cause.candidate_hypotheses = [
        h for h in root_cause.candidate_hypotheses if not ungrounded_entities(h.statement, base_source_text)
    ]

    for step in five_why.steps:
        if step.answer and ungrounded_entities(step.answer, base_source_text):
            trace.append(AgentTraceStep.warn("FINAL SWEEP: 5-Why answer referenced an ungrounded entity — replaced"))
            step.answer = "Could not be validated against this finding's evidence."
            step.status = "UNKNOWN"  # type: ignore[assignment]

    # CAPA/impact are allowed to reference the now-cleaned root cause narrative
    # as legitimate context (it has already passed its own check above).
    extended_source_text = build_source_text(
        state["request"].finding_text, evidence_ledger,
        [root_cause.narrative] if root_cause.narrative else [],
    )
    capa.potential_areas, _ = filter_list_field(capa.potential_areas, extended_source_text)
    capa.recommended_investigation, _ = filter_list_field(capa.recommended_investigation, extended_source_text)

    impact.areas, _ = filter_list_field(impact.areas, extended_source_text)
    if impact.narrative and ungrounded_entities(impact.narrative, extended_source_text):
        trace.append(AgentTraceStep.warn("FINAL SWEEP: impact narrative referenced an ungrounded entity — removed"))
        impact.narrative = None


def _compute_observation_quality(state: AgentState) -> str:
    quality = state.get("observation_quality")
    if not quality:
        return "INSUFFICIENT"
    return quality.status.value


def _compute_confidence_scores(state: AgentState) -> tuple[str, str, str]:
    """Returns (observation_confidence, root_cause_confidence, overall_confidence)."""
    root_cause = state.get("root_cause")
    quality = state.get("observation_quality")
    evidence_ledger = state.get("evidence_ledger", [])

    # Observation confidence
    obs_conf = "HIGH" if quality and quality.status == ObservationQualityStatus.SUFFICIENT else "LOW"

    # Root cause confidence (Requirements 5 & 20)
    # If root_cause status is NOT_ESTABLISHED, confidence MUST be LOW (never MEDIUM/HIGH).
    rc_conf = "LOW"
    if root_cause:
        status_val = getattr(root_cause.status, "value", root_cause.status)
        if status_val in ("NOT_ESTABLISHED", "STATED_UNVERIFIED", "CONTRADICTED"):
            rc_conf = "LOW"
        elif hasattr(root_cause, "confidence") and root_cause.confidence:
            rc_conf = root_cause.confidence
        elif status_val in ("VERIFIED", "SUPPORTED"):
            rc_conf = "HIGH" if any(e.status == EvidenceStatus.VERIFIED for e in evidence_ledger) else "MEDIUM"
        else:
            rc_conf = "LOW"

    # Overall analytical confidence
    if obs_conf == "HIGH" and rc_conf == "HIGH":
        overall_conf = "HIGH"
    elif obs_conf == "HIGH" or rc_conf == "MEDIUM":
        overall_conf = "MEDIUM"
    else:
        overall_conf = "LOW"

    return obs_conf, rc_conf, overall_conf


def _compute_investigation_required(state: AgentState) -> str:
    root_cause = state.get("root_cause")
    capa = state.get("capa_analysis")

    if not root_cause:
        return "YES"
    if root_cause.status in ("NOT_ESTABLISHED", "STATED_UNVERIFIED", "CONTRADICTED"):
        return "YES"
    if capa and capa.status in (CapaStatus.INVESTIGATION_REQUIRED, CapaStatus.INSUFFICIENT_EVIDENCE):
        return "LIMITED"
    return "NO"


def _compute_final_state(state: AgentState) -> AgentFinalState:
    iteration_count = state.get("iteration_count", 0)
    max_iter = 10  # from settings but accessed statically here
    if iteration_count >= max_iter:
        return AgentFinalState.MAX_ITERATIONS_REACHED

    root_cause = state.get("root_cause")
    capa = state.get("capa_analysis")
    errors = state.get("errors", [])

    if any("tool failure" in e.lower() or "permission" in e.lower() for e in errors):
        return AgentFinalState.TOOL_FAILURE

    if not root_cause or root_cause.status == "NOT_ESTABLISHED":
        return AgentFinalState.INVESTIGATION_REQUIRED

    if capa and capa.status == CapaStatus.INSUFFICIENT_EVIDENCE:
        return AgentFinalState.INSUFFICIENT_EVIDENCE

    return AgentFinalState.READY_FOR_HUMAN_REVIEW


async def generate_report_node(state: AgentState) -> AgentState:
    """Assemble the InvestigationReport from agent state — no LLM call."""
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))

    # Assemble all components
    observation_quality = _compute_observation_quality(state)
    obs_conf, rc_conf, overall_conf = _compute_confidence_scores(state)
    investigation_required = _compute_investigation_required(state)
    final_state = _compute_final_state(state)

    # Fallback objects are deliberately generic/entity-free: they only fire when an
    # upstream node failed to produce a result, and must never inject another
    # case's facts (or any invented entity) into this finding's output.
    root_cause = state.get("root_cause") or RootCauseAnalysis(
        status="NOT_ESTABLISHED",
        category=None,
        narrative="Root cause analysis could not be completed for this finding. Manual investigation is required.",
        evidence_status=EvidenceStatus.UNKNOWN,
        verification_needed="Full auditor investigation required — AI root cause analysis was unavailable for this finding.",
    )

    investigation_plan = state.get("investigation_plan") or InvestigationPlan()

    from app.models.agent import FiveWhyAnalysis
    five_why = state.get("five_why") or FiveWhyAnalysis(
        steps=[], is_complete=False, status_note="INCOMPLETE — ROOT CAUSE NOT ESTABLISHED"
    )

    from app.models.agent import CapaAnalysis
    capa = state.get("capa_analysis") or CapaAnalysis(
        status=CapaStatus.INVESTIGATION_REQUIRED,
        potential_areas=[],
        recommended_investigation=["Full manual investigation required — AI CAPA analysis was unavailable for this finding."],
    )

    from app.models.agent import ImpactAssessment
    impact = state.get("impact_assessment") or ImpactAssessment(
        status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT,
        areas=["Determine scope and period of potential impact — auditor assessment required."],
        narrative=None,
    )

    _final_grounding_sweep(state, root_cause, five_why, capa, impact, trace)

    canonical = state.get("canonical_finding_state")
    evidence_claims = canonical.evidence_claims if canonical else []
    evidence_conflicts = canonical.evidence_conflicts if canonical else []

    report = InvestigationReport(
        observation_quality=observation_quality,  # type: ignore[arg-type]
        observation_confidence=obs_conf,  # type: ignore[arg-type]
        root_cause_confidence=rc_conf,  # type: ignore[arg-type]
        overall_confidence=overall_conf,  # type: ignore[arg-type]
        confidence=overall_conf,  # type: ignore[arg-type]
        investigation_required=investigation_required,  # type: ignore[arg-type]
        root_cause=root_cause,
        contributing_factors=state.get("contributing_factors", []),
        investigation=investigation_plan,
        five_why=five_why,
        capa=capa,
        impact_assessment=impact,
        evidence_gaps=state.get("evidence_gaps", []),
        evidence=state.get("evidence_ledger", []),
        evidence_claims=evidence_claims,
        evidence_conflicts=evidence_conflicts,
        human_review_required=True,  # always
        analysis_mode=state.get("analysis_mode", "LLM"),  # type: ignore[arg-type]
        analysis_engine=state.get("analysis_engine", "LLM"),  # type: ignore[arg-type]
        provider_used=state.get("provider_used"),
        fallback_used=bool(state.get("fallback_used", False)),
        provider_attempts=state.get("provider_attempts", []),
        critic_status=state.get("critic_status"),
    )
    if report.analysis_mode == "DETERMINISTIC":
        trace.append(AgentTraceStep.ok(
            "Report generated using evidence-grounded deterministic synthesis."
        ))
    elif report.analysis_mode == "DEGRADED":
        trace.append(AgentTraceStep.warn(
            "Report generated in DEGRADED MODE — full auditor review required."
        ))

    trace.append(AgentTraceStep.ok("Investigation report generated"))
    trace.append(AgentTraceStep.warn("Auditor review required"))

    return {
        **state,
        "report": report,
        "final_state": final_state,
        "trace": trace,
        "errors": errors,
    }
