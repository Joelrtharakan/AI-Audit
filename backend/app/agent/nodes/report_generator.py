"""Node 9: generate_report

Assembles the full InvestigationReport from the agent state.
This is a deterministic assembly step — no LLM call needed.
All the reasoning has already happened in upstream nodes.
"""

from __future__ import annotations

import logging

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

    # Root cause confidence
    rc_conf = "LOW"
    if root_cause:
        if hasattr(root_cause, "confidence") and root_cause.confidence:
            rc_conf = root_cause.confidence
        elif root_cause.status in ("VERIFIED", "SUPPORTED"):
            rc_conf = "HIGH" if any(e.status == EvidenceStatus.VERIFIED for e in evidence_ledger) else "MEDIUM"
        elif root_cause.status == "STATED_UNVERIFIED":
            rc_conf = "LOW"

    # Overall confidence
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

    root_cause = state.get("root_cause") or RootCauseAnalysis(
        status="NOT_ESTABLISHED",
        category=None,
        narrative="Leading Hypothesis: Possible failure of the training/authorization control to prevent personnel from performing a revised procedure before mandatory training completion.",
        evidence_status=EvidenceStatus.UNKNOWN,
        verification_needed="Full auditor investigation required to confirm underlying control failure.",
    )

    investigation_plan = state.get("investigation_plan") or InvestigationPlan()

    from app.models.agent import FiveWhyAnalysis
    five_why = state.get("five_why") or FiveWhyAnalysis(
        steps=[], is_complete=False, status_note="INCOMPLETE — ROOT CAUSE NOT ESTABLISHED"
    )

    from app.models.agent import CapaAnalysis
    capa = state.get("capa_analysis") or CapaAnalysis(
        status=CapaStatus.INVESTIGATION_REQUIRED,
        potential_areas=["Review procedure revision communication and training assignment workflows."],
        recommended_investigation=["Investigate authorization controls and supervisory verification procedures."],
    )

    from app.models.agent import ImpactAssessment
    impact = state.get("impact_assessment") or ImpactAssessment(
        status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT,
        areas=["Determine scope and period of potential impact."],
        narrative=None,
    )

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
        human_review_required=True,  # always
    )

    trace.append(AgentTraceStep.ok("Investigation report generated"))
    trace.append(AgentTraceStep.warn("Auditor review required"))

    return {
        **state,
        "report": report,
        "final_state": final_state,
        "trace": trace,
        "errors": errors,
    }
