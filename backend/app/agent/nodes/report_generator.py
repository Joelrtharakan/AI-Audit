"""Node 9: generate_report

Assembles the full InvestigationReport from the agent state.
This is a deterministic assembly step — no LLM call needed.
All the reasoning has already happened in upstream nodes.
"""

from __future__ import annotations

import logging
import time

from app.agent.grounding_guard import build_source_text, filter_list_field, ungrounded_entities
from app.agent.state import AgentState
from app.config import get_settings
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
    canonical = state.get("canonical_finding_state")
    if canonical and not getattr(canonical, "is_actionable", True):
        return "NO"

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
    canonical = state.get("canonical_finding_state")
    if canonical and not getattr(canonical, "is_actionable", True):
        return AgentFinalState.READY_FOR_HUMAN_REVIEW

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
    evidence_claims = getattr(canonical, "evidence_claims", []) if canonical else []
    evidence_conflicts = getattr(canonical, "evidence_conflicts", []) if canonical else []
    referenced_documents = getattr(canonical, "referenced_documents", []) if canonical else []
    propositions = getattr(canonical, "propositions", []) if canonical else []
    cost_impact = state.get("cost_impact") or (getattr(canonical, "cost_impact", None) if canonical else None)
    if cost_impact and not getattr(cost_impact, "cost_factor_detected", False):
        cost_impact = None

    from app.models.agent import InvestigationMode
    investigation_mode = getattr(canonical, "investigation_mode", InvestigationMode.NORMAL) if canonical else InvestigationMode.NORMAL

    semantic_graph = getattr(canonical, "semantic_graph", None)
    if not semantic_graph:
        from app.models.agent import SemanticGraph
        semantic_graph = SemanticGraph()

    # Build comprehensive SemanticTraceabilityMatrix linking all primary report fields
    # to source propositions, claims, and semantic graph nodes
    from app.models.agent import (
        SemanticTraceabilityEntry,
        SemanticTraceabilityMatrix,
    )
    trace_entries: list[SemanticTraceabilityEntry] = []
    all_prop_ids = [p.id for p in propositions] if propositions else []
    all_claim_ids = [c.claim_id for c in evidence_claims if getattr(c, "claim_id", None)]

    if root_cause and getattr(root_cause, "statement", None):
        lead_hyps = getattr(root_cause, "candidate_hypotheses", []) or []
        lead_claim_ids = lead_hyps[0].supporting_claim_ids if lead_hyps else (all_claim_ids[:1] if all_claim_ids else [])
        trace_entries.append(SemanticTraceabilityEntry(
            field_name="root_cause.statement",
            concept=root_cause.statement,
            source_proposition_ids=lead_claim_ids,
            epistemic_status=getattr(root_cause, "evidence_status", "UNKNOWN"),
            provenance="INFERRED" if getattr(root_cause.status, "value", root_cause.status) != "VERIFIED" else "OBJECTIVE_RECORD",
            derivation_type="CAUSAL_PROGRESSION",
        ))

    if impact and getattr(impact, "affected_object", None):
        matching_props = [p.id for p in propositions if p.subject and p.subject.lower() in (impact.affected_object or "").lower()]
        trace_entries.append(SemanticTraceabilityEntry(
            field_name="impact_assessment.affected_object",
            concept=impact.affected_object,
            source_proposition_ids=matching_props or (all_prop_ids[:1] if all_prop_ids else all_claim_ids[:1]),
            epistemic_status="VERIFIED",
            provenance="AUDIT_OBSERVATION",
        ))

    if impact and getattr(impact, "process_at_risk", None):
        trace_entries.append(SemanticTraceabilityEntry(
            field_name="impact_assessment.process_at_risk",
            concept=impact.process_at_risk,
            source_proposition_ids=all_prop_ids[:1] if all_prop_ids else all_claim_ids[:1],
            epistemic_status="VERIFIED",
            provenance="AUDIT_OBSERVATION",
        ))

    if five_why and getattr(five_why, "steps", None):
        for idx, step in enumerate(five_why.steps):
            trace_entries.append(SemanticTraceabilityEntry(
                field_name=f"five_why.steps[{idx}]",
                concept=f"Q: {step.question} | A: {step.answer}",
                source_proposition_ids=all_prop_ids[:1] if all_prop_ids else all_claim_ids[:1],
                epistemic_status=str(step.status),
                provenance="CAUSAL_TRAVERSAL",
            ))

    if capa and getattr(capa, "immediate_actions", None):
        for idx, act in enumerate(capa.immediate_actions):
            act_text = getattr(act, "action", "") or getattr(act, "description", "")
            trace_entries.append(SemanticTraceabilityEntry(
                field_name=f"capa.immediate_actions[{idx}]",
                concept=act_text,
                source_proposition_ids=all_prop_ids[:1] if all_prop_ids else all_claim_ids[:1],
                epistemic_status="VERIFIED",
                provenance="CAPA_GROUNDING",
            ))

    semantic_traceability = SemanticTraceabilityMatrix(
        entries=trace_entries,
        is_valid=True,
        untraced_concepts=[],
    )

    # ------------------------------------------------------------------
    # Cost analysis.
    #
    # REMEDIATION COST ESTIMATE is the single AUTHORITATIVE, auditor-facing
    # cost analysis ("what will it cost to correct/prevent the finding?").
    # It owns its own canonical result, validator, and deterministic
    # calculator, and fails closed to an honest professional NOT_ASSESSABLE
    # result on any error -- never a fabricated number, never an internal
    # diagnostic string.
    #
    # The legacy Cost & Financial Exposure analysis ("what has the finding
    # already cost?") is NO LONGER rendered to the auditor and no longer runs
    # its LLM semantic-interpretation stage. Only the fast, deterministic
    # regex engine still runs, purely to derive the internal `cost_impact` /
    # `financial_amount` fields that the Risk & Impact Assessment narrative
    # consumes (app.agent.nodes.final_evidence_verification). It is fully
    # isolated from the remediation pipeline and from the report renderer.
    # ------------------------------------------------------------------
    from app.financial.engine import analyze_financial_exposure
    finding_text = state["request"].finding_text if state.get("request") else ""
    settings = get_settings()
    evidence_ledger = state.get("evidence_ledger", [])

    # -- Internal-only financial context (never rendered as its own section).
    financial_analysis = state.get("financial_analysis")
    if financial_analysis is None:
        financial_analysis = analyze_financial_exposure(
            finding_text=finding_text,
            evidence_ledger=evidence_ledger,
            evidence_claims=evidence_claims,
        )
        financial_analysis.reasoning_source = "DETERMINISTIC_REGEX"

    # -- Remediation Cost Estimate: the authoritative auditor-facing cost section.
    remediation_cost = state.get("remediation_cost")
    _rem_reused = remediation_cost is not None
    _rem_ms = None
    if remediation_cost is None and settings.remediation_cost_estimation_enabled:
        _t = time.monotonic()
        try:
            from app.remediation.engine import estimate_remediation_cost
            remediation_cost = await estimate_remediation_cost(
                finding_text=finding_text,
                evidence_ledger=evidence_ledger,
                root_cause=root_cause,
                capa=capa,
                impact=impact,
            )
        except Exception as exc:  # noqa: BLE001 - last-resort crash guard only
            logger.warning("Remediation cost estimation crashed unexpectedly (%s); reporting honestly.", exc)
            from app.remediation.engine import honest_not_assessable
            remediation_cost = honest_not_assessable("LLM_UNAVAILABLE")
        _rem_ms = int((time.monotonic() - _t) * 1000)

    logger.info(
        "REMEDIATION COST TIMING ms=%s reused=%s status=%s semantic_status=%s",
        _rem_ms, _rem_reused,
        getattr(getattr(remediation_cost, "status", None), "value", None),
        getattr(remediation_cost, "remediation_semantic_status", None),
    )

    # Canonical semantic finding context -- promoted to authoritative
    # downstream input (investigation planner / Five-Why, see
    # plan_investigation_node/core_synthesis_node/final_evidence_
    # verification_node) when `canonical_semantic_shadow_enabled` is on.
    # Computed ONCE, early, in plan_investigation_node and reused here via
    # state -- never recomputed (avoids a second LLM call for the same
    # finding). Disagreement recording is kept for diagnostics: it now
    # compares what was ACTUALLY used (the canonical value, when promoted)
    # against what the pure legacy raw-text path would independently have
    # produced, never the reverse.
    semantic_pipeline_disagreements: list[dict] = []
    settings = get_settings()
    if settings.canonical_semantic_shadow_enabled:
        try:
            from app.services.shadow_semantic_comparison import compare_deterministic_vs_canonical

            canonical_context = state.get("canonical_semantic_context")
            if canonical_context is not None:
                det_canonical_state = state.get("canonical_finding_state")
                deterministic_subject = (
                    getattr(det_canonical_state, "affected_object", None)
                    or getattr(det_canonical_state, "finding_subject", None)
                    if det_canonical_state is not None else None
                )
                disagreements = compare_deterministic_vs_canonical(
                    finding_text, deterministic_subject, canonical_context
                )
                semantic_pipeline_disagreements = [d.model_dump() for d in disagreements]
        except Exception as exc:  # noqa: BLE001 - diagnostics must never affect the authoritative report
            logger.warning("Canonical semantic comparison failed unexpectedly (%s); ignored.", exc)
            semantic_pipeline_disagreements = []

    report = InvestigationReport(
        observation_quality=observation_quality,  # type: ignore[arg-type]
        observation_confidence=obs_conf,  # type: ignore[arg-type]
        root_cause_confidence=rc_conf,  # type: ignore[arg-type]
        overall_confidence=overall_conf,  # type: ignore[arg-type]
        confidence=overall_conf,  # type: ignore[arg-type]
        investigation_required=investigation_required,  # type: ignore[arg-type]
        investigation_mode=investigation_mode,
        root_cause=root_cause,
        contributing_factors=state.get("contributing_factors", []),
        investigation=investigation_plan,
        five_why=five_why,
        capa=capa,
        impact_assessment=impact,
        cost_impact=cost_impact,
        financial_analysis=financial_analysis,
        remediation_cost=remediation_cost,
        evidence_gaps=state.get("evidence_gaps", []),
        evidence=state.get("evidence_ledger", []),
        propositions=propositions,
        evidence_claims=evidence_claims,
        evidence_conflicts=evidence_conflicts,
        referenced_documents=referenced_documents,
        semantic_graph=semantic_graph,
        semantic_traceability=semantic_traceability,
        human_review_required=True,  # always
        analysis_mode=state.get("analysis_mode", "LLM"),  # type: ignore[arg-type]
        analysis_engine=state.get("analysis_engine", "LLM"),  # type: ignore[arg-type]
        provider_used=state.get("provider_used"),
        fallback_used=bool(state.get("fallback_used", False)),
        provider_attempts=state.get("provider_attempts", []),
        critic_status=state.get("critic_status"),
        semantic_pipeline_disagreements=semantic_pipeline_disagreements,
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
        "remediation_cost": remediation_cost,
        "final_state": final_state,
        "trace": trace,
        "errors": errors,
    }
