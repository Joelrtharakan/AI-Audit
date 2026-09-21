"""Derive the unified, immutable report-level AI provenance (spec Phase 6).

Consolidates the scattered per-field provenance into one object the ASP.NET
layer can persist verbatim. Built once from settled report/state; never
re-resolved at render time. A stage that did not run is represented explicitly
(`attempted=False`) rather than stamped with a model it never called.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from app.agent.autonomy import build_current_execution_identity, evaluate_autonomy
from app.config import get_settings
from app.models.agent import (
    AnalysisProvenance,
    EvidenceCompleteness,
    InvestigationReport,
    RootCauseStatus,
    StageProvenance,
)
from app.models.autonomy import AutonomyDecision


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _required_tasks_for(report: InvestigationReport) -> list[str]:
    """Structural presence check ONLY -- which sections of the ALREADY
    SETTLED report actually exist -- never an inspection of finding text
    (spec §35)."""
    tasks = ["RCA", "CAPA"]
    if getattr(report.five_why, "steps", None):
        tasks.append("FIVE_WHY")
    if getattr(report.investigation, "questions", None):
        tasks.append("INVESTIGATION")
    if report.remediation_cost is not None:
        tasks.append("REMEDIATION_COSTING")
    return tasks


def _derive_autonomy_decision(report: InvestigationReport, state: dict[str, Any]) -> AutonomyDecision | None:
    """Evaluates the autonomy framework (app.agent.autonomy) for THIS
    settled report/state, from structured signals only:
        - required tasks: which report sections are actually present
        - evidence_sufficient: `report.evidence_completeness`
        - causal_consistent: `report.root_cause.status` is not a
          CONTRADICTED/CONFLICTED terminal state
        - structural_safety_ok: no node recorded a structural error in
          `state["errors"]`
    Never inspects finding text or introduces a new semantic rule -- every
    signal reads a field the pipeline already computed. Returns None (never
    a fabricated decision) if evaluation itself fails for any reason.
    """
    try:
        identity = build_current_execution_identity()
        evidence_sufficient = report.evidence_completeness == EvidenceCompleteness.COMPLETE
        causal_consistent = report.root_cause.status not in (
            RootCauseStatus.CONTRADICTED, RootCauseStatus.CONFLICTED,
        )
        structural_safety_ok = not bool(state.get("errors"))
        return evaluate_autonomy(
            identity,
            _required_tasks_for(report),
            evidence_sufficient=evidence_sufficient,
            structural_safety_ok=structural_safety_ok,
            epistemic_consistent=True,
            causal_consistent=causal_consistent,
            provenance_ok=True,
        )
    except Exception:
        # Fail-closed for the autonomy ANNOTATION only -- never lets a
        # provenance/autonomy computation error block report generation.
        # A missing `autonomy` field is treated by every consumer exactly
        # like NOT_CERTIFIED (see app.agent.autonomy fail-closed default).
        return None


def derive_analysis_provenance(report: InvestigationReport, state: dict[str, Any]) -> AnalysisProvenance:
    s = get_settings()

    # --- canonical semantic stage ---------------------------------------
    canon_status = str(state.get("canonical_semantic_status", report.canonical_semantic_status or "NOT_ATTEMPTED"))
    canon_attempted = canon_status not in ("NOT_ATTEMPTED", "NOT_ATTEMPTED_FLAG_OFF", "", "DISABLED")
    canonical = StageProvenance(
        attempted=canon_attempted,
        model=(s.canonical_semantic_model or s.ollama_model) if canon_attempted else None,
        prompt_version=s.canonical_semantic_prompt_version if canon_attempted else None,
        status=canon_status,
    )

    # --- core synthesis stage -----------------------------------------
    synth_exec = state.get("synthesis_execution") or {}
    synth_source = synth_exec.get("source")  # PRIMARY_LLM / RECOVERY_LLM / CANONICAL_STATE / DETERMINISTIC
    synth_attempted = synth_source in ("PRIMARY_LLM", "RECOVERY_LLM")
    synthesis = StageProvenance(
        attempted=synth_attempted,
        model=(s.llm_model or s.ollama_model) if synth_attempted else None,
        prompt_version=s.analysis_prompt_version if synth_attempted else None,
        status=synth_source or ("DETERMINISTIC" if report.analysis_mode != "LLM" else "UNKNOWN"),
    )

    # --- remediation stage (echo the authoritative nested ai_provenance) --
    rc = report.remediation_cost
    rc_prov = getattr(rc, "ai_provenance", None) or {} if rc is not None else {}
    rem_status = str(getattr(rc, "remediation_semantic_status", "NOT_ATTEMPTED")) if rc is not None else "NOT_ATTEMPTED"
    rem_attempted = bool(rc_prov) and rem_status not in ("NOT_ATTEMPTED", "")
    remediation = StageProvenance(
        attempted=rem_attempted,
        model=rc_prov.get("remediation_model"),
        prompt_version=rc_prov.get("remediation_prompt_version"),
        schema_version=rc_prov.get("semantic_schema_version"),
        status=rem_status,
    )

    return AnalysisProvenance(
        provider=state.get("provider_used") or s.llm_provider,
        provider_attempts=list(state.get("provider_attempts", []) or []),
        execution_mode=report.analysis_mode,
        analysis_engine=report.analysis_engine,
        semantic_mode=report.semantic_mode,
        fallback_used=bool(report.fallback_used),
        review_required=bool(report.review.required),
        reason_codes=list(report.review.reason_codes),
        generated_at=_now(),
        canonical=canonical,
        synthesis=synthesis,
        remediation=remediation,
        autonomy=_derive_autonomy_decision(report, state),
    )
