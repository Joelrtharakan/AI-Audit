"""Derive the structured human-review contract from a finished report.

Human-reviewed production is the operating model: `required` is always True.
The value of this module is the machine-readable *reason set* -- an integrator
(e.g. the ASP.NET LQMS UI) branches on `report.review.reason_codes` and cannot
silently drop a mandatory-review condition.

Every reason is derived from an authoritative, already-settled field on the
report object. Nothing here re-runs semantics or inspects raw finding text.
"""

from __future__ import annotations

from app.models.agent import InvestigationReport, ReviewReason, ReviewState

# Sub-analysis semantic-status tokens that mean "the model call failed" rather
# than "assessed and genuinely not determinable".
_TRANSIENT_LLM = {"LLM_UNAVAILABLE", "LLM_TIMEOUT", "LLM_INVALID", "LLM_INCOMPLETE"}


def _add(reasons: list[ReviewReason], code: str, detail: str) -> None:
    if not any(r.code == code for r in reasons):
        reasons.append(ReviewReason(code=code, detail=detail))


def derive_review_state(report: InvestigationReport) -> ReviewState:
    reasons: list[ReviewReason] = []

    # --- AI availability / analysis mode -----------------------------------
    if report.analysis_mode == "DEGRADED":
        _add(reasons, "AI_ANALYSIS_DEGRADED",
             "The analysis ran in degraded mode after an AI failure; treat all conclusions as unverified.")
    elif report.analysis_mode == "DETERMINISTIC":
        _add(reasons, "DETERMINISTIC_ANALYSIS",
             "No AI synthesis was used; a deterministic evidence-grounded analysis produced this report.")

    if report.semantic_mode == "DETERMINISTIC_FALLBACK":
        _add(reasons, "SEMANTIC_LAYER_UNAVAILABLE",
             "The canonical semantic AI was unavailable; the deterministic floor produced the interpretation.")

    status = (report.canonical_semantic_status or "").upper()
    if any(tok in status for tok in ("TIMEOUT", "UNAVAILABLE", "INVALID", "ERROR", "FAILED")):
        _add(reasons, "CANONICAL_SEMANTIC_FAILURE",
             f"Canonical semantic interpretation status: {report.canonical_semantic_status}.")

    if report.fallback_used:
        _add(reasons, "PROVIDER_FALLBACK_USED",
             "A provider/model fallback was exercised during analysis.")

    # --- observation / evidence ------------------------------------------
    if report.observation_quality == "INSUFFICIENT":
        _add(reasons, "OBSERVATION_INSUFFICIENT",
             "The finding statement does not establish enough to analyze confidently.")
    elif report.observation_quality == "CONFLICTING":
        _add(reasons, "OBSERVATION_CONFLICTING",
             "The finding statement contains conflicting information.")

    if report.evidence_completeness.value not in ("COMPLETE",):
        _add(reasons, "EVIDENCE_INCOMPLETE",
             f"Evidence completeness is {report.evidence_completeness.value}.")

    # --- root cause -----------------------------------------------------
    rc_status = getattr(report.root_cause.status, "value", str(report.root_cause.status))
    if rc_status not in ("ESTABLISHED", "VERIFIED"):
        _add(reasons, "ROOT_CAUSE_NOT_ESTABLISHED",
             f"Root cause is {rc_status}; corrective/preventive actions remain conditional.")

    # --- remediation cost --------------------------------------------------
    rc = report.remediation_cost
    if rc is not None:
        sem = getattr(rc, "remediation_semantic_status", "OK")
        if sem in _TRANSIENT_LLM:
            _add(reasons, "REMEDIATION_MODEL_FAILURE",
                 f"Remediation-cost model status: {sem}.")
        pricing = getattr(rc, "pricing_status", None)
        if pricing == "PARTIAL_ESTIMATE":
            _add(reasons, "REMEDIATION_PARTIAL_ESTIMATE",
                 "Remediation cost is only partially priced.")
        elif pricing == "RANGE_ESTIMATE":
            _add(reasons, "REMEDIATION_RANGE_ESTIMATE",
                 "Remediation cost is a range, not an exact figure.")
        rc_stat = getattr(getattr(rc, "status", None), "value", None)
        if rc_stat == "NOT_ASSESSABLE":
            _add(reasons, "REMEDIATION_COST_NOT_ASSESSABLE",
                 "Remediation cost could not be assessed from the available information.")
        for comp in (getattr(rc, "cost_components", []) or []):
            if getattr(comp, "recurrence", "ONE_TIME") == "UNKNOWN":
                _add(reasons, "RECURRENCE_UNKNOWN",
                     "At least one remediation cost component has unresolved recurrence.")
                break

    # --- financial exposure ---------------------------------------------
    fin = report.financial_analysis
    if fin is not None and getattr(fin, "financial_semantic_status", "OK") in _TRANSIENT_LLM:
        _add(reasons, "FINANCIAL_MODEL_FAILURE",
             f"Financial-exposure model status: {fin.financial_semantic_status}.")

    # --- baseline: human-reviewed production is always the operating mode --
    _add(reasons, "HUMAN_REVIEWED_PRODUCTION",
         "This system operates under mandatory human review; AI output is advisory.")

    return ReviewState(required=True, status="PENDING_HUMAN_REVIEW", reasons=reasons)
