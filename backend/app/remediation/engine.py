"""Remediation Cost Estimation orchestrator.

LLM interpretation -> provider-neutral normalization (in the interpreter) ->
deterministic structural validation -> deterministic arithmetic -> ONE canonical
`RemediationCostResult`.

Honest-failure contract (spec sections 6, 14, 15, 18): every non-OK LLM outcome
becomes an explicit, number-free result whose `not_assessable_reason` is a
professional user-facing sentence -- never an internal diagnostic. The real
machine status is kept on `remediation_semantic_status` for logs/invariants
only. This function ALWAYS returns a `RemediationCostResult` (never None), so
the report section always renders something professional.
"""

from __future__ import annotations

import logging
from typing import Any

from app.remediation.calculator import assemble_estimate
from app.remediation.models import (
    CostBasis,
    RemediationConfidence,
    RemediationCostResult,
    RemediationEstimateStatus,
)
from app.remediation.validator import validate_and_plan

logger = logging.getLogger(__name__)

_PROFESSIONAL_REASON = {
    "IMPLEMENTATION_SCOPE_UNKNOWN": (
        "Remediation cost cannot be reliably estimated because the implementation scope "
        "implied by this finding is not yet sufficiently defined."
    ),
    "QUANTITY_UNKNOWN": (
        "Remediation cost cannot be reliably estimated because the quantities of work, "
        "materials, or resources required are not established by the available evidence."
    ),
    "PRICING_BASIS_UNAVAILABLE": (
        "Remediation cost cannot be reliably estimated because the available evidence does "
        "not provide a defensible pricing basis for the required implementation work."
    ),
    "REMEDIATION_NOT_DEFINED": (
        "Remediation cost cannot be reliably estimated because the corrective and preventive "
        "action required to address this finding is not yet sufficiently defined."
    ),
    "CONFLICTING_EVIDENCE": (
        "Remediation cost cannot be reliably estimated because the evidence contains "
        "conflicting information about the required implementation work or its cost."
    ),
    "INSUFFICIENT_EVIDENCE": (
        "Remediation cost cannot be reliably estimated from the available evidence because "
        "the implementation scope and pricing basis are not sufficiently established."
    ),
    "": (
        "Remediation cost cannot be reliably estimated from the available evidence because "
        "the implementation scope and pricing basis are not sufficiently established."
    ),
}


def honest_not_assessable(semantic_status: str, machine_reason: str = "") -> RemediationCostResult:
    return RemediationCostResult(
        status=RemediationEstimateStatus.NOT_ASSESSABLE,
        confidence=RemediationConfidence.NOT_ASSESSABLE,
        estimate_classification=CostBasis.NOT_ESTABLISHED,
        not_assessable_reason=_PROFESSIONAL_REASON.get(machine_reason, _PROFESSIONAL_REASON[""]),
        reasoning_source="LLM_SEMANTIC" if semantic_status != "NO_EVIDENCE" else "NONE",
        remediation_semantic_status=semantic_status,
        review_required=True,
    )


def _hypothesis_ids(root_cause: Any) -> set[str]:
    ids: set[str] = set()
    for h in getattr(root_cause, "candidate_hypotheses", []) or []:
        hid = getattr(h, "hypothesis_id", None) or getattr(h, "id", None)
        if hid:
            ids.add(str(hid))
    return ids


def _capa_refs(capa: Any) -> set[str]:
    n = len(getattr(capa, "conditional_actions", []) or [])
    return {f"CAPA{i}" for i in range(n)}


async def estimate_remediation_cost(
    finding_text: str,
    evidence_ledger: list[Any] | None = None,
    root_cause: Any = None,
    capa: Any = None,
    impact: Any = None,
    financial_analysis: Any = None,  # accepted for back-compat; NOT sent to the LLM
    client=None,
) -> RemediationCostResult:
    # `financial_analysis` is intentionally not forwarded to the interpreter:
    # remediation cost does not depend on the financial LLM interpretation, so
    # the two run concurrently (see report_generator). The prompt already
    # instructs the model to distinguish incurred loss from future spend.
    evidence_ledger = evidence_ledger or []

    from app.remediation.interpreter import interpret_remediation

    status, interp = await interpret_remediation(
        finding_text=finding_text,
        evidence_ledger=evidence_ledger,
        root_cause=root_cause,
        capa=capa,
        impact=impact,
        client=client,
    )

    if status == "NO_EVIDENCE":
        return honest_not_assessable("NO_EVIDENCE", "INSUFFICIENT_EVIDENCE")
    if interp is None:
        return honest_not_assessable(status)

    try:
        valid_evidence_ids = {f"E{i}" for i in range(len(evidence_ledger))}
        components, proposals, outcome = validate_and_plan(
            interp,
            valid_evidence_ids=valid_evidence_ids,
            valid_hypothesis_ids=_hypothesis_ids(root_cause),
            valid_capa_refs=_capa_refs(capa),
        )
        est = assemble_estimate(components, proposals, outcome.traces)
    except Exception as exc:  # fail-closed: a bug must never fabricate a number
        logger.warning("Remediation cost validation/calculation failed unexpectedly (%s).", exc)
        return honest_not_assessable("LLM_INVALID")

    strategy = interp.strategy
    activities = [a.description for a in interp.activities if a.description]

    # --- Partial-coverage bookkeeping (spec section 14): activities/drivers the
    #     evidence identifies as required but cannot price. The priced portion is
    #     still reported; the result is NOT forced to NOT_ASSESSABLE.
    _unpriced_ids = set(est.unpriced_component_ids)
    unpriced_activities = _dedup([
        c.description for c in components if c.component_id in _unpriced_ids and c.description
    ])
    priced_count = len(components) - len(_unpriced_ids)

    # --- Overall status.
    has_evidence_backed_component = any(
        c.unit_cost_basis in ("VERIFIED", "REPORTED") or c.quantity_basis == "EVIDENCED"
        for c in components
    )
    bounded = (
        est.most_likely is not None
        or est.low is not None
        or est.high is not None
        or est.recurring_cost is not None
    )
    is_partial = bounded and bool(unpriced_activities) and priced_count > 0

    if not bounded and est.estimate_classification == CostBasis.NOT_ESTABLISHED:
        result = honest_not_assessable(
            "OK", interp.not_assessable_reason or "PRICING_BASIS_UNAVAILABLE"
        )
        # keep the qualitative reasoning the LLM produced even though no number survived
        result.remediation_strategy = strategy.remediation_summary or ""
        result.remediation_rationale = strategy.remediation_type or ""
        result.established_basis = strategy.established_basis or ""
        result.hypothetical_basis = strategy.hypothetical_basis or ""
        result.alternative_strategies = list(strategy.alternative_strategies)
        result.implementation_activities = activities
        result.cost_components = est.component_results
        result.unpriced_activities = unpriced_activities
        result.uncertainty_reasons = _dedup(interp.uncertainty_reasons + est.uncertainty_reasons)
        result.evidence_improves_estimate = list(interp.evidence_improves_estimate)
        result.calculation_traces = list(outcome.traces)
        result.rejected_items = list(outcome.rejected)
        return result

    if has_evidence_backed_component and bounded:
        overall = RemediationEstimateStatus.EVIDENCE_BACKED
    elif bounded:
        overall = RemediationEstimateStatus.ASSUMPTION_BASED
    else:
        overall = RemediationEstimateStatus.NOT_ASSESSABLE

    # --- Confidence.
    if est.estimate_classification == CostBasis.VERIFIED:
        confidence = RemediationConfidence.HIGH
    elif overall == RemediationEstimateStatus.EVIDENCE_BACKED:
        confidence = RemediationConfidence.MEDIUM
    elif overall == RemediationEstimateStatus.ASSUMPTION_BASED:
        confidence = RemediationConfidence.LOW
    else:
        confidence = RemediationConfidence.NOT_ASSESSABLE

    # If root cause is not established the remediation itself is contingent.
    rc_status = getattr(getattr(root_cause, "status", None), "value", getattr(root_cause, "status", None))
    contingent = rc_status in ("NOT_ESTABLISHED", "STATED_UNVERIFIED", "CONTRADICTED", None)

    assumptions = _dedup(
        [a for c in interp.cost_components for a in c.assumptions]
        + list(interp.range_assumptions)
    )
    if contingent:
        assumptions.append(
            "Root cause is not fully established; the remediation scope and therefore this "
            "estimate are contingent on confirming the cause."
        )
        if confidence == RemediationConfidence.HIGH:
            confidence = RemediationConfidence.MEDIUM

    uncertainty = _dedup(interp.uncertainty_reasons + est.uncertainty_reasons)
    if is_partial:
        uncertainty.append(
            f"{len(unpriced_activities)} implementation activit"
            f"{'y' if len(unpriced_activities) == 1 else 'ies'} could not be priced from the "
            "available evidence; the amounts shown cover only the priced portion."
        )
        if confidence == RemediationConfidence.HIGH:
            confidence = RemediationConfidence.MEDIUM

    result = RemediationCostResult(
        status=overall,
        remediation_strategy=strategy.remediation_summary or "",
        remediation_rationale=strategy.remediation_type or "",
        established_basis=strategy.established_basis or "",
        hypothetical_basis=strategy.hypothetical_basis or "",
        alternative_strategies=list(strategy.alternative_strategies),
        implementation_activities=activities,
        cost_components=est.component_results,
        currency=est.currency,
        one_time_cost=est.one_time_cost,
        recurring_cost=est.recurring_cost,
        recurring_period=est.recurring_period,
        low_estimate=est.low,
        most_likely_estimate=est.most_likely,
        high_estimate=est.high,
        unpriced_activities=unpriced_activities,
        is_partial_estimate=is_partial,
        estimate_classification=est.estimate_classification,
        confidence=confidence,
        assumptions=assumptions,
        range_assumptions=list(interp.range_assumptions),
        uncertainty_reasons=uncertainty,
        evidence_basis=_dedup(
            [r for c in components for r in c.source_reference_ids]
        ),
        estimation_method=est.estimation_method,
        evidence_improves_estimate=list(interp.evidence_improves_estimate),
        review_required=True,
        reasoning_source="LLM_SEMANTIC",
        remediation_semantic_status="OK",
        calculation_traces=list(outcome.traces),
        rejected_items=list(outcome.rejected),
    )
    return result


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        s = (it or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out
