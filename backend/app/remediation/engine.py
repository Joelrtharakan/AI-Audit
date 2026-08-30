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
import re
from typing import Any

from app.remediation import activities as _act
from app.remediation.activities import (
    build_canonical_activities,
    is_unsupported_concrete_intervention,
)
from app.remediation.calculator import assemble_estimate
from app.remediation.models import (
    CostBasis,
    RemediationConfidence,
    RemediationCostResult,
    RemediationEstimateStatus,
    RemediationUnresolvedDriver,
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
        "Remediation cost cannot be reliably estimated because the remediation scope has "
        "not yet been established from the available evidence — the identified investigation "
        "activities must first determine whether corrective implementation is required and, "
        "if so, define its scope."
    ),
    "INSUFFICIENT_PRICING_INFORMATION": (
        "The required remediation activities are established, but the available evidence "
        "does not yet provide a defensible pricing basis (rates, quantities, quotations or "
        "scope) for them."
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


def _conditional_systemic_sentence(scope: Any, canonical_state: Any) -> str:
    """The canonical, appropriately-abstract conditional systemic action used
    to replace an unsupported concrete prescription."""
    for a in getattr(scope, "activities", []) or []:
        if a.kind == "SYSTEMIC_STRENGTHENING":
            return a.description
    proc = (getattr(canonical_state, "affected_process", "") or "").strip()
    tail = f" over {proc[0].lower() + proc[1:]}" if proc else ""
    return (
        "Subject to confirmation of the underlying cause, determine whether "
        f"strengthened controls{tail} are required and implement the change the "
        "confirmed cause identifies"
    )


def _canon_semantic_type(canonical_state: Any) -> str | None:
    return getattr(canonical_state, "semantic_type", None)


def _derive_activity_fields(
    result: RemediationCostResult, canon: list["_act.CanonicalActivity"],
) -> None:
    """The ONLY place implementation_activities / conditional_activities /
    unpriced_activities / evidence_improves_estimate are written. All four
    come from the ONE canonical collection, so they cannot diverge and no
    evidence item is orphaned from an activity."""
    result.implementation_activities = _act.implementation_activities(canon)
    result.conditional_activities = _act.conditional_activities(canon)
    result.unpriced_activities = _act.unpriced_activities(canon)
    result.evidence_improves_estimate = _act.evidence_improves_estimate(canon)


def _enforce_result_consistency(result: RemediationCostResult) -> RemediationCostResult:
    """Final structural cross-check (spec §24): the semantic fields of the
    result must not contradict one another. Downgrades / clears -- never
    escalates, never invents. Runs on every path just before return."""
    _inv = {str(a).strip().lower() for a in (result.investigation_activities or [])}

    # 1. An activity the canonical interpretation declared investigation can
    #    never also be a priced/implementation activity.
    if _inv:
        result.implementation_activities = [
            a for a in result.implementation_activities if str(a).strip().lower() not in _inv
        ]
        result.unpriced_activities = [
            a for a in result.unpriced_activities if str(a).strip().lower() not in _inv
        ]
        result.conditional_activities = [
            a for a in result.conditional_activities if str(a).strip().lower() not in _inv
        ]

    # 2. conditional_activities must be a subset of implementation_activities.
    _impl = {str(a).strip().lower() for a in result.implementation_activities}
    result.conditional_activities = [
        a for a in result.conditional_activities if str(a).strip().lower() in _impl
    ]

    # 3. NOT_ASSESSABLE / REMEDIATION_NOT_DEFINED  <=>  no priced numbers, no
    #    implementation activities. (INSUFFICIENT_PRICING_INFORMATION keeps the
    #    activities visible but still carries no number.)
    _reason = _reason_code_of(result)
    if result.status == RemediationEstimateStatus.NOT_ASSESSABLE:
        result.one_time_cost = None
        result.recurring_cost = None
        result.low_estimate = None
        result.most_likely_estimate = None
        result.high_estimate = None
        if _reason == "REMEDIATION_NOT_DEFINED":
            result.implementation_activities = []
            result.conditional_activities = []
            result.cost_components = []
            result.unresolved_pricing_drivers = []

    # 4. A calculated estimate cannot coexist with zero implementation activities.
    if (
        result.status != RemediationEstimateStatus.NOT_ASSESSABLE
        and not result.implementation_activities
    ):
        result.status = RemediationEstimateStatus.NOT_ASSESSABLE
        result.confidence = RemediationConfidence.NOT_ASSESSABLE
        result.not_assessable_reason = _PROFESSIONAL_REASON["REMEDIATION_NOT_DEFINED"]
        result.one_time_cost = result.recurring_cost = None
        result.low_estimate = result.most_likely_estimate = result.high_estimate = None

    return result


def _reason_code_of(result: RemediationCostResult) -> str:
    """Recover the machine reason code from the professional sentence (the
    result only stores the rendered text)."""
    txt = result.not_assessable_reason or ""
    for code, sentence in _PROFESSIONAL_REASON.items():
        if code and sentence == txt:
            return code
    return ""


def _rc_established(root_cause: Any) -> bool:
    """Mirror of the engine's existing `contingent` test: the cause is
    'established' for remediation-scope purposes only when it is NOT one of
    the contingent statuses."""
    st = getattr(getattr(root_cause, "status", None), "value", getattr(root_cause, "status", None))
    return st not in ("NOT_ESTABLISHED", "STATED_UNVERIFIED", "CONTRADICTED", None)


def _derive_scope_from_canonical(canonical_state: Any, impact: Any, root_cause: Any):
    """Deterministic, finding-specific remediation SCOPE from the canonical
    semantic state (fail-closed fallback / weak-model backstop). Returns a
    RemediationScope; empty when no substantive subject is established."""
    from app.remediation.scope import derive_remediation_scope

    cs = canonical_state
    subject = (
        getattr(cs, "finding_subject", None)
        or getattr(cs, "affected_object", None)
        or getattr(impact, "affected_object", None)
    )
    condition = getattr(cs, "deviation_condition", None) or getattr(cs, "condition", None)
    semantic_type = getattr(cs, "semantic_type", None)
    affected_process = (
        getattr(cs, "affected_process", None)
        or getattr(impact, "process_at_risk", None)
    )
    is_recurring = bool(
        getattr(cs, "previous_capa_referenced", False)
        or getattr(cs, "occurrence_population", None)
        or (semantic_type or "").upper() == "RECURRENCE"
    )
    try:
        return derive_remediation_scope(
            subject=subject,
            condition=condition,
            semantic_type=semantic_type,
            affected_process=affected_process,
            root_cause_established=_rc_established(root_cause),
            is_recurring=is_recurring,
        )
    except Exception as exc:  # pragma: no cover - never fatal
        logger.warning("Deterministic remediation scope derivation failed (%s).", exc)
        from app.remediation.scope import RemediationScope
        return RemediationScope()


async def estimate_remediation_cost(
    finding_text: str,
    evidence_ledger: list[Any] | None = None,
    root_cause: Any = None,
    capa: Any = None,
    impact: Any = None,
    financial_analysis: Any = None,  # accepted for back-compat; NOT sent to the LLM
    client=None,
    canonical_state: Any = None,
    semantic_context: Any = None,
) -> RemediationCostResult:
    # `financial_analysis` is intentionally not forwarded to the interpreter:
    # remediation cost does not depend on the financial LLM interpretation, so
    # the two run concurrently (see report_generator). The prompt already
    # instructs the model to distinguish incurred loss from future spend.
    evidence_ledger = evidence_ledger or []

    # ------------------------------------------------------------------
    # CANONICAL REMEDIATION DECISION IS AUTHORITATIVE (spec §2/§6/§7/§11/§12).
    # When canonical_semantic_llm_primary is active, `semantic_context` carries
    # the single canonical interpretation's own remediation decision. This
    # layer is a CONSUMER of that decision, not a second RCA/remediation
    # reasoner. If canonical reasoning engaged and established NO remediation
    # activity and NO affirmative corrective obligation, then:
    #   - the independent `interpret_remediation` call must NOT run (it would
    #     re-derive corrective actions from the raw finding);
    #   - the deterministic `derive_remediation_scope` fallback must NOT run
    #     (it would manufacture scope-assessment / causal-investigation work);
    #   - the result is NOT_ASSESSABLE -- the remediation scope has not yet
    #     been established. Investigation work is never priced here.
    # This is §7's exact condition (empty remediation + valid canonical
    # reasoning + no independently established corrective obligation) -- a
    # legitimate MIXED case (canonical remediation_activities non-empty) falls
    # straight through to the normal path.
    # (semantic_context is only ever non-None when the flag is enabled AND the
    # LLM returned a valid interpretation -> flag-off behaviour is unchanged.)
    # ------------------------------------------------------------------
    # A non-None `semantic_context` IS a valid, validated canonical
    # interpretation (the interpreter returns None on any failure). When it
    # exists it is the semantic authority for the investigation/remediation
    # distinction -- an absent/empty `remediation_activities` is a meaningful
    # decision ("remediation not yet established"), not an omission to be
    # back-filled by a second interpreter or the deterministic scope.
    _sc = semantic_context
    _sc_engaged = _sc is not None
    _sc_remediation = list(getattr(_sc, "remediation_activities", []) or [])
    _sc_investigation = list(getattr(_sc, "investigation_activities", []) or [])
    _sc_obligation = getattr(_sc, "remediation_obligation", "NOT_DETERMINED")
    _affirmative_obligation = _sc_obligation in (
        "ESTABLISHED_CORRECTIVE_OBLIGATION", "IMMEDIATE_CORRECTION_ONLY",
    )
    if _sc_engaged and not _sc_remediation and not _affirmative_obligation:
        result = honest_not_assessable("OK", "REMEDIATION_NOT_DEFINED")
        result.remediation_strategy = (
            "Remediation scope has not yet been established from the available evidence. "
            "The identified investigation activities must first determine whether corrective "
            "implementation is required and, if so, define its scope."
        )
        _reasons: list[str] = []
        _why = getattr(_sc, "remediation_obligation_rationale", None)
        if _why:
            _reasons.append(str(_why))
        _inv_descs = [
            a.activity for a in _sc_investigation if getattr(a, "activity", None)
        ]
        for _d in _inv_descs:
            _reasons.append(f"Investigation required: {_d}")
        result.investigation_activities = _inv_descs
        result.uncertainty_reasons = _dedup(_reasons)
        # What evidence would establish a remediation scope -- the canonical
        # information gaps, verbatim (LLM-authored, finding-specific). Never
        # priced, never an activity.
        result.evidence_improves_estimate = _dedup([
            str(g) for g in (getattr(_sc, "information_gaps", []) or []) if str(g).strip()
        ])
        return _enforce_result_consistency(result)

    from app.remediation.interpreter import interpret_remediation

    status, interp = await interpret_remediation(
        finding_text=finding_text,
        evidence_ledger=evidence_ledger,
        root_cause=root_cause,
        capa=capa,
        impact=impact,
        client=client,
        canonical_state=canonical_state,
        semantic_context=semantic_context,
    )

    _scope = _derive_scope_from_canonical(canonical_state, impact, root_cause)
    _semantic_type = _canon_semantic_type(canonical_state)
    _sys_sentence = _conditional_systemic_sentence(_scope, canonical_state)
    rc_status = getattr(getattr(root_cause, "status", None), "value", getattr(root_cause, "status", None))
    contingent = rc_status in ("NOT_ESTABLISHED", "STATED_UNVERIFIED", "CONTRADICTED", None)

    def _kw(text: str | None) -> frozenset:
        return frozenset(re.findall(r"[a-z0-9]{3,}", (text or "").lower()))

    _inv_keys = {_kw(getattr(a, "activity", "")) for a in _sc_investigation}
    _inv_keys.discard(frozenset())

    def _drop_canonical_investigation(canon: list) -> list:
        """Remove any activity the canonical interpretation EXPLICITLY declared
        to be investigation (identity match against the LLM's own list -- not a
        verb rule). Spec §3/§10: investigation is never a priced remediation
        activity, even if a downstream step re-listed it."""
        if not _inv_keys:
            return canon
        return [a for a in canon if _kw(getattr(a, "description", "")) not in _inv_keys]

    def _scope_only_result(reason_status: str, machine_reason: str) -> RemediationCostResult:
        """Second-interpretation output unusable. When the canonical
        interpretation supplied remediation activities, THOSE are authoritative
        (spec §8/§22) -- otherwise the deterministic scope is the fail-closed
        floor. Every field derives from one source, never a mix."""
        r = honest_not_assessable(reason_status, machine_reason)
        canon, _ = build_canonical_activities(
            llm_activities=[], interp_components=[], validated_component_results=[],
            priced_component_ids=set(), scope=_scope, semantic_type=_semantic_type,
            contingent=contingent, use_scope_as_canonical=True,
            conditional_systemic_sentence=_sys_sentence,
            canonical_activities=(_sc_remediation or None),
        )
        canon = _drop_canonical_investigation(canon)
        if _sc_remediation:
            pass  # canonical activities carry their own framing
        elif _scope is not None and getattr(_scope, "approach", ""):
            r.remediation_strategy = _scope.approach
        if _sc_investigation:
            r.investigation_activities = _dedup([
                a.activity for a in _sc_investigation if getattr(a, "activity", None)
            ])
        _derive_activity_fields(r, canon)
        return _enforce_result_consistency(r)

    if status == "NO_EVIDENCE":
        return _scope_only_result("NO_EVIDENCE", "INSUFFICIENT_EVIDENCE")
    if interp is None:
        # Semantic interpretation unavailable/invalid -> fail closed on the
        # NUMBER, but still give the auditor the finding-specific scope.
        return _scope_only_result(status, "")

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

    from app.remediation.scope import looks_like_prompt_echo

    strategy = interp.strategy
    activities = [a.description for a in interp.activities if a.description]

    _priced_ids = {c.component_id for c in components} - set(est.unpriced_component_ids)

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

    # --- THE ONE canonical activity collection. LLM activities + LLM cost
    #     components are reconciled into it; deterministic scope is the
    #     fallback when the LLM output is unusable/echoed/empty. EVERY
    #     downstream activity/evidence field derives from `canon` -- there is
    #     no second source.
    _llm_weak = looks_like_prompt_echo(
        strategy.remediation_summary, activities, list(interp.evidence_improves_estimate)
    )
    # When the LLM proposed no implementation activities at all, prefer the
    # deterministic finding-aware scope (the LLM's cost components then attach
    # to it) rather than letting cost-driver phrases stand in for work.
    #
    # BUT when canonical reasoning is available (flag on) it OWNS the
    # investigation/remediation decision (spec §11): the deterministic
    # `derive_remediation_scope` fallback -- which always emits scope-assessment
    # / causal-investigation activities -- must not stand in for it and
    # manufacture investigation work as remediation. We reach this line only in
    # the legitimate MIXED case (canonical DID establish >=1 remediation
    # activity); trust the LLM interpretation + canonical, never the
    # deterministic scope.
    _use_scope = bool((_llm_weak or not activities) and _scope.activities) and not _sc_engaged
    # SINGLE SOURCE OF TRUTH (spec §8/§22): when the canonical interpretation
    # established the remediation activities, THEY define the work -- the second
    # interpretation contributes only cost components (pricing). The canonical
    # LLM's `disposition` is the authoritative semantic role; the deterministic
    # verb/purpose classifier is not re-run.
    canon, _unresolved_drivers = build_canonical_activities(
        llm_activities=interp.activities,
        interp_components=components,
        validated_component_results=est.component_results,
        priced_component_ids=_priced_ids,
        scope=_scope,
        semantic_type=_semantic_type,
        contingent=contingent,
        use_scope_as_canonical=_use_scope,
        conditional_systemic_sentence=_sys_sentence,
        canonical_activities=(_sc_remediation or None),
    )

    _before = len(canon)
    canon = _drop_canonical_investigation(canon)
    if len(canon) != _before:
        logger.info(
            "Remediation cost: dropped %d activity(ies) the canonical interpretation "
            "declared to be investigation, not remediation.", _before - len(canon),
        )

    _priced_acts = [a for a in canon if a.is_priced]
    _unpriced_acts = [a for a in canon if not a.is_priced]

    if not bounded:
        overall = RemediationEstimateStatus.NOT_ASSESSABLE
    elif has_evidence_backed_component:
        overall = RemediationEstimateStatus.EVIDENCE_BACKED
    else:
        overall = RemediationEstimateStatus.ASSUMPTION_BASED

    # Partial: some cost driver is priced and some remaining work (an unpriced
    # activity OR an unresolved pricing driver) is not. Component-level so it
    # does not depend on how a driver was reconciled.
    _unpriced_component_ids = set(est.unpriced_component_ids)
    is_partial = bounded and bool(_priced_ids) and (
        bool(_unpriced_acts) or bool(_unpriced_component_ids)
    )

    # --- Confidence.
    if est.estimate_classification == CostBasis.VERIFIED:
        confidence = RemediationConfidence.HIGH
    elif overall == RemediationEstimateStatus.EVIDENCE_BACKED:
        confidence = RemediationConfidence.MEDIUM
    elif overall == RemediationEstimateStatus.ASSUMPTION_BASED:
        confidence = RemediationConfidence.LOW
    else:
        confidence = RemediationConfidence.NOT_ASSESSABLE

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
    _unpriced_count = len(_unpriced_acts) + len(_unpriced_component_ids)
    if is_partial and _unpriced_count:
        uncertainty.append(
            f"{_unpriced_count} remediation item{'' if _unpriced_count == 1 else 's'} "
            "could not be priced from the available evidence; the amounts shown cover "
            "only the priced portion."
        )
        if confidence == RemediationConfidence.HIGH:
            confidence = RemediationConfidence.MEDIUM
    if _unresolved_drivers:
        uncertainty.append(
            f"{len(_unresolved_drivers)} cost driver"
            f"{'' if len(_unresolved_drivers) == 1 else 's'} could not be associated with a "
            "specific implementation activity and appear under cost drivers only."
        )

    # --- Strategy wording.
    # SINGLE SOURCE OF TRUTH (spec §20/§22): when the canonical interpretation
    # established the remediation activities, the approach describes THOSE --
    # never the secondary interpreter's summary, which is a fresh reading of
    # the raw finding and can drift into investigation language.
    _summary = strategy.remediation_summary or ""
    _canon_approach = (getattr(_sc, "remediation_obligation_rationale", None) or "").strip()
    if _sc_remediation:
        _framed_strategy = _canon_approach or _frame_strategy(
            "; ".join(a.description for a in canon) if canon else _summary, contingent
        )
    elif _use_scope:
        _framed_strategy = _scope.approach or _frame_strategy(_summary, contingent)
    elif contingent and is_unsupported_concrete_intervention(_summary):
        # the LLM headline prescribes a concrete intervention the evidence has
        # not established the need for -> use the abstract deterministic approach.
        _framed_strategy = _scope.approach or _frame_strategy(_summary, contingent)
    else:
        _framed_strategy = _frame_strategy(_summary, contingent)

    if overall == RemediationEstimateStatus.NOT_ASSESSABLE:
        # Spec INVARIANT 12/13: a NOT_ASSESSABLE where genuine remediation
        # activities ARE established but nothing can be priced is a DIFFERENT
        # state from "remediation not defined" -- name it so.
        _reason_code = interp.not_assessable_reason or "PRICING_BASIS_UNAVAILABLE"
        if canon and _sc_remediation:
            _reason_code = "INSUFFICIENT_PRICING_INFORMATION"
        result = honest_not_assessable("OK", _reason_code)
        result.remediation_rationale = strategy.remediation_type or ""
        result.established_basis = strategy.established_basis or ""
        result.hypothetical_basis = strategy.hypothetical_basis or ""
        result.alternative_strategies = list(strategy.alternative_strategies)
        result.cost_components = est.component_results
        result.uncertainty_reasons = uncertainty
        result.calculation_traces = list(outcome.traces)
        result.rejected_items = list(outcome.rejected)
    else:
        result = RemediationCostResult(
            status=overall,
            remediation_rationale=strategy.remediation_type or "",
            established_basis=strategy.established_basis or "",
            hypothetical_basis=strategy.hypothetical_basis or "",
            alternative_strategies=list(strategy.alternative_strategies),
            cost_components=est.component_results,
            currency=est.currency,
            one_time_cost=est.one_time_cost,
            recurring_cost=est.recurring_cost,
            recurring_period=est.recurring_period,
            low_estimate=est.low,
            most_likely_estimate=est.most_likely,
            high_estimate=est.high,
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
            review_required=True,
            reasoning_source="LLM_SEMANTIC",
            remediation_semantic_status="OK",
            calculation_traces=list(outcome.traces),
            rejected_items=list(outcome.rejected),
        )

    result.remediation_strategy = _framed_strategy
    result.unresolved_pricing_drivers = [
        RemediationUnresolvedDriver(component_id=d.component_id, description=d.description)
        for d in _unresolved_drivers
    ]

    # Consistency with the canonical remediation-obligation verdict (spec §8/§13):
    # in the mixed case (a genuine remediation activity exists alongside an
    # unresolved discrepancy / open cause), say plainly that no SYSTEMIC action
    # is yet justified. Surfacing the canonical verdict only; no remediation
    # content is generated here. The "no remediation activity at all" case has
    # already returned NOT_ASSESSABLE above.
    _oblig = getattr(semantic_context, "remediation_obligation", "NOT_DETERMINED")
    _oblig_line = {
        "RECONCILIATION_REQUIRED":
            "The finding is an unresolved discrepancy; the work required is to reconcile "
            "the values and establish their comparability. No systemic corrective action "
            "is currently justified.",
        "INVESTIGATION_REQUIRED":
            "The cause and scope must be established by investigation before a corrective "
            "action can be identified. No systemic corrective action is currently justified.",
        "NO_SYSTEMIC_REMEDIATION_JUSTIFIED":
            "The evidence does not establish that a systemic control failure exists; no "
            "systemic corrective action is currently justified.",
    }.get(_oblig)
    if _oblig_line and _oblig_line not in result.uncertainty_reasons:
        result.uncertainty_reasons = [_oblig_line, *result.uncertainty_reasons]

    # Surface the canonical investigation activities alongside the (mixed-case)
    # remediation -- separate collection, never priced.
    if _sc_investigation:
        result.investigation_activities = _dedup([
            a.activity for a in _sc_investigation if getattr(a, "activity", None)
        ])

    _derive_activity_fields(result, canon)
    return _enforce_result_consistency(result)


_CONTINGENT_MARKERS = (
    "potential", "contingent", "subject to", "pending", "if confirmed",
    "once the cause", "assuming", "provisional", "may require", "would likely",
)


def _frame_strategy(summary: str, contingent: bool) -> str:
    """Spec section 5: when the root cause is not established the remediation
    approach must be presented as a candidate, never as a directive tied to a
    confirmed cause. Structural, domain-agnostic -- only reframes a summary
    that reads as a directive and doesn't already carry contingency language."""
    s = (summary or "").strip()
    if not s or not contingent:
        return s
    if any(m in s.lower() for m in _CONTINGENT_MARKERS):
        return s
    lead = s[0].lower() + s[1:]
    return f"Potential implementation approach, subject to confirming the underlying cause: {lead}"


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        s = (it or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out
