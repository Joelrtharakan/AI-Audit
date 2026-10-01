"""Phase 9.8 -- structural cross-section consistency review of the canonical
semantic state.

BOUNDARY (category A/C: structural + provenance validation). This module
compares the LLM's OWN structured fields against each other and checks that
evidence references resolve. It reads NO prose: no keywords, no regex, no
vocabulary overlap. It never rewrites the LLM output -- it reports issues so
the caller can mark the result review-required and/or request regeneration
(fail closed, never silently patch).

Each issue is a short stable code plus the offending ids, e.g.
`CORRECTIVE_ACTION_WITHOUT_ESTABLISHED_CAUSE:A2`.
"""

from __future__ import annotations

from app.services.canonical_semantic_models import CanonicalFindingContext

_SELF_REVIEW_FLAGS = (
    "introduced_unsupported_fact", "confused_observation_with_cause",
    "promoted_belief_to_verified", "lost_material_evidence", "lost_cost_component",
    "merged_one_time_and_recurring", "collapsed_range", "invented_horizon",
    "invented_process_or_object", "unsupported_investigation_question",
    "claimed_action_authorized_or_completed_without_evidence",
    "went_past_evidence_boundary",
)


def _valid_ids(evidence_count: int) -> set[str]:
    return {f"E{i}" for i in range(evidence_count)}


def review_canonical_consistency(
    ctx: CanonicalFindingContext, evidence_count: int
) -> list[str]:
    """Return a list of structural contradiction codes (empty = consistent)."""
    issues: list[str] = []
    valid = _valid_ids(evidence_count)
    hyp_ids = {h.hypothesis_id for h in ctx.candidate_hypotheses}
    verified_cause = any(
        c.is_causal and c.evidence_status == "VERIFIED" and c.cause_ref
        for c in ctx.causal_claims
    )

    # --- root cause <-> hypotheses <-> actions ---------------------------
    if ctx.root_cause_status == "ESTABLISHED" and not verified_cause:
        issues.append("ROOT_CAUSE_ESTABLISHED_WITHOUT_VERIFIED_CAUSAL_CLAIM")
    if ctx.leading_hypothesis_id:
        if ctx.leading_hypothesis_id not in hyp_ids:
            issues.append(f"LEADING_HYPOTHESIS_DANGLING:{ctx.leading_hypothesis_id}")
        if ctx.root_cause_status != "ESTABLISHED" or ctx.causal_alternatives_unresolved:
            issues.append("LEADING_HYPOTHESIS_WITHOUT_ESTABLISHED_CAUSE")
    for h in ctx.candidate_hypotheses:
        if h.epistemic == "SUPPORTED" and not verified_cause:
            issues.append(f"HYPOTHESIS_SUPPORTED_WITHOUT_VERIFIED_CLAIM:{h.hypothesis_id}")
    for a in ctx.remediation_activities:
        if a.disposition == "CORRECTIVE_ACTION" and ctx.root_cause_status != "ESTABLISHED":
            issues.append(f"CORRECTIVE_ACTION_WITHOUT_ESTABLISHED_CAUSE:{a.action_id}")
    if ctx.remediation_obligation in ("RECONCILIATION_REQUIRED", "INVESTIGATION_REQUIRED"):
        for a in ctx.remediation_activities:
            if a.disposition == "CORRECTIVE_ACTION":
                issues.append(f"OBLIGATION_CONTRADICTS_ACTION:{a.action_id}")

    # --- causal claims provenance ---------------------------------------
    for c in ctx.causal_claims:
        if c.evidence_status == "VERIFIED" and not c.source_evidence_ids:
            issues.append(f"VERIFIED_CLAIM_WITHOUT_EVIDENCE:{c.claim_id}")
        if any(e not in valid for e in c.source_evidence_ids):
            issues.append(f"CLAIM_EVIDENCE_UNRESOLVED:{c.claim_id}")

    # --- investigation plan <-> hypotheses ------------------------------
    for i, s in enumerate(ctx.investigation_plan):
        if any(h not in hyp_ids for h in s.related_hypothesis_ids):
            issues.append(f"PLAN_STEP_HYPOTHESIS_DANGLING:{i}")

    # --- investigation gap identity (structure only: ids / exact-string identity)
    seen_gap: dict[str, int] = {}
    ev_owner: dict[str, str] = {}
    for i, st in enumerate(ctx.investigation_plan):
        gid = (st.gap_id or "").strip()
        if gid:
            if gid in seen_gap:
                issues.append(f"GAP_ID_DUPLICATE:{gid}")
            seen_gap[gid] = i
        ev = " ".join((st.evidence_that_would_resolve or "").split()).casefold()
        if ev and ev != "not specified":
            if ev in ev_owner:
                issues.append(f"GAP_DUPLICATE_EVIDENCE:{ev_owner[ev]},{gid or i}")
            else:
                ev_owner[ev] = gid or str(i)
    if ctx.investigation_plan and ctx.information_gaps:
        if {g.strip() for g in ctx.information_gaps} != {s.unknown.strip() for s in ctx.investigation_plan}:
            issues.append("GAPS_LIST_DIVERGES_FROM_PLAN")

    # --- action status provenance (proposal is never authorization) -----
    for a in list(ctx.remediation_activities) + list(ctx.investigation_activities):
        if a.action_status in ("AUTHORIZED", "COMPLETED"):
            if not a.action_status_evidence_ids:
                issues.append(f"ACTION_STATUS_WITHOUT_EVIDENCE:{a.action_id}")
            elif any(e not in valid for e in a.action_status_evidence_ids):
                issues.append(f"ACTION_STATUS_EVIDENCE_UNRESOLVED:{a.action_id}")

    # --- impact ----------------------------------------------------------
    if ctx.impact is not None and ctx.impact.status == "OBSERVED":
        if not ctx.impact.evidence_ids:
            issues.append("OBSERVED_IMPACT_WITHOUT_EVIDENCE")
        elif any(e not in valid for e in ctx.impact.evidence_ids):
            issues.append("IMPACT_EVIDENCE_UNRESOLVED")

    # --- pricing <-> actions --------------------------------------------
    rem_ids = {a.action_id for a in ctx.remediation_activities}
    for p in ctx.pricing_information:
        if p.action_id and p.action_id not in rem_ids:
            issues.append(f"PRICING_FOR_NON_REMEDIATION_ACTION:{p.action_id}")

    # --- the model's own self-critique ----------------------------------
    if ctx.self_review is not None:
        for flag in _SELF_REVIEW_FLAGS:
            if getattr(ctx.self_review, flag, False):
                issues.append(f"SELF_REVIEW_ADMITS:{flag}")

    return issues


def apply_review(ctx: CanonicalFindingContext, evidence_count: int) -> list[str]:
    """Run the review and record the result on `ctx` (in place). The LLM's
    content is untouched; only `consistency_issues` / `review_required` are
    written. Returns the issues."""
    issues = review_canonical_consistency(ctx, evidence_count)
    ctx.consistency_issues = issues
    ctx.review_required = bool(issues)
    return issues


# --------------------------------------------------------------------------- #
# Phase 9.9 -- regeneration safety
# --------------------------------------------------------------------------- #
# Issue count is NOT a semantic-quality score: a retry with fewer structural
# issues can be semantically worse (e.g. it "fixes" a contradiction by
# promoting a cause to ESTABLISHED). So a retry is adopted only if it is a
# STRUCTURAL repair: every high-risk semantic conclusion is identical to the
# first output and no structured item was lost. Anything else keeps the first
# output and records the divergence for human review.


def _hypothesis_fingerprint(ctx: CanonicalFindingContext) -> tuple:
    return tuple(sorted((h.hypothesis_id, h.epistemic, h.semantic_role) for h in ctx.candidate_hypotheses))


def high_risk_semantic_state(ctx: CanonicalFindingContext) -> dict:
    """The structured conclusions a retry must not change. Enums, ids and
    counts only -- no prose is read or compared."""
    rec = ctx.recurrence
    cmp_ = ctx.comparison
    return {
        "root_cause_status": ctx.root_cause_status,
        "leading_hypothesis_id": ctx.leading_hypothesis_id,
        "causal_alternatives_unresolved": ctx.causal_alternatives_unresolved,
        "epistemic_status": ctx.epistemic_status,
        "missing_record_status": ctx.missing_record_status,
        "remediation_obligation": ctx.remediation_obligation,
        "recurrence": (rec.count, rec.event, rec.period) if rec else None,
        "comparison_status": cmp_.status if cmp_ else None,
        "impact_status": ctx.impact.status if ctx.impact else None,
        "explicit_previous_capa_reference": ctx.explicit_previous_capa_reference,
        "hypotheses": _hypothesis_fingerprint(ctx),
        "causal_claim_status": tuple(sorted((c.claim_id, c.is_causal, c.evidence_status) for c in ctx.causal_claims)),
        "action_dispositions": tuple(sorted(
            (a.action_id, a.disposition, a.action_status)
            for a in list(ctx.remediation_activities) + list(ctx.investigation_activities))),
        "pricing": tuple(sorted(
            (p.action_id or "", p.evidence_available, p.observed_value_is_remediation_cost)
            for p in ctx.pricing_information)),
        "n_gaps": len(ctx.information_gaps),
        "n_plan_steps": len(ctx.investigation_plan),
    }


def retry_is_structural_repair(first: CanonicalFindingContext, retry: CanonicalFindingContext) -> tuple[bool, list[str]]:
    """(adopt?, divergent_fields). Adopt only when no high-risk semantic field
    differs. The comparison is on structured state, never on prose."""
    a, b = high_risk_semantic_state(first), high_risk_semantic_state(retry)
    diverged = sorted(k for k in a if a[k] != b[k])
    return (not diverged, diverged)
