"""Phase 19: evidence acquisition + deterministic evidence -> claim ->
hypothesis reconciliation node.

Wired into `app/agent/graph.py` as a new node, `acquire_evidence`, reached
via a conditional edge from `causal_investigation_planner` (Stage B) and
looping back to it -- the actual missing link Phase 18 disclosed. The loop
only activates when `state["evidence_provider"]` is set (Section 1: the
core engine depends on the abstract interface, never a concrete backend);
every one of the 1564 pre-Phase-19 tests leaves that key unset, so the new
conditional edge's default branch is identical to Phase 17/18 behavior --
zero regression risk by construction, not by omission.

Pipeline (Section 4/6/7 — never skipped, never reordered):

    Stage-B question
        -> EvidenceRequest   (build_evidence_requests, deterministic)
        -> EvidenceProvider.acquire()   (retrieval only, no causal judgment)
        -> EvidenceItem   (raw, provider-supplied hypothesis_relevance)
        -> reconcile_hypothesis_from_evidence()   (deterministic status transition)
        -> HypothesisStatusChange   (appended, never overwrites history)
        -> updated CandidateHypothesis.status/evidence_strength
        -> loop back to causal_investigation_planner_node (rebuilds CausalGraph,
           increments graph_version, produces the next plan from the NEW state)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from app.agent.claim_extractor import detect_evidence_conflicts
from app.agent.evidence_interpreter import derive_hypothesis_relevance
from app.agent.state import AgentState
from app.models.agent import AgentTraceStep, EvidenceClaim, EvidenceRequest, HypothesisStatusChange

logger = logging.getLogger(__name__)

# Mirrors app.agent.causal_graph.EVIDENCE_STRENGTH_RANK / HYPOTHESIS_STATUS_RANK
# -- the SAME ordinal vocabulary the rest of the architecture already uses,
# never a competing one.
_EVIDENCE_ITEM_STATUS_STRENGTH = {"VERIFIED": "VERIFIED", "REPORTED": "REPORTED"}


def build_evidence_requests(plan: Any, graph_version: int, iteration_id: int) -> list[EvidenceRequest]:
    """Deterministic: one EvidenceRequest per Stage-B question. Never built
    from raw finding text -- every field traces to the question's own
    structured targets (Section 11)."""
    requests = []
    for q in getattr(plan, "questions", None) or []:
        requests.append(EvidenceRequest(
            request_id=f"REQ_{uuid.uuid4().hex[:8]}",
            question_id=str(getattr(q, "question_id", None) or getattr(q, "id", None) or ""),
            target_node_id=getattr(q, "target_node_id", None),
            target_edge_id=getattr(q, "target_edge_id", None),
            hypothesis_ids=list(getattr(q, "target_hypothesis_ids", None) or []),
            evidence_types=[],
            required_artifacts=[getattr(q, "evidence_required", None) or getattr(q, "evidence", "")],
            objective=getattr(q, "objective", "") or getattr(q, "purpose", ""),
            iteration_id=iteration_id,
            graph_version=graph_version,
        ))
    return requests


def request_identity(req: Any) -> tuple:
    """Phase 22 Part B: canonical, deterministic, wording-independent
    identity for an EvidenceRequest. Built ONLY from structured semantic
    targets that already exist on the model -- never from `objective` or
    `question_id` (free text/labels) and never a hash of question wording.
    Two differently-worded questions that target the same node/edge,
    hypotheses, and evidence requirement collide on this identity (the
    same logical request); a question with a genuinely different
    evidence_types/required_artifacts target does not, even if it targets
    the same node/hypothesis (a legitimate refinement, not a duplicate --
    Part C)."""
    return (
        getattr(req, "target_node_id", None),
        getattr(req, "target_edge_id", None),
        tuple(sorted(getattr(req, "hypothesis_ids", None) or [])),
        tuple(sorted(getattr(req, "evidence_types", None) or [])),
        tuple(sorted(a for a in (getattr(req, "required_artifacts", None) or []) if a)),
    )


def reconcile_evidence_request(new_req: EvidenceRequest, existing_requests: list[EvidenceRequest]) -> tuple[EvidenceRequest, str]:
    """Phase 22 Part C/D: deterministic request-lifecycle reconciliation.
    Returns (request_to_use, action) where action is one of:

      REUSED   -- an outstanding or already-fulfilled request with the
                  IDENTICAL structured identity already exists; return that
                  SAME request object (never issue a second one for it --
                  Part F idempotency) instead of `new_req`.
      REFINED  -- `new_req` targets the same node/hypothesis as an existing
                  UNRESOLVED request but asks for something structurally
                  different (a genuine refinement, Part C's worked
                  example) -- `new_req.parent_request_id` is set and the
                  parent is marked SUPERSEDED (never deleted, never
                  overwritten -- Part D).
      CREATED  -- a genuinely new request; no relationship to any existing
                  one.
    """
    identity = request_identity(new_req)
    for existing in existing_requests:
        if existing.status in ("FULFILLED", "REQUESTED", "PARTIALLY_FULFILLED") and request_identity(existing) == identity:
            return existing, "REUSED"

    for existing in existing_requests:
        if existing.status != "UNRESOLVED" or request_identity(existing) == identity:
            continue
        same_target = bool(
            (new_req.target_node_id and new_req.target_node_id == existing.target_node_id)
            or (new_req.hypothesis_ids and set(new_req.hypothesis_ids) & set(existing.hypothesis_ids))
        )
        if same_target:
            new_req.parent_request_id = existing.request_id
            existing.status = "SUPERSEDED"
            return new_req, "REFINED"

    return new_req, "CREATED"


def reconcile_hypothesis_from_evidence(
    hypothesis: Any, evidence_items: list[Any], claims: list[Any] | None = None,
) -> tuple[str, str, str]:
    """Phase 19 Section 6/7, extended by Phase 21 Part M: deterministic
    evidence -> hypothesis status reconciliation. Returns (new_status,
    new_evidence_strength, reason). Never invents a stronger status than the
    evidence's OWN verification level supports -- REPORTED evidence can
    never promote a hypothesis to SUPPORTED (the same rule INV-CAUSAL-006
    already enforces elsewhere in this architecture; this function is
    simply the one place that decides the transition, not a second,
    competing rule).

    `claims` (Phase 21, optional; per-hypothesis relation reading added
    Phase 23 Part K): the EvidenceClaim[] extracted by
    app.agent.evidence_interpreter for THIS hypothesis's evidence batch.
    Still the SAME authoritative evaluator/decision procedure as Phase 20
    -- claims never write status directly. Phase 23: because a single
    EvidenceItem can now be interpreted once against MULTIPLE hypotheses
    (Part K -- "SUPPORTING H1, CONTRADICTING H2, INSUFFICIENT H3" from the
    same evidence), EvidenceItem.hypothesis_relevance (a single-valued
    field) can no longer reliably carry a different verdict per hypothesis
    when several hypotheses share the same EvidenceItem. When `claims`
    carries explicit per-hypothesis `hypothesis_relations` data for THIS
    hypothesis.id, that is the authoritative relevance signal; only when
    no such relation data exists does this function fall back to the
    legacy `evidence_items[i].hypothesis_relevance` (provider-set values,
    e.g. test fixtures/providers that don't use the interpreter at all --
    Phase 19/20 behavior, unchanged for them).
    """
    current_status = str(getattr(hypothesis, "status", "POSSIBLE"))
    current_strength = str(getattr(hypothesis, "evidence_strength", "NONE"))
    hid = str(getattr(hypothesis, "id", ""))

    have_relation_data = bool(claims) and any(
        r.hypothesis_id == hid for c in claims for r in (getattr(c, "hypothesis_relations", None) or [])
    )
    if have_relation_data:
        relation_votes = {
            r.relation for c in claims for r in (getattr(c, "hypothesis_relations", None) or [])
            if r.hypothesis_id == hid
        }
        has_supporting = "SUPPORTING" in relation_votes
        has_contradicting = "CONTRADICTING" in relation_votes
        has_conflicting = has_supporting and has_contradicting
    else:
        relevances = [getattr(e, "hypothesis_relevance", None) for e in evidence_items]
        has_supporting = "SUPPORTING" in relevances
        has_contradicting = "CONTRADICTING" in relevances
        has_conflicting = "CONFLICTING" in relevances or (has_supporting and has_contradicting)

    claim_suffix = ""
    if claims:
        n = len(claims)
        claim_suffix = f" (grounded in {n} extracted claim{'s' if n != 1 else ''})"

    if has_contradicting and not has_supporting:
        return "REFUTED", "VERIFIED", f"objective evidence contradicts this hypothesis{claim_suffix}"

    if has_conflicting:
        return current_status, "CONFLICTING", f"evidence both supports and contradicts this hypothesis -- unresolved{claim_suffix}"

    if has_supporting:
        # Phase 23: in relation-data mode the SUPPORTING vote came from
        # `claims`, not evidence_items[i].hypothesis_relevance (which may
        # be a single-valued aggregate across several hypotheses, or
        # deliberately left unset -- Part K) -- so every evidence_item in
        # this hypothesis's own batch is eligible for the VERIFIED/REPORTED
        # strength check, not just ones matching a legacy relevance label.
        supporting_items = evidence_items if have_relation_data else [
            e for e in evidence_items if getattr(e, "hypothesis_relevance", None) == "SUPPORTING"
        ]
        # Phase 24: exact equality against the EvidenceStatus values, never
        # `str(x).endswith("VERIFIED")` -- that substring check is true for
        # "EvidenceStatus.UNVERIFIED" too (Python: "UNVERIFIED".endswith(
        # "VERIFIED") is True), which would have let UNVERIFIED evidence
        # promote a hypothesis to SUPPORTED.
        item_statuses = [getattr(e, "status", None) for e in supporting_items]
        has_verified_item = any(s == "VERIFIED" for s in item_statuses)
        has_reported_item = any(s == "REPORTED" for s in item_statuses)
        if has_verified_item:
            return "SUPPORTED", "VERIFIED", f"objective (VERIFIED) evidence supports this hypothesis{claim_suffix}"
        if has_reported_item:
            # REPORTED evidence strengthens the hypothesis's standing but
            # must NOT promote status to SUPPORTED (mirrors INV-CAUSAL-006).
            return current_status, "REPORTED", f"reported (non-VERIFIED) evidence supports this hypothesis{claim_suffix}"

    # INSUFFICIENT / UNAVAILABLE / UNRESOLVED only: no change.
    return current_status, current_strength, "no evidence retrieved that changes this hypothesis's standing"


async def acquire_evidence_node(state: AgentState) -> AgentState:
    """The real LangGraph node. Reached only via the conditional edge from
    causal_investigation_planner when a real evidence_provider is
    configured in state AND actionable uncertainty remains AND the
    iteration budget is not exhausted (see graph.py's
    `stage_b_loop_decision`)."""
    trace = list(state.get("trace", []))
    provider = state.get("evidence_provider")
    plan = state.get("causal_investigation_plan")
    root_cause = state.get("root_cause")
    evidence_ledger = list(state.get("evidence_ledger", []))
    hypothesis_history = list(state.get("hypothesis_history", []))
    evidence_claims = list(state.get("evidence_claims", []))
    evidence_conflicts = list(state.get("evidence_conflicts", []))
    graph_version = state.get("causal_graph_version", 0)
    iteration = state.get("investigation_iteration", 0)
    # Phase 21: optional -- an app.agent.evidence_interpreter.EvidenceInterpreter
    # instance, or None (the default for every pre-Phase-21 caller and the
    # entire existing test corpus). When None the claim/interpretation step
    # is skipped entirely and behavior is byte-for-byte identical to Phase
    # 19/20 -- zero regression risk by construction, same pattern as
    # `evidence_provider` itself.
    interpreter = state.get("evidence_interpreter")

    if provider is None or plan is None or root_cause is None:
        trace.append(AgentTraceStep.warn("Evidence acquisition skipped — no provider/plan/root_cause available"))
        return {**state, "trace": trace}

    raw_requests = build_evidence_requests(plan, graph_version, iteration)
    hyp_by_id = {str(getattr(h, "id", "")): h for h in getattr(root_cause, "candidate_hypotheses", None) or []}
    evidence_by_hyp: dict[str, list] = {}
    claims_by_hyp: dict[str, list[EvidenceClaim]] = {}

    # Phase 22 Part B/C/D/F: request-lifecycle reconciliation. The
    # evidence_requests ledger is append-only across iterations (mirrors
    # evidence_ledger/hypothesis_history) -- REUSED requests are never
    # re-appended (they already live in the ledger), so acquiring evidence
    # twice for byte-for-byte the same structured target is impossible by
    # construction, not by a special case.
    evidence_requests_ledger = list(state.get("evidence_requests", []))
    ledger_by_id = {r.request_id: r for r in evidence_requests_ledger}
    requests: list[EvidenceRequest] = []
    for raw in raw_requests:
        req, action = reconcile_evidence_request(raw, evidence_requests_ledger)
        trace.append(AgentTraceStep.ok(f"EvidenceRequest {action}: {req.request_id} (target={request_identity(req)})"))
        # Observability (Part Q): IDs and structured metadata only -- never
        # evidence/claim/finding TEXT (that's what `req.objective`,
        # `item.claim`, etc. would leak).
        logger.info(
            "event=evidence_request_%s request_id=%s parent_request_id=%s target_node_id=%s "
            "hypothesis_ids=%s iteration=%d",
            action.lower(), req.request_id, req.parent_request_id, req.target_node_id,
            req.hypothesis_ids, iteration,
        )
        if action == "REUSED" and req.status == "FULFILLED":
            # Already resolved by an earlier iteration -- nothing to
            # re-acquire, re-interpret, or re-reconcile (Part F idempotency).
            continue
        if req.request_id not in ledger_by_id:
            evidence_requests_ledger.append(req)
            ledger_by_id[req.request_id] = req
        requests.append(req)

    for req in requests:
        try:
            item = await provider.acquire(req)
        except Exception as exc:  # never let a provider failure crash the pipeline
            logger.warning("event=evidence_provider_failure request_id=%s provider=%s error=%s",
                            req.request_id, type(provider).__name__, exc)
            trace.append(AgentTraceStep.warn(f"Evidence acquisition failed for {req.request_id}: {exc}"))
            req.status = "UNRESOLVED"
            continue
        logger.info(
            "event=evidence_item_acquired request_id=%s evidence_id=%s source=%s status=%s "
            "provider=%s",
            req.request_id, item.evidence_id, item.source, item.status, type(provider).__name__,
        )
        if not item.evidence_id:
            item.evidence_id = f"EV_{req.request_id}"
        item.request_id = req.request_id
        item.question_id = req.question_id
        item.iteration_id = iteration
        if not item.collection_timestamp:
            item.collection_timestamp = datetime.now(timezone.utc).isoformat()
        evidence_ledger.append(item)

        # Phase 21 Part G/M, batched Phase 23 Part K/W: interpret the
        # evidence into claims ONCE per evidence item, classified against
        # EVERY target hypothesis in a single call (never one LLM call per
        # hypothesis -- Part W: "prefer one structured interpretation call
        # per evidence batch"), but ONLY when the provider itself did not
        # already supply a relevance judgment (provider authority is
        # preserved -- when a provider DOES supply a relevance, e.g. a test
        # fixture or a future provider with its own domain logic, that
        # judgment wins).
        req_hyps = [hyp_by_id[h] for h in req.hypothesis_ids if hyp_by_id.get(h) is not None]
        if interpreter is not None and item.hypothesis_relevance in (None, "UNRESOLVED") and req_hyps:
            try:
                new_claims = await interpreter.interpret(
                    item,
                    hypotheses=[
                        {"id": str(h.id), "statement": getattr(h, "statement", "") or "",
                         "status": str(getattr(h, "status", "") or "")}
                        for h in req_hyps
                    ],
                    question=req.objective,
                )
            except Exception as exc:  # fail safe -- provider failure != evidence (Part O)
                logger.warning("event=evidence_interpreter_failure evidence_id=%s error=%s", item.evidence_id, exc)
                new_claims = []

            if new_claims:
                new_conflicts = detect_evidence_conflicts(evidence_claims + new_claims)
                # detect_evidence_conflicts recomputes the full conflict set
                # deterministically each call (never silently pick a
                # winner) -- merge in only conflicts not already recorded.
                existing_conflict_claim_sets = {tuple(sorted(c.claims)) for c in evidence_conflicts}
                for conflict in new_conflicts:
                    key = tuple(sorted(conflict.claims))
                    if key not in existing_conflict_claim_sets:
                        evidence_conflicts.append(conflict)
                        existing_conflict_claim_sets.add(key)
                        logger.info(
                            "event=evidence_conflict_detected conflict_id=%s claim_ids=%s status=%s",
                            conflict.conflict_id, conflict.claims, conflict.status,
                        )
                evidence_claims.extend(new_claims)
                # Phase 23 Part K: the SAME claim list is attributed to
                # every target hypothesis -- derive_hypothesis_relevance/
                # reconcile_hypothesis_from_evidence filter it down to
                # relations for ONE hypothesis_id each, so H1 and H2 can
                # (and legitimately do) reach different verdicts from the
                # same underlying claims.
                for h in req_hyps:
                    claims_by_hyp.setdefault(str(h.id), []).extend(new_claims)
                trace.append(AgentTraceStep.ok(
                    f"Evidence interpretation: {len(new_claims)} claim(s) extracted for {item.evidence_id} "
                    f"-> {[str(h.id) for h in req_hyps]}"
                ))

            # Best-effort single-valued aggregate on the EvidenceItem itself
            # (kept for observability/back-compat with any code reading
            # item.hypothesis_relevance directly) -- NOT the reconciliation
            # authority for multi-hypothesis requests; that reads
            # claims_by_hyp[hid] per-hypothesis below.
            relevance = derive_hypothesis_relevance(new_claims)
            if relevance is not None:
                item.hypothesis_relevance = relevance

        for hid in req.hypothesis_ids:
            evidence_by_hyp.setdefault(hid, []).append(item)

        # Phase 22 Part C, corrected for Phase 23 multi-hypothesis batching:
        # a request resolves once evidence actually says something DECISIVE
        # about at least one of its target hypotheses (checked per-hid via
        # claims_by_hyp, not the single-valued item.hypothesis_relevance
        # aggregate above) -- INSUFFICIENT/UNAVAILABLE/UNRESOLVED/None on
        # every target leaves it open (UNRESOLVED) so a future genuine
        # refinement can supersede it, never re-issued verbatim.
        decisive = any(
            derive_hypothesis_relevance(claims_by_hyp.get(hid, []), hid) in ("SUPPORTING", "CONTRADICTING", "CONFLICTING")
            for hid in req.hypothesis_ids
        ) or item.hypothesis_relevance in ("SUPPORTING", "CONTRADICTING", "CONFLICTING")
        req.status = "FULFILLED" if decisive else "UNRESOLVED"

    for hid, items in evidence_by_hyp.items():
        hyp = hyp_by_id.get(hid)
        if hyp is None:
            continue
        prev_status = str(getattr(hyp, "status", "POSSIBLE"))
        hyp_claims = claims_by_hyp.get(hid, [])
        new_status, new_strength, reason = reconcile_hypothesis_from_evidence(hyp, items, hyp_claims)
        if new_status != prev_status or new_strength != str(getattr(hyp, "evidence_strength", "NONE")):
            hyp.status = new_status
            hyp.evidence_strength = new_strength
            # Phase 20 Part B/C: this IS the single authoritative epistemic
            # evaluator. Lock the hypothesis so no downstream re-derivation
            # (final_evidence_verification_node's eligibility/promotion
            # loops) can silently overwrite this decision.
            hyp.status_locked = True
            logger.info(
                "event=hypothesis_status_changed hypothesis_id=%s previous_status=%s new_status=%s "
                "status_locked=True iteration=%d",
                hid, prev_status, new_status, iteration,
            )
            hypothesis_history.append(HypothesisStatusChange(
                hypothesis_id=hid, previous_status=prev_status, new_status=new_status,
                changed_by_evidence_ids=[str(getattr(i, "evidence_id", "")) for i in items],
                changed_by_claim_ids=[str(getattr(c, "claim_id", "")) for c in hyp_claims],
                graph_version=graph_version, iteration_id=iteration, reason=reason,
            ))
            trace.append(AgentTraceStep.ok(
                f"Evidence reconciliation: hypothesis {hid} {prev_status} -> {new_status} ({reason})"
            ))

    return {
        **state,
        "evidence_ledger": evidence_ledger,
        "hypothesis_history": hypothesis_history,
        "evidence_claims": evidence_claims,
        "evidence_conflicts": evidence_conflicts,
        "evidence_requests": evidence_requests_ledger,
        "trace": trace,
    }
