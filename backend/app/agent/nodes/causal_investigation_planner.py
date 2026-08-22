"""Phase 17: Stage B — post-synthesis causal investigation planner.

Audited runtime constraint (read before trusting any claim below): the
existing investigation "re-investigation loop" (critic send-back ->
execute_tool -> record_evidence -> core_synthesis -> critic) is driven by
`_planned_tool_calls`, which is populated ONLY by the LLM tool-planning
path in `plan_investigation_node` (the branch used when ASP.NET LQMS tool
integration is configured). In the deterministic fast-path — the path
`plan_investigation_node` takes whenever ASP.NET is not configured, which
is every test in this suite and any deployment without that integration —
`needs_investigation` and `planned_tools` are set to `False`/`[]`
unconditionally. `should_investigate` therefore always routes straight to
`core_synthesis`, `critic_decision`'s "are there pending tools?" check is
always false, and the send-back branch never actually reaches
`execute_tool` in that mode. Wiring Stage B onto the send-back edge would
therefore be a mechanism that is real but practically unreachable in the
dominant runtime configuration — the exact "vacuous" failure mode Phase 15
flagged, now one level deeper.

Stage B is instead wired into `app/agent/graph.py` as a new node that runs
UNCONDITIONALLY between `core_synthesis` and `critic` — i.e., on every
single graph execution, immediately after `CandidateHypothesis` objects
first exist. This is the earliest point in the actual LangGraph topology
where a genuine causal graph can be built (via the same
`app.agent.causal_graph.build_causal_graph` the final verification stage
uses) and reasoned over.

Honest scope of what this actually changes (see the Phase 17 report for
the full disclosure): Stage B's output is stored in the ADDITIVE
`state["causal_investigation_plan"]` field, not written over
`state["investigation_plan"]` (the field `InvestigationReport.investigation`
is rendered from). A full content-authority swap for the terminal report
was assessed and rejected for the same reason Phase 16 rejected it for
Stage A: hundreds of existing tests assert on the legacy planner's
specific question wording, and swapping it wholesale is an unbounded-
blast-radius rewrite, not a safe migration, within one session. What IS
real: Stage B's `planner_mode` genuinely affects control flow — see
`causal_investigation_planner_node`'s effect on `critic_decision` in
graph.py, which will not send an investigation back for more tool calls
when Stage B has already determined NO_ACTIONABLE_UNCERTAINTY, regardless
of what the LLM-based critic itself concluded.
"""
from __future__ import annotations

import logging
from typing import Any

from app.agent.state import AgentState
from app.models.agent import AgentTraceStep, InvestigationPlan

logger = logging.getLogger(__name__)


def _causal_level_depth(level: Any) -> int:
    from app.agent.causal_graph import _causal_level_depth as _depth
    return _depth(level)


def decide_investigation_state(
    live_unresolved_ids: list[str], iteration: int, max_iterations: int, causal_graph: Any,
) -> str:
    """Phase 18 Section 17: deterministic decision function. Evaluates only
    structural signals (unresolved hypothesis count, iteration budget,
    graph validity) -- never natural-language similarity, never an LLM
    call. Returns one of CONTINUE / REPLAN / RESOLVED / EVIDENCE_BOUNDARY /
    EXHAUSTED / FAIL_CLOSED.
    """
    if causal_graph is None:
        return "FAIL_CLOSED"
    if not live_unresolved_ids:
        return "RESOLVED"
    if iteration >= max_iterations:
        return "EXHAUSTED"
    if iteration == 0:
        return "CONTINUE"
    return "REPLAN"


def plan_investigation_causal(
    canonical_state: Any, root_cause: Any, evidence_ledger: list[Any] | None = None,
) -> tuple[InvestigationPlan, Any, list[str]]:
    """The real Stage B decision function. Returns (plan, causal_graph).

    Deterministic; never inspects raw finding text; never uses lexical
    keyword matching. Operates purely on:
      - CandidateHypothesis.status/evidence_strength/causal_level/
        supporting_claim_ids/deepens_hypothesis_id (already computed by the
        existing eligibility engine, never re-derived here)
      - the licensed CausalGraph built from those hypotheses via the same
        `build_causal_graph` the rest of the architecture already uses.
    """
    from app.agent.causal_graph import build_causal_graph, is_graph_authoritative_for_five_why
    from app.agent.nodes.graph_investigation_planner import build_discriminating_questions_for_competing_hypotheses

    hypotheses = list(getattr(root_cause, "candidate_hypotheses", None) or []) if root_cause else []
    causal_graph = build_causal_graph(canonical_state, root_cause, evidence_ledger or [])

    # 1. Nothing to reason over at all.
    if not hypotheses:
        # Release-gate fix (Check 1): zero candidate_hypotheses is
        # structurally IDENTICAL whether (a) the finding is genuinely
        # non-actionable, or (b) core_synthesis's hypothesis generation
        # simply didn't produce anything for this finding's shape -- a
        # coverage gap, not a verified absence of uncertainty (confirmed
        # reproducible via the real compiled graph: substantive findings
        # like "the calibration certificate was found expired at time of
        # use" or "badge access records show no entry corresponding to the
        # reported visit" produced zero hypotheses under deterministic
        # fallback synthesis). Conflating the two here means a synthesis
        # coverage gap silently presents as "nothing to investigate" --
        # exactly the outcome this system must never produce (mirrors the
        # identical Stage-A correction already applied in
        # app.agent.nodes.graph_investigation_planner.
        # plan_investigation_graph_first). Only when the finding was
        # ALREADY determined non-actionable by the one existing
        # deterministic upstream judgment (CanonicalFindingState.
        # is_actionable, set in app.agent.nodes.understanding) is
        # NO_ACTIONABLE_UNCERTAINTY actually justified; otherwise this
        # honestly reports TARGET_UNRESOLVED (the same vocabulary Stage A
        # already uses for "cannot determine the target").
        is_actionable = getattr(canonical_state, "is_actionable", True) if canonical_state is not None else True
        if is_actionable:
            plan = InvestigationPlan(
                areas=[], questions=[], evidence_to_collect=[],
                status="QUESTIONS_GENERATED", planner_mode="TARGET_UNRESOLVED",
                planner_stage="STAGE_B",
                fallback_reason="no candidate hypotheses were generated for this finding -- not a verified absence of uncertainty",
            )
            return plan, causal_graph, []
        plan = InvestigationPlan(
            areas=[], questions=[], evidence_to_collect=[],
            status="NO_ACTIONABLE_UNCERTAINTY", planner_mode="NO_ACTIONABLE_UNCERTAINTY",
            planner_stage="STAGE_B",
            fallback_reason=None,
        )
        return plan, causal_graph, []

    # 2. Any hypothesis still POSSIBLE/UNVERIFIED (not yet SUPPORTED or
    #    REFUTED) represents live, actionable uncertainty.
    live_unresolved = [h for h in hypotheses if str(getattr(h, "status", "")) not in ("SUPPORTED", "REFUTED")]

    if not live_unresolved:
        plan = InvestigationPlan(
            areas=[], questions=[], evidence_to_collect=[],
            status="NO_ACTIONABLE_UNCERTAINTY", planner_mode="NO_ACTIONABLE_UNCERTAINTY",
            planner_stage="STAGE_B",
            fallback_reason=None,
        )
        return plan, causal_graph, []

    # 3. Competing hypotheses (>=2 independent, unresolved siblings):
    #    generate discriminating evidence requests, never collapse them.
    questions: list = []
    if len(live_unresolved) >= 2:
        questions = build_discriminating_questions_for_competing_hypotheses(live_unresolved, causal_graph)
        for q in questions:
            q.planner_stage = "STAGE_B"

    # 4. Every live hypothesis NOT already covered by a discriminating pair
    #    question above gets its own nearest-unresolved-transition question
    #    (Section 3/INV-INVEST-023: a hypothesis that discrimination logic
    #    happened not to pair up -- e.g. because it shares grounding
    #    evidence with every sibling -- must still be represented, never
    #    silently dropped in favor of a single "the" question).
    covered_ids = {hid for q in questions for hid in (q.target_hypothesis_ids or [])}
    uncovered = [h for h in live_unresolved if str(getattr(h, "id", "")) not in covered_ids]
    from app.models.agent import InvestigationQuestion
    for h in sorted(uncovered, key=lambda h: _causal_level_depth(getattr(h, "causal_level", None))):
        node_id = getattr(h, "causal_node_id", None)
        edge_id = getattr(h, "causal_edge_id", None)
        source_node_id = getattr(h, "causal_edge_source_node_id", None)
        concept = (getattr(h, "name", None) or "").replace("_", " ").title() or str(getattr(h, "id", ""))
        questions.append(InvestigationQuestion(
            question_id=f"SB_{getattr(h, 'id', '1')}",
            id=f"SB_{getattr(h, 'id', '1')}",
            question=f"What objective evidence establishes or refutes {concept} as the causal mechanism?",
            purpose=f"Resolve the nearest unresolved causal transition ({concept})",
            objective=f"Resolve the nearest unresolved causal transition ({concept})",
            evidence="Objective records establishing or refuting the mechanism",
            evidence_required="Objective records establishing or refuting the mechanism",
            target_type="HYPOTHESIS",
            target_node_id=node_id,
            target_edge_id=edge_id,
            source_node_id=source_node_id,
            causal_level=str(getattr(h, "causal_level", None)),
            target_hypothesis_ids=[str(getattr(h, "id", ""))],
            hypotheses_tested=[str(getattr(h, "id", ""))],
            information_gain_band="MEDIUM",
            information_gain_reason="nearest unresolved causal transition for this hypothesis, no competing sibling to discriminate against",
            planner_stage="STAGE_B",
            priority="HIGH",
        ))

    plan = InvestigationPlan(
        questions=questions,
        areas=[f"Resolve hypothesis {q.target_hypothesis_ids}" for q in questions],
        evidence_to_collect=[q.evidence for q in questions],
        status="QUESTIONS_GENERATED", planner_mode="GRAPH_FIRST", planner_stage="STAGE_B",
        graph_targets_considered=len(hypotheses),
        graph_targets_selected=len(questions),
    )
    return plan, causal_graph, [str(getattr(h, "id", "")) for h in live_unresolved]


_DEFAULT_MAX_INVESTIGATION_ITERATIONS = 5


async def causal_investigation_planner_node(state: AgentState) -> AgentState:
    """The real LangGraph node. Wired unconditionally between
    core_synthesis and critic in app/agent/graph.py — executes on EVERY
    graph run, including every pass of the critic-send-back loop (Phase 17
    disclosed this loop is inert in the ASP.NET-free deterministic
    fast-path; when evidence DOES enter via that loop, e.g. the ASP.NET
    tool-planning path or a test that injects evidence directly, this node
    re-runs with the updated evidence_ledger/root_cause and produces a
    genuinely different plan — see the Phase 18 adaptive-loop test).

    Phase 18: adds real iteration/version/history bookkeeping (Section 2/6)
    and the deterministic decide_investigation_state gate (Section 17), on
    top of Phase 17's plan-construction logic, which is unchanged."""
    from app.models.agent import InvestigationIterationRecord

    trace = list(state.get("trace", []))
    canonical = state.get("canonical_finding_state")
    root_cause = state.get("root_cause")
    evidence_ledger = state.get("evidence_ledger", [])

    prev_plan = state.get("causal_investigation_plan")
    prev_unresolved_ids = set(
        hid for q in (getattr(prev_plan, "questions", None) or []) for hid in (getattr(q, "target_hypothesis_ids", None) or [])
    ) if prev_plan is not None else set()
    iteration = state.get("investigation_iteration", 0)
    graph_version = state.get("causal_graph_version", 0)
    history = list(state.get("investigation_history", []))

    try:
        plan, causal_graph, live_ids = plan_investigation_causal(canonical, root_cause, evidence_ledger)
        graph_version += 1
        after_ids = set(live_ids)
        newly_resolved = sorted(prev_unresolved_ids - after_ids)
        newly_created = sorted(after_ids - prev_unresolved_ids)

        decision_state = decide_investigation_state(live_ids, iteration, _DEFAULT_MAX_INVESTIGATION_ITERATIONS, causal_graph)
        if plan.planner_mode == "NO_ACTIONABLE_UNCERTAINTY":
            planner_decision = "NO_ACTIONABLE_UNCERTAINTY"
        elif iteration == 0:
            planner_decision = "NEW_TARGET" if after_ids else "NO_ACTIONABLE_UNCERTAINTY"
        elif newly_created:
            planner_decision = "NEW_UNCERTAINTY"
        elif newly_resolved and not after_ids:
            # every previously-unresolved hypothesis is gone: check whether
            # any survivor was REFUTED (elimination) vs simply resolved.
            refuted_now = any(
                str(getattr(h, "id", "")) in newly_resolved and str(getattr(h, "status", "")) == "REFUTED"
                for h in (getattr(root_cause, "candidate_hypotheses", None) or [])
            )
            planner_decision = "HYPOTHESIS_ELIMINATED" if refuted_now else "TARGET_RESOLVED"
        elif newly_resolved:
            planner_decision = "TARGET_RESOLVED"
        elif after_ids == prev_unresolved_ids:
            planner_decision = "SAME_TARGET"
        else:
            planner_decision = "NEW_TARGET"

        termination_reason = None
        if decision_state == "RESOLVED":
            termination_reason = "CAUSAL_CHAIN_RESOLVED" if newly_resolved else "NO_ACTIONABLE_UNCERTAINTY"
        elif decision_state == "EXHAUSTED":
            termination_reason = "MAX_ITERATIONS"
        elif decision_state == "FAIL_CLOSED":
            termination_reason = "FAIL_CLOSED"

        record = InvestigationIterationRecord(
            iteration_id=iteration,
            planner_stage="STAGE_B",
            investigation_question_ids=[str(getattr(q, "question_id", "") or getattr(q, "id", "")) for q in plan.questions],
            target_node_ids=[str(getattr(q, "target_node_id", None)) for q in plan.questions if getattr(q, "target_node_id", None)],
            target_edge_ids=[str(getattr(q, "target_edge_id", None)) for q in plan.questions if getattr(q, "target_edge_id", None)],
            hypothesis_ids=sorted(after_ids),
            evidence_ids=[str(getattr(e, "claim_id", None) or getattr(e, "source", "")) for e in evidence_ledger],
            causal_graph_version=graph_version,
            unresolved_targets_before=sorted(prev_unresolved_ids),
            unresolved_targets_after=sorted(after_ids),
            newly_resolved_targets=newly_resolved,
            newly_created_uncertainties=newly_created,
            planner_decision=planner_decision,
            termination_reason=termination_reason,
        )
        history.append(record)

        trace.append(AgentTraceStep.ok(
            f"Stage B causal investigation planner: iteration={iteration}, decision={planner_decision}, "
            f"state={decision_state}, planner_mode={plan.planner_mode}, {len(plan.questions)} question(s), "
            f"graph_version={graph_version}"
        ))
        return {
            **state,
            "causal_investigation_plan": plan,
            "causal_graph": causal_graph,
            "causal_graph_version": graph_version,
            "investigation_iteration": iteration + 1,
            "investigation_history": history,
            "trace": trace,
        }
    except Exception as exc:  # pragma: no cover — defensive, never fatal to the pipeline
        logger.warning("Stage B causal investigation planner failed (non-fatal): %s", exc)
        trace.append(AgentTraceStep.warn(f"Stage B causal investigation planner skipped (error): {exc}"))
        return {**state, "trace": trace}
