"""Final output hardening, Part 2/3/4: causal_sufficiency now controls
narrative generation. Closes the exact gap disclosed in the prior turn
("causal_sufficiency is populated but never consulted by narrative
generation") by reusing the existing narrative-safety-gate pattern in
final_evidence_verification_node (previously only applied for
rc.status == NOT_ESTABLISHED) and extending it to also fire when
rc.status == ESTABLISHED but the licensed causal graph only reaches
mechanism-level depth, never root-cause-level depth.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import CausalLevel, InvestigateRequest, RootCauseStatus

_FOUR_EMPLOYEES_FINDING = (
    "Four employees failed to complete the revised inspection checklist. "
    "One employee reported insufficient training. "
    "Another employee reported workload pressure. "
    "The supervisor reported poor discipline."
)


async def _build_established_but_mechanism_only_state():
    """Runs the real deterministic pipeline for canonical/hypothesis
    scaffolding, then constructs the exact adversarial mismatch this gate
    exists to catch: rc.status=ESTABLISHED with a narrative claiming root
    cause, but the hypothesis is only mechanism-level (causal_level
    L3_IMMEDIATE_MECHANISM, never promoted to a root-cause-level node) --
    simulating a genuine drift between whatever set rc.status and the
    causal graph's own licensed depth."""
    req = InvestigateRequest(finding_text=_FOUR_EMPLOYEES_FINDING)
    state = {
        "request": req, "evidence_ledger": [], "iteration_count": 0, "tool_call_count": 0,
        "critic_iteration": 0, "trace": [], "errors": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
        state = await core_synthesis_node(state)
        state = await generate_report_node(state)

    rc = state.get("root_cause")
    assert rc is not None and rc.candidate_hypotheses
    hyp = rc.candidate_hypotheses[0]
    # Reproduces a genuine cross-derivation disagreement: select_
    # authoritative_leading_hypothesis (app.agent.causal_graph) trusts the
    # hypothesis's OWN evidence_strength="VERIFIED" label directly and
    # returns ESTABLISHED for an L4_ROOT_CAUSE-level hypothesis -- but
    # build_causal_graph applies a STRICTER, independent rule (only an
    # actual VERIFIED item in this run's evidence_ledger may license a
    # VERIFIED edge; state["evidence_ledger"]=[] here) and downgrades the
    # edge to POSSIBLE. Two authoritative-sounding derivations, same
    # inputs, genuinely disagreeing -- exactly what this narrative gate
    # exists to catch.
    hyp.causal_level = CausalLevel.L4_ROOT_CAUSE
    hyp.status = "SUPPORTED"
    hyp.evidence_strength = "VERIFIED"
    hyp.causal_node_id = None  # let build_causal_graph re-derive the node fresh
    hyp.causal_edge_id = None
    hyp.status_locked = True

    rc.leading_hypothesis = hyp.id
    rc.leading_hypothesis_status = "SELECTED"
    rc.narrative = "The evidence establishes the root cause of the deviation."
    rc.root_cause_basis = "The evidence establishes the root cause of the deviation."
    state["evidence_ledger"] = []  # no VERIFIED ledger item -- the actual source of the disagreement
    return state


def test_established_root_cause_narrative_downgraded_when_only_mechanism_licensed():
    state = asyncio.run(_build_established_but_mechanism_only_state())
    final_state = asyncio.run(final_evidence_verification_node(state))
    rc = final_state.get("root_cause")
    assert "root cause" not in rc.narrative.lower() or "does not establish" in rc.narrative.lower(), (
        f"narrative still claims an established root cause despite only mechanism-level causal "
        f"sufficiency: {rc.narrative!r}"
    )
    assert "mechanism" in rc.narrative.lower()


def test_established_root_cause_narrative_unchanged_when_root_cause_actually_licensed():
    """Confirms the gate is not overcorrecting: a genuinely root-cause-
    level licensed graph must keep its narrative untouched by this check."""
    req = InvestigateRequest(finding_text="Audit trail logs confirmed that the security interlock on valve "
                                           "V-101 was disabled on August 12 without required change-management "
                                           "authorization.")
    state = {
        "request": req, "evidence_ledger": [], "iteration_count": 0, "tool_call_count": 0,
        "critic_iteration": 0, "trace": [], "errors": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = asyncio.run(understand_finding_node(state))
        state = asyncio.run(plan_investigation_node(state))
        state = asyncio.run(core_synthesis_node(state))
        state = asyncio.run(generate_report_node(state))
        state = asyncio.run(final_evidence_verification_node(state))
    rc = state.get("root_cause")
    assert rc.status == RootCauseStatus.ESTABLISHED
    assert rc.causal_sufficiency.root_cause_sufficiency == "ESTABLISHED"
    # Real established case: narrative was never touched by the new
    # mechanism-only-downgrade branch (must not contain the fallback text).
    assert "does not establish the underlying root cause" not in rc.narrative
