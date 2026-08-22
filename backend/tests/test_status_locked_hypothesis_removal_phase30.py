"""Final intelligence-hardening pass: regression test for a real,
live-Ollama-reproduced defect. final_evidence_verification_node's ~14
prose-pattern hypothesis-removal checks (is_evidence_state_not_hypothesis,
hypothesis_statement_asserts_unsupported_causation, etc.) examine
h.statement as originally proposed by core_synthesis, BEFORE evidence was
ever attached -- they did not check status_locked, so a hypothesis the
authoritative evidence-reconciliation evaluator (app.agent.nodes.
evidence_acquisition.reconcile_hypothesis_from_evidence) had already
promoted to SUPPORTED on genuinely VERIFIED evidence could still be
silently REMOVED from the final report merely because its statement text
contained a causal connective like "because"/"due to".

Live reproduction (not fabricated): a real compiled-graph run with real
Ollama qwen3:8b produced exactly this -- trace showed
"Evidence reconciliation: hypothesis H1 POSSIBLE -> SUPPORTED" immediately
followed by "Final Evidence Verification: removed hypothesis H1 --
statement asserts causation ... not grounded in a VERIFIED fact", deleting
the very hypothesis the authoritative evaluator had just locked.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import EvidenceItem, EvidenceStatus, HypothesisStatusChange, InvestigateRequest

_FOUR_EMPLOYEES_FINDING = (
    "Four employees failed to complete the revised inspection checklist. "
    "One employee reported insufficient training. "
    "Another employee reported workload pressure. "
    "The supervisor reported poor discipline."
)


async def _build_state_with_locked_hypothesis(finding_text: str, causal_statement: str):
    """Runs the real deterministic pipeline up through core_synthesis, then
    simulates exactly what the Phase 19-21 adaptive evidence loop does when
    it authoritatively locks a hypothesis: sets status=SUPPORTED,
    status_locked=True, and records a HypothesisStatusChange -- never
    mutating anything final_evidence_verification_node itself owns."""
    req = InvestigateRequest(finding_text=finding_text)
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
    assert rc is not None and rc.candidate_hypotheses, "test setup requires at least one generated hypothesis"
    hyp = rc.candidate_hypotheses[0]
    # Simulate the real authoritative evidence-reconciliation outcome.
    hyp.statement = causal_statement
    hyp.status = "SUPPORTED"
    hyp.evidence_strength = "VERIFIED"
    hyp.status_locked = True

    verified_item = EvidenceItem(
        claim=causal_statement, source="system_record", status=EvidenceStatus.VERIFIED,
        evidence_id="EV_LOCKED_1",
    )
    evidence_ledger = list(state.get("evidence_ledger", [])) + [verified_item]
    hypothesis_history = list(state.get("hypothesis_history", [])) + [HypothesisStatusChange(
        hypothesis_id=hyp.id, previous_status="POSSIBLE", new_status="SUPPORTED",
        changed_by_evidence_ids=["EV_LOCKED_1"], reason="objective (VERIFIED) evidence supports this hypothesis",
    )]
    state = {**state, "evidence_ledger": evidence_ledger, "hypothesis_history": hypothesis_history}
    return state, hyp.id


def test_locked_supported_hypothesis_with_causal_language_survives_final_verification():
    """The exact reproduced defect: a status_locked, VERIFIED-evidence-
    SUPPORTED hypothesis whose statement contains a causal connective
    ('because') must NOT be removed by
    hypothesis_statement_asserts_unsupported_causation or any sibling
    prose-pattern check."""
    causal_statement = "The required step was omitted because the workflow configuration did not include it in the sequence."
    state, hyp_id = asyncio.run(_build_state_with_locked_hypothesis(
        _FOUR_EMPLOYEES_FINDING,
        causal_statement,
    ))
    final_state = asyncio.run(final_evidence_verification_node(state))
    rc = final_state.get("root_cause")
    surviving_ids = {h.id for h in rc.candidate_hypotheses}
    assert hyp_id in surviving_ids, (
        f"locked, evidence-SUPPORTED hypothesis {hyp_id} was removed by final_evidence_verification_node "
        f"despite status_locked=True; surviving hypotheses: {surviving_ids}"
    )
    surviving = next(h for h in rc.candidate_hypotheses if h.id == hyp_id)
    assert surviving.status == "SUPPORTED"
    assert surviving.status_locked is True


def test_locked_hypothesis_still_flows_through_downstream_processing():
    """The fix must not simply short-circuit the whole loop body for
    locked hypotheses -- confirms it still participates in normal
    processing (e.g. remains a single, non-duplicated entry) rather than
    being appended twice or bypassing structural bookkeeping."""
    causal_statement = "The required step was omitted because the workflow configuration did not include it in the sequence."
    state, hyp_id = asyncio.run(_build_state_with_locked_hypothesis(
        _FOUR_EMPLOYEES_FINDING,
        causal_statement,
    ))
    final_state = asyncio.run(final_evidence_verification_node(state))
    rc = final_state.get("root_cause")
    matching = [h for h in rc.candidate_hypotheses if h.id == hyp_id]
    assert len(matching) == 1, f"expected exactly one surviving entry for {hyp_id}, got {len(matching)}"


def test_invariants_still_pass_with_locked_hypothesis_preserved():
    causal_statement = "The required step was omitted because the workflow configuration did not include it in the sequence."
    state, hyp_id = asyncio.run(_build_state_with_locked_hypothesis(
        _FOUR_EMPLOYEES_FINDING,
        causal_statement,
    ))
    final_state = asyncio.run(final_evidence_verification_node(state))
    is_valid, violations = evaluate_all_invariants(final_state)
    assert not any("INV-INVEST-028" in v for v in violations), violations
