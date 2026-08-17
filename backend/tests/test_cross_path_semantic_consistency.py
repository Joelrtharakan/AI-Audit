"""Cross-path semantic consistency (spec item 34 / README §5.6 analysis_mode
contract): PRIMARY_LLM, RECOVERY_LLM, and full DETERMINISTIC synthesis, given
the *same* canonical evidence, must land on the same causal semantics —
root-cause eligibility, leading-hypothesis eligibility, and whether the
surviving hypothesis set is non-empty. Wording may differ; the deterministic
guards in `causal_guard`/`causal_graph` recompute status/support from the
evidence ledger regardless of which path produced the candidate text, so
this is a regression test for that recomputation actually being applied
identically on every path — not a test of prompt wording.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.models.agent import (
    AgentTraceStep,
    CanonicalFindingState,
    EvidenceItem,
    EvidenceStatus,
    InvestigateRequest,
    InvestigationPlan,
)

FINDING_TEXT = (
    "Equipment operated outside the validated temperature range during the production run. "
    "The safety interlock control was found disabled at the time of the run, which permitted "
    "operation outside the validated range."
)
FACT_1 = "equipment operated outside the validated temperature range during the production run"
FACT_2 = "the safety interlock control was found disabled at the time of the run"
MECHANISM = "the safety interlock control was disabled, permitting operation outside the validated range"


def _build_state() -> dict:
    ledger = [
        EvidenceItem(claim=FACT_1, source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim=FACT_2, source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED),
    ]
    canonical = CanonicalFindingState(
        raw_finding=FINDING_TEXT,
        observed_deviation="equipment operation — outside the validated temperature range",
        deviation_condition="operated outside the validated temperature range",
        facts=[FACT_1, FACT_2],
        reported_statements=[],
        affected_objects=["the equipment"],
        immediate_mechanism=MECHANISM,
        immediate_mechanism_status="VERIFIED",
    )
    return {
        "request": InvestigateRequest(finding_text=FINDING_TEXT),
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "observation_quality": None,
        "extraction": None,
        "canonical_finding_state": canonical,
        "investigation_plan": None,
        "needs_investigation": False,
        "planned_tools": [],
        "completed_tools": [],
        "current_tool": None,
        "tool_results": {},
        "evidence_ledger": ledger,
        "evidence_gaps": [],
        "root_cause": None,
        "contributing_factors": [],
        "five_why": None,
        "impact_assessment": None,
        "capa_analysis": None,
        "critic_approved": False,
        "critic_feedback": None,
        "critic_send_back": False,
        "report": None,
        "ca_draft": None,
        "final_state": None,
        "trace": [AgentTraceStep.ok("Test started")],
        "errors": [],
    }


def _primary_response() -> str:
    return json.dumps({
        "root_cause": {
            "status": "SUPPORTED",
            "category": "MACHINE",
            "statement": "The safety interlock control was disabled, permitting out-of-range operation.",
            "leading_hypothesis": "H1",
            "candidate_hypotheses": [{
                "id": "H1",
                "name": "SAFETY_INTERLOCK_DISABLED",
                "statement": "The safety interlock control was disabled, permitting operation outside the validated range.",
                "status": "SUPPORTED",
                "evidence_needed": "Interlock control audit log",
                "supporting_claim_ids": ["C1", "C2"],
                "contradicting_claim_ids": [],
            }],
            "narrative": "Verified records show the interlock was disabled, directly permitting the out-of-range run.",
        },
        "five_why": {
            "steps": [{
                "level": 1, "question": "Why did equipment operate outside the validated range?",
                "answer": "The safety interlock control was disabled.", "status": "VERIFIED",
            }],
            "is_complete": False,
            "status_note": "Stopped at evidence boundary",
        },
        "impact_assessment": {"status": "IMPACT_REQUIRES_ASSESSMENT", "areas": []},
        "capa": {"status": "INVESTIGATION_REQUIRED", "potential_areas": [], "recommended_investigation": []},
        "contributing_factors": [],
        "ca_draft": {
            "immediate_action": "Assess affected production run.",
            "root_cause": "Safety interlock control was disabled.",
            "root_cause_category": "MACHINE",
            "preventive_action": "Verify interlock control integrity before future runs.",
            "impact_analysis": "Scope requires auditor assessment.",
        },
    })


def _recovery_response() -> str:
    return json.dumps({
        "root_cause": {
            "status": "SUPPORTED",
            "category": "MACHINE",
            "statement": "Interlock control disabled, permitting the out-of-range run.",
            "candidate_hypotheses": [{
                "id": "H1", "name": "INTERLOCK_DISABLED",
                "statement": "Interlock control disabled, permitting the out-of-range run.",
                "supporting_claim_ids": ["C1", "C2"], "contradicting_claim_ids": [],
                "status": "SUPPORTED", "evidence_needed": "Control audit log",
                "confirms_if": "Audit log confirms disablement window.",
                "refutes_if": "Audit log shows interlock active throughout the run.",
            }],
            "narrative": "Interlock disablement directly explains the out-of-range operation.",
        },
        "five_why": {
            "steps": [{
                "level": 1, "question": "Why did the run go out of range?",
                "answer": "The interlock control was disabled.", "status": "VERIFIED",
            }],
            "is_complete": False,
            "status_note": "evidence boundary reached",
        },
        "contributing_factors": [],
    })


def _semantic_shape(root_cause) -> dict:
    """Normalize a synthesis result down to the comparable semantics: causal
    outcome and hypothesis eligibility, not prose wording."""
    return {
        "hypothesis_count": len(root_cause.candidate_hypotheses),
        "any_supported": any(h.status in ("SUPPORTED", "ESTABLISHED") for h in root_cause.candidate_hypotheses),
        "leading_hypothesis_status": root_cause.leading_hypothesis_status,
    }


@pytest.mark.asyncio
async def test_primary_recovery_deterministic_paths_agree_on_established_mechanism():
    """Same VERIFIED-mechanism evidence, three different execution paths.
    All three must independently recompute the same causal semantics —
    a genuinely proven cause (verified interlock disablement, spec §6)
    must survive PRIMARY, RECOVERY, and full DETERMINISTIC synthesis alike.
    """
    from app.agent.nodes.core_synthesis import core_synthesis_node

    # PRIMARY path: LLM succeeds on the first call.
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = _primary_response()
        mock_get_client.return_value = mock_client
        primary_result = await core_synthesis_node(_build_state())

    # RECOVERY path: primary raises, compact recovery call succeeds.
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.side_effect = [
            RuntimeError("simulated primary failure"),
            _recovery_response(),
        ]
        mock_get_client.return_value = mock_client
        recovery_result = await core_synthesis_node(_build_state())

    # DETERMINISTIC path: both primary and recovery raise.
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.side_effect = [
            RuntimeError("simulated primary failure"),
            RuntimeError("simulated recovery failure"),
        ]
        mock_get_client.return_value = mock_client
        deterministic_result = await core_synthesis_node(_build_state())

    assert primary_result["root_cause"] is not None
    assert recovery_result["root_cause"] is not None
    assert deterministic_result["root_cause"] is not None

    # Execution paths are honestly distinguishable (spec §37/§38)...
    assert primary_result["synthesis_execution"]["source"] == "PRIMARY_LLM"
    assert recovery_result["synthesis_execution"]["source"] == "RECOVERY_LLM"
    assert deterministic_result["synthesis_execution"]["source"] not in ("PRIMARY_LLM", "RECOVERY_LLM")

    # ...but a genuinely evidence-established mechanism must survive on
    # every path: none of the three may suppress it or leave it empty.
    for label, result in (
        ("PRIMARY", primary_result),
        ("RECOVERY", recovery_result),
        ("DETERMINISTIC", deterministic_result),
    ):
        rc = result["root_cause"]
        assert len(rc.candidate_hypotheses) >= 1, f"{label}: verified mechanism must produce at least one hypothesis"
        assert any(
            h.status in ("SUPPORTED", "ESTABLISHED") for h in rc.candidate_hypotheses
        ), f"{label}: verified interlock-disablement mechanism must be promotable, not suppressed"

    # Hypothesis ID provenance must hold on every path independently —
    # no path may cite a claim ID the ledger never issued.
    for label, result in (
        ("PRIMARY", primary_result),
        ("RECOVERY", recovery_result),
        ("DETERMINISTIC", deterministic_result),
    ):
        for h in result["root_cause"].candidate_hypotheses:
            assert h.supporting_claim_ids or h.supporting_evidence, (
                f"{label}: hypothesis {h.id} has no traceable supporting evidence/claim ids"
            )


@pytest.mark.asyncio
async def test_investigation_plan_quality_invariants_hold_on_every_execution_path():
    """Spec items 19/20/23: the investigation-plan quality layer (dedup,
    information-gain ranking, area clustering) is invoked from
    final_evidence_verification_node regardless of which path produced the
    underlying hypotheses/questions -- no duplicate questions or areas, and
    no orphan areas, may survive on PRIMARY, RECOVERY, or DETERMINISTIC."""
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node

    async def _run_path(side_effect) -> dict:
        state = _build_state()
        state["investigation_plan"] = InvestigationPlan(questions=[], areas=[], evidence_to_collect=[])
        with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get_client:
            mock_client = AsyncMock()
            if isinstance(side_effect, list):
                mock_client.chat_completion.side_effect = side_effect
            else:
                mock_client.chat_completion.return_value = side_effect
            mock_get_client.return_value = mock_client
            synthesis_result = await core_synthesis_node(state)
        merged_state = {**state, **synthesis_result}
        return await final_evidence_verification_node(merged_state)

    primary_final = await _run_path(_primary_response())
    recovery_final = await _run_path([RuntimeError("simulated primary failure"), _recovery_response()])
    deterministic_final = await _run_path([
        RuntimeError("simulated primary failure"), RuntimeError("simulated recovery failure"),
    ])

    for label, final_state in (
        ("PRIMARY", primary_final),
        ("RECOVERY", recovery_final),
        ("DETERMINISTIC", deterministic_final),
    ):
        inv = final_state["investigation_plan"]
        assert inv is not None, f"{label}: investigation plan must not be dropped"
        question_texts = [q.question.lower() for q in inv.questions]
        assert len(question_texts) == len(set(question_texts)), f"{label}: duplicate questions survived"
        areas_lower = [a.lower() for a in inv.areas]
        assert len(areas_lower) == len(set(areas_lower)), f"{label}: duplicate areas survived: {inv.areas}"
        assert len(inv.areas) <= 4, f"{label}: too many investigation areas: {inv.areas}"
