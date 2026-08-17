"""Tests for the rebuilt LLM synthesis boundary (core_synthesis_node): the
strict claim-id provenance contract, the hypothesis count cap, timeout/
recovery/fallback convergence, analysis_mode truthfulness, and the
root-cause status evidence-justification firewall. Mocks the LLM client
directly (never a real Ollama call) so every failure mode is deterministic
and fast.
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
)
from app.services import llm_metrics
from app.services.llm_client import LLMTimeoutError


def _build_state() -> dict:
    """Training-conflict-shaped state: C1/C2/C3 in that order, matching
    the claim ids core_synthesis_node will assign (_assign_claim_ids)."""
    finding_text = (
        "The operator stated that they had not received training on the revised procedure. "
        "The department supervisor stated that the operator completed the required training "
        "before the procedure became effective. "
        "No training attendance record was available during the audit."
    )
    ledger = [
        EvidenceItem(
            claim="The operator stated that they had not received training on the revised procedure.",
            source="REPORTED_STATEMENT", status=EvidenceStatus.REPORTED,
        ),
        EvidenceItem(
            claim="The department supervisor stated that the operator completed the required "
            "training before the procedure became effective.",
            source="REPORTED_STATEMENT", status=EvidenceStatus.REPORTED,
        ),
        EvidenceItem(
            claim="No training attendance record was available during the audit.",
            source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED,
        ),
    ]
    canonical = CanonicalFindingState(
        raw_finding=finding_text,
        observed_deviation="training completion disputed for the revised procedure",
        finding_subject="training for the revised procedure",
        deviation_condition="disputed",
        facts=["No training attendance record was available during the audit."],
        reported_statements=[ledger[0].claim, ledger[1].claim],
        affected_objects=["training record"],
    )
    return {
        "request": InvestigateRequest(finding_text=finding_text),
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
        "observation_quality": None, "extraction": None,
        "canonical_finding_state": canonical, "investigation_plan": None,
        "needs_investigation": False, "planned_tools": [], "completed_tools": [],
        "current_tool": None, "tool_results": {}, "evidence_ledger": ledger,
        "evidence_gaps": [], "root_cause": None, "contributing_factors": [],
        "five_why": None, "impact_assessment": None, "capa_analysis": None,
        "critic_approved": False, "critic_feedback": None, "critic_send_back": False,
        "report": None, "ca_draft": None, "final_state": None,
        "trace": [AgentTraceStep.ok("Test started")], "errors": [],
    }


def _valid_llm_json(hyp_count: int = 1) -> str:
    hyps = []
    templates = [
        {"id": "H1", "name": "TRAINING_NOT_COMPLETED",
         "statement": "Required training may not have been completed before the revised procedure became effective.",
         "supporting_claim_ids": ["C1"], "contradicting_claim_ids": ["C2"],
         "status": "POSSIBLE", "evidence_needed": "Authenticated training completion record",
         "confirms_if": "No completion record exists.", "refutes_if": "A completion record confirms timely completion."},
        {"id": "H2", "name": "TRAINING_RECORD_UNAVAILABLE",
         "statement": "The training record may exist but was not available for verification during the audit.",
         "supporting_claim_ids": ["C3"], "contradicting_claim_ids": [],
         "status": "POSSIBLE", "evidence_needed": "Authenticated training record",
         "confirms_if": "A record is located.", "refutes_if": "No record exists anywhere."},
        {"id": "H3", "name": "TRAINING_SCHEDULE_GAP",
         "statement": "Training scheduling for the revised procedure may have a gap.",
         "supporting_claim_ids": ["C3"], "contradicting_claim_ids": [],
         "status": "POSSIBLE", "evidence_needed": "Training schedule",
         "confirms_if": "Schedule shows a gap.", "refutes_if": "Schedule confirms coverage."},
        {"id": "H4", "name": "SUPERVISOR_STATEMENT_UNRELIABLE",
         "statement": "The supervisor's statement may be inaccurate.",
         "supporting_claim_ids": ["C1"], "contradicting_claim_ids": [],
         "status": "POSSIBLE", "evidence_needed": "n/a",
         "confirms_if": "n/a", "refutes_if": "n/a"},
        {"id": "H5", "name": "AUTHORIZATION_GAP",
         "statement": "Authorization to perform the revised procedure without training may have been granted in error.",
         "supporting_claim_ids": ["C1"], "contradicting_claim_ids": [],
         "status": "POSSIBLE", "evidence_needed": "Authorization records",
         "confirms_if": "n/a", "refutes_if": "n/a"},
    ]
    hyps = templates[:hyp_count]
    return json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED", "category": "TO_BE_CONFIRMED", "statement": None,
            "candidate_hypotheses": hyps,
            "narrative": "Conflicting reports exist regarding training completion.",
        },
        "five_why": {
            "steps": [
                {"question": "Why was training completion disputed?", "answer": "Reports conflict on completion.", "status": "MIXED"},
            ],
            "is_complete": False, "status_note": "Stopped at evidence boundary",
        },
        "contributing_factors": [],
    })


def _mock_client(*, side_effect=None, return_value=None):
    mock_client = AsyncMock()
    if side_effect is not None:
        mock_client.chat_completion.side_effect = side_effect
    else:
        mock_client.chat_completion.return_value = return_value
    return mock_client


@pytest.fixture(autouse=True)
def _reset_metrics():
    llm_metrics.reset()
    yield
    llm_metrics.reset()


# ---------------------------------------------------------------------------
# TEST 1: valid structured response -> LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_structured_response_yields_llm_mode():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(return_value=_valid_llm_json(2))
        result = await core_synthesis_node(state)
    assert result["analysis_mode"] == "LLM"
    assert len(result["root_cause"].candidate_hypotheses) == 2
    snap = llm_metrics.snapshot()
    assert snap["llm_primary_attempted"] == 1
    assert snap["llm_primary_success"] == 1


# ---------------------------------------------------------------------------
# TEST 2: malformed JSON -> recovery or fallback, never crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_malformed_json_never_crashes():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(side_effect=["{not valid json", "{also not valid"])
        result = await core_synthesis_node(state)
    assert result["analysis_mode"] == "DETERMINISTIC"
    assert result["root_cause"] is not None


# ---------------------------------------------------------------------------
# TEST 3: empty response -> recovery/fallback, never crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_response_never_crashes():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(side_effect=["", ""])
        result = await core_synthesis_node(state)
    assert result["root_cause"] is not None
    assert result["analysis_mode"] in ("DETERMINISTIC", "LLM")


# ---------------------------------------------------------------------------
# TEST 4: invalid enum value -> normalization, never crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_enum_value_normalizes_without_crashing():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    bad_json = json.dumps({
        "root_cause": {
            "status": "TOTALLY_MADE_UP_STATUS", "category": "TO_BE_CONFIRMED", "statement": None,
            "candidate_hypotheses": [{
                "id": "H1", "name": "X", "statement": "Training may not have been completed.",
                "status": "ALSO_MADE_UP", "supporting_claim_ids": ["C1"], "contradicting_claim_ids": [],
                "evidence_needed": "record",
            }],
            "narrative": "x",
        },
        "five_why": {"steps": [], "is_complete": False, "status_note": "x"},
        "contributing_factors": [],
    })
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(return_value=bad_json)
        result = await core_synthesis_node(state)
    assert result["root_cause"].status is not None
    for h in result["root_cause"].candidate_hypotheses:
        assert h.status in ("POSSIBLE", "SUPPORTED", "REFUTED", "UNRESOLVED", "UNVERIFIED")


# ---------------------------------------------------------------------------
# TEST 5: unknown claim ID -> hypothesis rejected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hypothesis_citing_nonexistent_claim_id_is_rejected():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    bad_json = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED", "category": "TO_BE_CONFIRMED", "statement": None,
            "candidate_hypotheses": [{
                "id": "H1", "name": "INVENTED", "statement": "A mechanism the ledger never mentioned.",
                "status": "POSSIBLE", "supporting_claim_ids": ["C99"], "contradicting_claim_ids": [],
                "evidence_needed": "record",
            }],
            "narrative": "x",
        },
        "five_why": {"steps": [], "is_complete": False, "status_note": "x"},
        "contributing_factors": [],
    })
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(return_value=bad_json)
        result = await core_synthesis_node(state)
    names = {h.name for h in result["root_cause"].candidate_hypotheses}
    assert "INVENTED" not in names
    assert llm_metrics.aggregated()["provenance_rejections"] >= 1


# ---------------------------------------------------------------------------
# TEST 6: zero provenance -> hypothesis rejected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hypothesis_with_zero_provenance_is_rejected():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    bad_json = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED", "category": "TO_BE_CONFIRMED", "statement": None,
            "candidate_hypotheses": [{
                "id": "H1", "name": "UNGROUNDED", "statement": "A plausible-sounding but uncited mechanism.",
                "status": "POSSIBLE", "supporting_claim_ids": [], "contradicting_claim_ids": [],
                "evidence_needed": "record",
            }],
            "narrative": "x",
        },
        "five_why": {"steps": [], "is_complete": False, "status_note": "x"},
        "contributing_factors": [],
    })
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(return_value=bad_json)
        result = await core_synthesis_node(state)
    names = {h.name for h in result["root_cause"].candidate_hypotheses}
    assert "UNGROUNDED" not in names
    assert llm_metrics.aggregated()["provenance_rejections"] >= 1


# ---------------------------------------------------------------------------
# TEST 7: LLM says CONFIRMED without evidence -> cannot become CONFIRMED
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_confirmed_status_without_verified_evidence_is_downgraded():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    state = _build_state()
    bad_json = json.dumps({
        "root_cause": {
            "status": "CONFIRMED", "category": "TRAINING", "statement": "Training was not completed.",
            "candidate_hypotheses": [{
                "id": "H1", "name": "TRAINING_NOT_COMPLETED",
                "statement": "Required training was not completed.",
                "status": "SUPPORTED", "supporting_claim_ids": ["C1"], "contradicting_claim_ids": ["C2"],
                "evidence_needed": "record",
            }],
            "narrative": "x",
        },
        "five_why": {"steps": [], "is_complete": False, "status_note": "x"},
        "contributing_factors": [],
    })
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(return_value=bad_json)
        synthesis_result = await core_synthesis_node(state)

    fv_state = {
        **synthesis_result,
        "canonical_finding_state": state["canonical_finding_state"],
    }
    final_result = await final_evidence_verification_node(fv_state)
    rc = final_result["root_cause"]
    assert str(rc.status) not in ("RootCauseStatus.VERIFIED", "VERIFIED")
    assert rc.status in ("NOT_ESTABLISHED",) or str(rc.status) == "RootCauseStatus.NOT_ESTABLISHED"


# ---------------------------------------------------------------------------
# TEST 8: LLM invents an unsupported-domain mechanism -> firewall removes it
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_invented_unsupported_domain_mechanism_is_removed():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    bad_json = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED", "category": "TO_BE_CONFIRMED", "statement": None,
            "candidate_hypotheses": [{
                "id": "H1", "name": "SUPPLIER_QUALIFICATION_GAP",
                "statement": "A supplier qualification failure may have caused this deviation.",
                "status": "POSSIBLE", "supporting_claim_ids": ["C1"], "contradicting_claim_ids": [],
                "evidence_needed": "supplier qualification record",
            }],
            "narrative": "x",
        },
        "five_why": {"steps": [], "is_complete": False, "status_note": "x"},
        "contributing_factors": [],
    })
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(return_value=bad_json)
        result = await core_synthesis_node(state)
    names = {h.name for h in result["root_cause"].candidate_hypotheses}
    assert "SUPPLIER_QUALIFICATION_GAP" not in names


# ---------------------------------------------------------------------------
# TEST 9: LLM proposes more than the cap -> only the cap survives
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_excess_hypotheses_are_capped():
    from app.agent.nodes.core_synthesis import core_synthesis_node, _MAX_LLM_HYPOTHESES
    state = _build_state()
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(return_value=_valid_llm_json(5))
        result = await core_synthesis_node(state)
    assert len(result["root_cause"].candidate_hypotheses) <= _MAX_LLM_HYPOTHESES


# ---------------------------------------------------------------------------
# TEST 10: LLM proposes CAPA content -> ignored; CAPA is deterministic-only
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_capa_content_is_never_used():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    raw = json.loads(_valid_llm_json(1))
    raw["capa"] = {
        "conditional_actions": [{"recommended_action": "LLM-INVENTED UNCONDITIONAL ACTION TEXT"}]
    }
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(return_value=json.dumps(raw))
        result = await core_synthesis_node(state)
    capa = result["capa_analysis"]
    for a in capa.conditional_actions:
        assert "LLM-INVENTED UNCONDITIONAL ACTION TEXT" not in a.recommended_action
        assert "IF" in a.if_cause_confirmed.upper()


# ---------------------------------------------------------------------------
# TEST 11: primary timeout -> recovery attempted and can succeed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_primary_timeout_triggers_recovery_attempt():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(side_effect=[LLMTimeoutError("timed out"), _valid_llm_json(1)])
        result = await core_synthesis_node(state)
    snap = llm_metrics.snapshot()
    assert snap["llm_primary_timeout"] == 1
    assert snap["llm_recovery_attempted"] == 1
    assert snap["llm_recovery_success"] == 1
    assert result["analysis_mode"] == "LLM"


# ---------------------------------------------------------------------------
# TEST 12: primary + recovery timeout -> DETERMINISTIC
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_primary_and_recovery_timeout_yields_deterministic():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(
            side_effect=[LLMTimeoutError("timed out"), LLMTimeoutError("timed out again")]
        )
        result = await core_synthesis_node(state)
    assert result["analysis_mode"] == "DETERMINISTIC"
    snap = llm_metrics.snapshot()
    assert snap["llm_primary_timeout"] == 1
    assert snap["llm_recovery_timeout"] == 1
    assert snap["deterministic_fallback"] == 1
    assert result["root_cause"].status == "NOT_ESTABLISHED" or str(result["root_cause"].status) == "RootCauseStatus.NOT_ESTABLISHED"


# ---------------------------------------------------------------------------
# TEST 13: primary invalid, recovery valid -> LLM (current semantics)
#
# NOTE: the spec asks this be "HYBRID or LLM according to the defined
# semantics" -- a distinct HYBRID analysis_mode was NOT implemented this
# turn (see final report: adding a new Literal value to analysis_mode
# requires matching frontend branch handling too, or an unhandled value
# silently falls into the "else" branch and would incorrectly render as
# "DEEP LLM SYNTHESIS ACTIVE" for a hybrid case -- worse than the current
# accurate binary). Current, verified semantics: recovery success keeps
# analysis_mode == "LLM" (still a genuine LLM analysis, just via the
# smaller recovery schema).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_primary_invalid_recovery_valid_yields_llm_mode():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(side_effect=["{not valid json", _valid_llm_json(1)])
        result = await core_synthesis_node(state)
    assert result["analysis_mode"] == "LLM"
    assert len(result["root_cause"].candidate_hypotheses) >= 1


# ---------------------------------------------------------------------------
# TEST 14: primary valid but requires deterministic repair -> LLM mode,
# repair visible in trace (see note on TEST 13 -- no distinct HYBRID mode).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_primary_valid_with_repair_stays_llm_mode_and_repairs_are_traced():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    raw = json.loads(_valid_llm_json(1))
    # Force a repair: cite a nonexistent claim id on a SECOND hypothesis
    # alongside a valid one, so one survives untouched and one is repaired
    # (dropped) -- analysis_mode must still faithfully read "LLM" (the
    # primary call itself succeeded; a hypothesis-level repair is not a
    # provider failure).
    raw["root_cause"]["candidate_hypotheses"].append({
        "id": "H2", "name": "BAD_PROVENANCE", "statement": "An uncited claim.",
        "status": "POSSIBLE", "supporting_claim_ids": ["C77"], "contradicting_claim_ids": [],
        "evidence_needed": "record",
    })
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(return_value=json.dumps(raw))
        result = await core_synthesis_node(state)
    assert result["analysis_mode"] == "LLM"
    names = {h.name for h in result["root_cause"].candidate_hypotheses}
    assert "BAD_PROVENANCE" not in names
    assert any("cited claim id" in step.message for step in result["trace"] if hasattr(step, "message"))
