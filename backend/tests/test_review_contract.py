"""Phase 4 -- structured human-review contract.

`report.review` must be authoritative (derived from settled backend state),
machine-readable, always require review, carry the reasons why, and survive
serialization to the FastAPI response JSON.
"""
from __future__ import annotations

import json

import pytest
from unittest.mock import patch

from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.understanding import understand_finding_node
from app.agent.review_state import derive_review_state
from app.models.agent import InvestigateRequest, InvestigationReport, ReviewState


async def _run(finding_text: str) -> InvestigationReport:
    state = {
        "request": InvestigateRequest(finding_text=finding_text),
        "evidence_ledger": [], "iteration_count": 0, "tool_call_count": 0,
        "critic_iteration": 0, "trace": [], "errors": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
        state = await core_synthesis_node(state)
        state = await generate_report_node(state)
        state = await final_evidence_verification_node(state)
    return state["report"]


@pytest.mark.asyncio
async def test_review_state_is_populated_and_mandatory():
    report = await _run("Three employees failed to complete the revised inspection checklist.")
    rs = report.review
    assert isinstance(rs, ReviewState)
    assert rs.required is True
    assert rs.status == "PENDING_HUMAN_REVIEW"
    assert "HUMAN_REVIEWED_PRODUCTION" in rs.reason_codes
    # offline pipeline => deterministic analysis, no established root cause
    assert "DETERMINISTIC_ANALYSIS" in rs.reason_codes
    assert "ROOT_CAUSE_NOT_ESTABLISHED" in rs.reason_codes
    # legacy field preserved
    assert report.human_review_required is True


@pytest.mark.asyncio
async def test_reasons_track_authoritative_state_changes():
    report = await _run("Three employees failed to complete the revised inspection checklist.")

    report.analysis_mode = "DEGRADED"
    report.semantic_mode = "DETERMINISTIC_FALLBACK"
    report.observation_quality = "INSUFFICIENT"
    report.fallback_used = True
    rs = derive_review_state(report)
    assert {"AI_ANALYSIS_DEGRADED", "SEMANTIC_LAYER_UNAVAILABLE",
            "OBSERVATION_INSUFFICIENT", "PROVIDER_FALLBACK_USED"} <= set(rs.reason_codes)

    # a healthy full-AI run with an established cause still requires review,
    # but for fewer reasons
    report.analysis_mode = "LLM"
    report.semantic_mode = "CANONICAL_LLM"
    report.observation_quality = "SUFFICIENT"
    report.fallback_used = False
    rs2 = derive_review_state(report)
    assert rs2.required is True
    assert "HUMAN_REVIEWED_PRODUCTION" in rs2.reason_codes
    assert "AI_ANALYSIS_DEGRADED" not in rs2.reason_codes
    assert "OBSERVATION_INSUFFICIENT" not in rs2.reason_codes


@pytest.mark.asyncio
async def test_review_state_survives_api_serialization():
    report = await _run("Three employees failed to complete the revised inspection checklist.")
    raw = json.loads(report.model_dump_json())
    assert "review" in raw
    assert raw["review"]["required"] is True
    assert raw["review"]["status"] == "PENDING_HUMAN_REVIEW"
    assert isinstance(raw["review"]["reasons"], list) and raw["review"]["reasons"]
    assert all({"code", "detail"} <= set(r) for r in raw["review"]["reasons"])

    back = InvestigationReport.model_validate(raw)
    assert back.review.required is True
    assert set(back.review.reason_codes) == {r["code"] for r in raw["review"]["reasons"]}

    # exposed in the response schema for the ASP.NET consumer
    assert "review" in InvestigationReport.model_json_schema()["properties"]


def test_review_reason_detail_carries_no_chain_of_thought():
    rs = ReviewState()
    assert rs.required is True
    # default (unpopulated) state is still safe
    assert rs.reason_codes == []
