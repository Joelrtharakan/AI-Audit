"""Phase 6 -- unified, immutable report-level AI provenance.

`report.provenance` consolidates the scattered per-field provenance, represents
stages that did not run explicitly (never fabricated), is immutable, and
survives serialization to the FastAPI response JSON.
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
from app.agent.provenance import derive_analysis_provenance
from app.models.agent import AnalysisProvenance, InvestigateRequest, InvestigationReport


async def _run(text: str):
    state = {
        "request": InvestigateRequest(finding_text=text),
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
    return state, state["report"]


@pytest.mark.asyncio
async def test_provenance_present_and_reflects_deterministic_offline_run():
    _state, report = await _run("Three employees failed to complete the revised inspection checklist.")
    p = report.provenance
    assert isinstance(p, AnalysisProvenance)
    assert p.execution_mode in ("DETERMINISTIC", "DEGRADED", "LLM")
    assert p.review_required is True
    assert p.generated_at.endswith("+00:00")
    # offline: no semantic LLM ran -> stages explicitly not attempted, no fake model
    assert p.canonical.attempted is False and p.canonical.model is None
    assert p.synthesis.attempted is False and p.synthesis.model is None
    assert "HUMAN_REVIEWED_PRODUCTION" in p.reason_codes


@pytest.mark.asyncio
async def test_provenance_is_immutable():
    _state, report = await _run("Three employees failed to complete the revised inspection checklist.")
    with pytest.raises(Exception):
        report.provenance.provider = "something-else"
    with pytest.raises(Exception):
        report.provenance.canonical.attempted = True


@pytest.mark.asyncio
async def test_provenance_does_not_fabricate_a_stage_that_did_not_run():
    _state, report = await _run("A single checkout terminal must be replaced.")
    p = report.provenance
    # a stage that was not attempted must not carry a model/prompt id
    for stage in (p.canonical, p.synthesis, p.remediation):
        if not stage.attempted:
            assert stage.model is None
            assert stage.prompt_version is None


@pytest.mark.asyncio
async def test_provenance_survives_api_serialization_round_trip():
    _state, report = await _run("Three employees failed to complete the revised inspection checklist.")
    raw = json.loads(report.model_dump_json())
    assert "provenance" in raw
    prov = raw["provenance"]
    for key in ("provider", "execution_mode", "semantic_mode", "fallback_used",
                "review_required", "generated_at", "canonical", "synthesis", "remediation"):
        assert key in prov
    back = InvestigationReport.model_validate(raw)
    assert back.provenance.generated_at == report.provenance.generated_at
    assert back.provenance.reason_codes == report.provenance.reason_codes
    assert "provenance" in InvestigationReport.model_json_schema()["properties"]


@pytest.mark.asyncio
async def test_provenance_not_re_resolved_after_config_change(monkeypatch):
    from app.config import get_settings

    _state, report = await _run("Three employees failed to complete the revised inspection checklist.")
    stamped = report.provenance.model_dump()

    monkeypatch.setattr(get_settings(), "llm_provider", "some-other-provider")
    monkeypatch.setattr(get_settings(), "analysis_prompt_version", "9.9")
    # serializing/reading the already-generated report must not change provenance
    assert json.loads(report.model_dump_json())["provenance"] == stamped


@pytest.mark.asyncio
async def test_legacy_provenance_fields_retained_for_backward_compat():
    _state, report = await _run("Three employees failed to complete the revised inspection checklist.")
    raw = json.loads(report.model_dump_json())
    for legacy in ("analysis_mode", "analysis_engine", "provider_used",
                   "fallback_used", "provider_attempts", "critic_status",
                   "human_review_required"):
        assert legacy in raw
