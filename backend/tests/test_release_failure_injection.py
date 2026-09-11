"""Phase 2 release verification -- failure-injection + provenance/review integrity.

For every failure state the backend can reach, verify:
  * analysis_mode / semantic_mode are honest
  * review is always required, with the right machine-readable reason codes
  * provenance never fabricates a model / provider / prompt version
  * an unexecuted stage is attempted=False (not stamped with a fake model)
  * generated_at is explicit UTC
  * provenance does not change when runtime config changes afterwards
  * degraded / transient-failure results are excluded from the cache
"""
from __future__ import annotations

import json

import pytest
from unittest.mock import patch

from app.agent import cache as cache_mod
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.understanding import understand_finding_node
from app.agent.provenance import derive_analysis_provenance
from app.agent.review_state import derive_review_state
from app.models.agent import InvestigateRequest, InvestigationReport
from app.remediation.engine import honest_not_assessable


_FINDING = "A quantity of units did not meet a stated requirement during a period."


async def _run_offline():
    state = {
        "request": InvestigateRequest(finding_text=_FINDING),
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


def _assert_no_fabrication(prov):
    for stage in (prov.canonical, prov.synthesis, prov.remediation):
        if not stage.attempted:
            assert stage.model is None, stage
            assert stage.prompt_version is None, stage
    assert prov.generated_at.endswith("+00:00")
    assert prov.review_required is True


@pytest.mark.asyncio
async def test_offline_deterministic_path_is_honest():
    state, report = await _run_offline()
    assert report.analysis_mode in ("DETERMINISTIC", "DEGRADED")
    assert report.semantic_mode in ("DETERMINISTIC", "DETERMINISTIC_FALLBACK")
    assert report.human_review_required is True
    assert report.review.required is True
    _assert_no_fabrication(report.provenance)
    # nothing claims a clean LLM synthesis happened
    assert report.provenance.synthesis.attempted is False


@pytest.mark.asyncio
@pytest.mark.parametrize("sem_status, machine_reason, expect_code", [
    ("LLM_TIMEOUT", "MODEL_TIMEOUT", "REMEDIATION_MODEL_FAILURE"),
    ("LLM_UNAVAILABLE", "MODEL_UNAVAILABLE", "REMEDIATION_MODEL_FAILURE"),
    ("LLM_INVALID", "MODEL_OUTPUT_INVALID", "REMEDIATION_MODEL_FAILURE"),
])
async def test_remediation_failure_surfaces_in_review_and_provenance(sem_status, machine_reason, expect_code):
    _state, report = await _run_offline()
    report.remediation_cost = honest_not_assessable(sem_status, machine_reason)

    rs = derive_review_state(report)
    assert rs.required is True
    assert expect_code in rs.reason_codes

    prov = derive_analysis_provenance(report, _state)
    # remediation provenance echoes the real status, never a fabricated success
    assert prov.remediation.status == sem_status
    assert "OK" != prov.remediation.status
    _assert_no_fabrication(prov)


@pytest.mark.asyncio
async def test_degraded_and_fallback_states_force_review_reasons():
    _state, report = await _run_offline()

    report.analysis_mode = "DEGRADED"
    report.semantic_mode = "DETERMINISTIC_FALLBACK"
    report.canonical_semantic_status = "PROVIDER_ERROR"
    rs = derive_review_state(report)
    assert {"AI_ANALYSIS_DEGRADED", "SEMANTIC_LAYER_UNAVAILABLE",
            "CANONICAL_SEMANTIC_FAILURE"} <= set(rs.reason_codes)
    assert rs.required is True


@pytest.mark.asyncio
async def test_provenance_frozen_against_later_config_change(monkeypatch):
    from app.config import get_settings

    _state, report = await _run_offline()
    report.provenance = derive_analysis_provenance(report, _state)
    before = report.provenance.model_dump_json()

    monkeypatch.setattr(get_settings(), "llm_provider", "changed-provider")
    monkeypatch.setattr(get_settings(), "analysis_prompt_version", "99.9")
    # the already-stamped provenance object is immutable and unchanged
    assert report.provenance.model_dump_json() == before
    with pytest.raises(Exception):
        report.provenance.provider = "x"


@pytest.mark.asyncio
async def test_review_required_cannot_be_cleared():
    _state, report = await _run_offline()
    with pytest.raises(Exception):
        report.review.required = False  # frozen? no -- but validator forbids reconstruct
    from app.models.agent import ReviewState
    with pytest.raises(Exception):
        ReviewState(required=False)
    with pytest.raises(Exception):
        InvestigationReport.model_validate({**json.loads(report.model_dump_json()),
                                            "human_review_required": False})


@pytest.mark.asyncio
async def test_failed_states_are_never_cached():
    cache_mod.clear_cache()
    _state, report = await _run_offline()

    for mode_patch in (
        {"report": {"analysis_mode": "DEGRADED", "semantic_mode": "DETERMINISTIC"}},
        {"report": {"analysis_mode": "LLM", "semantic_mode": "DETERMINISTIC_FALLBACK"}},
        {"report": {"analysis_mode": "LLM", "semantic_mode": "CANONICAL_LLM",
                    "remediation_cost": {"remediation_semantic_status": "LLM_TIMEOUT"}}},
    ):
        assert cache_mod.set_cached_analysis("k", mode_patch) is False
        assert cache_mod.get_cached_analysis("k") is None
    cache_mod.clear_cache()
