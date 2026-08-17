"""Tests for the read-only LLM metrics endpoint (GET /health/llm-metrics)
and the structured metric-event wiring it exposes."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.agent import (
    AgentTraceStep,
    CanonicalFindingState,
    EvidenceItem,
    EvidenceStatus,
    InvestigateRequest,
)
from app.services import llm_metrics
from app.services.llm_client import LLMTimeoutError

# Sensitive-content markers that must never appear in the endpoint response.
_FORBIDDEN_MARKERS = [
    "operator stated",
    "training on the revised procedure",
    "No training attendance record",
    "supporting_claim_ids",
    "root_cause",
    "candidate_hypotheses",
]


def _build_state() -> dict:
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


def _valid_llm_json() -> str:
    return json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED", "category": "TO_BE_CONFIRMED", "statement": None,
            "candidate_hypotheses": [{
                "id": "H1", "name": "TRAINING_NOT_COMPLETED",
                "statement": "Required training may not have been completed before the revised procedure became effective.",
                "supporting_claim_ids": ["C1"], "contradicting_claim_ids": ["C2"],
                "status": "POSSIBLE", "evidence_needed": "Authenticated training completion record",
            }],
            "narrative": "Conflicting reports exist regarding training completion.",
        },
        "five_why": {"steps": [], "is_complete": False, "status_note": "Stopped at evidence boundary"},
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


# --- A/B/C: endpoint returns 200, valid JSON, required fields present ---

def test_endpoint_returns_200_valid_json_with_required_fields():
    client = TestClient(app)
    resp = client.get("/health/llm-metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    metrics = body["metrics"]
    for field in (
        "llm_success_count", "llm_timeout_count", "llm_invalid_json_count",
        "recovery_attempt_count", "recovery_success_count", "recovery_timeout_count",
        "deterministic_fallback_count", "hypotheses_generated", "hypotheses_accepted",
        "hypotheses_rejected", "provenance_rejections", "causal_guard_rejections",
        "validation_repairs", "average_primary_latency_ms", "average_recovery_latency_ms",
        "recent_execution_count",
    ):
        assert field in metrics, f"missing field: {field}"


# --- J: no sensitive content in the endpoint response ---

def test_endpoint_contains_no_sensitive_content():
    client = TestClient(app)
    resp = client.get("/health/llm-metrics")
    body_text = resp.text
    for marker in _FORBIDDEN_MARKERS:
        assert marker not in body_text, f"leaked sensitive marker: {marker!r}"


# --- D/E: calling an investigation increments the expected counters ---

@pytest.mark.asyncio
async def test_investigation_increments_expected_counters():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(return_value=_valid_llm_json())
        await core_synthesis_node(state)

    metrics = llm_metrics.aggregated()
    assert metrics["primary_attempt_count"] == 1
    assert metrics["primary_success_count"] == 1
    assert metrics["llm_success_count"] == 1

    client = TestClient(app)
    resp = client.get("/health/llm-metrics")
    assert resp.json()["metrics"]["primary_success_count"] == 1


# --- F: primary timeout increments timeout exactly once ---

@pytest.mark.asyncio
async def test_primary_timeout_increments_timeout_exactly_once():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(
            side_effect=[LLMTimeoutError("timed out"), LLMTimeoutError("timed out again")]
        )
        await core_synthesis_node(state)

    metrics = llm_metrics.aggregated()
    assert metrics["primary_timeout_count"] == 1
    assert metrics["recovery_timeout_count"] == 1
    assert metrics["llm_timeout_count"] == 2  # primary + recovery, each counted once
    assert metrics["deterministic_fallback_count"] == 1


# --- G: recovery success increments recovery success exactly once ---

@pytest.mark.asyncio
async def test_recovery_success_increments_exactly_once():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(side_effect=[LLMTimeoutError("timed out"), _valid_llm_json()])
        await core_synthesis_node(state)

    metrics = llm_metrics.aggregated()
    assert metrics["recovery_success_count"] == 1
    assert metrics["primary_timeout_count"] == 1
    assert metrics["deterministic_fallback_count"] == 0


# --- H: double timeout increments deterministic fallback exactly once ---

@pytest.mark.asyncio
async def test_double_timeout_increments_fallback_exactly_once():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(
            side_effect=[LLMTimeoutError("timed out"), LLMTimeoutError("timed out again")]
        )
        await core_synthesis_node(state)

    assert llm_metrics.aggregated()["deterministic_fallback_count"] == 1


# --- I: validation rejection increments the appropriate rejection metric ---

@pytest.mark.asyncio
async def test_provenance_rejection_increments_rejection_metric():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    bad_json = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED", "category": "TO_BE_CONFIRMED", "statement": None,
            "candidate_hypotheses": [{
                "id": "H1", "name": "UNGROUNDED", "statement": "An uncited plausible-sounding mechanism.",
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
        await core_synthesis_node(state)

    metrics = llm_metrics.aggregated()
    assert metrics["provenance_rejections"] >= 1
    assert metrics["validation_rejections_total"] >= 1


# --- K: existing metrics behavior remains backwards compatible ---

def test_snapshot_and_reset_remain_backwards_compatible():
    llm_metrics.increment("llm_primary_attempted")
    assert llm_metrics.snapshot()["llm_primary_attempted"] == 1
    llm_metrics.reset()
    assert llm_metrics.snapshot()["llm_primary_attempted"] == 0


# --- Endpoint performs no LLM call / no investigation (Section 9) ---

def test_endpoint_never_calls_ollama():
    with patch("app.services.ollama_client.OllamaClient.chat_completion") as mock_chat:
        client = TestClient(app)
        resp = client.get("/health/llm-metrics")
        assert resp.status_code == 200
        mock_chat.assert_not_called()


# --- Request correlation (Section 7): same request_id across stages ---

@pytest.mark.asyncio
async def test_request_id_correlates_primary_and_recovery():
    from app.agent.nodes.core_synthesis import core_synthesis_node
    state = _build_state()
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_get:
        mock_get.return_value = _mock_client(side_effect=[LLMTimeoutError("timed out"), _valid_llm_json()])
        result = await core_synthesis_node(state)

    request_id = result["synthesis_execution"]["request_id"]
    recent = llm_metrics.recent_executions()
    ids_seen = {e["request_id"] for e in recent}
    assert request_id in ids_seen
    phases_for_request = {e["phase"] for e in recent if e["request_id"] == request_id}
    assert "primary" in phases_for_request
    assert "recovery" in phases_for_request
