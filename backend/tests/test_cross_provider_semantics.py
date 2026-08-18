"""Cross-Provider Semantic Invariant Tests.

Proves that provider selection (Ollama vs GitHub Copilot) does NOT change the
deterministic safety architecture, evidence ledger rules, causal eligibility
firewalls, semantic ownership, or fallback behavior.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
import pytest

from app.agent.graph import get_agent_graph
from app.models.agent import EvidenceItem, EvidenceStatus, InvestigateRequest, RootCauseStatus
from app.services.llm.base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    """Custom mock provider returning controlled payloads for semantic invariant validation."""

    def __init__(self, provider_name: str, payload_content: str) -> None:
        self._provider_name = provider_name
        self._payload_content = payload_content

    async def generate(self, *, node: str, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(
            content=self._payload_content,
            provider=self._provider_name,
            model="mock-model",
            latency_ms=50,
            input_tokens=100,
            output_tokens=50,
            finish_reason="stop",
            raw_metadata={"node": node},
        )


class MockFailingProvider(LLMProvider):
    """Mock provider simulating provider failure (timeout/unavailable)."""

    def __init__(self, provider_name: str) -> None:
        self._provider_name = provider_name

    async def generate(self, *, node: str, prompt: str, **kwargs) -> LLMResponse:
        from app.services.llm.exceptions import LLMUnavailableError
        raise LLMUnavailableError(f"Mock failure from {self._provider_name}")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["ollama", "copilot"])
async def test_semantic_invariant_unestablished_evidence_remains_not_established(provider_name: str):
    """Semantic invariant: If the LLM proposes an ungrounded or self-reported cause,

    the deterministic validation engine MUST keep root_cause_status as NOT_ESTABLISHED
    regardless of whether Ollama or GitHub Copilot generated the proposal.
    """
    finding_text = "The required balance calibration was overdue by 3 days during audit."

    # LLM attempts to claim ESTABLISHED despite lack of verified causal mechanism in evidence
    llm_payload = json.dumps({
        "root_cause": {
            "status": "ESTABLISHED",
            "category": "CALIBRATION",
            "candidate_hypotheses": [
                {
                    "id": "H1",
                    "name": "CALIBRATION",
                    "statement": "Operator forgot calibration due to high workload.",
                    "status": "SUPPORTED",
                    "supporting_claim_ids": ["C1"],
                }
            ],
            "narrative": "Operator forgot calibration due to high workload.",
        },
        "five_why": {"steps": [], "is_complete": False, "status_note": "Incomplete"},
        "impact_assessment": {"status": "IMPACT_REQUIRES_ASSESSMENT", "areas": ["Calibration scope"]},
        "capa": {"status": "INVESTIGATION_REQUIRED", "potential_areas": [], "recommended_investigation": []},
        "contributing_factors": [],
        "ca_draft": {
            "immediate_action": "Calibrate balance.",
            "root_cause": "Operator oversight.",
            "root_cause_category": "CALIBRATION",
            "preventive_action": "Retrain operator.",
            "impact_analysis": "Check past measurements.",
        },
    })

    mock_provider = MockProvider(provider_name, llm_payload)

    with patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=mock_provider), \
         patch("app.agent.nodes.understanding.get_llm_client", return_value=mock_provider), \
         patch("app.agent.nodes.critic.get_llm_client", return_value=mock_provider):

        graph = get_agent_graph()
        initial_state = {
            "request": InvestigateRequest(finding_text=finding_text),
            "iteration_count": 0,
            "tool_call_count": 0,
            "critic_iteration": 0,
            "evidence_ledger": [
                EvidenceItem(claim="balance calibration was overdue by 3 days", source="finding_text", status=EvidenceStatus.VERIFIED)
            ],
            "evidence_gaps": [],
            "trace": [],
            "errors": [],
            "completed_tools": [],
            "planned_tools": [],
            "tool_results": {},
            "contributing_factors": [],
        }

        result = await graph.ainvoke(initial_state)
        report = result.get("report")

        assert report is not None
        # Invariant: Root cause status must NOT be established because high workload was not evidenced
        assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
        assert report.capa.status == "INVESTIGATION_REQUIRED"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["ollama", "copilot"])
async def test_semantic_invariant_fallback_on_provider_failure(provider_name: str):
    """Semantic invariant: If the provider fails, the system transitions to deterministic fallback.

    analysis_mode MUST be DETERMINISTIC across both providers without crashing.
    """
    finding_text = "The weekly review log was missing signatures for week 42."
    failing_provider = MockFailingProvider(provider_name)

    with patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=failing_provider), \
         patch("app.agent.nodes.understanding.get_llm_client", return_value=failing_provider), \
         patch("app.agent.nodes.critic.get_llm_client", return_value=failing_provider):

        graph = get_agent_graph()
        initial_state = {
            "request": InvestigateRequest(finding_text=finding_text),
            "iteration_count": 0,
            "tool_call_count": 0,
            "critic_iteration": 0,
            "evidence_ledger": [],
            "evidence_gaps": [],
            "trace": [],
            "errors": [],
            "completed_tools": [],
            "planned_tools": [],
            "tool_results": {},
            "contributing_factors": [],
        }

        result = await graph.ainvoke(initial_state)
        report = result.get("report")

        assert report is not None
        # Invariant: Must fall back to DETERMINISTIC mode when provider fails
        assert report.analysis_mode == "DETERMINISTIC"
        assert report.root_cause.status == RootCauseStatus.NOT_ESTABLISHED
