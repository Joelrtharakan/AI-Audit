"""Tests for the critic node's semantic enforcement (as opposed to the
deterministic keyword-stem guards in grounding_guard.py).

The deterministic guards (mentions_unsupported_domain, ungrounded_entities,
etc.) catch cheap, obvious cases for free but cannot generalize to every
fabricated mechanism a model might invent using only ordinary vocabulary
that happens to already be grounded (e.g. "the operator miscommunicated the
status to the auditor" for a finding that never mentions any auditor
interaction, but does mention "operator" and "status"). The critic is the
model-judgment-based layer meant to catch that class of error, and it now
actually acts on what it finds instead of only logging feedback.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.models.agent import (
    AgentTraceStep,
    CandidateHypothesis,
    CapaAnalysis,
    CapaStatus,
    FiveWhyAnalysis,
    ImpactAssessment,
    ImpactStatus,
    InvestigateRequest,
    RootCauseAnalysis,
)


def _build_state(finding_text: str, root_cause: RootCauseAnalysis) -> dict:
    return {
        "request": InvestigateRequest(finding_text=finding_text),
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "observation_quality": None,
        "extraction": None,
        "investigation_plan": None,
        "needs_investigation": False,
        "planned_tools": [],
        "completed_tools": [],
        "current_tool": None,
        "tool_results": {},
        "evidence_ledger": [],
        "evidence_gaps": [],
        "root_cause": root_cause,
        "contributing_factors": [],
        "five_why": FiveWhyAnalysis(steps=[], is_complete=False, status_note="INCOMPLETE"),
        "impact_assessment": ImpactAssessment(status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT, areas=[], narrative=None),
        "capa_analysis": CapaAnalysis(status=CapaStatus.INVESTIGATION_REQUIRED, potential_areas=[], recommended_investigation=[]),
        "critic_approved": False,
        "critic_feedback": None,
        "critic_send_back": False,
        "report": None,
        "ca_draft": None,
        "final_state": None,
        "trace": [AgentTraceStep.ok("Test started")],
        "errors": [],
    }


@pytest.mark.asyncio
async def test_critic_drops_hypothesis_it_flags_as_semantically_unsupported():
    """The critic identifies an ungrounded hypothesis by ID using its own
    semantic judgment (not a keyword match) and the node actually removes it."""
    from app.agent.nodes.critic import critic_node

    root_cause = RootCauseAnalysis(
        status="STATED_UNVERIFIED",
        category=None,
        narrative="Root cause not established.",
        candidate_hypotheses=[
            CandidateHypothesis(
                id="H1", name="LABEL_NOT_UPDATED",
                statement="The calibration status label may not have been updated after calibration was performed.",
                evidence_needed="Calibration certificate and label record",
            ),
            CandidateHypothesis(
                id="H2", name="VENDOR_DISPUTE",
                statement="A billing dispute with the calibration vendor may have delayed label issuance.",
                evidence_needed="Vendor correspondence",
            ),
        ],
    )
    state = _build_state(
        "Calibration label missing from equipment EQ-104. Operator says equipment was recently calibrated. Certificate unavailable.",
        root_cause,
    )

    critic_response = json.dumps({
        "approved": False,
        "send_back_for_investigation": False,
        "issues": [{"category": "DOMAIN_RELEVANCE", "severity": "BLOCKING", "description": "H2 invents a vendor billing dispute not present in the finding"}],
        "feedback": "H2 is fabricated.",
        "corrections_required": ["Remove H2"],
        "unsupported_hypothesis_ids": ["H2"],
        "root_cause_narrative_unsupported": False,
    })

    with patch("app.agent.nodes.critic.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = critic_response
        mock_get_client.return_value = mock_client
        result = await critic_node(state)

    names = [h.name for h in result["root_cause"].candidate_hypotheses]
    assert "VENDOR_DISPUTE" not in names
    assert "LABEL_NOT_UPDATED" in names
    assert any("dropped hypothesis" in t.model_dump()["message"] for t in result["trace"])


@pytest.mark.asyncio
async def test_critic_replaces_unsupported_root_cause_narrative():
    from app.agent.nodes.critic import critic_node

    root_cause = RootCauseAnalysis(
        status="STATED_UNVERIFIED",
        category="MAN",
        statement="The operator miscommunicated the calibration status to the auditor.",
        narrative="The operator miscommunicated the calibration status to the auditor, leading to the missing label.",
    )
    state = _build_state(
        "Calibration label missing from equipment EQ-104. Operator says equipment was recently calibrated. Certificate unavailable.",
        root_cause,
    )

    critic_response = json.dumps({
        "approved": False,
        "send_back_for_investigation": False,
        "issues": [{"category": "DOMAIN_RELEVANCE", "severity": "BLOCKING", "description": "Narrative invents an auditor interaction not in the finding"}],
        "feedback": "Narrative fabricates a mechanism.",
        "corrections_required": ["Remove the auditor-communication claim"],
        "unsupported_hypothesis_ids": [],
        "root_cause_narrative_unsupported": True,
    })

    with patch("app.agent.nodes.critic.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = critic_response
        mock_get_client.return_value = mock_client
        result = await critic_node(state)

    rc = result["root_cause"]
    assert rc.status == "NOT_ESTABLISHED"
    assert rc.statement is None
    assert "miscommunicated" not in rc.narrative.lower()
    assert "auditor" not in rc.narrative.lower()
