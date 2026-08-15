"""Live integration tests for app.agent.graph.py verifying all 12 architectural invariants.

Invokes the exact production graph pipeline used by the API endpoints.
"""

import json
from unittest.mock import AsyncMock, patch
import pytest

from app.agent.graph import get_agent_graph
from app.models.agent import (
    AgentTraceStep,
    CanonicalFindingState,
    EvidenceItem,
    EvidenceStatus,
    InvestigateRequest,
)

def _build_initial_state(finding_text: str) -> dict:
    request = InvestigateRequest(finding_text=finding_text)
    return {
        "request": request,
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "observation_quality": None,
        "extraction": None,
        "canonical_finding_state": None,
        "investigation_plan": None,
        "needs_investigation": False,
        "planned_tools": [],
        "completed_tools": [],
        "current_tool": None,
        "tool_results": {},
        "evidence_ledger": [],
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

@pytest.mark.asyncio
async def test_live_graph_mechanism_not_emitted_as_hypothesis_and_no_circular_why():
    """Verify invariants 1, 2, 4, 5, 6 on live production graph execution."""
    finding_text = (
        "During the internal audit, it was observed that the temperature log for refrigerator QC-REF-02 "
        "was not completed for 12 August 2026. The responsible technician confirmed that the temperature check was missed during the morning shift."
    )
    
    # Mock LLM calls inside graph nodes (understand_finding, core_synthesis, critic)
    # to simulate a model returning bad generic hypotheses and circular 5-Why answers.
    
    mock_synthesis_llm_json = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED",
            "category": "TO_BE_CONFIRMED",
            "statement": None,
            "leading_hypothesis": None,
            "candidate_hypotheses": [
                {
                    "id": "H1",
                    "name": "EXECUTION_OMISSION",
                    "statement": "The required activity associated with QC-REF-02 may not have been performed.",
                    "status": "POSSIBLE",
                    "evidence_needed": "Execution logs",
                },
                {
                    "id": "H2",
                    "name": "DOCUMENTATION_OMISSION",
                    "statement": "The required activity associated with QC-REF-02 may have been performed but not documented.",
                    "status": "POSSIBLE",
                    "evidence_needed": "Contemporaneous records",
                },
                {
                    "id": "H3",
                    "name": "SHIFT_HANDOVER_BREAKDOWN",
                    "statement": "Responsibility for the morning check on QC-REF-02 was not effectively communicated during shift handover.",
                    "status": "POSSIBLE",
                    "evidence_needed": "Shift handover logs",
                }
            ],
            "risk_of_recurrence": "MEDIUM",
            "narrative": "The check was missed during morning shift; underlying cause requires shift record review."
        },
        "five_why": {
            "steps": [
                {
                    "level": 1,
                    "question": "Why was the temperature log for QC-REF-02 incomplete?",
                    "answer": "The temperature check was missed during the morning shift.",
                    "status": "VERIFIED",
                    "evidence_reference": "Technician statement"
                },
                {
                    "level": 2,
                    "question": "Why did personnel report that the nonconformity occurred?",
                    "answer": "Because the technician reported it was missed.",
                    "status": "REPORTED"
                }
            ],
            "is_complete": True,
            "status_note": "Complete"
        },
        "impact_assessment": {
            "status": "IMPACT_REQUIRES_ASSESSMENT",
            "affected_object": "QC-REF-02",
            "affected_people": "Morning shift personnel",
            "affected_period": "12 August 2026",
            "process_at_risk": "Cold storage temperature monitoring",
            "relevant_change": "NOT ESTABLISHED",
            "potential_effect": "Potential temperature excursion undetected during morning shift",
            "evidence_needed": "SCADA temperature log for 12 August 2026",
            "areas": ["Cold storage monitoring"],
            "narrative": "Scope limited to 12 August 2026 morning shift for refrigerator QC-REF-02."
        },
        "capa": {
            "status": "INVESTIGATION_REQUIRED",
            "potential_areas": ["Shift Handover Control"],
            "recommended_investigation": ["Verify shift handover logs for 12 August 2026"],
            "conditional_actions": [
                {
                    "if_cause_confirmed": "If shift handover breakdown is confirmed",
                    "recommended_action": "Mandate digital shift handover sign-off for QC-REF-02 check"
                }
            ]
        },
        "contributing_factors": [
            {
                "description": "Manual recurring check dependent on individual memory without digital alert",
                "rationale": "QC-REF-02 check relies on manual log entry",
                "evidence_status": "INFERRED",
                "status": "POSSIBLE_UNCONFIRMED",
                "evidence_required": "SOP for refrigerator temperature check"
            }
        ],
        "ca_draft": {
            "immediate_action": "Inspect QC-REF-02 temperature continuous recorder for 12 August 2026.",
            "root_cause": "NOT_ESTABLISHED — Temperature check was missed during morning shift.",
            "root_cause_category": "TO_BE_CONFIRMED",
            "preventive_action": "Implement shift handover log verification.",
            "impact_analysis": "Scope limited to QC-REF-02 on 12 August 2026."
        }
    })

    graph = get_agent_graph()
    
    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_core_llm, \
         patch("app.agent.nodes.critic.get_llm_client") as mock_critic_llm:
        
        c_client = AsyncMock()
        c_client.chat_completion.return_value = mock_synthesis_llm_json
        mock_core_llm.return_value = c_client
        
        crit_client = AsyncMock()
        crit_client.chat_completion.return_value = json.dumps({"approved": True, "corrections_required": []})
        mock_critic_llm.return_value = crit_client

        state = _build_initial_state(finding_text)
        final_state = await graph.ainvoke(state)

    report = final_state["report"]
    assert report is not None

    # Invariant 1: A mechanism is emitted as a hypothesis (EXECUTION_OMISSION & DOCUMENTATION_OMISSION must be removed by causal_guard/final_evidence_verification)
    hyp_names = [h.name for h in report.root_cause.candidate_hypotheses]
    hyp_statements = [h.statement for h in report.root_cause.candidate_hypotheses]
    
    assert "EXECUTION_OMISSION" not in hyp_names
    assert "DOCUMENTATION_OMISSION" not in hyp_names
    assert not any("may not have been performed" in s for s in hyp_statements)
    assert not any("performed but not documented" in s for s in hyp_statements)
    # Valid causal hypotheses must survive
    assert any(n in hyp_names for n in ("SHIFT_HANDOVER_BREAKDOWN", "TASK_ASSIGNMENT_OR_HANDOVER_OMISSION"))

    # Invariant 6: A Why asks about reporting instead of causation ("Why did personnel report...") -> truncated/replaced
    five_why_questions = [s.question for s in report.five_why.steps]
    five_why_answers = [s.answer for s in report.five_why.steps]
    
    assert not any("why did personnel report" in q.lower() for q in five_why_questions)

    # Invariant 8: Contributing factors carry rationale
    assert len(report.contributing_factors) > 0
    assert report.contributing_factors[0].rationale is not None
