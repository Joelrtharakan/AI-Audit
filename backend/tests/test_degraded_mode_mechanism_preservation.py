"""Critical regression: when the LLM is completely unavailable, a causal
mechanism already explicitly present in the finding text (an attributed
statement — "X stated/confirmed/reported that...", or an awareness-gap
construction — "X was unaware that...") must survive all the way to the
final report's 5-Why chain.

This is the fix for the reported bug where degraded mode produced "Could
not be validated against this finding's evidence" for a finding that
already explicitly contained a REPORTED mechanism. Two independent causes
were found and fixed:

1. app/services/attribution_extraction.py: when the LLM extraction call
   fails, the previous fallback collapsed every sentence to a plain
   VERIFIED fact, destroying the REPORTED/attribution distinction.
2. app/agent/grounding_guard.py: the entity-fabrication heuristic flagged
   this system's own shouting-case boundary language ("NOT ESTABLISHED
   FROM AVAILABLE EVIDENCE") as fabricated identifiers and stripped it.

Run through the actual live graph (get_agent_graph()) with every
LLM-calling node mocked to fail, across multiple unrelated QMS domains to
prove the fix is structural, not tuned to the bug report's specific
wording (checklist/operator/procedure revision).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.models.agent import InvestigateRequest

_FORBIDDEN_PHRASE = "could not be validated against this finding's evidence"

FINDINGS = [
    "The daily equipment inspection checklist was not completed for three consecutive days. "
    "The operator stated that they were unaware that the checklist procedure had been revised.",

    "The calibration record for the measuring instrument contained a blank entry. "
    "The analyst reported that the calibration reminder notification was never received.",

    "The training file for the new laboratory technician lacked a signed acknowledgement. "
    "The supervisor confirmed that the revised training module had not been assigned.",

    "The environmental monitoring trend report was not generated for the month. "
    "The reviewer indicated that they did not know the trending requirement applied to this location.",

    "The supplier requalification review was six months overdue. "
    "The quality manager noted that the requalification schedule had not been communicated to the team.",
]


def _build_initial_state(finding_text: str) -> dict:
    return {
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


@pytest.mark.asyncio
@pytest.mark.parametrize("finding_text", FINDINGS)
async def test_reported_mechanism_survives_total_llm_outage(finding_text):
    from app.agent.graph import get_agent_graph

    graph = get_agent_graph()

    with patch("app.agent.nodes.core_synthesis.get_llm_client") as mock_cs, \
         patch("app.agent.nodes.understanding.get_llm_client") as mock_u, \
         patch("app.services.extraction.get_llm_client") as mock_e, \
         patch("app.agent.nodes.investigation_planner.get_llm_client") as mock_p, \
         patch("app.agent.nodes.critic.get_llm_client") as mock_c:
        for mock in (mock_cs, mock_u, mock_e, mock_p, mock_c):
            client = AsyncMock()
            client.chat_completion.side_effect = RuntimeError("simulated total provider outage")
            mock.return_value = client
        result = await graph.ainvoke(_build_initial_state(finding_text))

    report = result["report"]
    # Total LLM outage with a complete, evidence-grounded deterministic
    # result is analysis_mode="DETERMINISTIC", not "DEGRADED" — DEGRADED is
    # reserved for when the deterministic engine ALSO fails to produce a
    # safe result (see app/agent/nodes/core_synthesis.py _classify_failure /
    # the DETERMINISTIC-vs-DEGRADED contract).
    assert report.analysis_mode == "DETERMINISTIC"
    assert report.root_cause.status == "NOT_ESTABLISHED"

    five_why = report.five_why
    assert len(five_why.steps) >= 2, "the chain must include the mechanism, not stop at the bare observation"

    joined_answers = " ".join((s.answer or "") for s in five_why.steps).lower()
    assert _FORBIDDEN_PHRASE not in joined_answers, (
        "the reported mechanism was dropped instead of preserved — this is the exact bug being regression-tested"
    )

    # The second step must carry REPORTED status (the attributed statement),
    # not just repeat the VERIFIED observation from step 1.
    assert five_why.steps[1].status == "REPORTED"
    assert five_why.steps[1].answer != five_why.steps[0].answer
