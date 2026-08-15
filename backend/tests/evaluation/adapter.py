"""Agent Adapter for connecting evaluation framework to the LQMS AI Agent graph.
"""

import asyncio
import logging
from typing import Any, Dict

from app.agent.graph import get_agent_graph
from app.agent.state import AgentState
from app.models.agent import AgentTraceStep, InvestigateRequest


logger = logging.getLogger(__name__)


class AgentAdapter:
    """Adapter wrapper around the LQMS agentic investigation graph."""

    def __init__(self, timeout_seconds: float = 120.0, offline: bool = False):
        self.timeout_seconds = timeout_seconds
        self.offline = offline
        self.graph = get_agent_graph()

    async def analyze(self, finding_text: str, departments: list[str] | None = None) -> Dict[str, Any]:
        """Runs the investigation agent on the given finding text and returns a unified state dict."""
        payload = InvestigateRequest(
            finding_text=finding_text,
            departments=departments or [],
        )

        if self.offline:
            from app.agent.nodes.five_why_fallback import build_deterministic_five_why
            from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
            from app.models.agent import (
                CADraft,
                CapaAnalysis,
                EvidenceItem,
                EvidenceStatus,
                ImpactAssessment,
                RootCauseAnalysis,
            )
            from app.models.analysis import ObservationQualityResult, ObservationQualityStatus

            ledger = [EvidenceItem(claim=finding_text[:100], source="finding_text", status=EvidenceStatus.VERIFIED)]
            fw = build_deterministic_five_why(finding_text, ledger)
            hyps, plan = build_deterministic_investigation_plan(finding_text, ledger)

            from app.agent.analytical_validator import select_leading_hypothesis

            rc = RootCauseAnalysis(
                status="NOT_ESTABLISHED",
                category="TO_BE_CONFIRMED",
                narrative="Root cause not established from available evidence.",
                candidate_hypotheses=hyps,
                leading_hypothesis=select_leading_hypothesis(hyps),
            )
            capa = CapaAnalysis(status="INVESTIGATION_REQUIRED")
            impact = ImpactAssessment(
                status="IMPACT_REQUIRES_ASSESSMENT",
                narrative="Impact scope requires verification.",
            )

            ca_draft = CADraft(
                immediate_action="Identify specific incomplete entries and preserve original records.",
                root_cause="NOT_ESTABLISHED — Evidence is insufficient to determine cause.",
                root_cause_category="TO_BE_CONFIRMED",
                preventive_action="Investigate execution vs documentation controls.",
                impact_analysis="Affected records pending verification.",
            )

            return {
                "request": payload,
                "observation_quality": ObservationQualityResult(status=ObservationQualityStatus.SUFFICIENT),
                "root_cause": rc,
                "five_why": fw,
                "investigation_plan": plan,
                "capa_analysis": capa,
                "impact_assessment": impact,
                "ca_draft": ca_draft,
                "evidence_ledger": ledger,
                "trace": [AgentTraceStep.ok("Offline evaluation mode executed")],
                "errors": [],
            }


        initial_state: AgentState = {
            "request": payload,
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
            "trace": [AgentTraceStep.ok("Evaluation agent adapter started")],
            "errors": [],
        }

        try:
            final_state = await asyncio.wait_for(
                self.graph.ainvoke(initial_state),
                timeout=self.timeout_seconds,
            )
            return final_state
        except Exception as exc:
            logger.error("Agent execution failed in adapter: %s", exc)
            return {
                "error": str(exc),
                "request": payload,
                "errors": [str(exc)],
            }
