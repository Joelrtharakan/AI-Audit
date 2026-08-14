"""LangGraph StateGraph definition for the LQMS Corrective Action Investigation Agent.

Graph structure:
  START
    → understand_finding
    → plan_investigation
    → [should_investigate?]
       YES → execute_tool → record_evidence → [more_tools?]
              YES → execute_tool (loop)
              NO  → root_cause_analysis
       NO  → root_cause_analysis
    → impact_assessment
    → capa_analysis
    → critic
    → [critic_decision?]
       SEND_BACK + iterations_left → execute_tool (re-investigate)
       APPROVED or MAX_ITERATIONS → generate_report
    → generate_report
    → generate_ca_draft
    → END

Conditional edges implement the control flow. Nodes never raise — they record
errors in state and let the graph route gracefully.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.ca_draft_generator import ca_draft_generator_node
from app.agent.nodes.capa import capa_analysis_node
from app.agent.nodes.critic import critic_node
from app.agent.nodes.evidence_recorder import record_evidence_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.impact import impact_assessment_node
from app.agent.nodes.impact_capa_parallel import impact_capa_parallel_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.rca import root_cause_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.tool_executor import execute_tool_node
from app.agent.nodes.understanding import understand_finding_node
from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def should_investigate(state: AgentState) -> str:
    """Route after plan_investigation: investigate or go straight to RCA."""
    settings = get_settings()
    if (
        state.get("needs_investigation")
        and state.get("planned_tools")
        and state.get("iteration_count", 0) < settings.agent_max_iterations
    ):
        return "execute_tool"
    return "root_cause_analysis"


def more_tools_needed(state: AgentState) -> str:
    """After recording evidence: are there more tools to call?"""
    settings = get_settings()
    planned = state.get("_planned_tool_calls", [])
    completed = state.get("completed_tools", [])
    tool_call_count = state.get("tool_call_count", 0)

    pending = [t for t in planned if t.get("tool") not in completed]
    iteration_count = state.get("iteration_count", 0) + 1

    if (
        pending
        and tool_call_count < settings.agent_max_tool_calls
        and iteration_count < settings.agent_max_iterations
    ):
        return "execute_tool"
    return "root_cause_analysis"


def critic_decision(state: AgentState) -> str:
    """After critic: proceed to generate_report or send back for more investigation."""
    settings = get_settings()
    send_back = state.get("critic_send_back", False)
    critic_iter = state.get("critic_iteration", 0)

    if send_back and critic_iter <= settings.agent_max_critic_iterations:
        # Only send back if there are tools that could actually be called
        planned = state.get("_planned_tool_calls", [])
        completed = state.get("completed_tools", [])
        pending = [t for t in planned if t.get("tool") not in completed]
        if pending:
            return "execute_tool"

    return "generate_report"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_agent_graph() -> StateGraph:
    """Construct and compile the investigation agent graph."""
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("understand_finding", understand_finding_node)
    graph.add_node("plan_investigation", plan_investigation_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("record_evidence", record_evidence_node)
    graph.add_node("root_cause_analysis", root_cause_node)
    graph.add_node("impact_capa", impact_capa_parallel_node)  # impact + CAPA run in parallel
    graph.add_node("critic", critic_node)
    graph.add_node("final_evidence_verification", final_evidence_verification_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("generate_ca_draft", ca_draft_generator_node)

    # Entry point
    graph.add_edge(START, "understand_finding")
    graph.add_edge("understand_finding", "plan_investigation")

    # Conditional: investigate or skip
    graph.add_conditional_edges(
        "plan_investigation",
        should_investigate,
        {
            "execute_tool": "execute_tool",
            "root_cause_analysis": "root_cause_analysis",
        },
    )

    # Tool execution → evidence recording
    graph.add_edge("execute_tool", "record_evidence")

    # Conditional: more tools or move to RCA
    graph.add_conditional_edges(
        "record_evidence",
        more_tools_needed,
        {
            "execute_tool": "execute_tool",
            "root_cause_analysis": "root_cause_analysis",
        },
    )

    # Analysis chain (impact + CAPA run in parallel for speed)
    graph.add_edge("root_cause_analysis", "impact_capa")
    graph.add_edge("impact_capa", "critic")

    # Conditional: critic decision
    graph.add_conditional_edges(
        "critic",
        critic_decision,
        {
            "execute_tool": "execute_tool",
            "generate_report": "generate_report",
        },
    )

    # Output chain
    graph.add_edge("generate_report", "generate_ca_draft")
    graph.add_edge("generate_ca_draft", "final_evidence_verification")
    graph.add_edge("final_evidence_verification", END)

    return graph.compile()


# Singleton compiled graph — built lazily on first import
_graph = None


def get_agent_graph():
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph
