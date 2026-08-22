"""LangGraph StateGraph definition for the LQMS Corrective Action Investigation Agent.

Graph structure:
  START
    → understand_finding
    → plan_investigation
    → [should_investigate?]
       YES → execute_tool → record_evidence → [more_tools?]
              YES → execute_tool (loop)
              NO  → core_synthesis
       NO  → core_synthesis
    → causal_investigation_planner (Phase 17 Stage B — unconditional)
    → critic
    → [critic_decision?]
       SEND_BACK + iterations_left → execute_tool (re-investigate)
       APPROVED or MAX_ITERATIONS → generate_report
    → generate_report
    → final_evidence_verification
    → END

core_synthesis is the SOLE authoritative implementation of RCA, 5-Why,
contributing factors, impact assessment, CAPA, and CA-draft generation --
it replaced the older serial root_cause_node / impact_assessment_node /
capa_analysis_node / ca_draft_generator_node one-LLM-call-per-stage nodes in
a single consolidated call. Those older node modules still exist (and are
still exercised directly by some unit tests as isolated-guard tests) but are
NOT part of the live graph and must not be re-registered here without
retiring core_synthesis first -- having two independent implementations of
the same reasoning both live would silently diverge.

final_evidence_verification runs LAST (after generate_report, not before):
it is the analytical validation firewall -- grounding sweep, causal
consistency checks, and app/agent/analytical_validator.py's structural
invariants (root-cause certainty monotonicity, 5-Why mechanism-skip repair,
CAPA causal linkage, leading-hypothesis re-derivation) -- and mutates the
same object instances already embedded in `report` in place, so no
re-assembly step is needed after it runs.

Conditional edges implement the control flow. Nodes never raise — they record
errors in state and let the graph route gracefully.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from app.agent.nodes.causal_investigation_planner import causal_investigation_planner_node
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.critic import critic_node
from app.agent.nodes.evidence_acquisition import acquire_evidence_node
from app.agent.nodes.evidence_recorder import record_evidence_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
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
    """Route after plan_investigation: investigate or go straight to core synthesis."""
    settings = get_settings()
    if (
        state.get("needs_investigation")
        and state.get("planned_tools")
        and state.get("iteration_count", 0) < settings.agent_max_iterations
    ):
        return "execute_tool"
    return "core_synthesis"


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
    return "core_synthesis"


def stage_b_loop_decision(state: AgentState) -> str:
    """Phase 19: the actual missing link. Runs after every
    causal_investigation_planner (Stage B) execution and deterministically
    decides whether to loop back for more evidence or proceed to critic.

    Only activates the loop when ALL of:
      - a real evidence_provider is configured (Section 1: the engine
        depends on the abstract interface; when unset -- every pre-
        Phase-19 caller and the entire existing test corpus -- this
        function ALWAYS returns "critic", identical to Phase 17/18
        behavior, by construction rather than by a special case),
      - Stage B's own plan has real, actionable questions to answer,
      - the deterministic decide_investigation_state() says CONTINUE/REPLAN,
      - the iteration budget is not exhausted.

    Never an LLM decision (Section 14/17): every input here is structural
    state, never generated prose.
    """
    provider = state.get("evidence_provider")
    if provider is None:
        return "critic"

    plan = state.get("causal_investigation_plan")
    if plan is None or not getattr(plan, "questions", None):
        return "critic"

    from app.agent.nodes.causal_investigation_planner import (
        _DEFAULT_MAX_INVESTIGATION_ITERATIONS,
        decide_investigation_state,
    )

    live_ids = sorted({
        hid for q in plan.questions for hid in (getattr(q, "target_hypothesis_ids", None) or [])
    })
    iteration = state.get("investigation_iteration", 0)
    causal_graph = state.get("causal_graph")
    decision = decide_investigation_state(live_ids, iteration, _DEFAULT_MAX_INVESTIGATION_ITERATIONS, causal_graph)
    if decision in ("CONTINUE", "REPLAN"):
        return "acquire_evidence"
    return "critic"


def critic_decision(state: AgentState) -> str:
    """After critic: proceed to generate_report or send back for more investigation."""
    settings = get_settings()
    send_back = state.get("critic_send_back", False)
    critic_iter = state.get("critic_iteration", 0)

    # Phase 17: Stage B's deterministic causal-graph judgment overrides the
    # (LLM-assisted) critic's own send-back heuristic when Stage B has
    # already determined NO_ACTIONABLE_UNCERTAINTY -- a real, tested
    # control-flow effect (Section 43's "can Stage B terminate
    # investigation?"), never the reverse: this can only SUPPRESS a
    # send-back the critic would otherwise request, never manufacture one
    # the critic didn't. The LLM never gets authority to override a
    # deterministic "nothing left to investigate" graph judgment.
    stage_b_plan = state.get("causal_investigation_plan")
    if send_back and stage_b_plan is not None and getattr(stage_b_plan, "planner_mode", None) == "NO_ACTIONABLE_UNCERTAINTY":
        return "generate_report"

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
    graph.add_node("core_synthesis", core_synthesis_node)  # Single consolidated synthesis
    graph.add_node("causal_investigation_planner", causal_investigation_planner_node)  # Phase 17 Stage B
    graph.add_node("acquire_evidence", acquire_evidence_node)  # Phase 19 adaptive evidence loop
    graph.add_node("critic", critic_node)
    graph.add_node("final_evidence_verification", final_evidence_verification_node)
    graph.add_node("generate_report", generate_report_node)

    # Entry point
    graph.add_edge(START, "understand_finding")
    graph.add_edge("understand_finding", "plan_investigation")

    # Conditional: investigate or skip to core synthesis
    graph.add_conditional_edges(
        "plan_investigation",
        should_investigate,
        {
            "execute_tool": "execute_tool",
            "core_synthesis": "core_synthesis",
        },
    )

    # Tool execution → evidence recording
    graph.add_edge("execute_tool", "record_evidence")

    # Conditional: more tools or move to core synthesis
    graph.add_conditional_edges(
        "record_evidence",
        more_tools_needed,
        {
            "execute_tool": "execute_tool",
            "core_synthesis": "core_synthesis",
        },
    )

    # Core Synthesis → Stage B causal investigation planner → [loop?] → Critic
    # (Phase 17: Stage B runs unconditionally, on every graph execution,
    # immediately after CandidateHypothesis objects first exist — the
    # earliest point in the actual topology where a real causal graph can
    # be built and reasoned over. See causal_investigation_planner.py's
    # module docstring for the honest scope of what this changes.)
    graph.add_edge("core_synthesis", "causal_investigation_planner")
    # Phase 19: the actual adaptive evidence loop, deterministically gated
    # by stage_b_loop_decision. acquire_evidence loops back to Stage B
    # (not forward to critic) so the SECOND Stage-B invocation genuinely
    # consumes the graph/hypothesis state evidence acquisition just
    # updated — this is what makes Plan A -> evidence -> Plan B a real,
    # compiled-graph behavior rather than two independent function calls.
    graph.add_conditional_edges(
        "causal_investigation_planner",
        stage_b_loop_decision,
        {
            "acquire_evidence": "acquire_evidence",
            "critic": "critic",
        },
    )
    graph.add_edge("acquire_evidence", "causal_investigation_planner")

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
    graph.add_edge("generate_report", "final_evidence_verification")
    graph.add_edge("final_evidence_verification", END)

    return graph.compile()


# Singleton compiled graph — built lazily on first import
_graph = None


def get_agent_graph():
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph
