"""LangGraph agent state definition.

AgentState is the single TypedDict that flows through the entire graph.
Every node reads from and writes to this dict. LangGraph persists the full
state between node executions, enabling the conditional investigation loop.

Design rules:
  - No global mutable state. All state lives here.
  - Nodes never raise; they set errors[] and let the graph route to a graceful end.
  - iteration_count and tool_call_count are checked at every conditional edge.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from app.models.agent import (
    AgentFinalState,
    AgentTraceStep,
    CADraft,
    CapaAnalysis,
    ContributingFactor,
    EvidenceGap,
    EvidenceItem,
    FiveWhyAnalysis,
    ImpactAssessment,
    InvestigateRequest,
    InvestigationPlan,
    InvestigationReport,
    RootCauseAnalysis,
)
from app.models.analysis import ExtractionResult, ObservationQualityResult


class AgentState(TypedDict, total=False):
    # ------------------------------------------------------------------
    # Input (set once at graph entry, never modified)
    # ------------------------------------------------------------------
    request: InvestigateRequest

    # ------------------------------------------------------------------
    # Iteration / safety counters
    # ------------------------------------------------------------------
    iteration_count: int
    tool_call_count: int
    critic_iteration: int

    # ------------------------------------------------------------------
    # Finding understanding phase
    # ------------------------------------------------------------------
    observation_quality: ObservationQualityResult | None
    extraction: ExtractionResult | None

    # ------------------------------------------------------------------
    # Investigation planning
    # ------------------------------------------------------------------
    investigation_plan: InvestigationPlan | None
    needs_investigation: bool
    planned_tools: list[str]          # ordered list of tool names to call
    completed_tools: list[str]        # tools already called this run
    current_tool: str | None          # tool being executed right now

    # ------------------------------------------------------------------
    # Tool results (raw, keyed by tool name)
    # Each tool accumulates a list of results (can be called >once with
    # different args, e.g. get_training_record for different departments)
    # ------------------------------------------------------------------
    tool_results: dict[str, list[Any]]

    # ------------------------------------------------------------------
    # Evidence ledger
    # ------------------------------------------------------------------
    evidence_ledger: list[EvidenceItem]
    evidence_gaps: list[EvidenceGap]

    # ------------------------------------------------------------------
    # Analysis results
    # ------------------------------------------------------------------
    root_cause: RootCauseAnalysis | None
    contributing_factors: list[ContributingFactor]
    five_why: FiveWhyAnalysis | None
    impact_assessment: ImpactAssessment | None
    capa_analysis: CapaAnalysis | None

    # ------------------------------------------------------------------
    # Critic
    # ------------------------------------------------------------------
    critic_approved: bool
    critic_feedback: str | None
    critic_send_back: bool           # True → critic wants more investigation

    # ------------------------------------------------------------------
    # Final outputs
    # ------------------------------------------------------------------
    report: InvestigationReport | None
    ca_draft: CADraft | None
    final_state: AgentFinalState | None

    # ------------------------------------------------------------------
    # Trace & errors
    # ------------------------------------------------------------------
    trace: list[AgentTraceStep]
    errors: list[str]
