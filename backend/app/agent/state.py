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
    CanonicalFindingState,
    CapaAnalysis,
    ContributingFactor,
    CostImpact,
    EvidenceClaim,
    EvidenceConflict,
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
    canonical_finding_state: CanonicalFindingState | None



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
    cost_impact: CostImpact | None
    capa_analysis: CapaAnalysis | None
    # "LLM" (normal path) or "DEGRADED" (core_synthesis's LLM call failed and
    # a deterministic fallback ran) -- must be surfaced to the final report
    # so degraded analysis is never mistaken for a normal result.
    analysis_mode: str
    # Internal execution-state record (not part of the public report
    # contract): which path actually produced the result
    # (PRIMARY_LLM/RECOVERY_LLM/DETERMINISTIC) plus validation repair/
    # rejection counts, distinct from the public analysis_mode string.
    synthesis_execution: dict
    # Infrastructure metadata from the LLM provider router (app/services/
    # llm_router.py) — which provider actually answered core_synthesis's
    # call, whether a fallback was needed, and the full attempt order. No
    # analytical meaning; purely observability.
    provider_used: str | None
    fallback_used: bool
    provider_attempts: list[str]

    # ------------------------------------------------------------------
    # Critic
    # ------------------------------------------------------------------
    critic_approved: bool
    critic_feedback: str | None
    critic_send_back: bool           # True → critic wants more investigation
    # "SKIPPED" (deterministic pre-gate found nothing to check — 0ms fast
    # path), "OK" (LLM critic ran and returned a usable result), or
    # "UNAVAILABLE" (LLM critic call failed/timed out). The critic is a
    # secondary quality check, not the primary source of truth — its own
    # unavailability must never demote a valid core_synthesis result to
    # DEGRADED/DETERMINISTIC. See app/agent/nodes/critic.py.
    critic_status: str

    # ------------------------------------------------------------------
    # Final outputs
    # ------------------------------------------------------------------
    report: InvestigationReport | None
    ca_draft: CADraft | None
    final_state: AgentFinalState | None

    # ------------------------------------------------------------------
    # Epistemic transition tracking
    # ------------------------------------------------------------------
    # Append-only log of compact epistemic snapshots (see
    # app.agent.causal_graph.capture_epistemic_snapshot), one per
    # core_synthesis_node pass (usually just one; a second appears only on
    # the critic-send-back re-investigation loop). Lets
    # app.agent.invariants._check_epistemic_status_transitions compare
    # consecutive passes and catch an actual regression (a hypothesis
    # silently downgrading, or root_cause status/causal_readiness
    # regressing) rather than only inspecting the latest snapshot in
    # isolation.
    epistemic_snapshot_history: list[dict]

    # ------------------------------------------------------------------
    # Trace & errors
    # ------------------------------------------------------------------
    trace: list[AgentTraceStep]
    errors: list[str]
