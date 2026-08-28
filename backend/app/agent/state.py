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
    CausalGraph,
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
    # LLM canonical semantic finding context (promotion pass) -- validated
    # (never raw), computed once in plan_investigation_node, reused by
    # every later node instead of independently re-interpreting raw text.
    # None whenever unavailable/disabled (see canonical_semantic_shadow_
    # enabled); downstream consumers must fall back to unchanged legacy
    # behavior in that case, never conflating it with an explicit semantic
    # NOT_ESTABLISHED.
    canonical_semantic_context: Any | None



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
    # Phase 3: real runtime causal graph, built by
    # app.agent.causal_graph.build_causal_graph and attached by
    # final_evidence_verification_node. MUST be declared here — LangGraph
    # only persists keys present in the AgentState schema across node
    # executions; an undeclared key is silently dropped from the merged
    # state even though the node returned it, which caused
    # state["five_why"] steps to retain causal_edge_id references into a
    # graph that never actually reached the final state (caught by
    # INV-CAUSAL-004 in end-to-end regression runs).
    causal_graph: CausalGraph | None
    # Phase 4: pre-investigation graph, built from the semantic graph alone
    # (before CandidateHypothesis data exists) and consumed by the
    # deterministic investigation planner. Declared here for the same
    # reason `causal_graph` had to be — an undeclared key is silently
    # dropped by LangGraph's state merge.
    causal_uncertainty_graph: CausalGraph | None
    # Phase 9: real derived causal-path objects (one per hypothesis with a
    # licensed edge chain back to the deviation). Declared here for the
    # same reason causal_graph/causal_uncertainty_graph had to be —
    # LangGraph drops any state key not in this schema.
    causal_paths: list[Any]
    # Phase 17: Stage B post-synthesis causal investigation plan, built by
    # app.agent.nodes.causal_investigation_planner AFTER CandidateHypothesis
    # objects exist. Additive — never overwrites `investigation_plan` (see
    # that module's docstring for the disclosed scope). Declared here for
    # the same LangGraph state-merge reason as causal_graph above.
    causal_investigation_plan: InvestigationPlan | None
    # Phase 18: adaptive-loop bookkeeping. `causal_graph_version` increments
    # every time causal_investigation_planner_node rebuilds the causal graph
    # (Section 6); `investigation_iteration` increments every time it runs
    # (Section 2); `investigation_history` accumulates one
    # InvestigationIterationRecord per run, giving the loop (and any test)
    # a real, inspectable trace of what changed between Stage-B invocations
    # rather than only the latest snapshot. Declared here for the same
    # LangGraph state-merge reason as the fields above.
    causal_graph_version: int
    investigation_iteration: int
    investigation_history: list[Any]
    # Phase 19: pluggable evidence-acquisition abstraction (Section 1). An
    # `app.services.evidence_provider.EvidenceProvider` instance, or None
    # (the default for every pre-Phase-19 caller and the entire existing
    # test corpus) -- when None, the adaptive evidence loop in graph.py
    # never activates and control flow is byte-for-byte identical to
    # Phase 17/18. Never a raw dict/backend detail; the engine only ever
    # calls `.acquire()` on this.
    evidence_provider: Any | None
    # Phase 21: optional app.agent.evidence_interpreter.EvidenceInterpreter
    # instance, or None (the default for every pre-Phase-21 caller and the
    # entire existing test corpus) -- when None, acquire_evidence_node's
    # claim-interpretation step never runs and behavior is byte-for-byte
    # identical to Phase 19/20. Same optionality pattern as evidence_provider.
    evidence_interpreter: Any | None
    # Phase 21: real runtime claim ledger, populated by
    # app.agent.nodes.evidence_acquisition from app.agent.evidence_interpreter
    # output. Reuses the pre-existing app.models.agent.EvidenceClaim model
    # (see that module's docstring) -- not a second claim representation.
    # Declared here for the same LangGraph state-merge reason as
    # evidence_ledger above.
    evidence_claims: list[EvidenceClaim]
    # Phase 22 Part B/C/D/E: append-only EvidenceRequest lifecycle ledger --
    # extends the existing evidence-acquisition state rather than creating
    # a second history system (Part E). Every EvidenceRequest ever built
    # (across every acquire_evidence_node execution this run) lives here
    # exactly once, with its `status`/`parent_request_id` updated in place
    # as evidence resolves it -- see app.agent.nodes.evidence_acquisition.
    # reconcile_evidence_request.
    evidence_requests: list[Any]
    # Phase 21: reuses the pre-existing app.models.agent.EvidenceConflict
    # model / app.agent.claim_extractor.detect_evidence_conflicts machinery
    # as this architecture's first-class contradiction representation
    # (Part L) -- never a second, competing one.
    evidence_conflicts: list[EvidenceConflict]
    # Phase 19 Section 8: append-only hypothesis status-transition history,
    # populated by app.agent.nodes.evidence_acquisition.
    hypothesis_history: list[Any]
    impact_assessment: ImpactAssessment | None
    cost_impact: CostImpact | None
    financial_analysis: Any | None
    # Canonical RemediationCostResult (app.remediation) -- expected cost to
    # correct/prevent the finding. Computed lazily in report_generator_node
    # (like financial_analysis). Declared here so LangGraph persists it.
    remediation_cost: Any | None
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
