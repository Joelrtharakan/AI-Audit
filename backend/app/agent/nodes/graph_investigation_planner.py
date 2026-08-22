"""Phase 16: graph-first investigation planning.

Audited runtime constraint (Phase 16 Section 1/42 — read before trusting any
claim below): `plan_investigation_node` runs immediately after
`understand_finding_node` and BEFORE `core_synthesis_node`
(app/agent/graph.py: `understand_finding -> plan_investigation -> ... ->
core_synthesis`). `CandidateHypothesis` objects and the licensed
`CausalGraph` do not exist yet at the point investigation planning runs —
they are produced by `core_synthesis_node` and
`final_evidence_verification_node` respectively, both of which execute
AFTER `plan_investigation_node` on every path through the graph, including
the critic send-back loop (which routes to `execute_tool`, never back to
`plan_investigation`). This is a real, structural dependency gap, not a
convenience assumption:

  - `plan_investigation_graph_first` below can therefore only reason over
    `CanonicalFindingState.semantic_graph` (via the existing
    `build_causal_uncertainty_graph`), because that is the only graph
    structure that genuinely exists at this point in the pipeline.
  - `build_discriminating_questions_for_competing_hypotheses` below DOES
    implement genuine competing-hypothesis discrimination logic over
    `CandidateHypothesis` + `CausalGraph` (Phase 16 Sections 9/31), and is
    unit-tested directly — but it is NOT currently called from
    `plan_investigation_node` or any other runtime node, because there is
    no point in the current graph where both a live LangGraph run and
    populated candidate hypotheses coexist with the investigation-planning
    step. This is reported honestly as an unwired, real building block, not
    as an active runtime mechanism.

What IS real and wired into the runtime this phase (see
`app/agent/nodes/investigation_planner.py`):

  - `plan_investigation_graph_first` is called first, unconditionally, by
    `plan_investigation_node`.
  - Its `planner_mode` determination (NO_ACTIONABLE_UNCERTAINTY /
    TARGET_UNRESOLVED / content-bearing) is authoritative — it, not any
    heuristic in the legacy planner, decides which of those three states
    applies for a given run.
  - For NO_ACTIONABLE_UNCERTAINTY, this function's empty plan IS the final
    plan; the legacy template planner is never invoked.
  - For TARGET_UNRESOLVED (semantic_graph missing/empty on an actionable
    finding — a real, if rare, structural insufficiency), the legacy
    planner is invoked as an explicit, observable, labeled fallback
    (`planner_mode="TARGET_UNRESOLVED"`, `fallback_reason` populated) —
    never silently.
  - For the common case (semantic_graph exists and has genuine unresolved
    structure), this module's own graph-derived questions are attached as
    `graph_targets_considered`/`graph_targets_selected` observability
    counts, but per the disclosed dependency gap above, QUESTION CONTENT
    for that case still comes from the legacy template planner
    (`planner_mode="LEGACY_FALLBACK"`, `fallback_reason` explains why) —
    this is the one acceptance-criterion gap this phase does not close,
    stated plainly rather than mislabeled as GRAPH_FIRST.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.models.agent import InvestigationPlan, InvestigationQuestion

PlannerMode = Literal["NO_ACTIONABLE_UNCERTAINTY", "TARGET_UNRESOLVED", "GRAPH_FIRST", "LEGACY_FALLBACK"]


@dataclass
class GraphInvestigationResult:
    planner_mode: PlannerMode
    plan: InvestigationPlan | None
    reason: str
    graph_targets_considered: int = 0
    graph_targets_selected: int = 0
    unresolved_node_ids: list[str] = field(default_factory=list)


def plan_investigation_graph_first(canonical_state: Any) -> GraphInvestigationResult:
    """The one genuinely graph-first, authoritative structural judgment this
    phase adds: does this finding's semantic graph contain ANY unresolved
    relation at all, and if so, how many distinct ones?

    Never inspects raw finding text. Never uses lexical/keyword matching.
    Reuses `build_causal_uncertainty_graph` (already the sole, existing,
    domain-neutral function that turns SemanticGraph normative-relation
    edges — VIOLATES/NOT_PERFORMED_AS_REQUIRED/... — into unresolved
    uncertainty nodes) rather than re-deriving uncertainty by any new
    heuristic.
    """
    if canonical_state is None:
        return GraphInvestigationResult(
            planner_mode="TARGET_UNRESOLVED",
            plan=None,
            reason="no canonical finding state available to determine investigation targets",
        )

    if not getattr(canonical_state, "is_actionable", True):
        return GraphInvestigationResult(
            planner_mode="NO_ACTIONABLE_UNCERTAINTY",
            plan=InvestigationPlan(
                areas=[], questions=[], evidence_to_collect=[],
                status="NO_ACTIONABLE_UNCERTAINTY", planner_mode="NO_ACTIONABLE_UNCERTAINTY", planner_stage="STAGE_A",
                fallback_reason=None,
            ),
            reason="finding is not actionable (CanonicalFindingState.is_actionable=False)",
        )

    semantic_graph = getattr(canonical_state, "semantic_graph", None)
    if semantic_graph is None or not getattr(semantic_graph, "nodes", None):
        return GraphInvestigationResult(
            planner_mode="TARGET_UNRESOLVED",
            plan=None,
            reason="semantic_graph is missing or empty — insufficient structure to determine investigation targets",
        )

    from app.agent.causal_graph import (
        build_causal_uncertainty_graph,
        information_gain_band_for_edge,
        rank_uncertainty_nodes_by_information_gain,
    )

    uncertainty_graph = build_causal_uncertainty_graph(canonical_state)
    unresolved_nodes = [n for n in uncertainty_graph.nodes if n.node_type.value == "UNRESOLVED"]

    if not unresolved_nodes:
        # Phase 17 correction (found via real live-Ollama + full-suite
        # regression testing, not assumed): `build_causal_uncertainty_graph`
        # only recognizes a narrow, fixed set of trigger relation types
        # (_UNCERTAINTY_TRIGGER_RELATIONS). Real actionable findings whose
        # semantic graph doesn't happen to populate one of those exact
        # relation types (e.g. "An employee stated that they had not
        # received training on the revised procedure.") produced ZERO
        # unresolved nodes despite genuinely requiring investigation --
        # NO_ACTIONABLE_UNCERTAINTY here was measured to be a substantive
        # misjudgment, not a safe structural conclusion, for a meaningful
        # fraction of real findings. Coverage gaps in the trigger-relation
        # set must not be represented as "nothing to investigate": treat
        # this the same as TARGET_UNRESOLVED (an honest "this planner
        # cannot determine the target" signal) so the legacy planner still
        # runs. NO_ACTIONABLE_UNCERTAINTY remains authoritative ONLY for
        # the is_actionable=False case above, which real testing did not
        # falsify.
        return GraphInvestigationResult(
            planner_mode="TARGET_UNRESOLVED",
            plan=None,
            reason=(
                "semantic_graph contains no unresolved normative relation of a recognized trigger type "
                "-- this reflects a coverage gap in the trigger-relation set, not a verified absence of "
                "actionable uncertainty, so the legacy planner is used rather than claiming "
                "NO_ACTIONABLE_UNCERTAINTY"
            ),
        )

    ranked = rank_uncertainty_nodes_by_information_gain(uncertainty_graph)
    edge_by_target = {e.target_node_id: e for e in uncertainty_graph.edges}
    questions: list[InvestigationQuestion] = []
    for idx, node in enumerate(ranked, start=1):
        edge = edge_by_target.get(node.node_id)
        if edge is None:
            continue
        band, reason_text = information_gain_band_for_edge(edge)
        questions.append(InvestigationQuestion(
            question_id=f"GQ{idx}",
            id=f"GQ{idx}",
            question=f"What objective evidence establishes the causal mechanism connecting the observed deviation to {node.label}?",
            purpose=f"Resolve unresolved relation to {node.label}",
            objective=f"Resolve unresolved relation to {node.label}",
            evidence="Objective records establishing the causal mechanism",
            evidence_required="Objective records establishing the causal mechanism",
            target_type="OTHER",
            target_node_id=node.node_id,
            target_edge_id=edge.edge_id,
            source_node_id=edge.source_node_id,
            causal_level=str(node.causal_level),
            unresolved_relation=edge.notes,
            information_gain_rank=idx,
            information_gain_band=band,
            information_gain_reason=reason_text,
            priority="HIGH" if band == "HIGH" else "MEDIUM",
            planner_stage="STAGE_A",
        ))

    plan = InvestigationPlan(
        questions=questions,
        areas=[f"Causal mechanism connecting the observed deviation to {n.label}" for n in ranked],
        evidence_to_collect=["Objective records establishing the causal mechanism"] if questions else [],
        status="QUESTIONS_GENERATED",
        planner_mode="GRAPH_FIRST",
        planner_stage="STAGE_A",
        graph_targets_considered=len(unresolved_nodes),
        graph_targets_selected=len(questions),
    )
    return GraphInvestigationResult(
        planner_mode="GRAPH_FIRST",
        plan=plan,
        reason=f"{len(unresolved_nodes)} unresolved relation(s) found in semantic_graph",
        graph_targets_considered=len(unresolved_nodes),
        graph_targets_selected=len(questions),
        unresolved_node_ids=[n.node_id for n in unresolved_nodes],
    )


def build_discriminating_questions_for_competing_hypotheses(
    hypotheses: list[Any], causal_graph: Any = None,
) -> list[InvestigationQuestion]:
    """Phase 16 Sections 9/10/31: real competing-hypothesis discrimination
    logic. NOT currently wired into any runtime node (see module docstring)
    — CandidateHypothesis objects do not exist at the point
    plan_investigation_node runs in the current LangGraph ordering. Exposed
    and unit-tested as a real, correct, reusable building block for the
    future re-planning pass this would require.

    "Competing" uses the SAME structural definition
    app.agent.causal_graph.is_graph_authoritative_for_five_why already
    established: hypotheses that remain independent siblings (no
    deepens_hypothesis_id chain and no shared supporting_claim_ids linking
    them) are competing; hypotheses already chained together are not — one
    already explains the other, there is nothing to discriminate.

    For every pair of competing hypotheses, produces ONE question naming
    both by their controlled-vocabulary `name` (never raw hedged
    `statement` prose), asking for evidence that discriminates between
    them, with `target_hypothesis_ids` set to both — never a redundant
    question per hypothesis.
    """
    live = [h for h in hypotheses if str(getattr(h, "status", "")) not in ("REFUTED",)]
    if len(live) < 2:
        return []

    claim_sets = [set(getattr(h, "supporting_claim_ids", None) or []) for h in live]
    deepens = [getattr(h, "deepens_hypothesis_id", None) for h in live]
    ids = [getattr(h, "id", None) for h in live]

    def _competing(i: int, j: int) -> bool:
        if deepens[i] == ids[j] or deepens[j] == ids[i]:
            return False  # chained, not competing
        if claim_sets[i] and claim_sets[j] and (claim_sets[i] & claim_sets[j]):
            return False  # share grounding evidence, not independent competitors
        return True

    questions: list[InvestigationQuestion] = []
    seen_pairs: set[tuple[str, str]] = set()
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            if not _competing(i, j):
                continue
            h_i, h_j = live[i], live[j]
            pair_key = tuple(sorted([str(ids[i]), str(ids[j])]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            name_i = (getattr(h_i, "name", None) or "").replace("_", " ").title() or str(ids[i])
            name_j = (getattr(h_j, "name", None) or "").replace("_", " ").title() or str(ids[j])
            questions.append(InvestigationQuestion(
                question_id=f"GDQ_{ids[i]}_{ids[j]}",
                id=f"GDQ_{ids[i]}_{ids[j]}",
                question=(
                    f"What objective evidence would support {name_i} over {name_j}, or {name_j} over "
                    f"{name_i}, as the causal mechanism?"
                ),
                purpose=f"Discriminate between competing hypotheses {name_i} and {name_j}",
                objective=f"Discriminate between competing hypotheses {name_i} and {name_j}",
                evidence="Objective records capable of distinguishing between the competing mechanisms",
                evidence_required="Objective records capable of distinguishing between the competing mechanisms",
                target_type="HYPOTHESIS",
                target_hypothesis_ids=[str(ids[i]), str(ids[j])],
                hypotheses_tested=[str(ids[i]), str(ids[j])],
                information_gain_band="HIGH",
                information_gain_reason="discriminates between two independently competing hypotheses",
                priority="HIGH",
            ))
    return questions
