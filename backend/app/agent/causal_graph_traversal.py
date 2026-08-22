"""Phase 3: CausalGraph consumers — 5-Why grounding and graph traversal.

Two real, additive capabilities over the typed `CausalGraph` built by
`app.agent.causal_graph.build_causal_graph`:

1. `ground_five_why_steps` — retroactively attaches causal-graph node/edge
   IDs to the EXISTING prose-generated `FiveWhyStep` objects when a step's
   answer text can be matched to a real graph node. This never changes a
   step's question/answer/status (generated text stays generated text —
   the Section 26 firewall) — it only populates the traceability fields
   added to `FiveWhyStep` in this phase, and only when a real match exists.
   An unmatched step is left with those fields unset, honestly reporting
   "no causal-graph grounding available for this step" rather than
   fabricating one.

2. `build_graph_grounded_five_why` — a genuine graph TRAVERSAL (Section 13):
   walks outgoing edges from the OBSERVED_DEVIATION node and emits one Why
   step per licensed edge, entirely independent of the prose generator.
   HONEST LIMITATION: the current `CausalGraph` builder only creates
   deviation->hypothesis edges (hypotheses are not themselves chained to
   deeper hypotheses — that data does not exist upstream), so this
   traversal is at most depth 1 today. It does not fabricate a chain
   between independent hypotheses to reach 5 steps — Section 13 is explicit
   that 5 is a maximum, not a requirement, and stopping at the evidence
   boundary is correct behavior, not a defect.
"""
from __future__ import annotations

import re
from typing import Any

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "was", "were", "is", "are",
    "be", "been", "being", "not", "may", "might", "could", "with", "by", "as", "that", "this",
    "did", "does", "do", "it", "its", "why", "what", "which", "how",
})


def _significant_words(text: str | None) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in _STOPWORDS and len(w) > 3}


def _stem(word: str) -> str:
    return word[:6] if len(word) > 6 else word


def _stemmed(words: set[str]) -> set[str]:
    return {_stem(w) for w in words}


def _best_matching_node(text: str, causal_graph: Any) -> Any | None:
    """Find the CausalGraphNode whose label best overlaps `text`, requiring
    at least 2 overlapping significant-word stems to avoid spurious matches
    on a single common word."""
    text_words = _stemmed(_significant_words(text))
    if not text_words:
        return None
    best_node = None
    best_overlap = 1  # require > 1, i.e. at least 2
    for node in causal_graph.nodes:
        node_words = _stemmed(_significant_words(node.label))
        overlap = len(text_words & node_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_node = node
    return best_node


def ground_five_why_steps(five_why: Any, causal_graph: Any) -> int:
    """Attach real causal-graph IDs to matching FiveWhyStep objects in place.

    Returns the number of steps successfully grounded. Never modifies
    question/answer/status — only the additive traceability fields.

    Idempotent by construction: always clears any PRIOR grounding first. The
    critic retry loop can re-run final_evidence_verification_node, which
    rebuilds `causal_graph` from scratch each time (edge IDs like "CE1" are
    positional, not content-addressed, so they are not stable across
    rebuilds) while `five_why` may carry over from an earlier pass. Without
    this reset, a step grounded against a since-discarded graph would keep
    citing a causal_edge_id that no longer exists — exactly what
    INV-CAUSAL-004 exists to catch. Clearing first guarantees every
    surviving grounding is consistent with the CURRENT graph.
    """
    if not five_why or not getattr(five_why, "steps", None):
        return 0

    for step in five_why.steps:
        step.source_node_id = None
        step.target_node_id = None
        step.causal_edge_id = None
        step.evidence_ids = []
        step.proposition_ids = []
        step.causal_level = None
        step.provenance = None

    if not causal_graph or not causal_graph.edges:
        return 0

    node_by_id = {n.node_id: n for n in causal_graph.nodes}
    grounded = 0
    for step in five_why.steps:
        target_node = _best_matching_node(step.answer or "", causal_graph)
        if target_node is None:
            continue
        matching_edge = next(
            (e for e in causal_graph.edges if e.target_node_id == target_node.node_id),
            None,
        )
        if matching_edge is None:
            continue
        source_node = node_by_id.get(matching_edge.source_node_id)
        if source_node is None or source_node.node_id == target_node.node_id:
            continue
        step.source_node_id = source_node.node_id
        step.target_node_id = target_node.node_id
        step.causal_edge_id = matching_edge.edge_id
        step.evidence_ids = list(matching_edge.evidence_ids)
        step.proposition_ids = list(matching_edge.proposition_ids)
        step.causal_level = str(target_node.causal_level)
        step.provenance = str(matching_edge.provenance)
        grounded += 1
    return grounded


def render_causal_transition_answer(source_node: Any, target_node: Any, edge_status: Any) -> str:
    """Phase 11 Step 2/6: epistemically correct rendering of a causal
    graph transition.

    A 5-Why answer for a graph-derived step must never be the raw
    `CandidateHypothesis.statement`/node label copied verbatim for a
    non-VERIFIED edge — that statement can carry language ("a reported
    factor possibly contributed...") stronger than the edge's actual
    epistemic backing, which is exactly the failure mode Phase 10 measured
    (44 test failures against INV-5WHY-CAUSAL-001/002: an unverified
    hypothesis's own wording — including its own embedded modal hedges —
    presented as if it were the renderer's established explanation).

    Only VERIFIED edges render the target node's full `label` directly —
    the one case where doing so is honest (the graph has already licensed
    it as an established fact; see build_causal_graph's edge licensing).
    Every other status renders a structurally generic, domain-neutral
    boundary statement using `concept_ref` (a short, controlled-vocabulary
    noun phrase derived mechanically from the hypothesis's own `name`, NOT
    its prose `statement`) — this is what actually closes the Phase 10 gap:
    `label` may legitimately carry hedged prose because it's the LLM's own
    sentence, but embedding that prose inside a hedged template still
    leaks its modal words back out. `concept_ref` never carries modal
    vocabulary because it is a mechanical case/underscore transform of a
    short identifier, not generated prose. Falls back to `label` only if
    `concept_ref` is unavailable (e.g. a node not derived from a named
    hypothesis) — a real, disclosed residual gap for that case.
    """
    from app.models.agent import CausalGraphEdgeStatus

    if edge_status == CausalGraphEdgeStatus.VERIFIED:
        return target_node.label
    source_ref = getattr(source_node, "concept_ref", None) or source_node.label
    target_ref = getattr(target_node, "concept_ref", None) or target_node.label

    # Phase 12 Step 7: when the source node carries structured
    # comparison/measurement context (populated only from
    # CanonicalFindingState's own typed fields — never re-derived from
    # prose), reuse the SAME comparison-subtype-keyed mechanism-category
    # renderer the deterministic prose generator already uses, instead of
    # falling back to the generic template and losing quantitative
    # precision. Measurement precision is descriptive content ONLY here —
    # it does not change which branch fires; the epistemic gate above
    # (VERIFIED vs not) is unaffected by whether a measurement exists.
    if getattr(source_node, "comparison_type", None) and getattr(source_node, "comparison_left", None) and getattr(source_node, "comparison_right", None):
        from app.agent.causal_guard import build_causal_boundary_answer
        return build_causal_boundary_answer(
            comparison_type=source_node.comparison_type,
            comparison_left=source_node.comparison_left,
            comparison_right=source_node.comparison_right,
            comparison_left_qualifier=source_node.comparison_left_qualifier,
            comparison_subtype=source_node.comparison_subtype,
            measurement_value=source_node.measurement_value,
            measurement_unit=source_node.measurement_unit,
            measurement_qualifier=source_node.measurement_qualifier,
            observed_deviation=source_node.label,
        )

    if edge_status == CausalGraphEdgeStatus.REPORTED:
        return f"{target_ref} was reported as a contributing factor to {source_ref}."
    return (
        f"Available evidence does not establish whether {source_ref} resulted from "
        f"{target_ref} or another mechanism."
    )


def _graph_node_why_question(node: Any) -> str:
    """Build the "why" question for one graph-derived 5-Why step from the
    node's own structured comparison fields when present (Phase 12 Step 7,
    mirroring the same guard used for the answer), falling back to the
    generic occurrence-question template otherwise. Domain-general: the
    verb phrase is looked up by comparison_type, never hardcoded per
    finding."""
    if getattr(node, "comparison_type", None) and getattr(node, "comparison_left", None) and getattr(node, "comparison_right", None):
        from app.services.semantic_subject import format_comparison_why_question
        question = format_comparison_why_question(
            node.comparison_type, node.comparison_left, node.comparison_right,
            node.comparison_left_qualifier,
        )
        if question:
            return question
    return f"Why did the following occur: {node.label}?"


def build_graph_grounded_five_why(causal_graph: Any) -> Any | None:
    """Real SEQUENTIAL traversal of the CausalGraph, one hop at a time.

    Phase 8 rewrite: the previous version only ever looked at edges sourced
    directly from OBSERVED_DEVIATION, so a genuine multi-hop chain (Phase 7's
    EXPLICIT edges, Phase 4's EVIDENCE_CORRELATED re-parented edges) was
    never actually walked past depth 1 — a real gap caught while extending
    the 5-Why authority swap to fire on EVIDENCE_CORRELATED chains, not just
    EXPLICIT ones (see final_evidence_verification_node).

    Starting at the deviation, at each node this follows the SINGLE
    best-licensed outgoing edge (VERIFIED > REPORTED > POSSIBLE; ties broken
    by deepest causal_level, then edge_id for determinism) to the next node,
    and repeats from there — a genuine sequential walk down ONE causal path,
    never treating sibling/competing hypotheses as sequential Whys. Stops
    (silently, correctly) the moment no outgoing edge is licensed from the
    current node — Section F: "stop at the first unsupported causal
    transition." Visited nodes are tracked so a construction-time invariant
    violation (a cycle) cannot cause infinite traversal here, defense in
    depth on top of INV-CGRAPH-008.

    Returns None (not an empty analysis) when zero edges are licensed at
    all — the evidence boundary was reached immediately.
    """
    from app.models.agent import CausalGraphEdgeStatus, FiveWhyAnalysis, FiveWhyStep

    if not causal_graph or not causal_graph.nodes:
        return None

    node_by_id = {n.node_id: n for n in causal_graph.nodes}
    deviation_nodes = [n for n in causal_graph.nodes if n.node_type.value == "OBSERVED_DEVIATION"]
    if not deviation_nodes:
        return None
    deviation = deviation_nodes[0]

    _TRAVERSABLE = {
        CausalGraphEdgeStatus.VERIFIED,
        CausalGraphEdgeStatus.REPORTED,
        CausalGraphEdgeStatus.POSSIBLE,
    }
    _STATUS_RANK = {
        CausalGraphEdgeStatus.VERIFIED: 0,
        CausalGraphEdgeStatus.REPORTED: 1,
        CausalGraphEdgeStatus.POSSIBLE: 2,
    }
    _CAUSAL_LEVEL_ORDER = [
        "L0_OBSERVATION", "EVIDENCE_STATE", "L1_EVENT",
        "L2_IMMEDIATE_MECHANISM", "L2_REPORTED_MECHANISM",
        "L3_CONTRIBUTING_CAUSE", "L3_IMMEDIATE_MECHANISM",
        "L4_ROOT_CAUSE", "L5_SYSTEMIC_CAUSE",
    ]

    def _level_rank(node: Any) -> int:
        val = node.causal_level.value if hasattr(node.causal_level, "value") else str(node.causal_level)
        try:
            return _CAUSAL_LEVEL_ORDER.index(val)
        except ValueError:
            return len(_CAUSAL_LEVEL_ORDER)

    status_map = {
        CausalGraphEdgeStatus.VERIFIED: "VERIFIED",
        CausalGraphEdgeStatus.REPORTED: "REPORTED",
        # Phase 12 Step 9: a licensed-but-unconfirmed (POSSIBLE) causal edge
        # is an evidence gap, not a hypothesis awaiting a specific piece of
        # evidence -- "UNKNOWN" matches the epistemic status the same
        # condition already gets everywhere else in this module (the
        # EVIDENCE_BOUNDARY marker step below, and Phase 11's boundary
        # phrasing), so a POSSIBLE-licensed step and an unlicensed boundary
        # step read identically to a caller inspecting status alone.
        CausalGraphEdgeStatus.POSSIBLE: "UNKNOWN",
    }

    steps = []
    visited = {deviation.node_id}
    current_id = deviation.node_id
    stopped_at_boundary = False
    for i in range(1, 6):
        outgoing = [
            e for e in causal_graph.edges
            if e.source_node_id == current_id
            and e.status in _TRAVERSABLE
            and e.target_node_id not in visited
        ]
        if not outgoing:
            stopped_at_boundary = True
            break
        outgoing.sort(key=lambda e: (
            _STATUS_RANK.get(e.status, 99),
            -_level_rank(node_by_id.get(e.target_node_id, deviation)),
            e.edge_id,
        ))
        edge = outgoing[0]
        target = node_by_id.get(edge.target_node_id)
        if target is None:
            stopped_at_boundary = True
            break
        source_node = node_by_id.get(current_id, deviation)
        steps.append(FiveWhyStep(
            question=_graph_node_why_question(source_node),
            answer=render_causal_transition_answer(source_node, target, edge.status),
            status=status_map.get(edge.status, "UNKNOWN"),
            step_id=f"GW{i}",
            source_node_id=current_id,
            target_node_id=target.node_id,
            causal_edge_id=edge.edge_id,
            evidence_ids=list(edge.evidence_ids),
            proposition_ids=list(edge.proposition_ids),
            causal_level=str(target.causal_level),
            provenance=str(edge.provenance),
            boundary_status="TRANSITION",
        ))
        visited.add(target.node_id)
        current_id = target.node_id

    if not steps:
        return None

    # Phase 13 Step 8: a POSSIBLE-licensed edge already renders as status
    # "UNKNOWN" (see status_map above) -- that transition step's answer
    # ALREADY IS the evidence-boundary statement ("available evidence does
    # not establish..."). Appending a second, separate EVIDENCE_BOUNDARY
    # marker step after it does not add information; it restates the same
    # boundary as an extra step, which is exactly the padding INV-WHY-007
    # exists to catch (Phase 12 Step 10's 60-regression single-hop
    # widening was dominated by this one bug: a single POSSIBLE edge from
    # the deviation produced a 1-step UNKNOWN transition, then this block
    # unconditionally appended a 2nd UNKNOWN marker step). The last
    # emitted transition step already terminates the chain in that case --
    # relabel it EVIDENCE_BOUNDARY in place instead of duplicating it.
    if steps[-1].status == "UNKNOWN":
        steps[-1].boundary_status = "EVIDENCE_BOUNDARY"
        stopped_at_boundary = False  # already represented; nothing to append

    # Phase 11 Step 8: when the walk stopped because no further causal edge
    # is licensed (as opposed to hitting the 5-step cap with more real
    # edges still available), append ONE explicit terminal marker step so
    # "the chain legitimately ends here" is a real, inspectable structural
    # fact (status="UNKNOWN", boundary_status="EVIDENCE_BOUNDARY") rather
    # than only an implicit absence a caller has to infer from length.
    # Never fabricates a causal claim: no target_node_id/causal_edge_id.
    # This only fires when the LAST transition step was itself VERIFIED/
    # REPORTED (a real causal step happened) and the walk then ran out of
    # further licensed edges -- i.e. the boundary is genuinely one hop
    # PAST the last real transition, not the same hop restated.
    if stopped_at_boundary and len(steps) < 5:
        boundary_node = node_by_id.get(current_id, deviation)
        boundary_ref = getattr(boundary_node, "concept_ref", None) or boundary_node.label
        if getattr(boundary_node, "comparison_type", None) and getattr(boundary_node, "comparison_left", None) and getattr(boundary_node, "comparison_right", None):
            from app.agent.causal_guard import build_causal_boundary_answer
            boundary_answer = build_causal_boundary_answer(
                comparison_type=boundary_node.comparison_type,
                comparison_left=boundary_node.comparison_left,
                comparison_right=boundary_node.comparison_right,
                comparison_left_qualifier=boundary_node.comparison_left_qualifier,
                comparison_subtype=boundary_node.comparison_subtype,
                measurement_value=boundary_node.measurement_value,
                measurement_unit=boundary_node.measurement_unit,
                measurement_qualifier=boundary_node.measurement_qualifier,
                observed_deviation=boundary_node.label,
            )
        else:
            boundary_answer = f"Available evidence does not establish whether a deeper cause exists for {boundary_ref}."
        steps.append(FiveWhyStep(
            question=_graph_node_why_question(boundary_node),
            answer=boundary_answer,
            status="UNKNOWN",
            step_id=f"GW{len(steps) + 1}",
            source_node_id=current_id,
            target_node_id=None,
            causal_edge_id=None,
            causal_level=str(boundary_node.causal_level),
            boundary_status="EVIDENCE_BOUNDARY",
        ))

    return FiveWhyAnalysis(
        steps=steps,
        is_complete=False,
        status_note=(
            "Graph-grounded sequential traversal: each step follows one licensed causal-graph "
            "edge to the next node on a single path. Traversal stopped where no further edge "
            "is licensed from the current node — this is an evidence boundary, not an omission."
        ),
    )
