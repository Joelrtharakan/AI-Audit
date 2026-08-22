"""Structural Output Quality Scorer (Section 21).

Produces a composite OutputQualityScore from 10 independent structural
dimensions. This scorer is deliberately decoupled from natural-language
fluency: a well-worded output that violates structural invariants still
receives a low score and triggers the fail-closed gate.

Score thresholds:
  >= 80  → PASS   (full investigation report may be emitted)
  60-79  → WARNING (report emitted with quality warnings in trace)
  < 60   → FAIL   (fail-closed gate fires; structured validation error returned)

No keyword, domain, regulation, or industry specificity is encoded here.
All scoring is based on graph structure, type correctness, and provenance.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_AFFECTED_OBJECT_ADMISSIBLE_NODE_TYPES = frozenset({
    "ENTITY", "RECORD", "ACTOR", "CONTROL", "PROCESS",
    "MEASUREMENT", "EVENT", "STATE", "OBSERVATION",
})
_INADMISSIBLE_AFFECTED_OBJECT_TYPES = frozenset({"REQUIREMENT", "ATTRIBUTE"})


def _score_semantic_fidelity(state: dict[str, Any]) -> tuple[int, int, str | None]:
    """D1 — Semantic fidelity: affected_object matches an admissible graph node (15 pts)."""
    canonical = state.get("canonical_finding_state")
    if not canonical:
        return 0, 15, "No canonical_finding_state in state"
    val = (getattr(canonical, "affected_object", None) or "").strip()
    if not val or val.upper().startswith(("UNKNOWN", "UNRESOLVED")):
        return 15, 15, None
    sem_graph = getattr(canonical, "semantic_graph", None)
    if not sem_graph:
        if len(val.split()) >= 1 and val.lower() not in ("process compliance", ""):
            return 10, 15, "Semantic graph absent; partial credit for non-generic affected_object"
        return 5, 15, "Semantic graph absent and affected_object is generic"
    node_map = {
        n.label.strip().lower(): str(getattr(n, "node_type", "ENTITY"))
        for n in (getattr(sem_graph, "nodes", None) or [])
    }
    node_type = node_map.get(val.lower())
    if node_type is None:
        return 8, 15, f"affected_object={val!r} has no matching node in semantic graph"
    if node_type in _INADMISSIBLE_AFFECTED_OBJECT_TYPES:
        return 0, 15, f"affected_object={val!r} is a {node_type} node (inadmissible type)"
    return 15, 15, None


def _score_role_accuracy(state: dict[str, Any]) -> tuple[int, int, str | None]:
    """D2 — Role accuracy: key semantic nodes have correct types (10 pts)."""
    canonical = state.get("canonical_finding_state")
    if not canonical:
        return 0, 10, "No canonical_finding_state"
    sem_graph = getattr(canonical, "semantic_graph", None)
    if not sem_graph or not getattr(sem_graph, "nodes", None):
        return 5, 10, "No semantic graph nodes to verify"
    nodes = sem_graph.nodes
    req_node_labels = {
        n.label.strip().lower()
        for n in nodes
        if str(getattr(n, "node_type", "")) == "REQUIREMENT"
    }
    entity_labels_that_are_reqs = [
        n.label
        for n in nodes
        if str(getattr(n, "node_type", "")) == "ENTITY"
        and n.label.strip().lower() in req_node_labels
    ]
    if entity_labels_that_are_reqs:
        return 3, 10, (
            f"Nodes {entity_labels_that_are_reqs!r} are typed as both ENTITY and REQUIREMENT "
            "(type-upgrade did not propagate correctly)"
        )
    return 10, 10, None


def _score_relation_preservation(state: dict[str, Any]) -> tuple[int, int, str | None]:
    """D3 — Relation preservation: normative and causal edges are distinct (10 pts)."""
    from app.models.agent import SemanticRelationType
    normative_types = SemanticRelationType.normative_relation_types()
    causal_rels = state.get("causal_relationships") or []
    contaminated = [
        str(getattr(cr, "relation_type", ""))
        for cr in causal_rels
        if str(getattr(cr, "relation_type", "")) in normative_types
    ]
    if contaminated:
        return 0, 10, f"Causal relations use normative types: {contaminated!r}"
    canonical = state.get("canonical_finding_state")
    sem_graph = getattr(canonical, "semantic_graph", None) if canonical else None
    if sem_graph and getattr(sem_graph, "edges", None) and getattr(sem_graph, "nodes", None):
        node_map = {n.id: n for n in sem_graph.nodes}
        for e in sem_graph.edges:
            rel = str(getattr(e, "relation_type", ""))
            if rel in ("VIOLATES", "SATISFIES", "LACKS_REQUIRED_ATTRIBUTE", "REQUIRES_ATTRIBUTE"):
                target = node_map.get(getattr(e, "target_id", ""))
                if target and str(getattr(target, "node_type", "")) not in (
                    "REQUIREMENT", "ATTRIBUTE", "CONTROL"
                ):
                    return 4, 10, (
                        f"Normative edge {rel!r} targets {target.node_type!r} node "
                        f"(label={target.label!r}); must target REQUIREMENT/ATTRIBUTE/CONTROL"
                    )
    return 10, 10, None


def _score_epistemic_correctness(state: dict[str, Any]) -> tuple[int, int, str | None]:
    """D4 — Epistemic correctness: no VERIFIED assigned to REPORTED claims (15 pts)."""
    canonical = state.get("canonical_finding_state")
    evidence_ledger = state.get("evidence_ledger", [])
    if not canonical:
        return 0, 15, "No canonical_finding_state"
    mech_status = getattr(canonical, "mechanism_status", None)
    verified_claims = [
        e for e in evidence_ledger
        # Phase 25 Rule 7: exact equality, not `.endswith("VERIFIED")`
        # ("UNVERIFIED".endswith("VERIFIED") is True in Python).
        if getattr(e, "status", None) == "VERIFIED"
    ]
    if mech_status == "VERIFIED" and not verified_claims:
        return 0, 15, "mechanism_status=VERIFIED but no VERIFIED claims in evidence ledger"
    rc = state.get("root_cause")
    if rc:
        rc_status = str(getattr(rc, "status", "")).rsplit(".", 1)[-1]
        if rc_status in ("ESTABLISHED", "SUPPORTED") and mech_status != "VERIFIED":
            return 5, 15, (
                f"root_cause.status={rc_status!r} but mechanism_status={mech_status!r}; "
                "causal promotion requires a verified mechanism"
            )
    return 15, 15, None


def _score_causal_validity(state: dict[str, Any]) -> tuple[int, int, str | None]:
    """D5 — Causal validity: causal graph uses only causal edge types (15 pts)."""
    from app.models.agent import SemanticRelationType
    normative_types = SemanticRelationType.normative_relation_types()
    causal_rels = state.get("causal_relationships") or []
    rc = state.get("root_cause")
    if not causal_rels and not rc:
        return 15, 15, None
    for cr in causal_rels:
        cr_type = str(getattr(cr, "relation_type", ""))
        if cr_type in normative_types:
            return 0, 15, f"CausalRelationship uses normative type {cr_type!r}"
    return 15, 15, None


def _score_question_relevance(state: dict[str, Any]) -> tuple[int, int, str | None]:
    """D6 — Question relevance: investigation questions target unresolved edges (10 pts)."""
    inv = state.get("investigation_plan")
    if not inv or not getattr(inv, "questions", None):
        return 10, 10, None
    questions = inv.questions
    if not questions:
        return 10, 10, None
    unanchored = [
        getattr(q, "question_id", f"Q{i}")
        for i, q in enumerate(questions)
        if not getattr(q, "target_proposition_id", None) and not getattr(q, "target_type", None)
    ]
    if len(unanchored) > len(questions) // 2:
        return 5, 10, f"More than half of investigation questions lack a target ({unanchored!r})"
    if unanchored:
        return 8, 10, f"Some investigation questions lack a target proposition: {unanchored!r}"
    return 10, 10, None


def _score_action_grounding(state: dict[str, Any]) -> tuple[int, int, str | None]:
    """D7 — Action grounding: CAPA actions traced to evidence (10 pts)."""
    capa = state.get("capa_analysis")
    if not capa:
        return 10, 10, None
    canonical = state.get("canonical_finding_state")
    if not canonical:
        return 5, 10, "CAPA present but no canonical_finding_state"
    from app.agent.grounding_guard import build_source_text, ungrounded_entities
    evidence_ledger = state.get("evidence_ledger", [])
    source_text = build_source_text(
        getattr(canonical, "raw_finding", ""),
        evidence_ledger,
        [],
    )
    ungrounded = []
    for action in (getattr(capa, "immediate_actions", None) or []):
        text = getattr(action, "action", None) or getattr(action, "description", None) or ""
        violations = ungrounded_entities(text, source_text)
        if violations:
            ungrounded.extend(violations)
    if ungrounded:
        return 3, 10, f"CAPA actions introduce ungrounded entities: {ungrounded!r}"
    return 10, 10, None


def _score_impact_grounding(state: dict[str, Any]) -> tuple[int, int, str | None]:
    """D8 — Impact grounding: no invented downstream harm (10 pts)."""
    impact = state.get("impact_assessment")
    if not impact:
        return 10, 10, None
    canonical = state.get("canonical_finding_state")
    if not canonical:
        return 5, 10, "Impact present but no canonical_finding_state"
    from app.agent.grounding_guard import build_source_text, ungrounded_entities
    evidence_ledger = state.get("evidence_ledger", [])
    source_text = build_source_text(
        getattr(canonical, "raw_finding", ""),
        evidence_ledger,
        [],
    )
    areas = getattr(impact, "areas", None) or []
    narrative = getattr(impact, "narrative", None) or ""
    ungrounded = []
    for area in areas:
        violations = ungrounded_entities(area, source_text)
        if violations:
            ungrounded.extend(violations)
    if narrative:
        violations = ungrounded_entities(narrative, source_text)
        if violations:
            ungrounded.extend(violations)
    if ungrounded:
        return 4, 10, f"Impact section references ungrounded entities: {ungrounded!r}"
    return 10, 10, None


def _score_capa_grounding(state: dict[str, Any]) -> tuple[int, int, str | None]:
    """D9 — CAPA grounding: conditional on causal epistemic certainty (10 pts)."""
    capa = state.get("capa_analysis")
    if not capa:
        return 10, 10, None
    canonical = state.get("canonical_finding_state")
    rc = state.get("root_cause")
    rc_status = str(getattr(rc, "status", "NOT_ESTABLISHED")).rsplit(".", 1)[-1] if rc else "NOT_ESTABLISHED"
    corrective_actions = getattr(capa, "corrective_actions", None) or []
    if rc_status == "NOT_ESTABLISHED" and corrective_actions:
        unconditional = [
            str(getattr(a, "action", "") or getattr(a, "description", ""))
            for a in corrective_actions
            if not getattr(a, "conditional", False) and not getattr(a, "pending_investigation", False)
        ]
        if unconditional:
            return 4, 10, (
                f"{len(unconditional)} corrective action(s) are not marked as conditional "
                "despite root cause being NOT_ESTABLISHED"
            )
    return 10, 10, None


def _score_traceability_completeness(state: dict[str, Any]) -> tuple[int, int, str | None]:
    """D10 — Traceability completeness: key output fields have source_claim_ids and validated trace matrix (5 pts)."""
    canonical = state.get("canonical_finding_state")
    report = state.get("report")
    if not canonical:
        return 0, 5, "No canonical_finding_state"

    # Check SemanticTraceabilityMatrix if present on report or canonical state
    matrix = getattr(report, "semantic_traceability", None) or getattr(canonical, "semantic_traceability", None)
    if matrix and getattr(matrix, "untraced_concepts", None):
        return 0, 5, f"Untraced concepts in SemanticTraceabilityMatrix: {matrix.untraced_concepts}"

    evidence_claims = getattr(canonical, "evidence_claims", None) or []
    if not evidence_claims:
        is_actionable = getattr(canonical, "is_actionable", True)
        if not is_actionable:
            return 5, 5, None
        return 2, 5, "No evidence_claims in canonical_finding_state despite actionable finding"
    claim_ids = {getattr(c, "claim_id", None) for c in evidence_claims if getattr(c, "claim_id", None)}
    sem_graph = getattr(canonical, "semantic_graph", None)
    if sem_graph and getattr(sem_graph, "nodes", None):
        nodes_with_claims = sum(
            1 for n in sem_graph.nodes
            if any(cid in claim_ids for cid in (getattr(n, "source_claim_ids", None) or []))
        )
        total_nodes = len(sem_graph.nodes)
        if total_nodes > 0 and nodes_with_claims == 0:
            return 2, 5, "No semantic graph nodes have source_claim_ids set"
        if total_nodes > 0 and nodes_with_claims < total_nodes // 2:
            return 3, 5, (
                f"Only {nodes_with_claims}/{total_nodes} semantic nodes have source_claim_ids"
            )
    return 5, 5, None


def _score_causal_graph_integrity(state: dict[str, Any]) -> tuple[int, int, str | None]:
    """D11 — Causal graph integrity (Phase 3, 10 pts): checks the REAL
    runtime state["causal_graph"] (app.models.agent.CausalGraph), not the
    always-empty legacy state["causal_relationships"] key D5/D3 above check.
    A graph with only an OBSERVED_DEVIATION node and zero edges is a valid,
    full-score state (evidence boundary reached honestly) — this dimension
    penalizes structural defects, not shallow depth."""
    from app.models.agent import CausalGraphEdgeStatus
    cg = state.get("causal_graph")
    if cg is None:
        # No causal graph was built at all for this run (e.g. no root_cause
        # yet, or construction failed) — neutral, not penalized, since not
        # every execution path reaches final_evidence_verification_node.
        return 10, 10, None
    node_ids = {n.node_id for n in cg.nodes}
    for e in cg.edges:
        if e.source_node_id == e.target_node_id:
            return 0, 10, f"Causal graph edge {e.edge_id} is a self-loop"
        if e.source_node_id not in node_ids or e.target_node_id not in node_ids:
            return 0, 10, f"Causal graph edge {e.edge_id} references a non-existent node"
        if e.status == CausalGraphEdgeStatus.VERIFIED and not e.evidence_ids:
            from app.models.agent import EpistemicSource
            if e.provenance != EpistemicSource.OBJECTIVE_RECORD:
                return 3, 10, f"Edge {e.edge_id} is VERIFIED with no evidence_ids and non-objective provenance"
    # Orphan check: any non-deviation node with zero incident edges is a
    # dangling candidate that should not silently exist without being
    # surfaced as unresolved — informational deduction only (not a defect,
    # since Section 4/8 explicitly allow unresolved candidate nodes to exist
    # without an edge), so this stays a small deduction, not a failure.
    incident = {e.source_node_id for e in cg.edges} | {e.target_node_id for e in cg.edges}
    orphans = [n for n in cg.nodes if n.node_type.value != "OBSERVED_DEVIATION" and n.node_id not in incident]
    if orphans and len(orphans) == len(cg.nodes) - 1 and len(cg.nodes) > 1:
        return 7, 10, f"All {len(orphans)} non-deviation causal node(s) are unconnected (no licensed edges yet)"
    return 10, 10, None


def _score_capa_graph_grounding(state: dict[str, Any]) -> tuple[int, int, str | None]:
    """D12 — CAPA graph grounding (Phase 5, 5 pts). Delegates to
    INV-CGRAPH-006 (app.agent.invariants) rather than duplicating its logic —
    a Phase 4 version of this check independently re-implemented (and
    mis-implemented) the same rule; Phase 5 fixed the root bug in the
    invariant and this now reuses that single corrected implementation."""
    from app.agent.invariants import _check_capa_definitive_action_requires_verified_causal_root
    is_valid, reason = _check_capa_definitive_action_requires_verified_causal_root(state)
    if is_valid:
        return 5, 5, None
    return 0, 5, reason


def _score_investigation_question_graph_grounding(state: dict[str, Any]) -> tuple[int, int, str | None]:
    """D13 — Investigation question graph grounding (Phase 4, 5 pts):
    informational signal, NOT a hard gate. Prototyped as an invariant
    (INV-CGRAPH-007-equivalent) and demoted after it fired on legitimate
    pipeline runs where the graph-grounded question path (Phase 4 Section
    3/4) has known coverage gaps (LLM-authored planning path does not
    populate target_node_id at all yet)."""
    ug = state.get("causal_uncertainty_graph")
    inv = state.get("investigation_plan")
    if not ug or not ug.edges or not inv or not inv.questions:
        return 5, 5, None
    from app.models.agent import CausalGraphEdgeStatus
    unresolved = [e for e in ug.edges if e.status == CausalGraphEdgeStatus.UNKNOWN]
    if not unresolved:
        return 5, 5, None
    grounded = [q for q in inv.questions if getattr(q, "target_node_id", None)]
    if not grounded:
        return 2, 5, (
            f"{len(unresolved)} unresolved causal-uncertainty edge(s) exist but zero "
            "investigation questions are graph-grounded (informational)"
        )
    return 5, 5, None


_SCORERS = [
    ("Semantic fidelity", _score_semantic_fidelity, 13),
    ("Role accuracy", _score_role_accuracy, 5),
    ("Relation preservation", _score_relation_preservation, 9),
    ("Epistemic correctness", _score_epistemic_correctness, 13),
    ("Causal validity", _score_causal_validity, 5),
    ("Question relevance", _score_question_relevance, 5),
    ("Action grounding", _score_action_grounding, 9),
    ("Impact grounding", _score_impact_grounding, 9),
    ("CAPA grounding", _score_capa_grounding, 5),
    ("Traceability completeness", _score_traceability_completeness, 5),
    ("Causal graph integrity", _score_causal_graph_integrity, 12),
    ("CAPA graph grounding", _score_capa_graph_grounding, 5),
    ("Investigation question graph grounding", _score_investigation_question_graph_grounding, 5),
]
assert sum(w for _, _, w in _SCORERS) == 100, "Dimension weights must sum to 100"


def compute_output_quality_score(state: dict[str, Any]) -> "OutputQualityScore":
    """Compute the composite structural quality score for the current agent state.

    Called by final_evidence_verification_node as the Section 21/22 fail-closed gate.
    """
    from app.models.agent import OutputQualityDimension, OutputQualityScore

    dimensions: list[OutputQualityDimension] = []
    total_score = 0
    blocker_violations: list[str] = []
    critical_violations: list[str] = []

    for name, scorer_fn, max_score in _SCORERS:
        try:
            score, max_s, reason = scorer_fn(state)
            score = max(0, min(score, max_s))
            passed = score >= max_s
            if not passed and reason:
                if score == 0:
                    blocker_violations.append(f"{name}: {reason}")
                else:
                    critical_violations.append(f"{name}: {reason}")
            dimensions.append(OutputQualityDimension(
                name=name,
                score=score,
                max_score=max_s,
                passed=passed,
                reason=reason,
            ))
            total_score += score
        except Exception as exc:
            logger.warning("Quality scorer %r failed: %s", name, exc)
            dimensions.append(OutputQualityDimension(
                name=name,
                score=0,
                max_score=max_score,
                passed=False,
                reason=f"Scorer error: {exc}",
            ))
            critical_violations.append(f"{name}: scorer raised {type(exc).__name__}: {exc}")

    if total_score >= 80:
        grade = "PASS"
    elif total_score >= 60:
        grade = "WARNING"
    else:
        grade = "FAIL"

    return OutputQualityScore(
        dimensions=dimensions,
        total_score=total_score,
        max_possible=100,
        grade=grade,
        blocker_violations=blocker_violations,
        critical_violations=critical_violations,
        computed_at=datetime.now(tz=timezone.utc).isoformat(),
    )
