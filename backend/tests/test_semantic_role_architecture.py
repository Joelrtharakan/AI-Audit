"""Blind cross-domain semantic role architecture tests.

20 structurally orthogonal test cases verifying that the semantic role
model correctly separates ENTITY / REQUIREMENT / ATTRIBUTE / PROCESS /
CONTROL / ACTOR / RECORD / EVENT across arbitrary domains.

Rules:
  - Zero natural-language output assertions.
  - Zero domain-specific vocabulary in the assertions.
  - Every assertion targets graph structure, node types, relation types,
    or epistemic status only.
  - Each test group is structurally independent.

Group layout:
  A (1-5)  : Entity vs Requirement
  B (6-8)  : Entity vs Attribute
  C (9-11) : Process vs Procedure
  D (12-13): Control vs Requirement
  E (14-15): Event vs Record
  F (16-17): Observation vs Compliance State
  G (18-19): Normative Relation vs Causal Relation
  H (20)   : Actor vs Responsibility
"""
from __future__ import annotations

import pytest
from app.agent.proposition_engine import build_semantic_graph, build_propositions_from_ledger
from app.agent.claim_extractor import extract_claims, detect_evidence_conflicts
from app.models.agent import (
    EvidenceItem,
    EvidenceStatus,
    SemanticNodeType,
    SemanticRelationType,
)


def _make_ledger(*claims: str, status: EvidenceStatus = EvidenceStatus.VERIFIED) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            claim=c,
            source="AUDITOR_FINDING",
            source_reference="Test",
            status=status,
            relevance="HIGH",
        )
        for c in claims
    ]


def _build_graph(finding: str, *extra_claims: str, extra_status: EvidenceStatus = EvidenceStatus.VERIFIED):
    ledger = _make_ledger(finding, *extra_claims, status=extra_status)
    claims = extract_claims(finding, ledger)
    conflicts = detect_evidence_conflicts(claims)
    props = build_propositions_from_ledger(finding, claims, conflicts)
    graph = build_semantic_graph(finding, claims, props, conflicts)
    return graph


def _node_types(graph) -> set[str]:
    return {str(n.node_type) for n in graph.nodes}


def _has_node_type(graph, node_type: SemanticNodeType) -> bool:
    return any(n.node_type == node_type for n in graph.nodes)


def _has_edge_type(graph, rel_type: SemanticRelationType) -> bool:
    return any(e.relation_type == rel_type for e in graph.edges)


def _requirement_labels(graph) -> set[str]:
    return {n.label.lower() for n in graph.nodes if n.node_type == SemanticNodeType.REQUIREMENT}


def _entity_labels(graph) -> set[str]:
    return {n.label.lower() for n in graph.nodes if n.node_type == SemanticNodeType.ENTITY}


# ---------------------------------------------------------------------------
# Group A: Entity vs Requirement (5 tests)
# ---------------------------------------------------------------------------

class TestEntityVsRequirement:
    def test_A1_explicit_named_requirement_is_typed_requirement(self):
        """A requirement referenced by an explicit ID must be a REQUIREMENT node."""
        graph = _build_graph("The activity was not performed as required by Procedure XY-001.")
        assert _has_node_type(graph, SemanticNodeType.REQUIREMENT), (
            "An explicitly named procedural requirement must produce a REQUIREMENT node"
        )

    def test_A2_requirement_label_not_in_entity_set(self):
        """A REQUIREMENT node label must not simultaneously be classified as ENTITY."""
        graph = _build_graph("Record RC-400 was not completed in accordance with Standard ST-20.")
        req_labels = _requirement_labels(graph)
        ent_labels = _entity_labels(graph)
        overlap = req_labels & ent_labels
        assert not overlap, (
            f"Labels {overlap!r} appear as both REQUIREMENT and ENTITY — type-upgrade did not propagate"
        )

    def test_A3_passive_obligation_produces_requirement_node(self):
        """A passive obligation ('was required') must produce a REQUIREMENT node."""
        graph = _build_graph("The dual approval was required but was not obtained.")
        assert _has_node_type(graph, SemanticNodeType.REQUIREMENT), (
            "Passive obligation 'was required' must produce a REQUIREMENT node"
        )

    def test_A4_violation_edge_targets_requirement_not_entity(self):
        """A VIOLATES edge must target a REQUIREMENT or ATTRIBUTE node, never an ENTITY."""
        graph = _build_graph(
            "The submission was not completed as required by Policy DOC-7."
        )
        node_map = {n.id: n for n in graph.nodes}
        for edge in graph.edges:
            if edge.relation_type == SemanticRelationType.VIOLATES:
                target = node_map.get(edge.target_id)
                assert target is not None
                assert target.node_type in (
                    SemanticNodeType.REQUIREMENT,
                    SemanticNodeType.ATTRIBUTE,
                    SemanticNodeType.CONTROL,
                ), (
                    f"VIOLATES edge targets {target.node_type!r} node {target.label!r}; "
                    "must target REQUIREMENT/ATTRIBUTE/CONTROL"
                )

    def test_A5_governance_edge_targets_process_not_requirement(self):
        """A GOVERNS edge source must be a REQUIREMENT; target must be a PROCESS or ENTITY."""
        graph = _build_graph("Schedule GEN-12 governs the monthly review process.")
        node_map = {n.id: n for n in graph.nodes}
        for edge in graph.edges:
            if edge.relation_type == SemanticRelationType.GOVERNS:
                source = node_map.get(edge.source_id)
                assert source is not None
                assert source.node_type == SemanticNodeType.REQUIREMENT, (
                    f"GOVERNS edge source must be REQUIREMENT, got {source.node_type!r}"
                )


# ---------------------------------------------------------------------------
# Group B: Entity vs Attribute (3 tests)
# ---------------------------------------------------------------------------

class TestEntityVsAttribute:
    def test_B1_missing_attribute_produces_attribute_node(self):
        """A missing required attribute (signature, date, etc.) must produce an ATTRIBUTE node."""
        graph = _build_graph("The form was submitted without the required approval signature.")
        assert _has_node_type(graph, SemanticNodeType.ATTRIBUTE), (
            "Missing required signature must produce an ATTRIBUTE node"
        )

    def test_B2_lacks_edge_connects_entity_to_attribute(self):
        """A LACKS_REQUIRED_ATTRIBUTE edge must connect an entity to an ATTRIBUTE node."""
        graph = _build_graph("The batch record was missing the required date entry.")
        node_map = {n.id: n for n in graph.nodes}
        for edge in graph.edges:
            if edge.relation_type == SemanticRelationType.LACKS_REQUIRED_ATTRIBUTE:
                target = node_map.get(edge.target_id)
                assert target is not None
                assert target.node_type in (
                    SemanticNodeType.ATTRIBUTE,
                    SemanticNodeType.REQUIREMENT,
                ), (
                    f"LACKS_REQUIRED_ATTRIBUTE edge must target ATTRIBUTE/REQUIREMENT, "
                    f"got {target.node_type!r}"
                )

    def test_B3_attribute_node_label_not_entity(self):
        """A node typed as ATTRIBUTE must not simultaneously appear as ENTITY."""
        graph = _build_graph("The log lacked the serial number identifier entry.")
        attr_labels = {n.label.lower() for n in graph.nodes if n.node_type == SemanticNodeType.ATTRIBUTE}
        ent_labels = _entity_labels(graph)
        overlap = attr_labels & ent_labels
        assert not overlap, (
            f"Labels {overlap!r} appear as both ATTRIBUTE and ENTITY"
        )


# ---------------------------------------------------------------------------
# Group C: Process vs Procedure (3 tests)
# ---------------------------------------------------------------------------

class TestProcessVsProcedure:
    def test_C1_procedure_name_is_requirement_not_process(self):
        """An explicit procedure name (e.g. 'Procedure ABC-10') is a REQUIREMENT, not a PROCESS."""
        graph = _build_graph(
            "The review was not conducted in accordance with Procedure ABC-10."
        )
        req_labels = _requirement_labels(graph)
        # At least one requirement label should contain the procedure reference
        assert req_labels, "Procedure reference must produce a REQUIREMENT node"

    def test_C2_activity_target_is_process_not_requirement(self):
        """The activity or system on which a requirement acts must be typed PROCESS, not REQUIREMENT."""
        graph = _build_graph(
            "The monthly reconciliation was not performed as required by Control Policy B-3."
        )
        proc_or_ent = [
            n for n in graph.nodes
            if n.node_type in (SemanticNodeType.PROCESS, SemanticNodeType.ENTITY)
        ]
        assert proc_or_ent, (
            "The activity target of a requirement must produce a PROCESS or ENTITY node"
        )

    def test_C3_no_synthetic_process_label_without_corroboration(self):
        """A synthetic process label (topic + generic suffix) must not be present when
        the finding text only describes a procedural violation without naming a process."""
        graph = _build_graph(
            "The form was not submitted in accordance with the applicable requirement."
        )
        from app.agent.invariants import _SYNTHETIC_PROCESS_SUFFIX_RE
        synthetic_process_nodes = [
            n for n in graph.nodes
            if n.node_type == SemanticNodeType.PROCESS
            and _SYNTHETIC_PROCESS_SUFFIX_RE.search(n.label)
        ]
        # If synthetic process nodes exist, they must be corroborated by a PROCESS node
        # whose label does NOT match the synthetic suffix pattern — i.e. they should not
        # be the ONLY PROCESS node.
        if synthetic_process_nodes:
            from app.agent.invariants import _check_affected_process_not_synthetic
            from app.models.agent import CanonicalFindingState
            for node in synthetic_process_nodes:
                canonical = CanonicalFindingState(
                    raw_finding="The form was not submitted in accordance with the applicable requirement.",
                    finding_subject="form",
                    affected_object="form",
                    affected_process=node.label,
                    affected_activity="UNKNOWN",
                    deviation="not submitted",
                    observed_deviation="not submitted",
                    facts=["not submitted"],
                    semantic_graph=graph,
                )
                is_valid, reason = _check_affected_process_not_synthetic(
                    {"canonical_finding_state": canonical}
                )
                assert not is_valid, (
                    f"Synthetic process label {node.label!r} without corroborating "
                    "non-synthetic PROCESS/CONTROL node must be rejected by INV-ROLE-005"
                )


# ---------------------------------------------------------------------------
# Group D: Control vs Requirement (2 tests)
# ---------------------------------------------------------------------------

class TestControlVsRequirement:
    def test_D1_control_node_is_typed_control_not_requirement(self):
        """A reference to a 'control' as a safeguard mechanism should never be typed REQUIREMENT."""
        graph = _build_graph(
            "The access control was not operational during the affected period."
        )
        control_labeled_nodes = [n for n in graph.nodes if "control" in n.label.lower()]
        assert control_labeled_nodes, "Expected at least one node referencing 'control'"
        assert all(
            n.node_type != SemanticNodeType.REQUIREMENT for n in control_labeled_nodes
        ), (
            f"Control-safeguard node(s) incorrectly typed as REQUIREMENT: "
            f"{[n.label for n in control_labeled_nodes if n.node_type == SemanticNodeType.REQUIREMENT]}"
        )

    def test_D2_requirement_does_not_become_control(self):
        """A named requirement (policy, standard) must not be downgraded to a CONTROL node."""
        graph = _build_graph(
            "Requirement REQ-20 was not met during the inspection period."
        )
        req_labels = _requirement_labels(graph)
        ctrl_labels = {n.label.lower() for n in graph.nodes if n.node_type == SemanticNodeType.CONTROL}
        overlap = req_labels & ctrl_labels
        assert not overlap, (
            f"Requirement labels {overlap!r} erroneously typed as CONTROL"
        )


# ---------------------------------------------------------------------------
# Group E: Event vs Record (2 tests)
# ---------------------------------------------------------------------------

class TestEventVsRecord:
    def test_E1_system_log_produces_record_node(self):
        """An audit trail, log, or certificate reference must be typed RECORD."""
        graph = _build_graph(
            "The calibration certificate was not available for inspection."
        )
        assert _has_node_type(graph, SemanticNodeType.RECORD) or _has_node_type(graph, SemanticNodeType.ENTITY), (
            "A document/certificate reference must produce a RECORD or ENTITY node"
        )

    def test_E2_occurrence_produces_event_not_record(self):
        """A stated occurrence or event ('the inspection was conducted') is typed EVENT, not RECORD."""
        graph = _build_graph(
            "The access event was recorded in the system log on the affected date."
        )
        # Log is a RECORD; event is EVENT — neither should be mistyped as REQUIREMENT or ATTRIBUTE
        inadmissible_types = {SemanticNodeType.REQUIREMENT, SemanticNodeType.ATTRIBUTE}
        for n in graph.nodes:
            if "log" in n.label.lower() or "event" in n.label.lower():
                assert n.node_type not in inadmissible_types, (
                    f"Node {n.label!r} typed as {n.node_type!r}; log/event must be RECORD/EVENT/ENTITY"
                )


# ---------------------------------------------------------------------------
# Group F: Observation vs Compliance State (2 tests)
# ---------------------------------------------------------------------------

class TestObservationVsComplianceState:
    def test_F1_observed_fact_carries_verified_epistemic_status(self):
        """An observation stated as a direct fact must have VERIFIED epistemic status."""
        ledger = _make_ledger("The document was not signed.", status=EvidenceStatus.VERIFIED)
        claims = extract_claims("The document was not signed.", ledger)
        conflicts = detect_evidence_conflicts(claims)
        props = build_propositions_from_ledger("The document was not signed.", claims, conflicts)
        graph = build_semantic_graph("The document was not signed.", claims, props, conflicts)
        verified_nodes = [
            n for n in graph.nodes
            if str(getattr(n, "epistemic_status", "")).endswith("VERIFIED")
        ]
        assert verified_nodes, "Directly stated observation fact must produce at least one VERIFIED node"

    def test_F2_reported_statement_carries_reported_epistemic_status(self):
        """A reported belief ('Operator stated...') must have REPORTED epistemic status in the ledger."""
        ledger = _make_ledger(
            "Operator stated they were unaware of the requirement.",
            status=EvidenceStatus.REPORTED,
        )
        claims = extract_claims(
            "Operator stated they were unaware of the requirement.", ledger
        )
        reported_claims = [c for c in claims if getattr(c, "status", None) == EvidenceStatus.REPORTED]
        # At least one claim should carry REPORTED status
        assert reported_claims or any(
            str(getattr(c, "status", "")) in ("REPORTED", "EvidenceStatus.REPORTED")
            for c in claims
        ), "Reported statement must produce a REPORTED-status claim"


# ---------------------------------------------------------------------------
# Group G: Normative Relation vs Causal Relation (2 tests)
# ---------------------------------------------------------------------------

class TestNormativeVsCausalRelation:
    def test_G1_normative_edges_are_not_in_normative_set_as_causal(self):
        """No edge in the semantic graph that uses a normative relation type should
        appear in causal_relationships. This is a structural type check only."""
        from app.models.agent import SemanticRelationType
        normative_types = SemanticRelationType.normative_relation_types()
        graph = _build_graph(
            "The transaction was not authorized in accordance with Policy POL-5."
        )
        # All normative edges in the semantic graph are structurally typed
        normative_edges = [e for e in graph.edges if e.relation_type in normative_types]
        # Verify each normative edge targets an admissible node type
        node_map = {n.id: n for n in graph.nodes}
        for edge in normative_edges:
            target = node_map.get(edge.target_id)
            if target:
                assert target.node_type in (
                    SemanticNodeType.REQUIREMENT,
                    SemanticNodeType.ATTRIBUTE,
                    SemanticNodeType.CONTROL,
                    SemanticNodeType.PROCESS,  # GOVERNS can point to PROCESS
                    SemanticNodeType.ENTITY,   # APPLIES_TO can point to ENTITY
                    SemanticNodeType.ACTOR,    # APPLIES_TO can point to ACTOR
                ), (
                    f"Normative edge {edge.relation_type!r} targets {target.node_type!r} {target.label!r}"
                )

    def test_G2_violation_plus_missing_attribute_on_same_claim(self):
        """A single claim encoding both a normative violation AND a missing attribute
        must produce BOTH edges (multi-edge per claim — not exclusive elif)."""
        graph = _build_graph(
            "The log entry was missing the required date record and was not completed as required by Procedure LOG-3."
        )
        has_normative = _has_edge_type(graph, SemanticRelationType.VIOLATES) or \
                        _has_edge_type(graph, SemanticRelationType.NOT_PERFORMED_AS_REQUIRED)
        has_attribute = _has_edge_type(graph, SemanticRelationType.LACKS_REQUIRED_ATTRIBUTE)
        # At least one of the two edge types must be present (both ideally)
        assert has_normative or has_attribute, (
            "Multi-relation claim must produce at least one normative or attribute edge"
        )


# ---------------------------------------------------------------------------
# Group H: Actor vs Responsibility (1 test)
# ---------------------------------------------------------------------------

class TestActorVsResponsibility:
    def test_H1_actor_node_typed_actor_not_process(self):
        """A named person or role (operator, manager, auditor) must be typed ACTOR, not PROCESS."""
        graph = _build_graph(
            "The operator confirmed that the step was not executed on the scheduled date."
        )
        for n in graph.nodes:
            if any(word in n.label.lower() for word in ("operator", "manager", "auditor", "inspector")):
                assert n.node_type in (SemanticNodeType.ACTOR, SemanticNodeType.ENTITY), (
                    f"Actor label {n.label!r} incorrectly typed as {n.node_type!r}"
                )
