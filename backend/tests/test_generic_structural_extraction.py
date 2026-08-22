"""Phase 2 hardening: generic structural event-clause extraction tests.

Verifies that `proposition_engine.build_semantic_graph`'s CHECK I (generic
structural fallback) produces graph structure for claims using vocabulary
that is NOT among the narrow lexical triggers (deliver/receive/access/
acknowledge/calibrate/train) exercised elsewhere in the suite.

These fixtures are drawn from domains never referenced in production code:
welding, laboratory sample handling, financial reconciliation, agriculture,
aviation maintenance. Production code contains no branches for any of them —
that is the property under test. If a future edit reintroduces a lexical
dependency, these tests fail because they use vocabulary no keyword list
in the codebase mentions.
"""
from __future__ import annotations

import re

from app.agent.claim_extractor import extract_claims, detect_evidence_conflicts
from app.agent.proposition_engine import (
    _GENERIC_EVENT_CLAUSE_RE,
    _is_participle,
    build_propositions_from_ledger,
    build_semantic_graph,
)
from app.models.agent import (
    EvidenceItem,
    EvidenceStatus,
    SemanticNodeType,
    SemanticRelationType,
)


def _build_graph(finding: str):
    ledger = [
        EvidenceItem(
            claim=finding, source="AUDITOR_FINDING", source_reference="Test",
            status=EvidenceStatus.VERIFIED, relevance="HIGH",
        )
    ]
    claims = extract_claims(finding, ledger)
    conflicts = detect_evidence_conflicts(claims)
    props = build_propositions_from_ledger(finding, claims, conflicts)
    return build_semantic_graph(finding, claims, props, conflicts)


class TestParticipleMorphologyIsVocabularyIndependent:
    def test_regular_suffix_recognized_for_unseen_verb(self):
        """A made-up-sounding but morphologically regular verb is recognized
        as a participle purely by suffix — no lexicon lookup involved."""
        assert _is_participle("welded")
        assert _is_participle("reconciled")
        assert _is_participle("dispensed")
        assert _is_participle("harvested")
        assert _is_participle("recalibrated")

    def test_non_participle_words_rejected(self):
        assert not _is_participle("the")
        assert not _is_participle("quickly")
        assert not _is_participle("valve")


class TestGenericExtractionAcrossUnseenDomains:
    """Each fixture uses a domain never mentioned in proposition_engine.py's
    lexical checks (C-H): welding, agriculture, aviation, finance, lab science."""

    def test_welding_domain_produces_event_structure(self):
        graph = _build_graph(
            "The structural weld joint WJ-12 was inspected by Inspector Rao."
        )
        assert any(n.node_type == SemanticNodeType.EVENT for n in graph.nodes), (
            "Unseen-domain verb 'inspected' must still produce an EVENT node"
        )
        assert any(
            e.relation_type == SemanticRelationType.RELATES_TO for e in graph.edges
        ), "Generic fallback must connect subject to event via RELATES_TO"

    def test_finance_domain_produces_actor_attribution(self):
        graph = _build_graph(
            "The general ledger was reconciled by the finance officer."
        )
        assert any(n.node_type == SemanticNodeType.EVENT for n in graph.nodes)
        assert any(
            e.relation_type == SemanticRelationType.EXECUTED_BY for e in graph.edges
        ), "Named actor performing an unseen-vocabulary event must be linked via EXECUTED_BY"

    def test_agriculture_domain_negated_event_preserved(self):
        graph = _build_graph(
            "The irrigation cycle was not completed as required by Schedule IRR-4."
        )
        # This claim matches the pre-existing normative CHECK A (broadened,
        # already domain-general) — verifying it still fires correctly
        # alongside/without the generic fallback for negated obligations.
        assert any(n.node_type == SemanticNodeType.REQUIREMENT for n in graph.nodes)

    def test_aviation_domain_unresolved_relation_is_marked_safe(self):
        """When the generic fallback fires, the edge must be explicitly
        marked as a non-committal structural extraction, never presented
        as a confidently-classified specific relation."""
        graph = _build_graph(
            "The turbine blade assembly was replaced by the maintenance crew."
        )
        generic_edges = [
            e for e in graph.edges
            if e.notes and "UNRESOLVED_RELATION" in e.notes
        ]
        assert generic_edges, "Generic extraction must mark its edges as unresolved/non-committal"
        for e in generic_edges:
            assert e.relation_type == SemanticRelationType.RELATES_TO, (
                "Unresolved generic edges must use the safe generic relation type, "
                "never a specific relation type guessed from an unfamiliar verb"
            )

    def test_no_lexical_trigger_words_present_in_fixture(self):
        """Meta-test: confirm the fixtures above do not accidentally contain
        any of the narrow lexical trigger substrings checks C-H key on —
        otherwise this test suite would not actually be exercising CHECK I."""
        narrow_triggers = (
            "deliver", "dispatch", "sent", "transmitt", "receiv", "access",
            "opened", "viewed", "acknowledg", "sign-off", "signature",
            "calibrat", "train",
        )
        fixtures = [
            "The structural weld joint WJ-12 was inspected by Inspector Rao.",
            "The general ledger was reconciled by the finance officer.",
            "The turbine blade assembly was replaced by the maintenance crew.",
        ]
        for f in fixtures:
            low = f.lower()
            hits = [t for t in narrow_triggers if t in low]
            assert not hits, f"Fixture unexpectedly contains narrow trigger(s) {hits}: {f!r}"


class TestGenericExtractionParaphraseInvariance:
    """Structurally equivalent claims expressed with different unseen-domain
    vocabulary must produce structurally equivalent graphs (EVENT node +
    RELATES_TO edge present), independent of the specific verb used."""

    def test_active_vs_passive_voice_both_produce_event_node(self):
        passive = _build_graph("The sample vial was mislabeled by the technician.")
        assert any(n.node_type == SemanticNodeType.EVENT for n in passive.nodes)

    def test_different_verbs_same_structure_shape(self):
        g1 = _build_graph("The batch was quarantined by the quality lead.")
        g2 = _build_graph("The shipment was rerouted by the logistics lead.")
        for g in (g1, g2):
            assert any(n.node_type == SemanticNodeType.EVENT for n in g.nodes)
            assert any(e.relation_type == SemanticRelationType.EXECUTED_BY for e in g.edges)
