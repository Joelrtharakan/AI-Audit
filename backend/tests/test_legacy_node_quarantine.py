"""Phase 5 -- legacy synthesis nodes are quarantined.

core_synthesis_node is the single authoritative implementation of RCA, 5-Why,
contributing factors, impact, CAPA and CA-draft. The older per-stage nodes
still exist (isolated guard-behaviour unit tests import them directly) but MUST
NOT be reachable from the production graph, and future accidental re-wiring
must fail a test.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

_LEGACY_MODULES = [
    "app/agent/nodes/rca.py",
    "app/agent/nodes/impact.py",
    "app/agent/nodes/capa.py",
    "app/agent/nodes/ca_draft_generator.py",
    "app/agent/nodes/impact_capa_parallel.py",
]
_BACKEND = Path(__file__).resolve().parent.parent


def test_production_graph_contains_only_the_authoritative_synthesis_node():
    from app.agent.graph import get_agent_graph

    nodes = set(get_agent_graph().get_graph().nodes.keys())
    forbidden = {"root_cause", "root_cause_analysis", "impact_assessment",
                 "capa_analysis", "capa", "impact", "generate_ca_draft",
                 "ca_draft_generator", "impact_capa_parallel"}
    assert nodes & forbidden == set(), f"legacy node wired into graph: {nodes & forbidden}"
    assert "core_synthesis" in nodes


def test_graph_module_does_not_import_legacy_synthesis_nodes():
    src = (_BACKEND / "app/agent/graph.py").read_text(encoding="utf-8")
    for banned in (
        "from app.agent.nodes.rca import",
        "from app.agent.nodes.impact import",
        "from app.agent.nodes.capa import",
        "from app.agent.nodes.ca_draft_generator import",
        "from app.agent.nodes.impact_capa_parallel import",
        "import app.agent.nodes.rca",
    ):
        assert banned not in src, f"graph.py imports a legacy node: {banned!r}"


def test_core_synthesis_does_not_import_legacy_nodes():
    src = (_BACKEND / "app/agent/nodes/core_synthesis.py").read_text(encoding="utf-8")
    for banned in ("from app.agent.nodes.rca import",
                   "from app.agent.nodes.impact import",
                   "from app.agent.nodes.capa import",
                   "from app.agent.nodes.ca_draft_generator import"):
        assert banned not in src


@pytest.mark.parametrize("rel", _LEGACY_MODULES)
def test_legacy_modules_carry_the_quarantine_marker(rel):
    head = (_BACKEND / rel).read_text(encoding="utf-8")[:400]
    assert "LEGACY NODE" in head and "NOT PART OF LIVE GRAPH" in head


def test_no_new_production_caller_of_legacy_node_functions():
    """Only test modules and the legacy modules themselves may reference the
    legacy node callables by name."""
    import app  # noqa: F401

    names = ("root_cause_node", "impact_assessment_node", "capa_analysis_node",
             "ca_draft_generator_node", "impact_capa_parallel_node")
    offenders = []
    for path in (_BACKEND / "app").rglob("*.py"):
        rel = path.relative_to(_BACKEND).as_posix()
        if rel in _LEGACY_MODULES:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            for n in names:
                # an actual import or call, not a prose mention in a comment/docstring
                if f"import {n}" in code or f"{n}(" in code or f"= {n}" in code:
                    offenders.append((rel, lineno, n))
    assert offenders == [], f"legacy node function used in production code: {offenders}"


def test_legacy_nodes_are_still_importable_for_unit_tests():
    # quarantine != deletion -- the guard-behaviour unit tests still need them
    from app.agent.nodes.rca import root_cause_node
    from app.agent.nodes.impact import impact_assessment_node
    from app.agent.nodes.capa import capa_analysis_node
    from app.agent.nodes.ca_draft_generator import ca_draft_generator_node

    for fn in (root_cause_node, impact_assessment_node, capa_analysis_node, ca_draft_generator_node):
        assert inspect.iscoroutinefunction(fn)
