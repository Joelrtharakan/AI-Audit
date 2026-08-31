"""Regression: `semantic_mode` / `canonical_semantic_status` must be declared
on the LangGraph state schema.

`understand_finding_node` returns these keys so the report can state plainly
whether the canonical LLM drove the semantics or the deterministic floor did
(spec §K). LangGraph drops any returned key that is not in the `AgentState`
TypedDict, so an omission here silently makes every report say
`semantic_mode = DETERMINISTIC` even when the canonical interpretation fully
succeeded. Direct-call node tests use a plain dict and never catch this; only
the compiled-graph production path does.
"""

from __future__ import annotations

from app.agent.state import AgentState


def test_state_schema_declares_semantic_provenance_keys():
    ann = AgentState.__annotations__
    assert "semantic_mode" in ann, (
        "AgentState must declare `semantic_mode` or LangGraph drops "
        "understand_finding_node's value and the report defaults to DETERMINISTIC"
    )
    assert "canonical_semantic_status" in ann


def test_understand_node_return_keys_are_all_in_schema():
    """Every key understand_finding_node writes must survive the graph schema."""
    import inspect

    from app.agent.nodes import understanding

    src = inspect.getsource(understanding.understand_finding_node)
    # the node's final `return {**state, ...}` includes these:
    for key in ("canonical_semantic_context", "canonical_semantic_status", "semantic_mode"):
        assert f'"{key}"' in src  # sanity: the node does write it
        assert key in AgentState.__annotations__, (
            f"understand_finding_node writes {key!r} but AgentState does not declare it"
        )
