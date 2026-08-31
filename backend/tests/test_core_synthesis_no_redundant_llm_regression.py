"""Pass 29 §2/§24: when the canonical semantic interpretation succeeds and
carries causal state, `core_synthesis_node` MUST NOT make its own synthesis
LLM call -- it consumes the canonical structured state directly. The downstream
root-cause / 5-Why / hypotheses report is still produced.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.config import get_settings
from app.models.agent import InvestigateRequest


_BASE = {
    "primary_deviation": None, "primary_deviation_confidence": "NOT_ESTABLISHED",
    "finding_subject": "safety guards", "observed_condition": "damaged safety guards",
    "epistemic_status": "VERIFIED", "root_cause_status": "NOT_ESTABLISHED",
    "candidate_hypotheses": [], "information_gaps": ["mechanism of guard damage"],
    "remediation_obligation": "IMMEDIATE_CORRECTION_ONLY",
    "remediation_activities": [{
        "action_id": "RA001", "activity": "Replace the two damaged safety guards",
        "disposition": "IMMEDIATE_CORRECTION", "depends_on_root_cause": False,
    }],
    "entities": [], "causal_claims": [], "stated_causal_alternatives": [],
    "evidence_boundaries": [], "unresolved_ambiguities": [],
}


class _CanonLLM:
    def __init__(self, payload):
        self._p = payload
    async def chat_completion(self, *a, **k):
        return json.dumps(self._p)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr(get_settings(), "canonical_semantic_llm_primary", True)
    monkeypatch.setattr(
        "app.services.canonical_finding_interpreter.get_llm_client",
        lambda **kw: _CanonLLM(_BASE),
    )
    # A spy that RAISES if core_synthesis tries to make its own LLM call.
    calls = {"core_synthesis": 0}

    def _boom(**kw):
        calls["core_synthesis"] += 1
        raise AssertionError(
            "core_synthesis made a synthesis LLM call although the canonical "
            "semantic interpretation already supplied the causal state"
        )

    for mod in ("investigation_planner",):
        monkeypatch.setattr(f"app.agent.nodes.{mod}.get_llm_client", lambda **kw: None)
    monkeypatch.setattr("app.agent.nodes.core_synthesis.get_llm_client", _boom)
    return calls


@pytest.mark.asyncio
async def test_core_synthesis_makes_zero_llm_calls_when_canonical_succeeds(flag_on):
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.understanding import understand_finding_node

    st = {
        "request": InvestigateRequest(
            finding_text="Two machines were found with damaged safety guards; both require replacement."
        ),
        "evidence_ledger": [], "trace": [], "errors": [],
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
    }
    st = await understand_finding_node(st)
    assert st.get("canonical_semantic_context") is not None
    assert st.get("semantic_mode") == "CANONICAL_LLM"

    st = await plan_investigation_node(st)
    st = await core_synthesis_node(st)          # must NOT raise (no _boom call)

    assert flag_on["core_synthesis"] == 0
    se = st.get("synthesis_execution") or {}
    assert se.get("source") == "CANONICAL_STATE"
    assert se.get("synthesis_llm_calls") == 0

    # downstream causal report still produced
    rc = st.get("root_cause")
    fw = st.get("five_why")
    assert rc is not None and str(rc.status.value if hasattr(rc.status, "value") else rc.status) == "NOT_ESTABLISHED"
    assert rc.leading_hypothesis in (None, "")          # no fabricated leading hypothesis
    assert fw is not None and len(fw.steps) >= 1        # 5-Why still generated (evidence boundary)
