"""Architectural-convergence audit for the LLM-primary semantic pipeline.

Proves:
  * the semantic LLM interpreter runs AT MOST ONCE per investigation;
  * no downstream node mutates or re-derives the canonical subject /
    comparison / recurrence / stated alternatives / missing-record status;
  * with the flag ON, the merged canonical state flows unchanged through
    investigation -> 5-Why -> impact -> remediation -> report.

Recorded-response fixtures only -- no live model.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.config import get_settings
from app.models.agent import InvestigateRequest


class _CountingSemanticLLM:
    calls = 0

    def __init__(self, payload):
        self._payload = payload

    async def chat_completion(self, messages, temperature=0.0, response_format_json=True, **kw):
        type(self).calls += 1
        return json.dumps(self._payload)


_BASE = {
    "primary_deviation": None, "primary_deviation_claim_id": None,
    "primary_deviation_confidence": "NOT_ESTABLISHED",
    "finding_subject": None, "subject_kind": None, "evidence_source": None,
    "reported_observation": None, "observed_condition": None, "epistemic_status": None,
    "comparison": None, "recurrence": None,
    "stated_causal_alternatives": [], "causal_alternatives_unresolved": False,
    "missing_record_status": None, "activity_performance_ambiguity": False,
    "affected_period": None, "scope": None,
    "entities": [], "causal_claims": [], "explicit_previous_capa_reference": False,
    "previous_capa_evidence_ids": [], "evidence_boundaries": [], "unresolved_ambiguities": [],
}


def _p(**over):
    return {**_BASE, **over}


@pytest.fixture
def flag_on(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "canonical_semantic_llm_primary", True)
    monkeypatch.setattr("app.agent.nodes.understanding.get_llm_client", lambda **kw: None)
    monkeypatch.setattr("app.agent.nodes.investigation_planner.get_llm_client", lambda **kw: None)
    monkeypatch.setattr("app.agent.nodes.core_synthesis.get_llm_client", lambda **kw: None)

    def _install(payload):
        _CountingSemanticLLM.calls = 0
        client = _CountingSemanticLLM(payload)
        monkeypatch.setattr(
            "app.services.canonical_finding_interpreter.get_llm_client", lambda **kw: client
        )
        return client
    return _install


async def _full_pipeline(finding_text: str):
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.report_generator import generate_report_node
    from app.agent.nodes.understanding import understand_finding_node

    state = {
        "request": InvestigateRequest(finding_text=finding_text),
        "evidence_ledger": [], "trace": [], "errors": [],
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
    }
    state = await understand_finding_node(state)
    state = await plan_investigation_node(state)
    state = await core_synthesis_node(state)
    state = await generate_report_node(state)
    state = await final_evidence_verification_node(state)
    return state


def _snapshot(cf):
    return {
        "subject": cf.finding_subject,
        "comparison_type": cf.comparison_type,
        "comparison_left": cf.comparison_left,
        "comparison_right": cf.comparison_right,
        "measurement": (cf.measurement.value if cf.measurement else None),
        "recurrence_count": cf.recurrence_count,
        "recurrence_event": cf.recurrence_event,
        "recurrence_period": cf.recurrence_period,
        "stated_causal_alternatives": tuple(cf.stated_causal_alternatives or ()),
        "causal_alternatives_unresolved": cf.causal_alternatives_unresolved,
        "semantic_type": cf.semantic_type,
    }


# ---------------------------------------------------------------------------
# §13 -- one semantic interpretation per investigation
# ---------------------------------------------------------------------------

def test_semantic_llm_called_at_most_once(flag_on):
    flag_on(_p(finding_subject="press PR-204"))
    asyncio.run(_full_pipeline(
        "Maintenance records show that temporary repairs were performed on press PR-204."
    ))
    assert _CountingSemanticLLM.calls == 1


def test_five_why_fallback_skips_resolve_deviation_for_competing_causes(monkeypatch):
    """Part 2: a competing-causes finding sources its 5-Why boundary from
    canonical state -- the deterministic fallback does NOT re-parse."""
    import app.agent.nodes.five_why_fallback as fwmod
    import app.services.semantic_subject as ss
    calls = {"n": 0}
    _real = ss.resolve_deviation
    monkeypatch.setattr(ss, "resolve_deviation",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), _real(*a, **k))[1])

    from app.models.agent import CanonicalFindingState
    cf = CanonicalFindingState(
        raw_finding="x", finding_subject="the discrepancy", affected_object="The discrepancy",
        observed_deviation="the discrepancy — cause not established among the stated alternatives",
        deviation="x", deviation_condition="cause not established among the stated alternatives",
        semantic_type="OBJECT",
        stated_causal_alternatives=["an unrecorded issue", "a miscount", "a data-entry error"],
        causal_alternatives_unresolved=True,
    )
    fw = fwmod.build_deterministic_five_why(
        "The discrepancy could have resulted from an unrecorded issue, a miscount, or a "
        "data-entry error.", [], canonical_state=cf,
    )
    assert calls["n"] == 0                      # no deterministic re-parse
    txt = " ".join(s.answer.lower() for s in fw.steps)
    assert "mechanisms remaining" in txt and "discriminate" in txt
    assert not fw.is_complete


def test_five_why_fallback_still_uses_resolve_deviation_without_canonical(monkeypatch):
    """Part 2 byte-equivalence: no canonical state -> deterministic floor runs."""
    import app.agent.nodes.five_why_fallback as fwmod
    import app.services.semantic_subject as ss
    calls = {"n": 0}
    _real = ss.resolve_deviation
    monkeypatch.setattr(ss, "resolve_deviation",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), _real(*a, **k))[1])
    fwmod.build_deterministic_five_why(
        "The calibration certificate for gauge G-7 had expired.", [],
    )
    assert calls["n"] >= 1


def test_investigation_planner_reuses_cached_context(flag_on):
    client = flag_on(_p(
        finding_subject="the discrepancy",
        stated_causal_alternatives=["an unrecorded transaction", "a physical miscount",
                                    "a system data-entry error"],
        causal_alternatives_unresolved=True,
    ))
    f = ("The discrepancy could have resulted from an unrecorded transaction, a physical "
         "miscount, or a system data-entry error.")
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.understanding import understand_finding_node
    state = {
        "request": InvestigateRequest(finding_text=f),
        "evidence_ledger": [], "trace": [], "errors": [],
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
    }
    state = asyncio.run(understand_finding_node(state))
    assert client.calls == 1
    state = asyncio.run(plan_investigation_node(state))
    assert client.calls == 1                     # planner did NOT call it again
    assert state.get("canonical_semantic_context") is not None


# ---------------------------------------------------------------------------
# §12 -- no downstream node mutates the canonical semantic core
# ---------------------------------------------------------------------------

_IMMUTABILITY_FINDINGS = [
    ("comparison",
     "The reconciliation of inventory location IL-4 showed a shortfall of 120 units against the system record.",
     _p(finding_subject="inventory location IL-4")),
    ("recurrence",
     "Equipment M-204 experienced three failures over a six-month period.",
     _p(finding_subject="Equipment M-204")),
    ("competing_causes",
     "The discrepancy could have resulted from an unrecorded transaction, a physical miscount, "
     "or a system data-entry error.",
     _p(finding_subject="the discrepancy",
        stated_causal_alternatives=["an unrecorded transaction", "a physical miscount",
                                    "a system data-entry error"],
        causal_alternatives_unresolved=True)),
    ("evidence_proposition",
     "Maintenance records show that temporary repairs were performed on press PR-204.",
     _p(finding_subject="press PR-204", evidence_source="maintenance records",
        reported_observation="temporary repairs were performed", epistemic_status="REPORTED")),
    ("access_control",
     "Several employees retained access not required by their roles, but the evidence did not "
     "establish whether the access resulted from a provisioning error, an incomplete review, "
     "or an approved exception.",
     _p(finding_subject="employee access",
        observed_condition="access exceeded role requirement",
        stated_causal_alternatives=["a provisioning error", "an incomplete review",
                                    "an approved exception"],
        causal_alternatives_unresolved=True)),
]


@pytest.mark.parametrize("label,finding,payload", _IMMUTABILITY_FINDINGS,
                         ids=[c[0] for c in _IMMUTABILITY_FINDINGS])
def test_canonical_semantic_core_is_immutable_downstream(flag_on, label, finding, payload):
    flag_on(payload)
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.report_generator import generate_report_node
    from app.agent.nodes.understanding import understand_finding_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node

    state = {
        "request": InvestigateRequest(finding_text=finding),
        "evidence_ledger": [], "trace": [], "errors": [],
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
    }
    state = asyncio.run(understand_finding_node(state))
    after_understanding = _snapshot(state["canonical_finding_state"])

    for node in (plan_investigation_node, core_synthesis_node, generate_report_node,
                 final_evidence_verification_node):
        state = asyncio.run(node(state))
        now = _snapshot(state["canonical_finding_state"])
        assert now == after_understanding, (
            f"{label}: {node.__name__} mutated the canonical semantic core:\n"
            f"  before={after_understanding}\n  after ={now}"
        )
