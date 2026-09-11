"""LLM-PRIMARY semantic architecture -- end-to-end (recorded-response).

Runs findings through understand_finding_node with
`canonical_semantic_llm_primary` ON and a FAKE semantic LLM client (no live
model). Proves the merged canonical state reaches downstream and that the
deterministic validator constrains bad LLM output. `resolve_deviation`
remains the floor for every field the LLM omits.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.config import get_settings
from app.models.agent import InvestigateRequest


class _FakeSemanticLLM:
    def __init__(self, payload):
        self._payload = payload

    async def chat_completion(self, messages, temperature=0.0, response_format_json=True, **kw):
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
def llm_primary(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "canonical_semantic_llm_primary", True)
    # segment-classification LLM off; only the semantic interpreter is faked
    monkeypatch.setattr("app.agent.nodes.understanding.get_llm_client", lambda **kw: None)

    def _install(payload):
        monkeypatch.setattr(
            "app.services.canonical_finding_interpreter.get_llm_client",
            lambda **kw: _FakeSemanticLLM(payload),
        )
    return _install


async def _understand(finding_text: str):
    from app.agent.nodes.understanding import understand_finding_node
    state = {
        "request": InvestigateRequest(finding_text=finding_text),
        "evidence_ledger": [], "trace": [], "errors": [],
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
    }
    return await understand_finding_node(state)


# ---------------------------------------------------------------------------

def test_llm_subject_wins_over_bad_floor(llm_primary):
    f = ("Several employees retained access not required by their roles, but the evidence did "
         "not establish whether the access resulted from a provisioning error, an incomplete "
         "review, or an approved exception.")
    llm_primary(_p(
        finding_subject="employee access", subject_kind="ENTITY",
        observed_condition="access exceeded role requirement",
        stated_causal_alternatives=["a provisioning error", "an incomplete review",
                                    "an approved exception"],
        causal_alternatives_unresolved=True,
    ))
    s = asyncio.run(_understand(f))
    cf = s["canonical_finding_state"]
    assert cf.finding_subject == "employee access"
    assert cf.subject_unresolved is False
    assert len(cf.stated_causal_alternatives) == 3
    assert cf.causal_alternatives_unresolved is True
    assert s["canonical_semantic_context"] is not None      # reused downstream


def test_llm_cause_subject_rejected_and_floor_also_unsafe_fails_closed(llm_primary):
    f = "An investigation invalidated an OOS result, but the record did not establish the assignable laboratory cause."
    llm_primary(_p(finding_subject="assignable laboratory cause", subject_kind="CAUSE"))
    s = asyncio.run(_understand(f))
    cf = s["canonical_finding_state"]
    subj = (cf.finding_subject or "").lower()
    assert "cause" not in subj or subj.startswith(("unknown", "unresolved", "finding subject not"))


def test_llm_omission_of_comparison_is_not_established_not_floor_guessed(llm_primary):
    # CONTRACT (Pass 37/38, final-hardening Phase 1): the canonical LLM is the
    # sole semantic authority on canonical success. When it OMITS a comparison,
    # that means NOT_ESTABLISHED -- the deterministic raw-text floor's guess is
    # NOT resurrected. (Superseded the pre-Pass-37 "floor conserves omission"
    # behaviour, which was a raw-text semantic re-derivation past LLM success.)
    f = "The reconciliation of inventory location IL-4 showed a shortfall of 120 units against the system record."
    llm_primary(_p(finding_subject="inventory location IL-4"))   # LLM drops the comparison
    s = asyncio.run(_understand(f))
    ctx = s["canonical_semantic_context"]
    from app.services.canonical_semantic_models import comparison_is_active
    assert not comparison_is_active(getattr(ctx, "comparison", None))
    cf = s["canonical_finding_state"]
    assert cf.semantic_type != "COMPARISON"


def test_llm_omission_of_recurrence_is_not_established_not_floor_guessed(llm_primary):
    # Same contract: LLM omission of recurrence -> NOT established on the
    # canonical-success path (no raw-text detect_recurrence re-derivation).
    f = "Equipment M-204 experienced three failures over a six-month period."
    llm_primary(_p(finding_subject="Equipment M-204"))
    s = asyncio.run(_understand(f))
    cf = s["canonical_finding_state"]
    assert not cf.recurrence_count
    ctx = s["canonical_semantic_context"]
    assert getattr(ctx, "recurrence", None) is None


def test_llm_manufactured_number_blocked_end_to_end(llm_primary):
    f = "The measured result differed from the approved value."
    llm_primary(_p(
        finding_subject="the measured result",
        comparison={"left": "measured result", "right": "approved value",
                    "reference": "approved value", "direction": "BELOW",
                    "magnitude": 9.9, "unit": "%"},
    ))
    s = asyncio.run(_understand(f))
    cf = s["canonical_finding_state"]
    assert cf.measurement is None                # 9.9 not in the finding text
    ctx = s["canonical_semantic_context"]
    assert ctx.comparison.direction == "MISMATCH"


def test_missing_record_not_promoted_end_to_end(llm_primary):
    f = "The activity was not documented, but it is unclear whether it was performed."
    llm_primary(_p(finding_subject="the activity", missing_record_status="ACTIVITY_NOT_PERFORMED"))
    s = asyncio.run(_understand(f))
    ctx = s["canonical_semantic_context"]
    assert ctx.missing_record_status == "ACTIVITY_NOT_RECORDED"
    assert ctx.activity_performance_ambiguity is True


def test_flag_off_is_noop(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "canonical_semantic_llm_primary", False)
    monkeypatch.setattr("app.agent.nodes.understanding.get_llm_client", lambda **kw: None)
    f = "The reconciliation of inventory location IL-4 showed a shortfall of 120 units against the system record."
    st = asyncio.run(_understand(f))
    assert st["canonical_semantic_context"] is None
    assert st["canonical_finding_state"].finding_subject == "inventory location IL-4"
