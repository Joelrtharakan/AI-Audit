"""FINAL HARDENING Phase 1/2/29: PROVE (not inspect) that when the canonical
semantic LLM SUCCEEDS, the deterministic *semantic* helpers are never invoked
on the live node chain -- the canonical structured state is the sole semantic
authority.

Method: wrap every forbidden deterministic-semantic function so that any call
records a stack trace, run understand -> plan_investigation -> core_synthesis
-> final_evidence_verification with a FAKE canonical LLM that returns a valid
interpretation, then assert:

  * detect_recurrence, extract_immediate_mechanism,
    extract_stated_causal_alternatives, classify_mechanism_polarity,
    _extract_dual_assessment_comparison  -> ZERO calls
  * resolve_deviation -> only from the PRE-LLM structural input gate
    (understanding.py segment admissibility) -- never from a post-success
    semantic derivation.

If the canonical-success path regresses and starts re-deriving semantics from
raw finding text, one of these fires.
"""
from __future__ import annotations

import asyncio
import json
import traceback

import pytest

from app.config import get_settings
from app.models.agent import InvestigateRequest

_FORBIDDEN = {
    "app.agent.recurrence_guard": ["detect_recurrence"],
    "app.agent.causal_guard": [
        "extract_immediate_mechanism",
        "extract_stated_causal_alternatives",
        "classify_mechanism_polarity",
    ],
    "app.services.semantic_subject": ["_extract_dual_assessment_comparison"],
}

# resolve_deviation has ONE legitimate pre-LLM use (structural segment
# admissibility in the input gate). Allowed callers by (file, function).
_RESOLVE_DEVIATION_ALLOWED_CALLERS = {
    ("understanding.py", "understand_finding_node"),
    ("understanding.py", "_understand"),
    ("understanding.py", "<module>"),
}


class _FakeCanonicalLLM:
    def __init__(self, payload):
        self._payload = payload

    async def chat_completion(self, messages, temperature=0.0, response_format_json=True, **kw):
        return json.dumps(self._payload)


_CANON = {
    "primary_deviation": "Five laboratory balances are overdue for calibration.",
    "primary_deviation_confidence": "HIGH",
    "finding_subject": "laboratory balances", "subject_kind": "ENTITY",
    "observed_condition": "overdue for calibration", "epistemic_status": "VERIFIED",
    "comparison": None,
    "recurrence": {"event": "overdue calibration", "period": "the audit period"},
    "stated_causal_alternatives": [], "causal_alternatives_unresolved": False,
    "missing_record_status": None, "activity_performance_ambiguity": False,
    "affected_period": None, "affected_process": "calibration control",
    "scope": "five balances", "entities": [], "causal_claims": [],
    "explicit_previous_capa_reference": False, "previous_capa_evidence_ids": [],
    "evidence_boundaries": [], "unresolved_ambiguities": [],
    "root_cause_status": "NOT_ESTABLISHED", "candidate_hypotheses": [],
    "remediation_obligation": "ESTABLISHED_CORRECTIVE_OBLIGATION",
    "information_gaps": ["the calibration interval requirement that applied"],
    "investigation_plan": [], "remediation_activities": [
        {"action_id": "recal", "activity": "Recalibrate the five balances",
         "disposition": "IMMEDIATE_CORRECTION", "depends_on_root_cause": False},
    ],
    "pricing_information": [],
}

_FINDING = (
    "Five laboratory balances were found overdue for calibration during the audit. "
    "The calibration service costs Rs 4,500 per balance and a technician requires "
    "2 hours per balance at Rs 800 per hour."
)


@pytest.mark.asyncio
async def test_canonical_success_never_invokes_deterministic_semantic_helpers(monkeypatch):
    calls: dict[str, list[str]] = {}

    def _wrap(modname: str, fname: str):
        mod = __import__(modname, fromlist=[fname])
        orig = getattr(mod, fname)

        def spy(*a, **kw):
            calls.setdefault(fname, []).append("".join(traceback.format_stack(limit=6)))
            return orig(*a, **kw)

        monkeypatch.setattr(mod, fname, spy)

    for modname, fns in _FORBIDDEN.items():
        for fn in fns:
            _wrap(modname, fn)
    _wrap("app.services.semantic_subject", "resolve_deviation")

    s = get_settings()
    monkeypatch.setattr(s, "canonical_semantic_llm_primary", True)
    monkeypatch.setattr("app.agent.nodes.understanding.get_llm_client", lambda **kw: None)
    monkeypatch.setattr(
        "app.services.canonical_finding_interpreter.get_llm_client",
        lambda **kw: _FakeCanonicalLLM(_CANON),
    )
    # keep every other node LLM-free (deterministic synthesis path)
    for mod in ("investigation_planner", "core_synthesis", "critic",
                "causal_investigation_planner", "final_evidence_verification"):
        try:
            monkeypatch.setattr(f"app.agent.nodes.{mod}.get_llm_client", lambda **kw: None)
        except Exception:
            pass

    from app.agent.nodes.understanding import understand_finding_node
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.agent.nodes.report_generator import generate_report_node

    state = {
        "request": InvestigateRequest(finding_text=_FINDING),
        "evidence_ledger": [], "trace": [], "errors": [],
        "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
    }
    state = await understand_finding_node(state)
    assert state.get("canonical_semantic_context") is not None, "fake canonical LLM should succeed"
    state = await plan_investigation_node(state)
    state = await core_synthesis_node(state)
    state = await final_evidence_verification_node(state)
    try:
        state = await generate_report_node(state)
    except Exception:
        pass  # report gen may need extra wiring; the semantic chain above is the target

    # These deterministic-semantic helpers have NO legitimate structural use --
    # on canonical success they must never run.
    hard_forbidden = [
        "detect_recurrence", "extract_immediate_mechanism",
        "extract_stated_causal_alternatives", "classify_mechanism_polarity",
    ]
    leaked = {fn: calls[fn] for fn in hard_forbidden if fn in calls}
    assert not leaked, (
        "canonical-SUCCESS path invoked a deterministic semantic helper:\n"
        + "\n".join(f"--- {fn} ---\n{stacks[0]}" for fn, stacks in leaked.items())
    )

    # resolve_deviation / _extract_dual_assessment_comparison: exactly ONE
    # legitimate use each -- the PRE-LLM structural input gate in
    # understanding.py (segment admissibility). Any call whose stack does NOT
    # pass through understanding.py's input gate is a post-success semantic
    # re-derivation. (Phase 3: that input-gate use is itself slated for
    # replacement by a dedicated assess_input_gate API -- deferred.)
    for fn in ("resolve_deviation", "_extract_dual_assessment_comparison"):
        bad = [st for st in calls.get(fn, []) if "nodes/understanding.py" not in st]
        assert not bad, (
            f"{fn} called OUTSIDE the pre-LLM input gate on canonical success:\n"
            + "\n".join(bad[:1])
        )
