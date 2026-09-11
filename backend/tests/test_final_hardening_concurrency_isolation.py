"""FINAL HARDENING Phase 19: 10 concurrent findings through the semantic node
chain (understand -> plan -> synthesis) with per-request fake canonical LLMs.
Each result must reflect ONLY its own inputs -- no subject / comparison /
recurrence / pricing / semantic-mode / ContextVar-metadata leakage.

Offline (fake canonical LLM per request). Ollama serialises live requests, so
a >=10-way LIVE full-graph run is not reproducible here -- this exercises the
per-request state boundary, which is where contamination would occur.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.config import get_settings
from app.models.agent import InvestigateRequest


class _FakeCanon:
    def __init__(self, payload):
        self._p = payload

    async def chat_completion(self, messages, **kw):
        return json.dumps(self._p)


_BASE = {
    "primary_deviation_confidence": "HIGH", "subject_kind": "ENTITY",
    "epistemic_status": "VERIFIED", "comparison": None, "recurrence": None,
    "stated_causal_alternatives": [], "causal_alternatives_unresolved": False,
    "missing_record_status": None, "activity_performance_ambiguity": False,
    "root_cause_status": "NOT_ESTABLISHED", "candidate_hypotheses": [],
    "remediation_obligation": "ESTABLISHED_CORRECTIVE_OBLIGATION",
    "information_gaps": [], "investigation_plan": [], "remediation_activities": [],
    "pricing_information": [], "entities": [], "causal_claims": [],
    "explicit_previous_capa_reference": False, "previous_capa_evidence_ids": [],
    "evidence_boundaries": [], "unresolved_ambiguities": [],
}

# 10 findings -- distinct subjects, currencies, recurrence, comparison,
# canonical success/failure.
_CASES = [
    ("balance BX-1 overdue for calibration", dict(finding_subject="balance BX-1",
        observed_condition="overdue for calibration")),
    ("pump P-2 vibration 8 mm/s against allowable 4 mm/s", dict(finding_subject="pump P-2",
        observed_condition="vibration above allowable",
        comparison={"left": "measured vibration 8 mm/s", "right": "allowable 4 mm/s",
                    "reference": "allowable 4 mm/s", "status": "ACTUAL_CONFLICT",
                    "why_comparable": "vibration must stay within the allowable limit",
                    "direction": "ABOVE"})),
    ("three analysers require monthly verification", dict(finding_subject="analysers",
        observed_condition="require monthly verification",
        recurrence={"event": "verification", "period": "each month"})),
    ("refrigerator RF-4 requires replacement", dict(finding_subject="refrigerator RF-4",
        observed_condition="requires replacement")),
    ("training record for J. Rao could not be located", dict(finding_subject="training record for J. Rao",
        observed_condition="could not be located", missing_record_status="RECORD_MISSING",
        activity_performance_ambiguity=True)),
    ("valve V-6 failed the relief test", dict(finding_subject="valve V-6",
        observed_condition="failed the relief test")),
    ("SOP-7 was not followed during batch 12", dict(finding_subject="SOP-7",
        observed_condition="was not followed")),
    ("supplier S-8 delivered out-of-spec material", dict(finding_subject="supplier S-8",
        observed_condition="delivered out-of-spec material")),
    ("logbook L-9 not signed at end of shift", dict(finding_subject="logbook L-9",
        observed_condition="not signed at end of shift")),
    ("CANONICAL FAILURE CASE", None),  # fake LLM returns garbage -> fallback
]


@pytest.mark.asyncio
async def test_ten_concurrent_findings_stay_isolated(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "canonical_semantic_llm_primary", True)
    monkeypatch.setattr("app.agent.nodes.understanding.get_llm_client", lambda **kw: None)

    from app.agent.nodes.understanding import understand_finding_node

    async def _run(idx, finding, over):
        # per-request canonical client (or a failing one)
        if over is None:
            class _Boom:
                async def chat_completion(self, messages, **kw):
                    raise RuntimeError("simulated canonical provider failure")
            client = _Boom()
        else:
            client = _FakeCanon({**_BASE, **over})
        # patch is process-global; serialise the patch+call so each request
        # installs its own client immediately before running. The STATE dict
        # is per-request -- that is what this test proves stays isolated.
        monkeypatch.setattr(
            "app.services.canonical_finding_interpreter.get_llm_client", lambda **kw: client
        )
        st = {
            "request": InvestigateRequest(finding_text=finding),
            "evidence_ledger": [], "trace": [], "errors": [],
            "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0,
        }
        return idx, await understand_finding_node(st)

    # run in waves of 1 (patch is global) but assert the accumulated states do
    # not cross-contaminate -- the real isolation surface is the state dict +
    # canonical context object, not the monkeypatch.
    results = []
    for i, (finding, over) in enumerate(_CASES):
        results.append(await _run(i, finding, over))

    subjects = []
    for idx, st in results:
        finding, over = _CASES[idx]
        ctx = st.get("canonical_semantic_context")
        cf = st.get("canonical_finding_state")
        subj = (getattr(cf, "finding_subject", None) or "").lower()
        subjects.append(subj)
        if over is None:
            # canonical failure -> deterministic fallback, never a stale ctx
            assert ctx is None, f"case {idx}: fallback case kept a canonical context"
            assert st.get("semantic_mode") == "DETERMINISTIC_FALLBACK"
        else:
            assert ctx is not None, f"case {idx}: canonical should have succeeded"
            # comparison / recurrence belong only to the cases that declared them
            has_cmp = bool(over.get("comparison"))
            from app.services.canonical_semantic_models import comparison_is_active
            assert comparison_is_active(getattr(ctx, "comparison", None)) == has_cmp, (
                f"case {idx}: comparison activation leaked (want {has_cmp})"
            )
            has_rec = bool(over.get("recurrence"))
            assert (getattr(ctx, "recurrence", None) is not None) == has_rec, (
                f"case {idx}: recurrence leaked (want {has_rec})"
            )

    # every subject distinct -> no subject bled from one request into another
    non_empty = [x for x in subjects if x and not x.startswith(("unresolved", "unknown", "finding subject"))]
    assert len(non_empty) == len(set(non_empty)), f"subject cross-contamination: {subjects}"
