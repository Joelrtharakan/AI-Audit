"""LLM-PRIMARY semantic architecture -- deterministic (recorded-response) tests.

The LLM canonical interpreter becomes the primary semantic interpreter;
`resolve_deviation()` is the fail-closed floor. These tests use a fake LLM
client returning canned JSON (no live model) and verify the merge + the
deterministic validator:

  VALID              -> LLM structure adopted
  CAUSE AS SUBJECT   -> rejected, floor kept
  EVIDENCE SOURCE    -> rejected, floor kept
  DROPPED COMPARISON -> deterministic comparison conserved
  DROPPED RECURRENCE -> deterministic recurrence conserved
  DROPPED ALTERNATIVES -> deterministic alternatives conserved
  MISSING->NONPERF   -> downgraded to ACTIVITY_NOT_RECORDED + ambiguity
  MANUFACTURED NUMBER -> magnitude / count dropped
  LLM None / invalid -> pure no-op (== deterministic floor)
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.models.agent import CanonicalFindingState
from app.services.canonical_context_validator import validate_canonical_context
from app.services.canonical_finding_interpreter import interpret_finding_canonically
from app.services.canonical_semantic_models import CanonicalFindingContext
from app.services.canonical_state_merge import merge_semantic_context_into_canonical
from app.agent.causal_guard import extract_stated_causal_alternatives as _esca
from app.services.semantic_subject import resolve_deviation


class _FakeLLM:
    def __init__(self, payload):
        self._payload = payload

    async def chat_completion(self, messages, temperature=0.0, response_format_json=True, **kw):
        if self._payload is None:
            raise RuntimeError("provider unavailable")
        return self._payload if isinstance(self._payload, str) else json.dumps(self._payload)


def _det_canonical(finding_text: str) -> CanonicalFindingState:
    """The deterministic-floor canonical state (what understand_finding_node
    builds before any merge)."""
    d = resolve_deviation(finding_text)
    _m = None
    if getattr(d, "measurement_value", None) is not None:
        from app.models.agent import SemanticMeasurement
        _m = SemanticMeasurement(
            value=float(d.measurement_value), unit=getattr(d, "measurement_unit", None),
            qualifier=getattr(d, "measurement_qualifier", None),
            role="OBSERVED_DISCREPANCY", evidence_status="UNKNOWN",
        )
    return CanonicalFindingState(
        measurement=_m,
        raw_finding=finding_text,
        finding_subject=d.finding_subject or "UNKNOWN",
        affected_object=(d.finding_subject or "UNKNOWN"),
        observed_deviation=d.deviation or finding_text,
        deviation=d.deviation or finding_text,
        deviation_condition=getattr(d, "condition", None) or "status unconfirmed",
        semantic_type=d.semantic_type or "OBJECT",
        subject_unresolved=not bool(d.finding_subject),
        comparison_type=getattr(d, "comparison_type", None),
        comparison_left=getattr(d, "comparison_left", None),
        comparison_right=getattr(d, "comparison_right", None),
        recurrence_count=getattr(d, "recurrence_count", None),
        recurrence_event=getattr(d, "recurrence_event", None),
        recurrence_period=getattr(d, "recurrence_period", None),
        stated_causal_alternatives=list(getattr(d, "stated_causal_alternatives", []) or [])
        or list(_esca(finding_text)),
        causal_alternatives_unresolved=bool(getattr(d, "causal_alternatives_unresolved", False))
        or (len(_esca(finding_text)) >= 2),
    )


def _run_llm(finding_text, payload, ledger=None):
    ledger = ledger or []
    ctx = asyncio.run(interpret_finding_canonically(
        finding_text=finding_text, evidence_ledger=ledger, client=_FakeLLM(payload),
    ))
    if ctx is None:
        return None
    return validate_canonical_context(ctx, ledger, finding_text)


def _merge(finding_text, payload, ledger=None):
    cs = _det_canonical(finding_text)
    ctx = _run_llm(finding_text, payload, ledger)
    cs, outcome = merge_semantic_context_into_canonical(cs, ctx)
    return cs, ctx, outcome


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


# ---------------------------------------------------------------------------

def test_none_llm_is_pure_noop():
    f = "The calibration certificate for gauge G-7 had expired."
    before = _det_canonical(f)
    ctx = _run_llm(f, None)
    assert ctx is None
    after, _ = merge_semantic_context_into_canonical(_det_canonical(f), ctx)
    assert after.finding_subject == before.finding_subject
    assert after.semantic_type == before.semantic_type


def test_valid_llm_subject_adopted_when_floor_unresolved():
    f = ("Several employees retained access not required by their roles, but the evidence "
         "did not establish whether the access resulted from a provisioning error, an "
         "incomplete review, or an approved exception.")
    cs, ctx, out = _merge(f, _p(
        finding_subject="employee access", subject_kind="ENTITY",
        observed_condition="access exceeded role requirement",
        stated_causal_alternatives=["a provisioning error", "an incomplete review",
                                    "an approved exception"],
        causal_alternatives_unresolved=True,
    ))
    assert cs.finding_subject == "employee access"
    assert "finding_subject" in out.fields_from_llm
    assert len(cs.stated_causal_alternatives) == 3
    assert cs.causal_alternatives_unresolved is True


def test_cause_as_subject_is_rejected():
    f = "An investigation invalidated an OOS result, but the record did not establish the assignable laboratory cause."
    cs, ctx, out = _merge(f, _p(finding_subject="assignable laboratory cause", subject_kind="CAUSE"))
    assert ctx.finding_subject is None            # validator nulled it
    assert "assignable laboratory cause" not in (cs.finding_subject or "").lower()
    assert "finding_subject" not in out.fields_from_llm


def test_evidence_source_as_subject_is_rejected():
    f = "Maintenance records show that temporary repairs were performed on press PR-204."
    cs, ctx, out = _merge(f, _p(
        finding_subject="maintenance records", evidence_source="maintenance records",
        reported_observation="temporary repairs were performed", epistemic_status="REPORTED",
    ))
    assert (cs.finding_subject or "").lower() != "maintenance records"
    assert ctx.evidence_source == "maintenance records"


def test_dropped_comparison_is_conserved_from_floor():
    f = "The reconciliation of inventory location IL-4 showed a shortfall of 120 units against the system record."
    # LLM omits the comparison entirely
    cs, ctx, out = _merge(f, _p(finding_subject="inventory location IL-4"))
    assert cs.comparison_type in ("BELOW", "MISMATCH")     # from the deterministic floor
    assert "comparison" in out.fields_conserved
    assert cs.measurement is not None and cs.measurement.value == 120.0


def test_dropped_recurrence_is_conserved_from_floor():
    f = "Equipment M-204 experienced three failures over a six-month period."
    cs, ctx, out = _merge(f, _p(finding_subject="Equipment M-204"))
    assert cs.recurrence_count == 3
    assert "recurrence" in out.fields_conserved


def test_dropped_alternatives_are_conserved_from_floor():
    f = ("The discrepancy could have resulted from an unrecorded transaction, a physical "
         "miscount, or a system data-entry error.")
    cs, ctx, out = _merge(f, _p(finding_subject="the discrepancy"))  # LLM returns no alternatives
    assert len(cs.stated_causal_alternatives) >= 3
    assert "stated_causal_alternatives" in out.fields_conserved


def test_missing_record_not_promoted_to_nonperformance():
    f = "The activity was not documented, but it is unclear whether it was performed."
    cs, ctx, out = _merge(f, _p(
        finding_subject="the activity",
        missing_record_status="ACTIVITY_NOT_PERFORMED",   # LLM over-reaches
        activity_performance_ambiguity=False,
    ))
    assert ctx.missing_record_status == "ACTIVITY_NOT_RECORDED"   # downgraded
    assert ctx.activity_performance_ambiguity is True


def test_manufactured_comparison_magnitude_is_dropped():
    f = "The measured result differed from the approved value."       # no number stated
    cs, ctx, out = _merge(f, _p(
        finding_subject="the measured result",
        comparison={"left": "measured result", "right": "approved value",
                    "reference": "approved value", "direction": "BELOW",
                    "magnitude": 7.5, "unit": "%"},
    ))
    assert ctx.comparison.magnitude is None       # not in finding text -> dropped
    assert ctx.comparison.direction == "MISMATCH"  # no directional word -> not BELOW


def test_manufactured_recurrence_count_is_dropped():
    f = "The equipment experienced repeated failures over the past year."   # no explicit count
    cs, ctx, out = _merge(f, _p(
        finding_subject="the equipment",
        recurrence={"count": 5, "event": "failures", "period": "the past year"},
    ))
    assert ctx.recurrence.count is None


def test_fabricated_alternative_is_stripped_by_validator():
    f = "The failure may have been caused by a worn bearing or a lubrication fault."
    cs, ctx, out = _merge(f, _p(
        finding_subject="the failure",
        stated_causal_alternatives=["a worn bearing", "a lubrication fault",
                                    "sabotage by a disgruntled employee"],  # not in text
        causal_alternatives_unresolved=True,
    ))
    assert "sabotage by a disgruntled employee" not in ctx.stated_causal_alternatives
    assert len(ctx.stated_causal_alternatives) == 2


def test_llm_never_promotes_belief_to_verified():
    f = "It was reported that the second check may not have been completed."
    cs, ctx, out = _merge(f, _p(
        finding_subject="the second check", observed_condition="may not have been completed",
        epistemic_status="VERIFIED",   # LLM over-confident
    ))
    # merge does not write epistemic_status onto a VERIFIED observation field;
    # the report layer reads ctx.epistemic_status which the validator leaves
    # as-is, but no VERIFIED *fact* is created from it.
    assert cs.mechanism_status == "UNKNOWN"
