"""Remediation Pass B — per-activity conditionality and activity-specific
pricing evidence.

Two properties, both domain-agnostic (entities are test data only):

  1. Conditionality is a property of the ACTIVITY, not just the headline.
     Systemic strengthening is CONDITIONAL when the root cause is not
     established; immediate correction / scope assessment / effectiveness
     verification are CONFIRMED regardless.

  2. The "what evidence would improve the estimate" list is derived from each
     unpriced activity's STRUCTURAL role. It is not always "internal labour
     rate + effort in hours": a physical-asset correction asks for a service /
     parts quotation; an analysis activity asks for a rate + effort; a
     systemic change asks for the confirmed control change first.
"""

from __future__ import annotations

import asyncio

import pytest

from app.models.agent import CanonicalFindingState, EvidenceItem, EvidenceStatus
from app.remediation.engine import estimate_remediation_cost
from app.remediation.models import RemediationEstimateStatus
from app.remediation.scope import derive_remediation_scope


class _RC:
    def __init__(self, status):
        self.status = status
        self.candidate_hypotheses = []


class _Down:
    async def chat_completion(self, *a, **k):
        raise RuntimeError("provider unavailable")


def _cs(**kw):
    d = dict(raw_finding="f", finding_subject="?", semantic_type="OBJECT",
             observed_deviation="x", deviation="x")
    d.update(kw)
    return CanonicalFindingState(**d)


def _run(canon, rc="NOT_ESTABLISHED"):
    return asyncio.run(estimate_remediation_cost(
        finding_text="f",
        evidence_ledger=[EvidenceItem(claim="c", status=EvidenceStatus.VERIFIED, source="t")],
        root_cause=_RC(rc), canonical_state=canon, client=_Down(),
    ))


# ---- 1. per-activity conditionality -------------------------------------

def test_only_systemic_activity_is_conditional_when_cause_unknown():
    scope = derive_remediation_scope(
        subject="the calibration certificate for gauge G-7", condition="expired",
        semantic_type="RECORD", affected_process="Calibration control",
        root_cause_established=False,
    )
    by_kind = {a.kind: a.conditionality for a in scope.activities}
    assert by_kind["IMMEDIATE_CORRECTION"] == "CONFIRMED"
    assert by_kind["SCOPE_ASSESSMENT"] == "CONFIRMED"
    assert by_kind["EFFECTIVENESS_VERIFICATION"] == "CONFIRMED"
    assert by_kind["SYSTEMIC_STRENGTHENING"] == "CONDITIONAL"
    assert scope.conditional_activity_descriptions == [
        a.description for a in scope.activities if a.kind == "SYSTEMIC_STRENGTHENING"
    ]


def test_no_activity_is_conditional_when_cause_established():
    scope = derive_remediation_scope(
        subject="the second-person verification step", condition="not performed",
        semantic_type="MISSING_RECORD", affected_process="Batch record review control",
        root_cause_established=True,
    )
    assert scope.conditional_activity_descriptions == []
    assert all(a.conditionality == "CONFIRMED" for a in scope.activities)
    assert not any(a.kind == "CAUSAL_INVESTIGATION" for a in scope.activities)


def test_conditional_activity_reaches_the_result_object():
    res = _run(_cs(finding_subject="the supplier audit for vendor V-12",
                   deviation_condition="not conducted", semantic_type="CONTROL",
                   affected_process="Supplier oversight control"))
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    assert res.conditional_activities
    # every conditional activity is also listed as an implementation activity
    for c in res.conditional_activities:
        assert c in res.implementation_activities
    # and it is phrased as contingent, not as a confirmed directive
    assert all("subject to confirming the underlying cause" in c.lower()
               for c in res.conditional_activities)


# ---- 2. activity-specific pricing evidence -----------------------------

def _evidence_for(kind, **kw):
    scope = derive_remediation_scope(root_cause_established=False, **kw)
    return next(a.pricing_evidence_needed.lower() for a in scope.activities if a.kind == kind)


def test_physical_asset_correction_asks_for_a_quotation_not_a_labour_rate():
    ev = _evidence_for(
        "IMMEDIATE_CORRECTION",
        subject="the pressure relief valve on vessel PV-4", condition="overdue",
        semantic_type="EQUIPMENT", affected_process="Preventive maintenance control",
    )
    assert "quotation" in ev
    assert "effort required (hours or days)" not in ev


def test_record_reconstruction_asks_for_an_internal_rate_and_effort():
    ev = _evidence_for(
        "IMMEDIATE_CORRECTION",
        subject="the training records for team T-3", condition="incomplete",
        semantic_type="MISSING_RECORD", affected_process="Training control",
    )
    assert "internal rate" in ev and "effort" in ev
    assert "service or contractor quotation" not in ev


def test_analysis_activities_ask_for_rate_and_effort_regardless_of_finding_type():
    ev = _evidence_for(
        "SCOPE_ASSESSMENT",
        subject="the pressure relief valve on vessel PV-4", condition="overdue",
        semantic_type="EQUIPMENT", affected_process="Preventive maintenance control",
    )
    assert "internal rate" in ev and "effort" in ev


def test_systemic_change_evidence_defers_to_the_confirmed_control_change():
    ev = _evidence_for(
        "SYSTEMIC_STRENGTHENING",
        subject="the change risk assessment for CR-9", condition="not updated",
        semantic_type="MISSING_RECORD", affected_process="Change-control linkage",
    )
    assert "once the cause is confirmed" in ev


def test_evidence_needed_list_is_not_a_single_repeated_line():
    scope = derive_remediation_scope(
        subject="the pressure relief valve on vessel PV-4", condition="overdue",
        semantic_type="EQUIPMENT", affected_process="Preventive maintenance control",
        root_cause_established=False,
    )
    assert len(scope.evidence_needed) >= 2  # dedup collapsed nothing to 1


@pytest.mark.parametrize("st_a,st_b", [("EQUIPMENT", "MISSING_RECORD")])
def test_two_finding_types_yield_different_immediate_correction_evidence(st_a, st_b):
    a = _evidence_for("IMMEDIATE_CORRECTION", subject="asset X-1", condition="overdue",
                      semantic_type=st_a, affected_process="p")
    b = _evidence_for("IMMEDIATE_CORRECTION", subject="record X-1", condition="incomplete",
                      semantic_type=st_b, affected_process="p")
    assert a != b


# ---- 3. adversarial: weak/echoed LLM must not erase per-activity data ---

def test_echoed_llm_scope_is_replaced_and_conditionality_preserved():
    import json

    class _Echo:
        async def chat_completion(self, *a, **k):
            return json.dumps({
                "strategy": {"remediation_summary": "draft the procedure",
                             "remediation_type": "procedure definition + execution"},
                "activities": [{"activity_id": "A0", "description": "procedure drafting effort",
                                "derived_from": "FINDING"}],
                "estimability": "NOT_ASSESSABLE",
                "not_assessable_reason": "PRICING_BASIS_UNAVAILABLE",
                "evidence_improves_estimate": ["an internal labor rate", "hours"],
            })

    res = asyncio.run(estimate_remediation_cost(
        finding_text="f",
        evidence_ledger=[EvidenceItem(claim="c", status=EvidenceStatus.VERIFIED, source="t")],
        root_cause=_RC("NOT_ESTABLISHED"),
        canonical_state=_cs(finding_subject="the fire-damper inspection for zone Z-2",
                            deviation_condition="overdue", semantic_type="EQUIPMENT",
                            affected_process="Facilities inspection control"),
        client=_Echo(),
    ))
    blob = " ".join([res.remediation_strategy] + list(res.implementation_activities)).lower()
    assert "draft the procedure" not in blob and "procedure drafting effort" not in blob
    assert "zone z-2" in blob
    assert res.conditional_activities
    assert any("quotation" in e.lower() for e in res.evidence_improves_estimate)
