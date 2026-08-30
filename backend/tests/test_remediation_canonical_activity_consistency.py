"""Canonical remediation-result self-consistency.

The engine derives implementation_activities / conditional_activities /
unpriced_activities / evidence_improves_estimate from ONE canonical activity
list (`app.remediation.activities`). These tests lock the invariant:

    set(evidence_improves_estimate) ==
        unique(pricing_evidence(a) for a in price-relevant final activities)

and the structural properties around it, across 12 domain-agnostic finding
shapes and both root-cause epistemic states. Deterministic (LLM patched off
or simulated) -> provider independent.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.models.agent import CanonicalFindingState, EvidenceItem, EvidenceStatus
from app.remediation.activities import (
    CONDITIONAL,
    classify_activity_kind,
    evidence_needed_for,
    normalize_remediation_activities,
)
from app.remediation.engine import estimate_remediation_cost
from app.remediation.models import RemediationEstimateStatus
from app.remediation.scope import _pricing_need, derive_remediation_scope


class _RC:
    def __init__(self, status):
        self.status = status
        self.candidate_hypotheses = []


class _Down:
    async def chat_completion(self, *a, **k):
        raise RuntimeError("unavailable")


def _llm(summary, acts, evidence=("a quotation",), components=()):
    class _C:
        async def chat_completion(self, *a, **k):
            return json.dumps({
                "strategy": {"remediation_summary": summary, "remediation_type": "x"},
                "activities": [
                    {"activity_id": f"A{i}", "description": d, "derived_from": "FINDING"}
                    for i, d in enumerate(acts)
                ],
                "cost_components": list(components),
                "estimability": "NOT_ASSESSABLE" if not components else "ESTIMABLE",
                "not_assessable_reason": "PRICING_BASIS_UNAVAILABLE",
                "evidence_improves_estimate": list(evidence),
            })
    return _C()


def _cs(**kw):
    d = dict(raw_finding="f", finding_subject="?", semantic_type="OBJECT",
             observed_deviation="x", deviation="x")
    d.update(kw)
    return CanonicalFindingState(**d)


def _run(canon, client, rc="NOT_ESTABLISHED"):
    return asyncio.run(estimate_remediation_cost(
        finding_text="f",
        evidence_ledger=[EvidenceItem(claim="c", status=EvidenceStatus.VERIFIED, source="t")],
        root_cause=_RC(rc), canonical_state=canon, client=client,
    ))


# --- the 12 structural finding categories (spec section TESTING) -----------

_CATEGORIES = [
    ("equipment_physical", dict(finding_subject="the pressure relief valve on vessel PV-4",
                                deviation_condition="overdue inspection", semantic_type="EQUIPMENT",
                                affected_process="Preventive maintenance control")),
    ("record_deficiency", dict(finding_subject="the batch record for lot L-88",
                               deviation_condition="incomplete", semantic_type="MISSING_RECORD",
                               affected_process="Batch record control")),
    ("access_control", dict(finding_subject="privileged access to the ERP",
                            deviation_condition="weak segregation", semantic_type="CONTROL",
                            affected_process="Identity and access management")),
    ("training_competency", dict(finding_subject="operator competency for line L-5",
                                 deviation_condition="not reassessed", semantic_type="OBSERVATION_VERIFICATION",
                                 affected_process="Training and qualification control")),
    ("process_control", dict(finding_subject="the aseptic gowning procedure",
                             deviation_condition="inconsistent compliance", semantic_type="CONTROL",
                             affected_process="Contamination-control practice")),
    ("change_management", dict(finding_subject="the risk assessment for change CR-9",
                               deviation_condition="not updated", semantic_type="MISSING_RECORD",
                               affected_process="Change-control linkage")),
    ("analytical_investigation", dict(finding_subject="the OOS investigation for result R-231",
                                      deviation_condition="did not document sufficient justification",
                                      semantic_type="RECORD", affected_process="Deviation investigation control")),
    ("supplier_qualification", dict(finding_subject="qualification of laboratory ELAB-4",
                                    deviation_condition="not renewed", semantic_type="CONTROL",
                                    affected_process="Supplier oversight control")),
    ("environmental_monitoring", dict(finding_subject="viable monitoring for cleanroom CR-2",
                                      deviation_condition="not performed", semantic_type="MISSING_RECORD",
                                      affected_process="Environmental monitoring control")),
    ("software_configuration", dict(finding_subject="the audit-trail configuration of the LIMS",
                                    deviation_condition="disabled", semantic_type="CONTROL",
                                    affected_process="Computerised-system control")),
    ("financial_reconciliation", dict(finding_subject="the bank reconciliation for account AC-9",
                                      deviation_condition="not performed", semantic_type="MISSING_RECORD",
                                      affected_process="Financial reconciliation control")),
    ("calibration", dict(finding_subject="calibration of balance BAL-7",
                         deviation_condition="overdue", semantic_type="EQUIPMENT",
                         affected_process="Calibration control")),
]


@pytest.mark.parametrize("cid,kw", _CATEGORIES, ids=[c[0] for c in _CATEGORIES])
@pytest.mark.parametrize("rc", ["NOT_ESTABLISHED", "ESTABLISHED"])
def test_canonical_result_is_self_consistent(cid, kw, rc):
    canon = _cs(**kw)
    res = _run(canon, _Down(), rc=rc)  # deterministic scope path

    impl = res.implementation_activities
    assert impl, f"{cid}/{rc}: no activities"

    # 1. conditional_activities is a strict subset of implementation_activities
    assert set(res.conditional_activities) <= set(impl)

    # 2. unpriced_activities on a NOT_ASSESSABLE result covers exactly the
    #    implementation activities (nothing lost, nothing foreign)
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    assert set(res.unpriced_activities) == set(impl)

    # 3. THE INVARIANT: evidence-needed == unique pricing need per final activity
    st = kw["semantic_type"]
    expected = evidence_needed_for(normalize_remediation_activities(
        llm_activities=[], scope=derive_remediation_scope(
            subject=kw["finding_subject"], condition=kw["deviation_condition"],
            semantic_type=st, affected_process=kw["affected_process"],
            root_cause_established=(rc == "ESTABLISHED"),
        ),
        semantic_type=st, contingent=(rc != "ESTABLISHED"),
        use_scope_as_canonical=True, conditional_systemic_sentence="",
    ))
    assert set(res.evidence_improves_estimate) == set(expected)

    # 4. every evidence line maps to an activity actually present
    for e in res.evidence_improves_estimate:
        assert any(_pricing_need(a, classify_activity_kind(a), st) == e for a in impl)

    # 5. no invented numbers
    assert res.low_estimate is None and res.most_likely_estimate is None
    assert res.currency is None

    # 6. conditionality follows the epistemic state
    if rc == "ESTABLISHED":
        assert res.conditional_activities == []
    else:
        assert any("subject to conf" in c.lower() or "once the cause" in c.lower()
                   or "determine whether" in c.lower() for c in res.conditional_activities)


def test_invariant_holds_when_llm_activities_are_disciplined():
    """LLM proposes a concrete systemic prescription -> it is dropped and the
    evidence list still corresponds ONLY to the final activities."""
    canon = _cs(finding_subject="temperature records for freezer FZ-2",
                deviation_condition="unavailable", semantic_type="RECORD",
                affected_process="Cold-chain record control")
    res = _run(canon, _llm(
        "Install a redundant monitoring system and retrain all staff.",
        ["Recover any available controller data and document the current state",
         "Install a redundant temperature monitoring system",
         "Retrain all cold-storage staff",
         "Investigate why the records were unavailable"],
    ))
    impl = res.implementation_activities
    assert not any("install" in a.lower() or "retrain" in a.lower() for a in impl)
    assert any("recover" in a.lower() for a in impl)            # immediate correction kept
    assert any("investigate why" in a.lower() for a in impl)    # causal investigation kept, concrete
    assert res.conditional_activities and all(c in impl for c in res.conditional_activities)

    st = "RECORD"
    for e in res.evidence_improves_estimate:
        assert any(_pricing_need(a, classify_activity_kind(a), st) == e for a in impl), e
    assert len(set(res.evidence_improves_estimate)) == len(res.evidence_improves_estimate)


def _evidence_target(e: str) -> str | None:
    import re
    m = re.search(r'to cost [“"](.+?)[”"]\s*$', e)
    return m.group(1) if m else None


def test_no_orphan_evidence_when_llm_components_differ_from_llm_activities():
    """The reported defect: LLM `activities` describe one set, LLM
    `cost_components` describe another (e.g. generic 'reconstruct records',
    'investigate cause'). Every evidence-needed item must still map to a
    FINAL activity -- nothing orphaned."""
    canon = _cs(finding_subject="the periodic access review for the ERP",
                deviation_condition="incomplete", semantic_type="MISSING_RECORD",
                affected_process="Access review control")

    class _C:
        async def chat_completion(self, *a, **k):
            return json.dumps({
                "strategy": {"remediation_summary": "Re-establish the access review and fix the records"},
                "activities": [
                    {"activity_id": "A0", "description": "Review and update the periodic access-review process", "derived_from": "FINDING"},
                    {"activity_id": "A1", "description": "Audit current access rights against approved roles", "derived_from": "FINDING"},
                ],
                "cost_components": [
                    {"component_id": "K0", "description": "reconstruct or complete the affected records",
                     "cost_category": "labor", "amount_type": "COMPONENT", "recurrence": "ONE_TIME",
                     "unit_cost_basis": "NOT_ESTABLISHED"},
                    {"component_id": "K1", "description": "investigate why the condition occurred",
                     "cost_category": "labor", "amount_type": "COMPONENT", "recurrence": "ONE_TIME",
                     "unit_cost_basis": "NOT_ESTABLISHED"},
                ],
                "estimability": "NOT_ASSESSABLE", "not_assessable_reason": "PRICING_BASIS_UNAVAILABLE",
                "evidence_improves_estimate": ["a rate", "hours"],
            })

    res = _run(canon, _C())
    impl = set(res.implementation_activities)
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE

    # every evidence item maps to exactly one final activity
    for e in res.evidence_improves_estimate:
        t = _evidence_target(e)
        assert t is not None, f"evidence item has no activity anchor: {e!r}"
        assert t in impl, f"orphan evidence -> {t!r} not in implementation_activities"

    # the LLM component work is not orphaned -- it is a visible activity
    assert any("reconstruct" in a.lower() for a in impl)
    assert any("investigate why" in a.lower() for a in impl)
    # nothing lost
    assert any("access-review process" in a.lower() for a in impl)
    assert any("audit current access" in a.lower() for a in impl)
    # unpriced == implementation on a fully NOT_ASSESSABLE result
    assert set(res.unpriced_activities) == impl
    # no fabricated numbers
    assert res.low_estimate is None and res.currency is None


def test_conditionality_and_pricing_are_independent_axes():
    """confirmed+priced, confirmed+unpriced, conditional+unpriced can co-exist."""
    canon = _cs(finding_subject="the calibration of gauge G-7", deviation_condition="overdue",
                semantic_type="EQUIPMENT", affected_process="Calibration control")

    class _C:
        async def chat_completion(self, *a, **k):
            return json.dumps({
                "strategy": {"remediation_summary": "Recalibrate and strengthen the control"},
                "activities": [
                    {"activity_id": "A0", "description": "Recalibrate gauge G-7 against a traceable standard", "derived_from": "FINDING"},
                    {"activity_id": "A1", "description": "Assess the calibration status of other gauges on the same schedule", "derived_from": "FINDING"},
                    {"activity_id": "A2", "description": "Strengthen the calibration scheduling control to prevent recurrence", "derived_from": "ROOT_CAUSE_HYPOTHESIS"},
                ],
                "cost_components": [
                    {"component_id": "K0", "description": "external calibration service", "activity_ids": ["A0"],
                     "cost_category": "service", "unit_cost": 6000, "unit_cost_basis": "REPORTED",
                     "currency": "INR", "amount_type": "COMPONENT", "recurrence": "ONE_TIME",
                     "source_reference_ids": ["E0"]},
                ],
                "overall_status": "EVIDENCE_BACKED",
            })

    res = _run(canon, _C(), rc="NOT_ESTABLISHED")
    # priced activity -> not in unpriced list
    assert any("recalibrate" in a.lower() for a in res.implementation_activities)
    assert not any("recalibrate" in a.lower() for a in res.unpriced_activities)
    # the systemic action is conditional (cause unknown) AND unpriced -> the
    # two axes are independent and both hold at once
    assert res.conditional_activities
    for c in res.conditional_activities:
        assert c in res.unpriced_activities
        assert c in res.implementation_activities
    # every evidence item still maps to a final activity
    impl = set(res.implementation_activities)
    for e in res.evidence_improves_estimate:
        t = _evidence_target(e)
        assert t is None or t in impl


def test_partial_estimate_evidence_needed_targets_only_the_unpriced_gap():
    canon = _cs(finding_subject="the calibration of gauge G-7", deviation_condition="overdue",
                semantic_type="EQUIPMENT", affected_process="Calibration control")
    res = _run(canon, _llm(
        "Recalibrate the gauge and revalidate the affected method.",
        ["Recalibrate gauge G-7 against a traceable standard",
         "Revalidate the analytical method that used the gauge"],
        evidence=["a verified revalidation effort"],
        components=[
            {"component_id": "C0", "description": "external calibration service",
             "cost_category": "service", "unit_cost": 6000, "unit_cost_basis": "REPORTED",
             "currency": "INR", "amount_type": "COMPONENT", "recurrence": "ONE_TIME",
             "source_reference_ids": ["E0"]},
            {"component_id": "C1", "description": "method revalidation effort",
             "cost_category": "labor", "quantity_basis": "NOT_ESTABLISHED",
             "unit_cost_basis": "NOT_ESTABLISHED", "amount_type": "COMPONENT",
             "recurrence": "ONE_TIME"},
        ],
    ), rc="ESTABLISHED")

    # priced portion survives, unpriced revalidation work stays visible
    assert res.is_partial_estimate is True
    assert any("revalidat" in a.lower() for a in res.unpriced_activities)
    # evidence-needed is about the unpriced gap, not the already-priced calibration
    joined = " ".join(res.evidence_improves_estimate).lower()
    assert joined  # non-empty
    assert "external calibration service" not in joined
