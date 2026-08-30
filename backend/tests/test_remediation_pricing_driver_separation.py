"""WORK vs PRICING-DRIVER separation in canonical remediation activities.

Semantic invariant under test:

    An IMPLEMENTATION ACTIVITY describes work to be performed.
    A COST COMPONENT / PRICING DRIVER describes what prices an activity.

A pricing-driver phrase ("labour for X", "contractor cost", "analyst effort")
must never become a second implementation activity when the work it prices
already exists -- but no pricing information may be lost: an unattached driver
stays in `cost_components` and is recorded in `unresolved_pricing_drivers`.

All paths deterministic (simulated / patched-off LLM). Entities span many
domains as TEST DATA only; assertions are on STRUCTURAL invariants.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from app.models.agent import CanonicalFindingState, EvidenceItem, EvidenceStatus
from app.remediation.activities import (
    is_pricing_driver_phrase,
    strip_pricing_frame,
)
from app.remediation.engine import estimate_remediation_cost
from app.remediation.models import RemediationEstimateStatus


class _RC:
    def __init__(self, status="ESTABLISHED"):
        self.status = status
        self.candidate_hypotheses = []


class _Down:
    async def chat_completion(self, *a, **k):
        raise RuntimeError("unavailable")


def _cs(**kw):
    d = dict(raw_finding="f", finding_subject="the affected item", semantic_type="OBJECT",
             deviation_condition="deficient", affected_process="a control",
             observed_deviation="x", deviation="x")
    d.update(kw)
    return CanonicalFindingState(**d)


def _llm(activities, components, *, summary="Address the finding",
         estimability="NOT_ASSESSABLE"):
    payload = {
        "strategy": {"remediation_summary": summary, "remediation_type": "x"},
        "activities": [
            {"activity_id": a.get("id", f"A{i}"), "description": a["d"],
             "derived_from": a.get("df", "FINDING"),
             "is_hypothetical": a.get("hyp", False)}
            for i, a in enumerate(activities)
        ],
        "cost_components": [
            dict({"component_id": c.get("id", f"C{i}"), "description": c["d"],
                  "cost_category": c.get("cat", "labor"),
                  "amount_type": "COMPONENT", "recurrence": "ONE_TIME"}, **c.get("extra", {}))
            for i, c in enumerate(components)
        ],
        "estimability": estimability,
        "not_assessable_reason": "PRICING_BASIS_UNAVAILABLE",
        "evidence_improves_estimate": ["a rate", "hours"],
    }

    class _C:
        async def chat_completion(self, *a, **k):
            return json.dumps(payload)
    return _C()


def _run(client, canon=None, rc="ESTABLISHED"):
    return asyncio.run(estimate_remediation_cost(
        finding_text="f",
        evidence_ledger=[EvidenceItem(claim="c", status=EvidenceStatus.VERIFIED, source="t")],
        root_cause=_RC(rc), canonical_state=canon or _cs(), client=client,
    ))


def _evidence_anchor(e: str) -> str | None:
    m = re.search(r'to cost [“"](.+?)[”"]\s*$', e)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# The structural predicate (no domain vocabulary).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrase", [
    "Labour for completing the assessment",
    "Internal labor for the record reconstruction",
    "Contractor cost",
    "Contractor cost for the repair",
    "Analyst effort",
    "Engineering hours for the redesign",
    "Material cost for the replacement",
    "Parts cost",
    "Testing cost for the requalification",
    "Documentation cost",
    "Training cost for the affected staff",
    "Service quotation for the outstanding inspection",
    "External consultant fees",
    "Cost of the corrective work",
])
def test_pricing_driver_phrases_detected(phrase):
    assert is_pricing_driver_phrase(phrase) is True


@pytest.mark.parametrize("phrase", [
    "Complete the required assessment",
    "Reconstruct the affected records",
    "Replace the failed valve",
    "Retrain the affected operators",
    "Investigate why the deviation occurred",
    "Validate the analytical method",
    "Configure the audit-trail setting",
    "Review the access rights",
    "Strengthen the scheduling control",
    "Determine the extent of the deviation",
])
def test_work_phrases_not_flagged_as_drivers(phrase):
    assert is_pricing_driver_phrase(phrase) is False


# ---------------------------------------------------------------------------
# Category matrix: activity + <driver of type X>  ->  ONE activity.
# ---------------------------------------------------------------------------

_DRIVER_CATEGORIES = [
    ("labor", "Internal labour for completing the assessment"),
    ("contractor", "Contractor cost for completing the assessment"),
    ("material", "Material cost for the assessment materials"),
    ("testing", "Testing cost for the assessment"),
    ("documentation", "Documentation cost for the assessment report"),
    ("training", "Training cost for the assessment briefing"),
    ("service", "Service quotation for completing the assessment"),
    ("effort", "Analyst effort for the assessment"),
]


@pytest.mark.parametrize("cat,driver", _DRIVER_CATEGORIES, ids=[c[0] for c in _DRIVER_CATEGORIES])
def test_activity_plus_driver_yields_one_activity(cat, driver):
    res = _run(_llm(
        [{"d": "Complete the required assessment"}],
        [{"d": driver, "cat": cat}],
    ))
    impl = res.implementation_activities
    assert impl == ["Complete the required assessment"], impl
    assert driver not in impl
    # nothing lost: the component is still in cost_components
    assert any(c.description == driver for c in res.cost_components)
    # evidence (if any) points at the real activity, never the driver phrase
    for e in res.evidence_improves_estimate:
        a = _evidence_anchor(e)
        assert a is None or a in impl
        assert "labour for" not in (a or "").lower() and "cost for" not in (a or "").lower()


def test_multiple_activities_multiple_drivers():
    res = _run(_llm(
        [{"d": "Reconstruct the affected training records"},
         {"d": "Re-verify the competency of the affected operators"}],
        [{"d": "Internal labour for reconstructing the affected training records", "id": "K0"},
         {"d": "Analyst effort for re-verifying the competency of the affected operators", "id": "K1"}],
    ))
    impl = res.implementation_activities
    assert len(impl) == 2
    assert not any(is_pricing_driver_phrase(a) for a in impl)
    assert any("reconstruct" in a.lower() for a in impl)
    assert any("re-verify" in a.lower() or "competency" in a.lower() for a in impl)


def test_unmatched_component_that_is_genuine_independent_work_becomes_activity():
    res = _run(_llm(
        [{"d": "Recalibrate the affected instrument"}],
        [{"d": "Requalify the downstream analytical method", "id": "K0"}],  # genuine separate work
    ))
    impl = [a.lower() for a in res.implementation_activities]
    assert any("recalibrate" in a for a in impl)
    assert any("requalify" in a or "downstream analytical method" in a for a in impl)


def test_unmatched_pure_pricing_driver_is_recorded_not_an_activity():
    res = _run(_llm(
        [{"d": "Review the affected access rights"},
         {"d": "Correct the excess entitlements"}],
        [{"d": "Third-party contractor cost", "id": "K0"}],  # pure driver, no work reference
    ))
    impl = res.implementation_activities
    assert "Third-party contractor cost" not in impl
    assert not any(is_pricing_driver_phrase(a) for a in impl)
    # preserved + auditable: recorded with the component_id that carries the money
    assert any(d.description == "Third-party contractor cost" for d in res.unresolved_pricing_drivers)
    _u = next(d for d in res.unresolved_pricing_drivers if d.description == "Third-party contractor cost")
    assert any(c.component_id == _u.component_id for c in res.cost_components)


def test_explicit_activity_ids_are_honoured():
    res = _run(_llm(
        [{"d": "Repair the failed actuator", "id": "A0"},
         {"d": "Assess other actuators on the same line", "id": "A1"}],
        [{"d": "Parts and labour", "id": "K0", "extra": {"activity_ids": ["A0"]}}],
    ))
    assert "Parts and labour" not in res.implementation_activities
    assert any("repair the failed actuator" == a.lower() for a in res.implementation_activities)


def test_overlapping_activities_distinct_drivers_stay_one_each():
    res = _run(_llm(
        [{"d": "Update the calibration procedure", "id": "A0"},
         {"d": "Update the calibration schedule", "id": "A1"}],
        [{"d": "Labour for updating the calibration procedure", "id": "K0", "extra": {"activity_ids": ["A0"]}},
         {"d": "Labour for updating the calibration schedule", "id": "K1", "extra": {"activity_ids": ["A1"]}}],
    ))
    impl = res.implementation_activities
    assert len(impl) == 2
    assert not any("labour for" in a.lower() for a in impl)


def test_conditional_activity_plus_pricing_driver():
    res = _run(_llm(
        [{"d": "Review the affected records", "id": "A0"},
         {"d": "Implement a new preventive monitoring tool", "id": "A1", "df": "ROOT_CAUSE_HYPOTHESIS"}],
        [{"d": "Internal labour for reviewing the affected records", "id": "K0", "extra": {"activity_ids": ["A0"]}}],
    ), rc="NOT_ESTABLISHED")
    impl = set(res.implementation_activities)
    # driver did not become an activity
    assert not any("labour for" in a.lower() for a in impl)
    # the concrete "new tool" prescription was abstracted -> conditional
    assert res.conditional_activities
    assert set(res.conditional_activities) <= impl
    assert set(res.unpriced_activities) <= impl
    # "Review the affected records" is confirmed, not conditional
    assert not any("review the affected records" == c.lower() for c in res.conditional_activities)


def test_priced_activity_plus_unpriced_component_is_partial_and_consistent():
    res = _run(_llm(
        [{"d": "Replace the failed sensor", "id": "A0"},
         {"d": "Requalify the affected batch", "id": "A1"}],
        [{"d": "Sensor replacement service", "id": "K0", "extra": {
            "activity_ids": ["A0"], "unit_cost": 8000, "unit_cost_basis": "REPORTED",
            "currency": "INR", "source_reference_ids": ["E0"]}},
         {"d": "Labour for requalifying the affected batch", "id": "K1", "extra": {
            "activity_ids": ["A1"], "unit_cost_basis": "NOT_ESTABLISHED"}}],
        estimability="ESTIMABLE",
    ))
    assert res.is_partial_estimate is True
    assert res.most_likely_estimate == 8000.0
    impl = set(res.implementation_activities)
    assert not any("labour for" in a.lower() for a in impl)
    assert set(res.unpriced_activities) <= impl
    for e in res.evidence_improves_estimate:
        a = _evidence_anchor(e)
        assert a is None or a in impl


def test_components_with_unsupported_pricing_remain_unpriced_and_auditable():
    res = _run(_llm(
        [{"d": "Perform the outstanding inspection"}],
        [{"d": "Inspection labour", "id": "K0", "extra": {
            "unit_cost": 9999, "unit_cost_basis": "ASSUMED"}}],  # unsupported -> stripped
    ))
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    assert res.low_estimate is None and res.currency is None
    assert res.implementation_activities == ["Perform the outstanding inspection"]
    assert any(c.component_id == "K0" for c in res.cost_components)


def test_component_with_no_currency_is_not_promoted_to_activity():
    res = _run(_llm(
        [{"d": "Restore the affected configuration"}],
        [{"d": "Restoration labour", "id": "K0", "extra": {
            "unit_cost": 5000, "unit_cost_basis": "REPORTED"}}],  # no currency
    ))
    assert res.currency is None
    assert res.implementation_activities == ["Restore the affected configuration"]
    assert "Restoration labour" not in res.implementation_activities


def test_scope_fallback_when_llm_unavailable_has_no_driver_activities():
    res = _run(_Down(), canon=_cs(
        finding_subject="the calibration certificate for gauge G-7",
        deviation_condition="expired", semantic_type="RECORD",
        affected_process="Calibration control"), rc="NOT_ESTABLISHED")
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    assert res.implementation_activities
    assert not any(is_pricing_driver_phrase(a) for a in res.implementation_activities)
    for e in res.evidence_improves_estimate:
        a = _evidence_anchor(e)
        assert a is None or a in res.implementation_activities


def test_prompt_echo_falls_back_to_scope_not_driver_activities():
    echo = _llm(
        [{"d": "draft the procedure"}],
        [{"d": "procedure drafting effort", "id": "K0"}],
        summary="define and perform the missing verification",
    )
    res = _run(echo, canon=_cs(
        finding_subject="the second-person verification step",
        deviation_condition="not performed", semantic_type="MISSING_RECORD",
        affected_process="Batch record review control"), rc="NOT_ESTABLISHED")
    impl = res.implementation_activities
    assert "draft the procedure" not in [a.lower() for a in impl]
    assert "procedure drafting effort" not in [a.lower() for a in impl]
    assert not any(is_pricing_driver_phrase(a) for a in impl)


# ---------------------------------------------------------------------------
# The required cross-cutting invariants (spec REQUIRED INVARIANTS 1-9).
# ---------------------------------------------------------------------------

_INVARIANT_CASES = [
    ("physical_asset", _cs(finding_subject="the pressure relief valve on vessel PV-4",
                           deviation_condition="overdue inspection", semantic_type="EQUIPMENT",
                           affected_process="Preventive maintenance control"),
     [{"d": "Carry out the overdue inspection of the relief valve"}],
     [{"d": "Contractor labour for the overdue inspection of the relief valve"}]),
    ("record", _cs(finding_subject="the batch record for lot L-88", deviation_condition="incomplete",
                   semantic_type="MISSING_RECORD", affected_process="Batch record control"),
     [{"d": "Reconstruct the incomplete batch record where objective evidence supports it"}],
     [{"d": "Internal labour for reconstructing the incomplete batch record"}]),
    ("procedural", _cs(finding_subject="the aseptic gowning procedure",
                       deviation_condition="inconsistent compliance", semantic_type="CONTROL",
                       affected_process="Contamination-control practice"),
     [{"d": "Revise the aseptic gowning procedure and retrain against it"}],
     [{"d": "Documentation cost for revising the aseptic gowning procedure"}]),
]


@pytest.mark.parametrize("cid,canon,acts,comps", _INVARIANT_CASES, ids=[c[0] for c in _INVARIANT_CASES])
@pytest.mark.parametrize("rc", ["ESTABLISHED", "NOT_ESTABLISHED"])
def test_cross_cutting_invariants(cid, canon, acts, comps, rc):
    res = _run(_llm(acts, comps), canon=canon, rc=rc)
    impl = set(res.implementation_activities)

    # 1. no pricing-driver phrase is an implementation activity
    assert not any(is_pricing_driver_phrase(a) for a in impl)
    # 4. implementation_activities are distinct remediation work
    assert impl
    # 5/6. subsets
    assert set(res.conditional_activities) <= impl
    assert set(res.unpriced_activities) <= impl
    # 2/3. every component is traceable: attached (evidence via activity),
    #      independent work (an activity), or unresolved driver (recorded)
    comp_ids = {c.component_id for c in res.cost_components}
    assert comp_ids  # nothing dropped from the pricing record
    # 7. evidence references actual implementation activities
    for e in res.evidence_improves_estimate:
        a = _evidence_anchor(e)
        assert a is None or a in impl
    # 14/15. no fabricated precision / currency
    assert res.low_estimate is None and res.most_likely_estimate is None
    assert res.currency is None
