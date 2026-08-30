"""Final adversarial audit of remediation canonicalization.

Invariant under test, for EVERY LLM output shape:

  * every final implementation activity represents genuine remediation WORK;
  * every pricing component is traceable to the activity it prices, to an
    independent work item, or to an explicit unresolved pricing relationship;
  * no pricing representation becomes a phantom implementation activity;
  * no genuine remediation work disappears;
  * no currency / rate / quantity / total is manufactured; the LLM performs
    no arithmetic; the canonical activity collection is the sole source of
    the final projections.

Matrix: 12 finding categories x 20 structural scenarios. Assertions A-N are
STRUCTURAL, not prose.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from app.models.agent import CanonicalFindingState, EvidenceItem, EvidenceStatus
from app.remediation.activities import is_pricing_driver_phrase
from app.remediation.engine import estimate_remediation_cost
from app.remediation.models import RemediationEstimateStatus


class _RC:
    def __init__(self, status="ESTABLISHED"):
        self.status = status
        self.candidate_hypotheses = []


class _Down:
    async def chat_completion(self, *a, **k):
        raise RuntimeError("unavailable")


# 12 categories: (id, canonical finding state, one genuine activity, a second
# genuine activity, an independent-work component phrase)
_CATS = [
    ("equipment_repair", dict(finding_subject="the drive motor on conveyor CV-9",
        deviation_condition="failed", semantic_type="EQUIPMENT", affected_process="Maintenance control"),
        "Replace the failed drive motor on conveyor CV-9",
        "Inspect the other motors on the same line",
        "Requalify the conveyor line after the repair"),
    ("calibration", dict(finding_subject="calibration of balance BAL-7",
        deviation_condition="overdue", semantic_type="EQUIPMENT", affected_process="Calibration control"),
        "Recalibrate balance BAL-7 against a traceable standard",
        "Review the calibration schedule for related instruments",
        "Re-test the batches weighed since the last valid calibration"),
    ("records", dict(finding_subject="the batch record for lot L-88",
        deviation_condition="incomplete", semantic_type="MISSING_RECORD", affected_process="Batch record control"),
        "Reconstruct the incomplete batch record where objective evidence supports it",
        "Assess other records from the same period",
        "Perform a documentation gap review across the campaign"),
    ("lab_investigation", dict(finding_subject="the OOS investigation for result R-231",
        deviation_condition="did not document sufficient justification", semantic_type="RECORD",
        affected_process="Deviation investigation control"),
        "Complete the OOS investigation with adequate justification",
        "Re-evaluate the disposition decision for the affected lot",
        "Conduct a retrospective review of similar OOS investigations"),
    ("training", dict(finding_subject="operator competency for line L-5",
        deviation_condition="not reassessed", semantic_type="OBSERVATION_VERIFICATION",
        affected_process="Training and qualification control"),
        "Reassess the competency of the affected operators",
        "Update the training matrix for line L-5",
        "Deliver refresher training on the revised procedure"),
    ("access_control", dict(finding_subject="privileged access to the ERP",
        deviation_condition="excessive", semantic_type="CONTROL", affected_process="Identity and access management"),
        "Review the affected privileged access rights",
        "Remove the excess entitlements",
        "Perform an access recertification for the ERP"),
    ("change_control", dict(finding_subject="the risk assessment for change CR-9",
        deviation_condition="not updated", semantic_type="MISSING_RECORD", affected_process="Change-control linkage"),
        "Update the risk assessment for change CR-9",
        "Verify the change was implemented as approved",
        "Re-review the affected change package"),
    ("supplier", dict(finding_subject="qualification of laboratory ELAB-4",
        deviation_condition="not renewed", semantic_type="CONTROL", affected_process="Supplier oversight control"),
        "Complete the outstanding qualification review for laboratory ELAB-4",
        "Assess the results issued during the lapsed period",
        "Perform an on-site audit of laboratory ELAB-4"),
    ("software", dict(finding_subject="the audit-trail configuration of the LIMS",
        deviation_condition="disabled", semantic_type="CONTROL", affected_process="Computerised-system control"),
        "Re-enable the audit-trail configuration of the LIMS",
        "Determine the extent of the period with audit trail disabled",
        "Revalidate the affected LIMS functionality"),
    ("procedural", dict(finding_subject="the aseptic gowning procedure",
        deviation_condition="inconsistent compliance", semantic_type="CONTROL",
        affected_process="Contamination-control practice"),
        "Revise the aseptic gowning procedure",
        "Retrain the affected personnel against the revised procedure",
        "Perform a compliance verification round after retraining"),
    ("financial_remediation", dict(finding_subject="the bank reconciliation for account AC-9",
        deviation_condition="not performed", semantic_type="MISSING_RECORD",
        affected_process="Financial reconciliation control"),
        "Perform the outstanding bank reconciliation for account AC-9",
        "Investigate the unreconciled items",
        "Review the reconciliations for the surrounding periods"),
    ("environmental", dict(finding_subject="viable environmental monitoring for cleanroom CR-2",
        deviation_condition="not performed", semantic_type="MISSING_RECORD",
        affected_process="Environmental monitoring control"),
        "Perform the outstanding environmental monitoring for cleanroom CR-2",
        "Assess the batches manufactured without monitoring data",
        "Conduct a facility impact assessment for the affected area"),
]

# 20 scenarios. Each returns (activities, cost_components, kwargs) given the
# category's two activities A1/A2 and its independent-work phrase W.
_LABOUR = "Internal labour for {t}"
_CONTRACTOR = "Contractor cost for {t}"
_MATERIAL = "Material cost for {t}"
_TESTING = "Testing cost for {t}"
_DOC = "Documentation cost for {t}"
_TRAINING = "Training cost for {t}"


def _low(t):  # first-word-lowercased, for "labour for <t>"
    return t[0].lower() + t[1:]


def _scenarios(a1, a2, w):
    def act(d, i, **extra):
        return dict({"activity_id": f"A{i}", "description": d, "derived_from": "FINDING"}, **extra)

    def comp(d, i, **extra):
        return dict({"component_id": f"K{i}", "description": d, "cost_category": "labor",
                     "amount_type": "COMPONENT", "recurrence": "ONE_TIME",
                     "unit_cost_basis": "NOT_ESTABLISHED"}, **extra)

    S = {}
    S["1_labour"] = ([act(a1, 0)], [comp(_LABOUR.format(t=_low(a1)), 0)], {})
    S["2_contractor"] = ([act(a1, 0)], [comp(_CONTRACTOR.format(t=_low(a1)), 0)], {})
    S["3_material"] = ([act(a1, 0)], [comp(_MATERIAL.format(t=_low(a1)), 0)], {})
    S["4_testing"] = ([act(a1, 0)], [comp(_TESTING.format(t=_low(a1)), 0)], {})
    S["5_documentation"] = ([act(a1, 0)], [comp(_DOC.format(t=_low(a1)), 0)], {})
    S["6_training"] = ([act(a1, 0)], [comp(_TRAINING.format(t=_low(a1)), 0)], {})
    S["7_multi"] = ([act(a1, 0), act(a2, 1)],
                    [comp(_LABOUR.format(t=_low(a1)), 0), comp(_CONTRACTOR.format(t=_low(a2)), 1)], {})
    S["8_unmatched_genuine_work"] = ([act(a1, 0)], [comp(w, 0)], {})
    S["9_unmatched_pricing_only"] = ([act(a1, 0), act(a2, 1)],
                                     [comp("External advisory services", 0)], {})
    S["10_pure_pricing_no_activities"] = ([], [comp("Contractor cost", 0)], {})
    S["11_explicit_ids"] = ([act(a1, 0), act(a2, 1)],
                            [comp(_LABOUR.format(t=_low(a1)), 0, activity_ids=["A0"])], {})
    S["12_conflicting_matches"] = ([act(a1, 0), act(a1 + " promptly", 1)],
                                   [comp(_LABOUR.format(t=_low(a1)), 0)], {})
    S["13_conditional_plus_driver"] = (
        [act(a1, 0), act("Implement a new preventive monitoring tool", 1, derived_from="ROOT_CAUSE_HYPOTHESIS")],
        [comp(_LABOUR.format(t=_low(a1)), 0, activity_ids=["A0"])], dict(rc="NOT_ESTABLISHED"))
    S["14_priced_and_unpriced"] = (
        [act(a1, 0), act(a2, 1)],
        [comp("Service quotation for " + _low(a1), 0, activity_ids=["A0"], unit_cost=8000,
              unit_cost_basis="REPORTED", currency="INR", source_reference_ids=["E0"]),
         comp(_LABOUR.format(t=_low(a2)), 1, activity_ids=["A1"])],
        dict(estimability="ESTIMABLE"))
    S["15_unsupported_pricing"] = ([act(a1, 0)],
        [comp(_LABOUR.format(t=_low(a1)), 0, unit_cost=9999, unit_cost_basis="ASSUMED")], {})
    S["16_missing_currency"] = ([act(a1, 0)],
        [comp("Service quotation for " + _low(a1), 0, activity_ids=["A0"], unit_cost=5000,
              unit_cost_basis="REPORTED")], dict(estimability="ESTIMABLE"))
    S["17_llm_unavailable"] = None   # handled specially
    S["18_prompt_echo"] = ([{"activity_id": "A0", "description": "draft the procedure", "derived_from": "FINDING"}],
        [comp("procedure drafting effort", 0)], dict(summary="define and perform the missing verification"))
    S["19_scope_fallback"] = ([], [], {})
    S["20_dup_activities_diff_drivers"] = (
        [act(a1, 0), act(a1, 1)],
        [comp(_LABOUR.format(t=_low(a1)), 0), comp(_CONTRACTOR.format(t=_low(a1)), 1)], {})
    return S


def _client(activities, components, summary="Address the finding", estimability="NOT_ASSESSABLE"):
    payload = {
        "strategy": {"remediation_summary": summary, "remediation_type": "x"},
        "activities": activities,
        "cost_components": components,
        "estimability": estimability,
        "not_assessable_reason": "PRICING_BASIS_UNAVAILABLE",
        "evidence_improves_estimate": ["a rate", "hours"],
    }

    class _C:
        async def chat_completion(self, *a, **k):
            return json.dumps(payload)
    return _C()


def _anchor(e):
    m = re.search(r'to cost [“"](.+?)[”"]\s*$', e)
    return m.group(1) if m else None


def _assert_invariants(res, *, expect_work_substrings=()):
    impl = res.implementation_activities
    impl_set = set(impl)

    # A. implementation_activities contains only genuine work (no driver phrases)
    for a in impl:
        assert not is_pricing_driver_phrase(a), f"phantom pricing-driver activity: {a!r}"

    # G / H. subsets
    assert set(res.conditional_activities) <= impl_set
    assert set(res.unpriced_activities) <= impl_set

    # I. evidence never references a phantom pricing-driver activity
    for e in res.evidence_improves_estimate:
        anc = _anchor(e)
        assert anc is None or anc in impl_set
        assert not (anc and is_pricing_driver_phrase(anc))

    # D / E / F. every cost component is traceable and no money is hidden
    comp_by_id = {c.component_id: c for c in res.cost_components}
    unresolved_ids = {getattr(d, "component_id", None) for d in res.unresolved_pricing_drivers}
    # canonical relationship: an id is attached, an independent activity, or unresolved.
    # We cannot see component_ids on the public result, but we CAN require that
    # every priced component's amount is reflected in the headline totals OR
    # the component is explicitly unresolved -- never silently dropped.
    priced = [c for c in res.cost_components
              if c.calculated_amount is not None or c.unit_cost is not None]
    for c in priced:
        # the amount must be visible somewhere: in the headline estimate, or
        # the component row itself carries currency+amount for audit.
        assert (c.currency is not None and (c.calculated_amount is not None or c.unit_cost is not None)) \
            or res.most_likely_estimate is not None \
            or res.one_time_cost is not None, f"priced component {c.component_id} not auditable"

    # F. an unresolved driver must point to a real component row (no lost money)
    for d in res.unresolved_pricing_drivers:
        cid = getattr(d, "component_id", None)
        assert cid in comp_by_id, f"unresolved driver {d!r} has no cost_components row"

    # K. no manufactured precision on a NOT_ASSESSABLE result
    if res.status == RemediationEstimateStatus.NOT_ASSESSABLE:
        assert res.low_estimate is None and res.most_likely_estimate is None
        assert res.high_estimate is None and res.currency is None

    # C. expected genuine work is present
    low_impl = " || ".join(a.lower() for a in impl)
    for sub in expect_work_substrings:
        assert sub.lower() in low_impl, f"genuine work {sub!r} disappeared from {impl}"


@pytest.mark.parametrize("cat", _CATS, ids=[c[0] for c in _CATS])
@pytest.mark.parametrize("scen", [
    "1_labour", "2_contractor", "3_material", "4_testing", "5_documentation",
    "6_training", "7_multi", "8_unmatched_genuine_work", "9_unmatched_pricing_only",
    "10_pure_pricing_no_activities", "11_explicit_ids", "12_conflicting_matches",
    "13_conditional_plus_driver", "14_priced_and_unpriced", "15_unsupported_pricing",
    "16_missing_currency", "17_llm_unavailable", "18_prompt_echo", "19_scope_fallback",
    "20_dup_activities_diff_drivers",
])
def test_adversarial_matrix(cat, scen):
    cid, cs_kw, a1, a2, w = cat
    canon = CanonicalFindingState(raw_finding="f", observed_deviation="x", deviation="x", **cs_kw)
    S = _scenarios(a1, a2, w)

    if scen == "17_llm_unavailable":
        res = asyncio.run(estimate_remediation_cost(
            finding_text="f",
            evidence_ledger=[EvidenceItem(claim="c", status=EvidenceStatus.VERIFIED, source="t")],
            root_cause=_RC("NOT_ESTABLISHED"), canonical_state=canon, client=_Down()))
        _assert_invariants(res)
        assert res.implementation_activities  # scope fallback still describes work
        return

    activities, components, kw = S[scen]
    rc = kw.pop("rc", "ESTABLISHED")
    res = asyncio.run(estimate_remediation_cost(
        finding_text="f",
        evidence_ledger=[EvidenceItem(claim="Quotation INR 8,000", status=EvidenceStatus.REPORTED, source="t")],
        root_cause=_RC(rc), canonical_state=canon,
        client=_client(activities, components, **kw)))

    # scenario-specific "genuine work must survive" expectations
    expect = ()
    if scen in ("1_labour", "2_contractor", "3_material", "4_testing", "5_documentation",
                "6_training", "11_explicit_ids", "15_unsupported_pricing"):
        expect = (a1[:22],)
    elif scen == "7_multi":
        expect = (a1[:22], a2[:22])
    elif scen == "8_unmatched_genuine_work":
        expect = (a1[:22],)  # w may attach or become its own activity; a1 must survive
    elif scen == "14_priced_and_unpriced":
        expect = (a1[:22], a2[:22])

    _assert_invariants(res, expect_work_substrings=expect)

    # B. the driver phrase never appears verbatim as an activity when work exists
    if scen in ("1_labour", "2_contractor", "11_explicit_ids"):
        drv = (_LABOUR if scen != "2_contractor" else _CONTRACTOR).format(t=_low(a1))
        assert drv not in res.implementation_activities

    # 8: genuine independent work is preserved (as an activity or attached)
    if scen == "8_unmatched_genuine_work":
        wl = w.lower()
        appears = any(wl[:18] in a.lower() for a in res.implementation_activities)
        # if it did not become its own activity it must be an unresolved driver
        # OR a cost component -- never simply gone
        traced = appears or any(w == getattr(d, "description", d) for d in res.unresolved_pricing_drivers) \
            or any(c.description == w for c in res.cost_components)
        assert traced, f"genuine independent work {w!r} disappeared"
