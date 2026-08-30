"""Finding-aware remediation SCOPE — domain-agnostic generalization matrix.

The remediation engine must produce a FINDING-SPECIFIC remediation approach
and activity list even when there is no pricing evidence (cost stays
NOT_ASSESSABLE). Different finding STRUCTURES must produce different scopes;
the same lack of pricing must NOT collapse every finding to the same text.

These tests drive the DETERMINISTIC path (LLM unavailable) so they are
provider-independent and stable. Entities span unrelated domains as TEST
DATA only -- the assertions check semantic role, never a domain keyword.
"""

from __future__ import annotations

import asyncio

import pytest

from app.models.agent import CanonicalFindingState, EvidenceItem, EvidenceStatus
from app.remediation.engine import estimate_remediation_cost
from app.remediation.models import RemediationEstimateStatus


class _RC:
    def __init__(self, status):
        self.status = status
        self.candidate_hypotheses = []


class _Down:
    async def chat_completion(self, *a, **k):  # LLM unavailable
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


# id, canonical kwargs, root_cause, substrings that MUST appear in the scope,
# substrings that must NOT
CASES = [
    ("missing_record",
     dict(finding_subject="temperature records for freezer FZ-2", deviation_condition="incomplete",
          semantic_type="MISSING_RECORD", affected_process="Temperature monitoring control"),
     "NOT_ESTABLISHED",
     ("reconstruct or complete temperature records for freezer fz-2",), ()),
    ("overdue_equipment_activity",
     dict(finding_subject="preventive maintenance for chiller CH-2", deviation_condition="overdue",
          semantic_type="OBJECT", affected_process="Maintenance scheduling control"),
     "NOT_ESTABLISHED",
     ("carry out the outstanding preventive maintenance for chiller ch-2",), ()),
    ("procedure_noncompliance",
     dict(finding_subject="the aseptic gowning procedure", deviation_condition="inconsistent compliance",
          semantic_type="CONTROL", affected_process="Contamination-control practice"),
     "NOT_ESTABLISHED",
     ("aseptic gowning procedure",), ()),
    ("competency_gap",
     dict(finding_subject="operator competency for line L-5", deviation_condition="not reassessed",
          semantic_type="OBSERVATION_VERIFICATION", affected_process="Training and qualification control"),
     "NOT_ESTABLISHED",
     ("operator competency for line l-5", "not reassessed"), ()),
    ("change_control_gap",
     dict(finding_subject="the risk assessment for change CR-9", deviation_condition="not updated",
          semantic_type="MISSING_RECORD", affected_process="Change-control linkage"),
     "NOT_ESTABLISHED",
     ("risk assessment for change cr-9",), ()),
    ("recurrence_trending",
     dict(finding_subject="the complaint recurrence evaluation", deviation_condition="not performed",
          semantic_type="RECURRENCE", occurrence_population="three complaints",
          affected_process="Complaint trending control"),
     "NOT_ESTABLISHED",
     ("retrospective review of the complaint recurrence evaluation across the affected population",), ()),
    ("supplier_qualification",
     dict(finding_subject="qualification of external analytical laboratory ELAB-4",
          deviation_condition="insufficient qualification", semantic_type="CONTROL",
          affected_process="Supplier oversight control"),
     "NOT_ESTABLISHED",
     ("external analytical laboratory elab-4",), ()),
    ("access_control_segregation",
     dict(finding_subject="segregation of privileged access", deviation_condition="weak segregation",
          semantic_type="CONTROL", affected_process="Identity and access management"),
     "NOT_ESTABLISHED",
     ("segregation of privileged access",), ()),
    ("comparison_discrepancy",
     dict(finding_subject="the recorded final yield for batch B-77", deviation_condition="differed from calculated",
          semantic_type="COMPARISON", affected_process="Yield reconciliation"),
     "NOT_ESTABLISHED",
     ("independently re-verify the recorded final yield for batch b-77",), ()),
    ("cause_established_is_direct",
     dict(finding_subject="the second-person verification step", deviation_condition="not performed",
          semantic_type="MISSING_RECORD", affected_process="Batch record review control"),
     "ESTABLISHED",
     ("strengthen batch record review control",),
     ("subject to confirming the underlying cause",)),  # cause known -> not hedged
    ("no_subject_stays_generic",
     dict(finding_subject="UNRESOLVED — the specific entity involved could not be isolated",
          semantic_type="NON_ACTIONABLE"),
     "NOT_ESTABLISHED",
     (), ("reconstruct", "retrospective review")),  # empty scope
]


@pytest.mark.parametrize("cid,kw,rc,must,must_not", CASES, ids=[c[0] for c in CASES])
def test_remediation_scope_is_finding_specific(cid, kw, rc, must, must_not):
    res = _run(_cs(**kw), rc=rc)

    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    # numbers are never fabricated
    assert res.low_estimate is None and res.most_likely_estimate is None and res.high_estimate is None
    assert res.currency is None
    assert res.one_time_cost is None and res.recurring_cost is None

    blob = " ".join(
        [res.remediation_strategy or ""]
        + list(res.implementation_activities or [])
        + list(res.unpriced_activities or [])
        + list(res.evidence_improves_estimate or [])
    ).lower()

    for m in must:
        assert m in blob, f"{cid}: expected {m!r} in remediation scope"
    for n in must_not:
        assert n not in blob, f"{cid}: {n!r} should not appear"

    # no internal diagnostics leaked
    for bad in ("llm", "schema", "parser", "traceback", "none", "not assessable reason"):
        assert bad not in (res.not_assessable_reason or "").lower()

    if cid == "no_subject_stays_generic":
        assert not res.implementation_activities
    else:
        assert len(res.implementation_activities) >= 3
        assert "verify that the completed remediation is effective" in blob
        # root-cause dependency
        if rc == "ESTABLISHED":
            assert "subject to confirming the underlying cause" not in blob
        else:
            assert "subject to confirming the underlying cause" in blob


def test_different_findings_do_not_produce_identical_scopes():
    a = _run(_cs(finding_subject="temperature records for freezer FZ-2", deviation_condition="incomplete",
                 semantic_type="MISSING_RECORD", affected_process="Temperature monitoring control"))
    b = _run(_cs(finding_subject="preventive maintenance for chiller CH-2", deviation_condition="overdue",
                 semantic_type="OBJECT", affected_process="Maintenance scheduling control"))
    assert a.implementation_activities != b.implementation_activities
    assert a.remediation_strategy != b.remediation_strategy
    assert a.evidence_improves_estimate != b.evidence_improves_estimate


def test_pricing_failure_does_not_destroy_semantic_scope_when_llm_gives_activities():
    """A weak/echoed LLM scope is replaced; a genuine LLM scope is kept."""
    import json

    class _Echo:
        async def chat_completion(self, *a, **k):
            return json.dumps({
                "strategy": {"remediation_summary": "define and perform the missing verification",
                             "remediation_type": "procedure definition + execution"},
                "activities": [{"activity_id": "A0", "description": "draft the procedure", "derived_from": "FINDING"}],
                "estimability": "NOT_ASSESSABLE", "not_assessable_reason": "PRICING_BASIS_UNAVAILABLE",
                "evidence_improves_estimate": ["an internal labor rate", "an effort estimate in hours"],
            })

    res = asyncio.run(estimate_remediation_cost(
        finding_text="f",
        evidence_ledger=[EvidenceItem(claim="c", status=EvidenceStatus.VERIFIED, source="t")],
        root_cause=_RC("NOT_ESTABLISHED"),
        canonical_state=_cs(finding_subject="the calibration certificate for gauge G-7",
                            deviation_condition="expired", semantic_type="RECORD",
                            affected_process="Calibration control"),
        client=_Echo(),
    ))
    blob = " ".join([res.remediation_strategy] + list(res.implementation_activities)).lower()
    assert "draft the procedure" not in blob
    assert "gauge g-7" in blob
    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE


def test_financial_and_remediation_stay_separate():
    res = _run(_cs(finding_subject="the reconciliation of account AC-3", deviation_condition="not performed",
                   semantic_type="MISSING_RECORD", affected_process="Financial reconciliation control"))
    # remediation result carries no financial-exposure field and no incurred-loss language
    blob = " ".join([res.remediation_strategy] + list(res.implementation_activities)
                    + list(res.evidence_improves_estimate)).lower()
    for bad in ("financial exposure", "financial loss", "incurred", "net loss", "recovery"):
        assert bad not in blob
    assert res.important_qualification  # the "expected cost, not incurred loss" disclaimer stays
