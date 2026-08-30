"""Domain-agnostic semantic + remediation integrity matrix.

Verifies that the SAME semantic principles hold across unrelated domains and
that the RESULT differs by semantic ROLE, not by vocabulary:

  * OBSERVED FACT != AFFECTED OBJECT != EVIDENCE != ROOT CAUSE != HYPOTHESIS
    != CORRECTIVE ACTION != PREVENTIVE/SYSTEMIC ACTION
  * an evidence gap never becomes a root cause
  * when root cause is NOT_ESTABLISHED, remediation offers direct corrections
    of the confirmed condition but only CONDITIONAL, appropriately-abstract
    systemic actions -- never a concrete unsupported intervention
  * remediation cost stays NOT_ASSESSABLE without defensible pricing evidence

All paths are deterministic (LLM patched off or simulated) -> provider
independent. Entities span many domains as TEST DATA only.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from app.models.agent import (
    CanonicalFindingState,
    EvidenceItem,
    EvidenceStatus,
    InvestigateRequest,
)
from app.remediation.activities import (
    is_unsupported_concrete_intervention as _is_unsupported_concrete_intervention,
)
from app.remediation.engine import estimate_remediation_cost
from app.remediation.models import RemediationEstimateStatus
from app.services.semantic_subject import resolve_deviation


# ---------------------------------------------------------------------------
# 1. Same noun, different semantic role -> different subject.
# ---------------------------------------------------------------------------

_ROLE_CASES = [
    # (finding, the noun under test, expected: is it the subject?)
    ("The calibration certificate for balance BAL-7 had expired.",
     "calibration certificate", True),
    ("The calibration certificate confirms that balance BAL-7 was within tolerance "
     "at the last check.", "calibration certificate", False),
    ("Evidence needed: the calibration certificate for balance BAL-7.",
     "calibration certificate", False),
    ("Balance BAL-7 drift was possibly caused by an overdue calibration certificate "
     "review.", "calibration certificate", False),

    ("The maintenance log for pump P-4 was missing entries for March.",
     "maintenance log", True),
    ("The maintenance log shows that pump P-4 was repaired twice in March.",
     "maintenance log", False),

    ("The training record for operator O-12 was incomplete.",
     "training record", True),
    ("Training records indicate that operator O-12 completed the SOP-114 course.",
     "training record", False),
]


@pytest.mark.parametrize("finding,noun,is_subject", _ROLE_CASES,
                         ids=[f"{c[1]}-{i}" for i, c in enumerate(_ROLE_CASES)])
def test_same_noun_role_determines_subject(finding, noun, is_subject):
    d = resolve_deviation(finding)
    subj = (d.finding_subject or "").lower()
    if is_subject:
        assert noun in subj, f"{finding!r}: expected {noun!r} as subject, got {subj!r}"
    else:
        assert noun not in subj, f"{finding!r}: {noun!r} leaked into subject {subj!r}"


# ---------------------------------------------------------------------------
# 2. The new-resource verb class predicate (spec 4/5) -- structural, not
#    domain vocabulary.
# ---------------------------------------------------------------------------

_UNSUPPORTED = [
    "Install a redundant temperature monitoring system with battery backup",
    "Procure and deploy automated data-logger units for all cold-storage units",
    "Retrain all pharmacy personnel on record retention procedures",
    "Upgrade the LIMS to a validated version with audit-trail enforcement",
    "Replace the pressure sensor on vessel PV-4 with a dual-redundant unit",
    "Hire an additional QA reviewer to cover the second shift",
    "Build an automated reconciliation interface between the ERP and the bank feed",
]
_ACCEPTABLE = [
    "Recover any available data from the device controller and document the current state",
    "Reconstruct the affected records for the period where objective evidence supports it",
    "Subject to confirming the underlying cause, strengthen the record-retention control",
    "Determine whether additional monitoring controls are required once the cause is established",
    "Implement the corrective control that the confirmed cause identifies",
    "Re-verify the recorded value against its reference and correct the discrepant record",
    "Quarantine the affected inventory pending assessment",
]


@pytest.mark.parametrize("text", _UNSUPPORTED)
def test_concrete_intervention_detected(text):
    assert _is_unsupported_concrete_intervention(text) is True


@pytest.mark.parametrize("text", _ACCEPTABLE)
def test_correction_or_abstract_action_not_flagged(text):
    assert _is_unsupported_concrete_intervention(text) is False


# ---------------------------------------------------------------------------
# 3. Remediation epistemic discipline across domains (LLM over-eager).
# ---------------------------------------------------------------------------

class _RC:
    def __init__(self, status="NOT_ESTABLISHED"):
        self.status = status
        self.candidate_hypotheses = []


def _overeager_llm(summary, activities):
    class _LLM:
        async def chat_completion(self, *a, **k):
            return json.dumps({
                "strategy": {"condition_identified": "x", "remediation_summary": summary,
                             "remediation_type": "systemic", "established_basis": "the condition",
                             "hypothetical_basis": "the cause"},
                "activities": [
                    {"activity_id": f"A{i}", "description": d,
                     "derived_from": "ROOT_CAUSE_HYPOTHESIS" if i else "FINDING"}
                    for i, d in enumerate(activities)
                ],
                "cost_components": [], "estimability": "NOT_ASSESSABLE",
                "not_assessable_reason": "PRICING_BASIS_UNAVAILABLE",
                "evidence_improves_estimate": ["a quotation", "effort"],
            })
    return _LLM()


_REMEDIATION_DOMAINS = [
    dict(subject="temperature records for vaccine refrigerator FR-3", condition="unavailable",
         semantic_type="RECORD", process="Cold-chain record control",
         summary="Install a redundant temperature monitoring system and retrain all staff.",
         acts=["Recover available controller data and document the current state",
               "Install a redundant temperature monitoring system",
               "Retrain all pharmacy staff on record retention"]),
    dict(subject="the bank reconciliation for account AC-9", condition="not performed",
         semantic_type="MISSING_RECORD", process="Financial reconciliation control",
         summary="Build an automated reconciliation interface and hire a reviewer.",
         acts=["Perform the outstanding reconciliation for the affected period",
               "Build an automated reconciliation interface between the ERP and the bank feed",
               "Hire an additional reviewer"]),
    dict(subject="access provisioning for the LIMS", condition="weak segregation",
         semantic_type="CONTROL", process="Identity and access management",
         summary="Deploy a privileged-access management platform.",
         acts=["Review current LIMS access against role requirements and remove excess rights",
               "Deploy a privileged-access management platform"]),
    dict(subject="the pressure relief valve on vessel PV-4", condition="overdue inspection",
         semantic_type="EQUIPMENT", process="Preventive maintenance control",
         summary="Replace all relief valves with dual-redundant units.",
         acts=["Carry out the overdue inspection of the relief valve now",
               "Replace all relief valves on the unit with dual-redundant units"]),
    dict(subject="supplier qualification for vendor V-12", condition="not renewed",
         semantic_type="CONTROL", process="Supplier oversight control",
         summary="Procure a third-party supplier audit service for all vendors.",
         acts=["Complete the outstanding qualification review for vendor V-12",
               "Procure a third-party supplier audit service for the whole vendor base"]),
]


@pytest.mark.parametrize("case", _REMEDIATION_DOMAINS,
                         ids=[c["semantic_type"] + "-" + c["subject"][:12] for c in _REMEDIATION_DOMAINS])
def test_unestablished_cause_yields_no_concrete_systemic_prescription(case):
    canon = CanonicalFindingState(
        raw_finding="f", finding_subject=case["subject"], deviation_condition=case["condition"],
        semantic_type=case["semantic_type"], affected_process=case["process"],
        observed_deviation="x", deviation="x",
    )
    res = asyncio.run(estimate_remediation_cost(
        finding_text="f",
        evidence_ledger=[EvidenceItem(claim="c", status=EvidenceStatus.VERIFIED, source="t")],
        root_cause=_RC("NOT_ESTABLISHED"), canonical_state=canon,
        client=_overeager_llm(case["summary"], case["acts"]),
    ))

    assert res.status == RemediationEstimateStatus.NOT_ASSESSABLE
    assert res.low_estimate is None and res.most_likely_estimate is None
    assert res.currency is None

    blob = " ".join([res.remediation_strategy] + list(res.implementation_activities)).lower()
    for banned in ("install a redundant", "build an automated", "deploy a privileged",
                   "replace all relief valves", "procure a third-party", "hire an additional",
                   "retrain all"):
        assert banned not in blob, f"{case['subject']}: leaked concrete prescription {banned!r}"

    # a conditional systemic action is present and marked conditional
    assert res.conditional_activities
    assert all("subject to conf" in c.lower() or "once the cause" in c.lower()
               or "determine whether" in c.lower()
               for c in res.conditional_activities)
    # the immediate correction of the confirmed condition survived
    assert len(res.implementation_activities) >= 2


def test_established_cause_allows_a_concrete_action():
    """When the cause IS confirmed, a concrete systemic action is legitimate
    and must NOT be abstracted away."""
    canon = CanonicalFindingState(
        raw_finding="f", finding_subject="the pressure relief valve on vessel PV-4",
        deviation_condition="failed to lift at set pressure", semantic_type="EQUIPMENT",
        affected_process="Preventive maintenance control", observed_deviation="x", deviation="x",
    )
    res = asyncio.run(estimate_remediation_cost(
        finding_text="f",
        evidence_ledger=[EvidenceItem(claim="c", status=EvidenceStatus.VERIFIED, source="t")],
        root_cause=_RC("ESTABLISHED"), canonical_state=canon,
        client=_overeager_llm(
            "Replace the failed relief valve with a like-for-like unit.",
            ["Replace the failed relief valve on vessel PV-4 with a like-for-like qualified unit"],
        ),
    ))
    blob = " ".join([res.remediation_strategy] + list(res.implementation_activities)).lower()
    assert "replace the failed relief valve" in blob
    assert not res.conditional_activities


# ---------------------------------------------------------------------------
# 4. Full pipeline: evidence gap stays a gap, 5-Why stops, impact hedged.
# ---------------------------------------------------------------------------

async def _pipeline(finding_text: str):
    from app.agent.invariants import evaluate_all_invariants
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.report_generator import generate_report_node
    from app.agent.nodes.understanding import understand_finding_node

    state = {
        "request": InvestigateRequest(finding_text=finding_text),
        "evidence_ledger": [], "iteration_count": 0, "tool_call_count": 0,
        "critic_iteration": 0, "trace": [], "errors": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
        state = await core_synthesis_node(state)
        state = await generate_report_node(state)
        state = await final_evidence_verification_node(state)
    ok, violations = evaluate_all_invariants(state)
    return state, ok, violations


_EVIDENCE_GAP_FINDINGS = [
    "Temperature records for vaccine refrigerator FR-3 were unavailable for a two-week "
    "period, and the actual storage temperature could not be established.",
    "The environmental monitoring data for cleanroom CR-2 could not be located for "
    "the last quarter.",
    "The access log for the payment system was not retained for the audit period, so "
    "it could not be confirmed who approved the batch of overrides.",
]


@pytest.mark.parametrize("finding", _EVIDENCE_GAP_FINDINGS)
def test_evidence_gap_never_becomes_a_root_cause(finding):
    state, ok, violations = asyncio.run(_pipeline(finding))
    assert ok, f"invariants violated: {violations}"

    rc = state.get("root_cause_result") or state.get("root_cause")
    status = getattr(getattr(rc, "status", None), "value", getattr(rc, "status", None))
    assert status in ("NOT_ESTABLISHED", None), f"evidence gap promoted to cause: {status}"

    fw = state.get("five_why_analysis") or state.get("five_why")
    if fw is not None:
        assert not getattr(fw, "is_complete", False)
        for step in getattr(fw, "steps", []):
            assert step.status in ("UNKNOWN", "HYPOTHESIS", "POSSIBLE", "UNVERIFIED", None), step.status

    report = state.get("report")
    if report is not None and report.impact_assessment is not None:
        eff = (getattr(report.impact_assessment, "potential_effect", "") or "").lower()
        for claim in ("was invalid", "was lost", "was contaminated", "was rejected",
                      "resulted in financial loss", "customers were harmed"):
            assert claim not in eff
