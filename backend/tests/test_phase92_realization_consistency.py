"""Phase 9.2 -- semantic realization consistency (finding-agnostic).

Fixes locked in:
  Issue 1  -- 5-Why evidence-boundary text: no "X — Y" fragment, no enum token,
              complete sentence, no fabricated mechanism
  Issue 2/9-- exact arithmetic over a BELIEF/REPORTED/ESTIMATED price never
              upgrades the epistemic classification to VERIFIED
  Issue 3/4-- a PER_MONTH recurring cost with no horizon renders the periodic
              rate directly ("₹X per month"), never "per occurrence", never
              "1 month × ₹X"; total stays NOT ASSESSABLE
  Issue 6  -- investigation-gap questions are natural, not "What does the
              evidence establish about: <gap>?"
  Issue 8  -- a partially-priced activity is never both priced and unpriced

All synthetic, domain-varied. No benchmark text. No raw-finding-text keywords.
"""
from __future__ import annotations

import asyncio
import json
import re

import pytest
from unittest.mock import patch

from app.models.agent import EvidenceItem, EvidenceStatus, InvestigateRequest
from app.remediation.engine import estimate_remediation_cost

_ENUM_TOKEN = re.compile(
    r"(?<![A-Za-z_])(UNKNOWN|NOT_ESTABLISHED|TO_BE_CONFIRMED|POSSIBLE|VERIFIED|SUPPORTED|"
    r"BELIEF|MIXED|REPORTED|NOT_ASSESSABLE|EXACT_ESTIMATE|EVIDENCE_BOUNDARY)(?![A-Za-z])")


class _Fake:
    def __init__(self, payload):
        self._p = json.dumps(payload)

    async def chat_completion(self, messages, **kw):
        return self._p


async def _state(finding_text: str):
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.report_generator import generate_report_node
    from app.agent.nodes.understanding import understand_finding_node

    st = {"request": InvestigateRequest(finding_text=finding_text), "evidence_ledger": [],
          "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0, "trace": [], "errors": []}
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        for n in (understand_finding_node, plan_investigation_node, core_synthesis_node,
                  generate_report_node, final_evidence_verification_node):
            st = await n(st)
    return st


def _report(ft):
    return asyncio.run(_state(ft))["report"]


def _cost(components, activities=None, proposals=None, evidence=None, finding="x"):
    interp = {"strategy": {"remediation_summary": "x"},
              "activities": activities or [{"activity_id": "A0", "description": "corrective work",
                                            "disposition": "CORRECTIVE_ACTION", "depends_on_root_cause": False,
                                            "derived_from": "FINDING"}],
              "cost_components": components, "calculation_proposals": proposals or [],
              "overall_status": "EVIDENCE_BACKED"}
    ledger = evidence or [EvidenceItem(claim="q", status=EvidenceStatus.REPORTED, source="t")]
    return asyncio.run(estimate_remediation_cost(finding_text=finding, evidence_ledger=ledger,
                                                 client=_Fake(interp)))


def _c(cid, desc, **over):
    d = {"component_id": cid, "description": desc, "activity_ids": ["A0"], "cost_category": "x",
         "value_kind": "REMEDIATION_COST", "amount_type": "COMPONENT", "recurrence": "ONE_TIME",
         "unit_cost_basis": "REPORTED", "currency": "INR", "source_reference_ids": ["E0"]}
    d.update(over)
    return d


_5WHY_DOMAINS = [
    "Preventive-maintenance controls were found to be inadequate for the compressor fleet.",
    "The order-verification step in the fulfilment workflow was not being performed.",
    "Cleanroom gowning compliance on the night shift was found to be deficient.",
    "The reconciliation of the sub-ledger to the general ledger was incomplete for March.",
    "Firmware on the pedestrian-crossing controllers had not been updated to the approved baseline.",
]


# --------------------------------------------------------------------------- #
# Issue 1 -- 5-Why evidence-boundary realization
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("finding", _5WHY_DOMAINS)
def test_five_why_evidence_boundary_is_a_clean_complete_sentence(finding):
    r = _report(finding)
    assert str(r.root_cause.status).endswith("NOT_ESTABLISHED")
    for s in r.five_why.steps:
        for text in (s.question or "", s.answer or ""):
            assert " — " not in text and " -- " not in text, f"em-dash fragment: {text!r}"
            assert not _ENUM_TOKEN.search(text), f"enum token leaked into prose: {text!r}"
            assert not re.search(r"[a-z)]\.[A-Z]{3,}", text), f"glued status token: {text!r}"
            # a real sentence (or a question) -- not a bare fragment
            t = text.strip()
            if t:
                assert t.endswith((".", "?")) or ":" in t, f"fragment (no terminal punctuation): {t!r}"
    # boundary answer states the observed condition once and stops
    joined = " ".join(s.answer or "" for s in r.five_why.steps).lower()
    assert "does not establish" in joined or "not established" in joined
    assert r.five_why.is_complete is False


def test_five_why_does_not_emit_a_causal_mechanism_when_rca_not_established():
    r = _report("The seal on pump P-9 failed and the maintenance history could not be located.")
    assert str(r.root_cause.status).endswith("NOT_ESTABLISHED")
    for s in r.five_why.steps:
        # no step asserts an established cause
        if s.status == "VERIFIED":
            a = (s.answer or "").lower()
            assert not re.search(r"\b(caused by|because of|due to|resulted from)\b", a), \
                f"VERIFIED step asserts a cause: {s.answer!r}"


def test_appended_unsupported_cause_does_not_resume_the_five_why_chain():
    base = _report("The chilled-water loop lost pressure during the night shift.")
    withcause = _report("The chilled-water loop lost pressure during the night shift. "
                        "This was clearly caused by inadequate preventive maintenance.")
    assert str(base.root_cause.status) == str(withcause.root_cause.status)
    assert withcause.five_why.is_complete is False
    for s in withcause.five_why.steps:
        if s.status == "VERIFIED":
            assert "caused by" not in (s.answer or "").lower()


# --------------------------------------------------------------------------- #
# Issue 2 / 9 -- exact arithmetic never upgrades epistemic status
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ev_status, ev_label", [
    (EvidenceStatus.BELIEF, "BELIEF"),
    (EvidenceStatus.REPORTED, "REPORTED"),
])
def test_exact_arithmetic_over_a_non_verified_price_is_not_classified_verified(ev_status, ev_label):
    r = _cost(
        [_c("C0", "part", quantity=5, quantity_unit="unit", quantity_basis="EVIDENCED",
            unit_cost=4000, unit_cost_basis="VERIFIED", amount_type="PER_UNIT")],
        proposals=[{"calculation_id": "K0", "target_component_id": "C0", "operation": "MULTIPLY",
                    "component_ids": ["C0"], "produces": "MOST_LIKELY", "frequency": "ONE_TIME"}],
        evidence=[EvidenceItem(claim="an estimate of Rs 4,000 per unit for 5 units",
                               status=ev_status, source="eng")],
        finding="5 units at an estimated Rs 4,000 each.")
    comp = r.cost_components[0]
    assert str(comp.unit_cost_basis).split(".")[-1] != "VERIFIED", ev_label
    assert str(r.estimate_classification).split(".")[-1] != "VERIFIED", ev_label
    # arithmetic can still be exact
    assert comp.calculated_amount == 20000.0
    assert r.review_required is True


def test_genuine_verified_evidence_still_yields_verified():
    r = _cost(
        [_c("C0", "part", unit_cost=45000, unit_cost_basis="VERIFIED", amount_type="COMPONENT")],
        evidence=[EvidenceItem(claim="purchase order confirms Rs 45,000",
                               status=EvidenceStatus.VERIFIED, source="po")],
        finding="Replacement part, PO confirms Rs 45,000.")
    assert str(r.cost_components[0].unit_cost_basis).split(".")[-1] == "VERIFIED"


# --------------------------------------------------------------------------- #
# Issue 3 / 4 -- periodic pricing
# --------------------------------------------------------------------------- #

def test_recurring_per_month_no_horizon_renders_periodic_rate_not_occurrence_or_fabricated_quantity():
    r = _cost([_c("C0", "monthly service", unit_cost=9000, unit_cost_basis="REPORTED",
                  recurrence="RECURRING", recurring_period="month")],
              finding="Ongoing monthly service at Rs 9,000, no end date.")
    assert r.recurring_cost == 9000.0
    assert r.recurring_period == "month"
    assert r.recurring_horizon_total is None
    assert r.one_time_cost is None and r.most_likely_estimate is None
    comp = r.cost_components[0]
    assert "occurrence" not in (comp.calculation_formula or "").lower()
    assert not re.search(r"1\s*(month|×|x)\s*", comp.calculation_formula or "", re.IGNORECASE)
    assert comp.calculation_formula == "9000 (stated amount)"


@pytest.mark.parametrize("period", ["month", "week", "quarter", "year"])
def test_periodic_basis_never_becomes_per_occurrence(period):
    r = _cost([_c("C0", "recurring check", unit_cost=1200, unit_cost_basis="REPORTED",
                  recurrence="RECURRING", recurring_period=period)],
              finding=f"A recurring check at Rs 1,200 per {period}.")
    assert r.recurring_period == period
    blob = " ".join([r.remediation_strategy or ""] + list(r.uncertainty_reasons or [])
                    + [c.calculation_formula or "" for c in r.cost_components]).lower()
    assert "per occurrence" not in blob


# --------------------------------------------------------------------------- #
# Issue 6 -- natural investigation-gap realization
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("finding", _5WHY_DOMAINS)
def test_investigation_questions_are_not_mechanical_frames(finding):
    r = _report(finding)
    for q in r.investigation.questions:
        text = (q.question or "").strip()
        assert not text.lower().startswith("what does the evidence establish about:"), text
        assert ":" not in text or text.count(":") == 0 or text.endswith("?"), text
        assert text.endswith("?"), text


# --------------------------------------------------------------------------- #
# Issue 8 -- activity / cost-component alignment
# --------------------------------------------------------------------------- #

def test_initial_plus_recurring_components_stay_separate_and_aligned():
    r = _cost(
        [_c("C0", "initial program setup", unit_cost=85000, unit_cost_basis="REPORTED",
            amount_type="COMPONENT", recurrence="ONE_TIME"),
         _c("C1", "ongoing monthly inspection", unit_cost=9000, unit_cost_basis="REPORTED",
            amount_type="COMPONENT", recurrence="RECURRING", recurring_period="month")],
        activities=[{"activity_id": "A0", "description": "revised inspection programme",
                     "disposition": "CORRECTIVE_ACTION", "depends_on_root_cause": False, "derived_from": "FINDING"}],
        finding="Set up the revised programme (Rs 85,000 one-time) and run monthly inspections (Rs 9,000/month).")
    assert r.one_time_cost == 85000.0
    assert r.recurring_cost == 9000.0
    assert r.recurring_period == "month"
    priced = {a.strip().lower() for a in r.implementation_activities}
    unpriced = {a.strip().lower() for a in r.unpriced_activities}
    assert priced & unpriced == set()


def test_partially_priced_activity_not_both_priced_and_unpriced():
    r = _cost(
        [_c("C0", "setup labour", quantity=8, quantity_unit="hour", quantity_basis="EVIDENCED",
            unit_cost=1500, amount_type="PER_HOUR"),
         _c("C1", "post-setup validation", unit_cost_basis="NOT_ESTABLISHED")],
        proposals=[{"calculation_id": "K0", "target_component_id": "C0", "operation": "MULTIPLY",
                    "component_ids": ["C0"], "produces": "MOST_LIKELY", "frequency": "ONE_TIME"}],
        finding="Setup (8h at Rs 1,500/h) plus validation afterwards.")
    priced = {a.strip().lower() for a in r.implementation_activities}
    unpriced = {a.strip().lower() for a in r.unpriced_activities}
    assert priced & unpriced == set()
    assert r.pricing_status == "PARTIAL_ESTIMATE"


# --------------------------------------------------------------------------- #
# Metamorphic
# --------------------------------------------------------------------------- #

def test_domain_change_does_not_change_the_epistemic_rule():
    for ft in ("The valve on skid A failed and no inspection record was on file.",
               "The signature on batch record B was missing and no reviewer log was on file.",
               "The access grant for account C was not documented and no approval record was on file."):
        r = _report(ft)
        assert str(r.root_cause.status).endswith("NOT_ESTABLISHED")
        assert r.review.required is True


def test_amount_change_does_not_change_evidence_status():
    # Phase 9.4 Defect D: the ONLY cited evidence is BELIEF-status (an
    # epistemic stance, "strictly weaker than REPORTED") -- the VERIFIED cap
    # (Phase 9.2) still applies first, but a belief-only citation cannot
    # support REPORTED either, so the basis caps one tier further to
    # ESTIMATED. The invariant under test (amount does not change the
    # classification) is unaffected by which tier it lands on.
    for amt in (1000, 50000, 999999):
        r = _cost([_c("C0", "part", unit_cost=amt, unit_cost_basis="VERIFIED", amount_type="COMPONENT")],
                  evidence=[EvidenceItem(claim=f"estimate ~Rs {amt}", status=EvidenceStatus.BELIEF, source="e")],
                  finding=f"A part estimated at Rs {amt}.")
        assert str(r.cost_components[0].unit_cost_basis).split(".")[-1] == "ESTIMATED"


def test_periodic_rate_without_horizon_never_creates_a_finite_total():
    for period in ("month", "week", "year"):
        r = _cost([_c("C0", "recurring", unit_cost=7500, unit_cost_basis="REPORTED",
                      recurrence="RECURRING", recurring_period=period)],
                  finding=f"Rs 7,500 per {period}, indefinitely.")
        assert r.recurring_horizon_total is None
        assert r.most_likely_estimate is None
        assert r.one_time_cost is None
