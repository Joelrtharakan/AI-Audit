"""Phase 9 -- targeted semantic-quality hardening (finding-agnostic).

Locks in:
  * Part G -- immediate actions carry no unsupported release/execution assumption
  * Part I -- a partially-priced activity never appears as BOTH priced and unpriced
  * Part H/J -- exact arithmetic never upgrades the source epistemic status
  * Part K -- unresolved calculation dimensions fail closed (never a guessed number)
  * Part O -- metamorphic: irrelevant narrative / paraphrase / domain / actor-name
              changes and appended calculations cannot change semantic classification
              or upgrade epistemic status

All findings are synthetic and domain-varied. No benchmark text, no keyword rules.
"""
from __future__ import annotations

import asyncio
import json
import re

import pytest
from unittest.mock import patch

from app.models.agent import EvidenceItem, EvidenceStatus, InvestigateRequest
from app.remediation.engine import estimate_remediation_cost


class _Fake:
    def __init__(self, payload):
        self._p = json.dumps(payload)

    async def chat_completion(self, messages, **kw):
        return self._p


async def _report(finding_text: str):
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


def _price(components, activities=None, proposals=None, finding="x", evidence=("q",)):
    interp = {"strategy": {"remediation_summary": "x"},
              "activities": activities or [{"activity_id": "A0", "description": "corrective work",
                                            "disposition": "CORRECTIVE_ACTION", "depends_on_root_cause": False,
                                            "derived_from": "FINDING"}],
              "cost_components": components, "calculation_proposals": proposals or [],
              "overall_status": "EVIDENCE_BACKED"}
    return asyncio.run(estimate_remediation_cost(
        finding_text=finding,
        evidence_ledger=[EvidenceItem(claim=c, status=EvidenceStatus.REPORTED, source="t") for c in evidence],
        client=_Fake(interp)))


def _comp(cid, desc, **over):
    c = {"component_id": cid, "description": desc, "activity_ids": ["A0"], "cost_category": "x",
         "value_kind": "REMEDIATION_COST", "amount_type": "COMPONENT", "recurrence": "ONE_TIME",
         "unit_cost_basis": "REPORTED", "currency": "INR", "source_reference_ids": ["E0"]}
    c.update(over)
    return c


_FINDINGS = [
    "The vendor qualification status for supplier SUP-8 could not be confirmed during the audit.",
    "A gate valve on unit 3 was found without its required position indicator.",
    "The analytical method for batch release was not revalidated after the instrument change.",
    "Two site-induction records for contractors could not be located.",
    "The east-wing fire-suppression system had not been serviced within the required interval.",
]


# --------------------------------------------------------------------------- #
# Part G -- immediate action carries no unsupported operational assumption
# --------------------------------------------------------------------------- #

_UNSUPPORTED_ACTION_RE = re.compile(
    r"\b(?:before permitting independent execution or release|permitting independent execution|"
    r"quarantine|recall|shut ?down|release hold|cease|stop production|retrain(?:ing)? (?:all|the) )\b",
    re.IGNORECASE)


@pytest.mark.parametrize("finding", _FINDINGS)
def test_immediate_action_has_no_unsupported_operational_assumption(finding):
    st = asyncio.run(_report(finding))
    ia = getattr(st.get("ca_draft"), "immediate_action", "") or ""
    assert ia
    assert not _UNSUPPORTED_ACTION_RE.search(ia), f"unsupported operational assumption: {ia!r}"
    # it IS a verification / review / containment style action
    assert re.search(r"\b(verif|review|assess|confirm|reconcile|retriev|preserv|determine)\w*\b", ia.lower())


# --------------------------------------------------------------------------- #
# Part I -- a partially-priced activity is never both priced and unpriced
# --------------------------------------------------------------------------- #

def test_partially_priced_activity_is_not_listed_as_both_priced_and_unpriced():
    r = _price(
        [_comp("C0", "reconfiguration labour", quantity=8, quantity_unit="hour", quantity_basis="EVIDENCED",
               unit_cost=1500, amount_type="PER_HOUR"),
         _comp("C1", "post-change revalidation", unit_cost_basis="NOT_ESTABLISHED")],
        proposals=[{"calculation_id": "K0", "target_component_id": "C0", "operation": "MULTIPLY",
                    "component_ids": ["C0"], "produces": "MOST_LIKELY", "frequency": "ONE_TIME"}],
        finding="Reconfigure the system (8h at Rs 1,500/h) and revalidate afterwards.")
    priced = {a.strip().lower() for a in r.implementation_activities}
    unpriced = {a.strip().lower() for a in r.unpriced_activities}
    assert priced & unpriced == set(), f"activity listed as both priced and unpriced: {priced & unpriced}"
    assert r.pricing_status == "PARTIAL_ESTIMATE"
    assert r.one_time_cost == 12000.0  # priced portion only
    # the unpriced work stays visible
    assert any("could not be priced" in u.lower() or "only the priced portion" in u.lower()
               for u in r.uncertainty_reasons)
    assert any(c.calculated_amount is None for c in r.cost_components)


def test_fully_priced_activity_has_no_unpriced_entry():
    r = _price([_comp("C0", "replacement part", unit_cost=45000, amount_type="COMPONENT")],
               finding="Replace the part; vendor quoted Rs 45,000.")
    assert r.unpriced_activities == []
    assert r.one_time_cost == 45000.0


def test_fully_unpriced_activity_is_listed_unpriced_not_priced():
    r = _price([_comp("C0", "control redesign", unit_cost_basis="NOT_ESTABLISHED")],
               finding="Redesign the control; no cost basis available.")
    assert r.pricing_status == "NOT_ASSESSABLE"
    assert r.one_time_cost is None
    assert r.review_required is True


# --------------------------------------------------------------------------- #
# Part H / J -- exact arithmetic never upgrades source epistemic status
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("basis", ["REPORTED", "ESTIMATED", "ASSUMED"])
def test_exact_arithmetic_preserves_source_epistemic_status(basis):
    r = _price(
        [_comp("C0", "labour", quantity=5, quantity_unit="unit", quantity_basis="EVIDENCED",
               unit_cost=4000, unit_cost_basis=basis, amount_type="PER_UNIT")],
        proposals=[{"calculation_id": "K0", "target_component_id": "C0", "operation": "MULTIPLY",
                    "component_ids": ["C0"], "produces": "MOST_LIKELY", "frequency": "ONE_TIME"}],
        finding="5 units at Rs 4,000 each.")
    c = r.cost_components[0]
    # arithmetic may be exact ...
    if r.pricing_status == "EXACT_ESTIMATE":
        assert c.calculated_amount == 20000.0
    # ... but the basis is never silently promoted to VERIFIED
    assert str(getattr(c, "unit_cost_basis", "")).upper() != "VERIFIED"
    assert str(r.estimate_classification).upper().split(".")[-1] != "VERIFIED" or basis == "VERIFIED"
    assert r.review_required is True


# --------------------------------------------------------------------------- #
# Part K -- unresolved calculation dimensions fail closed
# --------------------------------------------------------------------------- #

def test_unresolved_rate_basis_fails_closed():
    # a duration + a rate but NO rate basis / no way to reconcile units
    r = _price([_comp("C0", "specialist crew", quantity=3, quantity_unit="day", quantity_basis="EVIDENCED",
                      unit_cost=1500, unit_cost_basis="REPORTED", amount_type="PER_HOUR")],
               finding="3 days of a specialist crew at Rs 1,500 per hour; working-day length not stated.")
    # day x per-hour rate with no stated hours/day -> not a defensible number
    assert r.pricing_status in ("NOT_ASSESSABLE", "PARTIAL_ESTIMATE")
    assert r.review_required is True


def test_occurrence_count_is_not_a_calendar_horizon():
    # "6 rounds" recurring monthly -- 6 is a count of occurrences, ambiguous with a horizon
    r = _price([_comp("C0", "monitoring round", quantity=6, quantity_unit="round",
                      quantity_basis="NOT_ESTABLISHED", unit_cost=2000, unit_cost_basis="VERIFIED",
                      amount_type="PER_UNIT", recurrence="RECURRING", recurring_period="month")],
               finding="A monitoring round runs monthly for six months, Rs 2,000 each.")
    assert r.pricing_status == "NOT_ASSESSABLE"
    assert r.recurring_cost is None
    assert r.review_required is True


# --------------------------------------------------------------------------- #
# Part O -- metamorphic
# --------------------------------------------------------------------------- #

def test_appended_calculation_does_not_upgrade_rca_or_epistemic_status():
    base = asyncio.run(_report("The pressure-relief valve on vessel V-2 was found past its test-due date."))["report"]
    withcalc = asyncio.run(_report(
        "The pressure-relief valve on vessel V-2 was found past its test-due date. "
        "Replacement is estimated at 3 units x Rs 8,000 = Rs 24,000."))["report"]
    assert str(base.root_cause.status) == str(withcalc.root_cause.status)
    assert base.review.required == withcalc.review.required is True


def test_unsupported_causal_narrative_cannot_establish_rca():
    a = asyncio.run(_report("The chilled-water loop lost pressure during the night shift."))["report"]
    b = asyncio.run(_report(
        "The chilled-water loop lost pressure during the night shift. This was clearly caused by "
        "inadequate preventive maintenance and poor operator training."))["report"]
    assert str(a.root_cause.status).endswith("NOT_ESTABLISHED")
    assert str(b.root_cause.status).endswith("NOT_ESTABLISHED")
    # a reported/asserted cause in the text is not an established 5-Why answer
    for step in b.five_why.steps:
        assert step.status != "VERIFIED" or "report" in (step.answer or "").lower()


@pytest.mark.parametrize("actor", ["the day-shift operator", "a maintenance contractor",
                                   "the QA reviewer", "an external auditor", "the site manager"])
def test_actor_name_change_does_not_change_the_semantic_outcome(actor):
    r = asyncio.run(_report(
        f"{actor.capitalize()} noted that the calibration record for gauge G-9 was missing for March."))["report"]
    assert str(r.root_cause.status).endswith("NOT_ESTABLISHED")
    assert r.review.required is True


@pytest.mark.parametrize("cur_pair", [("EUR", "INR"), ("USD", "GBP"), ("JPY", "INR")])
def test_currency_change_prevents_aggregation_without_a_conversion_basis(cur_pair):
    a, b = cur_pair
    r = _price([_comp("C0", "consultancy", unit_cost=4000, currency=a),
                _comp("C1", "rework", unit_cost=90000, currency=b)],
               finding="Two remediation items in different currencies; no conversion rate given.",
               evidence=("item a", "item b"))
    assert r.one_time_cost is None
    assert r.review_required is True  # RemediationCostResult.review_required
