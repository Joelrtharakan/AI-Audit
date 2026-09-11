"""Phase 5 -- historical recurrence vs future recurrence risk.

Observed historical recurrence is a FACT about the past. A future-recurrence
RISK LEVEL (HIGH/MEDIUM/LOW) is a forward-looking assessment that needs an
evidence-grounded basis -- typically an established causal mechanism. These are
different propositions and must not be collapsed.

`assess_recurrence_risk(rec, root_cause_established)` is the single, epistemic
(not lexical) rule:
    recurring + cause ESTABLISHED  -> HIGH
    recurring + cause NOT established -> NOT_ASSESSABLE  (assessment required)

These tests use finding-agnostic wording across unrelated domains -- no
benchmark cases, no keyword rules.
"""
from __future__ import annotations

import asyncio

import pytest
from unittest.mock import patch

from app.agent.recurrence_guard import RecurrenceInfo, assess_recurrence_risk
from app.models.agent import InvestigateRequest


# --------------------------------------------------------------------------- #
# unit: the epistemic rule itself
# --------------------------------------------------------------------------- #

def test_recurring_without_established_cause_is_not_assessable():
    lvl, rat = assess_recurrence_risk(RecurrenceInfo(is_recurring=True, recurrence_count=5), False)
    assert lvl == "NOT_ASSESSABLE"
    r = rat.lower()
    assert "recurr" in r
    assert "not" in r and ("established" in r or "assessment" in r)


def test_recurring_with_established_cause_is_high():
    lvl, rat = assess_recurrence_risk(RecurrenceInfo(is_recurring=True, recurrence_count=2), True)
    assert lvl == "HIGH"
    assert "established" in rat.lower()


def test_large_count_alone_does_not_make_it_high():
    for n in (2, 12, 40, 500):
        lvl, _ = assess_recurrence_risk(RecurrenceInfo(is_recurring=True, recurrence_count=n), False)
        assert lvl == "NOT_ASSESSABLE", n


def test_previous_capa_reference_alone_does_not_make_it_high():
    lvl, _ = assess_recurrence_risk(
        RecurrenceInfo(is_recurring=True, capa_id="CAPA-2024-777"), False)
    assert lvl == "NOT_ASSESSABLE"


# --------------------------------------------------------------------------- #
# metamorphic: paraphrase / domain change with the same semantic structure
# --------------------------------------------------------------------------- #

async def _run(finding_text: str):
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.report_generator import generate_report_node
    from app.agent.nodes.understanding import understand_finding_node

    state = {"request": InvestigateRequest(finding_text=finding_text), "evidence_ledger": [],
             "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0, "trace": [], "errors": []}
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
        state = await core_synthesis_node(state)
        state = await generate_report_node(state)
        state = await final_evidence_verification_node(state)
    return state["report"]


_RECURRING_UNKNOWN_CAUSE = [
    # same semantic structure across unrelated domains: an explicitly recurring
    # condition with no causal mechanism established.
    "The same seal failure recurred on pump P-12 for the fourth time this year.",
    "The identical order-routing error recurred in the fulfilment system for the third time.",
    "The same medication-labelling mismatch recurred on ward 4 for the third time.",
    "The same packet-loss pattern recurred on network segment B for the fifth time.",
    "The same register omission recurred for the fourth consecutive term.",
    "The same calibration nonconformity recurred on balance BAL-3 for the third time.",
]


@pytest.mark.parametrize("finding", _RECURRING_UNKNOWN_CAUSE)
def test_recurring_unknown_cause_never_yields_a_confident_risk_level(finding):
    report = asyncio.run(_run(finding))
    rc = report.root_cause
    # deterministic offline path -> no established cause
    assert str(rc.status).endswith("NOT_ESTABLISHED")
    assert rc.risk_of_recurrence == "NOT_ASSESSABLE", finding
    assert rc.risk_of_recurrence_rationale
    # and the report still mandates review
    assert report.review.required is True


def test_metamorphic_irrelevant_wording_does_not_change_the_risk_decision():
    a = asyncio.run(_run("The same coupling defect recurred on line 3 for the third time."))
    b = asyncio.run(_run(
        "During a routine Tuesday walkthrough by the day-shift lead, it was again noted that "
        "the same coupling defect recurred on line 3 for the third time, as also mentioned last month."))
    assert a.root_cause.risk_of_recurrence == b.root_cause.risk_of_recurrence == "NOT_ASSESSABLE"


def test_non_recurring_finding_risk_is_left_to_the_semantic_layer():
    # a single-occurrence finding: the deterministic recurrence override does not
    # fire at all, so whatever the semantic layer produced stands (here: the
    # deterministic default).
    report = asyncio.run(_run("A fire door on level 2 was found wedged open during the inspection."))
    assert report.root_cause.risk_of_recurrence in ("NOT_ASSESSABLE", "LOW", "MEDIUM", "HIGH")
