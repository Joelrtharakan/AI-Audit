"""Quantitative / comparative information must survive semantic resolution.

The deterministic resolver must NOT collapse:
  * "shortfall of 120 units against the system record" -> "status unconfirmed"
  * "three failures over a six-month period"           -> "repeated failure"
  * "differed by 4.2 percent"                          -> "not matched"

when doing so destroys explicit numeric information.

Structural / domain-neutral -- the extractors key on grammatical comparison
and recurrence frames, never on domain vocabulary. Assertions are on semantic
invariants (magnitude / unit / reference / direction / count / period /
subject), not exact generated prose.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.understanding import understand_finding_node
from app.agent.recurrence_guard import extract_quantitative_recurrence
from app.models.agent import InvestigateRequest
from app.services.semantic_subject import (
    _extract_quantified_comparison,
    extract_measured_discrepancy,
    resolve_deviation,
)


# ---------------------------------------------------------------------------
# A. Comparison frames -- resolver level
# ---------------------------------------------------------------------------

# (finding, expected comparison_type, magnitude, unit, subject-substring)
_CMP = [
    ("The reconciliation of inventory location IL-4 showed a shortfall of 120 units "
     "against the system record.", "BELOW", 120.0, "units", "il-4"),
    ("Warehouse count for SKU-991 showed a variance of 37 units versus the WMS quantity.",
     "MISMATCH", 37.0, "units", "sku-991"),
    ("The measured result differed from the approved value by 4.2 percent.",
     "MISMATCH", 4.2, "%", "measured result"),
    ("Batch B-77 recorded yield exceeded the calculated yield by 3 percent.",
     "EXCEEDED", 3.0, "%", "yield"),
    ("The chamber temperature was 2 degrees above the approved setpoint.",
     "EXCEEDED", 2.0, "degrees", "temperature"),
    ("Physical cash on hand was 500 units below the ledger balance.",
     "BELOW", 500.0, "units", "cash"),
    ("The vendor invoice total exceeded the purchase order by 12 units.",
     "EXCEEDED", 12.0, "units", "invoice"),
    ("The reported headcount fell short of the approved establishment by 4.",
     "BELOW", 4.0, None, "headcount"),
]


@pytest.mark.parametrize("finding,ctype,mag,unit,subj", _CMP,
                         ids=[c[0][:32] for c in _CMP])
def test_comparison_magnitude_and_reference_preserved(finding, ctype, mag, unit, subj):
    d = resolve_deviation(finding)
    assert d.semantic_type == "COMPARISON", (finding, d.semantic_type, d.condition)
    assert d.comparison_type == ctype
    assert d.measurement_value == mag
    if unit is not None:
        assert d.measurement_unit == unit
    assert d.comparison_left and d.comparison_right
    assert subj in (d.finding_subject or "").lower()
    # the reference/baseline survives
    assert any(w in (d.comparison_right or "").lower()
               for w in ("system", "wms", "approved", "calculated", "ledger", "setpoint",
                         "purchase order", "establishment", "record", "quantity", "value", "balance"))
    # magnitude is in the condition text, not a bare "not matched"
    assert d.condition and d.condition != "status unconfirmed"
    assert str(int(mag)) in d.condition or f"{mag:g}" in d.condition
    # no fabricated direction for a bare "differed"
    if "differ" in finding and "by" not in finding.split("differed")[1][:6]:
        assert d.comparison_type == "MISMATCH"


def test_bare_differed_has_unknown_direction():
    d = resolve_deviation("The recorded value differed from the calculated value.")
    assert d.semantic_type == "COMPARISON"
    assert d.comparison_type == "MISMATCH"        # not BELOW / EXCEEDED
    assert d.measurement_value is None            # nothing invented


def test_measured_discrepancy_frames():
    assert extract_measured_discrepancy("a shortfall of 120 units")[:2] == (120.0, "units")
    assert extract_measured_discrepancy("the difference was approximately 4.2%")[:2] == (4.2, "%")
    assert extract_measured_discrepancy("a variance of 37 items")[:2] == (37.0, None)
    assert extract_measured_discrepancy("the pump was serviced on 3 May") is None
    assert extract_measured_discrepancy("3 batches were affected") is None


def test_quantified_comparison_helper_is_structural():
    r = _extract_quantified_comparison(
        "The stocktake showed a shortfall of 8 units against the register."
    )
    assert r and r["type"] == "BELOW" and r["magnitude"] == 8.0
    assert "register" in r["right"].lower()
    assert _extract_quantified_comparison("The training was not completed.") is None


# ---------------------------------------------------------------------------
# B. Recurrence frames -- resolver + recurrence_guard
# ---------------------------------------------------------------------------

_REC = [
    ("Equipment M-204 experienced three failures over a six-month period.", 3, "failures", "M-204"),
    ("Line L-9 had four incidents during the previous year.", 4, "incidents", "L-9"),
    ("The laboratory recorded five deviations in the last quarter.", 5, "deviations", "laboratory"),
    ("Three complaints were received over two months regarding product PX-2.", 3, "complaints", None),
    ("Server SRV-3 logged six alarms within the last week.", 6, "alarms", "SRV-3"),
]


@pytest.mark.parametrize("finding,count,event,subj", _REC, ids=[c[0][:30] for c in _REC])
def test_recurrence_count_event_period_preserved(finding, count, event, subj):
    d = resolve_deviation(finding)
    assert d.semantic_type == "RECURRENCE"
    assert d.recurrence_count == count
    assert event in (d.recurrence_event or "")
    assert d.recurrence_period and d.recurrence_period != "UNKNOWN"
    assert d.condition and d.condition != "status unconfirmed"
    assert str(count) in d.condition
    if subj is not None:
        assert subj.lower() in (d.finding_subject or "").lower()


def test_recurrence_guard_extractor():
    assert extract_quantitative_recurrence("three failures over six months") == (
        3, "failures", "six months")
    assert extract_quantitative_recurrence("the pump failed twice") is None      # no period
    assert extract_quantitative_recurrence("multiple failures over a year") is None  # no explicit count


def test_quantitative_recurrence_does_not_trigger_recurrence_risk():
    from app.agent.recurrence_guard import detect_recurrence
    r = detect_recurrence("Equipment M-204 experienced three failures over a six-month period.")
    assert r.recurrence_count == 3
    assert r.is_recurring is False        # count alone is NOT a recurrence-RISK signal


# ---------------------------------------------------------------------------
# C. End-to-end: information survives to the canonical state + downstream.
# ---------------------------------------------------------------------------

async def _pipeline(finding_text: str):
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


_E2E = [
    "The reconciliation of inventory location IL-4 showed a shortfall of 120 units against the system record.",
    "The measured result for assay A-7 differed from the approved value by 4.2 percent.",
    "Pump P-14 experienced three failures over a six-month period.",
    "Vendor V-3 delivered 6 shipments late over the last quarter.",
]


@pytest.mark.parametrize("finding", _E2E, ids=[f[:34] for f in _E2E])
def test_information_survives_end_to_end(finding):
    state, ok, violations = asyncio.run(_pipeline(finding))
    assert ok, f"invariants violated: {violations}"
    cf = state["canonical_finding_state"]

    rc = state.get("root_cause_result") or state.get("root_cause")
    status = getattr(getattr(rc, "status", None), "value", getattr(rc, "status", None))
    assert status in ("NOT_ESTABLISHED", None)          # no cause inferred from the number
    assert getattr(rc, "risk_of_recurrence", "NOT_ASSESSABLE") != "HIGH" or \
        "recur" in (state["request"].finding_text.lower())  # count alone -> not HIGH

    if cf.semantic_type == "COMPARISON":
        assert cf.comparison_type in ("BELOW", "EXCEEDED", "MISMATCH", "INCONSISTENT",
                                      "RECONCILIATION_FAILURE")
        assert cf.comparison_left and cf.comparison_right
        assert cf.measurement is not None and cf.measurement.value is not None
        assert cf.deviation_condition and cf.deviation_condition != "status unconfirmed"
    if cf.semantic_type == "RECURRENCE" and "experienced" in finding or "delivered" in finding:
        assert cf.recurrence_count is not None
        assert cf.recurrence_event and cf.recurrence_period
        assert cf.affected_period != "UNKNOWN"


def test_ordinary_finding_unaffected():
    state, ok, violations = asyncio.run(_pipeline(
        "The calibration certificate for gauge G-7 had expired."
    ))
    assert ok, violations
    cf = state["canonical_finding_state"]
    assert cf.semantic_type not in ("COMPARISON", "RECURRENCE")
    assert cf.measurement is None
    assert cf.recurrence_count is None


def test_stated_causal_alternatives_still_intact():
    """The prior causal-alternatives fix must not regress."""
    state, ok, _ = asyncio.run(_pipeline(
        "Physical stock at location IL-4 was 120 units below the system record. The "
        "discrepancy could have resulted from unrecorded issues, a miscount during the "
        "physical check, or a data-entry error in the system, and the records did not "
        "allow these to be distinguished."
    ))
    cf = state["canonical_finding_state"]
    assert len(cf.stated_causal_alternatives) >= 2
    rc = state.get("root_cause_result") or state.get("root_cause")
    assert len(getattr(rc, "candidate_hypotheses", []) or []) >= 2
