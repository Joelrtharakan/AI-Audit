"""Explicitly stated competing causal mechanisms must be PRESERVED, not
collapsed to a bare "root cause unknown".

When a finding text enumerates its own causal differential
("could have resulted from A, B, or C"), the pipeline must:
  * keep every alternative as a POSSIBLE / unranked candidate hypothesis;
  * keep root_cause NOT_ESTABLISHED, leading_hypothesis NONE;
  * produce a discrimination-focused investigation question naming them;
  * preserve them in the 5-Why evidence-boundary answer.

Structural / domain-neutral -- the extractor keys on the grammatical
"causal connector + coordinated list" frame, never on finding vocabulary.
These tests exercise the deterministic path (LLM unavailable), which is
where the defect was demonstrated.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.agent.causal_guard import (
    extract_stated_causal_alternatives,
    stated_alternatives_unresolved,
)
from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.report_generator import generate_report_node
from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import InvestigateRequest, RootCauseStatus


# ---------------------------------------------------------------------------
# 1. The extractor -- structural, domain-neutral.
# ---------------------------------------------------------------------------

_HAS_ALTS = [
    ("The stock discrepancy could have resulted from unrecorded issues, a miscount "
     "during the physical check, or a data-entry error in the system.", 3),
    ("The equipment failure may have been caused by a worn bearing or a lubrication fault.", 2),
    ("The variance could be due to a timing difference, an unrecorded adjustment, or a "
     "posting error.", 3),
    ("The access-log gap might have been caused by a session-logging fault or manual "
     "deletion.", 2),
    ("The out-of-tolerance reading may stem from sensor drift, an environmental excursion, "
     "or an incorrect setpoint.", 3),
    ("Possible causes include incomplete handover, an unclear procedure, and a missed "
     "notification.", 3),
    ("The delay could be attributable to a supplier scheduling issue or an internal "
     "approval bottleneck.", 2),
]

_NO_ALTS = [
    "The calibration certificate for gauge G-7 had expired.",
    "The required second-person verification was not documented.",
    "The batch was rejected because the assay result failed the specification.",
    "Root cause could not be determined from the available evidence.",
    "The supplier was not requalified within the required interval.",
    "The change was implemented without a documented risk assessment.",
]


@pytest.mark.parametrize("text,n", _HAS_ALTS)
def test_extractor_finds_stated_alternatives(text, n):
    alts = extract_stated_causal_alternatives(text)
    assert len(alts) == n, alts
    assert all(3 <= len(a) <= 120 for a in alts)
    assert not any(a.lower().startswith(("or ", "and ")) for a in alts)


@pytest.mark.parametrize("text", _NO_ALTS)
def test_extractor_does_not_fire_without_an_enumeration(text):
    assert extract_stated_causal_alternatives(text) == []


def test_unresolved_marker():
    assert stated_alternatives_unresolved(
        "the available records did not allow these to be distinguished"
    )
    assert stated_alternatives_unresolved("the two mechanisms remain indistinguishable")
    assert not stated_alternatives_unresolved("the cause was confirmed to be sensor drift")


# ---------------------------------------------------------------------------
# 2. End-to-end (deterministic path) across unrelated domains.
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


_E2E_FINDINGS = [
    ("inventory",
     "Physical stock at location IL-4 was 120 units below the system record. The discrepancy "
     "could have resulted from unrecorded issues, a miscount during the physical check, or a "
     "data-entry error in the system, and the records did not allow these to be distinguished."),
    ("it_access",
     "Audit-trail entries for the payment system were missing for a three-day window. The gap "
     "may have been caused by a session-logging service fault or by manual deletion of entries."),
    ("equipment",
     "Pump P-7 tripped on high vibration twice in one week. The trips could be due to bearing "
     "wear, shaft misalignment, or a resonance condition, and the available data did not "
     "distinguish them."),
    ("finance",
     "The bank reconciliation for account AC-9 showed an unexplained difference of 3,400. The "
     "difference could be attributable to a timing difference, an unrecorded fee, or a posting "
     "error."),
    ("lab",
     "The assay result for sample S-231 was outside the expected range. The deviation may stem "
     "from a sample-preparation error, instrument drift, or a reagent-quality issue."),
    ("supplier",
     "Two consecutive deliveries from vendor V-12 arrived late. The delays could be attributable "
     "to a supplier scheduling issue or an internal goods-receipt bottleneck."),
]


@pytest.mark.parametrize("dom,finding", _E2E_FINDINGS, ids=[c[0] for c in _E2E_FINDINGS])
def test_stated_alternatives_preserved_end_to_end(dom, finding):
    state, ok, violations = asyncio.run(_pipeline(finding))
    assert ok, f"{dom}: invariants violated: {violations}"

    cf = state["canonical_finding_state"]
    alts = list(cf.stated_causal_alternatives)
    assert len(alts) >= 2, f"{dom}: alternatives lost from canonical state"

    rc = state.get("root_cause_result") or state.get("root_cause")
    status = getattr(getattr(rc, "status", None), "value", getattr(rc, "status", None))
    assert status in ("NOT_ESTABLISHED", RootCauseStatus.NOT_ESTABLISHED, None)
    assert not getattr(rc, "leading_hypothesis", None) or "NONE" in str(
        getattr(rc, "leading_hypothesis", "")
    ).upper()

    hyps = getattr(rc, "candidate_hypotheses", []) or []
    assert len(hyps) >= len(alts), f"{dom}: expected one hypothesis per stated alternative"
    assert all(h.status in ("POSSIBLE", "UNVERIFIED", "UNRESOLVED") for h in hyps)
    # each stated alternative is reflected in some hypothesis statement
    _blob = " ".join(h.statement.lower() for h in hyps)
    for a in alts:
        key = a.lower().split()[-1]  # a distinctive token
        assert key in _blob, f"{dom}: alternative {a!r} not in any hypothesis"

    # a discrimination question exists and names the mechanisms
    qs = [q.question.lower() for q in state["investigation_plan"].questions]
    assert any("distinguish" in q or "discriminate" in q for q in qs), f"{dom}: no discrimination question"

    # the 5-Why boundary answer preserves the alternatives, does not fabricate a cause
    fw = state.get("five_why_analysis") or state.get("five_why")
    assert not getattr(fw, "is_complete", True)
    _fw_text = " ".join(s.answer.lower() for s in getattr(fw, "steps", []))
    assert "mechanisms remaining" in _fw_text or "discriminate between them" in _fw_text
    for banned in ("forgot", "was ignored", "was overlooked", "negligence"):
        assert banned not in _fw_text


def test_ordinary_finding_without_alternatives_is_unchanged():
    """No enumeration -> no hypotheses invented, generic boundary note kept."""
    state, ok, violations = asyncio.run(_pipeline(
        "The required environmental monitoring for cleanroom CR-2 was not performed during "
        "the second quarter."
    ))
    assert ok, violations
    cf = state["canonical_finding_state"]
    assert cf.stated_causal_alternatives == []
    fw = state.get("five_why_analysis") or state.get("five_why")
    _fw_text = " ".join(s.answer.lower() for s in getattr(fw, "steps", []))
    assert "mechanisms remaining" not in _fw_text
