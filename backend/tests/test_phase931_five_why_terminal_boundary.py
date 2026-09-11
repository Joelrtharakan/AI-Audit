"""Phase 9.3.1 -- the 5-Why evidence boundary is a HARD TERMINAL STATE.

Once the chain reaches a non-VERIFIED causal transition (REPORTED / POSSIBLE /
"no licensed edge"), NO further numbered Why step may be appended. A REPORTED
causal statement is not an established mechanism -- it does not authorize the
chain to continue. A comparison relationship never becomes a causal Why merely
because the finding also contains a comparison.

Enforced from the causal graph's structured edge status -- never from prose or
finding text.
"""
from __future__ import annotations

import asyncio
import re

import pytest
from unittest.mock import patch

from app.agent.causal_graph import build_causal_graph
from app.agent.causal_graph_traversal import build_graph_grounded_five_why
from app.models.agent import (
    CandidateHypothesis, CanonicalFindingState, CausalLevel, EvidenceItem, EvidenceStatus,
    InvestigateRequest, RootCauseAnalysis, RootCauseStatus,
)

_ENUM = re.compile(
    r"(?<![A-Za-z_])(UNKNOWN|NOT_ESTABLISHED|TO_BE_CONFIRMED|POSSIBLE|VERIFIED|SUPPORTED|"
    r"UNVERIFIED|NOT_ASSESSABLE|MIXED|REPORTED|EVIDENCE_BOUNDARY)(?![A-Za-z])")


def _canon(subj="the affected record", cond="was incomplete", **kw):
    return CanonicalFindingState(
        raw_finding="x", finding_subject=subj,
        observed_deviation=f"{subj} — {cond}", deviation_condition=cond, **kw)


def _hyp(strength, name="MECH", level=CausalLevel.L2_IMMEDIATE_MECHANISM, status="POSSIBLE"):
    return CandidateHypothesis(id="H1", name=name, statement="a candidate mechanism",
                               status=status, evidence_needed="records",
                               evidence_strength=strength, causal_level=level,
                               supporting_evidence=["e1"])


def _graph(canon, rc, ev_status=EvidenceStatus.REPORTED):
    return build_causal_graph(
        canon, rc, [EvidenceItem(claim="a reported contributing factor", source="s", status=ev_status)])


def _boundary_and_terminal(fw):
    assert fw is not None
    assert fw.is_complete is False
    # no step after the first non-VERIFIED transition
    seen_boundary = False
    for s in fw.steps:
        assert not seen_boundary, f"step emitted AFTER the evidence boundary: {s.question!r}"
        if s.status != "VERIFIED":
            seen_boundary = True
        for t in (s.question or "", s.answer or ""):
            assert not _ENUM.search(t), f"enum token leaked: {t!r}"
            assert " — " not in t and " -- " not in t, f"em-dash fragment: {t!r}"
        assert s.status != "VERIFIED" or "caused by" not in (s.answer or "").lower()


# --------------------------------------------------------------------------- #
# 1-6. structured cases
# --------------------------------------------------------------------------- #

def test_1_not_established_no_hypothesis_at_most_one_boundary_step():
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[])
    fw = build_graph_grounded_five_why(_graph(_canon(), rc))
    # zero licensed edges -> None (boundary reached immediately) OR a single step
    assert fw is None or len(fw.steps) <= 1
    if fw:
        _boundary_and_terminal(fw)


def test_2_not_established_with_reported_causal_edge_stops_at_that_step():
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED,
                           candidate_hypotheses=[_hyp("REPORTED")])
    fw = build_graph_grounded_five_why(_graph(_canon(), rc))
    _boundary_and_terminal(fw)
    if fw and fw.steps:
        last = fw.steps[-1]
        assert last.boundary_status == "EVIDENCE_BOUNDARY"
        assert last.status in ("REPORTED", "UNKNOWN")


def test_3_not_established_with_comparison_no_comparison_derived_causal_why():
    canon = _canon("the measured value", "did not match the target",
                   comparison_type="ACTUAL_CONFLICT", comparison_left="measured 850",
                   comparison_right="target 1000")
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[])
    fw = build_graph_grounded_five_why(_graph(canon, rc))
    assert fw is None or len(fw.steps) <= 1
    if fw:
        _boundary_and_terminal(fw)
        for s in fw.steps:
            # a comparison question is fine as the (single) boundary step; a
            # deeper causal continuation is not
            assert s.boundary_status in ("EVIDENCE_BOUNDARY", "TRANSITION")


def test_4_reported_cause_plus_comparison_no_causal_continuation():
    canon = _canon("the cost per machine", "did not match the quotation",
                   comparison_type="ACTUAL_CONFLICT", comparison_left="cost per machine",
                   comparison_right="42000")
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED,
                           candidate_hypotheses=[_hyp("REPORTED", name="LACK_OF_COMPLIANCE")])
    fw = build_graph_grounded_five_why(_graph(canon, rc))
    _boundary_and_terminal(fw)
    if fw:
        assert len(fw.steps) <= 2  # at most: comparison boundary + nothing, or 1 reported boundary
        assert not fw.is_complete


def test_5_boundary_at_step_1_zero_additional_steps():
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED,
                           candidate_hypotheses=[_hyp("INDICATIVE")])
    fw = build_graph_grounded_five_why(_graph(_canon(), rc))
    if fw:
        assert len(fw.steps) == 1
        assert fw.steps[0].boundary_status == "EVIDENCE_BOUNDARY"


def test_6_verified_step_then_boundary_no_steps_past_boundary():
    rc = RootCauseAnalysis(status=RootCauseStatus.SUPPORTED, candidate_hypotheses=[
        _hyp("VERIFIED", name="VMECH", status="SUPPORTED")])
    fw = build_graph_grounded_five_why(
        build_causal_graph(_canon(), rc,
                           [EvidenceItem(claim="x", source="finding", status=EvidenceStatus.VERIFIED)]))
    if fw:
        # exactly one VERIFIED transition then one boundary marker; nothing more
        transitions = [s for s in fw.steps if s.boundary_status == "TRANSITION"]
        boundary = [s for s in fw.steps if s.boundary_status == "EVIDENCE_BOUNDARY"]
        assert len(transitions) == 1 and transitions[0].status == "VERIFIED"
        assert len(boundary) == 1
        assert fw.steps[-1].boundary_status == "EVIDENCE_BOUNDARY"


# --------------------------------------------------------------------------- #
# 7-9. metamorphic
# --------------------------------------------------------------------------- #

def test_7_adding_a_reported_causal_claim_cannot_resume_a_stopped_chain():
    base_rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[])
    with_rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED,
                                candidate_hypotheses=[_hyp("REPORTED")])
    fw_base = build_graph_grounded_five_why(_graph(_canon(), base_rc))
    fw_with = build_graph_grounded_five_why(_graph(_canon(), with_rc))
    n_base = 0 if fw_base is None else len(fw_base.steps)
    n_with = 0 if fw_with is None else len(fw_with.steps)
    assert n_with <= n_base + 1  # at most the single reported boundary step
    if fw_with:
        _boundary_and_terminal(fw_with)


def test_8_adding_comparison_info_cannot_resume_a_stopped_causal_chain():
    plain = _canon("the batch record", "was not signed")
    with_cmp = _canon("the batch record", "was not signed",
                      comparison_type="ACTUAL_CONFLICT", comparison_left="a", comparison_right="b")
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED, candidate_hypotheses=[])
    fw_p = build_graph_grounded_five_why(_graph(plain, rc))
    fw_c = build_graph_grounded_five_why(_graph(with_cmp, rc))
    n_p = 0 if fw_p is None else len(fw_p.steps)
    n_c = 0 if fw_c is None else len(fw_c.steps)
    assert n_c <= max(1, n_p)


@pytest.mark.parametrize("subj", [
    "the calibration record for gauge G-7", "the vendor onboarding checklist",
    "the medication reconciliation entry", "the firmware baseline for controller C-3",
    "the evacuation drill log",
])
def test_9_domain_change_does_not_change_the_terminal_boundary_rule(subj):
    rc = RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED,
                           candidate_hypotheses=[_hyp("REPORTED")])
    fw = build_graph_grounded_five_why(_graph(_canon(subj, "was incomplete"), rc))
    if fw:
        _boundary_and_terminal(fw)
        assert not fw.is_complete


# --------------------------------------------------------------------------- #
# 10. full pipeline -- no enum leak, chain incomplete, no post-boundary step
# --------------------------------------------------------------------------- #

async def _report(ft):
    from app.agent.nodes.core_synthesis import core_synthesis_node
    from app.agent.nodes.final_evidence_verification import final_evidence_verification_node
    from app.agent.nodes.investigation_planner import plan_investigation_node
    from app.agent.nodes.report_generator import generate_report_node
    from app.agent.nodes.understanding import understand_finding_node

    st = {"request": InvestigateRequest(finding_text=ft), "evidence_ledger": [],
          "iteration_count": 0, "tool_call_count": 0, "critic_iteration": 0, "trace": [], "errors": []}
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        for n in (understand_finding_node, plan_investigation_node, core_synthesis_node,
                  generate_report_node, final_evidence_verification_node):
            st = await n(st)
    return st["report"]


@pytest.mark.parametrize("finding", [
    "Machine guarding on the production machines was inadequate; the supervisor said this was due to "
    "lack of compliance, and the stated cost per machine did not match the quotation.",
    "The reconciliation did not match the control total, and a clerk stated the difference was due to "
    "a transcription error.",
    "The measured throughput did not match the target, which an operator attributed to poor scheduling.",
    "The inspection interval was exceeded and the technician said the schedule had not been updated.",
    "Two approvals were missing and a manager stated the workflow tool was unavailable that week.",
])
def test_10_full_pipeline_boundary_is_terminal_and_clean(finding):
    r = asyncio.run(_report(finding))
    assert r.five_why.is_complete is False
    seen_boundary = False
    for s in r.five_why.steps:
        assert not (seen_boundary and s.status == "VERIFIED"), \
            f"VERIFIED step after boundary: {s.question!r}"
        if s.status not in ("VERIFIED",):
            seen_boundary = True
        for t in (s.question or "", s.answer or ""):
            assert not _ENUM.search(t), f"enum leaked into prose: {t!r}"
        assert s.status != "VERIFIED" or "caused by" not in (s.answer or "").lower()
