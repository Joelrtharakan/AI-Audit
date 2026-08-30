"""Domain-agnostic semantic-subject QUALITY matrix.

15 structurally-different findings across unrelated domains (entities are
TEST DATA, never logic). For each: (A) the canonical affected subject is the
substantive audited object -- not an event/action clause, evidence source,
belief, hypothesis, requirement phrase, or whole sentence; and (B) the
downstream investigation questions embed that canonical subject (no clause
fragment, no "for was", no belief/hypothesis phrase) and the final report
carries no leaked BELIEF/HYPOTHESIS/EVIDENCE-source subject.

Grammar/semantic-role assertions only -- no domain keyword is tested.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.investigation_planner import plan_investigation_node
from app.agent.nodes.understanding import understand_finding_node
from app.models.agent import InvestigateRequest
from app.services.semantic_subject import reject_subject_if_clause


async def _pipeline(finding: str):
    state = {
        "request": InvestigateRequest(finding_text=finding),
        "evidence_ledger": [], "iteration_count": 0, "tool_call_count": 0,
        "critic_iteration": 0, "trace": [], "errors": [],
    }
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.investigation_planner.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        state = await understand_finding_node(state)
        state = await plan_investigation_node(state)
        state = await core_synthesis_node(state)
    return state


# id, finding, expected substring in subject | None (=> must be unresolved),
# phrases that must NOT appear anywhere in subject or questions
CASES = [
    ("1_entity_event",
     "Centrifuge CF-7 was operated above its validated speed on 12 March.",
     "cf-7", ()),
    ("2_entity_missing_control_evidence",
     "The maintenance of chiller CH-2 was performed, but the post-maintenance verification was not recorded.",
     "verification", ("the maintenance of chiller", "chiller ch-2 was")),
    ("3_process_control_deviation",
     "The invoice matching process bypassed the three-way match control for 14 payments.",
     "invoice matching process", ("14 payments",)),
    ("4_record_is_the_object",
     "The device history record for unit DH-4 contained an unapproved handwritten change.",
     "record", ()),
    ("5_transaction_is_the_object",
     "Journal entry JE-9920 was posted to a closed accounting period.",
     "je-9920", ("was posted",)),
    ("6_activity_is_the_object",
     "The annual supplier re-qualification was not completed for six critical suppliers.",
     "supplier re-qualification", ("six critical suppliers", "was not completed for")),
    ("7_evidence_source_reports_event",
     "Maintenance records show that compressor K-3 tripped twice during the shift.",
     "k-3", ("maintenance records", "records show")),
    ("8_belief_about_cause",
     "The engineer believed the pressure drop on loop L-8 was probably a calibration issue.",
     None, ("calibration issue", "pressure drop")),
    ("9_reported_accusation_concrete_event",
     "The auditor was told that operator badge access AC-3 was shared between two staff.",
     "ac-3", ()),
    ("10_unresolved_causal_statement",
     "The recurring stockouts at warehouse W-2 may be linked to an unreliable demand forecast.",
     "w-2", ("unreliable demand forecast",)),
    # A 3-clause finding: which of {batch, training, QA log} is primary is a
    # genuine ambiguity (§8). Only assert the subject is substantive (not a
    # clause fragment) and no raw clause leaks downstream.
    ("11_multiple_clauses",
     "Batch BR-88 was released; the release approver did not hold current training; the QA log was incomplete.",
     "", ("did not hold current",)),
    ("12_event_plus_requirement",
     "A firmware update was pushed to controller C-3, although the required rollback plan was not documented.",
     "rollback plan", ("was pushed", "firmware update")),
    ("13_no_identifiable_subject",
     "The process was not followed correctly during the last quarter.",
     None, ("not followed", "the process")),
    ("14_identifier_with_generic_phrase",
     "Records show that the item associated with reference R-4471 was disposed of without authorization.",
     "r-4471", ("records show", "the item associated")),
    ("15_fact_plus_hypothetical_cause",
     "Freezer FZ-5 lost power for three hours; this was likely caused by a tripped breaker.",
     "fz-5", ("tripped breaker", "likely caused")),

    # --- condition-nominalization class (the pass-8 defect) ---
    ("16_condition_nominalization_with_control",
     "There was inconsistent compliance with the gowning procedure in the fill area.",
     "gowning procedure", ("inconsistent compliance with", "in the fill area")),
    ("17_condition_nominalization_of_process",
     "Poor adherence to the change-control process was found across three departments.",
     "change-control process", ("poor adherence to", "three departments")),
    ("18_lack_of_relation",
     "Lack of oversight of contractor access to the server room was identified.",
     "contractor access", ("lack of oversight of",)),
    ("19_condition_causes_downstream",
     "Weak enforcement of the data-retention policy has resulted in premature deletion of records.",
     "data-retention policy", ("weak enforcement of",)),
    # "incomplete <artifact> for <entity> was <accepted>": whether the
    # subject is the artifact or the entity is a genuine ambiguity (§8);
    # only require a substantive, non-fragment subject.
    ("20_condition_is_genuinely_the_object",
     "The incomplete calibration certificate for balance BAL-2 was accepted at goods-in.",
     "", ("incomplete calibration",)),
    # segregation-of-duties: the fixed phrase stays whole
    ("21_segregation_of_duties",
     "Inadequate segregation of duties was observed in the invoice approval workflow.",
     "segregation of duties", ("inadequate segregation",)),
    # requirement + observed condition: the requirement's object is the subject
    # "The requirement to <verb> <object> was not met [for <entity>]" -- the
    # subject is the requirement's object (or the entity), NEVER the
    # requirement phrase itself.
    ("22_requirement_plus_condition",
     "The requirement to reconcile the sub-ledger monthly was not met for account AC-77.",
     "sub-ledger", ("the requirement to", "requirement to reconcile", "monthly")),
]


@pytest.mark.parametrize("cid,finding,must_contain,must_not", CASES, ids=[c[0] for c in CASES])
def test_subject_quality_and_downstream_propagation(cid, finding, must_contain, must_not):
    state = asyncio.run(_pipeline(finding))
    cf = state["canonical_finding_state"]
    subj = (cf.finding_subject or "")
    subj_l = subj.lower()

    # ---- A. canonical subject quality
    if must_contain is None:
        assert subj_l.startswith(("unresolved", "unknown")) or cf.semantic_type == "NON_ACTIONABLE", \
            f"{cid}: expected unresolved, got {subj!r}"
    else:
        assert not subj_l.startswith(("unresolved", "unknown")), f"{cid}: unexpectedly unresolved"
        assert not reject_subject_if_clause(subj), f"{cid}: subject is a clause: {subj!r}"
        assert must_contain in subj_l, f"{cid}: expected {must_contain!r} in subject {subj!r}"

    # ---- B. downstream propagation: questions + impact carry the canonical
    #        subject / semantic role, never a leaked clause or belief phrase
    plan = state.get("investigation_plan")
    q_text = " ".join(q.question for q in (plan.questions if plan else [])).lower()
    impact = state.get("impact_assessment")
    impact_text = " ".join(
        str(getattr(impact, f, "") or "")
        for f in ("affected_object", "process_at_risk", "potential_effect", "evidence_needed")
    ).lower() if impact else ""

    haystack = f"{subj_l} {q_text} {impact_text}"
    for bad in must_not:
        assert bad.lower() not in haystack, f"{cid}: leaked {bad!r} into subject/questions/impact"
    # never a grammatically broken template
    assert "for was" not in haystack and "for was?" not in haystack
    assert " the the " not in haystack
    # belief / hypothesis language must never appear as an impact/question subject
    for leak in ("believed", "probably a", "likely caused by", "may be linked to", "suspected that"):
        assert leak not in q_text, f"{cid}: epistemic phrase {leak!r} leaked into a question"
