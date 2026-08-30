"""Domain-agnostic semantic-role test matrix.

25 grammatical forms, entities drawn from unrelated domains AS TEST DATA
ONLY -- the resolver must generalise by grammatical/semantic role, never by
keyword. Each case asserts the AFFECTED OBJECT is the audited subject and
never the evidence source / a clause fragment / a generic noun / an
unresolved marker.
"""

from __future__ import annotations

import pytest

from app.services.semantic_subject import (
    is_actor_noun,
    reject_subject_if_clause,
    resolve_deviation,
)

# (id, finding, must_contain_in_subject | None, must_NOT_contain_in_subject)
CASES = [
    ("direct",            "The centrifuge CF-7 exceeded its validated speed limit.", "cf-7", ()),
    ("passive",           "The calibration certificate for scale SC-3 was not renewed.", "sc-3", ()),
    ("evsrc_frame",       "Maintenance records show that pump P-9 failed twice.", "p-9",
     ("record", "maintenance")),
    ("negated_frame",     "Complaint records did not demonstrate that recurrence of defect D-2 was evaluated.",
     "d-2", ("complaint record",)),
    ("reported_speech",   "The operator reported that the interlock on press PR-4 was bypassed.", "pr-4",
     ("operator",)),
    ("nested_prop",       "It was noted that the review concluded that training module TM-1 was outdated.",
     "tm-1", ("review", "concluded", "noted")),
    ("multi_entity",      "Valve V-12 and gauge G-8 on skid SK-2 were found out of calibration.", None,
     ("record",)),
    ("multi_evsrc",       "Both the SCADA log and the shift handover record failed to show that "
                          "alarm A-3 was acknowledged.", "a-3", ("scada log", "handover record")),
    ("missing_subject",   "The required verification was not performed before release.", "verification",
     ("record",)),
    ("generic_subject",   "The process was not followed correctly.", None, ("not followed",)),
    ("reported_claim",    "Staff stated that refresher training for procedure SOP-22 had lapsed.", "sop-22",
     ("staff",)),
    ("temporal",          "Between March and June, temperature excursions for freezer FZ-5 went unlogged.",
     "fz-5", ("march", "june")),
    ("multi_observation", "The fire extinguisher FE-9 was overdue for inspection and its tamper seal "
                          "was broken.", "fe-9", ()),
    ("records_show",      "Audit trail records show that access for user U-77 was not revoked on termination.",
     "u-77", ("audit trail", "on termination")),
    ("records_did_not",   "The training log did not record that competency for task T-9 was reassessed.",
     "t-9", ("training log",)),
    ("inspection_found",  "Inspection found excessive wear on bearing BR-2.", "br-2", ("inspection",)),
    ("record_contained",  "The device history record for unit DH-4 contained an unapproved change.",
     "dh-4", ()),
    ("employee_reported", "The technician reported that the pressure relief valve PRV-6 had not been tested.",
     "prv-6", ("technician",)),
    ("it_was_observed",   "It was observed that the emergency lighting on floor 3 was inoperative.",
     "emergency lighting", ("observed",)),
    ("evidence_indicates","Evidence indicates that segregation of duties for approver AP-2 was not enforced.",
     "segregation of duties", ("evidence indicates",)),
    ("procedure_not_followed", "The lockout-tagout procedure was not followed during maintenance of "
                               "conveyor CV-1.", "lockout-tagout procedure", ()),
]


@pytest.mark.parametrize("cid,finding,must_contain,must_not", CASES, ids=[c[0] for c in CASES])
def test_semantic_role_matrix(cid, finding, must_contain, must_not):
    r = resolve_deviation(finding, [finding])
    subj = (r.finding_subject or "").lower()

    # Never an internal marker leaking into the "subject"
    assert not subj.startswith(("unknown", "unresolved", "not established"))
    # Never a clause fragment
    assert not reject_subject_if_clause(r.finding_subject) if r.finding_subject else True

    for bad in must_not:
        assert bad.lower() not in subj, f"{cid}: evidence-source/fragment {bad!r} leaked into subject {subj!r}"

    if must_contain is not None:
        assert must_contain.lower() in subj, f"{cid}: expected {must_contain!r} in subject, got {subj!r}"


def test_actor_as_oblique_is_not_the_subject_head():
    # "user"/"approver" after a preposition is context, not the phrase head.
    assert is_actor_noun("access for user U-77") is False
    assert is_actor_noun("segregation of duties for approver AP-2") is False
    assert is_actor_noun("training for the operators") is False
    # bare actor phrases are still actor-headed
    assert is_actor_noun("four operators") is True
    assert is_actor_noun("the responsible technician") is True


def test_reporting_cognition_clause_is_rejected():
    for clause in (
        "review concluded that training module TM-1 was outdated",
        "the audit found that training was incomplete",
        "it determined that the control had lapsed",
        "not followed correctly",
        "not performed before release",
    ):
        assert reject_subject_if_clause(clause) is True
