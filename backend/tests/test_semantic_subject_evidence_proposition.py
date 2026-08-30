"""Regression: an evidence / reporting clause must never become the
canonical finding subject when the finding explicitly names an entity.

Reported failure:
    Finding: "Production equipment M-204 experienced four unplanned failures
             during the previous six months. Maintenance records show that
             temporary repairs were performed after each failure and no
             documented permanent corrective action was implemented."
    Wrong subject:   "Maintenance records show that temporary repairs"
    Correct subject: "Production equipment M-204"

Root cause: `_ENTITY_RE` required a 2+ letter prefix (missed "M-204"), and
`reject_subject_if_clause` / `_strip_framing` only recognised a narrow
hard-coded set of "system records show that" prefixes, so a generic
"<evidence source> <reporting verb> that <proposition>" second sentence was
picked up by the agentless-passive recovery rule.

Fix is structural (evidence-source head noun + reporting verb), never a
phrase blacklist -- see `looks_like_evidence_proposition`.
"""

from __future__ import annotations

import pytest

from app.services.semantic_subject import (
    extract_entities,
    is_established_subject,
    looks_like_evidence_proposition,
    reject_subject_if_clause,
    resolve_deviation,
    validate_semantic_subject,
)

EVIDENCE_CLAUSES = [
    "Maintenance records show that temporary repairs were performed",
    "Finance records confirm that two payments were processed",
    "Historical records show ten occurrences",
    "System logs indicate repeated authentication failures",
    "Production records demonstrate that output fell below target",
    "The review identified missing approvals",
    "Inspection found corrosion",
    "Testing confirmed the deviation",
    "Data indicates a downward trend",
    "History shows recurring shortages",
    "It was observed that the valve was left open",
    "It was found that no calibration record existed",
    "Audit trail records reveal that access was not revoked",
    "Our analysis suggests that controls were not applied",
]

REAL_SUBJECTS_WITH_EVIDENCE_HEAD_NOUN = [
    "batch record",
    "maintenance log",
    "training record",
    "inspection checklist",
    "audit trail",
    "test results",
    "calibration certificate",
    "the shipping documentation",
]


@pytest.mark.parametrize("clause", EVIDENCE_CLAUSES)
def test_evidence_clause_is_not_a_subject(clause):
    assert looks_like_evidence_proposition(clause) is True
    assert reject_subject_if_clause(clause) is True
    assert validate_semantic_subject(clause) is False
    assert is_established_subject(clause) is False


@pytest.mark.parametrize("subj", REAL_SUBJECTS_WITH_EVIDENCE_HEAD_NOUN)
def test_real_subject_with_evidence_head_noun_is_preserved(subj):
    # "batch record", "maintenance log" etc. are NOT reporting frames --
    # they are only flagged when followed by a reporting verb.
    assert looks_like_evidence_proposition(subj) is False
    assert validate_semantic_subject(subj) is True


def test_entity_first_over_later_evidence_clause():
    finding = (
        "Production equipment M-204 experienced four unplanned failures during the "
        "previous six months. Maintenance records show that temporary repairs were "
        "performed after each failure and no documented permanent corrective action "
        "was implemented."
    )
    assert "M-204" in extract_entities(finding)
    r = resolve_deviation(finding, [finding])
    subj = (r.finding_subject or "").lower()
    assert "m-204" in subj
    assert "maintenance records" not in subj
    assert "temporary repairs" not in subj


def test_entity_inside_reporting_clause_is_still_recovered():
    # "Inspection found corrosion on Tank T-14" -- the reporting frame is
    # stripped, but the entity lives INSIDE the clause, so it must survive.
    r = resolve_deviation("Inspection found corrosion on Tank T-14.", ["Inspection found corrosion on Tank T-14."])
    assert "t-14" in (r.finding_subject or "").lower()


def test_single_letter_asset_tags_are_entities():
    for tag, text in (
        ("M-204", "Equipment M-204 failed twice."),
        ("T-14", "Tank T-14 showed corrosion."),
        ("P-3", "Pump P-3 was offline."),
        ("L-5", "Line L-5 was stopped."),
    ):
        assert tag in extract_entities(text), text
    # ordinary hyphenated words are still not entities
    assert extract_entities("The in-process check was skipped on the well-documented line.") == []


def test_evidence_only_finding_falls_back_not_corrupts():
    # No entity anywhere, whole finding is a reporting proposition -> the
    # resolver must not emit the reporting clause as the subject.
    r = resolve_deviation(
        "Historical records show that ten similar occurrences were noted.",
        ["Historical records show that ten similar occurrences were noted."],
    )
    subj = r.finding_subject or ""
    assert not looks_like_evidence_proposition(subj)
    assert "records show" not in subj.lower()


# A REPORTED ACTION inside an evidence proposition ("... that <X> were
# performed / was implemented") is a remediation activity the source is
# reporting on -- never the audited entity. With no better entity in the
# finding the correct result is "not specifically identified", not the
# action phrase. A STATE participle ("was damaged / disabled / incomplete")
# is the opposite: the noun IS the affected entity and must still resolve.

@pytest.mark.parametrize("finding", [
    "Maintenance records show that temporary repairs were performed after each failure "
    "and no permanent corrective action was implemented.",
    "Audit records confirm that a review was conducted and that additional checks were added.",
    "System logs indicate that a patch was deployed and the service was restarted.",
])
def test_reported_action_object_is_not_the_subject(finding):
    r = resolve_deviation(finding, [finding])
    subj = (r.finding_subject or "").lower()
    for bad in ("temporary repairs", "corrective action", "a review", "additional checks",
                "a patch", "the service"):
        assert bad not in subj


@pytest.mark.parametrize("finding,expected", [
    ("Inspection found that the walkway surface was damaged.", "walkway surface"),
    ("It was observed that the fire door on level 3 was propped open.", "fire door"),
    ("System logs indicate that the backup job was disabled.", "backup job"),
    ("Audit records confirm that the reconciliation was not completed.", "reconciliation"),
])
def test_state_predicate_after_evidence_frame_still_resolves_the_entity(finding, expected):
    r = resolve_deviation(finding, [finding])
    assert expected in (r.finding_subject or "").lower()


# NEGATED reporting frame: "<record store> did not <reporting verb> that
# <proposition>". The finding is that the proposition (a required evaluation
# / action) was not established -- the record store is only the frame.

@pytest.mark.parametrize("finding,expected_subj,expected_cond", [
    ("Complaint records did not demonstrate that recurring failures of product PX-9 were evaluated.",
     "px-9", "not evaluated"),
    ("Temperature logs did not show that corrective action was taken after the excursion.",
     "corrective action", "not taken"),
    ("Maintenance records did not confirm that calibration of gauge G-12 was completed.",
     "g-12", "not completed"),
    ("The audit trail did not record that access reviews were performed for the finance system FS-2.",
     "fs-2", "not performed"),
])
def test_negated_reporting_frame_resolves_the_proposition_not_the_record(finding, expected_subj, expected_cond):
    r = resolve_deviation(finding, [finding])
    subj = (r.finding_subject or "").lower()
    assert expected_subj in subj
    assert "record" not in subj and "log" not in subj and "audit trail" not in subj
    assert (r.condition or "").lower() == expected_cond


def test_negated_reporting_frame_with_no_entity_is_unresolved_not_the_record():
    r = resolve_deviation(
        "Complaint records did not demonstrate that recurring failures were evaluated.",
        ["Complaint records did not demonstrate that recurring failures were evaluated."],
    )
    assert r.finding_subject is None
    assert not (r.matched)


def test_negated_frame_does_not_hijack_a_real_activity_subject():
    # "reconciliation" is an audit ACTIVITY, not a record store -> the
    # negated-frame rule must not fire; "reconciliation" stays the subject.
    r = resolve_deviation("The reconciliation did not show a variance.",
                          ["The reconciliation did not show a variance."])
    assert "reconciliation" in (r.finding_subject or "").lower()
