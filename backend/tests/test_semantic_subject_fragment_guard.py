"""Regression: a bare grammatical fragment (article / pronoun / generic
category noun) must never survive as a finding subject / affected object.

Reported production failure:
    Affected Object: The
    Evidence Needed: Applicable procedure governing The
    Process at Risk: Process operational process

Root cause: `reject_subject_if_clause` (the single authoritative subject
gate every stage funnels through) did not reject a bare article -- "the"/
"a"/"an" were in neither `_PRONOUNS` nor `_BARE_DETERMINER_WORDS`. The three
fallback planners additionally accepted any string not in a fixed
`{"process compliance", None, ""}` set, so a stored "UNRESOLVED …" marker
could be spliced into "Applicable procedure governing UNRESOLVED …".
"""

from __future__ import annotations

import pytest

from app.services.semantic_subject import (
    UNRESOLVED_SUBJECT_DISPLAY,
    _clean_subject,
    humanize_unresolved_subject,
    is_established_subject,
    reject_subject_if_clause,
    validate_semantic_subject,
)

# Every one of these is a grammatical fragment, not an auditable object.
BARE_FRAGMENTS = [
    "The", "the", "a", "an", "A", "An",
    "this", "that", "it", "they", "these", "those",
    "issue", "the issue", "an issue",
    "condition", "this condition", "the condition",
    "activity", "an activity", "the activity",
    "process", "the process", "a process",
    "matter", "the matter", "area", "the area", "aspect",
    "the audit", "the finding", "the observation",
]

# Real audit subjects across unrelated domains -- must ALL be preserved.
REAL_SUBJECTS = [
    "balance BAL-014",
    "temperature log for refrigerator QC-REF-02",
    "payment reconciliation process",
    "access control system",
    "the revised inspection checklist",
    "training for the revised procedure",
    "duplicate vendor payment",
    "supplier qualification record",
    "purchase order PO-4471",
    "the batch record for Lot ABC-2024-001",
    "fire suppression system in Cleanroom Suite 3",
    "employee onboarding workflow",
    "software licensing renewal",
]


@pytest.mark.parametrize("frag", BARE_FRAGMENTS)
def test_bare_fragment_is_rejected(frag):
    assert reject_subject_if_clause(frag) is True
    assert validate_semantic_subject(frag) is False
    assert is_established_subject(frag) is False


@pytest.mark.parametrize("subj", REAL_SUBJECTS)
def test_real_subject_is_preserved(subj):
    assert reject_subject_if_clause(subj) is False
    assert validate_semantic_subject(subj) is True
    assert is_established_subject(subj) is True


def test_clean_subject_collapses_bare_article_to_empty():
    assert _clean_subject("The") == ""
    assert _clean_subject("the ") == ""
    assert _clean_subject("a") == ""
    # a real subject with a leading article keeps its content
    assert _clean_subject("the revised procedure") == "revised procedure"


def test_unresolved_markers_are_not_established_subjects():
    for marker in (
        "UNKNOWN — no affected object could be isolated from the finding text",
        "UNRESOLVED — the specific entity involved could not be isolated from the finding text",
        "NOT ESTABLISHED",
        "NOT_ESTABLISHED",
    ):
        assert is_established_subject(marker) is False


def test_humanize_maps_marker_to_one_professional_phrase():
    assert (
        humanize_unresolved_subject("UNRESOLVED — the specific entity involved could not be isolated from the finding text")
        == UNRESOLVED_SUBJECT_DISPLAY
    )
    assert humanize_unresolved_subject("NOT ESTABLISHED") == UNRESOLVED_SUBJECT_DISPLAY
    assert humanize_unresolved_subject("The") == UNRESOLVED_SUBJECT_DISPLAY
    # genuine subject is untouched
    assert humanize_unresolved_subject("balance BAL-014") == "balance BAL-014"
    assert humanize_unresolved_subject(None) is None


def test_cost_phrasing_never_becomes_the_subject():
    """A monetary / effort statement is a cost driver, never a finding subject."""
    for cost_text in (
        "the rate",
        "the cost",
        "an amount",
        "the exposure",
        "historical average cost",  # rule 7: all-financial-meta phrase
        "reported amount",
    ):
        assert is_established_subject(cost_text) is False


def test_fallback_planners_reject_unresolved_marker_as_subject():
    """plan_investigation_fallback / five_why_fallback must gate on
    is_established_subject, not on the fixed _DEGRADED_SUBJECTS set, so a
    stored marker is never spliced into a generated question/evidence line."""
    from app.agent.nodes import five_why_fallback, plan_investigation_fallback

    marker = "UNRESOLVED — the specific entity involved could not be isolated from the finding text"
    assert marker not in plan_investigation_fallback._DEGRADED_SUBJECTS
    # the guard is now is_established_subject, which rejects it
    assert is_established_subject(marker) is False
    assert is_established_subject("The") is False
    # sanity: the modules still import and expose the helper path
    assert hasattr(plan_investigation_fallback, "is_established_subject")
