"""Regression coverage for a domain-specific-keyword entity fabrication bug
found while auditing the pipeline for generic semantic-role safety.

Bug: app/services/semantic_subject.py's `_extract_activity_from_reported_
finding` (the structural fallback used when no entity/document/date-anchored
resolver succeeds) treated a bare "email"/"notification"/"dispatch"/
"message"/"notice"/"alert" keyword occurring ANYWHERE in the finding text as
grounds to invent a specific activity noun phrase ("email notification",
"notification dispatch", "notification delivery") -- with no requirement
that the finding actually name such an object. app/agent/nodes/
five_why_fallback.py additionally duplicated the same keyword-triggered
guess independently in three places, and a fourth hardcoded, domain-
specific Five-Why boundary answer keyed off the same keyword set.

This is exactly the finding-vocabulary-triggered entity fabrication the
system must avoid generically (see the "domain generality" and "prefer
NOT ESTABLISHED over fabricated entity" hardening requirements) -- a vague
finding that merely mentions "notification" should never resolve to a
confident-sounding "notification delivery" entity it never actually named.

Unlike this branch, the adjacent "payment"/"shipment" keyword fallbacks in
the same function are intentionally left alone: they are covered by an
existing mutation-invariance test suite (test_semantic_fidelity_mutations.py)
that depends on them as the least-bad available fallback when the
alternative is an ACTOR noun (e.g. "finance clerk") winning instead --
removing them regressed that suite, so they are out of scope for this fix.
"""

from __future__ import annotations

from app.agent.nodes.five_why_fallback import build_deterministic_five_why
from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.semantic_subject import (
    _extract_activity_from_reported_finding,
    reject_subject_if_clause,
    resolve_deviation,
)


def test_vague_notification_keyword_does_not_fabricate_entity():
    finding = (
        "It was noted during the audit that something related to "
        "notification seemed unclear over the review period."
    )
    activity = _extract_activity_from_reported_finding(finding)
    assert activity not in ("email notification", "notification dispatch", "notification delivery")


def test_vague_dispatch_alert_keywords_do_not_fabricate_entity():
    finding = "An alert was raised but the underlying dispatch process could not be confirmed as the cause."
    activity = _extract_activity_from_reported_finding(finding)
    assert activity not in ("email notification", "notification dispatch", "notification delivery")


def test_five_why_fallback_does_not_fabricate_notification_subject():
    # No real entity, no equipment/document identifier -- only the bare
    # keyword "message" appears, which must not become the causal subject.
    finding = "A message was expected but its status could not be confirmed by the reviewer."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="review")]
    result = build_deterministic_five_why(finding, ledger)
    combined = " ".join((s.question or "") + " " + (s.answer or "") for s in result.steps).lower()
    assert "email notification" not in combined
    assert "notification dispatch" not in combined


def test_bare_quantifier_plus_generic_noun_never_becomes_subject():
    """A quantifier attached to nothing but a generic occurrence noun
    ("each event", "several failures", "multiple incidents") is a FACT
    about count, not a specific entity -- it must never resolve as the
    affected object/subject."""
    for phrase in ("Each event", "several failures", "multiple incidents", "every occurrence"):
        assert reject_subject_if_clause(phrase) is True


def test_quantifier_noun_finding_does_not_fabricate_subject():
    finding = "Each event was reviewed but the reported amount could not be confirmed."
    r = resolve_deviation(finding, [])
    assert r.subject != "Each event"
    assert r.finding_subject != "Each event"


def test_grounded_specific_subject_still_accepted():
    # Contrast case: a real noun phrase naming an actual entity must
    # still pass -- this fix only rejects bare quantifier+generic-noun
    # combinations, not legitimate specific subjects.
    assert reject_subject_if_clause("the calibration certificate") is False


def test_frequency_modifier_plus_generic_noun_never_becomes_subject():
    """A recurrence/frequency descriptor attached to nothing but a generic
    occurrence noun ("recurring failures", "repeated incidents") names a
    CHARACTERISTIC of the finding, not a specific entity -- it must never
    resolve as the affected object/subject."""
    for phrase in ("Recurring failures", "repeated incidents", "frequent errors", "isolated defects"):
        assert reject_subject_if_clause(phrase) is True


def test_recurring_finding_does_not_fabricate_subject():
    finding = "Recurring failures were noted but the specific cause could not be determined."
    r = resolve_deviation(finding, [])
    assert r.subject != "Recurring failures"
    assert r.finding_subject != "Recurring failures"


def test_subject_ending_in_bare_modal_verb_is_rejected():
    """A candidate ending in a bare modal/auxiliary verb ("...specific
    cause could", "...the record would") is always a truncated clause
    fragment, never a complete noun phrase -- structural, not tied to any
    specific finding's wording."""
    for phrase in ("specific cause could", "the record would", "root cause should"):
        assert reject_subject_if_clause(phrase) is True


def test_recurring_finding_five_why_fallback_produces_grammatical_question():
    """Regression guard for the modal-truncation defect surfaced by fixing
    the 'recurring failures' fabrication: once that candidate is rejected,
    the NEXT fallback candidate must not itself be a truncated clause
    fragment ending mid-sentence."""
    finding = "Recurring failures were noted but the specific cause could not be determined."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="review")]
    result = build_deterministic_five_why(finding, ledger)
    combined = " ".join(s.question for s in result.steps).lower()
    assert "specific cause could" not in combined
    assert " could deviate" not in combined
    assert " could not deviate" not in combined


def test_financial_metric_phrase_never_becomes_subject():
    """A phrase composed entirely of financial-metric modifiers and
    meta-nouns ("historical average cost", "reported amount", "annualized
    exposure") names a MEASUREMENT, not a specific affected object."""
    for phrase in ("historical average cost", "reported amount", "annualized exposure", "verified exposure", "projected loss"):
        assert reject_subject_if_clause(phrase) is True


def test_bare_temporal_reference_never_becomes_subject():
    """A bare temporal-unit noun preceded only by a temporal modifier
    ("past year", "last quarter", "this month") is a date/period
    reference, not an entity."""
    for phrase in ("past year", "last quarter", "this month", "next week"):
        assert reject_subject_if_clause(phrase) is True


def test_financial_metric_finding_does_not_fabricate_subject():
    finding = "The historical average cost was unclear and could not be confirmed."
    r = resolve_deviation(finding, [])
    assert r.subject != "historical average cost"


def test_entity_containing_financial_word_still_accepted():
    """Contrast case: a real entity that happens to contain a financial
    word alongside a concrete identifier/noun must still be accepted --
    rule 7 only rejects a candidate composed ENTIRELY of financial
    modifiers/meta-nouns."""
    assert reject_subject_if_clause("cost center") is False
    assert reject_subject_if_clause("vendor payment") is False


def test_additional_financial_metric_sentences_never_fabricate_subject():
    """Adversarial sentences from the financial/semantic contamination
    hardening pass: financial metrics stated as the grammatical subject
    of a sentence must not become the resolved affected-object subject."""
    for finding in (
        "The proposed remediation cost is INR 60,000.",
        "Historical losses were USD 120,000 per year.",
        "The annual recurrence loss was INR 100,000.",
        "Recovery of USD 20,000 was confirmed.",
        "Implementation cost was GBP 30,000.",
        "ROI is projected at 40%.",
    ):
        r = resolve_deviation(finding, [])
        assert r.subject is None, finding


def test_bare_financial_acronym_never_becomes_subject():
    assert reject_subject_if_clause("ROI") is True
    assert reject_subject_if_clause("payback") is True


def test_grounded_notification_entity_still_extracted_when_actually_named():
    # Contrast case: when the finding genuinely names a notification
    # artifact (with an identifier), that IS legitimate grounded
    # extraction and must still work -- this fix only removes the
    # ungrounded bare-keyword guess, not real extraction.
    finding = "System dispatch logs show email notification NOTIF-901 was transmitted but never received."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="system log")]
    result = build_deterministic_five_why(finding, ledger)
    assert result.steps
