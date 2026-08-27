"""Regression coverage for the semantic-parsing-integrity hardening pass:
decimal-number corruption in subject extraction, and the resulting class
of financial-metric-as-subject fabrications this bug enabled.

Root cause traced precisely: the "Active Transitive SVO Pattern" resolver
in resolve_deviation (the m_svo regex) captured its object noun phrase
with a character class ([a-zA-Z0-9\\s/-]) that excluded the period
character, terminating on `(?:\\.|$)`. Because the object quantifier is
lazy, the regex found the SHORTEST match satisfying that terminator --
and a decimal point inside a number (e.g. the "." in "0.5") satisfied
`\\.` just as validly as a real sentence-ending period, so "Payback is
estimated at 0.5 years." produced obj="estimated at 0", silently
dropping the "5 years" and worse, treating a numeric fragment as if it
were an entity.

Fixed by (1) allowing "." inside the object character class so a
decimal number can be captured whole, and changing the terminator to
`\\.(?!\\d)` so only a period NOT followed by a digit ends the phrase --
a genuine sentence-ending period is unaffected, since it is never
followed by a digit continuing the same number; and (2) adding two new,
narrowly-scoped rejection rules to the shared reject_subject_if_clause
gate: a bare number carrying "%" or immediately followed by a
measurement/duration unit word (0.5 years, 40%, 2 million) is a
measured VALUE, and a candidate that IS a bare number and nothing else
(2.5) has no entity context at all -- both rejected, while a legitimate
bare-number identifier paired with a location/container noun in the
same candidate ("Room 102", "Line 3", "Batch B205") is deliberately
left untouched (regression-guarded below).

Uses abstract, domain-neutral test fixtures.
"""

from __future__ import annotations

from app.services.semantic_subject import reject_subject_if_clause, resolve_deviation


def test_decimal_number_no_longer_truncated_in_svo_object():
    """The exact reproduction: '0.5' must never become '0'."""
    r = resolve_deviation("Payback is estimated at 0.5 years.", [])
    assert r.subject != "estimated at 0"
    if r.subject:
        assert "0.5" in r.subject or r.subject is None


def test_financial_metric_sentences_never_fabricate_subject():
    """All six adversarial sentences from the financial + semantic
    contamination hardening pass."""
    for finding in (
        "Payback is estimated at 0.5 years.",
        "ROI is projected at 40%.",
        "Annual avoided exposure is USD 120,000.",
        "Historical annualized exposure is INR 100,000.",
        "Benefit-cost ratio is 2.5.",
        "Expected recurrence loss is USD 50,000/year.",
    ):
        r = resolve_deviation(finding, [])
        assert r.subject is None, finding


def test_percent_value_never_becomes_subject():
    assert reject_subject_if_clause("40%") is True
    assert reject_subject_if_clause("projected at 40%") is True


def test_decimal_with_duration_unit_never_becomes_subject():
    assert reject_subject_if_clause("0.5 years") is True
    assert reject_subject_if_clause("2 million") is True


def test_bare_number_alone_never_becomes_subject():
    assert reject_subject_if_clause("2.5") is True


def test_location_identifiers_with_bare_numbers_still_accepted():
    """Regression guard for the over-broad first version of this fix:
    'Room 102'/'Line 3'/'Batch 205' style identifiers -- a number paired
    with a preceding location/container noun and NO following
    measurement unit -- must remain valid, unaffected by the new
    numeric-value rejection rules."""
    for phrase in ("Room 102", "Line 3", "Batch 205", "cleaning checklist for Room 102"):
        assert reject_subject_if_clause(phrase) is False, phrase


def test_cross_domain_entity_resolution_unaffected_by_numeric_fix():
    """Regression guard: an unrelated finding containing a bare location
    number ('assembly line 3') must still resolve its real entity
    subject, not fail to match at all."""
    r = resolve_deviation(
        "Chassis robotic weld controller logged weld cycle execution on assembly line 3, "
        "but the welding inspector reported ultrasonic non-destructive testing record was missing.",
        [],
    )
    assert r.matched is True
    assert r.subject is not None
