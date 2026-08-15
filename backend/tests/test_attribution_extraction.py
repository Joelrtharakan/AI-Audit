"""Tests for app/services/attribution_extraction.py — the deterministic,
LLM-free recovery of the VERIFIED-fact vs REPORTED-statement distinction
when the extraction LLM call is unavailable.

Domain-diverse on purpose: attribution/awareness-gap structures generalize
across arbitrary QMS findings, not just the "checklist"/"operator" example
from the bug report.
"""

from __future__ import annotations

import pytest

from app.services.attribution_extraction import split_facts_and_attributed_statements

DOMAINS = [
    # (finding, expected_attributed_substring)
    (
        "The equipment maintenance log was incomplete. The technician stated that they were unaware the maintenance schedule had changed.",
        "unaware",
    ),
    (
        "The calibration record was missing an entry. The analyst reported that the calibration due-date reminder was never received.",
        "reminder",
    ),
    (
        "The training file lacked a signed acknowledgement. The supervisor confirmed that the revised training module had not been assigned.",
        "assigned",
    ),
    (
        "The batch record contained a blank field. The operator indicated that the electronic system was inaccessible during the shift.",
        "inaccessible",
    ),
    (
        "The supplier qualification review was overdue. The quality manager noted that the requalification schedule had not been communicated.",
        "communicated",
    ),
    (
        "The environmental monitoring result was not trended. The reviewer explained that they did not know the trending requirement applied.",
        "trending",
    ),
]


@pytest.mark.parametrize("finding,expected_substring", DOMAINS)
def test_attribution_extracted_across_domains(finding, expected_substring):
    facts, attributed = split_facts_and_attributed_statements(finding)
    assert len(attributed) == 1, f"expected one attributed statement in: {finding!r}"
    assert expected_substring in attributed[0]["claim"].lower()
    assert len(facts) == 1


def test_no_attribution_present_all_facts():
    finding = "The record was incomplete. The entry was missing a signature."
    facts, attributed = split_facts_and_attributed_statements(finding)
    assert attributed == []
    assert len(facts) == 2


def test_speaker_captured():
    finding = "The record was incomplete. The night-shift reviewer confirmed that the checklist had not been distributed."
    facts, attributed = split_facts_and_attributed_statements(finding)
    assert attributed[0]["speaker"].lower().startswith("the night-shift reviewer")


def test_empty_finding_no_crash():
    facts, attributed = split_facts_and_attributed_statements("")
    assert facts == []
    assert attributed == []
