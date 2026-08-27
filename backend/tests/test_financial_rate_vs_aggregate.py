"""Regression coverage for the "rate treated as aggregate amount" defect
class: quantity x rate (hours x hourly cost, days x daily cost, units x
per-unit cost, etc.) must be recognized as a RATE and multiplied out
deterministically, never left standing as if the rate itself were the
total exposure.

Root cause traced to app/financial/extractor.py:
  - _PER_EVENT_RE only recognized a small fixed set of per-X phrasings
    ("per event", "per delivery", "per batch", "per unit", "per
    incident") and no "/unit" shorthand -- "INR 12,000 per hour" or
    "INR 12,000/hour" fell through as a flat, unqualified amount.
  - _EVENT_COUNT_WORD_RE's noun list did not include duration nouns
    (hour/day/week/month), so a stated quantity like "10 hours" was
    never recognized as a linkable count at all.
  - Fixing the noun list surfaced a second, previously-latent regex
    defect: the count-word alternation's `\\d+` could partially match
    inside a comma-grouped monetary amount (e.g. matching "12" or "000"
    out of "12,000" as if it were itself a standalone count of hours),
    corrupting the extracted quantity. Fixed with lookaround guards.

Uses abstract, domain-varied test data (maintenance downtime, staffing
hours, logistics units, lab batches) -- no finding-specific production
logic was introduced; all fixes are in the shared, generic unit-noun
vocabulary and regex safety.
"""

from __future__ import annotations

import math

from app.financial.engine import analyze_financial_exposure
from app.financial.extractor import extract_financial_observations
from app.financial.models import FinancialEpistemicStatus
from app.models.agent import EvidenceItem, EvidenceStatus


def test_hourly_rate_is_not_treated_as_aggregate_same_clause():
    finding = "Downtime of 10 hours occurred, at a verified cost rate of INR 12,000 per hour."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="maintenance log")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 120000.0


def test_hourly_rate_shorthand_slash_notation():
    finding = "Downtime of 10 hours occurred, at a verified cost rate of INR 12,000/hour."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="maintenance log")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 120000.0


def test_daily_rate_cross_evidence_linking_stays_bounded():
    """VERIFIED duration + REPORTED daily rate, stated in separate
    evidence items, must derive the product but stay bounded to the
    weaker (REPORTED) provenance -- mirrors the quantity x unit-cost
    linking fix, extended to duration-based rates."""
    ledger = [
        EvidenceItem(claim="Staffing coverage was reduced for 5 days, verified against attendance records.", status=EvidenceStatus.VERIFIED, source="attendance"),
        EvidenceItem(claim="The reported cost impact is INR 8,000 per day.", status=EvidenceStatus.REPORTED, source="estimate"),
    ]
    res = analyze_financial_exposure("finding", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None
    assert res.confirmed_impact.reported_financial_exposure == 40000.0
    assert res.status != FinancialEpistemicStatus.VERIFIED_EXPOSURE


def test_per_unit_rate_not_confused_with_aggregate_logistics_domain():
    finding = "12 units were affected, each verified at a replacement cost of INR 2,500 per unit."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="logistics record")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 30000.0


def test_rate_amount_extracted_as_unit_amount_not_flat_amount():
    """Structural check on the extractor itself: a monetary value stated
    as a rate must populate unit_amount, never the flat `amount` field,
    regardless of which unit the rate is expressed against."""
    for text, expected in [
        ("The cost rate is INR 12,000 per hour.", 12000.0),
        ("The cost rate is INR 8,000 per day.", 8000.0),
        ("The replacement cost is INR 2,500 per unit.", 2500.0),
        ("The verified cost is INR 3,000 per batch.", 3000.0),
    ]:
        obs, *_ = extract_financial_observations(text)
        assert len(obs) == 1
        assert obs[0].amount is None
        assert obs[0].unit_amount == expected


def test_quantity_word_regex_does_not_match_inside_grouped_number():
    """Regression guard for the comma-grouped-number regex defect: a
    monetary amount like "INR 12,000 per hour" must never be
    misinterpreted as a standalone count of hours (e.g. "12" or "000")."""
    from app.financial.extractor import _EVENT_COUNT_WORD_RE

    for text in (
        "The reported cost rate is INR 12,000 per hour.",
        "The verified amount is INR 45,000 per day.",
        "A total of INR 100,000 per batch was recorded.",
    ):
        m = _EVENT_COUNT_WORD_RE.search(text)
        assert m is None, f"unexpectedly matched a count inside a monetary amount: {m.group(0) if m else None!r}"

    # Contrast: a genuine standalone quantity must still match.
    m2 = _EVENT_COUNT_WORD_RE.search("10 hours of downtime were recorded.")
    assert m2 is not None
    assert m2.group("word_count") == "10"


def test_no_invalid_numbers_reach_result_for_rate_based_finding():
    finding = "Downtime of 10 hours occurred, at a verified cost rate of INR 12,000 per hour."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="maintenance log")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    d = res.model_dump()

    def _check(obj):
        if isinstance(obj, float):
            assert not math.isnan(obj) and not math.isinf(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _check(v)
        elif isinstance(obj, list):
            for v in obj:
                _check(v)

    _check(d)
