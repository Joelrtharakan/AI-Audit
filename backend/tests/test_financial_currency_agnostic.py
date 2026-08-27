"""Regression coverage for the currency-agnostic financial hardening pass.

Defect reproduced: _STRICT_AMOUNT_PATTERN/_RANGE_AMOUNT_PATTERN in
app/financial/extractor.py only recognized a fixed, enumerated list of
currency symbols/codes (INR, USD, EUR, GBP plus a few symbols). Any
finding stating an amount in any OTHER valid ISO 4217 currency (AED,
JPY, CHF, SGD, ...) was invisible to the extractor entirely -- the
monetary fact was silently dropped, not merely mislabeled. Additionally,
_normalize_currency defaulted every unrecognized token to "INR" (a
second, distinct silent-fallback defect for the rare code paths where an
unrecognized value reached it).

Fix: replaced the enumerated currency-code list with a generic 3-letter
alphabetic capture, validated post-match against a static ISO 4217
alpha-3 reference set (_ISO_4217_CODES) -- pure reference DATA, not
per-currency branching logic. Recognizing a new ISO code requires only a
potential future addition to that data set, never new extraction code.
_normalize_currency now returns None (never "INR") when a token cannot
be reliably resolved, and callers skip creating an observation rather
than fabricating a currency.

A regression surfaced and was fixed during this pass: making the code
pattern generic caused incidental 3-letter word fragments adjacent to a
number (e.g. "8 inc[idents]", "for 5 [days]") to structurally match the
amount pattern, which corrupted the event-count/quantity extraction
paths that check "does this text already contain an amount". Fixed by
validating currency resolution before treating any such structural match
as a genuine amount (see _has_valid_amount and the corrected
_amount_spans construction).

Uses abstract, domain-neutral test fixtures across a broad currency
matrix -- these currencies are examples proving generic handling, not
individually implemented.
"""

from __future__ import annotations

import math

from app.financial.engine import analyze_financial_exposure
from app.financial.extractor import extract_financial_observations
from app.models.agent import EvidenceItem, EvidenceStatus

# A representative sample spanning the full matrix named in the task --
# deliberately includes currencies that were NEVER previously recognized
# by this codebase (AED, JPY, CHF, SGD, ZAR, BRL, KRW, TRY, ...).
_CURRENCY_MATRIX = [
    "USD", "INR", "EUR", "GBP", "JPY", "CNY", "CHF", "CAD", "AUD", "SGD",
    "AED", "SAR", "QAR", "KWD", "BHD", "OMR", "HKD", "NZD", "ZAR", "BRL",
    "MXN", "THB", "MYR", "IDR", "KRW", "VND", "PHP", "TRY", "SEK", "NOK",
    "DKK", "PLN", "CZK", "HUF", "ILS", "EGP",
]


def test_full_currency_matrix_explicit_iso_code_parsing():
    for code in _CURRENCY_MATRIX:
        text = f"A verified exposure of {code} 10,000 was identified."
        obs, *_ = extract_financial_observations(text)
        assert len(obs) == 1, code
        assert obs[0].amount == 10000.0, code
        assert obs[0].currency == code, code


def test_currency_never_silently_falls_back_to_inr_for_new_codes():
    for code in ("AED", "JPY", "CHF", "SGD", "ZAR"):
        text = f"A verified exposure of {code} 5,000 was identified."
        ledger = [EvidenceItem(claim=text, status=EvidenceStatus.VERIFIED, source="finance")]
        res = analyze_financial_exposure(text, evidence_ledger=ledger)
        assert res.confirmed_impact.currency == code
        assert res.confirmed_impact.currency != "INR"


def test_same_currency_addition_generic_across_matrix():
    for code in ("AED", "JPY", "SGD"):
        text = f"A verified exposure of {code} 10,000 was identified, and {code} 3,000 was recovered."
        ledger = [EvidenceItem(claim=text, status=EvidenceStatus.VERIFIED, source="finance")]
        res = analyze_financial_exposure(text, evidence_ledger=ledger)
        assert res.confirmed_impact.currency == code
        assert res.confirmed_impact.verified_gross_exposure == 10000.0
        assert res.confirmed_impact.verified_recovery == 3000.0
        assert res.confirmed_impact.confirmed_net_loss == 7000.0


def test_same_currency_quantity_times_rate_generic():
    text = "Three verified batches were affected, each verified at AED 2,000 per batch."
    ledger = [EvidenceItem(claim=text, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(text, evidence_ledger=ledger)
    assert res.confirmed_impact.currency == "AED"
    assert res.confirmed_impact.verified_gross_exposure == 6000.0


def test_historical_annualization_generic_for_new_currency():
    text = "Historical records show verified losses of JPY 1,200,000 per year."
    ledger = [EvidenceItem(claim=text, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(text, evidence_ledger=ledger)
    assert res.annualized_exposure.is_assessable is True
    assert res.annualized_exposure.currency == "JPY"
    assert res.annualized_exposure.annualized_amount == 1200000.0


def test_cross_currency_rejection_for_previously_unsupported_pair():
    ledger = [
        EvidenceItem(claim="A verified exposure of AED 40,000 was identified.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="A separate verified expense of ZAR 20,000 was identified.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    by_curr = {c.currency: c for c in res.currency_breakdown}
    assert set(by_curr) == {"AED", "ZAR"}
    assert res.confirmed_impact.verified_gross_exposure is None


def test_unknown_future_iso_style_code_flows_generically_without_new_code():
    """A syntactically valid ISO 4217 code this codebase's test suite has
    never referenced (BOB, currently unused elsewhere in these tests)
    must still be handled purely via the reference-data lookup -- proving
    the mechanism is generic, not a hardcoded per-currency whitelist."""
    text = "A verified exposure of BOB 8,000 was identified."
    obs, *_ = extract_financial_observations(text)
    assert len(obs) == 1
    assert obs[0].currency == "BOB"


def test_incidental_three_letter_word_never_becomes_a_currency():
    """A 3-letter word structurally adjacent to a number in ordinary
    prose (not a real currency statement) must never fabricate an
    observation."""
    for text in (
        "The 500 employees were surveyed.",
        "Approximately 200 the units were tested.",
    ):
        obs, *_ = extract_financial_observations(text)
        assert obs == [], text


def test_event_count_extraction_not_corrupted_by_generic_currency_pattern():
    """Regression guard for the defect surfaced while implementing this
    pass: a generic 3-letter currency pattern must not swallow the first
    3 letters of an unrelated word (e.g. '8 inc[idents]') and thereby
    block legitimate event-count extraction."""
    text = "Historically, 8 incidents occurred over the past year, each verified at INR 15,000."
    ledger = [EvidenceItem(claim=text, status=EvidenceStatus.VERIFIED, source="log")]
    res = analyze_financial_exposure(text, evidence_ledger=ledger)
    assert res.annualized_exposure.annualized_amount == 120000.0
    assert res.annualized_exposure.observed_event_rate_per_year == 8.0


def test_duration_rate_linking_not_corrupted_by_generic_currency_pattern():
    """Second regression guard: a bare quantity statement ('for 5 days')
    must not be treated as if it already contained an amount merely
    because a trailing 3-letter word fragment structurally resembles a
    currency code."""
    ledger = [
        EvidenceItem(claim="Staffing coverage was reduced for 5 days, verified against attendance records.", status=EvidenceStatus.VERIFIED, source="attendance"),
        EvidenceItem(claim="The reported cost impact is INR 8,000 per day.", status=EvidenceStatus.REPORTED, source="estimate"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.reported_financial_exposure == 40000.0


def test_no_invalid_numbers_across_currency_matrix():
    for code in ("JPY", "KWD", "BHD", "AED", "HUF"):
        text = f"A verified exposure of {code} 10,000 was identified."
        ledger = [EvidenceItem(claim=text, status=EvidenceStatus.VERIFIED, source="finance")]
        res = analyze_financial_exposure(text, evidence_ledger=ledger)
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
