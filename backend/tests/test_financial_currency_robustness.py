"""Regression coverage for the currency robustness hardening pass:
native-symbol resolution via an explicit code prefix, code/symbol
conflict detection, and a broader currency-matrix sanity sweep.

Defects reproduced:
  1. "CNY ¥10,000" / "JPY ¥10,000" failed to extract at all -- the
     disclosed gap from the previous pass. The native symbol ¥ is
     genuinely ambiguous (represents multiple currencies) and was
     correctly never aliased on its own, but when an explicit ISO code
     immediately precedes it, that code should resolve the amount.
  2. "USD ₹10,000" silently resolved to INR -- the explicit "USD" code
     token never matched anything (not adjacent to a digit), so only
     the "₹10,000" portion matched, silently discarding the conflicting
     "USD" prefix and picking INR. This is exactly the "misleading
     currency" defect the task named: a stated USD amount must never
     become INR.

Fix: extended _STRICT_AMOUNT_PATTERN with an optional leading
code_prefix group, and added _resolve_currency_with_prefix() which:
  - uses the code_prefix when the symbol/code alone doesn't resolve
    (e.g. ¥ alone is ambiguous, but "JPY ¥10,000" resolves via the
    code),
  - rejects the match entirely (never picks either side) when the
    prefix and the symbol/code resolve to DIFFERENT currencies (a
    genuine conflict, e.g. "USD ₹10,000"),
  - is a no-op when no code_prefix is present, preserving all existing
    bare-symbol behavior ("$10,000" -> USD, "₹10,000" -> INR, etc.)
    unchanged.

Uses abstract, domain-neutral test fixtures.
"""

from __future__ import annotations

import math

from app.financial.engine import analyze_financial_exposure
from app.financial.extractor import extract_financial_observations
from app.models.agent import EvidenceItem, EvidenceStatus


def test_cny_native_symbol_resolves_via_explicit_code():
    obs, *_ = extract_financial_observations("CNY ¥10,000 was verified.")
    assert len(obs) == 1
    assert obs[0].currency == "CNY"
    assert obs[0].amount == 10000.0


def test_jpy_native_symbol_resolves_via_explicit_code():
    obs, *_ = extract_financial_observations("JPY ¥10,000 was verified.")
    assert len(obs) == 1
    assert obs[0].currency == "JPY"


def test_bare_ambiguous_native_symbol_without_context_stays_unresolved():
    obs, *_ = extract_financial_observations("¥10,000 was verified.")
    assert obs == []


def test_conflicting_code_and_symbol_never_silently_resolved():
    """'USD ₹10,000' must never become INR (nor USD) -- the conflict is
    detected and the match is rejected rather than either side winning."""
    obs, *_ = extract_financial_observations("USD ₹10,000 was verified.")
    assert obs == []
    # Specifically must never fabricate INR from a stated USD amount.
    assert all(o.currency != "INR" for o in obs)


def test_existing_bare_symbol_behavior_unchanged():
    """Regression guard: the established, heavily-tested bare-symbol
    convention ($ -> USD, ₹ -> INR) must be completely unaffected by the
    code_prefix addition."""
    for text, expected in [("$10,000 was verified.", "USD"), ("₹10,000 was verified.", "INR"), ("€10,000 was verified.", "EUR")]:
        obs, *_ = extract_financial_observations(text)
        assert len(obs) == 1, text
        assert obs[0].currency == expected, text


def test_explicit_code_plus_matching_symbol_still_resolves():
    obs, *_ = extract_financial_observations("USD $10,000 was verified.")
    assert len(obs) == 1
    assert obs[0].currency == "USD"


def test_current_historical_remediation_three_populations_independent():
    ledger = [
        EvidenceItem(claim="A verified current expense of EUR 20,000 was identified.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="Historical records show verified losses of USD 100,000 per year.", status=EvidenceStatus.VERIFIED, source="C2"),
        EvidenceItem(claim="Proposed remediation will cost GBP 30,000.", status=EvidenceStatus.VERIFIED, source="C3"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    by_curr = {c.currency: c for c in res.currency_breakdown}
    assert by_curr["EUR"].gross_amount == 20000.0
    assert by_curr["USD"].historical_annualized_amount == 100000.0
    # Remediation cost must never be summed with current/historical, and
    # must never populate EUR/USD's own gross figures.
    assert by_curr["EUR"].historical_annualized_amount is None
    assert by_curr["USD"].gross_amount is None


def test_five_plus_simultaneous_currencies_all_independent():
    ledger = [
        EvidenceItem(claim="A verified exposure of USD 10,000 was identified.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="A separate verified exposure of EUR 5,000 was identified.", status=EvidenceStatus.VERIFIED, source="C2"),
        EvidenceItem(claim="A separate verified exposure of JPY 1,000,000 was identified.", status=EvidenceStatus.VERIFIED, source="C3"),
        EvidenceItem(claim="A separate verified exposure of AED 20,000 was identified.", status=EvidenceStatus.VERIFIED, source="C4"),
        EvidenceItem(claim="A separate verified exposure of CHF 3,000 was identified.", status=EvidenceStatus.VERIFIED, source="C5"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    by_curr = {c.currency: c for c in res.currency_breakdown}
    assert set(by_curr) == {"USD", "EUR", "JPY", "AED", "CHF"}
    assert by_curr["USD"].gross_amount == 10000.0
    assert by_curr["EUR"].gross_amount == 5000.0
    assert by_curr["JPY"].gross_amount == 1000000.0
    assert by_curr["AED"].gross_amount == 20000.0
    assert by_curr["CHF"].gross_amount == 3000.0


def test_payback_never_divides_across_currencies():
    """Annual avoided exposure and remediation cost in DIFFERENT
    currencies must never produce a payback figure."""
    ledger = [
        EvidenceItem(claim="Historical records show verified losses of USD 120,000 per year.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="Proposed remediation will cost INR 60,000.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    # capa_economics is computed on the pooled (unsegregated) observation
    # set only when a single currency exists; with a currency mismatch
    # present, the top-level result short-circuits to NOT_ASSESSABLE
    # before any cross-currency division could occur.
    assert res.status.value == "NOT_ASSESSABLE"
    assert res.capa_economics.is_assessable is False


def test_large_currency_amounts_never_become_event_counts():
    for amount_text, expected in [("JPY 1,500,000", 1500000.0), ("KRW 5,000,000", 5000000.0), ("VND 20,000,000", 20000000.0)]:
        code, amt = amount_text.split(" ", 1)
        text = f"A verified exposure of {amount_text} was identified."
        obs, *_ = extract_financial_observations(text)
        assert len(obs) == 1, text
        assert obs[0].event_count is None, text
        assert obs[0].amount == expected, text


def test_range_amounts_still_generic_across_currencies():
    for text, curr in [
        ("A potential exposure of USD 10,000-20,000 was identified.", "USD"),
        ("A potential exposure of EUR 5,000 to 8,000 was identified.", "EUR"),
    ]:
        obs, *_ = extract_financial_observations(text)
        range_obs = [o for o in obs if o.amount_min is not None]
        assert len(range_obs) == 1, text
        assert range_obs[0].currency == curr, text


def test_zero_amount_does_not_create_false_exposure():
    obs, *_ = extract_financial_observations("A cost of USD 0 was verified.")
    assert obs == []


def test_no_invalid_numbers_for_conflict_and_symbol_cases():
    ledger = [
        EvidenceItem(claim="USD ₹10,000 was verified.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="CNY ¥5,000 was verified.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
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
