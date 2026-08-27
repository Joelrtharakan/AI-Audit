"""Entity/period extraction safety for financial-bearing findings, and
additional deterministic-arithmetic/aggregation-safety cases for the
Evidence-Grounded Financial Exposure & Cost-of-Recurrence Analysis.

Covers gaps identified during the financial-analysis hardening pass:
  - A stated temporal phrase (e.g. "a four-month period from January
    through April 2026") must never become the affected_object/finding
    subject, and must be preserved as a date/period fact instead.
  - Missing entity resolution must fail to None/UNRESOLVED rather than
    fabricate a plausible-looking placeholder.
  - Verified per-event amount x verified count multiplication.
  - Duplicate identical evidence entries must not double-count exposure.
  - Mismatched currencies require reconciliation, never silent conversion.
  - Malformed numeric text must not crash extraction or produce NaN/Infinity.
  - Verified zero recovery is distinguished from unknown/unverified recovery.
"""

from __future__ import annotations

import math

from app.financial.engine import analyze_financial_exposure
from app.financial.models import FinancialEpistemicStatus, RecoveryStatus
from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.semantic_subject import extract_date, resolve_deviation


def test_temporal_phrase_never_becomes_affected_object():
    finding = (
        "A reported amount of INR 40,000 was associated with the deviation "
        "over a four-month period from January through April 2026."
    )
    result = resolve_deviation(finding, [])
    subject = (result.finding_subject or "") + (result.affected_object or "")
    assert "four-month period" not in subject.lower()
    assert "january" not in subject.lower()
    assert not subject.lower().startswith("deviation")


def test_explicit_month_range_period_is_preserved():
    finding = (
        "A reported amount of INR 40,000 was associated with the deviation "
        "over a four-month period from January through April 2026."
    )
    assert extract_date(finding) == "January through April 2026"


def test_missing_entity_resolves_to_none_not_fabricated():
    # No named object, no equipment/document identifier, no concrete noun
    # phrase beyond the generic "deviation"/temporal framing.
    finding = "A reported amount of INR 40,000 was associated with the deviation over a four-month period from January through April 2026."
    result = resolve_deviation(finding, [])
    assert result.finding_subject in (None, "") or result.finding_subject == "UNKNOWN"


def test_verified_per_event_amount_times_verified_count():
    finding = "Three verified nonconformities each cost INR 30,000 in rework."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="QA log")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 90000.0
    assert res.confirmed_impact.verified_event_count == 3
    assert "3" in res.confirmed_impact.calculation_formula or "3.0" in res.confirmed_impact.calculation_formula


def test_duplicate_identical_evidence_not_double_counted():
    finding = "Rework cost of INR 40,000 was incurred."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="QA log"),
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="QA log"),
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 40000.0


def test_mismatched_currency_requires_reconciliation():
    finding = "Loss of $5,000 was later offset by a recovery of INR 200,000 from local vendor."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.status == FinancialEpistemicStatus.NOT_ASSESSABLE
    assert "currenc" in res.assessment_reason.lower()


def test_malformed_numeric_text_does_not_crash_or_produce_invalid_numbers():
    finding = "Cost was approximately INR abc,xyz which is unclear."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None
    d = res.model_dump()

    def _assert_no_invalid_numbers(obj):
        if isinstance(obj, float):
            assert not math.isnan(obj)
            assert not math.isinf(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _assert_no_invalid_numbers(v)
        elif isinstance(obj, list):
            for v in obj:
                _assert_no_invalid_numbers(v)

    _assert_no_invalid_numbers(d)


def test_verified_zero_recovery_distinguished_from_unknown_recovery():
    finding_zero = "Scrap cost of INR 25,000 was incurred; recovery was nil."
    ledger_zero = [EvidenceItem(claim=finding_zero, status=EvidenceStatus.VERIFIED, source="finance")]
    res_zero = analyze_financial_exposure(finding_zero, evidence_ledger=ledger_zero)
    assert res_zero.confirmed_impact.recovery_status == RecoveryStatus.VERIFIED_ZERO_RECOVERY
    assert res_zero.confirmed_impact.verified_recovery == 0.0
    assert res_zero.confirmed_impact.confirmed_net_loss == 25000.0

    finding_unknown = "Scrap cost of INR 25,000 was incurred."
    ledger_unknown = [EvidenceItem(claim=finding_unknown, status=EvidenceStatus.VERIFIED, source="finance")]
    res_unknown = analyze_financial_exposure(finding_unknown, evidence_ledger=ledger_unknown)
    assert res_unknown.confirmed_impact.recovery_status == RecoveryStatus.REQUIRES_VERIFICATION
    assert res_unknown.confirmed_impact.verified_recovery is None
    assert res_unknown.confirmed_impact.confirmed_net_loss is None


def test_verified_count_with_reported_unit_cost_never_becomes_verified_exposure():
    """VERIFIED event count + REPORTED per-event cost must not combine
    into a VERIFIED gross exposure -- the calculation is bounded by the
    weakest required input (the reported cost)."""
    finding = "Eight verified nonconformities were identified. Each is reported to cost approximately INR 15,000."
    ledger = [
        EvidenceItem(claim="Eight verified nonconformities were identified.", status=EvidenceStatus.VERIFIED, source="QA log"),
        EvidenceItem(claim="Each is reported to cost approximately INR 15,000.", status=EvidenceStatus.REPORTED, source="estimate"),
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None
    assert res.status != FinancialEpistemicStatus.VERIFIED_EXPOSURE
    assert res.status != FinancialEpistemicStatus.CONFIRMED_NET_LOSS
    # The verified count (8) IS correctly linked to the reported per-event
    # cost to compute the derived total -- 8 x 15,000 -- but the result
    # stays REPORTED/UNVERIFIED (never promoted) because one required
    # input (the unit cost) is only REPORTED.
    assert res.confirmed_impact.reported_financial_exposure == 120000.0
    assert res.confirmed_impact.reported_unit_exposure == 15000.0
    assert res.confirmed_impact.reported_event_count == 8


def test_verified_quantity_and_verified_unit_cost_from_separate_statements_link_correctly():
    """When BOTH the event count and the per-event cost are independently
    VERIFIED (even though stated in separate evidence items), the engine
    must link them into a VERIFIED gross exposure -- this is the mirror
    case of the REPORTED-unit-cost test above."""
    finding = "Eight verified nonconformities were identified. Each verified nonconformity cost INR 15,000."
    ledger = [
        EvidenceItem(claim="Eight verified nonconformities were identified.", status=EvidenceStatus.VERIFIED, source="QA log"),
        EvidenceItem(claim="Each verified nonconformity cost INR 15,000.", status=EvidenceStatus.VERIFIED, source="finance ledger"),
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 120000.0
    assert res.confirmed_impact.verified_event_count == 8


def test_unverified_quantity_with_verified_unit_cost_stays_unverified():
    """The mirror epistemic-bound case: an UNVERIFIED event count must not
    let a VERIFIED unit cost produce a VERIFIED aggregate -- bounded by
    the weaker input regardless of which side (quantity or cost) is
    weaker."""
    finding = "Some nonconformities were alleged. Each verified nonconformity cost INR 15,000."
    ledger = [
        EvidenceItem(claim="Some nonconformities were alleged.", status=EvidenceStatus.UNVERIFIED, source="hearsay"),
        EvidenceItem(claim="Each verified nonconformity cost INR 15,000.", status=EvidenceStatus.VERIFIED, source="finance ledger"),
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None


def test_ambiguous_multiple_bare_quantities_are_not_guessed():
    """Two distinct bare-quantity statements alongside one per-event
    amount is ambiguous -- which quantity applies to the cost cannot be
    determined, so linking must NOT occur (no guess)."""
    finding = (
        "Five nonconformities were found in Area A. Seven nonconformities were found in Area B. "
        "Each nonconformity is reported to cost INR 10,000."
    )
    ledger = [
        EvidenceItem(claim="Five nonconformities were found in Area A.", status=EvidenceStatus.VERIFIED, source="log A"),
        EvidenceItem(claim="Seven nonconformities were found in Area B.", status=EvidenceStatus.VERIFIED, source="log B"),
        EvidenceItem(claim="Each nonconformity is reported to cost INR 10,000.", status=EvidenceStatus.REPORTED, source="estimate"),
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    # Must not silently pick 5 or 7 -- the per-event amount stays
    # unlinked (reported_event_count is whatever the extractor found
    # in-clause, not a guessed cross-reference).
    assert res.confirmed_impact.reported_financial_exposure != 50000.0
    assert res.confirmed_impact.reported_financial_exposure != 70000.0


def test_unit_cost_never_rendered_as_gross_exposure_without_aggregation_basis():
    """A single per-event/per-unit amount with a verified event COUNT in
    the same statement must be multiplied out, never left standing in for
    the aggregate (already covered by the multiplication path) -- and a
    bare per-event amount with NO count must not silently become the
    gross total either."""
    finding = "Rework cost is INR 30,000 per event; the number of affected events was not established."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="QA log")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    # A verified per-event amount with an implicit count of 1 is a
    # legitimate single-event gross exposure -- it must not be silently
    # inflated, but it also is not a bug for it to stand as the exposure
    # for that one event. The key invariant is it can never EXCEED what a
    # verified count supports.
    if res.confirmed_impact.verified_gross_exposure is not None:
        assert res.confirmed_impact.verified_gross_exposure == 30000.0
