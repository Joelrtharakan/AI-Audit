"""Regression coverage for the financial temporal-semantics hardening
pass: generic event-frequency derivation, observation-period phrasing
robustness across paraphrases, current/historical population isolation
under mixed evidence, and conflicting historical-frequency detection.

Bugs fixed in this pass:
  1. "N months"/"N years" inside an observation-period phrase could be
     misread as an EVENT COUNT by the same regex used for genuine
     occurrence counts (e.g. "Records covering 12 months identified 8
     events" incorrectly derived 12 x 15,000 instead of 8 x 15,000) --
     fixed by masking the period span (month/year/quarter units only)
     before event-count extraction runs.
  2. Several common paraphrases of "N events over T months" ("during the
     previous T months", "across the preceding year", "a T-month review
     period") were not recognized as HISTORICAL population framing at
     all, silently letting historical costs populate current exposure --
     fixed by reusing the observation-period regex's own retrospective
     modifier (past/last/previous/preceding) as an additional signal
     rather than a second disjoint keyword list.
  3. A cross-source dedup guard in the extractor was scoped globally
     across ALL evidence, so two genuinely different verified claims
     that happened to share the same per-event amount (but disagreed on
     event count) had the second one silently discarded -- this made a
     real frequency conflict invisible. Fixed by scoping the dedup guard
     per-source-statement and adding an explicit historical-frequency
     conflict check, plus a value-based corroboration dedup at the
     calculator layer (two sources stating IDENTICAL numbers still
     collapse to one fact, as before) so conflict detection now works
     without regressing duplicate-evidence handling.

Uses abstract, domain-varied test fixtures.
"""

from __future__ import annotations

from app.financial.engine import analyze_financial_exposure
from app.financial.extractor import _extract_observation_period_months
from app.financial.models import FinancialEpistemicStatus
from app.models.agent import EvidenceItem, EvidenceStatus


def _historical(count: int, months: int, per_event: int = 15000) -> str:
    return f"Historically, {count} incidents occurred over the past {months} months, each verified at INR {per_event:,}."


def test_frequency_derivation_8_events_12_months():
    ledger = [EvidenceItem(claim=_historical(8, 12), status=EvidenceStatus.VERIFIED, source="log")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.annualized_exposure.observed_event_rate_per_year == 8.0
    assert res.annualized_exposure.annualized_amount == 120000.0


def test_frequency_derivation_8_events_6_months():
    ledger = [EvidenceItem(claim=_historical(8, 6), status=EvidenceStatus.VERIFIED, source="log")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.annualized_exposure.observed_event_rate_per_year == 16.0
    assert res.annualized_exposure.annualized_amount == 240000.0


def test_frequency_derivation_8_events_24_months():
    ledger = [EvidenceItem(claim=_historical(8, 24), status=EvidenceStatus.VERIFIED, source="log")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.annualized_exposure.observed_event_rate_per_year == 4.0
    assert res.annualized_exposure.annualized_amount == 60000.0


def test_frequency_derivation_15_events_30_months():
    ledger = [EvidenceItem(claim=_historical(15, 30), status=EvidenceStatus.VERIFIED, source="log")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.annualized_exposure.observed_event_rate_per_year == 6.0
    assert res.annualized_exposure.annualized_amount == 90000.0


def test_period_phrase_never_misread_as_event_count():
    """Root defect: 'Records covering 12 months identified 8 events'
    must derive the exposure from 8 events, not from the 12-month
    period phrase."""
    finding = "Records covering 12 months identified 8 events, each verified at INR 15,000."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="log")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.annualized_exposure.annualized_amount == 120000.0
    assert res.annualized_exposure.observed_event_rate_per_year == 8.0


def test_duration_rate_quantity_unaffected_by_period_masking():
    """The period-phrase masking is scoped to month/year/quarter units
    only -- a genuine day/hour duration-rate quantity ('for 5 days' x
    'INR 8,000/day') must remain fully extractable, including when
    linked across separate evidence items."""
    ledger = [
        EvidenceItem(claim="Staffing coverage was reduced for 5 days, verified against attendance records.", status=EvidenceStatus.VERIFIED, source="attendance"),
        EvidenceItem(claim="The reported cost impact is INR 8,000 per day.", status=EvidenceStatus.REPORTED, source="estimate"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.reported_financial_exposure == 40000.0


def test_paraphrase_equivalence_produces_same_structured_result():
    """Five differently-worded but semantically equivalent statements of
    '8 incidents, INR 15,000 each, over a 12-month period' must all
    produce the same core numeric result (8 events, INR 15,000/event,
    12-month period, INR 120,000/year annualized), even though not all
    five carry an explicit retrospective marker word."""
    variants = [
        "8 incidents during the previous 12 months, each verified at INR 15,000.",
        "8 incidents over the last year, each verified at INR 15,000.",
        "During a twelve-month review period, 8 incidents occurred, each verified at INR 15,000.",
        "8 occurrences were recorded across the preceding year, each verified at INR 15,000.",
        "Records covering 12 months identified 8 events, each verified at INR 15,000.",
    ]
    for finding in variants:
        ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="log")]
        res = analyze_financial_exposure(finding, evidence_ledger=ledger)
        assert res.annualized_exposure.annualized_amount == 120000.0, finding
        assert res.annualized_exposure.observed_event_rate_per_year == 8.0, finding


def test_explicit_retrospective_phrasing_excludes_current_exposure():
    """Variants carrying an explicit retrospective marker (previous,
    last, preceding) must be classified HISTORICAL and never populate
    current gross exposure."""
    variants = [
        "8 incidents during the previous 12 months, each verified at INR 15,000.",
        "8 incidents over the last year, each verified at INR 15,000.",
        "8 occurrences were recorded across the preceding year, each verified at INR 15,000.",
    ]
    for finding in variants:
        ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="log")]
        res = analyze_financial_exposure(finding, evidence_ledger=ledger)
        assert res.confirmed_impact.verified_gross_exposure is None, finding


def test_bare_past_year_and_hyphenated_month_period_parsing():
    assert _extract_observation_period_months("over the past year") == 12.0
    assert _extract_observation_period_months("during a twelve-month review period") == 12.0
    assert _extract_observation_period_months("across the preceding year") == 12.0
    assert _extract_observation_period_months("during the previous 12 months") == 12.0


def test_conflicting_historical_frequency_requires_reconciliation():
    """Two VERIFIED claims stating the same per-event cost and period but
    a DIFFERENT event count describe the recurrence rate inconsistently
    -- must never be silently resolved by picking one."""
    ledger = [
        EvidenceItem(claim="Historically, 6 incidents occurred over the past year, each verified at INR 15,000.", status=EvidenceStatus.VERIFIED, source="Log A"),
        EvidenceItem(claim="Historically, 8 incidents occurred over the past year, each verified at INR 15,000.", status=EvidenceStatus.VERIFIED, source="Log B"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.status == FinancialEpistemicStatus.FINANCIAL_CONFLICT_REQUIRES_RECONCILIATION


def test_corroborating_identical_historical_claims_not_double_counted():
    """Two VERIFIED claims stating the IDENTICAL historical fact (same
    count, cost, and period) from different sources corroborate one
    fact -- they must not be summed into double the frequency/exposure."""
    ledger = [
        EvidenceItem(claim="Historically, 8 incidents occurred over the past year, each verified at INR 15,000.", status=EvidenceStatus.VERIFIED, source="Log A"),
        EvidenceItem(claim="Historically, 8 incidents occurred over the past year, each verified at INR 15,000.", status=EvidenceStatus.VERIFIED, source="Log B"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.status == FinancialEpistemicStatus.ANNUALIZED_EXPOSURE
    assert res.annualized_exposure.observed_event_rate_per_year == 8.0
    assert res.annualized_exposure.annualized_amount == 120000.0


def test_duplicate_identical_current_finding_evidence_still_not_double_counted():
    """Regression guard: the per-source dedup-scoping change (needed to
    let genuinely different historical claims survive extraction) must
    not resurrect double-counting of the SAME current-finding fact
    restated across two evidence items -- the calculator-level
    value-based corroboration dedup must catch this instead."""
    finding = "Rework cost of INR 40,000 was incurred."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="QA log"),
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="QA log"),
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 40000.0


def test_current_and_historical_still_isolated_after_this_pass():
    """Regression guard for the previous pass's core fix: current +
    historical mixed evidence must remain separated after this pass's
    changes."""
    ledger = [
        EvidenceItem(claim="A verified rework cost of INR 22,000 was incurred for the current nonconformity.", status=EvidenceStatus.VERIFIED, source="QA log"),
        EvidenceItem(claim="Historically, similar incidents occurred 4 times over the past year, each verified at INR 15,000.", status=EvidenceStatus.VERIFIED, source="CAPA database"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 22000.0
    assert res.annualized_exposure.annualized_amount == 60000.0
