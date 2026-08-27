"""Regression coverage for the current-finding vs historical financial
population separation defect.

Bug: a financial fact drawn from evidence explicitly framed as
backward-looking historical context ("historically", "similar incidents
occurred N times over the past year", ...) was being aggregated into the
CURRENT finding's verified gross exposure alongside (or instead of) any
actual current-finding financial fact -- a historical per-incident cost
could surface as "Verified Gross Exposure" for a finding that itself
establishes no financial amount at all.

Root cause traced to app/financial/extractor.py and calculator.py: no
concept of "which population does this fact belong to" existed --
observations from ANY evidence claim, however framed, were pooled
together before current-exposure aggregation.

Fix: a domain-neutral historical-framing marker (_HISTORICAL_MARKER_RE)
tags observations extracted from backward-looking statements as
financial_population="HISTORICAL"; calculate_confirmed_impact and
calculate_potential_exposure now only aggregate CURRENT_FINDING-tagged
observations. Historical observations remain fully available to
calculate_recurrence_exposure / calculate_annualized_exposure, which is
their intended use.

Uses abstract, domain-varied test fixtures -- no finding-specific
production logic was introduced.
"""

from __future__ import annotations

from app.financial.engine import analyze_financial_exposure
from app.financial.models import FinancialEpistemicStatus
from app.models.agent import EvidenceItem, EvidenceStatus


def test_historical_cost_never_becomes_current_gross_exposure():
    finding = (
        "A nonconformity was observed. Historically, similar incidents occurred 8 times "
        "over the past year, each verified at INR 15,000."
    )
    ledger = [
        EvidenceItem(claim="A nonconformity was observed.", status=EvidenceStatus.VERIFIED, source="audit"),
        EvidenceItem(
            claim="Historically, similar incidents occurred 8 times over the past year, each verified at INR 15,000.",
            status=EvidenceStatus.VERIFIED,
            source="CAPA database",
        ),
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None
    assert res.confirmed_impact.is_confirmed_event is False


def test_historical_data_still_produces_correct_annualized_exposure():
    """The same historical facts excluded from current exposure must still
    feed the historical annualization calculation correctly."""
    finding = (
        "A nonconformity was observed. Historically, similar incidents occurred 8 times "
        "over the past year, each verified at INR 15,000."
    )
    ledger = [
        EvidenceItem(claim="A nonconformity was observed.", status=EvidenceStatus.VERIFIED, source="audit"),
        EvidenceItem(
            claim="Historically, similar incidents occurred 8 times over the past year, each verified at INR 15,000.",
            status=EvidenceStatus.VERIFIED,
            source="CAPA database",
        ),
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.annualized_exposure.is_assessable is True
    assert res.annualized_exposure.annualized_amount == 120000.0
    assert res.annualized_exposure.observed_event_rate_per_year == 8.0
    assert res.status == FinancialEpistemicStatus.ANNUALIZED_EXPOSURE


def test_current_finding_fact_alongside_historical_fact_stays_separated():
    """When the CURRENT finding itself also has a verified financial fact,
    that fact -- and only that fact -- populates current gross exposure;
    the historical fact contributes only to annualization."""
    finding = (
        "A verified rework cost of INR 22,000 was incurred for the current nonconformity. "
        "Historically, similar incidents occurred 4 times over the past year, each verified at INR 15,000."
    )
    ledger = [
        EvidenceItem(claim="A verified rework cost of INR 22,000 was incurred for the current nonconformity.", status=EvidenceStatus.VERIFIED, source="QA log"),
        EvidenceItem(
            claim="Historically, similar incidents occurred 4 times over the past year, each verified at INR 15,000.",
            status=EvidenceStatus.VERIFIED,
            source="CAPA database",
        ),
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 22000.0
    assert res.annualized_exposure.is_assessable is True
    assert res.annualized_exposure.annualized_amount == 60000.0  # 4 x 15,000


def test_historical_fact_without_current_loss_produces_not_established_status():
    finding = "A control gap was noted. In the past, similar gaps led to losses of INR 9,000 per occurrence, several times over the past year."
    ledger = [
        EvidenceItem(claim="A control gap was noted.", status=EvidenceStatus.VERIFIED, source="audit"),
        EvidenceItem(
            claim="In the past, similar gaps led to losses of INR 9,000 per occurrence, several times over the past year.",
            status=EvidenceStatus.VERIFIED,
            source="incident log",
        ),
    ]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None
    assert res.confirmed_impact.is_confirmed_loss is False


def test_observation_period_recognizes_bare_past_year_phrasing():
    from app.financial.extractor import _extract_observation_period_months

    assert _extract_observation_period_months("over the past year") == 12.0
    assert _extract_observation_period_months("during the last month") == 1.0
    assert _extract_observation_period_months("over a period of 4 months") == 4.0
    assert _extract_observation_period_months("during the last 2 years") == 24.0


def test_frequency_word_times_recognized_as_event_count():
    from app.financial.extractor import _EVENT_COUNT_WORD_RE

    m = _EVENT_COUNT_WORD_RE.search("similar incidents occurred 8 times over the past year")
    assert m is not None
    assert m.group("word_count") == "8"
