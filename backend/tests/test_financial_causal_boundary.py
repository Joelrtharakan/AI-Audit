"""Regression coverage for the financial/causal boundary hardening pass:
remediation cost misclassified as loss, historical annual-rate phrasing
not recognized, and additional financial-metric phrases fabricating an
affected object.

Bug 1 -- "X will cost Y" remediation phrasing not recognized: "Proposed
remediation will cost INR 60,000" did not match the remediation-cost
classifier (which only recognized the rigid noun phrase "remediation
cost"), so the amount defaulted to DIRECT_LOSS and populated the
CURRENT finding's Verified Gross Exposure -- exactly the forbidden
"remediation cost treated as loss" case. Broadened the classifier to
recognize "remediation/corrective action/CAPA ... will cost / is
estimated to cost" phrasing and "cost of/to implement/remediate/fix",
while still not matching unrelated cost phrases ("cost of goods sold",
"scrap cost").

Bug 2 -- "X per year" not recognized as an already-annual figure: a
statement like "recurring losses of INR 120,000 per year" has no
"over/during a period" phrase, so no observation period was established
and Historical Annualized Exposure stayed NOT ASSESSABLE despite the
amount already being explicitly annual. Added recognition of "per
year"/"/year" as directly establishing a 12-month period.

Bug 3 -- additional financial-metric phrases not covered by the
affected-object firewall: "financial exposure", "implementation cost",
"remediation cost", "historical loss", "annual loss" all passed the
subject-acceptance gate despite being financial metrics, not entities
(the modifier words "financial"/"remediation"/"implementation" were
missing from the recognized modifier set).

Together these let the worked example from the task spec (historical
annualized exposure + remediation cost -> payback calculation) function
correctly end to end.

Uses abstract, domain-neutral test fixtures.
"""

from __future__ import annotations

from app.financial.engine import analyze_financial_exposure
from app.financial.extractor import _REMEDIATION_RE
from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.semantic_subject import reject_subject_if_clause


def test_remediation_will_cost_phrasing_recognized():
    for text in (
        "Proposed remediation will cost INR 60,000.",
        "The corrective action is estimated to cost INR 45,000.",
        "The cost to implement the fix was verified at INR 30,000.",
    ):
        assert _REMEDIATION_RE.search(text) is not None, text


def test_unrelated_cost_phrasing_not_misclassified_as_remediation():
    for text in (
        "The cost of goods sold was INR 5,000.",
        "A scrap cost of INR 5,000 was incurred.",
    ):
        assert _REMEDIATION_RE.search(text) is None, text


def test_remediation_cost_never_populates_current_gross_exposure():
    finding = "Proposed remediation will cost INR 60,000."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="CAPA proposal")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None


def test_per_year_phrasing_recognized_as_annual_period():
    finding = "Historical records show recurring losses of INR 120,000 per year."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="finance report")]
    res = analyze_financial_exposure(finding, evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None
    assert res.annualized_exposure.is_assessable is True
    assert res.annualized_exposure.annualized_amount == 120000.0


def test_historical_exposure_and_remediation_cost_produce_correct_payback():
    """The task's own worked example: historical annualized exposure of
    INR 120,000/year and remediation cost of INR 60,000 must produce a
    0.5-year (6-month) indicative payback, with current exposure staying
    NOT ESTABLISHED and no double-counting."""
    ledger = [
        EvidenceItem(claim="Historical records show recurring losses of INR 120,000 per year.", status=EvidenceStatus.VERIFIED, source="finance report"),
        EvidenceItem(claim="Proposed remediation will cost INR 60,000.", status=EvidenceStatus.VERIFIED, source="CAPA proposal"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None
    assert res.annualized_exposure.annualized_amount == 120000.0
    assert res.capa_economics.is_assessable is True
    assert res.capa_economics.remediation_cost == 60000.0
    assert res.capa_economics.annual_avoided_exposure == 120000.0
    assert res.capa_economics.indicative_payback_years == 0.5
    # The economic comparison is explicitly qualified as indicative/
    # conditional, never presented as a confirmed guaranteed outcome.
    assert "not guaranteed" in res.capa_economics.qualification.lower()


def test_additional_financial_metric_phrases_never_become_affected_object():
    for phrase in ("financial exposure", "implementation cost", "remediation cost", "historical loss", "annual loss"):
        assert reject_subject_if_clause(phrase) is True, phrase


def test_real_entities_containing_financial_words_still_accepted():
    assert reject_subject_if_clause("cost center") is False
    assert reject_subject_if_clause("the calibration certificate") is False
