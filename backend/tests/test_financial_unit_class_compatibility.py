"""Regression coverage for a rate/quantity unit-compatibility defect
found while auditing basis and calculation-eligibility hardening.

Bug: the cross-evidence quantity-linking mechanism (added to correctly
derive "quantity x unit cost" when the two facts are stated in separate
evidence items) linked ANY single unambiguous bare quantity to ANY
single unambiguous unit-only rate, regardless of whether their units
were actually compatible. "Ten verified nonconformities were identified"
(an EVENT count) linked to "The downtime cost rate is INR 12,000 per
hour" (an HOURLY rate), producing a fabricated 10 x 12,000 = 120,000
gross exposure -- multiplying an event count by an hourly rate is not a
meaningful calculation.

Root cause: the linking logic checked population and status compatibility
but never checked that the rate's stated unit (hour/day/event/batch/...)
matched the quantity's stated unit.

Fix: capture the specific unit noun on both sides (rate_unit_class on
the FinancialObservation, quantity_unit_class on the bare-quantity
tuple), normalized to a broad compatibility class (OCCURRENCE / TIME /
ITEM), and require them to match (or either side be an unspecified
wildcard, e.g. bare "each") before linking.

Uses abstract, domain-neutral test fixtures.
"""

from __future__ import annotations

from app.financial.engine import analyze_financial_exposure
from app.models.agent import EvidenceItem, EvidenceStatus


def test_event_count_not_multiplied_by_incompatible_hourly_rate():
    ledger = [
        EvidenceItem(claim="Ten verified nonconformities were identified.", status=EvidenceStatus.VERIFIED, source="QA log"),
        EvidenceItem(claim="The downtime cost rate is INR 12,000 per hour.", status=EvidenceStatus.VERIFIED, source="finance"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure != 120000.0


def test_compatible_hour_quantity_still_links_to_hourly_rate():
    """Contrast case: a quantity genuinely stated in the SAME unit as the
    rate (hours x per-hour) must still link and multiply correctly."""
    ledger = [
        EvidenceItem(claim="Ten hours of downtime were verified from system logs.", status=EvidenceStatus.VERIFIED, source="logs"),
        EvidenceItem(claim="The downtime cost rate is INR 12,000 per hour.", status=EvidenceStatus.VERIFIED, source="finance"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 120000.0


def test_time_duration_not_multiplied_by_incompatible_batch_rate():
    ledger = [
        EvidenceItem(claim="Eight verified hours of downtime were recorded.", status=EvidenceStatus.VERIFIED, source="QA log"),
        EvidenceItem(claim="The verified remediation rate is INR 3,000 per batch.", status=EvidenceStatus.VERIFIED, source="finance"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure != 24000.0


def test_compatible_batch_quantity_still_links_to_batch_rate():
    ledger = [
        EvidenceItem(claim="Eight verified batches were affected.", status=EvidenceStatus.VERIFIED, source="QA log"),
        EvidenceItem(claim="The verified remediation rate is INR 3,000 per batch.", status=EvidenceStatus.VERIFIED, source="finance"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 24000.0


def test_bare_each_wildcard_still_links_to_any_quantity_unit():
    """A rate stated with bare "each" (no specific unit noun) is a
    wildcard and must still link to a quantity of any unit class,
    preserving pre-existing "each"-based linking behavior."""
    ledger = [
        EvidenceItem(claim="Eight verified nonconformities were identified.", status=EvidenceStatus.VERIFIED, source="QA log"),
        EvidenceItem(claim="Each is reported to cost approximately INR 15,000.", status=EvidenceStatus.REPORTED, source="estimate"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.reported_financial_exposure == 120000.0
