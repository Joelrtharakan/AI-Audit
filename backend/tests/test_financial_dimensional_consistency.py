"""Regression coverage for two defects found while hardening dimensional
compatibility between financial quantities and rates.

Bug 1 -- rate numeral misread as event count: "The verified cost rate is
INR 500 per day" was extracted with event_count=500 (!) because the
event-count regex matched "500 per day" as if "500" were a count of
"500 days" -- the SAME numeral that is the rate's own value. This
corrupted the observation's own event_count field directly (not just
the cross-clause linking path), producing wildly wrong totals (e.g.
10 hours x [500-events-of-]INR 500/day = 250,000 instead of a blocked
or correctly-scoped result). Fixed by rejecting an event-count match
whose digit span overlaps a monetary-amount match in the same text.

Bug 2 -- overly coarse dimension classes: quantity/rate unit
compatibility previously grouped "unit"/"item"/"batch"/"delivery" into
one ITEM class and "hour"/"day"/"week"/"month" into one TIME class,
letting "8 units" incorrectly multiply a "per batch" rate, or "10 hours"
incorrectly multiply a "per day" rate -- despite these NOT being
interchangeable measurement dimensions (a batch may contain many units;
a day is not an hour). Fixed by splitting into fully distinct classes
(UNIT, ITEM, BATCH, DELIVERY, TIME_HOUR, TIME_DAY, TIME_WEEK,
TIME_MONTH) that are never automatically treated as equivalent, plus a
related pluralization bug ("batches" -> "batche" via naive "-s"
stripping never matched "batch", silently falling through to the
wildcard/unrecognized case and defeating the compatibility check
entirely).

OCCURRENCE-class nouns (event/incident/occurrence/defect/nonconformity/
transaction) are deliberately left grouped as one class -- unlike the
batch/unit/delivery containment hierarchy, these are near-synonyms for
"a thing that happened" with no demonstrated false-calculation defect,
and splitting them further was not attempted given no reproduced bug
supports it (disclosed as a scoping decision in the accompanying report,
not silently assumed complete).

Uses abstract, domain-neutral test fixtures.
"""

from __future__ import annotations

from app.financial.engine import analyze_financial_exposure
from app.financial.extractor import _unit_noun_class
from app.models.agent import EvidenceItem, EvidenceStatus


def _run(claims: list[str]) -> float | None:
    ledger = [EvidenceItem(claim=c, status=EvidenceStatus.VERIFIED, source="src") for c in claims]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    return res.confirmed_impact.verified_gross_exposure


def test_rate_numeral_never_misread_as_its_own_event_count():
    """The exact reproduction: 'INR 500 per day' must not internally
    acquire event_count=500 from misreading its own numeral."""
    from app.financial.extractor import extract_financial_observations

    obs, *_ = extract_financial_observations(
        "x",
        evidence_ledger=[
            EvidenceItem(claim="The verified cost rate is INR 500 per day.", status=EvidenceStatus.VERIFIED, source="finance"),
        ],
    )
    assert len(obs) == 1
    assert obs[0].event_count != 500


def test_hour_quantity_not_multiplied_by_incompatible_daily_rate():
    assert _run(["Ten verified hours of downtime were recorded.", "The verified cost rate is INR 500 per day."]) != 5000.0


def test_hour_quantity_still_multiplies_compatible_hourly_rate():
    assert _run(["Ten verified hours of downtime were recorded.", "The verified cost rate is INR 500 per hour."]) == 5000.0


def test_unit_quantity_not_multiplied_by_incompatible_batch_rate():
    assert _run(["Eight verified units were reworked.", "The cost per batch was verified at INR 20,000."]) != 160000.0


def test_unit_quantity_still_multiplies_compatible_unit_rate():
    assert _run(["Eight verified units were reworked.", "The cost per unit was verified at INR 450."]) == 3600.0


def test_batch_quantity_not_multiplied_by_incompatible_delivery_rate():
    assert _run(["Eight verified batches were affected.", "The rework cost was verified at INR 3,000 per delivery."]) != 24000.0


def test_batch_quantity_still_multiplies_compatible_batch_rate():
    assert _run(["Eight verified batches were affected.", "The rework cost was verified at INR 3,000 per batch."]) == 24000.0


def test_pluralization_of_ch_ending_noun_normalizes_correctly():
    """Regression guard for the 'batches' -> wildcard-fallback defect:
    standard English -es pluralization after ch/sh/x/s/z must resolve to
    the correct singular class, not silently fall through unrecognized."""
    assert _unit_noun_class("batches") == "BATCH"
    assert _unit_noun_class("batch") == "BATCH"
