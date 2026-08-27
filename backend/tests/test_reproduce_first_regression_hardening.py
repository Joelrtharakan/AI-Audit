"""Regression coverage for the reproduce-first regression hardening pass on
the AI Finding Investigation Analysis pipeline.

Defects reproduced directly against the pipeline BEFORE any code change,
root-caused, and fixed with the smallest deterministic change:

1. QUANTITY x RATE COLLAPSE (app/financial/extractor.py):
   `_EVENT_COUNT_WORD_RE`'s comma-fragment guard rejected ANY comma-grouped
   number outright (not just a fragment of a larger amount), so "1,000
   units required rework at INR 250 per unit." silently fell back to a
   count of 1, calculating INR 250 instead of INR 250,000. `_parse_count_
   token` also could not parse a comma-containing string at all (`int(
   "1,000")` raises). Fixed by matching a WHOLE well-formed comma-grouped
   number (rejecting only a genuine fragment via lookaround, and a "per
   <unit>"/"/" rate-denominator context via negative lookahead), stripping
   commas in `_parse_count_token`, and masking recognized amount spans out
   of the search text before hunting for a count (so an amount's own
   comma-grouped number, e.g. "INR 15,000", can never be misread as the
   count of a LATER unit noun in the same sentence -- this exact ambiguity
   surfaced as a regression while fixing the comma-number matching and was
   caught by the pre-existing test_average_cost_with_explicit_count_
   calculates_normally before being shipped).

2. AVERAGE-RATE DOWNGRADE RACE (app/financial/extractor.py): the "average
   rate without a local count" VERIFIED->UNVERIFIED downgrade was applied
   INSIDE the per-source-statement loop, before the cross-evidence
   quantity-linking pass (later in the same function) had a chance to
   attach a compatible, independently VERIFIED count from a SEPARATE
   evidence statement. This meant "8 incidents were verified..." (C1) +
   "The average cost per incident was INR 15,000." (C2, both VERIFIED)
   permanently downgraded to UNVERIFIED even though both underlying facts
   were VERIFIED and compatible. Fixed by deferring the downgrade decision
   to AFTER cross-evidence linking runs, only downgrading an observation
   that STILL has no count once linking has had its chance.

3. RECOVERY DISAPPEARING (app/financial/calculator.py,
   app/financial/models.py): `calculate_confirmed_impact` only recognized
   a RECOVERY-type observation when its verification_status was exactly
   VERIFIED; a REPORTED/UNVERIFIED recovery observation matched neither
   the verified-recovery filter NOR the reported-loss filter (which
   explicitly excludes amount_type == RECOVERY), so it vanished from the
   result entirely with no trace in any field. Fixed by adding a
   `reported_recovery` field, populated from REPORTED/UNVERIFIED recovery
   observations, never fed into confirmed_net_loss, and rendered in the
   frontend labeled "(REPORTED)" distinct from a verified recovery.

4. GRAMMATICAL-FRAGMENT / HISTORICAL-CLAUSE SUBJECT FABRICATION (app/
   services/semantic_subject.py): `_SELF_REFERENTIAL_EVIDENCE_PREFIX_RE`
   (the single shared "records show that..."-style framing-prefix
   stripper) required the self-referential noun phrase to appear with NO
   leading modifier, so "Historical records show..." (modifier "Historical"
   before "records") failed to match and left the literal verb "show" as
   part of the resolved subject ("show the same type of failure"). Fixed
   by allowing an optional historical/prior/previous/recent/current
   modifier before the noun phrase, and adding "audit log(s)" to the
   recognized noun list. A second, previously-latent defect surfaced once
   the framing was correctly stripped: "same type of failure" (a category
   comparison, not a named entity) then resolved AS the subject -- fixed
   with a new reject_subject_if_clause rule for "same type/kind of
   <generic-occurrence-noun>", plus two narrow rules for the truncated
   remainders this can produce ("same type" alone; a bare discourse
   adverb like "previously" left over once the noun phrase before it is
   rejected).

5. RECURRING WORDING FABRICATING A PREVIOUS CAPA (app/agent/nodes/
   plan_investigation_fallback.py): the RECURRENCE block in
   build_deterministic_investigation_plan generated previous-CAPA
   implementation/effectiveness/scope investigation questions whenever
   `recurrence.is_recurring` was true, with NO check on `recurrence.
   has_previous_capa_reference` (unlike the equivalent Five-Why fallback
   branch, which already correctly checks both). A finding merely
   containing "the same X occurred" (matched by detect_recurrence's own
   structural pattern) fabricated an entire previous-CAPA investigation
   narrative for a finding that never mentioned a CAPA. Fixed by gating
   the previous-CAPA-specific branch on `has_previous_capa_reference` too,
   with a new non-CAPA recurrence-history branch covering the
   recurring-without-previous-CAPA case instead of silently doing nothing.

Uses abstract, domain-neutral test fixtures.
"""

from __future__ import annotations

from app.agent.nodes.five_why_fallback import build_deterministic_five_why
from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
from app.financial.calculator import calculate_confirmed_impact
from app.financial.engine import analyze_financial_exposure
from app.financial.extractor import _parse_count_token, extract_financial_observations
from app.financial.models import FinancialAmountType, FinancialObservation
from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.semantic_subject import reject_subject_if_clause, resolve_deviation


# ---------------------------------------------------------------------------
# 1. Current quantity x current rate integrity
# ---------------------------------------------------------------------------

def test_current_quantity_times_current_rate_same_source():
    """The exact reproduction: '1,000 units ... INR 250 per unit' must
    calculate 1,000 x 250 = 250,000, never collapse to a single event."""
    ledger = [EvidenceItem(claim="1,000 units required rework at INR 250 per unit.", status=EvidenceStatus.VERIFIED, source="C1")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 250000.0


def test_current_quantity_times_current_rate_cross_evidence():
    ledger = [
        EvidenceItem(claim="1,000 units were identified as requiring rework.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="The rework cost is INR 250 per unit.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 250000.0


def test_parse_count_token_handles_comma_grouped_numbers():
    assert _parse_count_token("1,000") == 1000
    assert _parse_count_token("12,500") == 12500
    assert _parse_count_token("10") == 10


def test_rate_denominator_number_never_misread_as_event_count():
    """'INR 12,000 per day' must never read '12,000' as a count of days."""
    obs, _, _ = extract_financial_observations("A loss of INR 12,000 per day was confirmed.", evidence_ledger=None)
    assert all(o.event_count != 12000 for o in obs)


# ---------------------------------------------------------------------------
# 2. Average rate safety (regression matrix items 3, 4, 5, 6, 7, 8)
# ---------------------------------------------------------------------------

def test_average_rate_without_quantity_never_verified():
    ledger = [EvidenceItem(claim="The average cost per incident was INR 15,000.", status=EvidenceStatus.VERIFIED, source="C1")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None
    assert res.confirmed_impact.reported_financial_exposure == 15000.0


def test_average_rate_with_same_source_quantity_calculates():
    ledger = [EvidenceItem(claim="The average cost per incident was verified at INR 15,000 across 8 verified incidents.", status=EvidenceStatus.VERIFIED, source="C1")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 120000.0


def test_average_rate_with_cross_evidence_verified_quantity_calculates():
    """The exact reproduction of the deferred-downgrade race: both facts
    VERIFIED and compatible must be able to combine into a VERIFIED total,
    not get stuck downgraded to UNVERIFIED by the premature per-source
    average-without-count check."""
    ledger = [
        EvidenceItem(claim="8 incidents were verified during the audit.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="The average cost per incident was INR 15,000.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure == 120000.0


def test_reported_average_rate_with_verified_quantity_stays_unverified():
    """A REPORTED average rate must never become VERIFIED merely because
    the quantity paired with it is independently VERIFIED -- bounded by
    the weaker of the two provenances."""
    ledger = [
        EvidenceItem(claim="8 incidents were verified during the audit.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="The average cost per incident was INR 15,000.", status=EvidenceStatus.REPORTED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None
    assert res.confirmed_impact.reported_financial_exposure == 120000.0


def test_incompatible_quantity_rate_units_never_linked():
    ledger = [
        EvidenceItem(claim="8 batches were identified.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="The average cost per incident was INR 15,000.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None


def test_historical_rate_never_multiplied_by_current_quantity():
    ledger = [
        EvidenceItem(claim="8 incidents were verified during the audit.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="Historically, the average cost per incident was INR 15,000.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None


def test_current_rate_never_multiplied_by_historical_quantity():
    ledger = [
        EvidenceItem(claim="Historically, 8 incidents were verified.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="The average cost per incident was INR 15,000.", status=EvidenceStatus.VERIFIED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.verified_gross_exposure is None


# ---------------------------------------------------------------------------
# 3. Recovery must never disappear
# ---------------------------------------------------------------------------

def test_reported_recovery_preserved_not_discarded():
    """The exact reproduction: a REPORTED recovery observation matched
    neither the verified-recovery filter nor the reported-loss filter and
    vanished from the result entirely."""
    ledger = [EvidenceItem(claim="INR 40,000 was reportedly recovered from the supplier.", status=EvidenceStatus.REPORTED, source="C1")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.reported_recovery == 40000.0
    assert res.confirmed_impact.verified_recovery is None


def test_reported_recovery_never_becomes_verified_or_feeds_net_loss():
    ledger = [
        EvidenceItem(claim="A direct loss of INR 100,000 was confirmed.", status=EvidenceStatus.VERIFIED, source="C1"),
        EvidenceItem(claim="INR 40,000 was reportedly recovered from the supplier.", status=EvidenceStatus.REPORTED, source="C2"),
    ]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.confirmed_impact.reported_recovery == 40000.0
    assert res.confirmed_impact.verified_recovery is None
    assert res.confirmed_impact.confirmed_net_loss is None
    assert res.confirmed_impact.potential_unrecovered_exposure == 100000.0


def test_verified_recovery_still_works_unaffected():
    obs = [FinancialObservation(observation_id="o1", amount=40000.0, currency="INR", amount_type=FinancialAmountType.RECOVERY, verification_status="VERIFIED")]
    result = calculate_confirmed_impact(obs)
    assert result.verified_recovery == 40000.0
    assert result.reported_recovery is None


# ---------------------------------------------------------------------------
# 4. Affected-object integrity: historical clause / grammatical fragment
# ---------------------------------------------------------------------------

def test_historical_records_show_clause_never_becomes_subject():
    """The exact reproduction: 'Historical records show...' left the verb
    'show' inside the resolved subject due to a framing-prefix regex that
    could not tolerate the leading 'Historical' modifier."""
    for finding in (
        "Historical records show the same type of failure.",
        "Historical records show the same type of failure occurred previously.",
        "Records show the same type of failure.",
    ):
        r = resolve_deviation(finding, [])
        assert r.subject is None, finding


def test_same_type_of_failure_phrase_rejected():
    for phrase in ("same type of failure", "the same type of failure", "same kind of incident", "same type of failure occurred previously"):
        assert reject_subject_if_clause(phrase) is True, phrase


def test_truncated_remainders_of_same_type_phrase_rejected():
    for phrase in ("same type", "type", "previously", "historically"):
        assert reject_subject_if_clause(phrase) is True, phrase


def test_grounded_entity_still_extracted_from_reporting_verb_sentence():
    """Contrast case: a real entity named after a reporting verb must
    still be extracted -- this fix only strips the framing prefix, it
    does not suppress genuine extraction."""
    r = resolve_deviation("The audit log shows the calibration was overdue.", [])
    assert r.subject is not None
    assert "shows" not in (r.subject or "").lower()


def test_historical_clause_finding_no_fabricated_investigation_subject():
    """No 'What procedure governs Historical records show...'-style
    malformed question -- the generic fallback subject must be used
    instead of an arbitrary raw clause."""
    finding = "Historical records show the same type of failure occurred previously."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="audit")]
    hyps, plan = build_deterministic_investigation_plan(finding, ledger)
    all_text = " ".join(plan.areas) + " " + " ".join(q.question for q in plan.questions)
    assert "historical records show" not in all_text.lower()
    assert "governs historical" not in all_text.lower()


# ---------------------------------------------------------------------------
# 5. Investigation-plan contamination: recurrence != previous CAPA
# ---------------------------------------------------------------------------

def test_recurring_wording_without_previous_capa_never_fabricates_capa_questions():
    """The exact reproduction: 'the same X occurred' (recurrence-shaped
    wording) with NO previous-CAPA reference anywhere in the evidence must
    never trigger previous-CAPA implementation/effectiveness/scope
    questions."""
    finding = "Historical records show the same type of failure occurred previously."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="audit")]
    hyps, plan = build_deterministic_investigation_plan(finding, ledger)
    all_text = " ".join(plan.areas) + " " + " ".join(q.question for q in plan.questions)
    assert "capa" not in all_text.lower()
    assert "previous" not in all_text.lower()


def test_ten_previous_failures_without_capa_reference_never_fabricates_capa():
    finding = "10 previous failures of the same type were identified in the log."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="audit")]
    hyps, plan = build_deterministic_investigation_plan(finding, ledger)
    all_text = " ".join(plan.areas) + " " + " ".join(q.question for q in plan.questions)
    assert "capa" not in all_text.lower()


def test_genuine_previous_capa_reference_still_generates_capa_questions():
    """Contrast case: this fix must not suppress the LEGITIMATE case where
    the evidence actually references a previous corrective action."""
    finding = "The nonconformity recurred after the previous corrective action for calibration drift."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="audit")]
    hyps, plan = build_deterministic_investigation_plan(finding, ledger)
    all_text = " ".join(plan.areas) + " " + " ".join(q.question for q in plan.questions)
    assert "capa" in all_text.lower()


# ---------------------------------------------------------------------------
# 6. Five-Why causal adjacency: recovery/historical facts never substitute
# for a causal mechanism answer (spot checks; deterministic fallback path)
# ---------------------------------------------------------------------------

def test_five_why_never_answers_with_unrelated_recovery_fact():
    finding = "A packaging failure occurred during the shift."
    ledger = [
        EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="audit"),
        EvidenceItem(claim="INR 40,000 was recovered from the supplier.", status=EvidenceStatus.VERIFIED, source="fin"),
        EvidenceItem(claim="10 historical failures occurred last year.", status=EvidenceStatus.VERIFIED, source="hist"),
    ]
    result = build_deterministic_five_why(finding, ledger)
    combined = " ".join((s.answer or "") for s in result.steps).lower()
    assert "recovered" not in combined
    assert "40,000" not in combined
    assert result.steps[0].status == "UNKNOWN"
