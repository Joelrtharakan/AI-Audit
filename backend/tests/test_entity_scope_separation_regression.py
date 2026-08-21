"""Regression suite for entity/attribute/scope separation in the generic
extraction fallback (app.services.semantic_subject).

Locks in the fix for a structural conflation defect: _clean_subject() only
stripped LEADING noise (articles, quantity words, framing prefixes) from a
subject captured by the generic _CONDITION_PATTERNS/_PREFIX_CONDITION_
PATTERNS/_SHORT_DEVIATION_PATTERNS fallbacks -- an embedded date range or
population clause inside the grammatical subject ("The checklist for 15
employees between 7 and 9 August was incomplete") ended up baked into
affected_object as one string instead of three separate typed fields
(affected_object / occurrence_population / date).

Scenarios deliberately vary domain/vocabulary/sentence structure (not one
worked example) per the anti-overfitting requirement: population-only,
date-only, both together, and a document-identifier case where entity
enrichment is intentional and must NOT be stripped.
"""

from __future__ import annotations

from app.services.semantic_subject import extract_semantic_subject


def test_population_and_date_range_kept_out_of_entity_name():
    text = "The pre-operational checklist for 15 employees between 7 and 9 August was incomplete."
    d = extract_semantic_subject(text)
    assert d.matched
    assert "for 15" not in d.affected_object.lower()
    assert "august" not in d.affected_object.lower()
    assert d.occurrence_population and "15" in d.occurrence_population
    assert d.date and "august" in d.date.lower()


def test_population_only_kept_separate_across_domains():
    text = "The training record for 8 technicians was missing."
    d = extract_semantic_subject(text)
    assert d.matched
    assert "technicians" not in d.affected_object.lower()
    assert d.occurrence_population and "8" in d.occurrence_population


def test_no_population_or_date_clause_unaffected():
    """Baseline: a plain finding with neither clause must extract normally
    -- the new stripping must not remove legitimate subject words."""
    text = "The batch release checklist was incomplete."
    d = extract_semantic_subject(text)
    assert d.matched
    assert d.affected_object.lower() == "batch release checklist"
    assert d.occurrence_population is None


def test_financial_domain_population_clause_separated():
    """Different domain (financial vs. quality) -- confirms the generic
    for-N-noun pattern generalizes and isn't tied to quality-domain nouns."""
    text = "The reconciliation worksheet for 12 suppliers was not reviewed."
    d = extract_semantic_subject(text)
    assert d.matched
    assert "suppliers" not in d.affected_object.lower()
    assert d.occurrence_population and "12" in d.occurrence_population


def test_document_identifier_enrichment_still_intentional():
    """A document-code enrichment onto a 'log'/'label' subject is
    deliberate existing behavior (adds needed disambiguating context) --
    must NOT be treated as conflation and stripped."""
    text = "The calibration log for SOP-ENG-002 was not completed."
    d = extract_semantic_subject(text)
    assert d.matched
    assert "SOP-ENG-002" in d.affected_object
    assert "SOP-ENG-002" in d.entities
