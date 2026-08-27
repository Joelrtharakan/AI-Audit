"""Regression for a real defect found while diagnosing golden-scenario
financial regressions: "note" is structurally ambiguous between the
report-verb sense ("X noted that Y") and the second half of a standard
accounting-document compound noun ("credit note", "delivery note", "debit
note", "cover note"). _REPORT_VERB_RE's non-greedy speaker group cannot
disambiguate on its own -- "Credit note confirms X" was matching with
speaker="Credit", verb="note", claim="confirms X", silently dropping "note"
and folding the real verb ("confirms") into the claim text. This corrupted
the evidence text fed into every downstream consumer, including the
financial semantic interpreter.

Fixtures here deliberately use different document nouns and different
predicate verbs than any scenario in tests/test_golden_20_scenarios.py.
"""

from __future__ import annotations

from app.agent.claim_extractor import _SYSTEM_RECORD_SPEAKER_RE
from app.services.attribution_extraction import classify_finding_segments


class TestNoteAsDocumentNounNotVerb:
    def test_credit_note_confirms_is_not_misparsed(self):
        segs = classify_finding_segments("Credit note confirms a refund of the disputed charge.")
        assert len(segs) == 1
        assert segs[0].speaker == "Credit note"
        assert segs[0].claim == "a refund of the disputed charge"

    def test_delivery_note_states_is_not_misparsed(self):
        segs = classify_finding_segments("Delivery note states 40 cartons were received.")
        assert len(segs) == 1
        assert segs[0].speaker == "Delivery note"
        assert segs[0].claim == "40 cartons were received"

    def test_debit_note_indicates_is_not_misparsed(self):
        segs = classify_finding_segments("Debit note indicates an additional charge was applied.")
        assert len(segs) == 1
        assert segs[0].speaker == "Debit note"
        assert segs[0].claim == "an additional charge was applied"

    def test_genuine_note_as_verb_still_works(self):
        # The fix must not break the ordinary "X noted that Y" case --
        # here "note" really is the predicate, not a noun.
        segs = classify_finding_segments("The reviewer noted that the checklist was incomplete.")
        assert len(segs) == 1
        assert segs[0].speaker == "The reviewer"
        assert segs[0].claim == "the checklist was incomplete"

    def test_person_named_after_a_document_word_context_unaffected(self):
        # A plain "<Person> confirmed Y" sentence (no "note" involved at
        # all) must be completely unaffected by this fix.
        segs = classify_finding_segments("The supervisor confirmed the deviation occurred.")
        assert len(segs) == 1
        assert segs[0].speaker == "The supervisor"
        assert segs[0].claim == "the deviation occurred"


class TestSystemRecordVocabularyIncludesAccountingNotes:
    """`_SYSTEM_RECORD_SPEAKER_RE` already treats other financial-document
    phrases (e.g. 'bank credit memo') as VERIFIED system records -- 'note'
    documents of the same accounting category ('credit note', 'debit
    note') were simply missing from that vocabulary, an inconsistency
    unrelated to any specific finding."""

    def test_credit_note_recognized_as_system_record(self):
        assert _SYSTEM_RECORD_SPEAKER_RE.search("Credit note")

    def test_debit_note_recognized_as_system_record(self):
        assert _SYSTEM_RECORD_SPEAKER_RE.search("Debit note")

    def test_unrelated_person_name_not_recognized(self):
        assert not _SYSTEM_RECORD_SPEAKER_RE.search("The site manager")


class TestPassiveVoiceReportAttribution:
    """Real defect: a REPORTED finding written in passive voice ("X was
    reported") fell through to plain FACT classification (-> VERIFIED by
    default) because the existing attribution pattern only recognized
    active voice ("X reported that Y"). This silently upgraded evidence
    status entirely upstream of financial semantics."""

    def test_passive_report_without_named_agent_is_attributed(self):
        segs = classify_finding_segments("A supplier overpayment of INR 200,000 was reported during the reconciliation review.")
        assert len(segs) == 1
        assert segs[0].kind == "ATTRIBUTED"
        assert segs[0].speaker is None

    def test_passive_report_with_named_agent_captures_speaker(self):
        segs = classify_finding_segments("An inventory shortfall of 40 units was reported by the warehouse supervisor.")
        assert len(segs) == 1
        assert segs[0].kind == "ATTRIBUTED"
        assert segs[0].speaker is not None and "warehouse supervisor" in segs[0].speaker.lower()

    def test_plain_verified_fact_is_unaffected(self):
        # A genuinely VERIFIED-shaped sentence (no passive report marker)
        # must remain a plain FACT.
        segs = classify_finding_segments("Bank records verify that INR 50,000 was received against the supplier account.")
        assert len(segs) == 1
        assert segs[0].kind == "FACT"
