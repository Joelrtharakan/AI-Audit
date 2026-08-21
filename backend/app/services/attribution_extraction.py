"""Deterministic, LLM-free structural attribution extraction.

Used only when the LLM extraction call is unavailable (degraded mode). The
LLM-based extractor (app/services/extraction.py) normally distinguishes
VERIFIED stated_facts from REPORTED attributed_statements; when that call
fails, the previous fallback collapsed everything to plain sentence-level
VERIFIED facts, silently destroying the REPORTED/attribution distinction
for any causal statement already explicitly present in the finding text
(e.g. "The operator stated that they were unaware the procedure had been
revised"). This module recovers that distinction structurally -- verb-of-
report constructions and awareness/knowledge-gap constructions -- without
any finding-specific or domain-specific vocabulary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# "<Speaker> stated/reported/claimed/... (that) <claim>"
_REPORT_VERB_RE = re.compile(
    r"^(?P<speaker>[A-Z][\w\s-]{0,50}?)\s+"
    r"(?:stated|reported|confirmed|indicated|noted|mentioned|claimed|said|"
    r"explained|advised|acknowledged|attributed\s+(?:this|it|the\s+issue|the\s+failure)?\s*to|cited|"
    r"states|reports|confirms|indicates|notes|mentions|claims|says|"
    r"explains|advises|acknowledges|"
    r"state|report|confirm|indicate|note|mention|claim|say|"
    r"explain|advise|acknowledge)\s+"
    r"(?:that\s+)?(?P<claim>.+)$",
    re.IGNORECASE,
)

# "<Speaker> was/were unaware/not aware/not informed/not notified (that|of) <claim>"
_AWARENESS_GAP_RE = re.compile(
    r"^(?P<speaker>[A-Z][\w\s-]{0,50}?)\s+(?:was|were)\s+"
    r"(?P<claim>(?:unaware|not\s+aware|not\s+informed|not\s+notified|not\s+told).*)$",
    re.IGNORECASE,
)

# "<Speaker> did not know / didn't know (that) <claim>"
_DID_NOT_KNOW_RE = re.compile(
    r"^(?P<speaker>[A-Z][\w\s-]{0,50}?)\s+(?:did\s+not|didn't)\s+know\s+"
    r"(?P<claim>.*)$",
    re.IGNORECASE,
)

_ATTRIBUTION_PATTERNS = (_REPORT_VERB_RE, _AWARENESS_GAP_RE, _DID_NOT_KNOW_RE)

# A sentence that is neither an instruction, a stance, nor an attribution
# still isn't automatically audit evidence -- social/expressive speech acts
# (thanks, praise, greetings, well-wishes) carry no verifiable proposition
# about any real-world entity or condition, yet would otherwise fall
# through to plain "FACT" and enter the evidence ledger. This is only
# excluded when BOTH conditions hold: the sentence performs a recognized
# expressive speech act (structural verb class, not a specific phrase) AND
# it contains no evidentiary marker of its own -- so an ordinary terse
# audit statement is never at risk, only content that is purely social.
_EXPRESSIVE_SPEECH_ACT_RE = re.compile(
    r"\b(?:thank(?:s|ed|ing)?|appreciat(?:e|ed|ion)|grateful|praise[ds]?|commend(?:ed)?|"
    r"congratulat(?:e|ed|ions)|apolog(?:ize|ise|ized|ised|y|ies)|welcome[ds]?|"
    r"good\s+job|great\s+job|well\s+done|nice\s+work|wanted\s+to\s+say|"
    r"hope\s+(?:you|everyone)|good\s+(?:morning|afternoon|evening)|happy\s+(?:to|holidays|new\s+year))\b",
    re.IGNORECASE,
)
_EVIDENTIARY_CONTENT_RE = re.compile(
    r"\b(?:not|no|never|without|missing|absent|overdue|unauthoriz(?:ed|ation)|duplicat(?:e|ed|ion)|"
    r"discrepanc(?:y|ies)|deviat(?:e|ed|ion)|nonconform(?:ing|ance|ity)|exceed(?:s|ed|ing)?|"
    r"below|above|breach(?:ed)?|violat(?:e|ed|ion)|expired?|incomplete|could\s+not|fail(?:ed|s)?\s+to|"
    r"record|records|log|logs|certificate|report|checklist|procedure|requirement|specification|"
    r"inspection|audit\s+trail|invoice|payment|permit|calibrat|signature|approv(?:al|ed)|reconcil|"
    r"batch|lot|sample|equipment|machine|device|system|contractor|employee|vendor|supplier|"
    r"training|schedule|deadline|revision|version)\b",
    re.IGNORECASE,
)


def _is_expressive_non_substantive(sentence: str) -> bool:
    if any(ch.isdigit() for ch in sentence):
        return False
    if _EVIDENTIARY_CONTENT_RE.search(sentence):
        return False
    return bool(_EXPRESSIVE_SPEECH_ACT_RE.search(sentence))


# ---------------------------------------------------------------------------
# Generalized segment classification (Defects 1, 2, 6).
#
# The historic splitter answered exactly one question -- "does this sentence
# match a speech-attribution marker?" -- and defaulted everything else to
# VERIFIED fact. That default is what made a belief ("the security team
# believes ...") and a counterfactual ("if the permit had been issued, the
# reading would have been logged") both come out as observed fact.
#
# `classify_finding_segments` replaces the single boolean with four
# independent structural questions, each answered by its own generalized
# mechanism, so the VERIFIED default is now the RESIDUE of four exclusions
# rather than the unexamined fallback:
#   1. is the segment a system-directed instruction?      -> UNTRUSTED
#   2. does an epistemic-stance predicate govern it?      -> STANCE
#   3. does a speech-attribution predicate govern it?     -> ATTRIBUTED
#   4. what is its grammatical mood?                      -> modality axis
# ---------------------------------------------------------------------------


@dataclass
class FindingSegment:
    """One independently classified proposition from the finding text."""
    text: str
    kind: str = "FACT"              # FACT | ATTRIBUTED | STANCE | UNTRUSTED | NON_SUBSTANTIVE
    speaker: str | None = None      # attribution speaker / stance holder
    claim: str | None = None        # the embedded proposition, when there is one
    stance: str | None = None       # BELIEF|DOUBT|SUSPICION|ASSUMPTION|OPINION
    modality: str = "ACTUAL"        # ACTUAL | CONDITIONAL | COUNTERFACTUAL
    modality_marker: str | None = None
    security_classification: str = "NORMAL"


def split_sentences(finding_text: str) -> list[str]:
    """Sentence-level segmentation shared by every classifier here."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split((finding_text or "").strip()) if s.strip()]


def classify_finding_segments(finding_text: str) -> list[FindingSegment]:
    """Segment a finding and classify each segment on all four axes."""
    from app.services.epistemic_modality import classify_epistemic_stance, classify_modality
    from app.services.instruction_detector import classify_instruction

    segments: list[FindingSegment] = []
    for sentence in split_sentences(finding_text):
        # 1. Untrusted / system-directed instruction -> quarantine.
        instr = classify_instruction(sentence)
        if instr.is_untrusted:
            segments.append(FindingSegment(
                text=sentence, kind="UNTRUSTED",
                security_classification=instr.classification,
            ))
            continue

        # 4. Grammatical mood (computed for every kind; orthogonal axis).
        mood = classify_modality(sentence)

        # 1b. Purely expressive/social content -- checked before stance and
        # attribution because a filler opener like "just wanted to say ..."
        # or "really appreciated ..." otherwise surface-matches the same
        # speech-attribution shape as genuine reported speech ("X said
        # that ..."). Only content with no evidentiary marker of its own is
        # ever excluded here, so a real (if informally phrased) audit
        # statement is never at risk.
        if _is_expressive_non_substantive(sentence):
            segments.append(FindingSegment(
                text=sentence, kind="NON_SUBSTANTIVE",
                modality=mood.modality, modality_marker=mood.marker,
            ))
            continue

        # 2. Epistemic stance -- checked BEFORE speech attribution because
        # "X believes that Y" and "X stated that Y" are the same surface
        # shape and only the predicate's semantic class separates them.
        stance = classify_epistemic_stance(sentence)
        if stance is not None:
            segments.append(FindingSegment(
                text=sentence, kind="STANCE",
                speaker=stance.holder, claim=stance.proposition,
                stance=stance.stance,
                modality=mood.modality, modality_marker=mood.marker,
            ))
            continue

        # 3. Speech attribution.
        attributed = False
        for pattern in _ATTRIBUTION_PATTERNS:
            m = pattern.match(sentence)
            if m:
                claim = m.group("claim").strip().rstrip(".")
                if claim:
                    segments.append(FindingSegment(
                        text=sentence, kind="ATTRIBUTED",
                        speaker=m.group("speaker").strip(), claim=claim,
                        modality=mood.modality, modality_marker=mood.marker,
                    ))
                    attributed = True
                break
        if attributed:
            continue

        # 5. Purely expressive/social content (thanks, praise, greetings)
        # with no evidentiary marker of its own -- excluded so it can't
        # supply the "affected object" or enter the evidence ledger.
        if _is_expressive_non_substantive(sentence):
            segments.append(FindingSegment(
                text=sentence, kind="NON_SUBSTANTIVE",
                modality=mood.modality, modality_marker=mood.marker,
            ))
            continue

        segments.append(FindingSegment(
            text=sentence, kind="FACT",
            modality=mood.modality, modality_marker=mood.marker,
        ))
    return segments


def split_facts_and_attributed_statements(finding_text: str) -> tuple[list[str], list[dict]]:
    """Deterministic, LLM-free split of a finding into (VERIFIED sentence
    facts, REPORTED attributed statements) using structural attribution
    patterns -- verb-of-report constructions and awareness-gap
    constructions -- never domain vocabulary. Each attributed statement is
    returned as {"speaker": ..., "claim": ...} matching AttributedStatement.

    Falls back to treating a sentence as a plain fact when it doesn't match
    an attribution pattern -- this never loses information, it only adds
    the REPORTED classification where the sentence structure supports it.

    `stated_facts` now carries ONLY segments that are genuinely assertions of
    actual fact: epistemic-stance segments, non-actual (conditional/
    counterfactual) segments and quarantined instruction segments are
    withheld here so they cannot be re-promoted to VERIFIED by any consumer
    of this legacy 2-tuple. Callers that need those segments (the evidence
    ledger builder) use `classify_finding_segments` instead, which preserves
    every proposition with its own classification."""
    facts: list[str] = []
    attributed: list[dict] = []
    for seg in classify_finding_segments(finding_text):
        if seg.kind == "ATTRIBUTED" and seg.claim:
            attributed.append({"speaker": seg.speaker or "", "claim": seg.claim})
        elif seg.kind == "FACT" and seg.modality == "ACTUAL":
            facts.append(seg.text)
    return facts, attributed
