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

# Passive-voice report construction: "<content> was/were/is/are/has been/
# have been reported/stated/claimed/alleged/indicated (by <speaker>)?".
# _REPORT_VERB_RE only recognizes ACTIVE voice ("X reported that Y") --
# but "A supplier overpayment of INR 200,000 was reported during the
# reconciliation review" is exactly as much an attribution/hedge as its
# active-voice equivalent ("Finance reported that a supplier overpayment
# of INR 200,000 occurred"), just without a named agent performing the
# reporting. Missing this construction meant a REPORTED finding written
# in passive voice fell through to plain FACT (-> VERIFIED by default),
# silently upgrading its evidence status -- a defect entirely upstream of
# financial semantics, which only ever preserves whatever status the
# evidence ledger already assigned it.
_PASSIVE_REPORT_RE = re.compile(
    r"\b(?:was|were|is|are|has\s+been|have\s+been)\s+"
    r"(?:reported|stated|claimed|alleged|indicated)\b",
    re.IGNORECASE,
)
# A trailing named agent ("... was reported BY the warehouse supervisor.")
# is a separate, optional sub-match -- kept independent of the marker
# above so trailing modifier text with no agent at all ("... was reported
# during the reconciliation review.") still matches the marker without
# needing to also satisfy an end-of-string constraint right after the verb.
_PASSIVE_REPORT_AGENT_RE = re.compile(r"\bby\s+(?P<speaker>[\w-]+(?:\s+[\w-]+){0,6})\s*[.,]?\s*$", re.IGNORECASE)

# Every bare-form report verb _REPORT_VERB_RE can match as the predicate
# (mirrors the third alternative group inside _REPORT_VERB_RE). "note" is
# structurally ambiguous: it is also the second half of a compound noun
# ("credit note", "delivery note", "cover note", "promissory note") that
# names a financial/business document, not a person reporting something.
# _REPORT_VERB_RE's non-greedy speaker group cannot tell these apart on
# its own -- "Credit note confirms X" matches with speaker="Credit",
# verb="note", claim="confirms X", silently discarding "note" and folding
# the real verb ("confirms") into the claim text. Detected and corrected
# below rather than in the regex, since the fix needs to inspect what
# follows the matched verb, not just what precedes it.
_BARE_REPORT_VERBS = frozenset({
    "stated", "reported", "confirmed", "indicated", "noted", "mentioned",
    "claimed", "said", "explained", "advised", "acknowledged", "cited",
    "states", "reports", "confirms", "indicates", "notes", "mentions",
    "claims", "says", "explains", "advises", "acknowledges",
    "state", "report", "confirm", "indicate", "note", "mention", "claim",
    "say", "explain", "advise", "acknowledge",
})

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
                speaker = m.group("speaker").strip()
                # A genuine speaker/entity never ends on a bare copula or
                # auxiliary verb ("...40 units was", "...the shipment
                # has") -- that shape means the match actually straddles
                # a PASSIVE construction ("40 units was reported by X"),
                # where "reported" (a recognized active-voice report verb)
                # got matched against text that is really the tail of a
                # passive clause, not an active speaker. Reject the match
                # here so it falls through to the dedicated passive-voice
                # check below instead of misattributing a name.
                if pattern is _REPORT_VERB_RE and speaker.split()[-1:] and speaker.split()[-1].lower() in (
                    "was", "is", "were", "are", "been", "be"
                ):
                    continue
                if claim:
                    # "note" mis-consumed as the report verb, immediately
                    # followed by the ACTUAL report verb -- reattach "note"
                    # to the speaker (as the noun it really is: "Credit
                    # note", "Delivery note", ...) and let the following
                    # verb govern the claim instead. Harmless no-op for a
                    # genuine "X noted that <claim>" sentence, since a real
                    # claim practically never begins with another
                    # attribution verb.
                    claim_words = claim.split(None, 1)
                    if (
                        pattern is _REPORT_VERB_RE
                        and claim_words
                        and claim_words[0].lower() in _BARE_REPORT_VERBS
                        and sentence[len(speaker):].lstrip().lower().startswith("note ")
                    ):
                        speaker = f"{speaker} note"
                        claim = claim_words[1] if len(claim_words) > 1 else ""
                    if claim:
                        segments.append(FindingSegment(
                            text=sentence, kind="ATTRIBUTED",
                            speaker=speaker, claim=claim,
                            modality=mood.modality, modality_marker=mood.marker,
                        ))
                        attributed = True
                break
        if attributed:
            continue

        # 3b. Passive-voice report construction ("X was reported", with no
        # named reporting agent) -- same epistemic weight as active-voice
        # attribution, just a different surface form. The whole sentence
        # is kept as the claim (unlike active-voice attribution, there is
        # no clean "subject vs. embedded proposition" split to make
        # without losing trailing context), and speaker is only set when
        # an explicit "by <name>" is present.
        m = _PASSIVE_REPORT_RE.search(sentence)
        if m:
            agent_m = _PASSIVE_REPORT_AGENT_RE.search(sentence)
            speaker = agent_m.group("speaker").strip() if agent_m else None
            segments.append(FindingSegment(
                text=sentence, kind="ATTRIBUTED",
                speaker=speaker, claim=sentence.rstrip("."),
                modality=mood.modality, modality_marker=mood.marker,
            ))
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
