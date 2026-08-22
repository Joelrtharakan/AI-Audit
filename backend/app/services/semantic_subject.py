"""General, non-finding-specific semantic subject/condition extraction and validation.

This module provides the single authoritative semantic resolution layer for ANY audit finding:
"what is the actual affected object / process / activity of the deviation, and what deviation
condition is being asserted about it?"

Critical architectural rules:
  1. SUBJECT EXTRACTION MUST NOT RETURN VERB PHRASES OR CLAUSES.
     "they had not received training" is an ACTION/CLAIM, never a finding subject.
  2. A valid subject is a noun phrase, entity, process, record, or object.
  3. Pronouns ("they", "he", "she", "it", "we") are never the finding subject.
  4. Finding subject != Reported mechanism != Evidence != Root cause.
  5. If an extracted subject contains a clause, it is rejected by `validate_semantic_subject()`.
"""

from __future__ import annotations

from enum import Enum
import re
from dataclasses import dataclass, field

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|,\s*while\s+|,\s*whereas\s+", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Pronouns and clause stop-words
# ---------------------------------------------------------------------------
_PRONOUNS = {
    "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "there", "whose",
}

# Verbs/auxiliaries that indicate a finite verb clause rather than a noun phrase
_FINITE_VERB_PATTERNS = [
    re.compile(r"\b(?:had|have|has|did|do|does|was|were|is|are)\s+not\b", re.IGNORECASE),
    re.compile(r"\b(?:had|have|has)\s+(?:been|received|completed|performed|conducted|attended|seen|missed|followed)\b", re.IGNORECASE),
    re.compile(r"\b(?:was|were|is|are)\b", re.IGNORECASE),
    re.compile(r"\b(?:did|do|does)\s+not\s+(?:receive|complete|perform|conduct|attend|follow|know|have)\b", re.IGNORECASE),
    re.compile(r"\b(?:could|should|would|might|may|must|cannot|can)\s+(?:not\s+)?(?:have|be|receive|complete)\b", re.IGNORECASE),
    re.compile(r"\b(?:stated|claimed|reported|said|mentioned|noted|indicated|confirmed|acknowledged)\b", re.IGNORECASE),
    re.compile(r"\b(?:they|he|she|we|i|you)\s+(?:had|have|has|was|were|did|could|should|would|stated|reported|claimed|said|received|completed|missed|failed|were|was|are|is)\b", re.IGNORECASE),
    re.compile(r"\b(?:but|however|although|whereas|because|while|since)\b", re.IGNORECASE),
]

# Single authoritative pattern for a "self-referential evidence" clause
# ("system records show that", "evidence indicates that", "the audit found
# that", "records confirm that", "the finding states that") prefixed onto an
# otherwise-clean proposition. This is deliberately the ONE place this shape
# is defined -- app/agent/causal_guard.py's hypothesis-vetting guard imports
# it from here rather than keeping its own narrower duplicate, and it is
# applied at BOTH extraction layers below (_strip_framing, on the whole
# sentence before pattern matching, and _clean_subject, on the captured
# subject group after matching) so a proposition sentence can never survive
# as an entity/subject value regardless of which extraction path produced
# it. Broader than the old hardcoded scada/server/dispatch-logs-only list it
# replaces: "records"/"data"/"evidence"/"the finding"/"the audit" are exactly
# as common a self-referential-evidence framing as "system logs".
_SELF_REFERENTIAL_EVIDENCE_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"(?:system\s+)?records?|the\s+logs?|logs?|data|the\s+system|"
    r"(?:the\s+)?evidence|the\s+finding|the\s+audit|"
    r"(?:scada|server|dispatch|distribution|security\s+badge|error|audit\s+trail|"
    r"[a-z]+\s+trail)\s+(?:system\s+|error\s+)?logs?"
    r")\s+"
    r"(?:show|shows|showed|state|states|stated|indicate|indicates|indicated|"
    r"confirm|confirms|confirmed|reveal|reveals|revealed|establish|establishes|"
    r"prove|proves|proved|find|finds|found)\s+(?:that\s+)?",
    re.IGNORECASE,
)

_ACCORDING_TO_EVIDENCE_PREFIX_RE = re.compile(
    r"^\s*according\s+to\s+(?:the\s+)?(?:evidence|records?|the\s+audit|the\s+finding|system\s+records?)\s*,?\s*",
    re.IGNORECASE,
)

# Reporting verbs & discourse preamble prefix patterns
_FRAMING_PREFIXES = [
    _SELF_REFERENTIAL_EVIDENCE_PREFIX_RE,
    _ACCORDING_TO_EVIDENCE_PREFIX_RE,
    re.compile(r"^\s*(?:just\s+)?wanted\s+to\s+(?:let\s+everyone\s+know|notify|inform\s+you)\s+(?:that\s+)?", re.IGNORECASE),
    re.compile(r"^\s*(?:please\s+note\s+that|note\s+that|be\s+advised\s+that)\s+", re.IGNORECASE),
    re.compile(r"^\s*during\s+.+?,\s*", re.IGNORECASE),
    re.compile(r"^\s*while\s+reviewing\s+.+?,\s*", re.IGNORECASE),
    re.compile(r"^\s*it\s+(?:was|is)\s+(?:observed|found|noted|identified|determined)\s+that\s+", re.IGNORECASE),
    re.compile(
        r"^\s*(?:the\s+)?(?:audit\s+)?observation\s+(?:states?|notes?|indicates?|reports?)\s+that\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:the\s+)?(?:auditor|inspector|reviewer|assessor)\s+"
        r"(?:identified|found|observed|noted|determined)\s+(?:that\s+)?",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*a\s+deviation\s+was\s+observed\s+(?:involving|regarding|in|with)\s+", re.IGNORECASE),
    re.compile(
        r"^\s*(?:the\s+)?(?:responsible\s+)?(?:technician|operator|staff|employee|supervisor|analyst|manager|trainer)\s+"
        r"(?:stated|confirmed|reported|said|noted|claimed|indicated)\s+(?:that\s+)?",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(?:under|per|as\s+per|in\s+accordance\s+with|pursuant\s+to)\s+(?:statutory\s+|regulatory\s+|applicable\s+)?(?:safety\s+|quality\s+|environmental\s+|compliance\s+|security\s+)?(?:standard|regulation|procedure|policy|rule|directive|guideline)\s+[A-Z0-9-]+\s*,\s*", re.IGNORECASE),
    re.compile(
        r"^\s*(?:(?:scada|server|system|audit|dispatch|distribution|security\s+badge|error)\s+(?:system\s+|error\s+)?logs?\s+(?:establish|proves?|shows?|confirms?|indicates?)\s+(?:that\s+)?)",
        re.IGNORECASE,
    ),
]


def reject_subject_if_clause(subject: str | None) -> bool:
    """Return True if `subject` is a verb phrase, clause, reported speech,
    or pronoun-led sentence fragment — meaning it must be REJECTED as a subject."""
    if not subject or not subject.strip():
        return True
    s = subject.strip()
    if len(s) < 3:
        return True

    words = s.lower().split()
    if not words:
        return True

    # 1. Starts with a pronoun
    if words[0] in _PRONOUNS:
        return True

    # 2. Pronoun anywhere followed by a verb or finite verb pattern
    for pattern in _FINITE_VERB_PATTERNS:
        if pattern.search(s):
            return True

    # 3. Starts with common finite past-tense / auxiliary / reporting verbs
    if words[0] in {"had", "have", "has", "was", "were", "did", "could", "should", "would", "is", "are", "stated", "claimed", "reported", "said"}:
        return True

    # 4. Long sentences with punctuation or subordinate connectors
    if len(words) > 12:
        return True

    return False


def validate_semantic_subject(subject: str | None) -> bool:
    """Return True if `subject` is a valid, clean noun phrase / entity / process.
    Rejects clauses, pronouns, or malformed fragments."""
    return not reject_subject_if_clause(subject)


def _strip_framing(sentence: str) -> str:
    out = sentence.strip()
    prev = None
    while out != prev:
        prev = out
        for pattern in _FRAMING_PREFIXES:
            out = pattern.sub("", out).strip()
    return out


_DECLARATIVE_AUX_RE = re.compile(
    r"^(?P<subject>.+?)\s+(?P<aux>was|were|is|are|had|has|have)\s+(?P<rest>.+)$", re.IGNORECASE
)
_DECLARATIVE_PAST_VERB_RE = re.compile(
    r"^(?P<subject>(?:an?|the)\s+[\w\s-]{1,40}?)\s+(?P<verb>[a-z]+ed)\s+(?P<rest>.+)$", re.IGNORECASE
)


def _lower_leading_word(text: str) -> str:
    """Lowercase a sentence-initial capital only when it's an ordinary
    word being moved mid-question -- never a short all-caps token (an
    acronym or a bare single-letter identifier like "X"/"H1"), where
    lowercasing would corrupt it rather than just de-capitalizing a
    sentence-start artifact. "A"/"I" are excluded from that protection --
    as single letters they're indistinguishable from an acronym by case
    alone, but as English words they are always the article/pronoun, never
    an identifier, so they must still lowercase."""
    if not text:
        return text
    m = re.match(r"^\S+", text)
    first_word = m.group(0) if m else ""
    if first_word.isupper() and first_word.lower() not in ("a", "i"):
        return text
    return text[0].lower() + text[1:]


def declarative_to_why_question(text: str) -> str:
    """Turn a declarative VERIFIED claim (e.g. "temperature monitoring
    records for refrigerator QC-REF-02 were incomplete for three
    consecutive days" or "An employee performed an activity covered by a revised procedure")
    into a natural "Why <aux> <subject> <rest>?" or "Why did <subject> <verb> <rest>?" question.
    """
    clause = _strip_framing(text).strip().rstrip(".")
    if not clause:
        return "Why did this occur?"
    m = _DECLARATIVE_AUX_RE.match(clause)
    if m:
        subject = m.group("subject").strip()
        aux = m.group("aux")
        rest = m.group("rest").strip()
        subject_lc = _lower_leading_word(subject) if subject else subject
        return f"Why {aux} {subject_lc} {rest}?"
    m_past = _DECLARATIVE_PAST_VERB_RE.match(clause)
    if m_past:
        subject = m_past.group("subject").strip()
        verb = m_past.group("verb").lower()
        rest = m_past.group("rest").strip()
        subject_lc = _lower_leading_word(subject) if subject else subject
        # Convert past tense verb to base form if simple rule applies
        base_verb = verb[:-1] if verb.endswith("ed") and not verb.endswith("eed") else verb
        if verb.endswith("ied"):
            base_verb = verb[:-3] + "y"
        elif verb.endswith("ed") and len(verb) > 4 and verb[-3] == verb[-4]:
            base_verb = verb[:-3]
        elif verb.endswith("ed"):
            base_verb = verb[:-2] if verb.endswith(("ted", "ded", "ked", "med", "ped", "red", "led")) and not verb.endswith(("ated", "ited", "oted", "uted")) else verb[:-1]
        # Common verb lookup
        _IRREG_OR_STD = {
            "performed": "perform", "completed": "complete", "conducted": "conduct",
            "executed": "execute", "carried": "carry", "operated": "operate",
            "processed": "process", "used": "use", "modified": "modify",
            "changed": "change", "accessed": "access", "received": "receive",
            "entered": "enter", "signed": "sign", "approved": "approve",
        }
        infinitive = _IRREG_OR_STD.get(verb, base_verb)
        return f"Why did {subject_lc} {infinitive} {rest}?"
    return f"Why {_lower_leading_word(clause)}?"


_PLURAL_SUBJECT_TAIL_RE = re.compile(
    r"\b(?:records?|logs?|entries|checks?|inspections?|reports?|certificates?|documents?|"
    r"attendance\s+sheets?|results?)\b\s*$",
    re.IGNORECASE,
)

# Closed vocabulary of condition ADJECTIVES/participles that follow "was/were"
# as a predicate (never as a verb an active-voice question could be built
# around) -- the same words the "adj_state" entry in _CONDITION_PATTERNS
# recognizes on the extraction side. Kept here as the single source both
# sides read, so format_deviation_why_question can tell "incomplete" (stays
# "was incomplete") apart from a bare verb infinitive like "distribute" (an
# object of a "failed to <verb>" match, needing active "did not <verb>"
# phrasing) even when upstream extraction has already stripped any leading
# "was/were" from the condition string by the time it gets here.
_CONDITION_ADJECTIVES = frozenset({
    "incomplete", "missing", "unavailable", "overdue", "expired", "nonconforming",
    "inaccurate", "inadequate", "outdated", "unverified", "missed", "disabled",
    "deactivated", "bypassed", "overridden", "blank", "undocumented", "unrecorded",
    "unresolved", "pending", "invalid", "absent", "insufficient", "unclear",
    "incorrect", "unconfirmed", "unassigned",
})

# Closed whitelist of infinitive-form verbs that legitimately signal an
# OMITTED-action verb phrase (captured from a "failed to <verb> <object>"
# source pattern, e.g. "distribute the revised SOP"). A bare leading-word
# match ("^[a-z]+\s+\S") is NOT enough on its own to classify a condition as
# an omitted action -- it also matches an already-inflected past-tense
# predicate describing something that DID happen ("processed the
# transaction twice"), which needs the opposite grammatical treatment.
# Shared by format_deviation_why_question below and
# app.agent.nodes.core_synthesis._reportedly_clause, which has the same
# classification problem and must agree on which conditions are which shape.
_TRANSITIVE_FAILED_TO_VERBS = frozenset({
    "complete", "perform", "execute", "conduct", "record", "log", "receive",
    "distribute", "notify", "deliver", "dispatch", "transmit", "forward",
    "issue", "submit", "verify", "approve", "review", "document", "report",
    "escalate", "sign", "file", "update", "process",
})


# Generalized comparison-type -> verb-phrase table (Section 7/9): a single
# source of truth both the 5-Why question builder and the impact-text
# renderer read, so adding a new comparison TYPE only means adding one row
# here rather than touching every place comparison grammar is generated.
_COMPARISON_VERB_PHRASE = {
    "MISMATCH": "differ from",
    "EXCEEDED": "exceed",
    "BELOW": "fall below",
    "INCONSISTENT": "be inconsistent with",
    "RECONCILIATION_FAILURE": "fail to reconcile with",
    "MISSING": "be missing from",
    "DUPLICATE": "duplicate",
}
_NUMBER_WORD = r"(?:\d+|two|three|four|five|six|seven|eight|nine|ten|multiple|several|numerous)"

_COMPARISON_VERB_PAST = {
    "MISMATCH": "differed from",
    "EXCEEDED": "exceeded",
    "BELOW": "was below",
    "INCONSISTENT": "was inconsistent with",
    "RECONCILIATION_FAILURE": "did not reconcile with",
    "MISSING": "was missing from",
    "DUPLICATE": "duplicated",
}

# Mechanism-category vocabulary a 5-Why causal-boundary answer lists as
# still-open possibilities, keyed by comparison SUBTYPE (Section 1/4) --
# each list uses only vocabulary relevant to that subtype's own domain
# ("calculation"/"formula" for a calculation mismatch, never for a
# parameter mismatch, and vice versa), so the boundary answer never
# introduces an unsupported-specificity claim the finding itself never
# raised.
# Only subtypes with their OWN dedicated hypothesis-generation branch (in
# plan_investigation_fallback.py) get their own entry here -- the 5-Why
# answer's mechanism-category vocabulary must always match the vocabulary
# of the hypotheses actually generated for this finding, or the causal-
# boundary guard (answer_selects_unverified_hypothesis) can spuriously
# fire on a coincidental word overlap between an unrelated subtype's
# category list and the generic calculation-shaped hypothesis names.
# Subtypes without their own branch (still routed through the generic
# calculation-shaped tree) intentionally fall through to the default.
COMPARISON_SUBTYPE_MECHANISM_CATEGORIES: dict[str, str] = {
    "CALCULATION_MISMATCH": "data entry, calculation, source-entry, formula/version",
    "PARAMETER_MISMATCH": "data entry, transcription, parameter configuration, revision mismatch, process execution",
}
_DEFAULT_COMPARISON_MECHANISM_CATEGORIES = "data entry, calculation, source-entry, formula/version"


def format_comparison_why_question(
    comparison_type: str | None, left: str | None, right: str | None, left_qualifier: str | None = None
) -> str | None:
    """Build "Why did <left> <verb> <right>?" directly from the canonical
    comparison event -- e.g. "Why did the recorded final yield differ from
    the calculated yield?" -- instead of routing a comparison finding
    through the generic subject/condition "Why was X not matched?"
    template, which reads as a category error (the comparison itself is
    VERIFIED; only its cause is in question)."""
    if not comparison_type or not left or not right:
        return None
    verb = _COMPARISON_VERB_PHRASE.get(comparison_type)
    if not verb:
        return None
    left_lc = left[0].lower() + left[1:] if left else left
    if left_qualifier:
        left_lc = f"{left_qualifier} {left_lc}"
    if comparison_type == "BELOW":
        return f"Why was the {left_lc} below the {right}?"
    return f"Why did the {left_lc} {verb} the {right}?"


def render_comparison_sentence(
    comparison_type: str | None,
    left: str | None,
    right: str | None,
    measurement_value: float | None = None,
    measurement_unit: str | None = None,
    measurement_qualifier: str | None = None,
    left_qualifier: str | None = None,
) -> str | None:
    """Render a grammatically correct, VERIFIED-observation sentence
    directly from the canonical comparison event (Section 6/7) -- e.g.
    "The recorded final yield differed from the calculated yield by
    approximately 4.2%." -- rather than the generic (and passive/incorrect
    for a VERIFIED observation) "X was reportedly not matched" template."""
    if not comparison_type or not left or not right:
        return None
    verb_past = _COMPARISON_VERB_PAST.get(comparison_type, "differed from")
    # "The " is always prepended below, so left_full must stay lowercase
    # (not Title-cased) regardless of whether a qualifier is present --
    # "The recorded final yield", never "The Recorded final yield".
    left_full = f"{left_qualifier} {left}" if left_qualifier else left
    left_cap = left_full[0].lower() + left_full[1:] if left_full else left_full
    magnitude = ""
    if measurement_value is not None:
        qual = f"{measurement_qualifier} " if measurement_qualifier else ""
        unit = measurement_unit or ""
        magnitude = f" by {qual}{measurement_value}{unit}"
    if comparison_type == "BELOW":
        return f"The {left_cap} was below the {right}{magnitude}."
    return f"The {left_cap} {verb_past} the {right}{magnitude}."


def format_deviation_why_question(
    subject: str | None, condition: str | None = None, temporal: str | None = None
) -> str:
    """Deterministically build a grammatically correct "Why was/were
    <subject> <condition> [<temporal>]?" question from canonical
    subject/condition/temporal fields.
    """
    raw_subj = (subject or "").strip()
    temporal_suffix = f" {temporal.strip()}" if temporal and temporal.strip() else ""
    if not raw_subj:
        return "Why did this deviation occur?"

    cond = (condition or "").strip()
    # A condition that IS a quantity/amount descriptor ("approximately ₹4
    # lakh of rework costs...") is not a predicate this template can attach
    # to a subject at all -- "Why was X approximately ₹4 lakh...?" is not a
    # question anyone would ask; the actual deviation is what caused that
    # cost, not the amount itself. Fall back to the generic deviation
    # question rather than forcing a quantity into an adjective/verb slot.
    if re.match(r"^(?:approximately|about|roughly|nearly|₹|\$|€|£|\d)", cond, re.IGNORECASE):
        raw_subj_cap = raw_subj[0].upper() + raw_subj[1:] if raw_subj else raw_subj
        subj_phrase = raw_subj_cap if re.match(r"^(?:the|a|an|this|that)\b", raw_subj, re.IGNORECASE) else f"The {raw_subj}"
        return f"Why did {subj_phrase[0].lower()}{subj_phrase[1:]} occur{temporal_suffix}?"
    cond_aux_match = re.match(r"^(?:was|were)\s+(.+)$", cond, re.IGNORECASE)
    # A condition that ALREADY starts with "was/were" (e.g. "was incomplete",
    # "was missing") is an adjective/participle PREDICATE -- "was" belongs
    # with it grammatically ("Why was the checklist incomplete?") and must
    # be put back rather than treated as a strippable verb-infinitive
    # marker. Only a condition with NO leading aux at all (e.g. "distribute
    # the revised SOP", captured from a "failed to <verb>" source pattern)
    # is an active-voice verb phrase needing the "did ... not <verb>"
    # treatment below.
    had_leading_aux = bool(cond_aux_match)
    if cond_aux_match:
        cond = cond_aux_match.group(1)

    if is_actor_noun(raw_subj):
        stripped_actor = strip_leading_article(raw_subj).lower()
        if not cond or cond.upper() == "UNKNOWN":
            return f"Why did the {stripped_actor} not complete the required activity{temporal_suffix}?"
        if cond.startswith("not "):
            return f"Why did the {stripped_actor} not {cond[4:].strip()}{temporal_suffix}?"
        return f"Why did the {stripped_actor} {cond}{temporal_suffix}?"

    if raw_subj and not re.match(r"^(?:the|a|an|my|your|his|her|its|our|their|this|that)\b", raw_subj, re.IGNORECASE):
        subj = f"the {raw_subj}"
    else:
        subj = raw_subj

    if not cond or cond.upper() == "UNKNOWN":
        return f"Why did {subj[0].lower()}{subj[1:]} deviate from the applicable requirement{temporal_suffix}?"

    aux = "were" if _PLURAL_SUBJECT_TAIL_RE.search(subj) else "was"
    # A condition already shaped as a negated/passive predicate -- either it
    # HAD a leading "was/were" stripped above (an adjective/participle like
    # "incomplete"/"missing", which "was/were" belongs directly in front of)
    # or it already starts with "not " (an already-negated participle like
    # "not completed"/"not calibrated", which likewise just needs "was/were"
    # put back in front) -- must NOT be routed through the active-voice
    # verb-phrase branch below; it renders correctly through the plain
    # "was/were <cond>" fallback at the bottom of this function.
    cond_first_word = cond.split()[0].lower() if cond else ""
    cond_is_negated_predicate = (
        had_leading_aux or cond.lower().startswith("not ") or cond_first_word in _CONDITION_ADJECTIVES
    )
    # Only a condition with NO leading aux, no leading "not", a RECOGNIZED
    # transitive verb as its first word, AND a following object (e.g.
    # "distribute the revised SOP", captured from a "failed to <verb>
    # <object>" source pattern) is an active-voice VERB PHRASE that cannot
    # follow "was/were" as-is -- "Why was the document-control system
    # distribute the SOP?" is ungrammatical.
    #
    # Deliberately a whitelist, not "any leading word": a bare single-word
    # condition (no object) is usually already a STATE verb describing the
    # deviation itself ("failed", "malfunctioned", "crashed") where "not"
    # would invert the meaning ("did X not fail?" asks the opposite
    # question), or a noun/adjective this module doesn't yet recognize
    # (e.g. "duplicate transaction processed") -- both fall through to the
    # plain "was/were <cond>" fallback below instead, which was already the
    # established, safe rendering for every condition shape not explicitly
    # handled here.
    verb_match = (
        re.match(r"^([a-z]+)\s+\S", cond, re.IGNORECASE)
        if not cond_is_negated_predicate else None
    )
    if verb_match and verb_match.group(1).lower() not in _TRANSITIVE_FAILED_TO_VERBS:
        verb_match = None
    if verb_match:
        v = verb_match.group(1).lower()
        if v == "complete":
            return f"Why {aux} {subj[0].lower()}{subj[1:]} not completed{temporal_suffix}?"
        elif v == "perform":
            return f"Why {aux} {subj[0].lower()}{subj[1:]} not performed{temporal_suffix}?"
        elif v == "receive":
            return f"Why {aux} {subj[0].lower()}{subj[1:]} not received{temporal_suffix}?"
        elif v in ("execute", "conduct", "record", "log"):
            return f"Why {aux} {subj[0].lower()}{subj[1:]} not {v}ed{temporal_suffix}?"
        return f"Why did {subj[0].lower()}{subj[1:]} not {v}{temporal_suffix}?"

    return f"Why {aux} {subj[0].lower()}{subj[1:]} {cond}{temporal_suffix}?"


_SPEAKER_CLAIM_RE = re.compile(r"^([A-Za-z][\w\s]{0,40}?):\s*(.+)$")
_WAS_WERE_LEAD_RE = re.compile(r"^(?:was|were)\s+(.+)$", re.IGNORECASE)
_HAD_LEAD_RE = re.compile(r"^had\s+(.+)$", re.IGNORECASE)
_DID_LEAD_RE = re.compile(r"^did\s+not\s+(.+)$", re.IGNORECASE)


def naturalize_reported_claim(text: str | None) -> str | None:
    """Convert a raw internal "speaker: claim" evidence-ledger fragment
    (e.g. "operator: were unaware that the checklist procedure had been
    revised") into a natural-language sentence ("The operator reported
    being unaware that the checklist procedure had been revised.") suitable
    for direct display as a 5-Why answer or mechanism statement.

    The "speaker: claim" format is an internal representation built when
    attributed statements are added to the evidence ledger -- it must never
    leak into report-facing text verbatim (Problem 9: "operator: were
    unaware..." is not a sentence). No-op (returns input unchanged) when
    the text doesn't match the speaker-prefixed shape, so this is always
    safe to apply defensively wherever a reported claim might reach a
    user-facing field.
    """
    if not text:
        return text
    m = _SPEAKER_CLAIM_RE.match(text.strip())
    if not m:
        return text
    speaker, claim = m.group(1).strip(), m.group(2).strip()
    if not speaker or not claim:
        return text
    speaker_phrase = (strip_leading_article(speaker) or speaker).strip().lower()
    for lead_re, gerund in ((_WAS_WERE_LEAD_RE, "being"), (_HAD_LEAD_RE, "having")):
        m2 = lead_re.match(claim)
        if m2:
            return f"The {speaker_phrase} reported {gerund} {m2.group(1)}.".replace("..", ".")
    m3 = _DID_LEAD_RE.match(claim)
    if m3:
        return f"The {speaker_phrase} reported not {m3.group(1)}.".replace("..", ".")
    return f"The {speaker_phrase} reported that {claim}.".replace("..", ".")


_QUANTITY_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "twenty", "thirty", "forty", "fifty",
    "hundred", "thousand", "first", "second", "third", "fourth", "fifth",
    "multiple", "several", "numerous", "various", "many", "few", "both", "each", "every",
    "some", "any", "no", "none", "all", "total", "count", "number", "numbers",
    "item", "items", "piece", "pieces", "set", "sets", "batch", "batches",
}

_QUANTITY_PREFIX_RE = re.compile(
    r"^(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"multiple|several|numerous|various|many|few|some)\s+",
    re.IGNORECASE,
)


def strip_quantity_prefix(text: str | None) -> str | None:
    """Strip leading quantity numbers or number-words (e.g. 'three ', '3 ', 'multiple ')
    from a subject or noun phrase so incidental quantities never contaminate process,
    topic, or entity names."""
    if not text:
        return text
    return _QUANTITY_PREFIX_RE.sub("", text.strip()).strip() or text.strip()


def _clean_subject(raw: str) -> str:
    s = raw.strip().strip("\"'").strip()
    s = re.sub(r"^(?:that|which|who|from|of|in|to)\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(
        r"^(?:audit\s+trail\s+(?:proves|shows|confirms|establishes)|"
        r"scada\s+system\s+logs?\s+(?:establish|proves|shows|confirms)|"
        r"server\s+(?:error\s+)?logs?\s+(?:establish|proves|shows|confirms)|"
        r"dispatch\s+system\s+logs?\s+(?:confirms|shows|establishes))\s+(?:that\s+)?",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    # Same self-referential-evidence firewall applied at the sentence level in
    # _strip_framing -- a captured subject group can still carry this prefix
    # when the matched pattern fell back to searching the UNSTRIPPED sentence
    # (see extract_semantic_subject's `pattern.search(stripped) or
    # pattern.search(sentence)` fallback), so this is defense-in-depth, not
    # dead code.
    s = _SELF_REFERENTIAL_EVIDENCE_PREFIX_RE.sub("", s).strip()
    s = _ACCORDING_TO_EVIDENCE_PREFIX_RE.sub("", s).strip()
    s = re.sub(r"^(?:a|an|the)\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^(?:from|of|in|to)\s+", "", s, flags=re.IGNORECASE).strip()
    s = _QUANTITY_PREFIX_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s).strip(" ,.;:")

    # Strip an embedded date range/date substring: temporal scope belongs in
    # DeviationInfo.date (already captured separately via extract_date() at
    # the top of extract_semantic_subject and threaded into every
    # DeviationInfo returned below), never duplicated into the entity/subject
    # string itself -- e.g. "checklist between 7 and 9 August" -> "checklist".
    m_date = _DATE_RANGE_RE.search(s) or _DATE_SINGLE_RE.search(s)
    if m_date:
        s = (s[: m_date.start()] + " " + s[m_date.end():]).strip()

    # Drop dangling connective words a stripped clause leaves behind
    # ("checklist for" / "checklist between" / "checklist and").
    s = re.sub(r"\s+", " ", s).strip(" ,.;:")
    s = re.sub(r"\s*\b(?:for|between|during|on|at|in|and)\s*$", "", s, flags=re.IGNORECASE).strip(" ,.;:")
    return s


# ---------------------------------------------------------------------------
# Entities, Dates, Actors
# ---------------------------------------------------------------------------
_ENTITY_RE = re.compile(
    # The bare "PREFIX-SUFFIX" alternative requires a DIGIT somewhere in
    # the match -- without it, re.IGNORECASE (needed for the other
    # alternatives below) makes `[A-Z]{2,5}-[A-Z0-9-]+` match any ordinary
    # hyphenated word pair ("in-process", "on-site", "well-documented") as
    # though it were an equipment/batch code, corrupting subject extraction
    # for any finding using such wording. A real code (EQ-104, BR-2026-0900,
    # SOP-ENG-002, MBR-4471) always contains a digit; a plain adjective
    # compound never does.
    #
    # F6 — Structural identifier validation: the same digit-requirement
    # guard applied to the bare PREFIX-SUFFIX alternative is now also applied
    # uniformly to Lot/Batch alternatives. A real lot/batch code always
    # contains a digit (Lot ABC-2024-001, Batch B102, Batch BR-2026-0815).
    # A generic English noun following the label word ("batch record", "lot
    # file", "batch process") never does -- without this guard those phrases
    # were falsely extracted as entity IDs, corrupting the subject field.
    r"\b([A-Z]{2,5}-[A-Z0-9-]*\d[A-Z0-9-]*"
    r"|Lot\s+[A-Z0-9-]*\d[A-Z0-9-]*"
    r"|Batch\s+[A-Z0-9-]*\d[A-Z0-9-]*"
    r"|Line\s+\d+|Room\s+\d+|Cleanroom\s+Suite\s+\d+|Suite\s+\d+|"
    r"Cleanroom\s+[A-Za-z0-9\s]+|Autoclave\s+#?\d+|AHU-\d+|CR-\d+|LF-\d+|VI-\d+|CP-\d+|PP-\d+|"
    r"Lyo-\d+|FH-\d+|SP-\d+|BAL-\d+|W-\d+|NC-\d+-\d+|CAPA-\d+-\d+|BRD-\d+|MBR-[A-Z0-9-]+|"
    r"WSC-\d+|API-[0-9]+|RM-[0-9]+|QC-REF-\d+|EQ-\d+|CH-\d+|PM-\d+|ECO-\d+|PO-\d+|PV-\d+|"
    r"WT-\d+|MIC-\d+|TC-\d+|FW-\d+|IP-\d+|PCB-\d+|INV-\d+|MSA-\d+|SCH-\d+|CF-\d+|R-\d+|db-[a-z0-9-]+|svc_[a-z0-9_]+)\b",
    re.IGNORECASE,
)

_MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"

_DATE_RANGE_RE = re.compile(
    rf"\b(?:between\s+(\d{{1,2}}(?:st|nd|rd|th)?)\s+(?:and|to|–|-)\s+(\d{{1,2}}(?:st|nd|rd|th)?)\s+({_MONTHS})(?:\s+(\d{{4}}))?|"
    rf"between\s+({_MONTHS})\s+(\d{{1,2}}(?:st|nd|rd|th)?)\s+(?:and|to|–|-)\s+(?:({_MONTHS})\s+)?(\d{{1,2}}(?:st|nd|rd|th)?)(?:\s+(\d{{4}}))?|"
    rf"(\d{{1,2}}(?:st|nd|rd|th)?)\s*(?:–|-|to|and)\s*(\d{{1,2}}(?:st|nd|rd|th)?)\s+({_MONTHS})(?:\s+(\d{{4}}))?|"
    rf"({_MONTHS})\s+(\d{{1,2}}(?:st|nd|rd|th)?)\s*(?:–|-|to|and)\s*(\d{{1,2}}(?:st|nd|rd|th)?)(?:,?\s+(\d{{4}}))?)\b",
    re.IGNORECASE,
)

_DATE_SINGLE_RE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}(?:\s+\d{{4}})?|{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|"
    rf"\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}}|"
    rf"(?:in|during)\s+{_MONTHS}(?:\s+\d{{4}})?)\b",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}\s+{_MONTHS}\s+\d{{4}}|{_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}|"
    rf"\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}})\b"
)
_ACTOR_RE = re.compile(
    r"\b(?:the\s+)?(?:responsible\s+)?(?:technician|operator|supervisor|analyst|manager|auditor|inspector|trainer|employee|personnel)\b",
    re.IGNORECASE,
)


def extract_entities(text: str) -> list[str]:
    """Extract explicit equipment/SOP/batch/room codes from text."""
    if not text:
        return []
    matches = _ENTITY_RE.findall(text)
    return list(dict.fromkeys(m.strip() for m in matches if m.strip()))


def extract_date(text: str) -> str | None:
    if not text:
        return None
    # 1. Match date ranges first (e.g. "between 10 and 12 August" -> "10–12 August")
    m_range = _DATE_RANGE_RE.search(text)
    if m_range:
        matched = m_range.group(0).strip()
        # Clean "between X and Y Month" -> "X–Y Month" or return cleanly
        m_between = re.match(rf"between\s+(\d{{1,2}}(?:st|nd|rd|th)?)\s+and\s+(\d{{1,2}}(?:st|nd|rd|th)?)\s+({_MONTHS})(?:\s+(\d{{4}}))?", matched, re.IGNORECASE)
        if m_between:
            d1, d2, month, yr = m_between.group(1), m_between.group(2), m_between.group(3), m_between.group(4)
            year_part = f" {yr}" if yr else ""
            return f"{d1}–{d2} {month}{year_part}"
        return matched

    # 2. Match single dates / month expressions
    m_single = _DATE_SINGLE_RE.search(text)
    if m_single:
        return m_single.group(0).strip()
    return None


# Generic "for N <noun>" population-scope clause -- deliberately built from
# the same _NUMBER_WORD used elsewhere rather than a fixed noun list (unlike
# the RECURRENCE branch's explicit batches/records/units/... list), so it
# generalizes to any domain instead of only the ones already enumerated
# there. Population scope (e.g. "for 15 employees") belongs in
# DeviationInfo.occurrence_population, never baked into the entity/subject
# string itself.
_POPULATION_CLAUSE_RE = re.compile(
    rf"\bfor\s+{_NUMBER_WORD}\s+[a-z][a-z-]*s\b",
    re.IGNORECASE,
)


def extract_population_clause(text: str) -> tuple[str | None, str]:
    """Detect and remove a generic "for N <noun...>" population-scope clause.

    Returns (population_text_or_None, text_with_clause_removed). The
    remaining text keeps the population scope out of whatever entity/subject
    string is built from it next, while the caller is responsible for
    threading the extracted population text into DeviationInfo.
    occurrence_population so it is preserved, not discarded.
    """
    if not text:
        return None, text
    m = _POPULATION_CLAUSE_RE.search(text)
    if not m:
        return None, text
    population_text = m.group(0).strip()
    remaining = (text[: m.start()] + " " + text[m.end():]).strip()
    remaining = re.sub(r"\s+", " ", remaining).strip()
    return population_text, remaining


def extract_actors(text: str) -> list[str]:
    if not text:
        return []
    matches = _ACTOR_RE.findall(text)
    return list(dict.fromkeys(m.strip() for m in matches if m.strip()))


_RELATIVE_TIME_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:consecutive\s+)?(?:day|days|week|weeks|month|months|shift|shifts)\b",
    re.IGNORECASE,
)


_DEGRADED_CONDITIONS = {None, "", "UNKNOWN", "status unconfirmed", "condition unconfirmed", "unconfirmed"}


def _degraded_condition_filler(subject: str | None) -> str:
    """Pick the generic 'degraded' condition filler ("status unconfirmed" /
    "condition unconfirmed") without repeating a noun the subject already
    ends with (e.g. subject "equipment calibration status" must not get
    "status unconfirmed" appended, which would read "status status
    unconfirmed" once joined) -- generalizes to any subject ending in
    "status" or "condition", not just calibration-specific phrasing."""
    last_word = re.findall(r"[a-zA-Z]+", subject or "")
    last_word = last_word[-1].lower() if last_word else ""
    if last_word in ("status", "condition"):
        return "unconfirmed"
    return "status unconfirmed"


def classify_finding_specificity(
    finding_text: str,
    reported_claims: list[str] | None = None,
    mechanism_status: str | None = None,
    deviation_condition: str | None = None,
) -> str:
    """Deterministic finding-specificity classification (HIGH/MEDIUM/LOW),
    structural only -- never tied to a domain word or a specific evaluated
    finding.

    A finding is only LOW specificity when it has NONE of: a specific
    entity/equipment/document identifier, a date or relative time period, a
    reported/attributed statement, an already-established immediate
    mechanism, or a structurally-captured deviation condition (e.g. "not
    completed", "operated outside its validated range" -- a concrete stated
    condition, not the degraded "status/condition unconfirmed" filler).
    That combination is exactly what a generic allegation like "the
    department is not following the required procedure correctly" looks
    like structurally -- no object, no period, no account, no mechanism, no
    stated condition -- versus a finding that names an affected object, a
    date, quotes someone, or states a concrete deviation condition (e.g.
    "the equipment was operated outside its validated range" -- clear even
    with no entity ID, date, or attributed statement).

    Used to gate hypothesis generation: a LOW-specificity finding must not
    receive fabricated, evidence-free causal hypotheses (Section 29) -- the
    correct output is NOT_ESTABLISHED plus a list of what's missing, not a
    guess dressed up as analysis.
    """
    text = finding_text or ""
    has_entity = bool(_ENTITY_RE.search(text))
    has_date_or_period = bool(
        _DATE_RE.search(text)
        or _DATE_SINGLE_RE.search(text)
        or _DATE_RANGE_RE.search(text)
        or _RELATIVE_TIME_RE.search(text)
    )
    has_reported = bool(reported_claims)
    has_mechanism = bool(mechanism_status) and mechanism_status not in ("UNKNOWN", "NONE")
    has_condition = bool(deviation_condition) and deviation_condition.strip() not in _DEGRADED_CONDITIONS
    has_financial_signal = bool(re.search(r"₹|\$|€|£|INR|USD|EUR|duplicate\s+payment|overpayment|batch\s+worth", text, re.IGNORECASE))
    has_system_record = bool(re.search(r"\b(?:server|system|audit|database|lms|calibration|distribution|maintenance|inspection|telemetry)\s+(?:logs?|records?|trail|history)\b|\b\d{1,2}:\d{2}\b", text, re.IGNORECASE))

    concrete_signals = sum([has_entity, has_date_or_period, has_reported, has_mechanism, has_condition, has_financial_signal, has_system_record])
    if concrete_signals == 0:
        return "LOW"
    if concrete_signals >= 2:
        return "HIGH"
    return "MEDIUM"


_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def strip_leading_article(text: str | None) -> str | None:
    """Strip a leading "the"/"a"/"an" so an extracted actor/subject phrase
    (which may have been captured mid-sentence, e.g. "The operator" at the
    start of a source sentence) can be re-embedded cleanly into a new
    sentence without carrying stray capitalization or a doubled article
    (e.g. "the the operator")."""
    if not text:
        return text
    return _LEADING_ARTICLE_RE.sub("", text).strip() or None


# A temporal clause already present in the finding text (e.g. "before the
# procedure became effective", "prior to the audit date") — structural only,
# no finding-specific vocabulary. Used so affected_period never discards
# temporal information the finding already states just because it isn't a
# calendar date.
_TEMPORAL_CLAUSE_RE = re.compile(
    r"\b(before\s+(?:the\s+)?[\w\s-]{1,60}?\s+(?:became|becomes|was|is|took)\s+[\w\s-]{1,30}?|"
    r"prior\s+to\s+[\w\s-]{1,60}?|"
    r"since\s+[\w\s-]{1,60}?|"
    r"after\s+(?:the\s+)?[\w\s-]{1,60}?\s+(?:became|becomes|was|is)\s+[\w\s-]{1,30}?)"
    r"(?=[.,;]|\s+(?:and|but|however)\b|$)",
    re.IGNORECASE,
)
# Bounded-duration expressions actually stated in the finding (never
# fabricated): "for three consecutive days", "during the morning shift",
# etc. -- a distinct shape from the relative-clause patterns above (those
# anchor to another EVENT, e.g. "before the procedure became effective";
# these anchor to a stated DURATION or named period). Checked in priority
# order below -- "during the audit" is a fallback, not preferred, since it
# describes when the finding was NOTICED, not the period the deviation
# itself actually spans (e.g. "for three consecutive days" is the
# meaningful affected period; "during the audit" is just framing).
_TEMPORAL_DURATION_RE = re.compile(
    r"\bfor\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"consecutive\s+(?:day|days|week|weeks|month|months|shift|shifts)\b|"
    r"\bduring\s+the\s+(?:morning|afternoon|evening|night|day)\s+shift\b",
    re.IGNORECASE,
)
_TEMPORAL_DURING_AUDIT_RE = re.compile(
    r"\bduring\s+the\s+(?:current\s+)?audit(?:\s+period)?\b", re.IGNORECASE,
)


def extract_detected_period(text: str) -> str | None:
    """Extract audit detection framing period (e.g. 'during the audit')."""
    if not text:
        return None
    m = _TEMPORAL_DURING_AUDIT_RE.search(text)
    if m:
        return m.group(0).strip().rstrip(".,;")
    return None


def extract_temporal_clause(text: str) -> str | None:
    """Extract a relative temporal clause or stated duration already
    present in the finding (e.g. 'before the procedure became effective',
    'for three consecutive days') when no absolute date is present. Never
    fabricates a date/period — returns None if nothing is stated."""
    if not text:
        return None
    m = _TEMPORAL_CLAUSE_RE.search(text)
    if m:
        clause = m.group(1).strip().rstrip(".,;")
        if clause:
            return clause
    m2 = _TEMPORAL_DURATION_RE.search(text)
    if m2:
        return m2.group(0).strip().rstrip(".,;")
    return None


_QUANTITY_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "twenty", "thirty", "forty", "fifty",
    "hundred", "thousand", "first", "second", "third", "fourth", "fifth",
    "multiple", "several", "numerous", "various", "many", "few", "both", "each", "every",
    "some", "any", "no", "none", "all", "total", "count", "number", "numbers",
    "item", "items", "piece", "pieces", "set", "sets", "batch", "batches",
}

# Generic stopwords stripped when deriving a short topic word from a subject
# phrase — deliberately domain-agnostic so this works for training,
# calibration, checklist, temperature-log, maintenance, documentation,
# inspection, communication, or any other QMS subject noun phrase.
_TOPIC_STOPWORDS = {
    "the", "a", "an", "for", "of", "on", "in", "to", "with", "status",
    "compliance", "record", "records", "log", "logs", "entry", "entries",
    "document", "documents", "documentation", "file", "files", "sheet", "sheets",
    *_QUANTITY_WORDS,
}


def strip_quantity_prefix(subject: str | None) -> str | None:
    if not subject:
        return subject
    return _QUANTITY_PREFIX_RE.sub("", subject.strip()).strip() or subject.strip()


def topic_word(subject: str | None) -> str:
    """Derive a short, lower-case topic word (e.g. "training", "calibration",
    "documentation") from a resolved subject phrase, for use in dynamically
    naming hypotheses/evidence instead of a hardcoded domain vocabulary.
    Falls back to "process" when no usable word is found."""
    if not subject:
        return "process"
    clean = strip_quantity_prefix(subject) or subject
    for word in re.findall(r"[A-Za-z]+", clean):
        low = word.lower()
        if low not in _TOPIC_STOPWORDS and len(low) > 2:
            return low
    return "process"


# Captures the concrete noun immediately governed by a negation trigger
# (e.g. "had not received retraining" -> "retraining", "did not attend the training" -> "training").
# Deliberately handles optional articles ("the", "a", "an") and base verbs.
_NEGATION_OBJECT_RE = re.compile(
    r"\bnot\s+(?:receive|received|complete|completed|perform|performed|conduct|conducted|do|done|provide|provided|give|given|attend|attended)\s+"
    r"(?:the\s+|an?\s+)?([a-z][\w-]*)",
    re.IGNORECASE,
)


def extract_conflict_topic(claim_a: str | None, claim_b: str | None, fallback_subject: str | None = None) -> str:
    """Derive the topic word for a DETECTED CONFLICT from the conflicting
    claims themselves, rather than from the finding's overall subject.

    This matters when a finding's overall deviation is about one thing
    (e.g. "temperature monitoring records were incomplete") while a
    conflict embedded in the SAME finding is about a completely different
    proposition (e.g. whether retraining occurred) -- using the finding-
    level topic_word() for hypothesis naming in that case produces
    nonsensical mechanisms like "TEMPERATURE_NOT_COMPLETED" for what is
    actually a training question. Tries, in order:
      1. The noun governed by an explicit negation ("had not received
         retraining" -> "retraining") in either claim -- this is usually
         the clearest signal of what specific thing is under dispute.
      2. The significant words the two claims share (the proposition's
         "aboutness"), filtering actor nouns, via topic_word on that shared vocabulary.
      3. The finding's overall subject, only as a last resort.
    """
    for text in (claim_a, claim_b):
        if not text:
            continue
        m = _NEGATION_OBJECT_RE.search(text)
        if m:
            candidate = m.group(1).lower()
            if candidate not in _TOPIC_STOPWORDS and not is_actor_noun(candidate) and len(candidate) > 2:
                return candidate
    if claim_a and claim_b:
        from app.services.text_grounding import significant_words
        ignore = {
            "stated", "claimed", "reported", "said", "operator", "supervisor",
            "auditor", "technician", "manager", "employee", "personnel", "staff",
            "trainer", "analyst", "inspector", *_QUANTITY_WORDS,
        }
        shared = {
            w for w in (significant_words(claim_a) & significant_words(claim_b)) - ignore
            if not is_actor_noun(w)
        }
        if shared:
            return topic_word(" ".join(sorted(shared)))
    return topic_word(fallback_subject)


_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}


def extract_incidental_quantity(text: str) -> int | None:
    """Extract an explicit record/item count (e.g. 'three temperature records' -> 3)
    so quantity is tracked cleanly in metadata without leaking into entity names."""
    if not text:
        return None
    m = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
        r"(?:[a-z-]+\s+)?(?:records?|logs?|entries|checks?|inspections?|items?|sheets?|samples?)\b",
        text,
        re.IGNORECASE,
    )
    if m:
        val = m.group(1).lower()
        if val.isdigit():
            return int(val)
        return _WORD_TO_NUM.get(val)
    return None


def split_topic_and_tail(subject: str | None, topic: str) -> str | None:
    """Strip a leading "<topic> [status/compliance] (for|of) " prefix from a
    resolved subject phrase, returning the remainder (e.g. "training for the
    revised procedure" + topic "training" -> "the revised procedure").

    This is what lets downstream wording say "<role> <topic> status for
    <tail>" (e.g. "Operator training status for the revised procedure")
    instead of concatenating the whole subject phrase a second time onto an
    already-topic-prefixed label (e.g. "Operator qualification for training
    compliance for the revised procedure") -- the exact double-concatenation
    defect this function exists to prevent. Domain-agnostic: works for any
    topic word this finding's subject actually contains, not just training.
    Returns None if the subject doesn't start with the topic word at all
    (nothing safe to strip)."""
    if not subject or not topic:
        return subject
    # The optional middle word must be an actual status/compliance-type
    # word (matching this function's own docstring), never an arbitrary
    # word -- allowing ANY word here (the previous `\s+\w+`) wrongly
    # matched record-shaped subjects like "temperature log for refrigerator
    # QC-REF-02" (where "log" filled that slot), misclassifying a
    # record/document subject as an activity/qualification one and
    # producing "<role> <topic> status for <tail>" nonsense like
    # "Technician temperature status for refrigerator QC-REF-02" instead
    # of treating the subject as already a clean, specific affected object.
    pattern = re.compile(
        rf"^{re.escape(topic)}\b(?:\s+(?:status|compliance))?\s*(?:for|of)\s+", re.IGNORECASE
    )
    m = pattern.match(subject)
    if m:
        tail = subject[m.end():].strip()
        if tail:
            d = extract_date(tail)
            if d and tail.startswith(d):
                rem = tail[len(d):].strip()
                if rem.lower().startswith("for "):
                    rem = rem[4:].strip()
                tail = rem or None
        return tail or None
    return None


def build_affected_object_phrase(subject: str | None, actor: str | None = None) -> str:
    """Build the ONE canonical 'affected object' phrase from a resolved
    subject and an optional actor.

    This is the single construction used everywhere an affected-object-style
    phrase is needed (deterministic synthesis, LLM-output repair) so a
    downstream field is never built by independently re-concatenating
    semantic parts (role + topic + tail) a second, inconsistent way.

    Two subject shapes require different treatment (the same distinction
    `_derive_deterministic_impact` already makes -- reused here rather than
    re-decided a second, inconsistent way):

    - ACTIVITY/QUALIFICATION-shaped ("training for the revised procedure"):
      split_topic_and_tail finds a "<topic> for <tail>" structure, and the
      affected object IS a role/qualification status -- built from
      role + topic + tail exactly once each, e.g. "Operator training
      status for the revised procedure".

    - ENTITY-shaped ("balance BAL-014", "temperature log for refrigerator
      QC-REF-02"): the subject is ALREADY a clean, specific object with no
      "<topic> for <tail>" structure to extract. Wrapping it in the
      actor/qualification template regardless produces exactly the
      "Personnel calibration status for calibration certificate" defect --
      entity + fabricated process + fabricated status + a default actor
      the finding never named. Used directly instead, capitalized.
    """
    if not subject or subject.startswith("UNKNOWN"):
        return "NOT ESTABLISHED"
    if is_actor_noun(subject):
        return "Required procedure compliance"
    topic = topic_word(subject)
    tail = split_topic_and_tail(subject, topic)
    if not tail:
        return subject[0].upper() + subject[1:]
    stripped_actor = strip_leading_article(actor)
    actor_word = stripped_actor.capitalize() if stripped_actor else "Personnel"
    return f"{actor_word} {topic} status for {tail}"


def subject_topic_matches(candidate: str | None, canonical_subject: str | None) -> bool:
    """True if `candidate` (e.g. a proposed affected_object/process_at_risk
    string) still relates to the finding's canonical subject topic, rather
    than having semantically drifted to an unrelated concept (e.g. an
    affected_object about "procedure compliance" when the finding's actual
    topic is "training").

    A grammar-only check (validate_semantic_subject) cannot catch this: "the
    procedure compliance" is a perfectly well-formed noun phrase, just about
    the wrong thing. This requires the canonical subject's topic word to
    appear literally in the candidate -- NOT a general significant-word
    overlap, which produces false negatives here: the canonical subject
    "training for the revised procedure" and the drifted text "procedure
    compliance" both incidentally contain the word "procedure", so a plain
    overlap check would wrongly call that a match even though the actual
    topic (training vs. compliance-with-a-procedure) differs. Returns True
    (no violation) when either side is empty -- nothing to compare against.
    """
    if not candidate or not canonical_subject:
        return True
    canon_topic = topic_word(canonical_subject)
    if topic_word(candidate) == canon_topic:
        return True
    candidate_words = {w.lower() for w in re.findall(r"[A-Za-z]+", candidate)}
    return canon_topic in candidate_words


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Structured Deviation Info & Semantic Roles
# ---------------------------------------------------------------------------
class SemanticRole(str, Enum):
    ACTOR = "ACTOR"
    AFFECTED_OBJECT = "AFFECTED_OBJECT"
    ACTIVITY = "ACTIVITY"
    PROCESS = "PROCESS"
    RECORD = "RECORD"
    ASSET = "ASSET"
    REQUIREMENT = "REQUIREMENT"
    SYSTEM = "SYSTEM"
    NOTIFICATION = "NOTIFICATION"


_ACTOR_NOUNS_RE = re.compile(
    r"\b(?:employees?|operators?|technicians?|personnel|analysts?|workers?|staff|inspectors?|"
    r"supervisors?|users?|authors?|reviewers?|approvers?|managers?|chemists?|engineers?)\b",
    re.IGNORECASE,
)


def is_actor_noun(text: str | None) -> bool:
    """True if text refers to personnel / actors rather than a controlled object."""
    if not text:
        return False
    return bool(_ACTOR_NOUNS_RE.search(text))


@dataclass
class DeviationInfo:
    subject: str | None = None
    finding_subject: str | None = None
    affected_object: str | None = None
    affected_process: str | None = None
    affected_activity: str | None = None
    deviation: str | None = None
    condition: str | None = None
    requirement: str | None = None
    date: str | None = None
    actor: str | None = None
    actors: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    semantic_type: str = "OBJECT"
    relevant_change: str | None = None
    reported_mechanism: str | None = None
    verified_mechanism: str | None = None
    mechanism_status: str = "UNKNOWN"
    mechanism_polarity: str | None = None
    reported_statements: list[str] = field(default_factory=list)
    verified_observations: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    matched: bool = False
    # Canonical comparison/relational-finding semantics (Section 0c): kept
    # as distinct typed fields rather than folded back into `deviation`
    # prose, so downstream nodes (5-Why, impact rendering) can generate
    # grammatically correct, evidence-state-aware sentences directly from
    # structure instead of re-parsing text.
    comparison_type: str | None = None  # MISMATCH/EXCEEDED/BELOW/INCONSISTENT/RECONCILIATION_FAILURE
    comparison_left: str | None = None
    comparison_left_qualifier: str | None = None  # e.g. "recorded", "measured", "logged"
    comparison_right: str | None = None
    comparison_basis: str | None = None
    # Comparison SUBTYPE (Section 4/5): which investigation FRAMEWORK
    # applies -- a parameter mismatch and a calculation mismatch are both
    # comparison_type=MISMATCH but need entirely different investigation
    # questions, so this is a distinct, more specific classification.
    comparison_subtype: str | None = None
    comparison_reference_type: str | None = None  # e.g. "APPROVED_PARAMETER"
    comparison_batch_id: str | None = None
    # Missing-record/missing-documentation semantic fields (Section 2): a
    # required ACTIVITY_OR_CONTROL is reported as undocumented/missing,
    # kept distinct from any batch/record CONTEXT identifier and from any
    # DOWNSTREAM_EVENT the finding also mentions.
    missing_record_activity: str | None = None
    missing_record_context: str | None = None  # e.g. batch/record id the activity relates to
    downstream_action_text: str | None = None
    downstream_action_present: bool = False
    # Recurrence semantic fields (Section 13): the repeated DEVIATION is
    # kept distinct from the POPULATION it recurred across, and from any
    # prior-action target -- never collapsed into one opaque object.
    occurrence_population: str | None = None
    # Attributed-claim semantics (Section 2/3): the SOURCE who made a
    # statement and the causal PROPOSITION they offered are kept distinct
    # from the affected object/activity itself -- a person's explanation
    # for why an activity was omitted, never folded into the object name.
    attributed_source: str | None = None
    attributed_proposition: str | None = None
    # Event-sequence / control-point semantic fields (Section 1/3/5): a
    # controlled TRANSITION (invalidation/override/exception/waiver/...)
    # kept distinct from whether its required justification/authorization
    # was documented -- never collapsed into "control failed".
    transition_type: str | None = None  # INVALIDATION/OVERRIDE/APPROVAL/EXCEPTION/...
    control_justification_missing: bool = False
    # Uncertainty-resolution semantic fields (Section 2/3): the applicable
    # REQUIREMENT is a distinct dimension from the OBSERVATION itself --
    # an observation can be VERIFIED while the requirement governing it
    # remains UNKNOWN, and compliance/root-cause conclusions must never be
    # drawn until both are resolved.
    # F4a — requirement_status semantics:
    #   UNKNOWN  = finding gives no basis for inferring a requirement exists
    #   STATED   = finding text asserts a requirement in force (word "required",
    #              "mandatory", "obligatory", "must", or cognates), even if the
    #              specific rule document is unnamed
    #   VERIFIED = requirement is explicitly cited and confirmed
    # Only UNKNOWN should trigger the 5-Why deferral gate in core_synthesis.
    requirement_status: str = "UNKNOWN"  # UNKNOWN | STATED | VERIFIED
    observed_entity: str | None = None
    measurement_value: float | None = None
    measurement_unit: str | None = None
    measurement_qualifier: str | None = None
    # Entity-fidelity (Defect 3). The resolver must never silently substitute
    # a generic placeholder ("process compliance") for an entity it failed to
    # extract: a plausible-looking false identity is strictly worse than an
    # honest "unresolved", because every downstream node (impact,
    # investigation, CAPA) treats affected_object as ground truth.
    #   RESOLVED   -- a concrete noun phrase was isolated from the text
    #   PARTIAL    -- only a best-effort noun-phrase fragment was recovered
    #   UNRESOLVED -- nothing usable; the caller must flag, not fabricate
    extraction_confidence: str = "RESOLVED"
    subject_unresolved: bool = False
    partial_subject_fragment: str | None = None


# ---------------------------------------------------------------------------
# Condition Patterns (Structural Extraction)
# ---------------------------------------------------------------------------
_CONDITION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "identified_as",
        re.compile(
            r"(?:auditor|inspector|reviewer|assessor)\s+identified\s+(?P<subject>.+?)\s+as\s+"
            r"(?P<cond>[a-z][a-z\s]*?)\s*\.?$",
            re.IGNORECASE,
        ),
    ),
    (
        "deviated_from",
        re.compile(
            r"^(?P<subject>.+?)\s+deviated\s+from\s+(?:the\s+)?(?:requirement\s+)?(?P<requirement>.+?)\s*\.?$",
            re.IGNORECASE,
        ),
    ),
    (
        "missing_from",
        re.compile(
            r"^(?P<subject>.+?)\s+(?:was|were)\s+missing\s+from\s+(?:the\s+)?(?:equipment\s+|instrument\s+)?(?P<entity>[A-Z0-9-]+|[a-z0-9\s-]+?)\s*\.?$",
            re.IGNORECASE,
        ),
    ),
    (
        "preventing",
        re.compile(
            r"^(?P<cause>.+?)\s+(?:preventing|precluding|delaying)\s+(?:the\s+|an?\s+)?(?P<subject>.+?)\s*\.?$",
            re.IGNORECASE,
        ),
    ),
    (
        "not_state",
        re.compile(
            r"^(?P<subject>.+?)\s+(?:was|were|did)\s+not\s+(?P<cond>[a-z]+(?:-[a-z0-9]+)?(?:\s+[a-z0-9-]+)*?)"
            r"\s*(?:\bfor\b.*|\bfrom\b.*|\bon\b.*|\bfollowing\b.*|\bafter\b.*|\bduring\b.*|\bprior\b.*|\bbefore\b.*)?\.?$",
            re.IGNORECASE,
        ),
    ),
    (
        "actor_action",
        re.compile(
            r"^(?P<subject>.+?)\s+(?P<cond>(?:misread|misrecorded|miscalculated|misapplied|mishandled|incorrectly\s+[a-z]+)\s+(?:the\s+|an?\s+)?(?P<obj>[a-z0-9\s-]+?))\s*\.?$",
            re.IGNORECASE,
        ),
    ),
    (
        "adj_state",
        re.compile(
            r"^(?P<subject>.+?)\s+(?:was|were)\s+"
            r"(?P<cond>incomplete|missing|unavailable|overdue|expired|nonconforming|inaccurate|"
            r"inadequate|out of date|outdated|unverified|missed|disabled|deactivated|bypassed|overridden)\b.*$",
            re.IGNORECASE,
        ),
    ),
    (
        "contained",
        re.compile(
            r"^(?P<subject>.+?)\s+contain(?:ed|s)\s+"
            r"(?P<cond>an\s+error|errors|an\s+incomplete\s+entry|incomplete\s+entries|"
            r"a\s+discrepancy|discrepancies)\b.*$",
            re.IGNORECASE,
        ),
    ),
    (
        "rejected_due_to",
        re.compile(
            r"^(?P<subject>.+?)\s+rejected\s+(?P<cond>.+?\s+due\s+to\s+[a-z][a-z0-9\s-]*?)\s*\.?$",
            re.IGNORECASE,
        ),
    ),
    (
        "failed_to",
        re.compile(r"^(?P<subject>.+?)\s+failed\s+to\s+(?P<cond>[a-z0-9][a-z0-9\s/.,-]*?)\s*\.?$", re.IGNORECASE),
    ),
    (
        "forgot_to",
        re.compile(r"^(?P<subject>.+?)\s+(?:forgot\s+to|neglected\s+to)\s+(?P<cond>[a-z0-9][a-z0-9\s/.,-]*?)\s*\.?$", re.IGNORECASE),
    ),
    (
        "causative_verb",
        re.compile(r"^(?P<subject>.+?)\s+(?:created|caused|resulted\s+in|led\s+to|generated)\s+(?P<cond>.*)$", re.IGNORECASE),
    ),
    (
        "outside_scope",
        re.compile(
            r"^(?P<subject>.+?)\s+(?:was|were)\s+(?P<verb>operated|used|performed|conducted|run|stored|handled)"
            r"\s+outside\s+(?:its|the|their)\s+(?P<cond>[a-z][a-z\s]*?(?:range|limits?|parameters?|specification|"
            r"tolerance|threshold))\b.*$",
            re.IGNORECASE,
        ),
    ),
    (
        "missed",
        re.compile(r"^(?P<subject>.+?)\s+(?:was|were)?\s*(?:missed|skipped|omitted)\s+(?:from\s+|by\s+|in\s+)?(?:the\s+|an?\s+)?(?P<obj>[a-z0-9\s-]+?)\s*\.?$", re.IGNORECASE),
    ),
    (
        "bypassed",
        re.compile(r"^(?P<subject>.+?)\s+(?:deliberately\s+|intentionally\s+|accidentally\s+)?(?:bypassed|disabled|deactivated|overrode|overridden|defeated)\s+(?:the\s+|an?\s+)?(?P<obj>[a-z0-9\s-]+?)\s*\.?$", re.IGNORECASE),
    ),
    (
        "system_operated",
        re.compile(
            r"^(?P<subject>.+?)\s+(?:was|were)\s+(?P<verb>operated|accessed|entered|modified|executed)\s+(?P<cond>.*)$",
            re.IGNORECASE,
        ),
    ),
    (
        "metric_drift",
        re.compile(r"^(?P<subject>.+?)\s+(?P<cond>(?:increased|decreased|rose|drifted|exceeded)\b.*?)\s*\.?$", re.IGNORECASE),
    ),
    (
        "out_of_spec",
        re.compile(r"^(?P<subject>.+?)\s+(?:was|were)\s+(?P<cond>out\s+of\s+specification|out\s+of\s+spec|out\s+of\s+limits?|out\s+of\s+tolerance)\b.*$", re.IGNORECASE),
    ),
    (
        "bare_failed",
        re.compile(r"^(?P<subject>.+?)\s+failed\b.*$", re.IGNORECASE),
    ),
]

_PREFIX_CONDITION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("failure to perform", re.compile(r"^failure\s+to\s+perform\s+(?P<subject>.+?)\s*\.?$", re.IGNORECASE)),
    ("absence", re.compile(r"^absence\s+of\s+(?P<subject>.+?)\s*\.?$", re.IGNORECASE)),
    ("incomplete", re.compile(r"^(?:an?\s+)?incomplete\s+(?P<subject>.+?)\s*\.?$", re.IGNORECASE)),
    ("missing", re.compile(r"^(?:an?\s+)?missing\s+(?P<subject>.+?)\s*\.?$", re.IGNORECASE)),
    ("nonconforming", re.compile(r"^(?:an?\s+)?nonconforming\s+(?P<subject>.+?)\s*\.?$", re.IGNORECASE)),
]

_SHORT_DEVIATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "postpositive_adj",
        re.compile(
            r"^(?P<subject>[a-z0-9\s-]+?)\s+(?P<cond>missing|absent|overdue|expired|incomplete|unavailable|unrecorded|undocumented|unauthorized|nonconforming|inaccurate|inadequate|outdated|unverified|missed|disabled|deactivated|bypassed|overridden|blank|invalid|unclear|incorrect|unconfirmed|unassigned|duplicated|omitted)\s*\.?$",
            re.IGNORECASE,
        ),
    ),
    (
        "postpositive_verb_mod",
        re.compile(
            r"^(?P<subject>[a-z0-9\s-]+?)\s+(?P<verb>used|applied|modified|altered|deleted|destroyed|released|approved|distributed|bypassed|skipped|performed|executed|processed)\s+(?P<mod>(?:without|outside|prior\s+to|before|after|in\s+violation\s+of|contrary\s+to|by|to|for|with)\b.+?)\s*\.?$",
            re.IGNORECASE,
        ),
    ),
    (
        "evaluative_subj_participle",
        re.compile(
            r"^(?P<subject>(?:wrong|incorrect|invalid|unauthorized|unapproved|outdated|obsolete|expired|duplicate|excessive|uncalibrated|unqualified)\s+[a-z0-9\s-]+?)\s+(?P<cond>used|applied|modified|altered|deleted|destroyed|released|approved|distributed|bypassed|skipped|performed|executed|processed|issued|paid)\s*\.?$",
            re.IGNORECASE,
        ),
    ),
]


_ENTITY_TYPE_NOUN_RE = re.compile(
    r"\b(?:the\s+)?([a-z][a-z\s-]{1,30}?)\s+(" + _ENTITY_RE.pattern[3:-3] + r")\b",
    re.IGNORECASE,
)


def _entity_noun_phrase(text: str, entity: str) -> str | None:
    """Structural (not keyword-list) affected-object resolution: when the
    finding names a tagged entity (an equipment/room/lot code the ENTITY_RE
    pattern already recognizes), the noun immediately preceding it in the
    RAW SENTENCE -- "balance BAL-014", "refrigerator QC-REF-02", "scale
    SC-04" -- is the actual concrete object, not a generic status-label
    template. This works for any entity+preceding-noun pairing the text
    happens to use, so it generalizes across domains without a per-domain
    keyword list (unlike the "if 'calibration' in text: return ..."
    fallback this function is meant to preempt)."""
    if not text or not entity:
        return None
    for match in _ENTITY_TYPE_NOUN_RE.finditer(text):
        if match.group(2).upper() != entity.upper():
            continue
        noun_phrase = match.group(1).strip()
        _stopwords = {"for", "the", "a", "an", "of", "to", "in", "on", "at", "and", "or", "with"}
        words = noun_phrase.split()
        last_stopword_idx = max(
            (i for i, w in enumerate(words) if w.lower() in _stopwords), default=-1
        )
        words = words[last_stopword_idx + 1:] or words
        noun_phrase = " ".join(words[-2:])
        if not noun_phrase or not validate_semantic_subject(noun_phrase):
            continue
        return f"{noun_phrase} {entity}"
    return None


# ---------------------------------------------------------------------------
# Referenced-Evidence Boundary
# ---------------------------------------------------------------------------
_DOC_REFERENCE_VERB_RE = re.compile(
    r"\b(?:referenced|cited|referred\s+to|refers?\s+to)\s+(?:an?\s+|the\s+)?(?:attached\s+)?"
    r"(?P<doc>[a-z][a-z\s-]{1,50}?)\b(?=\s*(?:,|\.|;|\s+that\b|\s+which\b|\s+but\b|\s+and\b|$))",
    re.IGNORECASE,
)
_DOC_MENTIONED_RE = re.compile(
    r"\b(?:the\s+)?(?:attached\s+)?(?P<doc>[a-z][a-z\s-]{1,50}?)\s+was\s+(?:mentioned|cited|referenced)\b",
    re.IGNORECASE,
)
_DOC_UNAVAILABLE_RE = re.compile(
    r"\b(?:was|were|is|are)\s+not\s+(?:available|attached|provided|accessible)\b|"
    r"\bcould\s+not\s+be\s+(?:retrieved|accessed|reviewed|provided|attached)\b|"
    r"\b(?:was|were|is|are)\s+unavailable\b|"
    r"\bcould\s+not\s+(?:access|retrieve|review)\b|"
    r"\bnot\s+available\s+to\s+the\b",
    re.IGNORECASE,
)
_DOC_TYPE_TRAILING_STOPWORDS = {"it", "that", "which", "was", "were"}


def _clean_document_type(raw: str) -> str:
    words = [w for w in raw.strip().split() if w.lower() not in _DOC_TYPE_TRAILING_STOPWORDS]
    return " ".join(words).strip(" ,.;:")


def detect_referenced_unavailable_documents(text: str) -> list[tuple[str, str]]:
    """Return [(document_type, raw_sentence_span), ...] for every sentence
    where a document is referenced/cited/mentioned/attached AND separately
    marked as unavailable/inaccessible/not provided within that same
    sentence."""
    if not text:
        return []
    results: list[tuple[str, str]] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        sentence = sentence.strip()
        if not sentence or not _DOC_UNAVAILABLE_RE.search(sentence):
            continue
        doc_type = None
        m = _DOC_REFERENCE_VERB_RE.search(sentence)
        if m:
            doc_type = _clean_document_type(m.group("doc"))
        else:
            m2 = _DOC_MENTIONED_RE.search(sentence)
            if m2:
                doc_type = _clean_document_type(m2.group("doc"))
        if doc_type and len(doc_type) > 2:
            results.append((doc_type, sentence))
    return results


def _mask_referenced_document_spans(text: str) -> str:
    refs = detect_referenced_unavailable_documents(text)
    if not refs:
        return text
    masked = text
    for _doc_type, span in refs:
        masked = masked.replace(span, "")
    return re.sub(r"\s+", " ", masked).strip()


def _extract_activity_from_reported_finding(text: str) -> str:
    """Extract a clean activity noun phrase when finding is purely reported speech."""
    if "payment" in text.lower():
        return "vendor payment"
    if "shipment" in text.lower() or "customs" in text.lower():
        return "shipment delivery"
    if any(w in text.lower() for w in ("email", "notification", "dispatch", "message", "notice", "alert")):
        if "email" in text.lower():
            return "email notification"
        elif "dispatch" in text.lower() and ("notification" in text.lower() or "alert" in text.lower()):
            return "notification dispatch"
        return "notification delivery"
    m_train = re.search(r"\btraining\s+on\s+((?:the\s+)?[a-z0-9\s-]+?)(?:,|\.|\s+but|\s+and|$)", text, re.IGNORECASE)
    if m_train:
        topic = m_train.group(1).strip()
        return f"training for {topic}"
    if "training" in text.lower() and ("revised" in text.lower() or "procedure" in text.lower()):
        return "training for the revised procedure"
    if "checklist" in text.lower() and "equipment" in text.lower():
        return "daily equipment inspection checklist"
    if "calibration" in text.lower() and "label" in text.lower():
        return "calibration status label"
    if "training" in text.lower():
        return "personnel training status"
    if "calibration" in text.lower():
        return "equipment calibration status"
    return "process compliance"


_DISCREPANCY_RE = re.compile(
    r"\b(?:difference|discrepancy|variance|deviation)\s+(?:of\s+|was\s+)?"
    r"(?P<qual>approximately|about|roughly|nearly)?\s*"
    r"(?P<val>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|percent|degrees?\s*[cf]?|°\s*[cf])?",
    re.IGNORECASE,
)


def extract_measured_discrepancy(text: str) -> tuple[float, str | None, str | None] | None:
    """Extract a measured DISCREPANCY (a comparison's own magnitude, e.g.
    "the difference was approximately 4.2%") as a typed (value, unit,
    qualifier) triple -- kept semantically separate from any financial
    amount extraction (Section 8): this number is never itself a currency
    figure and must never be handed to cost/financial analysis."""
    if not text:
        return None
    m = _DISCREPANCY_RE.search(text)
    if not m:
        return None
    try:
        val = float(m.group("val"))
    except (TypeError, ValueError):
        return None
    unit_raw = (m.group("unit") or "").strip().lower()
    if unit_raw in ("%", "percent"):
        unit = "%"
    else:
        unit = unit_raw or None
    qualifier = (m.group("qual") or "").strip().lower() or None
    return val, unit, qualifier


# Comparison SUBTYPE classification (Section 4): which investigation
# FRAMEWORK a comparison/mismatch finding needs -- checked in priority
# order against the compared phrases' own vocabulary (never the specific
# finding's identifiers), so "calculated yield" routes to the calculation
# tree while "approved process parameter" routes to the parameter tree,
# without hardcoding to any one domain like "temperature" or "yield".
_COMPARISON_SUBTYPE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("VERSION_MISMATCH", re.compile(r"\b(?:version|revision)\b", re.IGNORECASE)),
    ("APPROVAL_MISMATCH", re.compile(r"\b(?:approv\w*)\b.*\b(?:signatur\w*|authoriz\w*)\b|\b(?:signatur\w*|authoriz\w*)\b.*\b(?:approv\w*)\b", re.IGNORECASE)),
    ("CALCULATION_MISMATCH", re.compile(r"\bcalculat\w*\b", re.IGNORECASE)),
    ("PARAMETER_MISMATCH", re.compile(r"\b(?:approved|parameter|specification|setting|set[\s-]?point|limit|range|process\s+condition)\b", re.IGNORECASE)),
    ("RECORD_TO_SOURCE_MISMATCH", re.compile(r"\b(?:source\s+record|electronic\s+record|paper\s+record|reconcil\w*)\b", re.IGNORECASE)),
    ("TIMING_MISMATCH", re.compile(r"\b(?:time|date|schedule|deadline|timestamp|duration)\b", re.IGNORECASE)),
    ("CONFIGURATION_MISMATCH", re.compile(r"\b(?:configuration|config|system\s+setting)\b", re.IGNORECASE)),
    ("QUANTITY_MISMATCH", re.compile(r"\b(?:quantity|amount|count|number\s+of)\b", re.IGNORECASE)),
]


def classify_comparison_subtype(left: str | None, right: str | None, basis: str | None = None) -> str:
    """Classify a comparison/mismatch finding into a specific investigation
    FRAMEWORK based on the compared phrases' own vocabulary. Returns
    "UNKNOWN_COMPARISON" when nothing matches -- routed through the
    original generic calculation-flavored tree as a safe default, never
    left un-investigatable."""
    combined = " ".join(p for p in (left, right, basis) if p)
    for subtype, pattern in _COMPARISON_SUBTYPE_RULES:
        if pattern.search(combined):
            return subtype
    return "UNKNOWN_COMPARISON"


def extract_semantic_subject(text: str) -> DeviationInfo:
    """Extract the semantic affected object + condition from finding text.
    
    Deterministic semantic resolver enforces:
      - ACTOR is separated from AFFECTED_OBJECT, ACTIVITY, PROCESS, RECORD, NOTIFICATION.
      - Grammatical subject (e.g. 'Four employees') is never used as affected_object.
      - Returns strongly typed DeviationInfo with semantic_type and relevant_change.
    """
    if not text or not text.strip():
        return DeviationInfo(subject=None)

    entities = extract_entities(text)
    actors = extract_actors(text)
    date = extract_date(text)

    _referenced_doc_sentences = {
        span for _doc_type, span in detect_referenced_unavailable_documents(text)
    }

    t_low = text.lower()
    # 0. Specialized Domain-Level Semantic Classification (Section 1 & 11)
    if re.search(r"\b(?:duplicate\s+(?:supplier\s+|vendor\s+|invoice\s+)?payments?|paid\s+twice|double\s+payments?)\b", t_low):
        affected_obj = "Duplicate payment to supplier"
        finding_subj = "Duplicate payment to supplier"
        process_name = "Accounts Payable — Payment Processing"
        activity_name = "Supplier invoice payment processing"
        dev_str = "Duplicate payment to supplier identified"
        # Deliberately just "processed", not "duplicate transaction
        # processed": finding_subj above already says "Duplicate payment to
        # supplier" -- repeating "duplicate transaction" in the condition
        # too produces "Why was the Duplicate payment to supplier duplicate
        # transaction processed?" (word repetition, not a new predicate).
        cond_str = "processed"
        return DeviationInfo(
            subject=finding_subj,
            finding_subject=finding_subj,
            affected_object=affected_obj,
            affected_process=process_name,
            affected_activity=activity_name,
            deviation=dev_str,
            condition=cond_str,
            requirement="Accounts payable payment verification and duplicate-detection controls",
            date=date,
            actor="Accounts payable / authorized payment personnel",
            actors=actors,
            entities=entities,
            semantic_type="FINANCIAL",
            matched=True,
        )

    if "overpayment" in t_low or "overpaid" in t_low:
        affected_obj = "Supplier payment transaction"
        finding_subj = "Supplier overpayment"
        process_name = "Accounts payable and payment verification control"
        activity_name = "Supplier invoice payment processing"
        dev_str = "Overpayment to supplier identified"
        # Same fix as the duplicate-payment case above: finding_subj is
        # already "Supplier overpayment", so the condition must not repeat
        # "overpayment" too.
        cond_str = "processed"
        return DeviationInfo(
            subject=finding_subj,
            finding_subject=finding_subj,
            affected_object=affected_obj,
            affected_process=process_name,
            affected_activity=activity_name,
            deviation=dev_str,
            condition=cond_str,
            requirement="Accounts payable payment verification controls",
            date=date,
            actor=None,
            actors=actors,
            entities=entities,
            semantic_type="FINANCIAL",
            matched=True,
        )

    # 0b. F5 — Version/state comparison: a finding asserting that an entity
    # has a LOCAL STATE (obsolete/outdated/superseded/expired/withdrawn/
    # deprecated/archived/stale/prior/previous/former) that differs from an
    # AUTHORITATIVE STATE (current/approved/active) at a controlled source.
    # Structurally this is a COMPARISON proposition, not merely a noun phrase
    # -- the document name, local state, location, authoritative state, and
    # source must all be preserved as distinct fields so downstream nodes
    # can generate discriminating investigation questions.
    #
    # The semantic class is defined by the ROLES (local vs authoritative),
    # not by specific words like "revision" or "copy": any controlled-document
    # type word (revision, version, copy, copies, edition, draft, form,
    # document, procedure, instruction) in combination with any obsolete-state
    # qualifier triggers this path.
    _OBSOLETE_STATE_WORDS = (
        r"outdated|obsolete|superseded|expired|revoked|withdrawn|deprecated"
        r"|archived|stale|old|previous|prior|former"
    )
    _DOCUMENT_TYPE_WORDS = (
        r"revision|version|copy|copies|edition|draft|form|document|"
        r"procedure|instruction"
    )
    _obsolete_copy_match = re.search(
        rf"\b(?:{_OBSOLETE_STATE_WORDS})\s+(?:{_DOCUMENT_TYPE_WORDS})s?\s+of\s+"
        rf"(?:the\s+)?(?:same\s+)?(?P<doc>[a-zA-Z0-9][a-zA-Z0-9\s-]*?)"
        rf"\s+(?:were|was)\s+(?:found|identified|located|discovered|present|observed|in\s+use)"
        rf"(?:\s+(?:at|in|on)\s+(?P<loc>[^.]+?))?\s*\.?",
        text, re.IGNORECASE,
    )
    # Detect second clause asserting the authoritative/controlled state
    _auth_state_match = re.search(
        r"\b(?:current|approved|active|valid|latest|effective)\s+"
        r"(?:[a-z]+\s+)?(?:version|revision|copy|edition|procedure|instruction|form)?\s*"
        r"(?:was|is|were|are)\s+(?:available|located|stored|maintained|held|kept)?\s*"
        r"(?:in|on|at|within)?\s*(?:the\s+)?(?P<source>[a-zA-Z0-9\s-]{3,60}?)\s*(?:\.|$)",
        text, re.IGNORECASE,
    )
    if _obsolete_copy_match:
        doc_raw = _obsolete_copy_match.group("doc").strip()
        doc_name = _clean_subject(doc_raw) or "controlled document"
        doc_cap = doc_name[0].upper() + doc_name[1:]
        affected_obj = f"{doc_cap} copies"
        finding_subj = doc_name
        return DeviationInfo(
            subject=finding_subj,
            finding_subject=finding_subj,
            affected_object=affected_obj,
            affected_process="Controlled document distribution and obsolete-copy withdrawal",
            affected_activity="Controlled-copy distribution and withdrawal",
            deviation=f"Outdated versions of the {doc_name} were found in use",
            condition="outdated versions in use",
            date=date,
            actor=None,
            actors=actors,
            entities=entities,
            semantic_type="RECORD",
            matched=True,
            requirement_status="STATED",
        )

    # 0c. Relational/comparison findings ("X did not match Y", "X differed
    # from Y", "X exceeded Y", "X was below Y", "X did not reconcile with
    # Y", "X was inconsistent with Y", "X did not agree with Y", "X
    # conflicted with Y") -- a distinct finding SHAPE the generic
    # _CONDITION_PATTERNS loop below has no notion of: it has no pattern
    # for "two things were compared and disagreed", so it falls through to
    # generic clause-matching and (worse) sometimes captures the RIGHT-HAND
    # side of the comparison, plus its own "from <basis>" qualifier, as if
    # the whole thing were the subject ("Calculated yield from the
    # individual process entries execution"). Handled generically for any
    # domain (yield, temperature, invoice amount, quantity, batch result,
    # record reconciliation, ...), not specific to any one finding.
    _COMPARISON_VERB_RE = (
        r"(?:did\s+not\s+match|didn't\s+match|does\s+not\s+match|"
        r"differed\s+from|differs\s+from|"
        r"was\s+inconsistent\s+with|were\s+inconsistent\s+with|is\s+inconsistent\s+with|"
        r"exceeded|exceeds|"
        r"was\s+below|were\s+below|is\s+below|"
        r"did\s+not\s+reconcile\s+with|does\s+not\s+reconcile\s+with|"
        r"did\s+not\s+agree\s+with|does\s+not\s+agree\s+with|"
        r"conflicted\s+with|conflicts\s+with)"
    )
    # Maps the matched surface verb onto one of a small closed set of
    # comparison TYPES (Section 7: generalized semantic-event rendering) --
    # keyed on normalized verb text so any new verb phrase added to
    # _COMPARISON_VERB_RE above only needs one entry here, not a rewrite of
    # every downstream renderer.
    _COMPARISON_TYPE_BY_VERB = {
        "did not match": "MISMATCH", "didn't match": "MISMATCH", "does not match": "MISMATCH",
        "differed from": "MISMATCH", "differs from": "MISMATCH",
        "was inconsistent with": "INCONSISTENT", "were inconsistent with": "INCONSISTENT",
        "is inconsistent with": "INCONSISTENT",
        "exceeded": "EXCEEDED", "exceeds": "EXCEEDED",
        "was below": "BELOW", "were below": "BELOW", "is below": "BELOW",
        "did not reconcile with": "RECONCILIATION_FAILURE", "does not reconcile with": "RECONCILIATION_FAILURE",
        "did not agree with": "MISMATCH", "does not agree with": "MISMATCH",
        "conflicted with": "INCONSISTENT", "conflicts with": "INCONSISTENT",
    }
    _measurement = extract_measured_discrepancy(text)
    # Match against the single SENTENCE containing the comparison verb, not
    # the whole (possibly multi-sentence) finding text -- otherwise a
    # trailing sentence ("The difference was approximately 4.2%.") gets
    # swallowed into the right-hand side/basis capture group instead of
    # staying a separate measurement.
    _comparison_sentence = next(
        (s for s in _SENTENCE_SPLIT_RE.split(text.strip())
         if re.search(_COMPARISON_VERB_RE, s, re.IGNORECASE)),
        text,
    )
    _comparison_match = re.search(
        rf"^(?P<left>.+?)\s+(?P<verb>{_COMPARISON_VERB_RE})\s+(?:the\s+)?(?P<right>.+?)\s*\.?$",
        _strip_framing(_comparison_sentence.strip()), re.IGNORECASE,
    )
    if _comparison_match:
        left_raw = _comparison_match.group("left").strip()
        right_raw = _comparison_match.group("right").strip()
        comparison_type = _COMPARISON_TYPE_BY_VERB.get(
            re.sub(r"\s+", " ", _comparison_match.group("verb").lower()), "MISMATCH"
        )

        # Strip trailing relative pronoun clauses or state values from left_raw (e.g. "The recorded drying temperature was 65°C, which" -> "The recorded drying temperature")
        left_raw = re.sub(r",?\s*(?:which|and)\s*$", "", left_raw, flags=re.IGNORECASE).strip()
        left_raw = re.sub(r"\s+(?:was|is|were)\s+[\d.]+\s*°?[a-z%]*\s*$", "", left_raw, flags=re.IGNORECASE).strip()

        # Strip a trailing clause that names a KNOWN ENTITY (batch/lot/
        # equipment id already extracted from the full text) off either
        # side -- e.g. "temperature setting recorded for production batch
        # BR-2026-0815" -> object clause "temperature setting recorded",
        # batch "BR-2026-0815". Entity-driven rather than a fixed
        # preposition list, so it generalizes to "for", "on", "of", "in
        # batch", etc. without enumerating every phrasing.
        comparison_batch_id = None
        # Only entities that look like actual IDENTIFIERS (contain a digit,
        # e.g. "BR-2026-0815") qualify for this stripping -- a generic
        # document-type phrase entity like "batch record" must NOT trigger
        # it, or "specified in the batch record" gets wrongly truncated to
        # "specified" (there is no id to strip, just a reference-type noun
        # the basis-clause stripper below already handles correctly).
        _id_entities = [e for e in entities if re.search(r"\d", e)]
        for _side_name, _side_raw in (("left", left_raw), ("right", right_raw)):
            for _ent in _id_entities:
                _ent_clause = re.search(
                    rf"^(?P<obj>.+?)\s+(?:for|on|of|in|to|regarding)\s+(?:the\s+)?(?:production\s+)?"
                    rf"(?:batch|lot|equipment|record)?\s*{re.escape(_ent)}\b.*$",
                    _side_raw, re.IGNORECASE,
                )
                if _ent_clause:
                    comparison_batch_id = comparison_batch_id or _ent
                    if _side_name == "left":
                        left_raw = _ent_clause.group("obj").strip()
                    else:
                        right_raw = _ent_clause.group("obj").strip()
                    break

        # Strip an actor-attribution clause off the LEFT side ("the final
        # yield recorded by the operator" -> object "final yield", actor
        # "operator") -- the object being compared is never the same thing
        # as who recorded/reported/measured it. The "by <actor>" clause is
        # optional so a bare trailing provenance verb ("... recorded", with
        # its batch qualifier already stripped above) is still recognized.
        _actor_attrib = re.search(
            r"^(?P<obj>.+?)\s+(?P<qualifier>recorded|reported|entered|measured|logged|documented)"
            r"(?:\s+by\s+(?:the\s+)?(?P<actor>[a-z][a-z\s]*?))?$",
            left_raw, re.IGNORECASE,
        )
        comparison_actor = None
        comparison_left_qualifier = None
        if _actor_attrib:
            left_raw = _actor_attrib.group("obj").strip()
            comparison_actor = (_actor_attrib.group("actor") or "").strip() or None
            # The provenance verb ("recorded"/"measured"/"logged") is
            # meaningful when naming WHICH of the two compared values this
            # is (the "recorded" reading vs. the "calculated" one) -- kept
            # as a qualifier for 5-Why/impact rendering even though it's
            # stripped from affected_object (which must stay a bare noun
            # phrase, e.g. "Final yield", not "Recorded final yield").
            comparison_left_qualifier = _actor_attrib.group("qualifier").lower()

        # Strip a basis/reference-source clause off the RIGHT side ("the
        # calculated yield from the individual process entries" -> object
        # "calculated yield", basis "individual process entries"; "the
        # approved process parameter specified in the batch record" ->
        # object "approved process parameter", basis "the batch record")
        # -- provenance, not part of the object's own name.
        _basis_match = re.search(
            r"^(?P<obj>.+?)\s+(?:from|based\s+on|per|using|specified\s+in|documented\s+in|"
            r"defined\s+in|stated\s+in|noted\s+in)\s+(?:the\s+)?(?P<basis>[a-z][a-z0-9\s-]*?)$",
            right_raw, re.IGNORECASE,
        )
        comparison_basis = None
        if _basis_match:
            right_raw = _basis_match.group("obj").strip()
            comparison_basis = _basis_match.group("basis").strip()

        left_clean = _clean_subject(left_raw)
        right_clean = _clean_subject(right_raw)
        if left_clean and right_clean and validate_semantic_subject(left_clean):
            affected_obj = left_clean[0].upper() + left_clean[1:]
            # The HEAD noun of an "adjective(s) + noun" phrase ("final
            # yield", "batch record", "temperature reading") is normally
            # the LAST significant word, not the first (topic_word() picks
            # the first, which is right for a single-noun subject but wrong
            # here -- "final yield" -> "final" reads oddly as a process
            # name ("Final reconciliation") where "yield" is the actual
            # topic).
            _topic_words = [w for w in re.findall(r"[a-z]+", left_clean.lower()) if len(w) > 2]
            topic = _topic_words[-1] if _topic_words else topic_word(left_clean)
            basis_suffix = f" against {comparison_basis}" if comparison_basis else ""
            dev_str = f"The {left_clean} did not match the {right_clean}{basis_suffix}"
            comparison_subtype = classify_comparison_subtype(left_clean, right_clean, comparison_basis)
            comparison_reference_type = {
                "CALCULATION_MISMATCH": "CALCULATED_VALUE",
                "PARAMETER_MISMATCH": "APPROVED_PARAMETER",
                "VERSION_MISMATCH": "APPROVED_VERSION",
                "RECORD_TO_SOURCE_MISMATCH": "SOURCE_RECORD",
                "TIMING_MISMATCH": "SCHEDULED_TIME",
                "CONFIGURATION_MISMATCH": "APPROVED_CONFIGURATION",
                "APPROVAL_MISMATCH": "APPROVAL_RECORD",
                "QUANTITY_MISMATCH": "REQUIRED_QUANTITY",
            }.get(comparison_subtype, "REFERENCE_VALUE")
            return DeviationInfo(
                subject=left_clean,
                finding_subject=left_clean,
                affected_object=affected_obj,
                affected_process=f"{topic.capitalize()} reconciliation and verification",
                deviation=dev_str,
                condition="not matched",
                date=date,
                actor=comparison_actor,
                actors=([comparison_actor] if comparison_actor else []) + actors,
                entities=entities,
                semantic_type="COMPARISON",
                matched=True,
                comparison_type=comparison_type,
                comparison_left=left_clean,
                comparison_left_qualifier=comparison_left_qualifier,
                comparison_right=right_clean,
                comparison_basis=comparison_basis,
                measurement_value=_measurement[0] if _measurement else None,
                measurement_unit=_measurement[1] if _measurement else None,
                measurement_qualifier=_measurement[2] if _measurement else None,
                comparison_subtype=comparison_subtype,
                comparison_reference_type=comparison_reference_type,
                comparison_batch_id=comparison_batch_id,
            )

    # 0d. Missing-record / missing-documentation findings ("X was not
    # documented", "X was not recorded", "no record of X exists", "X lacks
    # documentation") -- a distinct finding SHAPE (Section 2): the affected
    # object is the ACTIVITY/CONTROL that should have been documented, not
    # a batch/record CONTEXT identifier the activity relates to, and not
    # the whole clause including any downstream action the finding also
    # reports. Generalizes across any domain (inspection, review, approval,
    # verification, calibration check, ...), not specific to any one
    # activity type.
    _MISSING_RECORD_VERB_RE = (
        r"(?:was\s+not\s+document(?:ed)?|were\s+not\s+document(?:ed)?|"
        r"was\s+not\s+recorded|were\s+not\s+recorded|"
        r"was\s+not\s+logged|were\s+not\s+logged|"
        r"has\s+no\s+record|have\s+no\s+record|"
        r"no\s+record\s+(?:exists|was\s+found|could\s+be\s+found)|"
        r"lacks?\s+documentation|"
        r"was\s+undocumented|were\s+undocumented|"
        r"is\s+not\s+evidenced|are\s+not\s+evidenced|was\s+not\s+evidenced|were\s+not\s+evidenced|"
        r"no\s+documented\s+evidence)"
    )
    _missing_record_sentence = next(
        (s for s in _SENTENCE_SPLIT_RE.split(text.strip())
         if re.search(_MISSING_RECORD_VERB_RE, s, re.IGNORECASE)),
        None,
    )
    if _missing_record_sentence:
        _mr_match = re.search(
            rf"^(?P<activity>.+?)\s+(?P<verb>{_MISSING_RECORD_VERB_RE})\b(?P<rest>.*)$",
            _strip_framing(_missing_record_sentence.strip()), re.IGNORECASE,
        )
        if _mr_match:
            activity_raw = _mr_match.group("activity").strip()
            _mr_id_entities = [e for e in entities if re.search(r"\d", e)]
            missing_record_context = None
            for _ent in _mr_id_entities:
                # Up to 3 lowercase context words ("new hire", "production
                # batch", "work order") may sit between the preposition and
                # the entity id -- not restricted to a fixed noun list, so
                # this generalizes to any domain's own context-object
                # vocabulary.
                _ctx_clause = re.search(
                    rf"^(?P<obj>.+?)\s+(?:for|on|of|in|to|regarding)\s+(?:the\s+)?"
                    rf"(?:[a-z]+\s+){{0,3}}{re.escape(_ent)}\b.*$",
                    activity_raw, re.IGNORECASE,
                )
                if _ctx_clause:
                    missing_record_context = _ent
                    activity_raw = _ctx_clause.group("obj").strip()
                    break
            activity_clean = _clean_subject(activity_raw)
            if activity_clean and validate_semantic_subject(activity_clean):
                # Downstream-action detection (Section 4): a contrastive
                # clause ("although/though/but/however/, and") containing a
                # passive-voice past-participle construction ("was
                # released", "was approved", "was dispatched", "was
                # completed", ...) describing a completed action AFTER the
                # missing-evidence assertion -- detected structurally (verb
                # SHAPE), never from a fixed list of domain verbs.
                _downstream_match = re.search(
                    r"\b(?:although|though|but|however|whereas)\b\s*,?\s*"
                    r"(?P<subj>(?:the\s+|it\s+)?[a-z][a-z\s]*?)\s+"
                    r"(?:"
                    r"(?:was|were|had\s+(?:already\s+)?been)\s+(?:subsequently\s+|later\s+|already\s+)?(?P<verb>[a-z]+(?:ed|en))\b|"
                    r"(?:subsequently\s+|later\s+|already\s+)?(?P<verb2>[a-z]+ed)\b"
                    r")"
                    r"(?P<downstream_rest>[^.]*)",
                    text, re.IGNORECASE,
                )
                downstream_present = bool(_downstream_match)
                downstream_text = _downstream_match.group(0).strip() if _downstream_match else None
                affected_obj = activity_clean[0].upper() + activity_clean[1:]
                # Preserve the ACTUAL verb the finding used ("logged" /
                # "recorded" / "documented" / "evidenced" / "no record")
                # rather than normalizing every phrasing to "not
                # documented" -- downstream word-overlap checks (e.g. the
                # analytical-validator "does this mechanism restate the
                # observation" guard) compare vocabulary, so silently
                # substituting a synonym would make an identical
                # observation look like a distinct new fact.
                _verb_matched = _mr_match.group("verb").lower()
                if "log" in _verb_matched:
                    _condition_word = "not logged"
                elif "record" in _verb_matched:
                    _condition_word = "missing a record" if "no record" in _verb_matched else "not recorded"
                elif "evidenc" in _verb_matched:
                    _condition_word = "not evidenced"
                elif "undocumented" in _verb_matched:
                    _condition_word = "undocumented"
                else:
                    _condition_word = "not documented"
                dev_str = f"The {activity_clean} was {_condition_word}"
                return DeviationInfo(
                    subject=activity_clean,
                    finding_subject=activity_clean,
                    affected_object=affected_obj,
                    affected_process=f"{topic_word(activity_clean).capitalize()} documentation and record control",
                    deviation=dev_str,
                    condition=_condition_word,
                    date=date,
                    actor=actors[0] if actors else None,
                    actors=actors,
                    entities=entities,
                    semantic_type="MISSING_RECORD",
                    matched=True,
                    missing_record_activity=activity_clean,
                    missing_record_context=missing_record_context,
                    downstream_action_text=downstream_text,
                    downstream_action_present=downstream_present,
                )

    # 0e. Recurrence findings ("the same X was identified in N separate
    # batches/records/units/...", "a similar X was observed across N
    # locations") -- a distinct finding SHAPE (Section 1/2/13): the
    # repeated DEVIATION (e.g. "process deviation") is the affected object,
    # never the generic population phrase ("three separate batches") and
    # never the degraded "process compliance" placeholder. Generalizes
    # across any domain/population noun (batches, records, transactions,
    # locations, periods, cases, ...), not specific to any one wording.
    _recurrence_match = re.search(
        rf"\b(?:the\s+same|a\s+similar|this)\s+(?P<deviation>[\w\s-]{{1,40}}?)\s+(?:was|were)\s+"
        rf"(?P<verb>identified|observed|found|noted|detected|reported)\s+"
        rf"(?P<prep>in|across|on)\s+(?P<population>{_NUMBER_WORD}\s+(?:separate\s+|different\s+|distinct\s+)?"
        rf"(?:[a-z]+\s+){{0,2}}"
        rf"(?:batches|records|units|transactions|locations|sites|periods|cases|instances|occasions|"
        rf"shipments|lots|samples|invoices|files|systems|documents|departments|suppliers|employees))\b",
        text, re.IGNORECASE,
    )
    if _recurrence_match:
        deviation_raw = _recurrence_match.group("deviation").strip()
        deviation_clean = _clean_subject(deviation_raw)
        if deviation_clean and validate_semantic_subject(deviation_clean):
            affected_obj = deviation_clean[0].upper() + deviation_clean[1:]
            population_text = _recurrence_match.group("population").strip()
            return DeviationInfo(
                subject=deviation_clean,
                finding_subject=deviation_clean,
                affected_object=affected_obj,
                affected_process=f"{topic_word(deviation_clean).capitalize()} operational process",
                deviation=f"The {deviation_clean} was {_recurrence_match.group('verb')} {_recurrence_match.group('prep')} {population_text}",
                condition=f"{_recurrence_match.group('verb')} {_recurrence_match.group('prep')} {population_text}",
                date=date,
                actor=actors[0] if actors else None,
                actors=actors,
                entities=entities,
                semantic_type="RECURRENCE",
                matched=True,
                occurrence_population=population_text,
            )

    # 0f. Attributed causal explanation ("X stated/reported/claimed/
    # explained that ACTIVITY was skipped/omitted/missed/bypassed/deferred
    # because REASON") -- a distinct finding SHAPE (Section 2/3): the
    # ACTIVITY is the affected object; the reporting SOURCE and the causal
    # REASON offered are kept as separate attributed-claim metadata, never
    # folded into the object name or promoted into the deviation clause
    # itself. Generalizes across any role/domain (operator, technician,
    # supervisor, employee, ...) and any omission verb, not specific to
    # any one activity type.
    _attribution_match = re.search(
        r"\b(?P<source>(?:an?|the)\s+[a-z][a-z\s]{1,25}?)\s+"
        r"(?P<report_verb>stated|reported|claimed|explained|indicated|noted|mentioned|believed|said)\s+that\s+"
        r"(?P<activity>(?:the\s+|a\s+)?(?:required|mandatory|necessary\s+)?[a-z][a-z\s-]{2,50}?)\s+"
        r"(?:was|were)\s+(?P<omit_verb>skipped|omitted|missed|bypassed|deferred|not\s+performed|not\s+completed|not\s+done)\b"
        r"(?:\s+because\s+(?P<reason>.+?))?\s*\.?$",
        text, re.IGNORECASE,
    )
    if _attribution_match:
        activity_clean = _clean_subject(_attribution_match.group("activity").strip())
        if activity_clean and validate_semantic_subject(activity_clean):
            affected_obj = activity_clean[0].upper() + activity_clean[1:]
            source_text = re.sub(r"^(?:an?|the)\s+", "", _attribution_match.group("source").strip(), flags=re.IGNORECASE)
            reason_text = (_attribution_match.group("reason") or "").strip().rstrip(".") or None
            omit_verb = _attribution_match.group("omit_verb").lower()
            return DeviationInfo(
                subject=activity_clean,
                finding_subject=activity_clean,
                affected_object=affected_obj,
                affected_process=f"{topic_word(activity_clean).capitalize()} operational process",
                deviation=f"The {activity_clean} was {omit_verb}",
                condition=omit_verb,
                date=date,
                actor=source_text,
                actors=([source_text] if source_text else []) + actors,
                entities=entities,
                semantic_type="ATTRIBUTED_EXPLANATION",
                matched=True,
                reported_mechanism=reason_text,
                mechanism_status="REPORTED" if reason_text else "UNKNOWN",
                mechanism_polarity="non_performance",
                attributed_source=source_text,
                attributed_proposition=reason_text,
            )

    # 0g. Event-sequence / control-point findings (Section 1/2/3/5): a
    # controlled TRANSITION (invalidation, override, waiver, exception,
    # escalation, transfer, disposition, ...) whose required justification/
    # authorization is reported as missing/undocumented -- a distinct
    # finding SHAPE from a plain missing-record finding, because the
    # missing artifact here documents a DECISION about a prior event, not
    # the event itself. Generalizes across any domain via a closed
    # RELATION-TYPE vocabulary (Section 5), never a keyword list for one
    # specific transition like "invalidation"/"OOS"/"retest".
    _TRANSITION_TYPE_WORDS = {
        "MODIFICATION": ("modifi", "altered", "altering", "change", "changed", "changing", "edit", "edited", "editing"),
        "INVALIDATION": ("invalidat",),
        "OVERRIDE": ("override", "overridden"),
        "EXCEPTION": ("exception", "excepted"),
        "APPROVAL": ("approv",),
        "RELEASE": ("released", "release"),
        "RETEST": ("retest", "repeat"),
        "REWORK": ("rework",),
        "ACCEPTANCE": ("accept",),
        "CLOSURE": ("closed", "closure"),
        "ESCALATION": ("escalat",),
        "TRANSFER": ("transferr", "transfer"),
        "DISPOSITION": ("disposition", "dispositioned"),
        "WAIVER": ("waiv",),
        "BYPASS": ("bypass",),
    }
    _control_gap_match = re.search(
        r"\b(?:no\s+(?:documented\s+|recorded\s+|approved\s+|written\s+)?(?P<missing_thing>justification|authorization|authorisation|approval|reason|"
        r"explanation|rationale)\s+(?:for\s+(?:the\s+|this\s+)?(?P<transition_ref>[a-z][a-z\s-]{2,40}?)\s+)?"
        r"(?:was|is|were)\s+(?:documented|recorded|provided|available|found)|"
        r"(?:with\s+no|without)\s+(?:documented\s+|recorded\s+|approved\s+|written\s+)?(?P<missing_thing2>justification|authorization|authorisation|approval|reason|rationale))\b",
        text, re.IGNORECASE,
    )
    if _control_gap_match:
        transition_ref = (_control_gap_match.group("transition_ref") or "").strip()
        _combined_context = f"{transition_ref} {text}"
        transition_type = next(
            (label for label, stems in _TRANSITION_TYPE_WORDS.items()
             if any(stem in _combined_context.lower() for stem in stems)),
            None,
        )
        if transition_type:
            # Check for leading observed entity (e.g. "A laboratory result was modified after its initial entry")
            _lead_obs_match = re.search(
                r"\b(?P<subj>(?:an?|the)\s+[a-z][a-z0-9\s-]{2,40}?)\s+"
                r"(?:was|were)\s+(?P<action>[a-z]+(?:ed|en)(?:\s+[^.,;]+)?)",
                text, re.IGNORECASE,
            )
            if _lead_obs_match:
                extracted_subj = _clean_subject(_lead_obs_match.group("subj"))
                observed_action = _lead_obs_match.group("action").strip()
                subject_clean = extracted_subj if (extracted_subj and validate_semantic_subject(extracted_subj)) else (transition_ref or transition_type.replace("_", " ").lower())
                deviation_str = f"The {subject_clean} was {observed_action} without documented justification"
                condition_str = observed_action
            else:
                subject_phrase = transition_ref or transition_type.replace("_", " ").lower()
                subject_clean = _clean_subject(subject_phrase) or subject_phrase
                deviation_str = f"The {subject_clean} {transition_type.lower()} occurred without documented justification"
                condition_str = "justification not documented"

            affected_obj = subject_clean[0].upper() + subject_clean[1:] if subject_clean else transition_type.capitalize()
            _downstream_match = re.search(
                r"\b(?:although|though|but|however|whereas)\b\s*,?\s*"
                r"(?P<subj>(?:the\s+|it\s+)?[a-z][a-z\s]*?)\s+"
                r"(?:"
                r"(?:was|were|had\s+(?:already\s+)?been)\s+(?:subsequently\s+|later\s+|already\s+)?(?P<verb>[a-z]+(?:ed|en))\b|"
                r"(?:subsequently\s+|later\s+|already\s+)?(?P<verb2>[a-z]+ed)\b"
                r")"
                r"(?P<downstream_rest>[^.]*)",
                text, re.IGNORECASE,
            )
            return DeviationInfo(
                subject=subject_clean,
                finding_subject=subject_clean,
                affected_object=affected_obj,
                affected_process=f"{transition_type.replace('_', ' ').capitalize()} control and authorization",
                deviation=deviation_str,
                condition=condition_str,
                date=date,
                actor=actors[0] if actors else None,
                actors=actors,
                entities=entities,
                semantic_type="EVENT_SEQUENCE_CONTROL",
                matched=True,
                transition_type=transition_type,
                control_justification_missing=True,
                downstream_action_present=bool(_downstream_match),
                downstream_action_text=_downstream_match.group(0).strip() if _downstream_match else None,
                observed_entity=subject_clean,
            )

    # 0h. Requirement-uncertain findings (Section 2/3/4): an OBSERVATION is
    # reported ("X was observed/found/noted in condition Y") while the
    # governing REQUIREMENT/specification/procedure/standard is separately
    # reported as unknown/unavailable/unclear/not established -- these are
    # two INDEPENDENT dimensions (Section 3), never collapsed into a single
    # "process compliance" placeholder or a premature deviation/root-cause
    # conclusion. Generalizes across any domain via a closed requirement-
    # noun and uncertainty-phrase vocabulary, never a keyword list for one
    # specific requirement type.
    _requirement_uncertain_match = re.search(
        r"\b(?:the\s+)?(?:applicable\s+|governing\s+|relevant\s+)?"
        r"(?P<req_noun>requirement|specification|procedure|standard|instruction|limit|criteria|criterion)\b"
        r"(?:(?!\.).){0,60}?\b(?:could\s+not\s+be\s+(?:determined|confirmed|located)|"
        r"is\s+unavailable|was\s+unavailable|is\s+unknown|was\s+unknown|"
        r"is\s+unclear|was\s+unclear|(?:is|was)\s+not\s+established|"
        r"(?:is|was)\s+uncertain)\b",
        text, re.IGNORECASE,
    )
    if _requirement_uncertain_match:
        _observed_match = re.search(
            r"\b(?P<observed_entity>(?:(?:an?|the)\s+)?[a-z][a-z0-9\s-]{2,50}?)\s+"
            r"(?:was|were)\s+(?:observed|found|noted|identified)\s+"
            r"(?P<observed_cond>[^;,\.]+)",
            text, re.IGNORECASE,
        )
        if _observed_match:
            observed_raw = _observed_match.group("observed_entity").strip()
            observed_cond = _observed_match.group("observed_cond").strip()
            observed_clean = _clean_subject(observed_raw)
            if observed_clean and validate_semantic_subject(observed_clean):
                affected_obj = observed_clean[0].upper() + observed_clean[1:]
                req_noun = _requirement_uncertain_match.group("req_noun").lower()
                return DeviationInfo(
                    subject=observed_clean,
                    finding_subject=observed_clean,
                    affected_object=affected_obj,
                    affected_process=f"Applicable {req_noun} and control",
                    deviation=f"The {observed_clean} was {observed_cond}; the applicable {req_noun} is unresolved",
                    condition=observed_cond or "in an unverified condition",
                    date=date,
                    actor=actors[0] if actors else None,
                    actors=actors,
                    entities=entities,
                    semantic_type="REQUIREMENT_UNCERTAIN",
                    matched=True,
                    requirement_status="UNKNOWN",
                    observed_entity=observed_clean,
                )

    # 0i. F7 — Temporal deviation / deadline / duration condition.
    # Findings asserting that an activity's actual timing differed from its
    # expected timing: "X was N UNIT overdue/late/expired/delayed/past due".
    # This is a distinct semantic class — TEMPORAL_DEVIATION — with its own
    # dimension set (ACTIVITY, TEMPORAL_RELATION, DURATION). Not hardcoded to
    # any domain: "requalification review", "inspection", "calibration",
    # "approval" all route through the same structural match.
    _TEMPORAL_DEVIATION_RE = re.compile(
        # Form A: <SUBJECT> was/is <N UNIT> overdue/late/delayed/past
        r"^(?:the\s+)?(?P<subj_a>[a-z][a-z\s/,()-]{2,60}?)\s+"
        r"(?:was|is|has\s+been|were|are)\s+"
        r"(?:(?P<qty_a>\w+(?:\s+\w+)?)\s+(?P<unit_a>days?|weeks?|months?|years?|hours?)\s+)?"
        r"(?P<rel_a>overdue|late|delayed|past\s+(?:its\s+)?(?:due\s+date|deadline)|past\s+due"
        r"|delinquent|outstanding|expired|lapsed)\b",
        re.IGNORECASE,
    )
    _TEMPORAL_EXPIRED_RE = re.compile(
        # Form B: <SUBJECT> expired [N UNIT ago / before use / before the activity]
        r"^(?:the\s+)?(?P<subj_b>[a-z][a-z\s/,()-]{2,60}?)\s+(?:was\s+)?expired"
        r"(?:\s+(?P<qty_b>\w+(?:\s+\w+)?)\s+(?P<unit_b>days?|weeks?|months?|years?)\s+ago)?"
        r"(?:\s+(?:before\s+use|prior\s+to\s+use|before\s+the\s+(?:activity|procedure|test)))?",
        re.IGNORECASE,
    )
    for _td_sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        _td_sentence = _td_sentence.strip()
        if not _td_sentence or _td_sentence in _referenced_doc_sentences:
            continue
        _td_stripped = _strip_framing(_td_sentence)
        _td_m = _TEMPORAL_DEVIATION_RE.match(_td_stripped) or _TEMPORAL_DEVIATION_RE.match(_td_sentence)
        _td_exp = None
        if not _td_m:
            _td_exp = _TEMPORAL_EXPIRED_RE.match(_td_stripped) or _TEMPORAL_EXPIRED_RE.match(_td_sentence)
        if _td_m or _td_exp:
            if _td_m:
                _td_subj_raw = _td_m.group("subj_a").strip()
                _td_qty = (_td_m.group("qty_a") or "").strip()
                _td_unit = (_td_m.group("unit_a") or "").strip()
                _td_rel = (_td_m.group("rel_a") or "overdue").strip().lower()
            else:
                _td_subj_raw = _td_exp.group("subj_b").strip()
                _td_qty = (_td_exp.group("qty_b") or "").strip()
                _td_unit = (_td_exp.group("unit_b") or "").strip()
                _td_rel = "expired"
            _td_subj = _clean_subject(_td_subj_raw)
            # Guard: reject if subj is a pronoun or too vague
            if not _td_subj or not validate_semantic_subject(_td_subj) or is_actor_noun(_td_subj):
                continue
            _td_subj_cap = _td_subj[0].upper() + _td_subj[1:]
            _td_duration = f"{_td_qty} {_td_unit}".strip() if _td_qty and _td_unit else ""
            _td_cond = (
                f"{_td_rel} by {_td_duration}" if _td_duration else _td_rel
            )
            return DeviationInfo(
                subject=_td_subj,
                finding_subject=_td_subj,
                affected_object=_td_subj_cap,
                affected_process=f"{topic_word(_td_subj_cap).capitalize()} scheduling and deadline control",
                affected_activity=_td_subj_cap,
                deviation=f"{_td_subj_cap} \u2014 {_td_cond}",
                condition=_td_cond,
                date=date,
                actor=actors[0] if actors else None,
                actors=actors,
                entities=entities,
                semantic_type="TEMPORAL_DEVIATION",
                matched=True,
                requirement_status="STATED",
            )

    # 0j. F3 — Passive / resultative event: a finding where ENTITY was
    # affected by an OUTCOME event, agent possibly absent.  "Batch was
    # scrapped", "Invoice was cancelled", "Sample was rejected", "Payment was
    # reversed", "Record was deleted", "Equipment was removed from service".
    # The pattern is structural: ENTITY + passive auxiliary + outcome-verb.
    # It does NOT hardcode a closed list of outcome verbs — it uses a curated
    # set of outcome-class verb stems (irreversible state-change events) and
    # supplements them with regular past-participle morphology for open-vocab
    # extension. A financial value qualifier ("worth ₹X", "valued at $Y") is
    # stripped from the entity to prevent the value from replacing the entity
    # as the subject.
    _OUTCOME_VERBS = (
        r"scrap(?:ped)?|reject(?:ed)?|cancel(?:led|ed)?|revers(?:ed)?|delet(?:ed)?|"
        r"dispos(?:ed)?\s+of|quarantin(?:ed)?|recall(?:ed)?|written\s+off|"
        r"remov(?:ed)\s+from\s+service|suspend(?:ed)?|terminat(?:ed)?|"
        r"withdrawn|withheld|destroy(?:ed)?|discard(?:ed)?"
    )
    # 0j. Continued obsolete document use ("SOP-014 was updated but technicians continued using SOP-014A and SOP-014-B")
    _OBSOLETE_USE_RE = re.compile(
        r"^(?P<doc>[A-Z0-9/-]+|[a-z][a-z0-9\s-]{2,40}?(?:\s+(?:procedure|sop|instruction|standard)))\s+"
        r"(?:was|is)\s+(?:updated|revised|superseded)\s+but\s+"
        r"(?P<actor>[a-z\s]+?)\s+continued\s+using\s+(?P<obsolete>.+?)\s*\.?$",
        re.IGNORECASE,
    )
    # 0k. Required rework / additional labor hours ("The batch required 40 additional rework hours due to improper packaging")
    _REWORK_REQUIRED_RE = re.compile(
        r"^(?:the\s+)?(?P<obj>[a-z0-9\s/-]{2,40}?)\s+"
        r"required\s+(?P<qty>\d+(?:\.\d+)?)\s+(?:additional\s+)?(?P<unit>rework\s+hours?|hours?|days?|labor\s+hours?)"
        r"(?:\s+(?:due\s+to|because\s+of|following|as\s+a\s+result\s+of)\s+(?P<cause>[^.]+?))?\s*\.?$",
        re.IGNORECASE,
    )
    for _sec_sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        _sec_sentence = _sec_sentence.strip()
        if not _sec_sentence or _sec_sentence in _referenced_doc_sentences:
            continue
        _sec_stripped = _strip_framing(_sec_sentence)
        _m_obs = _OBSOLETE_USE_RE.match(_sec_stripped) or _OBSOLETE_USE_RE.match(_sec_sentence)
        if _m_obs:
            _doc_raw = _clean_subject(_m_obs.group("doc").strip()) or _m_obs.group("doc").strip()
            _actor_raw = _m_obs.group("actor").strip()
            _obs_raw = _m_obs.group("obsolete").strip()
            _doc_cap = _doc_raw[0].upper() + _doc_raw[1:]
            return DeviationInfo(
                subject=_doc_raw,
                finding_subject=_doc_raw,
                affected_object=_doc_cap,
                affected_process="Controlled document distribution and obsolete-copy withdrawal",
                affected_activity=f"{_doc_cap} revision control",
                deviation=f"Obsolete versions ({_obs_raw}) continued in use after {_doc_raw} update",
                condition="obsolete version in use",
                relevant_change=f"Revision of {_doc_raw}",
                date=date,
                actor=_actor_raw,
                actors=[_actor_raw] + [a for a in actors if a != _actor_raw],
                entities=entities,
                semantic_type="RECORD",
                matched=True,
                requirement_status="STATED",
            )
        _m_rew = _REWORK_REQUIRED_RE.match(_sec_stripped) or _REWORK_REQUIRED_RE.match(_sec_sentence)
        if _m_rew:
            _obj_raw = _clean_subject(_m_rew.group("obj").strip())
            if _obj_raw and validate_semantic_subject(_obj_raw):
                _obj_cap = _obj_raw[0].upper() + _obj_raw[1:]
                _cause_raw = _clean_subject((_m_rew.group("cause") or "").strip()) or None
                _qty_str = _m_rew.group("qty")
                _unit_str = _m_rew.group("unit")
                return DeviationInfo(
                    subject=_obj_raw,
                    finding_subject=_obj_raw,
                    affected_object=_obj_cap,
                    affected_process=f"{topic_word(_obj_cap).capitalize()} processing and packaging control",
                    affected_activity=f"{_obj_cap} rework",
                    deviation=f"{_obj_cap} required {_qty_str} {_unit_str}" + (f" due to {_cause_raw}" if _cause_raw else ""),
                    condition="rework required",
                    relevant_change=_cause_raw,
                    date=date,
                    actor=actors[0] if actors else None,
                    actors=actors,
                    entities=entities,
                    semantic_type="ACTIVITY",
                    matched=True,
                    requirement_status="STATED",
                )

    _PASSIVE_RESULT_RE = re.compile(
        r"^(?:an?\s+|the\s+)?(?P<entity>[a-z][a-z0-9\s/-]{1,50}?)"
        r"(?:\s+(?:worth|valued\s+at|costing|of)\s+[^,]+?(?:,\d{3})*(?:\s*,\s*)?)?"
        r"\s+(?:was|were|has\s+been|have\s+been)\s+"
        r"(?P<outcome>" + _OUTCOME_VERBS + r")"
        r"(?:\s+(?:due\s+to|because\s+of|following|as\s+a\s+result\s+of)\s+"
        r"(?:a\s+|an\s+|the\s+)?(?P<cause>[^.]+?))?\s*\.?$",
        re.IGNORECASE,
    )
    for _pr_sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        _pr_sentence = _pr_sentence.strip()
        if not _pr_sentence or _pr_sentence in _referenced_doc_sentences:
            continue
        _pr_stripped = _strip_framing(_pr_sentence)
        _pr_m = _PASSIVE_RESULT_RE.match(_pr_stripped) or _PASSIVE_RESULT_RE.match(_pr_sentence)
        if _pr_m:
            _pr_entity_raw = (_pr_m.group("entity") or "").strip()
            _pr_entity = _clean_subject(_pr_entity_raw)
            if not _pr_entity or not validate_semantic_subject(_pr_entity):
                continue
            _pr_outcome = (_pr_m.group("outcome") or "").strip().lower()
            _pr_cause = _clean_subject((_pr_m.group("cause") or "").strip()) or None
            _pr_cap = _pr_entity[0].upper() + _pr_entity[1:]
            return DeviationInfo(
                subject=_pr_entity,
                finding_subject=_pr_entity,
                affected_object=_pr_cap,
                affected_process=f"{topic_word(_pr_cap).capitalize()} disposition and outcome control",
                affected_activity=_pr_cap,
                deviation=f"{_pr_cap} \u2014 {_pr_outcome}",
                condition=_pr_outcome,
                relevant_change=_pr_cause,
                date=date,
                actor=actors[0] if actors else None,
                actors=actors,
                entities=entities,
                semantic_type="PASSIVE_EVENT",
                matched=True,
                requirement_status="STATED" if _pr_cause else "UNKNOWN",
            )

    # 1. Try structural sentence patterns
    for sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        sentence = sentence.strip()
        if not sentence or sentence in _referenced_doc_sentences:
            continue
        stripped = _strip_framing(sentence)
        # Check for reported missing/absent/unperformed items in clause (e.g. "technician reported X was absent")
        m_rep_absent = re.search(
            r"(?:stated|reported|noted|indicated|confirmed|observed)\s+(?:that\s+)?(?:the\s+|an?\s+)?(?P<obj>[a-z0-9\s/-]+?)\s+(?:was|were)\s+(?P<cond>absent|missing|unavailable|not\s+recorded|not\s+completed|not\s+signed|not\s+performed)",
            sentence,
            re.IGNORECASE,
        )
        if m_rep_absent:
            raw_obj = m_rep_absent.group("obj").strip()
            subj = _clean_subject(raw_obj)
            cond_val = m_rep_absent.group("cond").strip().lower()
            if subj and validate_semantic_subject(subj):
                if entities and not any(e.lower() in subj.lower() for e in entities) and ("label" in subj.lower() or "log" in subj.lower() or "valve" in subj.lower() or "record" in subj.lower()):
                    subj = f"{subj} for {entities[0]}"
                clean_subj_cap = subj[0].upper() + subj[1:]
                sem_t = "RECORD" if any(w in subj.lower() for w in ("record", "log", "sheet", "form", "sign-off", "authorization", "approval", "report")) else "ACTIVITY"
                return DeviationInfo(
                    subject=subj,
                    finding_subject=subj,
                    affected_object=clean_subj_cap,
                    affected_process=f"{topic_word(subj).capitalize()} operational control",
                    affected_activity=clean_subj_cap,
                    deviation=f"{clean_subj_cap} — {cond_val}",
                    condition=cond_val,
                    date=date,
                    actor=actors[0] if actors else None,
                    actors=actors,
                    entities=entities,
                    semantic_type=sem_t,
                    matched=True,
                )

        for name, pattern in _CONDITION_PATTERNS:
            match = pattern.search(stripped) or pattern.search(sentence)
            if not match:
                continue
            groups = match.groupdict()
            raw_subj = groups.get("subject", "") or ""
            population_text, raw_subj_no_population = extract_population_clause(raw_subj)
            subj = _clean_subject(raw_subj_no_population)
            cond = groups.get("cond")
            req = groups.get("requirement")

            # Check if raw_subj is an ACTOR (e.g. "Four employees", "Operator", "Three operators")
            if is_actor_noun(raw_subj) or is_actor_noun(subj):
                actor_name = raw_subj.strip()
                _PREP_SPLIT_RE = re.compile(r"\s+(?:before|prior\s+to|during|after|following|on|in|at)\b", re.IGNORECASE)

                if name == "failed_to" and cond:
                    m_verb = re.match(
                        r"^(?P<verb>[a-z]+(?:\s+out)?)\s+(?:the\s+|an?\s+)?(?P<obj>.+)$",
                        cond.strip(),
                        re.IGNORECASE,
                    )
                    if m_verb:
                        v = m_verb.group("verb").lower()
                        raw_obj_clause = m_verb.group("obj").strip()
                        # Split out trailing temporal/prepositional qualifiers from the pure object noun phrase
                        obj_match = _PREP_SPLIT_RE.split(raw_obj_clause, maxsplit=1)
                        target_obj = _clean_subject(obj_match[0].strip())
                        if not target_obj:
                            target_obj = _clean_subject(raw_obj_clause)
                        clean_cap = target_obj[0].upper() + target_obj[1:] if target_obj else "Activity"
                        activity_name = f"{v.capitalize()} {target_obj}"

                        # Form natural condition and deviation strings
                        if v in ("complete", "fill", "fill out", "perform", "conduct", "execute", "sign", "log", "record"):
                            cond_str = "not completed"
                            dev_str = f"Required {target_obj} not completed"
                            sem_type = "RECORD" if any(w in target_obj.lower() for w in ("record", "log", "form", "sheet", "checklist", "entry", "entries", "report", "certificate")) else "ACTIVITY"
                        elif v in ("receive", "get", "obtain"):
                            cond_str = "not received"
                            dev_str = f"{actor_name} did not receive {target_obj}"
                            sem_type = "NOTIFICATION"
                        elif v in ("pass", "satisfy", "meet"):
                            cond_str = "failed"
                            dev_str = f"{clean_cap} failed"
                            sem_type = "ACTIVITY"
                        else:
                            cond_str = f"not {v}ed" if not v.endswith("e") else f"not {v}d"
                            dev_str = f"{actor_name} did not {v} {target_obj}"
                            sem_type = "ACTIVITY"

                        proc_topic = topic_word(target_obj).capitalize()
                        process_name = f"{proc_topic} process and operational control"

                        rel_change = None
                        if "revised" in sentence.lower() or "revision" in sentence.lower():
                            rel_change = f"Revision of the {target_obj}" if not target_obj.lower().startswith("revised") else f"Revision of the {target_obj[8:]}"

                        return DeviationInfo(
                            subject=target_obj,
                            finding_subject=target_obj,
                            affected_object=clean_cap,
                            affected_process=process_name,
                            affected_activity=activity_name,
                            deviation=dev_str,
                            condition=cond_str,
                            requirement=req or f"Applicable {target_obj} requirement",
                            date=date,
                            actor=actor_name,
                            actors=[actor_name] + [a for a in actors if a != actor_name],
                            entities=entities,
                            semantic_type=sem_type,
                            relevant_change=rel_change,
                            matched=True,
                            requirement_status="STATED",
                        )
                    else:
                        raw_cond_clause = cond.strip()
                        cond_match = _PREP_SPLIT_RE.split(raw_cond_clause, maxsplit=1)
                        target_obj = _clean_subject(cond_match[0].strip()) or _clean_subject(raw_cond_clause)
                        clean_cap = target_obj[0].upper() + target_obj[1:] if target_obj else "Activity"
                        return DeviationInfo(
                            subject=target_obj,
                            finding_subject=target_obj,
                            affected_object=clean_cap,
                            affected_process=f"{topic_word(target_obj).capitalize()} operational control",
                            affected_activity=f"{clean_cap} execution",
                            deviation=f"{actor_name} failed to {cond}",
                            condition="failed",
                            requirement=req or f"Applicable {target_obj} requirement",
                            date=date,
                            actor=actor_name,
                            actors=[actor_name] + [a for a in actors if a != actor_name],
                            entities=entities,
                            semantic_type="ACTIVITY",
                            relevant_change=None,
                            matched=True,
                            requirement_status="STATED",
                        )

                if (name == "not_state" or "did not" in sentence.lower() or "not received" in sentence.lower() or "not notified" in sentence.lower() or "was absent" in sentence.lower() or "were absent" in sentence.lower()) and cond:
                    m_did = re.search(r"(?:did\s+not\s+|was\s+not\s+|were\s+not\s+|was\s+|were\s+)(?P<verb>[a-z]+(?:\s+out)?)\s+(?:the\s+|an?\s+)?(?P<obj>[a-z0-9\s-]+?)(?:\s+(?:before|prior\s+to|during|after|following|on|in|at)\b|\.|$)", sentence, re.IGNORECASE)
                    m_absent = re.search(r"(?:the\s+|an?\s+)?(?P<obj>[a-z0-9\s-]+?)\s+(?:was|were)\s+absent", sentence, re.IGNORECASE)
                    if m_absent:
                        target_obj = _clean_subject(m_absent.group("obj"))
                        clean_cap = target_obj[0].upper() + target_obj[1:] if target_obj else "Record"
                        affected_obj = clean_cap
                        process_name = f"{topic_word(target_obj).capitalize()} operational control"
                        activity_name = clean_cap
                        dev_str = f"{clean_cap} was absent"
                        cond_str = "absent"
                        sem_type = "RECORD" if any(w in target_obj.lower() for w in ("record", "log", "sheet", "form", "sign-off", "authorization", "approval")) else "OBJECT"
                    elif m_did:
                        v = m_did.group("verb").lower()
                        target_obj = _clean_subject(m_did.group("obj"))
                        clean_cap = target_obj[0].upper() + target_obj[1:] if target_obj else "Activity"
                        activity_name = f"{v.capitalize()} {target_obj}"

                        if v in ("initial", "sign", "complete", "fill", "fill out", "perform", "conduct", "execute", "log", "record"):
                            cond_str = "not completed"
                            dev_str = f"Required {target_obj} not completed"
                            sem_type = "RECORD" if any(w in target_obj.lower() for w in ("record", "log", "sheet", "form", "checklist", "entry", "entries")) else "ACTIVITY"
                        else:
                            v_past = f"{v}d" if v.endswith("e") else f"{v}ed"
                            cond_str = f"not {v_past}"
                            dev_str = f"{actor_name} did not {v} {target_obj}"
                            sem_type = "ACTIVITY"

                        proc_topic = topic_word(target_obj).capitalize()
                        process_name = f"{proc_topic} process and operational control"
                        affected_obj = clean_cap
                    elif "notified" in cond or "receive notification" in cond or "notification" in sentence.lower():
                        affected_obj = "Notification"
                        process_name = "Controlled document notification and distribution control"
                        activity_name = "Notification distribution"
                        dev_str = f"{actor_name} did not receive notification"
                        cond_str = "not received"
                        sem_type = "NOTIFICATION"
                        target_obj = "notification"
                    elif "trained" in cond or "training" in sentence.lower():
                        affected_obj = "Personnel training"
                        process_name = "Training management and qualification control"
                        activity_name = "Personnel training"
                        dev_str = f"{actor_name} not trained"
                        cond_str = "not trained"
                        sem_type = "ACTIVITY"
                        target_obj = "training"
                    else:
                        m_did = re.search(r"did\s+not\s+(?P<verb>[a-z]+(?:\s+out)?)\s+(?:the\s+|an?\s+)?(?P<obj>[a-z0-9\s-]+?)(?:\s+(?:before|prior\s+to|during|after|following|on|in|at)\b|\.|$)", sentence, re.IGNORECASE)
                        if m_did:
                            v = m_did.group("verb").lower()
                            target_obj = _clean_subject(m_did.group("obj"))
                            clean_cap = target_obj[0].upper() + target_obj[1:] if target_obj else "Activity"
                            activity_name = f"{v.capitalize()} {target_obj}"

                            if v in ("initial", "sign", "complete", "fill", "fill out", "perform", "conduct", "execute", "log", "record"):
                                cond_str = "not completed"
                                dev_str = f"Required {target_obj} not completed"
                                sem_type = "RECORD" if any(w in target_obj.lower() for w in ("record", "log", "sheet", "form", "checklist", "entry", "entries")) else "ACTIVITY"
                            else:
                                v_past = f"{v}d" if v.endswith("e") else f"{v}ed"
                                cond_str = f"not {v_past}"
                                dev_str = f"{actor_name} did not {v} {target_obj}"
                                sem_type = "ACTIVITY"

                            proc_topic = topic_word(target_obj).capitalize()
                            process_name = f"{proc_topic} process and operational control"
                            affected_obj = clean_cap
                        else:
                            clean_cond = _clean_subject(cond)
                            affected_obj = clean_cond[0].upper() + clean_cond[1:] if clean_cond else "Activity"
                            target_obj = clean_cond
                            process_name = f"{topic_word(clean_cond).capitalize()} operational control"
                            activity_name = f"{clean_cond[0].upper() + clean_cond[1:]} activity"
                            dev_str = f"{actor_name} — not {cond}"
                            cond_str = f"not {cond}"
                            sem_type = "ACTIVITY"

                    return DeviationInfo(
                        subject=target_obj,
                        finding_subject=target_obj,
                        affected_object=affected_obj,
                        affected_process=process_name,
                        affected_activity=activity_name,
                        deviation=dev_str,
                        condition=cond_str,
                        requirement=req or f"Applicable {target_obj} requirement",
                        date=date,
                        actor=actor_name,
                        actors=[actor_name] + [a for a in actors if a != actor_name],
                        entities=entities,
                        semantic_type=sem_type,
                        relevant_change=None,
                        matched=True,
                        requirement_status="STATED",
                    )

                if name == "preventing" and groups.get("subject"):
                    target_subj = _clean_subject(groups["subject"])
                    cause_desc = groups.get("cause", "technical failure").strip()
                    clean_cap = target_subj[0].upper() + target_subj[1:]
                    affected_obj = clean_cap
                    return DeviationInfo(
                        subject=affected_obj,
                        finding_subject=affected_obj,
                        affected_object=affected_obj,
                        affected_process=f"{topic_word(target_subj).capitalize()} operational control",
                        affected_activity=f"{clean_cap} execution",
                        deviation=f"{clean_cap} prevented by {cause_desc}",
                        condition="prevented",
                        date=date,
                        actors=actors,
                        entities=entities,
                        semantic_type="CONTROL",
                        matched=True,
                    )

                if name == "actor_action" and cond:
                    m_obj = groups.get("obj")
                    target_obj = _clean_subject(m_obj) if m_obj else "equipment"
                    clean_cap = target_obj[0].upper() + target_obj[1:]
                    affected_obj = f"{clean_cap} operation and measurement" if any(w in target_obj.lower() for w in ("scale", "balance", "meter", "gauge", "instrument")) else f"{clean_cap} operational control"
                    process_name = f"{topic_word(target_obj).capitalize()} operation and measurement control"
                    activity_name = f"{clean_cap} measurement and adherence"
                    dev_str = f"{actor_name} — {cond}"
                    cond_str = cond
                    return DeviationInfo(
                        subject=affected_obj,
                        finding_subject=affected_obj,
                        affected_object=affected_obj,
                        affected_process=process_name,
                        affected_activity=activity_name,
                        deviation=dev_str,
                        condition=cond_str,
                        date=date,
                        actor=actor_name,
                        actors=[actor_name] + [a for a in actors if a != actor_name],
                        entities=entities,
                        semantic_type="ACTIVITY",
                        matched=True,
                    )

                if name == "forgot_to" and cond:
                    m_item = re.search(r"(?:initial|sign|complete|fill|record|log|perform)\s+(?:the\s+|an?\s+)?(?P<item>[a-z0-9\s-]+?)(?:\s*\.?$)", cond, re.IGNORECASE)
                    item_name = _clean_subject(m_item.group("item")) if m_item else "log sheet"
                    clean_cap = item_name[0].upper() + item_name[1:]
                    affected_obj = f"{clean_cap} verification" if not item_name.lower().endswith("verification") else clean_cap
                    return DeviationInfo(
                        subject=affected_obj,
                        finding_subject=affected_obj,
                        affected_object=affected_obj,
                        affected_process=f"{topic_word(item_name).capitalize()} completion and verification control",
                        affected_activity=f"{clean_cap} documentation",
                        deviation=f"{clean_cap} — {cond}",
                        condition=cond,
                        date=date,
                        actor=actor_name if is_actor_noun(raw_subj) else None,
                        actors=actors,
                        entities=entities,
                        semantic_type="RECORD",
                        matched=True,
                    )

                if name == "system_operated":
                    clean_cap = subj[0].upper() + subj[1:] if subj else "System"
                    v = groups.get("verb", "operation")
                    affected_obj = f"{clean_cap} {v}"
                    return DeviationInfo(
                        subject=affected_obj,
                        finding_subject=affected_obj,
                        affected_object=affected_obj,
                        affected_process=f"{clean_cap} access and operational control",
                        affected_activity=f"{clean_cap} {v}",
                        deviation=f"{clean_cap} — {v} {cond}",
                        condition=f"{v} {cond}",
                        date=date,
                        actors=actors,
                        entities=entities,
                        semantic_type="ACTIVITY",
                        matched=True,
                    )

                if name == "causative_verb":
                    clean_subj_cap = subj[0].upper() + subj[1:]
                    affected_obj = f"{clean_subj_cap} control"
                    return DeviationInfo(
                        subject=affected_obj,
                        finding_subject=affected_obj,
                        affected_object=affected_obj,
                        affected_process=f"{topic_word(subj).capitalize()} operational control",
                        affected_activity=f"{clean_subj_cap} review",
                        deviation=f"{clean_subj_cap} — {cond}",
                        condition=cond or "deviation",
                        date=date,
                        actors=actors,
                        entities=entities,
                        semantic_type="FINANCIAL" if any(w in subj.lower() for w in ("invoice", "payment", "fee", "cost")) else "ACTIVITY",
                        matched=True,
                    )

                if name == "missed" and groups.get("obj"):
                    target_obj = _clean_subject(groups["obj"])
                    clean_cap = target_obj[0].upper() + target_obj[1:]
                    affected_obj = f"{clean_cap} execution" if not target_obj.lower().endswith("execution") else clean_cap
                    return DeviationInfo(
                        subject=affected_obj,
                        finding_subject=affected_obj,
                        affected_object=affected_obj,
                        affected_process=f"{topic_word(target_obj).capitalize()} process and compliance control",
                        affected_activity=f"{clean_cap} activity",
                        deviation=f"{clean_cap} missed",
                        condition="missed",
                        date=date,
                        actor=actor_name if is_actor_noun(raw_subj) else None,
                        actors=actors,
                        entities=entities,
                        semantic_type="RECORD" if any(w in target_obj.lower() for w in ("log", "sheet", "record", "checklist", "check")) else "ACTIVITY",
                        matched=True,
                    )

                if name == "bypassed" and groups.get("obj"):
                    target_obj = _clean_subject(groups["obj"])
                    clean_cap = target_obj[0].upper() + target_obj[1:]
                    affected_obj = f"{clean_cap} compliance" if not target_obj.lower().endswith("compliance") else clean_cap
                    return DeviationInfo(
                        subject=affected_obj,
                        finding_subject=affected_obj,
                        affected_object=affected_obj,
                        affected_process=f"{topic_word(target_obj).capitalize()} operational control",
                        affected_activity=f"{clean_cap} enforcement",
                        deviation=f"{clean_cap} bypassed",
                        condition="bypassed",
                        date=date,
                        actor=actor_name if is_actor_noun(raw_subj) else None,
                        actors=actors,
                        entities=entities,
                        semantic_type="CONTROL",
                        matched=True,
                    )

                if name == "metric_drift":
                    clean_subj_cap = subj[0].upper() + subj[1:]
                    affected_obj = f"{clean_subj_cap} control"
                    return DeviationInfo(
                        subject=affected_obj,
                        finding_subject=affected_obj,
                        affected_object=affected_obj,
                        affected_process=f"{topic_word(subj).capitalize()} monitoring and process control",
                        affected_activity=f"{clean_subj_cap} monitoring",
                        deviation=f"{clean_subj_cap} — {cond}",
                        condition=cond or "out of range",
                        date=date,
                        actors=actors,
                        entities=entities,
                        semantic_type="PARAMETER",
                        matched=True,
                    )

                if name == "bare_failed":
                    m_bare = re.search(r"failed\s+(?:the\s+|an?\s+)?(?P<obj>[a-z][\w\s-]*?)(?:\s+because|\s+after|\s*\.|$)", sentence, re.IGNORECASE)
                    obj_name = _clean_subject(m_bare.group("obj")) if m_bare else "inspection"
                    clean_obj = obj_name[0].upper() + obj_name[1:]
                    affected_obj = f"{clean_obj} execution"
                    process_name = f"{topic_word(obj_name).capitalize()} process and qualification control"
                    activity_name = f"{clean_obj} execution"
                    dev_str = f"{clean_obj} failed"
                    cond_str = "failed"
                    return DeviationInfo(
                        subject=affected_obj,
                        finding_subject=affected_obj,
                        affected_object=affected_obj,
                        affected_process=process_name,
                        affected_activity=activity_name,
                        deviation=dev_str,
                        condition=cond_str,
                        requirement="Applicable inspection standard" if "inspection" in obj_name.lower() else None,
                        date=date,
                        actor=actor_name,
                        actors=[actor_name] + [a for a in actors if a != actor_name],
                        entities=entities,
                        semantic_type="ACTIVITY",
                        matched=True,
                    )

            # Non-actor path
            matched_ent = groups.get("entity")
            if matched_ent and re.match(r"^[A-Z0-9-]+$", matched_ent.strip()) and matched_ent not in subj:
                subj = f"{subj} for equipment {matched_ent}"
            elif entities and not any(e in subj for e in entities) and ("label" in subj.lower() or "log" in subj.lower()):
                subj = f"{subj} for {entities[0]}"

            if not validate_semantic_subject(subj):
                continue

            if name == "not_state" and cond:
                cond = f"not {cond.strip()}"
            elif name == "deviated_from" and req:
                cond = f"deviated from {req.strip()}"
            elif name == "missing_from":
                cond = "missing"
            elif name == "outside_scope" and cond:
                verb = (groups.get("verb") or "operated").strip()
                cond = f"{verb} outside its {cond.strip()}"

            clean_subj_cap = subj[0].upper() + subj[1:]
            topic_str = topic_word(subj).capitalize()
            if name == "outside_scope":
                proc_str = f"Validated {subj.lower()} operation" if not subj.lower().startswith("validated") else f"{clean_subj_cap} control"
                sem_type = "EQUIPMENT"
            elif any(w in subj.lower() for w in ("record", "log", "certificate", "sheet", "form", "report", "label")):
                proc_str = f"{topic_str} record control"
                sem_type = "RECORD"
            elif "notification" in subj.lower():
                proc_str = "Controlled document notification"
                sem_type = "NOTIFICATION"
            else:
                proc_str = f"{topic_str} operational process"
                sem_type = "OBJECT"

            return DeviationInfo(
                subject=subj,
                finding_subject=subj,
                affected_object=clean_subj_cap,
                affected_process=proc_str,
                affected_activity=clean_subj_cap,
                deviation=f"{subj} — {cond or 'condition unverified'}",
                condition=cond,
                requirement=req,
                date=date,
                actor=actors[0] if actors else None,
                actors=actors,
                entities=entities,
                semantic_type=sem_type,
                matched=True,
                occurrence_population=population_text,
            )

        for cond_label, pattern in _PREFIX_CONDITION_PATTERNS:
            match = pattern.match(stripped)
            if not match:
                continue
            _prefix_population, _prefix_subj_no_pop = extract_population_clause(match.group("subject"))
            subj = _clean_subject(_prefix_subj_no_pop)
            if not validate_semantic_subject(subj):
                continue
            clean_subj_cap = subj[0].upper() + subj[1:]
            return DeviationInfo(
                subject=subj,
                finding_subject=subj,
                affected_object=clean_subj_cap,
                affected_process=f"{topic_word(subj).capitalize()} operational process",
                affected_activity=clean_subj_cap,
                deviation=f"{subj} — {cond_label}",
                condition=cond_label,
                date=date,
                actor=actors[0] if actors else None,
                actors=actors,
                entities=entities,
                semantic_type="OBJECT",
                matched=True,
                occurrence_population=_prefix_population,
            )

        for cond_label, pattern in _SHORT_DEVIATION_PATTERNS:
            match = pattern.match(stripped)
            if not match:
                continue
            groups = match.groupdict()
            _short_population, _short_subj_no_pop = extract_population_clause(groups.get("subject", ""))
            subj = _clean_subject(_short_subj_no_pop)
            if not subj or not validate_semantic_subject(subj) or is_actor_noun(subj):
                continue
            clean_subj_cap = subj[0].upper() + subj[1:]
            cond_str = groups.get("cond") or (f"{groups.get('verb', '')} {groups.get('mod', '')}".strip())
            sem_t = "EVENT_SEQUENCE_CONTROL" if "without" in cond_str else "OBJECT"
            return DeviationInfo(
                subject=subj,
                finding_subject=subj,
                affected_object=clean_subj_cap,
                affected_process=f"{topic_word(subj).capitalize()} operational process",
                affected_activity=clean_subj_cap,
                deviation=f"{subj} — {cond_str}",
                condition=cond_str,
                date=date,
                actor=actors[0] if actors else None,
                actors=actors,
                entities=entities,
                semantic_type=sem_t,
                matched=True,
                occurrence_population=_short_population,
            )

        # Framing was stripped and what's left is a short, usable noun phrase
        if stripped != sentence and len(stripped) < 140:
            subj = _clean_subject(stripped)
            if validate_semantic_subject(subj) and not is_actor_noun(subj):
                clean_subj_cap = subj[0].upper() + subj[1:]
                return DeviationInfo(
                    subject=subj,
                    finding_subject=subj,
                    affected_object=clean_subj_cap,
                    affected_process=f"{topic_word(subj).capitalize()} operational process",
                    affected_activity=clean_subj_cap,
                    deviation=f"{subj} — nonconforming",
                    condition="nonconforming",
                    date=date,
                    actor=actors[0] if actors else None,
                    actors=actors,
                    entities=entities,
                    semantic_type="OBJECT",
                    matched=True,
                )

    # 2. Structural entity-noun resolution FIRST
    activity_subj = None
    if entities:
        activity_subj = _entity_noun_phrase(text, entities[0])
    if not activity_subj:
        masked_text = _mask_referenced_document_spans(text)
        act = _extract_activity_from_reported_finding(masked_text)
        if act and act != "process compliance":
            activity_subj = act
            if entities and not any(e in activity_subj for e in entities):
                activity_subj = f"{activity_subj} for {entities[0]}"

    if activity_subj and activity_subj != "process compliance":
        clean_act_cap = activity_subj[0].upper() + activity_subj[1:]
        return DeviationInfo(
            subject=activity_subj,
            finding_subject=activity_subj,
            affected_object=clean_act_cap,
            affected_process="training management" if "training" in text.lower() else f"{topic_word(clean_act_cap).capitalize()} operational process",
            affected_activity=clean_act_cap,
            deviation=f"{activity_subj} — {_degraded_condition_filler(activity_subj)}",
            condition=_degraded_condition_filler(activity_subj),
            date=date,
            actor=actors[0] if actors else None,
            actors=actors,
            entities=entities,
            semantic_type="ACTIVITY",
            matched=True,
        )

    # Non-actionable input fallback: do NOT manufacture false "process compliance" actionability
    return DeviationInfo(
        subject=None,
        finding_subject=None,
        affected_object=None,
        affected_process=None,
        affected_activity=None,
        deviation=None,
        condition=None,
        date=date,
        actors=actors,
        entities=entities,
        semantic_type="NON_ACTIONABLE",
        matched=False,
    )


# ---------------------------------------------------------------------------
# Broadened structural entity recovery (Defect 3).
#
# The resolver already handled canonical active/passive SVO shapes. The blind
# audit's 8 entity-collapse cases all fell OUTSIDE those shapes, but not
# randomly so -- they clustered into three grammatical families, each of
# which is recoverable by a general rule rather than a per-sentence pattern:
#
#   A. AGENTLESS PASSIVE. "<NP> was <past-participle> ..." with no "by ..."
#      agent. The grammatical subject of a passive IS the semantic patient,
#      so no agent phrase is needed for the extraction to succeed. When the
#      passive subject is itself an ACTOR ("an employee was observed not
#      wearing X"), the patient lives in the participial/infinitival
#      complement, so we descend into it instead of giving up.
#   B. ABSENCE / EXISTENTIAL NEGATION. "<NP> could not be located",
#      "no <NP> exists for ...", "<NP> was missing". The noun phrase is
#      still a perfectly good entity -- it is merely paired with an absence
#      predicate rather than an action predicate.
#   C. NOMINALIZATION. "non-adherence to <NP>", "failure to <verb> <NP>",
#      "the <verb>ing of <NP>", "lack/absence/omission of <NP>". The head
#      entity sits inside the nominalized construction; the outer clause is
#      not unparseable, it just has no finite verb to key on.
#
# Each family is one rule over an open noun-phrase slot, so it generalizes
# to any domain vocabulary. None of them enumerates a sample sentence.
# ---------------------------------------------------------------------------

_NP = r"(?P<np>[A-Za-z][A-Za-z0-9\s/&._-]{2,50}?)"
_NP_END = (
    r"(?=\s+(?:for|of|in|on|at|during|by|from|with|to|as|that|which|when|while|"
    r"prior|before|after|contrary|required|was|were|is|are|has|have|had|could|"
    r"would|should|will|shall|remain|remains|remained|and|or|but|despite|"
    r"notwithstanding)\b|[,.;:]|$)"
)

# A. agentless passive whose subject is an actor -> descend into the
#    participial / infinitival complement for the real patient.
_PASSIVE_ACTOR_COMPLEMENT_RE = re.compile(
    r"\b(?:was|were|is|are|been)\s+[a-z]+(?:ed|en)\s+"
    r"(?:to\s+be\s+)?(?:not\s+|never\s+|without\s+)?"
    r"(?:[a-z]+ing|to\s+[a-z]+|wearing|using|following)\s+"
    r"(?:the\s+|an?\s+|any\s+|their\s+|his\s+|her\s+|its\s+)?" + _NP + _NP_END,
    re.IGNORECASE,
)
# A'. agentless passive with a non-actor subject and ANY past participle
#     (the previous rule required a closed verb list).
_AGENTLESS_PASSIVE_RE = re.compile(
    r"^(?:the\s+|an?\s+|any\s+)?" + _NP.replace("(?P<np>", "(?P<np>") +
    r"\s+(?:was|were|is|are|has\s+been|have\s+been|had\s+been)\s+"
    r"(?:not\s+|never\s+)?(?P<part>[a-z]+(?:ed|en))\b",
    re.IGNORECASE,
)
# B. absence / existential negation.
_ABSENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:no|No)\s+" + _NP + r"\s+(?:exist|exists|existed|was|were|is|are|could|had)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bthere\s+(?:was|were|is|are)\s+no\s+" + _NP + _NP_END,
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:the\s+|an?\s+)?" + _NP +
        r"\s+(?:could|can|would)\s+not\s+be\s+"
        r"(?:located|found|produced|retrieved|provided|identified|traced|obtained|"
        r"presented|verified|evidenced|demonstrated|substantiated|reconciled)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:the\s+|an?\s+)?" + _NP +
        r"\s+(?:was|were|is|are|remained|remains)\s+(?:not\s+)?"
        r"(?:missing|absent|unavailable|incomplete|outstanding|blank|empty|"
        r"not\s+available|not\s+present|not\s+on\s+file|nowhere\s+to\s+be\s+found)\b",
        re.IGNORECASE,
    ),
]
# C. nominalizations.
_NOMINALIZATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\bfailure\s+to\s+[a-z]+\s+(?:the\s+|an?\s+|any\s+)?" + _NP + _NP_END,
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:non[-\s]?(?:adherence|compliance|conformance|conformity|completion|"
        r"performance|availability|submission|execution)|lack|absence|omission|"
        r"failure|inadequacy|insufficiency|deficiency|deviation|discrepancy)\s+"
        r"(?:to|with|of|in|for|from)\s+(?:the\s+|an?\s+|any\s+)?" + _NP + _NP_END,
        re.IGNORECASE,
    ),
    re.compile(
        r"\bthe\s+[a-z]+ing\s+of\s+(?:the\s+|an?\s+)?" + _NP + _NP_END,
        re.IGNORECASE,
    ),
]

_ABSENCE_CONDITION_RE = re.compile(
    r"\b(?:could\s+not\s+be\s+[a-z]+|was\s+missing|were\s+missing|was\s+absent|"
    r"were\s+absent|was\s+not\s+available|were\s+not\s+available|does\s+not\s+exist|"
    r"do\s+not\s+exist|was\s+incomplete|were\s+incomplete|was\s+unavailable)\b",
    re.IGNORECASE,
)


def _accept_entity_candidate(raw: str | None) -> str | None:
    """Clean and validate a candidate noun phrase; reject actors, clauses,
    pronouns and empty fragments."""
    if not raw:
        return None
    cand = _clean_subject(raw.strip())
    if not cand:
        return None
    cand = strip_leading_article(cand) or cand
    cand = cand.strip(" .,;:")
    if not cand or len(cand) < 3:
        return None
    if not validate_semantic_subject(cand):
        return None
    if is_actor_noun(cand):
        return None
    return cand


def recover_entity_structurally(text: str) -> tuple[str, str | None] | None:
    """Broadened grammatical recovery of the affected entity for sentence
    shapes outside the canonical SVO/passive patterns.

    Returns (entity_noun_phrase, condition_or_None) or None if no noun
    phrase could be isolated. Never returns a generic placeholder -- an
    honest None is the only failure mode.
    """
    if not text or not text.strip():
        return None
    for sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        sentence = _strip_framing(sentence.strip())
        if not sentence:
            continue
        # B. absence / existential negation -- checked first because an
        # absence predicate is the most specific signal available.
        for pattern in _ABSENCE_PATTERNS:
            m = pattern.search(sentence)
            if m:
                cand = _accept_entity_candidate(m.group("np"))
                if cand:
                    cond_m = _ABSENCE_CONDITION_RE.search(sentence)
                    return cand, (cond_m.group(0).lower() if cond_m else "not available")
        # C. nominalization -- the head entity inside the nominalized phrase.
        for pattern in _NOMINALIZATION_PATTERNS:
            m = pattern.search(sentence)
            if m:
                cand = _accept_entity_candidate(m.group("np"))
                if cand:
                    return cand, "not adhered to"
        # A. agentless passive with an actor subject -> descend to complement.
        m = _PASSIVE_ACTOR_COMPLEMENT_RE.search(sentence)
        if m:
            cand = _accept_entity_candidate(m.group("np"))
            if cand:
                return cand, "not used as required"
        # A'. agentless passive with a non-actor subject, ANY participle.
        m = _AGENTLESS_PASSIVE_RE.match(sentence)
        if m:
            cand = _accept_entity_candidate(m.group("np"))
            if cand:
                return cand, m.group("part").lower()
    return None


_STOPWORD_HEADS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "they", "there",
    "during", "in", "on", "at", "for", "of", "and", "or", "was", "were", "is",
    "are", "had", "has", "have", "no", "not", "any", "some", "all", "which",
    "who", "when", "while", "audit", "review", "observation", "finding",
}


def best_partial_noun_phrase(text: str) -> str | None:
    """Last-resort PARTIAL fragment: the longest plausible noun-phrase-like
    span in the text. Deliberately imperfect -- its purpose is to preserve
    the reader's connection to the actual finding while being explicitly
    flagged as low-confidence, which is strictly better than substituting a
    confident-sounding generic placeholder."""
    if not text or not text.strip():
        return None
    best: str | None = None
    for m in re.finditer(
        r"\b(?:the|a|an)\s+((?:[a-z][a-z0-9-]*\s+){0,3}[a-z][a-z0-9-]*)\b",
        text, re.IGNORECASE,
    ):
        frag = m.group(1).strip()
        head = frag.split()[0].lower()
        if head in _STOPWORD_HEADS:
            continue
        # Truncate the fragment at the first finite/participial verb or
        # complementizer: a noun phrase ends where the predicate begins, so
        # "dispensing radiographer adjudged that" must become
        # "dispensing radiographer".
        kept: list[str] = []
        for tok in frag.split():
            low = tok.lower()
            if low in _STOPWORD_HEADS or low in {"that", "which", "who", "whose", "whom"}:
                break
            if len(kept) >= 1 and re.search(r"(?:ed|ing)$", low) and low not in {
                "recording", "reading", "training", "briefing", "meeting", "finding",
                "building", "setting", "coating", "casing", "housing", "bearing",
                "feed", "shed", "bed", "record", "lead",
            }:
                break
            kept.append(tok)
        frag = " ".join(kept).strip()
        if not frag:
            continue
        cand = _accept_entity_candidate(frag)
        if cand and (best is None or len(cand) > len(best)):
            best = cand
    if best:
        return best
    # Fall back to a capitalized/identifier-looking token (e.g. "QC-REF-02").
    for m in re.finditer(r"\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)+\b", text):
        return m.group(0)
    return None


def resolve_referenced_documents(finding_text: str) -> list[dict]:
    """Public wrapper returning referenced-but-unavailable documents as
    plain dicts (document_type, raw_span) for the canonical state builder --
    the single authoritative producer of this metadata (Referenced-Evidence
    Boundary). Never returns content; a document's type/name is the only
    thing this can ever establish."""
    return [
        {"document_type": doc_type, "raw_span": span}
        for doc_type, span in detect_referenced_unavailable_documents(finding_text)
    ]


def resolve_deviation(finding_text: str, fact_claims: list[str] | None = None) -> DeviationInfo:
    """Best-effort semantic subject/condition resolution for a finding.
    Guaranteed to return a valid noun phrase subject when actionable."""
    if not finding_text or not finding_text.strip():
        return DeviationInfo(subject=None, matched=False, semantic_type="NON_ACTIONABLE")

    candidates = [finding_text] + [c for c in (fact_claims or []) if c]
    result = DeviationInfo(subject=None, matched=False, semantic_type="NON_ACTIONABLE")
    for candidate in candidates:
        result = extract_semantic_subject(candidate)
        if result.matched and result.subject and validate_semantic_subject(result.subject):
            break

    if not result.matched or not result.subject or not validate_semantic_subject(result.subject):
        activity_subj = _extract_activity_from_reported_finding(_mask_referenced_document_spans(finding_text))
        if activity_subj and activity_subj != "process compliance":
            result.subject = activity_subj
            result.finding_subject = activity_subj
            result.affected_object = activity_subj
            result.deviation = f"{activity_subj} — {_degraded_condition_filler(activity_subj)}"
            result.matched = True
            result.semantic_type = "ACTIVITY"
        else:
            # F2 — Pronoun-actor role recovery.
            # UNRESOLVED ACTOR != UNACTIONABLE FINDING.
            # A sentence like "During audit, they had not completed the log."
            # contains a valid auditable event (not-completed) and a clear
            # object (the log) even though the actor ("they") is unresolved.
            # The pattern engine correctly rejects the pronoun as a subject,
            # but must NOT discard the finding: recover EVENT + OBJECT from
            # the predicate clause, use the object as the subject (the
            # affected thing), and leave actor = None / UNRESOLVED.
            # Domain-agnostic: "they", "it", "someone", "personnel", and any
            # implicit-agent construction all route here.
            _PRONOUN_PREDICATE_RE = re.compile(
                r"^(?:during\s+\S+\s*,\s*)?"
                r"(?:they|it|he|she|we|someone|personnel|"
                r"the\s+(?:operator|technician|employee|staff|analyst|supervisor|user)s?)"
                r"\s+(?:had\s+not|has\s+not|have\s+not|did\s+not|does\s+not|"
                r"was\s+not|were\s+not|could\s+not|should\s+not|would\s+not|"
                r"failed\s+to)\s+"
                r"(?P<verb>[a-z]+(?:\s+out)?)\s+"
                r"(?:the\s+|an?\s+)?(?P<obj>[a-z][a-z0-9\s/-]{1,50}?)\s*(?:\.|$)",
                re.IGNORECASE,
            )
            _f2_recovered = False
            for _f2_sent in _SENTENCE_SPLIT_RE.split(finding_text.strip()):
                _f2_sent = _f2_sent.strip()
                if not _f2_sent:
                    continue
                _f2_m = (
                    _PRONOUN_PREDICATE_RE.match(_strip_framing(_f2_sent))
                    or _PRONOUN_PREDICATE_RE.match(_f2_sent)
                )
                if _f2_m:
                    _f2_verb = (_f2_m.group("verb") or "").strip().lower()
                    _f2_obj = _clean_subject((_f2_m.group("obj") or "").strip())
                    if _f2_obj and validate_semantic_subject(_f2_obj) and is_actor_noun(_f2_obj) == False:
                        _f2_cap = _f2_obj[0].upper() + _f2_obj[1:]
                        _f2_cond = f"not {_f2_verb}ed" if _f2_verb else "not completed"
                        result.subject = _f2_obj
                        result.finding_subject = _f2_obj
                        result.affected_object = _f2_cap
                        result.affected_process = (
                            f"{topic_word(_f2_cap).capitalize()} operational process"
                        )
                        result.affected_activity = _f2_cap
                        result.deviation = f"{_f2_cap} — {_f2_cond}"
                        result.condition = _f2_cond
                        result.actor = None  # UNRESOLVED — never invent the actor
                        result.semantic_type = "OBJECT"
                        result.matched = True
                        _f2_recovered = True
                        break
            if not _f2_recovered:
                result.matched = False
                result.subject = None
                result.finding_subject = None
                result.semantic_type = "NON_ACTIONABLE"

    # Universal Syntactic Head-Noun & SVO Recovery:
    # If the structured regexes did not match, parse the main independent clause
    # to extract the grammatical patient/theme noun phrase (affected object).
    if not result.matched or not result.subject or not validate_semantic_subject(result.subject):
        for sent in _SENTENCE_SPLIT_RE.split(finding_text.strip()):
            sent = _strip_framing(sent.strip())
            if not sent or len(sent.split()) < 3:
                continue
            # 1. Passive / Stative / Intransitive Pattern: <OBJECT> (was/were/tripped/leaked/failed/showed/...) <CONDITION>
            m_syn = re.match(
                r"^(?:the\s+|an?\s+)?(?P<obj>[a-zA-Z0-9][a-zA-Z0-9\s/-]{2,50}?)\s+"
                r"(?P<verb>was|were|tripped|leaked|failed|showed|displayed|exhibited|occurred|billed|approved|deleted|deployed|implemented)\b\s*"
                r"(?P<rest>[^.;]+)?",
                sent,
                re.IGNORECASE,
            )
            if m_syn:
                cand_obj = _clean_subject(m_syn.group("obj"))
                if cand_obj and validate_semantic_subject(cand_obj) and not is_actor_noun(cand_obj):
                    cand_cap = cand_obj[0].upper() + cand_obj[1:]
                    verb_str = m_syn.group("verb").lower()
                    rest_str = (m_syn.group("rest") or "").strip()
                    cond_str = f"{verb_str} {rest_str}".strip() if rest_str else verb_str
                    result.subject = cand_obj
                    result.finding_subject = cand_obj
                    result.affected_object = cand_cap
                    result.affected_process = f"{topic_word(cand_cap).capitalize()} operational process"
                    result.affected_activity = cand_cap
                    result.deviation = f"{cand_cap} — {cond_str}"
                    result.condition = cond_str
                    result.semantic_type = "OBJECT"
                    result.matched = True
                    break

            # 2. Active Transitive SVO Pattern: <ACTOR/SUBJECT> <TRANSITIVE_VERB> <PATIENT/OBJECT> <MODIFIER/CONTEXT>
            # e.g., "Technicians wiped optical lenses using cotton swabs" -> object = "optical lenses", condition = "wiped using cotton swabs"
            m_svo = re.match(
                r"^(?:the\s+|an?\s+)?(?P<subj>[a-zA-Z0-9\s-]{2,30}?)\s+"
                r"(?P<verb>[a-zA-Z]+(?:ed|s|d))\s+"
                r"(?:the\s+|an?\s+)?(?P<obj>[a-zA-Z0-9][a-zA-Z0-9\s/-]{2,40}?)\s*"
                r"(?P<mod>(?:using|with|without|contrary\s+to|instead\s+of|rather\s+than|before|after|outside)\b[^.;]+)?(?:\.|$)",
                sent,
                re.IGNORECASE,
            )
            if m_svo:
                actor_cand = m_svo.group("subj").strip()
                cand_obj = _clean_subject(m_svo.group("obj"))
                if cand_obj and validate_semantic_subject(cand_obj) and not is_actor_noun(cand_obj):
                    cand_cap = cand_obj[0].upper() + cand_obj[1:]
                    verb_str = m_svo.group("verb").lower()
                    mod_str = (m_svo.group("mod") or "").strip()
                    cond_str = f"{verb_str} {mod_str}".strip() if mod_str else verb_str
                    result.subject = cand_obj
                    result.finding_subject = cand_obj
                    result.affected_object = cand_cap
                    result.affected_process = f"{topic_word(cand_cap).capitalize()} operational process"
                    result.affected_activity = cand_cap
                    result.deviation = f"{cand_cap} — {cond_str}"
                    result.condition = cond_str
                    result.actor = actor_cand if is_actor_noun(actor_cand) else None
                    result.semantic_type = "OBJECT"
                    result.matched = True
                    break

    # Defect 3 — broadened structural recovery, then honest failure.
    # Only reached when every earlier resolver produced nothing usable.
    if not result.matched or not result.subject or not validate_semantic_subject(result.subject):
        # Try the VERIFIED fact claims as well as the raw finding: when the
        # finding opens with a belief or a counterfactual, the concrete
        # entity usually lives in the factual sentence, not the first one.
        _recovered = None
        for _cand in candidates:
            _recovered = recover_entity_structurally(_cand)
            if _recovered:
                break
        if _recovered:
            _ent, _cond = _recovered
            _ent_cap = _ent[0].upper() + _ent[1:]
            result.subject = _ent
            result.finding_subject = _ent
            result.affected_object = _ent_cap
            result.affected_process = f"{topic_word(_ent_cap).capitalize()} operational process"
            result.affected_activity = _ent_cap
            result.condition = _cond or "UNKNOWN"
            result.deviation = f"{_ent_cap} — {_cond}" if _cond else _ent_cap
            result.semantic_type = result.semantic_type if result.semantic_type not in (None, "NON_ACTIONABLE") else "OBJECT"
            result.matched = True
            result.extraction_confidence = "RESOLVED"
            result.subject_unresolved = False
        else:
            # Never fabricate. Keep the best real fragment we have and mark
            # it as low-confidence so the invariant layer and the
            # investigation planner can both see the uncertainty.
            _partial = best_partial_noun_phrase(finding_text)
            result.partial_subject_fragment = _partial
            result.subject_unresolved = True
            result.extraction_confidence = "PARTIAL" if _partial else "UNRESOLVED"
            if _partial:
                result.subject = _partial
                result.finding_subject = _partial
                result.affected_object = _partial[0].upper() + _partial[1:]
                result.matched = True

    # F4a — Universal requirement_status classification:
    # VERIFIED: Normative document/contract/revision/standard explicitly referenced.
    # STATED: Explicit normative constraint words, deontic modals, or violation predicates asserted
    #         ("required", "mandatory", "must", "exceeding", "limit", "target", "without", "before",
    #          "unauthorized", "unreconciled", "discrepancy", "overdue", "expired", "duplicate", "bridging", "leak").
    # UNKNOWN: Purely descriptive observation without normative assertion.
    if result.matched and getattr(result, "requirement_status", "UNKNOWN") == "UNKNOWN":
        _VERIFIED_REQ_RE = re.compile(
            r"\b(?:SOP-[A-Z0-9-]+|Revision\s+\d+|Rev\s+\d+|ISO\s+\d+|contract|agreement|manifest|specification|tolerance|spec)\b",
            re.IGNORECASE,
        )
        _STATED_REQ_RE = re.compile(
            r"\b(?:required|mandatory|obligatory|compulsory|must|shall|"
            r"is\s+required|are\s+required|was\s+required|were\s+required|"
            r"without\s+(?:the\s+)?(?:required\s+|signed\s+|prior\s+)?|"
            r"exceeding|target|limit|threshold|before|prior\s+to|unauthorized|unreconciled|"
            r"duplicate|overpayment|overpaid|bridging|short\s+circuit|open\s+circuit|padlocked|locked\s+shut|bypassed)\b",
            re.IGNORECASE,
        )
        if _VERIFIED_REQ_RE.search(finding_text):
            result.requirement_status = "VERIFIED"
        elif _STATED_REQ_RE.search(finding_text):
            result.requirement_status = "STATED"

    result.date = result.date or extract_date(finding_text)
    result.actors = result.actors or extract_actors(finding_text)
    result.actor = result.actor or (result.actors[0] if result.actors else None)
    result.entities = result.entities or extract_entities(finding_text)
    return result
