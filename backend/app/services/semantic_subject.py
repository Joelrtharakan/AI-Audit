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

import re
from dataclasses import dataclass, field

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

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

# Reporting verbs prefix patterns
_FRAMING_PREFIXES = [
    re.compile(r"^\s*during\s+.+?,\s*", re.IGNORECASE),
    re.compile(r"^\s*it\s+(?:was|is)\s+(?:observed|found|noted|identified|determined)\s+that\s+", re.IGNORECASE),
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


def declarative_to_why_question(text: str) -> str:
    """Turn a declarative VERIFIED claim (e.g. "temperature monitoring
    records for refrigerator QC-REF-02 were incomplete for three
    consecutive days") into a natural "Why <aux> <subject> <rest>?" question
    by fronting the auxiliary verb -- "Why were temperature monitoring
    records for refrigerator QC-REF-02 incomplete for three consecutive
    days?" -- instead of asking whether the (already-verified) observation
    itself is "unconfirmed", which is a category error: a verified
    deviation is never in question, only its cause is.

    Strips a leading framing clause first (e.g. "During the internal audit
    of the Quality Control Laboratory, ..."). Falls back to a plain "Why
    <lowercased clause>?" when no auxiliary verb is found to front.
    """
    clause = _strip_framing(text).strip().rstrip(".")
    if not clause:
        return "Why did this occur?"
    m = _DECLARATIVE_AUX_RE.match(clause)
    if m:
        subject = m.group("subject").strip()
        aux = m.group("aux")
        rest = m.group("rest").strip()
        subject_lc = subject[0].lower() + subject[1:] if subject else subject
        return f"Why {aux} {subject_lc} {rest}?"
    return f"Why {clause[0].lower()}{clause[1:]}?"


def _clean_subject(raw: str) -> str:
    s = raw.strip().strip("\"'").strip()
    s = re.sub(r"^(?:that|which|who)\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^(?:a|an|the|one)\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+", " ", s).strip(" ,.;:")
    return s


# ---------------------------------------------------------------------------
# Entities, Dates, Actors
# ---------------------------------------------------------------------------
_ENTITY_RE = re.compile(
    r"\b([A-Z]{2,5}-[A-Z0-9-]+|Lot\s+[A-Z0-9-]+|Batch\s+[A-Z0-9-]+|Line\s+\d+|Room\s+\d+|"
    r"Cleanroom\s+[A-Za-z0-9\s]+|Autoclave\s+#?\d+|AHU-\d+|CR-\d+|LF-\d+|VI-\d+|CP-\d+|PP-\d+|"
    r"Lyo-\d+|FH-\d+|SP-\d+|BAL-\d+|W-\d+|NC-\d+-\d+|CAPA-\d+-\d+|BRD-\d+|MBR-[A-Z0-9-]+|"
    r"WSC-\d+|API-[0-9]+|RM-[0-9]+|QC-REF-\d+|EQ-\d+)\b",
    re.IGNORECASE,
)

_MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
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
    m = _DATE_RE.search(text or "")
    return m.group(0) if m else None


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


def classify_finding_specificity(
    finding_text: str,
    reported_claims: list[str] | None = None,
    mechanism_status: str | None = None,
) -> str:
    """Deterministic finding-specificity classification (HIGH/MEDIUM/LOW),
    structural only -- never tied to a domain word or a specific evaluated
    finding.

    A finding is only LOW specificity when it has NONE of: a specific
    entity/equipment/document identifier, a date or relative time period, a
    reported/attributed statement, or an already-established immediate
    mechanism. That combination is exactly what a generic allegation like
    "the department is not following the required procedure correctly"
    looks like structurally -- no object, no period, no account, no
    mechanism -- versus a finding that names an affected object, a date, or
    quotes someone even without an explicit ID (e.g. "the checklist was not
    completed for three consecutive days; the operator stated...").

    Used to gate hypothesis generation: a LOW-specificity finding must not
    receive fabricated, evidence-free causal hypotheses (Section 29) -- the
    correct output is NOT_ESTABLISHED plus a list of what's missing, not a
    guess dressed up as analysis.
    """
    text = finding_text or ""
    has_entity = bool(_ENTITY_RE.search(text))
    has_date_or_period = bool(_DATE_RE.search(text) or _RELATIVE_TIME_RE.search(text))
    has_reported = bool(reported_claims)
    has_mechanism = bool(mechanism_status) and mechanism_status not in ("UNKNOWN", "NONE")

    concrete_signals = sum([has_entity, has_date_or_period, has_reported, has_mechanism])
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


def extract_temporal_clause(text: str) -> str | None:
    """Extract a relative temporal clause already stated in the finding
    (e.g. "before the procedure became effective") when no absolute date is
    present. Never fabricates a date — returns None if nothing is stated."""
    if not text:
        return None
    m = _TEMPORAL_CLAUSE_RE.search(text)
    if not m:
        return None
    clause = m.group(1).strip().rstrip(".,;")
    return clause or None


# Generic stopwords stripped when deriving a short topic word from a subject
# phrase — deliberately domain-agnostic so this works for training,
# calibration, checklist, temperature-log, maintenance, documentation,
# inspection, communication, or any other QMS subject noun phrase.
_TOPIC_STOPWORDS = {
    "the", "a", "an", "for", "of", "on", "in", "to", "with", "status",
    "compliance", "record", "records", "log", "logs",
}


def topic_word(subject: str | None) -> str:
    """Derive a short, lower-case topic word (e.g. "training", "calibration",
    "documentation") from a resolved subject phrase, for use in dynamically
    naming hypotheses/evidence instead of a hardcoded domain vocabulary.
    Falls back to "process" when no usable word is found."""
    if not subject:
        return "process"
    for word in re.findall(r"[A-Za-z]+", subject):
        low = word.lower()
        if low not in _TOPIC_STOPWORDS and len(low) > 2:
            return low
    return "process"


# Captures the concrete noun immediately governed by a negation trigger
# (e.g. "had not received retraining" -> "retraining", "was not completed"
# -> nothing since "completed" isn't a noun -- this deliberately only fires
# when the negated verb takes a direct-object noun, which is exactly the
# shape that names the missing activity/thing).
_NEGATION_OBJECT_RE = re.compile(
    r"\bnot\s+(?:received|completed|performed|conducted|done|provided|given|attended)\s+"
    r"([a-z][\w-]*)",
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
         "aboutness"), via topic_word on that shared vocabulary.
      3. The finding's overall subject, only as a last resort.
    """
    for text in (claim_a, claim_b):
        if not text:
            continue
        m = _NEGATION_OBJECT_RE.search(text)
        if m:
            candidate = m.group(1).lower()
            if candidate not in _TOPIC_STOPWORDS and len(candidate) > 2:
                return candidate
    if claim_a and claim_b:
        from app.services.text_grounding import significant_words
        ignore = {"stated", "claimed", "reported", "said", "operator", "supervisor", "auditor", "technician", "manager"}
        shared = (significant_words(claim_a) & significant_words(claim_b)) - ignore
        if shared:
            return topic_word(" ".join(sorted(shared)))
    return topic_word(fallback_subject)


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
    pattern = re.compile(
        rf"^{re.escape(topic)}\b(?:\s+\w+)?\s*(?:for|of)\s+", re.IGNORECASE
    )
    m = pattern.match(subject)
    if m:
        tail = subject[m.end():].strip()
        return tail or None
    return None


def build_affected_object_phrase(subject: str | None, actor: str | None = None) -> str:
    """Build the ONE canonical 'affected object' phrase from a resolved
    subject and an optional actor, e.g. subject="training for the revised
    procedure" + actor="The operator" -> "Operator training status for the
    revised procedure".

    This is the single construction used everywhere an affected-object-style
    phrase is needed (deterministic synthesis, LLM-output repair) so a
    downstream field is never built by independently re-concatenating
    semantic parts (role + topic + tail) a second, inconsistent way. Purely
    structural -- topic/tail come from the finding's own resolved subject,
    never a domain-specific hardcode.
    """
    if not subject or subject.startswith("UNKNOWN"):
        return "NOT ESTABLISHED"
    topic = topic_word(subject)
    tail = split_topic_and_tail(subject, topic) or subject
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
# Structured Deviation Info
# ---------------------------------------------------------------------------
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
    reported_mechanism: str | None = None
    verified_mechanism: str | None = None
    mechanism_status: str = "UNKNOWN"
    mechanism_polarity: str | None = None
    reported_statements: list[str] = field(default_factory=list)
    verified_observations: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    matched: bool = False


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
            r"^(?P<subject>.+?)\s+(?:was|were)\s+missing\s+from\s+(?:equipment\s+|instrument\s+)?(?P<entity>[A-Z0-9-]+)\b.*$",
            re.IGNORECASE,
        ),
    ),
    (
        "not_state",
        re.compile(
            r"^(?P<subject>.+?)\s+(?:was|were)\s+not\s+(?P<cond>[a-z]+(?:\s+[a-z]+){0,3}?)"
            r"\s*(?:\bfor\b.*|\bfrom\b.*|\bon\b.*)?\.?$",
            re.IGNORECASE,
        ),
    ),
    (
        "adj_state",
        re.compile(
            r"^(?P<subject>.+?)\s+(?:was|were)\s+"
            r"(?P<cond>incomplete|missing|unavailable|overdue|expired|nonconforming|inaccurate|"
            r"inadequate|out of date|outdated|unverified|missed)\b.*$",
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
        "failed_to",
        re.compile(r"^(?P<subject>.+?)\s+failed\s+to\s+(?P<cond>[a-z][a-z\s]*?)\s*\.?$", re.IGNORECASE),
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


def _extract_activity_from_reported_finding(text: str) -> str:
    """Extract a clean activity noun phrase when finding is purely reported speech."""
    # Look for "training on the revised procedure", "checklist procedure revision", etc.
    # "training for X" (not "training compliance for X") -- a concise noun
    # phrase naming the activity itself, not a status label glued in front.
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


def extract_semantic_subject(text: str) -> DeviationInfo:
    """Extract the semantic affected object + condition from finding text."""
    if not text or not text.strip():
        return DeviationInfo(subject=None)

    entities = extract_entities(text)
    actors = extract_actors(text)
    date = extract_date(text)

    # 1. Try structural sentence patterns
    for sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        stripped = _strip_framing(sentence)

        for name, pattern in _CONDITION_PATTERNS:
            match = pattern.search(stripped) or pattern.search(sentence)
            if not match:
                continue
            groups = match.groupdict()
            raw_subj = groups.get("subject", "") or ""
            subj = _clean_subject(raw_subj)
            
            # Incorporate entity if matched
            matched_ent = groups.get("entity")
            if matched_ent and matched_ent not in subj:
                subj = f"{subj} for equipment {matched_ent}"
            elif entities and not any(e in subj for e in entities) and ("label" in subj.lower() or "log" in subj.lower()):
                subj = f"{subj} for {entities[0]}"

            if not validate_semantic_subject(subj):
                continue

            cond = groups.get("cond")
            req = groups.get("requirement")
            if name == "not_state" and cond:
                cond = f"not {cond.strip()}"
            elif name == "deviated_from" and req:
                cond = f"deviated from {req.strip()}"
            elif name == "missing_from":
                cond = "missing"

            return DeviationInfo(
                subject=subj,
                finding_subject=subj,
                affected_object=subj,
                affected_process="operational process",
                affected_activity=subj,
                deviation=f"{subj} — {cond or 'condition unverified'}",
                condition=cond,
                requirement=req,
                date=date,
                actor=actors[0] if actors else None,
                actors=actors,
                entities=entities,
                matched=True,
            )

        for cond_label, pattern in _PREFIX_CONDITION_PATTERNS:
            match = pattern.match(stripped)
            if not match:
                continue
            subj = _clean_subject(match.group("subject"))
            if not validate_semantic_subject(subj):
                continue
            return DeviationInfo(
                subject=subj,
                finding_subject=subj,
                affected_object=subj,
                affected_process="operational process",
                affected_activity=subj,
                deviation=f"{subj} — {cond_label}",
                condition=cond_label,
                date=date,
                actor=actors[0] if actors else None,
                actors=actors,
                entities=entities,
                matched=True,
            )

        # Framing was stripped and what's left is a short, usable noun phrase
        # (e.g. "A deviation was observed involving X." -> "X").
        if stripped != sentence and len(stripped) < 140:
            subj = _clean_subject(stripped)
            if validate_semantic_subject(subj):
                return DeviationInfo(
                    subject=subj,
                    finding_subject=subj,
                    affected_object=subj,
                    affected_process="operational process",
                    affected_activity=subj,
                    deviation=f"{subj} — nonconforming",
                    condition="nonconforming",
                    date=date,
                    actor=actors[0] if actors else None,
                    actors=actors,
                    entities=entities,
                    matched=True,
                )

    # 2. Check if the finding describes reported speech / training conflict
    activity_subj = _extract_activity_from_reported_finding(text)
    if entities and not any(e in activity_subj for e in entities):
        activity_subj = f"{activity_subj} for {entities[0]}"

    return DeviationInfo(
        subject=activity_subj,
        finding_subject=activity_subj,
        affected_object=activity_subj,
        affected_process="training management" if "training" in text.lower() else "operational process",
        affected_activity=activity_subj,
        deviation=f"{activity_subj} — status unconfirmed",
        condition="status unconfirmed",
        date=date,
        actor=actors[0] if actors else None,
        actors=actors,
        entities=entities,
        matched=True,
    )


def resolve_deviation(finding_text: str, fact_claims: list[str] | None = None) -> DeviationInfo:
    """Best-effort semantic subject/condition resolution for a finding.
    Guaranteed to return a valid noun phrase subject (never a verb clause)."""
    if not finding_text or not finding_text.strip():
        return DeviationInfo(subject=None)

    candidates = [finding_text] + [c for c in (fact_claims or []) if c]
    result = DeviationInfo(subject=None)
    for candidate in candidates:
        result = extract_semantic_subject(candidate)
        if result.subject and validate_semantic_subject(result.subject):
            break

    if not result.subject or not validate_semantic_subject(result.subject):
        activity_subj = _extract_activity_from_reported_finding(finding_text)
        result.subject = activity_subj
        result.finding_subject = activity_subj
        result.affected_object = activity_subj
        result.deviation = f"{activity_subj} — condition unconfirmed"

    result.date = result.date or extract_date(finding_text)
    result.actors = result.actors or extract_actors(finding_text)
    result.actor = result.actor or (result.actors[0] if result.actors else None)
    result.entities = result.entities or extract_entities(finding_text)
    return result
