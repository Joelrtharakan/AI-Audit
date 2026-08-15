"""General, non-finding-specific semantic subject/condition extraction.

This module answers one question for ANY audit finding: "what is the actual
affected object of the deviation, and what condition is being asserted about
it?" It is a pure grammatical/structural extractor — no finding-specific
vocabulary (no "temperature log", no "refrigerator", no equipment IDs) is
hardcoded anywhere here. It recognizes SENTENCE STRUCTURES (framing clauses,
negated-state clauses, adjectival-state clauses, prefix-deviation clauses),
not specific finding content, so it generalizes to any subject: a checklist,
a procedure, a training record, a batch record, a supplier record, etc.

This replaces the previous "cut the first fact at the first was/were" regex
that was duplicated across four call sites and was the root cause of finding
subjects collapsing into framing fragments like "During the internal audit,
it" whenever the framing clause itself contained a "was" (e.g. "it *was*
observed that ...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# ---------------------------------------------------------------------------
# Step 1: strip reporting/framing clauses. These describe HOW the deviation
# was observed/reported, never WHAT was deviant, so they can never be the
# affected object. Structural (any framing verb + "that"), not finding-specific.
# ---------------------------------------------------------------------------
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
        r"^\s*(?:the\s+)?(?:responsible\s+)?(?:technician|operator|staff|employee|supervisor|analyst|manager)\s+"
        r"(?:stated|confirmed|reported|said|noted)\s+that\s+",
        re.IGNORECASE,
    ),
]


def _strip_framing(sentence: str) -> str:
    out = sentence.strip()
    prev = None
    while out != prev:
        prev = out
        for pattern in _FRAMING_PREFIXES:
            out = pattern.sub("", out).strip()
    return out


# ---------------------------------------------------------------------------
# Step 2: condition clauses. Each captures (subject, condition) from a
# structural pattern describing an audit deviation. Order matters — more
# specific patterns are tried before generic ones.
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
        "not_state",
        re.compile(
            r"^(?P<subject>.+?)\s+(?:was|were)\s+not\s+(?P<cond>[a-z]+(?:\s+[a-z]+){0,2}?)"
            r"\s*(?:\bfor\b.*)?\.?$",
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

# Prefix forms: "failure to perform X", "absence of X", "incomplete X", "missing X", "nonconforming X".
_PREFIX_CONDITION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("failure to perform", re.compile(r"^failure\s+to\s+perform\s+(?P<subject>.+?)\s*\.?$", re.IGNORECASE)),
    ("absence", re.compile(r"^absence\s+of\s+(?P<subject>.+?)\s*\.?$", re.IGNORECASE)),
    ("incomplete", re.compile(r"^(?:an?\s+)?incomplete\s+(?P<subject>.+?)\s*\.?$", re.IGNORECASE)),
    ("missing", re.compile(r"^(?:an?\s+)?missing\s+(?P<subject>.+?)\s*\.?$", re.IGNORECASE)),
    ("nonconforming", re.compile(r"^(?:an?\s+)?nonconforming\s+(?P<subject>.+?)\s*\.?$", re.IGNORECASE)),
]

_PRONOUN_ONLY = {"it", "this", "that", "there", "they", "he", "she", "which", "who"}


def _clean_subject(raw: str) -> str:
    s = raw.strip().strip("\"'").strip()
    s = re.sub(r"^(?:that|which|who)\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^(?:a|an|the|one)\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s+", " ", s).strip(" ,.;:")
    return s


def _is_usable_subject(subject: str) -> bool:
    if not subject or len(subject) < 3:
        return False
    if subject.lower() in _PRONOUN_ONLY:
        return False
    # Reject a subject that is nothing but stopwords/pronouns (e.g. "the it").
    words = re.findall(r"[a-zA-Z0-9]+", subject.lower())
    if not words:
        return False
    if all(w in _PRONOUN_ONLY for w in words):
        return False
    return True


# ---------------------------------------------------------------------------
# Date / actor extraction — general patterns, not tied to any specific finding.
# ---------------------------------------------------------------------------
_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
)
_DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}\s+{_MONTHS}\s+\d{{4}}|{_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}|"
    rf"\d{{1,2}}/\d{{1,2}}/\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}})\b"
)
_ACTOR_RE = re.compile(
    r"\bthe\s+responsible\s+\w+\b|"
    r"\bthe\s+\w+\s+(?:technician|operator|supervisor|analyst|manager|auditor|inspector)\b",
    re.IGNORECASE,
)


def extract_date(text: str) -> str | None:
    m = _DATE_RE.search(text or "")
    return m.group(0) if m else None


def extract_actor(text: str) -> str | None:
    m = _ACTOR_RE.search(text or "")
    return m.group(0).strip() if m else None


@dataclass
class DeviationInfo:
    subject: str | None
    condition: str | None = None
    requirement: str | None = None
    date: str | None = None
    actor: str | None = None
    matched: bool = False


def extract_semantic_subject(text: str) -> DeviationInfo:
    """Extract the semantic affected object + condition from a single piece
    of finding text, using structural sentence patterns only."""
    if not text or not text.strip():
        return DeviationInfo(subject=None)

    for sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        stripped = _strip_framing(sentence)

        for _name, pattern in _CONDITION_PATTERNS:
            match = pattern.search(stripped) or pattern.search(sentence)
            if not match:
                continue
            groups = match.groupdict()
            subject = _clean_subject(groups.get("subject", "") or "")
            if not _is_usable_subject(subject):
                continue
            condition = groups.get("cond")
            requirement = groups.get("requirement")
            if condition:
                condition = condition.strip()
            if _name == "not_state" and condition:
                condition = f"not {condition}"
            if _name == "deviated_from" and requirement:
                condition = f"deviated from {requirement.strip()}"
            return DeviationInfo(subject=subject, condition=condition, requirement=requirement, matched=True)

        for cond_label, pattern in _PREFIX_CONDITION_PATTERNS:
            match = pattern.match(stripped)
            if not match:
                continue
            subject = _clean_subject(match.group("subject"))
            if not _is_usable_subject(subject):
                continue
            return DeviationInfo(subject=subject, condition=cond_label, matched=True)

        # Framing was stripped and what's left is a short, usable noun phrase
        # (e.g. "A deviation was observed involving X." -> "X").
        if stripped != sentence and len(stripped) < 140:
            subject = _clean_subject(stripped)
            if _is_usable_subject(subject):
                return DeviationInfo(subject=subject, condition=None, matched=True)

    return DeviationInfo(subject=None)


def resolve_deviation(finding_text: str, fact_claims: list[str] | None = None) -> DeviationInfo:
    """Best-effort semantic subject/condition resolution for a finding.

    Tries the raw finding text first (most context), then each individually
    extracted fact as a fallback candidate. Date/actor are always resolved
    from the full finding text regardless of which candidate produced the
    subject, since they may appear in a different sentence than the subject.
    """
    candidates = [finding_text] + [c for c in (fact_claims or []) if c]
    result = DeviationInfo(subject=None)
    for candidate in candidates:
        result = extract_semantic_subject(candidate)
        if result.subject:
            break
    result.date = result.date or extract_date(finding_text)
    result.actor = result.actor or extract_actor(finding_text)
    return result
