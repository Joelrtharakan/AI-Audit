"""Temporal / recurrence reasoning.

Detects whether a finding describes a RECURRING nonconformity (a similar
finding previously identified) and whether a previous corrective action was
recorded as completed -- and, critically, keeps COMPLETION and
EFFECTIVENESS as two separate facts. "Previous CAPA completed" is never
treated as "previous CAPA effective": effectiveness stays NOT_VERIFIED
unless the finding/evidence explicitly states an effectiveness review
occurred.

Purely structural (regex over generic recurrence/CAPA vocabulary, no
domain-specific words like "temperature" or "training") so this
generalizes across any QMS domain a finding can describe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A similar finding was previously identified -- deliberately generic
# ("a similar <anything> finding/nonconformity/issue/observation/deviation
# was identified/observed/found/noted/reported"), never tied to one domain.
_SIMILAR_FINDING_RE = re.compile(
    r"\bsimilar\s+[\w\s-]{0,40}?(?:finding|nonconformity|non-conformity|issue|observation|deviation)\b"
    r"(?:(?!\.).){0,40}?\b(?:identified|observed|found|noted|reported|existed)\b",
    re.IGNORECASE,
)
_RECURRENCE_WORD_RE = re.compile(
    r"\b(recurrence|recurring|repeated\s+finding|repeat\s+finding|reoccur(?:red|rence)?)\b",
    re.IGNORECASE,
)
_PREVIOUS_AUDIT_TIME_RE = re.compile(
    r"\b(previous|prior|earlier|last)\s+audit\b|"
    r"\b\d+\s+(?:day|week|month|year)s?\s+(?:earlier|ago|prior|before)\b",
    re.IGNORECASE,
)
_PREVIOUS_CAPA_RE = re.compile(
    r"\b(?:previous|prior)\s+corrective\s+action\b|\b(?:previous|prior)\s+CAPA\b",
    re.IGNORECASE,
)
_CAPA_COMPLETED_RE = re.compile(
    r"\b(?:corrective\s+action|CAPA)\b(?:(?!\.).){0,40}?\b(?:was\s+)?"
    r"(?:recorded\s+as\s+|marked\s+as\s+|documented\s+as\s+)?(completed|closed|implemented)\b",
    re.IGNORECASE,
)
_EFFECTIVENESS_REVIEW_RE = re.compile(
    r"\b(effectiveness\s+(?:review|verification|check|assessment)|verified\s+(?:as\s+)?effective|"
    r"confirmed\s+(?:to\s+be\s+)?effective)\b",
    re.IGNORECASE,
)


@dataclass
class RecurrenceInfo:
    is_recurring: bool = False
    has_previous_capa_reference: bool = False
    # "COMPLETED" | None -- whether the finding states the previous
    # corrective action was recorded as completed.
    previous_capa_status: str | None = None
    # "EFFECTIVE" | "NOT_VERIFIED" -- NEVER derived from previous_capa_status
    # alone; requires its own explicit effectiveness-review language.
    previous_capa_effectiveness: str = "NOT_VERIFIED"
    rationale: str | None = None


def detect_recurrence(finding_text: str) -> RecurrenceInfo:
    """Deterministic, structural recurrence/previous-CAPA detector. Never
    infers effectiveness from completion status alone -- that conflation is
    exactly the defect this module exists to prevent."""
    if not finding_text:
        return RecurrenceInfo()

    is_recurring = bool(
        _SIMILAR_FINDING_RE.search(finding_text)
        or _RECURRENCE_WORD_RE.search(finding_text)
    )
    has_previous_capa = bool(_PREVIOUS_CAPA_RE.search(finding_text))

    capa_status = "COMPLETED" if (has_previous_capa and _CAPA_COMPLETED_RE.search(finding_text)) else None
    effectiveness = (
        "EFFECTIVE" if (has_previous_capa and _EFFECTIVENESS_REVIEW_RE.search(finding_text))
        else "NOT_VERIFIED"
    )

    rationale = None
    if is_recurring and capa_status == "COMPLETED":
        if effectiveness == "NOT_VERIFIED":
            rationale = (
                "A similar finding was previously identified and the prior corrective action was "
                "recorded as completed; effectiveness of that action has not been established from "
                "the available evidence."
            )
        else:
            rationale = (
                "A similar finding was previously identified; the prior corrective action was "
                "recorded as completed and an effectiveness review is referenced in the evidence."
            )
    elif is_recurring:
        rationale = "A similar finding was previously identified for this subject."
    elif has_previous_capa:
        rationale = "A previous corrective action is referenced, but no similar-finding recurrence is stated."

    return RecurrenceInfo(
        is_recurring=is_recurring,
        has_previous_capa_reference=has_previous_capa,
        previous_capa_status=capa_status,
        previous_capa_effectiveness=effectiveness,
        rationale=rationale,
    )


# A hypothesis whose CONTENT concerns whether the previous corrective action
# was implemented/verified/effective -- checked by content, not by name or
# id, so this catches such a hypothesis regardless of what a real LLM call
# happened to name or number it (an exact-name match like
# "*_CAPA_EFFECTIVENESS_GAP" only catches this system's OWN deterministic
# naming convention, not an LLM's free-form phrasing of the same mechanism).
_PREVIOUS_WORD_RE = re.compile(r"\b(?:previous|prior|earlier)\b", re.IGNORECASE)
_CAPA_WORD_RE = re.compile(r"\b(?:corrective\s+action|capa)\b", re.IGNORECASE)
_IMPLEMENTATION_EFFECTIVENESS_WORD_RE = re.compile(
    r"\b(?:implement\w*|effective\w*|ineffective\w*|verif\w*|recurrence)\b", re.IGNORECASE
)


def is_previous_capa_mechanism_hypothesis(statement: str | None) -> bool:
    """True if `statement` concerns the implementation/verification/
    effectiveness of a PREVIOUS corrective action -- used to gate hypothesis
    status promotion: word-overlap with the VERIFIED fact that a similar
    finding recurred and a previous CAPA was completed will always be high
    for a hypothesis like this (that fact is what grounds the hypothesis in
    the first place), but it never verifies the hypothesis's OWN specific
    claim about implementation/verification/effectiveness -- only a
    dedicated implementation or effectiveness-review claim could.

    Deliberately co-occurrence-based rather than requiring "previous" and
    "corrective action" to sit adjacent -- a hypothesis can phrase this as
    "the corrective action implemented following the previous finding..."
    just as validly as "the previous corrective action...", and both must
    be caught the same way.
    """
    if not statement:
        return False
    return bool(
        _PREVIOUS_WORD_RE.search(statement)
        and _CAPA_WORD_RE.search(statement)
        and _IMPLEMENTATION_EFFECTIVENESS_WORD_RE.search(statement)
    )
