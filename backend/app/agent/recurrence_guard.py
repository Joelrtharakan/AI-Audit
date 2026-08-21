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
_NUMBER_WORD = r"(?:\d+|two|three|four|five|six|seven|eight|nine|ten|multiple|several|numerous)"
# Population/occurrence-count evidence of recurrence (Section 1/2): "the
# same X was identified in three separate batches", "occurred across
# multiple locations", "identified on four separate occasions" -- a
# generalized structural signal (count/population noun, not any specific
# domain word like "batch") independent of whether the finding also uses
# an explicit word like "recurring"/"repeated".
_OCCURRENCE_POPULATION_RE = re.compile(
    rf"\b(?:the\s+same|a\s+similar|this)\s+[\w\s-]{{1,40}}?\b(?:was|were)\s+"
    rf"(?:identified|observed|found|noted|detected|reported)\s+"
    rf"(?:in|across|on)\s+{_NUMBER_WORD}\s+(?:separate\s+|different\s+|distinct\s+)?(?:[a-z]+\s+){{0,2}}"
    rf"(?:batches|records|units|transactions|locations|sites|periods|cases|instances|occasions|"
    rf"shipments|lots|samples|invoices|files|systems|documents|departments|suppliers|employees)\b",
    re.IGNORECASE,
)
_OCCURRENCE_COUNT_RE = re.compile(
    rf"\b{_NUMBER_WORD}\s+(?:separate\s+|different\s+|distinct\s+)?(?:occasions|times|instances)\b",
    re.IGNORECASE,
)
_RECURRENCE_WORD_RE = re.compile(
    r"\b(recurrence|recurring|recurred|repeated\s+finding|repeat\s+finding|reoccur(?:red|rence)?|the\s+same\s+[\w\s-]{1,30}?\s+occurred)\b",
    re.IGNORECASE,
)
_PREVIOUS_AUDIT_TIME_RE = re.compile(
    r"\b(previous|prior|earlier|last)\s+audit\b|"
    r"\b\d+\s+(?:day|week|month|year)s?\s+(?:earlier|ago|prior|before)\b",
    re.IGNORECASE,
)
_PREVIOUS_CAPA_RE = re.compile(
    r"\b(?:previous|prior)\s+(?:[\w\s-]{0,30}?\s+)?(?:corrective\s+action|CAPA|preventive\s+maintenance)\b|"
    # A corrective/preventive action taken "after"/"following" an earlier
    # occurrence is a PRIOR action relative to the current finding even
    # without the literal word "previous"/"prior" -- the temporal
    # relationship is expressed structurally instead.
    r"\b(?:corrective\s+action|CAPA|preventive\s+action|remediation|containment\s+action|action\s+plan)\b"
    r"(?:(?!\.).){0,40}?\b(?:after|following)\s+the\s+(?:first|initial|earlier|prior)\s+"
    r"(?:occurrence|instance|finding|deviation)\b",
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
# The mere PRESENCE of the phrase "effectiveness review" does not confirm
# effectiveness -- "the effectiveness review was not available" mentions
# the exact same words while stating the opposite. Checked as a negation
# window around the effectiveness-review match itself (not just a global
# "not" anywhere in the finding, which would incorrectly suppress
# genuinely confirmed effectiveness elsewhere in a longer finding).
_EFFECTIVENESS_NEGATED_RE = re.compile(
    r"\b(?:effectiveness\s+(?:review|verification|check|assessment))\b"
    r"(?:(?!\.).){0,40}?\b(?:not\s+available|unavailable|was\s+not\s+(?:conducted|performed|completed|done)|"
    r"could\s+not\s+be\s+(?:located|found|retrieved)|absent|missing|no\s+(?:record|evidence)\s+of)\b|"
    r"\b(?:no|not\s+yet|without)\b(?:(?!\.).){0,20}?\beffectiveness\s+(?:review|verification|check|assessment)\b",
    re.IGNORECASE,
)


@dataclass
class RecurrenceInfo:
    is_recurring: bool = False
    has_previous_capa_reference: bool = False
    capa_id: str | None = None
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
        or _OCCURRENCE_POPULATION_RE.search(finding_text)
        or _OCCURRENCE_COUNT_RE.search(finding_text)
    )
    has_previous_capa = bool(_PREVIOUS_CAPA_RE.search(finding_text))
    m_capa = re.search(r"\b(CAPA-\d{4}-\d+|CAPA-[A-Z0-9-]+)\b", finding_text, re.IGNORECASE)
    capa_id = m_capa.group(1).upper() if m_capa else None

    capa_status = "COMPLETED" if (has_previous_capa and _CAPA_COMPLETED_RE.search(finding_text)) else None
    effectiveness = (
        "EFFECTIVE"
        if (
            has_previous_capa
            and _EFFECTIVENESS_REVIEW_RE.search(finding_text)
            and not _EFFECTIVENESS_NEGATED_RE.search(finding_text)
        )
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
        capa_id=capa_id,
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


def build_recurrence_rationale(rec: RecurrenceInfo) -> str:
    """Deterministic recurrence-risk rationale text (Section 5/9 of the
    recurrence-hardening spec): recurrence after a previous CAPA's closure
    is a reason to flag HIGH recurrence RISK -- it is never, by itself,
    proof that the previous CAPA was ineffective (that requires its own
    explicit effectiveness-review evidence, tracked separately in
    previous_capa_effectiveness). The two dimensions must never be
    conflated into one assertion.
    """
    capa_ref = rec.capa_id or "a previous CAPA"
    if rec.previous_capa_effectiveness == "EFFECTIVE":
        # Recurring despite a verified-effective previous CAPA -- still a
        # HIGH-risk signal (something is still going wrong), but framed
        # around the recurrence itself, not a (contradicted) ineffectiveness
        # claim.
        return (
            f"The same issue recurred after {capa_ref} was verified effective; the current "
            "recurrence indicates a materially different mechanism or a control gap requiring investigation."
        )
    return (
        f"The same issue recurred after {capa_ref} was marked complete, while objective "
        "effectiveness-verification evidence is unavailable. This is a recurrence-risk signal, "
        "not proof that the previous corrective action was ineffective."
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
