"""General, structural causal-reasoning guards.

Two things live here, both deliberately content-free (no finding vocabulary
like "temperature"/"refrigerator"/"training" anywhere in this file — only
grammatical/verb-shape patterns that generalize across any QMS domain):

1. MECHANISM EXTRACTION: distinguishes the OBSERVATION (the artifact/state a
   finding describes, e.g. "the log was incomplete") from the IMMEDIATE
   MECHANISM (the action-level explanation of HOW it happened, when the
   finding/evidence directly states one, e.g. "the check was missed"). This
   is Layer 1 vs Layer 2 of the causal chain — see CausalGuard architecture
   note in app/agent/nodes/core_synthesis.py.

2. CONTRADICTION DETECTION: once a mechanism establishes NON-PERFORMANCE
   (the required activity did not happen at all), a competing hypothesis
   asserting NON-RECORDING (the activity happened but wasn't documented) is
   logically contradicted, not merely "less likely" — the two cannot both be
   true. This is enforced in code so it holds regardless of how faithfully
   the LLM followed the prompt's own instruction to do the same.

3. 5-WHY CIRCULARITY: a "why" step whose answer contributes essentially no
   vocabulary beyond its own question (or beyond the previous step's answer)
   is restating the question, not explaining it — a generic, content-free
   check that catches template-shaped non-answers regardless of domain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.text_grounding import significant_words

# ---------------------------------------------------------------------------
# 1. Mechanism extraction
# ---------------------------------------------------------------------------

# A required activity did not happen at all. The "not (been) VERB" forms
# allow for auxiliary chains ("was not performed", "may not have been
# performed", "had not been conducted") rather than requiring "not" to sit
# immediately before the verb.
_NON_PERFORMANCE_RE = re.compile(
    r"\b(missed|skipped|omitted|"
    r"not\s+(?:have\s+been\s+|been\s+)?(?:performed|conducted|carried\s+out|done|completed|followed|occurred|assigned)|"
    r"did\s+not\s+(?:occur|perform|conduct|complete|carry\s+out|follow|assign)|"
    r"failed\s+to\s+(?:perform|conduct|complete|carry\s+out|follow|assign)|"
    r"never\s+(?:performed|conducted|done|completed|occurred|assigned))\b",
    re.IGNORECASE,
)

# An activity may have happened but wasn't captured/documented.
_NON_RECORDING_RE = re.compile(
    r"\b(not\s+recorded|not\s+documented|not\s+logged|undocumented|not\s+captured|"
    r"no\s+record\s+(?:was\s+)?(?:made|kept)|entry\s+(?:was\s+)?omitted)\b",
    re.IGNORECASE,
)

# A person lacked awareness/notification of something (e.g. a revision, a
# requirement, a status change) -- distinct from either polarity above: the
# activity may not have happened *because* of this gap, but the gap itself
# is about knowledge/communication, not performance or recording.
_KNOWLEDGE_GAP_RE = re.compile(
    r"\b(unaware|not\s+aware|did\s+not\s+know|didn't\s+know|not\s+informed|"
    r"not\s+notified|not\s+told|no\s+knowledge\s+of)\b",
    re.IGNORECASE,
)


@dataclass
class MechanismInfo:
    statement: str | None = None
    status: str = "UNKNOWN"  # VERIFIED | REPORTED | UNKNOWN
    polarity: str | None = None  # "non_performance" | "non_recording" | None
    source_claim: str | None = None


def classify_mechanism_polarity(text: str) -> str | None:
    """Structural classification of a claim's mechanism shape. Returns None
    if the claim doesn't state an action-level mechanism at all (e.g. it's
    just describing the artifact's state, not what a person/system did)."""
    if not text:
        return None
    if _NON_PERFORMANCE_RE.search(text):
        return "non_performance"
    if _NON_RECORDING_RE.search(text):
        return "non_recording"
    if _KNOWLEDGE_GAP_RE.search(text):
        return "knowledge_gap"
    return None


def extract_immediate_mechanism(
    reported_statements: list[str],
    verified_facts: list[str],
) -> MechanismInfo:
    """Find the claim that states the immediate mechanism. Reported
    statements are checked before verified facts, since a person's account
    of HOW something happened is usually the reported statement.

    Two tiers:
    1. A claim whose verb shape matches one of the specific, recognized
       polarities (non_performance / non_recording / knowledge_gap).
    2. GENERAL CATCH-ALL (reported statements only): a reported/attributed
       statement is inherently a causal-explanation candidate -- that's
       what makes it "reported" rather than a plain observed fact -- even
       when its specific verb shape doesn't match a recognized polarity
       (e.g. "operator was unaware that the procedure had been revised"
       matches knowledge_gap, but arbitrary other attributed explanations
       might not match anything specific). Never silently drop it; a
       reported causal statement is real information the finding already
       gave us, and losing it (especially when the LLM is unavailable) is
       exactly the failure this function exists to prevent. Verified facts
       are NOT given this catch-all: a verified fact with no recognized
       mechanism shape is usually just the observation restated, and
       treating every verified fact as "the mechanism" would misclassify
       the observation itself as its own explanation.
    """
    for claim in reported_statements or []:
        polarity = classify_mechanism_polarity(claim)
        if polarity:
            return MechanismInfo(statement=claim, status="REPORTED", polarity=polarity, source_claim=claim)
    for claim in verified_facts or []:
        polarity = classify_mechanism_polarity(claim)
        if polarity:
            return MechanismInfo(statement=claim, status="VERIFIED", polarity=polarity, source_claim=claim)
    for claim in reported_statements or []:
        if claim and claim.strip():
            return MechanismInfo(statement=claim, status="REPORTED", polarity="general", source_claim=claim)
    return MechanismInfo()


# ---------------------------------------------------------------------------
# 2. Contradiction detection
# ---------------------------------------------------------------------------

# A hypothesis asserting the activity WAS carried out, just not captured.
_PERFORMED_BUT_NOT_RECORDED_RE = re.compile(
    r"\b(performed|conducted|carried\s+out|done|completed|followed|occurred|took\s+place)\b"
    r"(?:(?!\.).){0,60}?"
    r"\b(not\s+(?:documented|recorded|logged|captured)|"
    r"but\s+(?:was\s+)?(?:not\s+)?(?:documented|recorded|logged))\b",
    re.IGNORECASE,
)


def hypothesis_contradicts_mechanism(hypothesis_text: str, mechanism: MechanismInfo) -> bool:
    """True if `hypothesis_text` asserts the activity WAS performed (just
    not recorded) while the established mechanism directly states the
    activity was NOT performed at all -- the two cannot both be true."""
    if not hypothesis_text:
        return False
    if mechanism.status not in ("VERIFIED", "REPORTED") or mechanism.polarity != "non_performance":
        return False
    return bool(_PERFORMED_BUT_NOT_RECORDED_RE.search(hypothesis_text))


# A VERIFIED fact stating something WAS done ("training was completed",
# "calibration was performed", "review was conducted").
_POSITIVE_COMPLETION_RE = re.compile(
    r"\b([a-z][a-z\s]{1,40}?)\s+(?:was|were|has\s+been|have\s+been)\s+"
    r"(?:completed|provided|performed|conducted|delivered|given|received|attended)\b",
    re.IGNORECASE,
)

# A hypothesis asserting a deficiency/absence of that same thing.
_DEFICIENCY_LEADING_RE = re.compile(
    r"\b(?:lack\s+of|insufficient|inadequate|absence\s+of)\s+([a-z][a-z\s]{1,30}?)"
    r"(?=[.,]|\s+(?:may|might|could|contributed|was|is|caused|led)\b|$)",
    re.IGNORECASE,
)
_DEFICIENCY_SUFFIX_RE = re.compile(
    r"\b([a-z][a-z\s]{1,30}?)\s+(?:deficiency|gap)\b",
    re.IGNORECASE,
)
_NOT_PROVIDED_RE = re.compile(
    r"\b([a-z][a-z\s]{1,30}?)\s+(?:was|were)\s+not\s+"
    r"(?:provided|completed|performed|conducted|sufficient|adequate|given|received)\b",
    re.IGNORECASE,
)


def hypothesis_contradicts_verified_completion(hypothesis_text: str, verified_facts: list[str]) -> bool:
    """True if `hypothesis_text` proposes a deficiency/absence of some thing
    (e.g. "training deficiency", "lack of calibration") while a VERIFIED
    fact directly states that same thing WAS done (e.g. "training was
    completed") -- the two describe opposite states of the same topic and
    cannot both be true. General verb-shape + word-overlap check, not tied
    to any specific topic word (training, calibration, maintenance, ...)."""
    if not hypothesis_text or not verified_facts:
        return False

    deficiency_topics: list[str] = []
    for pattern in (_DEFICIENCY_LEADING_RE, _DEFICIENCY_SUFFIX_RE, _NOT_PROVIDED_RE):
        deficiency_topics.extend(pattern.findall(hypothesis_text))
    if not deficiency_topics:
        return False

    for fact in verified_facts:
        for completed_topic in _POSITIVE_COMPLETION_RE.findall(fact):
            completed_words = significant_words(completed_topic)
            if not completed_words:
                continue
            for topic in deficiency_topics:
                topic_words = significant_words(topic)
                if topic_words and (completed_words & topic_words):
                    return True
    return False


def mechanism_already_names_generic_hypothesis(hypothesis_statement: str, mechanism: MechanismInfo) -> bool:
    """True if a hypothesis is just re-describing the already-established
    mechanism as though it were still an open possibility (e.g. mechanism
    already establishes non-performance, and the hypothesis proposes
    "activity may not have been performed" as a new discovery). Structural:
    checks whether the hypothesis's own polarity matches the established
    mechanism's polarity with hedging language ("may have", "possibly",
    "it is possible that") wrapped around essentially the same claim."""
    if mechanism.status not in ("VERIFIED", "REPORTED") or not mechanism.polarity:
        return False
    if not hypothesis_statement:
        return False
    hedge_re = re.compile(r"\b(may|might|possibly|could\s+have|it\s+is\s+possible)\b", re.IGNORECASE)
    if not hedge_re.search(hypothesis_statement):
        return False
    return classify_mechanism_polarity(hypothesis_statement) == mechanism.polarity


# ---------------------------------------------------------------------------
# 3. 5-Why circularity
# ---------------------------------------------------------------------------

_WHY_STOPWORDS = {"why", "did", "was", "were", "the", "that", "this"}


def is_circular_why_answer(question: str, answer: str | None) -> bool:
    """True if `answer` contributes essentially no new vocabulary beyond its
    own question -- i.e. it restates the question rather than answering it
    (e.g. Q: "Why was the record incomplete?" A: "The record was
    incomplete.")."""
    if not answer:
        return False
    q_words = significant_words(question) - _WHY_STOPWORDS
    a_words = significant_words(answer) - _WHY_STOPWORDS
    if not a_words:
        return False
    new_words = a_words - q_words
    overlap = q_words & a_words
    return len(new_words) <= 1 and len(overlap) >= 2


def is_reporting_why_question(question: str) -> bool:
    """True if question asks about reporting behavior rather than causal mechanism
    (e.g., 'Why did personnel report...', 'Why was this observed during audit...')."""
    if not question:
        return False
    return bool(re.search(
        r"\bwhy\s+did\s+(?:the\s+|a\s+)?(?:personnel|staff|operator|technician|auditor)\s+report\b",
        question, re.IGNORECASE,
    )) or bool(re.search(r"\bwhy\s+was\s+(?:this|it|the\s+nonconformity)\s+reported\b", question, re.IGNORECASE))


def repeats_previous_why_answer(previous_answer: str | None, answer: str | None) -> bool:
    """True if this step's answer is essentially the same content as the
    prior step's answer -- the chain isn't advancing."""
    if not previous_answer or not answer:
        return False
    prev_words = significant_words(previous_answer)
    words = significant_words(answer)
    if not prev_words or not words:
        return False
    union = prev_words | words
    if not union:
        return False
    jaccard = len(prev_words & words) / len(union)
    return jaccard >= 0.75


def restates_observation(answer: str | None, observed_deviation: str | None) -> bool:
    """True if a Why-step's answer is essentially just the observation
    restated (e.g. "the log was incomplete" answering why the log was
    incomplete) rather than explaining it -- the same near-duplicate check
    used for previous-answer repetition, applied against Layer 1."""
    return repeats_previous_why_answer(observed_deviation, answer)


def question_reopens_mechanism(question: str, mechanism: MechanismInfo) -> bool:
    """True if a 5-Why QUESTION itself re-litigates whether the mechanism
    occurred (e.g. asking "was it performed but not documented?" once
    non-performance is already established, or vice versa) instead of
    asking why the established mechanism occurred. Structural: the question
    names the OPPOSITE polarity from the one already established."""
    if not question or mechanism.status not in ("VERIFIED", "REPORTED") or not mechanism.polarity:
        return False
    question_polarity = classify_mechanism_polarity(question)
    if question_polarity is None:
        return False
    return question_polarity != mechanism.polarity


# ---------------------------------------------------------------------------
# 4. Generic non-analysis filler
# ---------------------------------------------------------------------------

_GENERIC_FILLER_RE = re.compile(
    r"^\s*(additional\s+)?(contributing\s+factors?|root\s+cause|evidence)?\s*"
    r"(is|are|was|were)?\s*not\s+(yet\s+)?(established|available|determined|identified)\s*\.?\s*$",
    re.IGNORECASE,
)


def is_generic_non_analysis_filler(text: str | None) -> bool:
    """True if a field that's supposed to carry a specific, case-grounded
    analysis instead contains only a boilerplate non-answer (e.g.
    "Additional contributing factors are not established."). This is a
    structural sentence-shape check, not a lookup of any specific phrase --
    it fires on any sentence whose entire content is "X is/are not
    established/available" with nothing case-specific attached."""
    if not text:
        return False
    return bool(_GENERIC_FILLER_RE.match(text.strip()))
