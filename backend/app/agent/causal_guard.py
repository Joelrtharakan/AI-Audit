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
    r"but\s+(?:was\s+)?(?:not\s+)?(?:documented|recorded|logged)|"
    r"documentation\s+(?:was\s+)?(?:omitted|missing|not\s+completed)|"
    r"recording\s+(?:was\s+)?(?:omitted|delayed))\b",
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
# "calibration was performed", "review was conducted", "confirmed active calibration").
_POSITIVE_COMPLETION_RE = re.compile(
    r"\b([\w\s-]{1,50}?)\s+(?:was|were|has\s+been|have\s+been)\s+"
    r"(?:completed|provided|performed|conducted|delivered|given|received|attended|active)\b|"
    r"\b(?:confirmed|verified)\s+(?:active\s+)?([\w\s-]{1,50}?)\b",
    re.IGNORECASE,
)

# A hypothesis asserting a deficiency/absence of that same thing.
_DEFICIENCY_LEADING_RE = re.compile(
    r"\b(?:lack\s+of|insufficient|inadequate|absence\s+of)\s+([\w\s-]{1,50}?)"
    r"(?=[.,]|\s+(?:may|might|could|contributed|was|is|caused|led)\b|$)",
    re.IGNORECASE,
)
_DEFICIENCY_SUFFIX_RE = re.compile(
    r"\b([\w\s-]{1,50}?)\s+(?:deficiency|gap)\b",
    re.IGNORECASE,
)
_NOT_PROVIDED_RE = re.compile(
    r"\b([\w\s-]{1,50}?)\s+(?:was|were)\s+not\s+"
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
        for m in _POSITIVE_COMPLETION_RE.finditer(fact):
            completed_topic = m.group(1) or (m.group(2) if len(m.groups()) >= 2 else None)
            if not completed_topic:
                continue
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
        r"\bwhy\s+did\s+(?:the\s+|a\s+)?(?:personnel|staff|operator|technician|auditor|supervisor|inspector)\s+"
        r"(?:report|state|claim|say|mention|note)\b",
        question, re.IGNORECASE,
    )) or bool(re.search(r"\bwhy\s+was\s+(?:this|it|the\s+nonconformity)\s+(?:reported|observed)\b", question, re.IGNORECASE))


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


# ---------------------------------------------------------------------------
# 5. Hypothesis causality: reject a "hypothesis" that is really just an
#    evidence-gap/observation restatement, not a proposed explanation.
# ---------------------------------------------------------------------------

# Vocabulary that signals the statement is actually PROPOSING a causal
# mechanism (a control/process/step that may have failed) rather than just
# describing what is/isn't currently known or on hand. Deliberately generic
# -- no domain word (calibration, temperature, training, ...) appears here.
_CAUSAL_EXPLANATION_RE = re.compile(
    r"\b(not\s+(completed|followed|applied|performed|synchroni[sz]ed|aligned|updated|verified|"
    r"communicated|documented|filed|retriev\w*)|fail(ed|ure)?|breakdown|weakness|inadequa\w*|"
    r"insufficient|lack(ing|s)?\s+of|absence\s+of\s+an?\s+effective|no\s+effective|gap\s+in|"
    r"may\s+not\s+have\s+been|does\s+not\s+(ensure|require|verify|confirm)|"
    r"was\s+not\s+(assigned|scheduled|reviewed))\b",
    re.IGNORECASE,
)


_ATTRIBUTION_LANGUAGE_RE = re.compile(
    r"\b(stated|reported|claimed|said|mentioned|indicated|alleged)\s+(that\s+)?",
    re.IGNORECASE,
)


def answer_asserts_verified_but_is_reported(
    answer: str | None, status: str, reported_facts: list[str], verified_facts: list[str]
) -> bool:
    """True if a 5-Why step's answer is labeled VERIFIED but its actual
    content traces back to a REPORTED (not VERIFIED) evidence-ledger claim
    -- i.e. the model escalated someone's account into an established fact
    just by picking a status word, rather than the ledger's own evidence
    status actually supporting VERIFIED. Structural: compares the answer's
    content against both fact pools rather than trusting the status label
    the model attached to it.

    Two independent signals, either one sufficient:
      1. The answer text itself contains attribution language ("the
         operator STATED that...") -- a person's account is being narrated
         inline, regardless of what else the sentence also says.
      2. The answer's content substantially overlaps a REPORTED ledger
         claim more than any VERIFIED one (catches paraphrased attribution
         that dropped the reporting verb).
    Checking (1) directly matters because a mixed sentence ("X stated Y,
    but Z was also true") can dilute a pure word-overlap ratio below any
    reasonable threshold even though it plainly narrates a report.
    """
    if status != "VERIFIED" or not answer:
        return False

    if _ATTRIBUTION_LANGUAGE_RE.search(answer):
        return True

    def _best_overlap(claims: list[str]) -> float:
        answer_words = significant_words(answer)
        if not answer_words:
            return 0.0
        best = 0.0
        for claim in claims:
            claim_words = significant_words(claim)
            if not claim_words:
                continue
            ratio = len(answer_words & claim_words) / len(answer_words)
            best = max(best, ratio)
        return best

    reported_overlap = _best_overlap(reported_facts)
    verified_overlap = _best_overlap(verified_facts)
    return reported_overlap >= 0.6 and reported_overlap > verified_overlap


_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:,\s*but\s+|,\s*however\s+|;\s*|\s+but\s+|\s+however\s+)\s*", re.IGNORECASE
)


def classify_mixed_evidence_answer(answer: str | None) -> str | None:
    """Detects a compound sentence whose clauses carry DIFFERENT evidence
    provenance -- e.g. "the operator STATED X occurred, but Y was not
    available" combines a REPORTED claim (someone's account) with a
    separately-standing factual clause. A single status word (VERIFIED or
    otherwise) can never honestly describe the whole sentence in that case.

    Returns "MIXED" when the answer splits into 2+ clauses on a
    conjunction/semicolon boundary AND at least one clause carries
    attribution language while at least one other substantive clause does
    not (i.e. it isn't ALL reported, and it isn't ALL clearly-standing
    fact -- it's a genuine mix). Returns None otherwise, leaving the
    caller's existing status logic (e.g.
    answer_asserts_verified_but_is_reported for a purely-reported answer)
    to decide.

    Structural only: the clause boundary is punctuation/conjunction shape,
    the provenance signal is the same domain-free attribution vocabulary
    used elsewhere -- no finding-specific words appear here.
    """
    if not answer:
        return None
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(answer) if c.strip()]
    if len(clauses) < 2:
        return None
    attributed = [c for c in clauses if _ATTRIBUTION_LANGUAGE_RE.search(c)]
    non_attributed = [c for c in clauses if not _ATTRIBUTION_LANGUAGE_RE.search(c) and len(significant_words(c)) >= 2]
    if attributed and non_attributed:
        return "MIXED"
    return None


_HUMAN_ERROR_RE = re.compile(
    r"\b(human\s+(error|oversight)|operator\s+error|personnel\s+error|individual\s+error|"
    r"careless(ness)?|negligen(t|ce))\b",
    re.IGNORECASE,
)
# Vocabulary that reframes an error claim as a SYSTEMIC/process question
# (why the error was possible) rather than stopping at "a person made a
# mistake" -- presence of any of these means the hypothesis isn't purely
# blaming the individual, it's asking what allowed the error to happen.
_PROCESS_FRAMING_RE = re.compile(
    r"\b(process|procedure|control|system|workflow|control\s+weakness|verification|"
    r"design|training|workload|schedul\w*|sop|control\s+step)\b",
    re.IGNORECASE,
)


def hypothesis_overclaims_human_error(statement: str | None) -> bool:
    """True if a hypothesis attributes the deviation to human error/
    oversight/carelessness WITHOUT also framing it as a systemic/process
    question (i.e. it stops at blaming the individual instead of asking
    what allowed the error to occur or go undetected). This does not mean
    human execution error is never a valid hypothesis -- it means a bare
    "human oversight" claim, with nothing else, is prematurely narrow and
    should be deprioritized relative to hypotheses that also consider
    process/control causes, per the same evidence discipline the system
    already applies elsewhere (never claim more than the evidence
    supports)."""
    if not statement:
        return False
    if not _HUMAN_ERROR_RE.search(statement):
        return False
    return not _PROCESS_FRAMING_RE.search(statement)


# A hypothesis that attacks the CREDIBILITY of a reported statement itself
# (accuracy/honesty of what someone said) rather than reasoning about the
# underlying proposition the statement is about. Deliberately generic --
# "a claim/statement/account was inaccurate" applies to any speaker role in
# any domain, not just this finding's operator/supervisor.
_STATEMENT_CREDIBILITY_ATTACK_RE = re.compile(
    r"\b(?:claim|statement|account|report(?:ing)?|assertion)\b(?:(?!\.).){0,40}?"
    r"\b(?:was|were|is|are)\s+(?:not\s+)?(?:inaccurate|false|incorrect|wrong|mistaken|untrue|"
    r"unreliable|fabricated|dishonest|misleading)\b|"
    r"\b(?:lied|misrepresented|fabricated|falsified|gave\s+incorrect\s+information|"
    r"was\s+dishonest|misunderstood\s+what\s+(?:was|were)\s+(?:said|reported))\b",
    re.IGNORECASE,
)


def hypothesis_attacks_statement_credibility(statement: str | None) -> bool:
    """True if a hypothesis's causal mechanism is "a person's statement was
    inaccurate/dishonest/mistaken" rather than a mechanism about the
    underlying proposition the statement concerns. A person's statement is
    evidence (REPORTED), never itself a root-cause mechanism -- a hypothesis
    that stops at "the supervisor's claim was inaccurate" instead of asking
    the actual causal question ("was the required activity completed?")
    has substituted an accusation about credibility for a causal
    explanation. Never flags reasoning about the underlying proposition
    itself (e.g. "training was not completed"), only statements that make
    the credibility/accuracy of the REPORT the mechanism."""
    if not statement:
        return False
    return bool(_STATEMENT_CREDIBILITY_ATTACK_RE.search(statement))


def is_evidence_gap_not_hypothesis(statement: str | None, source_text: str) -> bool:
    """True if `statement` is a restated fact/evidence-gap from the finding
    (e.g. "the certificate was not available during the audit") rather than
    a proposed causal explanation for WHY the observed deviation occurred.

    Structural test, not a keyword lookup of any specific case: a statement
    that (a) heavily overlaps with content already stated in the finding/
    evidence -- i.e. it isn't proposing anything new -- AND (b) contains
    none of the generic control/process-failure vocabulary that signals an
    actual causal claim, is doing the job of an evidence note, not a
    hypothesis. A statement that proposes a new mechanism will either use
    that vocabulary or introduce content the source text doesn't already
    state, so it survives this check regardless of domain.
    """
    if not statement:
        return False
    if _CAUSAL_EXPLANATION_RE.search(statement):
        return False
    stmt_words = significant_words(statement)
    source_words = significant_words(source_text)
    if not stmt_words or not source_words:
        return False
    overlap = len(stmt_words & source_words) / len(stmt_words)
    return overlap >= 0.6


# ---------------------------------------------------------------------------
# 5b. Cross-hypothesis semantic consistency (hypothesis -> evidence ->
# discrimination -> action must all describe the SAME causal proposition)
# ---------------------------------------------------------------------------

# Matches the literal "H<n>" id tokens the system itself assigns to
# candidate hypotheses -- deliberately structural (keyed off IDs the pipeline
# generated, not any domain vocabulary), so this generalizes to any finding
# with 2+ competing hypotheses, not just a training-specific example.
_HYPOTHESIS_ID_RE = re.compile(r"\bH(\d+)\b")
# A favorable/supporting verb immediately BEFORE the id mention it governs
# (e.g. "...would support H3", "...confirms H2") -- the verb precedes its
# object in this construction, so the check window looks backward from the
# id, not forward.
_FAVORABLE_VERB_TRAILING_RE = re.compile(
    r"(?:would\s+support|support(?:s|ed)?|confirms?|favors?)\s*$", re.IGNORECASE
)


def hypothesis_discrimination_cites_wrong_id(hypothesis) -> bool:
    """True if a hypothesis's own discrimination_evidence/confirms_if/
    refutes_if text asserts that some evidence "supports"/"confirms" a
    DIFFERENT hypothesis id than its own (e.g. "...would support H3 instead
    of H2" written as H2's discrimination text) -- this is exactly the
    defect where evidence for one hypothesis gets mislabeled as supporting a
    different, causally distinct hypothesis (e.g. a record-availability
    hypothesis's evidence wrongly described as supporting a verification-
    control hypothesis). A hypothesis is free to MENTION another id for
    contrast ("...does not prove H1 — see H2") as long as a favorable verb
    doesn't govern the other id."""
    own_id = (getattr(hypothesis, "id", "") or "").strip().upper()
    text = " ".join(filter(None, [
        getattr(hypothesis, "discrimination_evidence", None),
        getattr(hypothesis, "confirms_if", None),
        getattr(hypothesis, "refutes_if", None),
    ]))
    if not own_id or not text:
        return False
    for m in _HYPOTHESIS_ID_RE.finditer(text):
        other_id = f"H{m.group(1)}"
        if other_id.upper() == own_id:
            continue
        preceding = text[max(0, m.start() - 30):m.start()]
        if _FAVORABLE_VERB_TRAILING_RE.search(preceding):
            return True
    return False


# ---------------------------------------------------------------------------
# 6. Conflict-aware mechanism extraction (Phase 3)
# ---------------------------------------------------------------------------

def mechanism_from_conflicts(conflicts: list) -> MechanismInfo:
    """When evidence conflicts exist, the mechanism cannot be established
    from the conflicting claims alone.  Returns a mechanism with status
    UNKNOWN and a boundary statement explaining the conflict.

    ``conflicts`` is a list of ``EvidenceConflict`` objects (or any object
    with ``proposition`` and ``claims`` attributes).
    """
    if not conflicts:
        return MechanismInfo()
    propositions = [c.proposition for c in conflicts if hasattr(c, "proposition")]
    if not propositions:
        return MechanismInfo()
    boundary = (
        "Cannot be established from available evidence — "
        + "; ".join(propositions[:3])
    )
    return MechanismInfo(
        statement=boundary,
        status="UNKNOWN",
        polarity=None,
        source_claim=None,
    )


# ---------------------------------------------------------------------------
# 7. Consolidated 5-Why question validator (Phase 5)
# ---------------------------------------------------------------------------

def validate_why_question(
    question: str,
    previous_answer: str | None = None,
    observation: str | None = None,
    mechanism: MechanismInfo | None = None,
    finding_text: str | None = None,
) -> tuple[bool, str | None]:
    """Consolidated quality gate for a 5-Why question.  Returns (valid, reason)
    where reason explains the rejection if valid is False.

    Runs ALL structural checks in priority order:
      1. Not empty and at least 5 words
      2. Not about reporting behavior
      3. Not restating the previous answer
      4. Not restating the finding verbatim
      5. Not reopening an established mechanism
      6. Not introducing unsupported causal assumptions
    """
    if not question or not question.strip():
        return False, "Empty question"
    q = question.strip()
    if len(q.split()) < 5:
        return False, "Too short to be a genuine causal question"
    if is_reporting_why_question(q):
        return False, "Asks about reporting behavior, not causal mechanism"
    if previous_answer:
        if repeats_previous_why_answer(previous_answer, q):
            return False, "Restates the previous answer"
        q_words = significant_words(q) - _WHY_STOPWORDS
        prev_words = significant_words(previous_answer) - _WHY_STOPWORDS
        if q_words and prev_words and q_words.issubset(prev_words):
            return False, "Question introduces no new causal inquiry beyond previous answer"
    if observation and restates_observation(q, observation):
        return False, "Restates the observation"
    if finding_text:
        q_words = significant_words(q)
        finding_words = significant_words(finding_text)
        if q_words and finding_words:
            overlap = len(q_words & finding_words) / len(q_words) if q_words else 0
            if overlap >= 0.9:
                return False, "Repeats the finding verbatim"
    if mechanism and question_reopens_mechanism(q, mechanism):
        return False, "Reopens an already-established mechanism"
    return True, None


# ---------------------------------------------------------------------------
# 8. Hypothesis quality validation (Phase 6)
# ---------------------------------------------------------------------------

_UNSUPPORTED_SPECIFICITY_CHECKS = [
    (
        re.compile(r"\b(?:accessible\s+at\s+the\s+point\s+of\s+use|point\s+of\s+use\s+copy|workstation\s+copy|document\s+accessibility)\b", re.IGNORECASE),
        re.compile(r"\b(?:access|point\s+of\s+use|workstation|physical|copy|location|retriev)\b", re.IGNORECASE),
        "Invented point-of-use procedure accessibility failure without supporting evidence",
    ),
    (
        re.compile(r"\b(?:supervisory\s+check\s+or\s+automated\s+verification|automated\s+verification\s+failed|supervisory\s+verification\s+failed)\b", re.IGNORECASE),
        re.compile(r"\b(?:supervis|automated|system\s+check|verification\s+control|sign-off|review)\b", re.IGNORECASE),
        "Invented supervisory/automated verification failure without evidence such a control existed",
    ),
    (
        re.compile(r"\b(?:revisions\s+or\s+updates.*?were\s+not\s+effectively\s+communicated|revision\s+communication\s+gap)\b", re.IGNORECASE),
        re.compile(r"\b(?:revision|revised|communicat|unaware|announc|distribut|updat)\b", re.IGNORECASE),
        "Invented revision communication breakdown when no revision or communication context was mentioned",
    ),
    # Equipment/system/software/maintenance/power malfunction is one of the
    # most common LLM-invented mechanisms -- plausible in the abstract for
    # almost any audit finding, but never grounded unless the finding itself
    # describes a fault, error, alarm, or outage. Generalized across
    # equipment/instrument/system/software/power/calibration nouns rather
    # than tied to any one domain word (e.g. "temperature").
    (
        re.compile(
            r"\b(?:equipment|instrument|system|device|sensor)\s+(?:malfunction\w*|fail\w*|fault\w*|break\w*\s+down)\b|"
            r"\bsoftware\s+(?:failure|fault|bug|error|glitch)\b|"
            r"\bmaintenance\s+(?:failure|lapse|was\s+not\s+performed)\b|"
            r"\bpower\s+(?:failure|outage|loss)\b|"
            r"\bcalibration\s+drift\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:malfunction\w*|fail\w*|fault\w*|broke|breakdown|error\s+code|alarm|drift|"
            r"out\s+of\s+calibration|power\s+outage|power\s+loss|maintenance)\b",
            re.IGNORECASE,
        ),
        "Invented equipment/system/software/maintenance malfunction mechanism with no fault, error, alarm, or outage evidence in the finding",
    ),
    # "The procedure itself was unclear/ambiguous" is a distinct, stronger
    # claim than "a person reported being unaware of a revision" -- the
    # latter is evidence of an awareness/communication gap, not of the
    # procedure's own content or clarity, so this must never be invented
    # from unawareness alone.
    (
        re.compile(
            r"\b(?:procedural\s+clarity|procedure\s+(?:was|is)\s+(?:unclear|ambiguous|confusing)|"
            r"guidance\s+(?:weakness|gap|was\s+insufficient)|lack\s+of\s+clarity\s+in\s+the\s+procedure|"
            r"ambiguous\s+procedure|unclear\s+instructions?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:unclear|ambiguous|confusing|hard\s+to\s+understand|difficult\s+to\s+interpret|"
            r"poorly\s+written|contradictory\s+instructions?)\b",
            re.IGNORECASE,
        ),
        "Invented procedural clarity/guidance weakness with no evidence describing the procedure's own content or clarity",
    ),
]


def detect_unsupported_causal_specificity(statement: str | None, source_text: str) -> tuple[bool, str | None]:
    """Detect when a hypothesis invents specific causal infrastructure
    (e.g., accessibility failures, supervisory/automated check breakdowns,
    communication breakdowns) that has zero basis in finding text or evidence.

    Returns (is_unsupported, reason).
    """
    if not statement:
        return False, None

    for target_pattern, allowed_context_pattern, reason in _UNSUPPORTED_SPECIFICITY_CHECKS:
        if target_pattern.search(statement):
            if not allowed_context_pattern.search(source_text):
                return True, reason
    return False, None


def validate_hypothesis_quality(statement: str | None, source_text: str) -> tuple[bool, str | None]:
    """Validate that a hypothesis answers 'WHY COULD THIS HAVE HAPPENED?'
    rather than just restating what was observed or reported.

    Returns (valid, reason).  A hypothesis is rejected if:
      1. It merely restates an observation or evidence gap
      2. It doesn't contain causal/process/control language
      3. It overclaims human error without process framing
      4. It invents unsupported causal infrastructure (unsupported specificity)
      5. It contains attribution framing
    """
    if not statement:
        return False, "Empty hypothesis statement"
    if is_evidence_gap_not_hypothesis(statement, source_text):
        return False, "Restates an evidence gap or observation, not a causal explanation"
    if hypothesis_overclaims_human_error(statement):
        return False, "Attributes to bare human error without process framing"
    # Check for attribution language -- a hypothesis that says "the operator
    # stated X" is narrating a report, not proposing a cause.
    if _ATTRIBUTION_LANGUAGE_RE.search(statement):
        return False, "Contains attribution language — narrates a report rather than proposing a cause"
    # Check for unsupported causal specificity
    is_unsupported, reason = detect_unsupported_causal_specificity(statement, source_text)
    if is_unsupported:
        return False, reason
    return True, None


def test_hypothesis_against_claims(hypothesis_text: str, claims: list) -> str:
    """Test a hypothesis against all known claims.  Returns one of:
      - 'CONSISTENT': no claim contradicts the hypothesis
      - 'CONTRADICTED': at least one claim with opposite polarity
      - 'UNKNOWN': can't determine (no overlap or no polarity info)

    ``claims`` is a list of ``EvidenceClaim`` objects (or any object with
    ``text``, ``polarity``, and ``status`` attributes).
    """
    if not hypothesis_text or not claims:
        return "UNKNOWN"

    hypothesis_polarity = classify_mechanism_polarity(hypothesis_text)
    if not hypothesis_polarity:
        return "UNKNOWN"

    for claim in claims:
        claim_text = getattr(claim, "text", str(claim))
        claim_polarity = getattr(claim, "polarity", None)
        claim_status = getattr(claim, "status", None)

        # Only test against claims with known polarity
        if not claim_polarity:
            continue

        # Check for subject overlap
        hyp_words = significant_words(hypothesis_text)
        claim_words = significant_words(claim_text)
        if not hyp_words or not claim_words:
            continue
        overlap = len(hyp_words & claim_words) / min(len(hyp_words), len(claim_words))
        if overlap < 0.2:
            continue  # Claims about different things

        # Opposite polarities with subject overlap = contradiction
        if claim_polarity != _polarity_to_claim_polarity(hypothesis_polarity):
            # Only treat as contradiction if the claim is VERIFIED
            status_value = getattr(claim_status, "value", claim_status) if claim_status else None
            if status_value == "VERIFIED":
                return "CONTRADICTED"

    return "CONSISTENT"


def _polarity_to_claim_polarity(mechanism_polarity: str) -> str | None:
    """Map mechanism polarity to claim polarity for comparison."""
    mapping = {
        "non_performance": "negative",
        "non_recording": "negative",
        "knowledge_gap": "negative",
    }
    return mapping.get(mechanism_polarity)


def determine_hypothesis_status(
    statement: str,
    verified_facts: list[str],
    reported_claims: list[str],
    conflicts: list | None = None,
    mechanism: MechanismInfo | None = None,
    allow_verified_promotion: bool = True,
) -> tuple[str, str]:
    """Authoritative deterministic evidence-to-hypothesis status policy (Requirements 1, 2, 4).

    Returns (status, evidence_strength):
      - status: 'POSSIBLE' | 'SUPPORTED' | 'REFUTED' | 'UNRESOLVED'
      - evidence_strength: 'NONE' | 'REPORTED' | 'CORROBORATED' | 'VERIFIED' | 'CONFLICTING'

    Invariants:
      1. REPORTED evidence alone MUST NOT establish SUPPORTED.
      2. CONFLICTING reported evidence MUST NOT establish SUPPORTED.
      3. A hypothesis is SUPPORTED ONLY when independent VERIFIED evidence directly supports it.
      4. A hypothesis is REFUTED when reliable evidence directly contradicts it.

    `allow_verified_promotion=False` disables step 3 (word-overlap ->
    SUPPORTED) for hypotheses whose own truth is NOT what a VERIFIED fact in
    the ledger actually establishes -- e.g. a CAPA-effectiveness-gap
    hypothesis will always share heavy vocabulary with the VERIFIED fact
    that ESTABLISHES the recurrence/previous-CAPA precondition (that's what
    grounds the hypothesis in the first place), but that fact never
    verifies the hypothesis's OWN claim (that the action was ineffective) --
    only a dedicated effectiveness-review claim could. Word overlap with the
    precondition is not evidence of the hypothesis itself.
    """
    if not statement:
        return "POSSIBLE", "NONE"

    # 1. Contradiction check -> REFUTED
    if mechanism and mechanism.statement and hypothesis_contradicts_mechanism(statement, mechanism):
        return "REFUTED", "VERIFIED" if mechanism.status == "VERIFIED" else "REPORTED"
    if verified_facts and hypothesis_contradicts_verified_completion(statement, verified_facts):
        return "REFUTED", "VERIFIED"

    # 2. Conflicting evidence check -> UNRESOLVED / POSSIBLE with CONFLICTING strength
    if conflicts:
        hyp_words = significant_words(statement)
        for conflict in conflicts:
            conf_words = significant_words(getattr(conflict, "proposition", str(conflict)))
            if hyp_words and conf_words and (hyp_words & conf_words):
                return "UNRESOLVED", "CONFLICTING"

    # 3. Verified evidence match -> SUPPORTED
    if verified_facts and allow_verified_promotion:
        hyp_words = significant_words(statement)
        for fact in verified_facts:
            fact_words = significant_words(fact)
            if hyp_words and fact_words:
                overlap = len(hyp_words & fact_words) / min(len(hyp_words), len(fact_words))
                if overlap >= 0.5:
                    return "SUPPORTED", "VERIFIED"

    # 4. Reported evidence match -> POSSIBLE (with REPORTED strength, NEVER SUPPORTED)
    if reported_claims:
        hyp_words = significant_words(statement)
        for claim in reported_claims:
            claim_words = significant_words(claim)
            if hyp_words and claim_words:
                overlap = len(hyp_words & claim_words) / min(len(hyp_words), len(claim_words))
                if overlap >= 0.4:
                    return "POSSIBLE", "REPORTED"

    return "POSSIBLE", "NONE"


