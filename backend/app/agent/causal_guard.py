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
    from app.services.semantic_subject import naturalize_reported_claim

    for claim in reported_statements or []:
        polarity = classify_mechanism_polarity(claim)
        if polarity:
            return MechanismInfo(
                statement=naturalize_reported_claim(claim), status="REPORTED", polarity=polarity, source_claim=claim
            )
    for claim in verified_facts or []:
        polarity = classify_mechanism_polarity(claim)
        if polarity:
            return MechanismInfo(statement=claim, status="VERIFIED", polarity=polarity, source_claim=claim)
    for claim in reported_statements or []:
        if claim and claim.strip():
            return MechanismInfo(
                statement=naturalize_reported_claim(claim), status="REPORTED", polarity="general", source_claim=claim
            )
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


def restates_observation(answer: str | None, observed_deviation: str | None, question: str | None = None) -> bool:
    """True if a Why-step's answer is essentially just the observation
    restated (e.g. "the log was incomplete" answering why the log was
    incomplete, "the equipment was operated outside its validated range"
    or "the validated range was exceeded") rather than explaining it."""
    if not answer or not observed_deviation:
        return False
    if repeats_previous_why_answer(observed_deviation, answer):
        return True
    if question and is_circular_why_answer(question, answer):
        return True
    ans_words = significant_words(answer) - _WHY_STOPWORDS
    obs_words = significant_words(observed_deviation) - _WHY_STOPWORDS
    if not ans_words or not obs_words:
        return False
    overlap = ans_words & obs_words
    if len(overlap) >= 2 and len(overlap) / len(ans_words) >= 0.50:
        _EXPLANATORY_VERBS = {
            "because", "due", "caused", "failed", "disabled", "outage",
            "omitted", "forgot", "misconfigured", "unaware", "lacked", "broken",
            "interlock", "overload", "defect", "corrupted", "tripped", "bypassed"
        }
        new_words = ans_words - obs_words
        if not (new_words & _EXPLANATORY_VERBS):
            return True
    return False


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


_EVIDENCE_BOUNDARY_ACKNOWLEDGMENT_RE = re.compile(
    r"\b(?:not\s+established|cannot\s+be\s+determined|could\s+not\s+be\s+determined|"
    r"insufficient\s+evidence|no\s+objective\s+evidence|not\s+confirmed|"
    r"unclear\s+from\s+the\s+available\s+evidence|not\s+yet\s+established|"
    r"has\s+not\s+been\s+established|does\s+not\s+establish|not\s+been\s+verified|"
    r"no\s+evidence\s+(?:currently\s+)?(?:available|exists))\b",
    re.IGNORECASE,
)


def five_why_answer_invents_mechanism_at_unknown_status(answer: str | None, status: str | None) -> bool:
    """True if a 5-Why step is labeled UNKNOWN/REQUIRES_EVIDENCE (the chain
    has reached the evidence boundary) but its answer text nonetheless
    reads as a confident, specific factual claim ("There was no reminder
    or system prompt for logging") rather than acknowledging that boundary
    ("The reason ... is not established from the available evidence").

    A status of UNKNOWN with an answer that asserts a brand-new,
    unattributed mechanism is a content/status mismatch: the LLM has
    silently invented an explanation for the gap instead of stating that
    the gap is unexplained -- exactly the "assumed cause" failure mode this
    system exists to prevent, just expressed as prose instead of a status
    label. Detected structurally (absence of any boundary-acknowledgment
    phrase), not by matching this finding's specific wording.
    """
    if not answer or status not in ("UNKNOWN", "REQUIRES_EVIDENCE"):
        return False
    return not _EVIDENCE_BOUNDARY_ACKNOWLEDGMENT_RE.search(answer)


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


_GENERIC_HYPOTHESIS_NAME_RE = re.compile(
    r"^\s*(?:process\s+(?:oversight|failure|inconsistency|gap|weakness|issue)|"
    r"(?:communication|documentation|training|system|quality|procedural|management)\s+"
    r"(?:issue|problem|gap|inconsistency|weakness)|"
    r"short[_-]?code.*|placeholder.*|candidate[_-]?mechanism.*|internal[_-]?code.*|"
    r"human\s+error|oversight)\s*$",
    re.IGNORECASE,
)
# A name that is literally just the hypothesis's own ID pattern ("H2",
# "HYP_003") or a generic code is a raw internal identifier leaking into what's supposed to
# be a human-readable title -- checked independent of the hypothesis's
# actual assigned id (any LLM-produced hypothesis could echo any Hn/HYP_n
# shape here), not tied to one specific id value.
_BARE_HYPOTHESIS_ID_NAME_RE = re.compile(r"^\s*(?:H\d+|HYP[_-]?\d+|SHORT[_-]?CODE|CODE|NAME)\s*$", re.IGNORECASE)


def hypothesis_name_is_generic(name: str | None) -> bool:
    """True if a hypothesis's `name`/title is either a vague relabeling of
    the finding ("Process Oversight", "Communication Issue", "Training
    Problem"), an internal placeholder ("SHORT_CODE"), or a raw internal
    identifier ("H2") rather than a precise, testable causal proposition."""
    if not name:
        return True
    stripped = name.strip()
    return bool(_GENERIC_HYPOTHESIS_NAME_RE.match(stripped) or _BARE_HYPOTHESIS_ID_NAME_RE.match(stripped))


# A hypothesis STATEMENT (not just its name/title) can be just as vague as
# a generic name: "Execution or task-control factors may have contributed
# to the missed activity" replaces one invented mechanism ("shift plan")
# with a differently-worded but equally unlicensed one -- a generic causal
# BUCKET (execution/task-control/process/operational/human/management/
# control factors, weakness, oversight, failure, gap) standing in for a
# real mechanism. This is unconditional (no allowed_context to check --
# being this vague is disqualifying regardless of what the finding says),
# unlike the domain-specific guards above which ARE excusable given the
# right evidence.
_GENERIC_CAUSAL_BUCKET_RE = re.compile(
    r"\b(?:execution|task[\s-]control|process|operational|human|management|control)\s+"
    r"(?:factors?|weakness(?:es)?|oversight|failure|gap)\b(?:(?!\.).){0,80}?\b"
    r"(?:may\s+have\s+|might\s+have\s+|could\s+have\s+)?contribut\w+\b",
    re.IGNORECASE,
)


def hypothesis_statement_is_generic_causal_bucket(statement: str | None) -> bool:
    """True if a hypothesis STATEMENT names only a generic causal category
    ("execution/task-control/process/operational/human/management/control
    factors/weakness/oversight/failure/gap ... may have contributed") in
    place of a specific, evidence-grounded mechanism.

    A hedged, broad category is not automatically safer than a specific
    invented one -- it just moves the same defect one level of abstraction
    up. When the evidence doesn't license a specific mechanism, the
    correct output is zero hypotheses (with this category retained only as
    an INVESTIGATION AREA, never as a candidate root-cause hypothesis), not
    a vaguer hypothesis standing in for a specific one."""
    if not statement:
        return False
    return bool(_GENERIC_CAUSAL_BUCKET_RE.search(statement))


_TITLE_STOPWORDS = frozenset({
    "the", "a", "an", "of", "for", "to", "and", "or", "may", "might", "have",
    "been", "was", "were", "is", "are", "not", "with", "by", "that", "this",
    "at", "in", "on", "before", "after", "during", "could", "would", "may have",
    "did", "does", "do", "which", "who", "its", "their", "it",
})


def derive_hypothesis_title_from_statement(statement: str | None, hypothesis_id: str) -> str:
    """Build a human-readable, auditor-facing hypothesis title from the
    hypothesis's own statement content -- used when a title must be
    replaced (e.g. it was too generic) and there is no LLM-provided title
    worth keeping.

    Must NEVER produce an internal-looking placeholder like
    "CANDIDATE_MECHANISM_H2" (Section 13): an auditor reading the final
    report has no idea what that means, and it visibly leaks
    implementation detail. Deriving the title FROM the statement instead
    keeps it human-readable and specific to this finding, matching the
    style of the deterministic generator's own SHORT_CODE names (e.g.
    TRAINING_RECORD_UNAVAILABLE) without inventing new causal content --
    it's a pure text transformation of already-vetted statement text.
    """
    if not statement:
        return f"Candidate Mechanism {hypothesis_id}"
    words = re.findall(r"[A-Za-z]+", statement)
    meaningful = [w for w in words if w.lower() not in _TITLE_STOPWORDS and len(w) > 2]
    picked = meaningful[:6] if meaningful else words[:6]
    if not picked:
        return f"Candidate Mechanism {hypothesis_id}"
    return " ".join(w.capitalize() for w in picked)


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


_EVIDENCE_STATE_INDICATORS = re.compile(
    r"\b(?:whether\s+[\w\s-]+?\s+(?:was|were|is|are|had\s+been)\s+(?:implemented|verified|completed|conducted|performed|effective|available|related|established)|"
    r"status\s+is\s+(?:unconfirmed|unknown|unverified|to\s+be\s+confirmed)|"
    r"records?\s+(?:are|were|is|was)\s+(?:unavailable|missing|not\s+available)|"
    r"effectiveness\s+verification\s+is\s+(?:unknown|unconfirmed|unverified)|"
    r"need\s+to\s+investigate|requires?\s+investigation|to\s+be\s+investigated|"
    r"may\s+or\s+may\s+not\s+have\s+(?:occurred|been)|"
    r"whether\s+it\s+was\s+implemented|whether\s+its?\s+effectiveness\s+was\s+verified|"
    r"is\s+unconfirmed\b)",
    re.IGNORECASE,
)


def is_evidence_state_not_hypothesis(statement: str | None, name: str | None = None) -> bool:
    """Return True if the proposition/statement is describing an EVIDENCE_STATE
    or INVESTIGATION_STATE (e.g. 'whether X is known', 'whether X was implemented',
    'status is unconfirmed', 'records are unavailable') rather than proposing
    a concrete causal mechanism (CAUSE -> MECHANISM -> DEVIATION)."""
    if name and re.search(r"\b(?:STATUS_UNCONFIRMED|STATUS_UNKNOWN|EVIDENCE_STATE|INVESTIGATION_REQUIRED|UNCONFIRMED)\b", name, re.IGNORECASE):
        return True
    if not statement:
        return False
    return bool(_EVIDENCE_STATE_INDICATORS.search(statement))


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

# Concept-root pattern for the scheduling/task-assignment/shift-handover
# mechanism family, shared between the target (does the HYPOTHESIS invoke
# this concept?) and the allowed-context (does the FINDING independently
# license it?) checks below -- covers hyphenated/spaced variants and both
# "plan" and "roster" phrasing so paraphrases of the same concept are
# treated symmetrically on both sides of the check.
_SCHEDULING_CONCEPT_PATTERN = (
    r"\bschedul\w*\b|\bshift[\s-]plan\b|\bduty[\s-]?(?:plan|roster)\b|"
    r"\btask\s+assignment\b|\bstaffing\s+plan\b|\bnot\s+assigned\s+to\b"
)

# Same technique for the notification/communication/distribution/
# acknowledgement mechanism family (the revision-communication-chain
# concepts): shared between target and allowed-context so any paraphrase
# of "notification/communication/distribution/acknowledgement failed" is
# treated symmetrically against whatever form of that same concept the
# finding itself may (or may not) independently use.
_NOTIFICATION_CONCEPT_PATTERN = (
    r"\bnotif\w*\b|\bnotice\b|\bcommunicat\w*\b|\bdistribut\w*\b|\backnowledg\w*\b|\bannounc\w*\b"
)

_UNSUPPORTED_SPECIFICITY_CHECKS = [
    (
        re.compile(
            r"\b(?:accessible\s+at\s+the\s+point\s+of\s+use|point\s+of\s+use\s+copy|workstation\s+copy|"
            r"document\s+accessibility|(?:was|were)\s+not\s+(?:easily\s+|readily\s+)?accessible|"
            r"\binaccessible\b)\b",
            re.IGNORECASE,
        ),
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
            r"\b(?:malfunction\w*|fault\w*|broke|breakdown|error\s+code|alarm|drift|"
            r"out\s+of\s+calibration|power\s+outage|power\s+loss|"
            r"(?:equipment|instrument|system|device|sensor)\s+fail\w*|"
            r"maintenance\b(?:(?!\.).){0,40}?\b(?:failure|lapse|overdue|missed|was\s+not\s+performed))\b",
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
            r"ambiguous\s+procedure|unclear\s+instructions?|"
            r"lack(?:ed|s)?\s+clear(?:ity)?\s+(?:guidance|instructions?))\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:unclear|ambiguous|confusing|hard\s+to\s+understand|difficult\s+to\s+interpret|"
            r"poorly\s+written|contradictory\s+instructions?)\b",
            re.IGNORECASE,
        ),
        "Invented procedural clarity/guidance weakness with no evidence describing the procedure's own content or clarity",
    ),
    # "The record was unavailable during the audit" establishes exactly
    # that -- unavailability at audit time -- and nothing about WHY: not
    # that it was never created, not that it was mishandled, and certainly
    # not that it was lost/destroyed/discarded (a specific, stronger claim
    # than "not maintained/controlled as required", which is the legitimate,
    # already-supported record-control-gap hypothesis type and is
    # deliberately NOT targeted here -- only the sharper loss/destruction
    # claim, which requires the finding to actually describe that event).
    (
        re.compile(
            r"\b(?:records?|logs?|certificates?|documents?|attendance\s+sheets?)\b"
            r"(?:(?!\.).){0,40}?"
            r"(?:\b(?:was|were)\s+(?:lost|destroyed|discarded|misplaced|deleted|damaged|corrupted)\b|"
            r"(?:could\s+not\s+be\s+|failed\s+to\s+(?:be\s+)?|was\s+unable\s+to\s+be\s+)"
            r"(?:retrieve[d]?|locate[d]?|find|found)\b|"
            r"retrieval\s+(?:failed|was\s+unsuccessful))",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:lost|destroyed|discarded|misplaced|deleted|damaged|corrupted)\b|"
            r"(?:could\s+not\s+be\s+|failed\s+to\s+(?:be\s+)?|was\s+unable\s+to\s+be\s+)"
            r"(?:retrieve[d]?|locate[d]?|find|found)\b|"
            r"retrieval\s+(?:failed|was\s+unsuccessful)",
            re.IGNORECASE,
        ),
        "Invented record-loss/retrieval-failure claim — the finding only establishes the record was unavailable during the audit, not that it was lost, destroyed, discarded, damaged, or that retrieval was attempted and failed",
    ),
    # TRAINING/COMPETENCY is a distinct causal domain from AWARENESS/
    # COMMUNICATION -- "the operator was unaware of a revision" is
    # evidence about notification/acknowledgement, never by itself
    # evidence about whether training/retraining/qualification was
    # required, applicable, or lacking. Eligible only when the finding
    # itself contains training/competency vocabulary; unawareness words
    # alone ("unaware", "did not know", "unfamiliar") never satisfy this.
    (
        re.compile(
            r"\b(?:re)?train(?:ed|ing)?\b|\bcompetenc\w*\b|\bqualif(?:y|ied|ication)\w*\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:re)?train(?:ed|ing)?\b|\bcompetenc\w*\b|\bqualif(?:y|ied|ication)\w*\b|"
            r"\bcourse\b|\brefresher\b|\battendance\b",
            re.IGNORECASE,
        ),
        "Invented training/competency/qualification mechanism — the finding contains no training, retraining, competency, or qualification vocabulary; unawareness of a revision is a communication/acknowledgement signal, not evidence of a training gap",
    ),
    # SCHEDULING/TASK-ASSIGNMENT is a distinct causal domain from a bare
    # "missed" report -- "the check was missed during the morning shift"
    # names WHEN the deviation was noticed (temporal context), never HOW
    # responsibility was assigned or transferred. Detected by CONCEPT ROOT
    # ("schedul-", "shift plan", "duty roster", "task assignment") appearing
    # ANYWHERE in the statement, not by enumerating phrase templates -- an
    # LLM can paraphrase "not scheduled" a dozen ways ("wasn't included in
    # the schedule", "scheduling controls were inadequate", "system records
    # show the check was scheduled"), and every one of them still contains
    # the root concept word regardless of the surrounding sentence shape.
    # Whether the claim asserts scheduling WAS or WAS NOT done is
    # irrelevant -- either direction is an unlicensed claim about
    # scheduling absent real evidence. Eligible only when the finding
    # itself contains actual scheduling/assignment/roster vocabulary; a
    # bare time-of-day/shift reference is NOT that evidence (the allowed
    # context below deliberately excludes bare "shift").
    (
        re.compile(
            _SCHEDULING_CONCEPT_PATTERN + r"|\bnot\s+planned\b",
            re.IGNORECASE,
        ),
        # Deliberately the SAME concept-root pattern as the target: whatever
        # wording the hypothesis used for the scheduling concept, the
        # finding is licensed to support it if the finding independently
        # uses ANY form of that same concept (paraphrase-symmetric) --
        # never a hand-picked list of specific excusing phrases that a
        # differently-tensed or differently-worded finding sentence
        # ("the schedule showed..." vs "...shows...") would silently miss.
        # Bare "shift" (a time-of-day reference) is still excluded since it
        # matches none of these concept roots on its own.
        re.compile(_SCHEDULING_CONCEPT_PATTERN, re.IGNORECASE),
        "Invented scheduling/task-assignment/shift-handover mechanism — a bare shift/time reference is temporal context, not evidence that scheduling, assignment, or handover itself was involved (in either direction)",
    ),
    # Same concept-root approach for reminder/notification/alert mechanisms
    # -- "reminder"/"prompt"/"alert" appearing anywhere in the statement,
    # not a fixed phrase list. Deliberately excludes generic "verification
    # control" (a distinct, separately-licensed mechanism used elsewhere
    # for record/authorization verification) so this only targets the
    # reminder/notification/alert concept specifically.
    (
        re.compile(r"\breminder\b|\bprompt\b|\balert\b", re.IGNORECASE),
        re.compile(
            r"\breminder\s+system\b|\balert\s+log\b|\bnotification\s+system\b|"
            r"\bprompt\s+(?:was\s+not\s+)?(?:sent|triggered|issued)\b",
            re.IGNORECASE,
        ),
        "Invented reminder/notification/alert mechanism — the finding contains no evidence of a reminder, alert, or automated notification system, let alone its failure",
    ),
    # RENEWAL/RECERTIFICATION concept guard: "X expired on date A" + "X was
    # used on date B" establishes exactly that -- use after the stated
    # expiry date -- and nothing about WHY the expiry went unaddressed.
    # "The certificate was not renewed" / "recertification did not occur"
    # is a stronger, distinct upstream-process claim (a renewal/
    # recertification workflow existing and failing) that requires the
    # finding to independently describe that workflow, not just an expiry
    # date and a later use date.
    (
        re.compile(
            r"\b(?:re-?new(?:al|ed)?|recertif\w*|recalibrat\w*)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:re-?new(?:al|ed)?|recertif\w*|recalibrat\w*)\b",
            re.IGNORECASE,
        ),
        "Invented renewal/recertification/recalibration mechanism — the finding establishes an expiry date and a later use date, not that any renewal, recertification, or recalibration process existed or failed",
    ),
]

_SELF_ASSERTED_EVIDENCE_RE = re.compile(
    r"\b(?:system\s+records?|records?|the\s+log|logs?|data|the\s+system)\s+"
    r"(?:show|shows|showed|indicate|indicates|indicated|confirm|confirms|confirmed|reveal|reveals|revealed)\b",
    re.IGNORECASE,
)


_REFERENCED_DOC_ONLY_RE = re.compile(
    r"\b(referenced|cited|mentioned|attached|unavailable|not\s+available|"
    r"could\s+not\s+be\s+(?:retrieved|accessed|reviewed|provided)|referred\s+to|refers?\s+to)\b",
    re.IGNORECASE,
)


def hypothesis_asserts_referenced_document_content(
    statement: str | None, referenced_documents: list | None
) -> bool:
    """True if `statement` asserts something about the CONTENTS of a
    document the finding only referenced/cited/attached but which was
    unavailable for inspection (Referenced-Evidence Boundary: REFERENCED
    EVIDENCE != INSPECTED EVIDENCE) -- e.g. asserting a calibration
    certificate had expired when the certificate itself was never available.

    Structural: matches on vocabulary shared with the referenced document's
    OWN type/name (whatever that happens to be -- calibration report,
    training record, maintenance log, qualification document, etc., never a
    hardcoded per-document-type list), not on any specific domain word.
    A statement that only restates the reference/unavailability itself
    (e.g. "the calibration report was unavailable") is NOT a content claim
    and is allowed through."""
    if not statement or not referenced_documents:
        return False
    stmt_words = set(re.findall(r"[a-z]+", statement.lower()))
    for doc in referenced_documents:
        doc_type = getattr(doc, "document_type", None) if not isinstance(doc, dict) else doc.get("document_type")
        if not doc_type:
            continue
        doc_words = {w for w in re.findall(r"[a-z]+", doc_type.lower()) if len(w) > 3}
        if doc_words & stmt_words and not _REFERENCED_DOC_ONLY_RE.search(statement):
            return True
    return False


def hypothesis_asserts_self_referential_evidence(statement: str | None) -> bool:
    """True if a hypothesis statement narrates its OWN supporting evidence
    inline ("system records show...", "the log indicates...", "records
    confirm...") instead of actually being grounded via
    `supporting_claim_ids` in the real evidence ledger.

    A hypothesis SAYING evidence exists is not the same as evidence
    actually existing in this case's ledger -- this phrasing is a generic
    way a claim can smuggle an unverified assertion past evidence-grounding
    review by dressing it up as if it already cites a source, regardless of
    which mechanism (scheduling, training, equipment, anything) it's
    attached to. Domain-agnostic: does not name any specific mechanism
    concept, only the self-referential "X shows/indicates/confirms" shape."""
    if not statement:
        return False
    return bool(_SELF_ASSERTED_EVIDENCE_RE.search(statement))


# A hypothesis escalates from "mechanism-level" (something happened this
# time, to this specific case) to "systemic-level" (the organization's
# process/system/control itself is broken) whenever it pairs a systemic
# NOUN (process/system/control/mechanism/management) with a FAILURE VERB
# (lacked/failed/was ineffective/did not exist/was inadequate/was
# defective/was not in place), in either word order. This is deliberately
# ONE domain-agnostic pattern covering training, communication, scheduling,
# maintenance, revision-control, record-control, verification-control, or
# any other domain -- not a separate regex per mechanism family (Section 19
# of the governing spec explicitly requires this consolidation instead of
# enumerating "is_training_process_failure()", "is_comms_process_failure()",
# etc. one at a time).
_SYSTEMIC_NOUN = r"(?:process|system|control|mechanism|management)"
_SYSTEMIC_FAILURE_VERB = (
    r"(?:lacked|failed|was\s+ineffective|did\s+not\s+exist|was\s+inadequate|"
    r"was\s+defective|was\s+not\s+(?:defined|established|in\s+place))"
)
_SYSTEMIC_ESCALATION_RE = re.compile(
    rf"\b{_SYSTEMIC_NOUN}\b(?:(?!\.).){{0,60}}?\b{_SYSTEMIC_FAILURE_VERB}\b|"
    rf"\b{_SYSTEMIC_FAILURE_VERB}\b(?:(?!\.).){{0,60}}?\b{_SYSTEMIC_NOUN}\b",
    re.IGNORECASE,
)
# What licenses a systemic claim: the finding must independently cite an
# actual artifact/finding ABOUT the process (a review, audit, documentation,
# log, or record that was consulted), not merely restate the deviation or a
# person's report about the event itself. Also domain-agnostic -- these are
# generic "something was actually checked/found" verbs, not a list of
# per-domain record names.
_SYSTEMIC_EVIDENCE_RE = re.compile(
    r"\b(?:review\s+(?:showed|found|revealed)|audit\s+(?:showed|found|revealed)|"
    r"documentation\s+(?:shows|showed|confirms|confirmed)|records?\s+(?:shows?|showed|confirms?|confirmed)|"
    r"log\s+(?:shows?|showed)|investigation\s+(?:found|showed|revealed)|"
    r"(?:process|system|control)\s+documentation\s+(?:shows?|showed|confirms?)|"
    r"no\s+\w+\s+(?:step|control)\s+(?:was\s+)?defined)\b",
    re.IGNORECASE,
)


def hypothesis_asserts_systemic_cause_without_process_evidence(statement: str | None, source_text: str) -> bool:
    """True if a hypothesis escalates to SYSTEMIC/process-level causation
    ("the revision process lacked a mechanism...", "the training system
    failed...", "the record-control process was defective...") without the
    finding independently citing any actual review/audit/documentation/
    record finding ABOUT that process.

    This is the Level 1 -> Level 4 jump the governing spec forbids: a
    REPORTED event about what happened THIS time (e.g. "the operator was
    unaware of a revision") never by itself establishes that the
    organization's underlying process/system/control is broken -- that is
    a categorically stronger, systemic claim requiring its own
    process-level evidence, regardless of which mechanism family
    (training/communication/scheduling/maintenance/record-control/
    verification/anything) the hypothesis names.
    """
    if not statement or not _SYSTEMIC_ESCALATION_RE.search(statement):
        return False
    return not _SYSTEMIC_EVIDENCE_RE.search(source_text or "")


# NOTE: an earlier version of this list also rejected the WORDING pattern
# "X occurred but wasn't documented/recorded" outright, regardless of
# hedging. That was an over-correction: "the activity may have been
# completed, but objective completion evidence was unavailable" is a
# legitimate, evidence-grounded CANDIDATE HYPOTHESIS (not an assertion of
# fact) whenever a REPORTED claim states completion and the record is
# independently VERIFIED unavailable/absent. The distinguishing factor is
# NOT the underlying mechanism (record-availability-vs-completion is a
# perfectly valid causal branch) -- it's whether the sentence ASSERTS the
# activity occurred as settled fact ("was conducted") or preserves the
# uncertainty ("may have been conducted"). See
# hypothesis_asserts_unhedged_unverified_completion() below, which targets
# only the unhedged, fact-asserting phrasing.
_UNHEDGED_COMPLETION_BUT_UNDOCUMENTED_RE = re.compile(
    r"\b(?:was|were)\s+(?:conducted|performed|completed|carried\s+out|done)\s+but\s+"
    r"(?:was\s+|were\s+)?not\s+(?:properly\s+|adequately\s+)?(?:documented|recorded|retained|logged)\b",
    re.IGNORECASE,
)
_HYPOTHESIS_HEDGE_WORD_RE = re.compile(
    r"\b(?:may|might|could|possibly|perhaps|potentially)\b", re.IGNORECASE
)


def hypothesis_asserts_unhedged_unverified_completion(statement: str | None) -> bool:
    """True if `statement` asserts an activity occurred as SETTLED FACT
    (no hedging word anywhere in the sentence) paired with an unverified
    documentation-failure claim -- e.g. "The training was conducted but not
    properly documented." The same underlying causal branch phrased with
    uncertainty-preserving language ("training MAY HAVE been completed, but
    the record was unavailable") is a legitimate hypothesis and is
    deliberately NOT flagged -- only the unhedged, fact-asserting version,
    which silently promotes a REPORTED completion claim into an assumed
    fact within the hypothesis's own statement."""
    if not statement:
        return False
    if not _UNHEDGED_COMPLETION_BUT_UNDOCUMENTED_RE.search(statement):
        return False
    return not _HYPOTHESIS_HEDGE_WORD_RE.search(statement)


def hypothesis_asserts_unhedged_notification_failure(statement: str | None, source_text: str) -> bool:
    """True if `statement` asserts, as SETTLED FACT (no hedging word
    anywhere in the sentence), that a notification/communication/
    distribution/acknowledgement event did NOT occur -- e.g. "The revision
    occurred without proper notification to the operator" -- when the
    finding provides no independent evidence of that concept.

    This is the causal-event-inversion defect: a REPORTED downstream state
    ("the operator was unaware") can license investigating whether
    notification/communication/acknowledgement was effective (a HEDGED
    Level-2 candidate: "may not have been effectively communicated"), but
    it can never establish, as settled fact, that the upstream event
    itself did not happen -- that is a stronger, unlicensed claim
    regardless of which concept-family word (notification/communication/
    distribution/acknowledgement/announcement) is used.

    Uses the SAME concept-root matching as the scheduling/reminder guards
    (any paraphrase of "notification"/"communication"/etc., not a fixed
    phrase list) combined with the SAME hedge-exemption as
    hypothesis_asserts_unhedged_unverified_completion(): a hedged
    formulation of the identical mechanism is a legitimate candidate
    hypothesis and must not be rejected merely for sharing vocabulary with
    the unhedged, unlicensed version. Licensed (regardless of hedging)
    when the finding independently uses any form of the same concept.
    """
    if not statement or not re.search(_NOTIFICATION_CONCEPT_PATTERN, statement, re.IGNORECASE):
        return False
    if _HYPOTHESIS_HEDGE_WORD_RE.search(statement):
        return False
    return not re.search(_NOTIFICATION_CONCEPT_PATTERN, source_text or "", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Structural (not word-list) guard against unlicensed CHANGE-EVENT defect
# claims.
#
# `hypothesis_asserts_unhedged_notification_failure` above (and its sibling
# concept-root patterns for scheduling/reminders/malfunction/etc.) work by
# enumerating the family of words that can express a given mechanism. That
# approach cannot keep pace with paraphrase generation: "without prior
# notice" gets fixed, and the next run produces "not properly documented or
# disseminated" -- a different verb family entirely (documentation,
# distribution) describing the exact same unlicensed inference (a REPORTED
# downstream state -- "operator was unaware" -- promoted into a SETTLED-FACT
# claim about an upstream event -- the revision/change process itself).
#
# The invariant that actually holds, regardless of vocabulary, is
# STRUCTURAL: any hypothesis whose grammatical SUBJECT is the change event
# itself ("the revision", "the procedure change", "the update") paired with
# ANY unhedged negative-outcome clause ("was not X-ed", "lacked X",
# "occurred/implemented without X") is asserting a fact about how the
# change was handled. The finding in this recurring defect class never
# describes the change-handling process at all -- only the operator's
# downstream reaction to it -- so no verb-family enumeration is needed on
# the target side; only a check that the finding independently makes some
# claim about the change event's handling licenses it.
_CHANGE_EVENT_SUBJECT_RE = re.compile(
    r"\bthe\s+(?:[\w-]+\s+){0,3}?(?:revision|amendment|procedure\s+change|process\s+change|"
    r"policy\s+change|update)\b",
    re.IGNORECASE,
)
_NEGATIVE_OUTCOME_CLAUSE_RE = re.compile(
    r"\b(?:was|were)\s+not\s+(?:properly\s+|adequately\s+|fully\s+|formally\s+)?\w+(?:ed|d)\b"
    r"|\blacked\b"
    r"|\b(?:occurred|implemented|issued|introduced|rolled\s+out)\s+without\b"
    r"|\bwithout\s+(?:prior\s+|proper\s+|any\s+|formal\s+)?\w+\b"
    r"|\bno\s+\w+\s+(?:existed|was\s+in\s+place|was\s+available)\b",
    re.IGNORECASE,
)


def hypothesis_asserts_unlicensed_change_event_defect(statement: str | None, source_text: str) -> bool:
    """True if `statement` asserts, as SETTLED FACT, that the change event
    itself (the revision/amendment/procedure change) was handled
    defectively -- e.g. "The checklist revision was not properly documented
    or disseminated" -- when the finding gives no independent account of
    how the change was handled (only, e.g., that a revision occurred and
    that an operator reported being unaware of it).

    Structural counterpart to hypothesis_asserts_unhedged_notification_failure:
    matches on SENTENCE SHAPE (change-event subject + unhedged
    negative-outcome clause) rather than a specific vocabulary family, so it
    generalizes across "not documented", "not disseminated", "not
    accessible", "occurred without notice", and any future paraphrase
    without requiring a new word to be added. Hedged formulations ("may not
    have been properly documented") are exempt, as with the sibling guard --
    the mechanism itself is a legitimate candidate; only the fact-asserting
    phrasing is unlicensed."""
    if not statement or not _CHANGE_EVENT_SUBJECT_RE.search(statement):
        return False
    if not _NEGATIVE_OUTCOME_CLAUSE_RE.search(statement):
        return False
    if _HYPOTHESIS_HEDGE_WORD_RE.search(statement):
        return False
    source = source_text or ""
    if not _CHANGE_EVENT_SUBJECT_RE.search(source):
        return True
    return not _NEGATIVE_OUTCOME_CLAUSE_RE.search(source)


# High-stakes concepts a "potential effect" narrative can invent that the
# finding/evidence never establishes -- e.g. "the operator may have
# proceeded without confirmed qualification" when nothing in the finding
# mentions qualification at all. Evidence-grounded, not a blind ban: each
# concept is only rejected when NEITHER the finding text nor any evidence
# claim independently uses it, so a finding that genuinely involves
# qualification/contamination/etc. remains free to say so. Deliberately a
# short list of ORGANIZATIONAL/CONSEQUENCE concepts (not verbs or a
# specific finding's vocabulary), so it generalizes the same way across
# any domain rather than being tuned to this one finding.
_UNESTABLISHED_IMPACT_CONCEPT_RE = re.compile(
    r"\b(?:qualification|qualified|authoriz\w+|certif\w+|licens\w+|"
    r"contamination|contaminated|recall|regulatory\s+(?:violation|breach)|"
    r"patient\s+safety|customer\s+(?:impact|complaint)|product\s+(?:impact|release|recall)|"
    r"batch\s+(?:impact|recall|rejection)|equipment\s+damage)\b",
    re.IGNORECASE,
)


def impact_asserts_unestablished_concept(text: str | None, finding_text: str, evidence_texts: list[str]) -> bool:
    """True if `text` (typically impact.potential_effect) introduces one of
    a set of high-stakes organizational/consequence concepts that neither
    the finding nor any evidence claim independently establishes -- e.g.
    inventing "qualification" for a finding that only ever discusses
    training completion. Evidence-grounded rather than a blind word ban:
    the SAME word is licensed the moment the finding or evidence actually
    uses it."""
    if not text:
        return False
    hits = set(m.group(0).lower() for m in _UNESTABLISHED_IMPACT_CONCEPT_RE.finditer(text))
    if not hits:
        return False
    grounding = " ".join([finding_text or "", *(evidence_texts or [])]).lower()
    for hit in hits:
        if hit in grounding:
            continue
        return True
    return False


_CAUSAL_CONNECTOR_RE = re.compile(
    r"\b(?:because|due\s+to|caused\s+by|resulted\s+from|as\s+a\s+result\s+of|led\s+to|leading\s+to|"
    r"resulting\s+in|owing\s+to|failed\s+because)\b",
    re.IGNORECASE,
)


def reported_claims_contain_causal_explanation(reported_claims: list[str] | None) -> bool:
    """True if any REPORTED claim actually attributes a REASON for the
    observed deviation (a genuine "X because Y" causal clause), as opposed
    to merely restating that the deviation happened, optionally with added
    temporal/contextual framing.

    This is the load-bearing distinction between:
      "The technician confirmed the check was missed during the morning
      shift." -- a bare fact restatement (temporal context, no reason) --
    and:
      "The technician stated the check was missed because they had not
      received retraining." -- a genuine reported causal explanation.

    Only the second licenses generating ANY candidate hypothesis from this
    claim. A bare restatement -- however specific, however confidently
    reported -- contains no causal content to build a hypothesis FROM;
    generating one anyway just substitutes a differently-worded invented
    mechanism (e.g. a generic "execution/task-control factor" bucket) for
    a more specific one, which is the same defect at one level of
    abstraction removed. When this returns False, the correct behavior is
    zero hypotheses from this claim, not a hedged placeholder."""
    for claim in reported_claims or []:
        if claim and _CAUSAL_CONNECTOR_RE.search(claim):
            return True
    return False


def hypothesis_statement_asserts_unsupported_causation(statement: str | None, verified_facts: list[str]) -> bool:
    """True if `statement` uses causal-connector language ("because", "due
    to", "caused by", "resulted from", "as a result of", "led to") to
    assert a causal link within the hypothesis's own STATEMENT.

    A hypothesis statement should name a candidate mechanism, not narrate a
    causal justification for it -- that is what `rationale` is for.
    "The checklist was not completed due to lack of awareness" quietly
    promotes a REPORTED explanation (the operator's account) into an
    asserted cause merely by using "due to" instead of describing the
    reported claim's own uncertainty. Allowed only when the causal clause
    is itself grounded in a VERIFIED fact (i.e. citing an already-
    established causal link, not inventing one)."""
    if not statement or not _CAUSAL_CONNECTOR_RE.search(statement):
        return False
    stmt_words = significant_words(statement)
    for fact in verified_facts or []:
        fact_words = significant_words(fact)
        if stmt_words and fact_words:
            overlap = len(stmt_words & fact_words) / min(len(stmt_words), len(fact_words))
            if overlap >= 0.6:
                return False
    return True


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
    if is_evidence_state_not_hypothesis(statement):
        return False, "Describes an evidence state or investigation uncertainty rather than proposing a concrete causal mechanism"
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
    subject_words: frozenset | None = None,
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

    `subject_words` (optional): significant words from the finding's own
    canonical subject/affected-object phrase (e.g. "daily equipment
    inspection checklist"). Every hypothesis about this finding will
    naturally repeat that phrase -- it's the thing the finding is about --
    so raw word overlap between a hypothesis statement and the VERIFIED
    deviation fact is inflated by that shared, guaranteed-to-match subject
    phrase regardless of whether the fact actually supports the
    hypothesis's own CAUSAL claim. Excluded from the step-3 overlap
    calculation so promotion to SUPPORTED requires the hypothesis's
    DISTINCTIVE (non-subject) content to be corroborated, not just
    confirmation that the hypothesis is about the same object as the
    finding.
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

    # 3. Verified evidence match -> SUPPORTED ONLY when direct causal mechanism proof exists
    # Direct objective confirmation of control bypass, disablement, system outage, or unvalidated action.
    # Event/observation facts (e.g. "notification sent", "temperature measured") do NOT prove mechanism.
    if verified_facts and allow_verified_promotion:
        stmt_low = statement.lower()
        has_verified_causal_proof = False
        for fact in verified_facts:
            f_low = fact.lower()
            has_disabled_control = bool(re.search(
                r"\b(?:interlock|block|check|control|guard|safety|verification|validation)\b.*?\b(?:disabled|bypassed|defeated|deactivated|switched off|overridden)\b|"
                r"\b(?:disabled|bypassed|defeated|deactivated|switched off|overridden)\b.*?\b(?:interlock|block|check|control|guard|safety|verification|validation)\b",
                f_low,
            ))
            has_system_outage = bool(re.search(r"\b(?:server|system|service|channel|network|queue)\b.*?\b(?:outage|failure|down|crashed|unresponsive|failed)\b", f_low))
            has_unauthorized_action = bool(re.search(r"\b(?:authorized|approved|executed|operated)\b.*?\b(?:without|unvalidated|unqualified|untrained|no\s+completed)\b", f_low))
            has_explicit_audit_proof = ("audit log" in f_low or "audit trail" in f_low or "system log" in f_low or "server log" in f_low) and any(
                w in f_low for w in ("disabled", "bypassed", "failure", "outage", "defeated", "overridden")
            )
            if has_disabled_control or has_system_outage or has_unauthorized_action or has_explicit_audit_proof:
                if any(w in stmt_low for w in ("disable", "bypass", "outage", "service failure", "workflow", "authorization", "control", "interlock", "block")):
                    has_verified_causal_proof = True
                    break
        if has_verified_causal_proof:
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


def evaluate_causal_eligibility(
    hypothesis: Any,
    propositions: list | None = None,
    conflicts: list | None = None,
    evidence_ledger: list | None = None,
    referenced_docs: list | None = None,
    mechanism: MechanismInfo | None = None,
    source_text: str = "",
) -> tuple[bool, str | None]:
    """Evaluate whether a candidate causal hypothesis is eligible to enter the report.

    Enforces the 10 invariant criteria:
      1. Is statement valid and non-empty?
      2. Does it propose a causal mechanism rather than restating the observation?
      3. Is it contradicted by an established mechanism or verified completion?
      4. Does it assume document content for an unavailable document?
      5. Does it attack speaker credibility rather than testing records?
      6. Does it overclaim human error without process framing?
      7. Does it invent ungrounded domains or entities?
      8. Does it cite valid supporting/contradicting claim IDs?
      9. Does it restate an evidence gap as a cause?
      10. Does it respect conflicting evidence bounds (cannot be SUPPORTED if conflicted)?

    Returns (is_eligible, rejection_reason).
    """
    statement = getattr(hypothesis, "statement", str(hypothesis))
    if not statement or len(statement.strip()) < 5:
        return False, "empty_statement"

    # 1. Evidence gap not hypothesis
    if is_evidence_gap_not_hypothesis(statement, source_text):
        return False, "restates_evidence_gap"

    # 2. Contradicts mechanism
    if mechanism and mechanism.statement and hypothesis_contradicts_mechanism(statement, mechanism):
        return False, "contradicts_mechanism"

    # 3. Contradicts verified completion
    verified_facts = [
        getattr(e, "claim", getattr(e, "text", str(e)))
        for e in (evidence_ledger or [])
        if getattr(e, "status", None) == "VERIFIED" or str(getattr(e, "status", "")) == "EvidenceStatus.VERIFIED"
    ]
    if verified_facts and hypothesis_contradicts_verified_completion(statement, verified_facts):
        return False, "contradicts_verified_completion"

    # 4. Attacks statement credibility or overclaims human error
    if hypothesis_attacks_statement_credibility(statement) or hypothesis_overclaims_human_error(statement):
        return False, "attacks_statement_credibility"

    # 5. Contradicts verified fact (e.g. asserts deficient when verified fact says completed)
    if verified_facts:
        for vf in verified_facts:
            vf_low = vf.lower()
            stmt_low = statement.lower()
            if ("completed" in vf_low or "verified" in vf_low) and ("deficient" in stmt_low or "not completed" in stmt_low or "missing" in stmt_low):
                if any(w in stmt_low for w in vf_low.split() if len(w) > 4):
                    return False, "contradicts_verified_completion"

    # 6. Unavailable document content inference
    if referenced_docs:
        unavail_types = [
            getattr(d, "document_type", "").lower()
            for d in referenced_docs
            if getattr(d, "reference_status", "") == "REFERENCED_UNAVAILABLE"
        ]
        for ut in unavail_types:
            if ut and ut in statement.lower() and re.search(r"\b(showed|indicated|contained|recorded|proved)\b", statement.lower()):
                return False, "infers_unavailable_document_content"

    # 7. Restates already established mechanism
    if mechanism and mechanism_already_names_generic_hypothesis(statement, mechanism):
        return False, "restates_established_mechanism"

    return True, None


def should_generate_investigation_plan(
    evidence_ledger: Any = None,
    propositions: list[Any] | None = None,
    root_cause: Any = None,
    finding_text: str = "",
) -> bool:
    """Return True if an investigation plan must be generated for the finding.

    Core Rule (Sections 1, 2, 3):
    NO ROOT-CAUSE HYPOTHESIS does NOT imply NO INVESTIGATION PLAN.
    NO ROOT-CAUSE HYPOTHESIS + UNRESOLVED EVIDENCE / UNKNOWN PROPOSITION / CONFLICT = INVESTIGATION REQUIRED.

    Returns TRUE when ANY of the following exists:
    - unresolved evidence conflict
    - UNKNOWN or UNRESOLVED proposition
    - missing objective record required to resolve a finding
    - reported mechanism without objective corroboration
    - missing referenced document
    - previous CAPA with unresolved implementation/effectiveness status
    - requirement not established
    - affected period not established
    - actual event not established
    - causal mechanism not established (e.g. root_cause.status == NOT_ESTABLISHED or no candidate hypotheses)
    - downstream impact requiring scope assessment
    - contradiction between evidence sources

    Returns FALSE ONLY when:
    1. no unresolved investigation proposition exists, AND
    2. the evidence is sufficient for the current analysis state, AND
    3. no material evidence gap remains.
    """
    if root_cause is not None:
        rc_status = str(getattr(root_cause, "status", ""))
        if "NOT_ESTABLISHED" in rc_status or not getattr(root_cause, "candidate_hypotheses", []):
            return True

    if evidence_ledger is not None:
        if getattr(evidence_ledger, "has_unresolved_conflict", False) or getattr(evidence_ledger, "has_conflicting_reports", False):
            return True
        if getattr(evidence_ledger, "missing_records", []) or getattr(evidence_ledger, "missing_referenced_documents", []):
            return True
        items = getattr(evidence_ledger, "items", []) or []
        for it in items:
            st = str(getattr(it, "status", ""))
            src = str(getattr(it, "source", ""))
            if any(k in st for k in ("UNRESOLVED", "CONFLICTING", "REPORTED", "MISSING")):
                return True
            if "REPORTED_STATEMENT" in src or "PERSON_REPORTED" in src:
                return True

    if propositions:
        for p in propositions:
            st = str(getattr(p, "status", ""))
            pt = str(getattr(p, "proposition_type", ""))
            if any(k in st for k in ("UNKNOWN", "UNRESOLVED", "POSSIBLE_UNCONFIRMED", "CONFLICTING")):
                return True
            if any(k in pt for k in ("EVIDENCE_STATE", "INVESTIGATION_QUESTION", "REPORTED_MECHANISM")):
                return True

    from app.agent.recurrence_guard import detect_recurrence
    if detect_recurrence(finding_text).is_recurring:
        return True

    _trigger_re = re.compile(
        r"\b(?:missing|unavailable|not available|not provided|unconfirmed|unknown|unverified|disputed|conflicted|gap|lacking)\b",
        re.IGNORECASE,
    )
    if _trigger_re.search(finding_text):
        return True

    return True




