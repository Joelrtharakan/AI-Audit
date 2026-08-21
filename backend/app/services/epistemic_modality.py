"""Generalized epistemic-stance and grammatical-modality classification.

Two ORTHOGONAL axes are computed here, both structurally (clause shape),
never by enumerating example sentences:

1. EPISTEMIC STANCE (Defect 1).  A proposition embedded under a
   cognitive/stance predicate ("<entity> believes/suspects/assumes (that)
   <proposition>") is the entity's MENTAL STATE about the world, not an
   observed fact about the world.  The classification decision keys off the
   *syntactic role* of the governing verb -- a finite verb whose subject is
   an animate/organizational entity and whose object is a finite
   ``that``-complement clause -- rather than on membership in a verb list.
   A seed lexicon exists only to (a) sub-type the stance
   (BELIEF/DOUBT/SUSPICION/ASSUMPTION/OPINION) and (b) license the harder
   no-``that`` case; a genuinely novel stance verb ("the auditor intuited
   that ...", "the reviewer harbours the view that ...") is still caught by
   the structural rule, because the rule is:

       animate subject + finite verb + "that" + finite clause
       AND the verb is NOT a verb-of-record / verb-of-report / verb of
       direct perception (confirmed, recorded, logged, shows, demonstrates,
       establishes, proves, verified, observed, stated, reported, ...)

   i.e. stance is the *residual* category of complement-taking verbs after
   the evidence-producing ones are excluded.  That is what makes it
   open-ended: new stance verbs need no code change, whereas a new
   *evidence* verb (which would be the unsafe direction) does.

2. GRAMMATICAL MODALITY / MOOD (Defect 2).  Whether a proposition asserts
   something that ACTUALLY happened, or something conditional/counterfactual
   ("if the permit HAD BEEN issued, the reading WOULD HAVE BEEN logged").
   This is detected from the AUXILIARY CLUSTER -- ``had`` + past participle
   in a protasis, ``would/could/might/should`` + ``have`` + past participle
   in an apodosis, plus the overt conditional subordinators -- so it
   generalizes across every lexical verb and every domain.

Modality is deliberately a SEPARATE axis from EvidenceStatus: a
counterfactual can itself be a corroborated fact about what was *said*
while remaining non-actual in *content*.  Callers therefore keep
``status`` for evidentiary weight and ``modality`` for mood.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Axis 1: epistemic stance
# ---------------------------------------------------------------------------

# Verbs that PRODUCE evidence rather than express a mental stance. These are
# the *closed* set: verbs of record, of direct perception, and of report.
# Anything complement-taking that is NOT here is treated as a stance verb.
# Keeping the evidence side closed (and the stance side open) is the
# fail-safe direction: an unrecognized verb degrades a claim to BELIEF, never
# promotes it to VERIFIED.
_EVIDENCE_PREDICATES = {
    # verbs of record / objective demonstration
    "record", "records", "recorded", "log", "logs", "logged", "document",
    "documents", "documented", "show", "shows", "showed", "shown",
    "demonstrate", "demonstrates", "demonstrated", "establish", "establishes",
    "established", "prove", "proves", "proved", "proven", "confirm",
    "confirms", "confirmed", "verify", "verifies", "verified", "evidence",
    "evidences", "evidenced", "reveal", "reveals", "revealed",
    # verbs of direct perception / audit act
    "observe", "observes", "observed", "find", "finds", "found", "identify",
    "identifies", "identified", "note", "notes", "noted", "determine",
    "determines", "determined", "detect", "detects", "detected", "measure",
    "measures", "measured", "witness", "witnesses", "witnessed",
    # verbs of report (already handled as REPORTED attribution upstream)
    "state", "states", "stated", "report", "reports", "reported", "say",
    "says", "said", "claim", "claims", "claimed", "explain", "explains",
    "explained", "advise", "advises", "advised", "acknowledge",
    "acknowledges", "acknowledged", "mention", "mentions", "mentioned",
    "indicate", "indicates", "indicated", "declare", "declares", "declared",
    "certify", "certifies", "certified", "testify", "testifies", "testified",
    # existential / copular fillers that can precede "that"
    "is", "was", "are", "were", "be", "been", "means", "meant", "requires",
    "required", "ensures", "ensured", "provided", "specifies", "specified",
    # passive-information verbs: "the team was informed that ..." is a report
    # of a communication event, not the team's own epistemic stance.
    "inform", "informs", "informed", "notify", "notifies", "notified", "tell",
    "tells", "told", "remind", "reminds", "reminded", "warn", "warns",
    "warned", "alert", "alerts", "alerted", "instruct", "instructs",
    "instructed", "aware", "unaware",
}

# Auxiliaries/copulas: if one immediately precedes the candidate verb, the
# clause is passive or progressive ("was informed that", "is reporting that"),
# which is a communication event, not a first-person mental state.
_AUXILIARIES = {
    "is", "was", "are", "were", "be", "been", "being", "am", "has", "have",
    "had", "having", "will", "would", "shall", "should", "may", "might",
    "must", "can", "could", "do", "does", "did", "not", "never",
}

# Seed lexicon: used ONLY to sub-type the stance and to license the weaker
# no-"that" construction. Not the gate for the "that"-clause rule.
_STANCE_SUBTYPE_SEEDS: list[tuple[str, re.Pattern[str]]] = [
    ("DOUBT", re.compile(r"^(?:doubt|doubts|doubted|question|questions|questioned|dispute|disputes|disputed|disbelieve\w*)$", re.I)),
    ("SUSPICION", re.compile(r"^(?:suspect|suspects|suspected|surmise\w*|conjecture\w*|speculate\w*|hypothesi[sz]e\w*|theori[sz]e\w*|posit\w*|postulate\w*)$", re.I)),
    ("ASSUMPTION", re.compile(r"^(?:assume|assumes|assumed|presume|presumes|presumed|presuppose\w*|take|takes|took|expect|expects|expected|anticipate\w*|infer|infers|inferred)$", re.I)),
    ("OPINION", re.compile(r"^(?:opine\w*|feel|feels|felt|consider|considers|considered|regard|regards|regarded|deem|deems|deemed|view|views|viewed|judge|judges|judged|maintain\w*|contend\w*|argue\w*)$", re.I)),
    ("BELIEF", re.compile(r"^(?:believe|believes|believed|think|thinks|thought|perceive\w*|understand|understands|understood|suppose\w*|reckon\w*|guess\w*|imagine\w*|trust|trusts|trusted|hold|holds|held|fear|fears|feared|hope|hopes|hoped|estimate\w*|conclude\w*|intuit\w*|sense|senses|sensed)$", re.I)),
]

# An animate / organizational entity capable of holding a belief. Deliberately
# a *shape* test (role noun, org noun, proper noun, personal pronoun), not a
# roster of job titles.
_ANIMATE_HEAD_RE = re.compile(
    r"\b(?:team|group|department|dept|committee|board|unit|function|management|"
    r"personnel|staff|crew|panel|management|leadership|division|section|office|"
    r"organi[sz]ation|company|client|customer|vendor|contractor|regulator|"
    r"auditor|inspector|reviewer|assessor|investigator|analyst|engineer|"
    r"technician|operator|supervisor|manager|director|officer|coordinator|"
    r"specialist|scientist|nurse|physician|doctor|pharmacist|clerk|"
    r"administrator|owner|lead|head|chief|representative|member|employee|"
    r"worker|person|people|witness|respondent|interviewee|candidate)s?\b",
    re.IGNORECASE,
)
# Agentive morphology: English derives person-denoting nouns with a small,
# closed set of suffixes (-er/-or/-ist/-ant/-ent/-ian/-ee/-man/-woman). Using
# the SUFFIX rather than a job-title roster is what lets a never-listed role
# ("the maintenance planner", "the radiographer", "the invigilator") count as
# a belief-capable entity without a code change.
_AGENTIVE_SUFFIX_RE = re.compile(
    r"\b[a-z]{3,}(?:er|or|ist|ant|ent|ian|ee|man|men|woman|women|ard|eer|smith)s?$",
    re.IGNORECASE,
)
_PERSONAL_PRONOUN_RE = re.compile(r"^(?:he|she|they|we|i|you|it)$", re.IGNORECASE)
_PROPER_NOUN_RE = re.compile(r"^(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|de|van|of|the))*$")


def _is_animate_entity(subject: str | None) -> bool:
    """Shape test for a belief-capable subject: a role/organizational head
    noun, a personal pronoun, or a proper noun. No job-title roster."""
    if not subject:
        return False
    s = subject.strip().strip(",")
    if not s or len(s.split()) > 8:
        return False
    if _PERSONAL_PRONOUN_RE.match(s):
        return True
    if _ANIMATE_HEAD_RE.search(s):
        return True
    if _AGENTIVE_SUFFIX_RE.search(s):
        return True
    stripped = re.sub(r"^(?:the|a|an|our|their|its|his|her)\s+", "", s, flags=re.I)
    return bool(_PROPER_NOUN_RE.match(stripped.strip()))


def _stance_subtype(verb: str) -> str | None:
    v = verb.strip().lower()
    for label, pattern in _STANCE_SUBTYPE_SEEDS:
        if pattern.match(v):
            return label
    return None


_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'’-]*$")


def _split_complement_clause(text: str) -> tuple[str, str, str, bool] | None:
    """Locate [subject] [finite verb] ("that") [complement] by SCANNING the
    token sequence, not by a single regex — a regex with a lazy subject group
    latches onto the first viable split and never backtracks to the real verb
    ("The security team believes ..." would parse as subject="The",
    verb="security").

    Returns (subject, verb, complement, has_that) or None.
    """
    tokens = text.split()
    if len(tokens) < 3:
        return None

    def _clean(tok: str) -> str:
        return tok.strip(",;:.\"'()").lower()

    # Route 1: an explicit "that"-complementizer pins the verb immediately
    # before it. This is the structural, open-ended route.
    for i, tok in enumerate(tokens):
        if _clean(tok) == "that" and i >= 2 and i + 1 < len(tokens):
            verb = tokens[i - 1].strip(",;:.")
            if not _WORD_RE.match(verb):
                continue
            prev = _clean(tokens[i - 2]) if i >= 2 else ""
            if prev in _AUXILIARIES:
                # passive/progressive => communication event, not stance,
                # unless the verb is itself a recognized stance predicate.
                if not _stance_subtype(verb):
                    return None
            return " ".join(tokens[: i - 1]), verb, " ".join(tokens[i + 1:]), True

    # Route 2: no complementizer. Too ambiguous structurally, so accept only
    # on a seed stance predicate (see classify_epistemic_stance docstring).
    for i in range(1, len(tokens) - 1):
        verb = tokens[i].strip(",;:.")
        if not _WORD_RE.match(verb) or not _stance_subtype(verb):
            continue
        if _clean(tokens[i - 1]) in _AUXILIARIES:
            continue
        return " ".join(tokens[:i]), verb, " ".join(tokens[i + 1:]), False
    return None

# Agentless / impersonal stance framings: "it is believed that ...",
# "it was assumed that ...", "in the opinion of X, ...", "X is of the view
# that ...". The stance holder may be unstated -- still not an observation.
_IMPERSONAL_STANCE_RE = re.compile(
    r"^(?:it\s+(?:is|was)\s+(?P<verb1>[a-z]+ed)\s+that\s+(?P<claim1>.+)"
    r"|in\s+the\s+(?:opinion|view|judg?ement|estimation|belief)\s+of\s+(?P<holder2>[^,]{2,60}),?\s+(?P<claim2>.+)"
    r"|(?P<holder3>[A-Za-z][\w\s'&-]{0,60}?)\s+(?:is|are|was|were)\s+of\s+the\s+(?:opinion|view|belief|impression|understanding)\s+that\s+(?P<claim3>.+)"
    r"|(?:according\s+to\s+)?(?P<holder4>[A-Za-z][\w\s'&-]{0,60}?)['’]s\s+(?:opinion|view|belief|assumption|suspicion|impression)\s+(?:is|was)\s+that\s+(?P<claim4>.+))$",
    re.IGNORECASE,
)


@dataclass
class StanceStatement:
    """An epistemic-stance proposition: what an entity BELIEVES, not what is."""
    holder: str | None
    proposition: str
    stance: str          # BELIEF | DOUBT | SUSPICION | ASSUMPTION | OPINION
    verb: str
    via: str             # "lexicon" | "structural" | "impersonal"

    @property
    def rendered(self) -> str:
        if self.holder:
            return f"{self.holder} {self.verb} {self.proposition}"
        return self.proposition


def classify_epistemic_stance(sentence: str | None) -> StanceStatement | None:
    """Return a StanceStatement if `sentence` embeds a proposition under a
    cognitive/epistemic-stance predicate, else None.

    Two acceptance routes:
      * STRUCTURAL (open-ended): animate subject + finite verb + explicit
        ``that``-complement, where the verb is not an evidence/report/
        perception predicate.  Novel stance verbs are caught here.
      * LEXICON (narrower): same shape WITHOUT ``that``, which is too
        ambiguous on its own ("the pump failed the test"), so it is only
        accepted when the verb matches a seed stance pattern.
    """
    if not sentence or not sentence.strip():
        return None
    text = sentence.strip().rstrip(".")

    m_imp = _IMPERSONAL_STANCE_RE.match(text)
    if m_imp:
        g = m_imp.groupdict()
        verb = (g.get("verb1") or "believed").lower()
        holder = next((g[k] for k in ("holder2", "holder3", "holder4") if g.get(k)), None)
        claim = next((g[k] for k in ("claim1", "claim2", "claim3", "claim4") if g.get(k)), None)
        if claim:
            return StanceStatement(
                holder=holder.strip() if holder else None,
                proposition=claim.strip(),
                stance=_stance_subtype(verb) or "BELIEF",
                verb=verb,
                via="impersonal",
            )

    parsed = _split_complement_clause(text)
    if not parsed:
        return None
    subject, verb, claim, has_that = parsed
    # A finding usually opens with an adverbial ("During the November review
    # of Line 7, the maintenance planner apprehended that ..."). The stance
    # HOLDER is the noun phrase in the final comma-delimited segment; the
    # preceding adjunct is setting, not subject.
    subject = subject.split(",")[-1].strip() if "," in subject else subject.strip()
    claim = claim.strip()
    if not claim or len(claim.split()) < 2:
        return None
    if verb.lower() in _EVIDENCE_PREDICATES:
        return None
    if not _is_animate_entity(subject):
        return None

    subtype = _stance_subtype(verb)
    if has_that:
        # Structural route: a complement-taking verb governed by an animate
        # subject, which is not an evidence/report/perception predicate, IS a
        # stance predicate -- whether or not we have ever seen this verb.
        return StanceStatement(
            holder=subject,
            proposition=claim,
            stance=subtype or "BELIEF",
            verb=verb,
            via="lexicon" if subtype else "structural",
        )
    if subtype:
        # No "that": accept only on a recognized stance verb, since bare
        # SVO is otherwise indistinguishable from an ordinary transitive.
        return StanceStatement(
            holder=subject, proposition=claim, stance=subtype, verb=verb, via="lexicon"
        )
    return None


# ---------------------------------------------------------------------------
# Axis 2: grammatical modality / mood
# ---------------------------------------------------------------------------

ACTUAL = "ACTUAL"
CONDITIONAL = "CONDITIONAL"
COUNTERFACTUAL = "COUNTERFACTUAL"

# would/could/might/should (+ not/never, + an optional adverb like
# "likely"/"probably"/"certainly") + have + past participle. The auxiliary
# CLUSTER is the signal -- the lexical verb (and any adverb inserted before
# "have") is irrelevant, so this fires identically for "would have been
# logged", "could have escalated", "would likely have passed", "might
# probably not have crystallised".
_MODAL_PERFECT_RE = re.compile(
    r"\b(?:would|could|might|should|may|must)\s+(?:not\s+|never\s+|n['’]t\s+)?"
    r"(?:\w+ly\s+)?(?:not\s+|never\s+)?"
    r"have\s+(?:been\s+)?[a-z]+(?:ed|en|n|t|ne|un|ung|ought|elt|ilt|ept)\b",
    re.IGNORECASE,
)
# "if X had been ..." / inverted "Had X been ..." -- past-perfect protasis.
# The "been + participle" cluster may be followed by an adverbial/temporal
# phrase before the clause boundary ("Had the wrench been calibrated THAT
# WEEK, ..."), so the match only anchors on the auxiliary cluster itself,
# not on the participle being the last word before the comma.
_PAST_PERFECT_PROTASIS_RE = re.compile(
    r"\bif\s+[^,.;]{1,80}?\bhad\s+(?:not\s+)?been\s+[a-z]+(?:ed|en)\b"
    r"|^\s*had\s+[^,.;]{1,80}?\bbeen\s+(?:not\s+)?[a-z]+(?:ed|en)\b",
    re.IGNORECASE,
)
# Overt conditional subordinators (non-past-perfect): open conditions and
# subjunctive framings.
_CONDITIONAL_SUBORDINATOR_RE = re.compile(
    r"^\s*if\b|[,;]\s*if\b|\bunless\b|\bprovided\s+that\b|\bproviding\s+that\b|"
    r"\bassuming\s+(?:that\b|the\b)|\bin\s+the\s+event\s+(?:that|of)\b|"
    r"\bwere\s+[a-z][\w\s-]{0,40}?\s+to\s+[a-z]+\b|\bshould\s+[a-z][\w\s-]{0,40}?\s+(?:occur|arise|fail|happen|be\s+)|"
    r"\bhypothetically\b|\bin\s+theory\b|\bsuppose\s+that\b|\bwhat\s+if\b|"
    r"\bhad\s+it\s+not\s+been\s+for\b|\bother\s+things\s+being\s+equal\b",
    re.IGNORECASE,
)


@dataclass
class ModalityInfo:
    modality: str             # ACTUAL | CONDITIONAL | COUNTERFACTUAL
    marker: str | None = None

    @property
    def is_actual(self) -> bool:
        return self.modality == ACTUAL


def classify_modality(text: str | None) -> ModalityInfo:
    """Classify a proposition's grammatical mood from its auxiliary structure.

    COUNTERFACTUAL: a modal-perfect cluster (would/could/might/should +
      have + past participle) and/or a past-perfect protasis ("if X had
      been ...", "Had X been ..., ...").  This is the canonical English
      counterfactual conditional and is recognized purely from the
      auxiliary chain.
    CONDITIONAL: an overt conditional subordinator with a non-past-perfect
      protasis ("if ...", "unless ...", "provided that ...", "were X to
      occur", "should X occur", "assuming that ...").
    ACTUAL: everything else.

    Deliberately does NOT fire on a bare non-perfect modal ("could not be
    located", "should be reviewed"): "could not be located" is a factual
    statement about an *actual* failed retrieval, and "should be reviewed"
    is a deontic recommendation -- neither is a counterfactual.  Restricting
    to the perfect-infinitive cluster and the overt subordinators keeps the
    detector precise without enumerating any sentence.
    """
    if not text or not text.strip():
        return ModalityInfo(ACTUAL)
    t = text.strip()

    m_perf = _MODAL_PERFECT_RE.search(t)
    m_prot = _PAST_PERFECT_PROTASIS_RE.search(t)
    if m_perf or m_prot:
        return ModalityInfo(COUNTERFACTUAL, (m_prot or m_perf).group(0).strip())

    m_cond = _CONDITIONAL_SUBORDINATOR_RE.search(t)
    if m_cond:
        return ModalityInfo(CONDITIONAL, m_cond.group(0).strip())

    return ModalityInfo(ACTUAL)


def is_non_actual(text: str | None) -> bool:
    """True when the proposition is conditional/counterfactual in mood and
    therefore must never be recorded as a VERIFIED observed event."""
    return not classify_modality(text).is_actual
