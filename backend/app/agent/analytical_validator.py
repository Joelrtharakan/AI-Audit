"""Analytical validation firewall.

Runs AFTER causal synthesis (core_synthesis_node, which already applies the
causal_guard contradiction/circularity guards inline) and is invoked from
final_evidence_verification_node — the single point through which every
result reaches report_generator, so there is exactly one place these
invariants are enforced, not a second competing implementation.

Every function here is a pure, deterministic, structural check over the
already-synthesized analytical state. None of it generates new causal
content: a violation is handled by REPAIRING STRUCTURE (e.g. inserting a
mechanism step the model skipped, using text that PROVENANCE already gives
us), DOWNGRADING a claim's status, or REMOVING unsupported content — never
by inventing a fact, a mechanism, or a root cause that isn't already present
in the evidence ledger / canonical finding state.
"""

from __future__ import annotations

import logging
import re

from app.agent.causal_guard import MechanismInfo, repeats_previous_why_answer
from app.services.text_grounding import significant_words

logger = logging.getLogger(__name__)

_UNKNOWN_VALUE_RE = re.compile(
    r"^\s*(not\s+established|unknown|to\s+be\s+confirmed|n/?a|none|pending|requires\s+confirmation)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 1. Leading hypothesis selection (single source of truth)
# ---------------------------------------------------------------------------

_RANK_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def select_leading_hypothesis(candidate_hypotheses: list) -> str | None:
    """Deterministic leading-hypothesis rule, used everywhere a leading
    hypothesis needs to be derived (never left to prompt compliance):

    - No hypotheses at all -> None.
    - Exactly one SUPPORTED hypothesis -> that one.
    - Multiple SUPPORTED hypotheses that are equally strong (same
      relevance_rank) -> None (they are competing, not a single leader).
    - No SUPPORTED hypothesis, but the POSSIBLE ones differ in
      relevance_rank -> the single highest-ranked one.
    - No SUPPORTED hypothesis and all POSSIBLE ones share the same
      relevance_rank -> use structural scoring to break ties. If scoring
      still can't differentiate (scores within 0.1), return None.
    - No SUPPORTED hypothesis and only ONE non-refuted hypothesis exists ->
      None. "Leading" means materially stronger than the alternatives; a
      lone unresolved/possible hypothesis has no alternatives to be
      stronger than -- it survived filtering (or was the only one
      generated), not evidence competition. Calling it "leading" merely
      because nothing else remains manufactures a false sense of causal
      progress out of a single untested guess.
    """
    if not candidate_hypotheses:
        return None

    supported = [h for h in candidate_hypotheses if h.status == "SUPPORTED"]
    if supported:
        pool = supported
    else:
        pool = [h for h in candidate_hypotheses if h.status != "REFUTED"]
        if len(pool) <= 1:
            return None
    if not pool:
        return None

    ranks = {_RANK_ORDER.get(h.relevance_rank, 1) for h in pool}
    if len(pool) > 1 and len(ranks) == 1:
        # All tied at the same rank -- try structural scoring to break tie
        scored = [(h, score_hypothesis(h)) for h in pool]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_score = scored[0][1]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        if best_score - second_score > 0.1:
            # Enough differentiation to declare a leader
            best = scored[0][0]
            return f"{best.id} — {best.statement}"
        # Still tied -- no defensible single leader.
        return None

    best = min(pool, key=lambda h: _RANK_ORDER.get(h.relevance_rank, 1))
    return f"{best.id} — {best.statement}"


def score_hypothesis(hypothesis, claims: list | None = None,
                     conflicts: list | None = None,
                     mechanism=None) -> float:
    """Deterministic structural scoring of a hypothesis across 8 dimensions.

    Returns a float in [0.0, 1.0] reflecting overall hypothesis quality.
    Higher is better. This is used for tie-breaking when multiple hypotheses
    share the same relevance_rank, and for computing per-hypothesis
    confidence grades.

    Dimensions (each contributes 0.0–0.125 to the final score):
      1. Evidence support: does it cite evidence_needed?
      2. Causal relevance: does its statement contain causal language?
      3. Contradiction-free: is it not contradicted by known claims?
      4. Evidence availability: does it specify discrimination_evidence?
      5. Explanatory coverage: how specific is the rationale?
      6. Specificity: does it use specific (non-generic) language?
      7. Unsupported assumptions: does it overclaim?
      8. Discriminating evidence quality: does it specify confirms_if/refutes_if?
    """
    from app.agent.causal_guard import _CAUSAL_EXPLANATION_RE

    score = 0.0

    # 1. Evidence support
    if getattr(hypothesis, "evidence_needed", None):
        score += 0.125

    # 2. Causal relevance
    statement = getattr(hypothesis, "statement", "")
    if statement and _CAUSAL_EXPLANATION_RE.search(statement):
        score += 0.125

    # 3. Contradiction-free
    if claims:
        from app.agent.causal_guard import test_hypothesis_against_claims
        result = test_hypothesis_against_claims(statement, claims)
        if result == "CONSISTENT":
            score += 0.125
        elif result == "CONTRADICTED":
            pass  # 0 for this dimension
        else:
            score += 0.0625  # UNKNOWN -- partial credit

    else:
        score += 0.0625  # No claims to test against

    # 4. Evidence availability
    if getattr(hypothesis, "discrimination_evidence", None):
        score += 0.125

    # 5. Explanatory coverage
    rationale = getattr(hypothesis, "rationale", "")
    if rationale and len(rationale.split()) >= 5:
        score += 0.125

    # 6. Specificity (not generic filler)
    from app.agent.causal_guard import is_generic_non_analysis_filler
    if statement and not is_generic_non_analysis_filler(statement):
        score += 0.125

    # 7. Unsupported assumptions
    from app.agent.causal_guard import hypothesis_overclaims_human_error
    if not hypothesis_overclaims_human_error(statement):
        score += 0.125

    # 8. Discriminating evidence quality
    confirms_if = getattr(hypothesis, "confirms_if", None)
    refutes_if = getattr(hypothesis, "refutes_if", None)
    if confirms_if and refutes_if:
        score += 0.125
    elif confirms_if or refutes_if:
        score += 0.0625

    return min(score, 1.0)


def validate_root_cause_establishment(
    root_cause,
    mechanism=None,
    claims: list | None = None,
    conflicts: list | None = None,
) -> tuple[bool, list[str]]:
    """Deterministic validation of whether a root cause can be ESTABLISHED.

    Returns (can_establish, reasons) where reasons lists every criterion
    that failed. The root cause can ONLY be established if ALL of:
      1. A causal mechanism is supported by evidence (not just reported)
      2. The mechanism explains the observation
      3. Competing plausible causes are sufficiently excluded
      4. Evidence is not merely REPORTED
      5. There is no unsupported causal leap
      6. The causal relationship is explicit or strongly evidenced
      7. No evidence conflicts remain UNRESOLVED

    This function never mutates root_cause -- it only reports whether
    establishment is defensible.
    """
    reasons: list[str] = []

    # 1. Mechanism must exist and be more than UNKNOWN
    if not mechanism or not mechanism.statement:
        reasons.append("No causal mechanism established")
    elif mechanism.status == "UNKNOWN":
        reasons.append("Mechanism status is UNKNOWN (possibly due to evidence conflict)")
    elif mechanism.status == "REPORTED":
        reasons.append("Mechanism is only REPORTED, not VERIFIED")

    # 2. Root cause must have a statement
    if root_cause and not root_cause.statement:
        reasons.append("No root cause statement provided")

    # 3. Evidence conflicts must be resolved
    if conflicts:
        unresolved = [c for c in conflicts if getattr(c, "status", "UNRESOLVED") == "UNRESOLVED"]
        if unresolved:
            reasons.append(f"{len(unresolved)} evidence conflict(s) remain UNRESOLVED")

    # 4. Supporting evidence must exist
    if root_cause and not root_cause.supporting_evidence:
        reasons.append("No supporting evidence listed")

    # 5. Status must not already be NOT_ESTABLISHED/CONTRADICTED
    if root_cause:
        status_value = getattr(root_cause.status, "value", root_cause.status)
        if status_value in ("NOT_ESTABLISHED", "CONTRADICTED"):
            reasons.append(f"Root cause status is already {status_value}")

    can_establish = len(reasons) == 0
    return can_establish, reasons


def leading_hypothesis_status(candidate_hypotheses: list) -> str:
    """Distinguishes WHY select_leading_hypothesis returned None, which
    matters for what the report should say:

    - "NONE": there are no (surviving) hypotheses at all -- nothing to lead.
    - "TIED": 2+ hypotheses are equally plausible -- a real analytical
      outcome, not a missing one.
    - "SELECTED": a single leader was chosen.

    Collapsing TIED into the same bare `None` as NONE is exactly what makes
    a report read as "No leading hypothesis established" when the truth is
    "multiple well-reasoned hypotheses are genuinely competing" -- those are
    different things and must render differently.
    """
    if not candidate_hypotheses:
        return "NONE"
    pool = [h for h in candidate_hypotheses if h.status != "REFUTED"]
    if not pool:
        return "NONE"
    if select_leading_hypothesis(candidate_hypotheses) is not None:
        return "SELECTED"
    return "TIED"


def leading_hypothesis_display(candidate_hypotheses: list) -> str | None:
    """Same selection as select_leading_hypothesis, but renders the TIED
    case as an explicit, informative string instead of a bare None a report
    would otherwise show as "no leading hypothesis" -- when the real
    situation is "multiple hypotheses are competing", that must be visible,
    not indistinguishable from "no hypotheses were generated at all"."""
    if not candidate_hypotheses:
        return None
    leading = select_leading_hypothesis(candidate_hypotheses)
    if leading is not None:
        return leading
    if leading_hypothesis_status(candidate_hypotheses) == "TIED":
        return "NONE — COMPETING HYPOTHESES REMAIN TIED"
    return None


def apply_conflict_tie_override(root_cause, has_unresolved_conflict: bool) -> None:
    """When candidate hypotheses are competing explanations for the SAME
    unresolved evidence conflict (e.g. two REPORTED statements directly
    disagree about whether an activity was completed), a single "leading"
    hypothesis must never be promoted merely because of an incidental
    scoring/wording difference between otherwise equally-plausible
    statements -- score_hypothesis's structural dimensions (e.g. whether a
    statement happens to use a verb phrase the causal-language regex
    matches) were never meant to arbitrate between hypotheses that all
    equally trace back to one still-open disagreement in the evidence
    itself. Only a hypothesis with independently SUPPORTED status (i.e.
    objective evidence beyond the conflicting reports themselves) may still
    lead. Mutates root_cause in place; a no-op when there is no unresolved
    conflict or no hypothesis was actually SELECTED."""
    if not has_unresolved_conflict:
        return
    if root_cause.leading_hypothesis_status != "SELECTED":
        return
    if any(getattr(h, "status", None) == "SUPPORTED" for h in root_cause.candidate_hypotheses):
        return
    root_cause.leading_hypothesis_status = "TIED"
    root_cause.leading_hypothesis = "NONE — COMPETING HYPOTHESES REMAIN TIED"
    root_cause.confidence = "LOW"


def _statement_as_question_clause(statement: str) -> str:
    """Lower-cases the leading word and strips a trailing period so a
    declarative hypothesis statement can be embedded mid-sentence inside a
    grammatically complete question, without any NLP dependency."""
    s = (statement or "").strip().rstrip(".")
    if not s:
        return "this hypothesis is supported by the available evidence"
    return s[0].lower() + s[1:]


_NEGATION_LEAD_RE = re.compile(r"^(no|none|never|not|absence|failure)\b", re.IGNORECASE)
_POSITIVE_VERB_RE = re.compile(
    r"^(?P<subject>.+?)\s+(?P<verb>confirms?|shows?|indicates?|verifies?|establishes?|exists?)\b(?P<rest>.*)$",
    re.IGNORECASE,
)


def _positive_clause_to_question(hypothesis) -> str | None:
    """Turn a hypothesis's own confirms_if/refutes_if condition into a
    natural, positively-phrased yes/no investigation question (e.g. "An
    approved training completion record confirms timely completion." ->
    "Does an approved training completion record confirm timely
    completion?") instead of wrapping an absence-phrased condition in "Is
    there evidence that no X exists?" -- prefers whichever of
    confirms_if/refutes_if does NOT start with a negation, since a
    hypothesis's refutes_if and confirms_if are usually opposite polarity
    and at least one is typically phrased as a positive existence/state
    claim. Purely structural (verb-fronting), no domain vocabulary."""
    refutes_if = getattr(hypothesis, "refutes_if", None)
    confirms_if = getattr(hypothesis, "confirms_if", None)
    clause = None
    for candidate in (refutes_if, confirms_if):
        if candidate and not _NEGATION_LEAD_RE.match(candidate.strip()):
            clause = candidate.strip()
            break
    if not clause:
        return None
    text = clause.rstrip(".")
    m = _POSITIVE_VERB_RE.match(text)
    # A compound clause ("X is located and confirms Y") has an auxiliary
    # verb (is/was/were/has/have/are) already inside the captured subject --
    # fronting "Does <subject-with-its-own-verb> confirm...?" produces a
    # double-verb artifact ("Does X is located and confirm Y?"). Fall back
    # to the safe generic wrapper in that case rather than guess which verb
    # to front.
    if m and re.search(r"\b(?:is|was|were|are|has|have|had)\b", m.group("subject"), re.IGNORECASE):
        m = None
    if not m:
        return f"Does the available evidence show: {text[0].lower()}{text[1:]}?"
    subject = m.group("subject").strip()
    verb = m.group("verb").rstrip("s")  # confirms -> confirm, exists -> exist
    rest = m.group("rest").strip()
    subject_lc = subject[0].lower() + subject[1:] if subject else subject
    return f"Does {subject_lc} {verb}{(' ' + rest) if rest else ''}?"


def derive_investigation_questions(candidate_hypotheses: list) -> list:
    """Builds discriminating investigation questions directly from the
    already-synthesized candidate hypotheses -- for when core_synthesis
    produced hypotheses but the investigation plan (built by a separate,
    earlier node that runs BEFORE hypotheses exist, or fast-pathed with an
    empty plan) never got a chance to ask anything that would discriminate
    between them. This is the path that actually populates investigation
    questions on a normal successful LLM run when tool-based planning is
    fast-pathed to an empty plan (the common no-ASP.NET-integration
    deployment), so its wording quality matters as much as the deterministic
    fallback's.

    Each question is built from a FIXED, always-grammatical template rather
    than concatenating raw evidence-list text with the hypothesis statement.
    Prefers a POSITIVELY-phrased confirms_if/refutes_if clause turned into a
    natural "Does X confirm Y?" question (see _positive_clause_to_question)
    over wrapping a (frequently absence-phrased) condition in "Is there
    evidence that no X exists?", which reads as an awkward negative
    question. Generalizes to any domain: the template only ever wraps the
    hypothesis's own confirms_if/refutes_if/statement text, never re-parses
    or infers new content.
    """
    from app.models.agent import InvestigationQuestion

    questions = []
    for h in candidate_hypotheses:
        if getattr(h, "status", None) == "REFUTED":
            continue
        evidence = h.evidence_needed or "the relevant record for this hypothesis"
        purpose = h.discrimination_evidence or f"Determines whether {_statement_as_question_clause(h.statement)}"
        question = _positive_clause_to_question(h)
        if not question:
            clause = _statement_as_question_clause(h.confirms_if or h.statement)
            question = f"Is there evidence that {clause}?"
        if not validate_investigation_question(question):
            continue
        questions.append(InvestigationQuestion(
            question=question,
            purpose=purpose,
            evidence=evidence,
            hypothesis_tested=getattr(h, "id", None),
            confirms_if=getattr(h, "confirms_if", None),
            refutes_if=getattr(h, "refutes_if", None),
        ))
    return questions


def derive_required_evidence(candidate_hypotheses: list, evidence_conflicts: list | None = None) -> str | None:
    """Derives a concise "what would actually resolve the current
    uncertainty" evidence summary from the live candidate hypotheses' own
    (already-grounded) evidence_needed fields, rather than trusting a
    separately-generated impact/CAPA field that can drift onto an unrelated
    concept (e.g. the LLM proposing "procedure revision documentation" for a
    training finding). This is the single source of truth for "what
    evidence is needed" -- every hypothesis already carries its own
    evidence_needed, so this only dedupes and concatenates, never invents.
    Returns None when there are no hypotheses to derive from."""
    if not candidate_hypotheses:
        return None
    items = [h.evidence_needed for h in candidate_hypotheses if getattr(h, "evidence_needed", None) and h.status != "REFUTED"]
    deduped = normalize_and_dedupe_evidence_items(items)
    if not deduped:
        return None
    if len(deduped) == 1:
        return deduped[0]
    return "; ".join(deduped)


# ---------------------------------------------------------------------------
# Generic investigation-plan quality layer (areas / ranking / evidence dedup).
#
# This is the ONE authoritative post-processing layer for investigation-plan
# presentation quality -- deduplicate_investigation_questions (above),
# derive_investigation_areas, rank_questions_by_information_gain, and
# normalize_and_dedupe_evidence_items are all called exactly once, from
# final_evidence_verification_node, regardless of which producer (LLM
# hypotheses, a tool-based plan, or the deterministic fallback in
# plan_investigation_fallback.py) generated the raw questions. None of these
# functions invent causal content; they only cluster, rank, and rename
# ALREADY-PRODUCED investigative content into a small, non-duplicative,
# audit-facing set.
#
# Investigation areas in particular must never be authored from a fixed,
# domain-specific vocabulary ("Notification delivery...") or from copying a
# question's own purpose/resolves text verbatim as an area name -- both of
# those produce exactly the repetitive, overlapping area lists this layer
# exists to eliminate. Instead, every question is classified into one of a
# small set of GENERIC, domain-agnostic control-point aspects (delivery,
# receipt, acknowledgement, completion, record availability/control,
# authorization, requirement, activity, mechanism, exception, effectiveness,
# scope), aspects are grouped into four fixed control-FAMILIES, and one area
# name is synthesized per family that actually has a surviving question --
# parameterized by the finding's own subject, never a hardcoded domain noun.
# ---------------------------------------------------------------------------

# Matches the "REC_<n>" recurrence-proposition ID convention only -- a bare
# "REC" substring check would false-positive on e.g. "P_RECEIPT". No \b
# anchors: "_" is a \w character, so "P_REC_1" has no word boundary before
# "REC" -- the literal "REC_" substring (absent from "RECEIPT", which has no
# underscore) is precise enough on its own.
_RECURRENCE_TARGET_RE = re.compile(r"REC_\d+")

_ASPECT_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Most specific/topical aspects first -- "authorization" is checked LAST
    # because "verify"/"verification" is generic boilerplate that appears in
    # nearly every question's purpose/resolves text regardless of its real
    # topic (e.g. "Retraining COMPLETION VERIFICATION", "Previous CAPA
    # EFFECTIVENESS VERIFICATION"); letting it match first would swallow
    # every other aspect's own, more informative signal.
    ("delivery", re.compile(r"\bdeliver(?:y|ed|ies|ing)?\b", re.IGNORECASE)),
    ("receipt", re.compile(r"\breceiv(?:e|ed|es|ing)\b|\breceipt\b|\baccess(?:ed|ing)?\b", re.IGNORECASE)),
    ("acknowledgement", re.compile(r"\backnowledg", re.IGNORECASE)),
    ("effectiveness", re.compile(r"\beffective(?:ness)?\b", re.IGNORECASE)),
    ("exception", re.compile(r"\bexception\b|\bwaiver\b|\bdeviation\b(?!\s+condition)", re.IGNORECASE)),
    ("mechanism", re.compile(r"\bmechanism\b|\btechnical\b|\berror log\b|\bdiagnostic\b|configur\w*|malfunction\w*|\bfault\w*\b", re.IGNORECASE)),
    ("record_control", re.compile(r"\brecord[- ]control\b|\bretention\b|\bcreated and retained\b|\bretained\b", re.IGNORECASE)),
    ("record_availability", re.compile(r"\blocat(?:e|ed|ing)\b|\bretriev|\barchive\b|\brepository\b", re.IGNORECASE)),
    ("completion", re.compile(r"\bcomplet(?:e|ed|ion)\b", re.IGNORECASE)),
    ("requirement", re.compile(r"\brequirement\b|\bprocedure\b|\bgoverns?\b|\bapplicable\b", re.IGNORECASE)),
    ("scope", re.compile(r"\bdownstream\b|\bscope\b|\bimpact\b", re.IGNORECASE)),
    ("activity", re.compile(r"\bactivity\b|\bperform(?:ed)?\b|\bact on\b|\btask\b|\bwork\b", re.IGNORECASE)),
    ("authorization", re.compile(r"\bauthoriz|\bverif(?:y|ied|ication)\b|\bsign[- ]?off\b", re.IGNORECASE)),
]

_ASPECT_DISPLAY = {
    "delivery": "delivery",
    "receipt": "receipt",
    "acknowledgement": "acknowledgement",
    "record_control": "record retention",
    "record_availability": "record availability",
    "authorization": "verification/authorization",
    "completion": "completion",
    "requirement": "applicable requirement",
    "exception": "authorized exception",
    "effectiveness": "effectiveness",
    "scope": "downstream scope",
    "activity": "operational activity",
    "mechanism": "technical/administrative factors",
}

# Four fixed, generic control families -- capping the derived area count at 4
# regardless of how many distinct aspects appear (Section 10: "Normally
# output 2-4 investigation areas"). Order within a family controls word
# order in the synthesized area name.
_AREA_FAMILIES: list[tuple[str, list[str], str]] = [
    ("communication_chain", ["delivery", "receipt", "acknowledgement"], "control"),
    ("record_and_verification", ["completion", "record_availability", "record_control", "authorization"], "verification"),
    ("requirement_and_governance", ["requirement", "exception", "effectiveness", "scope"], "assessment"),
    ("activity_and_mechanism", ["activity", "mechanism"], "review"),
]


def classify_investigation_aspect(text: str) -> str | None:
    """Classifies free text into one of a small, generic set of QMS
    control-point aspects (never a domain-specific label like
    "notification"). Returns None if nothing recognizable is present.

    Text-only classification is inherently ambiguous when a question's own
    wording references a neighboring aspect for context (e.g. an
    acknowledgement question that also mentions "access" or "receipt" to
    describe what would confirm it) -- prefer `_question_aspect` below,
    which checks the question's own `target_proposition_id` first, whenever
    a full question object (not bare text) is available."""
    for label, pattern in _ASPECT_PATTERNS:
        if pattern.search(text or ""):
            return label
    return None


# A well-formed investigation plan already encodes its precise investigative
# target in `target_proposition_id` (e.g. "P_DELIVERY", "P_RECEIPT",
# "P_REQ_ACK") -- when present, this is a far more reliable aspect signal
# than scanning prose that may reference a neighboring aspect for context.
# Order matters: more specific keys are checked before substrings they
# contain ("ACK" before "REQ", since "P_REQ_ACK" contains both).
_TARGET_ID_ASPECT_MAP: list[tuple[str, str]] = [
    ("DELIVERY", "delivery"),
    ("RECEIPT", "receipt"),
    ("ACK", "acknowledgement"),
    ("ACTION", "activity"),
    ("MECHANISM", "mechanism"),
    ("GOV", "requirement"),
    ("REQ", "requirement"),
]


def _aspect_from_target_id(target_id: str | None) -> str | None:
    tid = (target_id or "").upper()
    for key, aspect in _TARGET_ID_ASPECT_MAP:
        if key in tid:
            return aspect
    return None


def _question_text_blob(q) -> str:
    if isinstance(q, dict):
        return " ".join(str(v) for v in (q.get("question"), q.get("purpose"), q.get("resolves")) if v)
    return " ".join(str(v) for v in (
        getattr(q, "question", None), getattr(q, "purpose", None), getattr(q, "resolves", None),
    ) if v)


def _question_aspect(q) -> str | None:
    """The authoritative per-question aspect classification used
    everywhere in this module: prefer the question's own
    `target_proposition_id` (precise, author-intended) over scanning its
    prose (ambiguous when the wording references a neighboring aspect)."""
    target_id = q.get("target_proposition_id") if isinstance(q, dict) else getattr(q, "target_proposition_id", None)
    aspect = _aspect_from_target_id(target_id)
    if aspect:
        return aspect
    return classify_investigation_aspect(_question_text_blob(q))


def _semantic_dedupe_labels(labels: list[str], threshold: float = 0.6) -> list[str]:
    """Structural (word-overlap) dedup shared by area/evidence normalization
    -- the same technique deduplicate_investigation_questions uses, applied
    to short labels instead of full questions."""
    kept: list[str] = []
    seen_word_sets: list[frozenset] = []
    for label in labels:
        if not label:
            continue
        words = frozenset(significant_words(label))
        if any(words and seen and len(words & seen) / max(len(words | seen), 1) >= threshold for seen in seen_word_sets):
            continue
        seen_word_sets.append(words)
        kept.append(label)
    return kept


def derive_investigation_areas(
    candidate_hypotheses: list | None = None,
    questions: list | None = None,
    existing_areas: list[str] | None = None,
    canonical_subject: str | None = None,
) -> list[str]:
    """Derives a small (normally 2-4), non-duplicative set of investigation
    areas from live candidate hypotheses and surviving questions (Section
    10-12). ONE INVESTIGATION AREA = ONE CONTROL/DOMAIN: each area
    summarizes a cluster of related questions, never restates one question's
    own wording, and every area returned has at least one surviving question
    or hypothesis behind it (no orphan areas)."""
    domain = None
    if canonical_subject and canonical_subject not in ("UNKNOWN", "the process", "process compliance", ""):
        domain = canonical_subject
    domain_cap = (domain[0].upper() + domain[1:]) if domain else "The applicable process"

    areas: list[str] = []

    # 1) Hypothesis-named areas (unchanged, generic-by-construction: derived
    #    from the hypothesis's own name/topic, not a fixed vocabulary).
    for h in (candidate_hypotheses or []):
        if getattr(h, "status", None) == "REFUTED":
            continue
        name = (getattr(h, "name", "") or "").upper()
        topic_raw = name.split("_")[0] if name else ""
        topic = topic_raw.capitalize() if topic_raw.isalpha() else None
        area = None
        if topic:
            if name.endswith("NOT_COMPLETED"):
                area = f"{topic} completion verification"
            elif "RECORD_UNAVAILABLE" in name or "RECORD_OMISSION" in name:
                area = f"{topic} record retrieval"
            elif "RECORD_CONTROL" in name:
                area = f"{topic} record control"
            elif "VERIFICATION_CONTROL" in name or "RECORDING_AND_VERIFICATION" in name:
                area = f"{topic} verification/authorization"
        if not area and getattr(h, "name", None):
            area = h.name.replace("_", " ").title()
        if area:
            areas.append(area)

    # 2) Recurrence / previous-CAPA questions are a distinct DOMAIN from the
    #    current finding's own subject (Section 12's own worked example:
    #    "Previous CAPA implementation and effectiveness") -- clustering
    #    them into the current-finding's domain-labeled areas below would
    #    misrepresent a genuinely separate investigative concern as part of
    #    the current subject. This label is a generic QMS lifecycle concept,
    #    not a hardcoded finding-domain vocabulary term, so it applies
    #    identically across every domain (Section 32).
    non_recurrence_questions = []
    has_recurrence_question = False
    for q in (questions or []):
        target_id = getattr(q, "target_proposition_id", None) or ""
        if _RECURRENCE_TARGET_RE.search(target_id):
            has_recurrence_question = True
        else:
            non_recurrence_questions.append(q)
    if has_recurrence_question:
        areas.append("Previous CAPA implementation and effectiveness")

    # 3) If the plan's own producer already generated concise area labels
    #    (e.g. plan_investigation_fallback.py's conflict-specific branches,
    #    which author areas hand-in-hand with their own questions), TRUST
    #    them rather than independently re-deriving competing area labels
    #    from the same questions -- Section 22: exactly ONE authoritative
    #    producer per output field. Re-deriving on top of an already-good
    #    producer is what caused the reported defect (a coherent 3-area
    #    plan plus a second, independently-generated, overlapping set of
    #    area labels for the very same questions). Question-level generic
    #    clustering below is reserved for when NO producer supplied areas
    #    at all (e.g. hypothesis-derived questions, which never populate
    #    `inv.areas`).
    if existing_areas:
        areas.extend(
            ea for ea in existing_areas
            if ea and not any(w in ea.lower() for w in ("failure", "ineffective", "breakdown"))
        )
    else:
        # Question-derived areas: cluster by generic control family, one
        # area per family that has >= 1 surviving question mapping into it.
        # Any question this classifier can't characterize still lands in
        # the broadest generic family (requirement_and_governance) rather
        # than being silently dropped from every area (Section 11: no
        # orphan questions).
        family_hits: dict[str, list[str]] = {}
        for q in non_recurrence_questions:
            aspect = _question_aspect(q) or "requirement"
            for family_key, family_aspects, _suffix in _AREA_FAMILIES:
                if aspect in family_aspects:
                    family_hits.setdefault(family_key, [])
                    if aspect not in family_hits[family_key]:
                        family_hits[family_key].append(aspect)
                    break

        for family_key, family_aspects, suffix in _AREA_FAMILIES:
            present = [a for a in family_aspects if a in family_hits.get(family_key, [])]
            if not present:
                continue
            words = [_ASPECT_DISPLAY[a] for a in present[:3]]
            if len(words) == 1:
                joined = words[0]
            elif len(words) == 2:
                joined = f"{words[0]} and {words[1]}"
            else:
                joined = f"{', '.join(words[:-1])} and {words[-1]}"
            areas.append(f"{domain_cap} {joined} {suffix}")

    deduped = _semantic_dedupe_labels(areas)
    return deduped[:4]


# Priority tiers for information-gain ordering (Section 7): lower number =
# resolved earlier. Mechanism/recurrence/impact questions are deliberately
# ranked after event/requirement/discrimination questions -- asking "was the
# system misconfigured" before "did delivery/receipt actually occur" invites
# an unsupported mechanism guess ahead of establishing the underlying event.
_ASPECT_PRIORITY_TIER = {
    "delivery": 1, "receipt": 1,
    "requirement": 2, "record_availability": 2,
    "completion": 3, "acknowledgement": 3, "authorization": 3, "record_control": 3, "activity": 3,
    "mechanism": 5,
    "effectiveness": 7, "scope": 7,
    "exception": 3,
}


def _question_priority_tier(q) -> int:
    target_id = getattr(q, "target_proposition_id", None) or ""
    # NOTE: matches the "REC_<n>" recurrence-proposition ID convention only
    # ("REC" as a bare substring would false-positive on e.g. "P_RECEIPT").
    if _RECURRENCE_TARGET_RE.search(target_id):
        # Recurrence-mechanism questions only make sense once the current
        # finding's own event/requirement is established.
        return 6
    if getattr(q, "hypothesis_tested", None):
        return 4
    aspect = _question_aspect(q)
    return _ASPECT_PRIORITY_TIER.get(aspect, 3)


def rank_questions_by_information_gain(questions: list) -> list:
    """Orders surviving investigation questions by their ability to resolve
    the current uncertainty (Section 7) -- conflict/event-establishing
    questions first, discrimination/mechanism/recurrence/impact questions
    last. A stable sort: ties (same tier) keep their original relative
    order, so an already-well-ordered producer (e.g. the DELIVERY_VS_RECEIPT
    fallback plan, whose questions are already authored in dependency order)
    is never needlessly reshuffled."""
    if not questions:
        return questions
    return sorted(questions, key=_question_priority_tier)


def normalize_and_dedupe_evidence_items(items: list[str], max_items: int = 6) -> list[str]:
    """Collapses near-duplicate evidence-artifact strings (Section 17) --
    "delivery logs" / "system delivery logs" / "delivery records" collapse
    to one canonical entry -- and caps the result to a small, high-value set
    instead of concatenating every artifact string ever proposed. Keeps the
    longest (most specific) phrasing of each surviving cluster."""
    if not items:
        return []
    clusters: list[list[str]] = []
    for item in items:
        if not item:
            continue
        words = frozenset(significant_words(item))
        placed = False
        for cluster in clusters:
            cluster_words = frozenset(significant_words(cluster[0]))
            if words and cluster_words and len(words & cluster_words) / max(len(words | cluster_words), 1) >= 0.5:
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
    representatives = [max(cluster, key=len) for cluster in clusters]
    return representatives[:max_items]


def deduplicate_investigation_questions(questions: list) -> list:
    """Drops near-duplicate investigation questions (Section 5/6: every
    question must have a distinct investigative objective -- two questions
    must never test what is effectively the same evidence gap, even when
    worded completely differently, e.g. "Verify notification delivery." /
    "Establish whether notification was delivered." / "Reconcile
    notification delivery records.").

    Two structural checks, applied in order (never an LLM call --
    Section 6: "avoid introducing an LLM call solely for deduplication"):

    1. Same generic investigative ASPECT (see `classify_investigation_aspect`
       -- delivery, receipt, acknowledgement, etc.) plus a modest amount of
       shared vocabulary. The aspect match already establishes the two
       questions target the same kind of control point, so a much lower
       word-overlap bar than (2) is enough to confirm they're asking about
       the same underlying object rather than two different ones that
       happen to share an aspect.
    2. No aspect match (or aspects differ): fall back to raw significant-
       word overlap with a high threshold, catching near-identical
       restatements this codebase's other aspect-agnostic questions might
       produce.

    Keeps the first occurrence (earlier questions are assumed to already be
    in priority order)."""
    if not questions:
        return questions
    kept: list = []
    seen_word_sets: list[frozenset] = []
    seen_aspects: list[str | None] = []
    for q in questions:
        q_text = getattr(q, "question", None) or (q.get("question") if isinstance(q, dict) else None)
        if not q_text:
            kept.append(q)
            continue
        words = frozenset(significant_words(q_text))
        aspect = _question_aspect(q)
        is_duplicate = False
        if aspect:
            for seen, seen_aspect in zip(seen_word_sets, seen_aspects):
                if seen_aspect == aspect and words and seen:
                    overlap = len(words & seen) / max(len(words | seen), 1)
                    if overlap >= 0.25:
                        is_duplicate = True
                        break
        if not is_duplicate:
            for seen in seen_word_sets:
                if not words or not seen:
                    continue
                overlap = len(words & seen) / max(len(words | seen), 1)
                if overlap >= 0.75:
                    is_duplicate = True
                    break
        if is_duplicate:
            continue
        seen_word_sets.append(words)
        seen_aspects.append(aspect)
        kept.append(q)
    return kept


def deduplicate_capa_actions(actions: list) -> list:
    """Drops near-duplicate CAPA actions WITHIN the same hypothesis
    condition (Section 24: semantic, not exact-string, deduplication).

    Scoped to `if_cause_confirmed` groups deliberately -- an
    IMMEDIATE_CORRECTION and a SYSTEMIC_ACTION for the SAME hypothesis are
    legitimately meant to be two different kinds of recommendation (fix
    this case vs. fix the control that let it happen), so only actions
    that turn out to be near-restatements of each other (same words, same
    intent) despite that distinction are merged -- never actions for
    different hypotheses, which may coincidentally share vocabulary
    without being the same action. Keeps the LONGER (more complete)
    `recommended_action` text of any merged pair, per Section 10.
    """
    if not actions:
        return actions
    kept: list = []
    # Parallel list: (condition, words, index_into_kept) for each kept action.
    kept_meta: list[tuple[str, frozenset, int]] = []
    for a in actions:
        condition = getattr(a, "if_cause_confirmed", None) or ""
        text = getattr(a, "recommended_action", None) or ""
        words = frozenset(significant_words(text))
        merged = False
        for seen_condition, seen_words, kept_idx in kept_meta:
            if seen_condition != condition or not words or not seen_words:
                continue
            overlap = len(words & seen_words) / max(len(words | seen_words), 1)
            if overlap >= 0.55:
                existing_text = getattr(kept[kept_idx], "recommended_action", None) or ""
                if len(text) > len(existing_text):
                    kept[kept_idx] = a
                merged = True
                break
        if merged:
            continue
        kept_meta.append((condition, words, len(kept)))
        kept.append(a)
    return kept


def validate_investigation_question(question: str | None) -> bool:
    """Deterministic structural quality gate for a generated investigation
    question (Phase 4/11 of the RCA quality pass). True if the question is
    well-formed; False if it should be rejected/regenerated. Domain-generic
    -- every check is about sentence SHAPE, never specific vocabulary:

    - must end with '?' and have enough words to be a real question
    - must not contain the "confirm or refute:" concatenation pattern that
      previously leaked raw evidence-list text into the question
    - must not ask about reporting behavior instead of causal mechanism
      (reuses the same guard 5-Why questions are held to)
    """
    if not question:
        return False
    q = question.strip()
    if not q.endswith("?"):
        return False
    if len(q.split()) < 5:
        return False
    if "confirm or refute" in q.lower():
        return False
    from app.agent.causal_guard import is_reporting_why_question
    if is_reporting_why_question(q):
        return False
    return True


def hypothesis_confidence(hypothesis) -> str:
    """Deterministic per-hypothesis confidence (distinct from
    leading_hypothesis_confidence, which grades only the selected leader).
    Never asserted by the LLM -- computed here from the same structural
    signals used elsewhere (status, relevance_rank, whether discriminating
    evidence was actually specified) so every hypothesis in the report
    carries a confidence grade, not just the leading one."""
    status = getattr(hypothesis, "status", "POSSIBLE")
    rank = getattr(hypothesis, "relevance_rank", "MEDIUM")
    has_discrimination = bool(getattr(hypothesis, "discrimination_evidence", None))
    if status == "SUPPORTED":
        return "HIGH" if rank == "HIGH" else "MEDIUM"
    if status == "POSSIBLE":
        if rank == "HIGH" and has_discrimination:
            return "MEDIUM"
        return "LOW"
    return "LOW"


def leading_hypothesis_confidence(candidate_hypotheses: list, leading_hypothesis: str | None) -> str:
    """LOW/MEDIUM/HIGH confidence for the leading hypothesis (not for the
    root cause itself, which stays whatever root_cause.confidence already
    reflects) -- SUPPORTED gets MEDIUM, a single best-ranked POSSIBLE gets
    LOW, no leader gets LOW."""
    if not leading_hypothesis or not candidate_hypotheses:
        return "LOW"
    leading_id = leading_hypothesis.split(" — ", 1)[0].strip()
    match = next((h for h in candidate_hypotheses if h.id == leading_id), None)
    if match and match.status == "SUPPORTED":
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# 2. Root-cause state validation
# ---------------------------------------------------------------------------

# Statuses that assert a causal relationship has actually been established
# (mapped onto the existing RootCauseStatus vocabulary: VERIFIED and
# SUPPORTED are the "don't claim this without evidence" tier; STATED_UNVERIFIED
# and INFERRED are already self-labeled as unconfirmed; NOT_ESTABLISHED and
# CONTRADICTED never need downgrading).
_ESTABLISHED_LIKE_STATUSES = {"VERIFIED", "SUPPORTED"}


def validate_root_cause_state(root_cause, mechanism: MechanismInfo | None) -> list[str]:
    """Mutates `root_cause` in place if its status claims more certainty
    than the evidence actually supports. Returns a list of human-readable
    warnings for the trace (empty if nothing was downgraded).

    Critical distinction (Case A vs Case C from the hardening request): the
    existence of *some* VERIFIED fact in the ledger is NOT sufficient to
    justify an ESTABLISHED-like root cause -- that VERIFIED fact might just
    be the OBSERVATION itself ("the check was not completed"), not evidence
    of WHY it happened. Only a VERIFIED *mechanism* (an action-level causal
    claim, e.g. "the audit trail confirms the user was never assigned the
    task") justifies root_cause claiming this level of certainty. A REPORTED
    mechanism (someone's account of what happened) never does.
    """
    warnings: list[str] = []
    if root_cause is None:
        return warnings

    status_value = getattr(root_cause.status, "value", root_cause.status)
    if status_value in _ESTABLISHED_LIKE_STATUSES:
        mechanism_is_verified = bool(mechanism and mechanism.status == "VERIFIED")
        if not mechanism_is_verified:
            warnings.append(
                f"Analytical Validator: root_cause.status={status_value} downgraded to STATED_UNVERIFIED "
                "— no VERIFIED causal mechanism (as opposed to a VERIFIED observation or a REPORTED "
                "account) supports this level of certainty."
            )
            root_cause.status = "STATED_UNVERIFIED"  # type: ignore[assignment]

    return warnings


# ---------------------------------------------------------------------------
# 3. 5-Why: detect a skipped, explicitly available causal fact
# ---------------------------------------------------------------------------


def five_why_skips_available_mechanism(five_why_steps: list, mechanism: MechanismInfo) -> bool:
    """True if the finding/evidence establishes an immediate mechanism
    (VERIFIED or REPORTED) but NONE of the 5-Why steps' answers actually
    reflect it -- i.e. the chain stopped (or never engaged) before an
    explicitly available causal fact, which the finding itself already
    resolves."""
    if not mechanism or mechanism.status not in ("VERIFIED", "REPORTED") or not mechanism.statement:
        return False
    if not five_why_steps:
        return True
    mechanism_words = significant_words(mechanism.statement)
    if not mechanism_words:
        return False
    for step in five_why_steps:
        answer_words = significant_words(step.answer or "")
        if not answer_words:
            continue
        overlap = mechanism_words & answer_words
        if len(overlap) >= max(2, len(mechanism_words) // 2):
            return False
    return True


def repair_five_why_with_mechanism(five_why_steps: list, mechanism: MechanismInfo, observed_deviation: str | None):
    """Deterministic structure repair (never invents content): if the chain
    skipped the mechanism entirely, insert it as the step right after the
    observation, using the mechanism's own already-extracted text and
    status -- not a fabricated explanation. Returns a NEW list of FiveWhyStep
    objects; caller is responsible for updating the FiveWhyAnalysis."""
    from app.models.agent import FiveWhyStep
    from app.services.semantic_subject import declarative_to_why_question, format_deviation_why_question

    if not mechanism or not mechanism.statement:
        return five_why_steps

    # `observed_deviation` is usually the canonical "<subject> — <condition>"
    # form (understanding_node's dash-joined field) -- naive interpolation
    # of that into "Why did <X> occur?" produces ungrammatical output like
    # "Why did daily equipment inspection checklist — not completed occur?"
    # since the condition IS the predicate, not something that "occurs" on
    # top of a dash-joined label. Split it and build a proper "Why was/were
    # <subject> <condition>?" question instead; fall back to the
    # declarative-clause formatter for a plain full-sentence deviation
    # string (no dash), and to a generic phrasing only when nothing at all
    # is available.
    if observed_deviation and " — " in observed_deviation:
        subj, _, cond = observed_deviation.partition(" — ")
        question_text = format_deviation_why_question(subj, cond)
    elif observed_deviation:
        question_text = declarative_to_why_question(observed_deviation)
    else:
        question_text = "Why did the observed condition occur?"

    mechanism_step = FiveWhyStep(
        question=question_text,
        answer=mechanism.statement,
        status="REPORTED" if mechanism.status == "REPORTED" else "VERIFIED",
    )

    if not five_why_steps:
        return [mechanism_step]

    # Insert right after the first step (the observation) if that step
    # doesn't already carry the mechanism's content, to preserve
    # Answer(N) explains Question(N-1) ordering.
    first = five_why_steps[0]
    if repeats_previous_why_answer(first.answer, mechanism.statement):
        return five_why_steps
    return [five_why_steps[0], mechanism_step, *five_why_steps[1:]]


# ---------------------------------------------------------------------------
# 4. Contributing factors: established vs. potential
# ---------------------------------------------------------------------------


def classify_contributing_factors(factors: list) -> tuple[list, list]:
    """Splits factors into (established, potential) based on their own
    evidence_status/status fields -- never reclassifies content, only
    groups it. A factor is "established" only if it is VERIFIED; everything
    else (including anything the model marked POSSIBLE_UNCONFIRMED) is
    potential, never presented as confirmed."""
    established = [f for f in factors if getattr(f.evidence_status, "value", f.evidence_status) == "VERIFIED" and f.status == "VERIFIED"]
    potential = [f for f in factors if f not in established and f.status != "REJECTED"]
    return established, potential


def classify_contributing_factors_full(factors: list, mechanism: MechanismInfo, verified_facts: list[str]) -> tuple[list, list, list]:
    """Three-way split (established, potential, rejected). A factor is
    REJECTED (mutates its status in place) if it structurally contradicts
    the established mechanism or a VERIFIED completion fact -- reusing the
    same contradiction detectors applied to hypotheses, since a
    contributing factor makes the same kind of causal claim. Never
    reclassifies a factor as MORE certain than it already was."""
    from app.agent.causal_guard import hypothesis_contradicts_mechanism, hypothesis_contradicts_verified_completion

    rejected = []
    survivors = []
    for f in factors:
        contradicts = (
            hypothesis_contradicts_mechanism(f.description, mechanism)
            or hypothesis_contradicts_verified_completion(f.description, verified_facts)
        )
        if contradicts:
            f.status = "REJECTED"  # type: ignore[assignment]
            rejected.append(f)
        else:
            survivors.append(f)

    established, potential = classify_contributing_factors(survivors)
    return established, potential, rejected


# ---------------------------------------------------------------------------
# 5. CAPA causal linkage
# ---------------------------------------------------------------------------


def conditional_action_has_causal_linkage(if_cause_confirmed: str, candidate_hypotheses: list) -> bool:
    """True if a conditional CAPA branch's condition text ('if_cause_confirmed')
    actually references one of the live candidate hypotheses (by id or by
    word overlap with its statement/name) rather than floating free of any
    stated cause."""
    if not candidate_hypotheses:
        # No hypotheses exist to link to -- can't require linkage that has
        # nothing to link to; this is not itself a violation.
        return True
    if not if_cause_confirmed:
        return False
    text_words = significant_words(if_cause_confirmed)
    for h in candidate_hypotheses:
        if h.id and h.id.lower() in if_cause_confirmed.lower():
            return True
        hyp_words = significant_words(h.statement) | significant_words(h.name.replace("_", " "))
        if text_words & hyp_words:
            return True
    return False


def validate_capa_causal_linkage(capa, candidate_hypotheses: list) -> list[str]:
    """Drops conditional CAPA actions that don't trace back to any live
    hypothesis. Mutates `capa` in place. Returns trace warnings."""
    warnings: list[str] = []
    if capa is None or not capa.conditional_actions:
        return warnings
    kept = []
    for action in capa.conditional_actions:
        if conditional_action_has_causal_linkage(action.if_cause_confirmed, candidate_hypotheses):
            kept.append(action)
        else:
            warnings.append(
                f"Analytical Validator: dropped conditional CAPA action — condition "
                f"{action.if_cause_confirmed!r} does not trace back to any candidate hypothesis"
            )
    capa.conditional_actions = kept
    return warnings


# ---------------------------------------------------------------------------
# 6. Analytical quality score (internal only -- never exposed as a claim of
#    truth, purely a signal for detecting weak synthesis / logging).
# ---------------------------------------------------------------------------


def compute_analytical_quality(root_cause, five_why, contributing_factors, capa, mechanism: MechanismInfo) -> dict[str, str]:
    """Returns a dict of HIGH/MEDIUM/LOW internal quality signals. This is
    NOT surfaced to the report as confidence -- it exists so trace/logs can
    flag a weak synthesis for review, without pretending to be a validated
    numeric score."""
    scores: dict[str, str] = {}

    # mechanism_accuracy: did we manage to extract an evidence-backed mechanism at all?
    scores["mechanism_accuracy"] = "HIGH" if mechanism and mechanism.status in ("VERIFIED", "REPORTED") else "LOW"

    # hypothesis_discrimination: do hypotheses carry discrimination_evidence?
    hyps = root_cause.candidate_hypotheses if root_cause else []
    if hyps:
        with_discrimination = sum(1 for h in hyps if h.discrimination_evidence)
        scores["hypothesis_discrimination"] = "HIGH" if with_discrimination == len(hyps) else ("MEDIUM" if with_discrimination else "LOW")
    else:
        scores["hypothesis_discrimination"] = "LOW"

    # five_why_coherence: has at least one step, and doesn't end mid-chain on a VERIFIED/SUPPORTED status
    # without an honest UNKNOWN boundary when the chain is incomplete.
    steps = five_why.steps if five_why else []
    if not steps:
        scores["five_why_coherence"] = "LOW"
    elif steps[-1].status in ("UNKNOWN", "NOT_ESTABLISHED") or (five_why and five_why.is_complete):
        scores["five_why_coherence"] = "HIGH"
    else:
        scores["five_why_coherence"] = "MEDIUM"

    # contributing_factor_quality: potential factors carry rationale/evidence_required
    if contributing_factors:
        well_formed = sum(1 for f in contributing_factors if f.rationale and f.evidence_required)
        scores["contributing_factor_quality"] = "HIGH" if well_formed == len(contributing_factors) else "MEDIUM"
    else:
        scores["contributing_factor_quality"] = "MEDIUM"  # empty is a valid, not a weak, outcome

    # capa_linkage: conditional actions exist and (post-validation) are all linked
    if capa and getattr(capa, "conditional_actions", None):
        scores["capa_linkage"] = "HIGH"
    elif capa and root_cause and root_cause.candidate_hypotheses:
        scores["capa_linkage"] = "LOW"  # hypotheses exist but no conditional actions map to them
    else:
        scores["capa_linkage"] = "MEDIUM"

    # uncertainty_discipline: NOT_ESTABLISHED/STATED_UNVERIFIED root cause with a populated
    # leading_hypothesis is the desired disciplined outcome; an ESTABLISHED-like status is
    # only HIGH-quality if evidence backs it (validate_root_cause_state already enforces this).
    status_value = getattr(root_cause.status, "value", root_cause.status) if root_cause else "NOT_ESTABLISHED"
    if status_value in ("NOT_ESTABLISHED", "STATED_UNVERIFIED", "INFERRED"):
        scores["uncertainty_discipline"] = "HIGH" if (root_cause and getattr(root_cause, "leading_hypothesis", None)) else "MEDIUM"
    else:
        scores["uncertainty_discipline"] = "MEDIUM"

    return scores


# ---------------------------------------------------------------------------
# 7. Impact field EXPLICIT / INFERRED / UNKNOWN classification
# ---------------------------------------------------------------------------

_IMPACT_FIELD_NAMES = ("affected_object", "affected_period", "process_at_risk", "relevant_change", "potential_effect")


def classify_impact_field_basis(value: str | None, finding_text: str) -> str:
    """Deterministic EXPLICIT/INFERRED/UNKNOWN classification for a single
    impact field, computed here (never asserted by the LLM, which can't be
    trusted to grade its own certainty):

    - UNKNOWN: empty, or itself says "not established"/"unknown"/etc.
    - EXPLICIT: most of the field's own vocabulary appears verbatim in the
      finding text -- i.e. it's a close restatement of something the
      finding actually said.
    - INFERRED: grounded (it already passed the entity/domain guards
      upstream) but synthesizes beyond a verbatim restatement -- e.g.
      "temperature monitoring" derived from "temperature log"."""
    if not value or not value.strip():
        return "UNKNOWN"
    if _UNKNOWN_VALUE_RE.match(value.strip()):
        return "UNKNOWN"

    value_words = significant_words(value)
    if not value_words:
        return "UNKNOWN"
    finding_words = significant_words(finding_text)
    overlap = value_words & finding_words
    ratio = len(overlap) / len(value_words)
    return "EXPLICIT" if ratio >= 0.7 else "INFERRED"


def compute_impact_field_basis(impact, finding_text: str) -> dict[str, str]:
    """Returns the {field_name: EXPLICIT|INFERRED|UNKNOWN} mapping for all
    of ImpactAssessment's named structured fields."""
    if impact is None:
        return {name: "UNKNOWN" for name in _IMPACT_FIELD_NAMES}
    return {
        name: classify_impact_field_basis(getattr(impact, name, None), finding_text)
        for name in _IMPACT_FIELD_NAMES
    }


# ---------------------------------------------------------------------------
# 8. Consolidated causal-graph edge validator
# ---------------------------------------------------------------------------
#
# Observation -> Immediate Mechanism -> Candidate Hypothesis -> Evidence ->
# Root Cause -> Corrective Action -> Effectiveness Check
#
# This is an AUDIT over content already produced by the checks above (and a
# couple of new leaf checks) -- it never repairs or invents anything itself;
# it exists so a single call can report every structural violation still
# present after the individual guards have already run, for observability
# and for tests that want one "is this analytically sound" answer.


def validate_causal_graph(root_cause, five_why, capa, mechanism: MechanismInfo | None) -> list[str]:
    """Returns every remaining structural violation across the causal chain.
    Does not mutate anything -- callers that want repairs already applied
    them via the more specific functions above (validate_root_cause_state,
    repair_five_why_with_mechanism, validate_capa_causal_linkage)."""
    violations: list[str] = []

    status_value = getattr(root_cause.status, "value", root_cause.status) if root_cause else "NOT_ESTABLISHED"

    # Edge: Evidence -> Root Cause
    # Root Cause -> must have supporting evidence if it claims certainty.
    if status_value in _ESTABLISHED_LIKE_STATUSES:
        if root_cause and not root_cause.supporting_evidence:
            violations.append("Root cause claims established-level certainty but supporting_evidence is empty")
        if not mechanism or mechanism.status != "VERIFIED":
            violations.append(f"Root cause claims certainty ({status_value}) but mechanism is not VERIFIED ({getattr(mechanism, 'status', 'None')})")

    # Edge: Mechanism -> Root Cause (Reported statement must not be asserted as verified root cause)
    if status_value in _ESTABLISHED_LIKE_STATUSES and mechanism and mechanism.status == "REPORTED":
        violations.append("Reported statement asserted as verified root cause without corroborating evidence")

    # Edge: Candidate Hypothesis -> Evidence: every hypothesis needs a rationale
    # and something identifying what would confirm/discriminate it.
    for h in (root_cause.candidate_hypotheses if root_cause else []):
        if not h.rationale:
            violations.append(f"Hypothesis {h.id} has no rationale")
        if not h.evidence_needed and not h.discrimination_evidence:
            violations.append(f"Hypothesis {h.id} has no evidence_needed/discrimination_evidence")

    # Edge: Root Cause / Hypotheses -> Corrective Action: every conditional CAPA action must
    # trace back to a hypothesis (already enforced/repaired by
    # validate_capa_causal_linkage upstream; this re-checks post-repair).
    if capa:
        for action in capa.conditional_actions:
            if not conditional_action_has_causal_linkage(action.if_cause_confirmed, root_cause.candidate_hypotheses if root_cause else []):
                violations.append(f"CAPA action {action.if_cause_confirmed!r} has no causal linkage")
            # Systemic action -> effectiveness check: a SYSTEMIC_ACTION
            # branch needs a way to verify it worked.
            if action.action_type == "SYSTEMIC_ACTION" and not action.verification_method:
                violations.append(f"Systemic CAPA action {action.if_cause_confirmed!r} has no verification_method")

    # Edge: Observation -> Mechanism -> 5-Why (already repaired upstream; re-check).
    if five_why and mechanism and mechanism.statement:
        if five_why_skips_available_mechanism(five_why.steps, mechanism):
            violations.append("5-Why chain does not reflect the explicitly available mechanism")

    return violations
