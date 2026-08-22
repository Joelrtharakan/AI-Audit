"""Machine-Readable Invariant Registry for the LQMS AI Finding Investigation Agent.

Every production invariant is formalized here with an explicit invariant ID,
description, severity, category, and validation function.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from app.models.agent import RootCauseStatus


class InvariantSeverity(str, Enum):
    BLOCKER = "BLOCKER"
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"


class InvariantCategory(str, Enum):
    CAUSAL = "CAUSAL"
    FINANCIAL = "FINANCIAL"
    FIVE_WHY = "FIVE_WHY"
    HYPOTHESIS = "HYPOTHESIS"
    CAPA = "CAPA"
    SEMANTIC = "SEMANTIC"
    SECURITY = "SECURITY"


@dataclass
class InvariantRule:
    inv_id: str
    category: InvariantCategory
    severity: InvariantSeverity
    description: str
    validate: Callable[[dict[str, Any]], tuple[bool, str | None]]


def _check_indicative_not_promoted(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-005: a common-factor/pattern-derived hypothesis
    (evidence_strength=INDICATIVE) can never be SUPPORTED/ESTABLISHED --
    only genuine mechanism evidence (VERIFIED) earns that. A pattern across
    multiple observations is a lead, never proof."""
    rc = state.get("root_cause")
    if not rc or not getattr(rc, "candidate_hypotheses", None):
        return True, None
    for h in rc.candidate_hypotheses:
        if getattr(h, "evidence_strength", "") == "INDICATIVE" and getattr(h, "status", "") in ("SUPPORTED", "ESTABLISHED"):
            return False, f"Hypothesis {h.id} is {h.status} on INDICATIVE (pattern-only) evidence"
    return True, None


def _check_reported_not_promoted(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-006: a hypothesis grounded only in REPORTED evidence
    (someone's account -- e.g. a supervisor's characterization of an
    operator as careless) can never be SUPPORTED/ESTABLISHED, and REPORTED-
    or UNVERIFIED-only evidence can never make root_cause.status itself
    SUPPORTED/ESTABLISHED. A reported explanation is a candidate hypothesis
    at most, never established causal fact, regardless of how confidently
    it was worded or which node produced it."""
    rc = state.get("root_cause")
    if not rc:
        return True, None
    for h in getattr(rc, "candidate_hypotheses", None) or []:
        if getattr(h, "evidence_strength", "") in ("REPORTED", "NONE") and getattr(h, "status", "") in ("SUPPORTED", "ESTABLISHED"):
            return False, f"Hypothesis {h.id} is {h.status} on {h.evidence_strength} (non-verified) evidence"
    return True, None


def _check_causal_provenance(state: dict[str, Any]) -> tuple[bool, str | None]:
    rc = state.get("root_cause")
    if not rc:
        return True, None
    if rc.status in (RootCauseStatus.ESTABLISHED, RootCauseStatus.SUPPORTED):
        if not getattr(rc, "leading_hypothesis", None) and getattr(rc, "leading_hypothesis_status", "") != "SELECTED":
            return False, "Root cause marked ESTABLISHED/SUPPORTED without a selected leading hypothesis or objective verification"
    return True, None


def _check_causal_no_unsupported_promotion(state: dict[str, Any]) -> tuple[bool, str | None]:
    canonical = state.get("canonical_finding_state")
    rc = state.get("root_cause")
    conflicts = getattr(canonical, "evidence_conflicts", []) if canonical else []
    if conflicts and rc and rc.status in (RootCauseStatus.ESTABLISHED, RootCauseStatus.SUPPORTED):
        return False, "Root cause promoted despite unresolved evidence conflicts"
    return True, None


def _check_financial_visibility(state: dict[str, Any]) -> tuple[bool, str | None]:
    finding_text = (state.get("request") or getattr(state.get("canonical_finding_state"), "finding_text", None) or "")
    if hasattr(finding_text, "finding_text"):
        finding_text = finding_text.finding_text
    cost_impact = state.get("cost_impact") or getattr(state.get("report"), "cost_impact", None)

    # Delegates to the same deterministic gate cost_impact was computed
    # from (app.services.cost_analysis.has_cost_signals) rather than a
    # second, independent regex -- two different definitions of "is this
    # financial" is exactly how this invariant used to disagree with the
    # detector it is meant to be checking.
    from app.services.cost_analysis import has_cost_signals
    has_financial_signal = has_cost_signals(str(finding_text))
    if not has_financial_signal and cost_impact and getattr(cost_impact, "cost_factor_detected", False):
        return False, "Financial section displayed when no financial factor exists in finding text"
    return True, None


def _check_financial_loss_separation(state: dict[str, Any]) -> tuple[bool, str | None]:
    cost_impact = state.get("cost_impact") or getattr(state.get("report"), "cost_impact", None)
    if not cost_impact or not getattr(cost_impact, "cost_factor_detected", False):
        return True, None
    if getattr(cost_impact, "financial_factor", "") == "DUPLICATE PAYMENT":
        if getattr(cost_impact, "actual_loss_status", "") == "VERIFIED" and getattr(cost_impact, "recoverability_status", "") == "REQUIRES_VERIFICATION":
            return False, "Duplicate payment transaction amount was prematurely converted into a confirmed actual loss"
    return True, None


def _check_estimated_amount_not_actual_loss(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-FIN-002: an ESTIMATED/REPORTED financial amount (e.g. "approximately
    ₹4 lakh") can never automatically become a VERIFIED actual financial
    loss -- that requires its own independent loss-confirmation evidence,
    never just successful amount extraction/normalization."""
    cost_impact = state.get("cost_impact") or getattr(state.get("report"), "cost_impact", None)
    if not cost_impact or not getattr(cost_impact, "cost_factor_detected", False):
        return True, None
    fin_status = getattr(cost_impact, "financial_status", None)
    if fin_status in ("ESTIMATED", "REPORTED") and getattr(cost_impact, "actual_loss_status", None) == "VERIFIED":
        return False, f"financial_status={fin_status} but actual_loss_status=VERIFIED — an estimated/reported amount was promoted to a confirmed loss"
    return True, None


def _check_capa_effectiveness_not_overclaimed(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAPA-001/002: CAPA completion never proves effectiveness, and
    missing effectiveness evidence never proves ineffectiveness -- the
    recurrence-risk rationale (and root-cause narrative) must never assert
    a previous CAPA WAS ineffective as settled fact, only that effectiveness
    is unverified."""
    rc = state.get("root_cause")
    if not rc:
        return True, None
    texts = [getattr(rc, "risk_of_recurrence_rationale", None), getattr(rc, "narrative", None)]
    ineffective_re = re.compile(r"\b(?:was|were)\s+ineffective\b|\bineffective\s+in\s+preventing\b", re.IGNORECASE)
    # A match preceded (within a short window) by a hedge/negation phrase
    # ("not proof that", "does not establish", "never establishes") is the
    # CORRECT conservative wording this invariant exists to require, not a
    # violation of it -- only an UNQUALIFIED assertion is the defect.
    negation_re = re.compile(
        r"\b(?:not\s+proof|does\s+not\s+establish|is\s+not\s+established|never\s+establishes?|"
        r"not\s+evidence|no\s+evidence)\b",
        re.IGNORECASE,
    )
    for text in texts:
        if not text:
            continue
        for m in ineffective_re.finditer(text):
            window = text[max(0, m.start() - 60):m.start()]
            if not negation_re.search(window):
                return False, f"text asserts previous CAPA ineffectiveness as settled fact: {text!r}"
    return True, None


def _check_recurrence_investigation_coverage(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-CAPA: a finding describing BOTH a recurring/repeated
    issue AND a previous CAPA must generate at least one investigation
    question targeting CAPA implementation/effectiveness/scope, not rely
    solely on generic questions unrelated to that specific, stronger
    investigation lead."""
    from app.agent.recurrence_guard import detect_recurrence
    request = state.get("request")
    finding_text = getattr(request, "finding_text", None) if request else None
    if not finding_text:
        return True, None
    rec = detect_recurrence(finding_text)
    if not (rec.is_recurring and rec.has_previous_capa_reference):
        return True, None
    inv = state.get("investigation_plan")
    if not inv or not getattr(inv, "questions", None):
        return True, None
    capa_word_re = re.compile(r"\b(?:capa|corrective\s+action|effectiveness|implement)\w*\b", re.IGNORECASE)
    if not any(capa_word_re.search(q.question or "") for q in inv.questions):
        return False, "recurring finding with a previous CAPA reference has no CAPA-implementation/effectiveness investigation question"
    return True, None


def _check_five_why_no_repetition(state: dict[str, Any]) -> tuple[bool, str | None]:
    fw = state.get("five_why")
    if not fw or not getattr(fw, "steps", None):
        return True, None
    from app.agent.causal_guard import is_circular_why_answer, repeats_previous_why_answer
    for i, step in enumerate(fw.steps):
        ans = step.answer or ""
        if is_circular_why_answer(step.question, ans) and step.status not in ("UNKNOWN", "NOT_ESTABLISHED"):
            return False, f"5-Why step {i+1} answer circular-repeats its question"
        if i > 0 and repeats_previous_why_answer(fw.steps[i-1].answer, ans) and step.status not in ("UNKNOWN", "NOT_ESTABLISHED"):
            return False, f"5-Why step {i+1} repeats previous step answer"
    return True, None


def _check_hypothesis_citations(state: dict[str, Any]) -> tuple[bool, str | None]:
    rc = state.get("root_cause")
    if not rc or not getattr(rc, "candidate_hypotheses", None):
        return True, None
    for h in rc.candidate_hypotheses:
        if not getattr(h, "supporting_evidence", []) and not getattr(h, "supporting_claim_ids", []):
            return False, f"Hypothesis {h.id} lacks supporting evidence provenance"
    return True, None


def _check_capa_conditionality(state: dict[str, Any]) -> tuple[bool, str | None]:
    rc = state.get("root_cause")
    capa = state.get("capa_analysis")
    if not capa or not getattr(capa, "conditional_actions", None):
        return True, None
    if rc and rc.status == RootCauseStatus.NOT_ESTABLISHED:
        for a in capa.conditional_actions:
            if getattr(a, "action_type", "") in ("CORRECTIVE_ACTION", "SYSTEMIC_ACTION") and not getattr(a, "if_cause_confirmed", None):
                return False, "Systemic CAPA was recommended unconditionally when root cause was not established"
    return True, None


# Generic, content-free entity names. These are placeholders BY SHAPE: they
# name a category of processes rather than any object that exists in a
# particular finding, so they can never be a correct affected_object no
# matter which finding produced them. Kept as a small closed set of generic
# nouns (not a per-finding blocklist) plus a structural test for "generic
# adjective + generic process noun".
_GENERIC_PLACEHOLDER_ENTITY_RE = re.compile(
    r"^(?:the\s+)?(?:overall\s+|general\s+|relevant\s+|affected\s+|applicable\s+)?"
    r"(?:process|procedural|operational|organi[sz]ational|business|quality|"
    r"regulatory|system|systemic)?\s*"
    r"(?:compliance|process|procedure|activity|operation|control|practice|"
    r"system|workflow|area|function|requirement)s?$",
    re.IGNORECASE,
)
_EXPLICIT_PLACEHOLDERS = {"process compliance", "employees control", "the affected process"}


def is_generic_placeholder_entity(value: str | None) -> bool:
    """True when `value` is a generic process-category label rather than a
    concrete entity from the finding. Shape-based, so it catches
    'operational process', 'the relevant control', 'quality system' etc.
    without a per-case blocklist."""
    if not value:
        return False
    v = value.strip().strip(".").lower()
    if v in _EXPLICIT_PLACEHOLDERS:
        return True
    return bool(_GENERIC_PLACEHOLDER_ENTITY_RE.match(v))


def _check_semantic_cleanliness(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-002 (BLOCKER): affected_object / finding_subject must name a
    concrete entity from the finding, never a generic process-category
    placeholder. An UNRESOLVED/PARTIAL marker is explicitly ALLOWED -- an
    honest 'entity extraction was uncertain' is the correct output when
    extraction genuinely fails; a confident-sounding fake is not."""
    canonical = state.get("canonical_finding_state")
    impact = state.get("impact_assessment")
    checks: list[tuple[str, str]] = []
    if canonical:
        for name in ("affected_object", "finding_subject"):
            val = getattr(canonical, name, None)
            if val:
                checks.append((f"canonical.{name}", val))
    if impact and getattr(impact, "affected_object", None):
        checks.append(("impact.affected_object", impact.affected_object))
    for label, val in checks:
        if str(val).strip().upper().startswith(("UNKNOWN", "UNRESOLVED")):
            continue  # explicitly flagged as unresolved -- honest, not degraded
        if is_generic_placeholder_entity(val):
            return False, f"Degraded placeholder {label}: {val!r}"
    return True, None


def _check_entity_resolution_flagged(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-004 (CRITICAL): when entity extraction did not fully resolve,
    the state must SAY so (entity_resolution != RESOLVED implies an
    entity_resolution_note), so an uncertain entity never reaches the report
    looking like a confident one."""
    canonical = state.get("canonical_finding_state")
    if not canonical:
        return True, None
    res = getattr(canonical, "entity_resolution", "RESOLVED")
    if res not in ("RESOLVED", "PARTIAL", "UNRESOLVED"):
        return False, f"Unknown entity_resolution value: {res!r}"
    if res != "RESOLVED" and not getattr(canonical, "entity_resolution_note", None):
        return False, (
            f"entity_resolution={res} but no entity_resolution_note — an uncertain "
            "entity must be visibly flagged, not silently shipped"
        )
    return True, None


def _check_no_untrusted_content_in_ledger(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEC-003 (BLOCKER): instruction-like / prompt-injection content must
    never appear in the evidence ledger under ANY status -- not VERIFIED, not
    REPORTED, not BELIEF. Defense-in-depth, mirroring INV-SEC-002: the
    classifier is re-run over the ledger that was actually built, so a future
    extraction path that bypasses understand_finding_node's quarantine is
    still caught here rather than trusting the upstream exclusion alone."""
    from app.services.instruction_detector import classify_instruction

    for item in state.get("evidence_ledger") or []:
        text = getattr(item, "claim", None)
        if text and classify_instruction(text).is_untrusted:
            return False, (
                f"Evidence ledger contains instruction-like/untrusted content as "
                f"{getattr(item, 'status', '?')}: {text!r}"
            )
    canonical = state.get("canonical_finding_state")
    for claim in (getattr(canonical, "evidence_claims", None) or []):
        text = getattr(claim, "text", None)
        if text and classify_instruction(text).is_untrusted:
            return False, (
                f"evidence_claims contains instruction-like/untrusted content as "
                f"{getattr(claim, 'status', '?')}: {text!r}"
            )
    return True, None


def _check_epistemic_claims_not_verified(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-EPIS-001 (BLOCKER): a proposition that is an entity's epistemic
    STANCE (belief/suspicion/assumption/opinion) or that is grammatically
    non-actual (conditional/counterfactual) must never carry evidentiary
    status VERIFIED. Structural re-check of the ledger the pipeline actually
    built, using the same generalized classifiers as the extractor."""
    from app.models.agent import EvidenceStatus
    from app.services.epistemic_modality import (
        classify_epistemic_stance,
        classify_modality,
    )

    def _violates(text: str | None, status: Any) -> str | None:
        if not text:
            return None
        if status not in (EvidenceStatus.VERIFIED, "VERIFIED"):
            return None
        if classify_epistemic_stance(text) is not None:
            return "epistemic-stance (belief/opinion) proposition"
        if not classify_modality(text).is_actual:
            return "conditional/counterfactual proposition"
        return None

    for item in state.get("evidence_ledger") or []:
        why = _violates(getattr(item, "claim", None), getattr(item, "status", None))
        if why:
            return False, f"Evidence ledger records a {why} as VERIFIED: {item.claim!r}"
    canonical = state.get("canonical_finding_state")
    for claim in (getattr(canonical, "evidence_claims", None) or []):
        why = _violates(getattr(claim, "text", None), getattr(claim, "status", None))
        if why:
            return False, f"evidence_claims records a {why} as VERIFIED: {claim.text!r}"
    return True, None


def _check_entity_not_evidence_proposition(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-003: an entity/noun-phrase field (affected_object,
    finding_subject) must never be an entire evidence-proposition SENTENCE
    ("System records show that the document-control system") -- reuses the
    same self-referential-evidence-prefix pattern the extraction firewall in
    app/services/semantic_subject.py strips at the source, as a defense-in-
    depth check on the field the renderer/downstream nodes actually
    consume."""
    from app.services.semantic_subject import (
        _ACCORDING_TO_EVIDENCE_PREFIX_RE,
        _SELF_REFERENTIAL_EVIDENCE_PREFIX_RE,
    )
    canonical = state.get("canonical_finding_state")
    impact = state.get("impact_assessment")
    fields_to_check: list[tuple[str, str]] = []
    if canonical:
        for name in ("affected_object", "finding_subject"):
            val = getattr(canonical, name, None)
            if val:
                fields_to_check.append((f"canonical.{name}", val))
    if impact and getattr(impact, "affected_object", None):
        fields_to_check.append(("impact.affected_object", impact.affected_object))
    for label, val in fields_to_check:
        if _SELF_REFERENTIAL_EVIDENCE_PREFIX_RE.search(val) or _ACCORDING_TO_EVIDENCE_PREFIX_RE.search(val):
            return False, f"{label} is an evidence proposition, not a noun-phrase entity: {val!r}"
    return True, None


_CLAUSE_MARKER_RE = re.compile(
    r"\b(?:was|were|did|does|failed|reportedly|not\s+matched?)\b", re.IGNORECASE,
)


def _check_affected_object_not_full_clause(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEMANTIC-001: affected_object must be a clean noun phrase, never
    a complete causal/action clause -- either an obvious finite-verb/
    framing marker (was/were/did/failed/reportedly/execution) or an
    implausibly long phrase (a real noun phrase for an audit finding is
    short; a whole sentence is not)."""
    canonical = state.get("canonical_finding_state")
    impact = state.get("impact_assessment")
    fields_to_check: list[tuple[str, str]] = []
    if canonical and getattr(canonical, "affected_object", None):
        fields_to_check.append(("canonical.affected_object", canonical.affected_object))
    if impact and getattr(impact, "affected_object", None):
        fields_to_check.append(("impact.affected_object", impact.affected_object))
    for label, val in fields_to_check:
        # An EXPLICIT unresolved marker is an honest statement about
        # extraction failure, not a badly-shaped entity name. It is precisely
        # what INV-SEM-002 now requires instead of a generic placeholder, so
        # the two invariants must not contradict each other.
        if str(val).strip().upper().startswith(("UNKNOWN", "UNRESOLVED")):
            continue
        if _CLAUSE_MARKER_RE.search(val) or len(val.split()) > 8:
            return False, f"{label} reads as a full clause, not a noun phrase: {val!r}"
    return True, None


def _check_comparison_finding_preserves_both_sides(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEMANTIC-002: a COMPARISON-type finding must preserve both sides
    of the comparison in the canonical deviation text, not collapse to a
    single opaque phrase."""
    canonical = state.get("canonical_finding_state")
    if not canonical or getattr(canonical, "semantic_type", None) != "COMPARISON":
        return True, None
    deviation = getattr(canonical, "observed_deviation", None) or ""
    if "did not match" not in deviation.lower() and "not matched" not in deviation.lower():
        return False, f"COMPARISON finding's observed_deviation lost the comparison relationship: {deviation!r}"
    return True, None


def _check_investigation_targets_comparison(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-010: a COMPARISON-type finding must receive comparison-
    specific investigation coverage (verifying the values/calculation/
    entry point), not only generic questions."""
    canonical = state.get("canonical_finding_state")
    if not canonical or getattr(canonical, "semantic_type", None) != "COMPARISON":
        return True, None
    inv = state.get("investigation_plan")
    if not inv or not getattr(inv, "questions", None):
        return True, None
    comparison_word_re = re.compile(r"\b(?:compar\w*|calculat\w*|recalculat\w*|reconcil\w*|discrepancy|match\w*|entry|entries)\b", re.IGNORECASE)
    if not any(comparison_word_re.search(q.question or "") for q in inv.questions):
        return False, "COMPARISON finding has no comparison/calculation-targeted investigation question"
    return True, None


def _check_five_why_no_unsupported_actor_blame(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-WHY-012: a 5-Why answer cannot introduce an unsupported actor/
    mechanism claim (e.g. "the operator may have recorded it incorrectly")
    merely because an actor happens to appear in the observation -- an
    actor-blame statement is only legitimate when the step's own status is
    VERIFIED/SUPPORTED (i.e. backed by real evidence), never at UNKNOWN/
    hedged confidence."""
    fw = state.get("five_why")
    if not fw or not getattr(fw, "steps", None):
        return True, None
    blame_re = re.compile(
        r"\b(?:operator|technician|supervisor|analyst|reviewer|employee|staff)\b(?:(?!\.).){0,40}?"
        r"\b(?:may\s+have|might\s+have|could\s+have|possibly|likely)\b(?:(?!\.).){0,40}?"
        r"\b(?:incorrectly|wrongly|by\s+mistake|in\s+error|error)\b",
        re.IGNORECASE,
    )
    for i, step in enumerate(fw.steps):
        if blame_re.search(step.answer or ""):
            return False, f"5-Why step {i+1} introduces an unsupported actor-blame hypothesis as its answer: {step.answer!r}"
    return True, None


def _check_five_why_no_unverified_hypothesis_promotion(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-5WHY-CAUSAL-001: a 5-Why answer cannot promote or present an
    UNVERIFIED candidate hypothesis as the established explanation.

    Broader and more structural than INV-WHY-012 above (which only matches
    a fixed actor+hedge+error-word regex): this reuses the same vocabulary-
    overlap check core_synthesis and final_evidence_verification already
    apply to REPAIR the answer at generation time -- present here as the
    final, authoritative gate so no code path (LLM, recovery, deterministic
    fallback edited downstream) can slip an unverified-hypothesis-selecting
    answer past validation even if the repair itself were ever bypassed."""
    fw = state.get("five_why")
    if not fw or not getattr(fw, "steps", None):
        return True, None
    rc = state.get("root_cause")
    hypotheses = getattr(rc, "candidate_hypotheses", None) if rc else None
    if not hypotheses:
        return True, None
    from app.agent.causal_guard import answer_selects_unverified_hypothesis
    for i, step in enumerate(fw.steps):
        matched = answer_selects_unverified_hypothesis(step.answer, hypotheses, status=step.status)
        if matched is not None:
            return False, (
                f"5-Why step {i+1} answer selects unverified hypothesis "
                f"{getattr(matched, 'id', '?')} ({getattr(matched, 'name', '?')}) as the explanation: {step.answer!r}"
            )
    return True, None


def _check_five_why_reported_not_promoted(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-REPORTED-002 / INV-CAUSAL-REPORTED-005: a 5-Why step whose
    content traces to a REPORTED (attributed) evidence-ledger claim must
    never be labeled VERIFIED or SUPPORTED merely because the model chose
    that status word -- statement EXISTENCE verification (the person did
    say this) must never promote PROPOSITION verification (what they said
    is true), and an attributed causal explanation must never be labeled
    SUPPORTED without independent causal evidence."""
    fw = state.get("five_why")
    if not fw or not getattr(fw, "steps", None):
        return True, None
    evidence_ledger = state.get("evidence_ledger", [])
    reported_facts = [getattr(e, "claim", getattr(e, "text", "")) for e in evidence_ledger if getattr(e, "status", None) == "REPORTED"]
    verified_facts = [getattr(e, "claim", getattr(e, "text", "")) for e in evidence_ledger if getattr(e, "status", None) == "VERIFIED"]
    from app.agent.causal_guard import answer_asserts_verified_but_is_reported
    for i, step in enumerate(fw.steps):
        if answer_asserts_verified_but_is_reported(step.answer, step.status, reported_facts, verified_facts):
            return False, f"5-Why step {i+1} labeled {step.status} but its content traces to a REPORTED (attributed) claim, not independent evidence: {step.answer!r}"
    return True, None


_CONTROL_FAILED_ASSERTION_RE = re.compile(r"\bcontrol\s+(?:was\s+)?bypassed\b|\bwas\s+not\s+executed\b|\bwas\s+circumvented\b", re.IGNORECASE)


def _check_missing_artifact_not_prove_non_occurrence(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-EVENT-002: a missing control artifact (e.g. undocumented
    justification for a transition) cannot independently prove that the
    controlled activity/transition did not occur -- the transition itself
    stays whatever the evidence establishes it to be, independent of
    whether its paperwork exists."""
    canonical = state.get("canonical_finding_state")
    if not canonical or getattr(canonical, "semantic_type", None) != "EVENT_SEQUENCE_CONTROL":
        return True, None
    for t in _recurrence_texts(state):
        if re.search(r"\b(?:transition|override|invalidation|exception|waiver)\s+(?:did\s+not\s+occur|never\s+occurred)\b", t, re.IGNORECASE):
            return False, f"Missing control artifact presented as proving the transition did not occur: {t!r}"
    return True, None


def _check_transition_not_causally_established_by_occurrence(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-EVENT-003: a controlled transition cannot be labeled causally
    established solely because the transition occurred -- root cause must
    stay NOT_ESTABLISHED absent an independently verified mechanism."""
    canonical = state.get("canonical_finding_state")
    if not canonical or getattr(canonical, "semantic_type", None) != "EVENT_SEQUENCE_CONTROL":
        return True, None
    if getattr(canonical, "mechanism_status", None) == "VERIFIED" and getattr(canonical, "verified_mechanism", None):
        return True, None
    rc = state.get("root_cause")
    if not rc:
        return True, None
    status = str(getattr(rc, "status", "")).rsplit(".", 1)[-1]
    if status in ("SUPPORTED", "ESTABLISHED"):
        return False, f"Root cause promoted to {status!r} for an EVENT_SEQUENCE_CONTROL finding with no independently verified causal mechanism"
    return True, None


def _check_control_gap_hypothesis_stays_unverified(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-EVENT-004: a control-gap hypothesis must remain
    POSSIBLE/UNVERIFIED until objective evidence establishes the
    mechanism -- never SUPPORTED/ESTABLISHED merely because the
    transition + missing justification pattern was detected."""
    rc = state.get("root_cause")
    if not rc or not getattr(rc, "candidate_hypotheses", None):
        return True, None
    for h in rc.candidate_hypotheses:
        if "CONTROL" in str(getattr(h, "name", "")).upper() and str(getattr(h, "status", "")) in ("SUPPORTED", "ESTABLISHED"):
            if str(getattr(h, "evidence_strength", "")) in ("NONE", "REPORTED", "INDICATIVE", ""):
                return False, f"Control-gap hypothesis {getattr(h, 'id', '?')} promoted to {h.status} without sufficient independent evidence strength"
    return True, None


def _check_downstream_impact_not_from_transition_alone(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-EVENT-009: downstream impact cannot be inferred solely from the
    existence of a controlled transition -- impact text must not assert
    the downstream action was improper without objective evidence."""
    canonical = state.get("canonical_finding_state")
    if not canonical or not getattr(canonical, "downstream_action_present", False):
        return True, None
    if getattr(canonical, "semantic_type", None) not in ("EVENT_SEQUENCE_CONTROL", "MISSING_RECORD"):
        return True, None
    impact = state.get("impact_assessment")
    text = getattr(impact, "potential_effect", None) or ""
    if re.search(r"\bnot\s+establish\w*\s+that\b.{0,60}?\bimproper\b", text, re.IGNORECASE):
        return True, None
    if re.search(r"\b(?:was|is)\s+improper\b|\bimproperly\s+(?:released|approved|processed|dispositioned)\b", text, re.IGNORECASE):
        return False, f"Downstream impact asserted improper solely from transition existence: {text!r}"
    return True, None


def _check_hypothesis_not_reported_only(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-REPORTED-001: a candidate hypothesis derived solely from
    an attributed (REPORTED) claim must never be labeled SUPPORTED/
    ESTABLISHED unless independent objective evidence backs it -- the
    hypothesis's own evidence_strength must not be REPORTED/NONE while its
    status claims confirmation."""
    rc = state.get("root_cause")
    if not rc or not getattr(rc, "candidate_hypotheses", None):
        return True, None
    for h in rc.candidate_hypotheses:
        status = str(getattr(h, "status", ""))
        strength = str(getattr(h, "evidence_strength", ""))
        if status in ("SUPPORTED", "ESTABLISHED") and strength in ("REPORTED", "NONE", ""):
            return False, f"Hypothesis {getattr(h, 'id', '?')} labeled {status} with only {strength or 'no'} evidence strength (attributed claim alone cannot establish it)"
    return True, None


def _check_five_why_no_unverified_modal_causation(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-5WHY-CAUSAL-002: if no verified causal mechanism exists, a
    5-Why answer cannot contain an unverified MODAL causal claim ("may
    have been...", "could have been...", "likely...", "possibly...",
    "appears to have...", "was probably..."). Independent of, and broader
    than, INV-5WHY-CAUSAL-001 above -- that check requires a candidate
    hypothesis list to compare vocabulary against; this one fires even
    with ZERO candidate hypotheses (e.g. root_cause NOT_ESTABLISHED with
    an empty candidate_hypotheses list), which is exactly the shape a
    modal-hedged invented explanation can otherwise slip through in."""
    fw = state.get("five_why")
    if not fw or not getattr(fw, "steps", None):
        return True, None
    canonical = state.get("canonical_finding_state")
    has_verified_mechanism = getattr(canonical, "mechanism_status", None) == "VERIFIED"
    from app.agent.causal_guard import five_why_answer_contains_unverified_modal_causation
    for i, step in enumerate(fw.steps):
        if five_why_answer_contains_unverified_modal_causation(step.answer, step.status, has_verified_mechanism):
            return False, f"5-Why step {i+1} answer contains an unverified modal causal claim: {step.answer!r}"
    return True, None


def _check_five_why_no_restatement_after_boundary(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-5WHY-CAUSAL-003: once the 5-Why chain reaches an UNKNOWN/evidence
    boundary, no LATER step may merely restate the finding observation, the
    previous question, or the previous answer (including a tense/inflection
    or synonymic variant) as though it were a further causal step. The
    chain must STOP at the boundary rather than pad itself with a
    restated copy of the same fact -- generalized (structural word-overlap
    checks already used elsewhere for exactly this purpose), not tied to
    any one finding's domain or wording."""
    fw = state.get("five_why")
    if not fw or not getattr(fw, "steps", None) or len(fw.steps) < 2:
        return True, None
    canonical = state.get("canonical_finding_state")
    observed_deviation = getattr(canonical, "observed_deviation", None)
    from app.agent.causal_guard import is_circular_why_answer, repeats_previous_why_answer, restates_observation
    boundary_reached = False
    prev_answer = None
    for i, step in enumerate(fw.steps):
        if boundary_reached:
            if (
                (observed_deviation and restates_observation(step.answer, observed_deviation, step.question))
                or (prev_answer and repeats_previous_why_answer(prev_answer, step.answer))
                or is_circular_why_answer(step.question, step.answer)
            ):
                return False, f"5-Why step {i+1} restates the observation/previous step after the evidence boundary: {step.answer!r}"
        if step.status in ("UNKNOWN", "REQUIRES_EVIDENCE"):
            boundary_reached = True
        prev_answer = step.answer
    return True, None


def _check_parameter_mismatch_not_calculation_routed(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-COMP-004: a PARAMETER_MISMATCH comparison finding must not be
    routed through the calculation-specific investigation tree -- its
    questions must not reference "calculation"/"formula" vocabulary that
    has no basis in a parameter mismatch (there is no formula to
    re-derive)."""
    canonical = state.get("canonical_finding_state")
    if not canonical or getattr(canonical, "comparison_subtype", None) != "PARAMETER_MISMATCH":
        return True, None
    inv = state.get("investigation_plan")
    if not inv or not getattr(inv, "questions", None):
        return True, None
    calc_word_re = re.compile(r"\b(?:calculat\w*|formula)\b", re.IGNORECASE)
    for q in inv.questions:
        if calc_word_re.search(q.question or ""):
            return False, f"PARAMETER_MISMATCH finding routed through a calculation-specific question: {q.question!r}"
    return True, None


def _check_affected_object_no_unnecessary_identifiers(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-COMP-005: for a COMPARISON finding, affected_object must not
    also carry the batch/record identifier that comparison_batch_id
    already captures as separate metadata (e.g. "Temperature setting",
    not "Temperature setting for batch BR-2026-0815"). Scoped to
    COMPARISON findings only -- an equipment/batch identifier that is the
    finding's actual SUBJECT (e.g. "Balance BAL-014", "Batch record
    BR-2201") is semantically part of the object's own name for every
    other finding shape and must not be flagged."""
    canonical = state.get("canonical_finding_state")
    if not canonical or getattr(canonical, "semantic_type", None) != "COMPARISON":
        return True, None
    affected_object = getattr(canonical, "affected_object", None) or ""
    batch_id = getattr(canonical, "comparison_batch_id", None)
    if not batch_id:
        return True, None
    bare_id = re.sub(r"^(?:batch|lot|equipment|record)\s+", "", batch_id, flags=re.IGNORECASE)
    if bare_id and bare_id in affected_object:
        return False, f"affected_object contains the batch identifier, which belongs in metadata: {affected_object!r}"
    return True, None


def _check_hypothesis_has_investigation_path(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-011: every candidate hypothesis that declares which
    investigation step tests it (resolves_investigation) must have that
    step actually present in the investigation plan -- never generate a
    hypothesis with no way to confirm or refute it. Non-blocking (WARNING):
    hypothesis-generation branches that don't yet populate
    resolves_investigation are unaffected (nothing to check)."""
    rc = state.get("root_cause")
    hypotheses = getattr(rc, "candidate_hypotheses", None) if rc else None
    if not hypotheses:
        return True, None
    inv = state.get("investigation_plan")
    question_ids = {q.id for q in getattr(inv, "questions", []) if getattr(q, "id", None)} if inv else set()
    for h in hypotheses:
        target = getattr(h, "resolves_investigation", None)
        if target and target not in question_ids:
            return False, f"Hypothesis {h.id} ({h.name}) targets investigation step {target!r}, which is not in the investigation plan"
    return True, None


def _check_evidence_artifacts_deduplicated(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-EVIDENCE-001: evidence artifacts in the investigation plan must
    be deduplicated (near-duplicate/case-insensitive) before final
    rendering -- e.g. "logs and logs and records" must never appear."""
    inv = state.get("investigation_plan")
    items = getattr(inv, "evidence_to_collect", None) if inv else None
    if not items:
        return True, None
    seen = set()
    for item in items:
        key = re.sub(r"\s+", " ", (item or "").strip().lower())
        if key and key in seen:
            return False, f"Evidence artifact list contains an exact duplicate: {item!r}"
        seen.add(key)
    from app.agent.analytical_validator import normalize_and_dedupe_evidence_items
    deduped = normalize_and_dedupe_evidence_items(items, max_items=len(items))
    if len(deduped) < len(items) * 0.5 and len(items) > 2:
        return False, f"Evidence artifact list has substantial unresolved near-duplication: {items!r}"
    return True, None


_NON_PERFORMANCE_ASSERTION_RE = re.compile(
    r"\b(?:was\s+not\s+performed|were\s+not\s+performed|non[-\s]?performance|"
    r"activity\s+did\s+not\s+occur|was\s+not\s+carried\s+out)\b",
    re.IGNORECASE,
)
_CONTROL_FAILURE_ASSERTION_RE = re.compile(
    r"\b(?:control\s+fail(?:ed|ure)|review\s+fail(?:ed|ure)|authorization\s+fail(?:ed|ure))\b",
    re.IGNORECASE,
)


def _check_missing_evidence_not_non_performance(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-MISSING-001: for a MISSING_RECORD finding, missing evidence
    alone must never be presented as though it establishes that the
    underlying activity was NOT performed -- only objective evidence
    (a VERIFIED claim stating non-performance) can license that language."""
    canonical = state.get("canonical_finding_state")
    if not canonical or getattr(canonical, "semantic_type", None) != "MISSING_RECORD":
        return True, None
    verified_facts = [e.claim for e in state.get("evidence_ledger", []) if getattr(e, "status", None) == "VERIFIED"]
    has_verified_non_performance = any(_NON_PERFORMANCE_ASSERTION_RE.search(f) for f in verified_facts)
    if has_verified_non_performance:
        return True, None
    rc = state.get("root_cause")
    fw = state.get("five_why")
    texts = []
    if rc:
        texts.append(getattr(rc, "narrative", None) or "")
        for h in getattr(rc, "candidate_hypotheses", None) or []:
            texts.append(getattr(h, "statement", None) or "")
    if fw:
        for step in getattr(fw, "steps", None) or []:
            texts.append(step.answer or "")
    for t in texts:
        if _NON_PERFORMANCE_ASSERTION_RE.search(t):
            return False, f"Missing evidence presented as establishing non-performance without objective evidence: {t!r}"
    return True, None


def _check_missing_evidence_not_root_cause(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-MISSING-002: missing evidence cannot independently establish
    root cause -- a MISSING_RECORD finding with no other verified causal
    mechanism must keep root_cause.status NOT_ESTABLISHED."""
    canonical = state.get("canonical_finding_state")
    if not canonical or getattr(canonical, "semantic_type", None) != "MISSING_RECORD":
        return True, None
    if getattr(canonical, "mechanism_status", None) == "VERIFIED" and getattr(canonical, "verified_mechanism", None):
        return True, None
    rc = state.get("root_cause")
    if not rc:
        return True, None
    status = str(getattr(rc, "status", "")).rsplit(".", 1)[-1]
    if status in ("SUPPORTED", "ESTABLISHED"):
        return False, f"Root cause promoted to {status!r} from a MISSING_RECORD finding with no independently verified causal mechanism"
    return True, None


def _check_downstream_action_not_control_failure(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-MISSING-003: a downstream action's mere presence cannot
    independently establish that a control failed -- that requires
    objective evidence, not the co-occurrence of a missing record and a
    later action."""
    canonical = state.get("canonical_finding_state")
    if not canonical or not getattr(canonical, "downstream_action_present", False):
        return True, None
    verified_facts = [e.claim for e in state.get("evidence_ledger", []) if getattr(e, "status", None) == "VERIFIED"]
    if any(_CONTROL_FAILURE_ASSERTION_RE.search(f) for f in verified_facts):
        return True, None
    impact = state.get("impact_assessment")
    text = getattr(impact, "potential_effect", None) or ""
    if _CONTROL_FAILURE_ASSERTION_RE.search(text):
        return False, f"Downstream action presence used to assert control failure without objective evidence: {text!r}"
    return True, None


def _check_investigation_distinguishes_performance_from_documentation(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-MISSING-005: for a MISSING_RECORD finding, investigation
    questions must include coverage that distinguishes activity
    PERFORMANCE verification from documentation/record verification --
    never only ask about the record."""
    canonical = state.get("canonical_finding_state")
    if not canonical or getattr(canonical, "semantic_type", None) != "MISSING_RECORD":
        return True, None
    inv = state.get("investigation_plan")
    if not inv or not getattr(inv, "questions", None):
        return True, None
    performance_re = re.compile(r"\b(?:perform\w*|execut\w*|carried\s+out|actually\s+(?:done|occur\w*))\b", re.IGNORECASE)
    documentation_re = re.compile(r"\b(?:record\w*|document\w*|log\w*|evidence)\b", re.IGNORECASE)
    q_texts = [q.question or "" for q in inv.questions]
    if not any(performance_re.search(t) for t in q_texts):
        return False, "MISSING_RECORD investigation plan has no question verifying whether the activity was actually performed"
    if not any(documentation_re.search(t) for t in q_texts):
        return False, "MISSING_RECORD investigation plan has no question verifying the documentation/record itself"
    return True, None


def _check_downstream_investigation_present(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-MISSING-006: when a downstream action is objectively present,
    the investigation plan must include downstream-control coverage
    (precondition/review/authorization), not only activity/documentation
    questions."""
    canonical = state.get("canonical_finding_state")
    if not canonical or not getattr(canonical, "downstream_action_present", False):
        return True, None
    inv = state.get("investigation_plan")
    if not inv or not getattr(inv, "questions", None):
        return True, None
    downstream_re = re.compile(r"\bdownstream\s+action\b", re.IGNORECASE)
    if not any(downstream_re.search(q.question or "") for q in inv.questions):
        return False, "Downstream action is present but the investigation plan has no downstream-control question"
    return True, None


def _check_investigation_areas_not_restate_object(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-MISSING-007: investigation AREAS must represent investigation
    domains, not simply restate the affected object (e.g. "Verify
    compliance and control records for X" where X is just the finding's
    own affected_object)."""
    canonical = state.get("canonical_finding_state")
    inv = state.get("investigation_plan")
    if not canonical or not inv or not getattr(inv, "areas", None):
        return True, None
    affected_object = (getattr(canonical, "affected_object", None) or "").strip().lower()
    if not affected_object or affected_object == "unknown":
        return True, None
    from app.services.text_grounding import significant_words
    obj_words = significant_words(affected_object)
    if not obj_words:
        return True, None
    for area in inv.areas:
        area_words = significant_words(area or "")
        if not area_words:
            continue
        # An area that is EXACTLY the affected object's own vocabulary
        # (nothing else at all) restates the object instead of naming a
        # domain -- e.g. area text "X" where X literally is affected_object.
        # Conservative on purpose: an area that adds even one real domain
        # word ("...status and authorization", "...control effectiveness")
        # is a legitimate, more specific area name, not a restatement.
        if obj_words == area_words:
            return False, f"Investigation area restates the affected object rather than naming a domain: {area!r}"
    return True, None


_CAPA_EFFECTIVE_ASSERTION_RE = re.compile(r"\b(?:was|is|proved|remained)\s+effective\b", re.IGNORECASE)
_CAPA_INEFFECTIVE_ASSERTION_RE = re.compile(r"\bwas\s+ineffective\b|\bfailed\s+to\s+remain\s+effective\b|\bprior\s+action\s+failed\b", re.IGNORECASE)
_SYSTEMIC_ASSERTION_RE = re.compile(r"\bis\s+systemic\b|\bsystemic\s+(?:issue|failure|problem)\b", re.IGNORECASE)


def _recurrence_texts(state: dict[str, Any]) -> list[str]:
    texts = []
    rc = state.get("root_cause")
    if rc:
        texts.append(getattr(rc, "narrative", None) or "")
        for h in getattr(rc, "candidate_hypotheses", None) or []:
            texts.append(getattr(h, "statement", None) or "")
    fw = state.get("five_why")
    if fw:
        for step in getattr(fw, "steps", None) or []:
            texts.append(step.answer or "")
    impact = state.get("impact_assessment")
    if impact:
        texts.append(getattr(impact, "potential_effect", None) or "")
    return texts


def _check_verified_recurrence_not_denied(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-REC-001: verified recurrence cannot be represented as
    NOT_EVIDENCED elsewhere in the output."""
    canonical = state.get("canonical_finding_state")
    if not canonical or not getattr(canonical, "recurrence_signal", False):
        return True, None
    for t in _recurrence_texts(state):
        if re.search(r"\brecurrence\s+(?:is|has)\s+not\s+(?:been\s+)?(?:evidenced|established|confirmed)\b", t, re.IGNORECASE):
            return False, f"Recurrence is structurally evidenced but described as not evidenced: {t!r}"
    return True, None


def _check_recurrence_not_establish_future_risk(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-REC-002: recurrence evidence cannot independently establish
    future recurrence risk -- a HIGH risk_of_recurrence rating must carry
    its own rationale, not merely cite that recurrence already occurred."""
    rc = state.get("root_cause")
    if not rc or getattr(rc, "risk_of_recurrence", None) != "HIGH":
        return True, None
    rationale = (getattr(rc, "risk_of_recurrence_rationale", None) or "").strip()
    if not rationale:
        return False, "risk_of_recurrence=HIGH with no supporting rationale"
    return True, None


def _check_capa_completion_not_effectiveness(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-REC-003: prior CAPA completion cannot establish CAPA
    effectiveness -- COMPLETED status with unverified effectiveness must
    never be described as "effective"."""
    canonical = state.get("canonical_finding_state")
    if not canonical or getattr(canonical, "previous_capa_status", None) != "COMPLETED":
        return True, None
    if getattr(canonical, "previous_capa_effectiveness", None) == "EFFECTIVE":
        return True, None
    from app.agent.causal_guard import _NEGATED_CAUSAL_HEDGE_RE
    for t in _recurrence_texts(state):
        if _NEGATED_CAUSAL_HEDGE_RE.search(t):
            continue
        if _CAPA_EFFECTIVE_ASSERTION_RE.search(t):
            return False, f"Prior CAPA completion described as effectiveness without verified effectiveness evidence: {t!r}"
    return True, None


def _check_missing_effectiveness_not_ineffectiveness(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-REC-004: missing effectiveness evidence cannot establish CAPA
    ineffectiveness."""
    canonical = state.get("canonical_finding_state")
    if not canonical or getattr(canonical, "previous_capa_effectiveness", None) not in (None, "NOT_VERIFIED"):
        return True, None
    from app.agent.causal_guard import _NEGATED_CAUSAL_HEDGE_RE
    for t in _recurrence_texts(state):
        if _NEGATED_CAUSAL_HEDGE_RE.search(t):
            continue
        if _CAPA_INEFFECTIVE_ASSERTION_RE.search(t):
            return False, f"Missing effectiveness evidence presented as establishing ineffectiveness: {t!r}"
    return True, None


def _check_recurrence_not_establish_capa_causal_failure(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-REC-005: recurrence cannot independently establish that a prior
    CAPA causally failed -- root_cause must not be promoted to SUPPORTED/
    ESTABLISHED purely from a recurrence + prior-action co-occurrence with
    no independently verified causal mechanism."""
    canonical = state.get("canonical_finding_state")
    if not canonical or not (getattr(canonical, "recurrence_signal", False) and getattr(canonical, "previous_capa_referenced", False)):
        return True, None
    if getattr(canonical, "mechanism_status", None) == "VERIFIED" and getattr(canonical, "verified_mechanism", None):
        return True, None
    rc = state.get("root_cause")
    if not rc:
        return True, None
    status = str(getattr(rc, "status", "")).rsplit(".", 1)[-1]
    if status in ("SUPPORTED", "ESTABLISHED"):
        return False, f"Root cause promoted to {status!r} from recurrence + prior-action co-occurrence with no independently verified causal mechanism"
    return True, None


def _check_multiple_occurrences_not_auto_systemic(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-REC-010: multiple occurrences must not automatically be labeled
    SYSTEMIC without population/scope evidence."""
    canonical = state.get("canonical_finding_state")
    if not canonical or not getattr(canonical, "recurrence_signal", False):
        return True, None
    if getattr(canonical, "occurrence_population", None):
        return True, None
    for t in _recurrence_texts(state):
        if _SYSTEMIC_ASSERTION_RE.search(t):
            return False, f"Recurrence labeled systemic without population/scope evidence: {t!r}"
    return True, None


def _check_recurrence_investigation_coverage(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-REC-006: when recurrence AND a prior action are both
    structurally established, the investigation plan must include
    prior-action-relationship coverage (implementation/effectiveness/scope
    of the prior action), not only generic questions."""
    canonical = state.get("canonical_finding_state")
    if not canonical or not (getattr(canonical, "recurrence_signal", False) and getattr(canonical, "previous_capa_referenced", False)):
        return True, None
    inv = state.get("investigation_plan")
    if not inv or not getattr(inv, "questions", None):
        return True, None
    prior_action_re = re.compile(r"\b(?:previous|prior)\b.{0,30}?\b(?:capa|corrective\s+action)\b|\beffectiveness\b|\bimplement\w*\b", re.IGNORECASE)
    if not any(prior_action_re.search(q.question or "") for q in inv.questions):
        return False, "Recurrence + prior action finding has no prior-action-relationship investigation question"
    return True, None


def _check_investigation_plan_has_evidence(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-003: every investigation question must name the evidence
    to examine -- a question with no evidence/evidence_required target is
    structurally incomplete (no way to actually resolve it)."""
    inv = state.get("investigation_plan")
    if not inv or not getattr(inv, "questions", None):
        return True, None
    for q in inv.questions:
        if not (getattr(q, "evidence", None) or getattr(q, "evidence_required", None)):
            return False, f"Investigation question {q.question!r} has no evidence to examine"
    return True, None


def _check_investigation_question_traceability(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-012 (Phase 14 Section 18): a graph-grounded investigation
    question's target_node_id/target_edge_id/source_node_id must refer to a
    real node/edge that actually exists in this run's causal-uncertainty
    graph or causal graph -- never a dangling or fabricated reference. Only
    applies to questions that actually populate these fields (the honest
    "came from the causal-graph path" signal documented on
    InvestigationQuestion); a question with none of them set is not held to
    this bar, it is simply not claiming graph grounding."""
    inv = state.get("investigation_plan")
    if not inv or not getattr(inv, "questions", None):
        return True, None

    node_ids: set[str] = set()
    edge_ids: set[str] = set()
    for graph in (state.get("causal_uncertainty_graph"), state.get("causal_graph")):
        if graph is None:
            continue
        node_ids.update(n.node_id for n in getattr(graph, "nodes", None) or [])
        edge_ids.update(e.edge_id for e in getattr(graph, "edges", None) or [])

    for q in inv.questions:
        claims_graph_grounding = bool(
            getattr(q, "target_node_id", None) or getattr(q, "target_edge_id", None) or getattr(q, "source_node_id", None)
        )
        if not claims_graph_grounding:
            continue
        if not node_ids and not edge_ids:
            # This run never built any causal-uncertainty/causal graph at
            # all, yet a question claims a graph-grounded reference -- the
            # reference cannot be real.
            return False, (
                f"Investigation question {q.question!r} claims causal-graph grounding "
                "(target_node_id/target_edge_id/source_node_id set) but no causal graph exists in this run"
            )
        for field_name in ("target_node_id", "source_node_id"):
            ref = getattr(q, field_name, None)
            if ref and ref not in node_ids:
                return False, (
                    f"Investigation question {q.question!r} has {field_name}={ref!r}, "
                    "which does not exist in this run's causal graph — dangling reference"
                )
        ref = getattr(q, "target_edge_id", None)
        if ref and ref not in edge_ids:
            return False, (
                f"Investigation question {q.question!r} has target_edge_id={ref!r}, "
                "which does not exist in this run's causal graph — dangling reference"
            )
    return True, None


def _check_investigation_plan_status_consistency(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-013 (Phase 15 Section 9): InvestigationPlan.status is a
    machine-readable claim about the plan's contents -- NO_ACTIONABLE_
    UNCERTAINTY asserts there is nothing to investigate, which is only true
    if the plan genuinely carries no questions. A status/content mismatch
    (in either direction) means the claim cannot be trusted and must fail
    closed rather than let a caller believe an unverified claim."""
    inv = state.get("investigation_plan")
    if not inv:
        return True, None
    status = getattr(inv, "status", "QUESTIONS_GENERATED")
    has_questions = bool(getattr(inv, "questions", None))
    # Only the direction this phase actually produces is enforced: a plan
    # that explicitly CLAIMS no actionable uncertainty must not also carry
    # questions. The converse (QUESTIONS_GENERATED with zero questions) is
    # deliberately NOT flagged here -- "QUESTIONS_GENERATED" is this field's
    # default value, and countless existing unit tests across this suite
    # manually construct a bare InvestigationPlan() for unrelated
    # assertions; treating that default as an affirmative claim would turn
    # this into a blanket "every hand-built fixture must populate
    # questions" rule, which is not what this invariant is for.
    if status == "NO_ACTIONABLE_UNCERTAINTY" and has_questions:
        return False, "InvestigationPlan.status=NO_ACTIONABLE_UNCERTAINTY but questions were generated"
    return True, None


def _check_no_actionable_uncertainty_matches_graph(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-019 (Phase 16 Section 27): a plan claiming
    NO_ACTIONABLE_UNCERTAINTY must actually be consistent with this run's
    causal_uncertainty_graph -- if the graph itself still carries an
    UNRESOLVED node, the claim that there is nothing left to investigate is
    false and must fail closed rather than be trusted."""
    inv = state.get("investigation_plan")
    if not inv:
        return True, None
    is_no_actionable = getattr(inv, "status", None) == "NO_ACTIONABLE_UNCERTAINTY" or getattr(inv, "planner_mode", None) == "NO_ACTIONABLE_UNCERTAINTY"
    if not is_no_actionable:
        return True, None
    ug = state.get("causal_uncertainty_graph")
    if ug is None:
        return True, None
    unresolved = [n for n in getattr(ug, "nodes", None) or [] if str(getattr(n.node_type, "value", n.node_type)) == "UNRESOLVED"]
    if unresolved:
        return False, (
            f"InvestigationPlan claims NO_ACTIONABLE_UNCERTAINTY but the causal_uncertainty_graph "
            f"still carries {len(unresolved)} unresolved node(s): {[n.node_id for n in unresolved]}"
        )
    return True, None


def _check_stage_b_question_has_target(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-021 (Phase 17): every Stage-B question must resolve to a
    real causal/semantic target -- target_node_id or target_hypothesis_ids
    must be populated (Stage B never emits a generic question with
    neither)."""
    plan = state.get("causal_investigation_plan")
    if not plan or not getattr(plan, "questions", None):
        return True, None
    for q in plan.questions:
        if getattr(q, "planner_stage", None) != "STAGE_B":
            continue
        if not (getattr(q, "target_node_id", None) or getattr(q, "target_hypothesis_ids", None)):
            return False, f"Stage-B question {q.question!r} has no causal/semantic target"
    return True, None


def _check_stage_b_question_has_evidence_target(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-022 (Phase 17): every Stage-B question must declare what
    evidence would resolve it."""
    plan = state.get("causal_investigation_plan")
    if not plan or not getattr(plan, "questions", None):
        return True, None
    for q in plan.questions:
        if getattr(q, "planner_stage", None) != "STAGE_B":
            continue
        if not (getattr(q, "evidence", None) or getattr(q, "evidence_required", None)):
            return False, f"Stage-B question {q.question!r} has no evidence-resolution target"
    return True, None


def _check_stage_b_preserves_competing_hypotheses(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-023 (Phase 17): when root_cause carries 2+ hypotheses
    that are neither SUPPORTED nor REFUTED (live, unresolved), Stage B's
    plan must not silently drop any of them from its questions' combined
    target_hypothesis_ids -- either they are all represented, or the plan
    is empty (a genuine NO_ACTIONABLE_UNCERTAINTY judgment), never a
    partial subset standing in as "the" answer."""
    plan = state.get("causal_investigation_plan")
    rc = state.get("root_cause")
    if not plan or not rc:
        return True, None
    if getattr(plan, "planner_stage", None) != "STAGE_B":
        return True, None
    live_ids = {
        str(getattr(h, "id", ""))
        for h in getattr(rc, "candidate_hypotheses", None) or []
        if str(getattr(h, "status", "")) not in ("SUPPORTED", "REFUTED")
    }
    if len(live_ids) < 2:
        return True, None
    if not plan.questions:
        # A genuine "nothing actionable" judgment is legitimate even with
        # live hypotheses only if planner_mode says so explicitly.
        if getattr(plan, "planner_mode", None) == "NO_ACTIONABLE_UNCERTAINTY":
            return True, None
        return False, f"Stage B has {len(live_ids)} live competing hypotheses but produced zero questions"
    covered: set[str] = set()
    for q in plan.questions:
        covered.update(str(x) for x in (getattr(q, "target_hypothesis_ids", None) or []))
    missing = live_ids - covered
    if missing:
        return False, f"Stage B silently dropped competing hypothesis id(s) {sorted(missing)} from its questions"
    return True, None


def _check_investigation_history_monotonic(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-024 (Phase 18 Section 6/29): causal_graph_version and
    iteration_id in investigation_history must be strictly monotonically
    non-decreasing -- a version going backwards means graph provenance was
    lost or the history was corrupted, and must fail closed."""
    history = state.get("investigation_history")
    if not history:
        return True, None
    prev_version = -1
    prev_iter = -1
    for rec in history:
        version = getattr(rec, "causal_graph_version", None)
        it = getattr(rec, "iteration_id", None)
        if version is None or it is None:
            return False, "investigation_history record missing causal_graph_version/iteration_id"
        if version <= prev_version:
            return False, f"causal_graph_version did not increase (iteration {it}: {version} <= {prev_version})"
        if it < prev_iter:
            return False, f"investigation iteration_id went backwards ({it} < {prev_iter})"
        prev_version, prev_iter = version, it
    return True, None


def _check_no_actionable_uncertainty_consistent_with_history(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-025 (Phase 18 Section 16/29): NO_ACTIONABLE_UNCERTAINTY
    must not be emitted by Stage B while its own iteration record still
    lists unresolved targets -- the two fields on the SAME structural
    output must agree, or the claim is untrustworthy."""
    plan = state.get("causal_investigation_plan")
    history = state.get("investigation_history")
    if not plan or not history:
        return True, None
    if getattr(plan, "planner_mode", None) != "NO_ACTIONABLE_UNCERTAINTY":
        return True, None
    last = history[-1]
    unresolved_after = getattr(last, "unresolved_targets_after", None) or []
    if unresolved_after:
        return False, (
            f"Stage B emitted NO_ACTIONABLE_UNCERTAINTY but its own iteration record still lists "
            f"unresolved targets: {unresolved_after}"
        )
    return True, None


def _check_evidence_has_provenance(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-026 (Phase 19 Section 28): any EvidenceItem acquired
    through the Phase 19 adaptive loop (identified by having an
    evidence_id set at all -- pre-Phase-19 evidence never populates that
    field) must carry a request_id linking it back to the EvidenceRequest
    that produced it. An evidence item with an id but no request
    provenance is untraceable and must fail closed."""
    ledger = state.get("evidence_ledger")
    if not ledger:
        return True, None
    for item in ledger:
        evidence_id = getattr(item, "evidence_id", None)
        if not evidence_id:
            continue  # not a Phase-19-acquired item; not held to this bar
        if not getattr(item, "request_id", None):
            return False, f"EvidenceItem {evidence_id!r} has no request_id — missing acquisition provenance"
    return True, None


def _check_hypothesis_history_references_real_evidence(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-027 (Phase 19 Section 28): every evidence_id a
    HypothesisStatusChange cites as having caused the transition must
    actually exist in the evidence_ledger — a status change cannot be
    justified by a reference to evidence that isn't there."""
    history = state.get("hypothesis_history")
    if not history:
        return True, None
    ledger_ids = {getattr(e, "evidence_id", None) for e in (state.get("evidence_ledger") or [])}
    ledger_ids.discard(None)
    for change in history:
        for eid in getattr(change, "changed_by_evidence_ids", None) or []:
            if eid not in ledger_ids:
                return False, (
                    f"HypothesisStatusChange for {change.hypothesis_id} cites evidence_id {eid!r}, "
                    "which does not exist in evidence_ledger"
                )
    return True, None


def _check_single_epistemic_authority(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-028 (Phase 20 Part B/C): a hypothesis's CURRENT status
    must match the LAST status the authoritative evidence-reconciliation
    evaluator (app.agent.nodes.evidence_acquisition.
    reconcile_hypothesis_from_evidence, recorded in hypothesis_history)
    actually decided for it, whenever both exist. A mismatch means some
    OTHER code path silently overwrote the authoritative decision after
    the fact -- exactly the "second epistemic authority" defect this
    invariant exists to catch."""
    history = state.get("hypothesis_history")
    rc = state.get("root_cause")
    if not history or not rc:
        return True, None
    last_by_id: dict[str, str] = {}
    for change in history:
        last_by_id[change.hypothesis_id] = change.new_status
    for h in getattr(rc, "candidate_hypotheses", None) or []:
        hid = str(getattr(h, "id", ""))
        expected = last_by_id.get(hid)
        if expected is not None and str(getattr(h, "status", "")) != expected:
            return False, (
                f"Hypothesis {hid} status is {getattr(h, 'status', None)!r} but the authoritative evaluator "
                f"last decided {expected!r} — a second writer silently overrode the epistemic authority"
            )
    return True, None


def _check_evidence_claims_have_provenance(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-029 (Phase 21 Part C/Z): every EvidenceClaim in
    evidence_claims must cite at least one real evidence_ledger entry — a
    claim with no provenance, or a dangling evidence_id, must never reach
    runtime state (app.agent.evidence_interpreter fails closed and drops
    such claims before they are appended; this invariant is the checkable
    guarantee that no other code path can reintroduce one)."""
    claims = state.get("evidence_claims")
    if not claims:
        return True, None
    ledger_ids = {getattr(e, "evidence_id", None) for e in (state.get("evidence_ledger") or [])}
    ledger_ids.discard(None)
    for claim in claims:
        evidence_ids = getattr(claim, "evidence_ids", None) or []
        if not evidence_ids:
            return False, f"EvidenceClaim {getattr(claim, 'claim_id', None)!r} has no evidence_ids — missing provenance"
        for eid in evidence_ids:
            if eid not in ledger_ids:
                return False, (
                    f"EvidenceClaim {getattr(claim, 'claim_id', None)!r} cites evidence_id {eid!r}, "
                    "which does not exist in evidence_ledger"
                )
    return True, None


def _check_impact_never_verified_without_verified_evidence(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-IMPACT-002 (Phase 22 Part I): ImpactAssessment.status ==
    IMPACT_VERIFIED must never appear unless at least one VERIFIED item
    exists in the evidence ledger to ground it -- a forward-looking guard
    against any future code path asserting impact is proven merely because
    a hypothesis exists or an LLM narrative says so (this codebase
    currently never assigns IMPACT_VERIFIED at all; this invariant keeps
    it that way unless real objective evidence backs it)."""
    impact = state.get("impact_assessment")
    if not impact or getattr(impact, "status", None) != "IMPACT_VERIFIED":
        return True, None
    evidence_ledger = state.get("evidence_ledger") or []
    # Phase 24: exact equality, not `str(x).endswith("VERIFIED")` -- that
    # substring check also matches "EvidenceStatus.UNVERIFIED"
    # ("UNVERIFIED".endswith("VERIFIED") is True in Python), which would
    # have let UNVERIFIED evidence satisfy this "has real proof" check.
    has_verified = any(getattr(e, "status", None) == "VERIFIED" for e in evidence_ledger)
    if not has_verified:
        return False, "ImpactAssessment.status is IMPACT_VERIFIED but no VERIFIED evidence exists to ground it"
    return True, None


def _check_capa_excludes_refuted_hypotheses(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAPA-001 (Phase 22 Part G): a CAPA conditional action must never
    be conditioned on ("IF ... is confirmed") a hypothesis the authoritative
    evaluator has already REFUTED -- that question is already closed, not
    still open for confirmation."""
    capa = state.get("capa_analysis")
    rc = state.get("root_cause")
    if not capa or not rc:
        return True, None
    refuted_ids = {
        str(getattr(h, "id", "")) for h in (getattr(rc, "candidate_hypotheses", None) or [])
        if str(getattr(h, "status", "")) == "REFUTED"
    }
    if not refuted_ids:
        return True, None
    for action in getattr(capa, "conditional_actions", None) or []:
        hid = getattr(action, "root_cause_hypothesis_id", None)
        if hid and hid in refuted_ids:
            return False, f"CAPA conditional action is conditioned on {hid!r}, which is REFUTED"
    return True, None


def _check_impact_fields_normalized(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-IMPACT-001: impact fields (affected_object, process_at_risk,
    control_at_risk) must use normalized entities, never a raw evidence
    proposition sentence."""
    from app.services.semantic_subject import (
        _ACCORDING_TO_EVIDENCE_PREFIX_RE,
        _SELF_REFERENTIAL_EVIDENCE_PREFIX_RE,
    )
    impact = state.get("impact_assessment")
    if not impact:
        return True, None
    for field_name in ("affected_object", "process_at_risk", "control_at_risk"):
        val = getattr(impact, field_name, None)
        if val and (_SELF_REFERENTIAL_EVIDENCE_PREFIX_RE.search(val) or _ACCORDING_TO_EVIDENCE_PREFIX_RE.search(val)):
            return False, f"impact.{field_name} is an evidence proposition, not a normalized entity: {val!r}"
    return True, None


def _check_security_isolation(state: dict[str, Any]) -> tuple[bool, str | None]:
    canonical = state.get("canonical_finding_state")
    if canonical and getattr(canonical, "referenced_documents", None):
        unavail = [
            getattr(d, "document_type", "").lower()
            for d in canonical.referenced_documents
            if getattr(d, "reference_status", "") == "REFERENCED_UNAVAILABLE"
        ]
        rc = state.get("root_cause")
        if rc and getattr(rc, "candidate_hypotheses", None):
            for h in rc.candidate_hypotheses:
                stmt_low = h.statement.lower()
                for ut in unavail:
                    if ut and ut in stmt_low and re.search(r"\b(showed|indicated|proved|contained|recorded|stated)\b", stmt_low):
                        return False, f"Hypothesis {h.id} infers contents of unavailable document '{ut}'"
    return True, None


def _check_no_injection_in_causal_output(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEC-002: an instruction-like/prompt-injection proposition must
    never surface as causal support anywhere in the final output --
    root cause, hypotheses, 5-Why, CAPA, or impact. Defense-in-depth: the
    evidence-ledger population in understand_finding_node already excludes
    this content at the source (Section 3), so this should never actually
    fire in practice; it exists as a final backstop in case a future
    extraction path (e.g. a live-LLM path) reintroduces it."""
    from app.services.instruction_detector import classify_instruction

    def _flagged(text: str | None) -> bool:
        return bool(text) and classify_instruction(text).is_untrusted

    rc = state.get("root_cause")
    if rc:
        if _flagged(getattr(rc, "narrative", None)):
            return False, "root_cause.narrative contains instruction-like/untrusted content"
        for h in getattr(rc, "candidate_hypotheses", None) or []:
            if _flagged(h.statement):
                return False, f"Hypothesis {h.id} statement contains instruction-like/untrusted content"

    fw = state.get("five_why")
    if fw:
        for i, step in enumerate(getattr(fw, "steps", None) or []):
            if _flagged(step.answer):
                return False, f"5-Why step {i+1} answer contains instruction-like/untrusted content"

    capa = state.get("capa_analysis")
    if capa:
        for a in getattr(capa, "conditional_actions", None) or []:
            if _flagged(getattr(a, "recommended_action", None)) or _flagged(getattr(a, "if_cause_confirmed", None)):
                return False, "CAPA conditional action contains instruction-like/untrusted content"

    impact = state.get("impact_assessment")
    if impact and _flagged(getattr(impact, "potential_effect", None)):
        return False, "impact.potential_effect contains instruction-like/untrusted content"

    return True, None


def _check_five_why_no_hedging(state: dict[str, Any]) -> tuple[bool, str | None]:
    fw = state.get("five_why")
    if not fw or not getattr(fw, "steps", None):
        return True, None
    hedge_re = re.compile(r"\b(?:may\s+have|could\s+have|likely\s+caused|probably\s+caused|appears\s+to\s+have\s+caused)\b", re.IGNORECASE)
    for i, step in enumerate(fw.steps):
        ans = step.answer or ""
        if hedge_re.search(ans) and step.status not in ("UNKNOWN", "NOT_ESTABLISHED"):
            return False, f"5-Why step {i+1} asserts speculative causal mechanism using hedge language"
    return True, None


_CAUSAL_STEP_STATUSES = ("VERIFIED", "SUPPORTED")


def _check_five_why_evidence_absence_not_causal(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-WHY-006 / INV-WHY-009: a claim that itself states causal
    information is UNAVAILABLE ("no information is available about why...")
    can never be cited as VERIFIED/SUPPORTED causal support for a Why
    answer -- that is the exact reported defect (an evidence-absence claim
    being used as if it explained the mechanism it says is unknown)."""
    fw = state.get("five_why")
    if not fw or not getattr(fw, "steps", None):
        return True, None
    from app.agent.causal_guard import is_evidence_absence_claim
    for i, step in enumerate(fw.steps):
        if step.status in _CAUSAL_STEP_STATUSES and is_evidence_absence_claim(step.answer):
            return False, (
                f"5-Why step {i+1} is marked {step.status} but its answer is itself an "
                "evidence-absence statement, not causal evidence"
            )
    return True, None


def _check_five_why_not_padded_past_boundary(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-WHY-007: 5-Why depth is evidence-driven, not fixed at five -- if
    the chain hits the evidence boundary (UNKNOWN) on the FIRST step, no
    further steps may be appended just to fill out a template length."""
    fw = state.get("five_why")
    if not fw or not getattr(fw, "steps", None) or len(fw.steps) < 2:
        return True, None
    if fw.steps[0].status in ("UNKNOWN", "NOT_ESTABLISHED") and all(
        s.status in ("UNKNOWN", "NOT_ESTABLISHED") for s in fw.steps
    ):
        return False, (
            f"5-Why chain hit the evidence boundary on step 1 but was padded to {len(fw.steps)} "
            "UNKNOWN steps instead of stopping"
        )
    return True, None


def _check_five_why_no_progression_past_unknown(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-WHY-011: once a step in the chain reaches UNKNOWN (the evidence
    boundary), no LATER step may resume claiming VERIFIED/SUPPORTED -- the
    chain cannot "un-stall" without new independent evidence establishing
    the next mechanism, which this deterministic pipeline never fabricates."""
    fw = state.get("five_why")
    if not fw or not getattr(fw, "steps", None):
        return True, None
    hit_unknown = False
    for i, step in enumerate(fw.steps):
        if hit_unknown and step.status in _CAUSAL_STEP_STATUSES:
            return False, f"5-Why step {i+1} claims {step.status} after an earlier UNKNOWN step in the same chain"
        if step.status in ("UNKNOWN", "NOT_ESTABLISHED"):
            hit_unknown = True
    return True, None


def _check_detection_not_root_cause(state: dict[str, Any]) -> tuple[bool, str | None]:
    rc = state.get("root_cause")
    if not rc or not getattr(rc, "candidate_hypotheses", None):
        return True, None
    if rc.status in (RootCauseStatus.ESTABLISHED, RootCauseStatus.SUPPORTED) and rc.leading_hypothesis:
        for h in rc.candidate_hypotheses:
            if h.id == rc.leading_hypothesis and getattr(h, "causal_role", "") == "DETECTION_FAILURE":
                return False, f"Detection failure hypothesis {h.id} was erroneously promoted as primary root cause"
    return True, None


def _check_affected_period(state: dict[str, Any]) -> tuple[bool, str | None]:
    canonical = state.get("canonical_finding_state")
    impact = state.get("impact_assessment")
    if canonical and getattr(canonical, "timeframe", None):
        if canonical.timeframe and "during the audit" in str(canonical.timeframe).lower():
            return False, "Finding detection timeframe 'During the audit' was erroneously recorded as deviation timeframe"
    if impact and getattr(impact, "affected_period", None):
        if impact.affected_period and "during the audit" in str(impact.affected_period).lower():
            return False, "Finding detection timeframe 'During the audit' was erroneously recorded as affected_period"
    return True, None


def _check_rendering_and_malformed_amounts(state: dict[str, Any]) -> tuple[bool, str | None]:
    impact = state.get("impact_assessment")
    rc = state.get("root_cause")
    fw = state.get("five_why")
    
    texts_to_check: list[str] = []
    if impact:
        for val in (impact.potential_effect, impact.affected_object, impact.process_at_risk, impact.narrative):
            if val:
                texts_to_check.append(str(val))
    if rc:
        for val in (rc.statement, rc.narrative, rc.root_cause_basis):
            if val:
                texts_to_check.append(str(val))
    if fw and fw.steps:
        for s in fw.steps:
            if s.answer:
                texts_to_check.append(str(s.answer))
            if s.question:
                texts_to_check.append(str(s.question))

    malformed_patterns = [
        re.compile(r"\bof\s*,\s*", re.IGNORECASE),
        re.compile(r"\bof\s+was\s+identified\b", re.IGNORECASE),
        re.compile(r"₹\s*,\s*", re.IGNORECASE),
        re.compile(r"\bof\s+₹\s*,\b", re.IGNORECASE),
        re.compile(r"\bwas\s+reportedly\s+duplicate\s+transaction\s+processed\b", re.IGNORECASE),
        re.compile(r"\bwas\s+reportedly\s+failed\b", re.IGNORECASE),
        re.compile(r"\bnot\s+(?:incomplete|missing|unavailable|blank|disabled)\b", re.IGNORECASE),
        re.compile(r"\bwhy\s+(?:did|was|were)\s+.*?\bsystem\s+records?\s+show\s+that\b", re.IGNORECASE),
        re.compile(r"\breportedly\s+failed\s+to\s+(?:approximately|about|roughly|nearly|₹|\$|€|£|\d)", re.IGNORECASE),
        re.compile(r"\bwhy\s+(?:did|was|were)\s+.*?\b(?:approximately|about|roughly|nearly|₹|\$|€|£)\b", re.IGNORECASE),
        # Immediate word repetition -- deliberately a plain literal
        # alternation, NOT a generic \b(\w+)\s+\1\b backreference: the
        # generic form false-positives on legitimate entity-ID conventions
        # throughout this codebase ("Lot LOT-9988", "SOP SOP-014" -- the
        # generic noun label immediately followed by an ID that starts with
        # the same word, case-insensitively).
        re.compile(r"\b(overpayment|status|duplicate|payment|inspection|checklist)\s+\1\b", re.IGNORECASE),
    ]

    for text in texts_to_check:
        for pat in malformed_patterns:
            if pat.search(text):
                return False, f"Malformed sentence structure or blank financial amount detected: {text}"
    return True, None


def _leading_hypothesis_id(rc: Any) -> str | None:
    """rc.leading_hypothesis is a DISPLAY string ("H1 — <statement>"), not a
    bare id -- set by leading_hypothesis_display() in analytical_validator.py.
    Extract the leading "H<n>" token it's always prefixed with."""
    raw = getattr(rc, "leading_hypothesis", None)
    if not raw:
        return None
    m = re.match(r"^(H\d+)\b", raw)
    return m.group(1) if m else raw


def _leading_hypothesis(rc: Any) -> Any | None:
    lead_id = _leading_hypothesis_id(rc)
    if not lead_id:
        return None
    for h in getattr(rc, "candidate_hypotheses", None) or []:
        if h.id == lead_id:
            return h
    return None


def _five_why_step_matches_statement(step: Any, statement: str) -> bool:
    from app.services.text_grounding import significant_words
    stmt_words = significant_words(statement or "")
    answer_words = significant_words(getattr(step, "answer", None) or "")
    if not stmt_words or not answer_words:
        return False
    overlap = stmt_words & answer_words
    return len(overlap) >= max(2, len(stmt_words) // 2)


def _check_causal_status_not_unknown_downstream(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-001: the authoritative leading hypothesis cannot be
    SUPPORTED/ESTABLISHED while the 5-Why step reflecting the SAME
    proposition still shows UNKNOWN -- the exact production inconsistency
    ('Root Cause Status: ESTABLISHED' next to a 5-Why step for the same
    mechanism reading UNKNOWN)."""
    rc = state.get("root_cause")
    fw = state.get("five_why")
    if not rc or not fw or not getattr(fw, "steps", None):
        return True, None
    if getattr(rc.status, "value", rc.status) not in ("SUPPORTED", "ESTABLISHED", "VERIFIED"):
        return True, None
    leading = _leading_hypothesis(rc)
    if not leading or not leading.statement:
        return True, None
    for step in fw.steps:
        # Phase 13: a graph-derived EVIDENCE_BOUNDARY marker step legitimately
        # names the already-established mechanism it is asking "why" about
        # (there is no deeper node to reference instead) -- that is the
        # correct question for "what caused the established mechanism
        # itself", not a restatement of the mechanism's own status. Only a
        # step with no such structural boundary marking can be the real
        # defect this invariant exists to catch (a prose-generated step that
        # actually mislabels the SAME proposition as UNKNOWN).
        if getattr(step, "boundary_status", None) == "EVIDENCE_BOUNDARY":
            continue
        if step.status == "UNKNOWN" and _five_why_step_matches_statement(step, leading.statement):
            return False, (
                f"Root cause is {getattr(rc.status, 'value', rc.status)} via hypothesis {leading.id}, "
                f"but 5-Why step {step.question!r} restates the same mechanism and is marked UNKNOWN"
            )
    return True, None


def _check_causal_five_why_consistency(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-003: 5-Why status must be consistent with the authoritative
    causal state for EVERY SUPPORTED/ESTABLISHED hypothesis, not only the
    selected leading one -- a broader sweep than INV-CAUSAL-001."""
    rc = state.get("root_cause")
    fw = state.get("five_why")
    if not rc or not fw or not getattr(fw, "steps", None):
        return True, None
    for h in getattr(rc, "candidate_hypotheses", None) or []:
        if getattr(h, "status", "") not in ("SUPPORTED", "ESTABLISHED") or not h.statement:
            continue
        for step in fw.steps:
            # Phase 13: see the matching comment in
            # _check_causal_status_not_unknown_downstream -- a graph-derived
            # EVIDENCE_BOUNDARY marker step naming an established mechanism
            # is asking about a deeper cause, not restating the mechanism.
            if getattr(step, "boundary_status", None) == "EVIDENCE_BOUNDARY":
                continue
            if step.status == "UNKNOWN" and _five_why_step_matches_statement(step, h.statement):
                return False, (
                    f"Hypothesis {h.id} is {h.status} but 5-Why step {step.question!r} "
                    "restates the same mechanism and is marked UNKNOWN"
                )
    return True, None


def _check_causal_provenance_for_promoted_hypothesis(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-002: every SUPPORTED/ESTABLISHED proposition must have at
    least one valid supporting claim -- never SUPPORTED/ESTABLISHED with
    zero evidence provenance."""
    rc = state.get("root_cause")
    if not rc:
        return True, None
    for h in getattr(rc, "candidate_hypotheses", None) or []:
        if getattr(h, "status", "") in ("SUPPORTED", "ESTABLISHED"):
            if not getattr(h, "supporting_evidence", None) and not getattr(h, "supporting_claim_ids", None):
                return False, f"Hypothesis {h.id} is {h.status} with zero supporting evidence provenance"
    return True, None


def _check_investigation_targets_unresolved(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-004: an investigation question must not re-verify a
    proposition the authoritative causal layer already resolved as
    SUPPORTED/ESTABLISHED (a plain yes/no confirmation question testing the
    already-leading hypothesis) -- questions must target the NEXT unresolved
    causal layer, not re-prove what's already established."""
    rc = state.get("root_cause")
    inv = state.get("investigation_plan")
    if not rc or not inv or not getattr(inv, "questions", None):
        return True, None
    leading = _leading_hypothesis(rc)
    if not leading or getattr(leading, "status", "") not in ("SUPPORTED", "ESTABLISHED"):
        return True, None
    from app.services.text_grounding import significant_words
    confirm_re = re.compile(r"^\s*(?:did|was|were|has|have|is|are)\b", re.IGNORECASE)
    stmt_words = significant_words(leading.statement or "")
    for q in inv.questions:
        if getattr(q, "hypothesis_tested", None) != leading.id or not stmt_words:
            continue
        question_words = significant_words(q.question or "")
        overlap = stmt_words & question_words
        if confirm_re.match(q.question or "") and len(overlap) >= max(2, len(stmt_words) // 2):
            return False, (
                f"Investigation question {q.question!r} re-verifies already-{leading.status} "
                f"hypothesis {leading.id} instead of targeting the next unresolved causal layer"
            )
    return True, None


def _check_semantic_field_separation(state: dict[str, Any]) -> tuple[bool, str | None]:
    impact = state.get("impact_assessment")
    if not impact:
        return True, None
    obj = (impact.affected_object or "").strip().lower()
    proc = (impact.process_at_risk or "").strip().lower()
    ctrl = (getattr(impact, "control_at_risk", "") or "").strip().lower()

    if obj and proc and obj == proc and obj not in ("unknown", ""):
        return False, f"Affected Object and Process at Risk are identical: {impact.affected_object}"
    if proc and ctrl and proc == ctrl and proc not in ("unknown", ""):
        return False, f"Process at Risk and Control at Risk are identical: {impact.process_at_risk}"
    return True, None


def _check_uncertainty_classification_present(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-UNCERTAINTY-001: When root cause is not established, primary_uncertainty
    must be classified into an explicit semantic uncertainty category rather than
    generic unclassified fallback."""
    canonical = state.get("canonical_finding_state")
    rc = state.get("root_cause")
    if not canonical:
        return True, None
    rc_status = str(getattr(rc, "status", ""))
    if "NOT_ESTABLISHED" in rc_status or rc is None:
        primary = getattr(canonical, "primary_uncertainty", None)
        if not primary or primary in ("UNKNOWN", ""):
            return False, "Primary semantic uncertainty must be explicitly classified when root cause is not established"
    return True, None


def _check_requirement_uncertainty_blocks_hypotheses(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-UNCERTAINTY-002: When applicable requirement is uncertain, causal hypothesis
    generation is blocked (0 candidate hypotheses allowed) until requirement is resolved."""
    canonical = state.get("canonical_finding_state")
    rc = state.get("root_cause")
    if not canonical or not rc:
        return True, None
    if getattr(canonical, "primary_uncertainty", None) == "REQUIREMENT_UNCERTAIN" or getattr(canonical, "semantic_type", None) == "REQUIREMENT_UNCERTAIN":
        hyps = getattr(rc, "candidate_hypotheses", []) or []
        # Filter out recurrence hypotheses which are independent
        from app.agent.recurrence_guard import is_previous_capa_mechanism_hypothesis
        causal_hyps = [h for h in hyps if not is_previous_capa_mechanism_hypothesis(h.statement)]
        if causal_hyps:
            return False, "Causal hypotheses cannot be generated when applicable requirement is uncertain"
    return True, None



def _check_causal_readiness_gates_five_why(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-UNCERTAINTY-003: When causal readiness is NOT_READY, 5-Why analysis must be deferred."""
    canonical = state.get("canonical_finding_state")
    fw = state.get("five_why")
    if not canonical or not fw or not getattr(fw, "steps", None):
        return True, None
    if getattr(canonical, "primary_uncertainty", None) == "REQUIREMENT_UNCERTAIN" or getattr(canonical, "semantic_type", None) == "REQUIREMENT_UNCERTAIN":
        for s in fw.steps:
            if s.status in ("VERIFIED", "SUPPORTED"):
                return False, f"5-Why step claims {s.status} when causal readiness is NOT_READY"
    return True, None


def _check_investigation_maps_to_uncertainties(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-UNCERTAINTY-004: Investigation questions must resolve identified uncertainties."""
    inv = state.get("investigation_plan")
    if not inv or not getattr(inv, "questions", None):
        return True, None
    for q in inv.questions:
        if not getattr(q, "purpose", None) and not getattr(q, "objective", None):
            return False, f"Investigation question {q.question!r} lacks an investigative purpose/objective"
    return True, None


def _check_epistemic_status_transitions(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-UNCERTAINTY-005: Epistemic status transitions must adhere to directional validity
    (REPORTED cannot become VERIFIED without independent corroboration; UNKNOWN cannot become
    ESTABLISHED).

    Same-snapshot structural check (kept as-is -- still necessary, just not
    sufficient) plus a real cross-snapshot transition check using
    state["epistemic_snapshot_history"] (app.agent.causal_graph.
    capture_epistemic_snapshot), appended at the end of every
    core_synthesis_node pass. A run normally produces exactly one entry; a
    second only appears when the critic sends the investigation back for
    more evidence and core_synthesis re-runs -- comparing those two entries
    catches a hypothesis's evidence backing, the overall root_cause status,
    or causal_readiness silently regressing across THAT re-investigation
    without new conflicting evidence to justify it. Deliberately NOT
    compared against final_evidence_verification's own output: that node's
    job is exactly to reduce overstated certainty (grounding sweep,
    structural repairs), and treating its legitimate corrections as
    "regressions" would make this check fire on its own safety net.
    """
    rc = state.get("root_cause")
    if not rc:
        return True, None
    if rc.status in (RootCauseStatus.ESTABLISHED, "ESTABLISHED"):
        if not getattr(rc, "candidate_hypotheses", None):
            return False, "Root cause cannot be ESTABLISHED without candidate hypotheses"

    history = state.get("epistemic_snapshot_history") or []
    if len(history) < 2:
        return True, None
    prev, curr = history[-2], history[-1]
    prev_conflicts = set(prev.get("unresolved_conflict_ids", []))
    curr_conflicts = set(curr.get("unresolved_conflict_ids", []))
    new_conflict_evidence = bool(curr_conflicts - prev_conflicts)

    from app.agent.causal_graph import EVIDENCE_STRENGTH_RANK

    prev_hyps: dict[str, dict[str, str]] = prev.get("hypotheses", {})
    curr_hyps: dict[str, dict[str, str]] = curr.get("hypotheses", {})
    for name, prev_h in prev_hyps.items():
        curr_h = curr_hyps.get(name)
        if not curr_h:
            continue
        prev_rank = EVIDENCE_STRENGTH_RANK.get(prev_h.get("evidence_strength", "NONE"), 0)
        curr_rank = EVIDENCE_STRENGTH_RANK.get(curr_h.get("evidence_strength", "NONE"), 0)
        if curr_rank < prev_rank and curr_h.get("status") != "REFUTED" and not new_conflict_evidence:
            return False, (
                f"Hypothesis {name!r} evidence_strength regressed from "
                f"{prev_h.get('evidence_strength')} to {curr_h.get('evidence_strength')} "
                "without new conflicting evidence"
            )

    if (
        prev.get("root_cause_status") == "VERIFIED"
        and curr.get("root_cause_status") in ("NOT_ESTABLISHED", "UNKNOWN", "INSUFFICIENT_EVIDENCE")
        and not new_conflict_evidence
    ):
        return False, (
            f"Root cause status regressed from VERIFIED to {curr.get('root_cause_status')} "
            "without new conflicting evidence"
        )

    if (
        prev.get("causal_readiness") == "ESTABLISHED"
        and curr.get("causal_readiness") not in ("ESTABLISHED", None)
        and not new_conflict_evidence
    ):
        return False, (
            f"Causal readiness regressed from ESTABLISHED to {curr.get('causal_readiness')} "
            "without new conflicting evidence"
        )
    return True, None


def _check_requirement_resolution_priority_order(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-UNCERTAINTY-001: Requirement resolution questions must prioritize requirement
    resolution (P1), compliance comparison (P2), and deviation determination (P3) before mechanism."""
    canonical = state.get("canonical_finding_state")
    inv = state.get("investigation_plan")
    if not canonical or not inv or not getattr(inv, "questions", None):
        return True, None
    if getattr(canonical, "primary_uncertainty", None) == "REQUIREMENT_UNCERTAIN" or getattr(canonical, "semantic_type", None) == "REQUIREMENT_UNCERTAIN":
        p1_found = any(q.priority == "P1" or "P1" in str(q.question_id) for q in inv.questions)
        if not p1_found:
            return False, "Requirement resolution investigation plan must contain P1 requirement resolution question"
    return True, None


def _check_decision_tree_next_steps_defined(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-INVEST-UNCERTAINTY-002: Decision tree nodes must define branching rules or next steps."""
    inv = state.get("investigation_plan")
    if not inv or not getattr(inv, "questions", None):
        return True, None
    for q in inv.questions:
        if getattr(q, "category", "") == "REQUIREMENT_RESOLUTION" and getattr(q, "status", "ACTIVE") == "CONDITIONAL" and not getattr(q, "depends_on", None):
            return False, f"Conditional investigation question {q.question!r} must define depends_on prerequisite"
    return True, None



def _check_five_why_deferral_wording(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-5WHY-UNCERTAINTY-001: Deferred 5-Why must contain standard deferral statement."""
    canonical = state.get("canonical_finding_state")
    fw = state.get("five_why")
    if not canonical or not fw or not getattr(fw, "steps", None):
        return True, None
    if getattr(canonical, "primary_uncertainty", None) == "REQUIREMENT_UNCERTAIN" or getattr(canonical, "semantic_type", None) == "REQUIREMENT_UNCERTAIN":
        first_step = fw.steps[0]
        if "deferred" not in str(first_step.question).lower() and "deferred" not in str(first_step.answer).lower():
            return False, "5-Why analysis must be explicitly deferred when requirement/deviation status is not established"
    return True, None



def _check_semantic_roles_distinct(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-OBJECT-001: Semantic roles (observed_entity, affected_entity, control_entity) must be distinct and non-generic."""
    canonical = state.get("canonical_finding_state")
    impact = state.get("impact_assessment")
    if not canonical and not impact:
        return True, None
    proc = getattr(impact, "process_at_risk", None) or getattr(canonical, "affected_process", None)
    if proc and "process compliance" in str(proc).lower():
        return False, "process_at_risk must not be degraded generic placeholder 'Process compliance'"
    return True, None


def _check_non_actionable_input_cleanliness(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-ACTIONABLE-001: Non-actionable inputs must never produce candidate hypotheses, established root cause, investigation questions, or conditional CAPA actions."""
    canonical = state.get("canonical_finding_state")
    if not canonical or getattr(canonical, "is_actionable", True):
        return True, None
    rc = state.get("root_cause")
    if rc and getattr(rc, "candidate_hypotheses", None):
        return False, "Non-actionable input produced candidate causal hypotheses"
    if rc and rc.status not in (RootCauseStatus.NOT_APPLICABLE, "NOT_APPLICABLE"):
        return False, f"Non-actionable input produced root cause status {rc.status!r} instead of NOT_APPLICABLE"
    inv = state.get("investigation_plan")
    if inv and getattr(inv, "questions", None):
        return False, "Non-actionable input produced investigation questions"
    capa = state.get("capa_analysis")
    if capa and getattr(capa, "conditional_actions", None):
        return False, "Non-actionable input produced conditional CAPA actions"
    return True, None


def _check_semantic_role_preservation(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-ROLE-001: Semantic roles must not be corrupted: an actor/entity cannot become an activity/event in hypotheses or questions."""
    canonical = state.get("canonical_finding_state")
    if not canonical or not getattr(canonical, "is_actionable", True):
        return True, None
    rc = state.get("root_cause")
    if rc:
        from app.services.semantic_subject import is_actor_noun
        for h in getattr(rc, "candidate_hypotheses", []):
            name = getattr(h, "name", "")
            stmt = getattr(h, "statement", "")
            # e.g., EMPLOYEE_NOT_COMPLETED, OPERATOR_NOT_COMPLETED
            if name.endswith("_NOT_COMPLETED"):
                prefix = name[:-len("_NOT_COMPLETED")].lower()
                if is_actor_noun(prefix):
                    return False, f"Candidate hypothesis {name} treats actor noun {prefix!r} as the completed activity"
            if re.search(r"\bRequired\s+(?:employee|operator|technician|supervisor|analyst|manager|inspector)\b", stmt, re.IGNORECASE):
                return False, f"Candidate hypothesis statement {stmt!r} corrupts actor entity into required activity"
    inv = state.get("investigation_plan")
    if inv:
        for q in getattr(inv, "questions", []):
            q_text = getattr(q, "question", "")
            if re.search(r"\bcompleted\s+the\s+required\s+(?:employee|operator|technician|supervisor|analyst|manager|inspector)\b", q_text, re.IGNORECASE):
                return False, f"Investigation question {q_text!r} treats actor entity as required activity"
            if re.search(r"\bauthenticated\s+(?:employee|operator|technician|supervisor|analyst|manager|inspector)\s+record\b", q_text, re.IGNORECASE):
                return False, f"Investigation question {q_text!r} phrases record topic as actor record"
    return True, None


def _check_entity_atomicity(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-INV-001: No affected object or proposition subject may be a compound concatenation
    of multiple event/state concepts (e.g. 'A delivery and receipt', 'A (read-receipt)', 'A delivery delivery')."""
    canonical = state.get("canonical_finding_state")
    impact = state.get("impact_assessment")
    report = state.get("report")
    if not canonical:
        return True, None

    # Structural patterns matching synthetic compound entities across any domain
    compound_patterns = [
        # Repeated adjacent headwords (e.g. "delivery delivery", "control control")
        re.compile(r"\b([a-z]{3,})\s+\1\b", re.IGNORECASE),
        # Parenthetical qualifiers appended to entities (e.g. "delivery (read-receipt)")
        re.compile(r"\([^)]*(?:receipt|status|check|record|log|event|action|control)[^)]*\)", re.IGNORECASE),
        # Merged action/state compound phrases
        re.compile(r"\b\w+\s+(?:delivery|execution|receipt|approval|payment)\s+and\s+\w+\b", re.IGNORECASE),
        re.compile(r"\b(?:delivery|receipt|read|access|acknowledgement|payment|recovery|training|authorization)\s+(?:and|&|\+)\s+(?:delivery|receipt|read|access|acknowledgement|payment|recovery|training|authorization)\b", re.IGNORECASE),
    ]

    targets = [
        ("canonical.affected_object", canonical.affected_object),
        ("canonical.finding_subject", canonical.finding_subject),
        ("impact.affected_object", getattr(impact, "affected_object", None)),
        ("report.impact.affected_object", getattr(getattr(report, "impact_assessment", None), "affected_object", None)),
    ]

    for label, target in targets:
        if not target or target == "UNKNOWN":
            continue
        for pat in compound_patterns:
            if pat.search(str(target)):
                return False, f"{label} {target!r} is a synthetic compound concatenation of distinct concepts"
    return True, None


def _check_event_state_separation(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-INV-002: Event occurrences must never be equated with evidence states or compliance states."""
    canonical = state.get("canonical_finding_state")
    if not canonical:
        return True, None
    sem_type = getattr(canonical, "semantic_type", None)
    cond = str(getattr(canonical, "deviation_condition", "")).lower()
    if sem_type == "RECORD" and "did not perform" in cond:
        return False, "Missing record finding equates document availability with activity non-execution"
    return True, None


def _check_relation_preservation(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-INV-003: Conflicting reports must preserve individual participant relations rather than merging them."""
    canonical = state.get("canonical_finding_state")
    conflicts = getattr(canonical, "evidence_conflicts", []) if canonical else []
    if conflicts:
        for conf in conflicts:
            prop = getattr(conf, "proposition", "")
            if re.search(r"\b\w+\s+delivery\s+and\s+receipt\s+failed\b", prop, re.IGNORECASE):
                return False, f"Conflict collapses distinct transmission/receipt relations into compound failure: {prop!r}"
    return True, None


def _check_epistemic_orthogonality(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-INV-004: Epistemic status must be evaluated per proposition/relation, not globally collapsed."""
    props = state.get("propositions", [])
    if not props:
        canonical = state.get("canonical_finding_state")
        props = getattr(canonical, "propositions", []) if canonical else []
    if props:
        # Check that individual propositions carry valid explicit statuses
        for p in props:
            if not getattr(p, "status", None) and not getattr(p, "support_level", None):
                return False, f"Proposition {p.id} lacks an explicit epistemic status"
    return True, None


def _check_concept_provenance(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-INV-005: Every concept in RCA and Impact must have valid machine traceability."""
    report = state.get("report")
    if report and getattr(report, "semantic_traceability", None):
        matrix = report.semantic_traceability
        if getattr(matrix, "untraced_concepts", None):
            return False, f"Untraced concepts found in Semantic Traceability Matrix: {matrix.untraced_concepts}"
    return True, None


def _check_derivation_transparency(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-INV-006: Any derived concept must declare source = DERIVED with antecedent proposition IDs."""
    canonical = state.get("canonical_finding_state")
    props = getattr(canonical, "propositions", []) if canonical else []
    for p in props:
        if getattr(p, "derived_concept", False) and not getattr(p, "derived_from", None):
            return False, f"Derived proposition {p.id} does not declare antecedent proposition IDs"
    return True, None


def _check_graph_separation(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-INV-007: Semantic graph nodes/edges must remain distinct from Causal graph nodes/edges."""
    canonical = state.get("canonical_finding_state")
    sg = getattr(canonical, "semantic_graph", None)
    if sg and getattr(sg, "edges", None):
        for e in sg.edges:
            # Semantic edges should describe relations (TRANSMITTED_TO, APPLIES_TO, etc.), not causal causation
            rel = str(getattr(e, "relation_type", ""))
            if "CAUSED_BY" in rel or "ROOT_CAUSE_OF" in rel:
                return False, f"Semantic graph edge {e.id} contains causal progression link {rel!r}"
    return True, None


def _check_investigation_edge_gain(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-INV-008: Investigation questions must resolve unknown edges between known nodes."""
    inv = state.get("investigation_plan")
    if inv and getattr(inv, "questions", None):
        for q in inv.questions:
            if not getattr(q, "question", None) or len(q.question.strip()) < 5:
                return False, "Investigation question is empty or lacks information gain"
    return True, None


def _check_causal_edge_validity(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-INV-009: 5-Why steps must follow valid directed causal edges without jumping semantic domains."""
    fw = state.get("five_why")
    if fw and getattr(fw, "steps", None):
        for i, step in enumerate(fw.steps):
            if step.status in ("VERIFIED", "SUPPORTED") and not step.answer:
                return False, f"5-Why step {i+1} marked {step.status} without a valid explanation"
    return True, None


def _check_entity_event_separation(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-004: Entities must not merge multiple distinct action/event predicates (e.g. 'delivery and receipt', 'payment and recovery')."""
    canonical = state.get("canonical_finding_state")
    impact = state.get("impact_assessment")
    if not canonical:
        return True, None
    event_predicates = [
        re.compile(r"\b(?:delivery|receipt|read|access|acknowledgement|payment|recovery|training|authorization)\s+(?:and|&|\+)\s+(?:delivery|receipt|read|access|acknowledgement|payment|recovery|training|authorization)\b", re.IGNORECASE),
        re.compile(r"\b\w+\s+(?:delivery|execution|receipt|approval|payment)\s+and\s+\w+\b", re.IGNORECASE),
    ]
    for target in [canonical.affected_object, getattr(impact, "affected_object", None)]:
        if not target or target == "UNKNOWN" or target.startswith("UNRESOLVED"):
            continue
        for pat in event_predicates:
            if pat.search(str(target)):
                return False, f"Entity {target!r} contains a synthetic compound event merger"
    return True, None


def _check_generated_text_firewall(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-009: Generated text from investigation questions or 5-Why answers cannot mutate canonical state."""
    canonical = state.get("canonical_finding_state")
    inv = state.get("investigation_plan")
    if not canonical or not inv or not getattr(inv, "questions", None):
        return True, None
    # Verify canonical subject is not an exact copy of any full investigation question
    q_texts = {q.question.strip().lower() for q in inv.questions if getattr(q, "question", None)}
    if canonical.finding_subject and canonical.finding_subject.strip().lower() in q_texts:
        return False, "Canonical finding_subject was overwritten by generated investigation question text"
    return True, None


def _check_impact_grounding(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-014: Impact affected_object must be grounded in canonical entities or explicit source propositions."""
    impact = state.get("impact_assessment")
    canonical = state.get("canonical_finding_state")
    if not impact or not getattr(impact, "affected_object", None) or not canonical:
        return True, None
    if impact.affected_object in ("UNKNOWN", "NOT ESTABLISHED", ""):
        return True, None
    canon_obj = getattr(canonical, "affected_object", None) or getattr(canonical, "finding_subject", None)
    if canon_obj and canon_obj not in ("UNKNOWN", "NOT ESTABLISHED", "") and not canon_obj.startswith("UNRESOLVED"):
        if canon_obj.lower() not in impact.affected_object.lower() and impact.affected_object.lower() not in canon_obj.lower():
            return False, f"Impact affected_object {impact.affected_object!r} diverges from canonical {canon_obj!r}"
    return True, None


def _check_capa_grounding(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-SEM-015: CAPA actions must act exclusively on verified risks, mechanisms, or root causes."""
    capa = state.get("capa_analysis")
    rc = state.get("root_cause")
    if capa and getattr(capa, "conditional_actions", None) and rc:
        rc_status = str(getattr(rc, "status", ""))
        if "NOT_ESTABLISHED" in rc_status:
            for act in capa.conditional_actions:
                if not getattr(act, "if_cause_confirmed", None) and getattr(act, "action_type", None) == "CORRECTIVE_ACTION":
                    return False, "Corrective action is unconditional despite root cause not being established"
    return True, None


def _check_cross_section_consistency_graph(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CONSISTENCY-001: cross-section consistency pass.

    root_cause, five_why, CAPA, impact, and cost_impact all derive from the
    same canonical state but are written by different code paths (LLM
    synthesis, deterministic fallback, critic repair). The ~90 other rules
    in this registry are individually targeted point-checks; this is the one
    place that treats those sections as a small connected graph and checks
    the pairwise implications called out explicitly by the brief (root
    cause vs CAPA, cost vs impact narrative, 5-Why completion vs root cause)
    together, so an inconsistency between sections is caught even when no
    single narrower rule happens to cover that exact pair.
    """
    rc = state.get("root_cause")
    capa = state.get("capa_analysis")
    impact = state.get("impact_assessment")
    cost = state.get("cost_impact")
    fw = state.get("five_why")

    cause_unestablished = bool(rc) and str(getattr(rc, "status", "")) in (
        "NOT_ESTABLISHED", "INSUFFICIENT_EVIDENCE", "STATED_UNVERIFIED",
        "RootCauseStatus.NOT_ESTABLISHED", "RootCauseStatus.INSUFFICIENT_EVIDENCE", "RootCauseStatus.STATED_UNVERIFIED",
    )
    if cause_unestablished and capa and getattr(capa, "conditional_actions", None):
        for action in capa.conditional_actions:
            if getattr(action, "action_type", None) in ("CORRECTIVE_ACTION", "SYSTEMIC_ACTION") and not getattr(action, "if_cause_confirmed", None):
                return False, (
                    "CAPA proposes an unconditional corrective/systemic action while root_cause.status is "
                    f"{getattr(rc, 'status', None)!r} (not established)"
                )

    if (
        cost
        and str(getattr(cost, "recoverability_status", "") or getattr(cost, "recoverability", "")) == "RECOVERED"
        and impact
        and getattr(impact, "impact_observed", None)
        and "unrecovered" in str(impact.impact_observed).lower()
    ):
        return False, (
            "Financial recovery status disagrees between cost_impact (RECOVERED) and "
            "impact_assessment.impact_observed (references unrecovered amount)"
        )

    if (
        fw
        and getattr(fw, "is_complete", False)
        and getattr(fw, "steps", None)
        and str(fw.steps[-1].status) in ("VERIFIED", "SUPPORTED")
        and cause_unestablished
    ):
        return False, (
            "5-Why marked complete with a VERIFIED/SUPPORTED final step while root_cause.status remains "
            f"{getattr(rc, 'status', None)!r} (not established)"
        )

    return True, None


def _check_causal_level_consistency(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUS-004: CAUSAL_LEVEL_CONSISTENCY.

    Verifies across all reasoning layers that:
      - A verified immediate mechanism remains VERIFIED in 5-Why and evidence state even when the deeper root cause is NOT_ESTABLISHED.
      - Upstream certainty is not downgraded by downstream uncertainty (e.g. verified deviation/mechanism -> UNKNOWN).
      - Downstream uncertainty is not promoted to verified without objective corroboration.
      - 5-Why traversal halts at the first unknown edge with UNKNOWN status.
      - CAPA conditions match the verified causal level.
      - Investigation questions do not request verification of already-verified facts.
    """
    canonical = state.get("canonical_finding_state")
    fw = state.get("five_why")
    rc = state.get("root_cause")
    inv = state.get("investigation_plan")
    evidence_ledger = state.get("evidence_ledger", [])

    if not canonical:
        return True, None

    verified_claims = {
        getattr(e, "claim", getattr(e, "text", "")).strip().lower()
        for e in evidence_ledger
        if getattr(e, "status", None) == "VERIFIED"
    }

    # 1. Check that a genuine verified immediate mechanism (distinct from the primary observation) is preserved
    has_conflicts = bool(getattr(canonical, "evidence_conflicts", None))
    observed_dev = (getattr(canonical, "observed_deviation", "") or getattr(canonical, "deviation", "") or "").strip().lower()
    if not has_conflicts and getattr(canonical, "immediate_mechanism_status", None) == "VERIFIED" and getattr(canonical, "immediate_mechanism", None):
        mech_stmt = canonical.immediate_mechanism.strip().lower()
        # Phase 13: "distinct from observation" means substantively distinct,
        # not merely a different string -- `observed_deviation` is a
        # semantic-subject-normalized label (e.g. "Inspection execution —
        # missed") while `immediate_mechanism` is the raw stated-mechanism
        # sentence (e.g. "Technician missed the inspection."); these
        # routinely restate the exact same fact in different phrasing. Uses
        # the same majority-word-overlap equivalence test as
        # app.agent.causal_graph.is_graph_authoritative_for_five_why's
        # matching check, so both modules agree on what counts as "the same
        # fact" restated versus genuinely new mechanism-level information.
        from app.services.text_grounding import significant_words
        mech_words = significant_words(mech_stmt)
        dev_words = significant_words(observed_dev)
        is_distinct_mechanism = not (mech_words and dev_words and len(mech_words & dev_words) >= max(2, len(mech_words) // 2))
        # Only enforce when mechanism is distinct from observation (multi-level causal depth)
        if observed_dev and is_distinct_mechanism and len(evidence_ledger) > 1:
            if fw and getattr(fw, "steps", None) and len(fw.steps) >= 1:
                step_statuses = [str(getattr(s, "status", "")) for s in fw.steps]
                if "VERIFIED" not in step_statuses and "SUPPORTED" not in step_statuses:
                    return False, f"Verified immediate mechanism {canonical.immediate_mechanism!r} was degraded to all-UNKNOWN in 5-Why chain"

    # 2. Check that 5-Why does not claim complete/verified root cause when root_cause is NOT_ESTABLISHED
    if rc and str(getattr(rc, "status", "")) in ("NOT_ESTABLISHED", "RootCauseStatus.NOT_ESTABLISHED"):
        if fw and getattr(fw, "is_complete", False) and getattr(fw, "steps", None):
            if str(getattr(fw.steps[-1], "status", "")) in ("VERIFIED", "SUPPORTED", "ESTABLISHED"):
                return False, "5-Why final step asserts established cause while root cause status is NOT_ESTABLISHED"

    # 3. Check that RCA statement does not contradict verified immediate mechanism
    if getattr(canonical, "immediate_mechanism_status", None) == "VERIFIED" and getattr(canonical, "immediate_mechanism", None):
        if rc and rc.statement and "no causal mechanism" in rc.statement.lower():
            return False, f"RCA statement asserts 'no causal mechanism' despite verified immediate mechanism: {canonical.immediate_mechanism!r}"

    # 4. Check that verified normative requirement violations are not re-questioned as unknown requirements
    if inv and getattr(inv, "questions", None):
        sem_graph = getattr(canonical, "semantic_graph", None)
        if sem_graph and getattr(sem_graph, "nodes", None):
            known_reqs = [
                n.label.lower() for n in sem_graph.nodes
                if getattr(n, "node_type", None) == "REQUIREMENT" and getattr(n, "epistemic_status", None) == "VERIFIED"
            ]
            if known_reqs:
                for q in inv.questions:
                    q_text = getattr(q, "question", "").lower()
                    if "which approved procedure and specific requirement were applicable" in q_text:
                        return False, "Investigation plan asks to identify applicable requirement despite requirement being verified in canonical state"

    return True, None


def _check_normative_compliance_separation(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-NORM-001: NORMATIVE_COMPLIANCE_SEPARATION.

    Enforces that:
      1. A verified normative compliance violation does not automatically assert root cause.
      2. Compliance relations (VIOLATES/SATISFIES/OUTSIDE_REQUIREMENT) are NOT used as
         causal edges in CausalRelationship objects — normative and causal graphs are separate.
      3. Attributes are not promoted to entities.
      4. Known requirements are preserved and not investigated again.
    """
    canonical = state.get("canonical_finding_state")
    if not canonical:
        return True, None

    from app.models.agent import SemanticRelationType
    normative_types = SemanticRelationType.normative_relation_types()

    # CHECK 1: causal relationships must not use normative relation types
    causal_rels = state.get("causal_relationships") or []
    for cr in causal_rels:
        cr_type = str(getattr(cr, "relation_type", ""))
        if cr_type in normative_types:
            return False, (
                f"CausalRelationship uses normative relation type {cr_type!r}; "
                "normative relations are never causal"
            )

    return True, None



# ============================================================
# INV-ROLE-002 through INV-ROLE-010: Semantic Role Invariants
# ============================================================


def _check_requirement_not_affected_object(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-ROLE-002: A SemanticNodeType.REQUIREMENT node must never appear as the
    value of affected_object or finding_subject. Requirements are normative
    constructs; the affected_object must be the entity, record, or process that
    instantiated the requirement violation — not the requirement itself."""
    canonical = state.get("canonical_finding_state")
    if not canonical:
        return True, None
    sem_graph = getattr(canonical, "semantic_graph", None)
    if not sem_graph:
        return True, None
    req_labels = {
        n.label.strip().lower()
        for n in (getattr(sem_graph, "nodes", None) or [])
        if str(getattr(n, "node_type", "")) == "REQUIREMENT"
    }
    for field in ("affected_object", "finding_subject"):
        val = (getattr(canonical, field, None) or "").strip().lower()
        if val and val in req_labels:
            return False, (
                f"canonical.{field}={val!r} is a REQUIREMENT node label, not an entity. "
                "Requirements govern entities; they are not the entity itself."
            )
    impact = state.get("impact_assessment")
    if impact:
        val = (getattr(impact, "affected_object", None) or "").strip().lower()
        if val and val in req_labels:
            return False, (
                f"impact.affected_object={val!r} is a REQUIREMENT node label, not an entity."
            )
    return True, None


def _check_attribute_not_affected_object(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-ROLE-003: A SemanticNodeType.ATTRIBUTE node must never appear as the
    value of affected_object or finding_subject. Attributes describe properties
    of entities; they are not the entity being investigated."""
    canonical = state.get("canonical_finding_state")
    if not canonical:
        return True, None
    sem_graph = getattr(canonical, "semantic_graph", None)
    if not sem_graph:
        return True, None
    attr_labels = {
        n.label.strip().lower()
        for n in (getattr(sem_graph, "nodes", None) or [])
        if str(getattr(n, "node_type", "")) == "ATTRIBUTE"
    }
    for field in ("affected_object", "finding_subject"):
        val = (getattr(canonical, field, None) or "").strip().lower()
        if val and val in attr_labels:
            return False, (
                f"canonical.{field}={val!r} is an ATTRIBUTE node label, not an entity. "
                "Attributes describe properties; they are not the investigated entity."
            )
    return True, None


def _check_normative_relations_not_used_as_causal(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-ROLE-004: No SemanticRelationType in the normative set
    {VIOLATES, SATISFIES, REQUIRES, GOVERNED_BY, SUBJECT_TO, CONFORMS_TO, GOVERNS,
    REQUIRES_ATTRIBUTE, LACKS_REQUIRED_ATTRIBUTE, NOT_PERFORMED_AS_REQUIRED,
    NOT_DEMONSTRATED, INCONSISTENT_WITH, WITHIN_REQUIREMENT, OUTSIDE_REQUIREMENT}
    may appear as the relation_type of a CausalRelationship. Normative and causal
    graphs are structurally separate — no cross-contamination is permitted."""
    from app.models.agent import SemanticRelationType
    normative_types = SemanticRelationType.normative_relation_types()
    causal_rels = state.get("causal_relationships") or []
    for cr in causal_rels:
        cr_type = str(getattr(cr, "relation_type", ""))
        if cr_type in normative_types:
            return False, (
                f"CausalRelationship has normative relation_type {cr_type!r}; "
                "causal relationships must use causal edge types only"
            )
    # Also check root_cause narrative for overt normative-used-as-causal language
    rc = state.get("root_cause")
    if rc:
        narrative = getattr(rc, "narrative", "") or ""
        # Pattern: "the violation of X caused Y" — normative event presented as causal agent
        if re.search(r"\bthe\s+(?:violation|breach|noncompliance|non-compliance)\s+of\s+\S+.{0,40}?\b(?:caused?|led\s+to|resulted\s+in|triggered)\b", narrative, re.IGNORECASE):
            return False, (
                "Root cause narrative presents a normative violation as a causal agent "
                "(e.g. 'the violation of X caused Y'); normative facts constrain the investigation, "
                "they are not causal mechanisms"
            )
    return True, None


# Structural pattern that identifies synthetic process names constructed from
# topic_word + generic suffix. These are created by semantic_subject.py pattern
# branches and must not appear in affected_process unless a PROCESS node
# corroborates them in the semantic graph.
_SYNTHETIC_PROCESS_SUFFIX_RE = re.compile(
    r"\b(?:operational\s+control|scheduling\s+and\s+deadline\s+control|"
    r"execution\s+control|change\s+control|administration\s+control|"
    r"management\s+control|compliance\s+control|review\s+control|"
    r"monitoring\s+control|oversight\s+control)s?\b",
    re.IGNORECASE,
)


def _check_affected_process_not_synthetic(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-ROLE-005: affected_process must not be a synthetic process label without a corroborating node."""
    canonical = state.get("canonical_finding_state")
    if not canonical:
        return True, None
    proc = getattr(canonical, "affected_process", None) or ""
    if proc in ("", "UNKNOWN", "NOT_ESTABLISHED", "UNRESOLVED", "Operational process"):
        return True, None
    if not _SYNTHETIC_PROCESS_SUFFIX_RE.search(proc):
        return True, None
    sem_graph = getattr(canonical, "semantic_graph", None)
    if sem_graph:
        process_labels = {
            n.label.strip().lower()
            for n in (getattr(sem_graph, "nodes", None) or [])
            if str(getattr(n, "node_type", "")) in ("PROCESS", "CONTROL", "SemanticNodeType.PROCESS", "SemanticNodeType.CONTROL")
        }
        if proc.strip().lower() in process_labels:
            return True, None
    return False, f"affected_process={proc!r} is a synthetic process label with no corroborating PROCESS/CONTROL node in the semantic graph"


def _check_immediate_action_grounding(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-ROLE-006: Immediate actions must not introduce operational context
    (equipment, roles, systems, periods, locations) that is absent from the
    canonical finding state's verified evidence."""
    canonical = state.get("canonical_finding_state")
    if not canonical:
        return True, None
    capa = state.get("capa_analysis")
    if not capa:
        return True, None
    from app.agent.grounding_guard import build_source_text, ungrounded_entities
    evidence_ledger = state.get("evidence_ledger", [])
    source_text = build_source_text(
        getattr(canonical, "raw_finding", ""),
        evidence_ledger,
        [],
    )
    immediate_actions = getattr(capa, "immediate_actions", None) or []
    for action in immediate_actions:
        text = getattr(action, "action", None) or getattr(action, "description", None) or ""
        violations = ungrounded_entities(text, source_text)
        if violations:
            return False, (
                f"Immediate action introduces ungrounded operational context {violations!r}: {text!r}"
            )
    return True, None


def _check_verified_req_not_reinvestigated(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-ROLE-007: A requirement verified in the semantic graph must not be re-investigated."""
    canonical = state.get("canonical_finding_state")
    if not canonical:
        return True, None
    sem_graph = getattr(canonical, "semantic_graph", None)
    if not sem_graph:
        return True, None
    inv = state.get("investigation_plan")
    if not inv or not getattr(inv, "questions", None):
        return True, None
    verified_req_labels = [
        n.label.strip().lower()
        for n in (getattr(sem_graph, "nodes", None) or [])
        if str(getattr(n, "node_type", "")) == "REQUIREMENT"
        and getattr(n, "epistemic_status", None) == "VERIFIED"
    ]
    for req_label in verified_req_labels:
        req_words = {w for w in req_label.split() if len(w) > 3}
        if len(req_words) < 2:
            continue
        for q in inv.questions:
            q_text = (getattr(q, "question", "") or "").lower()
            if f"which approved procedure and specific requirement were applicable to {req_label}" in q_text:
                return False, f"Investigation question re-asks for verified requirement {req_label!r}"
    return True, None


def _check_affected_object_admissible_node_type(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-ROLE-009: affected_object must reference an admissible node type."""
    canonical = state.get("canonical_finding_state")
    if not canonical:
        return True, None
    val = (getattr(canonical, "affected_object", None) or "").strip()
    if not val or val.upper().startswith(("UNKNOWN", "UNRESOLVED")):
        return True, None
    sem_graph = getattr(canonical, "semantic_graph", None)
    if not sem_graph:
        return True, None
    _ADMISSIBLE = {"ENTITY", "RECORD", "ACTOR", "CONTROL", "PROCESS", "MEASUREMENT", "EVENT", "STATE", "OBSERVATION", "ROLE", "EVIDENCE"}
    node_map = {}
    for n in (getattr(sem_graph, "nodes", None) or []):
        nt = getattr(n, "node_type", None)
        nt_val = getattr(nt, "value", str(nt)).split(".")[-1]
        node_map[n.label.strip().lower()] = nt_val
    val_lower = val.strip().lower()
    if val_lower in node_map:
        node_type = node_map[val_lower]
        if node_type not in _ADMISSIBLE:
            return False, (
                f"affected_object={val!r} maps to SemanticNodeType.{node_type} which is not "
                f"admissible for affected_object"
            )
    return True, None


def _check_all_report_concepts_traced(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-ROLE-010: Traceability completeness for primary output fields."""
    report = state.get("report")
    if not report:
        return True, None
    matrix = getattr(report, "semantic_traceability", None)
    if not matrix:
        return True, None
    if getattr(matrix, "untraced_concepts", None):
        return False, f"Untraced concepts in report output: {matrix.untraced_concepts}"
    return True, None


# ============================================================
# INV-CAUSAL-001 through INV-CAUSAL-004: Real Runtime CausalGraph Invariants
# (Phase 3 — checks the ACTUAL state["causal_graph"] populated by
# app.agent.causal_graph.build_causal_graph, not the always-empty
# state["causal_relationships"] key INV-ROLE-004/_score_causal_validity
# check — see Remaining Limitations in the Phase 2 report for that gap.)
# ============================================================


def _check_causal_edge_evidence_licensing(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-001: A VERIFIED causal graph edge must carry non-empty
    evidence_ids OR reference a hypothesis with objective provenance —
    an edge cannot claim VERIFIED status while citing zero evidence."""
    from app.models.agent import CausalGraphEdgeStatus, EpistemicSource
    cg = state.get("causal_graph")
    if not cg or not getattr(cg, "edges", None):
        return True, None
    for e in cg.edges:
        if e.status == CausalGraphEdgeStatus.VERIFIED:
            if not e.evidence_ids and e.provenance != EpistemicSource.OBJECTIVE_RECORD:
                return False, (
                    f"CausalGraphEdge {e.edge_id} claims VERIFIED status with no evidence_ids "
                    f"and provenance={e.provenance!r} (not OBJECTIVE_RECORD)"
                )
    return True, None


def _check_causal_edge_no_self_loop(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-002: A causal edge's source and target must be distinct nodes —
    a self-loop is never a valid causal transition (Section 14: source != target)."""
    cg = state.get("causal_graph")
    if not cg or not getattr(cg, "edges", None):
        return True, None
    for e in cg.edges:
        if e.source_node_id == e.target_node_id:
            return False, f"CausalGraphEdge {e.edge_id} is a self-loop ({e.source_node_id})"
    return True, None


def _check_causal_graph_acyclic(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CGRAPH-008 (Phase 5 Section 3): the causal graph must be a DAG.
    A causal loop (X causes Y causes X) is never domain-independently valid —
    detected via a plain graph traversal (DFS cycle detection), not by
    string comparison or domain knowledge."""
    cg = state.get("causal_graph")
    if not cg or not getattr(cg, "edges", None):
        return True, None
    adjacency: dict[str, list[str]] = {}
    for e in cg.edges:
        adjacency.setdefault(e.source_node_id, []).append(e.target_node_id)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n.node_id: WHITE for n in cg.nodes}

    def _dfs(node_id: str, path: list[str]) -> str | None:
        color[node_id] = GRAY
        for neighbor in adjacency.get(node_id, []):
            if color.get(neighbor, WHITE) == GRAY:
                return " -> ".join(path + [neighbor])
            if color.get(neighbor, WHITE) == WHITE:
                cycle = _dfs(neighbor, path + [neighbor])
                if cycle:
                    return cycle
        color[node_id] = BLACK
        return None

    for node in cg.nodes:
        if color[node.node_id] == WHITE:
            cycle = _dfs(node.node_id, [node.node_id])
            if cycle:
                return False, f"Causal graph contains a cycle: {cycle}"
    return True, None


def _check_causal_edge_not_normative(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-003: A CausalGraphEdge's relation_type must never be one of
    the normative CausalEdgeType-adjacent relations — the graph builder only
    ever emits RESULTS_IN, but this guards against any future regression
    that starts deriving causal edges from normative semantic relations."""
    cg = state.get("causal_graph")
    if not cg or not getattr(cg, "edges", None):
        return True, None
    _NON_CAUSAL_EDGE_TYPES = {"REQUIRES", "ASSOCIATED_WITH"}
    for e in cg.edges:
        rel = str(getattr(e.relation_type, "value", e.relation_type))
        if rel in _NON_CAUSAL_EDGE_TYPES:
            return False, (
                f"CausalGraphEdge {e.edge_id} uses non-causal relation_type {rel!r}; "
                "only genuine causal edge types (e.g. RESULTS_IN) may appear in the causal graph"
            )
    return True, None


def _check_five_why_causal_edge_exists(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-004: If a FiveWhyStep declares a causal_edge_id, that edge
    must actually exist in state["causal_graph"] — a step must never
    reference a fabricated or stale edge ID."""
    fw = state.get("five_why")
    cg = state.get("causal_graph")
    if not fw or not getattr(fw, "steps", None):
        return True, None
    valid_edge_ids = {e.edge_id for e in (cg.edges if cg else [])}
    for step in fw.steps:
        edge_id = getattr(step, "causal_edge_id", None)
        if edge_id and edge_id not in valid_edge_ids:
            return False, (
                f"FiveWhyStep declares causal_edge_id={edge_id!r} which does not exist "
                "in the current causal graph"
            )
        source_id = getattr(step, "source_node_id", None)
        target_id = getattr(step, "target_node_id", None)
        if source_id and target_id and source_id == target_id:
            return False, (
                f"FiveWhyStep source_node_id == target_node_id ({source_id!r}) — "
                "a Why step must represent an actual causal transition, not a restatement"
            )
    return True, None


def _check_five_why_step_matches_its_own_edge(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-008 (Phase 26 Rule 5/7): a FiveWhyStep that declares a
    causal_edge_id must describe the SAME transition that edge actually
    licenses — step.source_node_id/target_node_id must equal the edge's
    own source_node_id/target_node_id. INV-CAUSAL-004 already proves the
    edge exists; this closes the remaining gap where a step could cite a
    real, valid edge ID while claiming it connects different nodes than it
    actually does (a graph traversal that is structurally valid in
    isolation but semantically disconnected from the edge it cites —
    exactly the class of defect Phase 26 targets)."""
    fw = state.get("five_why")
    cg = state.get("causal_graph")
    if not fw or not getattr(fw, "steps", None) or not cg:
        return True, None
    edges_by_id = {e.edge_id: e for e in (cg.edges or [])}
    for step in fw.steps:
        edge_id = getattr(step, "causal_edge_id", None)
        if not edge_id:
            continue
        edge = edges_by_id.get(edge_id)
        if edge is None:
            continue  # already caught by INV-CAUSAL-004
        step_source = getattr(step, "source_node_id", None)
        step_target = getattr(step, "target_node_id", None)
        if step_source and step_source != edge.source_node_id:
            return False, (
                f"FiveWhyStep cites causal_edge_id={edge_id!r} but its own source_node_id "
                f"{step_source!r} does not match the edge's actual source {edge.source_node_id!r}"
            )
        if step_target and step_target != edge.target_node_id:
            return False, (
                f"FiveWhyStep cites causal_edge_id={edge_id!r} but its own target_node_id "
                f"{step_target!r} does not match the edge's actual target {edge.target_node_id!r}"
            )
    return True, None


def _check_five_why_evidence_ids_resolve(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-TRACE-001 (Phase 26 Section 3: DOWNSTREAM_EPISTEMIC_STATUS_MUST_
    MATCH_AUTHORITATIVE_SOURCE, evidence-ledger half). Every FiveWhyStep.
    evidence_ids entry must resolve to a real evidence_ledger item (never a
    dangling/fabricated reference), AND a step whose own status is VERIFIED
    must cite at least one evidence item that is ITSELF VERIFIED in the
    ledger -- a downstream 5-Why renderer must not independently relabel a
    REPORTED/UNKNOWN ledger fact as VERIFIED just by attaching that status
    word to the step. This is the SAME authoritative evidence_ledger status
    every other layer (reconcile_hypothesis_from_evidence, CAPA, Impact)
    already treats as ground truth -- never re-derived here, only compared."""
    fw = state.get("five_why")
    if not fw or not getattr(fw, "steps", None):
        return True, None
    evidence_ledger = state.get("evidence_ledger") or []
    ledger_by_id = {getattr(e, "evidence_id", None): e for e in evidence_ledger}
    ledger_by_id.pop(None, None)
    if not ledger_by_id:
        # No evidence_id-addressable ledger this run (e.g. Phase pre-19
        # finding-text-only claims have no evidence_id) -- nothing to
        # cross-check against; not this invariant's concern.
        return True, None
    for step in fw.steps:
        evidence_ids = [eid for eid in (getattr(step, "evidence_ids", None) or []) if eid]
        if not evidence_ids:
            continue
        resolved = []
        for eid in evidence_ids:
            item = ledger_by_id.get(eid)
            if item is None:
                return False, f"FiveWhyStep cites evidence_id {eid!r}, which does not exist in evidence_ledger"
            resolved.append(item)
        if getattr(step, "status", None) == "VERIFIED":
            has_verified_source = any(getattr(item, "status", None) == "VERIFIED" for item in resolved)
            if not has_verified_source:
                return False, (
                    f"FiveWhyStep.status is VERIFIED but none of its cited evidence_ids "
                    f"{evidence_ids} are VERIFIED in the evidence_ledger — a downstream "
                    "renderer must not independently upgrade evidentiary status"
                )
    return True, None


def _check_evidence_artifacts_are_context_grounded(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-REPORT-003 (EVERY_EVIDENCE_ARTIFACT_MUST_BE_CONTEXT_GROUNDED).
    Every hypothesis-ID reference an investigation artifact makes -- an
    InvestigationQuestion's target_hypothesis_ids, an EvidenceRequest's
    hypothesis_ids, a CAPA ConditionalCapaAction's root_cause_hypothesis_id
    -- must resolve to a real CandidateHypothesis actually present in
    root_cause.candidate_hypotheses. A dangling reference means a
    downstream renderer (report, CAPA branch, evidence request) is citing
    a hypothesis that does not exist in this run's own analysis, which
    would surface as a fabricated or orphaned artifact in the final
    auditor-facing report. This is deliberately an ID-resolution check
    only (the same structural pattern as INV-TRACE-001's evidence_id
    check) -- never a vocabulary/word-overlap check against artifact text,
    since evidence artifacts are legitimately prospective (they name
    evidence not yet collected) and a text-similarity check would
    false-positive against the wording of a real, working request."""
    rc = state.get("root_cause")
    if not rc or not getattr(rc, "candidate_hypotheses", None):
        return True, None
    known_ids = {h.id for h in rc.candidate_hypotheses if getattr(h, "id", None)}
    if not known_ids:
        return True, None

    plan = state.get("investigation_plan")
    for q in getattr(plan, "questions", None) or []:
        for hid in getattr(q, "target_hypothesis_ids", None) or []:
            if hid and hid not in known_ids:
                return False, (
                    f"InvestigationQuestion {getattr(q, 'question_id', None) or getattr(q, 'id', None)!r} "
                    f"cites target_hypothesis_ids={hid!r}, which does not exist in "
                    f"root_cause.candidate_hypotheses"
                )

    for req in state.get("evidence_requests") or []:
        for hid in getattr(req, "hypothesis_ids", None) or []:
            if hid and hid not in known_ids:
                return False, (
                    f"EvidenceRequest {getattr(req, 'request_id', None)!r} cites hypothesis_ids="
                    f"{hid!r}, which does not exist in root_cause.candidate_hypotheses"
                )

    capa = state.get("capa")
    for action in getattr(capa, "conditional_actions", None) or []:
        hid = getattr(action, "root_cause_hypothesis_id", None)
        if hid and hid not in known_ids:
            return False, (
                f"ConditionalCapaAction cites root_cause_hypothesis_id={hid!r}, which does not "
                f"exist in root_cause.candidate_hypotheses"
            )

    # Claim-ID grounding: CandidateHypothesis.supporting_claim_ids/
    # contradicting_claim_ids and ConditionalCapaAction.supporting_claim_ids
    # must resolve to a real EvidenceClaim.claim_id in this run's own claim
    # registry (canonical_finding_state.evidence_claims — the actual
    # populated registry; state["evidence_claims"] is a separate, usually-
    # empty top-level key and is deliberately not used here). Skipped
    # entirely when no claim registry exists yet, same "nothing to ground
    # against" rule as the hypothesis-ID checks above.
    canonical = state.get("canonical_finding_state")
    known_claim_ids = {
        c.claim_id for c in (getattr(canonical, "evidence_claims", None) or []) if getattr(c, "claim_id", None)
    }
    if known_claim_ids:
        for h in rc.candidate_hypotheses:
            for cid in getattr(h, "supporting_claim_ids", None) or []:
                if cid and cid not in known_claim_ids:
                    return False, (
                        f"CandidateHypothesis {h.id!r} cites supporting_claim_ids={cid!r}, which "
                        f"does not exist in the evidence claim registry"
                    )
            for cid in getattr(h, "contradicting_claim_ids", None) or []:
                if cid and cid not in known_claim_ids:
                    return False, (
                        f"CandidateHypothesis {h.id!r} cites contradicting_claim_ids={cid!r}, "
                        f"which does not exist in the evidence claim registry"
                    )
        for action in getattr(capa, "conditional_actions", None) or []:
            for cid in getattr(action, "supporting_claim_ids", None) or []:
                if cid and cid not in known_claim_ids:
                    return False, (
                        f"ConditionalCapaAction cites supporting_claim_ids={cid!r}, which does "
                        f"not exist in the evidence claim registry"
                    )
    return True, None


# ============================================================
# INV-CAUSAL-005 through INV-CAUSAL-007: Phase 4 graph-consuming invariants
# ============================================================


def _check_established_root_cause_has_verified_causal_path(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-005 (Section 14): root_cause.status may only be ESTABLISHED
    when the real causal graph contains a VERIFIED edge reaching a node at
    UNDERLYING_CAUSE or SYSTEMIC_ROOT_CAUSE depth. A narrative or an LLM's
    confidence is never sufficient on its own — the graph must back it."""
    rc = state.get("root_cause")
    if not rc or str(getattr(rc, "status", "")).rsplit(".", 1)[-1] != "ESTABLISHED":
        return True, None
    cg = state.get("causal_graph")
    if not cg or not cg.edges:
        return False, "root_cause.status=ESTABLISHED but no causal graph edges exist to support it"
    from app.models.agent import CausalGraphEdgeStatus, CausalGraphNodeType
    node_by_id = {n.node_id: n for n in cg.nodes}
    for e in cg.edges:
        if e.status != CausalGraphEdgeStatus.VERIFIED:
            continue
        target = node_by_id.get(e.target_node_id)
        if target and target.node_type in (CausalGraphNodeType.UNDERLYING_CAUSE, CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE):
            return True, None
    return False, (
        "root_cause.status=ESTABLISHED but no VERIFIED causal graph edge reaches an "
        "UNDERLYING_CAUSE or SYSTEMIC_ROOT_CAUSE node"
    )


def _check_established_root_cause_consistent_with_five_why(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-REPORT-001 (final output consistency firewall): root_cause.status
    ESTABLISHED/SUPPORTED must not coexist with a 5-Why chain that itself
    reports no causal mechanism established (every step UNKNOWN/
    NOT_ESTABLISHED/REQUIRES_EVIDENCE). Two sections of the SAME report
    disagreeing about whether a mechanism was found is a genuine cross-
    section inconsistency INV-CAUSAL-005 (which only checks the causal
    GRAPH) does not catch on its own -- the graph can be correctly
    licensed while the rendered 5-Why narrative independently says "we
    don't know why", which is exactly the auditor-facing contradiction
    this invariant exists to reject."""
    rc = state.get("root_cause")
    if not rc or str(getattr(rc, "status", "")).rsplit(".", 1)[-1] not in ("ESTABLISHED", "SUPPORTED"):
        return True, None
    fw = state.get("five_why")
    steps = getattr(fw, "steps", None) if fw else None
    if not steps:
        # No 5-Why rendered at all is a separate concern (not this
        # invariant's) -- nothing to contradict.
        return True, None
    _NO_MECHANISM_STATUSES = ("UNKNOWN", "NOT_ESTABLISHED", "REQUIRES_EVIDENCE")
    if all(str(getattr(s, "status", "")) in _NO_MECHANISM_STATUSES for s in steps):
        return False, (
            f"root_cause.status={rc.status!r} but every 5-Why step reports no causal mechanism "
            "established (all UNKNOWN/NOT_ESTABLISHED/REQUIRES_EVIDENCE) -- the report's own "
            "sections disagree about whether a mechanism was found"
        )
    return True, None


def _check_established_root_cause_not_merely_mechanism(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-REPORT-002 (final output quality pass, Section 2/3: mechanism
    != root cause): when root_cause.causal_sufficiency is populated (see
    app.agent.causal_graph.derive_causal_sufficiency, wired in by
    final_evidence_verification_node), root_cause.status == ESTABLISHED
    must correspond to root_cause_sufficiency == ESTABLISHED, not merely
    mechanism_sufficiency == ESTABLISHED with the deeper root-cause tier
    unreached. This is the SAME depth requirement INV-CAUSAL-005 already
    enforces directly against the causal graph; this cross-checks the
    derived, report-facing field against that same authoritative source,
    catching drift if causal_sufficiency is ever populated by a different
    path than the one that also sets root_cause.status."""
    rc = state.get("root_cause")
    if not rc or str(getattr(rc, "status", "")).rsplit(".", 1)[-1] != "ESTABLISHED":
        return True, None
    suff = getattr(rc, "causal_sufficiency", None)
    if suff is None:
        return True, None  # not populated this run -- INV-CAUSAL-005 is the authority
    root_suff = str(getattr(suff, "root_cause_sufficiency", "")).rsplit(".", 1)[-1]
    if root_suff != "ESTABLISHED":
        mech_suff = str(getattr(suff, "mechanism_sufficiency", "")).rsplit(".", 1)[-1]
        return False, (
            f"root_cause.status=ESTABLISHED but causal_sufficiency.root_cause_sufficiency={root_suff!r} "
            f"(mechanism_sufficiency={mech_suff!r}) -- an established immediate mechanism must never be "
            "rendered as an established root cause"
        )
    return True, None


def _is_unconditional_capa_action(action: Any) -> bool:
    """An action is genuinely unconditional (a definitive claim, not a
    branch) only if its `if_cause_confirmed` field carries no real gating
    condition.

    CORRECTED in Phase 5 after source inspection: `ConditionalCapaAction`
    has NO `conditional`/`pending_investigation` boolean fields at all — an
    earlier version of this check (Phase 4) tested those non-existent
    fields via getattr-with-default, which silently always evaluated to
    False, misclassifying every hypothesis-gated action ("IF H1 is
    confirmed...") as unconditional and breaking real regression tests.
    The actual conditionality signal is the mandatory `if_cause_confirmed`
    string itself — verified against real pipeline output before fixing.
    """
    condition = (getattr(action, "if_cause_confirmed", "") or "").strip().lower()
    if not condition or condition in ("n/a", "none", "not applicable", "unconditional"):
        return True
    return not any(kw in condition for kw in ("if ", "when ", "pending", "confirmed", "upon"))


def _check_capa_definitive_action_requires_verified_causal_root(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CGRAPH-006 (Section 14): a CAPA corrective/systemic action that
    carries NO real hypothesis-confirmation gate (see
    `_is_unconditional_capa_action`) must be backed by a VERIFIED causal
    graph edge — CONTAINMENT/IMMEDIATE_CORRECTION actions are exempt since
    Section 14 explicitly allows them to be justified by the verified
    deviation alone, independent of root cause."""
    capa = state.get("capa_analysis")
    if not capa:
        return True, None
    definitive_actions = [
        a for a in (getattr(capa, "conditional_actions", None) or [])
        if getattr(a, "action_type", "") in ("CORRECTIVE_ACTION", "SYSTEMIC_ACTION")
        and _is_unconditional_capa_action(a)
    ]
    if not definitive_actions:
        return True, None
    cg = state.get("causal_graph")
    from app.models.agent import CausalGraphEdgeStatus
    has_verified_edge = bool(cg and any(e.status == CausalGraphEdgeStatus.VERIFIED for e in cg.edges))
    if not has_verified_edge:
        return False, (
            f"{len(definitive_actions)} unconditional root-cause CAPA action(s) exist "
            f"({[a.recommended_action[:40] for a in definitive_actions]}) but the causal "
            "graph has no VERIFIED edge to justify them"
        )
    return True, None


def _check_investigation_question_graph_grounding_ratio(state: dict[str, Any]) -> tuple[bool, str | None]:
    """INV-CAUSAL-007 (Section 4): when a causal_uncertainty_graph with
    unresolved edges exists, at least one investigation question must be
    graph-grounded (target_node_id set) — a plan consisting entirely of
    ungrounded generic questions while real structural uncertainty exists
    is exactly the "generic five-question checklist" Section 4 prohibits.
    CRITICAL, not BLOCKER: the LLM-authored planning path does not yet
    populate these fields (Phase 4 known limitation), so this cannot fail
    closed without blocking that entire path — it surfaces the gap instead.
    """
    ug = state.get("causal_uncertainty_graph")
    inv = state.get("investigation_plan")
    if not ug or not ug.edges or not inv or not inv.questions:
        return True, None
    from app.models.agent import CausalGraphEdgeStatus
    unresolved = [e for e in ug.edges if e.status == CausalGraphEdgeStatus.UNKNOWN]
    if not unresolved:
        return True, None
    grounded = [q for q in inv.questions if getattr(q, "target_node_id", None)]
    if not grounded:
        return False, (
            f"{len(unresolved)} unresolved causal-uncertainty edge(s) exist but zero "
            "investigation questions are graph-grounded (target_node_id set)"
        )
    return True, None


INVARIANT_REGISTRY: list[InvariantRule] = [
    InvariantRule(
        inv_id="INV-NORM-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Normative and compliance relations must remain structurally distinct from causal relations.",
        validate=_check_normative_compliance_separation,
    ),
    InvariantRule(
        inv_id="INV-ROLE-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Actor/entity must not be corrupted into an activity/event in hypotheses or investigation questions.",
        validate=_check_semantic_role_preservation,
    ),
    InvariantRule(
        inv_id="INV-ROLE-002",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="REQUIREMENT nodes must not appear as affected_object or finding_subject.",
        validate=_check_requirement_not_affected_object,
    ),
    InvariantRule(
        inv_id="INV-ROLE-003",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="ATTRIBUTE nodes must not appear as affected_object or finding_subject.",
        validate=_check_attribute_not_affected_object,
    ),
    InvariantRule(
        inv_id="INV-ROLE-004",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Normative relation types must never be used as causal edge types in CausalRelationship objects.",
        validate=_check_normative_relations_not_used_as_causal,
    ),
    InvariantRule(
        inv_id="INV-ROLE-005",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="affected_process must not be a synthetic process label without a corroborating PROCESS node in the semantic graph.",
        validate=_check_affected_process_not_synthetic,
    ),
    InvariantRule(
        inv_id="INV-ROLE-006",
        category=InvariantCategory.CAPA,
        severity=InvariantSeverity.CRITICAL,
        description="Immediate actions must not introduce operational context absent from canonical evidence.",
        validate=_check_immediate_action_grounding,
    ),
    InvariantRule(
        inv_id="INV-ROLE-007",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.CRITICAL,
        description="Investigation questions must not re-investigate requirements verified in the semantic graph.",
        validate=_check_verified_req_not_reinvestigated,
    ),
    InvariantRule(
        inv_id="INV-ROLE-009",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="affected_object must reference a node of type ENTITY/RECORD/ACTOR/CONTROL/PROCESS or be UNKNOWN.",
        validate=_check_affected_object_admissible_node_type,
    ),
    InvariantRule(
        inv_id="INV-ROLE-010",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Every asserted output concept must trace to at least one source claim in the evidence ledger.",
        validate=_check_all_report_concepts_traced,
    ),
    InvariantRule(
        # NOTE: namespaced INV-CGRAPH- rather than INV-CAUSAL- deliberately.
        # Source inspection during Phase 4 found the pre-existing registry
        # already used INV-CAUSAL-001 through INV-CAUSAL-006 for unrelated,
        # older invariants — this file's four Phase-3-added rules had
        # collided with those IDs since Phase 3, which silently broke the
        # BLOCKER-severity lookup in final_evidence_verification.py (a
        # dict-comprehension keyed by inv_id took the LAST-registered
        # entry's severity, so these were being read as WARNING/BLOCKER
        # from the wrong rule). Renamed to a namespace no other rule in
        # this file uses, verified by grep before assignment.
        inv_id="INV-CGRAPH-001",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="A VERIFIED causal graph edge must carry evidence_ids or objective-record provenance.",
        validate=_check_causal_edge_evidence_licensing,
    ),
    InvariantRule(
        inv_id="INV-CGRAPH-002",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="A causal graph edge's source and target nodes must be distinct (no self-loops).",
        validate=_check_causal_edge_no_self_loop,
    ),
    InvariantRule(
        inv_id="INV-CGRAPH-003",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Causal graph edges must use genuine causal relation types, never normative-adjacent ones.",
        validate=_check_causal_edge_not_normative,
    ),
    InvariantRule(
        inv_id="INV-CGRAPH-004",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="A FiveWhyStep's causal_edge_id must reference a real edge in the current causal graph; source_node_id must differ from target_node_id.",
        validate=_check_five_why_causal_edge_exists,
    ),
    InvariantRule(
        inv_id="INV-CAUSAL-008",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="A FiveWhyStep's source_node_id/target_node_id must match the actual source/target of the causal_edge_id it cites -- a step must not be semantically disconnected from the edge it references.",
        validate=_check_five_why_step_matches_its_own_edge,
    ),
    InvariantRule(
        inv_id="INV-TRACE-001",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="FiveWhyStep.evidence_ids must resolve to real evidence_ledger entries, and a VERIFIED step must cite at least one VERIFIED ledger item -- downstream rendering must not independently upgrade evidentiary status.",
        validate=_check_five_why_evidence_ids_resolve,
    ),
    InvariantRule(
        inv_id="INV-REPORT-003",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="EVERY_EVIDENCE_ARTIFACT_MUST_BE_CONTEXT_GROUNDED: hypothesis-ID references (InvestigationQuestion.target_hypothesis_ids, EvidenceRequest.hypothesis_ids, ConditionalCapaAction.root_cause_hypothesis_id) and claim-ID references (CandidateHypothesis.supporting_claim_ids/contradicting_claim_ids, ConditionalCapaAction.supporting_claim_ids) must resolve to a real hypothesis/claim -- no dangling/fabricated references.",
        validate=_check_evidence_artifacts_are_context_grounded,
    ),
    InvariantRule(
        inv_id="INV-CGRAPH-005",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="root_cause.status=ESTABLISHED requires a VERIFIED causal graph edge reaching an UNDERLYING_CAUSE/SYSTEMIC_ROOT_CAUSE node.",
        validate=_check_established_root_cause_has_verified_causal_path,
    ),
    InvariantRule(
        inv_id="INV-REPORT-001",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="root_cause.status=ESTABLISHED/SUPPORTED must not coexist with a 5-Why chain whose every step reports no causal mechanism established.",
        validate=_check_established_root_cause_consistent_with_five_why,
    ),
    InvariantRule(
        inv_id="INV-REPORT-002",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="root_cause.status=ESTABLISHED requires causal_sufficiency.root_cause_sufficiency=ESTABLISHED, not merely an established immediate mechanism.",
        validate=_check_established_root_cause_not_merely_mechanism,
    ),
    InvariantRule(
        # RE-PROMOTED in Phase 5. Phase 4 prototyped this, found it broke
        # regression tests, and demoted it to a scorer-only signal — but
        # root-caused the failure to a bug in the CHECK itself (it tested
        # `conditional`/`pending_investigation` boolean fields that don't
        # exist on ConditionalCapaAction, so every hypothesis-gated action
        # was misclassified as unconditional). Verified against real
        # pipeline output (`if_cause_confirmed` values) before fixing; see
        # `_is_unconditional_capa_action`. Now BLOCKER again.
        inv_id="INV-CGRAPH-006",
        category=InvariantCategory.CAPA,
        severity=InvariantSeverity.BLOCKER,
        description="A CAPA action with no real hypothesis-confirmation gate must be backed by a VERIFIED causal graph edge.",
        validate=_check_capa_definitive_action_requires_verified_causal_root,
    ),
    InvariantRule(
        inv_id="INV-CGRAPH-008",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="The causal graph must be acyclic.",
        validate=_check_causal_graph_acyclic,
    ),
    # NOTE: the question-graph-grounding-ratio check remains demoted —
    # unlike the CAPA check above, its Phase 4 failures were NOT due to a
    # bug in the check itself; the LLM-authored planning path genuinely
    # does not populate target_node_id yet (see Phase 4/5 reports). Exposed
    # as non-blocking quality-gate signal (D13) until that path is closed.
    InvariantRule(
        inv_id="INV-ACTIONABLE-001",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Non-actionable inputs must never manufacture hypotheses, investigation questions, or CAPA actions.",
        validate=_check_non_actionable_input_cleanliness,
    ),
    InvariantRule(
        inv_id="INV-CAUS-001",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.CRITICAL,
        description="Detection failure cannot be promoted as primary root cause.",
        validate=_check_detection_not_root_cause,
    ),
    InvariantRule(
        inv_id="INV-CAUS-002",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Root cause cannot be ESTABLISHED without supporting verified evidence provenance.",
        validate=_check_causal_provenance,
    ),
    InvariantRule(
        inv_id="INV-CAUS-003",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Root cause cannot be promoted if unresolved evidence conflicts undermine the mechanism.",
        validate=_check_causal_no_unsupported_promotion,
    ),
    InvariantRule(
        inv_id="INV-CAUS-004",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Causal level consistency must be maintained across all reasoning and output layers.",
        validate=_check_causal_level_consistency,
    ),
    InvariantRule(
        inv_id="INV-FIN-001",
        category=InvariantCategory.FINANCIAL,
        severity=InvariantSeverity.BLOCKER,
        description="Financial impact section cannot appear when no financial factor exists.",
        validate=_check_financial_visibility,
    ),
    InvariantRule(
        inv_id="INV-FIN-002",
        category=InvariantCategory.FINANCIAL,
        severity=InvariantSeverity.CRITICAL,
        description="Transaction amount must not automatically equal confirmed financial loss.",
        validate=_check_financial_loss_separation,
    ),
    InvariantRule(
        inv_id="INV-FIN-005",
        category=InvariantCategory.FINANCIAL,
        severity=InvariantSeverity.CRITICAL,
        description="An ESTIMATED/REPORTED financial amount cannot automatically become a VERIFIED actual financial loss.",
        validate=_check_estimated_amount_not_actual_loss,
    ),
    InvariantRule(
        inv_id="INV-CAPA-003",
        category=InvariantCategory.CAPA,
        severity=InvariantSeverity.CRITICAL,
        description="CAPA completion does not prove effectiveness, and missing effectiveness evidence does not prove ineffectiveness.",
        validate=_check_capa_effectiveness_not_overclaimed,
    ),
    InvariantRule(
        inv_id="INV-INVEST-009",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="A recurring finding with a previous CAPA reference must generate CAPA-effectiveness investigation coverage.",
        validate=_check_recurrence_investigation_coverage,
    ),
    InvariantRule(
        inv_id="INV-5WHY-001",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.CRITICAL,
        description="5-Why step cannot circular-repeat its question or previous answer.",
        validate=_check_five_why_no_repetition,
    ),
    InvariantRule(
        inv_id="INV-5WHY-002",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.CRITICAL,
        description="5-Why step cannot speculate candidate mechanisms using hedge words when unverified.",
        validate=_check_five_why_no_hedging,
    ),
    InvariantRule(
        inv_id="INV-HYP-001",
        category=InvariantCategory.HYPOTHESIS,
        severity=InvariantSeverity.BLOCKER,
        description="Candidate hypotheses must contain supporting evidence provenance.",
        validate=_check_hypothesis_citations,
    ),
    InvariantRule(
        inv_id="INV-CAPA-001",
        category=InvariantCategory.CAPA,
        severity=InvariantSeverity.CRITICAL,
        description="Systemic CAPA cannot be unconditional when root cause is unconfirmed.",
        validate=_check_capa_conditionality,
    ),
    InvariantRule(
        inv_id="INV-SEM-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.CRITICAL,
        description="Affected object, process at risk, and control at risk must be semantically distinct.",
        validate=_check_semantic_field_separation,
    ),
    InvariantRule(
        inv_id="INV-SEM-002",
        category=InvariantCategory.SEMANTIC,
        # Promoted from CRITICAL to BLOCKER: the audit found this violation was
        # detected, logged, and then shipped anyway. A false generic entity is
        # not a cosmetic defect -- it propagates into impact and CAPA.
        severity=InvariantSeverity.BLOCKER,
        description="Affected object and finding subject must name a concrete entity, never a generic process-category placeholder (an explicit UNRESOLVED/PARTIAL flag is allowed).",
        validate=_check_semantic_cleanliness,
    ),
    InvariantRule(
        inv_id="INV-SEM-004",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.CRITICAL,
        description="Uncertain entity extraction must be explicitly flagged (entity_resolution != RESOLVED requires an entity_resolution_note).",
        validate=_check_entity_resolution_flagged,
    ),
    InvariantRule(
        inv_id="INV-SEC-003",
        category=InvariantCategory.SECURITY,
        severity=InvariantSeverity.BLOCKER,
        description="Instruction-like/untrusted content must never appear in the evidence ledger or evidence_claims under any status.",
        validate=_check_no_untrusted_content_in_ledger,
    ),
    InvariantRule(
        inv_id="INV-EPIS-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="An epistemic-stance (belief/opinion) or conditional/counterfactual proposition must never be recorded as VERIFIED evidence.",
        validate=_check_epistemic_claims_not_verified,
    ),
    InvariantRule(
        inv_id="INV-TIME-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.CRITICAL,
        description="Affected period must not assert detection framing 'During the audit'.",
        validate=_check_affected_period,
    ),
    InvariantRule(
        inv_id="INV-RENDER-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="No malformed financial sentence or blank financial amount in rendered fields.",
        validate=_check_rendering_and_malformed_amounts,
    ),
    InvariantRule(
        inv_id="INV-SEC-001",
        category=InvariantCategory.SECURITY,
        severity=InvariantSeverity.BLOCKER,
        description="Unavailable referenced documents cannot have their contents asserted as evidence.",
        validate=_check_security_isolation,
    ),
    InvariantRule(
        inv_id="INV-SEC-002",
        category=InvariantCategory.SECURITY,
        severity=InvariantSeverity.BLOCKER,
        description="An instruction-like/prompt-injection proposition must never surface as causal support in root cause, 5-Why, CAPA, or impact.",
        validate=_check_no_injection_in_causal_output,
    ),
    InvariantRule(
        inv_id="INV-CAUSAL-001",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="A causal proposition cannot simultaneously be SUPPORTED/ESTABLISHED (root cause) and UNKNOWN (5-Why) for the same mechanism.",
        validate=_check_causal_status_not_unknown_downstream,
    ),
    InvariantRule(
        inv_id="INV-CAUSAL-002",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Every SUPPORTED/ESTABLISHED proposition must have valid evidence provenance.",
        validate=_check_causal_provenance_for_promoted_hypothesis,
    ),
    InvariantRule(
        inv_id="INV-CAUSAL-003",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="5-Why status must be consistent with the authoritative causal state for every SUPPORTED/ESTABLISHED hypothesis, not only the leading one.",
        validate=_check_causal_five_why_consistency,
    ),
    InvariantRule(
        inv_id="INV-CAUSAL-004",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Investigation questions must target the next unresolved causal layer, not re-verify an already-established proposition.",
        validate=_check_investigation_targets_unresolved,
    ),
    InvariantRule(
        inv_id="INV-CAUSAL-005",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="A common-factor/pattern-derived (INDICATIVE) hypothesis can never be SUPPORTED/ESTABLISHED.",
        validate=_check_indicative_not_promoted,
    ),
    InvariantRule(
        inv_id="INV-CAUSAL-006",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="REPORTED (or unverified) evidence can never make a hypothesis SUPPORTED/ESTABLISHED.",
        validate=_check_reported_not_promoted,
    ),
    InvariantRule(
        inv_id="INV-SEM-003",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Entity fields (affected_object, finding_subject) must be noun phrases, never a self-referential evidence proposition sentence.",
        validate=_check_entity_not_evidence_proposition,
    ),
    InvariantRule(
        inv_id="INV-SEMANTIC-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Affected object must not be a complete causal/action clause.",
        validate=_check_affected_object_not_full_clause,
    ),
    InvariantRule(
        inv_id="INV-SEMANTIC-002",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.CRITICAL,
        description="Comparison findings must preserve both sides of the comparison.",
        validate=_check_comparison_finding_preserves_both_sides,
    ),
    InvariantRule(
        inv_id="INV-INVEST-010",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Comparison/mismatch findings must receive comparison-specific investigation coverage.",
        validate=_check_investigation_targets_comparison,
    ),
    InvariantRule(
        inv_id="INV-WHY-012",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="A causal answer cannot introduce an unsupported actor/mechanism merely because the actor appears in the observation.",
        validate=_check_five_why_no_unsupported_actor_blame,
    ),
    InvariantRule(
        inv_id="INV-5WHY-CAUSAL-001",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="A 5-Why answer cannot promote or present an unverified candidate hypothesis as the established explanation.",
        validate=_check_five_why_no_unverified_hypothesis_promotion,
    ),
    InvariantRule(
        inv_id="INV-5WHY-CAUSAL-002",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="If no verified causal mechanism exists, a 5-Why answer cannot contain an unverified modal causal claim.",
        validate=_check_five_why_no_unverified_modal_causation,
    ),
    InvariantRule(
        inv_id="INV-5WHY-CAUSAL-003",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="No 5-Why step may restate the finding observation as an explanation after the evidence boundary.",
        validate=_check_five_why_no_restatement_after_boundary,
    ),
    InvariantRule(
        inv_id="INV-EVENT-001",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Every inferred event must have provenance.",
        validate=_check_hypothesis_has_investigation_path,
    ),
    InvariantRule(
        inv_id="INV-EVENT-002",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.CRITICAL,
        description="A missing control artifact cannot independently prove that the controlled activity did not occur.",
        validate=_check_missing_artifact_not_prove_non_occurrence,
    ),
    InvariantRule(
        inv_id="INV-EVENT-003",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="A controlled transition cannot be labeled causally established solely because the transition occurred.",
        validate=_check_transition_not_causally_established_by_occurrence,
    ),
    InvariantRule(
        inv_id="INV-EVENT-004",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.CRITICAL,
        description="A control-gap hypothesis must remain unverified until objective evidence establishes the mechanism.",
        validate=_check_control_gap_hypothesis_stays_unverified,
    ),
    InvariantRule(
        inv_id="INV-EVENT-005",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Every causal hypothesis must map to at least one investigation step.",
        validate=_check_hypothesis_has_investigation_path,
    ),
    InvariantRule(
        inv_id="INV-EVENT-006",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Every causal investigation step must identify the uncertainty/hypothesis it resolves.",
        validate=_check_investigation_plan_has_evidence,
    ),
    InvariantRule(
        inv_id="INV-EVENT-007",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="5-Why must not introduce causal mechanisms absent from the evidence graph.",
        validate=_check_five_why_no_unverified_modal_causation,
    ),
    InvariantRule(
        inv_id="INV-EVENT-008",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.WARNING,
        description="Investigation strategy must be selected from semantic relationships, not isolated domain nouns.",
        validate=_check_investigation_areas_not_restate_object,
    ),
    InvariantRule(
        inv_id="INV-EVENT-009",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.CRITICAL,
        description="Downstream impact cannot be inferred solely from the existence of a controlled transition.",
        validate=_check_downstream_impact_not_from_transition_alone,
    ),
    InvariantRule(
        inv_id="INV-CAUSAL-REPORTED-001",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="A REPORTED attributed claim cannot independently establish a causal hypothesis.",
        validate=_check_hypothesis_not_reported_only,
    ),
    InvariantRule(
        inv_id="INV-CAUSAL-REPORTED-002",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="Statement existence verification must not promote proposition verification.",
        validate=_check_five_why_reported_not_promoted,
    ),
    InvariantRule(
        inv_id="INV-CAUSAL-REPORTED-003",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="A hypothesis cannot use its originating attributed claim as independent causal support.",
        validate=_check_five_why_no_unverified_hypothesis_promotion,
    ),
    InvariantRule(
        inv_id="INV-CAUSAL-REPORTED-004",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Objective evidence confirming a condition does not automatically establish causality.",
        validate=_check_hypothesis_not_reported_only,
    ),
    InvariantRule(
        inv_id="INV-CAUSAL-REPORTED-005",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="5-Why cannot label an attributed causal explanation SUPPORTED without independent causal evidence.",
        validate=_check_five_why_reported_not_promoted,
    ),
    InvariantRule(
        inv_id="INV-CAUSAL-REPORTED-006",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.WARNING,
        description="Attribution must be preserved through understanding -> synthesis -> causal analysis -> 5-Why -> final verification.",
        validate=_check_five_why_reported_not_promoted,
    ),
    InvariantRule(
        inv_id="INV-REC-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.CRITICAL,
        description="Verified recurrence cannot be represented as NOT_EVIDENCED.",
        validate=_check_verified_recurrence_not_denied,
    ),
    InvariantRule(
        inv_id="INV-REC-002",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Recurrence evidence cannot independently establish future recurrence risk.",
        validate=_check_recurrence_not_establish_future_risk,
    ),
    InvariantRule(
        inv_id="INV-REC-003",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.CRITICAL,
        description="Prior CAPA completion cannot establish CAPA effectiveness.",
        validate=_check_capa_completion_not_effectiveness,
    ),
    InvariantRule(
        inv_id="INV-REC-004",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.CRITICAL,
        description="Missing effectiveness evidence cannot establish CAPA ineffectiveness.",
        validate=_check_missing_effectiveness_not_ineffectiveness,
    ),
    InvariantRule(
        inv_id="INV-REC-005",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Recurrence cannot independently establish causal failure of a prior CAPA.",
        validate=_check_recurrence_not_establish_capa_causal_failure,
    ),
    InvariantRule(
        inv_id="INV-REC-006",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Recurrence + prior action must trigger appropriate prior-action investigation coverage.",
        validate=_check_recurrence_investigation_coverage,
    ),
    InvariantRule(
        inv_id="INV-REC-007",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="5-Why cannot use unsupported modal causation when no verified mechanism exists.",
        validate=_check_five_why_no_unverified_modal_causation,
    ),
    InvariantRule(
        inv_id="INV-REC-008",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Investigation hypotheses must retain claim provenance.",
        validate=_check_hypothesis_has_investigation_path,
    ),
    InvariantRule(
        inv_id="INV-REC-009",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.CRITICAL,
        description="A definitive CAPA action cannot be generated when causal status is NOT_ESTABLISHED unless explicitly marked conditional.",
        validate=_check_capa_conditionality,
    ),
    InvariantRule(
        inv_id="INV-REC-010",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.WARNING,
        description="Multiple occurrences must not automatically be labeled SYSTEMIC without population evidence.",
        validate=_check_multiple_occurrences_not_auto_systemic,
    ),
    InvariantRule(
        inv_id="INV-MISSING-001",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.CRITICAL,
        description="Missing evidence cannot imply non-performance.",
        validate=_check_missing_evidence_not_non_performance,
    ),
    InvariantRule(
        inv_id="INV-MISSING-002",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Missing evidence cannot independently establish root cause.",
        validate=_check_missing_evidence_not_root_cause,
    ),
    InvariantRule(
        inv_id="INV-MISSING-003",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.CRITICAL,
        description="Downstream action cannot independently establish control failure.",
        validate=_check_downstream_action_not_control_failure,
    ),
    InvariantRule(
        inv_id="INV-MISSING-004",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="5-Why cannot repeat the observation after reaching the evidence boundary.",
        validate=_check_five_why_no_restatement_after_boundary,
    ),
    InvariantRule(
        inv_id="INV-MISSING-005",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Investigation steps must distinguish performance verification from documentation verification.",
        validate=_check_investigation_distinguishes_performance_from_documentation,
    ),
    InvariantRule(
        inv_id="INV-MISSING-006",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Downstream control investigation must be included when a downstream action is objectively present.",
        validate=_check_downstream_investigation_present,
    ),
    InvariantRule(
        inv_id="INV-MISSING-007",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.WARNING,
        description="Investigation areas must represent investigation domains rather than simply restating the affected object.",
        validate=_check_investigation_areas_not_restate_object,
    ),
    InvariantRule(
        inv_id="INV-COMP-004",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.CRITICAL,
        description="Parameter mismatch cannot automatically route through calculation-specific investigation.",
        validate=_check_parameter_mismatch_not_calculation_routed,
    ),
    InvariantRule(
        inv_id="INV-COMP-005",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.CRITICAL,
        description="Affected object must not contain full finding clauses or identifiers unless semantically necessary.",
        validate=_check_affected_object_no_unnecessary_identifiers,
    ),
    InvariantRule(
        inv_id="INV-INVEST-011",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Every hypothesis must have a corresponding investigation path.",
        validate=_check_hypothesis_has_investigation_path,
    ),
    InvariantRule(
        inv_id="INV-EVIDENCE-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.WARNING,
        description="Evidence artifacts must be deduplicated before final rendering.",
        validate=_check_evidence_artifacts_deduplicated,
    ),
    InvariantRule(
        inv_id="INV-INVEST-001",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Investigation questions must target unresolved causal propositions.",
        validate=_check_investigation_targets_unresolved,
    ),
    InvariantRule(
        inv_id="INV-INVEST-002",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="An established causal mechanism must not be re-investigated as an unknown fact.",
        validate=_check_investigation_targets_unresolved,
    ),
    InvariantRule(
        inv_id="INV-INVEST-003",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Investigation plan questions must name the evidence to examine.",
        validate=_check_investigation_plan_has_evidence,
    ),
    InvariantRule(
        inv_id="INV-WHY-001",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="5-Why cannot contradict the authoritative causal state (SUPPORTED/ESTABLISHED shown as UNKNOWN).",
        validate=_check_causal_status_not_unknown_downstream,
    ),
    InvariantRule(
        inv_id="INV-WHY-002",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.CRITICAL,
        description="5-Why cannot restate the question as its own answer.",
        validate=_check_five_why_no_repetition,
    ),
    InvariantRule(
        inv_id="INV-IMPACT-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Impact fields must use normalized entities, never a raw evidence-proposition sentence.",
        validate=_check_impact_fields_normalized,
    ),
    InvariantRule(
        inv_id="INV-IMPACT-002",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="ImpactAssessment.status must never be IMPACT_VERIFIED without VERIFIED evidence grounding it.",
        validate=_check_impact_never_verified_without_verified_evidence,
    ),
    InvariantRule(
        inv_id="INV-CAPA-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="A CAPA conditional action must never be conditioned on a REFUTED hypothesis.",
        validate=_check_capa_excludes_refuted_hypotheses,
    ),
    InvariantRule(
        inv_id="INV-WHY-006",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="Evidence stating that causal information is unavailable cannot be used as causal support for any Why answer.",
        validate=_check_five_why_evidence_absence_not_causal,
    ),
    InvariantRule(
        inv_id="INV-WHY-007",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.WARNING,
        description="5-Why depth is evidence-driven, not fixed at five -- a chain that hits the boundary on step 1 must not be padded further.",
        validate=_check_five_why_not_padded_past_boundary,
    ),
    InvariantRule(
        inv_id="INV-WHY-008",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="A causal (VERIFIED/SUPPORTED) Why answer must have actual causal evidence, not an evidence-absence statement.",
        validate=_check_five_why_evidence_absence_not_causal,
    ),
    InvariantRule(
        inv_id="INV-WHY-009",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="Evidence-absence claims cannot support causal mechanisms.",
        validate=_check_five_why_evidence_absence_not_causal,
    ),
    InvariantRule(
        inv_id="INV-WHY-011",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="A Why chain cannot resume VERIFIED/SUPPORTED progression after an UNKNOWN node without new independent evidence.",
        validate=_check_five_why_no_progression_past_unknown,
    ),
    InvariantRule(
        inv_id="INV-INVEST-007",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Investigation questions must target unresolved causal propositions.",
        validate=_check_investigation_targets_unresolved,
    ),
    InvariantRule(
        inv_id="INV-INVEST-008",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Investigation questions must not repeat already-established facts.",
        validate=_check_investigation_targets_unresolved,
    ),
    InvariantRule(
        inv_id="INV-INVEST-012",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="A graph-grounded investigation question's target/source node or edge id must resolve to a real object in this run's causal graph, never a dangling reference.",
        validate=_check_investigation_question_traceability,
    ),
    InvariantRule(
        inv_id="INV-INVEST-013",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="InvestigationPlan.status=NO_ACTIONABLE_UNCERTAINTY must not co-occur with populated questions.",
        validate=_check_investigation_plan_status_consistency,
    ),
    InvariantRule(
        inv_id="INV-INVEST-019",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="NO_ACTIONABLE_UNCERTAINTY cannot coexist with an unresolved node in this run's causal_uncertainty_graph.",
        validate=_check_no_actionable_uncertainty_matches_graph,
    ),
    InvariantRule(
        inv_id="INV-INVEST-021",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Every Stage-B question must resolve to a causal/semantic target.",
        validate=_check_stage_b_question_has_target,
    ),
    InvariantRule(
        inv_id="INV-INVEST-022",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Every Stage-B question must have an evidence-resolution target.",
        validate=_check_stage_b_question_has_evidence_target,
    ),
    InvariantRule(
        inv_id="INV-INVEST-023",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Stage B cannot silently collapse competing, unresolved hypotheses.",
        validate=_check_stage_b_preserves_competing_hypotheses,
    ),
    InvariantRule(
        inv_id="INV-INVEST-024",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="investigation_history's causal_graph_version/iteration_id must be strictly monotonic.",
        validate=_check_investigation_history_monotonic,
    ),
    InvariantRule(
        inv_id="INV-INVEST-025",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="NO_ACTIONABLE_UNCERTAINTY cannot be emitted while Stage B's own iteration record lists unresolved targets.",
        validate=_check_no_actionable_uncertainty_consistent_with_history,
    ),
    InvariantRule(
        inv_id="INV-INVEST-026",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Evidence acquired through the adaptive loop must carry request provenance.",
        validate=_check_evidence_has_provenance,
    ),
    InvariantRule(
        inv_id="INV-INVEST-027",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="HypothesisStatusChange evidence references must resolve to real evidence_ledger entries.",
        validate=_check_hypothesis_history_references_real_evidence,
    ),
    InvariantRule(
        inv_id="INV-INVEST-028",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="A hypothesis's current status must match the authoritative evidence-reconciliation evaluator's last decision -- no second epistemic authority may silently override it.",
        validate=_check_single_epistemic_authority,
    ),
    InvariantRule(
        inv_id="INV-INVEST-029",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Every EvidenceClaim must cite real evidence_ledger provenance -- a dangling or claim without evidence_ids must never reach runtime state.",
        validate=_check_evidence_claims_have_provenance,
    ),
    InvariantRule(
        inv_id="INV-OUTPUT-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="No malformed semantic fragments may reach the renderer.",
        validate=_check_rendering_and_malformed_amounts,
    ),
    InvariantRule(
        inv_id="INV-UNCERTAINTY-001",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Primary uncertainty classification required when root cause not established.",
        validate=_check_uncertainty_classification_present,
    ),
    InvariantRule(
        inv_id="INV-UNCERTAINTY-002",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Requirement uncertainty must block causal hypothesis generation.",
        validate=_check_requirement_uncertainty_blocks_hypotheses,
    ),
    InvariantRule(
        inv_id="INV-UNCERTAINTY-003",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="Causal readiness must gate 5-Why progression.",
        validate=_check_causal_readiness_gates_five_why,
    ),
    InvariantRule(
        inv_id="INV-UNCERTAINTY-004",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Investigation questions must map to identified uncertainties.",
        validate=_check_investigation_maps_to_uncertainties,
    ),
    InvariantRule(
        inv_id="INV-UNCERTAINTY-005",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.BLOCKER,
        description="Epistemic status transitions must adhere to directional validity.",
        validate=_check_epistemic_status_transitions,
    ),
    InvariantRule(
        inv_id="INV-INVEST-UNCERTAINTY-001",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.CRITICAL,
        description="Requirement resolution investigation priority order must follow P1 requirement -> P2 comparison -> P3 deviation.",
        validate=_check_requirement_resolution_priority_order,
    ),
    InvariantRule(
        inv_id="INV-INVEST-UNCERTAINTY-002",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Decision tree next steps and prerequisites must be defined for uncertainty resolution.",
        validate=_check_decision_tree_next_steps_defined,
    ),
    InvariantRule(
        inv_id="INV-5WHY-UNCERTAINTY-001",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="5-Why deferral wording must state why-chain is deferred when requirement/deviation is not established.",
        validate=_check_five_why_deferral_wording,
    ),
    InvariantRule(
        inv_id="INV-OBJECT-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.CRITICAL,
        description="Semantic roles must be distinct and process_at_risk must not be generic 'Process compliance'.",
        validate=_check_semantic_roles_distinct,
    ),
    InvariantRule(
        inv_id="INV-CONSISTENCY-001",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Cross-section consistency: root_cause/CAPA/impact/cost_impact/five_why must not contradict each other.",
        validate=_check_cross_section_consistency_graph,
    ),
    InvariantRule(
        inv_id="INV-SEM-INV-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="No affected object or proposition subject may be a compound concatenation of multiple event/state concepts.",
        validate=_check_entity_atomicity,
    ),
    InvariantRule(
        inv_id="INV-SEM-INV-002",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Event occurrences must never be equated with evidence states or compliance states.",
        validate=_check_event_state_separation,
    ),
    InvariantRule(
        inv_id="INV-SEM-INV-003",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Conflicting reports must preserve individual participant relations rather than merging them into a compound failure entity.",
        validate=_check_relation_preservation,
    ),
    InvariantRule(
        inv_id="INV-SEM-INV-004",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Epistemic status must be evaluated per proposition/relation, not globally collapsed.",
        validate=_check_epistemic_orthogonality,
    ),
    InvariantRule(
        inv_id="INV-SEM-INV-005",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Every concept in RCA and Impact must have valid machine traceability in the Semantic Traceability Matrix.",
        validate=_check_concept_provenance,
    ),
    InvariantRule(
        inv_id="INV-SEM-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="No affected object or proposition subject may be a compound concatenation of multiple distinct concepts.",
        validate=_check_entity_atomicity,
    ),
    InvariantRule(
        inv_id="INV-SEM-002",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Event occurrences must never be equated with evidence states or compliance states.",
        validate=_check_event_state_separation,
    ),
    InvariantRule(
        inv_id="INV-SEM-003",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Conflicting reports must preserve individual participant relations rather than merging them.",
        validate=_check_relation_preservation,
    ),
    InvariantRule(
        inv_id="INV-SEM-004",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Entities must not be named after event predicates.",
        validate=_check_entity_event_separation,
    ),
    InvariantRule(
        inv_id="INV-SEM-005",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Relations must remain explicit and preserved without mutation.",
        validate=_check_relation_preservation,
    ),
    InvariantRule(
        inv_id="INV-SEM-006",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Epistemic status must be evaluated per proposition/relation, not globally collapsed.",
        validate=_check_epistemic_orthogonality,
    ),
    InvariantRule(
        inv_id="INV-SEM-007",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Semantic graph nodes/edges must remain distinct from Causal graph nodes/edges.",
        validate=_check_graph_separation,
    ),
    InvariantRule(
        inv_id="INV-SEM-008",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Every concept in RCA and Impact must have valid machine traceability.",
        validate=_check_concept_provenance,
    ),
    InvariantRule(
        inv_id="INV-SEM-009",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Generated text from downstream nodes cannot mutate canonical semantic state.",
        validate=_check_generated_text_firewall,
    ),
    InvariantRule(
        inv_id="INV-SEM-010",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Any derived concept must declare source = DERIVED with valid antecedent proposition IDs.",
        validate=_check_derivation_transparency,
    ),
    InvariantRule(
        inv_id="INV-SEM-011",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="No synthetic entity creation via concatenation or parenthetical qualifier.",
        validate=_check_entity_atomicity,
    ),
    InvariantRule(
        inv_id="INV-SEM-012",
        category=InvariantCategory.FIVE_WHY,
        severity=InvariantSeverity.BLOCKER,
        description="5-Why steps must follow valid directed causal edges without jumping semantic domains.",
        validate=_check_causal_edge_validity,
    ),
    InvariantRule(
        inv_id="INV-SEM-013",
        category=InvariantCategory.CAUSAL,
        severity=InvariantSeverity.WARNING,
        description="Investigation questions must resolve unknown edges between known nodes.",
        validate=_check_investigation_edge_gain,
    ),
    InvariantRule(
        inv_id="INV-SEM-014",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Impact affected_object must be grounded in canonical entities or explicit source propositions.",
        validate=_check_impact_grounding,
    ),
    InvariantRule(
        inv_id="INV-SEM-015",
        category=InvariantCategory.CAPA,
        severity=InvariantSeverity.BLOCKER,
        description="CAPA actions must act exclusively on verified risks, mechanisms, or root causes.",
        validate=_check_capa_grounding,
    ),
]


def evaluate_all_invariants(state: dict[str, Any]) -> tuple[bool, list[str]]:
    """Evaluate all machine-readable invariants against the current analysis state.
    
    Returns:
        (is_valid, list_of_violations)
    """
    violations = []
    for rule in INVARIANT_REGISTRY:
        passed, error_msg = rule.validate(state)
        if not passed:
            violations.append(f"[{rule.inv_id}] {error_msg}")
    return len(violations) == 0, violations

