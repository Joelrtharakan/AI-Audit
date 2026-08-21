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
    reported_facts = [getattr(e, "claim", getattr(e, "text", "")) for e in evidence_ledger if str(getattr(e, "status", "")).endswith("REPORTED")]
    verified_facts = [getattr(e, "claim", getattr(e, "text", "")) for e in evidence_ledger if str(getattr(e, "status", "")).endswith("VERIFIED")]
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
    verified_facts = [e.claim for e in state.get("evidence_ledger", []) if str(getattr(e, "status", "")).endswith("VERIFIED")]
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
    verified_facts = [e.claim for e in state.get("evidence_ledger", []) if str(getattr(e, "status", "")).endswith("VERIFIED")]
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


INVARIANT_REGISTRY: list[InvariantRule] = [
    InvariantRule(
        inv_id="INV-ROLE-001",
        category=InvariantCategory.SEMANTIC,
        severity=InvariantSeverity.BLOCKER,
        description="Actor/entity must not be corrupted into an activity/event in hypotheses or investigation questions.",
        validate=_check_semantic_role_preservation,
    ),
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

