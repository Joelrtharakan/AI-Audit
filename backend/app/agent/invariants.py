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
    
    financial_pattern = re.compile(
        r"₹|\$|€|£|INR|USD|EUR|\b(?:duplicate\s+payment|overpayment|financial\s+loss|monetary|cost\s+of\s+rework|rework\s+hours|rework\s+cost|rework|scrap\s+cost|scrap|downtime\s+cost|downtime|penalty|fine|refund)\b",
        re.IGNORECASE,
    )
    has_financial_signal = bool(financial_pattern.search(str(finding_text)))
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


def _check_semantic_cleanliness(state: dict[str, Any]) -> tuple[bool, str | None]:
    canonical = state.get("canonical_finding_state")
    impact = state.get("impact_assessment")
    if canonical and getattr(canonical, "affected_object", None):
        if canonical.affected_object in ("Process compliance", "employees control"):
            return False, f"Degraded placeholder affected_object: {canonical.affected_object}"
    if impact and getattr(impact, "affected_object", None):
        if impact.affected_object in ("Process compliance", "employees control"):
            return False, f"Degraded placeholder impact affected_object: {impact.affected_object}"
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

    malformed_patterns = [
        re.compile(r"\bof\s*,\s*", re.IGNORECASE),
        re.compile(r"\bof\s+was\s+identified\b", re.IGNORECASE),
        re.compile(r"₹\s*,\s*", re.IGNORECASE),
        re.compile(r"\bof\s+₹\s*,\b", re.IGNORECASE),
        re.compile(r"\bwas\s+reportedly\s+duplicate\s+transaction\s+processed\b", re.IGNORECASE),
        re.compile(r"\bwas\s+reportedly\s+failed\b", re.IGNORECASE),
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


INVARIANT_REGISTRY: list[InvariantRule] = [
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
        severity=InvariantSeverity.CRITICAL,
        description="Affected object and process naming must not be degraded placeholders.",
        validate=_check_semantic_cleanliness,
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
