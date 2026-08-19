"""Formal Causal Proposition Graph & Root-Cause Eligibility Engine.

Implements the formal evidence-backed causal progression:
  OBSERVATION -> REPORTED_MECHANISM -> POSSIBLE_HYPOTHESIS -> SUPPORTED_HYPOTHESIS -> ESTABLISHED_ROOT_CAUSE

Enforces:
  1. Strict state separation across domains (DELIVERY != RECEIPT != ACKNOWLEDGEMENT != ACTION).
  2. evaluate_root_cause_eligibility() with deterministic promotion criteria.
  3. Deterministic leading hypothesis selection (0 candidates -> NONE, tied -> NONE/TIED, verified -> SELECTED).
  4. Structured conflict descriptions and neutral investigation questions.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from app.models.agent import (
    CandidateHypothesis,
    CausalEdgeType,
    CausalLevel,
    CausalRelationship,
    EvidenceConflict,
    EvidenceItem,
    EvidenceStatus,
    Proposition,
    PropositionType,
    ReferencedDocumentInfo,
    RootCauseAnalysis,
    RootCauseStatus,
    SupportLevel,
)
from app.services.text_grounding import significant_words


def generate_structured_conflict_text(
    conflict_type: str,
    subject: str = "the notification",
    proposition_a: str = "",
    proposition_b: str = "",
) -> str:
    """Generate precise, structurally accurate conflict descriptions."""
    if conflict_type in ("DELIVERY_VS_RECEIPT", "SYSTEM_RECORD_VS_HUMAN_REPORT") or (
        "deliver" in proposition_a.lower() and "receiv" in proposition_b.lower()
    ):
        return (
            "System delivery records indicate successful delivery, while affected recipients "
            "reported that they did not receive the notification. Objective records are required "
            "to reconcile delivery, receipt, access, and acknowledgement."
        )
    if "training" in subject.lower() or "training" in proposition_a.lower():
        return (
            "Available evidence contains conflicting statements regarding training completion: "
            f"one account indicates training was completed, while another reports it was not. "
            "Objective training and attendance records are required to resolve the discrepancy."
        )
    if "sop" in subject.lower() or "procedure" in subject.lower() or "checklist" in subject.lower():
        return (
            f"Available evidence contains conflicting accounts regarding implementation of {subject}. "
            "Objective distribution, acknowledgment, and execution logs are required to establish the factual timeline."
        )
    return (
        f"Available evidence contains conflicting accounts regarding {subject}. "
        "Objective records are required to resolve the discrepancy."
    )


def evaluate_root_cause_eligibility(
    hypothesis: CandidateHypothesis | Any,
    causal_level: CausalLevel = CausalLevel.L3_IMMEDIATE_MECHANISM,
    supporting_propositions: list[Proposition] | None = None,
    contradicting_propositions: list[Proposition] | None = None,
    evidence_items: list[EvidenceItem] | None = None,
    conflicts: list[EvidenceConflict] | None = None,
    referenced_docs: list[ReferencedDocumentInfo] | None = None,
    source_text: str = "",
) -> tuple[bool, SupportLevel, str | None, list[str], CausalLevel, bool]:
    """Evaluate whether a candidate causal hypothesis is eligible and calculate its support level.

    Returns:
      (eligible, support_level, rejection_reason, missing_evidence, causal_level, promotion_allowed)
    """
    statement = getattr(hypothesis, "statement", str(hypothesis))
    missing_evidence: list[str] = []

    if not statement or len(statement.strip()) < 5:
        return False, SupportLevel.REJECTED, "empty_statement", ["Valid causal hypothesis statement"], causal_level, False

    # 1. Check unavailable document content inference first
    if referenced_docs:
        unavail_types = [
            getattr(d, "document_type", "").lower()
            for d in referenced_docs
            if getattr(d, "reference_status", "") == "REFERENCED_UNAVAILABLE"
        ]
        for ut in unavail_types:
            if ut and ut in statement.lower() and re.search(r"\b(showed|indicated|contained|recorded|proved|stated|demonstrated)\b", statement.lower()):
                return False, SupportLevel.REJECTED, "infers_unavailable_document_content", [f"Verified content of {ut}"], causal_level, False

    from app.agent.causal_guard import (
        hypothesis_attacks_statement_credibility,
        hypothesis_contradicts_verified_completion,
        hypothesis_overclaims_human_error,
        is_evidence_gap_not_hypothesis,
        is_evidence_state_not_hypothesis,
    )
    if hypothesis_attacks_statement_credibility(statement) or hypothesis_overclaims_human_error(statement):
        return False, SupportLevel.REJECTED, "attacks_statement_credibility", ["Process and system control records"], causal_level, False

    # 3. Reject evidence-state / uncertainty propositions
    if is_evidence_state_not_hypothesis(statement, getattr(hypothesis, "name", None)):
        return False, SupportLevel.REJECTED, "evidence_state_not_causal_hypothesis", ["Underlying process failure mechanism"], CausalLevel.EVIDENCE_STATE, False

    # 4. Reject restating evidence gaps
    if is_evidence_gap_not_hypothesis(statement, source_text):
        return False, SupportLevel.REJECTED, "restates_evidence_gap", ["Underlying process failure mechanism"], causal_level, False

    # 4. Check contradiction with verified facts
    verified_facts = [
        getattr(e, "claim", getattr(e, "text", str(e)))
        for e in (evidence_items or [])
        if getattr(e, "status", None) == EvidenceStatus.VERIFIED or str(getattr(e, "status", "")) == "EvidenceStatus.VERIFIED"
    ]
    if verified_facts and hypothesis_contradicts_verified_completion(statement, verified_facts):
        return False, SupportLevel.CONTRADICTED, "contradicts_verified_completion", ["Consistent verified records"], causal_level, False

    # 5. Check conflict bounds: if there is an unresolved conflict on this subject, cannot be SUPPORTED or ESTABLISHED
    hyp_words = significant_words(statement)
    if conflicts:
        for conf in conflicts:
            conf_words = significant_words(getattr(conf, "proposition", str(conf)))
            if hyp_words and conf_words and (hyp_words & conf_words):
                missing_evidence.append(f"Objective verification resolving {conf.conflict_type}")
                return True, SupportLevel.UNRESOLVED, "conflicting_evidence_unresolved", missing_evidence, causal_level, False

    # 6. Evaluate Positive Grounding / Evidence Confirmation:
    # Check if objective evidence materially establishes the causal failure
    # (e.g. "audit log shows block was disabled", "server log shows service outage", "authorization record shows unvalidated approval")
    stmt_low = statement.lower()
    has_verified_causal_proof = False
    # A verified TECHNICAL/INFRASTRUCTURE event (service crashed, queue
    # outage) is an immediate mechanism, not a systemic cause -- it
    # answers "what happened", not "why", and always invites a further
    # "why did the service crash" question the evidence here does NOT
    # answer. A verified CONTROL/GOVERNANCE failure (a control was
    # disabled/bypassed, training/assignment was never completed) IS the
    # systemic-level finding in QMS terms: the control-design or
    # process-compliance gap itself is the audit-terminal cause, with
    # nothing evidence-backed deeper to ask. Conflating the two (treating
    # every verified mechanism as L5_SYSTEMIC_CAUSE) is exactly what
    # produced ESTABLISHED root-cause claims next to an UNKNOWN 5-Why step
    # for the same immediate-mechanism proposition.
    proven_causal_level = CausalLevel.L5_SYSTEMIC_CAUSE

    for fact in verified_facts:
        f_low = fact.lower()
        # Direct objective confirmation of control bypass, disablement, system outage, or unvalidated action
        is_post_event = bool(re.search(r"\b(?:released|executed|occurred|paid)\b.*?\b(?:disabled|bypassed|deactivated)\s+(?:on|after)\b", f_low))
        if is_post_event:
            continue
        has_disabled_control = bool(re.search(
            r"\b(?:interlock|block|check|control|guard|safety|verification|validation|rule|token|approval|workflow|override)\b.*?\b(?:disabled|bypassed|defeated|deactivated|switched off|overridden|bypass|override)\b|"
            r"\b(?:disabled|bypassed|defeated|deactivated|switched off|overridden|bypass|override)\b.*?\b(?:interlock|block|check|control|guard|safety|verification|validation|rule|token|approval|workflow|override)\b",
            f_low,
        ))
        has_system_outage = bool(re.search(r"\b(?:server|system|service|channel|network|queue|message queue)\b.*?\b(?:outage|failure|down|crashed|unresponsive|failed)\b", f_low))
        has_unauthorized_action = bool(re.search(r"\b(?:authorized|approved|executed|operated)\b.*?\b(?:without|unvalidated|unqualified|untrained|no\s+completed)\b", f_low))
        has_training_proof = bool(re.search(r"\b(?:lms|training log|training records?)\b.*?\b(?:no\s+operators|not\s+completed|never\s+completed|uncompleted|failed\s+to\s+complete)\b", f_low))
        has_change_mgmt_bypass = bool(re.search(r"\b(?:change[- ]management|sop-eng-\w+)\b.*?\b(?:bypassed|skipped|unvalidated|unconfigured)\b", f_low))
        has_task_assignment_proof = bool(re.search(r"\b(?:never\s+assigned|not\s+assigned|unassigned|assignment\s+(?:failed|not\s+configured))\b", f_low))
        has_explicit_audit_proof = ("audit log" in f_low or "audit trail" in f_low or "system log" in f_low or "server log" in f_low or "security_audit_log" in f_low or "security audit" in f_low) and any(
            w in f_low for w in ("disabled", "bypassed", "failure", "outage", "defeated", "overridden", "override", "never assigned", "not assigned", "unassigned", "crashed", "unconfigured")
        )

        is_bypass_hyp = any(w in stmt_low for w in ("bypass", "bypassed", "disabled", "defeated", "overridden", "override", "skipped", "without training validation", "without dual authorization"))
        is_training_hyp = any(w in stmt_low for w in ("training", "lms", "competenc")) and not is_bypass_hyp
        is_system_hyp = any(w in stmt_low for w in ("server", "service", "queue", "outage", "crashed"))
        is_assignment_hyp = any(w in stmt_low for w in ("assignment", "assigned"))

        if is_bypass_hyp and (has_disabled_control or has_change_mgmt_bypass or has_explicit_audit_proof or has_unauthorized_action):
            has_verified_causal_proof = True
            proven_causal_level = CausalLevel.L5_SYSTEMIC_CAUSE
            break
        elif is_training_hyp and has_training_proof:
            has_verified_causal_proof = True
            proven_causal_level = CausalLevel.L4_ROOT_CAUSE
            break
        elif is_system_hyp and (has_system_outage or has_explicit_audit_proof):
            has_verified_causal_proof = True
            proven_causal_level = CausalLevel.L2_IMMEDIATE_MECHANISM
            break
        elif is_assignment_hyp and (has_task_assignment_proof or has_explicit_audit_proof):
            has_verified_causal_proof = True
            proven_causal_level = CausalLevel.L4_ROOT_CAUSE
            break
        elif has_disabled_control or has_system_outage or has_change_mgmt_bypass or has_explicit_audit_proof:
            if any(w in stmt_low for w in ("disable", "bypass", "outage", "service failure", "workflow", "authorization", "control", "interlock", "block", "rule")):
                has_verified_causal_proof = True
                proven_causal_level = (
                    CausalLevel.L2_IMMEDIATE_MECHANISM if has_system_outage and not has_disabled_control
                    else CausalLevel.L5_SYSTEMIC_CAUSE
                )
                break

    if has_verified_causal_proof:
        return True, SupportLevel.SUPPORTED, None, [], proven_causal_level, True

    # 7. Standard plausible hypothesis (POSSIBLE)
    missing_evidence.append("Objective records confirming the causal mechanism")
    return True, SupportLevel.POSSIBLE, None, missing_evidence, CausalLevel.L3_IMMEDIATE_MECHANISM, False


def select_authoritative_leading_hypothesis(
    hypotheses: list[CandidateHypothesis],
    conflicts: list[EvidenceConflict] | None = None,
    evidence_ledger: list[EvidenceItem] | None = None,
) -> tuple[str | None, Literal["SELECTED", "POSSIBLE", "TIED", "NONE"], RootCauseStatus, str | None]:
    """Select the leading hypothesis deterministically.

    Rules:
      - 0 eligible candidates -> (None, "NONE", NOT_ESTABLISHED, "No eligible causal hypotheses identified")
      - If unresolved conflicts exist -> (None, "NONE", NOT_ESTABLISHED, "Available evidence contains unresolved conflicts requiring investigation")
      - 1 or more candidates:
        - If exactly 1 candidate is SUPPORTED with promotion allowed -> (H.id, "SELECTED", ESTABLISHED/SUPPORTED, H.rationale)
        - If multiple candidates are SUPPORTED -> (None, "TIED", RootCauseStatus.NOT_ESTABLISHED, "Multiple candidate hypotheses are supported by available evidence")
        - If exactly 1 candidate is POSSIBLE -> (H.id, "POSSIBLE", RootCauseStatus.NOT_ESTABLISHED, "Single leading hypothesis pending objective verification")
        - If multiple candidates are POSSIBLE:
          - If tied for rank/score -> (None, "TIED", RootCauseStatus.NOT_ESTABLISHED, "Multiple candidate hypotheses remain equally plausible pending investigation")

    ESTABLISHED vs SUPPORTED (causal-proof threshold, Section 1/7 of the
    causal-consistency spec): VERIFIED evidence alone proves the LEADING
    hypothesis's own mechanism -- it says nothing about whether that
    mechanism is itself the systemic/root cause or merely an immediate
    trigger with its own unresolved "why" beneath it (e.g. "the notification
    service failed" is VERIFIED but doesn't explain WHY the service failed).
    Conflating "well-evidenced" with "root cause" is exactly what produced
    ESTABLISHED root-cause claims next to an UNKNOWN 5-Why step for the same
    proposition. ESTABLISHED therefore requires BOTH: the causal chain has
    reached a root/systemic causal_level (L4_ROOT_CAUSE or
    L5_SYSTEMIC_CAUSE) AND that level is backed by VERIFIED evidence. An
    immediate-mechanism-level hypothesis (L0-L3), however well verified,
    caps out at SUPPORTED.
    """
    if not hypotheses:
        return None, "NONE", RootCauseStatus.NOT_ESTABLISHED, "No eligible causal hypotheses identified"

    if conflicts and len(conflicts) > 0:
        return None, "NONE", RootCauseStatus.NOT_ESTABLISHED, "Available evidence contains unresolved conflicts requiring investigation"

    # Separate primary root-cause candidates from detection failures and contributing factors
    primary_hyps = [h for h in hypotheses if getattr(h, "causal_role", "PRIMARY_CAUSE") == "PRIMARY_CAUSE" or getattr(h, "status", "") == "SUPPORTED"]
    candidates_to_evaluate = primary_hyps if primary_hyps else hypotheses

    supported_hyps = [h for h in candidates_to_evaluate if getattr(h, "status", "") == "SUPPORTED"]
    if len(supported_hyps) == 1:
        leading = supported_hyps[0]
        is_root_level_cause = getattr(leading, "causal_level", None) in (
            CausalLevel.L4_ROOT_CAUSE, CausalLevel.L5_SYSTEMIC_CAUSE,
        )
        has_verified_evidence = getattr(leading, "evidence_strength", "") == "VERIFIED"
        status_to_return = (
            RootCauseStatus.ESTABLISHED if (is_root_level_cause and has_verified_evidence)
            else RootCauseStatus.SUPPORTED
        )
        return leading.id, "SELECTED", status_to_return, getattr(leading, "rationale", "Supported by verified evidence")

    if len(supported_hyps) > 1:
        return None, "TIED", RootCauseStatus.NOT_ESTABLISHED, "Multiple candidate hypotheses are supported by available evidence"

    # All are POSSIBLE
    if len(candidates_to_evaluate) == 1:
        leading = candidates_to_evaluate[0]
        if evidence_ledger is not None and not getattr(leading, "supporting_evidence", None) and not getattr(leading, "supporting_claim_ids", None):
            return None, "NONE", RootCauseStatus.NOT_ESTABLISHED, "Candidate hypothesis lacks supporting evidence"
        return leading.id, "POSSIBLE", RootCauseStatus.NOT_ESTABLISHED, "Single candidate hypothesis pending objective verification"

    return None, "TIED", RootCauseStatus.NOT_ESTABLISHED, "Multiple candidate hypotheses remain equally plausible pending investigation"


def validate_final_analysis(state: dict[str, Any]) -> tuple[bool, list[str]]:
    """Canonical final validator validating all production invariants across all execution paths.

    Returns:
      (is_valid, warnings_or_violations)
    """
    from app.agent.invariants import evaluate_all_invariants

    violations: list[str] = []
    canonical = state.get("canonical_finding_state")
    rc = state.get("root_cause")
    fw = state.get("five_why")
    capa = state.get("capa_analysis")
    impact = state.get("impact_assessment")
    conflicts = getattr(canonical, "evidence_conflicts", []) if canonical else []
    ref_docs = getattr(canonical, "referenced_documents", []) if canonical else []

    # 1. Evaluate formalized invariant registry
    inv_valid, inv_violations = evaluate_all_invariants(state)
    violations.extend(inv_violations)

    # 2. Execution Mode Consistency: Evidence boundary is NOT degraded execution
    analysis_mode = state.get("analysis_mode", "LLM")
    if analysis_mode == "LLM" and fw and getattr(fw, "status_note", None):
        if "DEGRADED MODE" in fw.status_note.upper():
            violations.append("Reasoning evidence boundary was erroneously labeled as DEGRADED MODE in execution metadata")

    # 3. CAPA Traceability: 0 candidate hypotheses must yield 0 causal CAPAs
    if (not rc or not getattr(rc, "candidate_hypotheses", None)) and capa and getattr(capa, "conditional_actions", None):
        causal_actions = [c for c in capa.conditional_actions if getattr(c, "action_type", "") in ("CORRECTIVE_ACTION", "SYSTEMIC_ACTION")]
        if causal_actions:
            violations.append("Causal CAPA actions were generated with zero candidate hypotheses")

    # 4. Semantic Ownership: Subject, affected_object, affected_period must be consistent
    if canonical and impact:
        if getattr(canonical, "finding_subject", None) and canonical.finding_subject != "UNKNOWN" and getattr(impact, "affected_object", None):
            if impact.affected_object == "UNKNOWN" and canonical.finding_subject:
                violations.append("Impact affected_object drifted from canonical finding_subject")

    return len(violations) == 0, violations

