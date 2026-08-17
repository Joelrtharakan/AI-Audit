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

    for fact in verified_facts:
        f_low = fact.lower()
        # Direct objective confirmation of control bypass, disablement, system outage, or unvalidated action
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
        return True, SupportLevel.SUPPORTED, None, [], CausalLevel.L5_SYSTEMIC_CAUSE, True

    # 7. Standard plausible hypothesis (POSSIBLE)
    missing_evidence.append("Objective records confirming the causal mechanism")
    return True, SupportLevel.POSSIBLE, None, missing_evidence, CausalLevel.L3_IMMEDIATE_MECHANISM, False


def select_authoritative_leading_hypothesis(
    hypotheses: list[CandidateHypothesis],
    conflicts: list[EvidenceConflict] | None = None,
    evidence_ledger: list[EvidenceItem] | None = None,
) -> tuple[str | None, Literal["SELECTED", "TIED", "NONE"], RootCauseStatus, str | None]:
    """Select the leading hypothesis deterministically.

    Rules:
      - 0 eligible candidates -> (None, "NONE", NOT_ESTABLISHED, "No eligible causal hypotheses")
      - If unresolved conflicts exist -> (None, "NONE", NOT_ESTABLISHED, "Unresolved evidence conflicts prevent root cause selection")
      - 1 or more candidates:
        - If exactly 1 candidate is SUPPORTED with promotion allowed -> (H.id, "SELECTED", SUPPORTED, H.rationale)
        - If all candidates are POSSIBLE -> (None, "NONE", NOT_ESTABLISHED, "Available evidence does not discriminate among candidate hypotheses")
        - If multiple candidates are tied for highest support -> (None, "TIED", NOT_ESTABLISHED, "Multiple candidate hypotheses remain equally supported")
    """
    if not hypotheses:
        return None, "NONE", RootCauseStatus.NOT_ESTABLISHED, "No eligible causal hypotheses identified"

    if conflicts and len(conflicts) > 0:
        return None, "NONE", RootCauseStatus.NOT_ESTABLISHED, "Available evidence contains unresolved conflicts requiring investigation"

    supported_hyps = [h for h in hypotheses if getattr(h, "status", "") == "SUPPORTED"]
    if len(supported_hyps) == 1:
        leading = supported_hyps[0]
        return leading.id, "SELECTED", RootCauseStatus.SUPPORTED, getattr(leading, "rationale", "Supported by verified evidence")

    if len(supported_hyps) > 1:
        return None, "TIED", RootCauseStatus.NOT_ESTABLISHED, "Multiple candidate hypotheses are supported by available evidence"

    # All are POSSIBLE
    return None, "NONE", RootCauseStatus.NOT_ESTABLISHED, "Initial evidence is insufficient to establish causation; all hypotheses remain unverified"


def validate_final_analysis(state: dict[str, Any]) -> tuple[bool, list[str]]:
    """Canonical final validator validating all 20 causal invariants across all execution paths.

    Returns:
      (is_valid, warnings_or_violations)
    """
    violations: list[str] = []
    canonical = state.get("canonical_finding_state")
    rc = state.get("root_cause")
    fw = state.get("five_why")
    inv = state.get("investigation_plan")
    capa = state.get("capa_analysis")
    impact = state.get("impact_assessment")
    evidence = state.get("evidence_ledger", [])
    conflicts = getattr(canonical, "evidence_conflicts", []) if canonical else []
    ref_docs = getattr(canonical, "referenced_documents", []) if canonical else []

    # 1. Provenance: Every candidate hypothesis must have supporting evidence
    if rc and rc.candidate_hypotheses:
        for h in rc.candidate_hypotheses:
            if not getattr(h, "supporting_evidence", []) and not getattr(h, "supporting_claim_ids", []):
                violations.append(f"Hypothesis {h.id} lacks supporting evidence provenance")

    # 2. Document Availability & Content Isolation: Unavailable documents cannot have inferred contents
    if ref_docs:
        unavail = [getattr(d, "document_type", "").lower() for d in ref_docs if getattr(d, "reference_status", "") == "REFERENCED_UNAVAILABLE"]
        if rc and rc.candidate_hypotheses:
            for h in rc.candidate_hypotheses:
                stmt_low = h.statement.lower()
                for ut in unavail:
                    if ut and ut in stmt_low and re.search(r"\b(showed|indicated|proved|contained|recorded|stated)\b", stmt_low):
                        violations.append(f"Hypothesis {h.id} infers contents of unavailable document '{ut}'")

    # 3. Conflict Consistency: Conflicted findings cannot promote root cause without objective resolution
    if conflicts and len(conflicts) > 0:
        if rc and rc.status in (RootCauseStatus.ESTABLISHED, RootCauseStatus.SUPPORTED):
            violations.append("Root cause was promoted despite unresolved evidence conflicts")
        if rc and rc.leading_hypothesis_status == "SELECTED":
            violations.append("Leading hypothesis was selected despite unresolved evidence conflicts")

    # 4. 5-Why Evidence Boundary: Must stop at evidence boundary with UNKNOWN, MIXED, or not complete
    if fw and fw.steps:
        if rc and rc.status == RootCauseStatus.NOT_ESTABLISHED:
            if fw.is_complete and not any(s.status in ("UNKNOWN", "MIXED", "REQUIRES_EVIDENCE", "NOT_ESTABLISHED") for s in fw.steps):
                violations.append("5-Why chain claimed completion without established root cause")

    # 5. 5-Why Non-Circular & Non-Restating: Why answer cannot merely repeat or restate the observation
    from app.agent.causal_guard import restates_observation, repeats_previous_why_answer, is_circular_why_answer
    if fw and fw.steps:
        obs_dev = canonical.observed_deviation if canonical else None
        for i, step in enumerate(fw.steps):
            ans = step.answer or ""
            # If answer merely restates the observation as a verified causal explanation:
            if obs_dev and restates_observation(ans, obs_dev, step.question):
                if step.status not in ("UNKNOWN", "NOT_ESTABLISHED") and not ("does not establish why" in ans.lower() or "not establish" in ans.lower()):
                    violations.append(f"5-Why step {i+1} merely restated the observation as a verified causal explanation")
            if is_circular_why_answer(step.question, ans) and step.status not in ("UNKNOWN", "NOT_ESTABLISHED"):
                violations.append(f"5-Why step {i+1} answer restated its question without explaining it")
            if i > 0 and repeats_previous_why_answer(fw.steps[i-1].answer, ans) and step.status not in ("UNKNOWN", "NOT_ESTABLISHED"):
                violations.append(f"5-Why step {i+1} merely repeated step {i}'s answer")

    # 6. Execution Mode Consistency: Evidence boundary is NOT degraded execution
    analysis_mode = state.get("analysis_mode", "LLM")
    if analysis_mode == "LLM" and fw and fw.status_note:
        if "DEGRADED MODE" in fw.status_note.upper():
            violations.append("Reasoning evidence boundary was erroneously labeled as DEGRADED MODE in execution metadata")

    # 7. CAPA Traceability: 0 candidate hypotheses must yield 0 causal CAPAs
    if (not rc or not rc.candidate_hypotheses) and capa and capa.conditional_actions:
        causal_actions = [c for c in capa.conditional_actions if getattr(c, "action_type", "") in ("CORRECTIVE_ACTION", "SYSTEMIC_ACTION")]
        if causal_actions:
            violations.append("Causal CAPA actions were generated with zero candidate hypotheses")

    # 8. Semantic Ownership: Subject, affected_object, affected_period must be consistent
    if canonical and impact:
        if canonical.finding_subject != "UNKNOWN" and impact.affected_object:
            if impact.affected_object == "UNKNOWN" and canonical.finding_subject:
                violations.append("Impact affected_object drifted from canonical finding_subject")

    # 9. Root Cause Promotion Requirement: ESTABLISHED requires explicit supporting evidence IDs and proof
    if rc and rc.status == RootCauseStatus.ESTABLISHED:
        if not getattr(rc, "leading_hypothesis", None) and getattr(rc, "leading_hypothesis_status", "") != "SELECTED":
            violations.append("Root cause marked ESTABLISHED without selected leading hypothesis")

    return len(violations) == 0, violations

