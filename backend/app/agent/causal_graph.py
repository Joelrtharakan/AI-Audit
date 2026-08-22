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
    canonical_state: Any = None,
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

    # 5.5 If the finding is an EVENT_SEQUENCE_CONTROL transition with unverified mechanism, block promotion (INV-EVENT-003, INV-EVENT-004)
    if canonical_state and getattr(canonical_state, "semantic_type", None) == "EVENT_SEQUENCE_CONTROL":
        if getattr(canonical_state, "mechanism_status", "UNKNOWN") != "VERIFIED":
            missing_evidence.append("Objective records establishing whether the transition was authorized, bypassed, or omitted")
            return True, SupportLevel.POSSIBLE, None, missing_evidence, CausalLevel.L3_IMMEDIATE_MECHANISM, False

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
        has_instrument_proof = bool(re.search(r"\b(?:scada|event\s+log|telemetry|sensor|thermocouple|micrometer|instrument|open\s+circuit|seal\s+broken|calibration\s+record|tolerance\s+error)\b.*?\b(?:failed|open\s+circuit|error|broken|exceeded|out\s+of\s+calibration)\b", f_low))
        has_unauthorized_action = bool(re.search(r"\b(?:authorized|approved|executed|operated)\b.*?\b(?:without|unvalidated|unqualified|untrained|no\s+completed)\b", f_low))
        has_training_proof = bool(re.search(r"\b(?:lms|training log|training records?)\b.*?\b(?:no\s+operators|not\s+completed|never\s+completed|uncompleted|failed\s+to\s+complete)\b", f_low))
        has_change_mgmt_bypass = bool(re.search(r"\b(?:change[- ]management|sop-eng-\w+)\b.*?\b(?:bypassed|skipped|unvalidated|unconfigured)\b", f_low))
        has_task_assignment_proof = bool(re.search(r"\b(?:never\s+assigned|not\s+assigned|unassigned|assignment\s+(?:failed|not\s+configured))\b", f_low))
        has_explicit_audit_proof = ("audit log" in f_low or "audit trail" in f_low or "system log" in f_low or "server log" in f_low or "security_audit_log" in f_low or "security audit" in f_low or "scada" in f_low) and any(
            w in f_low for w in ("disabled", "bypassed", "failure", "outage", "defeated", "overridden", "override", "never assigned", "not assigned", "unassigned", "crashed", "unconfigured", "open circuit", "fault")
        )

        is_bypass_hyp = any(w in stmt_low for w in ("bypass", "bypassed", "disabled", "defeated", "overridden", "override", "skipped", "without training validation", "without dual authorization"))
        is_training_hyp = any(w in stmt_low for w in ("training", "lms", "competenc")) and not is_bypass_hyp
        is_system_hyp = any(w in stmt_low for w in ("server", "service", "queue", "outage", "crashed", "thermocouple", "sensor", "scada", "instrument", "open circuit", "hardware", "mechanical"))
        is_assignment_hyp = any(w in stmt_low for w in ("assignment", "assigned"))

        if is_bypass_hyp and (has_disabled_control or has_change_mgmt_bypass or (has_explicit_audit_proof and "override" not in f_low and "overridden" not in f_low) or has_unauthorized_action):
            has_verified_causal_proof = True
            proven_causal_level = CausalLevel.L5_SYSTEMIC_CAUSE
            break
        elif is_training_hyp and has_training_proof:
            has_verified_causal_proof = True
            proven_causal_level = CausalLevel.L4_ROOT_CAUSE
            break
        elif is_system_hyp and (has_system_outage or has_instrument_proof or has_explicit_audit_proof):
            has_verified_causal_proof = True
            proven_causal_level = CausalLevel.L2_IMMEDIATE_MECHANISM
            break
        elif is_assignment_hyp and (has_task_assignment_proof or has_explicit_audit_proof):
            has_verified_causal_proof = True
            proven_causal_level = CausalLevel.L4_ROOT_CAUSE
            break
        elif has_disabled_control or has_system_outage or has_instrument_proof or has_change_mgmt_bypass or has_explicit_audit_proof:
            if any(w in stmt_low for w in ("disable", "bypass", "outage", "service failure", "workflow", "authorization", "control", "interlock", "block", "rule", "thermocouple", "sensor", "fault", "hardware", "instrument")):
                has_verified_causal_proof = True
                proven_causal_level = (
                    CausalLevel.L2_IMMEDIATE_MECHANISM if (has_system_outage or has_instrument_proof) and not has_disabled_control
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

    # Only an UNRESOLVED conflict blocks establishment -- a conflict already
    # RESOLVED_FOR/RESOLVED_AGAINST elsewhere in the evidence set is not a
    # reason to collapse an otherwise-legitimate SUPPORTED immediate
    # mechanism to NOT_ESTABLISHED. Mirrors the same status-filtered check
    # already used at nodes/final_evidence_verification.py's
    # has_unresolved_conflict.
    unresolved_conflicts = [c for c in (conflicts or []) if getattr(c, "status", "UNRESOLVED") == "UNRESOLVED"]
    if unresolved_conflicts:
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


# ---------------------------------------------------------------------------
# Epistemic transition tracking & monotonic merge (INV-UNCERTAINTY-005)
# ---------------------------------------------------------------------------

# Ordinal rank so a hypothesis's evidence backing can be compared across two
# core_synthesis passes. CONFLICTING sits below CORROBORATED/VERIFIED
# deliberately: new conflicting evidence is exactly the kind of change that
# is ALLOWED to lower a hypothesis's standing (see merge_candidate_hypotheses).
EVIDENCE_STRENGTH_RANK: dict[str, int] = {
    "NONE": 0,
    "REPORTED": 1,
    "INDICATIVE": 2,
    "CONFLICTING": 2,
    "CORROBORATED": 3,
    "VERIFIED": 4,
}

HYPOTHESIS_STATUS_RANK: dict[str, int] = {
    "UNVERIFIED": 0,
    "UNRESOLVED": 0,
    "REFUTED": 0,
    "POSSIBLE": 1,
    "SUPPORTED": 2,
}


def capture_epistemic_snapshot(root_cause: RootCauseAnalysis | Any, canonical: Any = None) -> dict[str, Any]:
    """Compact, comparable snapshot of the current epistemic state.

    AgentState only ever holds the LATEST root_cause/canonical_finding_state
    -- there is no history. Appending one of these to
    AgentState["epistemic_snapshot_history"] at each point root_cause is
    (re)written gives app.agent.invariants._check_epistemic_status_transitions
    something to actually diff, instead of only inspecting a single snapshot.
    """
    hypotheses: dict[str, dict[str, str]] = {}
    for h in getattr(root_cause, "candidate_hypotheses", None) or []:
        key = getattr(h, "name", None) or getattr(h, "id", None) or ""
        hypotheses[key] = {
            "status": str(getattr(h, "status", "")),
            "evidence_strength": str(getattr(h, "evidence_strength", "NONE")),
        }
    conflicts = getattr(canonical, "evidence_conflicts", None) or []
    return {
        "root_cause_status": str(getattr(root_cause, "status", None)) if root_cause is not None else None,
        "causal_readiness": getattr(canonical, "causal_readiness", None) if canonical is not None else None,
        "unresolved_conflict_ids": sorted(
            getattr(c, "conflict_id", "") for c in conflicts if getattr(c, "status", "UNRESOLVED") == "UNRESOLVED"
        ),
        "hypotheses": hypotheses,
    }


def merge_candidate_hypotheses(
    previous: list[CandidateHypothesis] | None,
    new: list[CandidateHypothesis] | None,
) -> list[CandidateHypothesis]:
    """Monotonic merge guard for the critic-send-back re-investigation loop.

    core_synthesis_node runs a second time when the critic sends the
    investigation back for more evidence; that second pass regenerates
    candidate_hypotheses from scratch and fully replaces the first pass's
    result in AgentState. Nothing prevented a later, lower-confidence
    synthesis from silently under-stating a hypothesis an earlier pass had
    already established (e.g. VERIFIED evidence_strength quietly becoming
    REPORTED because the second LLM call phrased evidence more tentatively).

    This restores a matched hypothesis's evidence_strength/status to the
    higher of the two passes UNLESS the later pass explicitly REFUTED it --
    an explicit refutation is new evidence-driven information, not a
    regression, and is always respected.

    Phase 25 Rule 8 fix: a hypothesis the authoritative evidence-
    reconciliation evaluator already locked (`status_locked=True` --
    app.agent.nodes.evidence_acquisition.reconcile_hypothesis_from_evidence,
    INV-INVEST-028) MUST be carried forward VERBATIM regardless of the
    STATUS_RANK comparison below. That rank-based comparison alone is not
    sufficient: REFUTED and POSSIBLE/UNRESOLVED share the SAME rank tier
    (HYPOTHESIS_STATUS_RANK), so a fresh core_synthesis regeneration
    proposing POSSIBLE for a hypothesis the evidence loop had already
    REFUTED would NOT be caught by "prev_rank > new_rank" -- the freshly
    regenerated (unlocked, rank-1) POSSIBLE would silently outrank and
    overwrite the locked (rank-0) REFUTED, undoing an authoritative,
    evidence-grounded decision on the critic-send-back re-synthesis path.
    This was a real, confirmed defect (not hypothetical) -- reproduced by
    constructing exactly that previous/new pair before this fix.
    """
    if not previous or not new:
        return new or []
    prev_by_key = {(getattr(h, "name", None) or h.id): h for h in previous}
    for h in new:
        key = getattr(h, "name", None) or h.id
        prev = prev_by_key.get(key)
        if prev is None:
            continue
        if getattr(prev, "status_locked", False):
            # The authoritative evaluator already decided this hypothesis's
            # fate from real evidence -- a regenerated proposal from a
            # second core_synthesis pass is not new evidence and must never
            # override it (INV-INVEST-028: exactly one epistemic authority).
            h.status = prev.status
            h.evidence_strength = prev.evidence_strength
            h.status_locked = True
            continue
        if getattr(h, "status", "") == "REFUTED":
            continue
        prev_rank = EVIDENCE_STRENGTH_RANK.get(str(getattr(prev, "evidence_strength", "NONE")), 0)
        new_rank = EVIDENCE_STRENGTH_RANK.get(str(getattr(h, "evidence_strength", "NONE")), 0)
        if prev_rank > new_rank:
            h.evidence_strength = prev.evidence_strength
            if HYPOTHESIS_STATUS_RANK.get(str(prev.status), 0) > HYPOTHESIS_STATUS_RANK.get(str(h.status), 0):
                h.status = prev.status
    return new


# ---------------------------------------------------------------------------
# Phase 3: real runtime CausalGraph construction.
#
# Builds the typed CausalGraph strictly FROM already-epistemically-disciplined
# data: `CandidateHypothesis.status` / `.evidence_strength` (computed by the
# existing eligibility engine in this module and in causal_model.py — never
# re-derived here from raw text) and `RootCauseAnalysis`. It never reads
# SemanticGraph normative edges (VIOLATES/REQUIRES/GOVERNS/...), so normative
# relations structurally cannot leak into a causal edge — there is no code
# path by which they could.
# ---------------------------------------------------------------------------

def _humanize_concept_name(name: str | None) -> str | None:
    """Phase 11 Step 6: turn a controlled-vocabulary hypothesis name
    ("TRAINING_NOT_COMPLETED") into a short, neutral noun phrase
    ("Training Not Completed") for CausalGraphNode.concept_ref.

    Purely mechanical (case/underscore transform) — no domain vocabulary,
    no hedging words added or removed, applies identically to any
    hypothesis name from any domain."""
    if not name:
        return None
    return name.replace("_", " ").strip().title()


def _node_type_for_level(level: CausalLevel) -> Any:
    from app.models.agent import CausalGraphNodeType
    return {
        CausalLevel.L0_OBSERVATION: CausalGraphNodeType.OBSERVED_DEVIATION,
        CausalLevel.EVIDENCE_STATE: CausalGraphNodeType.OBSERVED_DEVIATION,
        CausalLevel.L1_EVENT: CausalGraphNodeType.IMMEDIATE_MECHANISM,
        CausalLevel.L2_IMMEDIATE_MECHANISM: CausalGraphNodeType.IMMEDIATE_MECHANISM,
        CausalLevel.L2_REPORTED_MECHANISM: CausalGraphNodeType.IMMEDIATE_MECHANISM,
        CausalLevel.L3_CONTRIBUTING_CAUSE: CausalGraphNodeType.CONTRIBUTING_FACTOR,
        CausalLevel.L3_IMMEDIATE_MECHANISM: CausalGraphNodeType.CONTRIBUTING_FACTOR,
        CausalLevel.L4_ROOT_CAUSE: CausalGraphNodeType.UNDERLYING_CAUSE,
        CausalLevel.L5_SYSTEMIC_CAUSE: CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE,
    }.get(level, CausalGraphNodeType.UNKNOWN)


# Evidence-strength -> node epistemic status. Strictly a lookup over values
# the eligibility engine already computed — never re-inferred from text.
def _epistemic_status_for_strength(strength: str) -> Any:
    from app.models.agent import EvidenceStatus
    return {
        "VERIFIED": EvidenceStatus.VERIFIED,
        "CORROBORATED": EvidenceStatus.VERIFIED,
        "REPORTED": EvidenceStatus.REPORTED,
        "INDICATIVE": EvidenceStatus.INFERRED,
        "CONFLICTING": EvidenceStatus.CONTRADICTED,
        "NONE": EvidenceStatus.UNKNOWN,
    }.get(strength, EvidenceStatus.UNKNOWN)


def _edge_status_for(status: str, strength: str) -> Any:
    """Section 6/8 edge licensing decision. Returns None when no licensed
    edge may be created — the caller must NOT invent one in that case."""
    from app.models.agent import CausalGraphEdgeStatus
    if strength == "CONFLICTING":
        return CausalGraphEdgeStatus.DISPUTED
    if status == "REFUTED":
        return CausalGraphEdgeStatus.REJECTED
    if status not in ("SUPPORTED", "POSSIBLE"):
        # UNVERIFIED / UNRESOLVED candidate: not yet licensed for any edge.
        return None
    if strength in ("VERIFIED", "CORROBORATED") and status == "SUPPORTED":
        return CausalGraphEdgeStatus.VERIFIED
    if strength == "REPORTED":
        return CausalGraphEdgeStatus.REPORTED
    if status == "POSSIBLE" or strength == "INDICATIVE":
        return CausalGraphEdgeStatus.POSSIBLE
    if strength == "NONE":
        # A SUPPORTED status with zero evidence backing is not licensable —
        # this should not happen given the upstream eligibility engine, but
        # the builder must fail closed (no edge) rather than trust the label.
        return None
    return CausalGraphEdgeStatus.POSSIBLE


def build_causal_graph(
    canonical_state: Any,
    root_cause: Any,
    evidence_ledger: list[EvidenceItem] | None = None,
) -> Any:
    """Build the real runtime CausalGraph from canonical state + root cause.

    Always contains at least an OBSERVED_DEVIATION node when
    `canonical_state.observed_deviation` (or `.deviation`) is set — the
    directly-observed finding is definitionally VERIFIED audit observation.
    Hypothesis nodes/edges are added only when the existing eligibility
    engine has already licensed them (Section 6) — this function performs
    NO new causal judgment of its own, it only projects already-computed
    epistemic state into explicit graph structure.
    """
    from app.models.agent import (
        CausalGraph,
        CausalGraphEdge,
        CausalGraphNode,
        CausalGraphNodeType,
        EpistemicSource,
        EvidenceStatus,
    )

    nodes: list[CausalGraphNode] = []
    edges: list[CausalGraphEdge] = []

    deviation_label = (
        getattr(canonical_state, "observed_deviation", None)
        or getattr(canonical_state, "deviation", None)
        or getattr(canonical_state, "finding_subject", None)
    ) if canonical_state else None
    if not deviation_label:
        return CausalGraph(nodes=[], edges=[])

    # Sanity backstop for the VERIFIED edge grade (Section 6: "an edge may
    # only be created when supported by an allowed source"): if the ledger
    # this run actually collected contains no VERIFIED item at all, no edge
    # in this graph may claim VERIFIED status regardless of what a
    # hypothesis's own evidence_strength label says — a label is not itself
    # evidence.
    _ledger = evidence_ledger or []
    _ledger_has_verified = any(
        str(getattr(item, "status", "")) in ("EvidenceStatus.VERIFIED", "VERIFIED")
        for item in _ledger
    )

    # Phase 12 Step 4/6: carry the finding's own typed comparison/
    # measurement fields (already extracted by understanding.py — never
    # re-derived here) onto the deviation node so the graph-derived
    # renderer has access to them without falling back to raw prose.
    _measurement = getattr(canonical_state, "measurement", None)
    _comparison_kwargs = dict(
        comparison_type=getattr(canonical_state, "comparison_type", None),
        comparison_left=getattr(canonical_state, "comparison_left", None),
        comparison_left_qualifier=getattr(canonical_state, "comparison_left_qualifier", None),
        comparison_right=getattr(canonical_state, "comparison_right", None),
        comparison_subtype=getattr(canonical_state, "comparison_subtype", None),
        measurement_value=getattr(_measurement, "value", None) if _measurement else None,
        measurement_unit=getattr(_measurement, "unit", None) if _measurement else None,
        measurement_qualifier=getattr(_measurement, "qualifier", None) if _measurement else None,
        measurement_evidence_status=getattr(_measurement, "evidence_status", None) if _measurement else None,
    )

    deviation_node_id = "CN_DEVIATION"
    nodes.append(CausalGraphNode(
        node_id=deviation_node_id,
        node_type=CausalGraphNodeType.OBSERVED_DEVIATION,
        label=deviation_label,
        causal_level=CausalLevel.L0_OBSERVATION,
        epistemic_status=EvidenceStatus.VERIFIED,
        provenance=EpistemicSource.AUDIT_OBSERVATION,
        **_comparison_kwargs,
    ))

    hypotheses = list(getattr(root_cause, "candidate_hypotheses", None) or [])
    n_idx = 1
    e_idx = 1
    # Phase 4 Section 7 (multi-hop): tracks (node, hypothesis, edge) triples
    # as they're created so a deterministic post-pass can re-parent a
    # deeper-level hypothesis's edge onto a shallower-level hypothesis node
    # when the evidence justifies it, instead of leaving every hypothesis a
    # direct, parallel child of the deviation.
    _created: list[tuple[Any, Any, Any]] = []
    for h in hypotheses:
        status = str(getattr(h, "status", "POSSIBLE"))
        strength = str(getattr(h, "evidence_strength", "NONE"))
        # Section 8: not every hypothesis under consideration is licensed to
        # exist as a graph node — REFUTED and zero-evidence UNVERIFIED
        # candidates are not part of the causal chain at all.
        if status in ("REFUTED", "UNVERIFIED") and strength == "NONE":
            continue

        level = getattr(h, "causal_level", CausalLevel.L3_IMMEDIATE_MECHANISM)
        node_id = f"CN{n_idx}"
        n_idx += 1
        nodes.append(CausalGraphNode(
            node_id=node_id,
            node_type=_node_type_for_level(level),
            label=getattr(h, "statement", "") or getattr(h, "name", ""),
            causal_level=level,
            epistemic_status=_epistemic_status_for_strength(strength),
            proposition_ids=list(getattr(h, "supporting_claim_ids", None) or []),
            evidence_ids=list(getattr(h, "supporting_evidence", None) or []),
            provenance=(
                EpistemicSource.OBJECTIVE_RECORD if strength in ("VERIFIED", "CORROBORATED")
                else EpistemicSource.REPORTED_STATEMENT if strength == "REPORTED"
                else EpistemicSource.INFERRED if strength == "INDICATIVE"
                else EpistemicSource.UNKNOWN_SOURCE
            ),
            confidence=getattr(h, "confidence", None),
            source_hypothesis_id=getattr(h, "id", None),
            concept_ref=_humanize_concept_name(getattr(h, "name", None)),
        ))

        # Phase 6 Section 3: stamp the hypothesis with its real graph
        # back-reference. This is the ONLY place CandidateHypothesis.
        # causal_node_id is ever set — never by the LLM, never elsewhere.
        if hasattr(h, "causal_node_id"):
            h.causal_node_id = node_id

        edge_status = _edge_status_for(status, strength)
        if edge_status is None:
            # Section 8: node exists (a candidate under consideration) but no
            # edge is licensed yet — this IS the evidence boundary, not a bug.
            _created.append((nodes[-1], h, None))
            continue
        from app.models.agent import CausalGraphEdgeStatus as _CGES
        _downgrade_note = ""
        if edge_status == _CGES.VERIFIED and not _ledger_has_verified:
            edge_status = _CGES.POSSIBLE
            _downgrade_note = " [downgraded: no VERIFIED item present in this run's evidence_ledger]"
        new_edge = CausalGraphEdge(
            edge_id=f"CE{e_idx}",
            source_node_id=deviation_node_id,
            target_node_id=node_id,
            evidence_ids=list(getattr(h, "supporting_evidence", None) or []),
            proposition_ids=list(getattr(h, "supporting_claim_ids", None) or []),
            provenance=(
                EpistemicSource.OBJECTIVE_RECORD if strength in ("VERIFIED", "CORROBORATED")
                else EpistemicSource.REPORTED_STATEMENT if strength == "REPORTED"
                else EpistemicSource.UNKNOWN_SOURCE
            ),
            confidence=getattr(h, "confidence", None),
            causal_level_transition=f"OBSERVED_DEVIATION->{_node_type_for_level(level).value}",
            status=edge_status,
            notes=(
                f"Derived from CandidateHypothesis {getattr(h, 'id', '?')} "
                f"(status={status}, evidence_strength={strength}){_downgrade_note}"
            ),
        )
        edges.append(new_edge)
        _created.append((nodes[-1], h, new_edge))
        if hasattr(h, "causal_edge_id"):
            h.causal_edge_id = new_edge.edge_id
        if hasattr(h, "causal_edge_source_node_id"):
            h.causal_edge_source_node_id = new_edge.source_node_id
        e_idx += 1

    _chain_explicit_causal_edges(_created)
    _chain_multihop_causal_edges(_created)

    return CausalGraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Phase 13 Step 12: deterministic single-hop 5-Why authority gate.
#
# Phase 12 Step 10 widened the graph-authoritative 5-Why swap to also fire on
# plain DIRECT (single-hop) edges and measured 60 regressions. Root-causing
# them (Phase 13 Step 1/2) found they fall into exactly three structural
# categories, none of them fixable by touching the 5-Why renderer alone:
#
#   1. Evidence-boundary padding (INV-WHY-007) -- fixed at the source in
#      causal_graph_traversal.build_graph_grounded_five_why (a POSSIBLE
#      single-hop transition step already IS the boundary; it must not be
#      followed by a second, duplicate marker step).
#   2. Competing-hypothesis collapse -- when more than one hypothesis is a
#      live, independent sibling of the observed deviation (no re-parenting
#      chained them together), the graph walk silently picks ONE via its
#      deterministic tie-break and discards the others, while the existing
#      prose generator represents each sibling as its own Why step. This is
#      exactly Section 10's "do not manufacture a winner": authority must not
#      activate while more than one hypothesis is still an independent
#      sibling of the deviation.
#   3. Structural information loss -- a directly-stated VERIFIED
#      immediate_mechanism (Section 2 of CanonicalFindingState) that has no
#      corresponding node in the causal graph (the graph is built only from
#      `root_cause.candidate_hypotheses`, never from `immediate_mechanism`
#      directly) would silently vanish from the rendered chain if the graph
#      became authoritative, and a semantic_type requiring transition-
#      specific phrasing (EVENT_SEQUENCE_CONTROL) that the generic graph
#      question/answer templates do not yet reproduce would lose real
#      information the deterministic generator states. Both are refused
#      rather than silently degraded.
#
# This predicate is the SINGLE place all three checks live -- it subsumes and
# replaces the old ad hoc `_has_multihop_chain` check (a multi-hop EXPLICIT/
# EVIDENCE_CORRELATED chain always re-parents every deeper edge off of
# `deviation`, leaving exactly one edge still sourced there, so condition (1)
# below is a strict generalization of the old gate, not a narrowing of it).
# ---------------------------------------------------------------------------


def is_graph_authoritative_for_five_why(causal_graph: Any, canonical_state: Any = None) -> tuple[bool, str]:
    """Deterministic predicate: may `build_graph_grounded_five_why`'s output
    replace the prose-generated 5-Why for this run?

    Returns (authoritative, reason). `reason` is always populated (even when
    authoritative=True) so callers can trace/log why the gate did or did not
    open, per Section 12's "no source files changed / no silent judgment"
    discipline.
    """
    from app.models.agent import CausalGraphNodeType

    if not causal_graph or not causal_graph.edges:
        return False, "no licensed causal edge exists"

    deviation_nodes = [n for n in causal_graph.nodes if n.node_type == CausalGraphNodeType.OBSERVED_DEVIATION]
    if not deviation_nodes:
        return False, "no observed-deviation node"
    deviation_id = deviation_nodes[0].node_id

    # (1) Exactly one independent branch off the deviation. A genuine
    # multi-hop chain (EXPLICIT/EVIDENCE_CORRELATED) already re-parents every
    # deeper edge away from the deviation node, so it naturally satisfies
    # this too -- there is no separate "is it multi-hop" branch anymore.
    root_edges = [e for e in causal_graph.edges if e.source_node_id == deviation_id]
    if len(root_edges) == 0:
        return False, "no edge licensed from the observed deviation"
    if len(root_edges) > 1:
        return False, (
            f"{len(root_edges)} independent hypotheses remain siblings of the observed deviation "
            "(competing, unresolved) -- selecting one would manufacture a winner"
        )

    if canonical_state is not None:
        # (3a) A VERIFIED, directly-stated immediate mechanism distinct from
        # the observation itself must already be represented as a VERIFIED
        # graph node, or authority does not activate (it would otherwise be
        # silently dropped from the rendered chain).
        imm = getattr(canonical_state, "immediate_mechanism", None)
        imm_status = getattr(canonical_state, "immediate_mechanism_status", "UNKNOWN")
        deviation_label = (getattr(canonical_state, "observed_deviation", None) or getattr(canonical_state, "deviation", None) or "")
        from app.services.text_grounding import significant_words
        imm_words = significant_words(imm) if imm else set()
        deviation_words = significant_words(deviation_label) if deviation_label else set()
        # "Distinct from the observation" means substantively distinct, not
        # merely a different string -- the same fact is routinely restated
        # in different phrasing between `observed_deviation` (the
        # semantic-subject-normalized label) and `immediate_mechanism` (the
        # raw stated-mechanism sentence). A majority word-overlap is the
        # same overlap threshold `_five_why_step_matches_statement` already
        # uses for the equivalent "is this the same proposition" judgment
        # elsewhere in this codebase (app.agent.invariants).
        _same_as_deviation = bool(imm_words and deviation_words and len(imm_words & deviation_words) >= max(2, len(imm_words) // 2))
        if imm and imm_status == "VERIFIED" and not _same_as_deviation:
            from app.models.agent import EvidenceStatus
            represented = any(
                n.epistemic_status == EvidenceStatus.VERIFIED
                and imm_words
                and len(imm_words & significant_words(n.label)) >= max(2, len(imm_words) // 2)
                for n in causal_graph.nodes
                if n.node_type != CausalGraphNodeType.OBSERVED_DEVIATION
            )
            if not represented:
                return False, (
                    f"verified immediate mechanism {imm!r} is not represented as a VERIFIED node in the "
                    "causal graph -- activating authority would silently drop it"
                )

        # (3b) Semantic types whose deterministic 5-Why phrasing encodes
        # structured fields (transition_type, control_justification_missing,
        # ...) the generic graph question/answer templates do not yet
        # reproduce. A disclosed residual gap (same category as the
        # comparison/measurement gap Phase 12 closed for comparison_type) --
        # not a domain keyword, a controlled architectural taxonomy value.
        semantic_type = getattr(canonical_state, "semantic_type", None)
        if semantic_type == "EVENT_SEQUENCE_CONTROL":
            return False, "semantic_type=EVENT_SEQUENCE_CONTROL requires transition-specific phrasing not yet in the graph templates"

    return True, "exactly one licensed branch from the observed deviation; no represented information would be lost"


def _causal_level_depth(level: Any) -> int:
    return {
        CausalLevel.L0_OBSERVATION: 0, CausalLevel.EVIDENCE_STATE: 0,
        CausalLevel.L1_EVENT: 1, CausalLevel.L2_IMMEDIATE_MECHANISM: 2,
        CausalLevel.L2_REPORTED_MECHANISM: 2, CausalLevel.L3_CONTRIBUTING_CAUSE: 3,
        CausalLevel.L3_IMMEDIATE_MECHANISM: 3, CausalLevel.L4_ROOT_CAUSE: 4,
        CausalLevel.L5_SYSTEMIC_CAUSE: 5,
    }.get(level, 3)


def _chain_explicit_causal_edges(created: list[tuple[Any, Any, Any]]) -> None:
    """Phase 7 Section 5: EXPLICIT sequential chaining from an asserted
    `deepens_hypothesis_id`.

    This is the PRIMARY chaining mechanism — a real structured causal
    candidate (source hypothesis id -> this hypothesis), validated against
    the same eligibility gate every hypothesis already passed through, not
    inferred from incidental evidence overlap. Runs BEFORE the evidence-
    overlap heuristic so an explicit assertion always wins when present.

    A dangling/self-referential/cyclic reference is silently ignored (the
    edge stays sourced from the deviation) — Section 26: never fake the
    feature when the referenced structure doesn't actually exist.
    """
    edged = [(node, h, edge) for node, h, edge in created if edge is not None]
    node_by_hyp_id = {getattr(h, "id", None): node for node, h, edge in edged}
    for deep_node, deep_h, deep_edge in edged:
        target_hyp_id = getattr(deep_h, "deepens_hypothesis_id", None)
        if not target_hyp_id or target_hyp_id == getattr(deep_h, "id", None):
            continue
        source_node = node_by_hyp_id.get(target_hyp_id)
        if source_node is None:
            continue  # dangling reference — not licensed, ignore
        # Reject a 2-cycle (A deepens B, B deepens A) — never create a loop.
        source_hyp_deepens = getattr(
            next((h for n, h, e in edged if n.node_id == source_node.node_id), None),
            "deepens_hypothesis_id", None,
        )
        if source_hyp_deepens == getattr(deep_h, "id", None):
            continue
        deep_edge.source_node_id = source_node.node_id
        deep_edge.causal_level_transition = f"{source_node.node_type.value}->{deep_node.node_type.value}"
        deep_edge.derivation = "EXPLICIT"
        deep_edge.notes = (deep_edge.notes or "") + (
            f" [explicit chain: asserted deepens_hypothesis_id={target_hyp_id!r}]"
        )
        if hasattr(deep_h, "causal_edge_source_node_id"):
            deep_h.causal_edge_source_node_id = source_node.node_id


def _chain_multihop_causal_edges(created: list[tuple[Any, Any, Any]]) -> None:
    """Phase 4 Section 7: deterministic, evidence-grounded multi-hop
    re-parenting.

    For each hypothesis-with-an-edge H_deep, if EXACTLY ONE other
    hypothesis-with-an-edge H_shallow in this same graph is at a
    strictly shallower causal level AND shares at least one
    supporting_claim_id with H_deep, H_deep's edge is re-sourced from
    H_shallow's node instead of the deviation — producing a genuine
    connected multi-hop chain (OBSERVATION -> MECHANISM -> ROOT_CAUSE)
    grounded in shared evidence, not in generation order or causal_level
    alone. When zero or more-than-one shallower candidate qualifies, the
    edge is left sourced from the deviation (no ambiguous chaining is
    ever manufactured) — this is the deliberate fail-safe default.

    Phase 7: skips any edge Phase 7's EXPLICIT pass already re-parented —
    an asserted transition is never overridden by an inferred one.
    """
    edged = [(node, h, edge) for node, h, edge in created if edge is not None]
    for deep_node, deep_h, deep_edge in edged:
        if deep_edge.derivation == "EXPLICIT":
            continue
        deep_claims = set(getattr(deep_h, "supporting_claim_ids", None) or [])
        if not deep_claims:
            continue
        deep_depth = _causal_level_depth(getattr(deep_h, "causal_level", None))
        candidates = []
        for shallow_node, shallow_h, shallow_edge in edged:
            if shallow_node.node_id == deep_node.node_id:
                continue
            shallow_depth = _causal_level_depth(getattr(shallow_h, "causal_level", None))
            if shallow_depth >= deep_depth:
                continue
            shallow_claims = set(getattr(shallow_h, "supporting_claim_ids", None) or [])
            if deep_claims & shallow_claims:
                candidates.append(shallow_node)
        if len(candidates) == 1:
            deep_edge.source_node_id = candidates[0].node_id
            deep_edge.causal_level_transition = f"{candidates[0].node_type.value}->{deep_node.node_type.value}"
            deep_edge.derivation = "EVIDENCE_CORRELATED"
            deep_edge.notes = (deep_edge.notes or "") + (
                f" [multi-hop: re-parented from deviation onto {candidates[0].node_id} "
                "based on shared supporting evidence]"
            )
            if hasattr(deep_h, "causal_edge_source_node_id"):
                deep_h.causal_edge_source_node_id = candidates[0].node_id


# ---------------------------------------------------------------------------
# Phase 4 Section 3: pre-investigation CausalUncertaintyGraph.
#
# Built BEFORE core_synthesis runs — from CanonicalFindingState.semantic_graph
# alone, which IS available at investigation-planning time (unlike
# CandidateHypothesis data, which only exists after synthesis). This is what
# makes graph-grounded investigation planning possible at all: a
# post-synthesis CausalGraph cannot drive planning because planning runs
# first (see Phase 4 report, "Primary Remaining Problem").
#
# Reuses the CausalGraph/CausalGraphNode/CausalGraphEdge models rather than
# introducing a parallel type family (Section 3 explicitly permits this).
# Every edge here is CausalGraphEdgeStatus.UNKNOWN by construction — this
# function invents NO causal relationship, only surfaces which semantic-graph
# facts are structurally unexplained. That is the entire contract.
# ---------------------------------------------------------------------------

# Normative relation types (already-established, not causal) whose presence
# on a semantic edge marks its target as "a fact requiring causal
# explanation" — the deviation from normal is on the record, but WHY it
# happened is, definitionally, not yet known before evidence is gathered.
_UNCERTAINTY_TRIGGER_RELATIONS = frozenset({
    "VIOLATES", "NOT_PERFORMED_AS_REQUIRED", "LACKS_REQUIRED_ATTRIBUTE",
    "NOT_DEMONSTRATED", "INCONSISTENT_WITH", "OUTSIDE_REQUIREMENT",
})


def build_causal_uncertainty_graph(canonical_state: Any) -> Any:
    """Build the pre-investigation CausalUncertaintyGraph from the semantic
    graph alone. Every non-deviation node is CausalGraphNodeType.UNRESOLVED
    with an incoming UNKNOWN-status edge — there is no VERIFIED/POSSIBLE
    edge in this graph by construction, because no evidence beyond the raw
    finding text has been gathered yet.
    """
    from app.models.agent import (
        CausalGraph,
        CausalGraphEdge,
        CausalGraphEdgeStatus,
        CausalGraphNode,
        CausalGraphNodeType,
        EpistemicSource,
        EvidenceStatus,
    )

    deviation_label = (
        getattr(canonical_state, "observed_deviation", None)
        or getattr(canonical_state, "deviation", None)
        or getattr(canonical_state, "finding_subject", None)
    ) if canonical_state else None
    if not deviation_label:
        return CausalGraph(nodes=[], edges=[])

    deviation_node_id = "UN_DEVIATION"
    nodes = [CausalGraphNode(
        node_id=deviation_node_id,
        node_type=CausalGraphNodeType.OBSERVED_DEVIATION,
        label=deviation_label,
        causal_level=CausalLevel.L0_OBSERVATION,
        epistemic_status=EvidenceStatus.VERIFIED,
        provenance=EpistemicSource.AUDIT_OBSERVATION,
    )]
    edges: list[Any] = []

    sem_graph = getattr(canonical_state, "semantic_graph", None)
    sem_edges = list(getattr(sem_graph, "edges", None) or []) if sem_graph else []
    sem_node_by_id = {n.id: n for n in (getattr(sem_graph, "nodes", None) or [])} if sem_graph else {}

    seen_targets: set[str] = set()
    n_idx = 1
    e_idx = 1
    for se in sem_edges:
        rel = str(getattr(se.relation_type, "value", se.relation_type))
        if rel not in _UNCERTAINTY_TRIGGER_RELATIONS:
            continue
        target = sem_node_by_id.get(se.target_id)
        if target is None or target.id in seen_targets:
            continue
        seen_targets.add(target.id)
        node_id = f"UNK{n_idx}"
        n_idx += 1
        nodes.append(CausalGraphNode(
            node_id=node_id,
            node_type=CausalGraphNodeType.UNRESOLVED,
            label=target.label,
            causal_level=CausalLevel.EVIDENCE_STATE,
            epistemic_status=EvidenceStatus.UNKNOWN,
            semantic_node_ids=[target.id],
            provenance=EpistemicSource.AUDIT_OBSERVATION,
        ))
        edges.append(CausalGraphEdge(
            edge_id=f"UNE{e_idx}",
            source_node_id=deviation_node_id,
            target_node_id=node_id,
            epistemic_status=EvidenceStatus.UNKNOWN,
            proposition_ids=list(getattr(se, "source_claim_ids", None) or []),
            provenance=EpistemicSource.AUDIT_OBSERVATION,
            causal_level_transition="OBSERVED_DEVIATION->UNRESOLVED",
            status=CausalGraphEdgeStatus.UNKNOWN,
            notes=(
                f"Unresolved: semantic relation {rel!r} establishes a normative fact about "
                f"{target.label!r}, but no causal mechanism connecting it to the observed "
                "deviation has been evidenced yet."
            ),
        ))
        e_idx += 1

    return CausalGraph(nodes=nodes, edges=edges)


def information_gain_band_for_edge(edge: Any) -> tuple[str, str]:
    """Phase 14 Section 5: deterministic ordinal information-gain
    classification for a single causal-uncertainty edge, using the SAME
    structural signal `rank_uncertainty_nodes_by_information_gain` already
    ranks by (number of proposition_ids backing the edge) -- more
    corroborating structural evidence pointing at an uncertainty means
    resolving it narrows more of the graph. Never a fabricated probability;
    never lexical similarity or question length.

    Returns (band, reason) where band is one of HIGH/MEDIUM/LOW.
    """
    prop_count = len(getattr(edge, "proposition_ids", None) or [])
    if prop_count >= 2:
        return "HIGH", f"{prop_count} proposition(s) structurally corroborate this unresolved edge"
    if prop_count == 1:
        return "MEDIUM", "exactly one proposition corroborates this unresolved edge"
    return "LOW", "no corroborating proposition is attached to this unresolved edge"


def rank_uncertainty_nodes_by_information_gain(uncertainty_graph: Any) -> list[Any]:
    """Phase 4 Section 5: deterministic discrimination-priority ranking over
    the unresolved nodes of a CausalUncertaintyGraph.

    NOT a quantitative information-gain calculation — the architecture has
    no probability model to justify one (Section 5 explicitly prohibits
    fabricating one). Instead a deterministic ranking using only data the
    graph actually has:
      1. number of source_claim_ids backing the underlying semantic relation
         (more corroborating structural evidence -> resolving it first
         narrows more of the graph)
      2. tie-break: node_id (stable, deterministic ordering)
    Returns nodes sorted highest-priority first; ties broken deterministically.
    """
    if not uncertainty_graph or not uncertainty_graph.edges:
        return []
    node_by_id = {n.node_id: n for n in uncertainty_graph.nodes}
    unresolved_edges = [e for e in uncertainty_graph.edges if str(e.status.value if hasattr(e.status, "value") else e.status) == "UNKNOWN"]

    def _rank_key(edge: Any) -> tuple[int, str]:
        return (-len(edge.proposition_ids), edge.target_node_id)

    ranked_edges = sorted(unresolved_edges, key=_rank_key)
    return [node_by_id[e.target_node_id] for e in ranked_edges if e.target_node_id in node_by_id]


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



# ---------------------------------------------------------------------------
# Phase 6 Section 11: RCA as a pure CausalGraph projection.
# ---------------------------------------------------------------------------


def build_rca_from_causal_graph(causal_graph: Any) -> Any:
    """Build a CausalGraphRCAProjection — a structural read of `causal_graph`,
    grouping nodes by type/level and edges by resolution status. Contains no
    generated text: every field is a graph reference. Competing hypotheses
    are nodes at the SAME causal_level that are both still viable (neither
    REJECTED) — the projection deliberately does not pick a winner; that
    remains the caller's job, done structurally (see `select_authoritative_
    leading_hypothesis`), never by this function.
    """
    from app.models.agent import (
        CausalGraphEdgeStatus,
        CausalGraphNodeType,
        CausalGraphRCAProjection,
        RCACausalEdgeRef,
        RCACausalNodeRef,
    )

    if not causal_graph or not causal_graph.nodes:
        return CausalGraphRCAProjection()

    def _ref(n: Any) -> Any:
        return RCACausalNodeRef(
            causal_node_id=n.node_id,
            label=n.label,
            causal_level=str(n.causal_level.value if hasattr(n.causal_level, "value") else n.causal_level),
            epistemic_status=str(n.epistemic_status.value if hasattr(n.epistemic_status, "value") else n.epistemic_status),
            evidence_ids=list(n.evidence_ids),
            proposition_ids=list(n.proposition_ids),
        )

    deviation = next((n for n in causal_graph.nodes if n.node_type == CausalGraphNodeType.OBSERVED_DEVIATION), None)
    by_type: dict[Any, list[Any]] = {}
    for n in causal_graph.nodes:
        if n.node_type == CausalGraphNodeType.OBSERVED_DEVIATION:
            continue
        by_type.setdefault(n.node_type, []).append(n)

    # Competing hypotheses: >1 node sharing a causal_level, neither backed
    # by a REJECTED edge — a genuine tie the projection must not resolve.
    level_groups: dict[str, list[Any]] = {}
    edge_by_target = {e.target_node_id: e for e in causal_graph.edges}
    for n in causal_graph.nodes:
        if n.node_type == CausalGraphNodeType.OBSERVED_DEVIATION:
            continue
        edge = edge_by_target.get(n.node_id)
        if edge is not None and edge.status == CausalGraphEdgeStatus.REJECTED:
            continue
        level_groups.setdefault(str(n.causal_level), []).append(n)
    competing = [n for group in level_groups.values() if len(group) > 1 for n in group]

    unresolved_edges = [
        RCACausalEdgeRef(
            causal_edge_id=e.edge_id, source_node_id=e.source_node_id,
            target_node_id=e.target_node_id,
            status=str(e.status.value if hasattr(e.status, "value") else e.status),
            evidence_ids=list(e.evidence_ids),
        )
        for e in causal_graph.edges if e.status == CausalGraphEdgeStatus.UNKNOWN
    ]

    root_established = any(
        e.status == CausalGraphEdgeStatus.VERIFIED
        and edge_by_target.get(e.target_node_id) is e
        and any(
            n.node_id == e.target_node_id
            and n.node_type in (CausalGraphNodeType.UNDERLYING_CAUSE, CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE)
            for n in causal_graph.nodes
        )
        for e in causal_graph.edges
    )

    return CausalGraphRCAProjection(
        observed_deviation=_ref(deviation) if deviation else None,
        immediate_mechanisms=[_ref(n) for n in by_type.get(CausalGraphNodeType.IMMEDIATE_MECHANISM, [])],
        contributing_factors=[_ref(n) for n in by_type.get(CausalGraphNodeType.CONTRIBUTING_FACTOR, [])],
        underlying_causes=[_ref(n) for n in by_type.get(CausalGraphNodeType.UNDERLYING_CAUSE, [])],
        systemic_root_causes=[_ref(n) for n in by_type.get(CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE, [])],
        competing_hypotheses=[_ref(n) for n in competing],
        unresolved_edges=unresolved_edges,
        evidence_boundary_reached=bool(unresolved_edges) or any(
            n.node_type != CausalGraphNodeType.OBSERVED_DEVIATION and n.node_id not in edge_by_target
            for n in causal_graph.nodes
        ),
        root_cause_status="ESTABLISHED" if root_established else "NOT_ESTABLISHED",
    )


# ---------------------------------------------------------------------------
# Phase 6 Section 14: Impact as a pure CausalGraph + EvidenceLedger projection.
# ---------------------------------------------------------------------------


def build_impact_from_graph(canonical_state: Any, causal_graph: Any) -> Any:
    """Build a CausalGraphImpactProjection.

    Never infers impact from the finding's wording/severity. OBSERVED is
    exactly the verified deviation node (the one fact already established
    without inference). POTENTIAL is every causal node whose mechanism is
    POSSIBLE/REPORTED (Phase 6 Section 14: "must identify the causal basis
    and uncertainty" — the node's own epistemic_status IS that basis).
    UNKNOWN is every unresolved edge, each surfaced as something requiring
    targeted investigation rather than silently dropped.
    """
    from app.models.agent import (
        CausalGraphEdgeStatus,
        CausalGraphImpactProjection,
        CausalGraphNodeType,
        EvidenceStatus,
        ImpactGraphRef,
    )

    if not causal_graph or not causal_graph.nodes:
        return CausalGraphImpactProjection()

    observed: list[Any] = []
    potential: list[Any] = []
    unknown: list[Any] = []

    for n in causal_graph.nodes:
        if n.node_type == CausalGraphNodeType.OBSERVED_DEVIATION:
            observed.append(ImpactGraphRef(
                description=n.label, causal_node_id=n.node_id,
                evidence_ids=list(n.evidence_ids), basis="OBSERVED",
            ))
        elif n.epistemic_status in (EvidenceStatus.REPORTED, EvidenceStatus.INFERRED, EvidenceStatus.UNVERIFIED):
            potential.append(ImpactGraphRef(
                description=n.label, causal_node_id=n.node_id,
                evidence_ids=list(n.evidence_ids), basis="POTENTIAL",
            ))
        elif n.epistemic_status == EvidenceStatus.VERIFIED:
            observed.append(ImpactGraphRef(
                description=n.label, causal_node_id=n.node_id,
                evidence_ids=list(n.evidence_ids), basis="OBSERVED",
            ))

    for e in causal_graph.edges:
        if e.status == CausalGraphEdgeStatus.UNKNOWN:
            unknown.append(ImpactGraphRef(
                description=f"Unresolved causal relationship ({e.source_node_id} -> {e.target_node_id})",
                causal_edge_id=e.edge_id, evidence_ids=list(e.evidence_ids), basis="UNKNOWN",
            ))

    return CausalGraphImpactProjection(
        observed=observed, potential=potential, unknown_investigation_required=unknown,
    )


# ---------------------------------------------------------------------------
# Phase 9 Step 6: real CausalPath objects — a pure derivation over
# already-licensed CausalGraph edges, never constructed from prose or
# fabricated to increase graph density.
# ---------------------------------------------------------------------------

_EDGE_STATUS_STRENGTH_RANK: dict[str, int] = {
    "VERIFIED": 0, "REPORTED": 1, "POSSIBLE": 2, "UNKNOWN": 3, "DISPUTED": 4, "REJECTED": 5,
}


def build_causal_paths(causal_graph: Any) -> list[Any]:
    """For every node that has exactly one incoming causal edge, walk
    backward to the OBSERVED_DEVIATION and materialize the real traversed
    chain as a CausalPath. Competing/independent hypotheses each get their
    OWN path — this function never merges two hypotheses' chains and never
    invents a path for a node with no incoming edge (an unresolved
    candidate correctly gets no path, not a fabricated one).

    Structural validation performed during the walk (defense in depth on
    top of INV-CGRAPH-008's construction-time cycle rejection):
      - a node visited twice aborts that path (cycle guard)
      - a dangling source (edge whose source node doesn't exist) aborts
        that path rather than silently truncating it
    """
    from app.models.agent import CausalGraphNodeType, CausalPath

    if not causal_graph or not causal_graph.nodes:
        return []

    node_by_id = {n.node_id: n for n in causal_graph.nodes}
    edge_by_target: dict[str, list[Any]] = {}
    for e in causal_graph.edges:
        edge_by_target.setdefault(e.target_node_id, []).append(e)

    deviation_nodes = [n for n in causal_graph.nodes if n.node_type == CausalGraphNodeType.OBSERVED_DEVIATION]
    if not deviation_nodes:
        return []
    deviation_id = deviation_nodes[0].node_id

    paths: list[Any] = []
    p_idx = 1
    for node in causal_graph.nodes:
        if node.node_id == deviation_id:
            continue
        incoming = edge_by_target.get(node.node_id, [])
        if len(incoming) != 1:
            # Zero incoming edges: an unresolved candidate — no path exists,
            # by design. More than one: not producible by the current
            # builder (each node has at most one licensed incoming edge),
            # so treat defensively as ambiguous and skip rather than guess.
            continue

        ordered_node_ids = [node.node_id]
        ordered_edge_ids: list[str] = []
        supporting: list[str] = []
        contradicting: list[str] = []
        weakest_rank = -1
        current_id = node.node_id
        visited = {node.node_id}
        valid = True
        while current_id != deviation_id:
            edges_in = edge_by_target.get(current_id, [])
            if len(edges_in) != 1:
                valid = False
                break
            edge = edges_in[0]
            ordered_edge_ids.append(edge.edge_id)
            supporting.extend(edge.evidence_ids)
            status_str = str(edge.status.value if hasattr(edge.status, "value") else edge.status)
            weakest_rank = max(weakest_rank, _EDGE_STATUS_STRENGTH_RANK.get(status_str, 3))
            source_id = edge.source_node_id
            if source_id not in node_by_id or source_id in visited:
                valid = False
                break
            ordered_node_ids.append(source_id)
            visited.add(source_id)
            current_id = source_id

        if not valid or ordered_node_ids[-1] != deviation_id:
            continue

        ordered_node_ids.reverse()
        ordered_edge_ids.reverse()
        weakest_status = next(
            (k for k, v in _EDGE_STATUS_STRENGTH_RANK.items() if v == weakest_rank),
            "UNKNOWN",
        )
        start_node = node_by_id[ordered_node_ids[0]]
        paths.append(CausalPath(
            path_id=f"PATH{p_idx}",
            hypothesis_id=node.source_hypothesis_id,
            ordered_node_ids=ordered_node_ids,
            ordered_edge_ids=ordered_edge_ids,
            starting_level=str(start_node.causal_level.value if hasattr(start_node.causal_level, "value") else start_node.causal_level),
            terminal_level=str(node.causal_level.value if hasattr(node.causal_level, "value") else node.causal_level),
            epistemic_status=weakest_status,
            supporting_evidence_ids=list(dict.fromkeys(supporting)),
            contradicting_evidence_ids=contradicting,
        ))
        p_idx += 1

    return paths


_SUFFICIENCY_RANK_TO_LABEL = {3: "ESTABLISHED", 2: "SUPPORTED", 1: "POSSIBLE"}


def derive_causal_sufficiency(causal_graph: Any) -> Any:
    """Final-output-quality pass: deterministically populates
    CausalSufficiencyAssessment (app.models.agent.RootCauseAnalysis.
    causal_sufficiency) -- an existing, correctly-typed field for
    representing exactly the distinction between mechanism/root-cause/
    systemic-cause depth that was declared on the model but never
    populated anywhere in production code. Reuses the SAME depth/status
    vocabulary INV-CAUSAL-005 already licenses (CausalGraphNodeType tiers,
    CausalGraphEdgeStatus strength) -- never a second, competing causal-
    sufficiency rule.

    Never invents a level the graph doesn't license: mechanism_sufficiency
    and systemic_sufficiency default to UNKNOWN when no edge reaches that
    tier at all; root_cause_sufficiency defaults to NOT_ESTABLISHED
    (matching the field's own declared default), so a report can honestly
    say "mechanism SUPPORTED, root cause NOT_ESTABLISHED, systemic cause
    UNKNOWN" rather than collapsing every depth into one status."""
    from app.models.agent import CausalGraphEdgeStatus, CausalGraphNodeType, CausalSufficiencyAssessment

    result = CausalSufficiencyAssessment()
    if not causal_graph or not getattr(causal_graph, "nodes", None):
        return result

    def _edge_rank(status: Any) -> int:
        s = str(status)
        if s.endswith("VERIFIED"):
            return 3
        if s.endswith("REPORTED"):
            return 2
        if s.endswith("POSSIBLE"):
            return 1
        return 0

    node_by_id = {n.node_id: n for n in causal_graph.nodes}
    tier_rank = {"MECHANISM": 0, "ROOT": 0, "SYSTEMIC": 0}
    for e in (causal_graph.edges or []):
        target = node_by_id.get(e.target_node_id)
        if target is None:
            continue
        rank = _edge_rank(e.status)
        if target.node_type in (CausalGraphNodeType.IMMEDIATE_MECHANISM, CausalGraphNodeType.CONTRIBUTING_FACTOR):
            tier_rank["MECHANISM"] = max(tier_rank["MECHANISM"], rank)
        elif target.node_type == CausalGraphNodeType.UNDERLYING_CAUSE:
            tier_rank["ROOT"] = max(tier_rank["ROOT"], rank)
        elif target.node_type == CausalGraphNodeType.SYSTEMIC_ROOT_CAUSE:
            tier_rank["SYSTEMIC"] = max(tier_rank["SYSTEMIC"], rank)

    result.mechanism_sufficiency = _SUFFICIENCY_RANK_TO_LABEL.get(tier_rank["MECHANISM"], "UNKNOWN")
    # A hypothesis licensed straight to SYSTEMIC_ROOT_CAUSE (skipping an
    # intermediate UNDERLYING_CAUSE node -- a real, observed production
    # shape: a single VERIFIED fact can directly establish the deepest
    # tier without a separate intermediate node ever being constructed)
    # must ALSO satisfy root-cause-level sufficiency -- the deeper tier
    # subsumes the shallower one, exactly mirroring INV-CAUSAL-005's own
    # OR condition (`node_type in (UNDERLYING_CAUSE, SYSTEMIC_ROOT_CAUSE)`)
    # for what licenses root_cause.status=ESTABLISHED. Reporting only the
    # UNDERLYING_CAUSE-specific tier here was a real bug: it disagreed
    # with the authoritative graph check and made the newly-introduced
    # INV-REPORT-002 reject 18 previously-correct production scenarios.
    root_rank = max(tier_rank["ROOT"], tier_rank["SYSTEMIC"])
    result.root_cause_sufficiency = _SUFFICIENCY_RANK_TO_LABEL.get(root_rank, "NOT_ESTABLISHED")
    result.systemic_sufficiency = _SUFFICIENCY_RANK_TO_LABEL.get(tier_rank["SYSTEMIC"], "UNKNOWN")
    return result
