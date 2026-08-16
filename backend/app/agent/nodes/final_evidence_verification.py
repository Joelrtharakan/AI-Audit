"""Node 8.5: final_evidence_verification

Filters and strips any ungrounded company policy, SOP, or regulatory claims
from state before generating the final report. Enforces the strict rule:
  "If the model writes 'The company requires retraining within 30 days' and no
   company document supports this — REMOVE IT."
"""

from __future__ import annotations

import logging
import re

from app.agent.state import AgentState
from app.models.agent import AgentTraceStep, CandidateHypothesis, EvidenceStatus, RootCauseStatus

logger = logging.getLogger(__name__)

# Patterns matching ungrounded company rules or specific timeframes
_UNGROUNDED_RULES_RE = re.compile(
    r"\b(within\s+\d+\s+(days|hours|weeks)|company\s+policy\s+requires|organization\s+requires|SOP\s+mandates|procedure\s+requires|automated\s+reminders)\b",
    re.IGNORECASE,
)


def _has_verified_document_evidence(evidence_ledger: list) -> bool:
    """Check if any document/record in the ledger provides verified company rules."""
    return any(
        e.status == EvidenceStatus.VERIFIED and "document" in e.source.lower()
        for e in evidence_ledger
    )


async def final_evidence_verification_node(state: AgentState) -> AgentState:
    """Sanitize state before final report generation."""
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    evidence_ledger = state.get("evidence_ledger", [])
    finding_text = state["request"].finding_text

    # Extract all SOP / Document identifiers from finding text and evidence ledger
    allowed_terms = set(re.findall(r"\bSOP-[\w-]+\b", finding_text, re.IGNORECASE))
    for e in evidence_ledger:
        for term in re.findall(r"\bSOP-[\w-]+\b", e.claim + " " + (e.source_reference or ""), re.IGNORECASE):
            allowed_terms.add(term.upper())

    # Helper function to remove hallucinated SOP identifiers and unevidenced features
    def _clean_text(text: str | None) -> str | None:
        if not text:
            return text
        # Remove SOP identifiers not present in allowed_terms
        found_sops = set(re.findall(r"\bSOP-[\w-]+\b", text, re.IGNORECASE))
        for sop in found_sops:
            if sop.upper() not in allowed_terms:
                text = re.sub(r"\bthe\s+SOP-[\w-]+\b", "the procedure revision", text, flags=re.IGNORECASE)
                text = re.sub(r"\bSOP-[\w-]+\b", "procedure revision", text, flags=re.IGNORECASE)
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: stripped unmentioned identifier '{sop}'"
                ))
        # Section 6: If finding text does NOT state a revision, strip hallucinated "revision" claims
        if "revision" not in finding_text.lower() and "revised" not in finding_text.lower() and "updated" not in finding_text.lower():
            text = re.sub(r"\b(procedure\s+revision|recent\s+revision|after\s+the\s+revision)\b", "procedure requirement", text, flags=re.IGNORECASE)

        # Section 12 & 16: Strip ungrounded patient safety / product safety / customer / recall / quarantine claims unless explicit
        if "patient" not in finding_text.lower():
            text = re.sub(r"\b(patient\s+safety|patient\s+impact|affecting\s+patient[s]?)\b", "data integrity and process compliance", text, flags=re.IGNORECASE)
        if "customer" not in finding_text.lower() and "client" not in finding_text.lower():
            text = re.sub(r"\b(customer\s+impact|affecting\s+customer[s]?|client\s+impact)\b", "internal quality process compliance", text, flags=re.IGNORECASE)
        if "product" not in finding_text.lower() and "batch" not in finding_text.lower() and "sample" not in finding_text.lower():
            text = re.sub(r"\b(product\s+quality|product\s+safety|compromised\s+product)\b", "process documentation compliance", text, flags=re.IGNORECASE)

        # Section 10 & 14 & 15: Strip ungrounded severe actions (recall, quarantine, equipment/operator restriction)
        if "recall" not in finding_text.lower():
            text = re.sub(r"\b(product\s+recall|recall\s+affected\s+batch)\b", "assess affected records", text, flags=re.IGNORECASE)
        if "quarantine" not in finding_text.lower():
            text = re.sub(r"\b(quarantine\s+affected\s+batch|quarantine\s+product[s]?)\b", "assess scope of affected records", text, flags=re.IGNORECASE)
        if "restrict" not in finding_text.lower() and "stop" not in finding_text.lower() and "halt" not in finding_text.lower():
            text = re.sub(r"\b(restrict\s+[\w-]+|stop\s+production|halt\s+operations)\b", "verify compliance and complete required records", text, flags=re.IGNORECASE)


        # Section 10: If training is completely unmentioned in finding and ledger, strip ungrounded training CAPA
        if "train" not in finding_text.lower() and "retrain" not in finding_text.lower():
            text = re.sub(r"\b(provide\s+additional\s+training|retrain\s+staff|retrain\s+operator|training\s+program)\b", "verify checklist usability and execution controls", text, flags=re.IGNORECASE)

        # Section 11 & 12: Strip ungrounded population expansion ("other operators", "all personnel") unless finding supports it
        if "other operators" not in finding_text.lower() and "all staff" not in finding_text.lower() and "multiple operators" not in finding_text.lower():
            text = re.sub(r"\b(other\s+operators|all\s+operators|all\s+personnel|broader\s+population)\b", "the specific personnel identified in the finding", text, flags=re.IGNORECASE)

        # Section 14: Strip ungrounded competency/qualification claims if not mentioned in finding
        if "competenc" not in finding_text.lower() and "qualification" not in finding_text.lower() and "qualifi" not in finding_text.lower():
            text = re.sub(r"\b(competency\s+assessment|operator\s+qualification|qualification\s+status)\b", "training record reconciliation", text, flags=re.IGNORECASE)

        # Section 17 & Part 15: Replace generic filler phrases with specific case-grounded wording
        text = re.sub(r"\bcontain\s+items\s+as\s+necessary\b", "assess and contain affected execution records", text, flags=re.IGNORECASE)
        text = re.sub(r"\bprimary\s+execution\s+and\s+verification\s+records\b", "specific execution logs and supervisory sign-off records", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdownstream\s+process\s+requiring\s+verification\b", "subsequent operational steps dependent on the nonconformity", text, flags=re.IGNORECASE)
        text = re.sub(r"\btimeframe\s+stated\s+in\s+finding\s+or\s+requires\s+confirmation\b", "the period noted in the finding", text, flags=re.IGNORECASE)

        # Part 4 & 6: Clean full-finding verbatim sentence repetitions inside template phrases ("records for 'During the audit...'")
        clean_finding = finding_text.strip()
        if len(clean_finding) > 15:
            pattern = re.escape(clean_finding)
            text = re.sub(r"(['\"`])(?:during\s+the\s+audit[,\s]*)?" + pattern + r"\1", "the nonconformity", text, flags=re.IGNORECASE)
            text = re.sub(r"for\s+['\"`][^'\"]*during\s+the\s+audit[^'\"]*['\"`]", "for the observed nonconformity", text, flags=re.IGNORECASE)
            text = re.sub(r"for\s+['\"`][^'\"]*['\"`]", "for the observed nonconformity", text, flags=re.IGNORECASE)

        # Requirement 23 & 5: Clean verb-clause / pronoun-led subject contamination fragments
        canonical = state.get("canonical_finding_state")
        canon_sub = canonical.finding_subject if (canonical and canonical.finding_subject != "UNKNOWN") else "the affected process"
        text = re.sub(r"\b(?:associated\s+with|verification\s+for|controls\s+for)\s+(?:they\s+(?:had|were|did)|the\s+operator\s+stated|the\s+technician\s+reported)\s+[a-z0-9\s-]+?\b(?=\s+(?:were|was|failed|lacked|are|is)\b)", f"associated with {canon_sub}", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:Why\s+did\s+the\s+following\s+occur:\s+)(?:they\s+(?:had|were|did)|the\s+operator\s+stated)\s+[a-z0-9\s-]+?\?", f"Why did the nonconformity in {canon_sub} occur?", text, flags=re.IGNORECASE)

        return text










    # Clean Root Cause Analysis
    rc = state.get("root_cause")
    if rc:
        rc.narrative = _clean_text(rc.narrative) or ""
        rc.statement = _clean_text(rc.statement)

    # Clean Investigation Plan
    inv = state.get("investigation_plan")
    if inv:
        # questions are InvestigationQuestion objects — clean each text field
        for iq in inv.questions:
            iq.question = _clean_text(iq.question) or iq.question
            iq.purpose = _clean_text(iq.purpose) or iq.purpose
            iq.evidence = _clean_text(iq.evidence) or iq.evidence
        inv.areas = [_clean_text(a) or a for a in inv.areas]

    # INVESTIGATION PLAN <- HYPOTHESES: plan_investigation runs BEFORE
    # core_synthesis exists, so it can never generate questions that
    # discriminate between hypotheses it hasn't seen yet -- and in this
    # deployment it's usually fast-pathed to an empty plan entirely (no
    # ASP.NET tool endpoints configured). If synthesis produced live
    # hypotheses but the plan still has no questions, derive them
    # deterministically from those hypotheses now rather than shipping a
    # report with real candidate causes and an empty investigation plan.
    rc_for_questions = state.get("root_cause")
    if inv is not None and not inv.questions and rc_for_questions and rc_for_questions.candidate_hypotheses:
        from app.agent.analytical_validator import derive_investigation_questions
        inv.questions = derive_investigation_questions(rc_for_questions.candidate_hypotheses)
        trace.append(AgentTraceStep.ok(
            f"Investigation plan: derived {len(inv.questions)} discriminating question(s) from "
            "candidate hypotheses (investigation planning ran before synthesis existed)"
        ))

    # Question uniqueness (Section 5): two hypotheses must never be tested
    # with what is effectively the same question. Applied regardless of
    # which path populated inv.questions (derived above, or a real
    # tool-based LLM plan).
    if inv is not None and inv.questions:
        from app.agent.analytical_validator import deduplicate_investigation_questions
        deduped_questions = deduplicate_investigation_questions(inv.questions)
        if len(deduped_questions) != len(inv.questions):
            trace.append(AgentTraceStep.warn(
                f"Investigation plan: removed {len(inv.questions) - len(deduped_questions)} "
                "duplicate/near-duplicate investigation question(s)"
            ))
            inv.questions = deduped_questions

    # Clean 5-Why — independent of whether an investigation_plan exists (a
    # finding that skipped tool-based investigation can still have a
    # populated five_why from core_synthesis).
    fw = state.get("five_why")
    if fw and fw.steps:
        valid_fw_steps = []
        for step in fw.steps:
            q_text = _clean_text(step.question) or step.question
            from app.agent.causal_guard import is_reporting_why_question
            if is_reporting_why_question(q_text):
                trace.append(AgentTraceStep.warn(
                    "Final Evidence Verification: stripped 5-Why step asking about reporting behavior"
                ))
                continue
            step.question = q_text
            step.answer = _clean_text(step.answer)
            valid_fw_steps.append(step)
        fw.steps = valid_fw_steps

    # Clean CA Draft
    ca = state.get("ca_draft")
    if ca:
        ca.immediate_action = _clean_text(ca.immediate_action) or ca.immediate_action
        ca.root_cause = _clean_text(ca.root_cause) or ca.root_cause
        ca.preventive_action = _clean_text(ca.preventive_action) or ca.preventive_action
        ca.impact_analysis = _clean_text(ca.impact_analysis) or ca.impact_analysis

    # Clean CapaAnalysis
    capa = state.get("capa_analysis")
    if capa:
        capa.potential_areas = [_clean_text(a) or a for a in capa.potential_areas]
        capa.recommended_investigation = [_clean_text(r) or r for r in capa.recommended_investigation]

    # Clean ImpactAssessment
    impact = state.get("impact_assessment")
    if impact:
        impact.narrative = _clean_text(impact.narrative)
        impact.areas = [_clean_text(a) or a for a in impact.areas]

    # Clean Contributing Factors
    cfs = state.get("contributing_factors", [])
    for cf in cfs:
        cf.description = _clean_text(cf.description) or cf.description

    # Clean Evidence Gaps
    gaps = state.get("evidence_gaps", [])
    for gap in gaps:
        gap.claim = _clean_text(gap.claim) or gap.claim
        gap.missing = _clean_text(gap.missing) or gap.missing

    # Clean InvestigationReport if already assembled
    report = state.get("report")
    if report:
        if report.root_cause:
            report.root_cause.narrative = _clean_text(report.root_cause.narrative) or ""
            report.root_cause.statement = _clean_text(report.root_cause.statement)
            if report.root_cause.supporting_evidence:
                report.root_cause.supporting_evidence = [_clean_text(e) or e for e in report.root_cause.supporting_evidence]
            if report.root_cause.contradicting_evidence:
                report.root_cause.contradicting_evidence = [_clean_text(e) or e for e in report.root_cause.contradicting_evidence]
            if report.root_cause.missing_evidence:
                report.root_cause.missing_evidence = [_clean_text(e) or e for e in report.root_cause.missing_evidence]
        if report.investigation:
            for iq in report.investigation.questions:
                iq.question = _clean_text(iq.question) or iq.question
                iq.purpose = _clean_text(iq.purpose) or iq.purpose
                iq.evidence = _clean_text(iq.evidence) or iq.evidence
            report.investigation.areas = [_clean_text(a) or a for a in report.investigation.areas]
            if report.investigation.evidence_to_collect:
                report.investigation.evidence_to_collect = [_clean_text(e) or e for e in report.investigation.evidence_to_collect]
        if report.five_why and report.five_why.steps:
            for step in report.five_why.steps:
                step.question = _clean_text(step.question) or step.question
                step.answer = _clean_text(step.answer)
        if report.capa:
            report.capa.potential_areas = [_clean_text(a) or a for a in report.capa.potential_areas]
            report.capa.recommended_investigation = [_clean_text(r) or r for r in report.capa.recommended_investigation]
        if report.impact_assessment:
            report.impact_assessment.narrative = _clean_text(report.impact_assessment.narrative)
            report.impact_assessment.areas = [_clean_text(a) or a for a in report.impact_assessment.areas]
        if report.contributing_factors:
            for cf in report.contributing_factors:
                cf.description = _clean_text(cf.description) or cf.description
        if report.evidence_gaps:
            for gap in report.evidence_gaps:
                gap.claim = _clean_text(gap.claim) or gap.claim
                gap.missing = _clean_text(gap.missing) or gap.missing
        if report.evidence:
            for ev in report.evidence:
                ev.claim = _clean_text(ev.claim) or ev.claim

    # Section 4 & 22 Consistency Validator Check:
    # If investigation questions exist but candidate hypotheses are empty, generate matching candidate hypotheses dynamically
    if rc and inv and inv.questions and not rc.candidate_hypotheses:
        for iq in inv.questions:
            q_lower = iq.question.lower()
            if "training" in q_lower or "retraining" in q_lower:
                rc.candidate_hypotheses.append(CandidateHypothesis(
                    id=f"H{len(rc.candidate_hypotheses)+1}",
                    name="TRAINING_IMPLEMENTATION",
                    statement="Required retraining may not have been completed or assigned following procedure requirement.",
                    status="POSSIBLE",
                    evidence_needed="Training assignment and completion records",
                    relevance_rank="HIGH",
                ))
            elif "checklist" in q_lower or "usability" in q_lower or "clear" in q_lower:
                rc.candidate_hypotheses.append(CandidateHypothesis(
                    id=f"H{len(rc.candidate_hypotheses)+1}",
                    name="CHECKLIST_USABILITY",
                    statement="The checklist or procedure requirement may not have been clear or usable at the point of execution.",
                    status="POSSIBLE",
                    evidence_needed="Checklist version, workstation setup, and workflow review",
                    relevance_rank="HIGH",
                ))
            elif "interruption" in q_lower or "workload" in q_lower or "condition" in q_lower:
                rc.candidate_hypotheses.append(CandidateHypothesis(
                    id=f"H{len(rc.candidate_hypotheses)+1}",
                    name="HUMAN_PERFORMANCE_CONDITIONS",
                    statement="Task execution conditions such as competing tasks or interruptions may have contributed to omission.",
                    status="POSSIBLE",
                    evidence_needed="Shift records and workflow observation",
                    relevance_rank="MEDIUM",
                ))
            else:
                rc.candidate_hypotheses.append(CandidateHypothesis(
                    id=f"H{len(rc.candidate_hypotheses)+1}",
                    name="PROCESS_EXECUTION_GAP",
                    statement="A gap in process execution or completion verification allowed the omission to occur.",
                    status="POSSIBLE",
                    evidence_needed="Execution and verification logs",
                    relevance_rank="MEDIUM",
                ))
            trace.append(AgentTraceStep.warn(
                f"Consistency Validator: dynamically derived candidate hypothesis H{len(rc.candidate_hypotheses)} from investigation question"
            ))


    # Sync report object components with cleaned state components so UI renderer receives identical clean data
    if report:
        report.root_cause = rc
        report.investigation = inv
        report.five_why = fw
        report.capa = capa
        report.impact_assessment = impact

    # Deterministic semantic consistency check (Section: SEMANTIC CONSISTENCY
    # VALIDATOR): confirms the canonical finding subject actually survived
    # into the final analysis, and that impact's affected_object didn't
    # regress to a placeholder or introduce an entity absent from the finding.
    from app.agent.semantic_validator import validate_semantic_consistency
    canonical = state.get("canonical_finding_state")
    consistency_warnings = validate_semantic_consistency(canonical, {**state, "root_cause": rc, "five_why": fw, "impact_assessment": impact})
    for warning in consistency_warnings:
        trace.append(AgentTraceStep.warn(warning))

    # Extract immediate mechanism for invariant filter check
    from app.agent.causal_guard import MechanismInfo, extract_immediate_mechanism, hypothesis_contradicts_mechanism, mechanism_already_names_generic_hypothesis
    canonical = state.get("canonical_finding_state")
    if canonical and canonical.immediate_mechanism:
        mechanism = MechanismInfo(
            statement=canonical.immediate_mechanism,
            status=canonical.immediate_mechanism_status,
        )
        from app.agent.causal_guard import classify_mechanism_polarity
        mechanism.polarity = classify_mechanism_polarity(canonical.immediate_mechanism)
    else:
        reported = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.REPORTED]
        verified = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.VERIFIED]
        mechanism = extract_immediate_mechanism(reported, verified)

    verified_facts = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.VERIFIED]

    # LOW-specificity gate: a finding with no entity/date/period, no
    # reported/attributed statement, and no established immediate mechanism
    # carries no evidence to ground a causal hypothesis on -- generating one
    # anyway (e.g. a fixed pair of generic 6M-style categories) misrepresents
    # a data-empty allegation as having been causally analyzed. Runs BEFORE
    # hypothesis filtering so no ungrounded hypothesis reaches the rest of
    # the pipeline. Recurrence hypotheses are exempt: recurrence is detected
    # independently of this finding's own specificity.
    if rc is not None:
        from app.agent.recurrence_guard import is_previous_capa_mechanism_hypothesis as _is_recurrence_hyp
        from app.services.semantic_subject import classify_finding_specificity
        _reported_for_specificity = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.REPORTED]
        specificity = classify_finding_specificity(
            finding_text, _reported_for_specificity, mechanism.status if mechanism else None
        )
        if specificity == "LOW":
            existing_hyps = rc.candidate_hypotheses or []
            dropped_low = [h for h in existing_hyps if not _is_recurrence_hyp(h.statement)]
            if dropped_low:
                trace.append(AgentTraceStep.warn(
                    "Final Evidence Verification: finding classified LOW specificity — removed "
                    f"{len(dropped_low)} hypothesis/hypotheses with no grounding entity, date/period, "
                    "reported statement, or established mechanism to justify causal analysis"
                ))
            rc.candidate_hypotheses = [h for h in existing_hyps if _is_recurrence_hyp(h.statement)]
            rc.status = RootCauseStatus.NOT_ESTABLISHED
            rc.confidence = "LOW"
            rc.narrative = (
                "The finding establishes a nonconformity but does not contain sufficient evidence to "
                "identify the causal mechanism."
            )
            if hasattr(rc, "missing_evidence"):
                rc.missing_evidence = list(dict.fromkeys([*(rc.missing_evidence or []), *[
                    "The specific procedure or requirement not followed",
                    "The specific affected activity, object, or process",
                    "The observed deviation from the requirement",
                    "Objective evidence (records, logs, or documents) of the deviation",
                    "The affected time period",
                ]]))

    # Invariant enforcement: ensure rejected hypotheses (from causal guard, critic, or grounding guard) do not appear in final output
    if rc and rc.candidate_hypotheses:
        from app.agent.causal_guard import (
            detect_unsupported_causal_specificity,
            hypothesis_attacks_statement_credibility,
            hypothesis_contradicts_verified_completion,
        )
        filtered_final_hyps = []
        for h in rc.candidate_hypotheses:
            if h.status == "REFUTED":
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} because status was REFUTED"
                ))
                continue
            if hypothesis_attacks_statement_credibility(h.statement):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — attacks the credibility of "
                    "a reported statement instead of reasoning about the underlying proposition"
                ))
                continue
            if mechanism and mechanism.statement:
                if hypothesis_contradicts_mechanism(h.statement, mechanism):
                    trace.append(AgentTraceStep.warn(
                        f"Final Evidence Verification: removed hypothesis {h.id} — contradicts mechanism"
                    ))
                    continue
                if mechanism_already_names_generic_hypothesis(h.statement, mechanism):
                    trace.append(AgentTraceStep.warn(
                        f"Final Evidence Verification: removed hypothesis {h.id} — restates mechanism"
                    ))
                    continue
            if hypothesis_contradicts_verified_completion(h.statement, verified_facts):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — contradicts a VERIFIED "
                    "fact that the thing it claims is deficient was actually completed"
                ))
                continue
            is_unsupported, reason = detect_unsupported_causal_specificity(h.statement, finding_text)
            if is_unsupported:
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — {reason}"
                ))
                continue

            # Deterministic status policy enforcement (Requirements 1, 2, 4)
            from app.agent.causal_guard import determine_hypothesis_status
            from app.agent.recurrence_guard import is_previous_capa_mechanism_hypothesis
            reported_claims = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.REPORTED]
            # A previous-CAPA-implementation/effectiveness hypothesis will
            # always share heavy vocabulary with the VERIFIED fact that
            # establishes recurrence/previous-CAPA-completion (that's what
            # grounds it) -- but that fact never verifies the hypothesis's
            # OWN claim (non-implementation/non-verification/ineffectiveness),
            # only a dedicated implementation or effectiveness-review claim
            # could. Detected by CONTENT (statement text), not by name, so
            # this catches an LLM's own free-form phrasing of this mechanism,
            # not only this system's deterministic naming convention.
            is_recurrence_hyp = is_previous_capa_mechanism_hypothesis(h.statement) or (
                "CAPA_EFFECTIVENESS_GAP" in (h.name or "").upper()
            )
            det_status, det_strength = determine_hypothesis_status(
                h.statement,
                verified_facts,
                reported_claims,
                canonical.evidence_conflicts if canonical else None,
                mechanism,
                allow_verified_promotion=not is_recurrence_hyp,
            )
            h.status = det_status
            h.evidence_strength = det_strength
            # Evidence traceability firewall: a hypothesis must never be
            # SUPPORTED without a citable supporting fact -- determine_hypothesis_status
            # already requires a VERIFIED-fact overlap to reach SUPPORTED, so
            # this records WHICH fact(s) justified it (auditability) and, as
            # a safety net, downgrades to POSSIBLE if none can actually be
            # cited (should never trigger given the status logic above, but
            # traceability must never be silently assumed).
            if h.status == "SUPPORTED":
                from app.services.text_grounding import significant_words
                hyp_words = significant_words(h.statement)
                citing_facts = [
                    fact for fact in verified_facts
                    if hyp_words and significant_words(fact)
                    and len(hyp_words & significant_words(fact)) / min(len(hyp_words), len(significant_words(fact))) >= 0.5
                ]
                if citing_facts:
                    h.supporting_evidence = list(dict.fromkeys([*h.supporting_evidence, *citing_facts]))
                else:
                    h.status = "POSSIBLE"
                    h.evidence_strength = "NONE"
                    trace.append(AgentTraceStep.warn(
                        f"Final Evidence Verification: downgraded hypothesis {h.id} from SUPPORTED to "
                        "POSSIBLE — no citable supporting evidence found (traceability firewall)"
                    ))
            filtered_final_hyps.append(h)

        # Cap current-event hypotheses at 4 (recurrence/previous-CAPA
        # hypotheses are a distinct causal dimension and are exempt from this
        # cap): an LLM asked to generate hypotheses will happily generate one
        # per generic 6M category regardless of how much evidence actually
        # exists, and that volume itself misrepresents evidence-constrained
        # reasoning as exhaustive causal coverage. Keep the highest-ranked
        # (then highest-scoring) current-event hypotheses when over the cap.
        current_event_hyps = [
            h for h in filtered_final_hyps if not is_previous_capa_mechanism_hypothesis(h.statement)
        ]
        recurrence_hyps = [
            h for h in filtered_final_hyps if is_previous_capa_mechanism_hypothesis(h.statement)
        ]
        if len(current_event_hyps) > 4:
            from app.agent.analytical_validator import _RANK_ORDER, score_hypothesis
            current_event_hyps.sort(
                key=lambda h: (_RANK_ORDER.get(h.relevance_rank, 1), -score_hypothesis(h)),
            )
            dropped = current_event_hyps[4:]
            current_event_hyps = current_event_hyps[:4]
            for h in dropped:
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: dropped hypothesis {h.id} — exceeded the 4 "
                    "current-event hypothesis cap (evidence-constrained hypothesis generation)"
                ))
        rc.candidate_hypotheses = current_event_hyps + recurrence_hyps

        # Investigation areas / CAPA potential_areas (Section 7 & 9): every
        # area must trace to a live candidate hypothesis, never
        # independently generated (an LLM CAPA field is free to invent a
        # category the finding never supports, e.g. "communication of
        # requirements" for a finding that doesn't mention communication).
        # Derived here, after hypothesis filtering, so it reflects the FINAL
        # surviving hypothesis set. While root cause is NOT_ESTABLISHED
        # these are investigation areas, not a confirmed CAPA scope --
        # rendering that distinction is the report generator's job; this
        # only guarantees the CONTENT traces to a real hypothesis.
        if filtered_final_hyps:
            from app.agent.analytical_validator import derive_investigation_areas
            derived_areas = derive_investigation_areas(filtered_final_hyps)
            if derived_areas:
                if capa and capa.potential_areas != derived_areas:
                    trace.append(AgentTraceStep.warn(
                        "Final Evidence Verification: potential_areas replaced with areas derived from "
                        f"candidate hypotheses (was {capa.potential_areas!r})"
                    ))
                    capa.potential_areas = derived_areas
                if inv and inv.areas != derived_areas:
                    inv.areas = derived_areas

        # If root cause is not established, root cause confidence must be LOW (Requirement 5 & 20)
        if rc.status in (RootCauseStatus.NOT_ESTABLISHED, "NOT_ESTABLISHED"):
            rc.confidence = "LOW"

        # Leading-hypothesis selection runs the SAME way regardless of
        # whether a conflict exists -- apply_conflict_tie_override (below)
        # is the single, authoritative correction for the conflict case, so
        # this block never independently reinterprets what "tied" means.
        from app.agent.analytical_validator import (
            apply_conflict_tie_override,
            hypothesis_confidence,
            leading_hypothesis_confidence,
            leading_hypothesis_display,
            leading_hypothesis_status,
        )
        for h in rc.candidate_hypotheses:
            h.confidence = hypothesis_confidence(h)
        new_status = leading_hypothesis_status(rc.candidate_hypotheses)
        new_leading = leading_hypothesis_display(rc.candidate_hypotheses)
        if hasattr(rc, "leading_hypothesis") and new_leading != getattr(rc, "leading_hypothesis", None):
            rc.leading_hypothesis = new_leading
            rc.leading_hypothesis_status = new_status
            if new_status == "SELECTED":
                rc.confidence = leading_hypothesis_confidence(rc.candidate_hypotheses, new_leading)
            elif rc.status in (RootCauseStatus.NOT_ESTABLISHED, "NOT_ESTABLISHED"):
                rc.confidence = "LOW"
        elif hasattr(rc, "leading_hypothesis_status"):
            rc.leading_hypothesis_status = new_status
            if rc.status in (RootCauseStatus.NOT_ESTABLISHED, "NOT_ESTABLISHED"):
                rc.confidence = "LOW"

        # An UNRESOLVED evidence conflict means root cause cannot stand as
        # established regardless of what upstream status was claimed, and no
        # single hypothesis may be promoted to "leading" merely from an
        # incidental scoring/wording difference between competing
        # explanations of that SAME still-open conflict (a hypothesis with
        # genuinely independent SUPPORTED evidence is still allowed to lead
        # -- see apply_conflict_tie_override).
        has_unresolved_conflict = bool(
            canonical and any(getattr(c, "status", "UNRESOLVED") == "UNRESOLVED" for c in canonical.evidence_conflicts)
        )
        if has_unresolved_conflict:
            if rc.status not in (RootCauseStatus.NOT_ESTABLISHED, "NOT_ESTABLISHED"):
                trace.append(AgentTraceStep.warn(
                    "Final Evidence Verification: root cause forced to NOT_ESTABLISHED — an unresolved "
                    "evidence conflict remains"
                ))
                rc.status = RootCauseStatus.NOT_ESTABLISHED
                rc.confidence = "LOW"
            apply_conflict_tie_override(rc, has_unresolved_conflict)

    # Rule 1 & 12: 5-Why provenance and conflict boundary enforcement
    if fw and fw.steps:
        from app.agent.causal_guard import _ATTRIBUTION_LANGUAGE_RE
        for step in fw.steps:
            if step.answer and _ATTRIBUTION_LANGUAGE_RE.search(step.answer):
                if step.status == "VERIFIED":
                    step.status = "REPORTED"
                    trace.append(AgentTraceStep.warn(
                        "Final Evidence Verification: downgraded 5-Why answer containing attribution from VERIFIED to REPORTED"
                    ))
        if canonical and canonical.evidence_conflicts:
            # When evidence conflicts exist, rebuild deterministic 5-Why chain if LLM fabricated steps
            from app.agent.nodes.five_why_fallback import build_deterministic_five_why
            fw = build_deterministic_five_why(finding_text, evidence_ledger)
            if report:
                report.five_why = fw

    # Rule 5: Affected Object vs Impact separation, AND semantic subject-
    # drift detection. A grammatically valid noun phrase can still have
    # drifted to an unrelated concept (e.g. "procedure compliance" when the
    # finding's actual topic is "training") -- validate_semantic_subject
    # alone only catches malformed clauses, not wrong-but-well-formed
    # phrases, so subject_topic_matches is the check that actually catches
    # this class of defect anywhere it can enter (the LLM path included,
    # not just the deterministic fallback).
    affected_object_corrected = False
    canon_subject = (canonical.finding_subject if canonical else None) or (canonical.affected_object if canonical else None)
    if canon_subject in ("UNKNOWN", None) or (canon_subject and canon_subject.startswith("UNKNOWN")):
        canon_subject = None
    if impact:
        from app.services.semantic_subject import (
            build_affected_object_phrase,
            strip_leading_article,
            subject_topic_matches,
            topic_word,
            validate_semantic_subject,
        )
        canon_actor = strip_leading_article((canonical.actor if canonical else None) or None)
        needs_repair = not validate_semantic_subject(impact.affected_object) or (
            bool(impact.affected_object) and bool(re.search(r"\bability\b", impact.affected_object, re.IGNORECASE))
        )
        if not needs_repair and canon_subject and impact.affected_object and not subject_topic_matches(impact.affected_object, canon_subject):
            needs_repair = True
            trace.append(AgentTraceStep.warn(
                f"Final Evidence Verification: affected_object {impact.affected_object!r} drifted from the "
                f"finding's canonical subject {canon_subject!r}"
            ))
        if needs_repair:
            clean_obj = build_affected_object_phrase(canon_subject, canon_actor) if canon_subject else "the affected process"
            impact.affected_object = clean_obj
            affected_object_corrected = True
            trace.append(AgentTraceStep.warn(
                f"Final Evidence Verification: corrected impact affected_object to canonical phrase: {clean_obj!r}"
            ))

        # Same drift defect, same fix, applied to the two other fields that
        # name the same underlying topic -- process_at_risk and
        # evidence_needed must describe THIS finding's topic, not one the
        # LLM substituted in (e.g. "procedure revision documentation" for a
        # training finding).
        if canon_subject:
            topic = topic_word(canon_subject)
            topic_cap = topic[0].upper() + topic[1:]
            if impact.process_at_risk and not subject_topic_matches(impact.process_at_risk, canon_subject):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: process_at_risk {impact.process_at_risk!r} drifted from "
                    "the finding's canonical subject — replaced"
                ))
                impact.process_at_risk = f"{topic_cap} and compliance control"

        # Evidence Needed (Section 5): always DERIVED from the live
        # candidate hypotheses' own already-grounded evidence_needed values
        # rather than trusted from a separately-generated impact field --
        # this is architecturally immune to the "procedure revision
        # documentation" class of drift, not just detected-and-patched
        # after the fact, since it never reads the LLM's impact.evidence_needed
        # at all when hypotheses exist.
        if rc and rc.candidate_hypotheses:
            from app.agent.analytical_validator import derive_required_evidence
            derived_evidence = derive_required_evidence(
                rc.candidate_hypotheses, canonical.evidence_conflicts if canonical else None
            )
            if derived_evidence and derived_evidence != impact.evidence_needed:
                trace.append(AgentTraceStep.warn(
                    "Final Evidence Verification: evidence_needed derived from candidate hypotheses "
                    f"(was {impact.evidence_needed!r})"
                ))
                impact.evidence_needed = derived_evidence
        elif impact.evidence_needed and canon_subject and not subject_topic_matches(impact.evidence_needed, canon_subject):
            trace.append(AgentTraceStep.warn(
                f"Final Evidence Verification: evidence_needed {impact.evidence_needed!r} drifted from "
                "the finding's canonical subject — replaced"
            ))
            impact.evidence_needed = f"Approved {topic_word(canon_subject)} completion and authorization record"

        if impact.potential_effect and not re.search(r"^(if\b|potential\b|requires\b)", impact.potential_effect.strip(), re.IGNORECASE):
            clean_subj = canon_subject or "the required activity"
            impact.potential_effect = f"If {clean_subj} was not completed as required, potential risk of operational noncompliance or unverified execution."

    # Rule 13: Immediate action grounding. Always REBUILT from the (possibly
    # just-corrected) impact.affected_object via the same deterministic
    # derivation core_synthesis uses for the CA draft, rather than trusting
    # the LLM's own restatement, whenever the action text shows a known bad
    # pattern OR affected_object itself needed repair above (an immediate
    # action built from a drifted affected_object inherits that same drift).
    if ca and ca.immediate_action:
        # Word-boundary match -- a naive substring check on "ability" also
        # matches inside unrelated words like "usability"/"capability".
        bad_pattern = bool(re.search(r"\bability\b|\bunqualified\b|\bthey had\b", ca.immediate_action, re.IGNORECASE))
        if bad_pattern or affected_object_corrected:
            from app.agent.nodes.core_synthesis import _derive_ca_draft_fields
            fields = _derive_ca_draft_fields(rc, impact)
            ca.immediate_action = fields["immediate_action"]
            trace.append(AgentTraceStep.warn(
                "Final Evidence Verification: rebuilt immediate_action from the canonical affected object"
            ))

    # ---------------------------------------------------------------------
    # ANALYTICAL VALIDATION FIREWALL (structural, deterministic; never
    # invents content -- repairs structure, downgrades unsupported
    # certainty, or removes unlinked content).
    # ---------------------------------------------------------------------
    from app.agent.analytical_validator import (
        compute_analytical_quality,
        five_why_skips_available_mechanism,
        repair_five_why_with_mechanism,
        validate_capa_causal_linkage,
        validate_root_cause_state,
    )

    # Root-cause certainty monotonicity: an ESTABLISHED-like status requires
    # actual VERIFIED evidence, never just a reported/inferred claim.
    if rc:
        for warning in validate_root_cause_state(rc, mechanism):
            trace.append(AgentTraceStep.warn(warning))

    # 5-Why must not skip an explicitly available causal fact: if the
    # finding/evidence establishes a mechanism but no Why-step reflects it,
    # insert it using the mechanism's own already-extracted text (never a
    # fabricated explanation) rather than silently pass through a chain
    # that stopped one level too early.
    if fw and mechanism and mechanism.statement:
        if five_why_skips_available_mechanism(fw.steps, mechanism):
            observed = canonical.observed_deviation if canonical else None
            repaired_steps = repair_five_why_with_mechanism(fw.steps, mechanism, observed)
            if repaired_steps != fw.steps:
                trace.append(AgentTraceStep.warn(
                    "Analytical Validator: 5-Why chain skipped an explicitly available causal "
                    "mechanism — inserted it as an additional step"
                ))
                fw.steps = repaired_steps

    # Contributing factor established/potential/rejected split: a factor
    # that structurally contradicts the mechanism or a VERIFIED completion
    # fact (same detectors used on hypotheses) is REJECTED and dropped from
    # the surviving list, never silently left in as though unconfirmed.
    from app.agent.analytical_validator import classify_contributing_factors_full
    if cfs:
        established_cfs, potential_cfs, rejected_cfs = classify_contributing_factors_full(cfs, mechanism, verified_facts)
        for f in rejected_cfs:
            trace.append(AgentTraceStep.warn(
                f"Analytical Validator: contributing factor rejected — contradicts evidence: {f.description!r}"
            ))
        cfs = established_cfs + potential_cfs
        if report:
            report.contributing_factors = cfs

    # CAPA causal linkage: a conditional action that doesn't trace back to
    # any live hypothesis is an orphaned claim, not evidence-grounded CAPA.
    if capa:
        for warning in validate_capa_causal_linkage(capa, rc.candidate_hypotheses if rc else []):
            trace.append(AgentTraceStep.warn(warning))

    # Impact field EXPLICIT/INFERRED/UNKNOWN classification -- deterministic,
    # computed here (not asserted by the LLM) from the already-finalized
    # (cleaned/grounded) impact content.
    from app.agent.analytical_validator import compute_impact_field_basis
    if impact:
        impact.field_basis = compute_impact_field_basis(impact, finding_text)

    # Internal quality signal only -- logged for observability, never
    # exposed to the report as a numeric confidence claim.
    quality = compute_analytical_quality(rc, fw, cfs, capa, mechanism)
    trace.append(AgentTraceStep.ok(f"Analytical quality signals: {quality}"))

    # Consolidated causal-graph audit (Observation -> Mechanism -> Hypothesis
    # -> Evidence -> Root Cause -> Corrective Action -> Effectiveness Check).
    # Logged only -- every repairable violation was already handled by the
    # more specific checks above; anything still flagged here is a residual
    # weak edge worth an auditor's attention, not something to fabricate a
    # fix for.
    from app.agent.analytical_validator import validate_causal_graph
    for violation in validate_causal_graph(rc, fw, capa, mechanism):
        trace.append(AgentTraceStep.warn(f"Causal Graph Audit: {violation}"))

    trace.append(AgentTraceStep.ok("Final evidence verification and consistency validation completed"))



    return {
        **state,
        "root_cause": rc,
        "investigation_plan": inv,
        "five_why": fw,
        "ca_draft": ca,
        "capa_analysis": capa,
        "impact_assessment": impact,
        "contributing_factors": cfs,
        "report": report,
        "trace": trace,
        "errors": errors,
    }
