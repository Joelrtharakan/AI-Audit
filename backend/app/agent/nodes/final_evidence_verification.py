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
from app.models.agent import AgentTraceStep, EvidenceStatus, RootCauseStatus, SupportLevel

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
    canonical = state.get("canonical_finding_state")

    # NON-ACTIONABLE FAST PATH: preserve clean not_applicable state without backfills
    if canonical and not getattr(canonical, "is_actionable", True):
        trace.append(AgentTraceStep.ok("Final evidence verification: non-actionable input verified (0ms fast path)"))
        return state

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
        # Synthetic artifact name sanitization (Section I & J)
        text = re.sub(r"\bSOP-[\w-]+\s+(?:execution\s+records?|maintenance/?status\s+logs?)\b", "approved procedure revision and distribution records", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:[A-Z0-9-]+(?:-[A-Z0-9]+)+)\s+execution\s+records?\b", "operational monitoring logs and execution records", text, flags=re.IGNORECASE)
        text = re.sub(r"\b([A-Za-z0-9_-]+)\s+execution\s+records?\b", r"\1 logs and records", text, flags=re.IGNORECASE)

        # Phantom hypothesis reference scrubbing (Section D)
        text = re.sub(r"\bResolves\s+H\d+\s*[-—:]\s*", "Resolves: ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bResolves\s+H\d+\b", "Resolves target proposition", text, flags=re.IGNORECASE)

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
    # deterministically from those hypotheses or fallback plan now rather than shipping a
    # report with real candidate causes and an empty investigation plan.
    rc_for_questions = state.get("root_cause")
    if inv is not None and not inv.questions:
        if rc_for_questions and rc_for_questions.candidate_hypotheses:
            from app.agent.analytical_validator import derive_investigation_questions
            inv.questions = derive_investigation_questions(rc_for_questions.candidate_hypotheses)
            trace.append(AgentTraceStep.ok(
                f"Investigation plan: derived {len(inv.questions)} discriminating question(s) from "
                "candidate hypotheses (investigation planning ran before synthesis existed)"
            ))
        else:
            from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
            _, fallback_plan = build_deterministic_investigation_plan(
                state["request"].finding_text,
                evidence_ledger,
                canonical_subject=getattr(canonical, "finding_subject", None),
                canonical_state=canonical,
            )
            inv.questions = fallback_plan.questions
            if not inv.areas:
                inv.areas = fallback_plan.areas
            if not inv.evidence_to_collect:
                inv.evidence_to_collect = fallback_plan.evidence_to_collect

    # Question uniqueness (Section 5): two hypotheses must never be tested
    # with what is effectively the same question. Applied regardless of
    # which path populated inv.questions (derived above, or a real
    # Question uniqueness (Section 5): two hypotheses must never be tested
    # with what is effectively the same question. Applied regardless of
    # which path populated inv.questions (derived above, or a real
    # tool-based LLM plan).
    if inv is not None and inv.questions:
        from app.agent.analytical_validator import (
            deduplicate_investigation_questions,
            normalize_investigation_decision_tree,
            rank_questions_by_information_gain,
        )
        deduped_questions = deduplicate_investigation_questions(inv.questions)
        if len(deduped_questions) != len(inv.questions):
            trace.append(AgentTraceStep.warn(
                f"Investigation plan: removed {len(inv.questions) - len(deduped_questions)} "
                "duplicate/near-duplicate investigation question(s)"
            ))
            inv.questions = deduped_questions

        # Normalize decision tree node properties (status, activation_condition, IDs, objectives)
        inv.questions = normalize_investigation_decision_tree(inv.questions)

        # Information-gain ordering (Section 7): conflict/event-establishing
        # questions before discrimination/mechanism/recurrence/impact ones,
        # respecting tree dependencies.
        inv.questions = rank_questions_by_information_gain(inv.questions)

    # Cap investigation questions: max 5 current-event + max 5 recurrence questions
    # to ensure all independent target propositions are preserved.
    if inv is not None and inv.questions:
        from app.agent.recurrence_guard import is_previous_capa_mechanism_hypothesis
        hyp_by_id = {h.id: h for h in (rc_for_questions.candidate_hypotheses if rc_for_questions else [])}

        import re as _re
        _recurrence_target_re = _re.compile(r"REC_\d+")

        def _is_recurrence_question(q) -> bool:
            tested_id = getattr(q, "hypothesis_tested", None)
            target_id = getattr(q, "target_proposition_id", None) or ""
            # NOTE: matches the "REC_<n>" recurrence-proposition ID
            # convention only -- a bare "REC" substring check false-
            # positives on e.g. "P_RECEIPT" (the DELIVERY_VS_RECEIPT
            # investigation plan's own target-proposition ID).
            if _recurrence_target_re.search(target_id):
                return True
            hyp = hyp_by_id.get(tested_id) if tested_id else None
            return bool(hyp and is_previous_capa_mechanism_hypothesis(hyp.statement))

        current_qs = [q for q in inv.questions if not _is_recurrence_question(q)]
        recurrence_qs = [q for q in inv.questions if _is_recurrence_question(q)]
        capped_questions = current_qs[:5] + recurrence_qs[:5]
        if len(capped_questions) != len(inv.questions):
            trace.append(AgentTraceStep.warn(
                f"Investigation plan: capped to {len(capped_questions)} question(s), dropped "
                f"{len(inv.questions) - len(capped_questions)}"
            ))
            inv.questions = capped_questions

    # Clean 5-Why — independent of whether an investigation_plan exists (a
    # finding that skipped tool-based investigation can still have a
    # populated five_why from core_synthesis).
    fw = state.get("five_why")
    if fw and fw.steps:
        _fw_canonical = state.get("canonical_finding_state")
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
            from app.agent.causal_guard import (
                answer_selects_unverified_hypothesis,
                build_causal_boundary_answer,
                five_why_answer_contains_unverified_modal_causation,
                detect_unsupported_causal_specificity,
                five_why_answer_invents_mechanism_at_unknown_status,
                hypothesis_asserts_referenced_document_content,
                hypothesis_asserts_systemic_cause_without_process_evidence,
                hypothesis_asserts_unhedged_notification_failure,
                hypothesis_asserts_unlicensed_change_event_defect,
                hypothesis_statement_asserts_unsupported_causation,
            )
            _fw_verified_facts = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.VERIFIED]
            _fw_referenced_docs = (_fw_canonical.referenced_documents if _fw_canonical else None)
            # A 5-Why ANSWER is exactly as capable of asserting an
            # unlicensed causal mechanism as a hypothesis STATEMENT is --
            # "Because staff may lack training or awareness" is the same
            # invented-training-mechanism defect the hypothesis firewall
            # already rejects, just relocated into a 5-Why step and (until
            # now) checked ONLY when the step's own status happened to be
            # UNKNOWN. A step mislabeled MIXED/REPORTED by the LLM must not
            # be exempt from the same evidence-licensing standard every
            # other causal sentence in the report is held to.
            _unsupported, _reason = detect_unsupported_causal_specificity(step.answer, finding_text)
            _unsupported_causation = hypothesis_statement_asserts_unsupported_causation(step.answer, _fw_verified_facts)
            _unestablished_at_unknown = five_why_answer_invents_mechanism_at_unknown_status(step.answer, step.status)
            _systemic_escalation = hypothesis_asserts_systemic_cause_without_process_evidence(step.answer, finding_text)
            _notification_inversion = hypothesis_asserts_unhedged_notification_failure(step.answer, finding_text)
            _change_event_inversion = hypothesis_asserts_unlicensed_change_event_defect(step.answer, finding_text)
            _referenced_doc_content = hypothesis_asserts_referenced_document_content(step.answer, _fw_referenced_docs)
            # MIXED MISUSE (Section 6/8): MIXED means actual evidence
            # CONFLICTS about the same proposition -- it is not a synonym
            # for "uncertain" or a way to phrase a plausible-but-unproven
            # mechanism without committing to VERIFIED. If this finding has
            # no detected evidence conflict at all, a step cannot
            # legitimately be MIXED; downgrade to UNKNOWN rather than let
            # an unhedged causal answer hide behind a status that sounds
            # more evidence-grounded than "unknown" but asserts nothing
            # falsifiable.
            _spurious_mixed = (
                step.status == "MIXED" and not (_fw_canonical and _fw_canonical.evidence_conflicts)
            )
            # CAUSAL BOUNDARY GUARD (INV-5WHY-CAUSAL-001), defense-in-depth:
            # re-checked here independent of core_synthesis's own guard so a
            # 5-Why step that reaches this node by any path (LLM recovery,
            # deterministic fallback edited downstream, etc.) is held to the
            # same standard -- never let an UNVERIFIED candidate hypothesis
            # be functionally selected as the explanation.
            _rc_hypotheses = getattr(rc, "candidate_hypotheses", None) if rc else None
            _selected_hyp = answer_selects_unverified_hypothesis(step.answer, _rc_hypotheses, status=step.status)
            _fev_has_verified_mechanism = getattr(_fw_canonical, "mechanism_status", None) == "VERIFIED"
            _modal_causation = five_why_answer_contains_unverified_modal_causation(
                step.answer, step.status, has_verified_mechanism=_fev_has_verified_mechanism
            )
            if _selected_hyp is not None or _modal_causation:
                if _selected_hyp is not None:
                    trace.append(AgentTraceStep.warn(
                        f"Final Evidence Verification: 5-Why step answer selected unverified hypothesis "
                        f"{getattr(_selected_hyp, 'id', '?')} ({getattr(_selected_hyp, 'name', '?')}) as the "
                        "explanation — replaced with the canonical evidence boundary"
                    ))
                else:
                    trace.append(AgentTraceStep.warn(
                        "Final Evidence Verification: 5-Why step answer contained an unverified modal "
                        "causal claim (INV-5WHY-CAUSAL-002) with no verified mechanism — replaced with "
                        "the canonical evidence boundary"
                    ))
                step.answer = build_causal_boundary_answer(
                    candidate_hypotheses=_rc_hypotheses,
                    comparison_type=getattr(_fw_canonical, "comparison_type", None),
                    comparison_left=getattr(_fw_canonical, "comparison_left", None),
                    comparison_right=getattr(_fw_canonical, "comparison_right", None),
                    comparison_left_qualifier=getattr(_fw_canonical, "comparison_left_qualifier", None),
                    comparison_subtype=getattr(_fw_canonical, "comparison_subtype", None),
                    measurement_value=getattr(getattr(_fw_canonical, "measurement", None), "value", None),
                    measurement_unit=getattr(getattr(_fw_canonical, "measurement", None), "unit", None),
                    measurement_qualifier=getattr(getattr(_fw_canonical, "measurement", None), "qualifier", None),
                    observed_deviation=getattr(_fw_canonical, "observed_deviation", None),
                )
                step.status = "UNKNOWN"
            elif _unsupported or _unsupported_causation or _unestablished_at_unknown or _systemic_escalation or _notification_inversion or _change_event_inversion or _referenced_doc_content:
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: 5-Why step answer asserted an unlicensed causal "
                    f"mechanism ({step.answer!r}, status {step.status}"
                    + (f", {_reason}" if _reason else "") +
                    ") — replaced with an evidence-boundary acknowledgment"
                ))
                step.answer = "The available evidence does not establish this — the underlying cause is not established from the evidence reviewed."
                step.status = "UNKNOWN"
            elif _spurious_mixed:
                # Content passed every unsupported-mechanism check above, but
                # the status itself was illegitimate -- MIXED without any
                # detected conflict. Downgrade the status only (never
                # promote to VERIFIED here: that would require positively
                # confirming the answer traces only to VERIFIED claims,
                # which this loop isn't set up to certify) -- conservative
                # per Section 46 ("when in doubt, do not guess").
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: 5-Why step labeled MIXED with no detected evidence "
                    "conflict for this finding — downgraded to UNKNOWN (MIXED requires an actual conflict, "
                    "not merely an uncertain answer)"
                ))
                step.status = "UNKNOWN"
            valid_fw_steps.append(step)
        fw.steps = valid_fw_steps

    # 5-Why empty-chain backfill: if every step the LLM proposed was
    # rejected (e.g. a circular/reporting-behavior WHY#1, which truncates
    # the whole chain to nothing since there's no valid step before it to
    # keep), the correct outcome is a deterministic, evidence-grounded
    # chain -- never an empty one. Mirrors the hypothesis backfill above:
    # this call's attempt failed, not the underlying evidence.
    if fw is not None and not fw.steps:
        from app.agent.nodes.five_why_fallback import build_deterministic_five_why
        _backfill_canonical = state.get("canonical_finding_state")
        backfilled_fw = build_deterministic_five_why(
            finding_text, evidence_ledger,
            canonical_subject=getattr(_backfill_canonical, "finding_subject", None),
        )
        if backfilled_fw.steps:
            trace.append(AgentTraceStep.warn(
                "Final Evidence Verification: 5-Why chain was empty after this call's steps were rejected "
                "— backfilled a deterministic, evidence-grounded chain"
            ))
            fw.steps = backfilled_fw.steps
            fw.is_complete = backfilled_fw.is_complete
            fw.status_note = backfilled_fw.status_note

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
                # _clean_text can map two DIFFERENT source items onto the
                # same canonical replacement text (e.g. two SOP-identifier
                # patterns both normalizing to "approved procedure revision
                # and distribution records") -- dedup AFTER rewriting so
                # the rewrite itself never reintroduces a literal duplicate
                # (Section 10/INV-EVIDENCE-001).
                report.investigation.evidence_to_collect = list(dict.fromkeys(
                    _clean_text(e) or e for e in report.investigation.evidence_to_collect
                ))
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

    # NOTE: a "Consistency Validator" block previously lived here that
    # invented a candidate hypothesis by keyword-matching investigation
    # QUESTION text (e.g. any question containing "checklist" produced a
    # canned "CHECKLIST_USABILITY" hypothesis) whenever candidate_hypotheses
    # was empty. This inverted the required causal pipeline (evidence ->
    # proposition -> hypothesis -> question) into (question -> hypothesis),
    # and reintroduced exactly the invented-mechanism defect the guards
    # elsewhere in this node exist to prevent -- an investigation question
    # is licensed to exist without any hypothesis backing it (a finding can
    # legitimately have investigation areas but zero causal hypotheses when
    # no evidence is eligible), so "a question exists" must never be
    # sufficient justification to manufacture one. Removed rather than
    # patched: no phrase-matching branch here is fixable, since the defect
    # is the existence of the branch itself, not its wording.

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
    specificity = "HIGH"
    if rc is not None:
        from app.agent.recurrence_guard import is_previous_capa_mechanism_hypothesis as _is_recurrence_hyp
        from app.services.semantic_subject import classify_finding_specificity
        _reported_for_specificity = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.REPORTED]
        specificity = classify_finding_specificity(
            finding_text, _reported_for_specificity, mechanism.status if mechanism else None,
            canonical.deviation_condition if canonical else None,
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

            # LOW specificity must not mean NO INVESTIGATION -- generate a
            # foundational, evidence-acquisition investigation plan (never
            # hypotheses; these questions establish WHAT is applicable, not
            # WHY a deviation occurred). Overrides whatever the LLM/
            # deterministic path produced, since anything generated before
            # this point may itself have presupposed an unlicensed cause.
            from app.models.agent import InvestigationQuestion
            _foundational_questions = [
                InvestigationQuestion(
                    question="Which approved procedure and specific requirement were applicable?",
                    purpose="Establish the specific requirement the finding alleges was not followed",
                    evidence="Applicable approved procedure and the specific requirement/step it defines",
                    hypothesis_tested=None,
                ),
                InvestigationQuestion(
                    question="What objective evidence demonstrates the deviation described in the finding?",
                    purpose="Establish objective (not merely alleged) evidence of the deviation",
                    evidence="Records, logs, or documents demonstrating the deviation",
                    hypothesis_tested=None,
                ),
                InvestigationQuestion(
                    question="Which specific step or activity was not performed as required?",
                    purpose="Identify the specific affected activity or control, not just the general allegation",
                    evidence="Process/activity records relevant to the applicable requirement",
                    hypothesis_tested=None,
                ),
                InvestigationQuestion(
                    question="What date or period was affected, and what is the scope of the deviation?",
                    purpose="Establish the affected time period and scope/extent",
                    evidence="Records establishing the affected date, period, and scope",
                    hypothesis_tested=None,
                ),
                InvestigationQuestion(
                    question="Which role or process was responsible for the applicable requirement?",
                    purpose="Establish responsibility and applicable control before any specific mechanism can be investigated",
                    evidence="Responsibility matrix or applicable control documentation",
                    hypothesis_tested=None,
                ),
            ]
            if inv is not None:
                inv.questions = _foundational_questions
                inv.areas = [
                    "Applicable requirement and objective evidence of the deviation",
                    "Affected activity, scope, and time period",
                    "Responsibility and applicable control",
                ]
                inv.evidence_to_collect = [
                    "Applicable approved procedure",
                    "Specific requirement/step alleged not followed",
                    "Objective evidence demonstrating the deviation",
                    "Affected date/period and scope of the deviation",
                    "Responsible role/process and applicable control",
                ]
            # Affected object/process/period/effect/evidence-needed must
            # stay foundational too -- a LOW-specificity finding provides no
            # basis for a specific object beyond "the alleged requirement",
            # and forcing it through the normal actor+topic template
            # produces exactly the "Department procedure status for
            # procedure" class of nonsense (concatenating actor + generic
            # noun + "status" to fill a field the evidence doesn't support).
            if impact is not None:
                impact.affected_object = "Required procedure compliance"
                impact.process_at_risk = "Procedure adherence and compliance control"
                impact.affected_period = "UNKNOWN"
                impact.relevant_change = "NOT_ESTABLISHED"
                impact.potential_effect = (
                    "The observed noncompliance may affect procedural adherence. Scope and downstream "
                    "consequences require assessment against the applicable requirement and objective records."
                )
                impact.evidence_needed = (
                    "Applicable approved procedure; specific requirement/step; objective evidence "
                    "demonstrating the deviation; affected date/period and scope; responsible role/process"
                )

            # 5-Why sanitization for LOW specificity: the per-answer content
            # filters above (detect_unsupported_causal_specificity /
            # hypothesis_statement_asserts_unsupported_causation) catch
            # known unsupported-mechanism SHAPES, but a hedged speculative
            # answer with no causal connector and no matching mechanism
            # pattern ("Documentation may be outdated or insufficient.",
            # "Monitoring logs may show no regular checks.") can still slip
            # through as an enumeration gap. LOW specificity has already
            # established -- independent of any per-answer wording check --
            # that there is no causal evidence in this case at all, so every
            # step beyond the first (the observation itself) is forced to
            # the same safe acknowledgment rather than trusting further
            # per-phrase pattern coverage to keep up with paraphrase drift.
            fw_low = state.get("five_why")
            if fw_low is not None and fw_low.steps:
                for _idx, _step in enumerate(fw_low.steps):
                    if _idx == 0:
                        continue
                    if _step.status not in ("UNKNOWN", "REQUIRES_EVIDENCE"):
                        trace.append(AgentTraceStep.warn(
                            f"Final Evidence Verification: LOW-specificity finding — 5-Why step {_idx + 1} "
                            f"({_step.status}, {_step.answer!r}) sanitized to the evidence boundary"
                        ))
                    _step.answer = "The available evidence does not establish this — the underlying cause is not established from the evidence reviewed."
                    _step.status = "UNKNOWN"
                # Never more than 2 steps for a LOW-specificity finding:
                # observation + one evidence-boundary acknowledgment is the
                # full defensible chain when there is no causal evidence at
                # all to reason further from.
                fw_low.steps = fw_low.steps[:2]
                fw_low.is_complete = False
                fw_low.status_note = "Evidence boundary reached — insufficient specificity to identify a causal mechanism."

    # Invariant enforcement: ensure rejected hypotheses (from causal guard, critic, or grounding guard) do not appear in final output.
    # Deliberately `rc is not None` (not `rc and rc.candidate_hypotheses`):
    # the block must still run -- and reach the evidence-grounded backfill
    # below -- even when candidate_hypotheses started EMPTY (an LLM call
    # that produced zero hypotheses to begin with), not only when a
    # non-empty list gets filtered down to empty during processing. An
    # empty starting list is inert through the loop/cap logic below.
    if rc is not None:
        from app.agent.causal_guard import (
            detect_unsupported_causal_specificity,
            hypothesis_asserts_referenced_document_content,
            hypothesis_asserts_self_referential_evidence,
            hypothesis_asserts_systemic_cause_without_process_evidence,
            hypothesis_asserts_unhedged_notification_failure,
            hypothesis_asserts_unhedged_unverified_completion,
            hypothesis_asserts_unlicensed_change_event_defect,
            hypothesis_attacks_statement_credibility,
            hypothesis_contradicts_verified_completion,
            hypothesis_name_is_generic,
            hypothesis_statement_asserts_unsupported_causation,
            hypothesis_statement_is_generic_causal_bucket,
        )
        from app.services import llm_metrics as _llm_metrics
        _referenced_docs = canonical.referenced_documents if canonical else None
        filtered_final_hyps = []
        for h in rc.candidate_hypotheses:
            if h.status == "REFUTED":
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} because status was REFUTED"
                ))
                _llm_metrics.record_validation_rejection(reason="invalid_hypothesis", node="final_evidence_verification")
                continue
            from app.agent.causal_guard import is_evidence_state_not_hypothesis
            if is_evidence_state_not_hypothesis(h.statement, h.name):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — describes an evidence state "
                    "or investigation uncertainty rather than proposing a concrete causal mechanism"
                ))
                _llm_metrics.record_validation_rejection(reason="invalid_hypothesis", node="final_evidence_verification")
                continue
            if hypothesis_asserts_referenced_document_content(h.statement, _referenced_docs):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — asserts content of a "
                    "document the finding only referenced/cited but which was unavailable for inspection "
                    "(Referenced-Evidence Boundary: a document being mentioned is not evidence of what it "
                    "contains)"
                ))
                _llm_metrics.record_validation_rejection(reason="unsupported_causal_specificity", node="final_evidence_verification")
                continue
            if hypothesis_asserts_self_referential_evidence(h.statement):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — statement narrates its own "
                    "'supporting evidence' inline (e.g. 'records show...') instead of being grounded in "
                    "the actual evidence ledger; a claim asserting evidence exists is not evidence"
                ))
                _llm_metrics.record_validation_rejection(reason="invalid_provenance", node="final_evidence_verification")
                continue
            if hypothesis_asserts_systemic_cause_without_process_evidence(h.statement, finding_text):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — escalates to a systemic "
                    "process/system/control-level claim without the finding citing any review, audit, "
                    "documentation, or record finding about that process (Level 1→4 jump)"
                ))
                _llm_metrics.record_validation_rejection(reason="unsupported_causal_specificity", node="final_evidence_verification")
                continue
            if hypothesis_statement_is_generic_causal_bucket(h.statement):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — statement names only a "
                    "generic causal category (execution/task-control/process/etc. factors) rather than "
                    "an evidence-grounded mechanism; a vague hedged bucket is not a valid substitute for "
                    "a specific one"
                ))
                _llm_metrics.record_validation_rejection(reason="invalid_hypothesis", node="final_evidence_verification")
                continue
            if hypothesis_name_is_generic(h.name):
                # A vague title ("Process Oversight") doesn't necessarily
                # mean the underlying statement is ungrounded -- rename
                # rather than discard a hypothesis whose statement content
                # still passes every other check below. Derived from the
                # statement itself (never an internal-looking placeholder
                # like "CANDIDATE_MECHANISM_H2" -- that leaks implementation
                # detail straight into the auditor-facing report).
                from app.agent.causal_guard import derive_hypothesis_title_from_statement
                new_name = derive_hypothesis_title_from_statement(h.statement, h.id)
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: hypothesis {h.id} had a generic, non-testable title "
                    f"({h.name!r}) — renamed to {new_name!r} derived from its own statement"
                ))
                h.name = new_name
            if hypothesis_attacks_statement_credibility(h.statement):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — attacks the credibility of "
                    "a reported statement instead of reasoning about the underlying proposition"
                ))
                _llm_metrics.record_validation_rejection(reason="invalid_hypothesis", node="final_evidence_verification")
                continue
            if mechanism and mechanism.statement:
                if hypothesis_contradicts_mechanism(h.statement, mechanism):
                    trace.append(AgentTraceStep.warn(
                        f"Final Evidence Verification: removed hypothesis {h.id} — contradicts mechanism"
                    ))
                    _llm_metrics.record_validation_rejection(reason="unsupported_causation", node="final_evidence_verification")
                    continue
                if mechanism_already_names_generic_hypothesis(h.statement, mechanism):
                    trace.append(AgentTraceStep.warn(
                        f"Final Evidence Verification: removed hypothesis {h.id} — restates mechanism"
                    ))
                    _llm_metrics.record_validation_rejection(reason="invalid_hypothesis", node="final_evidence_verification")
                    continue
            if hypothesis_contradicts_verified_completion(h.statement, verified_facts):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — contradicts a VERIFIED "
                    "fact that the thing it claims is deficient was actually completed"
                ))
                _llm_metrics.record_validation_rejection(reason="unsupported_causation", node="final_evidence_verification")
                continue
            is_unsupported, reason = detect_unsupported_causal_specificity(h.statement, finding_text)
            if is_unsupported:
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — {reason}"
                ))
                _llm_metrics.record_validation_rejection(reason="unsupported_causal_specificity", node="final_evidence_verification")
                continue
            if hypothesis_asserts_unhedged_unverified_completion(h.statement):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — asserts an unverified "
                    "activity as settled fact ('was conducted/completed but not documented') instead of "
                    "preserving uncertainty ('may have been completed, but the record was unavailable')"
                ))
                _llm_metrics.record_validation_rejection(reason="unsupported_causal_specificity", node="final_evidence_verification")
                continue
            if hypothesis_asserts_unhedged_notification_failure(h.statement, finding_text):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — asserts a notification/"
                    "communication/distribution/acknowledgement event did NOT occur as settled fact "
                    "(causal-event inversion: a downstream reported awareness state does not establish "
                    "the upstream communication event failed)"
                ))
                _llm_metrics.record_validation_rejection(reason="unsupported_causal_specificity", node="final_evidence_verification")
                continue
            if hypothesis_asserts_unlicensed_change_event_defect(h.statement, finding_text):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — asserts the change event "
                    "itself (revision/amendment/procedure change) was handled defectively as settled fact "
                    "(e.g. 'not properly documented or disseminated'), when the finding gives no "
                    "independent account of how the change was handled"
                ))
                _llm_metrics.record_validation_rejection(reason="unsupported_causal_specificity", node="final_evidence_verification")
                continue
            if hypothesis_statement_asserts_unsupported_causation(h.statement, verified_facts):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: removed hypothesis {h.id} — statement asserts causation "
                    "('because'/'due to'/'caused by'/etc.) not grounded in a VERIFIED fact"
                ))
                _llm_metrics.record_validation_rejection(reason="unsupported_causation", node="final_evidence_verification")
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
            from app.services.text_grounding import significant_words as _significant_words
            _subject_words = (
                _significant_words(canonical.finding_subject)
                if canonical and canonical.finding_subject else None
            )
            # Common-factor hypotheses (Section: COMMON-FACTOR CAUSAL
            # REASONING) are deliberately POSSIBLE/INDICATIVE by
            # construction -- a pattern across multiple claims, not a
            # single-claim overlap determine_hypothesis_status looks for.
            # Re-deriving status/strength from single-claim overlap here
            # would either strand it at status=POSSIBLE with the
            # provenance-losing generic "NONE" strength, or (worse) let a
            # coincidental wording overlap with one VERIFIED claim promote
            # it toward SUPPORTED on pattern evidence alone -- exactly the
            # premature-causality risk this hypothesis type must never take.
            if h.name == "SHARED_SYSTEM_COMMON_FACTOR":
                pass
            else:
                det_status, det_strength = determine_hypothesis_status(
                    h.statement,
                    verified_facts,
                    reported_claims,
                    canonical.evidence_conflicts if canonical else None,
                    mechanism,
                    allow_verified_promotion=not is_recurrence_hyp,
                    subject_words=_subject_words,
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
            # Semantic deduplication of identical/equivalent statements
            existing_match = None
            stmt_norm = re.sub(r"\s+", " ", h.statement.strip().lower())
            for existing in filtered_final_hyps:
                if re.sub(r"\s+", " ", existing.statement.strip().lower()) == stmt_norm:
                    existing_match = existing
                    break
            if existing_match:
                if h.supporting_claim_ids:
                    existing_match.supporting_claim_ids = list(dict.fromkeys([*(existing_match.supporting_claim_ids or []), *h.supporting_claim_ids]))
                if h.supporting_evidence:
                    existing_match.supporting_evidence = list(dict.fromkeys([*(existing_match.supporting_evidence or []), *h.supporting_evidence]))
                continue

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
                _llm_metrics.record_validation_rejection(reason="other", node="final_evidence_verification")
        rc.candidate_hypotheses = current_event_hyps + recurrence_hyps

        # Evidence-grounded hypothesis backfill: if EVERY hypothesis the
        # LLM proposed for this call turned out to be invalid (restated an
        # evidence gap, invented an ungrounded domain/mechanism, etc.) and
        # got removed above, the correct outcome is NOT an empty hypothesis
        # list -- that only means THIS attempt failed to articulate a valid
        # mechanism, not that the evidence is too thin to support one. A
        # finding with real specificity (an entity, a date/period, a
        # reported statement, or an established mechanism -- i.e. anything
        # above LOW specificity) can always be given the same deterministic,
        # evidence-grounded candidate hypotheses the fallback/recovery paths
        # already use for this exact situation (never invents anything --
        # reuses the one canonical generator). LOW-specificity findings are
        # correctly exempt: there genuinely isn't enough to hypothesize
        # about (handled by the LOW-specificity gate above).
        if not rc.candidate_hypotheses and specificity != "LOW":
            from app.agent.nodes.plan_investigation_fallback import (
                build_conditional_capa_actions,
                build_deterministic_investigation_plan,
            )
            backfill_hyps, backfill_plan = build_deterministic_investigation_plan(
                finding_text, evidence_ledger,
                canonical_subject=getattr(canonical, "finding_subject", None),
                canonical_state=canonical,
            )
            # Backfilled hypotheses must pass through the SAME unsupported-
            # specificity firewall as every other path (no output may bypass
            # the single authoritative checkpoint) -- defense in depth even
            # though the deterministic generator itself is expected to stay
            # evidence-conditioned; a future branch or edge phrasing that
            # slips past that expectation must still be caught here, not
            # silently reach the report just because it came from the
            # "trusted" fallback generator.
            backfill_hyps = [
                h for h in backfill_hyps
                if not detect_unsupported_causal_specificity(h.statement, finding_text)[0]
                and not hypothesis_statement_is_generic_causal_bucket(h.statement)
                and not hypothesis_asserts_self_referential_evidence(h.statement)
                and not hypothesis_asserts_systemic_cause_without_process_evidence(h.statement, finding_text)
                and not hypothesis_asserts_unhedged_notification_failure(h.statement, finding_text)
                and not hypothesis_asserts_unlicensed_change_event_defect(h.statement, finding_text)
                and not hypothesis_asserts_referenced_document_content(h.statement, _referenced_docs)
            ]
            if backfill_hyps:
                rc.candidate_hypotheses = backfill_hyps
                from app.services import llm_metrics as _llm_metrics
                _llm_metrics.increment("deterministic_hypotheses_generated", len(backfill_hyps))
                trace.append(AgentTraceStep.warn(
                    "Final Evidence Verification: this call's proposed hypotheses were all invalid "
                    f"(evidence-gap restatement / ungrounded domain / etc.) — backfilled {len(backfill_hyps)} "
                    "deterministic, evidence-grounded candidate hypothesis/hypotheses rather than reporting "
                    "none for a finding with sufficient specificity to support hypothesis generation"
                ))
                if capa is not None:
                    from app.services.semantic_subject import extract_semantic_subject, topic_word
                    backfill_subject = getattr(canonical, "finding_subject", None)
                    if not backfill_subject or backfill_subject == "UNKNOWN":
                        # canonical extraction isn't always populated on
                        # every path into this node -- falling back to the
                        # RAW finding text as "subject" makes topic_word()
                        # extract a garbled topic (e.g. "operator" from a
                        # training finding) and dumps the entire finding
                        # into the CAPA action's tail phrase. Extract a real
                        # subject directly from the finding text instead.
                        backfill_subject = extract_semantic_subject(finding_text).finding_subject or finding_text
                    capa.conditional_actions = build_conditional_capa_actions(
                        backfill_hyps, backfill_subject, topic_word(backfill_subject)
                    )
            # Investigation questions/areas are adopted from the backfill
            # plan whenever inv has no questions, REGARDLESS of whether any
            # hypothesis survived -- a finding can correctly produce zero
            # causal hypotheses (no reported explanation licenses one) while
            # still deserving a genuine, non-presupposing investigation plan
            # (Section 10/11: never fall back to "no questions generated"
            # just because no hypothesis exists to attach them to).
            if inv is not None and not inv.questions and backfill_plan.questions:
                inv.questions = backfill_plan.questions
                inv.areas = backfill_plan.areas
                inv.evidence_to_collect = backfill_plan.evidence_to_collect

        # Hypothesis-ID consistency: investigation questions test independent propositions (Section 6).
        # A question MUST NOT disappear merely because its related hypothesis was demoted or rejected.
        # Clear the dangling hypothesis_tested field if the hypothesis is not in candidate_hypotheses,
        # but the question and its target_proposition_id MUST SURVIVE.
        valid_hyp_ids = {h.id for h in rc.candidate_hypotheses}
        if inv is not None and inv.questions:
            for q in inv.questions:
                if getattr(q, "hypothesis_tested", None) and q.hypothesis_tested not in valid_hyp_ids:
                    q.hypothesis_tested = None
        if capa is not None and capa.conditional_actions:
            _hyp_id_re = re.compile(r"\bH\d+\b")
            kept_actions = []
            for a in capa.conditional_actions:
                m = _hyp_id_re.search(a.if_cause_confirmed or "")
                if m and m.group(0) not in valid_hyp_ids:
                    trace.append(AgentTraceStep.warn(
                        f"Final Evidence Verification: removed CAPA conditional action referencing "
                        f"{m.group(0)!r} — absent from the final hypothesis set (hypothesis-ID drift)"
                    ))
                    continue
                kept_actions.append(a)
            capa.conditional_actions = kept_actions

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
        # Investigation areas / CAPA potential_areas (Section K): aggregate
        # areas from both surviving hypotheses and surviving investigation questions.
        from app.agent.analytical_validator import derive_investigation_areas
        derived_areas = derive_investigation_areas(
            filtered_final_hyps,
            questions=inv.questions if inv else None,
            existing_areas=inv.areas if inv else None,
            canonical_subject=getattr(canonical, "finding_subject", None),
        )
        if derived_areas:
            if capa:
                capa.potential_areas = derived_areas
            if inv:
                inv.areas = derived_areas

        # Causal-proposition eligibility layer (primary enforcement, per the
        # structured causal-proposition architecture): every guard above is
        # a pattern-matcher on hypothesis PROSE and stays in place as
        # defense-in-depth, but this layer independently COMPUTES each
        # surviving hypothesis's evidence support from the structured claim
        # ledger rather than trusting that no guard missed a paraphrase. A
        # hypothesis whose support_level resolves to NONE/INDIRECT (no
        # VERIFIED or REPORTED_CAUSAL_MECHANISM claim actually supports it —
        # only topical relatedness, e.g. sharing the finding's own subject
        # nouns) is not dropped outright; it is demoted to an investigation
        # area, preserving the causal SPACE the LLM identified without
        # asserting it as a candidate cause the evidence doesn't support.
        from app.agent.causal_model import (
            HypothesisEligibility,
            build_causal_proposition,
            claims_from_evidence_ledger,
            derive_root_cause_status,
        )
        _claims = claims_from_evidence_ledger(evidence_ledger)
        from app.services.text_grounding import significant_words as _sig_words
        _subj_words = (
            _sig_words(canonical.finding_subject)
            if canonical and canonical.finding_subject and canonical.finding_subject != "UNKNOWN" else None
        )
        if not _subj_words:
            # canonical extraction isn't always populated on every path into
            # this node (e.g. a hand-built state in a fallback/recovery
            # branch) -- without SOME subject exclusion, a hypothesis that
            # merely repeats the finding's own subject nouns ("daily
            # equipment inspection checklist") would spuriously overlap the
            # finding's own VERIFIED claim and be scored VERIFIED_SUPPORT
            # for restating what the finding is about, not for being
            # corroborated. Fall back to extracting the subject directly
            # from the raw finding text.
            from app.services.semantic_subject import extract_semantic_subject
            _fallback_subject = extract_semantic_subject(finding_text).finding_subject
            _subj_words = _sig_words(_fallback_subject) if _fallback_subject else None
        _eligible_hyps = []
        _demoted_ids = set()
        _new_areas = []
        for h in rc.candidate_hypotheses:
            prop = build_causal_proposition(h, _claims, finding_text, subject_words=_subj_words)
            if prop.eligibility == HypothesisEligibility.ELIGIBLE:
                _eligible_hyps.append(h)
                continue
            _demoted_ids.add(h.id)
            _supp_str = getattr(prop.support_level, "value", str(prop.support_level))
            trace.append(AgentTraceStep.warn(
                f"Final Evidence Verification: demoted hypothesis {h.id} to investigation area — "
                f"computed support_level={_supp_str} (topical relatedness to the finding's "
                "own subject or a reported downstream state is not causal support; only a VERIFIED claim "
                "or a REPORTED_CAUSAL_MECHANISM claim licenses a candidate hypothesis)"
            ))
            # Derived from the STATEMENT, never the raw internal name --
            # a bucket-style name like "PROCESS_EXECUTION_COMPLIANCE_GAP"
            # humanizes into "Process Execution Compliance Gap", which
            # reads as an established finding/cause, not an investigation
            # target (Section 11: investigation areas must not be rendered
            # as if they were causes). derive_hypothesis_title_from_statement
            # is the same statement-derived title logic already used to
            # rename generic hypothesis NAMES elsewhere in this file.
            from app.agent.causal_guard import derive_hypothesis_title_from_statement as _derive_area_title
            area_title = _derive_area_title(h.statement, h.id) if h.statement else (h.name or "").replace("_", " ").strip().title()
            if area_title and area_title not in (inv.areas if inv else []) and area_title not in _new_areas:
                _new_areas.append(area_title)
        if _demoted_ids:
            rc.candidate_hypotheses = _eligible_hyps
            if inv is not None:
                inv.areas = [*inv.areas, *_new_areas]
                if inv.questions:
                    for q in inv.questions:
                        if getattr(q, "hypothesis_tested", None) in _demoted_ids:
                            q.hypothesis_tested = None
            if capa is not None and capa.conditional_actions:
                _demoted_re = re.compile(r"\bH\d+\b")
                _kept_actions = []
                for a in capa.conditional_actions:
                    m = _demoted_re.search(a.if_cause_confirmed or "")
                    if m and m.group(0) in _demoted_ids:
                        continue
                    _kept_actions.append(a)
                capa.conditional_actions = _kept_actions
            if not rc.candidate_hypotheses:
                rc.status = derive_root_cause_status([], has_unresolved_conflict=False)

        # Semantic CAPA deduplication (Section 24): two conditional actions
        # for the SAME hypothesis that turn out to be near-restatements of
        # each other (regardless of which generation path produced them —
        # deterministic template, LLM, or backfill) are merged, keeping the
        # more complete text. Runs unconditionally, not only when the
        # eligibility layer above demoted something, since duplicate CAPA
        # actions are an independent defect from hypothesis eligibility.
        if capa is not None and capa.conditional_actions:
            from app.agent.analytical_validator import deduplicate_capa_actions
            _deduped_actions = deduplicate_capa_actions(capa.conditional_actions)
            if len(_deduped_actions) != len(capa.conditional_actions):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: merged {len(capa.conditional_actions) - len(_deduped_actions)} "
                    "semantically duplicate CAPA action(s) targeting the same hypothesis"
                ))
                capa.conditional_actions = _deduped_actions

        # ROOT-CAUSE STATUS EVIDENCE JUSTIFICATION (LLM-boundary hardening):
        # the LLM can write status="CONFIRMED"/"VERIFIED" directly into its
        # JSON response, and status_normalizer maps that WORD onto
        # RootCauseStatus.VERIFIED -- but normalization is a syntactic
        # mapping (valid enum value), never a semantic check that the claim
        # is actually justified. A VERIFIED/SUPPORTED root-cause status must
        # be grounded EITHER by a hypothesis whose evidence_strength was
        # independently, deterministically computed as VERIFIED, OR --
        # when no hypothesis exists at all, e.g. a finding whose own
        # VERIFIED claim directly states the cause with no candidate
        # reasoning needed -- by a VERIFIED evidence-ledger claim whose
        # text substantially (>=2 significant words) overlaps the root
        # cause's own statement/narrative. Never trusted from the LLM's
        # own status word alone (Phase 7/8: "LLM must never decide
        # evidence status"). The >=2-word substantive-overlap threshold
        # matters: without it, a VERIFIED claim about something else
        # entirely (e.g. "no record was available") could spuriously
        # ground an unrelated causal statement merely by sharing one
        # incidental word.
        if rc.status in (RootCauseStatus.VERIFIED, RootCauseStatus.SUPPORTED, "VERIFIED", "SUPPORTED"):
            has_verified_hyp = any(h.evidence_strength == "VERIFIED" for h in rc.candidate_hypotheses)
            grounded_by_direct_claim = False
            if not has_verified_hyp:
                from app.services.text_grounding import significant_words as _sig_words_rc
                _rc_words = _sig_words_rc(rc.statement or rc.narrative or "")
                for _e in evidence_ledger:
                    if _e.status == EvidenceStatus.VERIFIED and len(_sig_words_rc(_e.claim) & _rc_words) >= 2:
                        grounded_by_direct_claim = True
                        break
            if not has_verified_hyp and not grounded_by_direct_claim:
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: root cause status {rc.status!r} was not backed by any "
                    "independently-verified hypothesis or a directly-overlapping VERIFIED claim — "
                    "downgraded to NOT_ESTABLISHED (a status word alone, from the LLM or otherwise, "
                    "never establishes root cause)"
                ))
                rc.status = RootCauseStatus.NOT_ESTABLISHED

        # Authoritative promotion check: if any candidate hypothesis is proven by verified evidence, promote it; otherwise enforce POSSIBLE
        from app.agent.causal_graph import evaluate_root_cause_eligibility, select_authoritative_leading_hypothesis
        for h in rc.candidate_hypotheses:
            el, supp, _, _, c_lvl, promo = evaluate_root_cause_eligibility(
                h,
                evidence_items=evidence_ledger,
                conflicts=canonical.evidence_conflicts if canonical else None,
                referenced_docs=canonical.referenced_documents if canonical else None,
                canonical_state=canonical,
            )
            if promo and supp == SupportLevel.SUPPORTED:
                h.status = "SUPPORTED"
                h.causal_level = c_lvl
                # evaluate_root_cause_eligibility only returns
                # (promo=True, SupportLevel.SUPPORTED) when it found direct
                # VERIFIED-evidence causal proof (has_verified_causal_proof) --
                # evidence_strength must reflect that authoritative finding,
                # never sit stale at whatever it was before promotion
                # (Section 8: SUPPORTED/ESTABLISHED must never have zero
                # evidence provenance).
                h.evidence_strength = "VERIFIED"
            else:
                h.status = "POSSIBLE"
                # INDICATIVE is preserved, never reset to NONE: a common-
                # factor/pattern-derived hypothesis (Section: COMMON-FACTOR
                # CAUSAL REASONING) legitimately has a real, citable signal
                # (the shared system/process across multiple locations) even
                # though it isn't promoted to SUPPORTED -- collapsing that
                # to NONE would indistinguishably conflate "a pattern points
                # here" with "there is no evidence at all here."
                if h.evidence_strength != "INDICATIVE":
                    h.evidence_strength = "NONE"

        lead_id, lead_mode, authoritative_rc_status, lead_rationale = select_authoritative_leading_hypothesis(
            rc.candidate_hypotheses,
            conflicts=canonical.evidence_conflicts if canonical else None,
            evidence_ledger=evidence_ledger,
        )
        lead_status_literal = "SELECTED" if lead_mode == "SELECTED" else ("TIED" if lead_mode == "TIED" else "NONE")
        if authoritative_rc_status in (RootCauseStatus.SUPPORTED, RootCauseStatus.ESTABLISHED) and lead_id:
            rc.status = authoritative_rc_status
            rc.leading_hypothesis = lead_id
            rc.leading_hypothesis_status = lead_status_literal
            if lead_rationale:
                rc.leading_hypothesis_rationale = lead_rationale
                rc.category = "TO_BE_CONFIRMED"
        else:
            rc.status = RootCauseStatus.NOT_ESTABLISHED
            rc.leading_hypothesis = None
            rc.leading_hypothesis_status = lead_status_literal
        from app.agent.recurrence_guard import build_recurrence_rationale, detect_recurrence
        _finding_txt = getattr(state.get("request"), "finding_text", "")
        _rec = detect_recurrence(_finding_txt)
        if _rec.is_recurring:
            rc.risk_of_recurrence = "HIGH"
            rc.risk_of_recurrence_rationale = build_recurrence_rationale(_rec)

        # Causal-verb firewall on the "Why" text (narrative/root_cause_basis):
        # the same "because"/"due to"/"caused by" over-claim that a
        # hypothesis STATEMENT can make is just as much a problem in the
        # free-text explanation of why root cause is/isn't established --
        # "The checklist was not completed due to lack of awareness" asserts
        # causation in prose even though every hypothesis stayed correctly
        # unresolved. Replaced with a safe, evidence-preserving fallback
        # rather than silently trusting whichever causal wording the LLM
        # happened to use.
        from app.agent.causal_guard import hypothesis_statement_asserts_unsupported_causation
        _conflicts = getattr(canonical, "evidence_conflicts", []) if canonical else []
        if _conflicts:
            _EVIDENCE_BOUNDARY_WHY = (
                "Available evidence establishes a conflict between recorded and reported information. "
                "The causal mechanism cannot yet be established. The investigation plan identifies the "
                "objective evidence required to resolve the conflict."
            )
        else:
            _EVIDENCE_BOUNDARY_WHY = (
                "The available evidence establishes the observed deviation and any reported statements "
                "about it, but does not establish the causal relationship between them."
            )
        for _field in ("narrative", "root_cause_basis"):
            _text = getattr(rc, _field, None)
            if _text and hypothesis_statement_asserts_unsupported_causation(_text, verified_facts):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: rc.{_field} asserted unsupported causation "
                    "('because'/'due to'/etc.) — replaced with an evidence-preserving statement"
                ))
                setattr(rc, _field, _EVIDENCE_BOUNDARY_WHY)

        # When root cause is NOT_ESTABLISHED, the "Why" text must explain
        # the evidence BOUNDARY, never propose a causal explanation -- even
        # a hedged one ("The deviation may stem from a communication gap").
        # A hedge word makes a HYPOTHESIS legitimate (uncertainty-preserving
        # candidate) but does not make a narrative explanation legitimate
        # when there is, by definition, no established cause for it to
        # tentatively point to: unlike a hypothesis, the narrative isn't
        # discriminated against evidence, so any causal attribution in it --
        # hedged or not -- silently reintroduces exactly the invented-
        # mechanism defect the hypothesis firewall above exists to prevent.
        # This check is unconditional on status alone, not on the specific
        # wording used, so it cannot be evaded by a new paraphrase the way
        # a "because"/"due to" connector list can.
        if rc.status in (RootCauseStatus.NOT_ESTABLISHED, "NOT_ESTABLISHED"):
            for _field in ("narrative", "root_cause_basis"):
                _text = getattr(rc, _field, None)
                if _text != _EVIDENCE_BOUNDARY_WHY:
                    if _text:
                        trace.append(AgentTraceStep.warn(
                            f"Final Evidence Verification: rc.{_field} proposed a causal explanation while "
                            "root cause is NOT_ESTABLISHED — replaced with an evidence-boundary statement"
                        ))
                    setattr(rc, _field, _EVIDENCE_BOUNDARY_WHY)

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
                for _field in ("narrative", "root_cause_basis"):
                    _text = getattr(rc, _field, None)
                    if _text != _EVIDENCE_BOUNDARY_WHY:
                        setattr(rc, _field, _EVIDENCE_BOUNDARY_WHY)
            apply_conflict_tie_override(rc, has_unresolved_conflict)

    # Rule 1 & 12: 5-Why provenance and conflict boundary enforcement
    if fw and fw.steps:
        from app.agent.causal_guard import _ATTRIBUTION_LANGUAGE_RE
        for step in fw.steps:
            if step.answer and _ATTRIBUTION_LANGUAGE_RE.search(step.answer):
                # SUPPORTED is just as capable of over-claiming an
                # attributed statement as VERIFIED -- checked here too
                # (defense-in-depth alongside core_synthesis's own guard)
                # so no path (LLM, recovery, downstream edit) can leave a
                # narrated report labeled as though it were an established
                # causal finding.
                if step.status in ("VERIFIED", "SUPPORTED"):
                    _old_status = step.status
                    step.status = "REPORTED"
                    trace.append(AgentTraceStep.warn(
                        f"Final Evidence Verification: downgraded 5-Why answer containing attribution from {_old_status} to REPORTED"
                    ))
        if canonical and canonical.evidence_conflicts:
            # When evidence conflicts exist, rebuild deterministic 5-Why chain if LLM fabricated steps
            from app.agent.nodes.five_why_fallback import build_deterministic_five_why
            fw = build_deterministic_five_why(
                finding_text, evidence_ledger,
                canonical_subject=getattr(canonical, "finding_subject", None),
            )
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

        # process_at_risk must describe THIS finding's topic, not one the
        # LLM substituted in (e.g. "procedure revision documentation" for a
        # training finding) -- but it legitimately names the CONTROL
        # DOMAIN (e.g. "calibration"), which is expected to use different
        # vocabulary than the entity/affected-object subject (e.g. "balance
        # BAL-014") by design (Phase 6: "do not let the entity resolver
        # determine the process name"). Comparing against canon_subject
        # alone made every correct, entity-independent process name look
        # like drift and get overwritten with a generic entity-derived
        # label -- exactly the second-producer defect this check was
        # supposed to prevent, just relocated to a different field.
        # Compare against the full finding text instead: legitimate
        # process vocabulary appears somewhere in the finding even when it
        # doesn't share words with the entity specifically; a genuinely
        # substituted, unrelated process (from a different finding
        if impact.process_at_risk:
            from app.services.semantic_subject import strip_quantity_prefix
            cleaned_proc = strip_quantity_prefix(impact.process_at_risk)
            if cleaned_proc and cleaned_proc != impact.process_at_risk:
                impact.process_at_risk = cleaned_proc[0].upper() + cleaned_proc[1:]
        if impact.affected_object:
            from app.services.semantic_subject import strip_quantity_prefix
            cleaned_obj = strip_quantity_prefix(impact.affected_object)
            if cleaned_obj and cleaned_obj != impact.affected_object:
                impact.affected_object = cleaned_obj[0].upper() + cleaned_obj[1:]

        from app.services.semantic_subject import is_actor_noun
        if impact.process_at_risk and (is_actor_noun(impact.process_at_risk) or is_actor_noun(impact.process_at_risk.replace(" control", "").strip())):
            impact.process_at_risk = "Inspection checklist completion and record control" if "checklist" in finding_text.lower() else "Required procedure compliance and control"
        if impact.affected_object and is_actor_noun(impact.affected_object):
            impact.affected_object = (
                canonical.affected_object if (canonical and canonical.affected_object and not is_actor_noun(canonical.affected_object))
                else ("Revised inspection checklist completion" if "checklist" in finding_text.lower() else "Required procedure compliance")
            )

        if canon_subject:
            topic = topic_word(canon_subject)
            topic_cap = topic[0].upper() + topic[1:]
            canon_proc = (canonical.affected_process if canonical else None) or ""
            if impact.process_at_risk and not is_actor_noun(impact.process_at_risk) and impact.process_at_risk != canon_proc:
                _process_topic = topic_word(impact.process_at_risk)
                _finding_words = {w.lower() for w in re.findall(r"[A-Za-z]+", finding_text)}
                if _process_topic not in _finding_words:
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
        elif inv is not None and inv.evidence_to_collect:
            # No candidate hypothesis exists (root cause NOT_ESTABLISHED,
            # correctly), but the investigation plan's own evidence
            # requirements are still real and must not be lost between
            # investigation_plan and risk_impact.evidence_needed -- without
            # this, the report shows a rich, multi-item investigation plan
            # next to an "Evidence Needed" field carrying only a single
            # generic placeholder line, silently dropping most of what was
            # actually identified as needed.
            from app.agent.analytical_validator import normalize_and_dedupe_evidence_items
            aggregated = "; ".join(normalize_and_dedupe_evidence_items(inv.evidence_to_collect))
            if aggregated != impact.evidence_needed:
                trace.append(AgentTraceStep.warn(
                    "Final Evidence Verification: evidence_needed aggregated from the investigation "
                    f"plan's own evidence requirements (was {impact.evidence_needed!r})"
                ))
                impact.evidence_needed = aggregated
        elif impact.evidence_needed and canon_subject and not subject_topic_matches(impact.evidence_needed, canon_subject):
            trace.append(AgentTraceStep.warn(
                f"Final Evidence Verification: evidence_needed {impact.evidence_needed!r} drifted from "
                "the finding's canonical subject — replaced"
            ))
            impact.evidence_needed = f"Approved {topic_word(canon_subject)} completion and authorization record"

        _clean_subj = canon_subject or "the required activity"
        _safe_potential_effect = (
            f"If {_clean_subj} was not completed as required, potential risk of operational "
            "noncompliance or unverified execution."
        )
        # Hedge check: the narrative must contain conditional/evidence-
        # bounded language SOMEWHERE, not necessarily as its first word --
        # a rigid "must start with if/potential/requires" check rejected
        # perfectly well-hedged deterministic narratives that happen to
        # lead with the affected object instead (e.g. "Equipment was
        # reportedly operated outside its validated range. This may
        # require assessment..."), silently discarding the single
        # authoritative producer's own output in favor of a generic
        # template for no semantic reason.
        _HEDGE_MARKERS_RE = re.compile(
            r"\b(if|potential(?:ly)?|require[sd]?|may|could|reportedly|does\s+not\s+establish|assessment)\b",
            re.IGNORECASE,
        )
        if impact.potential_effect and not _HEDGE_MARKERS_RE.search(impact.potential_effect.strip()):
            impact.potential_effect = _safe_potential_effect
        elif impact.potential_effect:
            from app.agent.causal_guard import impact_asserts_unestablished_concept
            _impact_evidence_texts = [e.claim for e in evidence_ledger]
            if impact_asserts_unestablished_concept(impact.potential_effect, finding_text, _impact_evidence_texts):
                trace.append(AgentTraceStep.warn(
                    f"Final Evidence Verification: potential_effect {impact.potential_effect!r} introduced "
                    "an organizational/consequence concept (qualification/authorization/contamination/etc.) "
                    "not established by the finding or evidence — replaced"
                ))
                impact.potential_effect = _safe_potential_effect

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
        repair_five_why_restatement,
        repair_five_why_with_mechanism,
        sync_five_why_status_with_causal_state,
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
        observed = canonical.observed_deviation if canonical else None
        if five_why_skips_available_mechanism(fw.steps, mechanism, observed):
            repaired_steps = repair_five_why_with_mechanism(fw.steps, mechanism, observed)
            if repaired_steps != fw.steps:
                trace.append(AgentTraceStep.warn(
                    "Analytical Validator: 5-Why chain skipped an explicitly available causal "
                    "mechanism — inserted it as an additional step"
                ))
                fw.steps = repaired_steps

    # Cross-node causal-status consistency (INV-CAUSAL-001/003): a 5-Why step
    # whose answer restates the mechanism/leading hypothesis that Root Cause
    # Status just authoritatively resolved as SUPPORTED/ESTABLISHED must
    # never be left showing UNKNOWN for that same proposition -- only ever
    # raises an inconsistent step's status word to match already-verified
    # evidence, never invents new answer text or touches a genuinely deeper,
    # still-unresolved "why" (the correct evidence boundary).
    if fw and fw.steps:
        synced_steps = sync_five_why_status_with_causal_state(fw.steps, mechanism, rc)
        if synced_steps != fw.steps:
            trace.append(AgentTraceStep.warn(
                "Analytical Validator: raised 5-Why step status to match the authoritative "
                "causal resolution — a step restating the established mechanism cannot remain UNKNOWN"
            ))
            fw.steps = synced_steps

    # Generalized restatement guard (Section 2): a Why answer that merely
    # repeats its own question's proposition is never causal reasoning,
    # regardless of which node produced it -- runs after the status sync
    # above so a step correctly promoted to VERIFIED/SUPPORTED is still
    # caught if its TEXT is circular (a wrong status was never the only
    # possible defect here).
    if fw and fw.steps:
        restated_steps = repair_five_why_restatement(fw.steps)
        if restated_steps != fw.steps:
            trace.append(AgentTraceStep.warn(
                "Analytical Validator: 5-Why step answer merely restated its own question — "
                "replaced with the evidence-boundary response"
            ))
            fw.steps = restated_steps

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

    # Section 15: INVESTIGATION_COMPLETENESS Rule
    # If root_cause.status == NOT_ESTABLISHED and material unresolved propositions exist,
    # investigation_plan.questions MUST NOT be empty.
    from app.agent.causal_guard import should_generate_investigation_plan
    if should_generate_investigation_plan(evidence_ledger, getattr(rc, "propositions", None), rc, state["request"].finding_text):
        if inv is None or not inv.questions:
            from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
            _, fallback_plan = build_deterministic_investigation_plan(
                state["request"].finding_text,
                evidence_ledger,
                canonical_subject=getattr(canonical, "finding_subject", None),
                canonical_state=canonical,
            )
            if inv is None:
                inv = fallback_plan
                state["investigation_plan"] = inv
            else:
                inv.questions = fallback_plan.questions
                if not inv.areas:
                    inv.areas = fallback_plan.areas
                if not inv.evidence_to_collect:
                    inv.evidence_to_collect = fallback_plan.evidence_to_collect
            trace.append(AgentTraceStep.ok(
                f"Investigation completeness rule: populated {len(inv.questions)} investigation question(s)"
            ))

    # Language hardening pass across all output fields
    def _clean_lang(text: str | None) -> str | None:
        if not text or not isinstance(text, str):
            return text
        t = re.sub(r"\b(employees|operators|personnel|technicians|auditors|analysts|supervisors)\s+was\b", r"\1 were", text, flags=re.IGNORECASE)
        t = re.sub(r"\bWhy\s+was\s+(the\s+)?(employees|operators|personnel|technicians)\s+complete\b", r"Why did \1\2 not complete", t, flags=re.IGNORECASE)
        t = re.sub(r"\b(employees|operators|personnel|technicians)\s+was\s+reportedly\s+complete\b", r"\1 reportedly did not complete", t, flags=re.IGNORECASE)
        t = re.sub(r"\bthat\s+(the\s+)?(employees|operators|personnel|technicians)\s+was\s+performed\b", r"that the required inspection activity was performed", t, flags=re.IGNORECASE)
        t = re.sub(r"\bwas\s+reportedly\s+duplicate\s+transaction\s+processed\b", "was identified as a duplicate transaction", t, flags=re.IGNORECASE)
        t = re.sub(r"\bwas\s+reportedly\s+failed\b", "reportedly failed", t, flags=re.IGNORECASE)
        t = re.sub(r"\bthe\s+the\b", "the", t, flags=re.IGNORECASE)
        t = re.sub(r"\bof\s*,\b", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\bof\s+was\b", "was", t, flags=re.IGNORECASE)
        t = re.sub(r"\s{2,}", " ", t).strip()
        return t

    # Cost & Financial Impact synchronization and verification (Sections 26-42)
    cost_impact = state.get("cost_impact") or (canonical.cost_impact if canonical else None)
    if not cost_impact:
        from app.services.cost_analysis import analyze_cost_and_financial_impact
        cost_impact = analyze_cost_and_financial_impact(finding_text, evidence_ledger)

    if cost_impact and not cost_impact.cost_factor_detected:
        cost_impact = None

    fin_amt = cost_impact.financial_amount if cost_impact else None
    if canonical:
        canonical.financial_amount = fin_amt
    if impact:
        impact.financial_amount = fin_amt
        if re.search(r"\b(?:duplicate\s+payment|paid\s+twice|double\s+payment)\b", finding_text, re.IGNORECASE):
            amt_txt = f" of {fin_amt.formatted}" if (fin_amt and fin_amt.formatted) else ""
            impact.potential_effect = f"A duplicate payment{amt_txt} was identified. The underlying cause and the extent of any unrecovered financial loss have not yet been established."
            if not getattr(impact, "control_at_risk", None):
                impact.control_at_risk = "Duplicate-Payment Prevention and Reconciliation"
        elif not getattr(impact, "control_at_risk", None) and canonical and getattr(canonical, "control_at_risk", None):
            impact.control_at_risk = canonical.control_at_risk
        impact.affected_object = _clean_lang(impact.affected_object)
        impact.process_at_risk = _clean_lang(impact.process_at_risk)
        impact.control_at_risk = _clean_lang(impact.control_at_risk)
        impact.potential_effect = _clean_lang(impact.potential_effect)
        impact.evidence_needed = _clean_lang(impact.evidence_needed)

    if rc:
        rc.root_cause_basis = _clean_lang(rc.root_cause_basis)
        rc.narrative = _clean_lang(rc.narrative)
        if rc.candidate_hypotheses:
            for h in rc.candidate_hypotheses:
                h.statement = _clean_lang(h.statement)
                h.evidence_needed = _clean_lang(h.evidence_needed)
                h.confirms_if = _clean_lang(h.confirms_if)
                h.refutes_if = _clean_lang(h.refutes_if)
                h.discrimination_evidence = _clean_lang(h.discrimination_evidence)

    if fw and fw.steps:
        for s in fw.steps:
            s.question = _clean_lang(s.question)
            s.answer = _clean_lang(s.answer)

    if inv and inv.questions:
        for q in inv.questions:
            q.question = _clean_lang(q.question)
            q.purpose = _clean_lang(q.purpose)
            q.evidence = _clean_lang(q.evidence)

    if ca:
        ca.immediate_action = _clean_lang(ca.immediate_action)
        if hasattr(ca, "proposed_ca") and ca.proposed_ca:
            ca.proposed_ca = _clean_lang(ca.proposed_ca)
        if hasattr(ca, "impact_analysis") and ca.impact_analysis:
            ca.impact_analysis = _clean_lang(ca.impact_analysis)

    if report:
        report.cost_impact = cost_impact
        report.financial_amount = fin_amt

    # ---------------------------------------------------------------------
    # FINAL OUTPUT VALIDATION LAYER: the formal invariant registry
    # (app.agent.invariants.INVARIANT_REGISTRY) is otherwise only exercised
    # by test harnesses that call evaluate_all_invariants() directly on a
    # hand-built state -- it was never actually invoked by the production
    # graph itself. Every individual invariant already has its own targeted
    # deterministic repair earlier in this function (root-cause certainty
    # downgrade, 5-Why mechanism/status sync, semantic cleanliness, etc.),
    # so this is a closing backstop, not a substitute for those: it re-runs
    # the full registry against the final, fully-hardened state and (a)
    # surfaces anything that still slipped through (never silently), and
    # (b) applies the one blanket-safe repair available at this point --
    # downgrading an inappropriately certain root-cause status. Reducing
    # certainty is always safe; inventing a fix for an arbitrary invariant
    # class here would not be.
    # -----------------------------------------------------------------
    # Pre-registry deterministic REPAIR for the three invariants that the
    # blind audit found were detected-but-shipped (INV-SEC-003, INV-EPIS-001,
    # INV-SEM-002). Per the audit brief, "any violation must be repaired
    # before report generation", so these are actively corrected here rather
    # than only warned about. All three repairs move the output in the
    # conservative direction (strip untrusted content, downgrade
    # over-claimed status, replace a false-confident entity with an honest
    # unresolved marker) -- none of them can loosen a causal conclusion.
    # -----------------------------------------------------------------
    _canonical = state.get("canonical_finding_state")
    _ledger_in = state.get("evidence_ledger") or []
    try:
        from app.models.agent import EvidenceStatus as _ES
        from app.services.epistemic_modality import (
            classify_epistemic_stance as _stance_of,
            classify_modality as _modality_of,
        )
        from app.services.instruction_detector import classify_instruction as _cls_instr

        _repaired_ledger = []
        for _item in _ledger_in:
            _txt = getattr(_item, "claim", None)
            if _txt and _cls_instr(_txt).is_untrusted:
                trace.append(AgentTraceStep.warn(
                    f"[INV-SEC-003 repaired] instruction-like content stripped from "
                    f"evidence ledger: {_txt!r}"
                ))
                continue
            if _txt and getattr(_item, "status", None) == _ES.VERIFIED:
                if _stance_of(_txt) is not None:
                    _item.status = _ES.BELIEF
                    trace.append(AgentTraceStep.warn(
                        f"[INV-EPIS-001 repaired] belief/opinion claim downgraded "
                        f"VERIFIED -> BELIEF: {_txt!r}"
                    ))
                elif not _modality_of(_txt).is_actual:
                    _item.status = _ES.UNVERIFIED
                    _item.modality = _modality_of(_txt).modality
                    trace.append(AgentTraceStep.warn(
                        f"[INV-EPIS-001 repaired] {_item.modality.lower()} claim downgraded "
                        f"VERIFIED -> UNVERIFIED: {_txt!r}"
                    ))
            _repaired_ledger.append(_item)
        _ledger_in = _repaired_ledger

        if _canonical is not None:
            _kept_claims = []
            for _c in (getattr(_canonical, "evidence_claims", None) or []):
                _ct = getattr(_c, "text", None)
                if _ct and _cls_instr(_ct).is_untrusted:
                    trace.append(AgentTraceStep.warn(
                        f"[INV-SEC-003 repaired] instruction-like content stripped from "
                        f"evidence_claims: {_ct!r}"
                    ))
                    continue
                if _ct and getattr(_c, "status", None) == _ES.VERIFIED:
                    if _stance_of(_ct) is not None:
                        _c.status = _ES.BELIEF
                    elif not _modality_of(_ct).is_actual:
                        _c.status = _ES.UNVERIFIED
                        _c.modality = _modality_of(_ct).modality
                _kept_claims.append(_c)
            _canonical.evidence_claims = _kept_claims

            # INV-SEM-002 repair: a generic placeholder entity is replaced by
            # an explicit unresolved marker + note. Never silently shipped.
            from app.agent.invariants import is_generic_placeholder_entity
            from app.services.semantic_subject import best_partial_noun_phrase
            for _fname in ("affected_object", "finding_subject"):
                _val = getattr(_canonical, _fname, None)
                if _val and is_generic_placeholder_entity(_val):
                    _frag = best_partial_noun_phrase(getattr(_canonical, "raw_finding", "") or "")
                    _new = _frag or "UNRESOLVED — the specific entity involved could not be isolated from the finding text"
                    setattr(_canonical, _fname, _new)
                    _canonical.entity_resolution = "PARTIAL" if _frag else "UNRESOLVED"
                    _canonical.entity_resolution_note = (
                        "Entity extraction was uncertain — this value was a generic "
                        "placeholder and has been replaced with a flagged best-effort value."
                    )
                    _canonical.subject_unresolved = True
            # Semantic Drift Gate: Ensure final report's affected object / process / root cause
            # align directly with the upstream canonical semantic authority.
            if _canonical is not None:
                if impact is not None:
                    # Enforce that impact affected_object matches canonical subject/affected_object
                    canon_obj = getattr(_canonical, "affected_object", None) or getattr(_canonical, "finding_subject", None)
                    if canon_obj and canon_obj != "UNKNOWN":
                        impact.affected_object = canon_obj
                    canon_proc = getattr(_canonical, "affected_process", None)
                    if canon_proc and canon_proc != "UNKNOWN" and getattr(impact, "process_at_risk", None) in (None, "", "UNKNOWN", "Operational process"):
                        impact.process_at_risk = canon_proc

                # Enforce that report carries the semantic graph and canonical traceability matrix
                if report is not None:
                    if getattr(_canonical, "semantic_graph", None) and not getattr(report, "semantic_graph", None):
                        report.semantic_graph = _canonical.semantic_graph

            if impact is not None and getattr(impact, "affected_object", None):
                if is_generic_placeholder_entity(impact.affected_object):
                    impact.affected_object = getattr(_canonical, "affected_object", None) or impact.affected_object
                    trace.append(AgentTraceStep.warn(
                        "[INV-SEM-002 repaired] generic placeholder impact.affected_object "
                        "replaced with the canonical entity value"
                    ))
    except Exception as _repair_err:  # never let a repair break the pipeline
        errors.append(f"Evidence-semantics repair skipped: {_repair_err}")

    final_check_state = {
        **state,
        "evidence_ledger": _ledger_in,
        "canonical_finding_state": _canonical,
        "root_cause": rc,
        "five_why": fw,
        "investigation_plan": inv,
        "capa_analysis": capa if capa else ca,
        "impact_assessment": impact,
        "cost_impact": cost_impact,
        "report": report,
    }
    from app.agent.invariants import evaluate_all_invariants
    _final_valid, _final_violations = evaluate_all_invariants(final_check_state)
    if not _final_valid:
        for _v in _final_violations:
            trace.append(AgentTraceStep.warn(f"Final Output Validation: {_v}"))
        if rc and rc.status in (RootCauseStatus.SUPPORTED, RootCauseStatus.ESTABLISHED) and any(
            "[INV-CAUS" in v or "[INV-WHY" in v for v in _final_violations
        ):
            trace.append(AgentTraceStep.warn(
                f"Final Output Validation: root_cause.status={rc.status} downgraded to NOT_ESTABLISHED "
                "— unresolved causal-consistency violation(s) at final validation"
            ))
            rc.status = RootCauseStatus.NOT_ESTABLISHED
            rc.leading_hypothesis = None
            if report:
                report.root_cause = rc

    trace.append(AgentTraceStep.ok("Final evidence verification and consistency validation completed"))

    return {
        **state,
        # Carry the repaired ledger/canonical state forward -- otherwise the
        # repairs above would be validated and then thrown away.
        "evidence_ledger": _ledger_in,
        "canonical_finding_state": _canonical,
        "root_cause": rc,
        "five_why": fw,
        "investigation_plan": inv,
        "ca_draft": ca,
        "capa_analysis": capa if capa else ca,
        "impact_assessment": impact,
        "cost_impact": cost_impact,
        "financial_amount": fin_amt,
        "contributing_factors": cfs,
        "report": report,
        "trace": trace,
        "errors": errors,
    }
