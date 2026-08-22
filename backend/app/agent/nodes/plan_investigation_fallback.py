"""Deterministic Fallback Investigation Planner.

Fired when investigation planning fails or produces zero questions/hypotheses.
Generates dynamic, case-grounded hypotheses and discriminating investigation questions
directly from canonical finding semantics, claim conflicts, and mechanism polarity.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

from app.models.agent import (
    CandidateHypothesis,
    ConditionalCapaAction,
    EvidenceItem,
    EvidenceStatus,
    InvestigationPlan,
    InvestigationQuestion,
)
from app.services.semantic_subject import (
    _clean_subject,
    extract_conflict_topic,
    extract_temporal_clause,
    resolve_deviation,
    split_topic_and_tail,
    strip_leading_article,
    topic_word,
)


_DEGRADED_SUBJECTS = {"process compliance", None, ""}

# Structural (verb-object, not keyword-list) extraction of what a
# transmission-shaped sentence ("System shows email delivered", "Access log
# shows document opened") is actually about -- used only as a last resort
# when neither the primary structural resolver nor the entity extractor
# recognized a DELIVERY_VS_RECEIPT finding's subject (e.g. "email"/
# "document" aren't in the generic keyword-priority fallback's list). Never
# used to override a subject the primary resolver already found.
# Tier 1: reporting/recording verbs ("shows", "indicates", "records",
# "confirms") -- checked first since these essentially never double as a
# leading noun-modifier the way "access"/"open"/"send" can ("Access log
# shows..." -- "Access" collides with the transmission verb "access" when
# scanned bag-of-words, capturing "log" instead of the actual object
# "document" later in the sentence).
_REPORTING_OBJECT_RE = re.compile(
    r"\b(?:shows?|indicat\w*|records?|confirms?)\s+(?:that\s+)?(?:the\s+|an?\s+)?([a-z][\w-]*)\b",
    re.IGNORECASE,
)
# Tier 2: broader transmission-verb fallback, only used when tier 1 finds
# nothing.
_TRANSMISSION_OBJECT_RE = re.compile(
    r"\b(?:deliver\w*|sen[dt]\w*|receiv\w*|access\w*|open\w*|retriev\w*|transmit\w*)\s+"
    r"(?:that\s+)?(?:the\s+|an?\s+)?([a-z][\w-]*)\b",
    re.IGNORECASE,
)
_TRANSMISSION_OBJECT_STOPWORDS = {
    "that", "the", "a", "an", "it", "this", "successfully", "never", "not",
    "was", "were", "is", "are", "has", "have", "been",
}


def _extract_transmission_object(text: str) -> str | None:
    for pattern in (_REPORTING_OBJECT_RE, _TRANSMISSION_OBJECT_RE):
        for match in pattern.finditer(text or ""):
            candidate = match.group(1).lower()
            if candidate not in _TRANSMISSION_OBJECT_STOPWORDS and len(candidate) > 2:
                return candidate
    return None


def build_deterministic_investigation_plan(
    finding_text: str,
    evidence_ledger: list[EvidenceItem],
    canonical_subject: str | None = None,
    canonical_state: Any = None,
) -> tuple[list[CandidateHypothesis], InvestigationPlan]:
    """Build dynamic, case-grounded hypotheses and discriminating investigation questions.

    `canonical_state` provides authoritative upstream epistemic state
    (primary_uncertainty, causal_readiness, semantic_type, recurrence, financial)
    without needing re-derivation. `canonical_subject` is used for subject continuity.
    Defaults to None for full backward compatibility with legacy callers.
    """
    hypotheses: list[CandidateHypothesis] = []
    questions: list[InvestigationQuestion] = []
    evidence_items: list[str] = []

    # Entity extraction (IDs, SOPs, BMRs, Lots, Equipment Tags, System Codes, Rooms, Lines)
    extracted_entities = re.findall(
        r"\b([A-Z]{2,5}-[A-Z0-9-]+|Lot\s+[A-Z0-9-]+|Batch\s+[A-Z0-9-]+|Line\s+\d+|Room\s+\d+|Cleanroom\s+[A-Za-z0-9\s]+|Autoclave\s+#?\d+|AHU-\d+|CR-\d+|LF-\d+|VI-\d+|CP-\d+|PP-\d+|Lyo-\d+|FH-\d+|SP-\d+|BAL-\d+|W-\d+|NC-\d+-\d+|CAPA-\d+-\d+|BRD-\d+|MBR-[A-Z0-9-]+|WSC-\d+|API-[0-9]+|RM-[0-9]+|QC-REF-\d+|EQ-\d+)\b",
        finding_text,
        re.IGNORECASE,
    )
    extracted_id = extracted_entities[0] if extracted_entities else ""

    fact_claims = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.VERIFIED]
    reported_claims = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.REPORTED]
    resolved = resolve_deviation(finding_text, fact_claims)
    if canonical_subject and canonical_subject not in _DEGRADED_SUBJECTS:
        subject = canonical_subject
    elif canonical_state and getattr(canonical_state, "finding_subject", None) and canonical_state.finding_subject not in _DEGRADED_SUBJECTS:
        subject = canonical_state.finding_subject
    else:
        subject = resolved.finding_subject or resolved.subject or "the affected process"
    actor = resolved.actor or (getattr(canonical_state, "actor", None) if canonical_state else None)

    # Areas defaults to the generic phrasing used by every non-conflict
    # branch below; the conflict branch overrides it with three specific,
    # investigation-oriented areas (Section 6) instead of one generic label.
    plan_areas = [f"Verify compliance and control records for {subject}"]

    # Defect 3 — entity fidelity surfaced as an investigation question.
    # When the deterministic resolver could not isolate the concrete entity,
    # the correct output is a question asking WHICH entity is involved, not a
    # confident generic placeholder. This is the mechanism that replaces the
    # old silent "process compliance" substitution.
    if canonical_state is not None and getattr(canonical_state, "subject_unresolved", False):
        questions.append(InvestigationQuestion(
            question_id="Q_ENTITY_IDENTIFICATION", id="Q_ENTITY_IDENTIFICATION",
            question=(
                "What is the specific entity (record, equipment item, document, batch, "
                "account, or system) that this finding concerns?"
            ),
            purpose="Resolve the affected object, which could not be isolated from the finding text",
            objective="Resolve the affected object, which could not be isolated from the finding text",
            evidence="The originating audit working paper, sample list, or observation sheet naming the item examined",
            evidence_required="The originating audit working paper, sample list, or observation sheet naming the item examined",
            priority="P1",
            target_type="OTHER",
            target_proposition_id="P_ENTITY_IDENTITY",
            status="ACTIVE",
            category="EVIDENCE_VERIFICATION",
            decision_rule=(
                "Until the specific entity is identified, no impact, root cause, or CAPA "
                "conclusion can be scoped to it."
            ),
        ))

    for ent in extracted_entities:
        ent_clean = ent.strip()
        if re.match(r"^(?:SOP|WI|POL|DOC|PRO)-", ent_clean, re.IGNORECASE):
            evidence_items.extend([f"{ent_clean} document distribution record", f"{ent_clean} revision history", f"{ent_clean} acknowledgment log"])
        elif re.match(r"^(?:Batch|Lot|MBR|BRD)-", ent_clean, re.IGNORECASE) or "batch" in ent_clean.lower() or "lot" in ent_clean.lower():
            evidence_items.extend([f"{ent_clean} processing log", f"{ent_clean} batch manufacturing record", f"{ent_clean} quality release record"])
        elif re.match(r"^(?:EQ|BAL|AUTOCLAVE|AHU|CR|LF|VI|CP|PP|LYO|FH|SP)-", ent_clean, re.IGNORECASE):
            evidence_items.extend([f"{ent_clean} calibration certificate", f"{ent_clean} operational logbook", f"{ent_clean} maintenance history"])
        else:
            evidence_items.extend([f"{ent_clean} operational record", f"{ent_clean} verification record", f"{ent_clean} audit trail"])

    from app.agent.claim_extractor import detect_evidence_conflicts, extract_claims
    from app.agent.causal_guard import extract_immediate_mechanism

    claims = extract_claims(finding_text, evidence_ledger)
    conflicts = detect_evidence_conflicts(claims)
    mechanism = extract_immediate_mechanism(reported_claims, fact_claims)

    text_low = finding_text.lower()

    # 1. Multiple Competing Reported Explanations Branch (e.g. training vs workload vs discipline)
    reported_claims_list = [c for c in claims if getattr(c, "status", None) == EvidenceStatus.REPORTED]
    if not conflicts and len(reported_claims_list) >= 2:
        plan_areas = [
            f"Training matrix, authorization, and LMS records for {subject}",
            f"Staffing, shift allocation, and workload distribution for {subject}",
            f"Objective execution history and supervisory logs for {subject}",
        ]
        h_idx = 1
        for rc in reported_claims_list:
            rc_text = rc.text.lower()
            rc_pred = (rc.predicate or "").lower()
            h_id = f"H{h_idx}"

            if "training" in rc_text or "training" in rc_pred or "unaware" in rc_text or "trained" in rc_text:
                h_name = "INSUFFICIENT_TRAINING_CONTRIBUTING_FACTOR"
                h_stmt = f"Insufficient training on {subject} contributed to the nonconformity."
                ev_need = f"Authenticated training records, LMS records, training matrix, relevant training requirement, effective date, and employee authorization records"
                conf_if = f"Authenticated records show required training for {subject} was not completed before the relevant date"
                ref_if = f"Authenticated records show all affected employees completed required training for {subject} before the relevant date"
                disc_crit = f"H{h_idx} strengthens if authenticated records show required training for {subject} was not completed; H{h_idx} weakens if authenticated records show all affected employees completed required training."

                questions.append(InvestigationQuestion(
                    question_id=f"Q{h_idx}_TRAINING_VERIFICATION",
                    id=f"Q{h_idx}_TRAINING_VERIFICATION",
                    question=f"Did each affected employee complete training required for {subject} before the relevant period?",
                    purpose=f"Verify whether required training for {subject} was completed",
                    objective=f"Verify whether required training for {subject} was completed",
                    evidence="Authenticated training records, LMS records, training matrix",
                    evidence_required="Authenticated training records, LMS records, training matrix",
                    target_type="HYPOTHESIS",
                    target_proposition_id=f"P_{h_id}",
                    priority="P1",
                    hypothesis_tested=h_id,
                    confirms_if=conf_if,
                    refutes_if=ref_if,
                ))
                evidence_items.extend(["Authenticated training records", "LMS completion logs", "Training matrix"])

            elif "workload" in rc_text or "workload" in rc_pred or "pressure" in rc_text or "staffing" in rc_text:
                h_name = "WORKLOAD_PRESSURE_CONTRIBUTING_FACTOR"
                h_stmt = f"Workload pressure or staffing constraints contributed to non-completion of {subject}."
                ev_need = f"Staffing rosters, shift logs, task assignment records, overtime records, production volume logs for the affected period"
                conf_if = f"Shift and staffing records confirm abnormal workload, concurrent conflicting assignments, or understaffing during the affected period"
                ref_if = f"Staffing logs confirm standard staffing levels and nominal workload during the affected period"
                disc_crit = f"H{h_idx} strengthens if records show abnormal workload or understaffing; H{h_idx} weakens if staffing and task volume were nominal."

                questions.append(InvestigationQuestion(
                    question_id=f"Q{h_idx}_WORKLOAD_RECORDS",
                    id=f"Q{h_idx}_WORKLOAD_RECORDS",
                    question="What staffing, workload, shift, and task-allocation records exist for the affected period?",
                    purpose=f"Assess whether workload, resource, or shift constraints contributed to non-completion of {subject}",
                    objective=f"Assess whether workload, resource, or shift constraints contributed to non-completion of {subject}",
                    evidence="Shift roster, workload records, task assignment records, overtime records",
                    evidence_required="Shift roster, workload records, task assignment records, overtime records",
                    target_type="HYPOTHESIS",
                    target_proposition_id=f"P_{h_id}",
                    priority="P2",
                    hypothesis_tested=h_id,
                    confirms_if=conf_if,
                    refutes_if=ref_if,
                ))
                evidence_items.extend(["Shift rosters", "Task assignment records", "Overtime records"])

            elif "discipline" in rc_text or "discipline" in rc_pred or "performance" in rc_text or "negligence" in rc_text:
                h_name = "PERFORMANCE_OR_DISCIPLINE_FACTOR"
                h_stmt = f"Performance or discipline issue contributed to the nonconformity in {subject}."
                ev_need = f"Prior execution records, documented supervisory records, objective performance logs independent of the current finding"
                conf_if = f"Documented history confirms repeated non-compliance by the same personnel under normal workload and confirmed training"
                ref_if = f"Objective records show consistent compliance history and no prior performance deviations"
                disc_crit = f"H{h_idx} strengthens if prior independent records establish repeated non-performance; H{h_idx} weakens if prior compliance history is consistent."

                questions.append(InvestigationQuestion(
                    question_id=f"Q{h_idx}_PERFORMANCE_HISTORY",
                    id=f"Q{h_idx}_PERFORMANCE_HISTORY",
                    question=f"Is there objective documented evidence of repeated failure to perform {subject} independent of the current finding?",
                    purpose="Determine whether the non-completion represents an isolated occurrence or repeated performance deviation",
                    objective="Determine whether the non-completion represents an isolated occurrence or repeated performance deviation",
                    evidence="Prior execution records, supervisory records, documented performance records",
                    evidence_required="Prior execution records, supervisory records, documented performance records",
                    target_type="HYPOTHESIS",
                    target_proposition_id=f"P_{h_id}",
                    priority="P3",
                    hypothesis_tested=h_id,
                    confirms_if=conf_if,
                    refutes_if=ref_if,
                ))
                evidence_items.extend(["Prior execution records", "Supervisory review logs"])

            else:
                pred_clean = (rc.predicate or rc.text).strip().rstrip(".")
                h_name = f"REPORTED_FACTOR_{h_idx}"
                h_stmt = f"Reported factor ({pred_clean}) contributed to non-completion of {subject}, but remains unverified."
                ev_need = f"Objective documentation and records capable of confirming or refuting whether {pred_clean} occurred"
                conf_if = f"Objective records confirm {pred_clean} during the affected period"
                ref_if = f"Objective records contradict {pred_clean}"
                disc_crit = f"H{h_idx} strengthens if objective records corroborate {pred_clean}; weakens if contradicted."

                questions.append(InvestigationQuestion(
                    question_id=f"Q{h_idx}_REPORTED_FACTOR",
                    id=f"Q{h_idx}_REPORTED_FACTOR",
                    question=f"Do objective records confirm or refute whether {pred_clean} affected performance of {subject}?",
                    purpose=f"Independently verify reported statement from {rc.speaker or 'personnel'}",
                    objective=f"Independently verify reported statement from {rc.speaker or 'personnel'}",
                    evidence=f"Objective operational records relevant to {pred_clean}",
                    evidence_required=f"Objective operational records relevant to {pred_clean}",
                    target_type="HYPOTHESIS",
                    target_proposition_id=f"P_{h_id}",
                    priority="P3",
                    hypothesis_tested=h_id,
                    confirms_if=conf_if,
                    refutes_if=ref_if,
                ))
                evidence_items.append(f"Objective records relevant to {pred_clean}")

            hypotheses.append(CandidateHypothesis(
                id=h_id,
                name=h_name,
                statement=h_stmt,
                status="POSSIBLE",
                evidence_needed=ev_need,
                confirms_if=conf_if,
                refutes_if=ref_if,
                discrimination_evidence=disc_crit,
                relevance_rank="HIGH",
                supporting_claim_ids=[rc.claim_id],
                evidence_strength="REPORTED",
            ))
            h_idx += 1

        return hypotheses, InvestigationPlan(
            areas=plan_areas,
            questions=questions,
            evidence_to_collect=list(dict.fromkeys(evidence_items)),
        )

    # 1b. Conflicting Evidence Branch — hypotheses, discrimination criteria and
    # investigation questions are all derived from the finding's own topic
    # word, subject phrase, temporal clause, and the actual reported claim
    # text that produced the conflict, so the wording is finding-specific
    # rather than a fixed generic template (works for training, calibration,
    # checklist, maintenance, documentation, inspection, communication, etc.)
    if conflicts:
        # Resolve the SPECIFIC conflict's own claims first -- a finding's
        # overall subject (e.g. "temperature monitoring records") can be
        # entirely different from what an embedded conflict is actually
        # about (e.g. whether retraining occurred). Prefer a
        # CONFLICTING_REPORTS-type conflict (two people disagreeing) over a
        # CONTRADICTED_BY_EVIDENCE one when both exist.
        primary_conflict = next((c for c in conflicts if getattr(c, "conflict_type", None) == "CONFLICTING_REPORTS"), conflicts[0])
        conflict_claim_ids = list(getattr(primary_conflict, "claims", []))
        conflict_claims_by_id = {c.claim_id: getattr(c, "text", str(c)) for c in claims}
        reported_texts = [
            getattr(c, "text", str(c)) for c in claims
            if getattr(c, "status", None) == EvidenceStatus.REPORTED
        ]
        claim_a = (
            conflict_claims_by_id.get(conflict_claim_ids[0]) if len(conflict_claim_ids) > 0 and conflict_claim_ids[0] in conflict_claims_by_id
            else (reported_texts[0] if reported_texts else f"one party reported that {subject} was not completed")
        )
        claim_b = (
            conflict_claims_by_id.get(conflict_claim_ids[1]) if len(conflict_claim_ids) > 1 and conflict_claim_ids[1] in conflict_claims_by_id
            else (reported_texts[1] if len(reported_texts) > 1 else f"another party reported that {subject} was completed")
        )

        # DELIVERY_VS_RECEIPT (Conflict-Center hardening): a conflict between
        # a system/record asserting something was delivered/sent/accessed and
        # a person reporting they never received/accessed it is NOT a
        # completion dispute -- collapsing it into a "{TOPIC}_NOT_COMPLETED"
        # hypothesis (the branch below, built for training/maintenance-style
        # "was the activity done" conflicts) fabricates a specific failure
        # mode (delivery/notification/acknowledgement failure) the evidence
        # never actually establishes. DELIVERY, RECEIPT, ACCESS, and
        # ACKNOWLEDGEMENT are distinct propositions (Section 2) -- this
        # branch preserves both sides, generates evidence-discriminating
        # questions that investigate the conflict without presupposing
        # either side, and returns ZERO hypotheses so root cause stays
        # NOT_ESTABLISHED / leading hypothesis stays NONE until independent
        # evidence resolves the conflict.
        if getattr(primary_conflict, "proposition_type", None) == "DELIVERY_VS_RECEIPT":
            # `subject` (the finding's OVERALL resolved subject) can itself
            # be the degraded "process compliance" placeholder when neither
            # the structural condition patterns nor the entity/keyword
            # fallback recognized the finding's vocabulary (e.g. "email",
            # "document" aren't in the keyword list) -- fall back to the
            # CONFLICT's own topic (derived from the conflicting claims'
            # actual wording, same mechanism the branch below already
            # relies on) rather than surfacing "process compliance" in
            # every question.
            dr_subject = subject
            if dr_subject in {"process compliance", None, ""}:
                dr_subject = _extract_transmission_object(finding_text) or extract_conflict_topic(claim_a, claim_b, subject)
            subject = dr_subject
            temporal = extract_temporal_clause(finding_text)
            temporal_suffix = f" {temporal}" if temporal else ""
            subject_cap = subject[0].upper() + subject[1:] if subject else "the affected communication"
            dr_questions = [
                InvestigationQuestion(
                    question_id="Q1_DELIVERY_CONFIRMED",
                    id="Q1_DELIVERY_CONFIRMED",
                    question=(
                        f"Do authenticated system records establish successful delivery of {subject} "
                        "to each affected recipient?"
                    ),
                    purpose="Verify delivery-side completion details",
                    objective="Verify delivery-side completion details",
                    evidence=f"Per-recipient delivery logs for {subject}, including recipient, destination, timestamp, and status",
                    evidence_required=f"Per-recipient delivery logs for {subject}, including recipient, destination, timestamp, and status",
                    target_type="PROPOSITION",
                    target_proposition_id="P_DELIVERY",
                    priority="P1",
                    status="ACTIVE",
                    next_question_if_true="Q2_RECEIPT_VERIFIED",
                    next_question_if_false="Q5_FAILURE_MECHANISM",
                    possible_outcomes=[
                        "Authenticated logs establish successful delivery to all recipients → proceed to verify receipt/access.",
                        "Logs show delivery failure or missing transmission records → investigate delivery failure mechanism.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q2_RECEIPT_VERIFIED",
                    id="Q2_RECEIPT_VERIFIED",
                    question=(
                        f"Do independent records establish actual receipt or access by the affected recipients{temporal_suffix}?"
                    ),
                    purpose="Establish whether delivery resulted in actual recipient receipt or access",
                    objective="Establish whether delivery resulted in actual recipient receipt or access",
                    evidence=f"Independent receipt, access, or read records for {subject}",
                    evidence_required=f"Independent receipt, access, or read records for {subject}",
                    target_type="PROPOSITION",
                    target_proposition_id="P_RECEIPT",
                    priority="P2",
                    depends_on="Q1_DELIVERY_CONFIRMED",
                    activation_condition="If delivery is confirmed",
                    status="CONDITIONAL",
                    next_question_if_true="Q3_ACKNOWLEDGEMENT_REQUIRED",
                    next_question_if_false="Q5_FAILURE_MECHANISM",
                    possible_outcomes=[
                        "Access or read records confirm actual receipt → proceed to verify acknowledgement requirement.",
                        "No receipt or access confirmed → investigate mechanism preventing receipt.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q3_ACKNOWLEDGEMENT_REQUIRED",
                    id="Q3_ACKNOWLEDGEMENT_REQUIRED",
                    question=f"Did the applicable procedure require acknowledgement before performing work under {subject}?",
                    purpose="Establish whether formal acknowledgement was required before work",
                    objective="Establish whether formal acknowledgement was required before work",
                    evidence=f"Applicable SOP/procedure and acknowledgement requirements for {subject}",
                    evidence_required=f"Applicable SOP/procedure and acknowledgement requirements for {subject}",
                    target_type="PROPOSITION",
                    target_proposition_id="P_REQ_ACK",
                    priority="P3",
                    depends_on="Q2_RECEIPT_VERIFIED",
                    activation_condition="If receipt/access is confirmed",
                    status="CONDITIONAL",
                    next_question_if_true="Q4_ACKNOWLEDGEMENT_COMPLETED",
                    next_question_if_false=None,
                    possible_outcomes=[
                        "Procedure required prior acknowledgement → proceed to verify completed acknowledgement records.",
                        "Acknowledgement was not mandatory before work → acknowledgement investigation not required.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q4_ACKNOWLEDGEMENT_COMPLETED",
                    id="Q4_ACKNOWLEDGEMENT_COMPLETED",
                    question=f"Do authenticated records establish that affected personnel acknowledged {subject} before performing the relevant activity?",
                    purpose="Verify timely completion of required acknowledgement",
                    objective="Verify timely completion of required acknowledgement",
                    evidence=f"Signed or electronic acknowledgement and activity records for {subject}",
                    evidence_required=f"Signed or electronic acknowledgement and activity records for {subject}",
                    target_type="PROPOSITION",
                    target_proposition_id="P_ACK_EXECUTION",
                    priority="P4",
                    depends_on="Q3_ACKNOWLEDGEMENT_REQUIRED",
                    activation_condition="If acknowledgement was required",
                    status="CONDITIONAL",
                    possible_outcomes=[
                        "Authenticated records confirm timely acknowledgement before work.",
                        "No acknowledgement recorded prior to activity execution.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q5_FAILURE_MECHANISM",
                    id="Q5_FAILURE_MECHANISM",
                    question=(
                        "What objective evidence establishes the mechanism that prevented or interrupted delivery or receipt"
                        f"{temporal_suffix}?"
                    ),
                    purpose="Identify an established mechanism if receipt did not occur",
                    objective="Identify an established mechanism if receipt did not occur",
                    evidence="Channel diagnostics, recipient account configuration, transmission errors, and related technical records",
                    evidence_required="Channel diagnostics, recipient account configuration, transmission errors, and related technical records",
                    target_type="PROPOSITION",
                    target_proposition_id="P_MECHANISM",
                    priority="P5",
                    depends_on="Q2_RECEIPT_VERIFIED",
                    activation_condition="If non-receipt is confirmed",
                    status="CONDITIONAL",
                    possible_outcomes=[
                        "Diagnostic logs establish technical transmission or filtering failure.",
                        "Account configuration or server logs explain non-receipt.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q6_ACTION_PRIOR_TO_CONFIRMATION",
                    id="Q6_ACTION_PRIOR_TO_CONFIRMATION",
                    question=(
                        f"Did affected personnel perform work governed by {subject} before receipt or "
                        "acknowledgement was confirmed?"
                    ),
                    purpose="Determine whether operational activity occurred prior to confirmed receipt",
                    objective="Determine whether operational activity occurred prior to confirmed receipt",
                    evidence=f"Operational activity records, batch records, or system execution logs relative to {subject}",
                    evidence_required=f"Operational activity records, batch records, or system execution logs relative to {subject}",
                    target_type="PROPOSITION",
                    target_proposition_id="P_ACTION",
                    priority="P4",
                    depends_on="Q2_RECEIPT_VERIFIED",
                    activation_condition="If operational activity could have occurred before confirmation",
                    status="CONDITIONAL",
                    possible_outcomes=[
                        "Activity performed prior to confirmed receipt/acknowledgement → assess scope and compliance impact.",
                        "No operational activity occurred prior to confirmation.",
                    ],
                ),
            ]
            dr_evidence = [
                f"Per-recipient delivery logs for {subject}",
                f"Independent receipt, access, or read records for {subject}",
                f"Applicable SOP/procedure and acknowledgement requirements for {subject}",
                f"Relevant operator activity records relative to {subject}",
                "Transmission diagnostics, account configuration, and error logs",
            ]
            dr_areas = [
                f"{subject_cap} control and distribution",
                f"Operational activity relative to {subject}",
                f"Technical and administrative transmission verification",
            ]
            return [], InvestigationPlan(
                questions=dr_questions,
                areas=dr_areas,
                evidence_to_collect=dr_evidence,
            )

        # Topic is derived from the CONFLICT's own claims, never from the
        # finding's overall subject -- see extract_conflict_topic's
        # docstring for why (a finding's deviation and an embedded
        # conflict's proposition can be about entirely different things).
        topic = extract_conflict_topic(claim_a, claim_b, subject)
        topic_cap = topic[0].upper() + topic[1:]
        # tail = a short "for <X>" qualifier. Try the finding subject first
        # (works when subject IS literally "<topic> for X", e.g. "training
        # for the revised procedure"), then either conflict claim, before
        # falling back to a neutral phrase -- never force the whole
        # (possibly unrelated) finding subject into this position.
        tail = (
            split_topic_and_tail(subject, topic)
            or split_topic_and_tail(claim_a, topic)
            or split_topic_and_tail(claim_b, topic)
            or "the applicable requirement"
        )
        temporal = extract_temporal_clause(finding_text)
        temporal_suffix = f" {temporal}" if temporal else ""
        effective_ref = temporal if temporal else "the applicable effective date"
        stripped_actor = strip_leading_article(actor)
        actor_phrase = f"the {stripped_actor.lower()}" if stripped_actor else "the responsible person"

        # CAUSAL LEVEL SEPARATION: only a genuine CAUSAL/EXECUTION-level
        # proposition (did the required activity actually happen?) is a
        # root-cause CANDIDATE HYPOTHESIS. Record-availability ("the record
        # wasn't there to check") and record-CONTROL-process propositions
        # ("the creation/retention/retrieval/verification process may be
        # weak") are a different causal LEVEL entirely -- an EVIDENCE-STATE
        # fact and a SYSTEMIC investigation area, respectively -- and must
        # never be pooled into `hypotheses` as competing root causes
        # alongside the execution-level one. They still matter, so their
        # content survives as investigation areas/questions; they just
        # never compete for "leading hypothesis" or generate their own
        # unconditional CAPA branch, since nothing here establishes WHICH
        # (if any) of them is the actual failure mode.
        h1 = CandidateHypothesis(
            id="H1",
            name=f"{topic.upper()}_NOT_COMPLETED",
            statement=f"Required {topic} for {tail} may not have been completed{temporal_suffix}.",
            status="POSSIBLE",
            evidence_needed=f"Authenticated {topic} attendance/completion record",
            confirms_if=f"No approved {topic} completion record exists for the affected period",
            refutes_if=f"An approved {topic} completion record confirms timely completion",
            discrimination_evidence=(
                f"A located record showing completion {effective_ref} weakens H1. Absence of a record alone does "
                "not prove H1 -- see the record-availability and record-control investigation areas below."
            ),
            rationale=f"Plausible because {claim_a}.",
            relevance_rank="HIGH",
            supporting_evidence=[claim_a],
            contradicting_evidence=[claim_b] if claim_b != claim_a else [],
        )
        hypotheses.append(h1)

        # P1 — LOCATE / ESTABLISH SOURCE (antecedent evidence-availability)
        questions.append(InvestigationQuestion(
            question_id="Q1_LOCATE_SOURCE_RECORD",
            id="Q1_LOCATE_SOURCE_RECORD",
            question=f"Can an authenticated {topic} record be located in the approved {topic} repository, system, or archive?",
            purpose=(
                "Resolves the antecedent evidence-availability question: whether the authoritative record exists and "
                "can be located, prior to evaluating what it shows"
            ),
            objective=f"Locate authenticated {topic} source records",
            evidence=f"Authenticated {topic} record from the approved repository, system, or archive",
            evidence_required=f"Authenticated {topic} record from the approved repository, system, or archive",
            target_type="PROPOSITION",
            target_proposition_id="P_RECORD_AVAILABILITY",
            priority="P1",
            status="ACTIVE",
            next_question_if_true="Q2_INTERPRET_SOURCE_RECORD",
            next_question_if_false="Q3_RECORD_CONTROL_REQUIREMENT",
            confirms_if=f"An authenticated {topic} record is located in the approved repository or archive",
            refutes_if=(
                f"No {topic} record can be located after a documented search of every applicable repository "
                "— this alone does not prove non-performance; evaluate record-control requirements"
            ),
            possible_outcomes=[
                "Record is located in repository → proceed to evaluate what the record establishes (Q2).",
                "Record cannot be located → evaluate whether the record was required and if record-control operated as required (Q3).",
            ],
        ))

        # P2 — INTERPRET SOURCE (evaluates H1 if record is located)
        questions.append(InvestigationQuestion(
            question_id="Q2_INTERPRET_SOURCE_RECORD",
            id="Q2_INTERPRET_SOURCE_RECORD",
            question=f"If located, do authenticated {topic} records show that {actor_phrase} completed the required {topic}{temporal_suffix}?",
            purpose="Resolves H1 — whether the required activity was actually completed as documented",
            objective=f"Evaluate completion status in located {topic} records",
            evidence=f"Authenticated {topic} attendance/completion record",
            evidence_required=f"Authenticated {topic} attendance/completion record",
            target_type="HYPOTHESIS",
            target_proposition_id="P_H1",
            priority="P2",
            depends_on="Q1_LOCATE_SOURCE_RECORD",
            activation_condition="If source record is located",
            status="CONDITIONAL",
            hypothesis_tested="H1",
            confirms_if=h1.confirms_if,
            refutes_if=h1.refutes_if,
            possible_outcomes=[
                f"Record confirms completion {effective_ref} → H1 refuted/weakened.",
                "Record shows no completion or non-attendance → H1 supported/strengthened.",
            ],
        ))

        # P3 — RECORD-CONTROL CHECK (if record cannot be located)
        questions.append(InvestigationQuestion(
            question_id="Q3_RECORD_CONTROL_REQUIREMENT",
            id="Q3_RECORD_CONTROL_REQUIREMENT",
            question=(
                f"If the record cannot be located, does the record-control audit trail establish whether the {topic} record for {tail} "
                "was required to exist, was created, and was retained in accordance with applicable requirements?"
            ),
            purpose="Determine whether the record-control process (creation/retention/control) itself operated as required",
            objective="Determine whether the record-control process (creation/retention) itself has a control weakness",
            evidence=f"{topic_cap} record-control procedure, retention requirements, and record audit trail",
            evidence_required=f"{topic_cap} record-control procedure, retention requirements, and record audit trail",
            target_type="PROPOSITION",
            target_proposition_id="P_RECORD_CONTROL",
            priority="P3",
            depends_on="Q1_LOCATE_SOURCE_RECORD",
            activation_condition="If source record cannot be located",
            status="CONDITIONAL",
            possible_outcomes=[
                "Audit trail shows the record was required but never created or retained → record-control weakness supported.",
                "Audit trail confirms proper creation and retention or establishes no record requirement → record-control weakness refuted.",
            ],
        ))

        # P4 — CONTROL / AUTHORIZATION CHECK
        questions.append(InvestigationQuestion(
            question_id="Q4_VERIFICATION_AUTHORIZATION",
            id="Q4_VERIFICATION_AUTHORIZATION",
            question=(
                f"Do verification records establish whether {topic} completion was authorized or verified "
                f"before {actor_phrase} performed the relevant activity{temporal_suffix}?"
            ),
            purpose="Determine whether a completion-verification or authorization control failed to operate",
            objective="Determine whether a completion-verification/authorization control failed to catch the gap",
            evidence=f"{topic_cap} authorization requirements and verification/sign-off records",
            evidence_required=f"{topic_cap} authorization requirements and verification/sign-off records",
            target_type="PROPOSITION",
            target_proposition_id="P_VERIFICATION",
            priority="P4",
            possible_outcomes=[
                "Verification/authorization step was required but not executed → verification-control weakness supported.",
                "Verification/authorization step was executed and documented → verification-control weakness refuted.",
            ],
        ))

        # P5 — SCOPE / IMPACT ASSESSMENT
        questions.append(InvestigationQuestion(
            question_id="Q5_SCOPE_DOWNSTREAM_IMPACT",
            id="Q5_SCOPE_DOWNSTREAM_IMPACT",
            question=(
                f"What downstream activities, decisions, outputs, or records could have been affected "
                f"by {actor_phrase} performing the activity without verified {topic} completion?"
            ),
            purpose="Establish the scope, affected time period, and downstream product/process impact",
            objective=f"Assess downstream consequences of unverified {topic} completion",
            evidence=f"Operational logs, batch records, released outputs, or downstream transactions relative to {tail}",
            evidence_required=f"Operational logs, batch records, released outputs, or downstream transactions relative to {tail}",
            target_type="PROPOSITION",
            target_proposition_id="P_SCOPE_IMPACT",
            priority="P5",
            possible_outcomes=[
                "Downstream outputs or records identified → evaluate product/process quality and compliance impact.",
                "No critical downstream outputs affected → scope contained to initial activity.",
            ],
        ))
        evidence_items.extend([
            f"Authenticated {topic} attendance/completion record",
            f"Authenticated {topic} record from the LMS, archive, or {topic} repository",
            f"{topic_cap} record-control procedure and retention requirements",
            f"{topic_cap} authorization/sign-off record",
            f"Downstream operational logs and output records for {tail}",
        ])
        # INVESTIGATION AREAS (Section 7 & 9): the causal hypothesis's own
        # area, plus the evidence-availability and systemic record-control
        # areas that exist independently of whether H1 itself survives.
        plan_areas = [
            f"{topic_cap} completion verification",
            f"{topic_cap} record availability",
            f"{topic_cap} record-control process",
        ]

    # 2. Knowledge / Revision Gap Branch (awareness/communication only --
    # "X was unaware of a revision" is evidence about notification/
    # acknowledgement, never by itself evidence about training/competency
    # or about the procedure's own clarity. Training and procedural-clarity
    # hypotheses are DELIBERATELY not generated here regardless of rank --
    # each is its own distinct causal domain requiring its own evidence
    # trigger (training/competency vocabulary; unclear/ambiguous/confusing
    # vocabulary respectively), neither of which this branch's trigger
    # condition (an unaware-of-revision statement) establishes on its own.
    elif mechanism.polarity == "knowledge_gap" or ("unaware" in text_low and "revision" in text_low):
        h1 = CandidateHypothesis(
            id="H1",
            name="REVISION_COMMUNICATION_OR_ACKNOWLEDGEMENT_GAP",
            statement=f"The revision affecting {subject} may not have been effectively communicated to or acknowledged by the affected personnel.",
            status="POSSIBLE",
            evidence_needed=f"{subject} revision distribution records, change notification records, acknowledgement records",
            confirms_if="No distribution, notification, or acknowledgement record exists confirming personnel received the revision",
            refutes_if="Distribution, notification, or acknowledgement records confirm personnel received and acknowledged the revision",
            discrimination_evidence="Distinguishes a communication/acknowledgement delivery gap from other causes of the deviation",
            relevance_rank="HIGH",
        )
        hypotheses.append(h1)

        questions.append(InvestigationQuestion(
            question_id="Q_REVISION_COMMUNICATION",
            id="Q_REVISION_COMMUNICATION",
            question=f"Do revision distribution, notification, or acknowledgement records show whether the affected personnel received and acknowledged the revision affecting {subject}?",
            purpose="Determine whether the revision was effectively communicated to and acknowledged by affected personnel",
            objective="Determine whether the revision was effectively communicated to and acknowledged by affected personnel",
            evidence=f"{subject} revision distribution records, change notification records, acknowledgement records",
            evidence_required=f"{subject} revision distribution records, change notification records, acknowledgement records",
            hypothesis_tested="H1",
            target_type="HYPOTHESIS",
            target_proposition_id="P_H1",
            priority="P4",
            confirms_if=h1.confirms_if,
            refutes_if=h1.refutes_if,
        ))
        questions.append(InvestigationQuestion(
            question_id="Q_COMMUNICATION_METHOD",
            id="Q_COMMUNICATION_METHOD",
            question=f"What records establish how the revision affecting {subject} was communicated to affected personnel?",
            purpose="Establish what communication, if any, is documented, without presupposing it failed",
            objective="Establish what communication, if any, is documented, without presupposing it failed",
            evidence=f"{subject} revision distribution and communication records",
            evidence_required=f"{subject} revision distribution and communication records",
            target_proposition_id="P_COMM",
            priority="P2",
        ))
        questions.append(InvestigationQuestion(
            question_id="Q_ACKNOWLEDGEMENT_REQUIREMENT",
            id="Q_ACKNOWLEDGEMENT_REQUIREMENT",
            question=f"Did the applicable procedure require acknowledgement of the revision affecting {subject} before work commenced?",
            purpose="Establish whether an acknowledgement requirement existed and was documented",
            objective="Establish whether an acknowledgement requirement existed and was documented",
            evidence=f"{subject} acknowledgement requirement and records",
            evidence_required=f"{subject} acknowledgement requirement and records",
            target_proposition_id="P_REQ_ACK",
            priority="P3",
        ))
        evidence_items.extend([f"{subject} revision distribution records", f"{subject} acknowledgement records"])

    # 3. Non-performance Branch: fires on any REPORTED "X was missed/not
    # performed" claim. CRITICAL DISTINCTION: a bare report that something
    # was missed (even with specific temporal/contextual framing, e.g.
    # "during the morning shift") contains NO causal content -- it restates
    # the deviation, it does not explain it. Generating ANY hypothesis from
    # a bare restatement just substitutes a differently-worded invented
    # mechanism (a generic "execution/task-control factor" bucket) for a
    # more specific one (e.g. "shift plan"); both are unlicensed. Only a
    # genuine reported causal clause ("...because they had not received
    # retraining") licenses building a hypothesis -- from THAT reported
    # reason specifically, not from a generic category.
    elif mechanism.polarity == "non_performance":
        from app.agent.causal_guard import reported_claims_contain_causal_explanation
        if reported_claims_contain_causal_explanation(reported_claims):
            # A genuine reported reason exists -- reflect it directly
            # (hedged) rather than inventing a separate generic bucket.
            nonperf_topic = topic_word(subject)
            h1 = CandidateHypothesis(
                id="H1",
                name="REPORTED_CAUSE_REQUIRES_VERIFICATION",
                statement=f"{mechanism.statement.rstrip('.')}, which may have contributed to the missed {subject} activity, but this has not been independently verified.",
                status="POSSIBLE",
                evidence_needed=f"Objective records capable of confirming or refuting the reported reason for {subject}",
                confirms_if="Objective records are consistent with the reported reason and no contradicting evidence exists",
                refutes_if="Objective records contradict the reported reason",
                discrimination_evidence="Distinguishes whether the specific reported reason is independently supported by objective evidence",
                relevance_rank="HIGH",
            )
            hypotheses.append(h1)

            questions.append(InvestigationQuestion(
                question=f"Do objective records confirm or contradict the reported reason for the missed {subject} activity?",
                purpose="Independently verify the specific reason reported for the deviation",
                evidence=f"Applicable {nonperf_topic} procedure and objective records relevant to the reported reason",
                hypothesis_tested="H1",
                confirms_if=h1.confirms_if,
                refutes_if=h1.refutes_if,
            ))
            evidence_items.append(f"Objective records relevant to the reported reason for {subject}")
        else:
            # No causal content in the reported claim -- correctly zero
            # hypotheses. Still contribute neutral, non-presupposing
            # investigation areas/evidence so the plan isn't empty; these
            # areas are investigation candidates, not asserted mechanisms.
            plan_areas = [
                f"Applicable requirement and responsibility controls for {subject}",
                f"Task assignment/scheduling controls for {subject}",
                f"Completion and secondary-record verification for {subject}",
            ]
            from app.services.semantic_subject import is_actor_noun
            if is_actor_noun(subject):
                q1 = "What procedure and responsibility requirements applied to personnel during the affected period?"
                q2 = "What evidence establishes whether the required activity was performed but not recorded by assigned personnel?"
                q3 = "Is there a documented event or constraint that could have affected execution during the affected period?"
            elif any(w in subject.lower() for w in ("checklist", "log", "record", "sheet", "form")):
                q1 = f"What requirement and responsibility applied to {subject} during the affected period?"
                q2 = f"What evidence establishes whether {subject} was performed but not recorded during the affected period?"
                q3 = f"Is there a documented event that could have affected completion of {subject} during the affected period?"
            elif any(w in subject.lower() for w in ("balance", "equipment", "instrument", "machine")):
                q1 = f"What requirement and specification applied to {subject} during the affected period?"
                q2 = f"What records establish whether the activity for {subject} was performed but not recorded?"
                q3 = f"Is there a documented event that could have affected operation of {subject} during the affected period?"
            else:
                q1 = f"What requirement and responsibility applied to {subject} during the affected period?"
                q2 = f"Is there objective evidence that the required activity for {subject} was performed but not recorded?"
                q3 = f"Is there a documented event that could have affected {subject} during the affected period?"

            questions.append(InvestigationQuestion(
                question=q1,
                purpose="Establish the applicable requirement and responsibility/assignment/scheduling controls before any specific mechanism can be investigated",
                evidence=f"Applicable procedure, responsibility matrix, duty/shift assignment records for {subject}",
                hypothesis_tested=None,
            ))
            questions.append(InvestigationQuestion(
                question=q2,
                purpose="Distinguishes non-performance from an unrecorded performance",
                evidence=f"Secondary records, electronic/instrument audit trail, supervisory verification for {subject}",
                hypothesis_tested=None,
            ))
            questions.append(InvestigationQuestion(
                question=q3,
                purpose="Identifies whether a contemporaneous event could explain the missed activity",
                evidence=f"Deviation, incident, equipment alarm, maintenance, or staffing records for {subject}, where applicable",
                hypothesis_tested=None,
            ))
            evidence_items.extend([
                f"Applicable procedure and responsibility matrix for {subject}",
                f"Secondary/independent verification records for {subject}",
                f"Deviation/incident/contemporaneous records for {subject}",
            ])

    # 3a. Requirement-resolution Branch (Section 4/6/7/8): when the
    # applicable requirement governing an observed condition is itself
    # reported unresolved, root-cause/mechanism questions are premature --
    # the FIRST priority is establishing what requirement applies, THEN
    # comparing the observed condition against it, and ONLY THEN (if a
    # deviation is actually confirmed) investigating cause. No hypotheses
    # are generated here: a hypothesis about WHY something happened is not
    # yet meaningful when it is not yet established THAT it happened.
    # Trigger is the structural REQUIREMENT_UNCERTAIN classification, never
    # a keyword list for one specific requirement type.
    elif resolved.semantic_type == "REQUIREMENT_UNCERTAIN" or (resolved.requirement_status == "UNKNOWN" and getattr(resolved, "semantic_type", None) in ("REQUIREMENT_UNCERTAIN", "OBSERVATION_VERIFICATION")):
        plan_areas = [
            f"Applicable requirement, specification, and standard for {subject}",
            f"Observed condition compliance determination and deviation confirmation for {subject}",
            f"Scope and downstream impact assessment for {subject}",
        ]
        questions.extend([
            InvestigationQuestion(
                question_id="P1_REQUIREMENT", id="P1_REQUIREMENT",
                question=f"What approved requirement, procedure, specification, instruction, or control governed {subject} during the relevant period?",
                purpose="Establish the applicable requirement before any compliance or causal determination can be made",
                objective="Establish the applicable requirement before any compliance or causal determination can be made",
                evidence=f"Applicable procedure/specification/standard governing {subject}, including version and scope of applicability",
                evidence_required=f"Applicable procedure/specification/standard governing {subject}, including version and scope of applicability",
                priority="P1",
                target_proposition_id="P_REQUIREMENT_UNCERTAIN",
                uncertainty_resolved="REQUIREMENT_UNCERTAIN",
                status="ACTIVE",
                category="REQUIREMENT_RESOLUTION",
                decision_rule="IF no applicable requirement can be located → the finding requires reassessment. IF located → proceed to P2_COMPARISON.",
                next_step_if_true="P2_COMPARISON",
                next_step_if_false="CLOSE_OR_REASSESS",
                next_question_if_true="P2_COMPARISON",
            ),
            InvestigationQuestion(
                question_id="P2_COMPARISON", id="P2_COMPARISON",
                question=f"Once the applicable requirement is established, does the observed condition of {subject} comply with it, including any approved exception?",
                purpose="Compare the observed condition against the applicable requirement, including permitted exceptions",
                objective="Compare the observed condition against the applicable requirement, including permitted exceptions",
                evidence=f"Observed-condition record for {subject} and any approved exception/waiver applicable at the relevant time",
                evidence_required=f"Observed-condition record for {subject} and any approved exception/waiver applicable at the relevant time",
                priority="P2",
                target_proposition_id="P_COMPLIANCE_DETERMINATION",
                uncertainty_resolved="CONTROL_EXECUTION_UNCERTAIN",
                depends_on="P1_REQUIREMENT",
                status="CONDITIONAL",
                category="REQUIREMENT_RESOLUTION",
                decision_rule=(
                    "IF the condition complies (or an approved exception applies) → the finding requires "
                    "reassessment; no deviation is confirmed. IF it does not comply → proceed to P3_DEVIATION."
                ),
                next_step_if_true="P3_DEVIATION",
                next_step_if_false="CLOSE_OR_REASSESS",
                next_question_if_true="P3_DEVIATION",
            ),
            InvestigationQuestion(
                question_id="P3_DEVIATION", id="P3_DEVIATION",
                question=f"Does the comparison in P2 confirm an actual deviation for {subject}, as distinct from the requirement merely being unresolved?",
                purpose="Determine whether a deviation actually exists before investigating its cause",
                objective="Determine whether a deviation actually exists before investigating its cause",
                evidence="The completed requirement-vs-condition comparison from P2",
                evidence_required="The completed requirement-vs-condition comparison from P2",
                priority="P3",
                target_proposition_id="P_DEVIATION_CONFIRMED",
                uncertainty_resolved="OBSERVATION_UNCERTAIN",
                depends_on="P2_COMPARISON",
                status="CONDITIONAL",
                category="REQUIREMENT_RESOLUTION",
                decision_rule="ONLY IF a deviation is confirmed here should mechanism/cause investigation begin.",
                next_step_if_true="P4_MECHANISM",
                next_step_if_false="CLOSE_OR_REASSESS",
            ),
            InvestigationQuestion(
                question_id="P4_MECHANISM", id="P4_MECHANISM",
                question=f"What operational, procedural, or control mechanism permitted the condition for {subject} to occur?",
                purpose="Investigate the causal mechanism if and only if a deviation is confirmed",
                objective="Investigate the causal mechanism if and only if a deviation is confirmed",
                evidence=f"Operational logs, execution records, and supervisory reviews for {subject}",
                evidence_required=f"Operational logs, execution records, and supervisory reviews for {subject}",
                priority="P4",
                target_proposition_id="P_MECHANISM_INVESTIGATION",
                uncertainty_resolved="MECHANISM_UNCERTAIN",
                depends_on="P3_DEVIATION",
                status="CONDITIONAL",
                category="MECHANISM_INVESTIGATION",
                decision_rule="IF mechanism identified → proceed to scope/impact. IF unknown → evidence gap remains.",
                next_step_if_true="P5_SCOPE_IMPACT",
            ),
            InvestigationQuestion(
                question_id="P5_SCOPE_IMPACT", id="P5_SCOPE_IMPACT",
                question=f"What is the extent of condition, population scope, and downstream impact of the condition for {subject}?",
                purpose="Establish scope and downstream consequence across the affected population",
                objective="Establish scope and downstream consequence across the affected population",
                evidence=f"Full period records, inventory/batch logs, and disposition documentation for {subject}",
                evidence_required=f"Full period records, inventory/batch logs, and disposition documentation for {subject}",
                priority="P5",
                target_proposition_id="P_SCOPE_AND_IMPACT",
                uncertainty_resolved="SCOPE_UNCERTAIN",
                depends_on="P3_DEVIATION",
                status="CONDITIONAL",
                category="IMPACT_ASSESSMENT",
                decision_rule="Establishes population boundary and downstream exposure.",
            ),
        ])
        evidence_items.extend([
            f"Applicable procedure/specification/standard governing {subject}",
            f"Observed-condition record for {subject}",
            "Approved exception/waiver records applicable at the relevant time",
            f"Operational logs and supervisory reviews for {subject}",
            f"Full period records and disposition documentation for {subject}",
        ])


    # 3b. Event-sequence / control-point Branch (Section 6/9): a controlled
    # TRANSITION (invalidation, override, exception, waiver, ...) whose
    # required justification/authorization is reported missing generates
    # exactly ONE testable control-gap hypothesis -- never a specific
    # mechanism (operator error, system error, intentional bypass) absent
    # supporting evidence. Trigger is the structural transition_type
    # classification, never a keyword list for one specific transition.
    elif resolved.semantic_type == "EVENT_SEQUENCE_CONTROL" and resolved.transition_type:
        _transition_label = resolved.transition_type.replace("_", " ").lower()
        h1 = CandidateHypothesis(
            id="H1",
            name="CONTROL_EXECUTION_GAP",
            statement=f"A control or authorization gap existed at the {_transition_label} transition for {subject}.",
            status="POSSIBLE",
            evidence_strength="INDICATIVE",
            evidence_needed=f"Applicable procedure governing the {_transition_label}, and any authorization/review record for this specific transition",
            confirms_if=f"No applicable authorization/review record exists for this {_transition_label}, or the applicable procedure required one that was not obtained",
            refutes_if=f"An authorization/review record for this {_transition_label} is located and satisfies the applicable procedure",
            discrimination_evidence=f"H1 is distinguished by locating (or failing to locate) the {_transition_label}'s own authorization/review record.",
            relevance_rank="MEDIUM",
            supporting_evidence=fact_claims[:1] if fact_claims else [],
            resolves_investigation="Q_CONTROL_EXECUTED",
        )
        hypotheses.append(h1)
        questions.extend([
            InvestigationQuestion(
                question_id="Q_CONFIRM_TRANSITION", id="Q_CONFIRM_TRANSITION",
                question=f"What objective evidence confirms that the {_transition_label} occurred as described?",
                purpose="Confirm the triggering transition before investigating its control",
                objective="Confirm the triggering transition before investigating its control",
                evidence=f"Records documenting the {_transition_label} event itself",
                evidence_required=f"Records documenting the {_transition_label} event itself",
                priority="P1",
                target_proposition_id="P_TRANSITION_CONFIRMED",
                status="ACTIVE",
                category="OBSERVATION_VERIFICATION",
                decision_rule=f"IF the {_transition_label} is not confirmed → resolve the observation independently. IF confirmed → proceed to Q_CONTROL_IDENTIFIED.",
                next_question_if_true="Q_CONTROL_IDENTIFIED",
            ),
            InvestigationQuestion(
                question_id="Q_CONTROL_IDENTIFIED", id="Q_CONTROL_IDENTIFIED",
                question=f"What approved requirement, authorization, review, or decision rule governed this {_transition_label}?",
                purpose="Identify the control that should have governed the transition",
                objective="Identify the control that should have governed the transition",
                evidence=f"Applicable procedure/policy governing a {_transition_label} of this kind",
                evidence_required=f"Applicable procedure/policy governing a {_transition_label} of this kind",
                priority="P1",
                target_proposition_id="P_CONTROL_IDENTIFIED",
                depends_on="Q_CONFIRM_TRANSITION",
                status="CONDITIONAL",
                category="MECHANISM_INVESTIGATION",
                decision_rule="Establishes which control applies before assessing whether it was executed.",
                next_question_if_true="Q_CONTROL_EXECUTED",
            ),
            InvestigationQuestion(
                question_id="Q_CONTROL_EXECUTED", id="Q_CONTROL_EXECUTED",
                question=f"Was the applicable control executed, and was it documented, for this {_transition_label}?",
                purpose="Determine whether the control was executed, bypassed, or simply not documented",
                objective="Determine whether the control was executed, bypassed, or simply not documented",
                evidence="Authorization/review/sign-off record for this specific transition",
                evidence_required="Authorization/review/sign-off record for this specific transition",
                priority="P2",
                target_proposition_id="P_CONTROL_EXECUTED",
                depends_on="Q_CONTROL_IDENTIFIED",
                status="CONDITIONAL",
                category="MECHANISM_INVESTIGATION",
                hypothesis_tested="H1",
                target_hypothesis_ids=["H1"],
                decision_rule=(
                    "EXECUTED → the control operated; investigate documentation practice only. "
                    "NOT_EXECUTED/BYPASSED → investigate the control-gap mechanism (H1). "
                    "NOT_DOCUMENTED → executed but undocumented; distinct from bypass. "
                    "UNKNOWN → evidence gap remains."
                ),
                possible_outcomes=[
                    "Executed and documented → the control operated as required.",
                    "Not executed or bypassed → investigate the control-gap mechanism (H1).",
                    "Executed but not documented → a documentation gap, not a control failure.",
                    "Cannot be determined → the evidence gap remains.",
                ],
            ),
        ])
        if getattr(resolved, "downstream_action_present", False):
            questions.append(InvestigationQuestion(
                question_id="Q_DOWNSTREAM_DEPENDENCY", id="Q_DOWNSTREAM_DEPENDENCY",
                question=f"Did the downstream action reported in the finding depend on the outcome of this {_transition_label}?",
                purpose="Assess downstream dependency without presuming the downstream action was improper",
                objective="Assess downstream dependency without presuming the downstream action was improper",
                evidence="Records showing what the downstream action relied on at the time it occurred",
                evidence_required="Records showing what the downstream action relied on at the time it occurred",
                priority="P3",
                target_proposition_id="P_DOWNSTREAM_DEPENDENCY",
                status="ACTIVE",
                category="IMPACT_ASSESSMENT",
                decision_rule="Establishes dependency, not fault -- never asserts the downstream action was improper absent objective evidence.",
            ))
        evidence_items.extend([
            f"Records documenting the {_transition_label} event",
            f"Applicable procedure/policy governing a {_transition_label} of this kind",
            f"Authorization/review/sign-off record for this specific transition",
        ])

    # 4. Non-recording Branch (MISSING_RECORD decision tree, Section 3/5/6):
    # STEP 1 confirms the missing evidence itself; STEP 2 distinguishes
    # PERFORMED_NOT_RECORDED / NOT_PERFORMED / UNKNOWN (never assumes
    # non-performance); STEP 3, added only when the canonical extraction
    # detected a downstream action, verifies whether the missing evidence
    # was required before that action and whether it remains appropriately
    # supported -- never asserts the downstream action was improper.
    # Generalizes across any domain: the branch trigger is
    # mechanism.polarity == "non_recording" (a structural classification),
    # not a keyword list for any one activity/record type.
    elif mechanism.polarity == "non_recording" or resolved.semantic_type == "MISSING_RECORD":
        h1 = CandidateHypothesis(
            id="H1",
            name="DOCUMENTATION_TIMELINESS_OR_COMPLIANCE_GAP",
            statement=f"The operational activity for {subject} was executed but contemporaneous recording was omitted or delayed.",
            status="POSSIBLE",
            evidence_needed=f"{subject} execution logs, system audit trail, secondary verification records",
            confirms_if="Secondary records or physical evidence confirm execution but primary log is blank",
            refutes_if="Secondary records confirm activity was not executed",
            discrimination_evidence="Distinguishes documentation omission from physical non-performance",
            relevance_rank="HIGH",
            resolves_investigation="Q_ACTIVITY_PERFORMANCE",
        )
        h2 = CandidateHypothesis(
            id="H2",
            name="RECORDING_MEDIA_OR_INTERFACE_AVAILABILITY_GAP",
            statement=f"Recording forms or digital interfaces for {subject} were temporarily unavailable during execution.",
            status="POSSIBLE",
            evidence_needed=f"{subject} workstation log, interface audit trail",
            confirms_if="Workstation log records interface downtime or form stockout",
            refutes_if="Workstation audit trail confirms recording interface was operational",
            discrimination_evidence="Distinguishes recording interface issue from documentation compliance delay",
            relevance_rank="HIGH",
            resolves_investigation="Q_ACTIVITY_PERFORMANCE",
        )
        hypotheses.extend([h1, h2])

        questions.append(InvestigationQuestion(
            question_id="Q_CONFIRM_MISSING_EVIDENCE", id="Q_CONFIRM_MISSING_EVIDENCE",
            question=f"Does the applicable record-keeping system confirm that the required record for {subject} is genuinely absent, not merely misfiled or delayed in entry?",
            purpose="Confirm the missing-evidence observation itself before investigating its cause",
            objective="Confirm the missing-evidence observation itself before investigating its cause",
            evidence=f"{subject} record repository/log index and record-control procedure",
            evidence_required=f"{subject} record repository/log index and record-control procedure",
            priority="P1",
            target_proposition_id="P_MISSING_EVIDENCE",
            status="ACTIVE",
            category="OBSERVATION_VERIFICATION",
            decision_rule="IF the record is found → the finding may require re-evaluation. IF genuinely absent → proceed to Q_ACTIVITY_PERFORMANCE.",
        ))
        questions.append(InvestigationQuestion(
            question_id="Q_ACTIVITY_PERFORMANCE", id="Q_ACTIVITY_PERFORMANCE",
            question=f"Do secondary physical records or electronic timestamps confirm execution of {subject} despite the missing log entry?",
            purpose="Evaluate contemporaneous recording compliance vs physical non-execution",
            objective="Evaluate contemporaneous recording compliance vs physical non-execution",
            evidence=f"{subject} execution logs, system audit trail",
            evidence_required=f"{subject} execution logs, system audit trail",
            priority="P1",
            target_proposition_id="P_ACTIVITY_PERFORMANCE",
            depends_on="Q_CONFIRM_MISSING_EVIDENCE",
            status="CONDITIONAL",
            category="MECHANISM_INVESTIGATION",
            hypothesis_tested="H1",
            target_hypothesis_ids=["H1", "H2"],
            confirms_if=h1.confirms_if,
            refutes_if=h1.refutes_if,
            decision_rule=(
                "IF activity confirmed performed → investigate the recording/documentation failure (H1/H2). "
                "IF activity confirmed not performed → investigate the execution/control failure. "
                "IF it cannot be determined → the evidence gap remains and root cause stays NOT_ESTABLISHED."
            ),
            possible_outcomes=[
                "Activity confirmed performed → investigate recording/documentation failure.",
                "Activity confirmed not performed → investigate execution/control failure.",
                "Cannot be determined → evidence gap remains.",
            ],
        ))
        evidence_items.extend([f"{subject} execution log", "system audit trail", f"{subject} record repository/log index"])

        # STEP 3: downstream-control investigation (Section 4/5/6) -- only
        # added when the canonical extraction objectively detected a
        # downstream action in the finding text itself, never inferred.
        if getattr(resolved, "downstream_action_present", False):
            questions.append(InvestigationQuestion(
                question_id="Q_DOWNSTREAM_EVIDENCE_REQUIRED", id="Q_DOWNSTREAM_EVIDENCE_REQUIRED",
                question=f"Was the missing record for {subject} required to be available before the downstream action reported in the finding?",
                purpose="Determine whether the missing evidence was a precondition for the downstream action",
                objective="Determine whether the missing evidence was a precondition for the downstream action",
                evidence="Applicable procedure defining prerequisite records/controls for the downstream action",
                evidence_required="Applicable procedure defining prerequisite records/controls for the downstream action",
                priority="P2",
                target_proposition_id="P_DOWNSTREAM_PRECONDITION",
                status="ACTIVE",
                category="DOWNSTREAM_AUTHORIZATION",
                decision_rule="Establishes whether the missing record was a required precondition, independent of whether it was actually reviewed.",
            ))
            questions.append(InvestigationQuestion(
                question_id="Q_DOWNSTREAM_REVIEW_EVIDENCE", id="Q_DOWNSTREAM_REVIEW_EVIDENCE",
                question="What evidence was reviewed before the downstream action occurred, and was the missing record identified during that review?",
                purpose="Establish what review actually occurred and whether the gap was already known at that time",
                objective="Establish what review actually occurred and whether the gap was already known at that time",
                evidence="Review/sign-off records and disposition documentation for the downstream action",
                evidence_required="Review/sign-off records and disposition documentation for the downstream action",
                priority="P2",
                target_proposition_id="P_DOWNSTREAM_REVIEW",
                depends_on="Q_DOWNSTREAM_EVIDENCE_REQUIRED",
                status="CONDITIONAL",
                category="DOWNSTREAM_AUTHORIZATION",
                decision_rule="IF the gap was identified and addressed before the action → the applicable control operated. IF not → investigate review/detection effectiveness.",
            ))
            questions.append(InvestigationQuestion(
                question_id="Q_DOWNSTREAM_AUTHORIZATION", id="Q_DOWNSTREAM_AUTHORIZATION",
                question="Was the downstream action authorized under the applicable procedure, and did the missing record affect that authorization decision?",
                purpose="Assess authorization effectiveness without presuming the downstream action was improper",
                objective="Assess authorization effectiveness without presuming the downstream action was improper",
                evidence="Applicable authorization procedure and the specific authorization record for the downstream action",
                evidence_required="Applicable authorization procedure and the specific authorization record for the downstream action",
                priority="P3",
                target_proposition_id="P_DOWNSTREAM_AUTHORIZATION",
                status="ACTIVE",
                category="IMPACT_ASSESSMENT",
                decision_rule="Establishes whether the downstream action remains appropriately supported -- never asserts it was improper absent objective evidence.",
            ))
            evidence_items.extend([
                "Applicable procedure defining prerequisite records for the downstream action",
                "Review/sign-off and authorization records for the downstream action",
            ])

    # 5. Duplicate Payment / Transaction Branch (Section 5 Hardening)
    elif re.search(r"\b(?:duplicate\s+payment|paid\s+twice|double\s+payment|overpayment)\b", finding_text, re.IGNORECASE):
        amt_str = ""
        from app.services.cost_analysis import extract_explicit_amounts, format_currency_amount
        explicit_amts = extract_explicit_amounts(finding_text)
        if explicit_amts:
            a_val, a_curr = explicit_amts[0]
            amt_str = f" of {format_currency_amount(a_val, a_curr)}"

        plan_areas = [
            "Payment transaction verification and duplicate detection controls",
            "ERP exception handling, approval, and workflow bypass controls",
            "Supplier master data and invoice indexing controls",
            "Accounts-payable reconciliation and recovery controls",
        ]

        hypotheses = [
            CandidateHypothesis(
                id="H1",
                name="DUPLICATE_DETECTION_MATCHING_GAP",
                statement="The second payment transaction was processed without triggering duplicate invoice or amount matching warnings.",
                status="POSSIBLE",
                evidence_needed="ERP duplicate-detection configuration and transaction processing logs",
                confirms_if="ERP audit logs show duplicate detection rules did not trigger on invoice/amount matching",
                refutes_if="ERP logs show duplicate detection operated and raised an exception",
                discrimination_evidence="Distinguishes automated detection gap from workflow bypass",
                relevance_rank="HIGH",
                causal_role="PRIMARY_CAUSE",
            ),
            CandidateHypothesis(
                id="H2",
                name="APPROVAL_OR_WORKFLOW_BYPASS",
                statement="The duplicate payment was executed under an exception override or authorization bypass.",
                status="POSSIBLE",
                evidence_needed="Payment approval logs, exception override records, and user authorization trails",
                confirms_if="Audit trail shows warning or exception was manually overridden without dual authorization",
                refutes_if="Audit trail confirms standard dual authorization was followed for both transactions",
                discrimination_evidence="Distinguishes workflow override from detection failure",
                relevance_rank="HIGH",
                causal_role="PRIMARY_CAUSE",
            ),
            CandidateHypothesis(
                id="H3",
                name="DUPLICATE_MASTER_DATA_OR_INVOICE_VARIANCE",
                statement="Duplicate vendor numbering or altered invoice reference data allowed transaction processing.",
                status="POSSIBLE",
                evidence_needed="Supplier master data records, invoice indexing records, and vendor ID audit logs",
                confirms_if="Supplier records reveal duplicate vendor IDs or altered invoice numbering",
                refutes_if="Master data audit confirms identical single vendor record and exact invoice number",
                discrimination_evidence="Distinguishes master data variance from detection rule failure",
                relevance_rank="HIGH",
                causal_role="CONTRIBUTING_CAUSE",
            ),
            CandidateHypothesis(
                id="H4",
                name="RECONCILIATION_DETECTION_GAP",
                statement="The duplicate payment was not identified or flagged during periodic accounts-payable reconciliation.",
                status="POSSIBLE",
                evidence_needed="Periodic AP reconciliation logs and supervisory review records",
                confirms_if="AP reconciliation records show monthly supplier sub-ledger was not reconciled against bank disbursements",
                refutes_if="Reconciliation records show timely matching or documented exception logging",
                discrimination_evidence="Distinguishes post-payment reconciliation delay from transaction-time controls",
                relevance_rank="HIGH",
                causal_role="DETECTION_FAILURE",
            ),
        ]

        questions = [
            # A. ROOT-CAUSE INVESTIGATION
            InvestigationQuestion(
                question="Do the two payment records correspond to the same supplier, invoice, purchase order, amount, or underlying obligation?",
                purpose="Establish whether the payments represent a true duplicate transaction vs distinct obligations",
                evidence="Payment records, supplier invoice records, purchase orders, and transaction IDs",
                question_type="ROOT_CAUSE",
                category="ROOT_CAUSE_INVESTIGATION",
                decision_effect="Establishes duplicate transaction validity",
            ),
            InvestigationQuestion(
                question="Did the ERP duplicate-payment control execute when the second transaction was processed?",
                purpose="Determine automated control execution status during transaction processing",
                evidence="ERP audit trail, duplicate detection logs, and exception logs",
                question_type="ROOT_CAUSE",
                category="ROOT_CAUSE_INVESTIGATION",
                hypothesis_tested="H1",
                decision_effect="Distinguishes detection rule failure from workflow override",
            ),
            InvestigationQuestion(
                question="Was the approval workflow bypassed or overridden using elevated authorization?",
                purpose="Evaluate potential workflow override or authorization bypass",
                evidence="Override logs, approval records, and user audit trail",
                question_type="ROOT_CAUSE",
                category="ROOT_CAUSE_INVESTIGATION",
                hypothesis_tested="H2",
                decision_effect="Identifies authorization override mechanism",
            ),
            InvestigationQuestion(
                question="Did duplicate supplier master data or altered invoice indexing permit transaction creation?",
                purpose="Verify master data integrity and invoice reference indexing",
                evidence="Supplier master data logs, vendor ID tables, and invoice indexing entries",
                question_type="ROOT_CAUSE",
                category="ROOT_CAUSE_INVESTIGATION",
                hypothesis_tested="H3",
                decision_effect="Identifies master data indexing discrepancy",
            ),
            # B. DETECTION-CONTROL INVESTIGATION
            InvestigationQuestion(
                question="Was the duplicate payment identified or flagged during periodic accounts-payable reconciliation?",
                purpose="Assess detective post-payment reconciliation controls",
                evidence="Accounts-payable reconciliation records, sub-ledger review evidence",
                question_type="DETECTION_CONTROL",
                category="DETECTION_CONTROL_INVESTIGATION",
                hypothesis_tested="H4",
                decision_effect="Evaluates effectiveness of post-payment detective controls",
            ),
            InvestigationQuestion(
                question="Did supervisory review or secondary approval identify the duplicate transaction prior to disbursement?",
                purpose="Verify secondary supervisory review execution and exception notification",
                evidence="Supervisory sign-off logs, banking batch approvals, and exception queue audit trails",
                question_type="DETECTION_CONTROL",
                category="DETECTION_CONTROL_INVESTIGATION",
                hypothesis_tested="H4",
                decision_effect="Assesses pre-disbursement detective review",
            ),
            # C. FINANCIAL INVESTIGATION
            InvestigationQuestion(
                question=f"Has the duplicate payment{amt_str} been reversed, recovered, or credited by the supplier?",
                purpose="Determine actual financial loss and current recoverability status",
                evidence="Bank records, supplier credit note, and recovery documentation",
                question_type="FINANCIAL_IMPACT",
                category="FINANCIAL_INVESTIGATION",
                decision_effect="Calculates net actual financial loss vs exposure",
            ),
            InvestigationQuestion(
                question="What amount remains unrecovered, if any?",
                purpose="Establish net unrecovered financial exposure",
                evidence="Bank reconciliation statement and supplier credit balance",
                question_type="FINANCIAL_IMPACT",
                category="FINANCIAL_INVESTIGATION",
                decision_effect="Bounds outstanding balance requiring recovery",
            ),
            # D. SYSTEMIC INVESTIGATION
            InvestigationQuestion(
                question="Were other duplicate payments processed during the same period population?",
                purpose="Assess whether the duplicate payment issue is isolated or systemic across accounts payable",
                evidence="Accounts payable audit trail and full-period payment reconciliation report",
                question_type="SYSTEMIC",
                category="SYSTEMIC_INVESTIGATION",
                decision_effect="Determines scope of population review and systemic recurrence risk",
            ),
        ]
        evidence_items.extend([
            "Payment transaction records",
            "Supplier invoice records and purchase orders",
            "ERP duplicate-detection logs and override audit trails",
            "Approval workflow and authorization records",
            "Bank/payment records and supplier credit notes",
            "Accounts-payable reconciliation records",
        ])
        all_cids = [item.claim_id for item in evidence_ledger if hasattr(item, "claim_id") and item.claim_id] or ["C1"]
        claim_map = {item.claim_id: (getattr(item, "claim", None) or getattr(item, "text", "")) for item in evidence_ledger if hasattr(item, "claim_id") and item.claim_id}
        for h in hypotheses:
            if not h.supporting_claim_ids:
                h.supporting_claim_ids = all_cids[:1]
            if not h.supporting_evidence and h.supporting_claim_ids:
                h.supporting_evidence = [claim_map[cid] for cid in h.supporting_claim_ids if cid in claim_map]

        return hypotheses, InvestigationPlan(
            areas=plan_areas,
            questions=questions,
            evidence_to_collect=list(dict.fromkeys(evidence_items)),
            root_cause_questions=[q for q in questions if q.question_type == "ROOT_CAUSE"],
            detection_control_questions=[q for q in questions if q.question_type == "DETECTION_CONTROL"],
            financial_questions=[q for q in questions if q.question_type == "FINANCIAL_IMPACT"],
            systemic_questions=[q for q in questions if q.question_type == "SYSTEMIC"],
        )

    # 6. General / Unresolved Branch: no conflict, no reported mechanism, no
    # recognized mechanism polarity -- by definition NOTHING in the finding
    # states or implies a candidate causal mechanism. Generating two
    # hypotheses here regardless (as this branch previously did) is exactly
    # the "specific-looking but invented RCA" anti-pattern the evidence
    # discipline throughout this codebase exists to prevent: "execution
    # compliance gap" / "verification control gap" READ as concrete
    # findings but are actually just the two halves of "something, somehow,
    # went wrong" with the finding's subject substituted in -- a generic
    # causal bucket wearing a specific-sounding name, which is why it was
    # evading the causal-bucket guard (that guard matches vaguer, more
    # obviously hedged phrasing, not this). Zero hypotheses is the correct,
    # evidence-honest output here (Section 21: "ZERO HYPOTHESES IS A VALID
    # AND OFTEN CORRECT OUTPUT") -- what this finding DOES support is a set
    # of foundational, non-presupposing questions about what's actually
    # established, which the code below asks instead of assuming an answer.
    else:
        # The subject phrase may already contain "status"/"condition"
        # anywhere in it (e.g. "equipment calibration status for BAL-014"),
        # not only at the end -- appending our own "status and
        # authorization"/"condition" regardless would double the word
        # ("...status for BAL-014 status and authorization"), the exact
        # "X status for X status" pattern this codebase's own style rules
        # elsewhere reject. Check for the word anywhere and adapt wording.
        subject_bare = re.sub(r"\b(?:status|condition)\b", "", subject or "", flags=re.IGNORECASE)
        subject_bare = re.sub(r"\s{2,}", " ", subject_bare).strip()
        subject_has_status_word = bool(re.search(r"\b(?:status|condition)\b", subject or "", re.IGNORECASE))
        subject_cap = subject[0].upper() + subject[1:] if subject else "The affected item"
        plan_areas = [
            f"{subject_cap} and authorization" if subject_has_status_word else f"{subject_cap} status and authorization",
            f"{subject_cap} governing procedure and control requirements",
            f"Downstream impact of the {subject_bare or subject} condition" if not subject_has_status_word
            else f"Downstream impact of the {subject}",
        ]
        is_operating_range_deviation = bool(re.search(
            r"\b(?:operated|used|run|performed)\s+outside\b|\b(?:validated|operating)\s+(?:range|limit|parameters?)\b",
            f"{finding_text} {resolved.condition or ''}",
            re.IGNORECASE,
        ))
        is_notification_or_dispatch_failure = bool(re.search(
            r"\b(?:notification|dispatch|alert|email|message)\b.*?\b(?:failed|not\s+sent|failure|not\s+dispatched|missing)\b|"
            r"\b(?:automated\s+notification|notification\s+system)\b|"
            # Generalized: any "<system/service> failed to <verb> <object>"
            # workflow-failure shape (e.g. "the document-control system
            # failed to distribute the revised SOP") is the SAME
            # trigger->queue->process->recipient->delivery investigation
            # shape as a notification/dispatch failure -- just different
            # domain vocabulary for the same failed-delivery-workflow class
            # of finding.
            r"\b(?:[a-z][a-z0-9\s-]*?\bsystem|[a-z][a-z0-9\s-]*?\bservice)\s+failed\s+to\s+"
            r"(?:distribute|deliver|transmit|dispatch|send|process|forward|route|publish)\b",
            f"{finding_text} {resolved.condition or ''}",
            re.IGNORECASE,
        ))
        is_financial_transaction_finding = bool(re.search(
            r"\b(?:overpaid|overpayment|duplicate\s+(?:supplier\s+|vendor\s+|invoice\s+)?payments?|"
            r"paid\s+twice|double\s+payments?|unauthorized\s+payment|unauthorised\s+payment)\b",
            finding_text, re.IGNORECASE,
        ))
        # Relational/comparison finding ("X did not match Y", "X differed
        # from Y", "X exceeded Y", "X was below Y", "X did not reconcile
        # with Y", "X was inconsistent with Y", "X did not agree with Y",
        # "X conflicted with Y") -- same shape recognized generically by
        # extract_semantic_subject's comparison block, re-checked here since
        # this function derives its own hypotheses/questions independently
        # from raw finding_text.
        is_comparison_mismatch_finding = bool(re.search(
            r"\b(?:did\s+not\s+match|didn't\s+match|does\s+not\s+match|differed\s+from|differs\s+from|"
            r"was\s+inconsistent\s+with|were\s+inconsistent\s+with|is\s+inconsistent\s+with|"
            r"exceeded|exceeds|was\s+below|were\s+below|is\s+below|"
            r"did\s+not\s+reconcile\s+with|does\s+not\s+reconcile\s+with|"
            r"did\s+not\s+agree\s+with|does\s+not\s+agree\s+with|"
            r"conflicted\s+with|conflicts\s+with)\b",
            finding_text, re.IGNORECASE,
        ))
        # Comparison SUBTYPE routing (Section 4/5): a PARAMETER_MISMATCH
        # (recorded value vs an approved parameter/specification/setting)
        # needs an entirely different investigation framework than a
        # CALCULATION_MISMATCH -- never route it through the
        # calculation-specific tree just because both are comparison_type
        # MISMATCH.
        is_parameter_mismatch_finding = (
            is_comparison_mismatch_finding and resolved.comparison_subtype == "PARAMETER_MISMATCH"
        )
        is_calculation_shaped_comparison_finding = (
            is_comparison_mismatch_finding and resolved.comparison_subtype != "PARAMETER_MISMATCH"
        )

        _control_proof_pattern = re.compile(
            r"\b(?:interlock|block|guard|safety|verification)\b.*?\b(?:disabled|bypassed|defeated|deactivated|switched off)\b|"
            r"\b(?:audit\s+logs?|server\s+logs?)\s+(?:establish|show|confirm)\b",
            re.IGNORECASE,
        )
        has_service_crash = bool(re.search(r"\b(?:server|service|message queue|queue)\b.*?\b(?:crashed|failure|outage|down)\b", text_low))
        # Generalized transitive technical-mechanism shape: "<system> failed
        # to <verb> <object>" (e.g. "the document-control system failed to
        # distribute the revised SOP") is the SAME class of verified
        # technical mechanism as an intransitive "service crashed" -- just a
        # different grammatical shape (a failed ACTION rather than a bare
        # state change). Detected generically by verb/object, not by any
        # domain-specific vocabulary (distribution/notification/etc.), so
        # this covers any future finding phrased this way, not only this one.
        _transitive_failure_match = re.search(
            r"\b(?P<system>[a-z][a-z0-9\s-]*?\bsystem|[a-z][a-z0-9\s-]*?\bservice)\s+failed\s+to\s+"
            r"(?P<verb>[a-z]+)\s+(?:the\s+)?(?P<obj>[a-z0-9\s-]+?)(?:\s+to\s+.+)?\s*\.?$",
            finding_text, re.IGNORECASE,
        )
        has_direct_control_proof = bool(_control_proof_pattern.search(finding_text))
        if has_service_crash:
            h1 = CandidateHypothesis(
                id="H1",
                name="NOTIFICATION_SERVICE_OUTAGE",
                statement=f"The notification message queue service crashed, preventing delivery of {subject}.",
                status="SUPPORTED",
                evidence_needed="Server crash error logs and message queue monitoring records",
                confirms_if="Server error logs confirm queue crash during dispatch window",
                refutes_if="Server logs show message queue was operational throughout",
                discrimination_evidence="H1 is established by server crash logs.",
                supporting_evidence=fact_claims,
                evidence_strength="VERIFIED",
                relevance_rank="HIGH",
                causal_role="PRIMARY_CAUSE",
            )
            hypotheses.append(h1)
        elif _transitive_failure_match:
            _sys_name = _clean_subject(_transitive_failure_match.group("system"))
            _verb = _transitive_failure_match.group("verb").lower()
            _obj = _clean_subject(_transitive_failure_match.group("obj"))
            h1 = CandidateHypothesis(
                id="H1",
                name="TECHNICAL_WORKFLOW_FAILURE",
                statement=(
                    f"The {_sys_name}'s {_verb} workflow failed, preventing {_obj} from completing "
                    "as required."
                ),
                status="SUPPORTED",
                evidence_needed=f"{_sys_name.capitalize()} workflow/audit logs and event-processing records covering the affected period",
                confirms_if=f"System logs confirm the {_sys_name} workflow failed during processing",
                refutes_if=f"System logs confirm the {_sys_name} workflow completed successfully",
                discrimination_evidence=f"H1 is established by {_sys_name} workflow/audit logs.",
                supporting_evidence=fact_claims,
                evidence_strength="VERIFIED",
                relevance_rank="HIGH",
                causal_role="PRIMARY_CAUSE",
            )
            hypotheses.append(h1)
        elif has_direct_control_proof:
            _citing_claims = [c for c in fact_claims if _control_proof_pattern.search(c)] or fact_claims
            h1 = CandidateHypothesis(
                id="H1",
                name="CONTROL_OR_INTERLOCK_DISABLED",
                statement=f"The operating control or validation rule for {subject} was disabled prior to operation, permitting the deviation.",
                status="SUPPORTED",
                evidence_needed="SCADA/system audit trail records",
                confirms_if="System audit trail records confirm the interlock or safety control was disabled",
                refutes_if="System audit trail records confirm all interlocks remained active",
                discrimination_evidence="Distinguishes active control bypass from operational parameter drift",
                supporting_evidence=_citing_claims,
                evidence_strength="VERIFIED" if _citing_claims else "NONE",
                relevance_rank="HIGH",
            )
            hypotheses.append(h1)
        elif is_parameter_mismatch_finding:
            # A recorded value differing from an APPROVED PARAMETER/
            # specification/setting (Section 3/4/15) is a distinct
            # investigation shape from a calculation mismatch -- there is
            # no "formula" to re-derive here. Each hypothesis maps 1:1 to
            # the investigation step that tests it (resolves_investigation),
            # per Section 15's hypothesis->investigation-path requirement.
            hypotheses.extend([
                CandidateHypothesis(
                    id="H1", name="DATA_ENTRY_TRANSCRIPTION_ERROR",
                    statement=f"The recorded value for {subject} was manually entered or transcribed differently from the approved parameter.",
                    status="POSSIBLE", evidence_strength="NONE", causal_role="CONTRIBUTING_CAUSE",
                    evidence_needed="Record audit trail showing how/when the value was entered",
                    confirms_if="The audit trail shows manual entry/transcription that does not match the approved parameter",
                    refutes_if="The audit trail shows the recorded value was auto-populated from a validated source",
                    discrimination_evidence="H1 is distinguished by the record's own entry-point audit trail.",
                    relevance_rank="MEDIUM", supporting_evidence=fact_claims[:1] if fact_claims else [],
                    resolves_investigation="P2",
                ),
                CandidateHypothesis(
                    id="H2", name="INCORRECT_PARAMETER_REVISION",
                    statement=f"The parameter/revision applied to {subject} was not the one approved and applicable at the time.",
                    status="POSSIBLE", evidence_strength="NONE", causal_role="CONTRIBUTING_CAUSE",
                    evidence_needed="Approved parameter/SOP revision history and the revision actually applicable",
                    confirms_if="The revision used does not match the one applicable at the relevant time",
                    refutes_if="The correct, currently-applicable revision was used",
                    discrimination_evidence="H2 is distinguished by comparing the applicable parameter/revision history.",
                    relevance_rank="MEDIUM", supporting_evidence=fact_claims[:1] if fact_claims else [],
                    resolves_investigation="P3",
                ),
                CandidateHypothesis(
                    id="H3", name="ACTUAL_PROCESS_OPERATED_AT_INCORRECT_PARAMETER",
                    statement=f"The actual process for {subject} was operated at a value that did not match either the recorded value or the approved parameter.",
                    status="POSSIBLE", evidence_strength="NONE", causal_role="CONTRIBUTING_CAUSE",
                    evidence_needed="Equipment/process historian, PLC/SCADA logs, and batch execution records",
                    confirms_if="Process execution records show operation at a value inconsistent with the recorded and/or approved value",
                    refutes_if="Process execution records confirm operation at the approved parameter",
                    discrimination_evidence="H3 is distinguished by independent equipment/process execution records.",
                    relevance_rank="MEDIUM", supporting_evidence=fact_claims[:1] if fact_claims else [],
                    resolves_investigation="P4",
                ),
                CandidateHypothesis(
                    id="H4", name="REVIEW_CONTROL_GAP",
                    statement=f"The discrepancy in {subject} existed but was not detected during the required review before disposition/release.",
                    status="POSSIBLE", evidence_strength="NONE", causal_role="DETECTION_FAILURE",
                    evidence_needed="Reviewer sign-off, review checklist, and disposition/release records",
                    confirms_if="Review records show the discrepancy was present but not flagged before disposition",
                    refutes_if="Review records show the discrepancy was identified and addressed before disposition",
                    discrimination_evidence="H4 is distinguished by the review/disposition record trail.",
                    relevance_rank="LOW", supporting_evidence=fact_claims[:1] if fact_claims else [],
                    resolves_investigation="P5",
                ),
            ])
        elif is_calculation_shaped_comparison_finding:
            # A verified DISCREPANCY between two values (Section 6/12) is
            # not, by itself, evidence of any ONE causal mechanism -- it is
            # equally consistent with a calculation error, a transcription/
            # data-entry error, an error in the underlying source entries,
            # a wrong calculation formula/version, or a review-control gap
            # that failed to catch it. Generate all as competing POSSIBLE
            # hypotheses (never auto-select one) with provenance to the
            # comparison claim itself, never SUPPORTED/ESTABLISHED status.
            hypotheses.extend([
                CandidateHypothesis(
                    id="H1", name="CALCULATION_FORMULA_ERROR",
                    statement=f"The calculated value for {subject} was generated using an incorrect formula, parameter, or calculation basis.",
                    status="POSSIBLE", evidence_strength="NONE", causal_role="PRIMARY_CAUSE",
                    evidence_needed="Approved calculation formula/worksheet and calculation audit trail",
                    confirms_if="The calculation worksheet/system does not reproduce the reported result using the approved formula",
                    refutes_if="The calculation correctly reproduces the reported result using the approved formula",
                    discrimination_evidence="H1 is distinguished by re-deriving the calculated value from the approved formula.",
                    relevance_rank="MEDIUM", supporting_evidence=fact_claims[:1] if fact_claims else [],
                ),
                CandidateHypothesis(
                    id="H2", name="TRANSCRIPTION_DATA_ENTRY_ERROR",
                    statement=f"The recorded value for {subject} was manually entered differently from the calculated/source result.",
                    status="POSSIBLE", evidence_strength="NONE", causal_role="CONTRIBUTING_CAUSE",
                    evidence_needed="Record audit trail showing how/when the recorded value was entered",
                    confirms_if="The audit trail shows manual entry that does not match the calculated/source value",
                    refutes_if="The audit trail shows the recorded value was auto-populated from the calculation",
                    discrimination_evidence="H2 is distinguished by the record's own entry-point audit trail.",
                    relevance_rank="MEDIUM", supporting_evidence=fact_claims[:1] if fact_claims else [],
                ),
                CandidateHypothesis(
                    id="H3", name="SOURCE_ENTRY_DISCREPANCY",
                    statement=f"One or more of the individual entries underlying the calculation for {subject} were incorrect or incomplete.",
                    status="POSSIBLE", evidence_strength="NONE", causal_role="CONTRIBUTING_CAUSE",
                    evidence_needed="The individual source entries/records the calculation was based on",
                    confirms_if="One or more source entries are shown to be incorrect or incomplete",
                    refutes_if="All source entries are confirmed correct and complete",
                    discrimination_evidence="H3 is distinguished by independently verifying each source entry.",
                    relevance_rank="MEDIUM", supporting_evidence=fact_claims[:1] if fact_claims else [],
                ),
                CandidateHypothesis(
                    id="H4", name="FORMULA_VERSION_MISMATCH",
                    statement=f"The calculation for {subject} used a different approved formula, calculation version, or revision than the one applicable at the time.",
                    status="POSSIBLE", evidence_strength="NONE", causal_role="CONTRIBUTING_CAUSE",
                    evidence_needed="Approved formula/SOP revision history and the version actually used",
                    confirms_if="The version/revision used does not match the one applicable at the relevant time",
                    refutes_if="The correct, currently-applicable version/revision was used",
                    discrimination_evidence="H4 is distinguished by comparing the formula version used against the applicable revision history.",
                    relevance_rank="LOW", supporting_evidence=fact_claims[:1] if fact_claims else [],
                ),
                CandidateHypothesis(
                    id="H5", name="REVIEW_CONTROL_GAP",
                    statement=f"The discrepancy in {subject} existed but was not detected during the required review before disposition/release.",
                    status="POSSIBLE", evidence_strength="NONE", causal_role="DETECTION_FAILURE",
                    evidence_needed="Reviewer sign-off, review checklist, and disposition/release records",
                    confirms_if="Review records show the discrepancy was present but not flagged before disposition",
                    refutes_if="Review records show the discrepancy was identified and addressed before disposition",
                    discrimination_evidence="H5 is distinguished by the review/disposition record trail.",
                    relevance_rank="LOW", supporting_evidence=fact_claims[:1] if fact_claims else [],
                ),
            ])

        if is_operating_range_deviation:
            plan_areas = [
                f"{subject_cap} operating range validation and limits",
                f"{subject_cap} operational records and actual operating conditions",
                f"{subject_cap} range controls and operational mechanism",
            ]
            questions.extend([
                InvestigationQuestion(
                    question_id="Q_OPERATING_SPECIFICATION",
                    id="Q_OPERATING_SPECIFICATION",
                    question=f"What approved validation or qualification record defines the permitted operating range for {subject}?",
                    purpose="Establish the approved specification and validated range requirements before assessing operation",
                    objective="Establish the approved specification and validated range requirements before assessing operation",
                    evidence=f"Approved validation protocol/report and qualification records for {subject}",
                    evidence_required=f"Approved validation protocol/report and qualification records for {subject}",
                    priority="P3",
                    target_proposition_id="P_SPEC",
                ),
                InvestigationQuestion(
                    question_id="Q_OPERATING_RECORDS",
                    id="Q_OPERATING_RECORDS",
                    question=f"What objective operating records establish the actual operating condition of {subject} during the affected period?",
                    purpose="Establish the objective operating parameter records, not merely what the finding text states",
                    objective="Establish the objective operating parameter records, not merely what the finding text states",
                    evidence=f"Equipment operating logs, SCADA/historian records, and batch execution records for {subject}",
                    evidence_required=f"Equipment operating logs, SCADA/historian records, and batch execution records for {subject}",
                    priority="P2",
                    target_proposition_id="P_ACTUAL_OPERATING_RECORDS",
                ),
                InvestigationQuestion(
                    question_id="Q_APPROVED_EXCEPTION",
                    id="Q_APPROVED_EXCEPTION",
                    question=f"Was any approved exception, deviation, extension, or authorization applicable to operation of {subject} outside the validated range?",
                    purpose="Determine whether an authorized departure from standard operating limits existed",
                    objective="Determine whether an authorized departure from standard operating limits existed",
                    evidence=f"Approved deviation, waiver, or planned exception records for {subject}",
                    evidence_required=f"Approved deviation, waiver, or planned exception records for {subject}",
                    priority="P3",
                    target_proposition_id="P_EXCEPTION",
                ),
                InvestigationQuestion(
                    question_id="Q_CONTROL_INTERLOCK_FUNCTION",
                    id="Q_CONTROL_INTERLOCK_FUNCTION",
                    question=f"Do control system audit trails establish whether the required interlock or alarm for {subject} operated as intended?",
                    purpose="Identify the active operational interlock, alarm, or procedural control point",
                    objective="Identify the active operational interlock, alarm, or procedural control point",
                    evidence=f"Control system logs, interlock verification records, and alarm audit trails for {subject}",
                    evidence_required=f"Control system logs, interlock verification records, and alarm audit trails for {subject}",
                    priority="P3",
                    target_proposition_id="P_CONTROL",
                ),
                InvestigationQuestion(
                    question_id="Q_OPERATIONAL_MECHANISM",
                    id="Q_OPERATIONAL_MECHANISM",
                    question=f"What objective evidence establishes the mechanism that allowed {subject} to be operated outside the validated range?",
                    purpose="Identify the specific operational or mechanical mechanism if operation outside limits occurred",
                    objective="Identify the specific operational or mechanical mechanism if operation outside limits occurred",
                    evidence=f"Parameter trends, operator run logs, and control failure records for {subject}",
                    evidence_required=f"Parameter trends, operator run logs, and control failure records for {subject}",
                    priority="P5",
                    target_proposition_id="P_MECHANISM",
                    depends_on="Q_OPERATING_RECORDS",
                    activation_condition="If operation outside validated range is confirmed",
                    status="CONDITIONAL",
                ),
            ])
        elif is_notification_or_dispatch_failure:
            # Generic workflow-term substitution: this decision tree
            # (trigger -> queue -> processing -> recipient resolution ->
            # delivery) applies equally to a notification/email failure and
            # a document-control DISTRIBUTION failure -- only the vocabulary
            # differs. Reflects whichever term the finding actually uses
            # instead of hardcoding "notification" for every domain.
            _workflow_term = "notification" if re.search(r"\bnotification\b", finding_text, re.IGNORECASE) else "distribution"
            plan_areas = [
                f"{_workflow_term.capitalize()} workflow and event-processing verification",
                "Recipient resolution and delivery verification",
                "Technical and configuration failure analysis",
            ]
            questions.extend([
                InvestigationQuestion(
                    question_id="Q_EVENT_TRIGGER",
                    id="Q_EVENT_TRIGGER",
                    question=f"Did the approved revision or release event generate the expected {_workflow_term} trigger for {subject}?",
                    purpose="Verify event generation and trigger initiation upon procedure release",
                    objective="Verify event generation and trigger initiation upon procedure release",
                    evidence="System event audit trail and revision dispatch triggers",
                    evidence_required="System event audit trail and revision dispatch triggers",
                    priority="P1",
                    target_proposition_id="P_TRIGGER",
                    status="ACTIVE",
                    possible_outcomes=[
                        "Event trigger generated successfully → verify queue creation and processing.",
                        "No trigger generated → investigate revision release workflow configuration.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_JOB_PROCESSING",
                    id="Q_JOB_PROCESSING",
                    question=f"Was a {_workflow_term} dispatch job or queue entry created and processed for {subject}?",
                    purpose="Verify queue creation and processing execution",
                    objective="Verify queue creation and processing execution",
                    evidence=f"{_workflow_term.capitalize()} service queue logs, worker task history, and processing status records",
                    evidence_required=f"{_workflow_term.capitalize()} service queue logs, worker task history, and processing status records",
                    priority="P2",
                    target_proposition_id="P_QUEUE",
                    depends_on="Q_EVENT_TRIGGER",
                    activation_condition="If event generation is confirmed",
                    status="CONDITIONAL",
                    possible_outcomes=[
                        "Job processed successfully → check recipient resolution and transmission.",
                        "Job failed or remained queued → investigate service error logs.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_SERVICE_ERRORS",
                    id="Q_SERVICE_ERRORS",
                    question=f"Did the {_workflow_term} service, transfer agent, or integration endpoint return any error or rejection records?",
                    purpose="Identify technical transmission errors or failure responses",
                    objective="Identify technical transmission errors or failure responses",
                    evidence=f"{_workflow_term.capitalize()} server error logs, delivery failure reports, and API integration diagnostics",
                    evidence_required=f"{_workflow_term.capitalize()} server error logs, delivery failure reports, and API integration diagnostics",
                    priority="P3",
                    target_proposition_id="P_ERRORS",
                    depends_on="Q_JOB_PROCESSING",
                    activation_condition="If dispatch job processing was attempted",
                    status="CONDITIONAL",
                    possible_outcomes=[
                        "Error logs confirm technical service outage or API failure.",
                        "No service errors logged → check recipient address and filtering rules.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_RECIPIENT_RESOLUTION",
                    id="Q_RECIPIENT_RESOLUTION",
                    question="Were recipient distribution lists, user account mappings, and delivery addresses resolved accurately?",
                    purpose="Verify recipient resolution and account mapping integrity",
                    objective="Verify recipient resolution and account mapping integrity",
                    evidence="User directory mappings, distribution list membership records, and recipient configuration logs",
                    evidence_required="User directory mappings, distribution list membership records, and recipient configuration logs",
                    priority="P3",
                    target_proposition_id="P_RECIPIENTS",
                    depends_on="Q_JOB_PROCESSING",
                    activation_condition="If dispatch job commenced",
                    status="CONDITIONAL",
                    possible_outcomes=[
                        "Recipient mappings resolved accurately.",
                        "Misconfiguration or missing user mapping identified.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_DOWNSTREAM_OPERATIONS",
                    id="Q_DOWNSTREAM_OPERATIONS",
                    question=f"Did affected personnel perform operational tasks prior to receiving or acknowledging {subject}?",
                    purpose="Determine whether operational activity occurred prior to revision awareness",
                    objective="Determine whether operational activity occurred prior to revision awareness",
                    evidence=f"Operational activity records, batch records, or system execution logs relative to {subject}",
                    evidence_required=f"Operational activity records, batch records, or system execution logs relative to {subject}",
                    priority="P4",
                    target_proposition_id="P_OPERATIONS",
                    depends_on="Q_SERVICE_ERRORS",
                    activation_condition="If notification failure is confirmed",
                    status="CONDITIONAL",
                    possible_outcomes=[
                        "Operational activity occurred prior to awareness → assess scope and compliance impact.",
                        "No operational activity occurred.",
                    ],
                ),
            ])
            evidence_items.extend([
                "System event audit trail and revision dispatch triggers",
                "Notification service queue logs and worker processing records",
                "Server error logs, delivery failure reports, and API integration diagnostics",
                "User directory mappings and distribution list membership records",
                "Batch production records and operational activity logs",
            ])
        elif is_financial_transaction_finding:
            plan_areas = [
                "Payment authorization and approval control verification",
                "Invoice/purchase-order/payment reconciliation",
                "Recovery and outstanding-balance verification",
            ]
            questions.extend([
                InvestigationQuestion(
                    question_id="Q_AMOUNT_MISMATCH",
                    id="Q_AMOUNT_MISMATCH",
                    question=f"Do the payment, invoice, purchase order, and supplier records establish why the amount paid for {subject} exceeded (or duplicated) the authorized amount?",
                    purpose="Determine whether the discrepancy originates in the invoice, PO, or payment instruction itself",
                    objective="Determine whether the discrepancy originates in the invoice, PO, or payment instruction itself",
                    evidence="Invoice, purchase order, and payment instruction records",
                    evidence_required="Invoice, purchase order, and payment instruction records",
                    priority="P1",
                    target_proposition_id="P_AMOUNT_MISMATCH",
                    status="ACTIVE",
                    possible_outcomes=[
                        "Invoice/PO/payment instruction amounts agree → investigate approval/authorization control.",
                        "A discrepancy exists between invoice, PO, or instruction → investigate data-entry/master-data source.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_AUTHORIZATION_CONTROL",
                    id="Q_AUTHORIZATION_CONTROL",
                    question="What payment authorization and approval controls were applicable to this transaction, and were they applied?",
                    purpose="Determine whether the applicable approval control operated as designed",
                    objective="Determine whether the applicable approval control operated as designed",
                    evidence="Payment approval workflow configuration and approval audit trail",
                    evidence_required="Payment approval workflow configuration and approval audit trail",
                    priority="P1",
                    target_proposition_id="P_AUTHORIZATION",
                    depends_on="Q_AMOUNT_MISMATCH",
                    status="CONDITIONAL",
                    possible_outcomes=[
                        "Approval control was applied and passed → investigate whether an override/exception was used.",
                        "Approval control was bypassed or not applied → investigate control-bypass mechanism.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_INDEPENDENT_VERIFICATION",
                    id="Q_INDEPENDENT_VERIFICATION",
                    question="Was the invoice amount, quantity, price, tax, or payment instruction independently verified before payment release?",
                    purpose="Determine whether a pre-payment verification step existed and was performed",
                    objective="Determine whether a pre-payment verification step existed and was performed",
                    evidence="Three-way match / invoice verification records",
                    evidence_required="Three-way match / invoice verification records",
                    priority="P2",
                    target_proposition_id="P_VERIFICATION",
                    status="ACTIVE",
                    possible_outcomes=[
                        "Verification was performed and passed → investigate why the discrepancy still occurred.",
                        "Verification was not performed or was skipped → investigate verification-control gap.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_SYSTEM_FLAG",
                    id="Q_SYSTEM_FLAG",
                    question="Did the accounts-payable system or approval workflow identify, flag, or block the transaction at any point?",
                    purpose="Determine whether an automated detection control existed and whether it fired",
                    objective="Determine whether an automated detection control existed and whether it fired",
                    evidence="ERP/accounts-payable system configuration and alert/exception logs",
                    evidence_required="ERP/accounts-payable system configuration and alert/exception logs",
                    priority="P2",
                    target_proposition_id="P_SYSTEM_DETECTION",
                    status="ACTIVE",
                    possible_outcomes=[
                        "System flagged the transaction but it was overridden → investigate override authorization.",
                        "System did not flag the transaction → investigate detection-control configuration.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_RECOVERY_VERIFICATION",
                    id="Q_RECOVERY_VERIFICATION",
                    question="What records establish the amount recovered to date, and what amount remains outstanding after reconciliation?",
                    purpose="Verify the recovered and outstanding amounts against independent records, not only the finding's own statement",
                    objective="Verify the recovered and outstanding amounts against independent records, not only the finding's own statement",
                    evidence="Supplier credit note, bank/payment records, and accounts-payable reconciliation records",
                    evidence_required="Supplier credit note, bank/payment records, and accounts-payable reconciliation records",
                    priority="P2",
                    target_proposition_id="P_RECOVERY",
                    status="ACTIVE",
                    possible_outcomes=[
                        "Reconciliation confirms the stated recovered/outstanding amounts.",
                        "Reconciliation identifies a different recovered/outstanding amount → update the financial exposure figures.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_OUTSTANDING_TREATMENT",
                    id="Q_OUTSTANDING_TREATMENT",
                    question="Does the remaining outstanding balance represent a confirmed actual loss, a receivable, a pending credit, or a recoverable balance?",
                    purpose="Determine the final accounting treatment of the outstanding balance before it is classified as a loss",
                    objective="Determine the final accounting treatment of the outstanding balance before it is classified as a loss",
                    evidence="Accounts-payable/receivable ledger and collection/write-off records",
                    evidence_required="Accounts-payable/receivable ledger and collection/write-off records",
                    priority="P3",
                    target_proposition_id="P_OUTSTANDING_TREATMENT",
                    depends_on="Q_RECOVERY_VERIFICATION",
                    status="CONDITIONAL",
                    possible_outcomes=[
                        "Confirmed recoverable/receivable → not an actual loss.",
                        "Confirmed uncollectible/written off → actual loss established.",
                        "Still under review → actual loss remains NOT_ESTABLISHED.",
                    ],
                ),
            ])
            evidence_items.extend([
                "Invoice, purchase order, and payment instruction records",
                "Payment approval workflow configuration and approval audit trail",
                "Three-way match / invoice verification records",
                "Supplier credit note and accounts-payable reconciliation records",
            ])
        elif is_parameter_mismatch_finding:
            # PARAMETER_MISMATCH decision tree (Section 5): distinct from
            # the calculation tree below -- confirm discrepancy -> source of
            # the recorded value -> applicable revision -> actual process
            # execution -> detection/review -> impact. Investigation AREA
            # names describe the DOMAIN being investigated, never restate
            # the observation itself (Section 8).
            plan_areas = [
                "Parameter source and entry control",
                "Applicable approved parameter and revision",
                "Actual process execution",
                "Review and detection control",
                "Batch impact assessment",
            ]
            questions.extend([
                InvestigationQuestion(
                    question_id="P1", id="P1",
                    question=f"What value was recorded for {subject}, and what approved parameter applied at the relevant time?",
                    purpose="Confirm the discrepancy before investigating cause",
                    objective="Confirm the discrepancy before investigating cause",
                    evidence=f"Batch record, approved process parameter, applicable SOP/master batch record, and revision history for {subject}",
                    evidence_required=f"Batch record, approved process parameter, applicable SOP/master batch record, and revision history for {subject}",
                    priority="P1",
                    target_proposition_id="P_DISCREPANCY",
                    status="ACTIVE",
                    category="OBSERVATION_VERIFICATION",
                    decision_rule=(
                        "IF values reconcile → the finding may require re-evaluation. "
                        "IF values differ → proceed to P2."
                    ),
                    next_question_if_false="CLOSE_OR_REASSESS",
                    next_question_if_true="P2",
                    possible_outcomes=[
                        "Values reconcile → the finding may require re-evaluation.",
                        "Values differ → continue to source-of-entry investigation.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="P2", id="P2",
                    question=f"How was the recorded value for {subject} entered or generated?",
                    purpose="Establish the source of the recorded value",
                    objective="Establish the source of the recorded value",
                    evidence="Record audit trail, electronic system logs, and manual entry records",
                    evidence_required="Record audit trail, electronic system logs, and manual entry records",
                    priority="P2",
                    target_proposition_id="P_ENTRY_SOURCE",
                    depends_on="P1",
                    activation_condition="If the discrepancy is confirmed",
                    status="CONDITIONAL",
                    category="MECHANISM_INVESTIGATION",
                    hypothesis_tested="H1",
                    target_hypothesis_ids=["H1"],
                    decision_rule=(
                        "IF manual entry → investigate transcription/data-entry controls (H1). "
                        "IF automatic system value → investigate system configuration/control logic. "
                        "IF transcribed from another record → compare source record. "
                        "IF unknown → obtain audit trail/source evidence."
                    ),
                    next_question_if_true="P3",
                    possible_outcomes=[
                        "Manual entry → investigate transcription/data-entry controls (H1).",
                        "Automatic system value → investigate system configuration/control logic.",
                        "Transcribed from another record → compare against the source record.",
                        "Unknown → obtain audit trail/source evidence.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="P3", id="P3",
                    question=f"Was the approved parameter/revision applicable to this batch used for {subject}?",
                    purpose="Verify the applicable revision",
                    objective="Verify the applicable revision",
                    evidence="Approved parameter/SOP revision history and the revision applied",
                    evidence_required="Approved parameter/SOP revision history and the revision applied",
                    priority="P3",
                    target_proposition_id="P_REVISION",
                    status="ACTIVE",
                    category="MECHANISM_INVESTIGATION",
                    hypothesis_tested="H2",
                    target_hypothesis_ids=["H2"],
                    decision_rule=(
                        "IF correct revision → continue to P4. "
                        "IF incorrect/outdated revision → investigate document/change-control mechanism (H2), "
                        "then continue to P4."
                    ),
                    next_question_if_true="P4",
                    next_question_if_false="P4",
                    possible_outcomes=[
                        "Correct revision applied → eliminate this hypothesis (H2).",
                        "Incorrect/outdated revision applied → investigate document/change-control mechanism.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="P4", id="P4",
                    question=f"Do equipment/process records show that the batch was actually operated at the recorded value, the approved parameter, or another value for {subject}?",
                    purpose="Verify actual process execution",
                    objective="Verify actual process execution",
                    evidence="Equipment historian, PLC/SCADA logs, batch execution records, calibration records, and electronic batch record",
                    evidence_required="Equipment historian, PLC/SCADA logs, batch execution records, calibration records, and electronic batch record",
                    priority="P2",
                    target_proposition_id="P_ACTUAL_EXECUTION",
                    status="ACTIVE",
                    category="MECHANISM_INVESTIGATION",
                    hypothesis_tested="H3",
                    target_hypothesis_ids=["H3"],
                    decision_rule="Actual process execution records establish what value the process was actually operated at, independent of what was recorded.",
                    next_question_if_true="P5",
                    possible_outcomes=[
                        "Process operated at the approved parameter → the discrepancy may be a recording issue.",
                        "Process operated at a different value → investigate the execution mechanism (H3).",
                    ],
                ),
                InvestigationQuestion(
                    question_id="P5", id="P5",
                    question="Was the discrepancy detected during the required review before batch disposition/release?",
                    purpose="Verify detection and review control effectiveness",
                    objective="Verify detection and review control effectiveness",
                    evidence="Reviewer sign-off, batch review checklist, deviation records, and release records",
                    evidence_required="Reviewer sign-off, batch review checklist, deviation records, and release records",
                    priority="P3",
                    target_proposition_id="P_REVIEW_CONTROL",
                    status="ACTIVE",
                    category="CONTROL_EFFECTIVENESS",
                    hypothesis_tested="H4",
                    target_hypothesis_ids=["H4"],
                    decision_rule=(
                        "IF detected → determine disposition and response taken, then proceed to P6. "
                        "IF not detected → investigate review-control effectiveness (H4), then proceed to P6."
                    ),
                    next_question_if_true="P6",
                    next_question_if_false="P6",
                    possible_outcomes=[
                        "Detected → determine disposition and response taken.",
                        "Not detected → investigate review-control effectiveness (H4).",
                    ],
                ),
                InvestigationQuestion(
                    question_id="P6", id="P6",
                    question="Did the temperature/parameter deviation affect process performance, product quality, batch disposition, or release decisions?",
                    purpose="Assess downstream impact without asserting that it occurred",
                    objective="Assess downstream impact without asserting that it occurred",
                    evidence="Process performance records, quality assessment, disposition records, and release records",
                    evidence_required="Process performance records, quality assessment, disposition records, and release records",
                    priority="P4",
                    target_proposition_id="P_DOWNSTREAM_IMPACT",
                    status="ACTIVE",
                    category="IMPACT_ASSESSMENT",
                    decision_rule=(
                        "IF no downstream impact identified → close the impact assessment line. "
                        "IF impact identified → quantify and assess scope."
                    ),
                    possible_outcomes=[
                        "No downstream impact identified.",
                        "Impact identified → quantify and assess scope.",
                    ],
                ),
            ])
            evidence_items.extend([
                "Batch record and approved process parameter",
                "Applicable SOP/master batch record and revision history",
                "Record audit trail and manual entry records",
                "Equipment historian, PLC/SCADA logs, and batch execution records",
                "Reviewer sign-off and disposition/release records",
            ])
        elif is_calculation_shaped_comparison_finding:
            # Generalized comparison/mismatch investigation dimensions
            # (Section 7/8): verify both compared values -> verify the
            # calculation/basis -> identify the point of data creation ->
            # identify the transformation/reconciliation step -> verify the
            # applicable control -> check detection/review -> assess
            # downstream impact. Works for any two-value comparison finding
            # (yield, temperature, invoice amount, quantity, batch result,
            # record reconciliation, ...), not hardcoded to any one domain.
            plan_areas = [
                f"{subject_cap} discrepancy verification and independent recalculation",
                f"{subject_cap} data-entry and calculation source-of-truth verification",
                f"{subject_cap} review/detection control effectiveness",
            ]
            questions.extend([
                InvestigationQuestion(
                    question_id="Q_ESTABLISH_DISCREPANCY",
                    id="Q_ESTABLISH_DISCREPANCY",
                    question=f"What were the two compared values for {subject}, and does an independent recalculation confirm the reported discrepancy?",
                    purpose="Quantify and independently verify the reported discrepancy before investigating its cause",
                    objective="Quantify and independently verify the reported discrepancy before investigating its cause",
                    evidence=f"Original record, underlying source entries, and approved calculation/reference basis for {subject}",
                    evidence_required=f"Original record, underlying source entries, and approved calculation/reference basis for {subject}",
                    priority="P1",
                    target_proposition_id="P_DISCREPANCY",
                    status="ACTIVE",
                    category="OBSERVATION_VERIFICATION",
                    decision_rule=(
                        "IF discrepancy is not reproduced → reconcile source records and determine whether the "
                        "finding resulted from a record interpretation/calculation issue. "
                        "IF discrepancy is reproduced → proceed to Q_VERIFY_CALCULATION."
                    ),
                    next_question_if_false="RECONCILE_SOURCE_RECORDS",
                    next_question_if_true="Q_VERIFY_CALCULATION",
                    possible_outcomes=[
                        "Values reconcile on independent recalculation → the finding may reflect a transcription/reporting issue.",
                        "Values do not reconcile → continue to mechanism investigation.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_VERIFY_CALCULATION",
                    id="Q_VERIFY_CALCULATION",
                    question=f"Do the underlying source entries mathematically produce the reported calculated/reference value for {subject}?",
                    purpose="Determine whether the calculation itself is correct",
                    objective="Determine whether the calculation itself is correct",
                    evidence="Raw source entries, approved calculation formula, and calculation worksheet/system audit trail",
                    evidence_required="Raw source entries, approved calculation formula, and calculation worksheet/system audit trail",
                    priority="P1",
                    target_proposition_id="P_CALCULATION",
                    depends_on="Q_ESTABLISH_DISCREPANCY",
                    activation_condition="If the discrepancy is confirmed",
                    status="CONDITIONAL",
                    category="MECHANISM_INVESTIGATION",
                    decision_rule=(
                        "IF calculation differs from source entries → investigate H1/H4 (calculation/formula). "
                        "IF calculation is correct → proceed to Q_DATA_ENTRY_POINT."
                    ),
                    next_question_if_false="Q_DATA_ENTRY_POINT",
                    next_question_if_true="Q_CALCULATION_CONFIGURATION",
                    possible_outcomes=[
                        "Calculation reproduces correctly → the discrepancy is not a calculation error.",
                        "Calculation does not reproduce → investigate the calculation mechanism (H1/H4).",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_DATA_ENTRY_POINT",
                    id="Q_DATA_ENTRY_POINT",
                    question=f"Was the recorded value for {subject} manually entered, automatically calculated, or transcribed from another record?",
                    purpose="Identify the point at which the recorded value entered the record",
                    objective="Identify the point at which the recorded value entered the record",
                    evidence="Record audit trail, electronic system logs, and manual entry records",
                    evidence_required="Record audit trail, electronic system logs, and manual entry records",
                    priority="P2",
                    target_proposition_id="P_ENTRY_POINT",
                    depends_on="Q_ESTABLISH_DISCREPANCY",
                    status="CONDITIONAL",
                    category="MECHANISM_INVESTIGATION",
                    decision_rule=(
                        "IF manual record differs → investigate H2 (transcription/data-entry). "
                        "IF source entry differs → investigate H3 (source-entry discrepancy). "
                        "IF source records and formula are correct but discrepancy remains → investigate the "
                        "system/transformation/reconciliation mechanism."
                    ),
                    next_question_if_true="Q_CALCULATION_CONFIGURATION",
                    possible_outcomes=[
                        "Manual entry → investigate transcription/data-entry mechanism (H2).",
                        "Automatic calculation → investigate system/formula behavior (H1).",
                        "Transcribed from another record → compare against the source record.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_CALCULATION_CONFIGURATION",
                    id="Q_CALCULATION_CONFIGURATION",
                    question=f"Was the approved calculation formula/version applicable to this record used for {subject}?",
                    purpose="Determine whether an incorrect or outdated calculation method contributed to the discrepancy",
                    objective="Determine whether an incorrect or outdated calculation method contributed to the discrepancy",
                    evidence="Approved formula/SOP revision history, system configuration, and calculation version used",
                    evidence_required="Approved formula/SOP revision history, system configuration, and calculation version used",
                    priority="P3",
                    target_proposition_id="P_CONFIGURATION",
                    status="ACTIVE",
                    category="MECHANISM_INVESTIGATION",
                    decision_rule=(
                        "IF correct formula/version used → eliminate H4 and proceed to Q_REVIEW_CONTROL. "
                        "IF incorrect/outdated formula/version used → investigate H4 (formula/version mismatch) "
                        "and configuration/change control."
                    ),
                    next_question_if_true="Q_REVIEW_CONTROL",
                    next_question_if_false="Q_REVIEW_CONTROL",
                    possible_outcomes=[
                        "Correct formula/version used → eliminate this hypothesis (H4).",
                        "Incorrect/outdated formula/version used → investigate configuration/change control.",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_REVIEW_CONTROL",
                    id="Q_REVIEW_CONTROL",
                    question="Was this discrepancy detected during the required review before disposition/release?",
                    purpose="Assess whether the applicable detective control operated effectively",
                    objective="Assess whether the applicable detective control operated effectively",
                    evidence="Reviewer sign-off, review checklist, deviation records, and disposition/release records",
                    evidence_required="Reviewer sign-off, review checklist, deviation records, and disposition/release records",
                    priority="P3",
                    target_proposition_id="P_REVIEW_CONTROL",
                    status="ACTIVE",
                    category="CONTROL_EFFECTIVENESS",
                    decision_rule=(
                        "IF detected → determine disposition and response taken, then proceed to "
                        "Q_DOWNSTREAM_IMPACT. IF not detected → investigate review-control effectiveness (H5) "
                        "and proceed to Q_DOWNSTREAM_IMPACT."
                    ),
                    next_question_if_true="Q_DOWNSTREAM_IMPACT",
                    next_question_if_false="Q_DOWNSTREAM_IMPACT",
                    possible_outcomes=[
                        "Detected → determine disposition and response taken.",
                        "Not detected → investigate review-control effectiveness (H5).",
                    ],
                ),
                InvestigationQuestion(
                    question_id="Q_DOWNSTREAM_IMPACT",
                    id="Q_DOWNSTREAM_IMPACT",
                    question="Did this discrepancy affect disposition, quality assessment, reporting, or other downstream decisions?",
                    purpose="Determine actual impact without assuming that impact occurred",
                    objective="Determine actual impact without assuming that impact occurred",
                    evidence="Disposition records, quality assessment, release records, and downstream reporting",
                    evidence_required="Disposition records, quality assessment, release records, and downstream reporting",
                    priority="P4",
                    target_proposition_id="P_DOWNSTREAM_IMPACT",
                    status="ACTIVE",
                    category="IMPACT_ASSESSMENT",
                    decision_rule=(
                        "IF no downstream impact identified → close the impact assessment line. "
                        "IF impact identified → quantify and assess scope."
                    ),
                    possible_outcomes=[
                        "No downstream impact identified.",
                        "Impact identified → quantify and assess scope.",
                    ],
                ),
            ])
            evidence_items.extend([
                "Original record and underlying source entries",
                "Approved calculation formula/reference basis and worksheet/system audit trail",
                "Record audit trail and manual entry records",
                "Reviewer sign-off and disposition/release records",
            ])
        else:
            # Dynamic uncertainty-driven question builder:
            # 1. If requirement is UNKNOWN -> investigate governing requirement.
            # 2. If requirement is STATED but source doc UNKNOWN -> investigate governing document/procedure.
            # 3. If requirement is already VERIFIED -> do NOT ask if requirement exists; investigate mechanism and authorization.
            req_status = getattr(canonical_state, "requirement_status", "UNKNOWN") if canonical_state else "UNKNOWN"
            has_normative_source = bool(re.search(r"\b(?:SOP-[A-Z0-9-]+|Revision\s+\d+|Rev\s+\d+|ISO\s+\d+|contract|agreement)\b", finding_text, re.IGNORECASE))
            
            if not has_normative_source and req_status != "VERIFIED":
                questions.append(
                    InvestigationQuestion(
                        question_id="Q_GOVERNING_REQUIREMENT",
                        id="Q_GOVERNING_REQUIREMENT",
                        question=f"What approved procedure, specification, or control defines the requirement for {subject} during the relevant period?",
                        purpose="Establish the governing source document and execution criteria",
                        objective="Establish the governing source document and execution criteria",
                        evidence=f"Applicable procedure, specification, or control governing {subject}",
                        evidence_required=f"Applicable procedure, specification, or control governing {subject}",
                        priority="P1",
                        target_proposition_id="P_GOV",
                        decision_rule="If governing procedure is located → compare observed status against specified limits; if no procedure exists → investigate requirement definition gap.",
                    )
                )

            questions.extend([
                InvestigationQuestion(
                    question_id="Q_STATUS_HISTORY",
                    id="Q_STATUS_HISTORY",
                    question=f"What objective records establish the operational status and execution logs for {subject_bare} at the time of the finding?",
                    purpose="Establish the objective operational status and execution logs from primary evidence",
                    objective="Establish the objective operational status and execution logs from primary evidence",
                    evidence=f"Status, maintenance, calibration, or execution logs for {subject}",
                    evidence_required=f"Status, maintenance, calibration, or execution logs for {subject}",
                    priority="P2",
                    target_proposition_id="P_STATUS",
                    decision_rule="If execution logs confirm nonconformity → proceed to mechanism investigation; if logs show compliant execution → investigate record reconciliation.",
                ),
                InvestigationQuestion(
                    question_id="Q_AUTHORIZED_EXCEPTION",
                    id="Q_AUTHORIZED_EXCEPTION",
                    question=f"Did any approved waiver, planned deviation, or authorized exception apply to {subject} during the affected period?",
                    purpose="Determine whether an authorized departure or change control was in effect",
                    objective="Determine whether an authorized departure or change control was in effect",
                    evidence=f"Approved waiver, deviation, or change control records for {subject}",
                    evidence_required=f"Approved waiver, deviation, or change control records for {subject}",
                    priority="P3",
                    target_proposition_id="P_EXCEPTION",
                    decision_rule="If authorized waiver exists → evaluate scope and validity of waiver; if no waiver exists → proceed to control failure analysis.",
                ),
                InvestigationQuestion(
                    question_id="Q_PROCESS_RESPONSIBILITY",
                    id="Q_PROCESS_RESPONSIBILITY",
                    question=f"What role, automated control, or verification step was responsible for preventing or detecting this condition in {subject}?",
                    purpose="Identify the specific preventive or detective control point",
                    objective="Identify the specific preventive or detective control point",
                    evidence=f"Process responsibility matrix and control assignment records for {subject}",
                    evidence_required=f"Process responsibility matrix and control assignment records for {subject}",
                    priority="P3",
                    target_proposition_id="P_RESPONSIBILITY",
                    decision_rule="If control point is identified → evaluate whether control design or control execution failed.",
                ),
                InvestigationQuestion(
                    question_id="Q_DOWNSTREAM_DEPENDENCIES",
                    id="Q_DOWNSTREAM_DEPENDENCIES",
                    question=f"What downstream batches, systems, or operational outputs were potentially affected by {subject}?",
                    purpose="Determine actual downstream impact without presupposing widespread loss",
                    objective="Determine actual downstream impact without presupposing widespread loss",
                    evidence=f"Downstream traceability, release records, and batch/system logs for {subject}",
                    evidence_required=f"Downstream traceability, release records, and batch/system logs for {subject}",
                    priority="P4",
                    target_proposition_id="P_SCOPE",
                    decision_rule="If downstream items are identified → quarantine and evaluate; if no downstream dependencies → close impact boundary.",
                ),
            ])
        evidence_items.extend([
            f"Applicable procedure/requirement governing {subject}",
            f"Status/history records for {subject}",
            f"Exception/waiver/deviation records for {subject}",
        ])

    if not hypotheses:
        # Check if verified evidence proves a specific causal mechanism
        has_training_proof = bool(re.search(r"\b(?:lms|training log|training records?)\b.*?\b(?:no\s+operators|not\s+completed|never\s+completed|uncompleted|failed\s+to\s+complete)\b", text_low))
        has_dual_approval_bypass = bool(re.search(r"\b(?:dual[- ]approval|dual[- ]authorization|mandatory|validation control)\b.*?\b(?:disabled|bypassed|deactivated)\b", text_low))
        has_service_crash = bool(re.search(r"\b(?:server|service|message queue|queue)\b.*?\b(?:crashed|failure|outage|down)\b", text_low))
        has_change_mgmt_bypass = bool(re.search(r"\b(?:change[- ]management|sop-eng-\w+)\b.*?\b(?:bypassed|skipped|unvalidated|unconfigured)\b", text_low))
        has_rule_disabled = bool(re.search(r"\b(?:duplicate detection rule|detection rule|validation rule)\b.*?\b(?:disabled|deactivated)\b", text_low))

        is_post_event_disablement = bool(re.search(
            r"\b(?:released|executed|occurred|paid)\b.*?\b(?:disabled|bypassed|deactivated)\s+(?:on|after)\b",
            text_low,
        ))
        if is_post_event_disablement:
            has_dual_approval_bypass = False
            has_rule_disabled = False

        first_cid = [item.claim_id for item in evidence_ledger if hasattr(item, "claim_id") and item.claim_id] or ["C1"]

        if has_training_proof:
            hypotheses.append(CandidateHypothesis(
                id="H1",
                name="TRAINING_COMPLETION_FAILURE",
                statement=f"Required training for {subject} was not completed by operators prior to performing the task.",
                status="SUPPORTED",
                evidence_strength="VERIFIED",
                causal_role="PRIMARY_CAUSE",
                evidence_needed="LMS training records and operator task assignment logs",
                confirms_if="LMS logs confirm training incomplete before task execution",
                refutes_if="LMS logs show valid training completed prior to task execution",
                discrimination_evidence="H1 is established by authenticated LMS training logs.",
                relevance_rank="HIGH",
                supporting_claim_ids=first_cid[:1],
            ))
        elif has_dual_approval_bypass:
            hypotheses.append(CandidateHypothesis(
                id="H1",
                name="MANDATORY_CONTROL_BYPASS",
                statement="Mandatory dual-approval validation control was disabled before payment release.",
                status="SUPPORTED",
                evidence_strength="VERIFIED",
                causal_role="PRIMARY_CAUSE",
                evidence_needed="Audit trail configuration logs for dual-approval control",
                confirms_if="Audit trail logs confirm control was disabled",
                refutes_if="Audit trail logs confirm control remained active",
                discrimination_evidence="H1 is established by audit trail configuration logs.",
                relevance_rank="HIGH",
                supporting_claim_ids=first_cid[:1],
            ))
        elif has_service_crash:
            hypotheses.append(CandidateHypothesis(
                id="H1",
                name="NOTIFICATION_SERVICE_OUTAGE",
                statement=f"Message queue service crashed, preventing dispatch of {subject}.",
                status="SUPPORTED",
                evidence_strength="VERIFIED",
                causal_role="PRIMARY_CAUSE",
                evidence_needed="Server crash error logs and message queue monitoring records",
                confirms_if="Server error logs confirm queue crash during dispatch window",
                refutes_if="Server logs show message queue was operational throughout",
                discrimination_evidence="H1 is established by server crash logs.",
                relevance_rank="HIGH",
                supporting_claim_ids=first_cid[:1],
            ))
        elif re.search(
            r"\b(?:[a-z][a-z0-9\s-]*?\bsystem|[a-z][a-z0-9\s-]*?\bservice)\s+failed\s+to\s+[a-z]+\b",
            finding_text, re.IGNORECASE,
        ):
            _transitive_failure_match2 = re.search(
                r"\b(?P<system>[a-z][a-z0-9\s-]*?\bsystem|[a-z][a-z0-9\s-]*?\bservice)\s+failed\s+to\s+"
                r"(?P<verb>[a-z]+)\s+(?:the\s+)?(?P<obj>[a-z0-9\s-]+?)(?:\s+to\s+.+)?\s*\.?$",
                finding_text, re.IGNORECASE,
            )
            _sys_name2 = _clean_subject(_transitive_failure_match2.group("system"))
            _verb2 = _transitive_failure_match2.group("verb").lower()
            _obj2 = _clean_subject(_transitive_failure_match2.group("obj"))
            hypotheses.append(CandidateHypothesis(
                id="H1",
                name="TECHNICAL_WORKFLOW_FAILURE",
                statement=(
                    f"The {_sys_name2}'s {_verb2} workflow failed, preventing {_obj2} from completing as required."
                ),
                status="SUPPORTED",
                evidence_strength="VERIFIED",
                causal_role="PRIMARY_CAUSE",
                evidence_needed=f"{_sys_name2.capitalize()} workflow/audit logs and event-processing records covering the affected period",
                confirms_if=f"System logs confirm the {_sys_name2} workflow failed during processing",
                refutes_if=f"System logs confirm the {_sys_name2} workflow completed successfully",
                discrimination_evidence=f"H1 is established by {_sys_name2} workflow/audit logs.",
                relevance_rank="HIGH",
                supporting_claim_ids=first_cid[:1],
            ))
        elif has_change_mgmt_bypass:
            hypotheses.append(CandidateHypothesis(
                id="H1",
                name="CHANGE_MANAGEMENT_BYPASS",
                statement="Change-management procedure was bypassed during upgrade, leaving critical controls unconfigured.",
                status="SUPPORTED",
                evidence_strength="VERIFIED",
                causal_role="PRIMARY_CAUSE",
                evidence_needed="Change control records and system configuration logs",
                confirms_if="Change audit trail confirms procedure was bypassed",
                refutes_if="Change records confirm complete validation prior to release",
                discrimination_evidence="H1 is established by change management audit trail.",
                relevance_rank="HIGH",
                supporting_claim_ids=first_cid[:1],
            ))
        elif has_rule_disabled:
            hypotheses.append(CandidateHypothesis(
                id="H1",
                name="AUTOMATED_DETECTION_RULE_DISABLED",
                statement="Automated duplicate detection rule was disabled in the ERP configuration.",
                status="SUPPORTED",
                evidence_strength="VERIFIED",
                causal_role="PRIMARY_CAUSE",
                evidence_needed="ERP rule configuration and audit trail logs",
                confirms_if="ERP configuration logs confirm detection rule was disabled",
                refutes_if="ERP configuration logs show detection rule was active and functional",
                discrimination_evidence="H1 is established by ERP configuration logs.",
                relevance_rank="HIGH",
                supporting_claim_ids=first_cid[:1],
            ))
        elif "duplicate payment" in text_low or "duplicate supplier payment" in text_low or "paid twice" in text_low:
            hypotheses.extend([
                CandidateHypothesis(
                    id="H1",
                    name="ACCOUNTS_PAYABLE_CONTROL_GAP",
                    statement="Accounts payable invoice verification and duplicate detection controls failed to flag the duplicate payment.",
                    status="POSSIBLE",
                    evidence_strength="REPORTED",
                    causal_role="PRIMARY_CAUSE",
                    evidence_needed="Invoice matching logs, approval audit trails, and ERP duplicate-check settings",
                    confirms_if="ERP duplicate validation controls were unconfigured or bypassed",
                    refutes_if="ERP automated controls were active and payment was intentionally released",
                    discrimination_evidence="H1 strengthens if ERP duplicate validation was disabled.",
                    relevance_rank="HIGH",
                    supporting_claim_ids=first_cid[:1],
                ),
                CandidateHypothesis(
                    id="H2",
                    name="MANUAL_INVOICE_PROCESSING_ERROR",
                    statement="Manual invoice entry without mandatory secondary review resulted in duplicate transaction submission.",
                    status="POSSIBLE",
                    evidence_strength="REPORTED",
                    causal_role="CONTRIBUTING_CAUSE",
                    evidence_needed="User entry audit logs and secondary review approval records",
                    confirms_if="Invoice entry logs show manual submission without secondary review",
                    refutes_if="Automated electronic invoice feed was used",
                    discrimination_evidence="H2 strengthens if manual entry occurred without secondary approval.",
                    relevance_rank="MEDIUM",
                    supporting_claim_ids=first_cid[:1],
                ),
            ])
        else:
            # Common-factor lead (Section: COMMON-FACTOR CAUSAL REASONING):
            # none of the direct-evidence branches above fired, but a
            # shared system/process/vendor/control across multiple
            # independently-affected departments/locations is itself a
            # strong investigation lead -- "no hypotheses" would be too
            # conservative here, but a common factor never proves
            # causality, so this is deliberately generated as
            # POSSIBLE/INDICATIVE, never SUPPORTED.
            from app.agent.common_factor import build_common_factor_hypothesis, detect_common_factor
            common_factor = detect_common_factor(evidence_ledger)
            if common_factor.detected:
                hypotheses.append(build_common_factor_hypothesis(common_factor, hyp_id="H1"))

    # Filter out any evidence-state propositions from hypotheses
    from app.agent.causal_guard import is_evidence_state_not_hypothesis
    hypotheses = [h for h in hypotheses if not is_evidence_state_not_hypothesis(h.statement, h.name)]

    # Investigation questions must TEST the common-factor hypothesis, not
    # ask generic "what requirement applied" questions unrelated to the
    # actual lead just identified (Section 6/7: the plan must follow the
    # hypothesis, not ignore it).
    if any(h.name == "SHARED_SYSTEM_COMMON_FACTOR" for h in hypotheses):
        common_hyp = next(h for h in hypotheses if h.name == "SHARED_SYSTEM_COMMON_FACTOR")
        factor_label = common_hyp.evidence_needed.split(" distribution")[0]
        plan_areas = [
            f"{factor_label.capitalize()} distribution and version-control verification",
            "Obsolete-copy withdrawal control effectiveness",
            "Systemic scope — other controlled documents/departments potentially affected",
        ]
        questions = [
            InvestigationQuestion(
                question_id="Q_REVISION_STATUS",
                id="Q_REVISION_STATUS",
                question=f"Was the current approved revision effective and available in the {factor_label} during the affected period?",
                purpose="Verify the approved revision was correctly released before assessing distribution",
                objective="Verify the approved revision was correctly released before assessing distribution",
                evidence=f"Document revision history, approval record, effective date, and document master for {factor_label}",
                evidence_required=f"Document revision history, approval record, effective date, and document master for {factor_label}",
                priority="P1",
                target_proposition_id="P_REVISION_STATUS",
                hypothesis_tested=common_hyp.id,
                status="ACTIVE",
                possible_outcomes=[
                    "Current revision correctly released → proceed to distribution verification.",
                    "Revision was not correctly released → investigate document-release control.",
                ],
            ),
            InvestigationQuestion(
                question_id="Q_DISTRIBUTION_VERIFICATION",
                id="Q_DISTRIBUTION_VERIFICATION",
                question=f"Did the {factor_label} distribute or make the current revision available to all affected departments?",
                purpose="Determine whether the shared system's distribution mechanism reached every affected location",
                objective="Determine whether the shared system's distribution mechanism reached every affected location",
                evidence="Distribution logs, recipient mappings, access records, and notification logs",
                evidence_required="Distribution logs, recipient mappings, access records, and notification logs",
                priority="P1",
                target_proposition_id="P_DISTRIBUTION",
                hypothesis_tested=common_hyp.id,
                depends_on="Q_REVISION_STATUS",
                activation_condition="If the current revision was correctly released",
                status="CONDITIONAL",
                possible_outcomes=[
                    "Distribution successful → investigate local obsolete-copy withdrawal control.",
                    "Distribution failure → investigate the shared system/process failure.",
                    "Distribution evidence unavailable → evidence boundary.",
                ],
            ),
            InvestigationQuestion(
                question_id="Q_WITHDRAWAL_CONTROL",
                id="Q_WITHDRAWAL_CONTROL",
                question="What control required obsolete controlled copies to be removed from workstations after the new revision became effective?",
                purpose="Determine whether an obsolete-copy withdrawal control existed and operated",
                objective="Determine whether an obsolete-copy withdrawal control existed and operated",
                evidence="Document-control SOP, controlled-copy register, withdrawal records, and workstation verification",
                evidence_required="Document-control SOP, controlled-copy register, withdrawal records, and workstation verification",
                priority="P2",
                target_proposition_id="P_WITHDRAWAL",
                hypothesis_tested=common_hyp.id,
                depends_on="Q_DISTRIBUTION_VERIFICATION",
                activation_condition="If distribution was successful",
                status="CONDITIONAL",
                possible_outcomes=[
                    "Withdrawal control existed and operated → investigate local deviation.",
                    "Withdrawal control was absent or ineffective → investigate process/control weakness.",
                ],
            ),
            InvestigationQuestion(
                question_id="Q_SYSTEMIC_SCOPE",
                id="Q_SYSTEMIC_SCOPE",
                question="Does the same document-control weakness affect other controlled procedures or departments?",
                purpose="Determine whether this is an isolated event or a systemic weakness",
                objective="Determine whether this is an isolated event or a systemic weakness",
                evidence="Obsolete-document audit, previous findings, CAPA records, and document-control reports",
                evidence_required="Obsolete-document audit, previous findings, CAPA records, and document-control reports",
                priority="P3",
                target_proposition_id="P_SYSTEMIC_SCOPE",
                hypothesis_tested=common_hyp.id,
                status="CONDITIONAL",
                possible_outcomes=[
                    "Systemic weakness confirmed across other documents/departments.",
                    "Isolated to this event.",
                    "Unresolved — insufficient evidence to determine scope.",
                ],
            ),
        ]
        evidence_items = [
            f"Document revision history and approval record for {factor_label}",
            "Distribution logs, recipient mappings, and notification logs",
            "Controlled-copy register and withdrawal records",
            "Prior obsolete-document findings and CAPA records",
        ]

    # RECURRENCE: mandatory investigation questions and areas when recurrence is detected
    from app.agent.recurrence_guard import detect_recurrence
    recurrence = detect_recurrence(finding_text)
    if recurrence.is_recurring:
        recurrence_topic = topic_word(subject)
        # Prepended, not appended: when a finding already gives a specific,
        # decision-oriented investigation path (recurrence + previous CAPA),
        # those targeted questions must take priority over the generic
        # fallback questions built above, not be buried after them (Section
        # 7: "do not generate generic questions... when the finding already
        # gives a much more specific investigation path").
        questions = build_recurrence_investigation_questions(subject, recurrence_topic) + questions
        plan_areas = [
            *plan_areas,
            f"{recurrence_topic[0].upper()}{recurrence_topic[1:]} previous CAPA implementation and effectiveness verification",
            f"Causal relationship between previous {recurrence_topic} CAPA and current deviation",
        ]
        evidence_items.extend([
            f"Previous {recurrence_topic} corrective action plan and implementation completion evidence",
            f"Previous {recurrence_topic} corrective action effectiveness review and verification records",
            f"Root cause analysis and scope documentation from previous {recurrence_topic} CAPA",
        ])

    # -----------------------------------------------------------------------
    # Phase 4 Section 3/4: graph-grounded investigation question(s), derived
    # from the pre-investigation CausalUncertaintyGraph (built from the
    # semantic graph alone — the only causal structure that actually exists
    # at planning time, since CandidateHypothesis data doesn't exist until
    # core_synthesis runs afterward). Appended to, never replacing, the
    # existing question set; capped at 2 to avoid flooding an already
    # complete plan, and only added when the underlying uncertainty node
    # is not already covered by an existing question's target vocabulary.
    # -----------------------------------------------------------------------
    if canonical_state is not None:
        try:
            from app.agent.causal_graph import (
                build_causal_uncertainty_graph,
                information_gain_band_for_edge,
                rank_uncertainty_nodes_by_information_gain,
            )
            _uncertainty_graph = build_causal_uncertainty_graph(canonical_state)
            _ranked = rank_uncertainty_nodes_by_information_gain(_uncertainty_graph)
            _edge_by_target = {e.target_node_id: e for e in _uncertainty_graph.edges}
            _existing_q_text = " ".join(q.question.lower() for q in questions)
            _added = 0
            for _node in _ranked:
                if _added >= 2:
                    break
                if _node.label.lower() in _existing_q_text:
                    continue
                _edge = _edge_by_target.get(_node.node_id)
                if _edge is None:
                    continue
                _added += 1
                _gain_band, _gain_reason = information_gain_band_for_edge(_edge)
                questions.append(InvestigationQuestion(
                    question_id=f"Q_GRAPH_{_added}",
                    id=f"Q_GRAPH_{_added}",
                    question=(
                        f"What objective evidence establishes the causal mechanism connecting "
                        f"the observed deviation to {_node.label}?"
                    ),
                    purpose=f"Resolve unresolved causal edge to {_node.label}",
                    objective=f"Resolve unresolved causal edge to {_node.label}",
                    evidence="Objective records establishing the causal mechanism",
                    evidence_required="Objective records establishing the causal mechanism",
                    target_type="OTHER",
                    target_node_id=_node.node_id,
                    target_edge_id=_edge.edge_id,
                    source_node_id=_edge.source_node_id,
                    causal_level=str(_node.causal_level),
                    unresolved_relation=_edge.notes,
                    information_gain_rank=_added,
                    information_gain_band=_gain_band,
                    information_gain_reason=_gain_reason,
                    priority="HIGH",
                ))
        except Exception as _ug_err:  # pragma: no cover — defensive, never fatal to planning
            logger.warning("Causal uncertainty graph question derivation failed: %s", _ug_err)

    # Standardize all question fields to satisfy Section 5 schema
    for i, q in enumerate(questions):
        if not q.id:
            q.id = f"Q{i+1}"
        if not q.target_proposition_id:
            q.target_proposition_id = f"P{i+1}"
        if not q.resolves:
            q.resolves = q.purpose or f"Resolves proposition {q.target_proposition_id}"
        if not q.evidence_required:
            q.evidence_required = q.evidence or "Applicable objective records"
        if not q.decision_rule:
            q.decision_rule = "Evaluate objective evidence to confirm or refute the target proposition"

    # Build investigation plan -- evidence artifacts get the same
    # near-duplicate normalization (Section 10/INV-EVIDENCE-001) already
    # used for impact.evidence_needed, applied here at the source so the
    # investigation plan itself never shows "logs and logs and records"-
    # style repetition, not just the downstream aggregated field.
    from app.agent.analytical_validator import normalize_and_dedupe_evidence_items
    investigation_plan = InvestigationPlan(
        questions=questions,
        evidence_to_collect=normalize_and_dedupe_evidence_items(evidence_items, max_items=12),
        areas=plan_areas,
        interviews=[f"Responsible personnel involved in {subject}"],
    )

    all_claim_ids = [item.claim_id for item in evidence_ledger if hasattr(item, "claim_id") and item.claim_id] or ["C1"]
    for h in hypotheses:
        if not h.supporting_claim_ids:
            h.supporting_claim_ids = all_claim_ids[:1]

    return hypotheses, investigation_plan


def build_conditional_capa_actions(
    hypotheses: list[CandidateHypothesis],
    subject: str,
    topic: str,
) -> list[ConditionalCapaAction]:
    """Map each surviving candidate hypothesis to conditional CAPA branch(es):
    IF the hypothesis is confirmed -> an organizational corrective/preventive
    action (never an evidence source dressed up as an action), plus how
    effectiveness would be verified. Never prescribes an action before the
    hypothesis is confirmed — every branch stays conditional.

    A completion-gap hypothesis (H1-shaped: "<topic>_NOT_COMPLETED") gets TWO
    branches, not one: an IMMEDIATE_CORRECTION (addresses the current finding
    directly) and a separate SYSTEMIC_ACTION (addresses the confirmed cause)
    — Section 5: immediate correction and systemic CAPA must never be mixed
    into a single recommendation.
    """
    from app.services.semantic_subject import split_topic_and_tail, is_actor_noun
    tail = split_topic_and_tail(subject, topic) or subject
    if tail and (is_actor_noun(tail) or re.match(r"^[A-Z]+-\d+$", tail)):
        tail = f"assigned activities for {tail}"
    topic_cap = topic[0].upper() + topic[1:]
    actions: list[ConditionalCapaAction] = []
    for h in hypotheses:
        # Phase 22 Part G: a REFUTED hypothesis is not "IF confirmed" --
        # it has already been determined not to be the cause (by the sole
        # authoritative evaluator; see app.agent.nodes.evidence_acquisition.
        # reconcile_hypothesis_from_evidence). Recommending a conditional
        # action for it would misrepresent an already-closed question as
        # still open.
        if str(getattr(h, "status", "")) == "REFUTED":
            continue
        _actions_before = len(actions)
        name = h.name.upper()
        condition = f"IF {h.id} ({h.name.replace('_', ' ').title()}) is confirmed"
        if name.endswith("NOT_COMPLETED"):
            actions.append(ConditionalCapaAction(
                if_cause_confirmed=condition,
                recommended_action=f"Complete the required {topic} before independent execution of {tail}, where applicable.",
                action_type="IMMEDIATE_CORRECTION",
                verification_method=f"Approved {topic} completion record confirming completion before use.",
                evidence_needed=f"Approved {topic} completion/attendance record",
            ))
            actions.append(ConditionalCapaAction(
                if_cause_confirmed=condition,
                # Deliberately a distinct CONTROL/PROCESS fix, not a
                # restatement of the immediate correction above (Section 5:
                # immediate correction and systemic CAPA must never be
                # mixed into one recommendation, but they must also not be
                # the same recommendation worded twice) -- the immediate
                # action fixes THIS case; this fixes the process gap that
                # let it go undetected, where required by the applicable
                # procedure.
                recommended_action=(
                    f"Where required by the applicable procedure, implement or strengthen a "
                    f"{topic}-completion verification step ahead of {tail} so future gaps are caught "
                    f"before personnel proceed, rather than discovered afterward."
                ),
                action_type="SYSTEMIC_ACTION",
                verification_method=f"{topic_cap} completion is documented and verified before authorization.",
                evidence_needed=f"Authenticated {topic} attendance/completion record",
            ))
            for branch in actions[_actions_before:]:
                branch.root_cause_hypothesis_id = h.id
                branch.supporting_claim_ids = list(getattr(h, "supporting_claim_ids", None) or [])
            continue
        elif "RECORD_UNAVAILABLE" in name or "RECORD_OMISSION" in name:
            action = f"Improve controlled storage, retrieval, and traceability of {topic} completion records."
            verification = (
                f"An authenticated {topic} record can be retrieved reliably during a controlled "
                "verification."
            )
            evidence_needed = f"{topic_cap} record repository or audit trail"
        elif "RECORD_CONTROL_GAP" in name:
            action = f"Strengthen the creation, retention, and traceability controls for required {topic} records."
            verification = (
                f"Required {topic} records are created, retained, and retrievable according to the "
                "applicable record-control requirements."
            )
            evidence_needed = f"{topic_cap} record-control procedure and retention requirements"
        elif "VERIFICATION_CONTROL_GAP" in name or "RECORDING_AND_VERIFICATION" in name:
            action = (
                f"Implement or strengthen mandatory verification and authorization of {topic} completion "
                f"before personnel are permitted to proceed with {tail}."
            )
            verification = "Documented completion and authorization check."
            evidence_needed = f"{topic_cap} authorization/sign-off record"
        elif "PREVIOUS_CAPA_NOT_FULLY_IMPLEMENTED" in name:
            action = (
                f"Complete implementation of the previous corrective action for {subject}, including any "
                "planned actions that were not carried out."
            )
            verification = (
                "TO_BE_DEFINED — effectiveness criterion to be defined once the completed implementation "
                "is confirmed."
            )
            evidence_needed = "Previous corrective action plan and implementation completion evidence"
        elif "PREVIOUS_CAPA_EFFECTIVENESS_NOT_VERIFIED" in name:
            action = (
                f"Establish and apply effectiveness criteria for the previous corrective action related "
                f"to {subject}, and conduct the effectiveness verification that was not performed."
            )
            verification = "TO_BE_DEFINED — effectiveness criteria to be defined by the assigned owner before verification."
            evidence_needed = "Previous corrective action effectiveness review or verification record"
        elif "PREVIOUS_CAPA_VERIFIED_BUT_INEFFECTIVE" in name:
            action = (
                f"Reassess the causal basis of the previous corrective action for {subject}, since a "
                "verified-effective action did not prevent recurrence, and implement additional or "
                "revised systemic controls."
            )
            verification = "TO_BE_DEFINED — revised effectiveness criterion to be defined given the prior verification did not hold."
            evidence_needed = "Previous corrective action effectiveness review and current recurrence evidence"
        else:
            # No fixed name pattern matched -- this is the common case for
            # an LLM-authored hypothesis, whose `name` is free-form rather
            # than this module's deterministic naming convention. Derive the
            # action from the hypothesis's OWN statement/refutes_if content
            # instead of a generic "address the condition in H2" template,
            # which names nothing about what would actually be done.
            statement_clause = (h.statement or "").strip().rstrip(".")
            if statement_clause and not statement_clause.split()[0].isupper():
                statement_clause = statement_clause[0].lower() + statement_clause[1:]
            if statement_clause:
                action = f"Implement corrective and preventive controls addressing the confirmed cause: {statement_clause}."
            else:
                action = f"Address the condition identified in {h.id} with a targeted corrective action once confirmed."
            refutes_clause = (h.refutes_if or "").strip().rstrip(".")
            if refutes_clause and not refutes_clause.split()[0].isupper():
                refutes_clause = refutes_clause[0].lower() + refutes_clause[1:]
            if refutes_clause:
                verification = f"Verification confirms {refutes_clause}."
            else:
                verification = f"Re-verification confirms the condition described in {h.id} no longer recurs."
            evidence_needed = h.evidence_needed
        is_recurrence_action = name.startswith("PREVIOUS_CAPA_") or "_PREVIOUS_CAPA_" in name
        actions.append(ConditionalCapaAction(
            if_cause_confirmed=condition,
            recommended_action=action,
            action_type="SYSTEMIC_ACTION",
            verification_method=verification,
            evidence_needed=evidence_needed,
            effectiveness_owner="TO_BE_ASSIGNED" if is_recurrence_action else None,
            effectiveness_review_period="TO_BE_DEFINED" if is_recurrence_action else None,
        ))
        # Phase 22 Part H: stamp every branch generated for this hypothesis
        # with its traceability -- which hypothesis, and which claims (if
        # any) grounded it -- rather than leaving CAPA's link back to the
        # evidence chain implicit in prose only.
        for branch in actions[_actions_before:]:
            branch.root_cause_hypothesis_id = h.id
            branch.supporting_claim_ids = list(getattr(h, "supporting_claim_ids", None) or [])
    return actions


def build_recurrence_investigation_questions(
    subject: str, topic: str, hypotheses: list[CandidateHypothesis] | None = None
) -> list[InvestigationQuestion]:
    """Structured conditional CAPA lifecycle investigation questions covering:
      1. Implementation verification
      2. Effectiveness requirement and definition
      3. Effectiveness objective verification
      4. Scope comparison
      5. Mechanism equivalence and causal linkage
    """
    return [
        InvestigationQuestion(
            id="Q_REC_1",
            target_proposition_id="P_REC_1",
            question=f"What objective records establish whether the actions required by the previous {topic} CAPA were fully implemented?",
            purpose="Determine whether the previous corrective action was fully implemented before this finding occurred",
            resolves="Implementation status of previous corrective action",
            evidence=f"Previous {topic} CAPA action plan and implementation completion evidence",
            evidence_required=f"Previous {topic} CAPA action plan and implementation completion evidence",
            decision_rule="If implementation records show actions were not completed or partial → implementation gap is possible; if fully completed → proceed to effectiveness verification.",
            possible_outcomes=[
                "Implementation completion record found → proceed to effectiveness review verification.",
                "No implementation completion record, or implementation was partial → implementation gap supported.",
            ],
        ),
        InvestigationQuestion(
            id="Q_REC_2",
            target_proposition_id="P_REC_2",
            question=f"Did the previous {topic} CAPA require a formal effectiveness review with defined success criteria?",
            purpose="Establish whether effectiveness verification was mandatory and what specific success criterion governed it",
            objective="Establish whether effectiveness verification was mandatory and what specific success criterion governed it",
            resolves="Effectiveness verification requirement and criterion",
            evidence=f"Previous {topic} CAPA plan, procedure requirements, and approved effectiveness criteria",
            evidence_required=f"Previous {topic} CAPA plan and approved effectiveness criteria",
            priority="P3",
            decision_rule="If effectiveness review was not required → do not evaluate effectiveness failure; if required → proceed to verification records.",
            possible_outcomes=[
                "Effectiveness criterion defined → evaluate verification records against criterion.",
                "Effectiveness review not required → effectiveness-verification failure is not applicable.",
            ],
        ),
        InvestigationQuestion(
            id="Q_REC_3",
            target_proposition_id="P_REC_3",
            question=f"What objective record establishes whether the previous {topic} CAPA satisfied its defined effectiveness criterion?",
            purpose="Determine whether an objective effectiveness verification was conducted and met the defined criterion",
            resolves="Effectiveness verification status and criteria satisfaction",
            evidence=f"Previous {topic} CAPA effectiveness review records, audit trail, and evaluation results",
            evidence_required=f"Previous {topic} CAPA effectiveness review records and evaluation results",
            decision_rule="If no effectiveness review exists when required → effectiveness verification gap; if verified effective → proceed to scope comparison.",
            possible_outcomes=[
                "Effectiveness review verified against criteria → proceed to scope comparison.",
                "No effectiveness review exists or criteria not satisfied → effectiveness verification gap supported.",
            ],
        ),
        InvestigationQuestion(
            id="Q_REC_4",
            target_proposition_id="P_REC_4",
            question=f"Does the current finding fall within the scope of the previous {topic} CAPA?",
            purpose="Evaluate whether the current equipment/process/condition was covered by the previous CAPA boundary",
            resolves="Applicability scope of previous CAPA to current finding",
            evidence=f"Previous {topic} CAPA scope definition and current finding operational scope",
            evidence_required=f"Previous {topic} CAPA scope definition and current finding operational scope",
            decision_rule="If current finding is outside previous CAPA scope → previous CAPA is not related to this deviation; if within scope → proceed to mechanism equivalence.",
            possible_outcomes=[
                "Current finding outside scope → previous CAPA not applicable to this finding.",
                "Current finding within scope → proceed to causal mechanism comparison.",
            ],
        ),
        InvestigationQuestion(
            id="Q_REC_5",
            target_proposition_id="P_REC_5",
            question=f"What objective evidence establishes whether the current deviation represents recurrence of the same causal mechanism addressed by the previous {topic} CAPA?",
            purpose="Determine whether the underlying causal mechanism is identical to the mechanism addressed by the previous CAPA",
            resolves="Causal mechanism equivalence between previous CAPA and current deviation",
            evidence=f"Previous {topic} CAPA root cause analysis compared with current deviation mechanism evidence",
            evidence_required=f"Previous {topic} CAPA root cause analysis and current mechanism evidence",
            decision_rule="If mechanisms differ → previous CAPA did not cause current deviation; if identical mechanism and scope → CAPA ineffectiveness hypothesis may be evaluated.",
            possible_outcomes=[
                "Different mechanism → previous CAPA is not causally connected to current deviation.",
                "Identical mechanism and within scope → genuine recurrence of previously addressed failure mechanism.",
            ],
        ),
    ]
