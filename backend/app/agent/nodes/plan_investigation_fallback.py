"""Deterministic Fallback Investigation Planner.

Fired when investigation planning fails or produces zero questions/hypotheses.
Generates dynamic, case-grounded hypotheses and discriminating investigation questions
directly from canonical finding semantics, claim conflicts, and mechanism polarity.
"""

from __future__ import annotations

import re

from app.models.agent import (
    CandidateHypothesis,
    ConditionalCapaAction,
    EvidenceItem,
    EvidenceStatus,
    InvestigationPlan,
    InvestigationQuestion,
)
from app.services.semantic_subject import (
    extract_conflict_topic,
    extract_temporal_clause,
    resolve_deviation,
    split_topic_and_tail,
    strip_leading_article,
    topic_word,
)


def build_deterministic_investigation_plan(
    finding_text: str,
    evidence_ledger: list[EvidenceItem],
) -> tuple[list[CandidateHypothesis], InvestigationPlan]:
    """Build dynamic, case-grounded hypotheses and discriminating investigation questions."""
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
    subject = resolved.finding_subject or resolved.subject or "the affected process"
    actor = resolved.actor

    if extracted_id and extracted_id not in subject:
        subject = f"{subject} ({extracted_id})"

    # Areas defaults to the generic phrasing used by every non-conflict
    # branch below; the conflict branch overrides it with three specific,
    # investigation-oriented areas (Section 6) instead of one generic label.
    plan_areas = [f"Verify compliance and control records for {subject}"]

    for ent in extracted_entities:
        evidence_items.extend([f"{ent} execution record", f"{ent} maintenance/status log", f"{ent} audit trail"])

    from app.agent.claim_extractor import detect_evidence_conflicts, extract_claims
    from app.agent.causal_guard import extract_immediate_mechanism

    claims = extract_claims(finding_text, evidence_ledger)
    conflicts = detect_evidence_conflicts(claims)
    mechanism = extract_immediate_mechanism(reported_claims, fact_claims)

    text_low = finding_text.lower()

    # 1. Conflicting Evidence Branch — hypotheses, discrimination criteria and
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
        verified_texts = [
            getattr(c, "text", str(c)) for c in claims
            if getattr(c, "status", None) == EvidenceStatus.VERIFIED
        ]
        claim_a = (
            conflict_claims_by_id.get(conflict_claim_ids[0]) if len(conflict_claim_ids) > 0 and conflict_claim_ids[0] in conflict_claims_by_id
            else (reported_texts[0] if reported_texts else f"one party reported that {subject} was not completed")
        )
        claim_b = (
            conflict_claims_by_id.get(conflict_claim_ids[1]) if len(conflict_claim_ids) > 1 and conflict_claim_ids[1] in conflict_claims_by_id
            else (reported_texts[1] if len(reported_texts) > 1 else f"another party reported that {subject} was completed")
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

        missing_record_claim = verified_texts[0] if verified_texts else f"no {topic} record was available during the audit"

        # STRICT HYPOTHESIS SEMANTIC CONSISTENCY: each hypothesis describes
        # exactly ONE causal mechanism, and every one of its fields
        # (statement / evidence_needed / discrimination_evidence /
        # confirms_if / refutes_if) stays about that SAME mechanism:
        #   H1 - the record exists but couldn't be located/retrieved
        #   H2 - the activity itself was not completed
        #   H3 - the record was never created/retained (a record-control
        #        process failure, distinct from H1's retrieval problem)
        #   H4 - the activity may have occurred, but the organization's
        #        verification/authorization control didn't confirm it
        # Never a hypothesis about whether a PERSON'S STATEMENT was
        # accurate/honest -- a statement is evidence, not a mechanism.
        h1 = CandidateHypothesis(
            id="H1",
            name=f"{topic.upper()}_RECORD_UNAVAILABLE",
            statement=(
                f"The required {topic} record for {tail} may exist but was unavailable or could not be "
                "retrieved during the audit."
            ),
            status="POSSIBLE",
            evidence_needed=f"Authenticated {topic} record from the LMS, archive, or {topic} repository",
            confirms_if=f"A located, authenticated {topic} record confirms completion despite its initial unavailability",
            refutes_if=f"No {topic} record exists in any repository, archived or otherwise",
            discrimination_evidence=(
                f"An authenticated {topic} record located after the audit that confirms completion supports "
                "H1; if no such record can be located anywhere, H1 is refuted."
            ),
            rationale=f"Plausible because {claim_b}, while {missing_record_claim}.",
            relevance_rank="HIGH",
        )
        h2 = CandidateHypothesis(
            id="H2",
            name=f"{topic.upper()}_NOT_COMPLETED",
            statement=f"Required {topic} for {tail} was not completed{temporal_suffix}.",
            status="POSSIBLE",
            evidence_needed=f"Authenticated {topic} attendance/completion record",
            confirms_if=f"No approved {topic} completion record exists for the affected period",
            refutes_if=f"An approved {topic} completion record confirms timely completion",
            discrimination_evidence=(
                f"A record showing completion {effective_ref} weakens H2. Absence of a record alone does "
                "not prove H2 — see H1 and H3."
            ),
            rationale=f"Plausible because {claim_a}.",
            relevance_rank="HIGH",
        )
        h3 = CandidateHypothesis(
            id="H3",
            name=f"{topic.upper()}_RECORD_CONTROL_GAP",
            statement=f"The required {topic} record for {tail} was not created, maintained, or controlled as required.",
            status="POSSIBLE",
            evidence_needed=f"{topic_cap} record-control procedure, retention requirements, and record audit trail",
            confirms_if=(
                f"Applicable record-control requirements show a {topic} record should have existed, but no "
                "such record was ever created or retained"
            ),
            refutes_if=f"The record-control audit trail confirms the {topic} record was properly created and retained",
            discrimination_evidence=(
                "Evidence that the record-control process itself failed to create or retain the record "
                "supports H3; evidence that a record was properly created and retained (even if temporarily "
                "hard to locate) weakens H3 in favor of H1."
            ),
            rationale=f"Plausible because {missing_record_claim}, which is consistent with a record-control failure and not distinguishable from H1 without further investigation.",
            relevance_rank="HIGH",
        )
        h4 = CandidateHypothesis(
            id="H4",
            name=f"{topic.upper()}_VERIFICATION_CONTROL_GAP",
            statement=(
                f"Required {topic} for {tail} may have been completed, but the organization's verification "
                f"or authorization control failed to ensure completion was confirmed{temporal_suffix}."
            ),
            status="POSSIBLE",
            evidence_needed=f"{topic_cap} authorization requirements and verification/sign-off records",
            confirms_if=f"No verification or authorization step exists confirming {topic} completion before use",
            refutes_if=f"A documented verification/authorization step confirms {topic} completion was checked before use",
            discrimination_evidence=(
                f"Evidence that {topic} was completed but required verification or authorization was absent "
                "or ineffective supports H4. Evidence that verification and authorization were correctly "
                "performed weakens H4."
            ),
            rationale=(
                "Plausible because conflicting completion claims could indicate insufficient verification "
                "controls, but this is not established."
            ),
            relevance_rank="HIGH",
        )
        hypotheses.extend([h1, h2, h3, h4])

        questions.append(InvestigationQuestion(
            question=f"Can an authenticated {topic} record be located in the approved {topic} repository or archive?",
            purpose="Resolves H1 — whether the record exists but was unavailable during the audit",
            evidence=f"Authenticated {topic} record from the LMS, archive, or {topic} repository",
            hypothesis_tested="H1",
            confirms_if=h1.confirms_if,
            refutes_if=h1.refutes_if,
            possible_outcomes=[
                "Record is located and confirms completion → H1 strengthened.",
                "No record exists in any repository → H1 refuted.",
            ],
        ))
        questions.append(InvestigationQuestion(
            question=f"Do authenticated {topic} records show that {actor_phrase} completed the required {topic}{temporal_suffix}?",
            purpose="Resolves H2 — whether the required activity was actually completed",
            evidence=f"Authenticated {topic} attendance/completion record",
            hypothesis_tested="H2",
            confirms_if=h2.confirms_if,
            refutes_if=h2.refutes_if,
            possible_outcomes=[
                f"Record confirms completion {effective_ref} → H2 weakened.",
                "Record shows no completion → H2 strengthened.",
            ],
        ))
        questions.append(InvestigationQuestion(
            question=(
                f"Do the applicable record-control requirements show that a {topic} record should have been "
                f"created and retained for {tail}, and is there evidence that this control failed?"
            ),
            purpose="Resolves H3 — whether record creation/retention was itself defective",
            evidence=f"{topic_cap} record-control procedure, retention requirements, and record audit trail",
            hypothesis_tested="H3",
            confirms_if=h3.confirms_if,
            refutes_if=h3.refutes_if,
            possible_outcomes=[
                "Record-control audit trail shows the record was never created → H3 strengthened.",
                "Record-control audit trail confirms proper creation and retention → H3 weakened.",
            ],
        ))
        questions.append(InvestigationQuestion(
            question=(
                f"Was {topic} completion required to be verified or authorized{temporal_suffix}, and is "
                "there evidence that this control was executed?"
            ),
            purpose="Resolves H4 — whether a verification/authorization control failed",
            evidence=f"{topic_cap} authorization requirements and verification/sign-off records",
            hypothesis_tested="H4",
            confirms_if=h4.confirms_if,
            refutes_if=h4.refutes_if,
            possible_outcomes=[
                "Verification/authorization step was required but not executed → H4 strengthened.",
                "Verification/authorization step was executed and documented → H4 weakened.",
            ],
        ))
        evidence_items.extend([
            f"Authenticated {topic} record from the LMS, archive, or {topic} repository",
            f"Authenticated {topic} attendance/completion record",
            f"{topic_cap} record-control procedure and retention requirements",
            f"{topic_cap} authorization/sign-off record",
        ])
        # INVESTIGATION AREAS (Section 7 & 9): one per hypothesis, so every
        # area traces to a live candidate hypothesis — never a fixed generic
        # label, and explicitly investigation-oriented (not a confirmed CAPA
        # scope) since root cause remains NOT_ESTABLISHED at this point.
        plan_areas = [
            f"{topic_cap} completion verification",
            f"{topic_cap} record retrieval",
            f"{topic_cap} record control",
            f"{topic_cap} verification and authorization",
        ]

    # 2. Knowledge / Revision Gap Branch
    elif mechanism.polarity == "knowledge_gap" or ("unaware" in text_low and "revision" in text_low):
        h1 = CandidateHypothesis(
            id="H1",
            name="REQUIREMENT_COMMUNICATION_OR_ACKNOWLEDGEMENT_GAP",
            statement=f"Requirements or revisions affecting {subject} were not effectively communicated to or acknowledged by responsible personnel.",
            status="POSSIBLE",
            evidence_needed=f"{subject} distribution records, training acknowledgement logs, change control file",
            confirms_if=f"Acknowledgement records confirm personnel were not trained on or notified of the requirement",
            refutes_if=f"Signed acknowledgement confirms personnel completed notification/training prior to execution",
            discrimination_evidence="Distinguishes communication/training delivery breakdown from intentional noncompliance",
            relevance_rank="HIGH",
        )
        h2 = CandidateHypothesis(
            id="H2",
            name="PROCEDURAL_CLARITY_OR_GUIDANCE_WEAKNESS",
            statement=f"The written instructions or operational controls for {subject} lacked clear guidance on required execution steps.",
            status="POSSIBLE",
            evidence_needed=f"Current approved procedure for {subject}, operator feedback logs",
            confirms_if="Procedure wording is ambiguous or omits required execution triggers",
            refutes_if="Procedure provides unambiguous, step-by-step execution instructions",
            discrimination_evidence="Distinguishes procedural ambiguity from communication delivery failure",
            relevance_rank="HIGH",
        )
        hypotheses.extend([h1, h2])

        questions.append(InvestigationQuestion(
            question=f"Do training and acknowledgment records show that responsible personnel received the applicable requirements for {subject} prior to execution?",
            purpose="Determine whether communication or training delivery contributed to the knowledge gap",
            evidence=f"{subject} training records, change notification logs, distribution matrix",
            hypothesis_tested="H1",
            confirms_if=h1.confirms_if,
            refutes_if=h1.refutes_if,
        ))
        evidence_items.extend([f"{subject} training acknowledgement log", f"{subject} approved procedure copy"])

    # 3. Non-performance Branch
    elif mechanism.polarity == "non_performance":
        h1 = CandidateHypothesis(
            id="H1",
            name="TASK_ASSIGNMENT_OR_HANDOVER_OMISSION",
            statement=f"Responsibility for executing {subject} was not effectively scheduled or transferred across operational shifts.",
            status="POSSIBLE",
            evidence_needed=f"{subject} shift handover logs, duty rosters, task assignment records",
            confirms_if="Duty roster or handover log shows no personnel assigned to the activity",
            refutes_if="Duty roster confirms designated personnel signed for task execution",
            discrimination_evidence="Distinguishes task assignment breakdown from procedural clarity weakness",
            relevance_rank="HIGH",
        )
        h2 = CandidateHypothesis(
            id="H2",
            name="WORKFLOW_REMINDER_OR_VERIFICATION_CONTROL_GAP",
            statement=f"The operational workflow for {subject} lacked an effective reminder or supervisory verification control.",
            status="POSSIBLE",
            evidence_needed=f"{subject} operating procedure, notification setup, execution log",
            confirms_if="Procedure contains no scheduled trigger or supervisory verification step",
            refutes_if="Procedure specifies clear scheduled controls with supervisory sign-off",
            discrimination_evidence="Distinguishes control design gap from personnel handover failure",
            relevance_rank="HIGH",
        )
        hypotheses.extend([h1, h2])

        questions.append(InvestigationQuestion(
            question=f"Did shift handover and task assignment logs assign responsibility for {subject} during the affected period?",
            purpose="Distinguish task assignment/handover breakdown from workflow reminder control weakness",
            evidence=f"{subject} shift handover log, duty roster, operating procedure",
            hypothesis_tested="H1",
            confirms_if=h1.confirms_if,
            refutes_if=h1.refutes_if,
        ))
        evidence_items.extend([f"{subject} duty roster", "shift handover log", "standard operating procedure"])

    # 4. Non-recording Branch
    elif mechanism.polarity == "non_recording":
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
        )
        hypotheses.extend([h1, h2])

        questions.append(InvestigationQuestion(
            question=f"Do secondary physical records or electronic timestamps confirm execution of {subject} despite the missing log entry?",
            purpose="Evaluate contemporaneous recording compliance vs physical non-execution",
            evidence=f"{subject} execution logs, system audit trail",
            hypothesis_tested="H1",
            confirms_if=h1.confirms_if,
            refutes_if=h1.refutes_if,
        ))
        evidence_items.extend([f"{subject} execution log", "system audit trail"])

    # 5. General / Unresolved Branch
    else:
        h1 = CandidateHypothesis(
            id="H1",
            name="PROCESS_EXECUTION_COMPLIANCE_GAP",
            statement=f"The operational controls associated with {subject} were not executed as specified in applicable procedures.",
            status="POSSIBLE",
            evidence_needed=f"{subject} execution logs, supervisory verification records",
            confirms_if="Execution logs show deviations from approved procedural steps",
            refutes_if="Execution logs confirm strict adherence to approved procedure",
            discrimination_evidence="Distinguishes execution noncompliance from supervisory detection weakness",
            relevance_rank="HIGH",
        )
        h2 = CandidateHypothesis(
            id="H2",
            name="VERIFICATION_OR_RECONCILIATION_CONTROL_GAP",
            statement=f"Verification or reconciliation controls for {subject} did not detect or prevent the nonconforming condition.",
            status="POSSIBLE",
            evidence_needed=f"{subject} supervisory sign-off records, review logs",
            confirms_if="Supervisory review signed off without identifying the nonconforming condition",
            refutes_if="Supervisory review timely documented and escalated the condition",
            discrimination_evidence="Distinguishes verification control weakness from initial execution failure",
            relevance_rank="HIGH",
        )
        hypotheses.extend([h1, h2])

        questions.append(InvestigationQuestion(
            question=f"Do execution and verification records for {subject} confirm adherence to approved procedural controls?",
            purpose="Evaluate operational compliance vs verification control effectiveness",
            evidence=f"{subject} execution logs, supervisory verification records",
            hypothesis_tested="H1",
            confirms_if=h1.confirms_if,
            refutes_if=h1.refutes_if,
        ))
        evidence_items.extend([f"{subject} execution logs", f"{subject} verification records"])

    # RECURRENCE (Section 8/28): mandatory additional hypothesis whenever a
    # similar finding was previously identified — appended after whichever
    # branch above fired, never in place of it, since recurrence is an
    # additional causal dimension (did the previous CAPA actually prevent
    # this), not a replacement for reasoning about the current deviation.
    from app.agent.recurrence_guard import detect_recurrence
    recurrence = detect_recurrence(finding_text)
    if recurrence.is_recurring and hypotheses:
        recurrence_topic = topic_word(subject)
        recurrence_hyps = build_recurrence_hypotheses(subject, recurrence_topic, len(hypotheses) + 1)
        hypotheses.extend(recurrence_hyps)
        questions.extend(build_recurrence_investigation_questions(subject, recurrence_topic, recurrence_hyps))
        recurrence_area = f"{recurrence_topic[0].upper()}{recurrence_topic[1:]} CAPA implementation and effectiveness verification"
        if recurrence_area not in plan_areas:
            plan_areas = [*plan_areas, recurrence_area]
        evidence_items.extend(h.evidence_needed for h in recurrence_hyps)

    # Build investigation plan
    investigation_plan = InvestigationPlan(
        questions=questions,
        evidence_to_collect=list(dict.fromkeys(evidence_items)),
        areas=plan_areas,
        interviews=[f"Responsible personnel involved in {subject}"],
    )

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
    from app.services.semantic_subject import split_topic_and_tail
    tail = split_topic_and_tail(subject, topic) or subject
    topic_cap = topic[0].upper() + topic[1:]
    actions: list[ConditionalCapaAction] = []
    for h in hypotheses:
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
                recommended_action=(
                    f"Ensure required {topic} is completed and verified before personnel are authorized "
                    f"to proceed with {tail}."
                ),
                action_type="SYSTEMIC_ACTION",
                verification_method=f"{topic_cap} completion is documented and verified before authorization.",
                evidence_needed=f"Authenticated {topic} attendance/completion record",
            ))
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
            action = (
                f"Address the condition identified in {h.id} "
                f"({h.name.replace('_', ' ').lower()}) with a targeted corrective action once confirmed."
            )
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
    return actions


def build_recurrence_hypotheses(subject: str, topic: str, start_id: int) -> list[CandidateHypothesis]:
    """Builds the mandatory RECURRENCE hypothesis set when a similar finding
    was previously identified (Section 8/9, and the current-turn Section 2).

    RECURRENCE (a similar finding happened again) and CAPA INEFFECTIVENESS
    (the previous corrective action failed to work) are two distinct facts,
    and "the previous CAPA was completed" proves neither non-implementation,
    non-verification, nor ineffectiveness on its own. So recurrence is never
    collapsed into a single "CAPA effectiveness gap" hypothesis -- it is
    split into three mutually exclusive mechanisms, each with its own
    evidence and CAPA mapping, so status determination never has to average
    across them into a false SUPPORTED:

      H_REC_1 -- the previous corrective action was never fully implemented.
      H_REC_2 -- it was implemented, but effectiveness was never verified.
      H_REC_3 -- it was implemented AND verified effective, yet the same or
                 a similar nonconformity still recurred (verified-but-failed).

    All three start UNRESOLVED (POSSIBLE) -- never SUPPORTED -- because
    recurrence plus prior completion is evidence that ONE of these three is
    true, not evidence for any single one of them.
    """
    id1, id2, id3 = f"H{start_id}", f"H{start_id + 1}", f"H{start_id + 2}"
    h_rec_1 = CandidateHypothesis(
        id=id1,
        name=f"{topic.upper()}_PREVIOUS_CAPA_NOT_FULLY_IMPLEMENTED",
        statement=(
            f"The corrective action from the previous {topic}-related finding for {subject} was not "
            "fully implemented before this finding occurred."
        ),
        status="POSSIBLE",
        evidence_needed=f"Previous {topic} corrective action plan and implementation completion evidence",
        confirms_if=(
            "Implementation records show the previous corrective action was not completed, was only "
            "partially completed, or was closed without completing its planned actions"
        ),
        refutes_if="Implementation records show the previous corrective action's planned actions were fully completed",
        discrimination_evidence=(
            f"An implementation completion record for the previous corrective action weakens {id1}. "
            f"An incomplete, partial, or missing implementation record supports {id1}."
        ),
        rationale=(
            "Plausible because a similar finding was previously identified and the prior corrective "
            "action was recorded as completed, but whether its planned implementation actions were "
            "actually carried out has not been established from the available evidence."
        ),
        relevance_rank="HIGH",
    )
    h_rec_2 = CandidateHypothesis(
        id=id2,
        name=f"{topic.upper()}_PREVIOUS_CAPA_EFFECTIVENESS_NOT_VERIFIED",
        statement=(
            f"The corrective action from the previous {topic}-related finding for {subject} was "
            "implemented, but its effectiveness was never objectively verified."
        ),
        status="POSSIBLE",
        evidence_needed=f"Previous {topic} corrective action effectiveness review or verification record",
        confirms_if="No effectiveness review or verification record exists for the previous corrective action",
        refutes_if="An effectiveness review or verification record exists for the previous corrective action",
        discrimination_evidence=(
            f"An effectiveness review or verification record weakens {id2}. The absence of any such "
            f"record supports {id2}."
        ),
        rationale=(
            "Plausible because the previous corrective action was recorded as completed, but completion "
            "does not itself establish that effectiveness was verified -- that is a separate, unestablished "
            "fact."
        ),
        relevance_rank="HIGH",
    )
    h_rec_3 = CandidateHypothesis(
        id=id3,
        name=f"{topic.upper()}_PREVIOUS_CAPA_VERIFIED_BUT_INEFFECTIVE",
        statement=(
            f"The corrective action from the previous {topic}-related finding for {subject} was "
            "implemented and objectively verified as effective, but the same or a similar nonconformity "
            "still recurred."
        ),
        status="POSSIBLE",
        evidence_needed=f"Previous {topic} corrective action effectiveness review and current recurrence evidence",
        confirms_if=(
            "An effectiveness review verified the previous corrective action as effective, and this "
            "finding nonetheless describes the same or a similar nonconformity"
        ),
        refutes_if="No effectiveness review verified the previous corrective action as effective",
        discrimination_evidence=(
            f"The combination of a documented effectiveness verification AND a recurrence of the same or "
            f"a similar nonconformity supports {id3}; absence of a prior effectiveness verification "
            f"weakens {id3} (in favor of {id2})."
        ),
        rationale=(
            "Plausible only if a documented effectiveness verification exists for the previous corrective "
            "action -- if it does not, this mechanism cannot be distinguished from unverified "
            f"effectiveness ({id2})."
        ),
        relevance_rank="MEDIUM",
    )
    return [h_rec_1, h_rec_2, h_rec_3]


def build_recurrence_investigation_questions(
    subject: str, topic: str, hypotheses: list[CandidateHypothesis]
) -> list[InvestigationQuestion]:
    """One well-formed, single-proposition investigation question per
    recurrence hypothesis (Section 7/28, and current-turn Section 12-14) --
    never one compound question covering all three mechanisms at once."""
    h_rec_1, h_rec_2, h_rec_3 = hypotheses
    return [
        InvestigationQuestion(
            question=(
                f"Was the corrective action from the previous {topic}-related finding for {subject} "
                "fully implemented?"
            ),
            purpose=f"Resolves {h_rec_1.id} — whether the previous CAPA's planned actions were actually carried out",
            evidence=f"Previous {topic} corrective action plan and implementation completion evidence",
            hypothesis_tested=h_rec_1.id,
            confirms_if=h_rec_1.confirms_if,
            refutes_if=h_rec_1.refutes_if,
            possible_outcomes=[
                "Implementation completion record found → hypothesis weakened.",
                "No implementation completion record, or implementation was partial → hypothesis strengthened.",
            ],
        ),
        InvestigationQuestion(
            question=(
                f"Was the effectiveness of the previous corrective action for {subject} objectively "
                "verified?"
            ),
            purpose=f"Resolves {h_rec_2.id} — whether an effectiveness review or verification record exists",
            evidence=f"Previous {topic} corrective action effectiveness review or verification record",
            hypothesis_tested=h_rec_2.id,
            confirms_if=h_rec_2.confirms_if,
            refutes_if=h_rec_2.refutes_if,
            possible_outcomes=[
                "Effectiveness review or verification record found → hypothesis weakened.",
                "No effectiveness review or verification record exists → hypothesis strengthened.",
            ],
        ),
        InvestigationQuestion(
            question=(
                f"If the previous corrective action for {subject} was verified effective, why did the "
                "same or a similar nonconformity still recur?"
            ),
            purpose=f"Resolves {h_rec_3.id} — whether a verified-effective action nonetheless failed to prevent recurrence",
            evidence=f"Previous {topic} corrective action effectiveness review and current recurrence evidence",
            hypothesis_tested=h_rec_3.id,
            confirms_if=h_rec_3.confirms_if,
            refutes_if=h_rec_3.refutes_if,
            possible_outcomes=[
                "No prior effectiveness verification exists → hypothesis not applicable, see H_REC_2.",
                "Prior effectiveness verification exists and recurrence is confirmed → hypothesis strengthened.",
            ],
        ),
    ]
