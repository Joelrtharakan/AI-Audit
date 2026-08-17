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
) -> tuple[list[CandidateHypothesis], InvestigationPlan]:
    """Build dynamic, case-grounded hypotheses and discriminating investigation questions.

    `canonical_subject` (understand_finding_node's already-resolved
    `canonical_finding_state.finding_subject`) is preferred over this
    function's own independent `resolve_deviation()` call whenever it is
    available and not itself degraded -- there must be ONE authoritative
    semantic-subject producer for a finding, not two that can silently
    disagree (Section 20: "canonical.finding_subject must remain the
    authoritative semantic source... Do not independently re-derive
    semantic subjects"). Defaults to None so every existing call site
    (production and test) keeps its current behavior unchanged unless it
    explicitly opts in.
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
    else:
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
                f"{subject_cap} delivery, receipt and acknowledgement control",
                f"Operational activity relative to {subject}",
                "Technical/administrative factors affecting delivery or receipt",
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
                f"A record showing completion {effective_ref} weakens H1. Absence of a record alone does "
                "not prove H1 -- see the record-availability and record-control investigation areas below."
            ),
            rationale=f"Plausible because {claim_a}.",
            relevance_rank="HIGH",
            supporting_evidence=[claim_a],
            contradicting_evidence=[claim_b] if claim_b != claim_a else [],
        )
        hypotheses.append(h1)

        questions.append(InvestigationQuestion(
            question=f"Do authenticated {topic} records show that {actor_phrase} completed the required {topic}{temporal_suffix}?",
            purpose="Resolves H1 — whether the required activity was actually completed",
            evidence=f"Authenticated {topic} attendance/completion record",
            hypothesis_tested="H1",
            confirms_if=h1.confirms_if,
            refutes_if=h1.refutes_if,
            possible_outcomes=[
                f"Record confirms completion {effective_ref} → H1 weakened.",
                "Record shows no completion → H1 strengthened.",
            ],
        ))

        # EVIDENCE-STATE investigation (not a hypothesis, not scored against
        # H1): the record was unavailable AT AUDIT TIME. That fact alone
        # cannot confirm OR refute H1 -- failing to locate a record never
        # establishes the record never existed, was lost, or that
        # retrieval failed (those are separate, evidence-requiring claims).
        # `hypothesis_tested` deliberately left unset -- this question does
        # not test a candidate hypothesis; it resolves an antecedent
        # evidence-availability question the hypothesis's own discrimination
        # depends on.
        questions.append(InvestigationQuestion(
            question=f"Can an authenticated {topic} record be located in the approved {topic} repository or archive?",
            purpose=(
                "Resolves the antecedent evidence-availability question: whether the record exists and "
                "can be located, prior to evaluating what it shows"
            ),
            evidence=f"Authenticated {topic} record from the LMS, archive, or {topic} repository",
            confirms_if=f"A located, authenticated {topic} record confirms completion despite its initial unavailability",
            refutes_if=(
                f"No {topic} record can be located after a documented, thorough search of every applicable "
                "repository — this alone still does not establish the activity did not occur; see the "
                "record-control investigation area if that remains unresolved"
            ),
            possible_outcomes=[
                "Record is located and confirms completion → H1 weakened.",
                "Record is located and does not confirm completion → H1 strengthened.",
                "No record can be located → availability remains unresolved unless record-control evidence "
                "establishes why.",
            ],
        ))

        # SYSTEMIC investigation area (record-control process): deliberately
        # NOT a single bundled hypothesis combining creation/retention/
        # retrieval/verification into one statement (that conflates four
        # distinct control points into one unfalsifiable claim) and
        # deliberately NOT a candidate hypothesis at all -- this is a
        # process-level investigation target, evaluated only if the
        # evidence-availability question above leaves the record
        # genuinely unresolved.
        questions.append(InvestigationQuestion(
            question_id="Q_RECORD_CONTROL_REQUIREMENT",
            id="Q_RECORD_CONTROL_REQUIREMENT",
            question=(
                f"Does the record-control audit trail establish whether the {topic} record for {tail} "
                "was created and retained in accordance with applicable requirements?"
            ),
            purpose="Determine whether the record-control process (creation/retention) itself has a control weakness",
            objective="Determine whether the record-control process (creation/retention) itself has a control weakness",
            evidence=f"{topic_cap} record-control procedure, retention requirements, and record audit trail",
            evidence_required=f"{topic_cap} record-control procedure, retention requirements, and record audit trail",
            target_type="PROPOSITION",
            target_proposition_id="P_RECORD_CONTROL",
            priority="P3",
            possible_outcomes=[
                "Audit trail shows the record was never created or retained as required → record-control weakness supported.",
                "Audit trail confirms proper creation and retention → record-control weakness refuted.",
            ],
        ))
        questions.append(InvestigationQuestion(
            question_id="Q_VERIFICATION_AUTHORIZATION",
            id="Q_VERIFICATION_AUTHORIZATION",
            question=(
                f"Do verification records establish whether {topic} completion was authorized or verified "
                f"before {actor_phrase} performed the relevant activity{temporal_suffix}?"
            ),
            purpose="Determine whether a completion-verification/authorization control failed to catch the gap",
            objective="Determine whether a completion-verification/authorization control failed to catch the gap",
            evidence=f"{topic_cap} authorization requirements and verification/sign-off records",
            evidence_required=f"{topic_cap} authorization requirements and verification/sign-off records",
            target_type="PROPOSITION",
            target_proposition_id="P_VERIFICATION",
            priority="P3",
            possible_outcomes=[
                "Verification/authorization step was required but not executed → verification-control weakness supported.",
                "Verification/authorization step was executed and documented → verification-control weakness refuted.",
            ],
        ))
        evidence_items.extend([
            f"Authenticated {topic} attendance/completion record",
            f"Authenticated {topic} record from the LMS, archive, or {topic} repository",
            f"{topic_cap} record-control procedure and retention requirements",
            f"{topic_cap} authorization/sign-off record",
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
            questions.append(InvestigationQuestion(
                question=f"What requirement and responsibility applied to {subject} during the affected period?",
                purpose="Establish the applicable requirement and responsibility/assignment/scheduling controls before any specific mechanism can be investigated",
                evidence=f"Applicable procedure, responsibility matrix, duty/shift assignment records for {subject}",
                hypothesis_tested=None,
            ))
            # Two independent causal branches (unrecorded performance vs. a
            # separate execution-affecting event) never share one question --
            # each is its own discriminating test with its own evidence.
            questions.append(InvestigationQuestion(
                question=f"Is there objective evidence that {subject} was performed but not recorded?",
                purpose="Distinguishes non-performance from an unrecorded performance",
                evidence=f"Secondary records, electronic/instrument audit trail, supervisory verification for {subject}",
                hypothesis_tested=None,
            ))
            questions.append(InvestigationQuestion(
                question=f"Is there a documented event that could have affected completion of {subject} during the affected period?",
                purpose="Identifies whether a contemporaneous event could explain the missed activity",
                evidence=f"Deviation, incident, equipment alarm, maintenance, or staffing records for {subject}, where applicable",
                hypothesis_tested=None,
            ))
            evidence_items.extend([
                f"Applicable procedure and responsibility matrix for {subject}",
                f"Secondary/independent verification records for {subject}",
                f"Deviation/incident/contemporaneous records for {subject}",
            ])

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

    # 5. General / Unresolved Branch: no conflict, no reported mechanism, no
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

        _control_proof_pattern = re.compile(
            r"\b(?:interlock|block|guard|safety|verification)\b.*?\b(?:disabled|bypassed|defeated|deactivated|switched off)\b|"
            r"\b(?:audit\s+logs?|server\s+logs?)\s+(?:establish|show|confirm)\b",
            re.IGNORECASE,
        )
        has_direct_control_proof = bool(_control_proof_pattern.search(finding_text))
        if has_direct_control_proof:
            # Provenance enforcement applies to every hypothesis producer,
            # not only the LLM-parsed path (core_synthesis._parse_causal_fields):
            # a SUPPORTED hypothesis must cite the verified claim(s) that
            # actually establish the control/interlock disablement it names,
            # never assert support with an empty evidence list.
            _citing_claims = [c for c in fact_claims if _control_proof_pattern.search(c)] or fact_claims
            h1 = CandidateHypothesis(
                id="H1",
                name="CONTROL_OR_INTERLOCK_DISABLED",
                statement=f"The operating control or interlock for {subject} was disabled prior to operation, permitting the out-of-range condition.",
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
        else:
            questions.extend([
                InvestigationQuestion(
                    question_id="Q_GOVERNING_REQUIREMENT",
                    id="Q_GOVERNING_REQUIREMENT",
                    question=f"What approved procedure or requirement governs {subject} during the relevant period?",
                    purpose="Establish the applicable requirement before evaluating whether it was met",
                    objective="Establish the applicable requirement before evaluating whether it was met",
                    evidence=f"Applicable procedure/requirement governing {subject}",
                    evidence_required=f"Applicable procedure/requirement governing {subject}",
                    priority="P3",
                    target_proposition_id="P_GOV",
                ),
                InvestigationQuestion(
                    question_id="Q_STATUS_HISTORY",
                    id="Q_STATUS_HISTORY",
                    question=f"What records establish the actual status of {subject_bare} at the time relevant to this finding?",
                    purpose="Establish the objective status/history, not merely what the finding itself states",
                    objective="Establish the objective status/history, not merely what the finding itself states",
                    evidence=f"Status/history records for {subject}",
                    evidence_required=f"Status/history records for {subject}",
                    priority="P2",
                    target_proposition_id="P_STATUS",
                ),
                InvestigationQuestion(
                    question_id="Q_AUTHORIZED_EXCEPTION",
                    id="Q_AUTHORIZED_EXCEPTION",
                    question=f"Did any approved exception, extension, waiver, or deviation apply to {subject} during the relevant period?",
                    purpose="Determine whether an authorized departure from the normal requirement existed",
                    objective="Determine whether an authorized departure from the normal requirement existed",
                    evidence=f"Exception/waiver/deviation records for {subject}",
                    evidence_required=f"Exception/waiver/deviation records for {subject}",
                    priority="P3",
                    target_proposition_id="P_EXCEPTION",
                ),
                InvestigationQuestion(
                    question_id="Q_PROCESS_RESPONSIBILITY",
                    id="Q_PROCESS_RESPONSIBILITY",
                    question=f"What process or role was responsible for monitoring and controlling {subject}?",
                    purpose="Identify the control point relevant to further investigation, without presupposing it failed",
                    objective="Identify the control point relevant to further investigation, without presupposing it failed",
                    evidence=f"Process ownership and responsibility records for {subject}",
                    evidence_required=f"Process ownership and responsibility records for {subject}",
                    priority="P3",
                    target_proposition_id="P_RESPONSIBILITY",
                ),
                InvestigationQuestion(
                    question_id="Q_DOWNSTREAM_DEPENDENCIES",
                    id="Q_DOWNSTREAM_DEPENDENCIES",
                    question=f"What downstream activities, decisions, or outputs depended on {subject} during the relevant period?",
                    purpose="Scope potential downstream impact for assessment, without asserting impact occurred",
                    objective="Scope potential downstream impact for assessment, without asserting impact occurred",
                    evidence=f"Records of activities/decisions dependent on {subject}",
                    evidence_required=f"Records of activities/decisions dependent on {subject}",
                    priority="P4",
                    target_proposition_id="P_SCOPE",
                ),
            ])
        evidence_items.extend([
            f"Applicable procedure/requirement governing {subject}",
            f"Status/history records for {subject}",
            f"Exception/waiver/deviation records for {subject}",
        ])

    # Filter out any evidence-state propositions from hypotheses
    from app.agent.causal_guard import is_evidence_state_not_hypothesis
    hypotheses = [h for h in hypotheses if not is_evidence_state_not_hypothesis(h.statement, h.name)]

    # RECURRENCE: mandatory investigation questions and areas when recurrence is detected
    from app.agent.recurrence_guard import detect_recurrence
    recurrence = detect_recurrence(finding_text)
    if recurrence.is_recurring:
        recurrence_topic = topic_word(subject)
        questions.extend(build_recurrence_investigation_questions(subject, recurrence_topic))
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
