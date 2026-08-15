"""Deterministic Fallback Investigation Planner (Section 4, 8 & 20).

Fired when investigation planning fails or produces zero questions/hypotheses.
Generates dynamic, finding-specific candidate hypotheses and investigation questions
directly from the extracted finding deviation instead of returning empty lists.
"""

from __future__ import annotations

import re

from app.models.agent import CandidateHypothesis, EvidenceItem, EvidenceStatus, InvestigationPlan, InvestigationQuestion
from app.services.semantic_subject import resolve_deviation



def build_deterministic_investigation_plan(
    finding_text: str,
    evidence_ledger: list[EvidenceItem],
) -> tuple[list[CandidateHypothesis], InvestigationPlan]:
    """Build dynamic, case-grounded hypotheses and investigation questions."""
    hypotheses: list[CandidateHypothesis] = []
    questions: list[InvestigationQuestion] = []
    evidence_items: list[str] = []

    # Generalized entity extraction (IDs, SOPs, BMRs, Lots, Equipment Tags, System Codes, Rooms, Lines)
    extracted_entities = re.findall(
        r"\b([A-Z]{2,5}-[A-Z0-9-]+|Lot\s+[A-Z0-9-]+|Batch\s+[A-Z0-9-]+|Line\s+\d+|Room\s+\d+|Cleanroom\s+[A-Za-z0-9\s]+|Autoclave\s+#?\d+|AHU-\d+|CR-\d+|LF-\d+|VI-\d+|CP-\d+|PP-\d+|Lyo-\d+|FH-\d+|SP-\d+|BAL-\d+|W-\d+|NC-\d+-\d+|CAPA-\d+-\d+|BRD-\d+|MBR-[A-Z0-9-]+|WSC-\d+|API-[0-9]+|RM-[0-9]+)\b",
        finding_text,
        re.IGNORECASE,
    )
    extracted_id = extracted_entities[0] if extracted_entities else ""

    fact_claims = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.VERIFIED]
    resolved = resolve_deviation(finding_text, fact_claims)
    subject = resolved.subject or "UNKNOWN — no affected object could be isolated from the finding text"

    if extracted_id:
        subject = extracted_id

    # Automatically add evidence items for every explicit entity referenced
    for ent in extracted_entities:
        evidence_items.extend([f"{ent} copy", f"{ent} log", f"{ent} audit trail"])


    # Step 4: Classify Finding Type (Section 4)
    text_low = finding_text.lower()
    if any(w in text_low for w in ("omitted the time", "missing timestamp", "omitted entry", "blank line", "incomplete entry", "lacked the required")):
        finding_type = "RECORD_COMPLETENESS"
    elif any(w in text_low for w in ("expired", "overdue", "out of service", "fluctuat", "malfunction", "calibration status")):
        finding_type = "CALIBRATION_EQUIPMENT_STATUS"
    elif any(w in text_low for w in ("policy", "procedure", "version", "sop", "outdated")):
        finding_type = "PROCEDURE_POLICY_VERIFICATION"
    elif any(w in text_low for w in ("tag", "sticker", "binder", "label", "missing tag")):
        finding_type = "EQUIPMENT_IDENTIFICATION_TAGGING"
    elif any(w in text_low for w in ("sequence", "method", "step")):
        finding_type = "PROCEDURE_SEQUENCE_NONCOMPLIANCE"
    elif any(w in text_low for w in ("correction fluid", "overwritten", "white-out")):
        finding_type = "DOCUMENT_INTEGRITY_ALCOA"
    else:
        finding_type = "RECORD_COMPLETENESS"

    # Evidence is derived from the hypotheses generated below (each hypothesis's
    # evidence_needed / evidence items reference this finding's own subject),
    # not from a blanket keyword-to-evidence mapping applied regardless of
    # whether any hypothesis actually needs it. The only evidence added before
    # a hypothesis exists is entity-specific (above) and recurrence-specific
    # (below), both tied to something explicitly present in the finding text.
    if any(w in text_low for w in ("capa", "recurrence", "nonconformity")):
        evidence_items.extend(["prior investigation file", "CAPA record"])

    from app.agent.causal_guard import extract_immediate_mechanism
    mechanism = extract_immediate_mechanism([e.claim for e in evidence_ledger if e.status == EvidenceStatus.REPORTED], fact_claims)

    if mechanism.polarity == "non_performance":
        hypotheses.append(CandidateHypothesis(
            id="H1",
            name="TASK_ASSIGNMENT_OR_HANDOVER_OMISSION",
            statement=f"Responsibility for performing the activity associated with {subject} was not effectively assigned or transferred across shifts.",
            status="POSSIBLE",
            evidence_needed=f"{subject} shift handover logs, duty roster, task assignment records",
            relevance_rank="HIGH",
        ))
        hypotheses.append(CandidateHypothesis(
            id="H2",
            name="PROCEDURAL_NOTIFICATION_OR_CONTROL_GAP",
            statement=f"The required schedule or procedure for {subject} lacked an effective reminder or verification control.",
            status="POSSIBLE",
            evidence_needed=f"{subject} standard operating procedure, notification setup, execution log",
            relevance_rank="HIGH",
        ))
        questions.append(InvestigationQuestion(
            question=f"Why was the required activity associated with {subject} missed?",
            purpose="Distinguish task assignment/handover breakdown from procedural clarity or notification control weakness",
            evidence=f"{subject} shift handover log, duty roster, standard operating procedure",
        ))
        evidence_items.extend([f"{subject} duty roster", "shift handover log", "standard operating procedure"])
    elif mechanism.polarity == "knowledge_gap":
        hypotheses.append(CandidateHypothesis(
            id="H1",
            name="PROCEDURE_REVISION_COMMUNICATION_GAP",
            statement=f"Revisions or updates affecting {subject} were not effectively communicated or acknowledged by personnel.",
            status="POSSIBLE",
            evidence_needed=f"{subject} document revision history, training acknowledgement records, distribution log",
            relevance_rank="HIGH",
        ))
        hypotheses.append(CandidateHypothesis(
            id="H2",
            name="DOCUMENT_DISTRIBUTION_OR_ACCESS_CONTROL_GAP",
            statement=f"The active version of the procedure for {subject} was not accessible at the point of use.",
            status="POSSIBLE",
            evidence_needed=f"{subject} point-of-use document audit, document management system log",
            relevance_rank="HIGH",
        ))
        questions.append(InvestigationQuestion(
            question=f"Was the procedure revision for {subject} communicated and accessible at the workstation?",
            purpose="Determine whether document distribution or training acknowledgment contributed to knowledge gap",
            evidence=f"{subject} document control index, training records, point-of-use document copy",
        ))
        evidence_items.extend([f"{subject} document control index", "training records", "point-of-use copy"])
    elif mechanism.polarity == "non_recording":
        hypotheses.append(CandidateHypothesis(
            id="H1",
            name="DOCUMENTATION_CONTROL_OR_TIMELINESS_GAP",
            statement=f"The activity for {subject} was executed but contemporaneous recording controls were omitted or delayed.",
            status="POSSIBLE",
            evidence_needed=f"{subject} execution logs, system audit trail, secondary verification records",
            relevance_rank="HIGH",
        ))
        hypotheses.append(CandidateHypothesis(
            id="H2",
            name="WORKSTATION_RECORDING_ACCESSIBILITY_GAP",
            statement=f"Recording forms or digital interfaces for {subject} were unavailable or inconvenient during task execution.",
            status="POSSIBLE",
            evidence_needed=f"{subject} workstation log, interface audit trail",
            relevance_rank="HIGH",
        ))
        questions.append(InvestigationQuestion(
            question=f"Were documentation controls or system interfaces for {subject} accessible at the time of execution?",
            purpose="Evaluate contemporaneous recording capability vs execution delay",
            evidence=f"{subject} execution logs, system audit trail",
        ))
        evidence_items.extend([f"{subject} execution log", "system audit trail"])
    else:
        hypotheses.append(CandidateHypothesis(
            id="H1",
            name="PROCESS_EXECUTION_OR_STATUS_GAP",
            statement=f"The process controls associated with {subject} were not executed as specified in the applicable procedure.",
            status="POSSIBLE",
            evidence_needed=f"{subject} execution logs, supervisory verification records",
            relevance_rank="HIGH",
        ))
        hypotheses.append(CandidateHypothesis(
            id="H2",
            name="VERIFICATION_CONTROL_WEAKNESS",
            statement=f"The supervisory check or automated verification for {subject} failed to detect or prevent the non-conformity.",
            status="POSSIBLE",
            evidence_needed=f"{subject} supervisory sign-off records, review logs",
            relevance_rank="HIGH",
        ))
        questions.append(InvestigationQuestion(
            question=f"What control or verification breakdown allowed the non-conformity in {subject} to occur?",
            purpose="Evaluate procedural compliance vs verification control weakness",
            evidence=f"{subject} execution logs, supervisory verification records",
        ))
        evidence_items.extend([f"{subject} execution logs", "supervisory review records", "audit trail"])






    deduped_evidence = list(dict.fromkeys(evidence_items))

    # Areas are derived from the hypotheses actually generated above (each
    # already grounded to this finding's own subject/mechanism), never a
    # fixed universal category list that would read as generic regardless
    # of what this finding is about.
    areas = [h.name.replace("_", " ").title() for h in hypotheses] or [subject]

    plan = InvestigationPlan(
        areas=areas,
        questions=questions,
        evidence_to_collect=deduped_evidence,
    )

    return hypotheses, plan
