"""Canonical Proposition Engine for LQMS Causal Reasoning.

Decomposes findings and extracted claims into formal Proposition objects,
assigns causal ladder levels (L0 to L5), classifies the investigation mode,
and maintains strict provenance between observations, mechanisms, and causes.
"""

from __future__ import annotations

import re
from typing import Any

from app.models.agent import (
    CausalLevel,
    ClaimAttribution,
    EpistemicSource,
    EvidenceClaim,
    EvidenceCompleteness,
    EvidenceConflict,
    EvidenceItem,
    EvidenceStatus,
    InvestigationMode,
    Proposition,
    PropositionType,
    ReferencedDocumentInfo,
    SupportLevel,
)


def classify_investigation_mode(
    finding_text: str,
    evidence_ledger: list[EvidenceItem] | list[EvidenceClaim] | None = None,
    conflicts: list[EvidenceConflict] | None = None,
    referenced_docs: list[ReferencedDocumentInfo] | None = None,
) -> InvestigationMode:
    """Classify the finding into its primary investigation mode.

    Modes:
      - CONFLICT: Conflicting accounts, delivery vs receipt, record vs statement.
      - DOCUMENT_UNAVAILABLE: Referenced document/report is missing/unavailable.
      - TEMPORAL_DEVIATION: Activity occurred after stated expiry / out of sequence.
      - REPORTED_MECHANISM: A person stated an execution omission / explanation.
      - LOW_SPECIFICITY: Vague assertion lacking specific requirement/evidence.
      - NORMAL: Standard single-observation finding.
    """
    text_lower = (finding_text or "").lower()

    # 1. Conflict detection (highest priority)
    if conflicts and len(conflicts) > 0:
        return InvestigationMode.CONFLICT

    # Check for delivery vs receipt or operator vs supervisor wording or system vs human statement
    if re.search(r"\b(delivered|sent|transmitted)\b", text_lower) and re.search(
        r"\b(not received|never received|did not receive|stated.*not received)\b", text_lower
    ):
        return InvestigationMode.CONFLICT

    if re.search(
        r"\b(recorded|states?|showed|shows?)\b.*?\bbut\b.*?\b(stated|claimed|reported)\b",
        text_lower,
    ):
        return InvestigationMode.CONFLICT

    if (
        re.search(r"\b(operator|technician|staff|personnel)\s+stated\b", text_lower)
        and re.search(r"\b(supervisor|manager|lead)\s+claimed\b", text_lower)
    ):
        return InvestigationMode.CONFLICT

    # 2. Document referenced but unavailable
    if referenced_docs and any(
        getattr(d, "reference_status", "") == "REFERENCED_UNAVAILABLE" for d in referenced_docs
    ):
        return InvestigationMode.DOCUMENT_UNAVAILABLE

    if re.search(r"\b(referenced|attached|cited)\b.*?\b(not available|unavailable|missing|could not be located)\b", text_lower):
        return InvestigationMode.DOCUMENT_UNAVAILABLE

    # 3. Temporal deviation: Expiry then use
    if re.search(r"\bexpir\w*\b", text_lower) and re.search(
        r"\b(used|performed|conducted|operated|executed)\b", text_lower
    ):
        return InvestigationMode.TEMPORAL_DEVIATION

    # 4. Reported mechanism by a person
    if re.search(
        r"\b(stated|claimed|reported|confirmed|explained)\s+(?:that\s+)?(?:the\s+[\w-]+\s+was\s+)?(?:they|he|she|i|it)?\s*(?:had\s+not|had\s+never|forgot|missed|was\s+missed|was\s+not)\b",
        text_lower,
    ) or (
        evidence_ledger and any(getattr(e, "status", None) == EvidenceStatus.REPORTED for e in evidence_ledger)
    ):
        return InvestigationMode.REPORTED_MECHANISM

    # 5. Low specificity
    if (
        len(text_lower.split()) < 12
        or re.search(r"^(the\s+)?(department|facility|team|staff|personnel)\s+is\s+not\s+following\s+the\s+required\s+procedure(\s+correctly)?\.?$", text_lower.strip())
        or ("not following procedure" in text_lower and not re.search(r"\b(sop|bal-|qc-|doc-|\d{2,})\b", text_lower))
    ):
        return InvestigationMode.LOW_SPECIFICITY

    return InvestigationMode.NORMAL


def classify_evidence_completeness(
    finding_text: str,
    evidence_ledger: list[EvidenceItem] | list[EvidenceClaim] | None = None,
    conflicts: list[EvidenceConflict] | None = None,
    referenced_docs: list[ReferencedDocumentInfo] | None = None,
) -> EvidenceCompleteness:
    """Classify the evidence completeness independently of individual proposition verification."""
    if conflicts and len(conflicts) > 0:
        return EvidenceCompleteness.CONFLICTED
    if referenced_docs and any(
        getattr(d, "reference_status", "") == "REFERENCED_UNAVAILABLE" for d in referenced_docs
    ):
        return EvidenceCompleteness.PARTIAL
    if re.search(r"\b(not\s+available\s+to\s+the\s+ai|unavailable\s+to\s+the\s+ai|report\s+was\s+not\s+available)\b", (finding_text or "").lower()):
        return EvidenceCompleteness.PARTIAL
    return EvidenceCompleteness.COMPLETE


def build_propositions_from_ledger(
    finding_text: str,
    evidence_ledger: list[EvidenceItem] | list[EvidenceClaim],
    conflicts: list[EvidenceConflict] | None = None,
) -> list[Proposition]:
    """Decompose extracted evidence into formal Proposition models with explicit CausalLevels."""
    propositions: list[Proposition] = []
    pid_counter = 1

    for item in evidence_ledger:
        claim_text = getattr(item, "claim", getattr(item, "text", str(item)))
        status = getattr(item, "status", EvidenceStatus.UNKNOWN)
        speaker = getattr(item, "speaker", None)
        claim_id = getattr(item, "claim_id", f"E{pid_counter}")
        attribution = getattr(item, "attribution", None)

        # Classify proposition type and causal level
        prop_type = PropositionType.OBSERVATION
        causal_lvl = CausalLevel.L0_OBSERVATION
        supp_lvl = SupportLevel.UNKNOWN

        claim_low = claim_text.lower()
        if re.search(r"\b(not\s+available|unavailable|could\s+not\s+be\s+located)\b", claim_low) and any(w in claim_low for w in ("report", "attachment", "record", "document", "ai")):
            prop_type = PropositionType.EVIDENCE_STATE
            causal_lvl = CausalLevel.EVIDENCE_STATE
            supp_lvl = SupportLevel.VERIFIED
            status = EvidenceStatus.VERIFIED
        elif "referenced" in claim_low or "attached" in claim_low or "cited" in claim_low:
            prop_type = PropositionType.DOCUMENT_REFERENCE
            causal_lvl = CausalLevel.L0_OBSERVATION
            supp_lvl = SupportLevel.VERIFIED
            status = EvidenceStatus.VERIFIED
        elif status == EvidenceStatus.VERIFIED:
            supp_lvl = SupportLevel.VERIFIED
            causal_lvl = CausalLevel.L0_OBSERVATION
            prop_type = PropositionType.OBSERVATION
        elif status == EvidenceStatus.REPORTED:
            supp_lvl = SupportLevel.REPORTED
            causal_lvl = CausalLevel.L2_REPORTED_MECHANISM
            prop_type = PropositionType.REPORTED_MECHANISM
        elif status == EvidenceStatus.INFERRED:
            supp_lvl = SupportLevel.POSSIBLE
            causal_lvl = CausalLevel.L3_IMMEDIATE_MECHANISM
            prop_type = PropositionType.IMMEDIATE_MECHANISM
        elif status == EvidenceStatus.CONTRADICTED:
            supp_lvl = SupportLevel.CONTRADICTED
            causal_lvl = CausalLevel.EVIDENCE_STATE
            prop_type = PropositionType.CONFLICTED_PROPOSITION

        source_type = EpistemicSource.AUDIT_OBSERVATION
        if attribution == ClaimAttribution.SYSTEM_EVIDENCE:
            source_type = EpistemicSource.SYSTEM_RECORD
        elif attribution in (ClaimAttribution.PERSON_REPORTED, ClaimAttribution.SUPERVISOR_REPORTED):
            source_type = EpistemicSource.REPORTED_STATEMENT
        elif attribution == ClaimAttribution.AI_INFERENCE or status == EvidenceStatus.INFERRED:
            source_type = EpistemicSource.INFERRED
        elif attribution == ClaimAttribution.DOCUMENTARY_EVIDENCE:
            source_type = EpistemicSource.OBJECTIVE_RECORD

        # Determine statement status vs underlying event status
        statement_status = "VERIFIED"
        underlying_event_status = "UNKNOWN"
        if source_type in (EpistemicSource.OBJECTIVE_RECORD, EpistemicSource.SYSTEM_RECORD) and status == EvidenceStatus.VERIFIED:
            underlying_event_status = "VERIFIED"
        elif source_type == EpistemicSource.AUDIT_OBSERVATION:
            underlying_event_status = "UNKNOWN"  # Audit assertion alone does not verify underlying event
        elif status == EvidenceStatus.REPORTED:
            statement_status = "REPORTED"
            underlying_event_status = "UNKNOWN"

        prop = Proposition(
            id=f"P{pid_counter}",
            statement=claim_text,
            type=prop_type,
            causal_level=causal_lvl,
            support_level=supp_lvl,
            source_type=source_type,
            supporting_evidence_ids=[claim_id],
            contradicting_evidence_ids=[],
            status=getattr(status, "value", str(status)),
            speaker=speaker,
            statement_status=statement_status,
            underlying_event_status=underlying_event_status,
        )
        propositions.append(prop)
        pid_counter += 1

    return propositions
