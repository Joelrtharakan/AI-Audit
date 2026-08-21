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

        # Extract atomic dimensions for Proposition
        subject_cand = getattr(item, "subject", None)
        pred_cand = getattr(item, "predicate", None)

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
            subject=subject_cand,
            predicate=pred_cand,
            speaker=speaker,
            statement_status=statement_status,
            underlying_event_status=underlying_event_status,
        )
        propositions.append(prop)
        pid_counter += 1

    return propositions


def build_semantic_graph(
    finding_text: str,
    evidence_claims: list[EvidenceClaim] | list[EvidenceItem],
    propositions: list[Proposition],
    conflicts: list[EvidenceConflict] | None = None,
) -> SemanticGraph:
    """Build the formal Semantic Graph with atomic entity/event/state nodes and directed relations."""
    from app.models.agent import (
        SemanticEdge,
        SemanticGraph,
        SemanticNode,
        SemanticNodeType,
        SemanticRelationType,
    )
    from app.services.semantic_subject import extract_actors, extract_entities

    nodes_by_label: dict[str, SemanticNode] = {}
    edges: list[SemanticEdge] = []
    n_idx = 1
    e_idx = 1

    def _get_or_create_node(
        label: str,
        node_type: SemanticNodeType = SemanticNodeType.ENTITY,
        status: EvidenceStatus = EvidenceStatus.UNKNOWN,
        source_type: EpistemicSource = EpistemicSource.AUDIT_OBSERVATION,
        claim_id: str | None = None,
    ) -> str:
        nonlocal n_idx
        clean_lbl = label.strip()
        key = clean_lbl.lower()
        if key in nodes_by_label:
            node = nodes_by_label[key]
            if node.node_type == SemanticNodeType.ENTITY and node_type != SemanticNodeType.ENTITY:
                node.node_type = node_type
            if claim_id and claim_id not in node.source_claim_ids:
                node.source_claim_ids.append(claim_id)
            return node.id
        node_id = f"N{n_idx}"
        n_idx += 1
        node = SemanticNode(
            id=node_id,
            label=clean_lbl,
            node_type=node_type,
            epistemic_status=status,
            provenance=source_type,
            source_claim_ids=[claim_id] if claim_id else [],
        )
        nodes_by_label[key] = node
        return node_id

    # 1. Base entities and actors from finding
    for act in extract_actors(finding_text):
        _get_or_create_node(act, SemanticNodeType.ACTOR, EvidenceStatus.VERIFIED, EpistemicSource.AUDIT_OBSERVATION)

    for ent in extract_entities(finding_text):
        _get_or_create_node(ent, SemanticNodeType.ENTITY, EvidenceStatus.VERIFIED, EpistemicSource.AUDIT_OBSERVATION)

    # 1b. Base nodes from claim subjects & speaker
    for claim in evidence_claims:
        subj = getattr(claim, "subject", None)
        spkr = getattr(claim, "speaker", None)
        c_status = getattr(claim, "status", EvidenceStatus.UNKNOWN)
        c_id = getattr(claim, "claim_id", None)
        if subj and len(subj.strip()) >= 3:
            _get_or_create_node(subj, SemanticNodeType.ENTITY, c_status, EpistemicSource.AUDIT_OBSERVATION, c_id)
        if spkr and len(spkr.strip()) >= 3:
            _get_or_create_node(spkr, SemanticNodeType.ACTOR, c_status, EpistemicSource.AUDIT_OBSERVATION, c_id)

    # 2. Add nodes and edges per claim / proposition
    for claim in evidence_claims:
        c_text = getattr(claim, "claim", getattr(claim, "text", str(claim)))
        c_id = getattr(claim, "claim_id", None)
        c_status = getattr(claim, "status", EvidenceStatus.UNKNOWN)
        c_speaker = getattr(claim, "speaker", None)
        c_low = c_text.lower()

        # Epistemic source
        source_type = EpistemicSource.AUDIT_OBSERVATION
        attr = getattr(claim, "attribution", None)
        if attr == ClaimAttribution.SYSTEM_EVIDENCE:
            source_type = EpistemicSource.SYSTEM_RECORD
        elif attr in (ClaimAttribution.PERSON_REPORTED, ClaimAttribution.SUPERVISOR_REPORTED):
            source_type = EpistemicSource.REPORTED_STATEMENT
        elif attr == ClaimAttribution.DOCUMENTARY_EVIDENCE:
            source_type = EpistemicSource.OBJECTIVE_RECORD

        # Structural relation & participant extraction across arbitrary domains
        # Normative / Requirement / Compliance relations
        m_req = re.search(r"\b(?:as\s+required\s+by|per|under|in\s+accordance\s+with|prescribed\s+by|mandated\s+by)\s+(?:procedure\s+|sop\s+|standard\s+|specification\s+|policy\s+)?([A-Z0-9-]+|[a-z0-9\s-]+?(?=\.|$|,|\s+and\b))", c_text, re.IGNORECASE)
        m_req_term = re.search(r"\b(?:procedure\s+|sop\s+|standard\s+|specification\s+|policy\s+)([A-Z0-9-]+)\b", c_text, re.IGNORECASE)
        if m_req or m_req_term:
            req_label = (m_req.group(1).strip() if m_req else m_req_term.group(1).strip())
            req_id = _get_or_create_node(req_label, SemanticNodeType.REQUIREMENT, EvidenceStatus.VERIFIED, EpistemicSource.AUDIT_OBSERVATION, c_id)
            target_lbl = subj if subj and req_label != subj else (c_speaker or "Process / Activity")
            target_id = _get_or_create_node(target_lbl, SemanticNodeType.PROCESS, c_status, source_type, c_id)
            if any(w in c_low for w in ("not completed", "missed", "deviat", "violat", "failed", "omitted", "exceeded", "below", "incomplete", "without", "lacked", "not")):
                edges.append(SemanticEdge(
                    id=f"E{e_idx}",
                    source_id=target_id,
                    target_id=req_id,
                    relation_type=SemanticRelationType.VIOLATES,
                    epistemic_status=c_status,
                    provenance=source_type,
                    source_claim_ids=[c_id] if c_id else [],
                    notes="Normative compliance deviation",
                ))
                e_idx += 1
            else:
                edges.append(SemanticEdge(
                    id=f"E{e_idx}",
                    source_id=req_id,
                    target_id=target_id,
                    relation_type=SemanticRelationType.GOVERNS,
                    epistemic_status=c_status,
                    provenance=source_type,
                    source_claim_ids=[c_id] if c_id else [],
                    notes="Normative governance",
                ))
                e_idx += 1

        elif any(w in c_low for w in ("lacks", "missing", "omitted", "without", "absent", "no record")) and any(w in c_low for w in ("attribute", "parameter", "identifier", "code", "tag", "serial", "batch", "lot", "label", "date", "entry", "signature", "sign-off")):
            ent_id = _get_or_create_node(subj or "Entity", SemanticNodeType.ENTITY, c_status, source_type, c_id)
            attr_id = _get_or_create_node("Required Attribute", SemanticNodeType.ATTRIBUTE, c_status, source_type, c_id)
            edges.append(SemanticEdge(
                id=f"E{e_idx}",
                source_id=ent_id,
                target_id=attr_id,
                relation_type=SemanticRelationType.LACKS_REQUIRED_ATTRIBUTE,
                epistemic_status=c_status,
                provenance=source_type,
                source_claim_ids=[c_id] if c_id else [],
                notes="Missing required attribute",
            ))
            e_idx += 1

        elif "deliver" in c_low or "dispatch" in c_low or "sent" in c_low or "transmitt" in c_low:
            sys_id = _get_or_create_node(c_speaker or "System", SemanticNodeType.ENTITY, c_status, source_type, c_id)
            target_lbl = subj or "Item"
            target_id = _get_or_create_node(target_lbl, SemanticNodeType.ENTITY, c_status, source_type, c_id)
            edges.append(SemanticEdge(
                id=f"E{e_idx}",
                source_id=sys_id,
                target_id=target_id,
                relation_type=SemanticRelationType.TRANSMITTED_TO,
                epistemic_status=c_status,
                provenance=source_type,
                source_claim_ids=[c_id] if c_id else [],
                notes="Transmission / dispatch recorded",
            ))
            e_idx += 1

        elif "receiv" in c_low:
            recp_id = _get_or_create_node(c_speaker or "Recipient", SemanticNodeType.ACTOR, c_status, source_type, c_id)
            target_lbl = subj or "Item"
            target_id = _get_or_create_node(target_lbl, SemanticNodeType.ENTITY, c_status, source_type, c_id)
            edges.append(SemanticEdge(
                id=f"E{e_idx}",
                source_id=recp_id,
                target_id=target_id,
                relation_type=SemanticRelationType.RECEIVED_BY,
                epistemic_status=c_status,
                provenance=source_type,
                source_claim_ids=[c_id] if c_id else [],
                notes="Receipt account",
            ))
            e_idx += 1

        elif "access" in c_low or "opened" in c_low or "viewed" in c_low:
            recp_id = _get_or_create_node(c_speaker or "Recipient", SemanticNodeType.ACTOR, c_status, source_type, c_id)
            target_lbl = subj or "Record"
            target_id = _get_or_create_node(target_lbl, SemanticNodeType.RECORD, c_status, source_type, c_id)
            edges.append(SemanticEdge(
                id=f"E{e_idx}",
                source_id=recp_id,
                target_id=target_id,
                relation_type=SemanticRelationType.ACCESSED_BY,
                epistemic_status=c_status,
                provenance=source_type,
                source_claim_ids=[c_id] if c_id else [],
            ))
            e_idx += 1

        elif "acknowledg" in c_low or "sign-off" in c_low or "signature" in c_low:
            recp_id = _get_or_create_node(c_speaker or "Personnel", SemanticNodeType.ACTOR, c_status, source_type, c_id)
            target_lbl = subj or "Record"
            target_id = _get_or_create_node(target_lbl, SemanticNodeType.RECORD, c_status, source_type, c_id)
            edges.append(SemanticEdge(
                id=f"E{e_idx}",
                source_id=recp_id,
                target_id=target_id,
                relation_type=SemanticRelationType.ACKNOWLEDGED_BY,
                epistemic_status=c_status,
                provenance=source_type,
                source_claim_ids=[c_id] if c_id else [],
            ))
            e_idx += 1

        elif "calibrat" in c_low:
            equip_id = _get_or_create_node(subj or "Equipment", SemanticNodeType.ENTITY, c_status, source_type, c_id)
            rec_id = _get_or_create_node("Calibration Record", SemanticNodeType.RECORD, c_status, source_type, c_id)
            edges.append(SemanticEdge(
                id=f"E{e_idx}",
                source_id=rec_id,
                target_id=equip_id,
                relation_type=SemanticRelationType.VERIFIES,
                epistemic_status=c_status,
                provenance=source_type,
                source_claim_ids=[c_id] if c_id else [],
            ))
            e_idx += 1

        elif "train" in c_low:
            actor_id = _get_or_create_node(c_speaker or "Personnel", SemanticNodeType.ACTOR, c_status, source_type, c_id)
            req_id = _get_or_create_node("Training Requirement", SemanticNodeType.REQUIREMENT, c_status, source_type, c_id)
            edges.append(SemanticEdge(
                id=f"E{e_idx}",
                source_id=req_id,
                target_id=actor_id,
                relation_type=SemanticRelationType.APPLIES_TO,
                epistemic_status=c_status,
                provenance=source_type,
                source_claim_ids=[c_id] if c_id else [],
            ))
            e_idx += 1

    # 3. Explicitly represent conflicting evidence relations as unresolved conflict edges
    if conflicts:
        for conf in conflicts:
            c_prop = getattr(conf, "proposition", "Disputed relation")
            c_claims = getattr(conf, "conflicting_claim_ids", [])
            n1 = _get_or_create_node(c_prop, SemanticNodeType.EVENT, EvidenceStatus.UNKNOWN, EpistemicSource.UNKNOWN_SOURCE)
            edges.append(SemanticEdge(
                id=f"E{e_idx}",
                source_id=n1,
                target_id=n1,
                relation_type=SemanticRelationType.RELATES_TO,
                epistemic_status=EvidenceStatus.UNKNOWN,
                provenance=EpistemicSource.UNKNOWN_SOURCE,
                source_claim_ids=c_claims,
                notes=f"Unresolved conflict: {c_prop}",
            ))
            e_idx += 1

    return SemanticGraph(nodes=list(nodes_by_label.values()), edges=edges)
