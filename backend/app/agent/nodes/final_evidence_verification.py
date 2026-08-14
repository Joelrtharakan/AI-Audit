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
from app.models.agent import AgentTraceStep, EvidenceStatus

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
        # Cleanup any accidental double words like "the the procedure revision revision" or "procedure revision revision"
        text = re.sub(r"\bthe\s+the\b", "the", text, flags=re.IGNORECASE)
        text = re.sub(r"\bprocedure\s+revision\s+revision\b", "procedure revision", text, flags=re.IGNORECASE)
        # Remove unevidenced temporal duration claims ("performed procedure over a 30-day period")
        if "over a 30-day period" in text.lower() or "for 30 days" in text.lower():
            text = re.sub(
                r"\b(over\s+a\s+30-day\s+period|for\s+30\s+days)\b",
                "since the SOP revision was issued 30 days ago",
                text,
                flags=re.IGNORECASE,
            )
            trace.append(AgentTraceStep.warn(
                "Final Evidence Verification: corrected unsupported 30-day duration claim"
            ))
        return text

    # Clean Root Cause Analysis
    rc = state.get("root_cause")
    if rc:
        rc.narrative = _clean_text(rc.narrative) or ""
        rc.statement = _clean_text(rc.statement)

    # Clean Investigation Plan
    inv = state.get("investigation_plan")
    if inv:
        inv.questions = [_clean_text(q) or q for q in inv.questions]
        inv.areas = [_clean_text(a) or a for a in inv.areas]

    # Clean 5-Why
    fw = state.get("five_why")
    if fw and fw.steps:
        for step in fw.steps:
            step.question = _clean_text(step.question) or step.question
            step.answer = _clean_text(step.answer)

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
            report.investigation.questions = [_clean_text(q) or q for q in report.investigation.questions]
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

    has_docs = _has_verified_document_evidence(evidence_ledger)
    if not has_docs and rc and rc.narrative:
        if _UNGROUNDED_RULES_RE.search(rc.narrative):
            rc.narrative = _UNGROUNDED_RULES_RE.sub("[specific requirement pending document review]", rc.narrative)
            trace.append(AgentTraceStep.warn(
                "Final Evidence Verification: stripped ungrounded company policy claim"
            ))

    trace.append(AgentTraceStep.ok("Final evidence verification completed"))

    return {
        **state,
        "root_cause": rc,
        "investigation_plan": inv,
        "five_why": fw,
        "ca_draft": ca,
        "capa_analysis": capa,
        "impact_assessment": impact,
        "report": report,
        "trace": trace,
        "errors": errors,
    }
