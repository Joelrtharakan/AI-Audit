"""Node 1: understand_finding

Runs the observation quality check and structured extraction on the finding text.
Reuses the existing extraction.py and observation_quality.py services, which are
already well-validated and have their own grounding checks.

This node never fails hard — on LLM errors it fails safe to INSUFFICIENT quality
and empty extraction (matching the existing pipeline's behavior).
"""

from __future__ import annotations

import logging
import re

from app.agent.state import AgentState
from app.models.agent import AgentTraceStep
from app.services.extraction import extract_finding
from app.services.llm_client import LLMError, get_llm_client
from app.services.observation_quality import check_observation_quality

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentence_fallback_facts(finding_text: str) -> list[str]:
    """Used only when the extraction LLM call fails entirely (e.g. it
    hallucinated an ungrounded entity on every retry). Extraction failing
    must never leave the evidence ledger completely empty -- every
    downstream node (RCA, investigation, impact, CAPA) reasons over the
    ledger, and an empty ledger starves the whole analysis into generic,
    low-value output even though the finding text itself is still right
    here and fully trustworthy. This is a deterministic, non-LLM fallback:
    just split the finding into sentences as raw stated facts."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(finding_text.strip()) if s.strip()]


async def understand_finding_node(state: AgentState) -> AgentState:
    """Load the finding, run extraction and observation quality check."""
    request = state["request"]
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))

    client = get_llm_client()

    # Parallelize observation quality and extraction calls for performance
    import asyncio
    quality_task = check_observation_quality(request.finding_text, client)
    extraction_task = extract_finding(request.finding_text, client)

    results = await asyncio.gather(quality_task, extraction_task, return_exceptions=True)
    
    res_quality, res_extraction = results[0], results[1]

    if isinstance(res_quality, Exception):
        logger.warning("Observation quality check failed: %s", res_quality)
        from app.models.analysis import ObservationQualityResult, ObservationQualityStatus
        quality = ObservationQualityResult(
            status=ObservationQualityStatus.INSUFFICIENT,
            missing_information=["Quality check unavailable — LLM error during assessment."],
        )
        trace.append(AgentTraceStep.warn("Observation quality check failed — defaulting to INSUFFICIENT"))
        errors.append(f"Quality check error: {res_quality}")
    else:
        quality = res_quality
        trace.append(AgentTraceStep.ok(
            f"Observation quality assessed: {quality.status.value}"
        ))

    if quality.missing_information:
        for gap in quality.missing_information:
            trace.append(AgentTraceStep.warn(f"Missing information: {gap}"))

    if isinstance(res_extraction, Exception):
        logger.warning("Finding extraction failed: %s", res_extraction)
        from app.models.analysis import ExtractionResult
        fallback_facts = _sentence_fallback_facts(request.finding_text)
        extraction = ExtractionResult(stated_facts=fallback_facts)
        trace.append(AgentTraceStep.warn(
            f"Extraction LLM call failed — falling back to {len(fallback_facts)} sentence-level "
            "facts split directly from the finding text so downstream analysis isn't starved of evidence"
        ))
        errors.append(f"Extraction error: {res_extraction}")
    else:
        extraction = res_extraction
        trace.append(AgentTraceStep.ok(
            f"Finding extracted: {len(extraction.stated_facts)} facts, "
            f"{len(extraction.attributed_statements)} attributed statements, "
            f"{len(extraction.referenced_records)} referenced records"
        ))

    # Populate initial evidence ledger directly from finding facts
    ledger = list(state.get("evidence_ledger", []))
    from app.models.agent import EvidenceItem, EvidenceStatus
    if extraction and extraction.stated_facts:
        for fact in extraction.stated_facts:
            if not any(e.claim == fact for e in ledger):
                ledger.append(EvidenceItem(
                    claim=fact,
                    source="AUDITOR_FINDING",
                    source_reference="Auditor finding text",
                    status=EvidenceStatus.VERIFIED,
                    relevance="HIGH",
                    notes="Fact stated directly in auditor finding text",
                ))

    return {
        **state,
        "observation_quality": quality,
        "extraction": extraction,
        "trace": trace,
        "errors": errors,
        "iteration_count": state.get("iteration_count", 0),
        "tool_call_count": state.get("tool_call_count", 0),
        "critic_iteration": state.get("critic_iteration", 0),
        "evidence_ledger": ledger,
        "evidence_gaps": state.get("evidence_gaps", []),
        "tool_results": state.get("tool_results", {}),
        "completed_tools": state.get("completed_tools", []),
        "contributing_factors": state.get("contributing_factors", []),
    }
