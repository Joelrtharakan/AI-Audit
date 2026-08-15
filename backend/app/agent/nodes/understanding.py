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


def _fallback_extraction_result(finding_text: str):
    """Deterministic degraded-mode extraction: splits the finding into
    VERIFIED facts AND recovers REPORTED attributed statements via
    structural attribution patterns (app.services.attribution_extraction),
    instead of collapsing every sentence to a plain VERIFIED fact. Losing
    the REPORTED/attribution distinction here is exactly what causes
    degraded mode to discard a causal mechanism that was already explicitly
    present in the finding text (e.g. "X stated that they were unaware...")."""
    from app.models.analysis import AttributedStatement, ExtractionResult
    from app.services.attribution_extraction import split_facts_and_attributed_statements

    facts, attributed = split_facts_and_attributed_statements(finding_text)
    return ExtractionResult(
        stated_facts=facts,
        attributed_statements=[AttributedStatement(**a) for a in attributed],
    )


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
        extraction = _fallback_extraction_result(request.finding_text)
        trace.append(AgentTraceStep.warn(
            f"Extraction LLM call failed — falling back to {len(extraction.stated_facts)} sentence-level "
            f"facts and {len(extraction.attributed_statements)} structurally-recovered attributed "
            "statements split directly from the finding text so downstream analysis isn't starved of evidence"
        ))
        errors.append(f"Extraction error: {res_extraction}")
    else:
        extraction = res_extraction
        trace.append(AgentTraceStep.ok(
            f"Finding extracted: {len(extraction.stated_facts)} facts, "
            f"{len(extraction.attributed_statements)} attributed statements, "
            f"{len(extraction.referenced_records)} referenced records"
        ))

    # Populate initial evidence ledger directly from finding facts (VERIFIED) & attributed statements (REPORTED)
    # Apply instruction detector guard to ensure prompt injection instructions are stripped from evidence ledger
    from app.services.instruction_detector import is_instruction
    ledger = list(state.get("evidence_ledger", []))
    from app.models.agent import EvidenceItem, EvidenceStatus
    if extraction:
        if extraction.stated_facts:
            for fact in extraction.stated_facts:
                if is_instruction(fact):
                    trace.append(AgentTraceStep.warn(f"Untrusted instruction in finding ignored: {fact!r}"))
                    continue
                if not any(e.claim == fact for e in ledger):
                    ledger.append(EvidenceItem(
                        claim=fact,
                        source="AUDITOR_FINDING",
                        source_reference="Auditor finding text",
                        status=EvidenceStatus.VERIFIED,
                        relevance="HIGH",
                        notes="Fact stated directly in auditor finding text",
                    ))
        if extraction.attributed_statements:
            for stmt in extraction.attributed_statements:
                if isinstance(stmt, dict):
                    speaker = stmt.get("speaker", "")
                    claim = stmt.get("claim", "")
                else:
                    speaker = getattr(stmt, "speaker", "")
                    claim = getattr(stmt, "claim", "")
                if is_instruction(claim):
                    trace.append(AgentTraceStep.warn(f"Untrusted instruction in attributed statement ignored: {claim!r}"))
                    continue
                text = f"{speaker}: {claim}" if speaker else str(claim)
                if text and not any(e.claim == text for e in ledger):
                    ledger.append(EvidenceItem(
                        claim=text,
                        source="REPORTED_STATEMENT",
                        source_reference="Attributed statement in finding text",
                        status=EvidenceStatus.REPORTED,
                        relevance="HIGH",
                        notes="Reported statement — unverified causal explanation",
                    ))

    # Build CanonicalFindingState as the single source of truth for all downstream nodes (Section 1)
    from app.models.agent import CanonicalFindingState
    fact_claims = [e.claim for e in ledger if e.status == EvidenceStatus.VERIFIED]
    reported_claims = [e.claim for e in ledger if e.status == EvidenceStatus.REPORTED]

    # Semantic subject/condition resolution (Section 2 & 3): prefer the LLM's
    # own structured extraction when it produced a grounded, non-pronoun
    # subject; otherwise fall back to the deterministic structural extractor.
    # Neither ever cuts a sentence at the first "was"/"were" -- that naive
    # approach is what previously collapsed findings like "During the
    # internal audit, it was observed that X was not completed" into the
    # framing fragment "During the internal audit, it" whenever the framing
    # clause itself contained a "was".
    from app.services.semantic_subject import resolve_deviation
    from app.services.text_grounding import phrase_is_grounded, significant_words

    source_words = significant_words(request.finding_text)
    llm_subject = extraction.deviation_subject if extraction else None
    if llm_subject and phrase_is_grounded(llm_subject, source_words) and llm_subject.strip().lower() not in {"it", "this", "that", "the audit", "the inspection"}:
        deviation_subject = llm_subject.strip()
        deviation_condition = (extraction.deviation_condition or "UNKNOWN") if extraction else "UNKNOWN"
        deviation_actor = extraction.deviation_actor if extraction else None
        deviation_date = extraction.timeframe if extraction else None
    else:
        resolved = resolve_deviation(request.finding_text, fact_claims)
        deviation_subject = resolved.subject
        deviation_condition = resolved.condition or "UNKNOWN"
        deviation_actor = resolved.actor
        deviation_date = resolved.date

    if not deviation_subject:
        # Genuinely nothing extractable -- say so plainly rather than
        # injecting a generic placeholder that reads as a real entity.
        deviation_subject = "UNKNOWN — no affected object could be isolated from the finding text"

    observed_deviation = deviation_subject
    if deviation_condition and deviation_condition != "UNKNOWN":
        observed_deviation = f"{deviation_subject} — {deviation_condition}"

    # Immediate mechanism extraction (Layer 2): does the finding/evidence
    # already state HOW the deviation happened (an action-level claim, e.g.
    # "the check was missed"), as opposed to just describing the artifact's
    # state (Layer 1, e.g. "the log was incomplete")? Structural detection
    # only -- see app/agent/causal_guard.py.
    from app.agent.causal_guard import extract_immediate_mechanism
    mechanism = extract_immediate_mechanism(reported_claims, fact_claims)

    canonical_state = CanonicalFindingState(
        raw_finding=request.finding_text,
        observed_deviation=observed_deviation,
        deviation_condition=deviation_condition,
        facts=fact_claims,
        reported_statements=reported_claims,
        unknowns=["Root cause unconfirmed from initial evidence"],
        affected_objects=[deviation_subject],
        affected_period=deviation_date or "UNKNOWN",
        actor=deviation_actor,
        immediate_mechanism=mechanism.statement,
        immediate_mechanism_status=mechanism.status,
        prompt_injection_detected=is_instruction(request.finding_text),
    )


    # Deterministically enforce Observation Quality Sufficiency (Section 1 & 2):
    # If a concrete affected object and observed deviation exist, quality is SUFFICIENT.
    # Root cause uncertainty must NEVER downgrade observation quality.
    from app.models.analysis import ObservationQualityResult, ObservationQualityStatus
    if not deviation_subject.startswith("UNKNOWN") and len(observed_deviation.strip()) >= 5:
        # Unless text is extremely vague ("something was wrong"), mark as SUFFICIENT
        if not re.search(r"^(something|anything|stuff)\s+(was|is)\s+(wrong|bad)", request.finding_text.strip(), re.IGNORECASE):
            quality = ObservationQualityResult(
                status=ObservationQualityStatus.SUFFICIENT,
                missing_information=[],
            )

    return {
        **state,
        "observation_quality": quality,
        "extraction": extraction,

        "canonical_finding_state": canonical_state,
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
