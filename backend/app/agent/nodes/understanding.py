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
from app.config import get_settings
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
    """Load the finding, run deterministic observation quality assessment and structured extraction."""
    request = state["request"]
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))

    settings = get_settings()
    client = get_llm_client(timeout_seconds=settings.ollama_extraction_timeout_seconds)

    # Deterministic observation quality assessment (0ms fast path)
    from app.models.analysis import ObservationQualityResult, ObservationQualityStatus
    finding_text = request.finding_text.strip()
    words = finding_text.split()
    if len(words) < 8:
        quality = ObservationQualityResult(
            status=ObservationQualityStatus.INSUFFICIENT,
            missing_information=["Observation is too brief to establish a verifiable deviation."],
        )
    else:
        quality = ObservationQualityResult(
            status=ObservationQualityStatus.SUFFICIENT,
            missing_information=[],
        )

    trace.append(AgentTraceStep.ok(f"Observation quality assessed deterministically: {quality.status.value}"))

    # Fast deterministic semantic extraction (0ms fast path)
    extraction = _fallback_extraction_result(request.finding_text)
    trace.append(AgentTraceStep.ok(
        f"Semantic extraction recovered {len(extraction.stated_facts)} facts and "
        f"{len(extraction.attributed_statements)} attributed statements directly from finding text."
    ))

    if quality.missing_information:
        for gap in quality.missing_information:
            trace.append(AgentTraceStep.warn(f"Missing information: {gap}"))

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
                from app.agent.claim_extractor import _SYSTEM_RECORD_SPEAKER_RE
                is_sys = bool(_SYSTEM_RECORD_SPEAKER_RE.search(speaker)) if speaker else False
                if text and not any(e.claim == text for e in ledger):
                    ledger.append(EvidenceItem(
                        claim=text,
                        source="SYSTEM_RECORD" if is_sys else "REPORTED_STATEMENT",
                        source_reference="System record in finding text" if is_sys else "Attributed statement in finding text",
                        status=EvidenceStatus.VERIFIED if is_sys else EvidenceStatus.REPORTED,
                        relevance="HIGH",
                        notes="System record verification" if is_sys else "Reported statement — unverified causal explanation",
                    ))

    # Build CanonicalFindingState as the single source of truth for all downstream nodes (Section 1)
    from app.models.agent import CanonicalFindingState
    fact_claims = [e.claim for e in ledger if e.status == EvidenceStatus.VERIFIED]
    reported_claims = [e.claim for e in ledger if e.status == EvidenceStatus.REPORTED]

    # Semantic subject/condition resolution: the deterministic structural
    # resolver (resolve_deviation -> extract_semantic_subject, including
    # its entity-noun-phrase extraction: "balance BAL-014", "pipette
    # P-200") is the SOLE authoritative producer of finding_subject/
    # affected_object. It previously competed with the LLM's own
    # extraction.deviation_subject, which won whenever it merely shared
    # vocabulary with the finding ("grounded") -- a much weaker bar than
    # actual correctness, and the source of a real production defect: the
    # LLM proposed "personnel calibration status for calibration
    # certificate" for a finding about "balance BAL-014", which passed the
    # groundedness check (shares "calibration"/"certificate") despite
    # confusing actor + process + document with the actual entity. The LLM
    # subject is now used ONLY as a last resort, when the deterministic
    # resolver itself produced no usable result -- never to override a
    # result the authoritative resolver actually found.
    from app.services.semantic_subject import resolve_deviation, validate_semantic_subject
    from app.services.text_grounding import phrase_is_grounded, significant_words

    resolved = resolve_deviation(request.finding_text, fact_claims)
    source_words = significant_words(request.finding_text)
    _DEGRADED_SUBJECTS = {"process compliance", None, ""}
    resolver_succeeded = (resolved.finding_subject or resolved.subject) not in _DEGRADED_SUBJECTS

    if resolver_succeeded:
        deviation_subject = resolved.finding_subject or resolved.subject
        deviation_condition = resolved.condition or "UNKNOWN"
        deviation_actor = resolved.actor or (extraction.deviation_actor if extraction else None)
        deviation_date = resolved.date or (extraction.timeframe if extraction else None)
    else:
        llm_subject = extraction.deviation_subject if extraction else None
        if llm_subject and phrase_is_grounded(llm_subject, source_words) and validate_semantic_subject(llm_subject):
            deviation_subject = llm_subject.strip()
            deviation_condition = (extraction.deviation_condition or "UNKNOWN") if extraction else "UNKNOWN"
            deviation_actor = extraction.deviation_actor if extraction else None
            deviation_date = extraction.timeframe if extraction else None
        else:
            deviation_subject = resolved.finding_subject or resolved.subject or "UNKNOWN — no affected object could be isolated from the finding text"
            deviation_condition = resolved.condition or "UNKNOWN"
            deviation_actor = resolved.actor or (extraction.deviation_actor if extraction else None)
            deviation_date = resolved.date or (extraction.timeframe if extraction else None)

    if not deviation_subject or not validate_semantic_subject(deviation_subject):
        deviation_subject = resolved.finding_subject or "process compliance"

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

    # Claim-level decomposition with full provenance (Phase 2): decomposes
    # the finding into individual claims, each with its own attribution and
    # status, then detects conflicts between claims about the same
    # proposition.  This is the foundation of the canonical causal evidence
    # model -- every downstream node reasons over these claims, not the raw
    # finding text.
    from app.agent.claim_extractor import detect_evidence_conflicts, extract_claims
    evidence_claims = extract_claims(request.finding_text, ledger)
    evidence_conflicts = detect_evidence_conflicts(evidence_claims)
    if evidence_conflicts:
        trace.append(AgentTraceStep.warn(
            f"Evidence conflict(s) detected: {len(evidence_conflicts)} — "
            + "; ".join(c.proposition for c in evidence_conflicts)
        ))
    # If evidence conflicts exist and the mechanism was derived from
    # REPORTED claims, the mechanism status must reflect the conflict.
    if evidence_conflicts and mechanism.status == "REPORTED":
        mechanism.status = "UNKNOWN"
        trace.append(AgentTraceStep.warn(
            "Mechanism status downgraded to UNKNOWN due to evidence conflict(s) "
            "— conflicting reported statements prevent establishing the mechanism"
        ))

    from app.services.semantic_subject import resolve_referenced_documents
    from app.models.agent import ReferencedDocumentInfo
    referenced_documents = [
        ReferencedDocumentInfo(document_type=d["document_type"], raw_span=d["raw_span"])
        for d in resolve_referenced_documents(request.finding_text)
    ]
    if referenced_documents:
        trace.append(AgentTraceStep.warn(
            "Referenced-but-unavailable document(s) detected: "
            + "; ".join(d.document_type for d in referenced_documents)
            + " — content treated as UNKNOWN, not usable as evidence"
        ))

    from app.agent.recurrence_guard import detect_recurrence
    recurrence = detect_recurrence(request.finding_text)
    if recurrence.is_recurring:
        trace.append(AgentTraceStep.warn(
            "Recurrence signal detected: " + (recurrence.rationale or "a similar finding was previously identified")
        ))

    from app.agent.proposition_engine import build_propositions_from_ledger, classify_investigation_mode
    investigation_mode = classify_investigation_mode(
        request.finding_text, evidence_claims, evidence_conflicts, referenced_documents
    )
    propositions = build_propositions_from_ledger(request.finding_text, evidence_claims, evidence_conflicts)

    canonical_state = CanonicalFindingState(
        raw_finding=request.finding_text,
        finding_subject=deviation_subject,
        affected_object=resolved.affected_object or deviation_subject,
        affected_process=resolved.affected_process or "UNKNOWN",
        affected_activity=resolved.affected_activity or "UNKNOWN",
        deviation=observed_deviation,
        observed_deviation=observed_deviation,
        deviation_condition=deviation_condition,
        facts=fact_claims,
        verified_observations=fact_claims,
        reported_statements=reported_claims,
        unknowns=["Root cause unconfirmed from initial evidence"],
        affected_objects=[deviation_subject],
        affected_period=deviation_date or "UNKNOWN",
        time_period=deviation_date or "UNKNOWN",
        actor=deviation_actor,
        actors=resolved.actors,
        entities=resolved.entities,
        immediate_mechanism=mechanism.statement,
        reported_mechanism=mechanism.statement if mechanism.status == "REPORTED" else None,
        verified_mechanism=mechanism.statement if mechanism.status == "VERIFIED" else None,
        immediate_mechanism_status=mechanism.status,
        mechanism_status=mechanism.status,
        mechanism_polarity=mechanism.polarity,
        prompt_injection_detected=is_instruction(request.finding_text),
        evidence_claims=evidence_claims,
        evidence_conflicts=evidence_conflicts,
        referenced_documents=referenced_documents,
        investigation_mode=investigation_mode,
        propositions=propositions,
        relevant_change=resolved.relevant_change,
        semantic_type=resolved.semantic_type,
        recurrence_signal=recurrence.is_recurring,
        previous_capa_referenced=recurrence.has_previous_capa_reference,
        previous_capa_status=recurrence.previous_capa_status,
        previous_capa_effectiveness=recurrence.previous_capa_effectiveness,
        recurrence_rationale=recurrence.rationale,
    )

    # Deterministic Cost & Financial Impact Analysis (Sections 26-42)
    from app.services.cost_analysis import analyze_cost_and_financial_impact
    cost_impact = analyze_cost_and_financial_impact(
        request.finding_text,
        ledger,
        evidence_claims=evidence_claims,
    )
    canonical_state.cost_impact = cost_impact
    if cost_impact and cost_impact.cost_factor_detected:
        trace.append(AgentTraceStep.ok(
            f"Cost & financial factor detected: status={cost_impact.financial_status}, "
            f"exposure={cost_impact.potential_cost_exposure or 'REQUIRES_ASSESSMENT'}"
        ))

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
        "cost_impact": cost_impact,
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

