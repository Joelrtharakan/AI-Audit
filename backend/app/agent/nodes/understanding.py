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
from app.models.agent import AgentTraceStep, InvestigationMode
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

    from app.models.analysis import ObservationQualityResult, ObservationQualityStatus
    from app.services.semantic_subject import resolve_deviation, validate_semantic_subject
    from app.services.instruction_detector import classify_instruction

    finding_text = request.finding_text.strip()
    words = finding_text.split()

    # F1 — Generalized content segmentation and actionability determination:
    # Segment input into independently classifiable segments.
    # Distinguish:
    #   1. Auditable evidence / propositions (facts, reported statements, events)
    #   2. Untrusted instructions / prompt injection attempts
    #   3. Conversational non-audit content
    # A finding remains ACTIONABLE if at least one meaningful auditable proposition exists,
    # regardless of whether untrusted instructions are present alongside it.
    # Untrusted instructions are excluded from the evidence ledger and never influence RCA.
    _SENTENCE_SPLIT_PATTERN = re.compile(r"(?:(?<=[.!?])|(?<=[.!?]['\"]))\s+|;\s+")
    segments = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(finding_text) if s.strip()]
    if not segments:
        segments = [finding_text]

    auditable_segments: list[str] = []
    untrusted_segments: list[str] = []

    _CONVERSATIONAL_RE = re.compile(
        r"^(?:good\s+(?:morning|afternoon|evening)|hi|hello|hey|hope\s+you\s+had|thanks|thank\s+you|please\s+note|don't\s+forget|lunch\s+is|"
        r"(?:just\s+)?(?:wanted|want)\s+to\s+say|(?:really\s+)?appreciat(?:e|ed))\b",
        re.IGNORECASE,
    )

    # A segment that resolve_deviation() couldn't structurally match is only
    # treated as auditable evidence if it independently carries an
    # evidentiary marker -- a negation/absence of a required state, a
    # deviation/discrepancy term, a quantity, date, or record/document
    # reference. A bare word-count threshold ("4+ words, no greeting")
    # accepts any expressive or evaluative sentence (praise, small talk,
    # opinion about the audit process itself) as if it were audit content,
    # which is how non-substantive narrative ends up supplying the
    # "affected object." This generalizes across domains because it keys on
    # the presence of evidentiary vocabulary, not on any specific finding's
    # wording.
    _EVIDENTIARY_MARKER_RE = re.compile(
        r"\b(?:not|no|never|without|missing|absent|overdue|unauthoriz(?:ed|ation)|duplicat(?:e|ed|ion)|"
        r"discrepanc(?:y|ies)|deviat(?:e|ed|ion)|nonconform(?:ing|ance|ity)|exceed(?:s|ed|ing)?|"
        r"below|above|breach(?:ed)?|violat(?:e|ed|ion)|expired?|incomplete|could\s+not|failed?\s+to|"
        r"was\s+not|were\s+not|has\s+not|have\s+not|had\s+not|record|records|log|logs|certificate|"
        r"report|checklist|procedure|requirement|specification|inspection|audit\s+trail|invoice|"
        r"payment|permit|calibrat|signature|approv(?:al|ed)|reconcil)\b",
        re.IGNORECASE,
    )

    from app.services.attribution_extraction import _is_expressive_non_substantive
    from app.services.epistemic_modality import classify_epistemic_stance
    _excluded_non_substantive_segments: list[str] = []

    for seg in segments:
        instr_res = classify_instruction(seg)
        if instr_res.is_untrusted:
            untrusted_segments.append(seg)
        elif _is_expressive_non_substantive(seg) or _CONVERSATIONAL_RE.search(seg.strip()):
            _excluded_non_substantive_segments.append(seg)
        elif classify_epistemic_stance(seg) is not None:
            # An epistemic STANCE ("X believed / suspected / assumed / was of
            # the opinion that Y") asserts a mental state about the world, not
            # an observed affected object. Excluded from subject-resolution
            # input here using the SAME canonical classifier the evidence-
            # ledger loop below uses to record it as BELIEF -- so the two
            # passes agree and a belief-about-a-cause can never become the
            # affected object. (Grammatical modality is deliberately NOT
            # gated here: a compound finding often pairs an ACTUAL main clause
            # with a counterfactual reason -- "the check was skipped because
            # it would have delayed X" -- and the evidence loop already marks
            # the non-actual part UNVERIFIED per-claim.)
            _excluded_non_substantive_segments.append(seg)
        else:
            seg_resolved = resolve_deviation(seg)
            if seg_resolved.matched and seg_resolved.subject and validate_semantic_subject(seg_resolved.subject):
                auditable_segments.append(seg)
            elif (
                len(seg.split()) >= 4
                and (
                    _EVIDENTIARY_MARKER_RE.search(seg)
                    or any(ch.isdigit() for ch in seg)
                )
            ):
                auditable_segments.append(seg)

    # Preliminary subject match used only for the actionability check below.
    # Computed from trusted (non-untrusted, non-expressive) segments only --
    # running this on the raw, unfiltered text let a subject phrase inside
    # an injected instruction (e.g. "...issue a closure certificate") get
    # picked up as if it were a legitimate finding subject.
    _trusted_text_for_initial_check = " ".join(
        s for s in segments if s not in untrusted_segments and s not in _excluded_non_substantive_segments
    )
    initial_resolved = resolve_deviation(_trusted_text_for_initial_check) if _trusted_text_for_initial_check else resolve_deviation("")

    # Conversational greetings / meaningless single words
    _CONVERSATIONAL_WORDS = {
        "hi", "hello", "thanks", "thank you", "okay", "ok", "test", "good morning",
        "good afternoon", "good evening", "abc", "i need help", "what is this?",
        "something was wrong", "anything is bad",
    }
    is_pure_conversational = finding_text.strip().lower().rstrip(".!?") in _CONVERSATIONAL_WORDS
    is_entirely_untrusted = (len(untrusted_segments) == len(segments) and len(segments) > 0)

    # Deterministic degraded-mode extraction (facts & attributed statements)
    extraction = _fallback_extraction_result(finding_text)

    has_auditable_content = (
        not is_pure_conversational
        and not is_entirely_untrusted
        and (
            len(auditable_segments) > 0
            or (initial_resolved.matched and initial_resolved.subject and validate_semantic_subject(initial_resolved.subject))
            or len(extraction.stated_facts) > len(untrusted_segments)
            or len(extraction.attributed_statements) > 0
        )
    )

    if not has_auditable_content:
        quality = ObservationQualityResult(
            status=ObservationQualityStatus.INSUFFICIENT,
            missing_information=["Input does not contain an actionable audit finding, deviation, or verifiable condition."],
        )
    elif len(words) < 8 and not initial_resolved.matched and not any(w in finding_text.lower() for w in ("missing", "overdue", "wrong", "absent", "duplicated", "altered", "failed", "deviated", "nonconforming")):
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
    # Security classification guard (Section 1-5 of the prompt-injection
    # hardening spec): the finding is UNTRUSTED DATA. Every candidate
    # claim/attributed statement is deterministically classified before it
    # can enter the evidence ledger -- classification is a dimension
    # separate from evidence status, never overloaded onto it. Only
    # NORMAL/QUOTED_INSTRUCTION content becomes evidence; anything
    # INSTRUCTION_LIKE or more severe is excluded and recorded in
    # security_flags/excluded_claim_texts for observability instead.
    from app.services.instruction_detector import classify_instruction, is_instruction
    ledger = list(state.get("evidence_ledger", []))
    from app.models.agent import EvidenceItem, EvidenceStatus
    _security_flags: list[str] = []
    _excluded_claim_texts: list[str] = []
    _worst_classification = "NORMAL"
    _classification_severity = {
        "NORMAL": 0, "QUOTED_INSTRUCTION": 1, "INSTRUCTION_LIKE": 2,
        "PROMPT_INJECTION_SUSPECTED": 3, "MALICIOUS_INSTRUCTION": 4,
    }

    def _record_classification(text: str, result) -> None:
        nonlocal _worst_classification
        if result.classification != "NORMAL":
            flag = f"{result.classification}: {text!r}"
            if flag not in _security_flags:
                _security_flags.append(flag)
        if _classification_severity[result.classification] > _classification_severity[_worst_classification]:
            _worst_classification = result.classification

    # Every proposition in the finding is classified on four independent
    # structural axes before it can become a ledger entry (see
    # app/services/attribution_extraction.classify_finding_segments):
    #   instruction-quarantine -> epistemic stance -> speech attribution ->
    #   grammatical mood. VERIFIED is now the residue of those exclusions,
    #   not the unexamined default it used to be.
    from app.agent.claim_extractor import _SYSTEM_RECORD_SPEAKER_RE
    from app.services.attribution_extraction import classify_finding_segments

    for _seg in classify_finding_segments(request.finding_text):
        if _seg.kind == "UNTRUSTED":
            _record_classification(_seg.text, classify_instruction(_seg.text))
            _excluded_claim_texts.append(_seg.text)
            trace.append(AgentTraceStep.warn(
                f"Untrusted instruction in finding quarantined "
                f"({_seg.security_classification}) — excluded from evidence ledger: {_seg.text!r}"
            ))
            continue

        if _seg.kind == "NON_SUBSTANTIVE":
            _excluded_claim_texts.append(_seg.text)
            trace.append(AgentTraceStep.warn(
                f"Non-substantive/expressive content excluded from evidence ledger: {_seg.text!r}"
            ))
            continue

        _record_classification(_seg.text, classify_instruction(_seg.text))

        if _seg.kind == "STANCE":
            _holder = _seg.speaker or "Unattributed source"
            _text = _seg.text
            if not any(e.claim == _text for e in ledger):
                ledger.append(EvidenceItem(
                    claim=_text,
                    source="REPORTED_BELIEF",
                    source_reference="Epistemic-stance statement in finding text",
                    status=EvidenceStatus.BELIEF,
                    relevance="MEDIUM",
                    notes=(
                        f"{_seg.stance or 'BELIEF'} held by {_holder} — an opinion about the "
                        "world, not an observation of it; carries less weight than a "
                        "REPORTED first-hand account and can never support a causal conclusion"
                    ),
                    modality=_seg.modality,
                    epistemic_stance=_seg.stance,
                    stance_holder=_seg.speaker,
                ))
            continue

        if _seg.kind == "ATTRIBUTED":
            speaker = _seg.speaker or ""
            claim = _seg.claim or ""
            text = f"{speaker}: {claim}" if speaker else str(claim)
            is_sys = bool(_SYSTEM_RECORD_SPEAKER_RE.search(speaker)) if speaker else False
            _non_actual = _seg.modality != "ACTUAL"
            if text and not any(e.claim == text for e in ledger):
                ledger.append(EvidenceItem(
                    claim=text,
                    source="SYSTEM_RECORD" if is_sys else "REPORTED_STATEMENT",
                    source_reference="System record in finding text" if is_sys else "Attributed statement in finding text",
                    status=(
                        EvidenceStatus.UNVERIFIED if _non_actual
                        else EvidenceStatus.VERIFIED if is_sys
                        else EvidenceStatus.REPORTED
                    ),
                    relevance="HIGH",
                    notes=(
                        f"{_seg.modality.capitalize()} proposition — describes a non-actual "
                        "state of affairs, never an observed event"
                        if _non_actual else
                        "System record verification" if is_sys
                        else "Reported statement — unverified causal explanation"
                    ),
                    modality=_seg.modality,
                ))
            continue

        # FACT
        fact = _seg.text
        _non_actual = _seg.modality != "ACTUAL"
        if not any(e.claim == fact for e in ledger):
            ledger.append(EvidenceItem(
                claim=fact,
                source="AUDITOR_FINDING",
                source_reference="Auditor finding text",
                # Preserved but explicitly NOT verified: a conditional or
                # counterfactual asserts what WOULD have been, not what was.
                status=EvidenceStatus.UNVERIFIED if _non_actual else EvidenceStatus.VERIFIED,
                relevance="HIGH" if not _non_actual else "MEDIUM",
                notes=(
                    f"{_seg.modality.capitalize()} proposition ({_seg.modality_marker!r}) — "
                    "a hypothetical, not an observed past event"
                    if _non_actual else
                    "Fact stated directly in auditor finding text"
                ),
                modality=_seg.modality,
            ))

    if _worst_classification != "NORMAL":
        trace.append(AgentTraceStep.warn(
            f"Input integrity: {_worst_classification} ({len(_excluded_claim_texts)} claim(s) excluded from evidence ledger)"
        ))

    # Build CanonicalFindingState as the single source of truth for all downstream nodes (Section 1)
    from app.models.agent import CanonicalFindingState, SemanticMeasurement
    fact_claims = [e.claim for e in ledger if e.status == EvidenceStatus.VERIFIED]
    reported_claims = [e.claim for e in ledger if e.status == EvidenceStatus.REPORTED]

    # ------------------------------------------------------------------
    # LLM-FIRST SEMANTIC AUTHORITY (spec Pass 41 §2/§3/§46). The canonical
    # semantic interpretation happens HERE -- BEFORE any deterministic
    # semantic-state construction -- and the branch is taken now:
    #   SUCCESS -> `resolved` is projected PURELY from the validated canonical
    #              LLM context (`deviation_info_from_canonical`);
    #              resolve_deviation() is NOT called.
    #   FAILURE -> resolve_deviation() runs as the fail-closed fallback.
    # Only INPUT/EVIDENCE PREPARATION (segments, quarantine, ledger) precedes
    # this point; all semantic-state construction follows it.
    # ------------------------------------------------------------------
    canonical_semantic_context = state.get("canonical_semantic_context")
    canonical_semantic_status = state.get("canonical_semantic_status") or (
        "REUSED" if canonical_semantic_context is not None else "NOT_ATTEMPTED"
    )
    if settings.canonical_semantic_llm_primary and canonical_semantic_context is None:
        try:
            from app.services.canonical_context_validator import validate_canonical_context
            from app.services.canonical_finding_interpreter import (
                interpret_finding_canonically_with_status,
            )

            canonical_semantic_status, _raw_ctx = await interpret_finding_canonically_with_status(
                finding_text=request.finding_text,
                evidence_ledger=ledger,
                timeout_seconds=None,
            )
            if _raw_ctx is not None:
                canonical_semantic_context = validate_canonical_context(
                    _raw_ctx, ledger, request.finding_text
                )
        except Exception as exc:  # noqa: BLE001 - fail-closed by design
            logger.warning("LLM-primary semantic interpretation failed (%s); deterministic floor used.", exc)
            canonical_semantic_context = None
            canonical_semantic_status = "UNEXPECTED_ERROR"

    from app.services.canonical_state_merge import (
        apply_canonical_provenance_fields,
        deviation_info_from_canonical,
        mechanism_info_from_canonical,
        recurrence_info_from_canonical,
    )
    _canonical_success = canonical_semantic_context is not None
    _resolved_from_canonical = deviation_info_from_canonical(canonical_semantic_context)

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
    from app.services.semantic_subject import (
        best_partial_noun_phrase,
        is_established_subject,
        resolve_deviation,
        validate_semantic_subject,
    )
    from app.services.text_grounding import phrase_is_grounded, significant_words

    # Security-quarantined text for subject/deviation resolution: if
    # untrusted instructions or non-substantive/expressive segments were
    # detected and stripped, pass only the auditable text spans to
    # resolve_deviation so injected or purely social text never becomes the
    # subject. Falling back to the raw finding_text is only safe when
    # nothing was deliberately excluded (i.e. segmentation just failed to
    # recognize otherwise-legitimate content) -- when every segment was
    # excluded as untrusted/non-substantive, there is no legitimate content
    # to fall back to, and resolve_deviation must simply find nothing.
    _all_segments_excluded = bool(segments) and (
        len(untrusted_segments) + len(_excluded_non_substantive_segments) >= len(segments)
    )
    if auditable_segments:
        resolution_text = " ".join(auditable_segments)
    elif _all_segments_excluded:
        resolution_text = ""
    else:
        resolution_text = request.finding_text

    # A "two independent quantified assessments, comparability explicitly
    # unresolved" finding carries its comparison semantics ACROSS sentence
    # boundaries ("Engineering estimated ... ₹3 lakh." / "Procurement
    # obtained a quotation of ₹4.2 lakh, but the scope had not been
    # confirmed to match ..."). Per-segment filtering (e.g. the epistemic-
    # stance gate dropping the "X estimated that ..." clause) can leave
    # `resolution_text` with only one side, destroying the comparison. When
    # the trusted full text (untrusted injections still excluded) exhibits
    # this shape, resolve on that instead so both assessments and their
    # provenance are conserved.
    # Spec Pass 41/43: on the canonical-SUCCESS path `resolved` is a straight
    # projection of the validated LLM context -- resolve_deviation() and every
    # other deterministic semantic helper (including the dual-assessment
    # comparison text-shape detector) are NOT invoked. They run ONLY on the
    # deterministic-fallback path.
    if _resolved_from_canonical is not None:
        resolved = _resolved_from_canonical
        source_words = set()
    else:
        from app.services.semantic_subject import _extract_dual_assessment_comparison
        _trusted_full = " ".join(s for s in segments if s not in untrusted_segments)
        if (
            _trusted_full
            and _extract_dual_assessment_comparison(_trusted_full)
            and not _extract_dual_assessment_comparison(resolution_text)
        ):
            resolution_text = _trusted_full
        resolved = resolve_deviation(resolution_text, fact_claims)
        source_words = significant_words(resolution_text)
    _DEGRADED_SUBJECTS = {"process compliance", None, ""}
    # A bare fragment ("The"), a generic placeholder, or an explicit
    # unresolved marker must all count as "resolver did NOT succeed" so the
    # LLM-subject / best-fragment / honest-marker fallback below runs.
    resolver_succeeded = is_established_subject(resolved.finding_subject or resolved.subject)

    if _canonical_success:
        # Spec Pass 41 §13/§14: the canonical LLM is authoritative. Its subject
        # (or its ABSENCE) stands -- NO raw-text noun-phrase extraction, NO
        # degraded-extraction fallback, NO grounding heuristics.
        _canon_subj = resolved.finding_subject or resolved.subject
        deviation_subject = _canon_subj or "UNRESOLVED — the canonical interpretation did not establish an affected object"
        deviation_condition = resolved.condition or "UNKNOWN"
        deviation_actor = resolved.actor
        deviation_date = resolved.date
        _entity_resolution = "RESOLVED" if _canon_subj else "UNRESOLVED"
        _entity_note = None if _canon_subj else (
            "The canonical semantic interpretation did not establish a specific "
            "affected object."
        )
        if _entity_note:
            trace.append(AgentTraceStep.warn(f"Entity fidelity: {_entity_note}"))
        observed_deviation = deviation_subject
        if deviation_condition and deviation_condition != "UNKNOWN":
            observed_deviation = f"{deviation_subject} — {deviation_condition}"
    elif resolver_succeeded:
        deviation_subject = resolved.finding_subject or resolved.subject
        deviation_condition = resolved.condition or "UNKNOWN"
        deviation_actor = resolved.actor or (extraction.deviation_actor if extraction else None)
        deviation_date = resolved.date or (extraction.timeframe if extraction else None)
        _entity_resolution = getattr(resolved, "extraction_confidence", "RESOLVED")
        _entity_note = None
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

    # Defect 3 — entity fidelity (FALLBACK PATH ONLY, spec Pass 41 §13). The
    # canonical-success branch above already set the subject / resolution /
    # observed_deviation directly from the LLM; this raw-text best-fragment
    # recovery must never run on that path.
    if not _canonical_success:
        _entity_resolution = getattr(resolved, "extraction_confidence", "RESOLVED")
        _entity_note = None
        if not deviation_subject or not validate_semantic_subject(deviation_subject):
            _fragment = (
                resolved.finding_subject
                or getattr(resolved, "partial_subject_fragment", None)
                or best_partial_noun_phrase(resolution_text)
            )
            if _fragment and validate_semantic_subject(_fragment):
                deviation_subject = _fragment
                _entity_resolution = "PARTIAL"
                _entity_note = (
                    "Entity extraction was uncertain — the specific affected "
                    "object/record/equipment could not be isolated from the finding text; "
                    f"'{_fragment}' is a best-effort fragment and must be confirmed."
                )
            else:
                deviation_subject = "UNRESOLVED — the specific entity involved could not be isolated from the finding text"
                _entity_resolution = "UNRESOLVED"
                _entity_note = (
                    "Entity extraction failed — the finding text does not name a "
                    "specific affected object, record, or equipment item."
                )
        elif getattr(resolved, "subject_unresolved", False):
            _entity_resolution = getattr(resolved, "extraction_confidence", "PARTIAL")
            _entity_note = (
                "Entity extraction was uncertain — the affected object is a best-effort "
                "noun-phrase fragment and must be confirmed during investigation."
            )
        if _entity_note:
            trace.append(AgentTraceStep.warn(f"Entity fidelity: {_entity_note}"))

        observed_deviation = deviation_subject
        if deviation_condition and deviation_condition != "UNKNOWN":
            observed_deviation = f"{deviation_subject} — {deviation_condition}"

    # Immediate mechanism (spec Pass 43 §8): LLM-owned on the canonical-success
    # path (a `causal_claim` the LLM marked `is_causal`, with its own
    # evidence_status). The deterministic `extract_immediate_mechanism`
    # (regex causal-signal detection over evidence claims) runs ONLY on the
    # fallback path.
    mechanism = mechanism_info_from_canonical(canonical_semantic_context)
    if mechanism is None:
        from app.agent.causal_guard import extract_immediate_mechanism
        mechanism = extract_immediate_mechanism(reported_claims, fact_claims)

    # Stated causal alternatives (spec Pass 44 §8): LLM-OWNED on the
    # canonical-success path -- the canonical LLM's own
    # `stated_causal_alternatives` / `causal_alternatives_unresolved`. The
    # deterministic text extractor runs ONLY on the fallback path. The raw
    # finding text itself is preserved on `raw_finding` for audit/provenance.
    if _canonical_success:
        _stated_causal_alternatives = list(
            getattr(canonical_semantic_context, "stated_causal_alternatives", None) or []
        )
        _causal_alternatives_unresolved = bool(
            getattr(canonical_semantic_context, "causal_alternatives_unresolved", False)
        )
    else:
        from app.agent.causal_guard import (
            extract_stated_causal_alternatives,
            stated_alternatives_unresolved,
        )
        _stated_causal_alternatives = extract_stated_causal_alternatives(request.finding_text)
        _causal_alternatives_unresolved = bool(_stated_causal_alternatives) and (
            stated_alternatives_unresolved(request.finding_text)
            or mechanism.status == "UNKNOWN"
        )

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
    # STRUCTURAL CONSISTENCY DOWNGRADE (spec Pass 46 §13): a REPORTED mechanism
    # cannot be treated as established while claims about the SAME proposition
    # structurally conflict. This is a validation downgrade, NOT a re-
    # interpretation -- the deterministic layer rejects an unestablishable
    # claim, it never invents a replacement mechanism. Provenance preserves
    # BOTH what was claimed and what validation did.
    if evidence_conflicts and mechanism.status == "REPORTED":
        _orig_mech_status = mechanism.status
        mechanism.status = "UNKNOWN"
        trace.append(AgentTraceStep.warn(
            "Mechanism validation: original_status=%s validated_status=UNKNOWN "
            "reason=EVIDENCE_CONFLICT action=DOWNGRADE — %d conflicting claim(s) about "
            "the same proposition prevent establishing the reported mechanism; no "
            "replacement mechanism is inferred."
            % (_orig_mech_status, len(evidence_conflicts))
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

    # Spec Pass 42 §4/§5: recurrence + previous-CAPA reference are LLM-owned on
    # the canonical-success path. `recurrence_guard.detect_recurrence`
    # (raw-finding-text regex) runs ONLY on the deterministic-fallback path.
    recurrence = recurrence_info_from_canonical(canonical_semantic_context)
    if recurrence is None:
        from app.agent.recurrence_guard import detect_recurrence
        recurrence = detect_recurrence(request.finding_text)
    if recurrence.is_recurring:
        trace.append(AgentTraceStep.warn(
            "Recurrence signal detected: " + (recurrence.rationale or "a similar finding was previously identified")
        ))

    from app.agent.proposition_engine import build_propositions_from_ledger, build_semantic_graph, classify_investigation_mode
    investigation_mode = classify_investigation_mode(
        request.finding_text, evidence_claims, evidence_conflicts, referenced_documents
    )
    propositions = build_propositions_from_ledger(request.finding_text, evidence_claims, evidence_conflicts)
    # Pass the already-computed DeviationInfo -- one semantic interpretation
    # per investigation (build_semantic_graph must not re-run resolve_deviation).
    # Only reuse it when it was computed from the SAME text the graph builder
    # would use (byte-equivalence for the segment-filtered path).
    # Spec Pass 47 §5: on the canonical-success path `resolved` is the
    # canonical LLM projection -- always pass it so `build_semantic_graph`
    # never falls back to its internal `resolve_deviation(finding_text)`.
    # On the fallback path keep the legacy quarantine guard.
    _rd_for_graph = resolved if (_canonical_success or resolution_text == request.finding_text) else None
    semantic_graph = build_semantic_graph(
        request.finding_text, evidence_claims, propositions, evidence_conflicts,
        resolved_deviation=_rd_for_graph,
    )

    # Typed measurement (Section 8): a discrepancy magnitude ("approximately
    # 4.2%") is semantically OBSERVED_DISCREPANCY, never a financial amount,
    # probability, or confidence score -- tag its own evidence status/claim
    # provenance from the SAME evidence ledger every other claim uses (the
    # "C1, C2, ..." positional convention core_synthesis assigns later), so
    # downstream nodes never need to re-derive it from raw text.
    semantic_measurement = None
    if getattr(resolved, "measurement_value", None) is not None:
        _meas_status = EvidenceStatus.UNKNOWN
        _meas_claim_id = None
        for _i, _e in enumerate(ledger):
            if str(resolved.measurement_value) in (_e.claim or ""):
                _meas_status = _e.status
                _meas_claim_id = f"C{_i + 1}"
                break
        canonical_state_measurement_status = (
            "VERIFIED" if _meas_status == EvidenceStatus.VERIFIED
            else "REPORTED" if _meas_status in (EvidenceStatus.REPORTED, EvidenceStatus.MIXED)
            else "UNKNOWN"
        )
        semantic_measurement = SemanticMeasurement(
            value=resolved.measurement_value,
            unit=resolved.measurement_unit,
            qualifier=resolved.measurement_qualifier,
            role="OBSERVED_DISCREPANCY",
            evidence_status=canonical_state_measurement_status,
            source_claim_id=_meas_claim_id,
        )

    # Align affected_object and affected_process with semantic graph nodes
    from app.models.agent import EpistemicSource, SemanticNode, SemanticNodeType
    _aff_obj = resolved.affected_object or deviation_subject
    if _aff_obj and not _aff_obj.startswith("UNKNOWN") and not _aff_obj.startswith("UNRESOLVED"):
        _existing_nodes = {n.label.strip().lower(): n for n in semantic_graph.nodes}
        if _aff_obj.strip().lower() not in _existing_nodes:
            semantic_graph.nodes.append(
                SemanticNode(
                    id=f"N{len(semantic_graph.nodes) + 1}",
                    label=_aff_obj,
                    node_type=SemanticNodeType.ENTITY,
                    epistemic_status=EvidenceStatus.VERIFIED,
                    provenance=EpistemicSource.AUDIT_OBSERVATION,
                )
            )

    _aff_proc = resolved.affected_process or "UNKNOWN"
    if _aff_proc and _aff_proc not in ("UNKNOWN", "NOT_ESTABLISHED", "UNRESOLVED"):
        _existing_nodes = {n.label.strip().lower(): n for n in semantic_graph.nodes}
        if _aff_proc.strip().lower() not in _existing_nodes:
            semantic_graph.nodes.append(
                SemanticNode(
                    id=f"N{len(semantic_graph.nodes) + 1}",
                    label=_aff_proc,
                    node_type=SemanticNodeType.PROCESS,
                    epistemic_status=EvidenceStatus.VERIFIED,
                    provenance=EpistemicSource.AUDIT_OBSERVATION,
                )
            )

    canonical_state = CanonicalFindingState(
        raw_finding=request.finding_text,
        finding_subject=deviation_subject,
        affected_object=_aff_obj,
        affected_process=_aff_proc,
        affected_activity=resolved.affected_activity or "UNKNOWN",
        deviation=observed_deviation,
        observed_deviation=observed_deviation,
        deviation_condition=deviation_condition,
        facts=fact_claims,
        verified_observations=fact_claims,
        reported_statements=reported_claims,
        unknowns=["Root cause unconfirmed from initial evidence"],
        affected_objects=[deviation_subject],
        affected_period=deviation_date or resolved.recurrence_period or recurrence.recurrence_period or "UNKNOWN",
        time_period=deviation_date or resolved.recurrence_period or recurrence.recurrence_period or "UNKNOWN",
        actor=deviation_actor,
        actors=resolved.actors,
        entities=resolved.entities,
        immediate_mechanism=mechanism.statement,
        reported_mechanism=mechanism.statement if mechanism.status == "REPORTED" else None,
        verified_mechanism=mechanism.statement if mechanism.status == "VERIFIED" else None,
        immediate_mechanism_status=mechanism.status,
        mechanism_status=mechanism.status,
        mechanism_polarity=mechanism.polarity,
        stated_causal_alternatives=_stated_causal_alternatives,
        causal_alternatives_unresolved=_causal_alternatives_unresolved,
        prompt_injection_detected=is_instruction(request.finding_text),
        input_integrity_status=_worst_classification,
        security_flags=_security_flags,
        instruction_like_claim_count=len(_excluded_claim_texts),
        excluded_claim_texts=_excluded_claim_texts,
        evidence_claims=evidence_claims,
        evidence_conflicts=evidence_conflicts,
        referenced_documents=referenced_documents,
        semantic_graph=semantic_graph,
        investigation_mode=investigation_mode,
        propositions=propositions,
        relevant_change=resolved.relevant_change,
        semantic_type=resolved.semantic_type,
        comparison_type=resolved.comparison_type,
        comparison_left=resolved.comparison_left,
        comparison_left_qualifier=resolved.comparison_left_qualifier,
        comparison_right=resolved.comparison_right,
        comparison_basis=resolved.comparison_basis,
        comparison_subtype=resolved.comparison_subtype,
        comparison_reference_type=resolved.comparison_reference_type,
        comparison_batch_id=resolved.comparison_batch_id,
        missing_record_activity=resolved.missing_record_activity,
        missing_record_context=resolved.missing_record_context,
        downstream_action_text=resolved.downstream_action_text,
        downstream_action_present=resolved.downstream_action_present,
        occurrence_population=resolved.occurrence_population,
        recurrence_count=resolved.recurrence_count or recurrence.recurrence_count,
        recurrence_event=resolved.recurrence_event or recurrence.recurrence_event,
        recurrence_period=resolved.recurrence_period or recurrence.recurrence_period,
        attributed_source=resolved.attributed_source,
        attributed_proposition=resolved.attributed_proposition,
        transition_type=resolved.transition_type,
        control_justification_missing=resolved.control_justification_missing,
        requirement_status=resolved.requirement_status,
        observed_entity=resolved.observed_entity or deviation_subject,
        affected_entity=deviation_subject,
        control_entity=resolved.affected_process if resolved.affected_process != "NOT_ESTABLISHED" else None,
        actor_entity=deviation_actor,
        measurement=semantic_measurement,
        recurrence_signal=recurrence.is_recurring,
        previous_capa_referenced=recurrence.has_previous_capa_reference,
        previous_capa_status=recurrence.previous_capa_status,
        previous_capa_effectiveness=recurrence.previous_capa_effectiveness,
        recurrence_rationale=recurrence.rationale,
        entity_resolution=_entity_resolution,
        entity_resolution_note=_entity_note,
        subject_unresolved=(_entity_resolution != "RESOLVED"),
        is_actionable=bool(quality.status == ObservationQualityStatus.SUFFICIENT),
        actionability_reason=None if quality.status == ObservationQualityStatus.SUFFICIENT else (quality.missing_information[0] if quality.missing_information else "Input does not contain an actionable audit finding, deviation, or verifiable condition."),
    )

    if not canonical_state.is_actionable:
        canonical_state.investigation_mode = InvestigationMode.NON_ACTIONABLE

    # Spec Pass 45 §4/§6/§7: on the canonical-success path uncertainty is
    # derived ONLY from LLM-established structured signals -- no raw finding
    # text. `canonical_success=True` makes `detect_uncertainties` ignore the
    # raw-text regex disjuncts and use `semantic_type` / `missing_record_
    # status` / `comparison_type` / `recurrence_signal` / `mechanism_status` /
    # `evidence_conflicts` (all LLM-owned on success).
    from app.agent.causal_guard import detect_uncertainties
    prim_unc, sec_unc, c_ready, blk_steps = detect_uncertainties(
        canonical_state=canonical_state,
        evidence_ledger=ledger,
        finding_text="" if _canonical_success else request.finding_text,
        canonical_success=_canonical_success,
        canonical_context=canonical_semantic_context if _canonical_success else None,
    )
    canonical_state.primary_uncertainty = prim_unc
    canonical_state.secondary_uncertainties = sec_unc
    canonical_state.causal_readiness = c_ready
    canonical_state.blocked_reasoning_steps = blk_steps


    # Cost & Financial Impact is no longer computed here. `report.cost_impact`
    # is now a compatibility projection of the canonical `financial_analysis`
    # (app.financial.models.FinancialAnalysisResult), derived once by
    # final_evidence_verification_node via app.financial.compatibility --
    # see that node for the single authoritative financial calculation
    # point. Nothing between this node and final_evidence_verification_node
    # reads state["cost_impact"], so no early computation is needed here.

    # Deterministically enforce Observation Quality Sufficiency (Section 1 & 2):
    # If a concrete affected object and observed deviation exist, quality is SUFFICIENT.
    # Root cause uncertainty must NEVER downgrade observation quality.
    #
    # Release-gate note: a variant of this fix was tried and reverted. The
    # `canonical_state.is_actionable` precondition below is NOT redundant --
    # it is a deliberate AND-gate: deviation_subject/observed_deviation can
    # carry non-trivial fallback/placeholder text even for genuinely
    # non-actionable input (e.g. "Hi"), so requiring is_actionable to
    # ALREADY be True before this correction fires is what prevents
    # exactly that over-correction. Removing it caused a real, confirmed
    # regression (test_input_actionability_regression.py) where
    # conversational/prompt-injection inputs were incorrectly upgraded to
    # SUFFICIENT/actionable. The originally-hypothesized failure mode this
    # was meant to fix (an LLM quality-check call failing and leaving
    # is_actionable stale) does not occur on this path in practice: the
    # LLM-based app.services.observation_quality.check_observation_quality
    # is not called here at all -- the quality determination above (lines
    # ~162-187) is fully deterministic. See Stage B's plan_investigation_causal
    # for where the actual reachable "zero hypotheses is not verified
    # non-actionability" gap was found and fixed instead.
    from app.models.analysis import ObservationQualityResult, ObservationQualityStatus
    if canonical_state.is_actionable and not deviation_subject.startswith("UNKNOWN") and len(observed_deviation.strip()) >= 5:
        # Unless text is extremely vague ("something was wrong"), mark as SUFFICIENT
        if not re.search(r"^(something|anything|stuff)\s+(was|is)\s+(wrong|bad)", request.finding_text.strip(), re.IGNORECASE):
            quality = ObservationQualityResult(
                status=ObservationQualityStatus.SUFFICIENT,
                missing_information=[],
            )
    elif not canonical_state.is_actionable:
        quality = ObservationQualityResult(
            status=ObservationQualityStatus.INSUFFICIENT,
            missing_information=["Input does not contain an actionable audit finding, deviation, or verifiable condition."],
        )

    # ------------------------------------------------------------------
    # Spec Pass 41-44: the canonical LLM interpretation already ran near the
    # top of this node, and on the SUCCESS path `canonical_state` was built
    # ENTIRELY from the validated LLM context (deviation_info_from_canonical /
    # recurrence_info_from_canonical / mechanism_info_from_canonical). This
    # step only copies the remaining context-only PROVENANCE fields
    # (evidence_source / reported_observation / finding_epistemic_status /
    # affected_period / scope) + the structural subject-safety re-check. It is
    # NOT a semantic merge -- there is no deterministic semantic state on this
    # path. On the FAILURE path `canonical_semantic_context` is None and this
    # is a no-op.
    # ------------------------------------------------------------------
    if canonical_semantic_context is not None:
        canonical_state, _prov = apply_canonical_provenance_fields(
            canonical_state, canonical_semantic_context
        )
        if _prov.fields_from_llm or _prov.fields_rejected:
            trace.append(AgentTraceStep.ok(
                "Understanding: canonical LLM semantic state — "
                f"from_llm={_prov.fields_from_llm} rejected={_prov.fields_rejected}"
            ))

    # Explicit, observable semantic mode (spec §3/§20): the report must never
    # imply canonical LLM reasoning occurred when the canonical call failed.
    _semantic_mode = "CANONICAL_LLM" if canonical_semantic_context is not None else (
        "DETERMINISTIC_FALLBACK" if settings.canonical_semantic_llm_primary else "DETERMINISTIC"
    )
    if settings.canonical_semantic_llm_primary:
        if canonical_semantic_context is not None:
            trace.append(AgentTraceStep.ok(
                f"Understanding: semantic mode = {_semantic_mode} "
                f"(canonical interpretation {canonical_semantic_status})"
            ))
        else:
            trace.append(AgentTraceStep.warn(
                f"Understanding: semantic mode = {_semantic_mode} — canonical interpretation "
                f"unavailable ({canonical_semantic_status}); deterministic floor is authoritative"
            ))

    return {
        **state,
        "observation_quality": quality,
        "extraction": extraction,
        "canonical_finding_state": canonical_state,
        "canonical_semantic_context": canonical_semantic_context,
        "canonical_semantic_status": canonical_semantic_status,
        "semantic_mode": _semantic_mode,
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

