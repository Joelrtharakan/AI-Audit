"""LLM-only finding analysis pipeline, decomposed into narrower LLM calls for
reliable calibration, with a mechanical post-generation guardrail layer:

    Observation
      -> Observation Quality Check              (observation_quality.py)
      -> Step 1: Extraction                     (extraction.py)
      -> Step 2: Causation Classification        (causation_classifier.py)
      -> Step 3: Generation (5-why/investigation/capa/impact)  (prompt_builder.py)
      -> Grounding validation (Step 2 vs Step 1, Step 3 vs source text)
         (grounding_validator.py) -- strips hallucinated claims, downgrades
         root_cause_status a tier when the hallucination is in the highest-stakes
         fields, enforces scope rules (recall language, named CAPA owners, quoted/
         numeric claims) as code, not model judgment
      -> 5-Why chain-threading check              (five_why_validator.py)
      -> Consistency check (ESTABLISHED only)      (rerun classification once)
      -> existing programmatic enforcement (capa status <-> root cause status)
      -> programmatic confidence scoring
      -> response

No document retrieval, embeddings, vector search, or historical-finding lookup
of any kind happens anywhere in this pipeline -- see
app/prompts/finding_analysis_system_prompt.txt for the anti-hallucination rules
the generation step depends on, and
app/prompts/causation_classification_fewshot.txt for the classifier's calibration.
"""

import datetime as dt
import logging
import uuid

from app.config import get_settings
from app.models.analysis import (
    AiMetadata,
    AnalysisSummary,
    AnalyzeFindingResponse,
    CapaStatus,
    CapaSuggestion,
    CausationClassification,
    ConfidenceLevel,
    ExtractionResult,
    GroundingReport,
    Investigation,
    ImpactAnalysis,
    ImpactStatus,
    ObservationQualityStatus,
    RiskOfRecurrence,
    RootCause,
    RootCauseStatus,
)
from app.models.requests import AnalyzeFindingRequest
from app.services.causation_classifier import classify_causation
from app.services.confidence import ConfidenceInputs, calculate_confidence
from app.services.extraction import extract_finding
from app.services.five_why_validator import validate_and_truncate_chain
from app.services.grounding_validator import validate_grounding
from app.services.llm_client import LLMClient, LLMError, get_llm_client
from app.services.llm_json import parse_llm_json, strip_html
from app.services.observation_quality import check_observation_quality
from app.services.prompt_builder import build_finding_analysis_messages
from app.services.text_grounding import significant_words

logger = logging.getLogger(__name__)

STATIC_AI_LIMITATIONS = [
    "This is AI-assisted audit analysis, not an evidence-grounded organizational CAPA -- "
    "the organization's actual SOPs, policies, and historical findings were not consulted.",
    "All root-cause, CAPA, and impact suggestions require auditor review before acceptance.",
]

_REQUIRED_TOP_KEYS = ["contributing_factors", "investigation", "capa", "impact_analysis"]

# spec 4.3: contributing_factors are conditions, never action items to go do -- and must
# not duplicate content already asked for in investigation.areas/evidence_required.
_ACTION_VERB_PREFIX_RE_PARTS = ("review", "verify", "confirm", "check", "pull", "collect")


def _validate(parsed: dict) -> list[str]:
    problems = [k for k in _REQUIRED_TOP_KEYS if k not in parsed]
    if "capa" in parsed and "status" not in parsed.get("capa", {}):
        problems.append("capa.status")
    return problems


def _strip_list(values: list) -> list[str]:
    return [strip_html(str(v)) for v in values]


def _looks_like_action_item(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(verb) for verb in _ACTION_VERB_PREFIX_RE_PARTS)


def _duplicates_investigation(text: str, investigation: Investigation) -> bool:
    text_words = significant_words(text)
    if not text_words:
        return False
    for other in (*investigation.areas, *investigation.evidence_required):
        other_words = significant_words(other)
        if not other_words:
            continue
        overlap = text_words & other_words
        if len(overlap) / len(text_words) >= 0.6:
            return True
    return False


def _filter_contributing_factors(factors: list[str], investigation: Investigation) -> list[str]:
    return [
        f for f in factors
        if not _looks_like_action_item(f) and not _duplicates_investigation(f, investigation)
    ]


def _build_root_cause_status(extraction: ExtractionResult, classification: CausationClassification) -> RootCauseStatus:
    """Enforcement, not just prompting: re-checks Step 2's classification against
    Step 1's extraction and downgrades it if the claimed status isn't actually
    supported -- e.g. a claimed ESTABLISHED with no referenced_records is not trusted,
    it's downgraded to what the extraction actually supports."""
    status = classification.root_cause_status

    if status == RootCauseStatus.ESTABLISHED and not extraction.referenced_records:
        logger.warning(
            "Classifier claimed ESTABLISHED with no referenced_records in the extraction; downgrading to SELF_REPORTED."
        )
        status = RootCauseStatus.SELF_REPORTED

    if status == RootCauseStatus.SELF_REPORTED and not extraction.attributed_statements:
        logger.warning(
            "Classifier claimed SELF_REPORTED with no attributed_statements in the extraction; downgrading to NOT_ESTABLISHED."
        )
        status = RootCauseStatus.NOT_ESTABLISHED

    return status


def _verification_needed_text(status: RootCauseStatus, classification: CausationClassification) -> str | None:
    if status != RootCauseStatus.SELF_REPORTED:
        return None
    reasoning = strip_html(classification.reasoning)
    return reasoning or "Verify the self-reported cause against supporting records before treating it as confirmed."


def _build_root_cause(
    status: RootCauseStatus,
    classification: CausationClassification,
    cleaned_generation: dict,
) -> RootCause:
    if status == RootCauseStatus.NOT_ESTABLISHED:
        return RootCause(
            status=status,
            category=None,
            category_other_text=None,
            statement=None,
            is_hypothesis=False,
            verification_needed=None,
            classification_reasoning=strip_html(classification.reasoning) or None,
            risk_of_recurrence=None,
            recommended_capa_owner=None,
        )

    risk_of_recurrence = None
    recommended_capa_owner = None
    if status == RootCauseStatus.ESTABLISHED:
        raw_risk = str(cleaned_generation.get("risk_of_recurrence") or "").strip().capitalize()
        if raw_risk in (RiskOfRecurrence.LOW.value, RiskOfRecurrence.MEDIUM.value, RiskOfRecurrence.HIGH.value):
            risk_of_recurrence = RiskOfRecurrence(raw_risk)
        recommended_capa_owner = strip_html(cleaned_generation.get("recommended_capa_owner")) or None

    return RootCause(
        status=status,
        category=classification.category,
        category_other_text=None,
        statement=strip_html(classification.reasoning) or None,
        is_hypothesis=status != RootCauseStatus.ESTABLISHED,
        verification_needed=_verification_needed_text(status, classification),
        classification_reasoning=strip_html(classification.reasoning) or None,
        risk_of_recurrence=risk_of_recurrence,
        recommended_capa_owner=recommended_capa_owner,
    )


def _build_capa(parsed: dict, root_cause_status: RootCauseStatus) -> CapaSuggestion:
    raw = parsed.get("capa") or {}
    llm_wants_ai_suggested = str(raw.get("status", "")).upper() == "AI_SUGGESTED"
    # Enforced in code, not just prompted: a real CAPA (AI_SUGGESTED) requires a truly
    # ESTABLISHED (corroborated) root cause -- never just SELF_REPORTED or
    # NOT_ESTABLISHED, regardless of what the LLM put in capa.status.
    status = (
        CapaStatus.AI_SUGGESTED
        if llm_wants_ai_suggested and root_cause_status == RootCauseStatus.ESTABLISHED
        else CapaStatus.INVESTIGATION_REQUIRED
    )

    if status == CapaStatus.INVESTIGATION_REQUIRED:
        return CapaSuggestion(
            status=status,
            corrective_actions_immediate=[],
            preventive_actions_recurrence=[],
            recommended_investigation=_strip_list(raw.get("recommended_investigation", [])),
            potential_corrective_action_areas=_strip_list(raw.get("potential_corrective_action_areas", [])),
        )

    return CapaSuggestion(
        status=status,
        corrective_actions_immediate=_strip_list(raw.get("corrective_actions_immediate", [])),
        preventive_actions_recurrence=_strip_list(raw.get("preventive_actions_recurrence", [])),
        recommended_investigation=[],
        potential_corrective_action_areas=[],
    )


def _build_impact_analysis(parsed: dict) -> ImpactAnalysis:
    raw = parsed.get("impact_analysis") or {}
    status = ImpactStatus.ASSESSED if str(raw.get("status", "")).upper() == "ASSESSED" else ImpactStatus.REQUIRES_ASSESSMENT
    return ImpactAnalysis(
        status=status,
        areas=_strip_list(raw.get("areas", [])),
        narrative=strip_html(raw.get("narrative")) or None,
    )


def _build_investigation(parsed: dict) -> Investigation:
    raw = parsed.get("investigation") or {}
    return Investigation(
        areas=_strip_list(raw.get("areas", [])),
        questions=_strip_list(raw.get("questions", [])),
        evidence_required=_strip_list(raw.get("evidence_required", [])),
    )


def _downgrade_confidence_one_tier(level: ConfidenceLevel) -> ConfidenceLevel:
    if level == ConfidenceLevel.HIGH:
        return ConfidenceLevel.MEDIUM
    if level == ConfidenceLevel.MEDIUM:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.LOW


class FindingAnalysisService:
    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or get_llm_client()
        self._settings = get_settings()
        self._model_name = (
            self._settings.ollama_model if self._settings.llm_provider == "ollama" else self._settings.openrouter_model
        )

    async def analyze(self, request: AnalyzeFindingRequest) -> AnalyzeFindingResponse:
        quality = await check_observation_quality(request.finding_text, self._client)

        extraction = await extract_finding(request.finding_text, self._client)
        classification = await classify_causation(extraction, self._client)

        status = _build_root_cause_status(extraction, classification)

        finding_dict = request.model_dump()
        messages = build_finding_analysis_messages(
            finding_dict,
            observation_quality_status=quality.status.value,
            observation_missing_information=quality.missing_information,
            extraction=extraction,
            classification=CausationClassification(
                root_cause_status=status, category=classification.category, reasoning=classification.reasoning
            ),
        )

        parsed = await self._generate_and_validate(messages)

        # Guardrail (Part 3): mechanically check every specific claim the generation
        # step introduced against the source text/extraction, strip what doesn't trace
        # back, enforce recall/quarantine/CAPA-owner/quoted-text scope rules as code,
        # and downgrade status a tier if the hallucination was in the highest-stakes
        # fields.
        grounding = validate_grounding(
            generation_output=parsed,
            finding_text=request.finding_text,
            extraction=extraction,
            root_cause_status=status,
            department=request.department,
        )
        cleaned = grounding.cleaned_output
        if grounding.downgraded_status:
            status = grounding.downgraded_status

        cleaned["five_why"] = validate_and_truncate_chain(cleaned.get("five_why", []))

        investigation = _build_investigation(cleaned)
        cleaned["contributing_factors"] = _filter_contributing_factors(
            cleaned.get("contributing_factors", []), investigation
        )

        # Part 4.1 backstop: a real classification (SELF_REPORTED/ESTABLISHED) means the
        # observation demonstrably had enough to work with, regardless of what the
        # cheap upfront quality heuristic guessed -- never let the two contradict in the
        # final response.
        if status != RootCauseStatus.NOT_ESTABLISHED and quality.status == ObservationQualityStatus.INSUFFICIENT:
            logger.warning(
                "observation_quality=INSUFFICIENT contradicts root_cause_status=%s; coercing to SUFFICIENT.",
                status.value,
            )
            quality.status = ObservationQualityStatus.SUFFICIENT

        # Part 3.3: for the highest-stakes (ESTABLISHED) results only, re-run
        # classification once more and check it agrees -- disagreement is a signal the
        # classification is on shaky ground, so confidence gets downgraded a tier.
        consistency_checked = False
        consistency_mismatch = False
        if status == RootCauseStatus.ESTABLISHED:
            consistency_checked = True
            try:
                repeat_classification = await classify_causation(extraction, self._client)
                if (
                    repeat_classification.root_cause_status != classification.root_cause_status
                    or repeat_classification.category != classification.category
                ):
                    consistency_mismatch = True
                    logger.warning(
                        "Consistency check mismatch: first classification %s/%s vs repeat %s/%s.",
                        classification.root_cause_status.value, classification.category,
                        repeat_classification.root_cause_status.value, repeat_classification.category,
                    )
            except LLMError as exc:
                logger.warning("Consistency check call failed (%s); skipping.", exc)
                consistency_checked = False

        root_cause = _build_root_cause(status, classification, cleaned)
        capa = _build_capa(cleaned, root_cause.status)
        impact_analysis = _build_impact_analysis(cleaned)
        contributing_factors = _strip_list(cleaned.get("contributing_factors", []))
        five_why = _strip_list(cleaned.get("five_why", []))

        # Anything short of a truly ESTABLISHED (corroborated) root cause still needs
        # auditor investigation -- including SELF_REPORTED, which is a strong hypothesis
        # but not yet confirmed.
        requires_investigation = (
            root_cause.status != RootCauseStatus.ESTABLISHED
            or capa.status == CapaStatus.INVESTIGATION_REQUIRED
        )

        confidence = calculate_confidence(
            ConfidenceInputs(
                finding_text=request.finding_text,
                observation_quality=quality.status,
                root_cause_status=root_cause.status,
                capa_status=capa.status,
                missing_information_count=len(quality.missing_information),
                open_questions_count=len(investigation.questions) + len(investigation.evidence_required),
                five_why_steps=len(five_why),
            )
        )
        if consistency_mismatch:
            confidence = _downgrade_confidence_one_tier(confidence)

        limitations = list(STATIC_AI_LIMITATIONS)
        if quality.status == ObservationQualityStatus.INSUFFICIENT:
            limitations.append(
                "The observation as written may be too vague for reliable analysis -- see missing_information."
            )
        if grounding.hard_violations:
            limitations.append(
                "Some AI-generated content was removed because it could not be traced back to the observation -- "
                "see grounding_report."
            )

        return AnalyzeFindingResponse(
            analysis=AnalysisSummary(
                observation_quality=quality.status,
                confidence=confidence,
                requires_investigation=requires_investigation,
            ),
            extraction=extraction,
            root_cause=root_cause,
            contributing_factors=contributing_factors,
            five_why=five_why,
            investigation=investigation,
            capa=capa,
            impact_analysis=impact_analysis,
            ai_limitations=limitations,
            grounding_report=GroundingReport(
                checked_entities=grounding.checked_entities,
                hard_violations=grounding.hard_violations,
                soft_violations=grounding.soft_violations,
                status_downgraded=bool(grounding.downgraded_status),
                consistency_checked=consistency_checked,
                consistency_mismatch=consistency_mismatch,
            ),
            ai_metadata=AiMetadata(
                model=self._model_name,
                prompt_version=self._settings.analysis_prompt_version,
                generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                suggestion_id=str(uuid.uuid4()),
            ),
        )

    async def _generate_and_validate(self, messages: list[dict[str, str]]) -> dict:
        # Generation has the largest response schema in the pipeline -- give it an
        # explicit token budget so a smaller local model doesn't run out of output
        # tokens partway through and silently drop trailing keys (e.g. impact_analysis).
        try:
            raw = await self._client.chat_completion(
                messages, temperature=0.2, response_format_json=True, max_tokens=2048
            )
            parsed = parse_llm_json(raw)
            problems = _validate(parsed)
            if not problems:
                return parsed
            logger.warning("Finding analysis response missing fields %s; retrying once.", problems)
        except (LLMError, ValueError, KeyError) as exc:
            logger.warning("Finding analysis generation failed (%s); retrying once.", exc)

        stricter_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Your previous response was invalid or incomplete. Return ONLY a single "
                    "valid JSON object matching the schema exactly, with every top-level key "
                    "present (use INVESTIGATION_REQUIRED / REQUIRES_ASSESSMENT and empty "
                    "lists where evidence is insufficient -- do not omit keys)."
                ),
            }
        ]
        try:
            raw = await self._client.chat_completion(
                stricter_messages, temperature=0.0, response_format_json=True, max_tokens=2048
            )
            parsed = parse_llm_json(raw)
        except (LLMError, ValueError) as exc:
            raise LLMError(f"Finding analysis failed after retry: {exc}") from exc

        problems = _validate(parsed)
        if problems:
            raise LLMError(f"Finding analysis response still invalid after retry: {problems}")
        return parsed
