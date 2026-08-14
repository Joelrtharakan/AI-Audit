"""Observation quality check (spec section 9): does the auditor's raw finding
text contain enough information for meaningful root-cause analysis?

This runs before the main LLM-first analysis so finding_analysis_service can
lean toward NOT_ESTABLISHED / INVESTIGATION_REQUIRED when quality is poor,
instead of guessing. Never invents missing_information -- if the LLM call
fails, we fail safe to INSUFFICIENT with a generic, honest message rather than
silently treating the observation as fine.
"""

import logging

from app.config import get_settings
from app.models.analysis import ObservationQualityResult, ObservationQualityStatus
from app.services.llm_client import LLMClient, LLMError, get_llm_client
from app.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)

_FALLBACK_MISSING_INFO = [
    "Automated observation-quality check was unavailable; auditor should confirm the "
    "finding includes what/where/when and directly observed evidence."
]


async def check_observation_quality(finding_text: str, client: LLMClient | None = None) -> ObservationQualityResult:
    settings = get_settings()
    template = (settings.prompts_dir / "finding_analysis_prompt.txt").read_text(encoding="utf-8")
    prompt = template.format(observation=finding_text)

    client = client or get_llm_client()
    try:
        raw = await client.chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format_json=True,
        )
        parsed = parse_llm_json(raw)
        sufficient = bool(parsed.get("sufficient"))
        missing = [str(x) for x in parsed.get("missing_information", [])]
        return ObservationQualityResult(
            status=ObservationQualityStatus.SUFFICIENT if sufficient else ObservationQualityStatus.INSUFFICIENT,
            missing_information=missing,
        )
    except (LLMError, ValueError, KeyError) as exc:
        logger.warning("Observation quality check failed (%s); failing safe to INSUFFICIENT.", exc)
        return ObservationQualityResult(
            status=ObservationQualityStatus.INSUFFICIENT,
            missing_information=list(_FALLBACK_MISSING_INFO),
        )
