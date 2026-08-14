"""Step 2 of the finding-analysis pipeline: causation classification.

Reasons ONLY over the Step 1 ExtractionResult (never raw finding_text) to decide
root_cause_status -- this forces classification to work from what was actually
extracted rather than re-deriving facts under the same breath as generating
prose. Calibrated against app/prompts/causation_classification_fewshot.txt.
"""

import logging

from app.config import get_settings
from app.models.analysis import CausationClassification, ExtractionResult, RootCauseStatus
from app.services.llm_client import LLMClient, LLMError, get_llm_client
from app.services.llm_json import parse_llm_json
from app.services.taxonomy import coerce_category, taxonomy_prompt_block

logger = logging.getLogger(__name__)

_STATUS_MAP = {
    "ESTABLISHED": RootCauseStatus.ESTABLISHED,
    "SELF_REPORTED": RootCauseStatus.SELF_REPORTED,
}


def _build_messages(extraction: ExtractionResult) -> list[dict[str, str]]:
    settings = get_settings()
    system_template = (settings.prompts_dir / "causation_classification_system_prompt.txt").read_text(
        encoding="utf-8"
    )
    fewshot = (settings.prompts_dir / "causation_classification_fewshot.txt").read_text(encoding="utf-8")
    system_prompt = system_template.format(taxonomy=taxonomy_prompt_block(), fewshot=fewshot)

    user_prompt = "EXTRACTION:\n" + extraction.model_dump_json(indent=2)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _to_classification(parsed: dict) -> CausationClassification:
    status = _STATUS_MAP.get(str(parsed.get("root_cause_status", "")).upper(), RootCauseStatus.NOT_ESTABLISHED)
    category = None
    if status != RootCauseStatus.NOT_ESTABLISHED and parsed.get("category"):
        category = coerce_category(parsed.get("category")).value
    return CausationClassification(
        root_cause_status=status,
        category=category,
        reasoning=str(parsed.get("reasoning", "")),
    )


async def classify_causation(
    extraction: ExtractionResult, client: LLMClient | None = None
) -> CausationClassification:
    client = client or get_llm_client()
    messages = _build_messages(extraction)

    try:
        raw = await client.chat_completion(messages, temperature=0.0, response_format_json=True)
        parsed = parse_llm_json(raw)
        if "root_cause_status" not in parsed:
            raise ValueError("missing root_cause_status")
        return _to_classification(parsed)
    except (LLMError, ValueError, KeyError) as exc:
        logger.warning("Causation classification failed (%s); retrying once.", exc)

    stricter_messages = messages + [
        {
            "role": "user",
            "content": (
                "Your previous response was invalid or incomplete. Return ONLY a single "
                "valid JSON object with root_cause_status, category, and reasoning present."
            ),
        }
    ]
    try:
        raw = await client.chat_completion(stricter_messages, temperature=0.0, response_format_json=True)
        parsed = parse_llm_json(raw)
        if "root_cause_status" not in parsed:
            raise ValueError("missing root_cause_status")
    except (LLMError, ValueError, KeyError) as exc:
        raise LLMError(f"Causation classification failed after retry: {exc}") from exc

    return _to_classification(parsed)
