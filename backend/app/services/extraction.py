"""Step 1 of the finding-analysis pipeline: extraction.

Pulls stated_facts / attributed_statements / referenced_records / timeframe /
asset_or_location out of the raw observation text, with no causal judgment at
all -- that's Step 2 (causation_classifier.py). Validates that the extraction
doesn't introduce content absent from the source text (a lightweight grounding
check, not full fact-checking) and retries once if it does, matching the
"one stricter retry" pattern used elsewhere in this pipeline.
"""

import logging

from app.config import get_settings
from app.models.analysis import AttributedStatement, ExtractionResult
from app.services.llm_client import LLMClient, LLMError, get_llm_client
from app.services.llm_json import parse_llm_json
from app.services.text_grounding import entity_is_grounded, phrase_is_grounded, significant_words

logger = logging.getLogger(__name__)


def _ungrounded_phrases(parsed: dict, finding_text: str) -> list[str]:
    source_words = significant_words(finding_text)
    candidates: list[str] = []
    candidates.extend(str(x) for x in parsed.get("stated_facts", []))
    for stmt in parsed.get("attributed_statements", []):
        if isinstance(stmt, dict):
            candidates.append(str(stmt.get("claim", "")))
    timeframe = parsed.get("timeframe")
    if timeframe:
        candidates.append(str(timeframe))
    asset = parsed.get("asset_or_location")
    if asset:
        candidates.append(str(asset))

    # For named systems or document IDs, enforce entity grounding check
    for doc in parsed.get("named_systems_or_documents", []):
        if doc and not entity_is_grounded(str(doc), finding_text):
            candidates.append(str(doc))

    return [c for c in candidates if c and not phrase_is_grounded(c, source_words)]


def _to_extraction_result(parsed: dict) -> ExtractionResult:
    statements = [
        AttributedStatement(speaker=str(s.get("speaker", "")), claim=str(s.get("claim", "")))
        for s in parsed.get("attributed_statements", [])
        if isinstance(s, dict) and s.get("claim")
    ]
    return ExtractionResult(
        stated_facts=[str(x) for x in parsed.get("stated_facts", [])],
        attributed_statements=statements,
        referenced_records=[str(x) for x in parsed.get("referenced_records", [])],
        named_systems_or_documents=[str(x) for x in parsed.get("named_systems_or_documents", [])],
        timeframe=parsed.get("timeframe") or None,
        asset_or_location=parsed.get("asset_or_location") or None,
        external_impact_stated=bool(parsed.get("external_impact_stated", False)),
    )


async def extract_finding(finding_text: str, client: LLMClient | None = None) -> ExtractionResult:
    settings = get_settings()
    template = (settings.prompts_dir / "extraction_prompt.txt").read_text(encoding="utf-8")
    prompt = template.format(observation=finding_text)
    client = client or get_llm_client()

    messages = [{"role": "user", "content": prompt}]

    async def _attempt(msgs: list[dict[str, str]], temperature: float) -> dict | None:
        try:
            raw = await client.chat_completion(msgs, temperature=temperature, response_format_json=True)
            parsed = parse_llm_json(raw)
        except (LLMError, ValueError, KeyError) as exc:
            logger.warning("Extraction call failed (%s).", exc)
            return None

        ungrounded = _ungrounded_phrases(parsed, finding_text)
        if ungrounded:
            logger.warning("Extraction introduced ungrounded content %s; will retry once.", ungrounded)
            return None
        return parsed

    parsed = await _attempt(messages, temperature=0.0)
    if parsed is not None:
        return _to_extraction_result(parsed)

    stricter_messages = messages + [
        {
            "role": "user",
            "content": (
                "Your previous response either was invalid JSON or included content not "
                "explicitly present in the OBSERVATION text. Extract ONLY words, phrases, "
                "and claims that literally appear (or are very close paraphrases of "
                "wording that appears) in the OBSERVATION. Return ONLY the JSON object."
            ),
        }
    ]
    parsed = await _attempt(stricter_messages, temperature=0.0)
    if parsed is not None:
        return _to_extraction_result(parsed)

    raise LLMError("Extraction failed after retry: LLM call failed or introduced ungrounded content.")
