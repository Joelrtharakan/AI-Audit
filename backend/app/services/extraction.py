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
    for field in ("deviation_subject", "deviation_condition", "deviation_actor"):
        value = parsed.get(field)
        if value:
            candidates.append(str(value))

    # For named systems or document IDs, enforce entity grounding check
    for doc in parsed.get("named_systems_or_documents", []):
        if doc and not entity_is_grounded(str(doc), finding_text):
            candidates.append(str(doc))

    return [c for c in candidates if c and not phrase_is_grounded(c, source_words)]


def _to_extraction_result(parsed: dict) -> ExtractionResult:
    statements: list[AttributedStatement] = []
    for s in parsed.get("attributed_statements", []):
        if isinstance(s, dict):
            claim = str(s.get("claim") or s.get("statement") or "").strip()
            if claim:
                speaker = str(s.get("speaker") or "").strip()
                statements.append(AttributedStatement(speaker=speaker, claim=claim))
        elif isinstance(s, str) and s.strip():
            statements.append(AttributedStatement(speaker="", claim=s.strip()))

    return ExtractionResult(
        stated_facts=[str(x).strip() for x in parsed.get("stated_facts", []) if str(x).strip()],
        attributed_statements=statements,
        referenced_records=[str(x).strip() for x in parsed.get("referenced_records", []) if str(x).strip()],
        named_systems_or_documents=[str(x).strip() for x in parsed.get("named_systems_or_documents", []) if str(x).strip()],
        timeframe=parsed.get("timeframe") or None,
        asset_or_location=parsed.get("asset_or_location") or None,
        external_impact_stated=bool(parsed.get("external_impact_stated", False)),
        deviation_subject=parsed.get("deviation_subject") or None,
        deviation_condition=parsed.get("deviation_condition") or None,
        deviation_actor=parsed.get("deviation_actor") or None,
    )


async def extract_finding(finding_text: str, client: LLMClient | None = None) -> ExtractionResult:
    settings = get_settings()
    template = (settings.prompts_dir / "extraction_prompt.txt").read_text(encoding="utf-8")
    prompt = template.format(observation=finding_text)
    client = client or get_llm_client()

    messages = [{"role": "user", "content": prompt}]

    async def _attempt(msgs: list[dict[str, str]], temperature: float) -> dict | None:
        try:
            raw = await client.chat_completion(
                msgs,
                temperature=temperature,
                response_format_json=True,
                max_tokens=settings.ollama_extraction_max_tokens,
                num_ctx=settings.ollama_extraction_num_ctx,
                node="extraction",
            )
            parsed = parse_llm_json(raw)
        except Exception as exc:
            # Broad catch: any provider failure (timeout, 429, malformed
            # response, etc.) must degrade to the caller's deterministic
            # fallback rather than crash the graph on its very first node.
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

    raise LLMError("Extraction failed: LLM call timed out or introduced ungrounded content.")
