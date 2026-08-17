"""Shared defensive handling for raw LLM chat completion output.

Never trust raw LLM output -- models occasionally wrap JSON in markdown code
fences despite instructions not to, prepend think tags, or append explanatory prose.
This module extracts the outermost valid JSON object/array cleanly.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")


def extract_json_str(raw: str) -> str:
    """Extract a candidate JSON string from raw LLM text.

    Handles:
      - <think>...</think> reasoning blocks
      - ```json ... ``` markdown code fences
      - Leading/trailing conversational text
      - Trailing commas before closing braces/brackets
    """
    if not raw:
        return ""

    # 1. Strip think tags
    text = _THINK_RE.sub("", raw).strip()

    # 2. Check for markdown code fences
    fence_matches = _CODE_FENCE_RE.findall(text)
    if fence_matches:
        # Prefer the largest code fence block if multiple exist
        text = max(fence_matches, key=len).strip()

    # 3. Locate outermost JSON object ({...}) or array ([...])
    if not (text.startswith("{") or text.startswith("[")):
        first_brace = text.find("{")
        first_bracket = text.find("[")

        start_idx = -1
        if first_brace != -1 and first_bracket != -1:
            start_idx = min(first_brace, first_bracket)
        elif first_brace != -1:
            start_idx = first_brace
        elif first_bracket != -1:
            start_idx = first_bracket

        if start_idx != -1:
            last_brace = text.rfind("}")
            last_bracket = text.rfind("]")
            end_idx = max(last_brace, last_bracket)
            if end_idx > start_idx:
                text = text[start_idx:end_idx + 1].strip()

    # 4. Remove illegal trailing commas before closing braces/brackets
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", text)
    return cleaned


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Parse raw LLM output into a dictionary with defensive JSON extraction."""
    if not raw or not raw.strip():
        raise ValueError("Empty LLM response received for JSON parsing.")

    extracted = extract_json_str(raw)
    try:
        parsed = json.loads(extracted)
        if not isinstance(parsed, dict):
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                return parsed[0]
            raise ValueError(f"Parsed JSON must be an object/dict, got {type(parsed).__name__}")
        return parsed
    except Exception as exc:
        logger.debug(
            "JSON parse failure: raw_length=%d extracted_length=%d error=%s first_200=%r last_200=%r",
            len(raw),
            len(extracted),
            str(exc),
            raw[:200] if raw else "",
            raw[-200:] if raw else "",
        )
        raise exc


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    return _TAG_RE.sub("", value).strip()

