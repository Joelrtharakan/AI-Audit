"""Provider-neutral defensive JSON extraction and schema validation.

Handles raw LLM completion output across all providers (Ollama, Copilot, etc.),
defending against markdown code fences, think reasoning blocks, conversational
preambles, trailing commas, and malformed wrapper payloads.
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
_DANGLING_KEY_RE = re.compile(r',\s*"(?:[^"\\]|\\.)*"\s*:?\s*$')
_TRAILING_KEY_RE = re.compile(r'\s*"(?:[^"\\]|\\.)*"\s*:\s*$')


def _close_open_structs(text: str) -> str:
    """Append the closing brackets/braces (and a closing quote, if a
    string is open) needed to balance `text`. Does not trim `text`."""
    stack: list[str] = []
    in_str = False
    escaped = False
    for ch in text:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    suffix = '"' if in_str else ""
    return text + suffix + "".join(reversed(stack))


def _repair_truncated_json(text: str) -> str:
    """Best-effort recovery of a response cut off by the model's output
    token limit (a real, provider-neutral failure mode: a semantically
    correct interpretation whose JSON simply stopped mid-structure).

    Walks truncation points backwards from the end; at each, strips a
    trailing comma / dangling `"key":`, balances the open structures, and
    tries to parse. Returns the first candidate that parses, else the
    original text unchanged. Never invents field values -- a list/object
    simply ends early, which the downstream schema + validator handle as a
    partial interpretation.
    """
    if not text or text[0] not in "{[":
        return text
    limit = min(len(text), 6000)
    for drop in range(0, limit):
        end = len(text) - drop
        cand = text[:end]
        # inside a string? cut back to before its opening quote
        m = re.search(r'"(?:[^"\\]|\\.)*$', cand)
        if m and (cand.count('"') - _escaped_quote_count(cand)) % 2 == 1:
            cand = cand[: m.start()]
        prev = None
        while prev != cand:
            prev = cand
            cand = cand.rstrip()
            cand = _DANGLING_KEY_RE.sub("", cand)
            cand = _TRAILING_KEY_RE.sub("", cand)
            cand = _TRAILING_COMMA_RE.sub(r"\1", cand)
            if cand.endswith(","):
                cand = cand[:-1]
        if not cand:
            break
        candidate = _close_open_structs(cand)
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        # Only accept a repair that recovered real content: a truncated
        # response keeps almost all of its text and at least one field. A
        # near-total rewrite (e.g. "{bad" -> "{}") means the failure was
        # malformed output, not truncation -- let that raise honestly.
        if isinstance(obj, dict) and obj and len(candidate) >= 0.5 * len(text):
            return candidate
    return text


def _escaped_quote_count(s: str) -> int:
    return len(re.findall(r'\\"', s))


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
                text = text[start_idx : end_idx + 1].strip()

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
    except json.JSONDecodeError:
        repaired = _repair_truncated_json(extracted)
        if repaired != extracted:
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    logger.info(
                        "Recovered a truncated LLM JSON response (%d -> %d chars) by closing "
                        "open structures; downstream schema/validator treats it as partial.",
                        len(extracted),
                        len(repaired),
                    )
                    return parsed
            except Exception:
                pass
        raise
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


def validate_llm_schema(data: dict[str, Any], required_keys: list[str]) -> list[str]:
    """Return a list of missing top-level keys from the parsed dictionary."""
    return [key for key in required_keys if key not in data]


def normalize_llm_output(raw: str) -> dict[str, Any]:
    """Convenience helper combining JSON extraction and normalization."""
    return parse_llm_json(raw)


def strip_html(value: str | None) -> str:
    """Remove HTML/XML tags from text fields."""
    if not value:
        return ""
    return _TAG_RE.sub("", value).strip()
