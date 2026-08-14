"""Shared defensive handling for raw LLM chat completion output.

Never trust raw LLM output -- models occasionally wrap JSON in markdown code
fences despite instructions not to, so this strips those before parsing, and
any free text placed into frontend form fields is HTML-stripped first.
"""

import json
import re

_TAG_RE = re.compile(r"<[^>]+>")


def parse_llm_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
    return json.loads(cleaned)


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    return _TAG_RE.sub("", value).strip()
