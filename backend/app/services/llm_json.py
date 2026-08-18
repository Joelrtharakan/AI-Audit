"""Defensive handling for raw LLM output and legacy compatibility module.

Delegates directly to the unified `app.services.llm.json_parser`.
"""

from __future__ import annotations

from app.services.llm.json_parser import (
    extract_json_str,
    normalize_llm_output,
    parse_llm_json,
    strip_html,
    validate_llm_schema,
)

__all__ = [
    "extract_json_str",
    "parse_llm_json",
    "validate_llm_schema",
    "normalize_llm_output",
    "strip_html",
]
