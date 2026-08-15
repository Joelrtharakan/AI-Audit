"""Instruction & Prompt-Injection Detector.

Treats all user-provided finding text strictly as DATA. Detects embedded
imperative instructions (e.g., "ignore previous instructions", "close the CAPA",
"mark as resolved", "set root cause to...", "blame operator") and separates
them from legitimate factual evidence before evidence extraction.
"""

from __future__ import annotations

import re

# Comprehensive regex patterns matching instruction-like imperatives
_INSTRUCTION_PATTERNS = (
    r"\bignore\s+(all\s+)?(previous\s+)?instructions?\b",
    r"\b(close|approve|resolve|finalize)\s+(the\s+)?(capa|finding|case|audit|deviation|investigation|corrective\s+action)\b",
    r"\b(set|mark|change|modify)\s+.*?\b(root\s+cause|severity|status|confidence|risk|quality|resolved)\b",
    r"\b(do\s+not|don't|bypass)\s+(investigate|investigation|mention|check|look\s+into|verify)\b",
    r"\b(consider\s+this|assume\s+the|treat\s+the)\s+(verified|resolved|approved|effective|trained|cause)\b",
    r"\b(delete|hide|suppress|remove)\s+(evidence|records?|logs?)\b",
    r"\b(blame|attribute\s+to)\s+(the\s+)?(operator|technician|personnel|individual)\b",
    r"\b(system\s+prompt|new\s+system\s+instruction)\b",
)

_INSTRUCTION_RE = re.compile("|".join(_INSTRUCTION_PATTERNS), re.IGNORECASE)



def is_instruction(text: str) -> bool:
    """True if text matches any prompt-injection or imperative instruction pattern."""
    if not text:
        return False
    return bool(_INSTRUCTION_RE.search(text.strip()))


def filter_untrusted_instructions(sentences: list[str]) -> tuple[list[str], list[str]]:
    """Splits sentences into (legitimate_data, untrusted_instructions)."""
    data: list[str] = []
    instructions: list[str] = []
    for s in sentences:
        if is_instruction(s):
            instructions.append(s)
        else:
            data.append(s)
    return data, instructions
