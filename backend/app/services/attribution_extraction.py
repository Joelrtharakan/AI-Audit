"""Deterministic, LLM-free structural attribution extraction.

Used only when the LLM extraction call is unavailable (degraded mode). The
LLM-based extractor (app/services/extraction.py) normally distinguishes
VERIFIED stated_facts from REPORTED attributed_statements; when that call
fails, the previous fallback collapsed everything to plain sentence-level
VERIFIED facts, silently destroying the REPORTED/attribution distinction
for any causal statement already explicitly present in the finding text
(e.g. "The operator stated that they were unaware the procedure had been
revised"). This module recovers that distinction structurally -- verb-of-
report constructions and awareness/knowledge-gap constructions -- without
any finding-specific or domain-specific vocabulary.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# "<Speaker> stated/reported/confirmed/indicated/... that <claim>"
_REPORT_VERB_RE = re.compile(
    r"^(?P<speaker>[A-Z][\w\s-]{0,50}?)\s+"
    r"(?:stated|reported|confirmed|indicated|noted|mentioned|claimed|said|explained|advised|acknowledged)\s+"
    r"that\s+(?P<claim>.+)$",
    re.IGNORECASE,
)

# "<Speaker> was/were unaware/not aware/not informed/not notified (that|of) <claim>"
_AWARENESS_GAP_RE = re.compile(
    r"^(?P<speaker>[A-Z][\w\s-]{0,50}?)\s+(?:was|were)\s+"
    r"(?P<claim>(?:unaware|not\s+aware|not\s+informed|not\s+notified|not\s+told).*)$",
    re.IGNORECASE,
)

# "<Speaker> did not know / didn't know (that) <claim>"
_DID_NOT_KNOW_RE = re.compile(
    r"^(?P<speaker>[A-Z][\w\s-]{0,50}?)\s+(?:did\s+not|didn't)\s+know\s+"
    r"(?P<claim>.*)$",
    re.IGNORECASE,
)

_ATTRIBUTION_PATTERNS = (_REPORT_VERB_RE, _AWARENESS_GAP_RE, _DID_NOT_KNOW_RE)


def split_facts_and_attributed_statements(finding_text: str) -> tuple[list[str], list[dict]]:
    """Deterministic, LLM-free split of a finding into (VERIFIED sentence
    facts, REPORTED attributed statements) using structural attribution
    patterns -- verb-of-report constructions and awareness-gap
    constructions -- never domain vocabulary. Each attributed statement is
    returned as {"speaker": ..., "claim": ...} matching AttributedStatement.

    Falls back to treating a sentence as a plain fact when it doesn't match
    an attribution pattern -- this never loses information, it only adds
    the REPORTED classification where the sentence structure supports it."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(finding_text.strip()) if s.strip()]

    facts: list[str] = []
    attributed: list[dict] = []
    for sentence in sentences:
        matched = False
        for pattern in _ATTRIBUTION_PATTERNS:
            m = pattern.match(sentence)
            if m:
                speaker = m.group("speaker").strip()
                claim = m.group("claim").strip().rstrip(".")
                if claim:
                    attributed.append({"speaker": speaker, "claim": claim})
                    matched = True
                break
        if not matched:
            facts.append(sentence)

    return facts, attributed
