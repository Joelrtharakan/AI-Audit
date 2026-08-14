"""Shared substring/fuzzy grounding primitives used by both extraction.py (Step 1
self-check) and grounding_validator.py (Step 3 post-generation check)."""

import re

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with", "was",
    "were", "is", "are", "not", "no", "that", "this", "had", "has", "have", "been",
    "being", "did", "does", "do", "after", "before", "during", "than", "then", "who",
    "what", "when", "where", "which", "their", "they", "them", "any", "all", "but",
}

_GROUNDING_THRESHOLD = 0.5


def significant_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def phrase_is_grounded(phrase: str, source_words: set[str]) -> bool:
    """Fuzzy check: is most of this phrase's vocabulary present in the source?"""
    phrase_words = significant_words(phrase)
    if not phrase_words:
        return True
    overlap = phrase_words & source_words
    return (len(overlap) / len(phrase_words)) >= _GROUNDING_THRESHOLD


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def entity_is_grounded(entity: str, source_text: str) -> bool:
    """Strict-ish check for a specific named entity (system name, ID, quantity): the
    normalized entity must appear as a substring of the normalized source. Deliberately
    stricter than phrase_is_grounded -- an invented specific shouldn't slip through on
    partial word overlap the way a paraphrased general phrase reasonably might."""
    norm_entity = normalize(entity)
    if not norm_entity:
        return True
    return norm_entity in normalize(source_text)
