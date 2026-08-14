"""RAG / knowledge-base tool stub.

Returns NOT_CONFIGURED until a real knowledge base is connected.
Never fabricates document results -- the agent records this gap in the
evidence ledger and continues without inventing SOP or policy content.

To connect a real knowledge base:
  1. Replace the body of search_knowledge() with your retrieval logic.
  2. Return a list of KnowledgeResult objects with real content.
  3. No other part of the agent architecture needs to change.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

NOT_CONFIGURED = "NOT_CONFIGURED"


class KnowledgeResult:
    def __init__(self, title: str, content: str, source: str, relevance_score: float):
        self.title = title
        self.content = content
        self.source = source
        self.relevance_score = relevance_score


async def search_knowledge(query: str, top_k: int = 3) -> list[KnowledgeResult] | str:
    """Search the organizational knowledge base.

    Returns NOT_CONFIGURED when no knowledge base is connected.
    The agent treats this as an evidence gap (source not available) and
    continues without inventing document content.
    """
    logger.info("Knowledge search called (query=%r) — knowledge base not configured.", query)
    return NOT_CONFIGURED
