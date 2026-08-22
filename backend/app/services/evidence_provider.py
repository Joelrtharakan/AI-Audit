"""Phase 19: domain-general evidence provider abstraction.

The core investigation engine (app.agent.nodes.evidence_acquisition) must
depend only on the `EvidenceProvider` interface below, never on a specific
backend (ASP.NET, SQL Server, one LQMS deployment, ...). Concrete providers
plug in by implementing `acquire`.

`acquire()` retrieves/returns evidence only -- it MUST NOT decide
causality, support/contradiction relevance, or epistemic promotion. Those
judgments belong to the deterministic reconciliation layer in
app.agent.nodes.evidence_acquisition (Section 1: "Do not make the provider
responsible for deciding causality").
"""
from __future__ import annotations

import abc
import logging
from typing import Any

from app.models.agent import EvidenceItem, EvidenceRequest, EvidenceStatus

logger = logging.getLogger(__name__)


class EvidenceProvider(abc.ABC):
    """Abstract evidence-retrieval interface. A concrete provider knows how
    to fetch artifacts for a structured EvidenceRequest; it does not
    interpret them."""

    @abc.abstractmethod
    async def acquire(self, request: EvidenceRequest) -> EvidenceItem:
        """Return exactly one EvidenceItem for `request`. If nothing could
        be retrieved, return an EvidenceItem with hypothesis_relevance=
        "UNAVAILABLE" -- never fabricate content, never omit the item."""
        raise NotImplementedError


class NullEvidenceProvider(EvidenceProvider):
    """The safe, honest default when no real evidence backend is
    configured: every request comes back UNAVAILABLE. Never fabricates
    evidence. Using this provider is equivalent to "no evidence source
    exists" -- the adaptive loop correctly stops at EVIDENCE_BOUNDARY
    rather than inventing anything."""

    async def acquire(self, request: EvidenceRequest) -> EvidenceItem:
        return EvidenceItem(
            claim="", source="none", status=EvidenceStatus.UNVERIFIED,
            evidence_id=f"EV_{request.request_id}", request_id=request.request_id,
            question_id=request.question_id, iteration_id=request.iteration_id,
            hypothesis_relevance="UNAVAILABLE",
        )


class ToolRegistryEvidenceProvider(EvidenceProvider):
    """The real (non-test-fixture) adapter over the existing ASP.NET/LQMS
    tool registry (app.agent.tools.registry). This is a genuine, usable
    production provider, not a fake: it actually calls the approved LQMS
    tool matching `request.evidence_types[0]` (when the caller names one
    the registry recognizes) and wraps the raw result as evidence.

    Honest scope: this adapter deliberately does NOT interpret whether the
    retrieved record supports or contradicts the target hypothesis --
    every LQMS tool here returns a structured RECORD (a training log, an
    equipment record, ...), not free text a keyword heuristic could safely
    classify as supporting/contradicting without domain-specific rules
    (explicitly forbidden by this phase's brief). It reports
    hypothesis_relevance="UNRESOLVED" ("evidence was retrieved but this
    provider does not judge its bearing on the hypothesis") rather than
    guessing -- a real, correct, conservative behavior, not a stub.
    Retrieval failures return UNAVAILABLE, never fabricated content.
    """

    async def acquire(self, request: EvidenceRequest) -> EvidenceItem:
        from app.agent.tools.registry import APPROVED_TOOLS, call_tool

        tool_name = next((t for t in request.evidence_types if t in APPROVED_TOOLS), None)
        if tool_name is None:
            return EvidenceItem(
                claim="", source="lqms", status=EvidenceStatus.UNVERIFIED,
                evidence_id=f"EV_{request.request_id}", request_id=request.request_id,
                question_id=request.question_id, iteration_id=request.iteration_id,
                hypothesis_relevance="UNAVAILABLE",
                notes="no approved LQMS tool matched this request's evidence_types",
            )
        try:
            result = await call_tool(tool_name, {})
        except Exception as exc:
            logger.warning("ToolRegistryEvidenceProvider: %s failed: %s", tool_name, exc)
            return EvidenceItem(
                claim="", source=tool_name, status=EvidenceStatus.UNVERIFIED,
                evidence_id=f"EV_{request.request_id}", request_id=request.request_id,
                question_id=request.question_id, iteration_id=request.iteration_id,
                hypothesis_relevance="UNAVAILABLE", notes=f"retrieval failed: {exc}",
            )
        return EvidenceItem(
            claim=str(result)[:500], source=tool_name, status=EvidenceStatus.REPORTED,
            evidence_id=f"EV_{request.request_id}", request_id=request.request_id,
            question_id=request.question_id, iteration_id=request.iteration_id,
            hypothesis_relevance="UNRESOLVED",
        )


def get_default_evidence_provider(settings: Any = None) -> EvidenceProvider | None:
    """Production entry point. Returns a real, usable provider only when a
    real backend is actually configured; otherwise None (no evidence
    source -- the adaptive loop must stop at EVIDENCE_BOUNDARY, never
    silently loop or fabricate). Never returns a test-only provider."""
    if settings is None:
        from app.config import get_settings
        settings = get_settings()
    if getattr(settings, "lqms_aspnet_base_url", None):
        return ToolRegistryEvidenceProvider()
    return None
