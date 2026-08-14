"""Parallel combinator: runs impact_assessment and capa_analysis concurrently.

Both nodes only depend on root_cause output and the evidence ledger — they are
completely independent of each other. Running them in parallel saves one full
Ollama round-trip (~10-30 seconds on a local model).
"""

from __future__ import annotations

import asyncio
import logging

from app.agent.nodes.capa import capa_analysis_node
from app.agent.nodes.impact import impact_assessment_node
from app.agent.state import AgentState
from app.models.agent import AgentTraceStep

logger = logging.getLogger(__name__)


async def impact_capa_parallel_node(state: AgentState) -> AgentState:
    """Run impact assessment and CAPA analysis in parallel."""
    trace = list(state.get("trace", []))

    impact_task = asyncio.create_task(impact_assessment_node(state))
    capa_task = asyncio.create_task(capa_analysis_node(state))

    impact_result, capa_result = await asyncio.gather(
        impact_task, capa_task, return_exceptions=True
    )

    merged: dict = dict(state)

    if isinstance(impact_result, Exception):
        logger.error("Parallel impact assessment failed: %s", impact_result)
        trace.append(AgentTraceStep.warn(f"Impact assessment failed in parallel: {impact_result}"))
        merged["errors"] = list(merged.get("errors", [])) + [str(impact_result)]
    else:
        merged.update({
            "impact_assessment": impact_result.get("impact_assessment"),
        })
        trace.extend([t for t in impact_result.get("trace", []) if t not in state.get("trace", [])])
        merged["errors"] = list(merged.get("errors", [])) + impact_result.get("errors", [])

    if isinstance(capa_result, Exception):
        logger.error("Parallel CAPA analysis failed: %s", capa_result)
        trace.append(AgentTraceStep.warn(f"CAPA analysis failed in parallel: {capa_result}"))
        merged["errors"] = list(merged.get("errors", [])) + [str(capa_result)]
    else:
        merged.update({
            "capa_analysis": capa_result.get("capa_analysis"),
        })
        trace.extend([t for t in capa_result.get("trace", []) if t not in state.get("trace", [])])
        merged["errors"] = list(merged.get("errors", [])) + capa_result.get("errors", [])

    merged["trace"] = trace
    return merged
