"""Node 4: record_evidence

After a tool executes, this node uses the LLM to classify each piece of information
from the tool result into EvidenceItems and adds them to the evidence ledger.

If the tool returned no data (is_available=False or empty), the node records
an EvidenceGap instead of fabricating evidence.

This is the hallucination firewall for tool results: the LLM is instructed to
classify, not fabricate, and the node validates the output before accepting it.
"""

from __future__ import annotations

import json
import logging

from app.agent.state import AgentState
from app.config import get_settings
from app.models.agent import AgentTraceStep, EvidenceGap, EvidenceItem, EvidenceStatus
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)

_VALID_STATUSES = {s.value for s in EvidenceStatus}


def _is_data_available(tool_result: object) -> bool:
    """Check if a tool result actually contains data."""
    if tool_result is None:
        return False
    if hasattr(tool_result, "is_available"):
        return bool(tool_result.is_available)
    if isinstance(tool_result, list):
        return len(tool_result) > 0
    if isinstance(tool_result, str):
        return tool_result not in ("NOT_CONFIGURED", "")
    return True


def _serialize_result(result: object) -> str:
    """Serialize tool result to JSON string for the LLM."""
    if result is None:
        return "null"
    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump(), default=str)
    if isinstance(result, list):
        items = []
        for item in result:
            if hasattr(item, "model_dump"):
                items.append(item.model_dump())
            else:
                items.append(str(item))
        return json.dumps(items, default=str)
    return str(result)


async def record_evidence_node(state: AgentState) -> AgentState:
    """Classify tool results into evidence items and add to ledger."""
    current_tool = state.get("current_tool")
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    evidence_ledger = list(state.get("evidence_ledger", []))
    evidence_gaps = list(state.get("evidence_gaps", []))

    if not current_tool:
        return {**state, "trace": trace, "errors": errors}

    tool_results_map = state.get("tool_results", {})
    tool_result_list = tool_results_map.get(current_tool, [])
    latest_result = tool_result_list[-1] if tool_result_list else None

    data_available = _is_data_available(latest_result)
    tool_result_str = _serialize_result(latest_result)

    if not data_available:
        # Record gap instead of fabricating
        gap = EvidenceGap(
            claim=f"Data from {current_tool}",
            missing=f"Tool {current_tool} returned no data — LQMS may not be configured or record not found.",
            evidence_status=EvidenceStatus.UNKNOWN,
        )
        evidence_gaps.append(gap)
        trace.append(AgentTraceStep.warn(f"No data from {current_tool} — gap recorded"))
        return {
            **state,
            "evidence_gaps": evidence_gaps,
            "trace": trace,
            "errors": errors,
        }

    settings = get_settings()
    client = get_llm_client()
    system_prompt = (settings.agent_prompts_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (settings.agent_prompts_dir / "evidence_recorder.txt").read_text(encoding="utf-8")

    prompt = template.format(
        tool_name=current_tool,
        tool_result=tool_result_str,
        data_available="YES — data was returned",
        finding_text=state["request"].finding_text,
        evidence_ledger_json=json.dumps(
            [e.model_dump() for e in evidence_ledger], default=str
        ),
    )

    try:
        raw = await client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format_json=True,
            max_tokens=1024,
        )
        parsed = parse_llm_json(raw)

        # Parse new evidence items
        for item_dict in parsed.get("new_evidence_items", []):
            try:
                status_str = str(item_dict.get("status", "UNKNOWN")).upper()
                if status_str not in _VALID_STATUSES:
                    status_str = "UNKNOWN"
                evidence_ledger.append(EvidenceItem(
                    claim=str(item_dict.get("claim", "")),
                    source=str(item_dict.get("source", current_tool)),
                    source_reference=item_dict.get("source_reference") or None,
                    status=EvidenceStatus(status_str),
                    relevance=str(item_dict.get("relevance", "")),
                    notes=item_dict.get("notes") or None,
                ))
            except Exception as exc:
                logger.warning("Failed to parse evidence item: %s", exc)

        # Parse new gaps
        for gap_dict in parsed.get("evidence_gaps", []):
            try:
                evidence_gaps.append(EvidenceGap(
                    claim=str(gap_dict.get("claim", "")),
                    missing=str(gap_dict.get("missing", "")),
                    evidence_status=EvidenceStatus.UNKNOWN,
                ))
            except Exception as exc:
                logger.warning("Failed to parse evidence gap: %s", exc)

        new_items = len(parsed.get("new_evidence_items", []))
        trace.append(AgentTraceStep.ok(
            f"Evidence recorded from {current_tool}: {new_items} items"
        ))

    except Exception as exc:
        # Broad catch: any LLM-path failure must degrade to a logged
        # evidence gap rather than crash the graph mid-investigation.
        logger.warning("Evidence recorder failed for %s: %s", current_tool, exc)
        trace.append(AgentTraceStep.warn(f"Evidence recorder failed for {current_tool} — gap noted"))
        errors.append(f"Evidence recorder error ({current_tool}): {exc}")
        evidence_gaps.append(EvidenceGap(
            claim=f"Classified evidence from {current_tool}",
            missing="Evidence classification LLM call failed.",
            evidence_status=EvidenceStatus.UNKNOWN,
        ))

    return {
        **state,
        "evidence_ledger": evidence_ledger,
        "evidence_gaps": evidence_gaps,
        "trace": trace,
        "errors": errors,
    }
