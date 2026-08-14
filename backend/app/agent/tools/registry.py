"""Tool registry — the strict allowlist of read-only tools the agent may invoke.

Security design:
  - The agent selects a tool NAME from APPROVED_TOOLS.
  - execute_tool() in tool_executor.py looks up the name in TOOL_REGISTRY.
  - Any name not in APPROVED_TOOLS raises PermissionError BEFORE the call.
  - All tools are read-only. Write operations are structurally impossible:
    the ASP.NET client only exposes GET methods.

To add a new tool:
  1. Implement it in aspnet_lqms_client.py
  2. Add a wrapper function here
  3. Add the name to APPROVED_TOOLS and TOOL_REGISTRY
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# The complete set of tool names the agent is allowed to invoke.
APPROVED_TOOLS: frozenset[str] = frozenset({
    "get_finding",
    "get_audit",
    "get_related_capa",
    "get_capa_history",
    "get_previous_findings",
    "get_audit_history",
    "get_equipment",
    "get_training_record",
    "get_department_information",
    "search_knowledge",
})

ToolFn = Callable[..., Coroutine[Any, Any, Any]]


def get_tool_registry() -> dict[str, ToolFn]:
    """Build the tool registry lazily so the LQMS client is only instantiated
    when the agent actually runs (allows config to be loaded first)."""
    from app.agent.tools.aspnet_lqms_client import AspNetLQMSClient
    from app.agent.tools.knowledge_tool import search_knowledge

    client = AspNetLQMSClient()

    return {
        "get_finding": client.get_finding,
        "get_audit": client.get_audit,
        "get_related_capa": client.get_related_capa,
        "get_capa_history": client.get_capa_history,
        "get_previous_findings": client.get_previous_findings,
        "get_audit_history": client.get_audit_history,
        "get_equipment": client.get_equipment,
        "get_training_record": client.get_training_record,
        "get_department_information": client.get_department_information,
        "search_knowledge": search_knowledge,
    }


async def call_tool(tool_name: str, kwargs: dict[str, Any]) -> Any:
    """Execute a named tool with the given arguments.

    Raises PermissionError if the tool name is not in APPROVED_TOOLS.
    This check is in code, not in a prompt.
    """
    if tool_name not in APPROVED_TOOLS:
        raise PermissionError(
            f"Agent attempted to call unauthorized tool '{tool_name}'. "
            f"Approved tools: {sorted(APPROVED_TOOLS)}"
        )
    registry = get_tool_registry()
    fn = registry[tool_name]
    logger.info("Calling tool: %s kwargs=%s", tool_name, list(kwargs.keys()))
    return await fn(**kwargs)
