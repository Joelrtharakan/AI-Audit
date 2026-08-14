"""Node 3: execute_tool

Executes the next planned tool from the tool registry.
Enforces:
  - Tool name must be in APPROVED_TOOLS (raises PermissionError if not — caught here)
  - Per-tool timeout from settings.agent_tool_timeout_seconds
  - Max tool calls from settings.agent_max_tool_calls

On any error (permission, timeout, HTTP), the error is recorded in the state
and the tool is marked complete so the loop continues.
"""

from __future__ import annotations

import asyncio
import logging

from app.agent.state import AgentState
from app.agent.tools.registry import APPROVED_TOOLS, call_tool
from app.config import get_settings
from app.models.agent import AgentTraceStep

logger = logging.getLogger(__name__)


async def execute_tool_node(state: AgentState) -> AgentState:
    """Execute the current planned tool and store the raw result."""
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    tool_results = dict(state.get("tool_results", {}))
    completed_tools = list(state.get("completed_tools", []))
    planned_tools = list(state.get("planned_tools", []))
    tool_call_count = state.get("tool_call_count", 0)
    settings = get_settings()

    # Find the next uncompleted tool
    next_tool = None
    next_tool_args = {}
    planned_tool_calls = state.get("_planned_tool_calls", [])

    for tool_spec in planned_tool_calls:
        tool_name = tool_spec.get("tool", "")
        if tool_name not in completed_tools:
            next_tool = tool_name
            next_tool_args = tool_spec.get("args", {})
            break

    if not next_tool:
        # No more tools to execute
        return {**state, "current_tool": None, "trace": trace, "errors": errors}

    # Hard limit on total tool calls
    if tool_call_count >= settings.agent_max_tool_calls:
        trace.append(AgentTraceStep.warn(
            f"Max tool calls ({settings.agent_max_tool_calls}) reached — skipping remaining tools"
        ))
        # Mark all remaining tools complete
        for tool_spec in planned_tool_calls:
            if tool_spec.get("tool") not in completed_tools:
                completed_tools.append(tool_spec.get("tool", ""))
        return {
            **state,
            "current_tool": None,
            "completed_tools": completed_tools,
            "trace": trace,
            "errors": errors,
        }

    # Security: validate tool name
    if next_tool not in APPROVED_TOOLS:
        msg = f"Security violation: agent attempted to call unauthorized tool '{next_tool}'"
        logger.error(msg)
        trace.append(AgentTraceStep.error(msg))
        errors.append(msg)
        completed_tools.append(next_tool)
        return {
            **state,
            "current_tool": None,
            "completed_tools": completed_tools,
            "tool_results": tool_results,
            "trace": trace,
            "errors": errors,
        }

    trace.append(AgentTraceStep.ok(f"Calling tool: {next_tool}"))

    try:
        result = await asyncio.wait_for(
            call_tool(next_tool, next_tool_args),
            timeout=settings.agent_tool_timeout_seconds,
        )
        if next_tool not in tool_results:
            tool_results[next_tool] = []
        tool_results[next_tool].append(result)
        tool_call_count += 1
        completed_tools.append(next_tool)
        trace.append(AgentTraceStep.ok(f"Tool {next_tool} completed"))

    except PermissionError as exc:
        msg = f"Tool permission error: {exc}"
        logger.error(msg)
        trace.append(AgentTraceStep.error(msg))
        errors.append(msg)
        completed_tools.append(next_tool)

    except asyncio.TimeoutError:
        msg = f"Tool {next_tool} timed out after {settings.agent_tool_timeout_seconds}s"
        logger.warning(msg)
        trace.append(AgentTraceStep.warn(msg))
        errors.append(msg)
        completed_tools.append(next_tool)

    except Exception as exc:
        msg = f"Tool {next_tool} failed: {exc}"
        logger.warning(msg)
        trace.append(AgentTraceStep.warn(msg))
        errors.append(msg)
        completed_tools.append(next_tool)

    return {
        **state,
        "current_tool": next_tool,
        "completed_tools": completed_tools,
        "tool_results": tool_results,
        "tool_call_count": tool_call_count,
        "trace": trace,
        "errors": errors,
    }
