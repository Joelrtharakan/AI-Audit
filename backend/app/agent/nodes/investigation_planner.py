"""Node 2: plan_investigation

Uses the LLM to decide which LQMS tools to call and in what order, based on the
finding content and extraction. Tool arguments are derived ONLY from the finding
context — the LLM cannot invent record IDs or department names not present.

If the LLM cannot determine useful tools, needs_investigation is set to False
and the agent proceeds directly to RCA with only the finding text as evidence.
"""

from __future__ import annotations

import json
import logging

from app.agent.state import AgentState
from app.config import get_settings
from app.models.agent import AgentTraceStep, InvestigationPlan
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)


async def plan_investigation_node(state: AgentState) -> AgentState:
    """Decide which tools to call and create the initial investigation plan."""
    request = state["request"]
    extraction = state.get("extraction")
    quality = state.get("observation_quality")
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    settings = get_settings()

    client = get_llm_client()
    system_prompt = (settings.agent_prompts_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (settings.agent_prompts_dir / "investigation_planner.txt").read_text(encoding="utf-8")

    departments_str = ", ".join(request.departments) if request.departments else "not provided"
    missing_str = "\n".join(f"- {m}" for m in (quality.missing_information if quality else [])) or "(none)"

    prompt = template.format(
        finding_text=request.finding_text,
        ca_number=request.ca_number or "not provided",
        audit_number=request.audit_number or "not provided",
        audit_date=request.audit_date or "not provided",
        clause_number=request.clause_number or "not provided",
        departments=departments_str,
        nature_of_nc=request.nature_of_nc or "not provided",
        audit_criteria=request.audit_criteria or "not provided",
        area_audited=request.area_audited or "not provided",
        finding_type=request.finding_type or "not provided",
        severity=request.severity or "not provided",
        likelihood=request.likelihood or "not provided",
        risk_result=request.risk_result or "not provided",
        observation_quality=quality.status.value if quality else "UNKNOWN",
        missing_information=missing_str,
        extraction_json=extraction.model_dump_json(indent=2) if extraction else "{}",
    )

    needs_investigation = False
    planned_tools: list[str] = []
    plan = InvestigationPlan()

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

        needs_investigation = bool(parsed.get("needs_investigation", False))
        raw_tools = parsed.get("planned_tools", [])

        # Validate and filter tool names against the approved list
        from app.agent.tools.registry import APPROVED_TOOLS
        valid_tools = []
        for t in raw_tools:
            if not isinstance(t, dict):
                continue
            tool_name = str(t.get("tool", ""))
            if tool_name not in APPROVED_TOOLS:
                logger.warning("Planner requested unapproved tool '%s' — skipped.", tool_name)
                continue
            args = t.get("args", {})
            # Sanitize args: only allow string/int values (no nested objects)
            safe_args = {k: v for k, v in args.items() if isinstance(v, (str, int, float))}
            valid_tools.append({"tool": tool_name, "args": safe_args})

        planned_tools = [t["tool"] for t in valid_tools]
        state["_planned_tool_calls"] = valid_tools  # store full spec with args

        raw_plan = parsed.get("investigation_plan", {})
        plan = InvestigationPlan(
            areas=[str(x) for x in raw_plan.get("areas", [])],
            questions=[str(x) for x in raw_plan.get("questions", [])],
            evidence_to_collect=[str(x) for x in raw_plan.get("evidence_to_collect", [])],
        )

        if needs_investigation and planned_tools:
            trace.append(AgentTraceStep.ok(
                f"Investigation plan created: {len(planned_tools)} tools planned"
            ))
        else:
            trace.append(AgentTraceStep.ok(
                "No LQMS tool investigation needed — proceeding to analysis"
            ))

    except (LLMError, ValueError, KeyError) as exc:
        logger.warning("Investigation planner failed: %s", exc)
        trace.append(AgentTraceStep.warn(f"Investigation planner failed — skipping tool phase: {exc}"))
        errors.append(f"Planner error: {exc}")
        needs_investigation = False

    return {
        **state,
        "needs_investigation": needs_investigation,
        "planned_tools": planned_tools,
        "completed_tools": [],
        "current_tool": None,
        "investigation_plan": plan,
        "trace": trace,
        "errors": errors,
    }
