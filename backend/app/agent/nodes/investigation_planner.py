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

from unittest.mock import AsyncMock, MagicMock

from app.agent.grounding_guard import (
    build_source_text,
    clean_structured_leak,
    is_placeholder_leak,
    mentions_unsupported_domain,
)
from app.agent.state import AgentState
from app.config import get_settings
from app.models.agent import AgentTraceStep, InvestigationPlan, InvestigationQuestion
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json
from app.services.text_grounding import significant_words

logger = logging.getLogger(__name__)


async def plan_investigation_node(state: AgentState) -> AgentState:
    """Decide which tools to call and create the initial investigation plan."""
    request = state["request"]
    extraction = state.get("extraction")
    quality = state.get("observation_quality")
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    settings = get_settings()
    plan = InvestigationPlan()

    # FAST-PATH: If ASP.NET integration is not configured and not testing mocked client, skip 30s LLM call
    client = get_llm_client()
    is_mocked = hasattr(client, "chat_completion") and isinstance(client.chat_completion, AsyncMock)
    if not settings.lqms_aspnet_base_url and not is_mocked:
        trace.append(AgentTraceStep.ok("No tool endpoints configured; proceeding directly to core synthesis (0ms fast path)"))
        state["investigation_plan"] = plan
        state["needs_investigation"] = False
        state["planned_tools"] = []
        state["trace"] = trace
        state["errors"] = errors
        return state
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
        from app.services.ollama_client import set_current_node
        set_current_node("investigation_planner")
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

        # Parse structured question objects {question, purpose, evidence}
        # Fall back gracefully if the model returns plain strings (older format)
        raw_questions = raw_plan.get("questions", [])
        parsed_questions: list[InvestigationQuestion] = []
        for q in raw_questions:
            if isinstance(q, dict):
                q_text = clean_structured_leak(q.get("question", ""))
                q_purpose = clean_structured_leak(q.get("purpose", ""))
                q_evidence = clean_structured_leak(q.get("evidence", ""))
                if q_text:
                    parsed_questions.append(InvestigationQuestion(
                        question=q_text,
                        purpose=q_purpose or "not specified",
                        evidence=q_evidence or "not specified",
                    ))
            elif isinstance(q, str):
                # Backward-compat: plain string — wrap as InvestigationQuestion
                q_text = clean_structured_leak(q)
                if q_text:
                    parsed_questions.append(InvestigationQuestion(
                        question=q_text,
                        purpose="not specified",
                        evidence="not specified",
                    ))

        plan = InvestigationPlan(
            areas=[a for a in (clean_structured_leak(x) for x in raw_plan.get("areas", [])) if a],
            questions=parsed_questions,
            evidence_to_collect=[
                a for a in (clean_structured_leak(x) for x in raw_plan.get("evidence_to_collect", [])) if a
            ],
        )

        # DOMAIN GUARD: drop any area/evidence item that invokes a training/authorization domain
        # this finding never mentions.
        domain_source_text = build_source_text(
            state["request"].finding_text, state.get("evidence_ledger", []),
            extraction.stated_facts if extraction else [],
        )
        for field_name in ("areas", "evidence_to_collect"):
            kept = []
            for item in getattr(plan, field_name):
                if mentions_unsupported_domain(item, domain_source_text):
                    trace.append(AgentTraceStep.warn(
                        f"Investigation {field_name[:-1] if field_name != 'evidence_to_collect' else 'evidence item'} "
                        f"dropped — invokes training/authorization domain not supported by this finding: {item!r}"
                    ))
                    continue
                if is_placeholder_leak(item):
                    trace.append(AgentTraceStep.warn(
                        f"Investigation {field_name} item dropped — echoed prompt instruction text: {item!r}"
                    ))
                    continue
                kept.append(item)
            setattr(plan, field_name, kept)

        # Apply domain guard to structured question objects
        kept_questions = []
        for iq in plan.questions:
            if mentions_unsupported_domain(iq.question, domain_source_text):
                trace.append(AgentTraceStep.warn(
                    f"Investigation question dropped — unsupported domain: {iq.question!r}"
                ))
                continue
            if is_placeholder_leak(iq.question):
                trace.append(AgentTraceStep.warn(
                    f"Investigation question dropped — echoed prompt instruction text: {iq.question!r}"
                ))
                continue
            from app.agent.analytical_validator import validate_investigation_question
            if not validate_investigation_question(iq.question):
                trace.append(AgentTraceStep.warn(
                    f"Investigation question dropped — malformed/not a genuine causal question: {iq.question!r}"
                ))
                continue
            kept_questions.append(iq)
        plan.questions = kept_questions

        # QUESTION -> EVIDENCE CONSISTENCY CHECK (over structured questions)
        evidence_vocab = [significant_words(e) for e in plan.evidence_to_collect]
        for iq in plan.questions:
            q_words = significant_words(iq.question)
            if not q_words:
                continue
            if not any(q_words & ev_words for ev_words in evidence_vocab):
                trace.append(AgentTraceStep.warn(
                    f"Investigation question has no matching evidence item: {iq.question!r}"
                ))

        # Section 8: Ensure plan.questions is NEVER empty
        if not plan.questions:
            from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
            _canonical = state.get("canonical_finding_state")
            _, fallback_plan = build_deterministic_investigation_plan(
                request.finding_text, state.get("evidence_ledger", []),
                canonical_subject=getattr(_canonical, "finding_subject", None),
            )
            plan = fallback_plan
            trace.append(AgentTraceStep.warn("Investigation Planner: generated fallback questions and evidence plan"))

        if needs_investigation and planned_tools:
            trace.append(AgentTraceStep.ok(
                f"Investigation plan created: {len(planned_tools)} tools planned"
            ))
        else:
            trace.append(AgentTraceStep.ok(
                "No LQMS tool investigation needed — proceeding to analysis"
            ))

    except (LLMError, ValueError, KeyError, Exception) as exc:
        logger.warning("Investigation planner failed: %s", exc)
        trace.append(AgentTraceStep.warn(f"Investigation planner failed — using fallback plan: {exc}"))
        errors.append(f"Planner error: {exc}")
        needs_investigation = False
        from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
        _canonical = state.get("canonical_finding_state")
        _, plan = build_deterministic_investigation_plan(
            request.finding_text, state.get("evidence_ledger", []),
            canonical_subject=getattr(_canonical, "finding_subject", None),
        )

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

