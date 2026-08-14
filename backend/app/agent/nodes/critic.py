"""Node 8: critic

Self-review of the entire analysis. Checks evidence grounding, root cause
validity, CAPA appropriateness, impact claims, and 5-Why integrity.

Can:
  - Approve the analysis
  - Reject with specific corrections needed (but continue to report generation)
  - Request another investigation round (capped by agent_max_critic_iterations)
"""

from __future__ import annotations

import json
import logging

from app.agent.state import AgentState
from app.config import get_settings
from app.models.agent import AgentTraceStep
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)


async def critic_node(state: AgentState) -> AgentState:
    """Self-review the analysis for evidence grounding and correctness."""
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    critic_iteration = state.get("critic_iteration", 0)
    evidence_ledger = state.get("evidence_ledger", [])
    root_cause = state.get("root_cause")
    five_why = state.get("five_why")
    capa = state.get("capa_analysis")
    impact = state.get("impact_assessment")
    settings = get_settings()
    client = get_llm_client()

    system_prompt = (settings.agent_prompts_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (settings.agent_prompts_dir / "critic.txt").read_text(encoding="utf-8")

    prompt = template.format(
        finding_text=state["request"].finding_text,
        evidence_ledger_json=json.dumps(
            [e.model_dump() for e in evidence_ledger], default=str
        ),
        root_cause_status=root_cause.status if root_cause else "NOT_ESTABLISHED",
        root_cause_narrative=root_cause.narrative if root_cause else "Not established.",
        contributing_factors_json=json.dumps(
            [cf.model_dump() for cf in state.get("contributing_factors", [])], default=str
        ),
        five_why_json=five_why.model_dump_json(indent=2) if five_why else "{}",
        capa_status=capa.status.value if capa else "INVESTIGATION_REQUIRED",
        impact_status=impact.status if impact else "REQUIRES_ASSESSMENT",
    )

    # Default: approve with warning if LLM fails
    approved = True
    send_back = False
    critic_feedback = "Critic review unavailable — LLM error. Proceeding to report generation."

    try:
        raw = await client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format_json=True,
            max_tokens=1024,
        )
        parsed = parse_llm_json(raw)

        approved = bool(parsed.get("approved", True))
        send_back = bool(parsed.get("send_back_for_investigation", False))
        critic_feedback = str(parsed.get("feedback", ""))
        corrections = [str(c) for c in parsed.get("corrections_required", [])]
        issues = parsed.get("issues", [])

        # Cap send_back at max_critic_iterations
        if send_back and critic_iteration >= settings.agent_max_critic_iterations:
            logger.warning("Critic wants to send back but max iterations reached — proceeding to report")
            send_back = False
            trace.append(AgentTraceStep.warn(
                "Critic review: max iterations reached — proceeding despite outstanding issues"
            ))

        if approved and not corrections:
            trace.append(AgentTraceStep.ok("Critic review: analysis approved"))
        else:
            blocking = [i for i in issues if i.get("severity") == "BLOCKING"]
            trace.append(AgentTraceStep.warn(
                f"Critic review: {len(blocking)} blocking issues, {len(corrections)} corrections needed"
            ))
            for correction in corrections[:5]:  # log first 5
                trace.append(AgentTraceStep.warn(f"  Critic: {correction}"))

    except (LLMError, ValueError, KeyError) as exc:
        logger.warning("Critic node failed: %s", exc)
        trace.append(AgentTraceStep.warn(f"Critic review failed — proceeding to report: {exc}"))
        errors.append(f"Critic error: {exc}")

    return {
        **state,
        "critic_approved": approved,
        "critic_send_back": send_back,
        "critic_feedback": critic_feedback,
        "critic_iteration": critic_iteration + 1,
        "trace": trace,
        "errors": errors,
    }
