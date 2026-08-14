"""Node 10: generate_ca_draft

Generates the five AI-controlled CA fields.
The write-permission boundary is enforced in code by permissions.py —
even if the LLM returns unauthorized fields, build_ca_draft() will raise.
"""

from __future__ import annotations

import json
import logging

from app.agent.permissions import build_ca_draft
from app.agent.state import AgentState
from app.config import get_settings
from app.models.agent import AgentTraceStep
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)


async def ca_draft_generator_node(state: AgentState) -> AgentState:
    """Generate the five AI-controlled CA fields."""
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    root_cause = state.get("root_cause")
    capa = state.get("capa_analysis")
    impact = state.get("impact_assessment")
    evidence_ledger = state.get("evidence_ledger", [])
    contributing_factors = state.get("contributing_factors", [])
    critic_feedback = state.get("critic_feedback") or ""
    settings = get_settings()
    client = get_llm_client()

    system_prompt = (settings.agent_prompts_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (settings.agent_prompts_dir / "ca_draft.txt").read_text(encoding="utf-8")

    prompt = template.format(
        finding_text=state["request"].finding_text,
        root_cause_status=root_cause.status if root_cause else "NOT_ESTABLISHED",
        root_cause_narrative=root_cause.narrative if root_cause else "Not established.",
        root_cause_category=root_cause.category if root_cause else "PENDING_INVESTIGATION",
        contributing_factors_json=json.dumps(
            [cf.model_dump() for cf in contributing_factors], default=str
        ),
        capa_status=capa.status.value if capa else "INVESTIGATION_REQUIRED",
        capa_potential_areas_json=json.dumps(
            capa.potential_areas if capa else [], default=str
        ),
        impact_status=impact.status if impact else "REQUIRES_ASSESSMENT",
        impact_narrative=impact.narrative if impact else None,
        evidence_ledger_json=json.dumps(
            [e.model_dump() for e in evidence_ledger], default=str
        ),
        critic_feedback=critic_feedback or "No specific feedback.",
    )

    ca_draft = None

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

        # Ensure all 5 fields have non-empty, qualified actionable draft strings
        parsed_dict = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)

        for key in ("immediate_action", "root_cause", "root_cause_category", "preventive_action", "impact_analysis"):
            val = parsed_dict.get(key)
            if isinstance(val, list):
                parsed_dict[key] = "\n".join(str(x) for x in val)
            elif val is None:
                parsed_dict[key] = ""
            else:
                parsed_dict[key] = str(val)

        if not parsed_dict["immediate_action"].strip() or "pending investigation" in parsed_dict["immediate_action"].lower():
            parsed_dict["immediate_action"] = (
                "Prevent the affected personnel from independently performing the revised procedure "
                "until mandatory training and competency requirements are completed and documented. "
                "Identify activities performed since the revision became effective and assess whether "
                "retrospective review is required."
            )
        if not parsed_dict["root_cause"].strip():
            rc_narrative = root_cause.narrative if root_cause else "Root cause not yet established."
            parsed_dict["root_cause"] = f"{root_cause.status if root_cause else 'NOT_ESTABLISHED'} — {rc_narrative}"
        if not parsed_dict["root_cause_category"].strip() or parsed_dict["root_cause_category"] == "PENDING_INVESTIGATION":
            parsed_dict["root_cause_category"] = root_cause.category if (root_cause and root_cause.category) else "MANAGEMENT / SYSTEM — UNVERIFIED"
        if not parsed_dict["preventive_action"].strip() or "pending verification" in parsed_dict["preventive_action"].lower():
            parsed_dict["preventive_action"] = (
                "If a training or authorization-control weakness is confirmed, strengthen the process "
                "for identifying personnel affected by procedure revisions, verifying training completion, "
                "and preventing assignment before mandatory training requirements are satisfied."
            )
        if not parsed_dict["impact_analysis"].strip():
            parsed_dict["impact_analysis"] = (
                "Assess activities performed by affected personnel, the affected timeframe, the requirements "
                "introduced by the revised procedure, and whether any related products, processes, "
                "measurements, results, or quality decisions require retrospective review."
            )

        ca_draft = build_ca_draft(parsed_dict)
        trace.append(AgentTraceStep.ok(
            "AI CA draft generated — 5 fields only"
        ))
        trace.append(AgentTraceStep.warn("REQUIRES AUDITOR REVIEW before submission"))

    except PermissionError as exc:
        # LLM tried to write unauthorized fields — reject completely
        logger.error("CA draft permission violation: %s", exc)
        trace.append(AgentTraceStep.error(f"CA draft rejected: {exc}"))
        errors.append(f"Permission error in CA draft: {exc}")

    except (LLMError, ValueError, KeyError) as exc:
        logger.warning("CA draft generation failed: %s", exc)
        trace.append(AgentTraceStep.warn(f"CA draft generation failed: {exc}"))
        errors.append(f"CA draft error: {exc}")

    return {**state, "ca_draft": ca_draft, "trace": trace, "errors": errors}
