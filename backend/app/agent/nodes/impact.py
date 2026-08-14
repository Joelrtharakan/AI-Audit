"""Node 6: impact_assessment

Assesses what is known and unknown about impact based on evidence.
Prevents recall/quarantine language without external impact evidence.
"""

from __future__ import annotations

import json
import logging

from app.agent.state import AgentState
from app.config import get_settings
from app.models.agent import AgentTraceStep, ImpactAssessment, ImpactStatus
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)


async def impact_assessment_node(state: AgentState) -> AgentState:
    """Assess impact from evidence ledger."""
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    root_cause = state.get("root_cause")
    evidence_ledger = state.get("evidence_ledger", [])
    extraction = state.get("extraction")
    settings = get_settings()
    client = get_llm_client()

    system_prompt = (settings.agent_prompts_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (settings.agent_prompts_dir / "impact.txt").read_text(encoding="utf-8")

    external_impact = extraction.external_impact_stated if extraction else False

    prompt = template.format(
        finding_text=state["request"].finding_text,
        root_cause_status=root_cause.status if root_cause else "NOT_ESTABLISHED",
        root_cause_narrative=root_cause.narrative if root_cause else "Not established.",
        evidence_ledger_json=json.dumps(
            [e.model_dump() for e in evidence_ledger], default=str
        ),
        extraction_json=extraction.model_dump_json(indent=2) if extraction else "{}",
    )

    # Safe default
    impact = ImpactAssessment(
        status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT,
        areas=["Determine scope and period of potential impact — auditor assessment required."],
        narrative=None,
    )

    try:
        raw = await client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format_json=True,
            max_tokens=512,
        )
        parsed = parse_llm_json(raw)
        raw_ia = parsed.get("impact_assessment", {})

        status_str = str(raw_ia.get("status", "IMPACT_REQUIRES_ASSESSMENT")).upper()
        try:
            status = ImpactStatus(status_str)
        except ValueError:
            status = ImpactStatus.IMPACT_REQUIRES_ASSESSMENT

        areas = [str(x) for x in raw_ia.get("areas", [])]
        narrative = raw_ia.get("narrative") or None

        # Firewall: strip recall/quarantine language if no external impact stated
        if not external_impact:
            import re
            recall_re = re.compile(r"\b(recall|quarantine|notify\s+\w*\s*customer|inform\s+\w*\s*client)\b", re.IGNORECASE)
            areas = [a for a in areas if not recall_re.search(a)]
            if narrative and recall_re.search(narrative):
                narrative = None
                trace.append(AgentTraceStep.warn(
                    "Recall/quarantine language removed from impact — no external impact stated"
                ))

        impact = ImpactAssessment(status=status, areas=areas, narrative=narrative)
        trace.append(AgentTraceStep.ok(f"Impact assessed: {impact.status.value}"))

    except (LLMError, ValueError, KeyError) as exc:
        logger.warning("Impact assessment failed: %s", exc)
        trace.append(AgentTraceStep.warn(f"Impact assessment failed — defaulting to REQUIRES_ASSESSMENT"))
        errors.append(f"Impact error: {exc}")

    return {**state, "impact_assessment": impact, "trace": trace, "errors": errors}
