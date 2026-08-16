"""LEGACY NODE — NOT PART OF LIVE GRAPH.
This module exists only for isolated unit-test guard checks.
core_synthesis_node is the SOLE authoritative synthesis path.

Node 6: impact_assessment (Legacy)
"""

from __future__ import annotations

import json
import logging

from app.agent.grounding_guard import (
    build_source_text,
    clean_structured_leak,
    filter_list_field,
    is_placeholder_leak,
    ungrounded_entities,
)
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

        # clean_structured_leak (not str()) here: a weak model asked for plain
        # strings sometimes returns nested dicts/objects instead.
        areas = [clean_structured_leak(x) for x in raw_ia.get("areas", [])]
        areas = [a for a in areas if a]
        narrative = clean_structured_leak(raw_ia.get("narrative")) or None

        # Parse new structured fields (all optional — only populated if model provides them)
        affected_object = clean_structured_leak(raw_ia.get("affected_object")) or None
        affected_people = clean_structured_leak(raw_ia.get("affected_people")) or None
        affected_period = clean_structured_leak(raw_ia.get("affected_period")) or None
        process_at_risk = clean_structured_leak(raw_ia.get("process_at_risk")) or None
        relevant_change = clean_structured_leak(raw_ia.get("relevant_change")) or None
        potential_effect = clean_structured_leak(raw_ia.get("potential_effect")) or None
        impact_evidence_needed = clean_structured_leak(raw_ia.get("evidence_needed")) or None


        # GROUNDING GUARD: drop any impact area, and null any narrative, that
        # references an entity/number not present in this finding's text or
        # evidence ledger — prevents another case's timeframe/entities from
        # surviving into this finding's impact assessment.
        extra_trusted = [root_cause.narrative] if root_cause and root_cause.narrative else []
        source_text = build_source_text(state["request"].finding_text, evidence_ledger, extra_trusted)
        areas, dropped = filter_list_field(areas, source_text)
        for item, violations in dropped:
            trace.append(AgentTraceStep.warn(
                f"Impact area dropped — referenced ungrounded entity/number {violations}"
            ))
        if narrative and ungrounded_entities(narrative, source_text):
            trace.append(AgentTraceStep.warn("Impact narrative dropped — referenced ungrounded entity/number"))
            narrative = None

        # PLACEHOLDER-LEAK GUARD: drop any area/narrative that echoes this
        # prompt's own instructional text instead of real, case-specific
        # content (observed in production: the model literally output "A
        # specific assessment pathway grounded in this finding's own affected
        # items/records." as an impact area).
        areas = [a for a in areas if not is_placeholder_leak(a)]
        if narrative and is_placeholder_leak(narrative):
            trace.append(AgentTraceStep.warn("Impact narrative dropped — echoed prompt instruction text"))
            narrative = None

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

        impact = ImpactAssessment(
            status=status,
            areas=areas,
            narrative=narrative,
            affected_object=affected_object,
            affected_people=affected_people,
            affected_period=affected_period,
            process_at_risk=process_at_risk,
            relevant_change=relevant_change,
            potential_effect=potential_effect,
            evidence_needed=impact_evidence_needed,
        )
        trace.append(AgentTraceStep.ok(f"Impact assessed: {impact.status.value}"))

    except (LLMError, ValueError, KeyError) as exc:
        logger.warning("Impact assessment failed: %s", exc)
        trace.append(AgentTraceStep.warn(f"Impact assessment failed — defaulting to REQUIRES_ASSESSMENT"))
        errors.append(f"Impact error: {exc}")

    return {**state, "impact_assessment": impact, "trace": trace, "errors": errors}
