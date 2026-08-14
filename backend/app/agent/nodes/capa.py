"""Node 7: capa_analysis

Determines CAPA status based on root cause and evidence.
Enforces the strict status rules in code — LLM cannot override them.
"""

from __future__ import annotations

import json
import logging

from app.agent.state import AgentState
from app.config import get_settings
from app.models.agent import AgentTraceStep, CapaAnalysis, CapaStatus, EvidenceStatus
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)


def _determine_capa_status_from_evidence(root_cause_status: str, evidence_ledger: list) -> CapaStatus:
    """Programmatic CAPA status enforcement — overrides LLM if needed."""
    has_verified = any(e.status == EvidenceStatus.VERIFIED for e in evidence_ledger)

    if root_cause_status == "ESTABLISHED" and has_verified:
        return CapaStatus.CAPA_RECOMMENDED
    if root_cause_status == "STATED_UNVERIFIED":
        return CapaStatus.CAPA_DRAFT_POSSIBLE
    return CapaStatus.INVESTIGATION_REQUIRED


async def capa_analysis_node(state: AgentState) -> AgentState:
    """Determine CAPA status and potential corrective action areas."""
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    root_cause = state.get("root_cause")
    evidence_ledger = state.get("evidence_ledger", [])
    settings = get_settings()
    client = get_llm_client()

    system_prompt = (settings.agent_prompts_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (settings.agent_prompts_dir / "capa.txt").read_text(encoding="utf-8")

    rc_status = root_cause.status if root_cause else "NOT_ESTABLISHED"
    rc_category = root_cause.category if root_cause else None
    rc_narrative = root_cause.narrative if root_cause else "Not established."

    prompt = template.format(
        root_cause_status=rc_status,
        root_cause_category=rc_category or "Not determined",
        root_cause_narrative=rc_narrative,
        evidence_ledger_json=json.dumps(
            [e.model_dump() for e in evidence_ledger], default=str
        ),
    )

    # Safe default
    capa = CapaAnalysis(
        status=CapaStatus.INVESTIGATION_REQUIRED,
        potential_areas=[],
        recommended_investigation=["Full manual investigation required — LLM error during CAPA analysis."],
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
        raw_capa = parsed.get("capa", {})

        llm_status_str = str(raw_capa.get("status", "INVESTIGATION_REQUIRED")).upper()
        potential_areas = [str(x) for x in raw_capa.get("potential_areas", [])]
        recommended_investigation = [str(x) for x in raw_capa.get("recommended_investigation", [])]

        # Programmatic enforcement: LLM cannot override the rules
        enforced_status = _determine_capa_status_from_evidence(rc_status, evidence_ledger)

        # Allow LLM to be more conservative (lower status) but never more liberal
        status_order = [
            CapaStatus.CAPA_RECOMMENDED,
            CapaStatus.CAPA_DRAFT_POSSIBLE,
            CapaStatus.INVESTIGATION_REQUIRED,
            CapaStatus.INSUFFICIENT_EVIDENCE,
            CapaStatus.NO_CAPA_RECOMMENDATION_YET,
        ]

        try:
            llm_status = CapaStatus(llm_status_str)
        except ValueError:
            llm_status = CapaStatus.INVESTIGATION_REQUIRED

        # Take the more conservative of LLM and programmatic
        llm_idx = status_order.index(llm_status) if llm_status in status_order else 2
        enforced_idx = status_order.index(enforced_status)
        final_status = status_order[max(llm_idx, enforced_idx)]

        if final_status != llm_status:
            trace.append(AgentTraceStep.warn(
                f"CAPA status enforced: {llm_status.value} → {final_status.value} (rule: root cause not ESTABLISHED)"
            ))

        capa = CapaAnalysis(
            status=final_status,
            potential_areas=potential_areas,
            recommended_investigation=recommended_investigation,
        )
        trace.append(AgentTraceStep.ok(f"CAPA analyzed: {capa.status.value}"))

    except (LLMError, ValueError, KeyError) as exc:
        logger.warning("CAPA analysis failed: %s", exc)
        trace.append(AgentTraceStep.warn("CAPA analysis failed — defaulting to INVESTIGATION_REQUIRED"))
        errors.append(f"CAPA error: {exc}")

    return {**state, "capa_analysis": capa, "trace": trace, "errors": errors}
