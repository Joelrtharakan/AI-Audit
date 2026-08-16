"""LEGACY NODE — NOT PART OF LIVE GRAPH.
This module exists only for isolated unit-test guard checks.
core_synthesis_node is the SOLE authoritative CA draft path.

Node 10: generate_ca_draft (Legacy)
"""

from __future__ import annotations

import json
import logging
import re

from app.agent.grounding_guard import (
    build_source_text,
    claims_unsupported_effectiveness,
    is_placeholder_leak,
    mentions_unsupported_domain,
    ungrounded_entities,
)
from app.agent.permissions import build_ca_draft
from app.agent.state import AgentState
from app.config import get_settings
from app.models.agent import AgentTraceStep
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json

_UNSUPPORTED_IMPACT_SEVERITY_RE = re.compile(
    r"\b(recall|quarantine|notify\s+\w*\s*customer|inform\s+\w*\s*client|product safety|"
    r"patient safety|regulatory requirement|non-compliance|noncompliance)\b",
    re.IGNORECASE,
)

_SAFE_FALLBACK_BY_FIELD = {
    "immediate_action": (
        "Review the specific records/items/activities described in this finding and "
        "determine whether containment or retrospective verification is required."
    ),
    "root_cause": "Root cause not yet established for this finding — further investigation is required.",
    "root_cause_category": "TO_BE_CONFIRMED",
    "preventive_action": (
        "Preventive action cannot be finalized until investigation establishes which "
        "control actually failed for this finding."
    ),
    "impact_analysis": "Impact requires assessment — auditor to determine scope for this finding.",
}

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

        # Fallbacks below are deliberately generic/entity-free — they only fire when
        # the model returned an empty or placeholder value, and must never inject a
        # fixed scenario's entities into a different finding's CA draft.
        if not parsed_dict["immediate_action"].strip() or "pending investigation" in parsed_dict["immediate_action"].lower():
            parsed_dict["immediate_action"] = (
                "Review the specific records/items/activities described in this finding and determine "
                "whether containment or retrospective verification is required before further "
                "auditor investigation is complete."
            )
        if not parsed_dict["root_cause"].strip():
            rc_narrative = root_cause.narrative if root_cause else "Root cause not yet established."
            parsed_dict["root_cause"] = f"{root_cause.status if root_cause else 'NOT_ESTABLISHED'} — {rc_narrative}"
        if not parsed_dict["root_cause_category"].strip() or parsed_dict["root_cause_category"] == "PENDING_INVESTIGATION":
            parsed_dict["root_cause_category"] = root_cause.category if (root_cause and root_cause.category) else "TO_BE_CONFIRMED"
        # CATEGORY DISCIPLINE: this is a SEPARATE LLM call from rca.py, so it can
        # (and in production testing, did) independently guess a definitive
        # category like "MEASUREMENT" even when the RCA step itself already
        # determined the cause isn't established and set TO_BE_CONFIRMED. The
        # CA draft must never be more confident than the root cause it's based on.
        if root_cause and root_cause.category == "TO_BE_CONFIRMED" and parsed_dict["root_cause_category"] != "TO_BE_CONFIRMED":
            trace.append(AgentTraceStep.warn(
                f"CA draft category '{parsed_dict['root_cause_category']}' overridden to TO_BE_CONFIRMED "
                "— root cause is not yet established"
            ))
            parsed_dict["root_cause_category"] = "TO_BE_CONFIRMED"
        if not parsed_dict["preventive_action"].strip() or "pending verification" in parsed_dict["preventive_action"].lower():
            parsed_dict["preventive_action"] = (
                "Once a root cause is confirmed for this finding, strengthen the specific control "
                "process implicated by that cause. Preventive action cannot be finalized until "
                "investigation establishes which control actually failed."
            )
        # Sanitize all 5 fields into clean human-readable strings free of python reprs
        def _clean_str(val: str) -> str:
            if not val:
                return ""
            s = str(val).strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                s = re.sub(r"'(?:status|id|name)':\s*'[^']*',?", "", s)
                s = re.sub(r"'(?:description|statement|text)':\s*", "", s)
                s = s.replace("{", "").replace("}", "").replace("[", "").replace("]", "").strip()
            return s

        for key in ("immediate_action", "root_cause", "root_cause_category", "preventive_action", "impact_analysis"):
            parsed_dict[key] = _clean_str(parsed_dict[key])

        # GROUNDING GUARD: the CA draft is what actually gets written into the
        # form, so this is the last chance to catch an entity/number that
        # doesn't trace back to THIS finding before it reaches the auditor.
        extra_trusted = []
        if root_cause and root_cause.narrative:
            extra_trusted.append(root_cause.narrative)
        if impact and impact.narrative:
            extra_trusted.append(impact.narrative)
        source_text = build_source_text(state["request"].finding_text, evidence_ledger, extra_trusted)
        finding_text = state["request"].finding_text
        extraction = state.get("extraction")
        external_impact_stated = bool(extraction and extraction.external_impact_stated)

        for key in ("immediate_action", "root_cause", "preventive_action", "impact_analysis"):
            violations = ungrounded_entities(parsed_dict[key], source_text)
            if violations:
                trace.append(AgentTraceStep.warn(
                    f"CA draft field '{key}' replaced — referenced ungrounded entity/number {violations}"
                ))
                logger.warning("CA draft field %s grounding violation: %s", key, violations)
                parsed_dict[key] = _SAFE_FALLBACK_BY_FIELD[key]
            elif claims_unsupported_effectiveness(parsed_dict[key], finding_text):
                trace.append(AgentTraceStep.warn(
                    f"CA draft field '{key}' replaced — claimed previous-CAPA effectiveness not "
                    "supported by this finding"
                ))
                parsed_dict[key] = _SAFE_FALLBACK_BY_FIELD[key]
            elif mentions_unsupported_domain(parsed_dict[key], source_text):
                trace.append(AgentTraceStep.warn(
                    f"CA draft field '{key}' replaced — invoked training/authorization domain not "
                    "supported by this finding"
                ))
                parsed_dict[key] = _SAFE_FALLBACK_BY_FIELD[key]
            elif is_placeholder_leak(parsed_dict[key]):
                trace.append(AgentTraceStep.warn(
                    f"CA draft field '{key}' replaced — echoed prompt instruction text instead of "
                    "real analysis"
                ))
                parsed_dict[key] = _SAFE_FALLBACK_BY_FIELD[key]
            elif key == "impact_analysis" and not external_impact_stated and _UNSUPPORTED_IMPACT_SEVERITY_RE.search(parsed_dict[key]):
                # Same firewall impact.py already applies to its own output —
                # the CA draft's impact_analysis is a SEPARATE LLM call and can
                # independently invent product-safety/recall/regulatory
                # language with no basis (observed in production testing).
                trace.append(AgentTraceStep.warn(
                    "CA draft field 'impact_analysis' replaced — invented unsupported severity "
                    "(recall/regulatory/product-safety) language with no external impact stated"
                ))
                parsed_dict[key] = _SAFE_FALLBACK_BY_FIELD[key]

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
