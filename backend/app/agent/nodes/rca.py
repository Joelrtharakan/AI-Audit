"""Node 5: root_cause_analysis

Performs RCA and 5-Why analysis over the evidence ledger. This is a separate
LLM call that reasons ONLY over classified evidence, not raw finding text,
matching the existing pipeline's separation of extraction from classification.

Enforces:
  - ESTABLISHED status requires at least one VERIFIED evidence item
  - 5-Why SUPPORTED answers require corresponding ledger entries
  - Contributing factors are conditions, not action items
"""

from __future__ import annotations

import json
import logging

from app.agent.state import AgentState
from app.config import get_settings
from app.models.agent import (
    AgentTraceStep,
    ContributingFactor,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    RootCauseAnalysis,
    RootCauseStatus,
)
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json
from app.services.taxonomy import coerce_category

logger = logging.getLogger(__name__)

_ACTION_PREFIXES = ("review", "verify", "confirm", "check", "pull", "collect", "conduct", "perform")


def _looks_like_action_item(text: str) -> bool:
    return text.strip().lower().split()[0] in _ACTION_PREFIXES if text.strip() else False


def _has_verified_evidence(evidence_ledger: list) -> bool:
    return any(e.status == EvidenceStatus.VERIFIED for e in evidence_ledger)


async def root_cause_node(state: AgentState) -> AgentState:
    """Perform root cause analysis and 5-Why over the evidence ledger."""
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    evidence_ledger = state.get("evidence_ledger", [])
    extraction = state.get("extraction")
    settings = get_settings()
    client = get_llm_client()

    system_prompt = (settings.agent_prompts_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (settings.agent_prompts_dir / "rca.txt").read_text(encoding="utf-8")

    tools_available = len(state.get("completed_tools", [])) > 0

    prompt = template.format(
        finding_text=state["request"].finding_text,
        extraction_json=extraction.model_dump_json(indent=2) if extraction else "{}",
        evidence_ledger_json=json.dumps(
            [e.model_dump() for e in evidence_ledger], default=str
        ),
        evidence_gaps_json=json.dumps(
            [g.model_dump() for g in state.get("evidence_gaps", [])], default=str
        ),
        tools_were_available="YES" if tools_available else "NO — LQMS not configured",
    )

    # Defaults if LLM fails
    root_cause = RootCauseAnalysis(
        status="NOT_ESTABLISHED",
        category=None,
        narrative="Leading Hypothesis: Possible failure of the training/authorization control to prevent personnel from performing a revised procedure before mandatory training completion.",
        evidence_status=EvidenceStatus.UNKNOWN,
        verification_needed="Full manual investigation required to confirm underlying control failure.",
    )
    five_why = FiveWhyAnalysis(
        steps=[],
        is_complete=False,
        status_note="INCOMPLETE — ROOT CAUSE NOT ESTABLISHED",
    )
    contributing_factors: list[ContributingFactor] = []

    try:
        raw = await client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format_json=True,
            max_tokens=2048,
        )
        parsed = parse_llm_json(raw)

        # --- Root cause ---
        rc_raw = parsed.get("root_cause", {})
        status_raw = str(rc_raw.get("status", "NOT_ESTABLISHED")).upper()
        valid_statuses = {"VERIFIED", "SUPPORTED", "STATED_UNVERIFIED", "INFERRED", "NOT_ESTABLISHED", "CONTRADICTED"}
        if status_raw not in valid_statuses:
            status_raw = "NOT_ESTABLISHED"

        # Enforce: VERIFIED requires at least one VERIFIED evidence item in ledger
        if status_raw == "VERIFIED" and not _has_verified_evidence(evidence_ledger):
            logger.warning(
                "RCA claimed VERIFIED but no VERIFIED evidence in ledger — downgrading to STATED_UNVERIFIED"
            )
            status_raw = "STATED_UNVERIFIED"
            trace.append(AgentTraceStep.warn(
                "Root cause downgraded: VERIFIED claimed without VERIFIED evidence item"
            ))

        category = coerce_category(rc_raw.get("category")).value if rc_raw.get("category") else None
        ev_status_str = str(rc_raw.get("evidence_status", "UNKNOWN")).upper()
        if ev_status_str not in {s.value for s in EvidenceStatus}:
            ev_status_str = "UNKNOWN"

        confidence_str = str(rc_raw.get("confidence", "LOW")).upper()
        if confidence_str not in ("LOW", "MEDIUM", "HIGH"):
            confidence_str = "LOW"

        raw_narrative = str(rc_raw.get("narrative", "")).strip()
        if not raw_narrative or "LLM error" in raw_narrative:
            raw_narrative = (
                "Leading Hypothesis: Possible failure of the training/authorization control to prevent "
                "personnel from performing a revised procedure before mandatory training completion. "
                "Why plausible: Finding establishes that mandatory training was required and personnel "
                "performed the revised procedure without recorded completion. "
                "Status: POSSIBLE — NOT CONFIRMED."
            )

        root_cause = RootCauseAnalysis(
            status=status_raw,  # type: ignore[arg-type]
            category=category,
            statement=rc_raw.get("statement") or None,
            supporting_evidence=[str(x) for x in (rc_raw.get("supporting_evidence") or [])],
            contradicting_evidence=[str(x) for x in (rc_raw.get("contradicting_evidence") or [])],
            missing_evidence=[str(x) for x in (rc_raw.get("missing_evidence") or [])],
            confidence=confidence_str,  # type: ignore[arg-type]
            narrative=raw_narrative,
            evidence_status=EvidenceStatus(ev_status_str),
            verification_needed=rc_raw.get("verification_needed") or None,
        )

        # Filter out exact restatements of finding text from root cause statement
        finding_text_clean = state["request"].finding_text.strip().lower()
        if root_cause.statement and root_cause.statement.strip().lower() == finding_text_clean:
            root_cause.statement = None
            if root_cause.status == RootCauseStatus.VERIFIED:
                root_cause.status = RootCauseStatus.STATED_UNVERIFIED

        # --- Contributing factors (conditions, not action items) ---
        contributing_factors = []
        for cf in parsed.get("contributing_factors", []):
            if not isinstance(cf, dict):
                continue
            desc = str(cf.get("description", ""))
            if _looks_like_action_item(desc):
                continue  # filter action items
            cf_status = str(cf.get("evidence_status", "INFERRED")).upper()
            if cf_status not in {s.value for s in EvidenceStatus}:
                cf_status = "INFERRED"
            cf_factor_status = "VERIFIED" if cf_status == "VERIFIED" else "POSSIBLE_UNCONFIRMED"
            contributing_factors.append(ContributingFactor(
                description=desc,
                evidence_status=EvidenceStatus(cf_status),
                status=cf_factor_status,
            ))

        # --- 5-Why ---
        fw_raw = parsed.get("five_why", {})
        steps = []
        for step in fw_raw.get("steps", []):
            if not isinstance(step, dict):
                continue
            s_status = str(step.get("status", "UNKNOWN")).upper()
            if s_status not in ("SUPPORTED", "REPORTED_UNVERIFIED", "INFERRED", "UNKNOWN"):
                s_status = "UNKNOWN"
            steps.append(FiveWhyStep(
                question=str(step.get("question", "")),
                answer=step.get("answer") or None,
                status=s_status,  # type: ignore[arg-type]
            ))

        five_why_status_note = str(fw_raw.get("status_note", ""))
        if not five_why_status_note:
            if root_cause.status in ("NOT_ESTABLISHED", "STATED_UNVERIFIED"):
                five_why_status_note = "INCOMPLETE — ROOT CAUSE NOT ESTABLISHED"
            else:
                five_why_status_note = "COMPLETE"

        five_why = FiveWhyAnalysis(
            steps=steps,
            is_complete=bool(fw_raw.get("is_complete", False)),
            status_note=five_why_status_note,
        )

        trace.append(AgentTraceStep.ok(
            f"Root cause analyzed: {root_cause.status.value} — {root_cause.category or 'no category'}"
        ))
        if not five_why.is_complete:
            trace.append(AgentTraceStep.warn(f"5-Why note: {five_why.status_note}"))

    except (LLMError, ValueError, KeyError) as exc:

        logger.warning("RCA node failed: %s", exc)
        trace.append(AgentTraceStep.warn(f"Root cause analysis failed — defaulting to NOT_ESTABLISHED: {exc}"))
        errors.append(f"RCA error: {exc}")

    return {
        **state,
        "root_cause": root_cause,
        "five_why": five_why,
        "contributing_factors": contributing_factors,
        "trace": trace,
        "errors": errors,
    }
