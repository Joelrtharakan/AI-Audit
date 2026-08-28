"""LLM semantic-understanding stage for Remediation Cost Estimation.

The ONLY place raw finding/evidence/CAPA context is handed to an LLM to reason
about what remediation a finding implies and what it would cost to implement.
Its output (`RemediationInterpretation`) is never trusted as a numeric authority
-- `app.remediation.validator` and `app.remediation.calculator` independently
validate structure and perform every calculation.

Never raises. Returns an honest, provider-independent status:

    ("OK", interpretation)         -- a structured interpretation was produced
    ("LLM_INCOMPLETE", partial)    -- only some pieces validated (salvaged)
    ("LLM_UNAVAILABLE", None)      -- no provider / network error / timeout
    ("LLM_INVALID", None)          -- unparseable JSON / nothing salvageable
    ("NO_EVIDENCE", None)          -- no finding text and no evidence at all
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from app.config import get_settings
from app.remediation.provider_normalization import normalize_to_canonical
from app.remediation.semantic_models import (
    ImplementationActivity,
    RemediationCalculationProposal,
    RemediationCostComponent,
    RemediationInterpretation,
    RemediationSemanticStatus,
    RemediationStrategy,
)
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)

RemediationInterpretationResult = tuple[RemediationSemanticStatus, RemediationInterpretation | None]

# Compact field reference. Every field is OPTIONAL unless noted -- the prompt
# instructs the model to omit any field it has no value for (this is what keeps
# the response small). `cost_category` and `remediation_type` are free text.
_SCHEMA_HINT = (
    '{'
    '"strategy":{"remediation_summary":str,"remediation_type":str,"established_basis":str,'
    '"hypothetical_basis":str,"alternative_strategies":[str],"interpretation_confidence":"HIGH|MEDIUM|LOW"},'
    '"activities":[{"activity_id":str(required),"description":str(required),'
    '"derived_from":"FINDING|EVIDENCE|ROOT_CAUSE_HYPOTHESIS|RECOMMENDED_CAPA|IMPACT|CONTEXT",'
    '"source_reference_ids":[str],"is_hypothetical":bool}],'
    '"cost_components":[{"component_id":str(required),"description":str(required),"activity_ids":[str],'
    '"cost_category":str,"quantity":num,"quantity_unit":str,'
    '"quantity_basis":"EVIDENCED|ASSUMED|NOT_ESTABLISHED",'
    '"unit_cost":num,"unit_cost_low":num,"unit_cost_high":num,'
    '"unit_cost_basis":"VERIFIED|REPORTED|ESTIMATED|ASSUMED|NOT_ESTABLISHED","currency":str,'
    '"amount_type":"PER_QUANTITY|PER_HOUR|PER_UNIT|PER_EVENT|PER_IMPLEMENTATION|COMPONENT|SUBTOTAL|TOTAL|ALTERNATIVE",'
    '"alternative_group":str(only for ALTERNATIVE),"is_primary_option":bool(only for ALTERNATIVE),'
    '"recurrence":"ONE_TIME|RECURRING","recurring_period":str,'
    '"source_reference_ids":[str],"assumptions":[str],"rationale":str}],'
    '"calculation_proposals":[{"calculation_id":str(required),"operation":"MULTIPLY|SUM|SUBTRACT",'
    '"component_ids":[str],"produces":"LOW|MOST_LIKELY|HIGH|COMPONENT_AMOUNT","reason":str}],'
    '"overall_status":"EVIDENCE_BACKED|ASSUMPTION_BASED|NOT_ASSESSABLE",'
    '"estimability":"ESTIMABLE|BOUNDED_ONLY|SINGLE_VERIFIED_COST|NOT_ASSESSABLE",'
    '"not_assessable_reason":"IMPLEMENTATION_SCOPE_UNKNOWN|QUANTITY_UNKNOWN|PRICING_BASIS_UNAVAILABLE|'
    'REMEDIATION_NOT_DEFINED|CONFLICTING_EVIDENCE|INSUFFICIENT_EVIDENCE",'
    '"range_assumptions":[str],"uncertainty_reasons":[str],"evidence_improves_estimate":[str]}'
)


_MAX_HYPOTHESES = 3
_MAX_CAPA_LINES = 6
_MAX_STMT_CHARS = 400


def _clip(s: Any, n: int = _MAX_STMT_CHARS) -> str:
    s = str(s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _context_block(
    finding_text: str,
    evidence_ledger: list[Any],
    root_cause: Any,
    capa: Any,
    impact: Any,
) -> str:
    """Compact, structured context -- only what remediation interpretation
    actually needs. No generated report prose, no financial LLM result, no
    duplicated narrative (spec: prompt context compression)."""
    lines: list[str] = [f"FINDING: {finding_text or '(none)'}", ""]

    ev_lines = []
    for idx, item in enumerate(evidence_ledger or []):
        status = getattr(getattr(item, "status", None), "value", None) or "UNVERIFIED"
        claim = getattr(item, "claim", None) or getattr(item, "text", "") or str(item)
        ev_lines.append(f"E{idx} [{status}]: {_clip(claim)}")
    lines.append("EVIDENCE:\n" + ("\n".join(ev_lines) if ev_lines else "(none)"))
    lines.append("")

    if root_cause is not None:
        status = getattr(getattr(root_cause, "status", None), "value", getattr(root_cause, "status", None))
        rc = [f"ROOT CAUSE (status={status}):"]
        stmt = getattr(root_cause, "statement", None) or getattr(root_cause, "narrative", None)
        if stmt:
            rc.append(f"  {_clip(stmt)}")
        for h in (getattr(root_cause, "candidate_hypotheses", []) or [])[:_MAX_HYPOTHESES]:
            hid = getattr(h, "hypothesis_id", None) or getattr(h, "id", None) or "H?"
            rc.append(f"  {hid}: {_clip(getattr(h, 'statement', ''))}")
        lines.append("\n".join(rc))
        lines.append("")

    if capa is not None:
        cp: list[str] = []
        for idx, ca in enumerate(getattr(capa, "conditional_actions", []) or []):
            cp.append(
                f"CAPA{idx} (if {_clip(getattr(ca, 'if_cause_confirmed', '?'), 160)}): "
                f"{_clip(getattr(ca, 'recommended_action', ''))}"
            )
        for a in (getattr(capa, "potential_areas", []) or []):
            cp.append(f"area: {_clip(a, 160)}")
        if cp:
            lines.append("RECOMMENDED CAPA:\n" + "\n".join(cp[:_MAX_CAPA_LINES]))
            lines.append("")

    if impact is not None:
        parts = [
            f"{f}={_clip(v, 160)}"
            for f in ("affected_object", "process_at_risk", "control_at_risk", "potential_effect")
            for v in (getattr(impact, f, None),)
            if v
        ]
        if parts:
            lines.append("IMPACT: " + "; ".join(parts))
            lines.append("")

    return "\n".join(lines).strip()


def _build_messages(
    finding_text: str,
    evidence_ledger: list[Any],
    root_cause: Any,
    capa: Any,
    impact: Any,
) -> list[dict[str, str]]:
    settings = get_settings()
    system_template = (
        settings.prompts_dir / "remediation_cost_interpretation_system_prompt.txt"
    ).read_text(encoding="utf-8")
    system_prompt = system_template.format(schema=_SCHEMA_HINT)
    user_prompt = _context_block(finding_text, evidence_ledger, root_cause, capa, impact)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def interpret_remediation(
    finding_text: str,
    evidence_ledger: list[Any] | None = None,
    root_cause: Any = None,
    capa: Any = None,
    impact: Any = None,
    client=None,
    timeout_seconds: float | None = None,
) -> RemediationInterpretationResult:
    evidence_ledger = evidence_ledger or []

    if not (finding_text or "").strip() and not evidence_ledger:
        return "NO_EVIDENCE", None

    settings = get_settings()
    effective_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else settings.remediation_cost_estimation_timeout_seconds
    )

    try:
        llm_client = client or get_llm_client(timeout_seconds=effective_timeout)
    except Exception as exc:  # no provider configured
        logger.info("Remediation cost interpretation: no LLM client available (%s).", exc)
        return "LLM_UNAVAILABLE", None

    try:
        messages = _build_messages(finding_text, evidence_ledger, root_cause, capa, impact)
    except Exception as exc:
        logger.warning("Remediation cost interpretation: prompt build failed (%s).", exc)
        return "LLM_UNAVAILABLE", None

    try:
        raw = await llm_client.chat_completion(
            messages,
            temperature=0.0,
            response_format_json=True,
            max_tokens=settings.remediation_cost_max_tokens,
            num_ctx=settings.remediation_cost_num_ctx,
            node="remediation_cost_interpretation",
            timeout_seconds=effective_timeout,
        )
    except (LLMError, Exception) as exc:  # noqa: BLE001 - fail-closed by design
        logger.info("Remediation cost interpretation unavailable (%s).", exc)
        return "LLM_UNAVAILABLE", None

    try:
        parsed = parse_llm_json(raw)
    except Exception as exc:
        logger.warning("Remediation cost interpretation returned unparseable JSON (%s).", exc)
        return "LLM_INVALID", None

    parsed = normalize_to_canonical(parsed)

    try:
        return "OK", RemediationInterpretation.model_validate(parsed)
    except ValidationError as exc:
        logger.info(
            "Remediation cost interpretation failed strict schema validation (%s); salvaging.", exc
        )
        interp = _salvage(parsed)
        if interp is None:
            return "LLM_INVALID", None
        has_content = bool(interp.cost_components or interp.activities)
        return ("OK" if has_content else "LLM_INCOMPLETE"), interp


def _salvage(parsed: Any) -> RemediationInterpretation | None:
    """Compositional recovery: validate each piece independently, keep what is
    individually well-formed, drop only the malformed pieces. Never fabricates."""
    if not isinstance(parsed, dict):
        return None

    def _one(model, item):
        try:
            return model.model_validate(item)
        except ValidationError:
            return None

    def _lst(v):
        return v if isinstance(v, list) else []

    activities = [a for a in (_one(ImplementationActivity, x) for x in _lst(parsed.get("activities"))) if a]
    components = [c for c in (_one(RemediationCostComponent, x) for x in _lst(parsed.get("cost_components"))) if c]
    kept_comp_ids = {c.component_id for c in components}

    proposals = []
    for x in _lst(parsed.get("calculation_proposals")):
        p = _one(RemediationCalculationProposal, x)
        if p is None:
            continue
        p.component_ids = [cid for cid in p.component_ids if cid in kept_comp_ids]
        proposals.append(p)

    strategy = _one(RemediationStrategy, parsed.get("strategy")) or RemediationStrategy()

    def _str_list(key):
        v = parsed.get(key)
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    overall = parsed.get("overall_status")
    if overall not in ("EVIDENCE_BACKED", "ASSUMPTION_BASED", "NOT_ASSESSABLE"):
        overall = "NOT_ASSESSABLE"
    estimability = parsed.get("estimability")
    if estimability not in ("ESTIMABLE", "BOUNDED_ONLY", "SINGLE_VERIFIED_COST", "NOT_ASSESSABLE"):
        estimability = "NOT_ASSESSABLE"
    reason = parsed.get("not_assessable_reason")
    if reason not in (
        "IMPLEMENTATION_SCOPE_UNKNOWN", "QUANTITY_UNKNOWN", "PRICING_BASIS_UNAVAILABLE",
        "REMEDIATION_NOT_DEFINED", "CONFLICTING_EVIDENCE", "INSUFFICIENT_EVIDENCE", "",
    ):
        reason = ""

    return RemediationInterpretation(
        strategy=strategy,
        activities=activities,
        cost_components=components,
        calculation_proposals=proposals,
        overall_status=overall,
        estimability=estimability,
        not_assessable_reason=reason,
        range_assumptions=_str_list("range_assumptions"),
        uncertainty_reasons=_str_list("uncertainty_reasons"),
        evidence_improves_estimate=_str_list("evidence_improves_estimate"),
    )
