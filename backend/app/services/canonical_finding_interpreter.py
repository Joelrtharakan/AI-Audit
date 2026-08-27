"""LLM canonical-semantic-interpretation stage.

This is the ONE place a finding + evidence ledger is handed to an LLM to
build the shared understanding every downstream module (financial engine,
investigation planner, Five-Why, risk/impact) should reason from.

Fails closed on every possible error -- returns None rather than raising,
so callers must treat a None result as "no canonical context available",
which existing deterministic modules (resolve_deviation, the regex
financial engine, build_deterministic_investigation_plan,
build_deterministic_five_why) already handle unmodified as their normal
operating mode. This function's result is NEVER required for the pipeline
to produce a valid report.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from app.config import get_settings
from app.models.agent import EvidenceItem
from app.services.canonical_semantic_models import CanonicalFindingContext
from app.services.llm_client import get_llm_client
from app.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)

_SCHEMA_HINT = (
    '{"primary_deviation": str|null, "primary_deviation_claim_id": str|null, '
    '"primary_deviation_confidence": "HIGH"|"MEDIUM"|"LOW"|"NOT_ESTABLISHED", '
    '"entities": [{"entity_id": str, "name": str, '
    '"kind": "ENTITY"|"STATE"|"EVENT"|"CONSEQUENCE"|"FINANCIAL_METRIC"|"HISTORICAL_CONTEXT"|'
    '"REMEDIATION"|"RECOVERY"|"CAUSE"|"HYPOTHESIS", "state": str|null, "source_evidence_ids": [str]}], '
    '"causal_claims": [{"claim_id": str, "statement": str, "is_causal": bool, '
    '"cause_ref": str|null, "effect_ref": str|null, "source_evidence_ids": [str], '
    '"evidence_status": "VERIFIED"|"REPORTED"|"UNVERIFIED"|"CONTRADICTED"}], '
    '"explicit_previous_capa_reference": bool, "previous_capa_evidence_ids": [str], '
    '"evidence_boundaries": [{"description": str, "related_claim_ids": [str]}], '
    '"unresolved_ambiguities": [str], '
    '"financial": {"finding": {...}, "claims": [...], "relationships": [...], '
    '"calculation_proposals": [...]} '
    '(financial sub-object schema: same as the financial-only semantic interpreter -- '
    'claims have claim_id/source_evidence_ids/fact_type/value/unit/currency/population/'
    'temporal_scope/evidence_status/explicit; relationships have relationship_id/type/'
    'source_claim/target_claim/confidence/evidence_basis; calculation_proposals have '
    'calculation_id/operation/inputs/relationship_ids/proposed_result_value/'
    'proposed_result_currency/reason)}'
)


def _build_messages(finding_text: str, evidence_ledger: list[EvidenceItem]) -> list[dict[str, str]]:
    settings = get_settings()
    system_template = (settings.prompts_dir / "canonical_finding_interpretation_system_prompt.txt").read_text(
        encoding="utf-8"
    )
    system_prompt = system_template.format(schema=_SCHEMA_HINT)

    evidence_lines = []
    for idx, item in enumerate(evidence_ledger):
        eid = f"E{idx}"
        status = item.status.value if getattr(item, "status", None) is not None else "UNVERIFIED"
        evidence_lines.append(f"{eid} [{status}]: {item.claim}")

    user_prompt = (
        f"FINDING:\n{finding_text}\n\n"
        f"EVIDENCE:\n" + ("\n".join(evidence_lines) if evidence_lines else "(none)")
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def interpret_finding_canonically(
    finding_text: str,
    evidence_ledger: list[EvidenceItem],
    client=None,
    timeout_seconds: float | None = 8.0,
) -> CanonicalFindingContext | None:
    """Ask the LLM to build the canonical semantic finding context.

    Returns None (never raises) on any failure: no provider configured,
    network/timeout error, invalid JSON, or schema violation.
    """
    if not evidence_ledger:
        return None

    try:
        llm_client = client or get_llm_client(timeout_seconds=timeout_seconds)
    except Exception as exc:
        logger.info("Canonical semantic interpretation skipped: no LLM client available (%s).", exc)
        return None

    try:
        messages = _build_messages(finding_text, evidence_ledger)
    except Exception as exc:
        logger.warning("Canonical semantic interpretation skipped: prompt build failed (%s).", exc)
        return None

    try:
        raw = await llm_client.chat_completion(messages, temperature=0.0, response_format_json=True)
    except Exception as exc:  # noqa: BLE001 - fail-closed by design
        logger.info("Canonical semantic interpretation unavailable (%s).", exc)
        return None

    try:
        parsed = parse_llm_json(raw)
    except Exception as exc:
        logger.warning("Canonical semantic interpretation returned unparseable JSON (%s).", exc)
        return None

    try:
        context = CanonicalFindingContext.model_validate(parsed)
    except ValidationError as exc:
        logger.warning("Canonical semantic interpretation failed schema validation (%s).", exc)
        return None

    return context
