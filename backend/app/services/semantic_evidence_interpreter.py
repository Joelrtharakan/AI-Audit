"""LLM semantic-understanding stage for financial evidence interpretation.

This is the ONLY place raw evidence text is handed to an LLM for financial
meaning-extraction. Its output (`SemanticFindingInterpretation`) is never
trusted as a numeric authority -- see `app/financial/relationship_
validator.py` and `app/financial/semantic_engine.py`, which independently
validate and calculate everything downstream.

Never raises. Returns an honest, provider-independent status alongside the
interpretation:

    ("OK", interpretation)      -- a structured interpretation was produced
    ("LLM_UNAVAILABLE", None)   -- no provider / network error / timeout
    ("LLM_INVALID", None)       -- unparseable JSON or schema violation
    ("NO_EVIDENCE", None)       -- there is no evidence ledger to interpret

The caller (`app.financial.semantic_engine`) turns every non-OK status into
an explicit, number-free `FinancialAnalysisResult` -- it never silently
substitutes a keyword/regex interpretation. Only "NO_EVIDENCE" lets the
caller defer to the deterministic text engine, because in that case there
is nothing for the LLM to read.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from app.config import get_settings
from app.financial.provider_normalization import normalize_to_canonical
from app.financial.semantic_models import FinancialSemanticStatus, SemanticFindingInterpretation
from app.models.agent import EvidenceItem
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)

SemanticInterpretationResult = tuple[FinancialSemanticStatus, SemanticFindingInterpretation | None]

_SCHEMA_HINT = (
    '{"finding": {"deviation": str|null, "affected_object": str|null, "process": str|null, '
    '"requirement": str|null, "affected_period": str|null, '
    '"interpretation_confidence": "HIGH"|"MEDIUM"|"LOW"}, '
    '"claims": [{"claim_id": str, "source_evidence_ids": [str], '
    '"fact_type": "QUANTITY"|"RATE"|"AMOUNT"|"RECOVERY"|"REMEDIATION_COST"|"PREVENTION_COST"|'
    '"OBSERVATION_PERIOD"|"PERCENTAGE"|"OTHER", "value": number|null, "unit": str|null, '
    '"currency": str|null, '
    '"population": "CURRENT_FINDING"|"HISTORICAL"|"RECOVERY"|"REMEDIATION"|"PREVENTION"|"OTHER", '
    '"temporal_scope": str|null, "evidence_status": "VERIFIED"|"REPORTED"|"UNVERIFIED"|"CONTRADICTED", '
    '"explicit": bool}], '
    '"relationships": [{"relationship_id": str, '
    '"relationship_type": str (free descriptive label, e.g. "per-unit rate", '
    '"each-event amount", "recovery against gross", "competing estimate of same loss"), '
    '"relationship_description": str, "semantic_basis": str, '
    '"is_conflict": bool (true ONLY when the two claims are competing/incompatible values '
    'for what should be one fact -- NOT for a gross amount and a smaller repayment against it), '
    '"source_claim": str, "target_claim": str, '
    '"confidence": "HIGH"|"MEDIUM"|"LOW", "evidence_basis": [str]}], '
    '"calculation_proposals": [{"calculation_id": str, "operation": "MULTIPLY"|"SUBTRACT"|"DIVIDE"|'
    '"ANNUALIZE"|"SUM", "inputs": [str], "relationship_ids": [str], '
    '"proposed_result_value": number|null, "proposed_result_currency": str|null, "reason": str}], '
    '"cost_factor": {"selected_factor": "DIRECT_LOSS"|"OVERPAYMENT"|"DUPLICATE_PAYMENT"|"REWORK_COST"|'
    '"SCRAP_COST"|"DOWNTIME_COST"|"CUSTOMER_COMPENSATION"|"PENALTY"|"REMEDIATION_COST"|"PREVENTION_COST"|'
    '"REVENUE_IMPACT"|"COST_AVOIDANCE"|"OTHER"|"NOT_ESTABLISHED", "supporting_claim_ids": [str], '
    '"confidence": "HIGH"|"MEDIUM"|"LOW", "rationale": str}, '
    '"financial_relevance": "NONE"|"POTENTIAL"|"MATERIAL"|"CONFIRMED" '
    '(is there a financial mechanism at all -- independent of whether an amount can be calculated; '
    '"18 additional rework hours" with no rate is still MATERIAL), '
    '"quantification": {"status": "QUANTIFIED"|"PARTIALLY_QUANTIFIED"|"UNQUANTIFIED"|"NOT_ASSESSABLE", '
    '"blocker": str, "missing_inputs": [str]}}'
)


def _build_messages(finding_text: str, evidence_ledger: list[EvidenceItem]) -> list[dict[str, str]]:
    settings = get_settings()
    system_template = (settings.prompts_dir / "financial_semantic_interpretation_system_prompt.txt").read_text(
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


async def interpret_evidence_semantically(
    finding_text: str,
    evidence_ledger: list[EvidenceItem],
    client=None,
    timeout_seconds: float | None = None,
) -> SemanticInterpretationResult:
    """Ask the LLM to structurally interpret the finding + evidence.

    Never raises. Returns `(status, interpretation)` -- see module docstring.
    A non-"OK" status is an honest failure signal, NOT permission to
    substitute a regex/keyword interpretation.
    """
    if not evidence_ledger:
        return "NO_EVIDENCE", None

    settings = get_settings()
    # Same operation-specific-timeout pattern as every other LLM call site
    # in the pipeline (extraction/core_synthesis/critic each have their
    # own settings.ollama_*_timeout_seconds) -- financial interpretation's
    # schema/prompt is comparable in size to core synthesis, so it gets
    # its own budget rather than the much tighter generic 8s default that
    # was silently causing every real-provider call to time out.
    effective_timeout = timeout_seconds if timeout_seconds is not None else settings.financial_semantic_reasoning_timeout_seconds

    try:
        llm_client = client or get_llm_client(timeout_seconds=effective_timeout)
    except Exception as exc:  # no provider configured, misconfiguration, etc.
        logger.info("Semantic financial interpretation: no LLM client available (%s).", exc)
        return "LLM_UNAVAILABLE", None

    try:
        messages = _build_messages(finding_text, evidence_ledger)
    except Exception as exc:
        logger.warning("Semantic financial interpretation: prompt build failed (%s).", exc)
        return "LLM_UNAVAILABLE", None

    try:
        raw = await llm_client.chat_completion(
            messages,
            temperature=0.0,
            response_format_json=True,
            max_tokens=settings.ollama_financial_semantic_max_tokens,
            num_ctx=settings.ollama_financial_semantic_num_ctx,
            node="financial_semantic_interpretation",
            timeout_seconds=effective_timeout,
        )
    except (LLMError, Exception) as exc:  # noqa: BLE001 - fail-closed by design
        logger.info("Semantic financial interpretation unavailable (%s).", exc)
        return "LLM_UNAVAILABLE", None

    try:
        parsed = parse_llm_json(raw)
    except Exception as exc:
        logger.warning("Semantic financial interpretation returned unparseable JSON (%s).", exc)
        return "LLM_INVALID", None

    # Provider-neutral schema normalization -- the ONLY place surface-form
    # differences between providers are reconciled. Everything after this
    # line is provider-independent.
    parsed = normalize_to_canonical(parsed)

    try:
        interpretation = SemanticFindingInterpretation.model_validate(parsed)
    except ValidationError as exc:
        logger.warning("Semantic financial interpretation failed schema validation (%s).", exc)
        return "LLM_INVALID", None

    return "OK", interpretation
