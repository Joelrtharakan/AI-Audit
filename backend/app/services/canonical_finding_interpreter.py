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
import time
from typing import Literal

from pydantic import ValidationError

from app.config import get_settings
from app.models.agent import EvidenceItem
from app.services.canonical_semantic_models import CanonicalFindingContext
from app.services.llm_client import get_llm_client
from app.services.llm_json import parse_llm_json

logger = logging.getLogger(__name__)

# Structured outcome of the canonical interpretation call (spec §3/§4/§18).
# A caller can tell WHY `context` is None -- a provider failure / fallback
# event is not the same as a valid interpretation with empty optional fields.
CanonicalInterpretStatus = Literal[
    "SUCCESS",                    # valid, no salvage needed
    "SALVAGED",                   # valid after dropping malformed optional pieces
    "PROVIDER_UNAVAILABLE",       # no LLM client configured
    "PROVIDER_ERROR",             # network / provider raised (incl. timeout)
    "EMPTY_RESPONSE",             # provider returned nothing usable
    "INVALID_JSON",               # response was not parseable JSON
    "SCHEMA_INVALID",             # parsed JSON could not be validated / salvaged
    "PROMPT_BUILD_FAILED",        # internal prompt assembly error
    "UNEXPECTED_ERROR",
]

_SCHEMA_HINT = (
    '{"primary_deviation": str|null, "primary_deviation_claim_id": str|null, '
    '"primary_deviation_confidence": "HIGH"|"MEDIUM"|"LOW"|"NOT_ESTABLISHED", '
    '"finding_subject": str|null (the substantive affected object/process/control/'
    'activity -- NEVER a causal mechanism, hypothesis, evidence-source noun, '
    'requirement text, or reported belief; null if none exists), '
    '"subject_kind": "ENTITY"|"STATE"|"EVENT"|null, '
    '"evidence_source": str|null (e.g. "maintenance records"), '
    '"reported_observation": str|null (an action/state attributed to a source), '
    '"observed_condition": str|null (what was observed about the subject), '
    '"epistemic_status": "VERIFIED"|"REPORTED"|"BELIEF"|"INFERRED"|"UNKNOWN"|null, '
    '"comparison": {"left": str|null, "right": str|null, "reference": str|null, '
    '"direction": "ABOVE"|"BELOW"|"MISMATCH"|"UNKNOWN", "magnitude": number|null, '
    '"unit": str|null}|null, '
    '"recurrence": {"count": int|null, "event": str|null, "period": str|null}|null, '
    '"stated_causal_alternatives": [str] (mechanisms the FINDING TEXT explicitly '
    'enumerates -- preserve every one, never rank, never erase), '
    '"causal_alternatives_unresolved": bool, '
    '"missing_record_status": "RECORD_EXISTS"|"RECORD_INCOMPLETE"|"RECORD_MISSING"|'
    '"RECORD_UNAVAILABLE"|"ACTIVITY_NOT_RECORDED"|"ACTIVITY_NOT_PERFORMED"|"UNKNOWN"|null, '
    '"activity_performance_ambiguity": bool (true when it is open whether the '
    'underlying activity occurred), '
    '"affected_period": str|null, "scope": str|null, '
    '"root_cause_status": "ESTABLISHED"|"NOT_ESTABLISHED"|"STATED_UNVERIFIED"|"CONTRADICTED" '
    '(ESTABLISHED only when a VERIFIED causal claim fixes ONE mechanism), '
    '"leading_hypothesis_id": str|null (null unless root_cause_status=ESTABLISHED), '
    '"candidate_hypotheses": [{"hypothesis_id": str, "statement": str, '
    '"epistemic": "POSSIBLE"|"SUPPORTED"|"REFUTED"|"UNKNOWN", "from_finding_text": bool, '
    '"rationale": str|null, "discriminating_evidence": str|null, "source_evidence_ids": [str]}], '
    '"information_gaps": [str], '
    '"investigation_plan": [{"unknown": str, "why_it_matters": str|null, '
    '"evidence_that_would_resolve": str|null, "decision_enabled": str|null, '
    '"related_hypothesis_ids": [str], "priority": "HIGH"|"MEDIUM"|"LOW"}], '
    '"remediation_obligation": "ESTABLISHED_CORRECTIVE_OBLIGATION"|'
    '"RECONCILIATION_REQUIRED"|"INVESTIGATION_REQUIRED"|"IMMEDIATE_CORRECTION_ONLY"|'
    '"NO_SYSTEMIC_REMEDIATION_JUSTIFIED"|"NOT_DETERMINED" '
    '(has the EVIDENCE established an obligation to remediate? do NOT assume '
    '"a finding exists therefore remediation exists"), '
    '"remediation_obligation_rationale": str|null, '
    '"investigation_activities": [{"action_id": str, "activity": str, '
    '"disposition": "INVESTIGATION", "addresses_condition": str|null, '
    '"justification": str|null}] '
    '(reconcile/compare/verify/determine/establish work whose PURPOSE is to find '
    'out what happened -- NOT remediation), '
    '"remediation_activities": [{"action_id": str, "activity": str, '
    '"disposition": "IMMEDIATE_CORRECTION"|"CONTAINMENT"|"CORRECTIVE_ACTION"|'
    '"CONDITIONAL_SYSTEMIC"|"EFFECTIVENESS_CHECK" '
    '(CORRECTIVE_ACTION only if root_cause_status=ESTABLISHED), '
    '"addresses_condition": str|null (the established condition / confirmed cause '
    'this corrects), "justification": str|null, "depends_on_root_cause": bool, '
    '"pricing_evidence_needed": str|null, "scope_evidence_needed": str|null}] '
    '(ONLY genuine correction -- MAY be [] when the task is still investigation), '
    '"immediate_actions": [str], "conditional_actions": [str], '
    '"pricing_information": [{"action_id": str|null, "pricing_basis": str|null, '
    '"rationale": str|null, "evidence_available": bool, '
    '"observed_value_in_finding": str|null, "observed_value_is_remediation_cost": bool}], '
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
    # Plain substitution (not str.format) -- the prompt body and the schema
    # hint both contain literal { } from JSON examples.
    system_prompt = system_template.replace("{schema}", _SCHEMA_HINT)

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
    timeout_seconds: float | None = None,
) -> CanonicalFindingContext | None:
    """Back-compat wrapper: returns the context (or None). Callers that need
    the failure reason should use `interpret_finding_canonically_with_status`."""
    _status, ctx = await interpret_finding_canonically_with_status(
        finding_text, evidence_ledger, client=client, timeout_seconds=timeout_seconds
    )
    return ctx


async def interpret_finding_canonically_with_status(
    finding_text: str,
    evidence_ledger: list[EvidenceItem],
    client=None,
    timeout_seconds: float | None = None,
) -> tuple[CanonicalInterpretStatus, CanonicalFindingContext | None]:
    """Ask the LLM to build the canonical semantic finding context.

    Never raises. Returns `(status, context)` -- `context` is None on every
    failure, and `status` records WHY (spec §3/§4): a provider failure /
    fallback event is distinguishable from a valid but sparse interpretation.

    Runs on the finding TEXT alone -- an evidence ledger is optional: the
    structured semantic fields come from the finding text; evidence-grounded
    fields (entities, causal_claims) are stripped by the validator when the
    ledger is empty.
    """
    settings = get_settings()
    # PRIMARY semantic authority: production timeout + token/context budget
    # (the same operation-specific pattern as the financial / remediation
    # interpreters). A shadow-mode 8s timeout or a truncated response was
    # silently failing this call for exactly the complex findings that matter,
    # dropping the whole pipeline back to the deterministic floor.
    effective_timeout = (
        timeout_seconds if timeout_seconds is not None
        else settings.canonical_semantic_primary_timeout_seconds
    )
    _t0 = time.monotonic()

    def _log(status: CanonicalInterpretStatus, *, resp_len: int = 0, detail: str = "") -> None:
        logger.info(
            "CANONICAL SEMANTIC INTERPRETATION status=%s latency_ms=%d timeout_s=%s "
            "max_tokens=%s num_ctx=%s response_chars=%d%s",
            status, int((time.monotonic() - _t0) * 1000), effective_timeout,
            settings.canonical_semantic_max_tokens, settings.canonical_semantic_num_ctx,
            resp_len, (f" detail={detail}" if detail else ""),
        )

    try:
        llm_client = client or get_llm_client(timeout_seconds=effective_timeout)
    except Exception as exc:
        _log("PROVIDER_UNAVAILABLE", detail=repr(exc))
        return "PROVIDER_UNAVAILABLE", None

    try:
        messages = _build_messages(finding_text, evidence_ledger)
    except Exception as exc:
        _log("PROMPT_BUILD_FAILED", detail=repr(exc))
        return "PROMPT_BUILD_FAILED", None

    try:
        raw = await llm_client.chat_completion(
            messages,
            temperature=0.0,
            response_format_json=True,
            max_tokens=settings.canonical_semantic_max_tokens,
            num_ctx=settings.canonical_semantic_num_ctx,
            node="canonical_semantic_interpretation",
            timeout_seconds=effective_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed by design
        _log("PROVIDER_ERROR", detail=type(exc).__name__)
        return "PROVIDER_ERROR", None

    if not raw or not str(raw).strip():
        _log("EMPTY_RESPONSE")
        return "EMPTY_RESPONSE", None

    try:
        parsed = parse_llm_json(raw)
    except Exception as exc:
        _log("INVALID_JSON", resp_len=len(str(raw)), detail=type(exc).__name__)
        return "INVALID_JSON", None

    context, salvaged = _validate_or_salvage(parsed)
    if context is None:
        _log("SCHEMA_INVALID", resp_len=len(str(raw)))
        return "SCHEMA_INVALID", None

    status: CanonicalInterpretStatus = "SALVAGED" if salvaged else "SUCCESS"
    _log(status, resp_len=len(str(raw)))
    return status, context


# Core semantic fields (spec §6): a malformed CORE field is NOT silently
# dropped-to-default -- that would let an unsafe interpretation through. If a
# core field cannot validate, the whole interpretation is rejected (fail
# closed). Optional reasoning collections are salvaged element-by-element.
_CORE_FIELDS = frozenset({
    "finding_subject", "observed_condition", "comparison", "epistemic_status",
    "root_cause_status", "remediation_obligation", "primary_deviation_confidence",
})


def _validate_or_salvage(parsed: object) -> tuple[CanonicalFindingContext | None, bool]:
    """Strict validation first; on failure, drop only malformed OPTIONAL
    elements / fields and re-validate (compositional salvage). One bad
    hypothesis or pricing item must never discard the whole interpretation.
    A malformed CORE semantic field fails closed instead of being defaulted.
    Returns (context|None, salvaged?)."""
    if not isinstance(parsed, dict):
        return None, False
    work = dict(parsed)
    salvaged = False
    for _ in range(24):
        try:
            return CanonicalFindingContext.model_validate(work), salvaged
        except ValidationError as exc:
            if any((e.get("loc") or (None,))[0] in _CORE_FIELDS for e in exc.errors()):
                return None, False
            changed = False
            _drop_idx: dict[str, set] = {}
            _drop_top: set = set()
            for err in exc.errors():
                loc = err.get("loc") or ()
                if not loc:
                    continue
                top = loc[0]
                if not isinstance(top, str) or top not in work:
                    continue
                # A bad element of a list field -> drop just that element.
                if len(loc) >= 2 and isinstance(loc[1], int) and isinstance(work.get(top), list):
                    _drop_idx.setdefault(top, set()).add(loc[1])
                else:
                    _drop_top.add(top)
            for top, idxs in _drop_idx.items():
                if top not in _drop_top:
                    work[top] = [v for i, v in enumerate(work[top]) if i not in idxs]
                    changed = True
            for top in _drop_top:
                work.pop(top, None)
                changed = True
            if not changed:
                return None, False
            salvaged = True
    return None, False
