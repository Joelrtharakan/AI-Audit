"""Provider-neutral schema normalization for the LLM financial semantic
interpretation.

This is the single named boundary the architecture requires:

    raw provider response (dict)
        |
        v   normalize_to_canonical()
        v
    canonical dict  ->  SemanticFindingInterpretation.model_validate(...)
        |
        v
    everything downstream is provider-independent

It performs SCHEMA normalization ONLY -- enum spelling / case, key aliases,
null handling, numeric-string coercion, and flattening of equivalent
nested shapes (e.g. a `proposed_result: {value, currency}` object into the
flat `proposed_result_value` / `proposed_result_currency` fields). It
performs NO financial reasoning: it never decides a cost factor, never
decides an operation, never decides which numbers combine, never invents a
claim, relationship, currency, or evidence status. Every transformation
here is a lossless restatement of what the provider already said,
differing only in surface form.

Different providers (structured-output APIs, tool-call wrappers, raw JSON
models) legitimately differ in casing, in key spelling (`source_claim` vs
`from` vs `source`), in whether an absent field is `null` or omitted, in
whether numbers arrive as JSON numbers or strings, and in whether a
compound value is nested or flattened. Normalizing those here keeps the
validator and calculator from having to know or care which provider
produced the interpretation -- and, critically, stops a harmless
formatting difference from being misread as a semantically invalid
interpretation (architecture spec section 15). Aliases are applied
per-section (claim aliases only to claims, relationship-endpoint aliases
only to relationships, etc.) so a generic key like `type` or `description`
is never rewritten on an object where it means something else.
"""

from __future__ import annotations

from typing import Any

_ENUM_KEYS_UPPER = frozenset(
    {
        "fact_type",
        "population",
        "evidence_status",
        "operation",
        "selected_factor",
        "status",
        "interpretation_confidence",
        "confidence",
        "financial_relevance",
    }
)

_NUMERIC_KEYS = frozenset({"value", "proposed_result_value"})


def _coerce_number(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        for sym in ("₹", "$", "€", "£", "¥"):
            s = s.removeprefix(sym).strip()
        parts = s.split()
        if len(parts) == 2 and parts[1].isalpha():  # trailing code: "250000 INR"
            s = parts[0]
        try:
            return float(s)
        except ValueError:
            return v
    return v


def _norm_scalar(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in _NUMERIC_KEYS:
        return _coerce_number(value)
    if key in _ENUM_KEYS_UPPER and isinstance(value, str):
        return value.strip().upper().replace(" ", "_").replace("-", "_")
    return value


def _walk(obj: Any) -> Any:
    """Depth-first scalar normalization ONLY (enum casing, numeric strings).
    No key renaming here."""
    if isinstance(obj, dict):
        return {k: (_walk(v) if isinstance(v, (dict, list)) else _norm_scalar(k, v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(x) for x in obj]
    return obj


def _as_string_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v if x is not None and str(x).strip()]
    return [str(v)]


def _alias(d: dict, canonical: str, *aliases: str) -> None:
    """If `canonical` is absent/empty but an alias key holds a value, move
    it under `canonical`. A populated canonical key always wins."""
    if d.get(canonical) not in (None, "", [], {}):
        for a in aliases:
            d.pop(a, None)
        return
    for a in aliases:
        if a in d and d.get(a) not in (None, "", [], {}):
            d[canonical] = d.pop(a)
            return
    for a in aliases:
        d.pop(a, None)


def _normalize_claims(parsed: dict) -> None:
    claims = parsed.get("claims")
    if not isinstance(claims, list):
        return
    for c in claims:
        if not isinstance(c, dict):
            continue
        _alias(c, "claim_id", "id", "claimId", "claimid")
        _alias(c, "source_evidence_ids", "evidence_ids", "source_evidence", "evidence", "source_evidence_id")
        if "source_evidence_ids" in c:
            c["source_evidence_ids"] = _as_string_list(c["source_evidence_ids"])


def _normalize_relationships(parsed: dict) -> None:
    rels = parsed.get("relationships")
    if not isinstance(rels, list):
        return
    for rel in rels:
        if not isinstance(rel, dict):
            continue
        _alias(rel, "relationship_type", "type", "kind", "label")
        _alias(rel, "relationship_description", "description")
        _alias(rel, "source_claim", "from", "source", "source_claim_id", "from_claim")
        _alias(rel, "target_claim", "to", "target", "target_claim_id", "to_claim")
        _alias(rel, "evidence_basis", "evidence_ids", "evidence")
        if "evidence_basis" in rel:
            rel["evidence_basis"] = _as_string_list(rel["evidence_basis"])
        raw_type = rel.get("relationship_type")
        if isinstance(raw_type, str) and raw_type.strip():
            lowered = raw_type.strip().lower()
            if "is_conflict" not in rel or rel.get("is_conflict") is None:
                rel["is_conflict"] = any(
                    tok in lowered for tok in ("conflict", "competing", "contradict", "incompatible value")
                )
            rel["relationship_type"] = raw_type.strip()
        else:
            rel["relationship_type"] = "related"


def _normalize_calculations(parsed: dict) -> None:
    for list_alias in ("calculations", "proposed_calculations", "calculation_plan"):
        if "calculation_proposals" not in parsed and list_alias in parsed:
            parsed["calculation_proposals"] = parsed.pop(list_alias)
    calcs = parsed.get("calculation_proposals")
    if not isinstance(calcs, list):
        return
    for calc in calcs:
        if not isinstance(calc, dict):
            continue
        _alias(calc, "calculation_id", "id", "calc_id")
        _alias(calc, "inputs", "input", "input_claims", "input_claim_ids", "claim_ids", "operands")
        _alias(calc, "relationship_ids", "relationship", "relationships_used", "relationship_id", "relationships")
        if "inputs" in calc:
            calc["inputs"] = _as_string_list(calc["inputs"])
        if "relationship_ids" in calc:
            calc["relationship_ids"] = _as_string_list(calc["relationship_ids"])
        # flatten {value, currency} nested result object
        nested = calc.get("proposed_result")
        if isinstance(nested, dict):
            if calc.get("proposed_result_value") is None and "value" in nested:
                calc["proposed_result_value"] = _coerce_number(nested.get("value"))
            if calc.get("proposed_result_currency") is None and "currency" in nested:
                calc["proposed_result_currency"] = nested.get("currency")
            calc.pop("proposed_result", None)
        if "proposed_result_value" in calc:
            calc["proposed_result_value"] = _coerce_number(calc["proposed_result_value"])


def _normalize_cost_factor(parsed: dict) -> None:
    cf = parsed.get("cost_factor")
    if not isinstance(cf, dict):
        return
    _alias(cf, "selected_factor", "factor", "cost_factor_type", "cost_factor", "value")
    _alias(cf, "supporting_claim_ids", "supporting_claims", "claim_ids", "claims")
    if "supporting_claim_ids" in cf:
        cf["supporting_claim_ids"] = _as_string_list(cf["supporting_claim_ids"])
    if isinstance(cf.get("selected_factor"), str):
        cf["selected_factor"] = cf["selected_factor"].strip().upper().replace(" ", "_").replace("-", "_")


def _normalize_quantification(parsed: dict) -> None:
    q = parsed.get("quantification")
    if not isinstance(q, dict):
        return
    _alias(q, "missing_inputs", "missing", "missing_input")
    if "missing_inputs" in q:
        q["missing_inputs"] = _as_string_list(q["missing_inputs"])


def _drop_unresolvable_relationships(parsed: dict) -> None:
    """A relationship missing an endpoint after aliasing carries no
    structural meaning the validator can act on. Dropping it -- rather than
    letting pydantic reject the ENTIRE interpretation over one malformed
    relationship -- preserves every valid claim, relationship, and
    calculation (spec section 16). A calculation that cited only a dropped
    relationship is still handled honestly downstream: rejected with an
    explicit reason, never silently executed."""
    rels = parsed.get("relationships")
    if not isinstance(rels, list):
        return
    parsed["relationships"] = [
        r for r in rels if isinstance(r, dict) and r.get("source_claim") and r.get("target_claim")
    ]


def normalize_to_canonical(parsed: Any) -> dict:
    """Return a canonical-shaped dict ready for
    `SemanticFindingInterpretation.model_validate`. Non-dict input is
    returned wrapped so validation fails cleanly downstream rather than
    here."""
    if not isinstance(parsed, dict):
        return {"claims": [], "relationships": [], "calculation_proposals": []}
    normalized = _walk(parsed)
    _alias(normalized, "financial_relevance", "relevance", "financial_relevance_level")
    if isinstance(normalized.get("financial_relevance"), str):
        normalized["financial_relevance"] = (
            normalized["financial_relevance"].strip().upper().replace(" ", "_").replace("-", "_")
        )
    _normalize_claims(normalized)
    _normalize_relationships(normalized)
    _normalize_calculations(normalized)
    _normalize_cost_factor(normalized)
    _normalize_quantification(normalized)
    _drop_unresolvable_relationships(normalized)
    return normalized
