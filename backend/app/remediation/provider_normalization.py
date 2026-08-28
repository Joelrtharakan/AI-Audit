"""Provider-neutral schema normalization for the LLM remediation-cost
interpretation.

The single named boundary the architecture requires:

    raw provider response (dict)
        |  normalize_to_canonical()
        v
    canonical dict  ->  RemediationInterpretation.model_validate(...)
        |
        v
    everything downstream is provider-independent

SCHEMA normalization ONLY -- enum spelling / case, key aliases, null handling,
numeric-string / currency-symbol coercion, bare-string -> list, and flattening
of equivalent nested shapes. It performs NO cost reasoning: it never decides a
cost category, an operation, a basis, which numbers combine, or invents a
component. Every transformation is a lossless restatement of what the provider
already said, differing only in surface form (spec section 13).
"""

from __future__ import annotations

from typing import Any

_ENUM_KEYS_UPPER = frozenset(
    {
        "derived_from", "quantity_basis", "unit_cost_basis", "amount_type",
        "recurrence", "operation", "produces", "overall_status", "estimability",
        "not_assessable_reason", "interpretation_confidence",
    }
)

_NUMERIC_KEYS = frozenset(
    {"quantity", "unit_cost", "unit_cost_low", "unit_cost_high", "proposed_result_value"}
)

_CURRENCY_SYMBOLS = ("₹", "$", "€", "£", "¥")


def _coerce_number(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        for sym in _CURRENCY_SYMBOLS:
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
    """Move an alias key's value under `canonical` when `canonical` is
    absent/empty. A populated canonical key always wins."""
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


def _flatten_nested_amount(d: dict, value_key: str, currency_key: str, nested_key: str) -> None:
    nested = d.get(nested_key)
    if isinstance(nested, dict):
        if d.get(value_key) is None and "value" in nested:
            d[value_key] = _coerce_number(nested.get("value"))
        if d.get(currency_key) is None and "currency" in nested:
            d[currency_key] = nested.get("currency")
        d.pop(nested_key, None)


def _normalize_strategy(parsed: dict) -> None:
    s = parsed.get("strategy")
    if not isinstance(s, dict):
        return
    _alias(s, "remediation_type", "type", "strategy_type", "approach_type")
    _alias(s, "remediation_summary", "summary", "approach", "strategy")
    _alias(s, "established_basis", "established", "established_vs_hypothetical")
    _alias(s, "hypothetical_basis", "hypothetical", "contingent")
    _alias(s, "deficient_requirement", "requirement", "deficient_control")
    _alias(s, "condition_identified", "condition", "deviation")
    _alias(s, "alternative_strategies", "alternatives", "alternative_approaches")
    if "alternative_strategies" in s:
        s["alternative_strategies"] = _as_string_list(s["alternative_strategies"])


def _normalize_activities(parsed: dict) -> None:
    acts = parsed.get("activities")
    if not isinstance(acts, list):
        return
    for a in acts:
        if not isinstance(a, dict):
            continue
        _alias(a, "activity_id", "id", "activityId")
        _alias(a, "description", "activity", "name", "text")
        _alias(a, "source_reference_ids", "source_refs", "references", "evidence_ids", "refs", "source_reference_id")
        if "source_reference_ids" in a:
            a["source_reference_ids"] = _as_string_list(a["source_reference_ids"])


def _normalize_components(parsed: dict) -> None:
    for list_alias in ("components", "cost_drivers", "drivers"):
        if "cost_components" not in parsed and list_alias in parsed:
            parsed["cost_components"] = parsed.pop(list_alias)
    comps = parsed.get("cost_components")
    if not isinstance(comps, list):
        return
    for c in comps:
        if not isinstance(c, dict):
            continue
        _alias(c, "component_id", "id", "componentId")
        _alias(c, "description", "name", "text", "label")
        _alias(c, "cost_category", "category", "type", "driver_type", "cost_type")
        _alias(c, "activity_ids", "activities", "activity_id", "linked_activities")
        _alias(c, "quantity_basis", "qty_basis", "quantity_source")
        _alias(c, "unit_cost", "rate", "price", "cost_per_unit", "unit_price")
        _alias(c, "unit_cost_basis", "cost_basis", "price_basis", "unit_cost_source", "basis")
        _alias(c, "amount_type", "value_type", "cost_structure", "component_type", "cost_relationship", "role")
        _alias(c, "alternative_group", "alt_group", "option_group", "scenario_group")
        _alias(c, "is_primary_option", "primary_option", "is_recommended", "recommended")
        _alias(c, "recurring_period", "period", "recurrence_period")
        _alias(c, "source_reference_ids", "source_refs", "references", "evidence_ids", "refs", "source_reference_id")
        _alias(c, "assumptions", "assumption")
        _flatten_nested_amount(c, "unit_cost", "currency", "unit_cost_obj")
        nested_range = c.get("unit_cost_range")
        if isinstance(nested_range, dict):
            if c.get("unit_cost_low") is None:
                c["unit_cost_low"] = _coerce_number(nested_range.get("low") or nested_range.get("min"))
            if c.get("unit_cost_high") is None:
                c["unit_cost_high"] = _coerce_number(nested_range.get("high") or nested_range.get("max"))
            c.pop("unit_cost_range", None)
        for k in ("activity_ids", "source_reference_ids", "assumptions"):
            if k in c:
                c[k] = _as_string_list(c[k])
        for k in ("quantity", "unit_cost", "unit_cost_low", "unit_cost_high"):
            if k in c:
                c[k] = _coerce_number(c[k])


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
        _alias(calc, "component_ids", "components", "inputs", "operands", "input", "component_id")
        if "component_ids" in calc:
            calc["component_ids"] = _as_string_list(calc["component_ids"])
        _flatten_nested_amount(calc, "proposed_result_value", "proposed_result_currency", "proposed_result")
        if "proposed_result_value" in calc:
            calc["proposed_result_value"] = _coerce_number(calc["proposed_result_value"])


def _drop_malformed(parsed: dict) -> None:
    """Drop a component/activity/proposal that is individually unusable (no id,
    or not even a dict) rather than letting pydantic reject the ENTIRE
    interpretation over one bad piece (spec section 15/16). Compositional
    salvage downstream handles the rest."""
    for key, id_field in (
        ("activities", "activity_id"),
        ("cost_components", "component_id"),
        ("calculation_proposals", "calculation_id"),
    ):
        items = parsed.get(key)
        if isinstance(items, list):
            parsed[key] = [
                it for it in items
                if isinstance(it, dict) and str(it.get(id_field) or "").strip()
            ]


def normalize_to_canonical(parsed: Any) -> dict:
    if not isinstance(parsed, dict):
        return {"activities": [], "cost_components": [], "calculation_proposals": []}
    normalized = _walk(parsed)
    _alias(normalized, "cost_components", "components", "cost_drivers", "drivers")
    _alias(normalized, "activities", "implementation_activities")
    _alias(normalized, "calculation_proposals", "calculations", "proposed_calculations")
    _alias(normalized, "overall_status", "status")
    _alias(normalized, "not_assessable_reason", "reason", "not_assessable")
    _alias(normalized, "range_assumptions", "range_assumption")
    _alias(normalized, "uncertainty_reasons", "uncertainty", "uncertainties")
    _alias(normalized, "evidence_improves_estimate", "evidence_needed", "improves_estimate")
    for k in ("range_assumptions", "uncertainty_reasons", "evidence_improves_estimate"):
        if k in normalized:
            normalized[k] = _as_string_list(normalized[k])
    for k in ("overall_status", "estimability", "not_assessable_reason"):
        if isinstance(normalized.get(k), str):
            normalized[k] = normalized[k].strip().upper().replace(" ", "_").replace("-", "_")
    _normalize_strategy(normalized)
    _normalize_activities(normalized)
    _normalize_components(normalized)
    _normalize_calculations(normalized)
    _drop_malformed(normalized)
    return normalized
