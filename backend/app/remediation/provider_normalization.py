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
        "derived_from", "quantity_basis", "unit_cost_basis", "amount_type", "value_kind",
        "recurrence", "frequency", "operation", "produces", "overall_status", "estimability",
        "not_assessable_reason", "interpretation_confidence", "horizon_basis",
    }
)

_NUMERIC_KEYS = frozenset(
    {"quantity", "unit_cost", "unit_cost_low", "unit_cost_high", "proposed_result_value", "horizon"}
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
        _alias(a, "disposition", "role", "activity_role", "action_type", "correction_type")
        _alias(a, "depends_on_root_cause", "depends_on_cause", "cause_dependent", "root_cause_dependent")
        _alias(a, "source_reference_ids", "source_refs", "references", "evidence_ids", "refs", "source_reference_id")
        if "source_reference_ids" in a:
            a["source_reference_ids"] = _as_string_list(a["source_reference_ids"])
        if "disposition" in a:
            _d = str(a["disposition"] or "").strip().upper().replace(" ", "_").replace("-", "_")
            a["disposition"] = _d if _d in (
                "IMMEDIATE_CORRECTION", "CONTAINMENT", "CORRECTIVE_ACTION",
                "CONDITIONAL_SYSTEMIC", "EFFECTIVENESS_CHECK", "INVESTIGATION",
            ) else ""


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
        _alias(c, "value_kind", "monetary_kind", "amount_kind", "money_kind", "kind")
        _alias(c, "activity_ids", "activities", "activity_id", "linked_activities")
        _alias(c, "quantity_basis", "qty_basis", "quantity_source")
        _alias(c, "unit_cost", "rate", "price", "cost_per_unit", "unit_price")
        _alias(c, "unit_cost_basis", "cost_basis", "price_basis", "unit_cost_source", "basis")
        _alias(c, "amount_type", "value_type", "cost_structure", "component_type", "cost_relationship", "role")
        _alias(c, "alternative_group", "alt_group", "option_group", "scenario_group")
        _alias(c, "is_primary_option", "primary_option", "is_recommended", "recommended")
        _alias(c, "recurring_period", "period", "recurrence_period", "frequency_period")
        _alias(c, "recurrence", "frequency", "cost_frequency", "cadence")
        _alias(c, "source_reference_ids", "source_refs", "references", "evidence_ids", "refs", "source_reference_id")
        _alias(c, "assumptions", "assumption")
        # A period-named recurrence ("MONTHLY", "PER_MONTH", "ANNUAL") is
        # RECURRING; carry the period into recurring_period. Enum spelling only.
        _rc = c.get("recurrence")
        if isinstance(_rc, str):
            _rcu = _rc.strip().upper().replace(" ", "_").replace("-", "_")
            if _rcu in ("ONE_TIME", "RECURRING", ""):
                c["recurrence"] = _rcu or "ONE_TIME"
            elif _rcu in ("ONETIME", "ONE_OFF", "SINGLE", "UPFRONT"):
                c["recurrence"] = "ONE_TIME"
            else:
                _p = _rcu.removeprefix("PER_").removesuffix("LY").lower()
                _res = next((v for k, v in {
                    "month": "month", "week": "week", "quarter": "quarter",
                    "year": "year", "annual": "year", "day": "day",
                }.items() if _p.startswith(k)), None)
                if _res:
                    c["recurrence"] = "RECURRING"
                    c.setdefault("recurring_period", _res)
                else:
                    c["recurrence"] = "ONE_TIME"
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
        # value_kind: keep a recognised value, else drop the key so the model
        # default (REMEDIATION_COST) applies -- an unrecognised label must not
        # discard the whole component via strict validation. Spelling only.
        if isinstance(c.get("value_kind"), str):
            _vk = c["value_kind"].strip().upper().replace(" ", "_").replace("-", "_")
            _VK = {"REMEDIATION_COST", "UNIT_RATE", "QUOTED_PRICE", "BUDGET", "ESTIMATE",
                   "OBSERVED_FINANCIAL_LOSS", "HISTORICAL_EXPENDITURE", "OTHER"}
            _VK_SYN = {"RATE": "UNIT_RATE", "UNIT_PRICE": "UNIT_RATE", "QUOTE": "QUOTED_PRICE",
                       "QUOTATION": "QUOTED_PRICE", "LOSS": "OBSERVED_FINANCIAL_LOSS",
                       "INCURRED_LOSS": "OBSERVED_FINANCIAL_LOSS",
                       "HISTORICAL_COST": "HISTORICAL_EXPENDITURE",
                       "PAST_EXPENDITURE": "HISTORICAL_EXPENDITURE"}
            _vk = _VK_SYN.get(_vk, _vk)
            if _vk in _VK:
                c["value_kind"] = _vk
            else:
                c.pop("value_kind", None)

        # amount_type: map an unrecognised label to the nearest valid enum
        # rather than let strict validation discard the whole component (spec
        # Pass 52: the model produced `PER_AREA` for a fixed complete-area
        # inspection price and the component -- Rs 6,000 -- was silently
        # dropped). Enum spelling only; the RELATIONSHIP the model expressed is
        # preserved (a fixed whole-scope price is a flat COMPONENT; an
        # unrecognised "per X" is a per-unit rate).
        if isinstance(c.get("amount_type"), str):
            _at = c["amount_type"].strip().upper().replace(" ", "_").replace("-", "_")
            _AT = {"PER_QUANTITY", "PER_HOUR", "PER_UNIT", "PER_EVENT",
                   "PER_IMPLEMENTATION", "TOTAL", "SUBTOTAL", "COMPONENT", "ALTERNATIVE"}
            _AT_SYN = {
                "PER_HR": "PER_HOUR", "HOURLY": "PER_HOUR",
                "PER_ITEM": "PER_UNIT", "PER_PIECE": "PER_UNIT", "PER_ASSET": "PER_UNIT",
                "PER_MACHINE": "PER_UNIT", "PER_DEVICE": "PER_UNIT", "PER_PANEL": "PER_UNIT",
                "PER_SENSOR": "PER_UNIT", "PER_LICENSE": "PER_UNIT", "PER_SEAT": "PER_UNIT",
                "PER_VISIT": "PER_EVENT", "PER_TRIP": "PER_EVENT", "PER_INSPECTION": "PER_EVENT",
                "PER_SERVICE": "PER_EVENT", "PER_JOB": "PER_EVENT", "PER_RUN": "PER_EVENT",
                "PER_PERSON": "PER_EVENT", "PER_HEAD": "PER_EVENT",
                # a fixed price for a whole area / site / scope / batch is one
                # flat additive line item, quantity implicitly 1
                "PER_AREA": "COMPONENT", "PER_SITE": "COMPONENT", "PER_LOCATION": "COMPONENT",
                "PER_SCOPE": "COMPONENT", "PER_PROJECT": "COMPONENT", "PER_BATCH": "COMPONENT",
                "PER_LOT": "COMPONENT", "PER_FACILITY": "COMPONENT", "PER_BUILDING": "COMPONENT",
                "FIXED": "COMPONENT", "FLAT": "COMPONENT", "FIXED_FEE": "COMPONENT",
                "FLAT_FEE": "COMPONENT", "LUMP_SUM": "COMPONENT", "LUMPSUM": "COMPONENT",
                "FIXED_PRICE": "COMPONENT", "ONE_OFF": "COMPONENT", "LINE_ITEM": "COMPONENT",
                "ITEM": "COMPONENT", "EQUIPMENT": "COMPONENT", "MATERIAL": "COMPONENT",
                "MATERIALS": "COMPONENT", "GRAND_TOTAL": "TOTAL", "SUB_TOTAL": "SUBTOTAL",
                "OPTION": "ALTERNATIVE", "SCENARIO": "ALTERNATIVE",
            }
            _at = _AT_SYN.get(_at, _at)
            if _at not in _AT:
                _at = "PER_QUANTITY" if _at.startswith("PER_") else "COMPONENT"
            c["amount_type"] = _at


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
        # NOTE: "operands" is a structured field of its own (Pass 32) -- only
        # fold it into component_ids when it is a bare id list, never when it
        # carries operand objects.
        if isinstance(calc.get("operands"), list) and not any(
            isinstance(x, dict) for x in calc["operands"]
        ):
            _alias(calc, "component_ids", "components", "inputs", "operands", "input", "component_id")
        else:
            _alias(calc, "component_ids", "components", "inputs", "input", "component_id")
        if "component_ids" in calc:
            calc["component_ids"] = _as_string_list(calc["component_ids"])
        _alias(calc, "result_represents", "result_type", "represents", "result_meaning")
        _alias(calc, "frequency", "recurrence")
        _alias(calc, "recurring_period", "period", "recurrence_period", "period_basis")
        _alias(calc, "horizon", "time_horizon", "horizon_periods", "periods")
        _alias(calc, "horizon_unit", "horizon_period", "horizon_units")
        _alias(calc, "horizon_basis", "horizon_source")
        if "horizon" in calc:
            calc["horizon"] = _coerce_number(calc["horizon"])
        for _k in ("frequency", "horizon_basis"):
            if isinstance(calc.get(_k), str):
                calc[_k] = calc[_k].strip().upper().replace(" ", "_").replace("-", "_")
        # A period-named frequency ("MONTHLY"/"PER_MONTH"/"ANNUAL"/...) IS
        # recurring -- fold the period out into recurring_period, keep the enum
        # to ONE_TIME|RECURRING. Spelling only; the LLM decided it recurs.
        _fq = calc.get("frequency")
        if isinstance(_fq, str) and _fq not in ("ONE_TIME", "RECURRING", ""):
            _p = _fq.removeprefix("PER_").removesuffix("LY").lower()
            _period_map = {"month": "month", "week": "week", "quarter": "quarter",
                           "year": "year", "annual": "year", "day": "day"}
            _resolved = next((v for k, v in _period_map.items() if _p.startswith(k)), None)
            if _resolved:
                calc["frequency"] = "RECURRING"
                calc.setdefault("recurring_period", _resolved)
            elif _fq in ("ONETIME", "ONE_OFF", "SINGLE"):
                calc["frequency"] = "ONE_TIME"
        _ops = calc.get("operands")
        if isinstance(_ops, list):
            for _op in _ops:
                if isinstance(_op, dict):
                    _alias(_op, "value", "amount", "operand_value", "number")
                    _alias(_op, "label", "name", "meaning", "description")
                    _alias(_op, "source_component_id", "component_id", "component")
                    _alias(_op, "evidence_refs", "source_reference_ids", "references", "refs", "evidence_ids")
                    if "evidence_refs" in _op:
                        _op["evidence_refs"] = _as_string_list(_op["evidence_refs"])
                    if "value" in _op:
                        _op["value"] = _coerce_number(_op["value"])
        # `produces` is a small closed enum (LOW / MOST_LIKELY / HIGH /
        # COMPONENT_AMOUNT). Weak models emit near-synonyms ("TOTAL", "SUM",
        # "GRAND_TOTAL", "SUBTOTAL") for the aggregate figure -- normalize the
        # malformed enum spelling so compositional salvage does not drop an
        # otherwise-valid SUM proposal. Pure spelling normalization, no
        # semantic decision (the calculator still executes the proposal).
        if "produces" in calc and isinstance(calc["produces"], str):
            _p = calc["produces"].strip().upper().replace(" ", "_").replace("-", "_")
            _PRODUCES_SYNONYM = {
                "TOTAL": "MOST_LIKELY", "SUM": "MOST_LIKELY", "GRAND_TOTAL": "MOST_LIKELY",
                "SUBTOTAL": "MOST_LIKELY", "TOTAL_COST": "MOST_LIKELY",
                "MOST_LIKELY_ESTIMATE": "MOST_LIKELY", "POINT": "MOST_LIKELY",
                "COMPONENT": "COMPONENT_AMOUNT", "AMOUNT": "COMPONENT_AMOUNT",
                "LINE_ITEM": "COMPONENT_AMOUNT", "LOW_ESTIMATE": "LOW", "HIGH_ESTIMATE": "HIGH",
            }
            calc["produces"] = _PRODUCES_SYNONYM.get(_p, _p if _p in (
                "LOW", "MOST_LIKELY", "HIGH", "COMPONENT_AMOUNT") else "MOST_LIKELY")
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
    _alias(normalized, "auditor_inputs_required", "auditor_inputs", "inputs_required",
           "auditor_evidence_required", "missing_pricing_inputs")
    for k in ("range_assumptions", "uncertainty_reasons", "evidence_improves_estimate"):
        if k in normalized:
            normalized[k] = _as_string_list(normalized[k])
    _air = normalized.get("auditor_inputs_required")
    if isinstance(_air, dict):
        _air = [_air]
    if isinstance(_air, list):
        _kept = []
        for it in _air:
            if isinstance(it, str):
                it = {"remediation_activity": "", "missing_input": it}
            if isinstance(it, dict):
                d = _walk(it)
                _alias(d, "remediation_activity", "activity", "activity_description")
                _alias(d, "missing_input", "missing", "required_input", "input")
                _alias(d, "current_pricing_evidence", "available", "current_evidence",
                       "available_evidence")
                _alias(d, "why_required", "reason", "rationale")
                _alias(d, "acceptable_evidence", "acceptable", "evidence_to_provide",
                       "acceptable_document")
                _alias(d, "enables_estimate_type", "enables", "estimate_type", "would_enable")
                _kept.append(d)
        normalized["auditor_inputs_required"] = _kept
    elif "auditor_inputs_required" in normalized:
        normalized["auditor_inputs_required"] = []
    for k in ("overall_status", "estimability", "not_assessable_reason"):
        if isinstance(normalized.get(k), str):
            normalized[k] = normalized[k].strip().upper().replace(" ", "_").replace("-", "_")
    _normalize_strategy(normalized)
    _normalize_activities(normalized)
    _normalize_components(normalized)
    _normalize_calculations(normalized)
    _drop_malformed(normalized)
    return normalized
