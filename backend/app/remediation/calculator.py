"""Deterministic arithmetic for Remediation Cost Estimation.

Arithmetic ONLY. No semantics, no keyword rules, no domain logic. Consumes the
structurally-validated components from `app.remediation.validator` and produces
the numeric fields of `RemediationCostResult`.

The one hard rule this module enforces (spec sections 4, 5, 7, 19, 23):

    Multiple monetary values are NOT automatically a low/most-likely/high range.

Before any range is produced, each component's declared semantic role
(`amount_type`) decides how it combines:

    COMPONENT / PER_*   additive line items  -> SUMMED into one total
    SUBTOTAL            a partial roll-up    -> reconciled, never re-added
    TOTAL              a complete total     -> the authoritative figure
    ALTERNATIVE         a mutually-exclusive option -> bracketed as a scenario

A low/most-likely/high spread is emitted ONLY when the evidence genuinely
carries it: ALTERNATIVE options, per-component `unit_cost_low/high`, or a stated
total that conflicts with its itemised parts. Additive components with single
point costs give `low == most_likely == high == their sum`. No invented spread,
no percentages. The LLM's `produces: LOW/HIGH` proposals are audit-only and
never drive these numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.remediation.models import (
    CostBasis,
    RemediationConfidence,
    RemediationCostComponentResult,
)
from app.remediation.semantic_models import (
    RemediationCalculationProposal,
    RemediationCalculationTrace,
    RemediationCostComponent,
)

_PER_X_TYPES = frozenset({"PER_QUANTITY", "PER_HOUR", "PER_UNIT", "PER_EVENT", "PER_IMPLEMENTATION"})
_ADDITIVE_TYPES = _PER_X_TYPES | {"COMPONENT"}
# scope_role values the MODEL declared to mean "this price does not belong to the
# current remediation total". Structural use of the model's own enum: the figure is
# still rendered as a priced cost driver, it is just not added into the total.
_OUT_OF_TOTAL_SCOPE_ROLES = frozenset({"UNRELATED", "OPTIONAL"})

_BASIS_RANK = {
    "VERIFIED": 4, "REPORTED": 3, "ESTIMATED": 2, "ASSUMED": 1, "NOT_ESTABLISHED": 0,
}
_CONF_FROM_BASIS = {
    "VERIFIED": RemediationConfidence.HIGH,
    "REPORTED": RemediationConfidence.MEDIUM,
    "ESTIMATED": RemediationConfidence.MEDIUM,
    "ASSUMED": RemediationConfidence.LOW,
    "NOT_ESTABLISHED": RemediationConfidence.LOW,
}


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 2)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1.0, abs(a) * 1e-6)


@dataclass
class _Row:
    """One priceable component in a single currency: point + range bounds.
    `point` is None for a component the evidence establishes ONLY as a range
    (no single best-estimate stated) -- Phase 9.7 §2: such a component still
    contributes to low/high, but the aggregate most-likely figure must not
    manufacture a midpoint to fill the gap."""
    c: RemediationCostComponent
    point: float | None
    low: float
    high: float


@dataclass
class AssembledEstimate:
    component_results: list[RemediationCostComponentResult] = field(default_factory=list)
    currency: str | None = None
    one_time_cost: float | None = None
    recurring_cost: float | None = None
    recurring_period: str | None = None
    low: float | None = None
    most_likely: float | None = None
    high: float | None = None
    # A recurring cost totalled over an EXPLICIT horizon the LLM supplied in a
    # calculation proposal (never inferred). Null otherwise -- the periodic
    # `recurring_cost` stands alone (spec Pass 33 §3/§7/§24).
    recurring_horizon_total: float | None = None
    recurring_horizon: float | None = None
    recurring_horizon_basis: str = ""
    estimate_classification: CostBasis = CostBasis.NOT_ESTABLISHED
    estimation_method: str = ""
    uncertainty_reasons: list[str] = field(default_factory=list)
    contributing_bases: list[str] = field(default_factory=list)
    unpriced_component_ids: list[str] = field(default_factory=list)
    # True when a stated SUBTOTAL/TOTAL does not reconcile with the itemised
    # components -- the figure is genuinely uncertain, never a clean EXACT
    # estimate (Pass 30 §8/§10). The engine reads this to set PARTIAL/RANGE.
    has_reconciliation_conflict: bool = False


def _multiplies(c: RemediationCostComponent) -> bool:
    """quantity x unit_cost applies ONLY when the LLM tagged the figure as a
    per-item RATE (PER_*). For COMPONENT / TOTAL / SUBTOTAL / ALTERNATIVE the
    `unit_cost` field carries that line's flat amount; any `quantity` is
    descriptive. `amount_type` is the LLM's explicit, deterministic signal --
    the calculator never re-guesses it (spec section 6). A quantity given
    only as a range (no single point value stated) still counts as "a
    quantity is present" -- Phase 9.7 §2."""
    has_qty = c.quantity is not None or (c.quantity_low is not None and c.quantity_high is not None)
    has_rate = c.unit_cost is not None or (c.unit_cost_low is not None and c.unit_cost_high is not None)
    return is_rate_figure(c) and has_qty and has_rate


def is_rate_figure(c: RemediationCostComponent) -> bool:
    """The figure is a per-item RATE: tagged PER_* by `amount_type`, OR classified
    UNIT_RATE by the model's own `value_kind` on a COMPONENT line. A rate the
    model classified as a rate but mis-tagged COMPONENT must still be multiplied
    by its stated quantity -- otherwise "45 hours at 1,600/hour" would surface
    as 1,600. Both fields are the model's enum declarations; no text is read."""
    return c.amount_type in _PER_X_TYPES or (
        c.amount_type == "COMPONENT" and getattr(c, "value_kind", "") == "UNIT_RATE"
    )


def _point_amount(c: RemediationCostComponent) -> float | None:
    if c.amount_type in _PER_X_TYPES:
        if not _multiplies(c):
            return None
        # A most-likely figure requires an actual POINT quantity AND a POINT
        # rate -- Phase 9.7 §2 forbids manufacturing a midpoint when either
        # side was stated only as a range. Range-only components contribute
        # to low/high but not to the point/most-likely figure.
        if c.quantity is not None and c.unit_cost is not None:
            return _round(c.quantity * c.unit_cost)
        return None
    if _multiplies(c) and c.quantity is not None and c.unit_cost is not None:
        return _round(c.quantity * c.unit_cost)  # UNIT_RATE line mis-tagged COMPONENT
    return _round(c.unit_cost) if c.unit_cost is not None else None


def _bound_amount(c: RemediationCostComponent, which: str) -> float | None:
    """Phase 9.7 §2: a multiplying (quantity x rate) component's range may
    come from either side independently -- a ranged QUANTITY at a fixed
    rate ("70-110 hours at Rs 1,900/hour"), a ranged RATE at a fixed
    quantity, or both. Each side's bound defaults to its own point value
    when that side has no stated range, so a range on only one factor is
    never silently collapsed to the point amount on that factor."""
    rate_edge = c.unit_cost_low if which == "low" else c.unit_cost_high
    qty_edge = c.quantity_low if which == "low" else c.quantity_high
    if _multiplies(c):
        if rate_edge is None and qty_edge is None:
            return _point_amount(c)
        rate = rate_edge if rate_edge is not None else c.unit_cost
        qty = qty_edge if qty_edge is not None else c.quantity
        if qty is None or rate is None:
            # e.g. only quantity_high stated (no quantity_low, no point
            # quantity) -- the "low" bound genuinely cannot be computed;
            # never guess rather than silently return a wrong number.
            return None
        return _round(qty * rate)
    if rate_edge is None:
        return _point_amount(c)
    if c.amount_type in _PER_X_TYPES:
        return None
    return _round(rate_edge)


def _pluralize_unit(unit: str, count: float) -> str:
    """General English count-agreement for a quantity unit noun ("hour",
    "machine", "visit") -- pure grammar, not a semantic classifier: it never
    inspects what the unit MEANS, only whether the stated count is 1. The
    unit word itself remains entirely the LLM's own semantic choice."""
    if not unit or abs(count - 1) < 1e-9:
        return unit
    lower = unit.lower()
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return unit + "es"
    if lower.endswith("y") and len(unit) > 1 and unit[-2].lower() not in "aeiou":
        return unit[:-1] + "ies"
    return unit + "s"


def _formula(c: RemediationCostComponent) -> str:
    """Phase 9.7 §2/§3/§4: preserve a range as a range in the displayed
    formula, whether the range lives on the quantity, the rate, or a flat
    (non-multiplying) amount -- never collapse it to a fabricated point,
    and never claim "no calculable amount" merely because no single point
    value was stated when a genuine range was."""
    cur = f"{c.currency} " if c.currency else ""
    if _multiplies(c):
        unit_word = c.quantity_unit or ""
        if c.quantity is not None:
            qty_desc = f"{c.quantity:g}" + (f" {_pluralize_unit(unit_word, c.quantity)}" if unit_word else "")
        elif c.quantity_low is not None and c.quantity_high is not None:
            qty_desc = f"{c.quantity_low:g}-{c.quantity_high:g}" + (
                f" {_pluralize_unit(unit_word, c.quantity_high)}" if unit_word else "")
        else:
            qty_desc = "an unspecified quantity"
        rate_desc = f"{cur}{c.unit_cost:g}" if c.unit_cost is not None else "an unspecified rate"
        low, high = _bound_amount(c, "low"), _bound_amount(c, "high")
        if low is not None and high is not None and not _close(low, high):
            return f"{qty_desc} x {rate_desc} = {cur}{low:g}-{cur}{high:g}"
        point = _point_amount(c)
        if point is not None:
            return f"{qty_desc} x {rate_desc} = {cur}{point:g}"
        return f"{qty_desc} x {rate_desc} (range not fully established)"
    if c.unit_cost is not None:
        return f"{c.unit_cost:g} (stated amount)"
    if c.unit_cost_low is not None and c.unit_cost_high is not None:
        return f"{c.unit_cost_low:g}-{c.unit_cost_high:g} (stated range)"
    return "no calculable amount"


def _basis(s: str) -> CostBasis:
    try:
        return CostBasis(s)
    except ValueError:
        return CostBasis.NOT_ESTABLISHED


def assemble_estimate(
    components: list[RemediationCostComponent],
    accepted_proposals: list[RemediationCalculationProposal],
    traces: list[RemediationCalculationTrace],
    scope_established: bool = True,
) -> AssembledEstimate:
    est = AssembledEstimate()

    currencies = {c.currency for c in components if c.currency}
    # The single working currency, if the evidence establishes exactly one.
    # Currency is NEVER invented or defaulted (spec section 7): a priced
    # component with no currency may only ADOPT this one when it exists;
    # with no currency anywhere its figure cannot be meaningfully expressed
    # and the component is carried as an unpriced cost driver instead.
    working_currency = next(iter(currencies), None) if len(currencies) == 1 else None
    est.currency = working_currency
    if len(currencies) > 1:
        est.uncertainty_reasons.append(
            f"Cost components are stated in multiple currencies ({sorted(currencies)}); they are "
            "reported individually and not combined into one total."
        )

    # --- Per-component render rows + the priceable rows used for roll-up
    #     (single working currency only; mixed-currency components still render).
    results: list[RemediationCostComponentResult] = []
    rows: list[_Row] = []
    _currency_dropped: list[str] = []
    _excluded_from_total: list[str] = []
    for c in components:
        pt = _point_amount(c)
        lo = _bound_amount(c, "low")
        hi = _bound_amount(c, "high")

        # Currency resolution for THIS component -- adopt the single working
        # currency when the component stated none, never invent one. A
        # range-only component (no point) needs a currency just as much as
        # a point one -- Phase 9.7 §2.
        _has_amount = pt is not None or (lo is not None and hi is not None)
        eff_currency = c.currency or (working_currency if _has_amount else None)
        # A figure with no currency and none to adopt is not a usable amount.
        currency_unusable = _has_amount and eff_currency is None
        render_amount = None if currency_unusable else pt

        results.append(RemediationCostComponentResult(
            component_id=c.component_id,
            description=c.description,
            cost_category=c.cost_category,
            quantity=c.quantity,
            quantity_unit=c.quantity_unit,
            quantity_basis=_basis(c.quantity_basis),
            unit_cost=None if currency_unusable else c.unit_cost,
            unit_cost_basis=_basis("NOT_ESTABLISHED" if currency_unusable else c.unit_cost_basis),
            currency=eff_currency,
            calculated_amount=render_amount,
            calculated_amount_low=(
                lo if (not currency_unusable and lo is not None and (render_amount is None or lo != render_amount))
                else None
            ),
            calculated_amount_high=(
                hi if (not currency_unusable and hi is not None and (render_amount is None or hi != render_amount))
                else None
            ),
            calculation_formula="pricing basis stated without a currency" if currency_unusable else _formula(c),
            recurrence=c.recurrence,
            recurring_period=c.recurring_period,
            scope_role=getattr(c, "scope_role", "NOT_ESTABLISHED"),
            confidence=RemediationConfidence.LOW if currency_unusable else _CONF_FROM_BASIS.get(c.unit_cost_basis, RemediationConfidence.LOW),
            source_reference_ids=list(c.source_reference_ids),
            assumptions=list(c.assumptions),
            rationale=c.rationale,
        ))
        if currency_unusable:
            _currency_dropped.append(c.component_id)
            est.unpriced_component_ids.append(c.component_id)
            continue
        if not _has_amount:
            est.unpriced_component_ids.append(c.component_id)
            continue
        if getattr(c, "scope_role", "NOT_ESTABLISHED") in _OUT_OF_TOTAL_SCOPE_ROLES:
            _excluded_from_total.append(c.component_id)  # known price, outside the current scope
            continue
        # A component contributes to the combined roll-up ONLY when the evidence
        # establishes exactly ONE working currency. With components in multiple
        # distinct currencies and no conversion basis, cross-currency addition is
        # a dimensional error -- each figure still renders individually (results
        # above), but no combined headline is produced. Structural: reads only
        # the model-authored `currency` fields, never the finding text.
        if eff_currency and working_currency is not None and eff_currency == working_currency:
            _row_c = c if c.currency else c.model_copy(update={"currency": eff_currency})
            rows.append(_Row(_row_c, pt, lo if lo is not None else pt, hi if hi is not None else pt))
        elif eff_currency and working_currency is None and len(currencies) > 1:
            est.unpriced_component_ids.append(c.component_id)

    if _excluded_from_total:
        est.uncertainty_reasons.append(
            f"{len(_excluded_from_total)} priced component(s) were classified as outside the current "
            "remediation scope; their prices are shown but are not included in the totals."
        )
    if _currency_dropped:
        est.uncertainty_reasons.append(
            f"{len(_currency_dropped)} cost component(s) provided a figure with no currency, and the "
            "evidence does not establish one; those figures are not expressed as an amount and the "
            "activities are carried as unpriced cost drivers."
        )

    _results_by_id = {r.component_id: r for r in results}

    # --- Fill MULTIPLY calculation traces for audit (executor value only).
    _fill_traces(accepted_proposals, {c.component_id: c for c in components}, traces)

    one_time = [r for r in rows if r.c.recurrence == "ONE_TIME"]
    recurring = [r for r in rows if r.c.recurrence == "RECURRING"]

    if recurring:
        # A range-only recurring component (no point) cannot contribute a
        # single recurring figure -- Phase 9.7 §2 forbids manufacturing one.
        if any(r.point is None for r in recurring):
            est.uncertainty_reasons.append(
                "A recurring cost driver is stated only as a range with no single best "
                "estimate; the recurring total is not expressed as one figure."
            )
        else:
            est.recurring_cost = _round(sum(r.point for r in recurring))
        periods = {r.c.recurring_period for r in recurring if r.c.recurring_period}
        est.recurring_period = next(iter(periods), None) if len(periods) == 1 else None
        if len(periods) > 1:
            est.uncertainty_reasons.append(
                "Recurring costs are stated over different periods; they are not combined into a "
                "single recurring figure."
            )
        # EXPLICIT horizon only (spec Pass 33 §3/§7): a proposal that targets a
        # recurring component and carries horizon_basis == "EXPLICIT" with a
        # matching unit lets the periodic cost be totalled. Nothing is inferred
        # -- absent an explicit horizon, the periodic amount stands alone.
        _rec_cids = {r.c.component_id for r in recurring}
        _horizons = [
            (p.horizon, (p.horizon_unit or "").strip().lower(), p.horizon_basis)
            for p in accepted_proposals
            if str(getattr(p, "horizon_basis", "")).upper() == "EXPLICIT"
            and getattr(p, "horizon", None)
            and (set(p.component_ids) & _rec_cids
                 or getattr(p, "target_component_id", None) in _rec_cids
                 or any(o.source_component_id in _rec_cids for o in getattr(p, "operands", [])))
        ]
        _rec_period = (est.recurring_period or "").strip().lower()
        _matched = [h for h in _horizons if not h[1] or not _rec_period or h[1].rstrip("s") == _rec_period.rstrip("s")]
        if est.recurring_cost is not None and len(_matched) == 1 and est.recurring_period:
            _h = _matched[0][0]
            est.recurring_horizon = _h
            est.recurring_horizon_basis = "EXPLICIT"
            est.recurring_horizon_total = _round(est.recurring_cost * _h)

    if one_time:
        low, ml, high, method = _aggregate_one_time(one_time, _results_by_id, est, scope_established)
        est.one_time_cost = ml
        est.low, est.most_likely, est.high = low, ml, high
        est.estimation_method = method
        est.contributing_bases = [r.c.unit_cost_basis for r in one_time]
        est.estimate_classification = _classify(one_time)
    elif recurring:
        est.estimation_method = "recurring cost only; no one-time implementation cost established"

    if est.most_likely is None and est.low is None and est.high is None and not est.estimation_method:
        est.estimation_method = "no calculable component cost"

    est.component_results = results
    return est


def _aggregate_one_time(
    rows: list[_Row],
    results_by_id: dict[str, RemediationCostComponentResult],
    est: AssembledEstimate,
    scope_established: bool = True,
) -> tuple[float | None, float | None, float | None, str]:
    """Combine one-time rows by their declared semantic role. Returns
    (low, most_likely, high, method_note)."""
    additive = [r for r in rows if r.c.amount_type in _ADDITIVE_TYPES]
    subtotals = [r for r in rows if r.c.amount_type == "SUBTOTAL"]
    grand_totals = [r for r in rows if r.c.amount_type == "TOTAL"]
    alternatives = [r for r in rows if r.c.amount_type == "ALTERNATIVE"]

    methods: list[str] = []

    # --- SUBTOTAL reconciliation: a subtotal that restates the sum of the
    #     itemised additive components is the SAME money -- drop it (mark
    #     derived). One that disagrees, or has no components to roll up, is
    #     kept as its own additive line and the discrepancy is flagged.
    additive_sum = sum(r.point for r in additive if r.point is not None)
    for st in subtotals:
        if additive and _close(st.point, additive_sum):
            if st.c.component_id in results_by_id:
                results_by_id[st.c.component_id].is_derived = True
            methods.append("stated subtotal matched the itemised components and was not double counted")
        elif additive:
            est.has_reconciliation_conflict = True
            est.uncertainty_reasons.append(
                f"A stated subtotal ({st.point:g}) does not reconcile with the sum of the itemised "
                f"components ({additive_sum:g}); the itemised components were used and the "
                "estimate is treated as PARTIAL pending reconciliation."
            )
        else:
            additive.append(st)  # nothing to roll up -> treat as a component

    # --- ALTERNATIVE options: bracket each group as a scenario, never sum.
    alt_low = alt_ml = alt_high = 0.0
    # Phase 9.9 §12: a most-likely figure for alternatives exists only when the
    # LLM DECLARED a primary option (with a single point value) for every
    # decision group. Picking "the first option" or adding 0 would invent one.
    _alt_ml_established = True
    if alternatives:
        groups: dict[str, list[_Row]] = {}
        for r in alternatives:
            groups.setdefault(r.c.alternative_group or "_default", []).append(r)
        for grp in groups.values():
            lows = [r.low for r in grp]
            highs = [r.high for r in grp]
            primary = next((r for r in grp if r.c.is_primary_option), None)
            alt_low += min(lows)
            alt_high += max(highs)
            _cand = primary.point if primary is not None else None
            if _cand is None:
                _alt_ml_established = False
            else:
                alt_ml += _cand
        methods.append(
            f"range reflects {sum(len(g) for g in groups.values())} alternative implementation "
            f"option(s) across {len(groups)} decision(s)"
        )

    _additive_missing_point = any(r.point is None for r in additive)
    add_ml = None if (_additive_missing_point or not _alt_ml_established) else sum(r.point for r in additive)
    add_low = sum(r.low for r in additive)
    add_high = sum(r.high for r in additive)
    if _additive_missing_point:
        methods.append(
            "one or more components are stated only as a range with no single best estimate; "
            "no most-likely figure is established for those components"
        )
    elif any(r.low != r.point or r.high != r.point for r in additive):
        methods.append("range assembled from component-level cost uncertainty")

    from_parts_ml = _round(add_ml + alt_ml) if add_ml is not None else None
    from_parts_low = _round(add_low + alt_low)
    from_parts_high = _round(add_high + alt_high)

    # --- Stated complete TOTAL(s).
    if grand_totals:
        if len(grand_totals) > 1:
            # Competing complete totals -> treat as alternatives to each other.
            pts = sorted(r.point for r in grand_totals if r.point is not None)
            est.uncertainty_reasons.append(
                f"The evidence states more than one complete implementation total ({', '.join(f'{p:g}' for p in pts)}); "
                "they are shown as a range rather than reconciled to a single figure."
            )
            return _round(pts[0]), _round(pts[0]), _round(pts[-1]), "; ".join(
                methods + ["multiple stated totals bracketed as a range"]
            ) or "multiple stated totals"

        gt = grand_totals[0]
        parts_present = bool(additive or alternatives or subtotals)
        if not parts_present:
            _b = gt.c.unit_cost_basis.lower()
            note = (
                f"single {_b} implementation total preserved"
                if _b in ("verified", "reported")
                else "single stated implementation total preserved"
            )
            return _round(gt.low), _round(gt.point), _round(gt.high), note
        if from_parts_ml is not None and _close(gt.point, from_parts_ml):
            methods.append("stated total reconciled with, and equals, the sum of the itemised parts")
            lo = min(gt.low, from_parts_low) if from_parts_low is not None else gt.low
            hi = max(gt.high, from_parts_high) if from_parts_high is not None else gt.high
            return _round(lo), _round(gt.point), _round(hi), "; ".join(methods)
        # Conflict: stated total vs itemised parts.
        est.has_reconciliation_conflict = True
        est.uncertainty_reasons.append(
            f"The stated implementation total ({gt.point:g}) does not reconcile with the sum of the "
            f"itemised parts ({from_parts_ml:g}); both figures are shown and the range spans them."
        )
        stronger_is_total = _BASIS_RANK.get(gt.c.unit_cost_basis, 0) >= _max_basis_rank(additive + alternatives)
        ml = gt.point if stronger_is_total else from_parts_ml
        lo = min(gt.low, from_parts_low if from_parts_low is not None else gt.low)
        hi = max(gt.high, from_parts_high if from_parts_high is not None else gt.high)
        return _round(lo), _round(ml), _round(hi), "; ".join(methods + ["stated total and itemised parts preserved without a forced reconciliation"])

    # --- No stated total: the answer is the sum of required parts (+ chosen
    #     alternative). A genuine range appears only from component uncertainty
    #     or alternative options.
    if from_parts_ml is None:
        # Phase 9.7 §2: a missing most-likely figure (one or more components
        # are range-only) must not blank out a genuinely computable low/high
        # range -- only the point/most-likely figure is unavailable.
        if from_parts_low is not None and from_parts_high is not None:
            return from_parts_low, None, from_parts_high, "; ".join(methods) or "range established; no single most-likely figure"
        return None, None, None, "; ".join(methods) or "no calculable one-time cost"
    if not methods:
        # "required" asserts the work is established remediation; only say it when
        # the model's scope declaration supports it (Phase 9.9 §7).
        _req = "required implementation" if scope_established else "priced activity"
        methods.append(
            f"sum of the {_req} components"
            if len(additive) > 1 else ("single implementation cost" if scope_established else "single priced activity cost")
        )
    return from_parts_low, from_parts_ml, from_parts_high, "; ".join(methods)


def _max_basis_rank(rows: list[_Row]) -> int:
    return max((_BASIS_RANK.get(r.c.unit_cost_basis, 0) for r in rows), default=0)


def _evaluate_plan(operation: str, values: list[float]) -> float | None:
    """Pure arithmetic execution of an LLM-supplied calculation plan. NO
    business meaning -- the LLM decided the operands and the operation; this
    only combines the numbers (spec Pass 32 §3/§4/§29)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if operation == "MULTIPLY":
        out = 1.0
        for v in vals:
            out *= v
        return _round(out)
    if operation == "SUBTRACT" and len(vals) >= 2:
        return _round(vals[0] - sum(vals[1:]))
    if operation == "DIVIDE" and len(vals) >= 2:
        denom = 1.0
        for v in vals[1:]:
            denom *= v
        return _round(vals[0] / denom) if denom else None
    return _round(sum(vals))


def _fill_traces(
    proposals: list[RemediationCalculationProposal],
    by_id: dict[str, RemediationCostComponent],
    traces: list[RemediationCalculationTrace],
) -> None:
    """Execute each LLM calculation plan for the audit trail. The plan's
    operands (explicit values the LLM supplied, or the point amounts of the
    referenced components) are combined with the LLM-declared operation.
    `produces: LOW/MOST_LIKELY/HIGH` is NOT applied to the estimate -- the
    authoritative numbers come from role-based component assembly
    (_aggregate_one_time). Spec Pass 32 §3-§5, §29: the LLM owns the plan, the
    executor owns the arithmetic, neither drives the headline figure alone."""
    trace_by_id = {t.calculation_id: t for t in traces}
    for p in proposals:
        # Rich form: explicit operand values the LLM supplied.
        explicit = [o.value for o in getattr(p, "operands", []) if o.value is not None]
        if explicit:
            values = explicit
            operand_desc = ", ".join(
                f"{o.label or '?'}={o.value:g}" for o in p.operands if o.value is not None
            )
        else:
            operands = [by_id[cid] for cid in p.component_ids if cid in by_id]
            values = [a for a in (_point_amount(o) for o in operands) if a is not None]
            operand_desc = str(p.component_ids)
        if not values:
            continue
        val = _evaluate_plan(p.operation, values)
        tr = trace_by_id.get(p.calculation_id)
        if tr is not None and val is not None:
            tr.executor_result = val
            _freq = f" [{p.frequency}]" if (getattr(p, "frequency", None) or "ONE_TIME") != "ONE_TIME" else ""
            _rep = f" = {p.result_represents}" if getattr(p, "result_represents", "") else ""
            tr.formula = f"{p.operation}({operand_desc}) -> {val:g}{_freq}{_rep} (audit only)"
            if getattr(p, "result_represents", ""):
                tr.result_represents = p.result_represents
            tr.frequency = getattr(p, "frequency", None) or "ONE_TIME"
            if tr.llm_proposed_result is not None and not _close(tr.llm_proposed_result, val):
                tr.disagreement = (
                    f"LLM proposed {tr.llm_proposed_result:g}; deterministic executor computed "
                    f"{val:g}. Neither figure sets the estimate -- the range is derived from "
                    "component structure."
                )


def _classify(rows: list[_Row]) -> CostBasis:
    if not rows:
        return CostBasis.NOT_ESTABLISHED
    bases = [r.c.unit_cost_basis for r in rows]
    if len(rows) == 1 and bases[0] == "VERIFIED":
        return CostBasis.VERIFIED
    weakest = min(bases, key=lambda b: _BASIS_RANK.get(b, 0))
    if weakest in ("ASSUMED", "NOT_ESTABLISHED"):
        return CostBasis.ASSUMED
    if all(b in ("VERIFIED", "REPORTED") for b in bases):
        return CostBasis.REPORTED if "REPORTED" in bases else CostBasis.VERIFIED
    return CostBasis.ESTIMATED
