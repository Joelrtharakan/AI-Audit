"""Deterministic structural validation for the LLM remediation-cost
interpretation.

The numeric-safety boundary described in the architecture: this module performs
NO arithmetic and NO cost reasoning. It decides, for every component and every
calculation proposal the LLM produced, whether the underlying references exist
and whether the claimed evidence basis is real -- downgrading any
VERIFIED/REPORTED claim that is not backed by a cited evidence item, and
stripping any unit cost that has no defensible basis at all (spec sections 5,
6). Arithmetic, double-count reconciliation, and range assembly happen
afterwards in `app.remediation.calculator`.

A rejected item is never silently dropped -- it is recorded with a neutral
reason so a reviewer can see why a number was not produced.
"""

from __future__ import annotations

import re

from app.remediation.semantic_models import (
    RemediationCalculationProposal,
    RemediationCalculationTrace,
    RemediationCostComponent,
    RemediationInterpretation,
    RemediationRejectedItem,
    RemediationValidationOutcome,
)

_EVIDENCE_ID_RE = re.compile(r"^E\d+$")
_PER_X_TYPES = frozenset({"PER_QUANTITY", "PER_HOUR", "PER_UNIT", "PER_EVENT", "PER_IMPLEMENTATION"})
_TOTAL_TYPES = frozenset({"TOTAL", "SUBTOTAL"})


def _finite_pos(v: float | None) -> bool:
    return v is not None and v > 0


def _is_evidence_ref(ref: str, valid_evidence_ids: set[str]) -> bool:
    return ref in valid_evidence_ids and bool(_EVIDENCE_ID_RE.match(ref))


def _validate_component(
    c: RemediationCostComponent,
    valid_reference_ids: set[str],
    valid_evidence_ids: set[str],
    outcome: RemediationValidationOutcome,
) -> RemediationCostComponent | None:
    """Return a possibly-adjusted copy of the component, or None if it must be
    dropped entirely. Adjustments (basis downgrade, pricing strip) are recorded
    in `outcome.llm_disagreements`."""
    # --- Observed-value / remediation-cost boundary (spec sections 16/17/25):
    # an observed ESTIMATE or QUOTATION carried over from the finding is
    # finding / financial evidence, not remediation work. It must never be
    # modelled as a remediation cost component regardless of how the LLM
    # labelled it. Structural test on the component's own quantity unit and
    # description -- no domain vocabulary.
    from app.services.semantic_subject import _measurement_artifact_head
    _qu = (c.quantity_unit or "").strip().lower().rstrip("s")
    _ARTIFACT_UNITS = {
        "estimate", "quotation", "quote", "projection", "forecast",
        "appraisal", "valuation", "bid", "tender",
    }
    if _qu in _ARTIFACT_UNITS or _measurement_artifact_head(c.description):
        outcome.rejected.append(
            RemediationRejectedItem(
                item_id=c.component_id, kind="COMPONENT",
                reason_code="OBSERVED_VALUE_NOT_REMEDIATION",
                detail=(
                    "This component identifies an observed estimate/quotation from the "
                    "finding, which is finding evidence rather than remediation work; "
                    "it was not priced as a remediation cost."
                ),
            )  # type: ignore[arg-type]
        )
        return None

    kept_refs = [r for r in c.source_reference_ids if r in valid_reference_ids]
    dropped_refs = [r for r in c.source_reference_ids if r not in valid_reference_ids]
    data = c.model_dump()
    data["source_reference_ids"] = kept_refs

    if dropped_refs:
        outcome.llm_disagreements.append(
            f"{c.component_id}: reference(s) {dropped_refs} do not exist and were ignored."
        )

    has_evidence_ref = any(_is_evidence_ref(r, valid_evidence_ids) for r in kept_refs)

    # --- Evidence-vs-assumption integrity (spec section 5): a VERIFIED/REPORTED
    # unit-cost claim must be backed by a real cited EVIDENCE item; otherwise it
    # is downgraded. The LLM's word is never taken on faith.
    if c.unit_cost_basis in ("VERIFIED", "REPORTED") and not has_evidence_ref:
        data["unit_cost_basis"] = "ESTIMATED" if c.assumptions or c.unit_cost is not None else "ASSUMED"
        outcome.llm_disagreements.append(
            f"{c.component_id}: unit cost marked {c.unit_cost_basis} but no cited evidence "
            f"supports it -- downgraded to {data['unit_cost_basis']}."
        )

    if c.quantity_basis == "EVIDENCED" and not has_evidence_ref:
        data["quantity_basis"] = "ASSUMED"
        outcome.llm_disagreements.append(
            f"{c.component_id}: quantity marked EVIDENCED but no cited evidence supports it -- "
            "downgraded to ASSUMED."
        )

    # --- "Do not manufacture precision." A monetary figure enters the
    # estimate ONLY when it is anchored to evidence -- never merely because
    # the LLM labelled an invented number ESTIMATED/ASSUMED. Anchored means:
    #
    #   * unit_cost_basis is VERIFIED or REPORTED  (an evidence item stated
    #     the price / rate / total), OR
    #   * unit_cost_basis is ESTIMATED *and* the component cites >=1 real
    #     evidence item  (a defensible inference from a cited source, e.g. a
    #     midpoint of two stated internal estimates).
    #
    # ADDITIONALLY, a per-unit RATE (PER_HOUR / PER_UNIT / ...) is anchored
    # only when the QUANTITY it multiplies is itself EVIDENCED -- an assumed
    # "N hours x rate" is fabricated effort (spec: no default effort quantity,
    # no default labour rate).
    #
    # Not anchored -> the figure and any range are removed; the component
    # SURVIVES as a named, unpriced cost driver (spec sections 1-8, 18, 20).
    eff_basis = data["unit_cost_basis"]
    _rate_anchored = eff_basis in ("VERIFIED", "REPORTED") or (
        eff_basis == "ESTIMATED" and has_evidence_ref
    )
    _is_rate = c.amount_type in ("PER_QUANTITY", "PER_HOUR", "PER_UNIT", "PER_EVENT", "PER_IMPLEMENTATION")
    _qty_anchored = data.get("quantity_basis") == "EVIDENCED"
    _anchored = _rate_anchored and (_qty_anchored or not _is_rate)

    if not _anchored and (
        data.get("unit_cost") is not None
        or data.get("unit_cost_low") is not None
        or data.get("unit_cost_high") is not None
    ):
        _why = (
            "an assumed per-unit quantity" if (_is_rate and _rate_anchored and not _qty_anchored)
            else "no evidence-backed pricing basis"
        )
        data["unit_cost"] = None
        data["unit_cost_low"] = None
        data["unit_cost_high"] = None
        data["unit_cost_basis"] = "NOT_ESTABLISHED"
        outcome.llm_disagreements.append(
            f"{c.component_id}: {_why} -- no monetary amount was produced; the cost driver is "
            "retained for the auditor with pricing not established."
        )

    # --- Non-positive / non-finite numeric guards.
    if data.get("unit_cost") is not None and not _finite_pos(data["unit_cost"]):
        data["unit_cost"] = None
        data["unit_cost_basis"] = "NOT_ESTABLISHED"
    if data.get("quantity") is not None and not _finite_pos(data["quantity"]):
        data["quantity"] = None
        data["quantity_basis"] = "NOT_ESTABLISHED"
    for k in ("unit_cost_low", "unit_cost_high"):
        if data.get(k) is not None and not _finite_pos(data[k]):
            data[k] = None

    # --- Double-count structural guard (spec section 11): a TOTAL/SUBTOTAL never
    # carries a live quantity multiplier.
    if c.amount_type in _TOTAL_TYPES and data.get("quantity") is not None:
        outcome.llm_disagreements.append(
            f"{c.component_id}: amount_type {c.amount_type} is an already-total figure; its "
            "quantity was ignored to prevent double counting."
        )
        data["quantity"] = None
        data["quantity_basis"] = "NOT_ESTABLISHED"

    return RemediationCostComponent(**data)


def _validate_proposal(
    calc: RemediationCalculationProposal,
    by_id: dict[str, RemediationCostComponent],
    outcome: RemediationValidationOutcome,
) -> bool:
    def reject(code: str, detail: str) -> None:
        outcome.rejected.append(
            RemediationRejectedItem(item_id=calc.calculation_id, kind="CALCULATION", reason_code=code, detail=detail)  # type: ignore[arg-type]
        )

    operands = [by_id[cid] for cid in calc.component_ids if cid in by_id]
    if len(operands) != len(calc.component_ids) or not operands:
        reject("UNKNOWN_COMPONENT", "A referenced cost component does not exist.")
        return False

    currencies = {o.currency for o in operands if o.currency}
    if len(currencies) > 1:
        reject("INCOMPATIBLE_CURRENCY", f"Operands span multiple currencies: {sorted(currencies)}.")
        return False

    if calc.operation == "MULTIPLY":
        # Either a single self-contained component (qty + unit_cost + per-X), or
        # exactly two components: one carrying the quantity, one the unit cost.
        if len(operands) == 1:
            o = operands[0]
            if not (_finite_pos(o.quantity) and _finite_pos(o.unit_cost) and o.amount_type in _PER_X_TYPES):
                reject(
                    "AMBIGUOUS_OPERANDS",
                    "MULTIPLY needs a component with both a quantity and a per-unit cost.",
                )
                return False
        elif len(operands) == 2:
            qty_side = [o for o in operands if _finite_pos(o.quantity)]
            cost_side = [o for o in operands if _finite_pos(o.unit_cost)]
            if len(qty_side) != 1 or len(cost_side) != 1 or qty_side[0] is cost_side[0]:
                reject(
                    "AMBIGUOUS_OPERANDS",
                    "MULTIPLY needs exactly one quantity-bearing and one unit-cost-bearing operand.",
                )
                return False
        else:
            reject("AMBIGUOUS_OPERANDS", "MULTIPLY takes one or two operands.")
            return False
    else:  # SUM / SUBTRACT
        priced = [
            o for o in operands
            if _finite_pos(o.unit_cost) or (_finite_pos(o.quantity) and _finite_pos(o.unit_cost))
        ]
        if len(priced) < 2:
            reject(
                "MISSING_UNIT_COST",
                f"{calc.operation} needs at least two operands with a usable monetary value.",
            )
            return False

    outcome.accepted_calculation_ids.append(calc.calculation_id)
    outcome.traces.append(
        RemediationCalculationTrace(
            calculation_id=calc.calculation_id,
            operation=calc.operation,
            component_ids=list(calc.component_ids),
            produces=calc.produces,
            currency=next(iter(currencies), None),
            llm_proposed_result=calc.proposed_result_value,
        )
    )
    if calc.proposed_result_value is not None:
        outcome.llm_disagreements.append(
            f"{calc.calculation_id}: LLM proposed {calc.proposed_result_value} -- the authoritative "
            "value is computed independently by the deterministic calculator, not this figure."
        )
    return True


def validate_and_plan(
    interpretation: RemediationInterpretation,
    valid_evidence_ids: set[str],
    valid_hypothesis_ids: set[str] | None = None,
    valid_capa_refs: set[str] | None = None,
) -> tuple[list[RemediationCostComponent], list[RemediationCalculationProposal], RemediationValidationOutcome]:
    """Validate structure + provenance. Returns
    (surviving_components, accepted_proposals, outcome). No arithmetic."""
    valid_hypothesis_ids = valid_hypothesis_ids or set()
    valid_capa_refs = valid_capa_refs or set()
    valid_reference_ids = valid_evidence_ids | valid_hypothesis_ids | valid_capa_refs

    outcome = RemediationValidationOutcome()

    components: list[RemediationCostComponent] = []
    for c in interpretation.cost_components:
        adjusted = _validate_component(c, valid_reference_ids, valid_evidence_ids, outcome)
        if adjusted is None:
            outcome.dropped_component_ids.append(c.component_id)
            if not any(r.item_id == c.component_id and r.kind == "COMPONENT" for r in outcome.rejected):
                outcome.rejected.append(
                    RemediationRejectedItem(item_id=c.component_id, kind="COMPONENT", reason_code="MISSING_PROVENANCE", detail="Component could not be validated.")  # type: ignore[arg-type]
                )
            continue
        components.append(adjusted)

    by_id = {c.component_id: c for c in components}
    accepted: list[RemediationCalculationProposal] = []
    for calc in interpretation.calculation_proposals:
        if _validate_proposal(calc, by_id, outcome):
            accepted.append(calc)

    return components, accepted, outcome
