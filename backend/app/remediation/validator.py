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

# A cited pricing source is either an enumerated evidence item (E0, E1, ...) or
# the FINDING itself. The finding text is a provided, auditor-authored source
# document -- a price / rate / quantity / recurrence stated IN the finding is
# evidence for pricing exactly as an evidence-ledger item would be. It is
# addressed by the reserved id "FINDING" (spec Pass 51 sections 3-6, 17-18).
# This is provenance plumbing, not semantic inference: the model still has to
# attribute the number to the finding, and a number it attributes to the
# finding that is not actually there is a hallucination handled no differently
# from a mis-cited E-id.
_EVIDENCE_ID_RE = re.compile(r"^(?:E\d+|FINDING)$")
_FINDING_REF_ID = "FINDING"
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
    # PRIMARY: the LLM's own `value_kind` (spec Pass 34 §21). A value it
    # classified as an incurred loss / historical spend is not remediation
    # expenditure -- reject it regardless of any other labelling.
    if c.value_kind in ("OBSERVED_FINANCIAL_LOSS", "HISTORICAL_EXPENDITURE"):
        outcome.rejected.append(
            RemediationRejectedItem(
                item_id=c.component_id, kind="COMPONENT",
                reason_code="OBSERVED_VALUE_NOT_REMEDIATION",
                detail=(
                    "This value was classified as an incurred loss / historical "
                    "expenditure that describes the finding, not remediation work; it "
                    "was not priced as a remediation cost."
                ),
            )  # type: ignore[arg-type]
        )
        return None
    # FAIL-CLOSED FLOOR (only when the LLM did NOT classify -- value_kind still
    # the default): a component whose own quantity unit is an assessment
    # artefact is finding evidence, not work. Subordinate to the LLM's
    # classification above; never overrides an explicit value_kind.
    if c.value_kind in ("REMEDIATION_COST", "NOT_ESTABLISHED"):
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

    # --- PASS 36 §A: a BUDGET / ESTIMATE figure is by definition not a
    # verified actual amount -- cap its basis at ESTIMATED so it can never
    # drive an EXACT / VERIFIED estimate classification downstream. Structural
    # consistency of the LLM's own two fields; no text inspection.
    if c.value_kind in ("BUDGET", "ESTIMATE") and data.get("unit_cost_basis") in ("VERIFIED", "REPORTED"):
        data["unit_cost_basis"] = "ESTIMATED"
        outcome.llm_disagreements.append(
            f"{c.component_id}: value_kind {c.value_kind} cannot also be a verified actual "
            "cost -- basis capped at ESTIMATED."
        )

    # A DERIVED quantity is the LLM's arithmetic combination of evidenced values
    # (spec Pass 32 §5). It is only trustworthy when (a) the LLM actually stated
    # the derivation and (b) the component cites the evidence the operands came
    # from. Missing either -> it is an unexplained number, downgrade to ASSUMED.
    if c.quantity_basis == "DERIVED" and (
        not (c.quantity_derivation or "").strip() or not has_evidence_ref
    ):
        data["quantity_basis"] = "ASSUMED"
        outcome.llm_disagreements.append(
            f"{c.component_id}: quantity marked DERIVED without a stated derivation and cited "
            "evidence -- downgraded to ASSUMED."
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
    _qty_anchored = data.get("quantity_basis") in ("EVIDENCED", "DERIVED")
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

    # --- PASS 36 §A: the LLM MUST classify the economic role of every monetary
    # value. An UNCLASSIFIED value (value_kind == NOT_ESTABLISHED, the
    # fail-closed default) is NEVER priced -- the number is stripped and the
    # component survives as an unpriced driver, regardless of its provenance.
    # Deterministic code does not recover the classification from words
    # ("cost"/"price"/"rate"/...) or from the evidence status. Other components
    # are untouched.
    if c.value_kind == "NOT_ESTABLISHED" and data.get("unit_cost") is not None:
        data["unit_cost"] = data["unit_cost_low"] = data["unit_cost_high"] = None
        data["unit_cost_basis"] = "NOT_ESTABLISHED"
        outcome.llm_disagreements.append(
            f"{c.component_id}: the economic role of this monetary value was not classified "
            "(value_kind); no amount was produced -- the driver is retained for the auditor."
        )
        outcome.rejected.append(
            RemediationRejectedItem(
                item_id=c.component_id, kind="COMPONENT", reason_code="MISSING_PROVENANCE",
                detail=(
                    "The economic role of this monetary value was not established (not "
                    "classified as a rate / quotation / budget / remediation cost). Confirm "
                    "what the value represents."
                ),
            )  # type: ignore[arg-type]
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

    # --- Recurrence / period consistency (spec Pass 33 §2/§6/§30). NOT a
    # semantic rule -- it reconciles two fields of the SAME LLM output that
    # must agree, and prevents the demonstrated failure (a monthly cost
    # mislabelled ONE_TIME, then multiplied by a period count and summed into
    # the one-time total). The number is stripped, never silently re-bucketed;
    # the driver survives unpriced so the auditor is asked for the frequency.
    _period = (data.get("recurring_period") or "").strip()
    _recurrence = data.get("recurrence")
    _has_amount = data.get("unit_cost") is not None
    _freq_conflict_detail = ""
    if _has_amount and _recurrence == "ONE_TIME" and _period:
        _freq_conflict_detail = (
            f"This cost is marked one-time but states a recurring period ('{_period}'); "
            "the frequency is contradictory, so no amount was produced. Confirm whether "
            "the cost is one-time or recurring."
        )
    elif _has_amount and _recurrence == "RECURRING" and not _period:
        _freq_conflict_detail = (
            "This cost is marked recurring but no recurrence period is stated, so the "
            "recurring amount cannot be expressed. Confirm the recurrence period."
        )
    if _freq_conflict_detail:
        data["unit_cost"] = data["unit_cost_low"] = data["unit_cost_high"] = None
        data["unit_cost_basis"] = "NOT_ESTABLISHED"
        outcome.llm_disagreements.append(f"{c.component_id}: {_freq_conflict_detail}")
        outcome.rejected.append(
            RemediationRejectedItem(
                item_id=c.component_id, kind="COMPONENT", reason_code="INVALID_NUMBER",
                detail=_freq_conflict_detail,
            )  # type: ignore[arg-type]
        )

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

    # --- Rich form (Pass 32): the LLM supplied explicit operand values. The
    #     executor just combines them; here we only check the plan is
    #     structurally evaluable and scrub unknown evidence refs. The number is
    #     audit-only, so we never reject on provenance -- we record it.
    _explicit = [o for o in (getattr(calc, "operands", []) or []) if o.value is not None]
    if _explicit:
        # Pass 35 §6/§15: a rich self-describing plan MUST state its recurrence
        # explicitly -- no semantic default. An absent `frequency` here is a
        # structurally incomplete plan; reject it (the LLM must say ONE_TIME or
        # RECURRING). Legacy component-id-only plans are exempt (below).
        if getattr(calc, "frequency", None) not in ("ONE_TIME", "RECURRING"):
            reject(
                "UNSUPPORTED_OPERATION",
                "The calculation plan does not state its recurrence (frequency ONE_TIME or "
                "RECURRING); it is structurally incomplete and was not used.",
            )
            return False
        for o in calc.operands:
            o.evidence_refs = [r for r in (o.evidence_refs or []) if r in by_id or _EVIDENCE_ID_RE.match(r)]
        if calc.operation in ("SUBTRACT", "DIVIDE") and len(_explicit) < 2:
            reject("AMBIGUOUS_OPERANDS", f"{calc.operation} needs at least two operands with a value.")
            return False
        _cur = calc.currency or next(
            (o.unit for o in _explicit if o.unit and o.unit.isalpha()), None
        )
        outcome.accepted_calculation_ids.append(calc.calculation_id)
        outcome.traces.append(
            RemediationCalculationTrace(
                calculation_id=calc.calculation_id,
                operation=calc.operation,
                component_ids=list(calc.component_ids),
                operands=list(calc.operands),
                produces=calc.produces,
                frequency=getattr(calc, "frequency", None) or "ONE_TIME",
                result_represents=getattr(calc, "result_represents", ""),
                currency=_cur,
                llm_proposed_result=calc.proposed_result_value,
            )
        )
        return True

    operands = [by_id[cid] for cid in calc.component_ids if cid in by_id]
    if len(operands) != len(calc.component_ids) or not operands:
        reject("UNKNOWN_COMPONENT", "A referenced cost component does not exist.")
        return False

    currencies = {o.currency for o in operands if o.currency}
    if len(currencies) > 1:
        reject("INCOMPATIBLE_CURRENCY", f"Operands span multiple currencies: {sorted(currencies)}.")
        return False

    if calc.operation == "DIVIDE":
        if len(operands) < 2:
            reject("AMBIGUOUS_OPERANDS", "DIVIDE needs at least two operands.")
            return False
    elif calc.operation == "MULTIPLY":
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
            frequency=getattr(calc, "frequency", "ONE_TIME"),
            result_represents=getattr(calc, "result_represents", ""),
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

    _enforce_single_recurrence_authority(accepted, by_id, outcome)

    return components, accepted, outcome


def _norm_period(p: str | None) -> str:
    return (p or "").strip().lower().rstrip("s")


def _enforce_single_recurrence_authority(
    accepted: list[RemediationCalculationProposal],
    by_id: dict[str, RemediationCostComponent],
    outcome: RemediationValidationOutcome,
) -> None:
    """The component is the ONE authoritative recurrence representation (spec
    Pass 34 §1/§8/§32). A calculation proposal that targets a component and
    declares a DIFFERENT frequency / period is an internally inconsistent LLM
    response -- it is NOT reinterpreted. That specific component's number is
    failed (stripped, driver retained) and the proposal dropped; every other
    component and proposal is preserved. Pure field comparison; no inference
    of what a period means."""
    _drop: set[str] = set()
    for calc in list(accepted):
        _cfreq = str(getattr(calc, "frequency", "") or "").strip().upper()
        _cperiod = _norm_period(getattr(calc, "recurring_period", None))
        targets = [by_id[cid] for cid in calc.component_ids if cid in by_id]
        targets += [
            by_id[o.source_component_id] for o in getattr(calc, "operands", [])
            if o.source_component_id in by_id
        ]
        # The component's recurrence is authoritative (spec Pass 34 §8). A
        # proposal creates a conflict ONLY when it POSITIVELY asserts a
        # different recurrence -- frequency RECURRING against a ONE_TIME
        # component, or an explicit period that differs. A proposal that simply
        # left `frequency` at its ONE_TIME default is an unset mirror, not a
        # contradiction, and never strips a component.
        conflict = False
        for comp in targets:
            _kfreq = "RECURRING" if comp.recurrence == "RECURRING" else "ONE_TIME"
            if _cfreq == "RECURRING" and _kfreq == "ONE_TIME":
                conflict = True
            _kperiod = _norm_period(comp.recurring_period)
            if _cperiod and _kperiod and _cperiod != _kperiod:
                conflict = True
        if conflict:
            _drop.add(calc.calculation_id)
            for comp in targets:
                if comp.unit_cost is not None:
                    comp.unit_cost = comp.unit_cost_low = comp.unit_cost_high = None
                    comp.unit_cost_basis = "NOT_ESTABLISHED"
                    outcome.rejected.append(
                        RemediationRejectedItem(
                            item_id=comp.component_id, kind="COMPONENT",
                            reason_code="INVALID_NUMBER",
                            detail=(
                                "The calculation plan for this cost states a different "
                                "recurrence/period than the cost component itself; the "
                                "frequency is internally inconsistent, so no amount was "
                                "produced. Confirm whether the cost is one-time or recurring "
                                "and, if recurring, over what period."
                            ),
                        )  # type: ignore[arg-type]
                    )
            outcome.llm_disagreements.append(
                f"{calc.calculation_id}: frequency/period disagrees with its target "
                "component(s); the calculation was dropped and the component(s) left unpriced."
            )
    if _drop:
        accepted[:] = [c for c in accepted if c.calculation_id not in _drop]
        outcome.accepted_calculation_ids = [
            cid for cid in outcome.accepted_calculation_ids if cid not in _drop
        ]
        outcome.traces[:] = [t for t in outcome.traces if t.calculation_id not in _drop]
