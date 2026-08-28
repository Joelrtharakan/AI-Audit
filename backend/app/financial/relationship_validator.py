"""Deterministic validation + materialization for LLM-proposed financial
relationships and calculations.

This module is the numeric-safety boundary described in the semantic
architecture: it NEVER performs arithmetic itself. Its job is to decide,
for every `CalculationProposal` the LLM produced, whether the underlying
evidence actually supports it (claim existence, provenance, evidence
status, population, units, currency, relationship support, conflicts,
ambiguity, numerical validity) -- and, for every proposal that passes,
translate the involved `SemanticClaim`s into the EXISTING
`app.financial.models.FinancialObservation` shape so the untouched,
already-tested `app.financial.calculator` functions perform the actual
arithmetic. This guarantees the semantic (LLM) path and the regex-
extraction path always share the same numeric engine.

A rejected proposal is never silently dropped -- it is recorded with a
reason code so the auditor can see exactly why a number was not produced.
"""

from __future__ import annotations

from app.financial.extractor import _valid_iso_code
from app.financial.models import FinancialAmountType, FinancialObservation
from app.financial.semantic_models import (
    CalculationTrace,
    RejectedCalculation,
    SemanticClaim,
    SemanticFindingInterpretation,
    SemanticRelationship,
    SemanticValidationOutcome,
)

_EVIDENCE_RANK = {"VERIFIED": 3, "REPORTED": 2, "UNVERIFIED": 1, "CONTRADICTED": 0}

_FACT_TYPE_TO_AMOUNT_TYPE: dict[str, FinancialAmountType] = {
    "RECOVERY": FinancialAmountType.RECOVERY,
    "REMEDIATION_COST": FinancialAmountType.REMEDIATION_COST,
    "PREVENTION_COST": FinancialAmountType.PREVENTION_COST,
}


def _claims_by_id(interpretation: SemanticFindingInterpretation) -> dict[str, SemanticClaim]:
    return {c.claim_id: c for c in interpretation.claims}


def _relationships_by_id(interpretation: SemanticFindingInterpretation) -> dict[str, SemanticRelationship]:
    return {r.relationship_id: r for r in interpretation.relationships}


def _valid_evidence_ids(evidence_count: int) -> set[str]:
    return {f"E{i}" for i in range(evidence_count)}


def _weakest_status(claims: list[SemanticClaim]) -> str:
    if not claims:
        return "UNVERIFIED"
    return min((c.evidence_status for c in claims), key=lambda s: _EVIDENCE_RANK.get(s, 0))


def _singularize(word: str) -> str:
    """Minimal, generic English-plural normalization ('shipments' ->
    'shipment', 'units' -> 'unit') -- not a lookup table for any specific
    noun, just the regular -s/-es rule, applied uniformly so a quantity
    claim naturally stated in the plural ('450 shipments') is not treated
    as a different unit from a rate claim naturally stated in the
    singular ('per shipment')."""
    if word.endswith("es") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        return word[:-1]
    return word


def _rate_denominator(unit: str) -> str:
    """A rate's unit is validly written either as a bare denominator
    ('hour', 'unit') or as a self-documenting compound ('INR/hour',
    'USD per unit') -- the LLM is not wrong to prefer the compound form,
    since it names the currency too. Only the denominator matters for
    compatibility with a quantity's unit, so it is extracted before
    comparison rather than requiring the two claims to spell their unit
    identically."""
    normalized = unit.strip().lower().replace(" per ", "/")
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1].strip()
    return _singularize(normalized)


_PERCENT_UNIT_STRINGS = frozenset({"%", "percent", "percentage", "pct"})


def _rate_value_as_multiplier(rate_claim: SemanticClaim) -> float | None:
    """A percentage-denominated rate (unit="%") is a real, universal
    mathematical fact independent of any domain or wording: "6%" always
    means the fraction 0.06, never the raw figure 6, when it participates
    in a MULTIPLY. The LLM correctly identifies THAT a claim is a
    percentage (fact_type=PERCENTAGE, or unit="%"/"percent"/"pct") --
    converting that to the multiplier a MULTIPLY needs is a mechanical
    arithmetic fact, not a semantic judgment, so it belongs in the
    deterministic executor exactly like any other unit normalization
    (see _rate_denominator above), not something the LLM must get exactly
    right in its own arithmetic."""
    if rate_claim.value is None:
        return None
    is_percent = rate_claim.fact_type == "PERCENTAGE" or (
        rate_claim.unit is not None and rate_claim.unit.strip().lower() in _PERCENT_UNIT_STRINGS
    )
    return rate_claim.value / 100.0 if is_percent else rate_claim.value


def _dimensional_unit(claim: SemanticClaim) -> str | None:
    """A claim's `unit` field sometimes just restates its currency code
    (e.g. unit="INR" on a RATE whose currency is also INR) rather than
    naming a real per-X denominator ("failure", "hour", "unit") -- a
    reasonable way for the LLM to fill the field when the "per X" is only
    implied grammatically (e.g. "each resulted in a cost of INR 18,000",
    where "each" carries the per-failure meaning but no unit word follows
    the amount). That is NOT a real dimensional unit and must not be
    compared against a quantity's unit as if it were one -- it is
    equivalent to no unit being stated at all (see _units_compatible's
    existing wildcard rule for a genuinely absent unit)."""
    if claim.unit is None:
        return None
    if _valid_iso_code(claim.unit) is not None or (
        claim.currency and claim.unit.strip().upper() == claim.currency.strip().upper()
    ):
        return None
    return claim.unit


def _amount_already_accounted(
    accounted: list[tuple[str, str, float]],
    population: str,
    currency: str | None,
    value: float | None,
) -> bool:
    """True when `value` (in `currency`, for `population`) has already been
    consumed as a component of -- or produced as the derived result of --
    an accepted calculation.

    Such a value is the SAME money re-expressed by a separate claim (e.g.
    an `amount_per_event` figure the LLM also emitted as a standalone
    gross-exposure claim, or the product N x X restated as a "total"),
    NOT an independent exposure, so it must not be summed a second time by
    the downstream calculator. This is pure structural value-identity --
    the same tolerance the calculator already uses for corroboration
    de-duplication -- and special-cases no scenario, wording, currency,
    cost factor, or amount."""
    if value is None or currency is None:
        return False
    for pop, cur, v in accounted:
        if pop == population and cur == currency and abs(v - value) <= max(1.0, abs(v) * 1e-6):
            return True
    return False


def _units_compatible(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return True  # a wildcard/unstated unit is compatible with anything
    return _rate_denominator(a) == _rate_denominator(b)


def _to_observation_population(population: str) -> str:
    """`FinancialObservation.financial_population` only recognizes
    CURRENT_FINDING/HISTORICAL/OTHER -- RECOVERY/REMEDIATION/PREVENTION
    are semantic-layer populations (used above for validation, e.g.
    rejecting a REMEDIATION claim mixed with a CURRENT_FINDING claim) but
    are represented on the materialized observation via `amount_type`
    (RECOVERY / REMEDIATION_COST / PREVENTION_COST) instead, exactly as
    the regex-extraction path already does."""
    if population == "HISTORICAL":
        return "HISTORICAL"
    if population == "OTHER":
        return "OTHER"
    return "CURRENT_FINDING"


def _amount_type_for(claim: SemanticClaim, validated_factor: FinancialAmountType | None = None) -> FinancialAmountType:
    # RECOVERY/REMEDIATION_COST/PREVENTION_COST already carry unambiguous
    # semantic meaning from their own fact_type -- a general cost-factor
    # selection (about the FINDING's overall nature) never overrides that.
    # A bare AMOUNT claim has no such intrinsic meaning, so the validated,
    # claim-grounded cost factor (if any) is used for it instead of the
    # generic VERIFIED/unverified default.
    if claim.fact_type == "AMOUNT" and validated_factor is not None:
        return validated_factor
    return _FACT_TYPE_TO_AMOUNT_TYPE.get(
        claim.fact_type,
        # No validated, claim-grounded cost factor and no more specific
        # fact_type mapping -- the honest answer is that the category
        # could not be confirmed (NOT_ESTABLISHED), not a fabricated
        # DIRECT_LOSS guess. The claim's own amount/evidence_status still
        # flow through unchanged, so this is a display-label difference
        # only, not a change to whether the amount is reported.
        FinancialAmountType.NOT_ESTABLISHED if claim.evidence_status == "VERIFIED" else FinancialAmountType.POTENTIAL_EXPOSURE,
    )


def _validate_cost_factor(interpretation: SemanticFindingInterpretation, claims: dict[str, SemanticClaim]) -> FinancialAmountType | None:
    """Independently check the LLM's CostFactorAssessment before trusting
    it: it must cite real, existing claims and carry at least MEDIUM
    confidence. An ungrounded, unsupported, LOW-confidence, or absent
    selection returns None -- the caller falls back to the existing
    deterministic default rather than fabricating a factor."""
    cf = interpretation.cost_factor
    if cf.selected_factor in ("NOT_ESTABLISHED", "OTHER"):
        return None
    # REMEDIATION_COST / PREVENTION_COST are IMPLEMENTATION-cost concepts -- what
    # it will cost to fix the finding -- not financial-exposure/loss factors.
    # They must never become the headline financial factor of the Cost &
    # Financial Exposure section (which answers "what has the finding already
    # cost?"). A finding whose only monetary content is a remediation quotation
    # yields Financial Factor: NOT ESTABLISHED here; the amount is analysed
    # separately by app.remediation (spec sections 1, 2, 15). The claims are
    # still materialized (via fact_type) for CAPA-payback economics.
    if cf.selected_factor in ("REMEDIATION_COST", "PREVENTION_COST"):
        return None
    if cf.confidence == "LOW":
        return None
    if not cf.supporting_claim_ids:
        return None
    if any(cid not in claims for cid in cf.supporting_claim_ids):
        return None
    try:
        return FinancialAmountType(cf.selected_factor)
    except ValueError:
        return None


def validate_and_materialize(
    interpretation: SemanticFindingInterpretation,
    evidence_count: int,
) -> tuple[list[FinancialObservation], SemanticValidationOutcome]:
    """Validate every calculation proposal; return the FinancialObservation
    list to feed into the existing deterministic calculator functions,
    plus a full audit of what was accepted/rejected."""

    claims = _claims_by_id(interpretation)
    relationships = _relationships_by_id(interpretation)
    valid_evidence_ids = _valid_evidence_ids(evidence_count)

    observations: list[FinancialObservation] = []
    outcome = SemanticValidationOutcome()
    obs_idx = 1
    consumed_claim_ids: set[str] = set()
    # (observation_population, currency, value) triples that an accepted
    # calculation has already ACCOUNTED FOR -- every monetary component it
    # consumed plus the derived result it produced. A later standalone
    # monetary claim whose value matches one of these is a restatement of
    # money already in the calculation graph, not a new exposure, and is
    # withheld from independent aggregation (spec: a component of a
    # calculation must not automatically become an additional exposure;
    # a derived result must not be re-aggregated).
    accounted_amounts: list[tuple[str, str, float]] = []

    validated_factor = _validate_cost_factor(interpretation, claims)
    # The LLM's own top-level relevance judgment gates the factor: if it
    # concluded there is no financial mechanism at all, a stray grounded
    # factor selection is not trusted.
    if interpretation.financial_relevance == "NONE":
        validated_factor = None
    outcome.validated_cost_factor = validated_factor.value if validated_factor else None

    # Precompute conflict pairs from relationships the LLM flagged as a
    # conflict (competing/incompatible values for one fact). `is_conflict`
    # is the ONLY structural signal the validator reads off a relationship
    # -- `relationship_type` is free-text metadata, never an operation
    # licence.
    conflict_pairs: set[frozenset[str]] = set()
    for rel in interpretation.relationships:
        if rel.is_conflict:
            conflict_pairs.add(frozenset({rel.source_claim, rel.target_claim}))

    def reject(calc_id: str, code: str, detail: str) -> None:
        outcome.rejected.append(RejectedCalculation(calculation_id=calc_id, reason_code=code, detail=detail))  # type: ignore[arg-type]

    for calc in interpretation.calculation_proposals:
        # A. Claim existence
        input_claims: list[SemanticClaim] = []
        missing = False
        for cid in calc.inputs:
            c = claims.get(cid)
            if c is None:
                reject(calc.calculation_id, "UNKNOWN_CLAIM", f"Input claim '{cid}' does not exist.")
                missing = True
                break
            input_claims.append(c)
        if missing:
            continue
        if not input_claims:
            reject(calc.calculation_id, "UNKNOWN_CLAIM", "No input claims specified.")
            continue

        # B. Provenance -- every source evidence ID must be real.
        bad_provenance = False
        for c in input_claims:
            if not c.source_evidence_ids:
                reject(calc.calculation_id, "MISSING_PROVENANCE", f"Claim '{c.claim_id}' cites no source evidence.")
                bad_provenance = True
                break
            for eid in c.source_evidence_ids:
                if eid not in valid_evidence_ids:
                    reject(calc.calculation_id, "MISSING_PROVENANCE", f"Claim '{c.claim_id}' cites nonexistent evidence '{eid}'.")
                    bad_provenance = True
                    break
            if bad_provenance:
                break
        if bad_provenance:
            continue

        # C. Evidence status -- CONTRADICTED claims never participate.
        if any(c.evidence_status == "CONTRADICTED" for c in input_claims):
            reject(calc.calculation_id, "EVIDENCE_STATUS_INELIGIBLE", "A contradicted claim cannot participate in calculation.")
            continue

        # I / J. Relationship support, confidence, conflicts.
        #
        # A calculation must still cite a real, valid relationship that
        # actually connects (some of) its own input claims -- this is
        # what prevents the LLM from combining two numbers merely because
        # they are numerically compatible or nearby in the text, per the
        # architecture's core safety requirement. What this validator no
        # longer requires is that relationship's TYPE to exactly match a
        # fixed per-operation lookup table (previously
        # _OPERATION_REQUIRES_RELATIONSHIP): the 6-member RelationshipType
        # taxonomy cannot cleanly label every real financial relationship
        # (e.g. "3 repeated identical events, each valued at X" is not
        # cleanly a "rate", but RATE_APPLIES_TO_QUANTITY was the only
        # available label), and rejecting a semantically sound,
        # well-grounded calculation merely because the LLM chose a
        # different-but-still-cited relationship label was exactly the
        # kind of deterministic second-guessing of the LLM's semantic
        # judgment this architecture must not do. The relationship's
        # EXISTENCE, its connection to these specific claims, and its
        # confidence are what get validated -- not whether its type name
        # matches a hardcoded expectation for this operation.
        claim_ids_in_calc = {c.claim_id for c in input_claims}
        rels = [relationships[rid] for rid in calc.relationship_ids if rid in relationships]
        if not rels:
            reject(calc.calculation_id, "UNSUPPORTED_RELATIONSHIP", "No valid relationship cited for this calculation.")
            continue
        if not any(
            {r.source_claim, r.target_claim} & claim_ids_in_calc
            for r in rels
        ):
            reject(calc.calculation_id, "UNSUPPORTED_RELATIONSHIP", "The cited relationship(s) do not connect any of this calculation's input claims.")
            continue
        if any(r.confidence == "LOW" for r in rels):
            reject(calc.calculation_id, "AMBIGUOUS_RELATIONSHIP", "A cited relationship has LOW confidence.")
            continue
        # Relationship evidence_basis must itself be real evidence.
        rel_evidence_bad = False
        for r in rels:
            for eid in r.evidence_basis:
                if eid not in valid_evidence_ids:
                    reject(calc.calculation_id, "MISSING_PROVENANCE", f"Relationship '{r.relationship_id}' cites nonexistent evidence '{eid}'.")
                    rel_evidence_bad = True
                    break
            if rel_evidence_bad:
                break
        if rel_evidence_bad:
            continue

        if any(pair.issubset(claim_ids_in_calc) for pair in conflict_pairs):
            reject(calc.calculation_id, "CONFLICTING_CLAIMS", "Two conflicting claims cannot be combined into one calculation.")
            continue
        # A claim that has an unresolved CONFLICTS_WITH relationship with
        # ANY other claim (not just one also cited in this calculation)
        # is contested data -- e.g. two competing rate claims for the
        # same quantity, where the LLM proposes a calculation using only
        # one of them. Using it here would be arbitrarily picking a side
        # in an unresolved conflict rather than the evidence establishing
        # which figure is correct.
        _conflict_members = {m for pair in conflict_pairs for m in pair}
        if claim_ids_in_calc & _conflict_members:
            reject(calc.calculation_id, "CONFLICTING_CLAIMS", "An input claim has an unresolved conflict with another claim; refusing to arbitrarily select one.")
            continue

        # D. Population.
        #  * MULTIPLY / ANNUALIZE: every input must describe ONE population
        #    -- a rate only applies to a quantity of the SAME population,
        #    and a historical count must never borrow a current rate.
        #  * SUBTRACT / SUM / DIVIDE: a RECOVERY / REMEDIATION / PREVENTION
        #    claim is DEFINITIONALLY a different population from the gross
        #    it nets against; that cross-population pairing is the entire
        #    point of the operation and must not be rejected as a
        #    "mismatch". What stays fatal for every operation is mixing
        #    CURRENT_FINDING with HISTORICAL (different time periods are
        #    never silently combined).
        populations = {c.population for c in input_claims}
        _NETTING_POPS = {"RECOVERY", "REMEDIATION", "PREVENTION"}
        _base_pops = {p for p in populations if p not in _NETTING_POPS}
        if calc.operation in ("MULTIPLY", "ANNUALIZE"):
            if len(populations) > 1:
                reject(calc.calculation_id, "POPULATION_MISMATCH", f"Claims span multiple populations: {sorted(populations)}.")
                continue
        else:
            if {"CURRENT_FINDING", "HISTORICAL"}.issubset(_base_pops) or len(_base_pops) > 1:
                reject(calc.calculation_id, "POPULATION_MISMATCH", f"Claims span multiple non-netting populations: {sorted(_base_pops)}.")
                continue

        # Rate/quantity ROLE is derived PRIMARILY from each claim's own
        # `fact_type` -- an unambiguous, per-claim signal the LLM assigns
        # directly ("this claim IS a RATE" / "this claim IS a QUANTITY"),
        # not subject to any positional convention. The relationship's
        # source_claim/target_claim ordering for RATE_APPLIES_TO_QUANTITY
        # is comparatively ambiguous: "rate applies to quantity" is
        # naturally read as source=rate/target=quantity, but a real model
        # sometimes emits it the other way around (source=quantity,
        # target=rate) since the schema itself never pins down which
        # slot means what -- trusting position over fact_type here
        # previously caused a real defect where a correctly fact_type-
        # tagged quantity/rate pair got silently swapped (event_count and
        # unit_amount reversed), a bug invisible in MULTIPLY's total
        # (commutative) but wrong in every other respect (the rendered
        # "N events x rate" breakdown, and any non-commutative operation).
        # The relationship's source/target position is used only as a
        # FALLBACK, for the case where fact_type alone can't disambiguate
        # (e.g. both claims are tagged AMOUNT).
        rate_claim = next((c for c in input_claims if c.fact_type == "RATE"), None)
        qty_claim = next((c for c in input_claims if c.fact_type == "QUANTITY"), None)
        # A claim tagged QUANTITY is just as unambiguous as one tagged
        # RATE -- if the LLM tagged exactly one input claim QUANTITY but
        # gave the per-unit-value claim some other fact_type (e.g. AMOUNT,
        # a legitimate choice for "INR 50,000 each" when it isn't phrased
        # with an explicit "per X" denominator), the QUANTITY tag alone
        # still identifies which of the two MULTIPLY inputs plays which
        # role -- by elimination, whichever OTHER claim is being
        # multiplied against it is necessarily the rate/multiplier,
        # regardless of what the LLM happened to label it or which slot
        # of the relationship it occupied. This still comes before the
        # positional fallback below for the same reason as the RATE/
        # QUANTITY case above: relationship source/target order is not a
        # reliable signal, and trusting it here previously reproduced the
        # exact same swapped event_count/unit_amount defect for an
        # AMOUNT+QUANTITY pair.
        if (rate_claim is None or qty_claim is None) and len(input_claims) == 2:
            a, b = input_claims
            # "Multiplier-ish" fact types (a per-X rate or a percentage)
            # unambiguously identify their own role; the OTHER input is the
            # base/quantity it applies to, by elimination -- a per-claim
            # fact_type signal, never the relationship's slot ordering.
            _MULTIPLIER = ("RATE", "PERCENTAGE")
            if a.fact_type in _MULTIPLIER and b.fact_type not in _MULTIPLIER:
                rate_claim, qty_claim = a, b
            elif b.fact_type in _MULTIPLIER and a.fact_type not in _MULTIPLIER:
                rate_claim, qty_claim = b, a
            elif a.fact_type == "QUANTITY" and b.fact_type != "QUANTITY":
                qty_claim, rate_claim = a, b
            elif b.fact_type == "QUANTITY" and a.fact_type != "QUANTITY":
                qty_claim, rate_claim = b, a
        # NOTE: roles are derived ONLY from each claim's own `fact_type`
        # (an unambiguous per-claim signal the LLM assigns) plus the
        # 2-input by-elimination rule above. There is deliberately NO
        # fallback to the relationship's source/target position or its
        # type label: `relationship_type` is free descriptive text and
        # carries no reliable role ordering, and trusting position here
        # previously caused silent event_count/unit_amount swaps.
        amount_claim = next((c for c in input_claims if c.fact_type in ("AMOUNT", "RECOVERY", "REMEDIATION_COST", "PREVENTION_COST")), None)
        period_claim = next((c for c in input_claims if c.fact_type == "OBSERVATION_PERIOD"), None)

        # E. Units -- pairwise compatibility for the RATE/QUANTITY pair
        # only (by ROLE, not raw fact_type -- see above). An
        # OBSERVATION_PERIOD claim (e.g. "12 months") describes a
        # completely separate dimension -- a time SPAN, not an occurrence
        # unit -- and must never be compared against the rate/quantity's
        # own unit; it legitimately differs (MONTH vs OCCURRENCE) without
        # that being any kind of incompatibility.
        if calc.operation in ("MULTIPLY", "ANNUALIZE"):
            _dimensional_claims = [c for c in (rate_claim, qty_claim) if c is not None]
            units = [u for u in (_dimensional_unit(c) for c in _dimensional_claims) if u]
            if len(units) >= 2 and not all(_units_compatible(units[0], u) for u in units[1:]):
                reject(calc.calculation_id, "INCOMPATIBLE_UNITS", f"Incompatible units: {units}.")
                continue

        # F. Currency -- must match across claims that carry one.
        currencies = {c.currency for c in input_claims if c.currency}
        if len(currencies) > 1:
            reject(calc.calculation_id, "INCOMPATIBLE_CURRENCY", f"Claims span multiple currencies: {sorted(currencies)}.")
            continue

        # K. Numerical validity.
        if any(c.value is None or c.value <= 0 for c in input_claims if c.fact_type != "OBSERVATION_PERIOD"):
            reject(calc.calculation_id, "INVALID_NUMBER", "A required claim value is missing, zero, or non-finite.")
            continue

        currency = next(iter(currencies), None)
        population = next(iter(populations))
        linked_status = _weakest_status(input_claims)
        source_ids = sorted({eid for c in input_claims for eid in c.source_evidence_ids if eid in valid_evidence_ids})

        # A MULTIPLY / ANNUALIZE over exactly two claims where neither the
        # fact_type tags nor the by-elimination rule identify a rate/
        # multiplier and a quantity (e.g. two bare AMOUNT claims) is
        # genuinely ambiguous -- there is no defensible way to know which
        # is the multiplicand. Reject rather than silently materialize just
        # one of them and drop the other.
        if (
            calc.operation in ("MULTIPLY", "ANNUALIZE")
            and len(input_claims) == 2
            and not (rate_claim and qty_claim)
        ):
            reject(
                calc.calculation_id,
                "AMBIGUOUS_RELATIONSHIP",
                "A MULTIPLY needs one claim identifiable as a rate/percentage and one as a quantity; "
                "neither the fact_type tags nor elimination establish those roles here.",
            )
            continue

        # Materialize -- reuse the EXISTING FinancialObservation shape so
        # the untouched calculator.py functions perform the real
        # arithmetic. Never compute a total here.

        if currency is None and (
            (calc.operation in ("MULTIPLY", "ANNUALIZE") and rate_claim and qty_claim)
            or amount_claim is not None
        ):
            # This materialization would need a monetary value, but no
            # participating claim stated a currency -- refuse rather than
            # fabricate one (spec: never silently default to INR).
            reject(calc.calculation_id, "INCOMPATIBLE_CURRENCY", "No claim states a currency; refusing to assume one.")
            continue

        if calc.operation in ("MULTIPLY", "ANNUALIZE") and rate_claim and qty_claim:
            observations.append(FinancialObservation(
                observation_id=f"SEM-OBS-{obs_idx:03d}",
                unit_amount=_rate_value_as_multiplier(rate_claim),
                event_count=int(qty_claim.value) if qty_claim.value else None,
                currency=currency,
                # A validated (grounded, sufficiently confident) LLM cost
                # factor is used when available; otherwise the honest
                # answer is that the specific category could not be
                # confirmed -- NOT_ESTABLISHED, never a fabricated
                # DIRECT_LOSS guess. The calculation itself still executes
                # and still contributes to gross exposure (see
                # calculator.py's verified_loss_obs, which explicitly
                # includes NOT_ESTABLISHED for exactly this reason).
                amount_type=validated_factor or FinancialAmountType.NOT_ESTABLISHED,
                observation_period_months=period_claim.value if period_claim else None,
                source_evidence_ids=source_ids,
                verification_status=linked_status,  # type: ignore[arg-type]
                financial_population=_to_observation_population(population),  # type: ignore[arg-type]
                is_derived=True,
                notes=f"Derived from semantic relationship(s): {', '.join(r.relationship_id for r in rels)}.",
            ))
            obs_idx += 1
            # Record what this derived result accounts for: the per-event /
            # rate value it CONSUMED as a component, and the product it
            # PRODUCED. Neither may be independently re-materialized from a
            # separate claim that merely restates the same figure.
            _derived_pop = _to_observation_population(population)
            _rv = _rate_value_as_multiplier(rate_claim)
            _qv = qty_claim.value
            if currency is not None and _rv is not None and _qv:
                accounted_amounts.append((_derived_pop, currency, float(_rv)))
                accounted_amounts.append((_derived_pop, currency, float(_rv) * float(_qv)))
        else:
            # Non-multiplicative operation (SUBTRACT / SUM / DIVIDE), or a
            # MULTIPLY that is really just a flat amount. Materialize EVERY
            # monetary input claim as its OWN observation, each keeping its
            # own population, evidence status, and semantic role. The
            # deterministic executor (calculate_confirmed_impact) then
            # combines them exactly -- e.g. verified_gross - verified_
            # recovery -- rather than this validator computing anything.
            # Materializing each claim independently is what prevents a
            # recovery / remediation input from being silently consumed
            # and dropped just because it shared a proposal with a gross
            # amount (spec: compositional, never lose a valid financial
            # population).
            monetary = [
                c
                for c in input_claims
                if c.fact_type in ("AMOUNT", "RECOVERY", "REMEDIATION_COST", "PREVENTION_COST")
                and c.value is not None
                and c.value > 0
                and c.currency is not None
            ]
            if not monetary:
                reject(
                    calc.calculation_id,
                    "UNSUPPORTED_OPERATION",
                    f"No monetary claim with a stated currency to materialize for operation '{calc.operation}'.",
                )
                continue
            for c in monetary:
                observations.append(FinancialObservation(
                    observation_id=f"SEM-OBS-{obs_idx:03d}",
                    amount=c.value,
                    currency=c.currency,
                    amount_type=_amount_type_for(c, validated_factor),
                    source_evidence_ids=[eid for eid in c.source_evidence_ids if eid in valid_evidence_ids],
                    verification_status=c.evidence_status,  # type: ignore[arg-type]
                    financial_population=_to_observation_population(c.population),  # type: ignore[arg-type]
                    notes=f"Semantic calculation {calc.calculation_id} ({calc.operation}); "
                    f"relationship(s): {', '.join(r.relationship_id for r in rels)}.",
                ))
                obs_idx += 1
                if c.currency is not None and c.value is not None:
                    accounted_amounts.append(
                        (_to_observation_population(c.population), c.currency, float(c.value))
                    )

        outcome.accepted_calculation_ids.append(calc.calculation_id)
        for c in input_claims:
            consumed_claim_ids.add(c.claim_id)
        if calc.proposed_result_value is not None:
            outcome.llm_disagreements.append(
                f"{calc.calculation_id}: LLM proposed {calc.proposed_result_value} -- "
                "authoritative value is computed independently by the deterministic calculator, not this figure."
            )

        # Auditable trace. `executor_result` is filled in later by
        # semantic_engine.py once the deterministic calculator has run over
        # the materialized observation (matched by observation_id).
        _roles: dict[str, str] = {}
        for c in input_claims:
            if c is rate_claim:
                _roles[c.claim_id] = "rate/multiplier"
            elif c is qty_claim:
                _roles[c.claim_id] = "quantity"
            elif c is amount_claim:
                _roles[c.claim_id] = f"amount ({c.fact_type.lower()})"
            elif c is period_claim:
                _roles[c.claim_id] = "observation period"
            else:
                _roles[c.claim_id] = c.fact_type.lower()
        if rate_claim and qty_claim:
            _formula = f"{qty_claim.value:g} {(qty_claim.unit or 'unit')} x {rate_claim.value:g} {(rate_claim.unit or '')}".strip()
        elif amount_claim:
            _formula = f"{amount_claim.value:g} {currency or ''}".strip()
        else:
            _formula = calc.operation
        outcome.traces.append(CalculationTrace(
            calculation_id=calc.calculation_id,
            cost_factor=(validated_factor.value if validated_factor else "NOT_ESTABLISHED"),
            input_claim_ids=[c.claim_id for c in input_claims],
            semantic_roles=_roles,
            relationship_ids=[r.relationship_id for r in rels],
            operation=calc.operation,
            currency=currency,
            evidence_status=linked_status,
            formula=_formula,
            llm_proposed_result=calc.proposed_result_value,
            executor_result=None,
            disagreement=None,
            observation_id=observations[-1].observation_id if observations else None,
            status={
                "VERIFIED": "VERIFIED_EXPOSURE",
                "REPORTED": "REPORTED_EXPOSURE",
            }.get(linked_status, "REQUIRES_VERIFICATION"),
        ))

    # Standalone self-sufficient facts (AMOUNT / RECOVERY / REMEDIATION_COST
    # / PREVENTION_COST) never need a relationship to mean something on
    # their own -- a stated recovery is a real observation whether or not
    # the LLM also proposed a SUBTRACT-against-gross calculation for it.
    # Materializing these unconditionally (subject only to the same
    # provenance/evidence-status/numeric-validity checks, since there is
    # no second claim to compare population/units/currency against) is
    # what prevents a recovery/remediation fact from silently disappearing
    # merely because the LLM didn't bundle it into a calculation proposal
    # -- the same failure mode the deterministic extractor path was fixed
    # for in an earlier hardening pass.
    for c in interpretation.claims:
        if c.claim_id in consumed_claim_ids:
            continue
        if c.fact_type not in ("AMOUNT", "RECOVERY", "REMEDIATION_COST", "PREVENTION_COST"):
            continue
        if c.evidence_status == "CONTRADICTED":
            continue
        if c.value is None or c.value <= 0:
            continue
        if not c.source_evidence_ids or any(eid not in valid_evidence_ids for eid in c.source_evidence_ids):
            continue
        if c.fact_type == "AMOUNT" and c.unit:
            # A flat AMOUNT carrying a per-X unit (e.g. "UNIT", "HOUR") is
            # structurally self-contradictory -- a lump sum has no
            # denominator. This is most plausibly a per-unit RATE the
            # interpreter mislabeled, not a genuine total; materializing
            # it unconsumed as a flat gross figure would silently drop the
            # multiplication a companion QUANTITY claim may have been
            # meant to combine with. Withhold rather than guess which it
            # is -- recorded for audit, never silently dropped.
            outcome.llm_disagreements.append(
                f"{c.claim_id}: AMOUNT claim carries a per-'{c.unit}' unit (a rate-shaped value); "
                "withheld as an unconsumed flat amount rather than materialized, since it was not "
                "linked to a quantity via a validated relationship."
            )
            continue
        if c.currency is None:
            # No stated currency -- refuse to fabricate one. Recorded (not
            # silently dropped) so the auditor can see this standalone fact
            # was withheld, not merely forgotten.
            outcome.llm_disagreements.append(
                f"{c.claim_id}: claim has no stated currency; withheld rather than defaulted to INR."
            )
            continue
        if c.fact_type == "AMOUNT" and _amount_already_accounted(
            accounted_amounts,
            _to_observation_population(c.population),
            c.currency,
            c.value,
        ):
            # This bare AMOUNT restates a monetary value already consumed
            # as a component of -- or produced as -- an accepted
            # calculation for the same population/currency. It is the same
            # money re-expressed, not an independent gross-exposure claim,
            # so it must NOT be materialized and summed again. RECOVERY /
            # REMEDIATION_COST / PREVENTION_COST are definitionally
            # separate financial roles and are never suppressed here.
            outcome.llm_disagreements.append(
                f"{c.claim_id}: this monetary value is already represented in an accepted "
                "calculation (consumed as a component or produced as its derived result); "
                "not aggregated again as an independent exposure."
            )
            continue
        observations.append(FinancialObservation(
            observation_id=f"SEM-OBS-{obs_idx:03d}",
            amount=c.value,
            currency=c.currency,
            amount_type=_amount_type_for(c, validated_factor),
            source_evidence_ids=list(c.source_evidence_ids),
            verification_status=c.evidence_status,  # type: ignore[arg-type]
            financial_population=_to_observation_population(c.population),  # type: ignore[arg-type]
        ))
        obs_idx += 1

    return observations, outcome
