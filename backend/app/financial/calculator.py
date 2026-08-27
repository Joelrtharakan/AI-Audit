"""Deterministic arithmetic and financial calculation logic.

Enforces:
  - Epistemic distinction: VERIFIED GROSS EXPOSURE != REPORTED FINANCIAL EXPOSURE != CONFIRMED NET LOSS.
  - Fail-closed arithmetic: Rejects NaN, Infinity, -Infinity, null, and empty operands.
  - Pure monotonic status propagation: Unverified/Reported inputs never produce Verified results.
  - Recovery safety: Never assume unverified recovery is 0; never compute net loss without verified recovery.
  - Potential additional events (e.g. 7 deliveries) tracked separately from verified events.
  - Annualization strictly requires verified exposure, verified count, and verified observation period > 0.
"""

from __future__ import annotations

import math
from app.financial.models import (
    AnnualizedExposure,
    CapaEconomicAnalysis,
    ConfirmedFinancialImpact,
    CostOfQualityBreakdown,
    DimensionalConfidence,
    FinancialAmountType,
    FinancialConfidenceLevel,
    FinancialEpistemicStatus,
    FinancialObservation,
    FinancialScenarioAnalysis,
    FinancialUncertainty,
    PotentialFinancialExposure,
    RecoveryStatus,
    RecurrenceAnalysis,
    ScenarioEstimate,
)


def _is_valid_positive_number(v: float | None) -> bool:
    if v is None:
        return False
    if not isinstance(v, (int, float)):
        return False
    if math.isnan(v) or math.isinf(v):
        return False
    return v > 0.0


def _dedupe_corroborating_observations(
    observations: list[FinancialObservation],
) -> list[FinancialObservation]:
    """Collapse multiple observations that state IDENTICAL financial
    values (same amount, unit amount, event count, period, currency) into
    one -- two independent evidence sources reporting the exact same
    numbers almost always corroborate the SAME underlying fact rather
    than describing two separate events, so summing them would double
    count. Genuinely differing values (a real conflict) are left intact
    for the caller's own conflict handling -- this only removes exact
    duplicates, never picks a "winner" between disagreeing values."""
    seen: set[tuple] = set()
    result: list[FinancialObservation] = []
    for o in observations:
        key = (o.amount, o.unit_amount, o.event_count, o.observation_period_months, o.currency, o.amount_type)
        if key in seen:
            continue
        seen.add(key)
        result.append(o)
    return result


def calculate_confirmed_impact(
    observations: list[FinancialObservation],
) -> ConfirmedFinancialImpact:
    """Calculate verified gross exposure, reported financial exposure, verified recovery, and confirmed net loss."""
    # CURRENT_FINDING-only: a fact drawn from evidence explicitly framed as
    # backward-looking historical context (a separate financial_population,
    # set by the extractor's _HISTORICAL_MARKER_RE) must never contribute
    # to the CURRENT finding's gross exposure, recovery, or net loss --
    # historical facts feed recurrence/annualization analysis instead
    # (calculate_recurrence_exposure / calculate_annualized_exposure),
    # which intentionally consider the full observation set.
    # Deduplicated before use (see _dedupe_corroborating_observations):
    # two evidence items stating IDENTICAL financial values corroborate
    # the same underlying fact, not two separate ones.
    verified_loss_obs = _dedupe_corroborating_observations([
        o for o in observations
        if o.verification_status == "VERIFIED"
        and o.financial_population == "CURRENT_FINDING"
        and o.amount_type in (
            FinancialAmountType.DIRECT_LOSS,
            FinancialAmountType.OVERPAYMENT,
            FinancialAmountType.DUPLICATE_PAYMENT,
            FinancialAmountType.REWORK_COST,
            FinancialAmountType.SCRAP_COST,
            FinancialAmountType.DOWNTIME_COST,
            FinancialAmountType.CUSTOMER_COMPENSATION,
            FinancialAmountType.PENALTY,
            # A calculation can be semantically valid (quantity x rate,
            # both VERIFIED, CURRENT_FINDING) even when the LLM's cost-
            # factor grounding specifically failed (e.g. an ungrounded
            # supporting_claim_id) -- the calculation itself is not in
            # question, only its display label. NOT_ESTABLISHED here
            # means "a real exposure exists, its specific cost-factor
            # category could not be confirmed" and must still be
            # eligible for gross-exposure aggregation, never silently
            # relabeled DIRECT_LOSS just to make it eligible.
            FinancialAmountType.NOT_ESTABLISHED,
        )
    ])

    reported_loss_obs = _dedupe_corroborating_observations([
        o for o in observations
        if o.verification_status in ("REPORTED", "UNVERIFIED")
        and o.financial_population == "CURRENT_FINDING"
        and o.amount_type != FinancialAmountType.RECOVERY
    ])

    verified_recovery_obs = _dedupe_corroborating_observations([
        o for o in observations
        if o.verification_status == "VERIFIED"
        and o.financial_population == "CURRENT_FINDING"
        and o.amount_type == FinancialAmountType.RECOVERY
    ])

    # A recovery amount stated but not (yet) VERIFIED must never simply
    # vanish -- it is a distinct financial observation and must remain
    # visible, labeled by its own evidence status, even though it can
    # never participate in confirmed_net_loss (reserved for the
    # both-sides-VERIFIED case) or be presented as VERIFIED.
    reported_recovery_obs = _dedupe_corroborating_observations([
        o for o in observations
        if o.verification_status in ("REPORTED", "UNVERIFIED")
        and o.financial_population == "CURRENT_FINDING"
        and o.amount_type == FinancialAmountType.RECOVERY
    ])

    # Verified calculations
    total_gross = 0.0
    loss_basis_parts = []
    source_ids = []
    curr = "INR"
    seen_events: set[str] = set()
    verified_event_count = 0
    financial_factor = "NOT_ESTABLISHED"

    for obs in verified_loss_obs:
        ev_key = obs.event_id or f"{obs.amount}_{obs.unit_amount}_{obs.quantity}_{obs.source_reference}"
        if ev_key in seen_events and obs.event_id:
            continue
        seen_events.add(ev_key)

        curr = obs.currency or curr
        source_ids.extend(obs.source_evidence_ids)
        cnt = obs.event_count or obs.quantity or 1
        verified_event_count += int(cnt)
        if financial_factor == "NOT_ESTABLISHED":
            financial_factor = obs.amount_type.value

        if _is_valid_positive_number(obs.amount):
            total_gross += obs.amount  # type: ignore[operator]
            loss_basis_parts.append(f"{obs.currency} {obs.amount:,.2f} ({obs.amount_type.value})")
        elif _is_valid_positive_number(obs.unit_amount):
            calc_val = obs.unit_amount * cnt  # type: ignore[operator]
            total_gross += calc_val
            loss_basis_parts.append(f"{cnt:g} verified event(s) × {obs.currency} {obs.unit_amount:,.2f}")

    # Reported calculations (fail-closed if not verified)
    total_reported = 0.0
    reported_unit_amt = None
    reported_event_count = None
    pot_additional_events = None

    for obs in reported_loss_obs:
        curr = obs.currency or curr
        source_ids.extend(obs.source_evidence_ids)
        if obs.event_count:
            reported_event_count = obs.event_count
        if obs.potential_event_count:
            pot_additional_events = obs.potential_event_count
        if _is_valid_positive_number(obs.amount):
            total_reported += obs.amount  # type: ignore[operator]
        elif _is_valid_positive_number(obs.unit_amount):
            reported_unit_amt = obs.unit_amount
            cnt = obs.event_count or 1
            total_reported += obs.unit_amount * cnt  # type: ignore[operator]

    # Process reported-but-not-verified recovery -- never fabricated as
    # VERIFIED, never summed into confirmed_net_loss, but never silently
    # dropped either.
    reported_recovery_val = None
    for obs in reported_recovery_obs:
        source_ids.extend(obs.source_evidence_ids)
        rec_val = obs.amount if obs.amount is not None else obs.recovery_amount
        if _is_valid_positive_number(rec_val):
            reported_recovery_val = (reported_recovery_val or 0.0) + rec_val  # type: ignore[operator]
            # A finding whose ONLY observation is a recovery (no gross-loss
            # observation to have already set `curr`) must still report its
            # OWN real currency -- never silently default to INR merely
            # because the loss-side loops above never ran.
            curr = obs.currency or curr

    # Process verified recovery
    has_verified_recovery = False
    verified_recovery_val = None
    recovery_status = RecoveryStatus.REQUIRES_VERIFICATION

    for obs in verified_recovery_obs:
        source_ids.extend(obs.source_evidence_ids)
        rec_val = obs.amount if obs.amount is not None else obs.recovery_amount
        if _is_valid_positive_number(rec_val) or rec_val == 0.0:
            has_verified_recovery = True
            verified_recovery_val = (verified_recovery_val or 0.0) + rec_val
            curr = obs.currency or curr

    for obs in verified_loss_obs:
        if obs.recovery_status == RecoveryStatus.VERIFIED_ZERO_RECOVERY:
            has_verified_recovery = True
            verified_recovery_val = 0.0
            recovery_status = RecoveryStatus.VERIFIED_ZERO_RECOVERY
        elif _is_valid_positive_number(obs.recovery_amount) and obs.verification_status == "VERIFIED" and obs not in verified_recovery_obs:
            has_verified_recovery = True
            verified_recovery_val = (verified_recovery_val or 0.0) + obs.recovery_amount

    if has_verified_recovery:
        if verified_recovery_val == 0.0:
            recovery_status = RecoveryStatus.VERIFIED_ZERO_RECOVERY
        elif total_gross > 0 and verified_recovery_val is not None and verified_recovery_val >= total_gross:
            recovery_status = RecoveryStatus.FULLY_RECOVERED
        else:
            recovery_status = RecoveryStatus.PARTIALLY_RECOVERED

    # Net loss calculation rule: ONLY when BOTH gross and recovery are VERIFIED
    confirmed_net_loss = None
    potential_unrecovered = None
    is_confirmed_loss = False

    if total_gross > 0.0:
        if has_verified_recovery and verified_recovery_val is not None:
            confirmed_net_loss = round(max(0.0, total_gross - verified_recovery_val), 2)
            is_confirmed_loss = True
        else:
            potential_unrecovered = round(total_gross, 2)

    is_confirmed_event = total_gross > 0.0 or has_verified_recovery
    has_reported = (total_reported > 0.0 or reported_unit_amt is not None) and total_gross == 0.0

    gross_basis = "; ".join(loss_basis_parts) if loss_basis_parts else ""

    calc_formula = ""
    basis_str = ""
    if is_confirmed_loss and verified_recovery_val is not None:
        gross_part = f"({gross_basis}) = {curr} {total_gross:,.2f}" if gross_basis else f"{curr} {total_gross:,.2f}"
        calc_formula = f"{gross_part} gross exposure − {curr} {verified_recovery_val:,.2f} verified recovery = {curr} {confirmed_net_loss:,.2f} confirmed net loss"
        basis_str = calc_formula
    elif total_gross > 0.0 and potential_unrecovered is not None:
        gross_part = f"{gross_basis} = {curr} {total_gross:,.2f}" if gross_basis else f"{curr} {total_gross:,.2f}"
        calc_formula = f"Verified Gross Exposure = {gross_part}; Recovery = NOT ESTABLISHED; Potential Unrecovered Exposure = UP TO {curr} {potential_unrecovered:,.2f}"
        basis_str = f"Verified gross exposure of {curr} {total_gross:,.2f} observed ({gross_basis}). Net loss requires recovery verification." if gross_basis else f"Verified gross exposure of {curr} {total_gross:,.2f} observed. Net loss requires recovery verification."
    elif has_reported:
        if reported_unit_amt and reported_event_count:
            calc_formula = f"{reported_event_count} reported event(s) × {curr} {reported_unit_amt:,.2f}/event = {curr} {total_reported:,.2f} reported exposure (UNVERIFIED)"
            basis_str = f"Evidence reports {reported_event_count} events at approximately {curr} {reported_unit_amt:,.2f} per event (unverified)."
        else:
            calc_formula = f"Reported financial exposure = {curr} {total_reported:,.2f} (UNVERIFIED)"
            basis_str = f"Reported exposure of {curr} {total_reported:,.2f} requires independent verification."

    return ConfirmedFinancialImpact(
        verified_gross_exposure=round(total_gross, 2) if total_gross > 0 else None,
        reported_financial_exposure=round(total_reported, 2) if total_reported > 0 else None,
        reported_unit_exposure=round(reported_unit_amt, 2) if reported_unit_amt is not None else None,
        reported_event_count=reported_event_count,
        verified_event_count=verified_event_count if verified_event_count > 0 else None,
        potential_additional_events=pot_additional_events,
        potential_additional_exposure=None,  # Unverified additional exposure remains NOT ESTABLISHED
        verified_recovery=round(verified_recovery_val, 2) if (has_verified_recovery and verified_recovery_val is not None) else None,
        reported_recovery=round(reported_recovery_val, 2) if reported_recovery_val is not None else None,
        confirmed_net_loss=confirmed_net_loss,
        potential_unrecovered_exposure=potential_unrecovered,
        recovery_status=recovery_status,
        financial_factor=financial_factor,
        currency=curr,
        observed_events_count=verified_event_count,
        calculation_formula=calc_formula,
        basis=basis_str,
        source_evidence_ids=list(dict.fromkeys(source_ids)),
        is_confirmed_event=is_confirmed_event,
        is_confirmed_loss=is_confirmed_loss,
        has_reported_exposure=has_reported,
    )


def calculate_potential_exposure(
    observations: list[FinancialObservation],
) -> PotentialFinancialExposure:
    """Calculate potential exposure ranges from unverified or population-expanded facts."""
    potential_obs = [
        o for o in observations
        if o.financial_population == "CURRENT_FINDING"
        and (
            o.amount_type == FinancialAmountType.POTENTIAL_EXPOSURE
            or o.verification_status in ("REPORTED", "UNVERIFIED")
            or (o.affected_population and o.affected_population > (o.event_count or 1))
        )
    ]

    if not potential_obs:
        return PotentialFinancialExposure(is_present=False)

    lower_total = 0.0
    upper_total = 0.0
    basis_parts = []
    source_ids = []
    curr = "INR"
    unverified_count = 0

    for obs in potential_obs:
        curr = obs.currency or curr
        source_ids.extend(obs.source_evidence_ids)
        if _is_valid_positive_number(obs.amount_min) and _is_valid_positive_number(obs.amount_max):
            lower_total += obs.amount_min  # type: ignore[operator]
            upper_total += obs.amount_max  # type: ignore[operator]
            basis_parts.append(f"{obs.currency} {obs.amount_min:,.2f}–{obs.amount_max:,.2f} range ({obs.verification_status})")
        elif _is_valid_positive_number(obs.amount):
            lower_total += obs.amount  # type: ignore[operator]
            upper_total += obs.amount  # type: ignore[operator]
            basis_parts.append(f"{obs.currency} {obs.amount:,.2f} ({obs.verification_status})")
        elif _is_valid_positive_number(obs.unit_amount):
            pop = obs.affected_population or obs.potential_event_count or obs.event_count or 1
            unit = obs.unit_amount
            lower_total += unit * pop  # type: ignore[operator]
            upper_total += unit * pop  # type: ignore[operator]
            basis_parts.append(f"{pop} potentially affected unit(s) × {obs.currency} {unit:,.2f}")
            unverified_count += int(pop)

    is_present = upper_total > 0.0
    basis_str = "; ".join(basis_parts) if basis_parts else "Potential exposure identified from unverified/reported evidence."

    return PotentialFinancialExposure(
        lower_bound=round(lower_total, 2) if lower_total > 0 else None,
        upper_bound=round(upper_total, 2) if upper_total > 0 else None,
        currency=curr,
        unverified_event_count=unverified_count if unverified_count > 0 else None,
        basis=basis_str,
        source_evidence_ids=list(dict.fromkeys(source_ids)),
        is_present=is_present,
    )


def calculate_annualized_exposure(
    observations: list[FinancialObservation],
    observation_period_months: float | None = None,
) -> AnnualizedExposure:
    """Calculate annualized exposure strictly requiring VERIFIED exposure, count, and period > 0."""
    # Historical annualization extrapolates a HISTORICAL recurrence rate --
    # when the evidence contains observations explicitly framed as
    # historical context, the current finding's own one-off fact (if any)
    # must not be smeared into that rate as if it recurred at the same
    # frequency. Falls back to using every observation when nothing is
    # historically framed, preserving prior behavior for a finding whose
    # own persisting condition spans a stated period.
    if any(o.financial_population == "HISTORICAL" for o in observations):
        observations = _dedupe_corroborating_observations(
            [o for o in observations if o.financial_population == "HISTORICAL"]
        )

    period = observation_period_months
    verified_events = 0
    has_unverified_events = False

    for obs in observations:
        if obs.observation_period_months and _is_valid_positive_number(obs.observation_period_months):
            if obs.verification_status == "VERIFIED" or period is None:
                period = obs.observation_period_months
        if obs.verification_status == "VERIFIED" and (obs.event_count or obs.quantity):
            verified_events += int(obs.event_count or obs.quantity or 1)
        elif obs.verification_status in ("REPORTED", "UNVERIFIED") or obs.potential_event_count:
            has_unverified_events = True

    # Annualization rule: Fails closed if period is missing/0, or if inputs are unverified
    if not period or not _is_valid_positive_number(period):
        reason = "A verified observation period and verified historical financial event dataset are not available."
        if has_unverified_events:
            reason += " Potentially affected events are excluded from annualization because their status is UNVERIFIED."
        return AnnualizedExposure(
            is_assessable=False,
            reason_if_not_assessable=reason,
            basis="Observation period is not specified or verified in evidence.",
        )

    verified_obs = [
        o for o in observations
        if o.verification_status == "VERIFIED"
        and (_is_valid_positive_number(o.amount) or _is_valid_positive_number(o.unit_amount))
    ]
    if not verified_obs:
        reason = "A verified observation period and verified historical financial event dataset are not available."
        if has_unverified_events:
            reason += " Potentially affected events are excluded from annualization because their status is UNVERIFIED."
        return AnnualizedExposure(
            is_assessable=False,
            reason_if_not_assessable=reason,
            basis="No verified financial exposure observed during the period.",
        )

    observed_exposure = 0.0
    curr = "INR"

    for o in verified_obs:
        curr = o.currency or curr
        if _is_valid_positive_number(o.amount):
            observed_exposure += o.amount  # type: ignore[operator]
        elif _is_valid_positive_number(o.unit_amount):
            cnt = o.quantity or o.event_count or 1.0
            observed_exposure += o.unit_amount * cnt  # type: ignore[operator]

    if observed_exposure <= 0:
        return AnnualizedExposure(
            is_assessable=False,
            reason_if_not_assessable="Observed exposure is zero.",
            basis="Observed exposure is zero.",
        )

    multiplier = 12.0 / period
    annualized = round(observed_exposure * multiplier, 2)
    rate_per_year = round((verified_events / period) * 12.0, 1) if verified_events > 0 else None

    if verified_events > 0:
        formula_str = f"{verified_events} verified event(s) / {period:g} months × 12 = {rate_per_year:g} events/year; {curr} {observed_exposure:,.2f} × 12 / {period:g} = {curr} {annualized:,.2f}/year"
    else:
        formula_str = f"{curr} {observed_exposure:,.2f} observed exposure × 12 / {period:g} months = {curr} {annualized:,.2f}/year"
    
    basis_str = f"Annualized from the verified historical rate of {curr} {observed_exposure:,.2f} over {period:g} month(s). This is not a confirmed future loss."

    return AnnualizedExposure(
        annualized_amount=annualized,
        currency=curr,
        observation_period_months=period,
        observed_exposure=round(observed_exposure, 2),
        observed_event_rate_per_year=rate_per_year,
        calculation_formula=formula_str,
        basis=basis_str,
        projection_type="ANNUALIZED_OBSERVED_EXPOSURE",
        is_assessable=True,
    )


def calculate_recurrence_exposure(
    observations: list[FinancialObservation],
    historical_frequency_per_year: float | None = None,
    frequency_range: tuple[float, float] | None = None,
) -> RecurrenceAnalysis:
    """Calculate recurrence-based expected annual loss: Expected Events/Year * Verified Avg Loss/Event."""
    # Same population-scoping rationale as calculate_annualized_exposure:
    # the average cost per event used for recurrence forecasting must come
    # from HISTORICAL-framed facts when they exist, not be diluted by the
    # current finding's own one-off amount.
    if any(o.financial_population == "HISTORICAL" for o in observations):
        observations = _dedupe_corroborating_observations(
            [o for o in observations if o.financial_population == "HISTORICAL"]
        )

    freq = historical_frequency_per_year if _is_valid_positive_number(historical_frequency_per_year) else None
    freq_min = frequency_range[0] if (frequency_range and _is_valid_positive_number(frequency_range[0])) else None
    freq_max = frequency_range[1] if (frequency_range and _is_valid_positive_number(frequency_range[1])) else None

    if not freq and not (freq_min and freq_max):
        return RecurrenceAnalysis(
            is_assessable=False,
            reason_if_not_assessable="Insufficient verified historical recurrence data to establish a recurrence rate.",
            basis="No verified historical recurrence frequency available.",
        )

    verified_losses = [
        o for o in observations
        if o.verification_status == "VERIFIED"
        and (_is_valid_positive_number(o.amount) or _is_valid_positive_number(o.unit_amount))
    ]
    if not verified_losses:
        return RecurrenceAnalysis(
            is_assessable=False,
            reason_if_not_assessable="No verified event loss data available to calculate average cost per event.",
            basis="No verified event loss data available to calculate average cost per event.",
        )

    total_loss = 0.0
    total_events = 0
    curr = "INR"
    for o in verified_losses:
        curr = o.currency or curr
        cnt = o.event_count or o.quantity or 1
        val = o.amount if o.amount is not None else (o.unit_amount * cnt if o.unit_amount else 0.0)
        total_loss += val
        total_events += int(cnt)

    avg_loss = total_loss / max(1, total_events)

    if freq_min and freq_max:
        exp_min = round(freq_min * avg_loss, 2)
        exp_max = round(freq_max * avg_loss, 2)
        exp_annual = round(((freq_min + freq_max) / 2.0) * avg_loss, 2)
        calc_formula = f"Recurrence rate ({freq_min:g}–{freq_max:g} events/year) × {curr} {avg_loss:,.2f}/event = {curr} {exp_min:,.2f}–{exp_max:,.2f}/year"
        basis_str = f"Historical recurrence of {freq_min:g}–{freq_max:g} events/year at verified average {curr} {avg_loss:,.2f}/event."
    else:
        exp_min = None
        exp_max = None
        exp_annual = round((freq or 0.0) * avg_loss, 2)
        calc_formula = f"{freq:g} verified event(s)/year × {curr} {avg_loss:,.2f} verified average loss/event = {curr} {exp_annual:,.2f} expected annual exposure"
        basis_str = calc_formula

    return RecurrenceAnalysis(
        historical_events_per_year=freq,
        historical_events_range_min=freq_min,
        historical_events_range_max=freq_max,
        average_loss_per_event=round(avg_loss, 2),
        expected_annual_exposure=exp_annual,
        expected_annual_range_min=exp_min,
        expected_annual_range_max=exp_max,
        currency=curr,
        calculation_formula=calc_formula,
        basis=basis_str,
        confidence=FinancialConfidenceLevel.MEDIUM,
        is_assessable=True,
    )


def calculate_scenarios(
    confirmed_impact: ConfirmedFinancialImpact,
    potential_exposure: PotentialFinancialExposure,
    annualized_exposure: AnnualizedExposure,
    recurrence_analysis: RecurrenceAnalysis,
) -> FinancialScenarioAnalysis:
    """Generate evidence-backed Conservative, Expected, and High scenarios."""
    # Prefer the currency of whichever sub-analysis is actually driving the
    # displayed figures (recurrence -> annualized -> confirmed -> potential,
    # matching the priority order used below to pick exp_amt), rather than
    # always defaulting to confirmed_impact's currency -- confirmed_impact
    # can be entirely empty (e.g. a purely historical/annualized finding)
    # while still carrying its model's own static default. Note: each
    # sub-model still carries its own static "INR" Pydantic default when
    # truly no currency was ever determined for it (a pre-existing
    # architectural property of these already-tested models, not rewritten
    # here); this selection only avoids UNCONDITIONALLY preferring
    # confirmed_impact's currency over a sub-analysis that actually has data.
    if recurrence_analysis.is_assessable:
        curr = recurrence_analysis.currency
    elif annualized_exposure.is_assessable:
        curr = annualized_exposure.currency
    elif confirmed_impact.is_confirmed_event or confirmed_impact.has_reported_exposure:
        curr = confirmed_impact.currency
    else:
        curr = potential_exposure.currency

    if (
        not confirmed_impact.is_confirmed_event
        and not potential_exposure.is_present
        and not annualized_exposure.is_assessable
        and not recurrence_analysis.is_assessable
    ):
        return FinancialScenarioAnalysis(is_assessable=False)

    cons_amt = confirmed_impact.confirmed_net_loss if confirmed_impact.confirmed_net_loss is not None else 0.0
    cons_basis = f"Confirmed net loss: {curr} {cons_amt:,.2f}" if confirmed_impact.is_confirmed_loss else "Net loss not yet established (recovery unverified)"

    if recurrence_analysis.is_assessable and recurrence_analysis.expected_annual_exposure:
        exp_amt = recurrence_analysis.expected_annual_exposure
        exp_basis = f"Historical recurrence rate: {recurrence_analysis.basis}"
    elif annualized_exposure.is_assessable and annualized_exposure.annualized_amount:
        exp_amt = annualized_exposure.annualized_amount
        exp_basis = f"Annualized observed rate: {annualized_exposure.basis}"
    elif confirmed_impact.verified_gross_exposure is not None:
        exp_amt = confirmed_impact.verified_gross_exposure
        exp_basis = f"Direct verified gross exposure: {curr} {exp_amt:,.2f}"
    else:
        exp_amt = potential_exposure.lower_bound or 0.0
        exp_basis = f"Potential lower bound: {curr} {exp_amt:,.2f}"

    high_amt = exp_amt
    if potential_exposure.is_present and potential_exposure.upper_bound:
        high_amt = max(high_amt, (confirmed_impact.verified_gross_exposure or 0.0) + potential_exposure.upper_bound)
    if annualized_exposure.is_assessable and annualized_exposure.annualized_amount:
        high_amt = max(high_amt, annualized_exposure.annualized_amount)
    if recurrence_analysis.is_assessable and recurrence_analysis.expected_annual_range_max:
        high_amt = max(high_amt, recurrence_analysis.expected_annual_range_max)
    high_basis = f"Upper bound projection including unverified exposure/upper annualization: {curr} {high_amt:,.2f}"

    return FinancialScenarioAnalysis(
        conservative=ScenarioEstimate(name="CONSERVATIVE", amount=round(cons_amt, 2), currency=curr, basis=cons_basis),
        expected=ScenarioEstimate(name="EXPECTED", amount=round(exp_amt, 2), currency=curr, basis=exp_basis),
        high=ScenarioEstimate(name="HIGH", amount=round(high_amt, 2), currency=curr, basis=high_basis),
        is_assessable=True,
    )


def calculate_cost_of_quality(
    observations: list[FinancialObservation],
) -> CostOfQualityBreakdown:
    """Classify observed costs into standard Prevention-Appraisal-Failure (PAF) categories."""
    internal_failure = 0.0
    external_failure = 0.0
    appraisal = 0.0
    prevention = 0.0
    components = []
    curr: str | None = None

    for obs in observations:
        amt = obs.amount or (obs.unit_amount * (obs.quantity or obs.event_count or 1.0) if obs.unit_amount else 0.0)
        if not _is_valid_positive_number(amt):
            continue
        # Every classified observation carries a real currency (never
        # fabricated -- see app.financial.extractor/relationship_validator);
        # `curr` is left None (never actively defaulted to "INR" here) when
        # nothing has been classified yet, and only ever set from an
        # observation's own currency.
        curr = obs.currency or curr

        if obs.amount_type in (FinancialAmountType.REWORK_COST, FinancialAmountType.SCRAP_COST, FinancialAmountType.DOWNTIME_COST):
            internal_failure += amt
            components.append(f"Internal Failure ({obs.amount_type.value}): {curr} {amt:,.2f}")
        elif obs.amount_type in (FinancialAmountType.CUSTOMER_COMPENSATION, FinancialAmountType.PENALTY, FinancialAmountType.REVENUE_IMPACT):
            external_failure += amt
            components.append(f"External Failure ({obs.amount_type.value}): {curr} {amt:,.2f}")
        elif obs.amount_type in (FinancialAmountType.REMEDIATION_COST, FinancialAmountType.PREVENTION_COST):
            prevention += amt
            components.append(f"Prevention/Remediation: {curr} {amt:,.2f}")

    kwargs = dict(
        internal_failure_cost=round(internal_failure, 2) if internal_failure > 0 else None,
        external_failure_cost=round(external_failure, 2) if external_failure > 0 else None,
        appraisal_cost=round(appraisal, 2) if appraisal > 0 else None,
        prevention_cost=round(prevention, 2) if prevention > 0 else None,
        classified_components=components,
    )
    if curr is not None:
        kwargs["currency"] = curr
    return CostOfQualityBreakdown(**kwargs)


def calculate_capa_payback(
    observations: list[FinancialObservation],
    annual_avoided_exposure: float | None = None,
) -> CapaEconomicAnalysis:
    """Calculate indicative payback period: Remediation Cost / Annual Avoided Exposure."""
    remediation_cost = None
    curr = "INR"
    # Evidence status of whichever observation(s) actually supplied
    # remediation_cost -- calculation eligibility and evidentiary strength
    # are separate dimensions (see calculate_confirmed_impact's identical
    # VERIFIED/REPORTED split): a BELIEF/REPORTED-sourced figure is still
    # usable as an estimate but must never be indistinguishable from a
    # VERIFIED one to a caller/renderer.
    remediation_cost_status = "NOT_ASSESSABLE"

    _remediation_obs = [
        o for o in observations
        if o.amount_type in (FinancialAmountType.REMEDIATION_COST, FinancialAmountType.PREVENTION_COST)
        and _is_valid_positive_number(o.amount)
    ]
    # Same-currency corroboration collapses to one value (two sources
    # stating the identical figure describe the same fact, not two costs)
    # -- reuses the existing value-identity dedup, never source identity.
    _remediation_obs = _dedupe_corroborating_observations(_remediation_obs)

    if len(_remediation_obs) == 1:
        remediation_cost = _remediation_obs[0].amount
        curr = _remediation_obs[0].currency or curr
        remediation_cost_status = _remediation_obs[0].verification_status
    elif len(_remediation_obs) > 1:
        _totals = [o for o in _remediation_obs if o.is_aggregate_total]
        if len(_totals) == 1:
            # An explicit "total program cost" statement is the single
            # signal trusted to mean "this figure already sums the
            # program's components" -- use it exclusively rather than
            # also adding the individual components it already covers.
            remediation_cost = _totals[0].amount
            curr = _totals[0].currency or curr
            remediation_cost_status = _totals[0].verification_status
        else:
            # Multiple differing remediation amounts with no explicit
            # total marker -- could be additive components, mutually
            # exclusive alternatives, or a genuine revision/conflict. The
            # evidence does not structurally establish which, so this
            # must never be resolved by summing, by picking the first,
            # or by any other guess.
            _distinct_amounts = sorted({round(o.amount, 2) for o in _remediation_obs if o.currency == _remediation_obs[0].currency})
            return CapaEconomicAnalysis(
                is_assessable=False,
                remediation_status="REQUIRES_RECONCILIATION",
                conflicting_remediation_amounts=_distinct_amounts,
                currency=_remediation_obs[0].currency or curr,
            )

    if not remediation_cost:
        return CapaEconomicAnalysis(is_assessable=False)

    # Remediation cost itself is always reported when found -- it is a
    # DISTINCT financial fact from the payback CALCULATION, which
    # additionally requires a same-currency annual_avoided_exposure.
    # Without that second operand, the cost remains visible but
    # is_assessable (the payback figure) stays False -- never silently
    # dropping the cost merely because a payback cannot be computed.
    if not _is_valid_positive_number(annual_avoided_exposure):
        return CapaEconomicAnalysis(
            remediation_cost=round(remediation_cost, 2),
            currency=curr,
            is_assessable=False,
            remediation_cost_status=remediation_cost_status,
        )

    payback = round(remediation_cost / annual_avoided_exposure, 2)  # type: ignore[operator]
    return CapaEconomicAnalysis(
        remediation_cost=round(remediation_cost, 2),
        annual_avoided_exposure=round(annual_avoided_exposure, 2),  # type: ignore[arg-type]
        indicative_payback_years=payback,
        currency=curr,
        is_assessable=True,
        remediation_cost_status=remediation_cost_status,
    )


def derive_dimensional_confidence(
    confirmed: ConfirmedFinancialImpact,
    potential: PotentialFinancialExposure,
    annualized: AnnualizedExposure,
    recurrence: RecurrenceAnalysis,
    observations: list[FinancialObservation],
) -> DimensionalConfidence:
    """Deterministically assign multi-dimensional confidence ratings."""
    if not observations or (not confirmed.is_confirmed_event and not confirmed.has_reported_exposure and not potential.is_present and not annualized.is_assessable and not recurrence.is_assessable):
        return DimensionalConfidence(
            transaction_confidence=FinancialConfidenceLevel.NOT_ASSESSABLE,
            amount_confidence=FinancialConfidenceLevel.NOT_ASSESSABLE,
            recovery_confidence=FinancialConfidenceLevel.NOT_ASSESSABLE,
            net_loss_confidence=FinancialConfidenceLevel.NOT_ASSESSABLE,
            recurrence_confidence=FinancialConfidenceLevel.NOT_ASSESSABLE,
            projection_confidence=FinancialConfidenceLevel.NOT_ASSESSABLE,
            overall=FinancialConfidenceLevel.NOT_ASSESSABLE,
            rationale="Insufficient financial evidence available.",
        )

    has_verified = any(o.verification_status == "VERIFIED" for o in observations)
    has_reported = any(o.verification_status in ("REPORTED", "UNVERIFIED") for o in observations)

    txn_conf = FinancialConfidenceLevel.HIGH if (has_verified and not has_reported) else (FinancialConfidenceLevel.LOW if has_reported else FinancialConfidenceLevel.MEDIUM)
    amt_conf = FinancialConfidenceLevel.HIGH if (confirmed.verified_gross_exposure is not None and not has_reported) else (FinancialConfidenceLevel.LOW if (has_reported or confirmed.has_reported_exposure) else FinancialConfidenceLevel.MEDIUM)

    if confirmed.recovery_status in (RecoveryStatus.FULLY_RECOVERED, RecoveryStatus.PARTIALLY_RECOVERED, RecoveryStatus.VERIFIED_RECOVERED, RecoveryStatus.VERIFIED_ZERO_RECOVERY):
        rec_conf = FinancialConfidenceLevel.HIGH
    else:
        rec_conf = FinancialConfidenceLevel.NOT_ASSESSABLE

    if confirmed.is_confirmed_loss:
        loss_conf = FinancialConfidenceLevel.HIGH
    else:
        loss_conf = FinancialConfidenceLevel.NOT_ASSESSABLE

    rec_conf_dim = FinancialConfidenceLevel.MEDIUM if recurrence.is_assessable else FinancialConfidenceLevel.NOT_ASSESSABLE
    proj_conf = FinancialConfidenceLevel.MEDIUM if annualized.is_assessable else FinancialConfidenceLevel.NOT_ASSESSABLE

    if confirmed.is_confirmed_loss and rec_conf == FinancialConfidenceLevel.HIGH:
        overall = FinancialConfidenceLevel.HIGH
        rationale = "Gross exposure and recovery are both verified in authoritative evidence."
    elif confirmed.is_confirmed_event:
        overall = FinancialConfidenceLevel.MEDIUM
        rationale = "Gross exposure is verified; net loss requires recovery confirmation."
    elif confirmed.has_reported_exposure or has_reported:
        overall = FinancialConfidenceLevel.LOW
        rationale = "Financial amounts or event populations are reported/unverified and require verification."
    else:
        overall = FinancialConfidenceLevel.NOT_ASSESSABLE
        rationale = "Insufficient structured financial evidence."

    return DimensionalConfidence(
        transaction_confidence=txn_conf,
        amount_confidence=amt_conf,
        recovery_confidence=rec_conf,
        net_loss_confidence=loss_conf,
        recurrence_confidence=rec_conf_dim,
        projection_confidence=proj_conf,
        overall=overall,
        rationale=rationale,
    )
