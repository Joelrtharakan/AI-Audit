"""PASS 50 -- Production Certification Dataset (semantic oracle).

This is a CERTIFICATION ORACLE ONLY. Nothing here is imported by runtime
code. Each scenario carries a human-readable expected semantic outcome and a
small set of machine-checkable expectations that the live runner
(`scripts/pass50_certify.py`) compares against the REAL model output.

The expectations encode SEMANTIC CORRECTNESS, not schema validity:
  - population/entity count is NOT recurrence count
  - multiple legitimate prices are NOT a comparison
  - a recurring activity is NOT one-time
  - a horizon is NOT the per-occurrence quantity
  - a budget/quote/loss is NOT a remediation-cost actual
  - an observation is NOT a verified root cause
  - remediation cost is NOT financial impact

`expect` keys (all optional; only assert what the scenario is about):
  comparison_active      : bool  -- comparison_is_active(ctx.comparison)
  recurrence_count_max   : int   -- ctx.recurrence.count must be <= this (population guard)
  cost_recurrence        : "ONE_TIME" | "RECURRING" | "NONE"
  recurring_period       : str   -- e.g. "month"
  has_recurring_cost     : bool
  has_one_time_cost      : bool
  has_horizon_total      : bool
  pricing_status_in      : list[str]
  value_not_priced       : bool  -- headline cost must NOT be presented as an EXACT actual
  root_cause_not_verified: bool  -- ctx.root_cause_status must not be a VERIFIED/ESTABLISHED value
  impact_not_from_cost   : bool  -- financial impact must not be asserted purely from remediation cost
  review_or_partial      : bool  -- if the model is uncertain, result must be partial/NOT_ASSESSABLE/review_required
"""

from __future__ import annotations

SCENARIOS: list[dict] = [
    # ================= A. SUBJECT / ENTITY vs RECURRENCE =================
    dict(
        id="A1_single_machine_onetime",
        section="A/B",
        material=True,
        finding=(
            "Analyser AX-1 was found operating with an expired calibration certificate. "
            "The unit requires recalibration by an external service provider at a cost of "
            "Rs 3,500 per unit."
        ),
        expected_text=(
            "population = 1 analyser; one-time corrective recalibration; "
            "one_time_cost = Rs 3,500; no recurrence; no comparison."
        ),
        expect=dict(
            comparison_active=False,
            recurrence_count_max=1,
            cost_recurrence="ONE_TIME",
            has_one_time_cost=True,
            has_recurring_cost=False,
            has_horizon_total=False,
        ),
    ),
    dict(
        id="A2_population_monthly_verification",
        section="A/B/E",
        material=True,
        finding=(
            "Three analysers require monthly verification. Each verification takes 2 hours "
            "at Rs 1,200 per hour."
        ),
        expected_text=(
            "population/entity count = 3 analysers (NOT recurrence count); "
            "recurrence = RECURRING, recurring_period = month; "
            "quantity = 2 hours per analyser per occurrence; rate = Rs 1,200/hour; "
            "periodic cost = 3 x 2 x 1,200 = Rs 7,200/month; "
            "recurring_cost = Rs 7,200/month; one_time_cost = none; horizon_total = none "
            "(no horizon stated -> never annualized); comparison = omitted."
        ),
        expect=dict(
            comparison_active=False,
            recurrence_count_max=1,   # "3" is population, must NOT surface as recurrence.count=3
            cost_recurrence="RECURRING",
            recurring_period="month",
            has_recurring_cost=True,
            has_one_time_cost=False,
            has_horizon_total=False,
        ),
    ),
    dict(
        id="A3_population_multi_department",
        section="A",
        material=False,
        finding=(
            "Four departments failed to complete the annual management review on schedule. "
            "Each department must hold a catch-up review session."
        ),
        expected_text=(
            "entity count = 4 departments; the annual review is an existing yearly cycle, "
            "not evidence of a repeated failure; recurrence.count must not be 4; "
            "no comparison."
        ),
        expect=dict(comparison_active=False, recurrence_count_max=1),
    ),
    dict(
        id="A4_batch_population",
        section="A",
        material=False,
        finding=(
            "Twelve production batches were released without the required final QC signature."
        ),
        expected_text=(
            "population = 12 batches; this is a batch population, not 12 recurrences of an "
            "event; recurrence not established from the count alone; no comparison."
        ),
        expect=dict(comparison_active=False, recurrence_count_max=1),
    ),

    # ================= B. RECURRENCE forms =================
    dict(
        id="B1_occurrence_count_three_times",
        section="B",
        material=True,
        finding=(
            "The environmental verification was performed three times during the audit period "
            "and each time the result was outside the acceptance limit. Corrective "
            "re-verification will be performed by an internal analyst (2 hours at Rs 1,200/hour)."
        ),
        expected_text=(
            "occurrence count = 3 (the event genuinely happened 3 times); "
            "this is NOT a population and NOT a horizon; "
            "no stated ongoing frequency -> corrective re-verification is ONE_TIME unless the "
            "model establishes an ongoing schedule; no comparison."
        ),
        expect=dict(comparison_active=False, cost_recurrence="ONE_TIME", has_one_time_cost=True),
    ),
    dict(
        id="B2_monthly_for_six_months",
        section="B/F",
        material=True,
        finding=(
            "To contain the issue, a supplementary manual check must be performed monthly for "
            "six months. Each check costs Rs 2,000."
        ),
        expected_text=(
            "recurrence = RECURRING, recurring_period = month; "
            "explicit horizon = 6 months (EXPLICIT basis); "
            "periodic cost = Rs 2,000/month; horizon_total = Rs 12,000 (2,000 x 6, arithmetic "
            "only, because horizon is EXPLICITLY stated); no comparison."
        ),
        expect=dict(
            comparison_active=False,
            cost_recurrence="RECURRING",
            recurring_period="month",
            has_recurring_cost=True,
            has_horizon_total=True,
        ),
    ),
    dict(
        id="B3_frequency_without_quantity",
        section="B",
        material=True,
        finding=(
            "A weekly reconciliation of the access log must be introduced. The finding does not "
            "state how long each reconciliation takes or its cost."
        ),
        expected_text=(
            "recurrence = RECURRING, recurring_period = week; "
            "quantity / unit cost NOT established -> component unpriced; "
            "result = PARTIAL_ESTIMATE or NOT_ASSESSABLE with a specific auditor input "
            "('confirm effort/rate per weekly reconciliation'); NEVER a guessed number; "
            "no comparison."
        ),
        expect=dict(
            comparison_active=False,
            pricing_status_in=["PARTIAL_ESTIMATE", "NOT_ASSESSABLE", "RANGE_ESTIMATE"],
            value_not_priced=True,
            review_or_partial=True,
        ),
    ),
    dict(
        id="B4_repeated_event_no_frequency",
        section="B",
        material=False,
        finding=(
            "The same deviation has been raised on this line in two previous audits. "
            "A root-cause investigation and a one-time corrective action are required."
        ),
        expected_text=(
            "recurrence is established qualitatively (repeat finding) but NO regular "
            "frequency/period; corrective action is ONE_TIME; previous-CAPA reference only "
            "if the model establishes it, not from the word 'previous'; no comparison."
        ),
        expect=dict(comparison_active=False, cost_recurrence="ONE_TIME"),
    ),

    # ================= C. COMPARISON (must be a relationship) =================
    dict(
        id="C1_two_independent_prices",
        section="C",
        material=True,
        finding=(
            "Correcting the finding requires two activities: recalibration of the analyser at "
            "Rs 3,500 and replacement of the worn sensor at Rs 2,000."
        ),
        expected_text=(
            "two independent line-item prices for different activities; "
            "comparison = OMITTED (not a comparison); one_time_cost = Rs 5,500; no recurrence."
        ),
        expect=dict(comparison_active=False, cost_recurrence="ONE_TIME", has_one_time_cost=True),
    ),
    dict(
        id="C2_quote_vs_estimate",
        section="C/D",
        material=True,
        finding=(
            "The supplier provided a quotation of Rs 20,000 for the corrective rework. "
            "The internal engineering estimate for the same rework is Rs 18,000."
        ),
        expected_text=(
            "a quotation and an internal estimate for the same scope; these are two pricing "
            "inputs, NOT a discrepancy finding; comparison should be OMITTED unless the model "
            "explicitly establishes the audit intent is to reconcile them; "
            "value_kind: one QUOTED_PRICE, one ESTIMATE (neither an observed loss); "
            "if the model cannot choose one, PARTIAL/auditor-input is acceptable, a fabricated "
            "single actual is not."
        ),
        expect=dict(comparison_active=False, value_not_priced=False),
    ),
    dict(
        id="C3_budget_vs_actual_remediation",
        section="C/D",
        material=True,
        finding=(
            "The approved CAPA budget for this corrective action is Rs 50,000. The actual "
            "remediation is expected to cost Rs 40,000."
        ),
        expected_text=(
            "a budget and a remediation estimate; comparison = OMITTED; "
            "the budget is NOT a verified actual (basis capped at ESTIMATED); "
            "headline remediation cost derives from the Rs 40,000 estimate, not the budget."
        ),
        expect=dict(comparison_active=False),
    ),
    dict(
        id="C4_component_vs_subtotal",
        section="C",
        material=True,
        finding=(
            "The corrective work comprises labour of Rs 5,000 and parts of Rs 7,000, for a "
            "stated subtotal of Rs 12,000."
        ),
        expected_text=(
            "component/subtotal relationship, NOT a comparison; comparison = OMITTED; "
            "one_time_cost = Rs 12,000 (subtotal reconciles with components -> no double count)."
        ),
        expect=dict(comparison_active=False, cost_recurrence="ONE_TIME", has_one_time_cost=True),
    ),
    dict(
        id="C5_genuine_comparison_actual_vs_spec",
        section="C",
        material=True,
        finding=(
            "The measured fill volume was recorded as 48.2 mL against the specification of "
            "50.0 mL +/- 0.5 mL, and the batch record marks the result as conforming."
        ),
        expected_text=(
            "GENUINE comparison: a measured value against a specification limit that it is "
            "expected to meet, with a recorded conformance conclusion that conflicts with the "
            "numbers; comparison SHOULD be active (ACTUAL_CONFLICT / UNRESOLVED_COMPARISON) "
            "with a real why_comparable; this is the one scenario where comparison_active=True "
            "is correct."
        ),
        expect=dict(comparison_active=True),
    ),
    dict(
        id="C6_two_unrelated_measurements",
        section="C",
        material=False,
        finding=(
            "The ambient temperature log showed 22 C and the calibration record showed the "
            "last calibration was 14 months ago."
        ),
        expected_text=(
            "two unrelated numbers (a temperature and an interval); no semantic relationship; "
            "comparison = OMITTED."
        ),
        expect=dict(comparison_active=False),
    ),

    # ================= D. VALUE KIND =================
    dict(
        id="D1_observed_loss_not_remediation",
        section="D",
        material=True,
        finding=(
            "A duplicate payment of Rs 250,000 was made to the supplier due to a control "
            "failure in the accounts payable process."
        ),
        expected_text=(
            "Rs 250,000 is an OBSERVED_FINANCIAL_LOSS, not a remediation cost; "
            "remediation cost is NOT_ASSESSABLE from this finding (recovery/process-fix effort "
            "not priced); the loss must NOT appear as the remediation headline; "
            "financial impact may reference the loss, remediation cost must not."
        ),
        expect=dict(
            comparison_active=False,
            pricing_status_in=["NOT_ASSESSABLE", "PARTIAL_ESTIMATE"],
            value_not_priced=True,
        ),
    ),
    dict(
        id="D2_historical_expenditure_not_estimate",
        section="D",
        material=True,
        finding=(
            "Last year the company spent Rs 90,000 on similar corrective rework. The current "
            "finding requires comparable rework but no current quotation is available."
        ),
        expected_text=(
            "Rs 90,000 is HISTORICAL_EXPENDITURE, not a current remediation estimate; "
            "it must be rejected as the priced actual; result = NOT_ASSESSABLE / PARTIAL with "
            "an auditor input for a current quotation."
        ),
        expect=dict(
            comparison_active=False,
            pricing_status_in=["NOT_ASSESSABLE", "PARTIAL_ESTIMATE"],
            value_not_priced=True,
        ),
    ),
    dict(
        id="D3_quotation_priceable",
        section="D",
        material=True,
        finding=(
            "A firm fixed-price quotation of Rs 20,000 has been received from an approved "
            "contractor to perform the required corrective repair in full."
        ),
        expected_text=(
            "Rs 20,000 is a QUOTED_PRICE for the full corrective scope -> priceable; "
            "one_time_cost = Rs 20,000; pricing_status EXACT/PARTIAL acceptable; no comparison."
        ),
        expect=dict(comparison_active=False, cost_recurrence="ONE_TIME", has_one_time_cost=True),
    ),

    # ================= E. QUANTITY vs HORIZON =================
    dict(
        id="E1_hours_per_month_quantity",
        section="E/F",
        material=True,
        finding=(
            "An additional monitoring task of 2 hours per month must be performed by an "
            "internal analyst charged at Rs 1,200 per hour, until the permanent fix is "
            "validated."
        ),
        expected_text=(
            "quantity = 2 (hours), unit = hour, recurring_period = month; "
            "quantity is NOT 12 and NOT 24; recurrence = RECURRING; "
            "horizon is NOT stated numerically ('until the permanent fix is validated' is not "
            "a number of months) -> horizon_basis != EXPLICIT, NO horizon_total; "
            "recurring_cost = Rs 2,400/month; no comparison; no implicit annualization."
        ),
        expect=dict(
            comparison_active=False,
            cost_recurrence="RECURRING",
            recurring_period="month",
            has_recurring_cost=True,
            has_one_time_cost=False,
            has_horizon_total=False,
        ),
    ),
    dict(
        id="E2_multi_factor_one_time",
        section="E",
        material=True,
        finding=(
            "Three machines must each undergo a 2-hour corrective inspection by a technician "
            "charged at Rs 500 per hour. This is a one-off corrective action."
        ),
        expected_text=(
            "population = 3 machines; quantity = 2 hours per machine; rate = Rs 500/hour; "
            "one-off -> ONE_TIME; one_time_cost = 3 x 2 x 500 = Rs 3,000; "
            "no recurrence, no horizon, no comparison."
        ),
        expect=dict(
            comparison_active=False,
            recurrence_count_max=1,
            cost_recurrence="ONE_TIME",
            has_one_time_cost=True,
            has_recurring_cost=False,
        ),
    ),

    # ================= G. CAUSALITY =================
    dict(
        id="G1_observation_no_cause",
        section="G",
        material=True,
        finding=(
            "During the audit, the pressure reading on vessel V-3 was found to be 1.8 bar, "
            "outside the operating range of 2.0-3.0 bar. No cause has been determined."
        ),
        expected_text=(
            "observation only; root cause NOT established / NOT_ESTABLISHED; "
            "an investigation is required; the out-of-range reading is an observation, not a "
            "cause; no comparison (a single reading vs its own operating range is a limit "
            "check the model may or may not flag as comparison -- either omitted or a genuine "
            "limit comparison is defensible; a fabricated multi-price comparison is not)."
        ),
        expect=dict(root_cause_not_verified=True),
    ),
    dict(
        id="G2_reported_cause_contradicted",
        section="G",
        material=True,
        finding=(
            "The operator stated the deviation was caused by a faulty sensor. The maintenance "
            "log shows the sensor passed verification the same day with no fault recorded."
        ),
        expected_text=(
            "a REPORTED cause that the objective record contradicts; "
            "root cause must NOT be presented as verified/established; "
            "evidence conflict surfaced; investigation required."
        ),
        expect=dict(root_cause_not_verified=True),
    ),

    # ================= H. IMPACT =================
    dict(
        id="H1_cost_without_loss",
        section="H",
        material=True,
        finding=(
            "A procedure gap was identified in the sample-retention process. Correcting it "
            "requires rewriting the procedure and retraining staff at an estimated Rs 15,000. "
            "No product was affected and no loss was incurred."
        ),
        expected_text=(
            "remediation cost ~ Rs 15,000 (estimate); financial IMPACT = none demonstrated "
            "('no loss was incurred'); impact must NOT be inferred from the Rs 15,000 "
            "remediation cost; no comparison."
        ),
        expect=dict(comparison_active=False, impact_not_from_cost=True),
    ),

    # ================= I. INVESTIGATION =================
    dict(
        id="I1_missing_document",
        section="I/J",
        material=True,
        finding=(
            "The training record for operator J. Rao could not be located during the audit."
        ),
        expected_text=(
            "missing-record status established; investigation question about the missing "
            "record; NOT a comparison; NOT a recurrence; remediation likely NOT_ASSESSABLE / "
            "reconstruction effort unpriced."
        ),
        expect=dict(comparison_active=False, recurrence_count_max=1),
    ),
    dict(
        id="I2_competing_causes",
        section="G/I",
        material=True,
        finding=(
            "The batch failed dissolution testing. This could have resulted from excessive "
            "compression force, incorrect excipient grade, or a degraded API lot."
        ),
        expected_text=(
            "three stated competing causal hypotheses, all POSSIBLE, none verified; "
            "causal_alternatives_unresolved = true; a discrimination plan is expected; "
            "root cause NOT verified; no comparison."
        ),
        expect=dict(comparison_active=False, root_cause_not_verified=True),
    ),

    # ================= J. PREVIOUS CAPA =================
    dict(
        id="J1_explicit_prior_capa_ineffective",
        section="J",
        material=True,
        finding=(
            "This is a recurrence of finding CA-2025-014. The corrective action from that "
            "finding (revised SOP-22) was implemented in March 2025 but the same deviation "
            "has now recurred."
        ),
        expected_text=(
            "explicit previous CAPA reference (CA-2025-014 / SOP-22); "
            "previous CAPA implemented but recurrence occurred -> effectiveness in question; "
            "recurrence established qualitatively; no comparison; corrective action ONE_TIME "
            "plus systemic re-evaluation."
        ),
        expect=dict(comparison_active=False),
    ),
    dict(
        id="J2_word_repeat_no_capa",
        section="J",
        material=False,
        finding=(
            "The auditor again observed that the logbook was not signed at end of shift."
        ),
        expected_text=(
            "the word 'again' alone does NOT establish a previous CAPA; "
            "explicit_previous_capa_reference must be false absent a real prior-CAPA citation; "
            "no comparison."
        ),
        expect=dict(comparison_active=False),
    ),

    # ================= FULL-GRAPH BASICS =================
    dict(
        id="F1_plain_deviation",
        section="18.1",
        material=False,
        finding=(
            "The calibration of balance BX-7 was not performed within the required interval."
        ),
        expected_text="plain deviation; no cost stated; no comparison; no recurrence.",
        expect=dict(comparison_active=False, recurrence_count_max=1),
    ),
]


def material_ids() -> list[str]:
    return [s["id"] for s in SCENARIOS if s.get("material")]


def by_id(sid: str) -> dict:
    for s in SCENARIOS:
        if s["id"] == sid:
            return s
    raise KeyError(sid)
