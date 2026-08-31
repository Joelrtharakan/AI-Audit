"""PASS 51 -- evidence-backed pricing / false-reconciliation certification set.

Oracle only. Not imported by runtime code.

`expect` keys (superset of pass50):
  comparison_active        : bool
  recurrence_count_max     : int
  cost_recurrence          : "ONE_TIME" | "RECURRING" | "NONE"
  recurring_period         : str
  has_recurring_cost       : bool
  has_one_time_cost        : bool
  has_horizon_total        : bool
  one_time_cost_approx     : (value, tol)   -- headline one-time cost within tol
  recurring_cost_approx    : (value, tol)
  horizon_total_approx     : (value, tol)
  pricing_status_in        : list[str]
  min_priced_components    : int
  no_auditor_pricing_input : bool           -- 0 auditor inputs asking for an
                                              already-present rate/price/quantity
  root_cause_not_verified  : bool
  value_kind_not           : list[str]      -- headline component value_kind must
                                              not be any of these
"""

from __future__ import annotations

SCENARIOS: list[dict] = [
    dict(
        id="P51_panels_evidence_backed",
        section="exact-failure",
        material=True,
        finding=(
            "Eight electrical panels require corrective labeling and inspection. "
            "New labels cost Rs 350 per panel. An electrician requires 1.5 hours per "
            "panel at Rs 900 per hour, followed by a safety inspection costing Rs 6,000 "
            "for the complete area."
        ),
        expected_text=(
            "population/scope = 8 panels (NOT recurrence.count=8); recurrence = ONE_TIME; "
            "no comparison (3 different pricing bases for 3 components is a cost "
            "composition, not a discrepancy); "
            "labels 8 x 350 = 2,800; electrician 8 x 1.5 x 900 = 10,800; "
            "safety inspection 1 x 6,000 = 6,000; TOTAL = 19,600 ONE_TIME; "
            "3 priceable evidence-backed components; NO auditor pricing input; "
            "RCA NOT_ESTABLISHED; no fabricated impact; no OBSERVED_FINANCIAL_LOSS."
        ),
        expect=dict(
            comparison_active=False,
            recurrence_count_max=1,
            cost_recurrence="ONE_TIME",
            has_one_time_cost=True,
            has_recurring_cost=False,
            has_horizon_total=False,
            one_time_cost_approx=(19600, 1),
            min_priced_components=3,
            no_auditor_pricing_input=True,
            root_cause_not_verified=True,
            value_kind_not=["OBSERVED_FINANCIAL_LOSS", "HISTORICAL_EXPENDITURE"],
        ),
    ),
    dict(
        id="P51_A_5machines_parts",
        section="15A",
        material=True,
        finding=(
            "Five machines require replacement of a worn part. The replacement part "
            "costs Rs 4,000 per machine."
        ),
        expected_text="5 x 4,000 = 20,000 ONE_TIME; no comparison; no auditor pricing input.",
        expect=dict(
            comparison_active=False,
            recurrence_count_max=1,
            cost_recurrence="ONE_TIME",
            has_one_time_cost=True,
            one_time_cost_approx=(20000, 1),
            min_priced_components=1,
            no_auditor_pricing_input=True,
        ),
    ),
    dict(
        id="P51_B_6machines_inspection_hours",
        section="15B",
        material=True,
        finding=(
            "Six machines require a corrective inspection. Each inspection requires "
            "2 hours at Rs 800 per hour."
        ),
        expected_text="6 x 2 x 800 = 9,600 ONE_TIME; no comparison; no auditor pricing input.",
        expect=dict(
            comparison_active=False,
            recurrence_count_max=1,
            cost_recurrence="ONE_TIME",
            has_one_time_cost=True,
            one_time_cost_approx=(9600, 1),
            min_priced_components=1,
            no_auditor_pricing_input=True,
        ),
    ),
    dict(
        id="P51_C_6machines_monthly",
        section="15C",
        material=True,
        finding=(
            "Six machines require a monthly inspection. Each inspection requires "
            "2 hours at Rs 800 per hour."
        ),
        expected_text=(
            "recurrence = RECURRING, recurring_period = month; 6 x 2 x 800 = 9,600 per month; "
            "recurring_cost = 9,600/month; no one-time; no horizon_total (no horizon stated); "
            "no comparison."
        ),
        expect=dict(
            comparison_active=False,
            recurrence_count_max=1,
            cost_recurrence="RECURRING",
            recurring_period="month",
            has_recurring_cost=True,
            has_one_time_cost=False,
            has_horizon_total=False,
            recurring_cost_approx=(9600, 1),
            no_auditor_pricing_input=True,
        ),
    ),
    dict(
        id="P51_D_6machines_monthly_4months",
        section="15D",
        material=True,
        finding=(
            "Six machines require a monthly inspection for four months. Each inspection "
            "requires 2 hours at Rs 800 per hour."
        ),
        expected_text=(
            "recurring_cost = 9,600/month; explicit horizon = 4 months; "
            "horizon_total = 38,400 (9,600 x 4, arithmetic only); no comparison."
        ),
        expect=dict(
            comparison_active=False,
            cost_recurrence="RECURRING",
            recurring_period="month",
            has_recurring_cost=True,
            has_horizon_total=True,
            recurring_cost_approx=(9600, 1),
            horizon_total_approx=(38400, 1),
            no_auditor_pricing_input=True,
        ),
    ),
    dict(
        id="P51_E_replacement_vs_quote",
        section="15E",
        material=True,
        finding=(
            "Replacement of the failed unit costs Rs 20,000. A supplier quotation of "
            "Rs 18,000 has also been received for the same replacement."
        ),
        expected_text=(
            "DO NOT automatically activate comparison; two independent pricing inputs "
            "for the same activity; priceable (a defensible single figure or a small "
            "range); no reconciliation investigation."
        ),
        expect=dict(comparison_active=False),
    ),
    dict(
        id="P51_F_actual_vs_spec",
        section="15F",
        material=True,
        finding=(
            "The actual calibration result was 7.2 against a specification limit of 5.0, "
            "and the record marks the instrument as within tolerance."
        ),
        expected_text="genuine comparison (measured vs specified, expected to conform); comparison active.",
        expect=dict(comparison_active=True),
    ),
    dict(
        id="P51_G_replacement_priceable",
        section="15G",
        material=True,
        finding=(
            "The failed pump must be replaced. Replacement of the pump costs Rs 20,000."
        ),
        expected_text="remediation activity established + priceable; one_time_cost = 20,000; no comparison.",
        expect=dict(
            comparison_active=False,
            cost_recurrence="ONE_TIME",
            has_one_time_cost=True,
            one_time_cost_approx=(20000, 1),
            min_priced_components=1,
            no_auditor_pricing_input=True,
        ),
    ),
    dict(
        id="P51_H_incurred_loss",
        section="15H",
        material=True,
        finding=(
            "The company incurred Rs 20,000 in additional costs as a direct result of "
            "the incident."
        ),
        expected_text=(
            "Rs 20,000 = OBSERVED_FINANCIAL_LOSS, not a remediation cost; remediation "
            "NOT_ASSESSABLE / unpriced; the 20,000 must not be the remediation headline."
        ),
        expect=dict(
            comparison_active=False,
            pricing_status_in=["NOT_ASSESSABLE", "PARTIAL_ESTIMATE"],
        ),
    ),
    dict(
        id="P51_I_replacement_no_price",
        section="15I",
        material=True,
        finding=(
            "The failed valve must be replaced. No price or quotation for the "
            "replacement is available."
        ),
        expected_text=(
            "unpriced activity; specific auditor input for the replacement price; "
            "NOT_ASSESSABLE (no other priceable component); number-free."
        ),
        expect=dict(
            comparison_active=False,
            pricing_status_in=["NOT_ASSESSABLE", "PARTIAL_ESTIMATE"],
        ),
    ),
]


# ---- Pass 52 §22 additional generalization cases ----
SCENARIOS += [
    dict(
        id="P52_multi_component_replace_plus_labor",
        section="52.22.4",
        material=True,
        finding=(
            "Six machines require replacement of a failed module. The module costs Rs 4,000 "
            "per machine, and a technician requires 2 hours per machine at Rs 800 per hour."
        ),
        expected_text="6x4000 + 6x2x800 = 24,000 + 9,600 = 33,600 ONE_TIME; no comparison.",
        expect=dict(
            comparison_active=False, recurrence_count_max=1, cost_recurrence="ONE_TIME",
            has_one_time_cost=True, one_time_cost_approx=(33600, 1),
            min_priced_components=2, no_auditor_pricing_input=True,
        ),
    ),
    dict(
        id="P52_fixed_service_complete_area",
        section="52.22.3",
        material=True,
        finding=(
            "Six machines require inspection. The inspection service costs Rs 6,000 for the "
            "complete area."
        ),
        expected_text="fixed-scope: quantity 1 x Rs 6,000 = 6,000; NOT 6 x 6,000; ONE_TIME; no comparison.",
        expect=dict(
            comparison_active=False, recurrence_count_max=1, cost_recurrence="ONE_TIME",
            has_one_time_cost=True, one_time_cost_approx=(6000, 1), no_auditor_pricing_input=True,
        ),
    ),
    dict(
        id="P52_multiple_prices_no_comparison",
        section="52.22.10",
        material=True,
        finding=(
            "Corrective work needs a replacement part costing Rs 4,000, technician labour at "
            "Rs 800 per hour, and an inspection costing Rs 6,000."
        ),
        expected_text="three independent pricing bases for one remediation; comparison omitted.",
        expect=dict(comparison_active=False),
    ),
    dict(
        id="P52_rca_unknown_cost_known",
        section="52.22.11",
        material=True,
        finding=(
            "Eight panels require replacement. Each replacement costs Rs 2,500. No cause has "
            "been determined."
        ),
        expected_text="RCA NOT_ESTABLISHED; direct replacement priceable; 8 x 2,500 = 20,000 ONE_TIME.",
        expect=dict(
            comparison_active=False, recurrence_count_max=1, cost_recurrence="ONE_TIME",
            has_one_time_cost=True, one_time_cost_approx=(20000, 1),
            root_cause_not_verified=True, no_auditor_pricing_input=True,
        ),
    ),
    dict(
        id="P52_bare_quote_ambiguous",
        section="52.22.12",
        material=False,
        finding="A supplier quoted Rs 20,000.",
        expected_text=(
            "no remediation activity or applicability established; preserve uncertainty -- "
            "NOT_ASSESSABLE / review, do NOT blindly price Rs 20,000 as the remediation."
        ),
        expect=dict(comparison_active=False, pricing_status_in=["NOT_ASSESSABLE", "PARTIAL_ESTIMATE"]),
    ),
]


def material_ids() -> list[str]:
    return [s["id"] for s in SCENARIOS if s.get("material")]


def by_id(sid: str) -> dict:
    for s in SCENARIOS:
        if s["id"] == sid:
            return s
    raise KeyError(sid)
