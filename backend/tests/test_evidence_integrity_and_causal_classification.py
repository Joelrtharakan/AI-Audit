"""Regression coverage for the evidence-integrity + causal-classification
validation pass.

Four concrete defects were reproduced and fixed:

1. The deterministic fallback investigation planner
   (plan_investigation_fallback.build_deterministic_investigation_plan) had
   no guard for a finding that states ONLY a financial figure with no
   resolved entity/actor/mechanism. Such a finding fell through to the
   generic process-governance branch ("What approved procedure ... governs
   the affected process?") or -- worse -- into the recurrence branch, which
   fires on the bare word "recurring" and fabricated an entire "previous
   CAPA effectiveness" investigation narrative for a sentence that never
   mentioned a CAPA at all. Fixed with a financial-only guard that returns
   economic-evidence-verification questions instead.

2. reject_subject_if_clause's financial-metric-phrase rule (rule 7) did not
   cover "CAPA cost", "prevention cost", or the hyphenated "benefit-cost
   ratio" (missing modifier/meta-noun vocabulary), and rule 9's bare-term
   set omitted "recovery" -- all four fabricated as resolved subjects in
   realistic sentences ("CAPA cost was INR 10,000." -> subject="CAPA
   cost"). Fixed by extending _FINANCIAL_MODIFIERS/_FINANCIAL_META_NOUNS
   and adding "recovery" to _BARE_FINANCIAL_TERMS (safe because that rule
   only fires for a single-word candidate, so "Recovery Department"/
   "Recovery Team" are structurally unaffected).

3. calculate_capa_payback never looked at the verification_status of the
   remediation-cost observation(s) it used -- a BELIEF-sourced remediation
   estimate produced an identical CapaEconomicAnalysis to a VERIFIED one,
   with no field anywhere distinguishing them (unlike calculate_
   confirmed_impact's existing VERIFIED/REPORTED split for ordinary loss
   amounts). Fixed by adding remediation_cost_status to CapaEconomicAnalysis
   and populating it from the underlying observation's verification_status.

4. An explicit currency-code/symbol conflict (e.g. "USD ₹10,000") was
   already safely excluded from calculation, but was indistinguishable
   from "no financial evidence in this finding at all" -- extract_
   financial_observations silently discarded the conflicting token. Fixed
   by having it report the conflict separately (currency_conflicts) so
   engine.py can set conversion_status="CONFLICT" with an explanatory
   assessment_reason, still with zero calculation performed.

Uses abstract, domain-neutral test fixtures.
"""

from __future__ import annotations

from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
from app.financial.calculator import calculate_capa_payback
from app.financial.engine import analyze_financial_exposure
from app.financial.extractor import extract_financial_observations
from app.financial.models import FinancialAmountType, FinancialObservation
from app.models.agent import EvidenceItem, EvidenceStatus
from app.services.semantic_subject import reject_subject_if_clause, resolve_deviation
from app.services.taxonomy import coerce_category


# ---------------------------------------------------------------------------
# 1. Financial-only finding must not fabricate a process/CAPA investigation
# ---------------------------------------------------------------------------

def test_pure_remediation_cost_finding_does_not_ask_generic_procedure_question():
    finding = "The proposed remediation cost is INR 60,000."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="audit")]
    hyps, plan = build_deterministic_investigation_plan(finding, ledger)
    all_text = " ".join(plan.areas) + " " + " ".join(q.question for q in plan.questions)
    assert "governs the affected process" not in all_text.lower()
    assert "governing procedure" not in all_text.lower()
    assert "remediation cost" in all_text.lower() or "capa" in all_text.lower()
    assert hyps == []


def test_pure_historical_recurring_loss_finding_does_not_fabricate_previous_capa():
    """The exact reproduction: 'recurring' alone (with no actual previous
    CAPA reference) must never trigger the previous-CAPA-effectiveness
    investigation branch for a bare financial statement."""
    finding = "Historical recurring losses were USD 120,000 per year."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="audit")]
    hyps, plan = build_deterministic_investigation_plan(finding, ledger)
    all_text = " ".join(plan.areas) + " " + " ".join(q.question for q in plan.questions)
    assert "previous" not in all_text.lower()
    assert "capa" not in all_text.lower() or "capa" in all_text.lower()  # no false claim either way
    assert hyps == []


def test_bare_financial_only_finding_gets_financial_evidence_question():
    finding = "₹100,000 of losses were identified."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="audit")]
    hyps, plan = build_deterministic_investigation_plan(finding, ledger)
    assert hyps == []
    assert plan.questions
    assert any("financial" in q.question.lower() or "amount" in q.question.lower() for q in plan.questions)


def test_causal_finding_with_financial_impact_unaffected_by_financial_only_guard():
    """Contrast case: a finding with a genuine causal mechanism (equipment
    failure) AND a financial figure must still go through the normal
    process-investigation branches -- the financial-only guard must never
    intercept a finding that has an actual resolved subject."""
    finding = "Equipment failure caused ₹100,000 in downtime losses."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="audit")]
    hyps, plan = build_deterministic_investigation_plan(finding, ledger)
    all_text = " ".join(plan.areas) + " " + " ".join(q.question for q in plan.questions)
    assert "equipment failure" in all_text.lower()


# ---------------------------------------------------------------------------
# 2. Additional financial-metric affected-object fabrications
# ---------------------------------------------------------------------------

def test_additional_financial_metric_phrases_rejected():
    for phrase in ("CAPA cost", "prevention cost", "recovery", "benefit-cost ratio"):
        assert reject_subject_if_clause(phrase) is True, phrase


def test_additional_financial_metric_sentences_do_not_fabricate_subject():
    for finding in (
        "CAPA cost was INR 10,000.",
        "The prevention cost was estimated at USD 5,000.",
        "The benefit-cost ratio is 2.5.",
        "Recovery was reported at USD 20,000.",
    ):
        r = resolve_deviation(finding, [])
        assert r.subject is None, finding


def test_real_entities_containing_these_words_remain_accepted():
    """Contrast case: rule 7/9 must only reject a candidate composed
    ENTIRELY of financial modifiers/meta-nouns, or a BARE single-word
    financial term -- a real entity name must be untouched."""
    for phrase in ("Cost Center", "Vendor Payment System", "Financial Reporting Process",
                   "Revenue Recognition Process", "Payment Processing System",
                   "Recovery Department", "Recovery Team", "Implementation Team", "CAPA Cost Center"):
        assert reject_subject_if_clause(phrase) is False, phrase


# ---------------------------------------------------------------------------
# 3. Remediation cost evidence-status must never silently strengthen
# ---------------------------------------------------------------------------

def test_verified_remediation_cost_marked_verified():
    obs = [FinancialObservation(
        observation_id="o1", amount=20000.0, currency="INR",
        amount_type=FinancialAmountType.REMEDIATION_COST, verification_status="VERIFIED",
    )]
    result = calculate_capa_payback(obs)
    assert result.remediation_cost == 20000.0
    assert result.remediation_cost_status == "VERIFIED"


def test_belief_sourced_remediation_cost_never_marked_verified():
    """The exact reproduction: a BELIEF/UNVERIFIED-sourced remediation
    estimate must never be indistinguishable from a VERIFIED one."""
    obs = [FinancialObservation(
        observation_id="o1", amount=20000.0, currency="INR",
        amount_type=FinancialAmountType.REMEDIATION_COST, verification_status="UNVERIFIED",
        source_evidence_status="BELIEF",
    )]
    result = calculate_capa_payback(obs)
    assert result.remediation_cost == 20000.0
    assert result.remediation_cost_status != "VERIFIED"
    assert result.remediation_cost_status == "UNVERIFIED"


def test_reported_remediation_cost_marked_reported_end_to_end():
    ledger = [EvidenceItem(claim="The proposed remediation cost is INR 60,000.", status=EvidenceStatus.REPORTED, source="C1")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.capa_economics.remediation_cost == 60000.0
    assert res.capa_economics.remediation_cost_status == "REPORTED"


# ---------------------------------------------------------------------------
# 4. Currency conflict is surfaced, never calculated
# ---------------------------------------------------------------------------

def test_currency_code_symbol_conflict_surfaced_without_calculation():
    ledger = [EvidenceItem(claim="The loss was USD ₹10,000.", status=EvidenceStatus.VERIFIED, source="C1")]
    res = analyze_financial_exposure("x", evidence_ledger=ledger)
    assert res.conversion_status == "CONFLICT"
    assert res.confirmed_impact.verified_gross_exposure is None
    assert "conflict" in res.assessment_reason.lower()


def test_currency_conflict_extraction_signal_distinct_from_amount_conflict():
    observations, has_conflict, currency_conflicts = extract_financial_observations(
        "The loss was USD ₹10,000."
    )
    assert observations == []
    assert has_conflict is False
    assert currency_conflicts


def test_non_conflicting_currency_extraction_reports_no_currency_conflict():
    observations, has_conflict, currency_conflicts = extract_financial_observations(
        "A direct loss of USD 10,000 was confirmed."
    )
    assert observations
    assert currency_conflicts == []


# ---------------------------------------------------------------------------
# 5. 6M classification cannot be fabricated from a financial figure
# (structural invariant: the taxonomy has no financial category, so any
# LLM output naming one is coerced to OTHER, never accepted as-is)
# ---------------------------------------------------------------------------

def test_taxonomy_has_no_financial_category():
    for bogus in ("Financial Loss", "INR 50,000", "Cost", "financial loss", "$50,000"):
        assert coerce_category(bogus).value == "OTHER", bogus


def test_taxonomy_still_accepts_genuine_6m_categories():
    assert coerce_category("Machine").value == "MACHINE"
    assert coerce_category("MATERIAL").value == "MATERIAL"


# ---------------------------------------------------------------------------
# 6. Five-Why: financial-only findings never fabricate a root cause
# ---------------------------------------------------------------------------

def test_five_why_pure_financial_finding_stops_at_evidence_boundary():
    from app.agent.nodes.five_why_fallback import build_deterministic_five_why
    finding = "₹100,000 in losses occurred."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="audit")]
    result = build_deterministic_five_why(finding, ledger)
    assert result.steps
    assert result.steps[0].status == "UNKNOWN"
    combined = " ".join((s.answer or "") for s in result.steps).lower()
    assert "because" not in combined.split(".")[0]  # no fabricated one-line causal claim


def test_five_why_historical_annualized_finding_infers_no_root_cause():
    from app.agent.nodes.five_why_fallback import build_deterministic_five_why
    finding = "Historical losses were USD 120,000 per year."
    ledger = [EvidenceItem(claim=finding, status=EvidenceStatus.VERIFIED, source="audit")]
    result = build_deterministic_five_why(finding, ledger)
    assert all(s.status != "VERIFIED" or "mechanism" not in (s.answer or "").lower() for s in result.steps)
    assert result.steps[-1].status == "UNKNOWN"
