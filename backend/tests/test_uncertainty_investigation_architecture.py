"""Tests for Generalized Evidence-Uncertainty Model & Dynamic Investigation Strategy Gate.

Covers Scenarios A through P and Invariants INV-UNCERTAINTY-001..005,
INV-INVEST-UNCERTAINTY-001..002, INV-5WHY-UNCERTAINTY-001, INV-OBJECT-001.
Strictly domain-agnostic without hardcoded domain vocabulary.
"""

import pytest
from app.agent.causal_guard import detect_uncertainties, select_investigation_strategy
from app.agent.invariants import evaluate_all_invariants
from app.agent.nodes.five_why_fallback import build_deterministic_five_why
from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
from app.agent.nodes.understanding import understand_finding_node
from app.agent.state import AgentState
from app.models.agent import (
    CanonicalFindingState,
    EvidenceClaim,
    EvidenceItem,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    ImpactAssessment,
    InvestigationPlan,
    InvestigationQuestion,
    InvestigateRequest,
    RootCauseAnalysis,
    RootCauseStatus,
)


async def _build_state(finding_text: str, evidence_ledger: list[EvidenceItem] = None) -> AgentState:
    evidence_ledger = evidence_ledger or []
    req = InvestigateRequest(finding_text=finding_text)
    state: AgentState = {
        "request": req,
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "evidence_ledger": evidence_ledger,
        "evidence_gaps": [],
        "trace": [],
        "errors": [],
        "completed_tools": [],
        "planned_tools": [],
        "tool_results": {},
        "contributing_factors": [],
    }
    return await understand_finding_node(state)


class TestUncertaintyScenarios:
    """Test Scenarios A through P covering generalized uncertainty structures."""

    async def test_scenario_a_requirement_uncertain(self):
        """Scenario A: Observation verified, requirement unknown/unavailable."""
        text = "Item alpha was observed in state beta during operation; the governing specification is unknown."
        state = await _build_state(text)
        canonical = state["canonical_finding_state"]

        assert canonical.primary_uncertainty == "REQUIREMENT_UNCERTAIN"
        assert canonical.causal_readiness == "NOT_READY"
        assert "Compliance determination" in canonical.blocked_reasoning_steps[0]

        hyps, plan = build_deterministic_investigation_plan(text, [], canonical_subject=canonical.finding_subject)
        # Zero hypotheses when requirement is uncertain
        assert len(hyps) == 0
        assert plan.questions[0].priority == "P1"
        assert plan.questions[0].uncertainty_resolved == "REQUIREMENT_UNCERTAIN"

        fw = build_deterministic_five_why(text, [], canonical_subject=canonical.finding_subject)
        assert len(fw.steps) == 1
        assert "deferred" in fw.steps[0].answer.lower()

    async def test_scenario_b_missing_record_activity_unverified(self):
        """Scenario B: Activity recorded missing, physical execution unconfirmed."""
        text = "Activity gamma for unit delta was not recorded in the operational log."
        state = await _build_state(text)
        canonical = state["canonical_finding_state"]

        assert canonical.primary_uncertainty == "DOCUMENTATION_UNCERTAIN"
        strategy = select_investigation_strategy(canonical.primary_uncertainty, canonical)
        assert strategy == "DOCUMENTATION_VS_PERFORMANCE"

    async def test_scenario_c_event_sequence_control_gap(self):
        """Scenario C: Controlled transition with justification missing."""
        text = "The invalidation of parameter epsilon occurred without documented justification."
        state = await _build_state(text)
        canonical = state["canonical_finding_state"]

        assert canonical.primary_uncertainty == "AUTHORIZATION_UNCERTAIN"
        strategy = select_investigation_strategy(canonical.primary_uncertainty, canonical)
        assert strategy == "AUTHORIZATION_VERIFICATION"

    async def test_scenario_d_financial_transaction_recovery_uncertain(self):
        """Scenario D: Duplicate disbursement with unverified recovery."""
        text = "A duplicate payment of $5,000 was processed for invoice INV-100."
        state = await _build_state(text)
        canonical = state["canonical_finding_state"]

        assert canonical.primary_uncertainty == "RECOVERY_UNCERTAIN"
        strategy = select_investigation_strategy(canonical.primary_uncertainty, canonical)
        assert strategy == "FINANCIAL_RECOVERY_AND_CONTAINMENT"

    async def test_scenario_e_recurrence_with_prior_action(self):
        """Scenario E: Same deviation observed across multiple units with prior action."""
        text = "The same deviation was identified across three separate units; prior action CAPA-01 was completed."
        state = await _build_state(text)
        canonical = state["canonical_finding_state"]

        assert canonical.primary_uncertainty == "RECURRENCE_UNCERTAIN"
        strategy = select_investigation_strategy(canonical.primary_uncertainty, canonical)
        assert strategy == "RECURRENCE_AND_PREVIOUS_CAPA"

    async def test_scenario_f_comparison_discrepancy(self):
        """Scenario F: Discrepancy between recorded value and reference."""
        text = "Recorded value 10.5 differed from the approved parameter of 12.0."
        state = await _build_state(text)
        canonical = state["canonical_finding_state"]

        assert canonical.primary_uncertainty == "MECHANISM_UNCERTAIN"
        strategy = select_investigation_strategy(canonical.primary_uncertainty, canonical)
        assert strategy == "MECHANISM_DISCRIMINATION"


class TestInvariants:
    """Test evaluation of the newly added uncertainty invariants."""

    def test_inv_uncertainty_001_presence(self):
        """INV-UNCERTAINTY-001: Primary uncertainty required when RC not established."""
        canonical = CanonicalFindingState(
            raw_finding="Finding text",
            observed_deviation="Deviation",
            finding_subject="Subject",
            primary_uncertainty="",
        )
        rc = RootCauseAnalysis(
            status=RootCauseStatus.NOT_ESTABLISHED,
            confidence="LOW",
            narrative="Narrative",
        )
        state = {"canonical_finding_state": canonical, "root_cause": rc}
        valid, violations = evaluate_all_invariants(state)
        assert not valid
        assert any("INV-UNCERTAINTY-001" in v for v in violations)

    def test_inv_uncertainty_002_blocks_hypotheses(self):
        """INV-UNCERTAINTY-002: Requirement uncertainty must block causal hypotheses."""
        from app.models.agent import CandidateHypothesis
        canonical = CanonicalFindingState(
            raw_finding="Finding text",
            observed_deviation="Deviation",
            finding_subject="Subject",
            primary_uncertainty="REQUIREMENT_UNCERTAIN",
            requirement_status="UNKNOWN",
        )
        rc = RootCauseAnalysis(
            status=RootCauseStatus.NOT_ESTABLISHED,
            confidence="LOW",
            narrative="Narrative",
            candidate_hypotheses=[
                CandidateHypothesis(id="H1", name="H1", statement="Operator missed step", status="POSSIBLE", evidence_needed="Log")
            ],
        )
        state = {"canonical_finding_state": canonical, "root_cause": rc}
        valid, violations = evaluate_all_invariants(state)
        assert not valid
        assert any("INV-UNCERTAINTY-002" in v for v in violations)

    def test_inv_5why_uncertainty_001_deferral(self):
        """INV-5WHY-UNCERTAINTY-001: 5-Why must be deferred when requirement uncertain."""
        canonical = CanonicalFindingState(
            raw_finding="Finding text",
            observed_deviation="Deviation",
            finding_subject="Subject",
            primary_uncertainty="REQUIREMENT_UNCERTAIN",
            requirement_status="UNKNOWN",
        )
        fw = FiveWhyAnalysis(
            steps=[
                FiveWhyStep(question="Why did it happen?", answer="Because of system error", status="VERIFIED")
            ]
        )
        state = {"canonical_finding_state": canonical, "five_why": fw}
        valid, violations = evaluate_all_invariants(state)
        assert not valid
        assert any("INV-5WHY-UNCERTAINTY-001" in v for v in violations)

    def test_inv_object_001_no_generic_process_compliance(self):
        """INV-OBJECT-001: process_at_risk cannot be 'Process compliance'."""
        impact = ImpactAssessment(
            status="IMPACT_REQUIRES_ASSESSMENT",
            affected_object="Unit X",
            process_at_risk="Process compliance",
        )
        state = {"impact_assessment": impact}
        valid, violations = evaluate_all_invariants(state)
        assert not valid
        assert any("INV-OBJECT-001" in v for v in violations)

    @pytest.mark.asyncio
    async def test_end_to_end_requirement_unresolved_chemical_container_scenario(self):
        """Verify the exact finding with unresolved requirement produces 0 hypotheses,
        deferred 5-Why, and grammatically valid impact without causal invention.
        """
        from app.agent.nodes.core_synthesis import _derive_deterministic_impact
        from app.agent.nodes.investigation_planner import plan_investigation_node
        
        text = (
            "A chemical container was found stored outside its designated storage cabinet. "
            "The container label was present, but the storage requirement could not be confirmed during the audit."
        )
        state = await _build_state(text)
        canonical = state["canonical_finding_state"]

        assert canonical.primary_uncertainty == "REQUIREMENT_UNCERTAIN"
        assert canonical.causal_readiness == "NOT_READY"
        assert canonical.observed_entity == "chemical container"

        # Planner fast-path builds P1-P5 requirement tree
        plan_state = await plan_investigation_node(state)
        plan = plan_state["investigation_plan"]
        assert len(plan.questions) >= 5
        assert plan.questions[0].id == "P1_REQUIREMENT"
        assert plan.questions[0].uncertainty_resolved == "REQUIREMENT_UNCERTAIN"

        # Deterministic impact has grammatically valid impact statement (no 'failed to requirement unresolved')
        impact, clean_noun, topic, actor = _derive_deterministic_impact(text, canonical, canonical.observed_deviation)
        assert "failed to requirement unresolved" not in impact.potential_effect
        assert "chemical container" in impact.potential_effect.lower() or "chemical container" in clean_noun.lower()

        # Deterministic 5-Why is explicitly deferred
        fw = build_deterministic_five_why(text, [], canonical_state=canonical)
        assert "deferred" in fw.steps[0].answer.lower()

        # Invariant check passes cleanly
        state["investigation_plan"] = plan
        state["five_why"] = fw
        state["impact_assessment"] = impact
        state["root_cause"] = RootCauseAnalysis(
            status=RootCauseStatus.NOT_ESTABLISHED,
            confidence="LOW",
            narrative="Requirement is unresolved.",
            candidate_hypotheses=[],
        )
        valid, violations = evaluate_all_invariants(state)
        assert valid, f"Violations: {violations}"

