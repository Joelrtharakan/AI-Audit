"""Regression and adversarial tests for finding understanding, semantic resolution,
investigation quality, 5-Why quality, and language hardening.
"""

import pytest
from app.models.agent import (
    CanonicalFindingState,
    EvidenceItem,
    EvidenceStatus,
    InvestigateRequest,
    RootCauseStatus,
)
from app.services.semantic_subject import (
    build_affected_object_phrase,
    extract_semantic_subject,
    format_deviation_why_question,
    is_actor_noun,
    resolve_deviation,
)
from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
from app.agent.nodes.five_why_fallback import build_deterministic_five_why
from app.agent.nodes.understanding import understand_finding_node
from app.agent.nodes.core_synthesis import core_synthesis_node
from app.agent.nodes.final_evidence_verification import final_evidence_verification_node


@pytest.mark.asyncio
async def test_four_employees_finding_full_pipeline():
    """Test the exact 4-employee failure example across the entire pipeline."""
    finding_text = (
        "Four employees failed to complete the revised inspection checklist. "
        "One employee reported insufficient training. "
        "Another employee reported workload pressure. "
        "The supervisor reported poor discipline."
    )
    
    # 1. Semantic subject extraction verification
    info = extract_semantic_subject(finding_text)
    assert is_actor_noun(info.actor)
    assert "Four employees" in info.actor or "employees" in info.actor.lower()
    assert "checklist" in info.affected_object.lower()
    assert "Employees" not in info.affected_object
    assert "Revised inspection checklist" in info.affected_object
    assert info.semantic_type == "RECORD"
    assert info.relevant_change == "Revision of the inspection checklist"

    # 2. Pipeline execution
    state = {
        "request": InvestigateRequest(finding_text=finding_text),
        "evidence_ledger": [],
        "trace": [],
        "errors": [],
    }

    from unittest.mock import patch
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        # Understand node
        state = await understand_finding_node(state)
        canonical: CanonicalFindingState = state["canonical_finding_state"]
        assert canonical.affected_object != "Employees"
        assert "checklist" in canonical.affected_object.lower()
        assert canonical.relevant_change == "Revision of the inspection checklist"
        assert len(canonical.reported_statements) >= 3

        # Core synthesis node (deterministic fallback or recovery)
        state = await core_synthesis_node(state)
        
        # Final verification node
        state = await final_evidence_verification_node(state)

    rc = state["root_cause"]
    fw = state["five_why"]
    impact = state["impact_assessment"]
    inv = state["investigation_plan"]

    # Invariant checks:
    # A. Affected Object & Process at Risk
    assert impact.affected_object != "Employees"
    assert "Employees" not in impact.affected_object
    assert "Employees control" not in impact.process_at_risk
    assert "checklist" in impact.affected_object.lower()

    # B. Causal conservatism: Root cause NOT_ESTABLISHED, no leading hypothesis
    assert rc.status == RootCauseStatus.NOT_ESTABLISHED
    assert rc.leading_hypothesis is None or rc.leading_hypothesis.startswith("NONE")

    # C. Hypotheses: all 3 reported explanations extracted with status POSSIBLE
    assert len(rc.candidate_hypotheses) == 3
    hyp_names = [h.name for h in rc.candidate_hypotheses]
    assert any("TRAINING" in name for name in hyp_names)
    assert any("WORKLOAD" in name for name in hyp_names)
    assert any("DISCIPLINE" in name or "PERFORMANCE" in name for name in hyp_names)
    for h in rc.candidate_hypotheses:
        assert h.status == "POSSIBLE"
        assert h.supporting_claim_ids  # Provenance present

    # D. 5-Why: Exactly 3 steps, stopping at evidence boundary
    assert len(fw.steps) == 3
    assert fw.steps[0].status == "VERIFIED"
    assert fw.steps[1].status in ("MIXED", "REPORTED")
    assert fw.steps[2].status == "UNKNOWN"
    assert not fw.is_complete

    # E. Investigation questions: Targeted to training, workload, performance
    assert len(inv.questions) >= 3
    q_texts = " ".join(q.question for q in inv.questions)
    assert "training" in q_texts.lower()
    assert "workload" in q_texts.lower() or "staffing" in q_texts.lower()
    assert "performance" in q_texts.lower() or "failure to perform" in q_texts.lower()

    # F. Language hardening checks (No malformed grammar)
    full_output_text = (
        f"{impact.affected_object} {impact.process_at_risk} {impact.potential_effect} "
        f"{rc.root_cause_basis} {rc.narrative} "
        + " ".join(h.statement for h in rc.candidate_hypotheses)
        + " ".join(s.question + " " + s.answer for s in fw.steps)
        + " ".join(q.question for q in inv.questions)
    )
    assert "employees was" not in full_output_text.lower()
    assert "why was the employees complete" not in full_output_text.lower()
    assert "employees was reportedly complete" not in full_output_text.lower()
    assert "that employees was performed" not in full_output_text.lower()


def test_adversarial_a_sanitation_log():
    """Adversarial A: Six operators did not sign the revised sanitation log."""
    text = "Six operators did not sign the revised sanitation log."
    info = extract_semantic_subject(text)
    assert is_actor_noun(info.actor)
    assert "operators" in info.actor.lower()
    assert "sanitation log" in info.affected_object.lower()
    assert "Operators" not in info.affected_object


def test_adversarial_b_calibration_verification():
    """Adversarial B: Two technicians failed to perform the daily calibration verification."""
    text = "Two technicians failed to perform the daily calibration verification."
    info = extract_semantic_subject(text)
    assert is_actor_noun(info.actor)
    assert "technicians" in info.actor.lower()
    assert "calibration verification" in info.affected_object.lower()


def test_adversarial_c_batch_temperature():
    """Adversarial C: Staff members did not record the batch temperature."""
    text = "Staff members did not record the batch temperature."
    info = extract_semantic_subject(text)
    assert is_actor_noun(info.actor)
    assert "batch temperature" in info.affected_object.lower()


def test_adversarial_d_single_reported_cause():
    """Adversarial D: Single reported cause."""
    finding_text = "The checklist was not completed. The operator stated training was missed."
    evidence_ledger = [
        EvidenceItem(claim="The checklist was not completed.", source="AUDIT_FINDING", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="The operator stated training was missed.", source="AUDIT_FINDING", status=EvidenceStatus.REPORTED),
    ]
    fw = build_deterministic_five_why(finding_text, evidence_ledger)
    assert len(fw.steps) in (2, 3)
    assert fw.steps[0].status == "VERIFIED"
    assert any(s.status == "REPORTED" for s in fw.steps)
    assert fw.steps[-1].status == "UNKNOWN"


def test_adversarial_e_conflicting_causes():
    """Adversarial E: Conflicting causes."""
    finding_text = "The log was not completed. Operator stated log was lost. Supervisor stated operator forgot."
    evidence_ledger = [
        EvidenceItem(claim="The log was not completed.", source="AUDIT_FINDING", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="Operator stated log was lost.", source="AUDIT_FINDING", status=EvidenceStatus.REPORTED),
        EvidenceItem(claim="Supervisor stated operator forgot.", source="AUDIT_FINDING", status=EvidenceStatus.REPORTED),
    ]
    fw = build_deterministic_five_why(finding_text, evidence_ledger)
    assert len(fw.steps) >= 2
    assert not fw.is_complete


def test_adversarial_f_zero_causal_explanations():
    """Adversarial F: Zero causal explanations."""
    finding_text = "Three analysts failed to submit the quarterly report."
    evidence_ledger = [
        EvidenceItem(claim=finding_text, source="AUDIT_FINDING", status=EvidenceStatus.VERIFIED),
    ]
    hyps, plan = build_deterministic_investigation_plan(finding_text, evidence_ledger)
    # Zero causal content -> 0 hypotheses
    assert len(hyps) == 0
    assert len(plan.questions) >= 2
    for q in plan.questions:
        assert "analysts was" not in q.question.lower()


def test_adversarial_g_unavailable_document():
    """Adversarial G: Referenced incident report could not be located."""
    finding_text = "Deviation noted in audit. Referenced incident report could not be located."
    evidence_ledger = []
    fw = build_deterministic_five_why(finding_text, evidence_ledger)
    assert len(fw.steps) == 1
    assert fw.steps[0].status == "UNKNOWN"
    assert not fw.is_complete


def test_adversarial_h_delivery_vs_receipt():
    """Adversarial H: Delivery vs receipt conflict."""
    finding_text = "System log confirms email dispatch to all personnel. Three operators report no notification received."
    evidence_ledger = [
        EvidenceItem(claim="System log confirms email dispatch to all personnel.", source="AUDIT_FINDING", status=EvidenceStatus.VERIFIED),
        EvidenceItem(claim="Three operators report no notification received.", source="AUDIT_FINDING", status=EvidenceStatus.REPORTED),
    ]
    fw = build_deterministic_five_why(finding_text, evidence_ledger)
    assert not fw.is_complete
    assert "delivery" in fw.steps[0].question.lower() or "dispatch" in fw.steps[0].question.lower()


@pytest.mark.asyncio
async def test_duplicate_payment_full_pipeline():
    """Sections 1-13: Full pipeline execution for duplicate payment finding."""
    from app.agent.nodes.report_generator import generate_report_node

    finding_text = "During the audit, duplicate payment of ₹1,25,000 to a supplier was identified."
    req = InvestigateRequest(finding_text=finding_text)
    state = {
        "request": req,
        "evidence_ledger": [],
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "trace": [],
        "errors": [],
    }

    from unittest.mock import patch
    with patch("app.agent.nodes.understanding.get_llm_client", return_value=None), \
         patch("app.agent.nodes.core_synthesis.get_llm_client", return_value=None):
        # Understand node
        state = await understand_finding_node(state)
        canonical: CanonicalFindingState = state["canonical_finding_state"]
        assert canonical.affected_object != "Process compliance"
        assert "duplicate" in canonical.affected_object.lower() and "payment" in canonical.affected_object.lower()
        assert "accounts payable" in canonical.affected_process.lower() or "payment" in canonical.affected_process.lower()

        # Core synthesis node
        state = await core_synthesis_node(state)

        # Final verification node
        state = await final_evidence_verification_node(state)

        # Report generator node
        state = await generate_report_node(state)

    report = state.get("report")
    assert report is not None

    # 1. Semantic resolution
    impact = report.impact_assessment
    assert impact.affected_object == "Duplicate payment to supplier"
    assert "accounts payable" in impact.process_at_risk.lower()
    assert impact.control_at_risk is not None and ("duplicate" in impact.control_at_risk.lower() or "prevention" in impact.control_at_risk.lower() or "reconciliation" in impact.control_at_risk.lower())
    assert impact.financial_amount is not None
    assert impact.financial_amount.amount == 125000.0
    assert "₹125,000" in impact.potential_effect
    assert "of ," not in impact.potential_effect
    assert "₹," not in impact.potential_effect

    # 2. Cost impact model
    cost = report.cost_impact
    assert cost is not None
    assert cost.cost_factor_detected is True
    assert cost.financial_factor == "DUPLICATE PAYMENT"
    assert cost.potential_exposure == 125000.0
    assert cost.actual_loss is None
    assert cost.actual_loss_status == "NOT_ESTABLISHED"
    assert cost.recoverability_status == "REQUIRES_VERIFICATION"
    assert cost.cost_drivers == []  # Rule 8: No invented secondary costs
    assert cost.financial_amount is not None
    assert cost.financial_amount.amount == 125000.0
    assert cost.financial_amount.currency == "INR"
    assert cost.financial_amount.support_status == "VERIFIED"

    # 3. Root cause & Hypotheses
    rc = report.root_cause
    assert rc.status == RootCauseStatus.NOT_ESTABLISHED
    assert rc.leading_hypothesis is None or rc.leading_hypothesis.startswith("NONE")
    assert len(rc.candidate_hypotheses) >= 2
    for h in rc.candidate_hypotheses:
        assert h.status == "POSSIBLE"
    for h in rc.candidate_hypotheses:
        assert h.status == "POSSIBLE"

    # 4. Investigation plan
    inv = report.investigation
    assert len(inv.questions) >= 4
    q_all = " ".join(q.question for q in inv.questions)
    assert "invoice" in q_all.lower() or "supplier" in q_all.lower()

    # 5. 5-Why
    fw = report.five_why
    assert len(fw.steps) == 3
    assert fw.steps[0].status == "VERIFIED"
    assert fw.steps[1].status == "UNKNOWN"
    assert fw.steps[2].status == "UNKNOWN"
    assert not fw.is_complete

    # 6. CA Draft immediate action
    ca = state.get("ca_draft")
    assert ca is not None
    assert "reversed" in ca.immediate_action.lower() or "reconciliation" in ca.immediate_action.lower() or "supplier" in ca.immediate_action.lower()


def test_cross_finding_semantic_architecture_domains():
    """Section 14: Cross-finding semantic resolution consistency across 5 domains."""
    # A. Equipment operated outside validated range
    info_a = extract_semantic_subject("Equipment was operated outside its validated range.")
    assert "equipment" in info_a.affected_object.lower() or "operating" in info_a.affected_object.lower() or "range" in info_a.affected_object.lower()

    # B. Notification failure
    info_b = extract_semantic_subject("Three operators did not receive the revised SOP notification.")
    assert is_actor_noun(info_b.actor)
    assert "notification" in info_b.affected_object.lower()

    # C. Checklist noncompliance
    info_c = extract_semantic_subject("Four employees failed to complete the revised inspection checklist.")
    assert is_actor_noun(info_c.actor)
    assert "checklist" in info_c.affected_object.lower()

    # D. Duplicate payment
    info_d = extract_semantic_subject("Duplicate payment of ₹125,000 was made to a supplier.")
    assert "duplicate" in info_d.affected_object.lower() and "payment" in info_d.affected_object.lower()

    # E. Batch scrap
    info_e = extract_semantic_subject("Batch worth ₹500,000 was scrapped due to a confirmed equipment failure.")
    assert "batch" in info_e.affected_object.lower() or "scrap" in info_e.affected_object.lower()

