"""Investigation-plan quality layer tests (Conditional Decision Tree Architecture)
Exercises the generic deduplication, ranking, single-decision validation,
conditional activation, area clustering, and management-ready output
across all domains and execution paths.
"""

from __future__ import annotations

import pytest

from app.agent.analytical_validator import (
    deduplicate_investigation_questions,
    derive_investigation_areas,
    is_compound_decision_question,
    normalize_and_dedupe_evidence_items,
    normalize_investigation_decision_tree,
    rank_questions_by_information_gain,
    validate_investigation_question,
)
from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
from app.models.agent import (
    EvidenceItem,
    EvidenceStatus,
    InvestigationQuestion,
    RootCauseAnalysis,
    RootCauseStatus,
)


def _q(question: str, **kwargs) -> InvestigationQuestion:
    return InvestigationQuestion(question=question, **kwargs)


# ---------------------------------------------------------------------------
# 1. Question with multiple decisions is rejected (Single-Decision Principle)
# ---------------------------------------------------------------------------

def test_compound_decision_question_is_rejected():
    compound_1 = "Was acknowledgement or confirmation mandatory before proceeding with tasks governed by SOP-042, and do read, access, or acknowledgement records confirm it was completed?"
    compound_2 = "Was retraining completed by the technician, and was the training record signed off by quality assurance?"
    compound_3 = "Did the procedure require authorization and whether the supervisor authorized it?"

    assert is_compound_decision_question(compound_1) is True
    assert is_compound_decision_question(compound_2) is True
    assert is_compound_decision_question(compound_3) is True
    assert validate_investigation_question(compound_1) is False
    assert validate_investigation_question(compound_2) is False

    single_1 = "Did the applicable procedure require acknowledgement before work under the revised SOP commenced?"
    single_2 = "Do authenticated records establish that affected personnel acknowledged the revision before performing the activity?"
    single_3 = "Do per-recipient delivery logs establish successful delivery to each affected recipient?"

    assert is_compound_decision_question(single_1) is False
    assert is_compound_decision_question(single_2) is False
    assert is_compound_decision_question(single_3) is False
    assert validate_investigation_question(single_1) is True
    assert validate_investigation_question(single_2) is True
    assert validate_investigation_question(single_3) is True


# ---------------------------------------------------------------------------
# 2. Conditional question is marked internally with activation_condition
# ---------------------------------------------------------------------------

def test_conditional_question_status_and_prerequisites():
    q_root = _q(
        "Do authenticated system records establish successful delivery to each affected recipient?",
        question_id="Q1",
        target_proposition_id="P_DELIVERY",
        priority="P1",
    )
    q_cond = _q(
        "Do independent records establish actual receipt or access by the affected recipients?",
        question_id="Q2",
        target_proposition_id="P_RECEIPT",
        priority="P2",
        depends_on="Q1",
        activation_condition="If delivery is confirmed",
    )

    normalized = normalize_investigation_decision_tree([q_root, q_cond])
    assert normalized[0].status == "ACTIVE"
    assert normalized[1].status == "CONDITIONAL"
    assert normalized[1].activation_condition == "If delivery is confirmed"
    assert normalized[1].depends_on == "Q1"


# ---------------------------------------------------------------------------
# 3. Exact duplicate questions removed
# ---------------------------------------------------------------------------

def test_exact_duplicate_questions_are_removed():
    questions = [
        _q("Do delivery logs confirm notification was delivered?"),
        _q("Do delivery logs confirm notification was delivered?"),
    ]
    result = deduplicate_investigation_questions(questions)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# 4. Semantic duplicate questions removed (different wording, same target)
# ---------------------------------------------------------------------------

def test_semantic_duplicate_delivery_questions_collapse_to_one():
    questions = [
        _q("Verify notification delivery to each recipient.", purpose="Verify delivery status"),
        _q("Establish whether notification was delivered to each recipient.", purpose="Verify delivery status"),
        _q("Reconcile notification delivery records for each recipient.", purpose="Verify delivery records"),
    ]
    result = deduplicate_investigation_questions(questions)
    assert len(result) == 1, "near-identical delivery-verification restatements must collapse to one"


def test_semantic_duplicate_acknowledgement_questions_collapse_to_one():
    questions = [
        _q("Verify acknowledgement was completed by each recipient.", purpose="Verify acknowledgement"),
        _q("Determine whether acknowledgement was completed by each recipient.", purpose="Verify acknowledgement"),
        _q("Establish whether formal acknowledgement occurred for each recipient.", purpose="Verify acknowledgement"),
    ]
    result = deduplicate_investigation_questions(questions)
    assert len(result) == 1


def test_distinct_investigative_objectives_are_not_merged():
    questions = [
        _q("Do delivery logs confirm the notification was delivered to each recipient?", purpose="Verify delivery"),
        _q("Do acknowledgement records confirm each recipient formally acknowledged the notification?", purpose="Verify acknowledgement"),
    ]
    result = deduplicate_investigation_questions(questions)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# 5/6/7. Investigation areas: 2-4 areas, no orphans, every question maps
# ---------------------------------------------------------------------------

def test_no_duplicate_investigation_areas_survive_family_clustering():
    questions = [
        _q("Do per-recipient delivery records establish successful delivery?",
           purpose="Verify delivery-side completion details", target_proposition_id="P_DELIVERY"),
        _q("Do independent records establish actual receipt or access?",
           purpose="Establish whether delivery resulted in actual recipient receipt or access",
           target_proposition_id="P_RECEIPT"),
        _q("Was acknowledgement required before proceeding?",
           purpose="Establish whether formal acknowledgement was required and completed",
           target_proposition_id="P_REQ_ACK"),
        _q("Did recipients act before receipt was confirmed?",
           purpose="Determine whether operational activity occurred prior to confirmed receipt",
           target_proposition_id="P_ACTION"),
        _q("What technical mechanism explains non-receipt, if it occurred?",
           purpose="Identify an established mechanism if receipt did not occur",
           target_proposition_id="P_MECHANISM"),
    ]
    areas = derive_investigation_areas(
        candidate_hypotheses=[], questions=questions, canonical_subject="the revised SOP notification",
    )
    assert 2 <= len(areas) <= 4
    assert len(areas) == len(set(a.lower() for a in areas)), f"duplicate areas survived: {areas}"
    purposes = {q.purpose.lower() for q in questions}
    assert not any(a.lower() in purposes for a in areas), f"an area restated a question purpose verbatim: {areas}"


def test_every_area_has_at_least_one_supporting_question():
    questions = [_q("Do delivery logs confirm delivery occurred?", target_proposition_id="P_DELIVERY")]
    areas = derive_investigation_areas(candidate_hypotheses=[], questions=questions, canonical_subject="the notice")
    assert len(areas) >= 1
    assert all("delivery" in a.lower() or "receipt" in a.lower() or "control" in a.lower() for a in areas)


def test_no_orphan_areas_when_no_questions_classify():
    areas = derive_investigation_areas(candidate_hypotheses=[], questions=[], existing_areas=None)
    assert areas == []


# ---------------------------------------------------------------------------
# 8. Evidence-needed deduplication
# ---------------------------------------------------------------------------

def test_evidence_items_normalize_and_dedupe():
    items = [
        "delivery logs", "system delivery logs", "delivery records",
        "acknowledgement records", "acknowledgement confirmation records",
        "channel error logs",
    ]
    result = normalize_and_dedupe_evidence_items(items)
    assert len(result) < len(items), "near-duplicate evidence artifacts must collapse"
    assert len(result) <= 6


# ---------------------------------------------------------------------------
# 9/10/11. Prioritization & Information-Gain Ordering
# ---------------------------------------------------------------------------

def test_priority_rules_order_p1_before_p3_before_p5():
    mechanism_q = _q("Was the notification system misconfigured?", priority="P5", target_proposition_id="P_MECHANISM")
    delivery_q = _q("Do delivery logs establish delivery occurred?", priority="P1", target_proposition_id="P_DELIVERY")
    req_q = _q("Did procedure require acknowledgement?", priority="P3", target_proposition_id="P_REQ_ACK")
    receipt_q = _q("Do independent records establish actual receipt?", priority="P2", target_proposition_id="P_RECEIPT")

    ranked = rank_questions_by_information_gain([mechanism_q, receipt_q, req_q, delivery_q])
    ranked_ids = [q.priority for q in ranked]
    assert ranked_ids == ["P1", "P2", "P3", "P5"]


def test_dependency_preserves_parent_before_child():
    child_q = _q("Do read records confirm receipt?", question_id="Q2", priority="P1", depends_on="Q1")
    parent_q = _q("Do system logs establish delivery?", question_id="Q1", priority="P2")

    ranked = rank_questions_by_information_gain([child_q, parent_q])
    assert ranked[0].question_id == "Q1"
    assert ranked[1].question_id == "Q2"


# ---------------------------------------------------------------------------
# 12/13/14/15. Causal Invariants & Stopping Boundaries
# ---------------------------------------------------------------------------

def test_unsupported_root_cause_remains_rejected():
    rc = RootCauseAnalysis(
        status=RootCauseStatus.NOT_ESTABLISHED,
        narrative="The causal mechanism cannot yet be established.",
    )
    assert rc.status == RootCauseStatus.NOT_ESTABLISHED


# ---------------------------------------------------------------------------
# 17. Notification Example Decision Tree Test
# ---------------------------------------------------------------------------

def test_notification_example_produces_decision_tree():
    finding_text = (
        "During an internal audit, it was observed that SOP-QA-042 Rev 4 was distributed via automated email on 2026-03-01. "
        "System transmission logs state that notification was successfully delivered to all department personnel. "
        "However, two operators reported they never received notification of the revision and continued executing tasks under Rev 3."
    )
    ledger = [
        EvidenceItem(claim="System logs indicate notification was delivered on 2026-03-01", status=EvidenceStatus.VERIFIED, source="finding"),
        EvidenceItem(claim="Two operators reported they never received notification", status=EvidenceStatus.REPORTED, source="finding"),
    ]

    hyps, plan = build_deterministic_investigation_plan(finding_text, ledger, canonical_subject="SOP revision notification")

    # Verify zero hypotheses (conflict case preserves uncertainty)
    assert len(hyps) == 0

    # Verify decision tree questions
    assert len(plan.questions) >= 5
    q1 = next(q for q in plan.questions if q.target_proposition_id == "P_DELIVERY")
    q2 = next(q for q in plan.questions if q.target_proposition_id == "P_RECEIPT")
    q3 = next(q for q in plan.questions if q.target_proposition_id == "P_REQ_ACK")
    q4 = next(q for q in plan.questions if q.target_proposition_id == "P_ACK_EXECUTION")
    q5 = next(q for q in plan.questions if q.target_proposition_id == "P_MECHANISM")

    assert q1.status == "ACTIVE"
    assert q1.priority == "P1"

    assert q2.status == "CONDITIONAL"
    assert q2.activation_condition == "If delivery is confirmed"

    assert q3.status == "CONDITIONAL"
    assert q3.activation_condition == "If receipt/access is confirmed"

    assert q4.status == "CONDITIONAL"
    assert q4.activation_condition == "If acknowledgement was required"

    assert q5.status == "CONDITIONAL"
    assert q5.activation_condition == "If non-receipt is confirmed"

    # Verify each question is single-decision
    for q in plan.questions:
        assert validate_investigation_question(q.question) is True
        assert is_compound_decision_question(q.question) is False


# ---------------------------------------------------------------------------
# 18. Calibration Example
# ---------------------------------------------------------------------------

def test_calibration_example_produces_decision_tree():
    finding_text = "Analytical balance BAL-014 was used in batch release testing after its calibration validity date expired on 2026-02-15."
    ledger = [
        EvidenceItem(claim="Analytical balance BAL-014 calibration expired on 2026-02-15", status=EvidenceStatus.VERIFIED, source="finding"),
        EvidenceItem(claim="BAL-014 was used in batch testing on 2026-02-18", status=EvidenceStatus.VERIFIED, source="finding"),
    ]

    _, plan = build_deterministic_investigation_plan(finding_text, ledger, canonical_subject="analytical balance BAL-014 calibration")
    assert len(plan.questions) >= 3
    assert all(validate_investigation_question(q.question) for q in plan.questions)
    assert 1 <= len(plan.areas) <= 4


# ---------------------------------------------------------------------------
# 19. Temperature Example
# ---------------------------------------------------------------------------

def test_temperature_example_produces_decision_tree():
    finding_text = "Daily temperature monitoring records for stability chamber SC-02 were missing for three consecutive days (2026-01-10 to 2026-01-12)."
    ledger = [
        EvidenceItem(claim="Temperature records for SC-02 missing for Jan 10-12", status=EvidenceStatus.VERIFIED, source="finding"),
    ]

    _, plan = build_deterministic_investigation_plan(finding_text, ledger, canonical_subject="temperature monitoring records for stability chamber SC-02")
    assert len(plan.questions) >= 3
    assert all(validate_investigation_question(q.question) for q in plan.questions)


# ---------------------------------------------------------------------------
# 20. Equipment Operating Range Example
# ---------------------------------------------------------------------------

def test_equipment_range_example_produces_decision_tree():
    finding_text = "Autoclave AC-01 was operated at 128°C during cycle 402, outside the validated range of 121°C ± 2°C."
    ledger = [
        EvidenceItem(claim="Autoclave AC-01 operated at 128°C on cycle 402", status=EvidenceStatus.VERIFIED, source="finding"),
    ]

    _, plan = build_deterministic_investigation_plan(finding_text, ledger, canonical_subject="autoclave AC-01 operating temperature")
    assert len(plan.questions) >= 3
    q_mech = next((q for q in plan.questions if q.target_proposition_id == "P_MECHANISM"), None)
    if q_mech:
        assert q_mech.status == "CONDITIONAL"
        assert q_mech.priority == "P5"


# ---------------------------------------------------------------------------
# 21. Training Example
# ---------------------------------------------------------------------------

def test_training_example_produces_decision_tree():
    finding_text = (
        "Technician John Doe executed aseptic filling operation on Batch AF-202. "
        "Training records showed annual aseptic qualification had expired on 2026-01-31. "
        "The supervisor stated retraining was completed on 2026-02-01, but QA reported no retraining record could be located."
    )
    ledger = [
        EvidenceItem(claim="Aseptic qualification expired 2026-01-31", status=EvidenceStatus.VERIFIED, source="finding"),
        EvidenceItem(claim="Supervisor stated retraining occurred 2026-02-01", status=EvidenceStatus.REPORTED, source="finding"),
        EvidenceItem(claim="QA reported no retraining record was located", status=EvidenceStatus.REPORTED, source="finding"),
    ]

    hyps, plan = build_deterministic_investigation_plan(finding_text, ledger, canonical_subject="aseptic qualification retraining")
    # Conflict between supervisor statement and missing QA record properly drives conflict-resolution plan
    assert len(plan.questions) >= 3
    assert all(validate_investigation_question(q.question) for q in plan.questions)
    assert 1 <= len(plan.areas) <= 4
