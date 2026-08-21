"""Regression and verification tests for Generalized Semantic Role & Subject-Predicate Preservation.

Ensures that:
1. Actor entities (e.g. employee EMP-104, operator, technician) are distinct from activities/events (training, inspection).
2. Topic words derived from conflicting claims are never actor nouns.
3. Candidate hypotheses (H1..H4), investigation questions (Q1..Q5), 5-Why steps, and CAPA actions preserve grammatical and semantic roles.
4. Proposition derivation correctly identifies negated clauses without converting actors to activities.
5. Invariant INV-ROLE-001 enforces semantic role preservation.
"""

import pytest
import re
from app.services.semantic_subject import (
    extract_conflict_topic,
    split_topic_and_tail,
    resolve_deviation,
    extract_semantic_subject,
    is_actor_noun,
)
from app.agent.claim_extractor import (
    extract_claims,
    detect_evidence_conflicts,
    _derive_conflict_proposition,
)
from app.agent.nodes.plan_investigation_fallback import (
    build_deterministic_investigation_plan,
    build_conditional_capa_actions,
)
from app.agent.nodes.five_why_fallback import build_deterministic_five_why
from app.models.agent import (
    InvestigateRequest,
    CanonicalFindingState,
    EvidenceItem,
    EvidenceStatus,
    CandidateHypothesis,
)
from app.agent.invariants import evaluate_all_invariants, _check_semantic_role_preservation


def test_conflict_topic_never_extracts_actor_noun():
    """Test that extract_conflict_topic extracts activities, not actors."""
    c1 = "The training management system shows that employee EMP-104 completed training on 5 August."
    c2 = "The employee states that they did not attend the training, and the signed attendance record is unavailable."
    
    topic = extract_conflict_topic(c1, c2, fallback_subject="training for 5 August for EMP-104")
    assert topic == "training"
    assert not is_actor_noun(topic)
    assert topic != "employee"


def test_derive_conflict_proposition_preserves_subject_predicate():
    """Test that _derive_conflict_proposition creates a grammatically correct affirmative proposition."""
    claims = extract_claims(
        "The training management system shows that employee EMP-104 completed training on 5 August. "
        "The employee states that they did not attend the training, and the signed attendance record is unavailable."
    )
    assert len(claims) >= 2
    prop = _derive_conflict_proposition(claims[0], claims[1])
    assert "Whether the employee attended the training" in prop
    assert not re.search(r"\bWhether employee was completed\b", prop)


def test_split_topic_and_tail_strips_metadata():
    """Test split_topic_and_tail correctly strips leading topic and dates without leaving dangling words."""
    tail = split_topic_and_tail("training for 5 August for EMP-104", "training")
    assert tail == "EMP-104"

    tail2 = split_topic_and_tail("training for the revised procedure", "training")
    assert tail2 == "the revised procedure"


def test_deterministic_plan_preserves_semantic_roles():
    """Test candidate hypotheses, questions, 5-Why, and CAPA generated for conflicting training finding."""
    text = (
        "The training management system shows that employee EMP-104 completed training on 5 August. "
        "The employee states that they did not attend the training, and the signed attendance record is unavailable."
    )
    resolved = resolve_deviation(text)
    claims = extract_claims(text)
    conflicts = detect_evidence_conflicts(claims)
    
    canonical = CanonicalFindingState(
        raw_finding=text,
        finding_subject=resolved.subject,
        observed_deviation=resolved.deviation,
        affected_process=resolved.affected_process,
        primary_uncertainty="CONFLICTING_EVIDENCE",
        evidence_conflicts=conflicts,
        is_actionable=True,
    )
    
    ledger = [
        EvidenceItem(id="C1", source="System", claim=claims[0].text, status=EvidenceStatus.VERIFIED),
        EvidenceItem(id="C2", source="Auditor", claim=claims[1].text, status=EvidenceStatus.REPORTED),
    ]

    hyps, plan = build_deterministic_investigation_plan(
        text,
        ledger,
        canonical_subject=resolved.subject,
        canonical_state=canonical,
    )

    # 1. Hypothesis check
    assert len(hyps) >= 1
    h1 = hyps[0]
    assert h1.name == "TRAINING_NOT_COMPLETED"
    assert "Required training" in h1.statement
    assert "Required employee" not in h1.statement
    assert "may not have been completed" in h1.statement

    # 2. Investigation questions check
    q_texts = [q.question for q in plan.questions]
    for q in q_texts:
        assert "completed the required employee" not in q
        assert "authenticated employee record" not in q

    # 3. 5-Why check
    fw = build_deterministic_five_why(
        text,
        ledger,
        canonical_subject=resolved.subject,
        canonical_state=canonical,
    )
    assert len(fw.steps) >= 1
    assert "employee" not in fw.steps[1].question if len(fw.steps) > 1 else True

    # 4. CAPA actions check
    capa_actions = build_conditional_capa_actions(hyps, resolved.subject, "training")
    for a in capa_actions:
        assert "execution of employee" not in a.recommended_action
        assert "Required employee" not in a.recommended_action


def test_invariant_inv_role_001_catches_corrupted_roles():
    """Test that INV-ROLE-001 catches corrupted candidate hypotheses or questions."""
    corrupted_state = {
        "canonical_finding_state": CanonicalFindingState(
            raw_finding="test",
            finding_subject="employee",
            observed_deviation="employee — status unconfirmed",
            is_actionable=True,
        ),
        "root_cause": type("RC", (), {
            "candidate_hypotheses": [
                CandidateHypothesis(
                    id="H1",
                    name="EMPLOYEE_NOT_COMPLETED",
                    statement="Required employee for the applicable requirement may not have been completed.",
                    evidence_needed="Training records",
                )
            ]
        })(),
    }
    valid, violation = _check_semantic_role_preservation(corrupted_state)
    assert not valid
    assert "treats actor noun" in violation or "corrupts actor entity" in violation


@pytest.mark.asyncio
async def test_full_pipeline_preserves_semantic_roles_end_to_end():
    """Test full agent graph execution on conflicting training finding."""
    from app.agent.graph import build_agent_graph
    
    text = (
        "The training management system shows that employee EMP-104 completed training on 5 August. "
        "The employee states that they did not attend the training, and the signed attendance record is unavailable."
    )
    req = InvestigateRequest(finding_text=text)
    state = {
        "request": req,
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "evidence_ledger": [],
        "errors": [],
        "trace": [],
    }
    graph = build_agent_graph()
    res = await graph.ainvoke(state)

    rc = res.get("root_cause")
    inv = res.get("investigation_plan")
    
    assert rc is not None
    for h in rc.candidate_hypotheses:
        assert not h.name.startswith("EMPLOYEE_")
        assert "Required employee" not in h.statement
        
    for q in inv.questions:
        assert "completed the required employee" not in q.question
        
    valid, violations = evaluate_all_invariants(res)
    assert valid, f"Invariants failed: {violations}"
