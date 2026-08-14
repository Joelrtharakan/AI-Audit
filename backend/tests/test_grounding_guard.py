"""Structural (code-level) tests for the grounding/causal-inference guard added
to the /investigate pipeline (app/agent/grounding_guard.py), wired into rca.py,
impact.py, capa.py, and ca_draft_generator.py.

These simulate a *malicious/careless LLM* that ignores its prompt instructions
and returns another case's entities, or performs the "reported + reported =
established cause" invalid inference. The point is that the guard catches this
in code regardless of what the model outputs -- these tests do not rely on the
prompt behaving, only on the enforcement layer.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.models.agent import AgentTraceStep, EvidenceItem, EvidenceStatus, InvestigateRequest


def _build_state(finding_text: str, evidence_ledger: list[EvidenceItem] | None = None) -> dict:
    return {
        "request": InvestigateRequest(finding_text=finding_text),
        "iteration_count": 0,
        "tool_call_count": 0,
        "critic_iteration": 0,
        "observation_quality": None,
        "extraction": None,
        "investigation_plan": None,
        "needs_investigation": False,
        "planned_tools": [],
        "completed_tools": [],
        "current_tool": None,
        "tool_results": {},
        "evidence_ledger": evidence_ledger or [],
        "evidence_gaps": [],
        "root_cause": None,
        "contributing_factors": [],
        "five_why": None,
        "impact_assessment": None,
        "capa_analysis": None,
        "critic_approved": False,
        "critic_feedback": None,
        "critic_send_back": False,
        "report": None,
        "ca_draft": None,
        "final_state": None,
        "trace": [AgentTraceStep.ok("Test started")],
        "errors": [],
    }


WORKLOAD_FINDING = (
    "Five production records contained incomplete entries. Operators reported "
    "unusually high workload during the affected shifts. The supervisor stated "
    "that the operators were insufficiently trained on the revised documentation "
    "procedure."
)


# ---------------------------------------------------------------------------
# 1. Causal-inference guard: reported + reported must never become an
#    established/causal claim.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_reported_facts_cannot_become_causal_claim():
    """The exact invalid pattern from the bug report: workload (REPORTED) +
    training (REPORTED) must never be combined into 'X and Y caused Z'."""
    from app.agent.nodes.rca import root_cause_node

    state = _build_state(WORKLOAD_FINDING)

    llm_response = json.dumps({
        "root_cause": {
            "status": "STATED_UNVERIFIED",
            "category": "MAN",
            "statement": "High workload and insufficient training caused the incomplete entries.",
            "leading_hypothesis": "High workload and insufficient training caused the incomplete entries.",
            "candidate_hypotheses": [],
            "narrative": "High workload and insufficient training caused the incomplete entries.",
            "confidence": "MEDIUM",
            "evidence_status": "REPORTED",
        },
        "contributing_factors": [],
        "five_why": {"steps": [], "is_complete": False, "status_note": "INCOMPLETE"},
    })

    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client

        result = await root_cause_node(state)

    root_cause = result["root_cause"]
    assert root_cause.status == "NOT_ESTABLISHED"
    assert root_cause.statement is None
    assert "caused" not in (root_cause.narrative or "").lower()
    assert root_cause.leading_hypothesis is None


@pytest.mark.asyncio
async def test_causal_claim_allowed_when_backed_by_verified_evidence():
    """The guard must not be so aggressive it blocks a genuinely established
    cause — Test 7/23 from the spec (explicit confirmed root cause)."""
    from app.agent.nodes.rca import root_cause_node

    finding = (
        "Investigation confirmed that the record-review checklist omitted the "
        "verification step introduced in the latest procedure revision. The "
        "checklist owner confirmed the checklist was not updated when the "
        "procedure was revised."
    )
    verified_evidence = [
        EvidenceItem(
            claim="The record-review checklist omitted the verification step introduced in the latest revision.",
            source="AUDITOR_FINDING",
            status=EvidenceStatus.VERIFIED,
        )
    ]
    state = _build_state(finding, verified_evidence)

    llm_response = json.dumps({
        "root_cause": {
            "status": "VERIFIED",
            "category": "METHOD",
            "statement": "The checklist was not updated when the procedure was revised, which caused the missing verification step.",
            "leading_hypothesis": None,
            "candidate_hypotheses": [],
            "narrative": "The checklist was not updated when the procedure was revised, which caused the missing verification step.",
            "confidence": "HIGH",
            "evidence_status": "VERIFIED",
        },
        "contributing_factors": [],
        "five_why": {"steps": [], "is_complete": True, "status_note": "COMPLETE"},
    })

    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client

        result = await root_cause_node(state)

    root_cause = result["root_cause"]
    assert root_cause.status == "VERIFIED"
    assert "caused" in (root_cause.narrative or "").lower()


@pytest.mark.asyncio
async def test_supported_status_requires_verified_evidence():
    """SUPPORTED ('strongly supported by verified facts') must be downgraded
    if the ledger has no VERIFIED item — matches VERIFIED's existing gate."""
    from app.agent.nodes.rca import root_cause_node

    state = _build_state(WORKLOAD_FINDING)  # no evidence ledger passed in

    llm_response = json.dumps({
        "root_cause": {
            "status": "SUPPORTED",
            "category": None,
            "statement": None,
            "leading_hypothesis": None,
            "candidate_hypotheses": [],
            "narrative": "Root cause is strongly supported.",
            "confidence": "MEDIUM",
            "evidence_status": "REPORTED",
        },
        "contributing_factors": [],
        "five_why": {"steps": [], "is_complete": False, "status_note": "INCOMPLETE"},
    })

    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client

        result = await root_cause_node(state)

    assert result["root_cause"].status == "STATED_UNVERIFIED"


# ---------------------------------------------------------------------------
# 2. Entity/number grounding guard: simulates an LLM that returns another
#    case's entities (SOP-OPS-014 / three operators / 30 days) for a finding
#    that never mentioned them.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rca_strips_entities_from_a_different_case():
    """Even if the model output entities from a different (training) case,
    the grounding guard must catch and remove them before they reach a
    workload/records finding's output."""
    from app.agent.nodes.rca import root_cause_node

    state = _build_state(WORKLOAD_FINDING)

    llm_response = json.dumps({
        "root_cause": {
            "status": "STATED_UNVERIFIED",
            "category": "MAN",
            "statement": None,
            "leading_hypothesis": None,
            "candidate_hypotheses": [
                {
                    "id": "H1",
                    "name": "AUTHORIZATION_CONTROL",
                    "statement": "Three operators performed the revised inspection procedure without SOP-OPS-014 completion.",
                    "status": "POSSIBLE",
                    "evidence_needed": "Authorization records",
                },
                {
                    "id": "H2",
                    "name": "WORKLOAD_PRESSURE",
                    "statement": "Unusually high workload may have contributed to the incomplete entries.",
                    "status": "POSSIBLE",
                    "evidence_needed": "Staffing and production volume records",
                },
            ],
            "narrative": "Root cause not established.",
            "confidence": "LOW",
            "evidence_status": "UNKNOWN",
        },
        "contributing_factors": [],
        "five_why": {"steps": [], "is_complete": False, "status_note": "INCOMPLETE"},
    })

    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client

        result = await root_cause_node(state)

    root_cause = result["root_cause"]
    names = [h.name for h in root_cause.candidate_hypotheses]
    # The contaminated hypothesis (references SOP-OPS-014, a different case's
    # entity) must be dropped; the clean workload hypothesis must survive.
    assert "AUTHORIZATION_CONTROL" not in names
    assert "WORKLOAD_PRESSURE" in names
    full_text = json.dumps([h.model_dump() for h in root_cause.candidate_hypotheses])
    assert "SOP-OPS-014" not in full_text


@pytest.mark.asyncio
async def test_ca_draft_strips_entities_from_a_different_case():
    """The final CA draft fields (what actually gets written to the form)
    must not carry an entity from a different case even if every upstream
    node somehow let it through."""
    from app.agent.nodes.ca_draft_generator import ca_draft_generator_node
    from app.models.agent import RootCauseAnalysis

    state = _build_state(WORKLOAD_FINDING)
    state["root_cause"] = RootCauseAnalysis(
        status="NOT_ESTABLISHED",
        category="TO_BE_CONFIRMED",
        narrative="Root cause not established for this finding.",
    )

    llm_response = json.dumps({
        "immediate_action": "Prevent the three operators from performing the revised inspection procedure.",
        "root_cause": "Root cause not established.",
        "root_cause_category": "TO_BE_CONFIRMED",
        "preventive_action": "Verify SOP-OPS-014 training completion before authorization.",
        "impact_analysis": "Impact requires assessment.",
    })

    with patch("app.agent.nodes.ca_draft_generator.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client

        result = await ca_draft_generator_node(state)

    ca_draft = result["ca_draft"]
    assert ca_draft is not None
    for field in ("immediate_action", "preventive_action"):
        value = getattr(ca_draft, field)
        assert "SOP-OPS-014" not in value
        assert "three operators" not in value.lower()


# ---------------------------------------------------------------------------
# 3. Full cross-case sequence: run case A (training) then case B (workload)
#    through root_cause_node with mocked LLM responses that are individually
#    correct for their own case, and confirm B's output contains none of A's
#    entities (the adversarial test from spec section 20/21).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_case_sequence_no_entity_bleed():
    from app.agent.nodes.rca import root_cause_node

    case_a_text = (
        "Three operators were observed performing the revised inspection "
        "procedure without having completed mandatory training. The SOP "
        "revision was issued 30 days ago."
    )
    case_a_response = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED",
            "category": None,
            "statement": None,
            "leading_hypothesis": None,
            "candidate_hypotheses": [
                {"id": "H1", "name": "TRAINING_ASSIGNMENT", "statement": "Training may not have been assigned to the three operators.", "status": "POSSIBLE", "evidence_needed": "Training assignment records"},
            ],
            "narrative": "Root cause not established for the training finding.",
            "confidence": "LOW",
            "evidence_status": "UNKNOWN",
        },
        "contributing_factors": [],
        "five_why": {"steps": [], "is_complete": False, "status_note": "INCOMPLETE"},
    })

    state_a = _build_state(case_a_text)
    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = case_a_response
        mock_get_client.return_value = mock_client
        result_a = await root_cause_node(state_a)

    assert "three operators" in (result_a["root_cause"].candidate_hypotheses[0].statement or "").lower()

    # Case B's LLM call returns ONLY case B content (a well-behaved model) —
    # the isolation guarantee under test is that nothing from case A's fresh
    # AgentState/prompt leaks in, since each call builds a brand-new state.
    state_b = _build_state(WORKLOAD_FINDING)
    case_b_response = json.dumps({
        "root_cause": {
            "status": "NOT_ESTABLISHED",
            "category": None,
            "statement": None,
            "leading_hypothesis": None,
            "candidate_hypotheses": [
                {"id": "H1", "name": "WORKLOAD_PRESSURE", "statement": "Unusually high workload may have contributed to the incomplete entries.", "status": "POSSIBLE", "evidence_needed": "Staffing and production volume records"},
            ],
            "narrative": "Root cause not established for the workload finding.",
            "confidence": "LOW",
            "evidence_status": "UNKNOWN",
        },
        "contributing_factors": [],
        "five_why": {"steps": [], "is_complete": False, "status_note": "INCOMPLETE"},
    })
    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = case_b_response
        mock_get_client.return_value = mock_client
        result_b = await root_cause_node(state_b)

    b_text = json.dumps(result_b["root_cause"].model_dump()).lower()
    for marker in ("three operators", "30 days", "inspection procedure"):
        assert marker not in b_text


# ---------------------------------------------------------------------------
# 4. Final report-level sweep: defense-in-depth catches a violation even if it
#    were (hypothetically) injected directly into state, bypassing the
#    per-node guards entirely.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_generator_final_sweep_catches_bypassed_violation():
    from app.agent.nodes.report_generator import generate_report_node
    from app.models.agent import CandidateHypothesis, RootCauseAnalysis

    state = _build_state(WORKLOAD_FINDING)
    # Simulate a violation that slipped past the per-node guards by writing
    # directly into state, as if some future code path bypassed rca.py.
    state["root_cause"] = RootCauseAnalysis(
        status="STATED_UNVERIFIED",
        category="MAN",
        narrative="Training matrix showed SOP-OPS-014 was not completed.",
        candidate_hypotheses=[
            CandidateHypothesis(
                id="H1", name="AUTHORIZATION_CONTROL",
                statement="Three operators lacked SOP-OPS-014 authorization.",
                evidence_needed="Authorization records",
            )
        ],
    )

    result = await generate_report_node(state)
    report = result["report"]

    assert "SOP-OPS-014" not in (report.root_cause.narrative or "")
    assert report.root_cause.status == "NOT_ESTABLISHED"
    assert report.root_cause.candidate_hypotheses == []
    assert any("FINAL SWEEP" in w for w in
               [t.model_dump()["message"] for t in result["trace"] if t.model_dump()["icon"] == "⚠"])


# ---------------------------------------------------------------------------
# 5. Contradictory evidence: conflicting claims about the same subject must
#    never be silently resolved in favor of one side.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contradictory_evidence_is_not_resolved_either_way():
    """Supervisor says training was completed; the training system says no
    completion record exists. The agent must not conclude either 'trained'
    or 'not trained' -- it must flag the conflict."""
    from app.agent.nodes.rca import root_cause_node

    finding = (
        "The supervisor stated that training was completed. Training records "
        "showed no completion. Signed attendance sheets were reportedly "
        "available but were not provided."
    )
    evidence_ledger = [
        EvidenceItem(claim="The supervisor stated that training was completed.", source="AUDITOR_FINDING", status=EvidenceStatus.REPORTED),
        EvidenceItem(claim="Training records showed no completion of the training.", source="AUDITOR_FINDING", status=EvidenceStatus.VERIFIED),
    ]
    state = _build_state(finding, evidence_ledger)

    llm_response = json.dumps({
        "root_cause": {
            "status": "VERIFIED",
            "category": "MAN",
            "statement": "Training was completed as stated by the supervisor.",
            "leading_hypothesis": None,
            "candidate_hypotheses": [],
            "narrative": "Training was completed as stated by the supervisor.",
            "confidence": "HIGH",
            "evidence_status": "VERIFIED",
        },
        "contributing_factors": [],
        "five_why": {"steps": [], "is_complete": False, "status_note": "INCOMPLETE"},
    })

    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await root_cause_node(state)

    root_cause = result["root_cause"]
    assert root_cause.status == "NOT_ESTABLISHED"
    narrative = (root_cause.narrative or "").lower()
    assert "conflict" in narrative
    # The LLM's one-sided claim must have been replaced, not passed through
    assert narrative != "training was completed as stated by the supervisor."


# ---------------------------------------------------------------------------
# 6. Recurrence / effectiveness: "recorded as completed" must never become
#    "was effective".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_previous_capa_completion_is_not_effectiveness():
    from app.agent.nodes.rca import root_cause_node

    finding = (
        "The current audit identified missing temperature records. The same "
        "issue was identified during the previous audit, and the previous "
        "corrective action was recorded as completed."
    )
    state = _build_state(finding)

    llm_response = json.dumps({
        "root_cause": {
            "status": "STATED_UNVERIFIED",
            "category": None,
            "statement": "The previous corrective action was effective and prevented recurrence.",
            "leading_hypothesis": None,
            "candidate_hypotheses": [],
            "narrative": "The previous corrective action was effective and prevented recurrence.",
            "confidence": "MEDIUM",
            "evidence_status": "REPORTED",
        },
        "contributing_factors": [],
        "five_why": {"steps": [], "is_complete": False, "status_note": "INCOMPLETE"},
    })

    with patch("app.agent.nodes.rca.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = llm_response
        mock_get_client.return_value = mock_client
        result = await root_cause_node(state)

    root_cause = result["root_cause"]
    assert root_cause.status == "NOT_ESTABLISHED"
    narrative = (root_cause.narrative or "").lower()
    # The LLM's unsupported claim of effectiveness must not survive verbatim
    assert narrative != "the previous corrective action was effective and prevented recurrence."
    assert "recorded" in narrative


def test_recurrence_detection_and_effectiveness_claim_helpers():
    from app.agent.grounding_guard import claims_unsupported_effectiveness, is_recurrence_finding

    recurrence_finding = "The previous corrective action was recorded as completed."
    non_recurrence_finding = "Five production records contained incomplete entries."

    assert is_recurrence_finding(recurrence_finding)
    assert not is_recurrence_finding(non_recurrence_finding)

    assert claims_unsupported_effectiveness("The action was effective.", recurrence_finding)
    assert not claims_unsupported_effectiveness("The action was recorded as completed.", recurrence_finding)
    # Never triggers outside a recurrence context
    assert not claims_unsupported_effectiveness("The action was effective.", non_recurrence_finding)
