"""Phase 3, Section 23: API fail-closed verification.

Source audit found `validation_failure` (set by
final_evidence_verification_node on a BLOCKER invariant violation or a
FAIL-grade structural quality score) had ZERO references outside
app/agent/ — the HTTP layer serialized the report exactly as if validation
had passed. These tests exercise the actual FastAPI route (not just the
agent state dict in isolation) to prove the fix in app/routers/investigate.py
actually intercepts a failed gate before it reaches the client, and that a
clean run is unaffected.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.models.agent import AgentFinalState, InvestigationReport

client = TestClient(app)
_HEADERS = {"X-Internal-Api-Key": "test-key"}


def _minimal_report() -> InvestigationReport:
    from app.models.agent import (
        CapaAnalysis,
        CapaStatus,
        FiveWhyAnalysis,
        ImpactAssessment,
        ImpactStatus,
        InvestigationPlan,
        RootCauseAnalysis,
        RootCauseStatus,
    )
    return InvestigationReport(
        observation_quality="SUFFICIENT",
        investigation_required="NO",
        root_cause=RootCauseAnalysis(status=RootCauseStatus.NOT_ESTABLISHED),
        investigation=InvestigationPlan(),
        five_why=FiveWhyAnalysis(),
        capa=CapaAnalysis(status=CapaStatus.INVESTIGATION_REQUIRED),
        impact_assessment=ImpactAssessment(status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT),
    )


def _patched_graph(final_state: dict):
    stub_graph = MagicMock()
    stub_graph.ainvoke = AsyncMock(return_value=final_state)
    return stub_graph


class TestFailClosedAPIEnforcement:
    def test_validation_failure_produces_422_not_200(self):
        """A BLOCKER-gate failure recorded in agent state must surface as an
        HTTP 422 with investigation_completed=False — never a 200 carrying
        the (invalid) report."""
        final_state = {
            "final_state": AgentFinalState.INVESTIGATION_REQUIRED,
            "report": _minimal_report(),
            "ca_draft": None,
            "trace": [],
            "validation_failure": "BLOCKER invariant violation(s): ['[INV-CGRAPH-002] test']",
        }
        with patch("app.routers.investigate.get_agent_graph", return_value=_patched_graph(final_state)):
            resp = client.post(
                "/api/v1/investigate",
                json={"finding_text": "Test finding for fail-closed verification."},
                headers=_HEADERS,
            )
        assert resp.status_code == 422
        body = resp.json()
        assert body["detail"]["investigation_completed"] is False
        assert body["detail"]["reason"] == "structural_quality_gate_failed"

    def test_clean_run_still_returns_200(self):
        """A run with no validation_failure set must be entirely unaffected
        by the new gate — regression guard against over-blocking."""
        final_state = {
            "final_state": AgentFinalState.INVESTIGATION_REQUIRED,
            "report": _minimal_report(),
            "ca_draft": None,
            "trace": [],
            "validation_failure": None,
        }
        with patch("app.routers.investigate.get_agent_graph", return_value=_patched_graph(final_state)):
            resp = client.post(
                "/api/v1/investigate",
                json={"finding_text": "Test finding for clean-path verification, unique text."},
                headers=_HEADERS,
            )
        assert resp.status_code == 200
        assert resp.json()["report"] is not None
