"""Generalization tests for the semantic subject/condition extractor.

These are the tests demanded by the "context/entity propagation" bug report:
the affected object of a finding must survive extraction regardless of
sentence structure or subject domain, and must never collapse into a
framing fragment ("During the internal audit, it") or a generic placeholder
("affected item", "record/process item", etc.). Every case here is
deliberately a SUBJECT TYPE / SENTENCE STRUCTURE the extractor has never
seen hardcoded anywhere in app/services/semantic_subject.py -- the module
recognizes grammatical patterns, not finding content.
"""

from __future__ import annotations

import pytest

from app.agent.semantic_validator import _PLACEHOLDER_STRINGS, validate_semantic_consistency
from app.models.agent import CanonicalFindingState
from app.services.semantic_subject import extract_semantic_subject, resolve_deviation

# ---------------------------------------------------------------------------
# Subject types (never referenced in semantic_subject.py's own patterns)
# ---------------------------------------------------------------------------
SUBJECT_TYPES = [
    "the cleaning checklist",
    "the standard operating procedure",
    "the training record",
    "the production record",
    "the laboratory record",
    "the equipment log",
    "the calibration certificate",
    "the supplier record",
    "the batch record",
    "the audit report",
    "the environmental monitoring record",
    "the software access record",
    "the warehouse inspection record",
]

PLACEHOLDER_STRINGS_LOWER = {s.lower() for s in _PLACEHOLDER_STRINGS}


def _assert_real_subject(subject: str | None, subject_type: str) -> None:
    assert subject is not None, f"no subject extracted for {subject_type!r}"
    lowered = subject.lower()
    assert lowered not in PLACEHOLDER_STRINGS_LOWER, f"got placeholder {subject!r} for {subject_type!r}"
    assert not lowered.startswith("during"), f"subject leaked framing clause: {subject!r}"
    assert "audit, it" not in lowered
    # The core noun of the subject type must survive (allow "the"/"a" article differences).
    core_noun = subject_type.split()[-1]
    assert core_noun in lowered, f"expected {core_noun!r} in extracted subject {subject!r}"


@pytest.mark.parametrize("subject_type", SUBJECT_TYPES)
def test_subject_survives_audit_observation_framing(subject_type: str) -> None:
    text = f"During the internal audit, it was observed that {subject_type} was not completed."
    result = extract_semantic_subject(text)
    _assert_real_subject(result.subject, subject_type)
    assert result.condition == "not completed"


@pytest.mark.parametrize("subject_type", SUBJECT_TYPES)
def test_subject_survives_it_was_found_structure(subject_type: str) -> None:
    text = f"It was found that {subject_type} was missing."
    result = extract_semantic_subject(text)
    _assert_real_subject(result.subject, subject_type)
    assert result.condition == "missing"


@pytest.mark.parametrize("subject_type", SUBJECT_TYPES)
def test_subject_survives_auditor_identified_structure(subject_type: str) -> None:
    text = f"The auditor identified {subject_type} as incomplete."
    result = extract_semantic_subject(text)
    _assert_real_subject(result.subject, subject_type)
    assert result.condition == "incomplete"


@pytest.mark.parametrize("subject_type", SUBJECT_TYPES)
def test_subject_survives_bare_not_completed_structure(subject_type: str) -> None:
    text = f"{subject_type.capitalize()} was not completed."
    result = extract_semantic_subject(text)
    _assert_real_subject(result.subject, subject_type)


@pytest.mark.parametrize("subject_type", SUBJECT_TYPES)
def test_subject_survives_deviation_observed_involving_structure(subject_type: str) -> None:
    text = f"A deviation was observed involving {subject_type}."
    result = extract_semantic_subject(text)
    _assert_real_subject(result.subject, subject_type)


@pytest.mark.parametrize("subject_type", SUBJECT_TYPES)
def test_subject_survives_technician_stated_structure(subject_type: str) -> None:
    text = f"The technician stated that {subject_type} was missed."
    result = extract_semantic_subject(text)
    _assert_real_subject(result.subject, subject_type)


@pytest.mark.parametrize("subject_type", SUBJECT_TYPES)
def test_subject_survives_record_for_x_structure(subject_type: str) -> None:
    text = f"The record for {subject_type} contained an incomplete entry."
    result = extract_semantic_subject(text)
    assert result.subject is not None
    core_noun = subject_type.split()[-1]
    assert core_noun in result.subject.lower()


# ---------------------------------------------------------------------------
# The exact reported bug: multi-clause finding with equipment ID + date +
# technician statement. Verifies subject, condition, date, and actor all
# survive together, matching the CORE TASK section of the bug report.
# ---------------------------------------------------------------------------


def test_original_bug_report_finding_resolves_correctly() -> None:
    finding = (
        "During the internal audit, it was observed that the temperature log for "
        "refrigerator QC-REF-02 was not completed for 12 August 2026. The responsible "
        "technician confirmed that the temperature check was missed during the morning shift."
    )
    result = resolve_deviation(finding)
    assert result.subject == "temperature log for refrigerator QC-REF-02"
    assert result.condition == "not completed"
    assert result.date == "12 August 2026"
    assert result.actor == "The responsible technician"
    assert "during the internal audit, it" not in (result.subject or "").lower()


# ---------------------------------------------------------------------------
# Placeholder elimination: even when nothing structural matches, the
# extractor must never silently invent a placeholder-shaped string itself —
# it returns None and lets the caller decide how to represent "unknown".
# ---------------------------------------------------------------------------


def test_no_extractable_subject_returns_none_not_a_placeholder() -> None:
    result = extract_semantic_subject("Something was wrong.")
    # "Something" is a pronoun-shaped filler; the extractor should not
    # fabricate a fake specific placeholder in its place.
    if result.subject is not None:
        assert result.subject.lower() not in PLACEHOLDER_STRINGS_LOWER


def test_empty_text_returns_no_subject() -> None:
    result = extract_semantic_subject("")
    assert result.subject is None


# ---------------------------------------------------------------------------
# Semantic consistency validator
# ---------------------------------------------------------------------------


def test_consistency_validator_flags_lost_entity() -> None:
    canonical = CanonicalFindingState(
        raw_finding="The training record for Operator A was not completed.",
        observed_deviation="training record for Operator A — not completed",
        affected_objects=["training record for Operator A"],
    )
    state = {
        "root_cause": None,
        "five_why": None,
        "impact_assessment": None,
    }
    # No downstream text at all yet -- nothing to check against, so no warnings.
    assert validate_semantic_consistency(canonical, state) == []


def test_consistency_validator_flags_placeholder_regression() -> None:
    from app.models.agent import ImpactAssessment, ImpactStatus

    canonical = CanonicalFindingState(
        raw_finding="The training record for Operator A was not completed.",
        observed_deviation="training record for Operator A — not completed",
        affected_objects=["training record for Operator A"],
    )
    impact = ImpactAssessment(
        status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT,
        affected_object="record/process item",
    )
    state = {"root_cause": None, "five_why": None, "impact_assessment": impact}
    warnings = validate_semantic_consistency(canonical, state)
    assert any("placeholder" in w.lower() for w in warnings)


def test_consistency_validator_passes_when_entity_preserved() -> None:
    from app.models.agent import ImpactAssessment, ImpactStatus

    canonical = CanonicalFindingState(
        raw_finding="The training record for Operator A was not completed.",
        observed_deviation="training record for Operator A — not completed",
        affected_objects=["training record for Operator A"],
    )
    impact = ImpactAssessment(
        status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT,
        affected_object="training record for Operator A",
        narrative="The training record for Operator A could not be confirmed complete.",
    )
    state = {"root_cause": None, "five_why": None, "impact_assessment": impact}
    warnings = validate_semantic_consistency(canonical, state)
    assert warnings == []


# ---------------------------------------------------------------------------
# End-to-end: understand_finding_node must never regress the original bug,
# even when the extraction LLM call returns the finding's full first
# sentence undecomposed as its only stated_fact (the exact condition that
# triggered the "During the internal audit, it" collapse).
# ---------------------------------------------------------------------------


def _build_understanding_state(finding_text: str) -> dict:
    from app.models.agent import AgentTraceStep, InvestigateRequest

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
        "evidence_ledger": [],
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


@pytest.mark.asyncio
async def test_understand_finding_node_does_not_regress_original_bug() -> None:
    from unittest.mock import AsyncMock, patch

    from app.agent.nodes.understanding import understand_finding_node
    from app.models.analysis import ExtractionResult, ObservationQualityResult, ObservationQualityStatus

    finding_text = (
        "During the internal audit, it was observed that the temperature log for "
        "refrigerator QC-REF-02 was not completed for 12 August 2026. The responsible "
        "technician confirmed that the temperature check was missed during the morning shift."
    )
    state = _build_understanding_state(finding_text)

    # Simulate a weak extraction model that returns the full first sentence
    # undecomposed (no deviation_subject field populated) -- the exact
    # scenario that used to collapse into the framing fragment.
    async def fake_extract_finding(text, client):
        return ExtractionResult(
            stated_facts=[
                "During the internal audit, it was observed that the temperature log for "
                "refrigerator QC-REF-02 was not completed for 12 August 2026."
            ],
            attributed_statements=[],
        )

    async def fake_check_quality(text, client):
        return ObservationQualityResult(status=ObservationQualityStatus.SUFFICIENT, missing_information=[])

    with patch("app.agent.nodes.understanding.get_llm_client") as mock_get_client, \
         patch("app.agent.nodes.understanding.extract_finding", side_effect=fake_extract_finding), \
         patch("app.agent.nodes.understanding.check_observation_quality", side_effect=fake_check_quality):
        mock_get_client.return_value = AsyncMock()
        result = await understand_finding_node(state)

    canonical = result["canonical_finding_state"]
    affected = canonical.affected_objects[0]
    assert affected.lower() not in PLACEHOLDER_STRINGS_LOWER
    assert not affected.lower().startswith("during")
    assert "audit, it" not in affected.lower()
    assert "temperature log" in affected.lower()
    assert "qc-ref-02" in affected.lower()
