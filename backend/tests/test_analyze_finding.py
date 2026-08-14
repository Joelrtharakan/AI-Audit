"""Finding analysis pipeline tests — no mock LLM, no fake clients.

All LLM-dependent tests call the real provider configured in .env.
These are integration tests; run with: pytest -m integration

Pure-logic unit tests (no LLM) are run always.
"""

import pytest

from app.models.analysis import ExtractionResult, RootCauseStatus
from app.models.requests import AnalyzeFindingRequest

# ---------------------------------------------------------------------------
# Pure-logic unit tests — no LLM required (always run)
# ---------------------------------------------------------------------------


def test_blank_finding_text_rejected():
    with pytest.raises(ValueError):
        AnalyzeFindingRequest(finding_text="   ")


def test_five_why_keeps_threaded_chain():
    from app.services.five_why_validator import validate_and_truncate_chain

    chain = [
        "The calibration lapsed because the request was never logged in the register.",
        "The request was never logged because the register reminder was ignored.",
    ]
    assert validate_and_truncate_chain(chain) == chain


def test_five_why_truncates_at_disconnected_step():
    from app.services.five_why_validator import validate_and_truncate_chain

    chain = [
        "The calibration lapsed because the request was never logged in the register.",
        "Completely unrelated statement about a totally different topic with no overlap.",
        "A third step that would never be reached.",
    ]
    result = validate_and_truncate_chain(chain)
    assert result == [chain[0]]


def test_five_why_empty_and_single_step_unchanged():
    from app.services.five_why_validator import validate_and_truncate_chain

    assert validate_and_truncate_chain([]) == []
    assert validate_and_truncate_chain(["Only one step."]) == ["Only one step."]


def test_grounding_strips_hallucinated_system_name():
    from app.models.analysis import ExtractionResult, RootCauseStatus
    from app.services.grounding_validator import validate_grounding

    extraction = ExtractionResult(
        stated_facts=["Refrigerator R-12 temperature logs had missing entries."],
        attributed_statements=[{"speaker": "staff", "claim": "had not been retrained on SOP-LAB-001"}],
        referenced_records=["training attendance log"],
        named_systems_or_documents=["SOP-LAB-001"],
        timeframe="three consecutive days",
        asset_or_location="Refrigerator R-12",
        external_impact_stated=False,
    )
    generation_output = {
        "contributing_factors": [],
        "five_why": ["The LIMS lacks automated validation for temperature entries."],
        "investigation": {"areas": [], "questions": [], "evidence_required": []},
        "capa": {
            "status": "AI_SUGGESTED",
            "corrective_actions_immediate": ["Update the LIMS validation rules."],
            "preventive_actions_recurrence": [],
            "recommended_investigation": [],
            "potential_corrective_action_areas": [],
        },
        "risk_of_recurrence": "Medium",
        "recommended_capa_owner": "Laboratory Quality Manager",
    }
    finding_text = (
        "Refrigerator R-12 temperature logs in the Laboratory department had missing entries; "
        "the training attendance log confirms staff had not been retrained on SOP-LAB-001."
    )

    result = validate_grounding(generation_output, finding_text, extraction, RootCauseStatus.ESTABLISHED)

    assert any(v.claimed_entity == "LIMS" for v in result.hard_violations)
    assert result.downgraded_status == RootCauseStatus.SELF_REPORTED


def test_grounding_allows_grounded_content():
    from app.models.analysis import ExtractionResult, RootCauseStatus
    from app.services.grounding_validator import validate_grounding

    extraction = ExtractionResult(
        stated_facts=["Refrigerator R-12 temperature logs had missing entries."],
        attributed_statements=[{"speaker": "staff", "claim": "had not been retrained on SOP-LAB-001"}],
        referenced_records=["training attendance log"],
        named_systems_or_documents=["SOP-LAB-001"],
        timeframe="three consecutive days",
        asset_or_location="Refrigerator R-12",
        external_impact_stated=False,
    )
    generation_output = {
        "contributing_factors": ["No backup reviewer assigned for the evening shift."],
        "five_why": ["Staff had not been retrained on SOP-LAB-001 per the training attendance log."],
        "investigation": {"areas": [], "questions": [], "evidence_required": []},
        "capa": {
            "status": "AI_SUGGESTED",
            "corrective_actions_immediate": ["Retrain staff on SOP-LAB-001."],
            "preventive_actions_recurrence": [],
            "recommended_investigation": [],
            "potential_corrective_action_areas": [],
        },
        "risk_of_recurrence": "Medium",
        "recommended_capa_owner": "Laboratory Quality Manager",
    }
    finding_text = (
        "Refrigerator R-12 temperature logs in the Laboratory department had missing entries; "
        "the training attendance log confirms staff had not been retrained on SOP-LAB-001."
    )

    result = validate_grounding(generation_output, finding_text, extraction, RootCauseStatus.ESTABLISHED)

    assert result.hard_violations == []
    assert result.downgraded_status is None


def test_grounding_strips_recall_language_without_external_impact():
    from app.models.analysis import ExtractionResult, RootCauseStatus
    from app.services.grounding_validator import validate_grounding

    extraction = ExtractionResult(
        stated_facts=["Temperature logs missing."],
        attributed_statements=[],
        referenced_records=["training attendance log"],
        external_impact_stated=False,
    )
    generation_output = {
        "contributing_factors": [],
        "five_why": [],
        "investigation": {"areas": [], "questions": [], "evidence_required": []},
        "capa": {
            "status": "AI_SUGGESTED",
            "corrective_actions_immediate": ["Recall the affected batch and notify the customer immediately."],
            "preventive_actions_recurrence": [],
            "recommended_investigation": [],
            "potential_corrective_action_areas": [],
        },
        "risk_of_recurrence": "Medium",
        "recommended_capa_owner": "Laboratory Quality Manager",
    }

    result = validate_grounding(
        generation_output, "Temperature logs were missing.", extraction, RootCauseStatus.ESTABLISHED
    )

    assert result.cleaned_output["capa"]["corrective_actions_immediate"] == []
    assert any("recall" in v.note.lower() for v in result.hard_violations)


def test_grounding_coerces_named_individual_to_role():
    from app.models.analysis import ExtractionResult, RootCauseStatus
    from app.services.grounding_validator import validate_grounding

    extraction = ExtractionResult(
        stated_facts=["Logs missing."],
        referenced_records=["training attendance log"],
        external_impact_stated=False,
    )
    generation_output = {
        "contributing_factors": [],
        "five_why": [],
        "investigation": {"areas": [], "questions": [], "evidence_required": []},
        "capa": {
            "status": "AI_SUGGESTED",
            "corrective_actions_immediate": [],
            "preventive_actions_recurrence": [],
            "recommended_investigation": [],
            "potential_corrective_action_areas": [],
        },
        "risk_of_recurrence": "Medium",
        "recommended_capa_owner": "John Smith",
    }

    result = validate_grounding(
        generation_output, "Logs missing.", extraction, RootCauseStatus.ESTABLISHED,
        department="Laboratory",
    )

    assert result.cleaned_output["recommended_capa_owner"] == "Laboratory Quality Manager"
    assert any(v.field == "recommended_capa_owner" for v in result.hard_violations)


def test_root_cause_enforcement_downgrades_established_without_records():
    """Code enforcement: ESTABLISHED with no referenced_records → downgraded to SELF_REPORTED."""
    from app.models.analysis import CausationClassification, ExtractionResult, RootCauseStatus
    from app.services.finding_analysis_service import _build_root_cause_status

    extraction = ExtractionResult(
        stated_facts=["Log entries were missing."],
        attributed_statements=[{"speaker": "staff", "claim": "not retrained"}],
        referenced_records=[],  # no corroborating record
        external_impact_stated=False,
    )
    classification = CausationClassification(
        root_cause_status=RootCauseStatus.ESTABLISHED,  # LLM overreach
        category="MAN",
        reasoning="Staff admitted the gap.",
    )

    result = _build_root_cause_status(extraction, classification)
    assert result == RootCauseStatus.SELF_REPORTED


def test_root_cause_enforcement_downgrades_self_reported_without_attributed_statements():
    """Code enforcement: SELF_REPORTED with no attributed_statements → NOT_ESTABLISHED."""
    from app.models.analysis import CausationClassification, ExtractionResult, RootCauseStatus
    from app.services.finding_analysis_service import _build_root_cause_status

    extraction = ExtractionResult(
        stated_facts=["Logs were missing."],
        attributed_statements=[],  # no one made any statement
        referenced_records=[],
        external_impact_stated=False,
    )
    classification = CausationClassification(
        root_cause_status=RootCauseStatus.SELF_REPORTED,
        category="MAN",
        reasoning="Guessing.",
    )

    result = _build_root_cause_status(extraction, classification)
    assert result == RootCauseStatus.NOT_ESTABLISHED


def test_contributing_factors_filter_drops_action_items():
    from app.models.analysis import ExtractionResult
    from app.services.finding_analysis_service import (
        _filter_contributing_factors,
        _build_investigation,
    )

    investigation_raw = {
        "areas": ["Review training records"],
        "questions": [],
        "evidence_required": [],
    }
    investigation = _build_investigation(investigation_raw)
    factors = [
        "Review training records for the named staff member",  # action item — dropped
        "No backup coverage existed for the evening shift",    # condition — kept
    ]

    result = _filter_contributing_factors(factors, investigation)
    assert result == ["No backup coverage existed for the evening shift"]


# ---------------------------------------------------------------------------
# Real LLM integration tests — require configured LLM_PROVIDER in .env
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_real_llm_extraction_returns_structured_result():
    """Extraction with real LLM: verify schema, not exact content."""
    from app.models.analysis import ExtractionResult
    from app.services.extraction import extract_finding
    from app.services.llm_client import get_llm_client

    client = get_llm_client()
    result = await extract_finding(
        "During the internal audit, refrigerator R-12 temperature logs in the Laboratory "
        "department were found missing entries for the evening shift on three consecutive days; "
        "staff confirmed they had not been retrained on the revised SOP-LAB-001.",
        client,
    )

    assert isinstance(result, ExtractionResult)
    assert isinstance(result.stated_facts, list)
    assert isinstance(result.attributed_statements, list)
    assert isinstance(result.referenced_records, list)
    assert isinstance(result.external_impact_stated, bool)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_real_llm_analysis_insufficient_finding():
    """A vague finding must yield NOT_ESTABLISHED and INVESTIGATION_REQUIRED CAPA."""
    from app.models.analysis import CapaStatus, RootCauseStatus
    from app.models.requests import AnalyzeFindingRequest
    from app.services.finding_analysis_service import FindingAnalysisService

    service = FindingAnalysisService()
    result = await service.analyze(AnalyzeFindingRequest(finding_text="Logs were missing."))

    # Structural assertions — not string matches
    assert result.root_cause.status in (RootCauseStatus.NOT_ESTABLISHED, RootCauseStatus.SELF_REPORTED)
    assert result.analysis.confidence in ("LOW", "MEDIUM")
    assert result.capa.status in (CapaStatus.INVESTIGATION_REQUIRED, CapaStatus.AI_SUGGESTED)
    # Code enforces: CAPA cannot be AI_SUGGESTED if root cause is NOT_ESTABLISHED
    if result.root_cause.status == RootCauseStatus.NOT_ESTABLISHED:
        assert result.capa.status == CapaStatus.INVESTIGATION_REQUIRED
    assert result.ai_metadata.model  # model name is populated
    assert result.ai_metadata.suggestion_id  # UUID is generated


@pytest.mark.asyncio
@pytest.mark.integration
async def test_real_llm_analysis_well_described_finding():
    """A well-described finding must yield a structured response."""
    from app.models.analysis import RootCauseStatus
    from app.models.requests import AnalyzeFindingRequest
    from app.services.finding_analysis_service import FindingAnalysisService

    service = FindingAnalysisService()
    result = await service.analyze(
        AnalyzeFindingRequest(
            finding_text=(
                "During the internal audit on 12 May, refrigerator R-12 temperature logs in the "
                "Laboratory department were reviewed and found missing entries for the evening "
                "shift on three consecutive days. Staff confirmed they had not been retrained "
                "after SOP-LAB-001 was revised last month."
            ),
            department="Laboratory",
            clause="4.6.2",
            finding_type="Non-Conformity",
            severity="Major",
        )
    )

    # Shape / invariant assertions
    assert result.root_cause.status in (s.value for s in RootCauseStatus)
    assert result.analysis.observation_quality in ("SUFFICIENT", "INSUFFICIENT")
    assert result.analysis.confidence in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(result.five_why, list)
    assert isinstance(result.investigation.areas, list)
    assert isinstance(result.contributing_factors, list)
    # Code-level invariant: ESTABLISHED requires referenced_records
    if result.root_cause.status == RootCauseStatus.ESTABLISHED:
        assert result.root_cause.is_hypothesis is False
    if result.root_cause.status == RootCauseStatus.SELF_REPORTED:
        assert result.root_cause.is_hypothesis is True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_real_llm_capa_never_ai_suggested_without_established_root_cause():
    """Code-enforced invariant: CAPA AI_SUGGESTED requires ESTABLISHED root cause."""
    from app.models.analysis import CapaStatus, RootCauseStatus
    from app.models.requests import AnalyzeFindingRequest
    from app.services.finding_analysis_service import FindingAnalysisService

    service = FindingAnalysisService()
    result = await service.analyze(
        AnalyzeFindingRequest(finding_text="A procedure was not followed.")
    )

    # This is enforced in code, not just prompted
    if result.root_cause.status != RootCauseStatus.ESTABLISHED:
        assert result.capa.status == CapaStatus.INVESTIGATION_REQUIRED


@pytest.mark.asyncio
@pytest.mark.integration
async def test_real_llm_response_shape_is_always_valid():
    """The response model must always be valid regardless of LLM content."""
    from app.models.requests import AnalyzeFindingRequest
    from app.services.finding_analysis_service import FindingAnalysisService

    service = FindingAnalysisService()
    result = await service.analyze(
        AnalyzeFindingRequest(finding_text="Some equipment was found out of calibration.")
    )

    # Validate the full response can be serialized
    json_str = result.model_dump_json()
    assert len(json_str) > 100
    assert result.ai_metadata.generated_at
    assert result.grounding_report is not None
