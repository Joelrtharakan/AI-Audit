"""Assembles the Step 3 (generation) LLM-first finding-analysis prompt. Kept
separate from openrouter_client.py / finding_analysis_service.py so prompt
changes are auditable independent of the generation/validation logic.

Step 1 (extraction) and Step 2 (classification) build their own prompts in
extraction.py and causation_classifier.py respectively -- this module is only
the final generation step, and it receives root_cause_status/category as
already-decided input, not something it re-derives.
"""

import json

from app.config import get_settings
from app.models.analysis import CausationClassification, ExtractionResult

GENERATION_RESPONSE_SCHEMA = {
    # Core fields first, deliberately -- on smaller local models a long schema can run
    # short on output tokens near the end, and it's far better to lose the two
    # ESTABLISHED-only extras below than a core field every response depends on.
    "five_why": ["string"],
    "investigation": {
        "areas": ["string"],
        "questions": ["string"],
        "evidence_required": ["string"],
    },
    "capa": {
        "status": "AI_SUGGESTED|INVESTIGATION_REQUIRED",
        "corrective_actions_immediate": ["string"],
        "preventive_actions_recurrence": ["string"],
        "recommended_investigation": ["string"],
        "potential_corrective_action_areas": ["string"],
    },
    "impact_analysis": {
        "status": "ASSESSED|REQUIRES_ASSESSMENT",
        "areas": ["string"],
        "narrative": "string or null",
    },
    "contributing_factors": ["string"],
    "risk_of_recurrence": "Low|Medium|High, or null if not ESTABLISHED",
    "recommended_capa_owner": "role string, or null if not ESTABLISHED",
}


def build_finding_analysis_messages(
    finding: dict,
    observation_quality_status: str,
    observation_missing_information: list[str],
    extraction: ExtractionResult,
    classification: CausationClassification,
) -> list[dict[str, str]]:
    """Build the Step 3 (generation) prompt. root_cause_status/category come from
    Step 2 and are passed as fixed context -- this step does not decide them."""
    settings = get_settings()
    system_prompt = (settings.prompts_dir / "finding_analysis_system_prompt.txt").read_text(encoding="utf-8")

    missing_info_block = (
        "\n".join(f"- {m}" for m in observation_missing_information)
        if observation_missing_information
        else "(none identified)"
    )

    user_prompt = f"""FINDING:
{finding.get('finding_text')}

CONTEXT:
- Department: {finding.get('department') or 'not provided'}
- Branch: {finding.get('branch') or 'not provided'}
- Standard/criteria: {finding.get('standard') or 'not provided'}
- Clause: {finding.get('clause') or 'not provided'}
- Finding type: {finding.get('finding_type') or 'not provided'}
- Nature of NC: {finding.get('nature_of_nc') or 'not provided'}
- Risk severity: {finding.get('risk_severity') or 'not provided'}
- Risk likelihood: {finding.get('risk_likelihood') or 'not provided'}
- Risk result: {finding.get('risk_result') or 'not provided'}

OBSERVATION QUALITY CHECK RESULT: {observation_quality_status}
Missing information already identified by the quality check:
{missing_info_block}

EXTRACTION (Step 1 -- what the observation actually says):
{extraction.model_dump_json(indent=2)}

ROOT CAUSE STATUS (Step 2 -- already decided, do not re-derive): {classification.root_cause_status.value}
ROOT CAUSE CATEGORY: {classification.category or 'null'}
ROOT CAUSE REASONING: {classification.reasoning or 'n/a'}
EXTERNAL_IMPACT_STATED: {extraction.external_impact_stated}

Respond with a single JSON object matching this schema exactly:
{json.dumps(GENERATION_RESPONSE_SCHEMA, indent=2)}
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
