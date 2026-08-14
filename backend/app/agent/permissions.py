"""Hard write-permission boundary for the AI agent.

Security MUST NOT depend on prompting. This module is the code-level enforcement
layer. Even if an LLM returns {"ca_status": "Closed"} or any other protected field,
validate_ai_output_fields() will raise PermissionError before it reaches the response.

Called by ca_draft_generator.py before constructing the CADraft response.
"""

from app.models.agent import CADraft

# The five fields the agent is permitted to generate. Nothing else.
AI_WRITABLE_FIELDS: frozenset[str] = frozenset({
    "immediate_action",
    "root_cause",
    "root_cause_category",
    "preventive_action",
    "impact_analysis",
})

# Fields the agent must NEVER touch. Reject any attempt proactively.
AI_FORBIDDEN_FIELDS: frozenset[str] = frozenset({
    "ca_status",
    "ca_attachments",
    "review_of_corrective_actions",
    "follow_up_details",
    "continuous_monitoring",
    "manager_approval",
    "closing_details",
    "status",
})


def validate_ai_output_fields(data: dict) -> dict:
    """Validate that the AI output only contains permitted writable fields.

    Raises PermissionError if any unauthorized field is present.
    Returns the validated dict on success.

    This runs in code, independent of the prompt. Even if the LLM produces
    a field like 'ca_status': 'Closed', this function will reject it.
    """
    for field in data:
        if field not in AI_WRITABLE_FIELDS:
            raise PermissionError(
                f"AI attempted to write to protected field '{field}'. "
                f"Only these fields are AI-writable: {sorted(AI_WRITABLE_FIELDS)}"
            )
    return data


def sanitize_form_text(val: str | None) -> str:
    if not val:
        return ""
    s = str(val).strip()
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        import re
        s = re.sub(r"'(?:status|id|name)':\s*'[^']*',?", "", s)
        s = re.sub(r"'(?:description|statement|text)':\s*", "", s)
        s = s.replace("{", "").replace("}", "").replace("[", "").replace("]", "").strip()
    return s


def build_ca_draft(data: dict) -> CADraft:
    """Validate field permissions then construct a CADraft with sanitized strings.

    Raises PermissionError if unauthorized fields are present.
    Raises ValueError if required fields are missing.
    """
    validated = validate_ai_output_fields(data)
    sanitized = {k: sanitize_form_text(v) for k, v in validated.items()}
    return CADraft(**sanitized)
