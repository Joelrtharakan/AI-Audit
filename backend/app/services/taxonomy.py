"""6M-based root cause taxonomy.

This is the single source of truth for root-cause categories. The LLM must never
invent categories outside this list — CAPA output is coerced/validated against it.
"""

from enum import Enum


class RootCauseCategory(str, Enum):
    MAN = "MAN"
    METHOD = "METHOD"
    MACHINE = "MACHINE"
    MATERIAL = "MATERIAL"
    MEASUREMENT = "MEASUREMENT"
    ENVIRONMENT_MGMT = "ENVIRONMENT_MGMT"
    OTHER = "OTHER"


TAXONOMY: dict[str, dict[str, str]] = {
    RootCauseCategory.MAN: {
        "label": "Man",
        "description": "Training gap / competency / human error / staffing & workload",
    },
    RootCauseCategory.METHOD: {
        "label": "Method",
        "description": "SOP not followed / SOP inadequate or outdated / process design flaw",
    },
    RootCauseCategory.MACHINE: {
        "label": "Machine",
        "description": "Equipment failure / calibration lapse / maintenance gap",
    },
    RootCauseCategory.MATERIAL: {
        "label": "Material",
        "description": "Reagent/supply quality issue / expired materials / supplier issue",
    },
    RootCauseCategory.MEASUREMENT: {
        "label": "Measurement",
        "description": "No validation/verification step / inadequate review process",
    },
    RootCauseCategory.ENVIRONMENT_MGMT: {
        "label": "Environment / Management",
        "description": "Resourcing decision / communication breakdown / management system gap",
    },
    RootCauseCategory.OTHER: {
        "label": "Other",
        "description": "Anything not covered above",
    },
}

VALID_CATEGORY_VALUES = {c.value for c in RootCauseCategory}


def coerce_category(value: str | None) -> RootCauseCategory:
    """Coerce an arbitrary (possibly LLM-produced) string into a valid category.

    Falls back to OTHER for anything unrecognized so downstream code never has
    to handle an out-of-taxonomy value; callers should log a warning when a
    coercion actually changes the value.
    """
    if not value:
        return RootCauseCategory.OTHER
    normalized = value.strip().upper().replace(" ", "_").replace("-", "_")
    if normalized in VALID_CATEGORY_VALUES:
        return RootCauseCategory(normalized)
    return RootCauseCategory.OTHER


def taxonomy_prompt_block() -> str:
    """Render the taxonomy as text for inclusion in LLM prompts."""
    lines = []
    for category, meta in TAXONOMY.items():
        lines.append(f"- {category.value}: {meta['description']}")
    return "\n".join(lines)
