"""Post-generation hallucination guardrail (spec Part 3).

Prompt instructions alone ("don't fabricate") are a request a model can silently
ignore. This module is the actual mechanical check: every specific claim the
GENERATION step introduces gets traced back to the finding text or the Step 1
extraction before it's allowed anywhere near the auditor. Anything that doesn't
trace back is a violation -- HARD violations get surgically stripped and, for
the highest-stakes fields, downgrade the root-cause status a tier, because a
hallucinated specific is itself evidence the classification was too confident.

Also enforces three rule-based scope guardrails (3.2) that are deliberately
*not* left to model judgment: recall/quarantine/notify language gated on
external impact, CAPA owners as roles not named individuals, and quoted/numeric
claims that must be verbatim from the source.
"""

import logging
import re

from app.models.analysis import ExtractionResult, GroundingViolation, RootCauseStatus
from app.services.text_grounding import entity_is_grounded

logger = logging.getLogger(__name__)

# Named-entity-shaped tokens: IDs like "SOP-LAB-001", "R-12", acronyms like "LIMS".
# This is deliberately narrow (not full NER) -- it targets exactly the fabrication
# pattern in the spec's own example ("the LIMS lacks automated validation" when no
# LIMS was ever mentioned), not general prose.
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)+\b|\b[A-Z]{2,}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_QUOTED_RE = re.compile(r'"([^"]{3,})"|\'([^\']{3,})\'')

_SOFT_GAP_PHRASES_RE = re.compile(
    r"\b(lacks?|does not have|doesn't have|no\s+\w+\s+(step|process|check)|insufficient|missing)\b",
    re.IGNORECASE,
)

_RECALL_RE = re.compile(r"\b(recall|quarantine|notify\s+\w*\s*customer|inform\s+\w*\s*client)\b", re.IGNORECASE)

_ROLE_KEYWORDS = (
    "manager", "officer", "lead", "supervisor", "director", "coordinator",
    "technician", "team", "head", "chief", "auditor", "specialist", "administrator",
)
_TWO_CAPITALIZED_WORDS_RE = re.compile(r"^[A-Z][a-z]+ [A-Z][a-z]+$")

_ROLE_FALLBACK_BY_DEPARTMENT = {
    "laboratory": "Laboratory Quality Manager",
    "pharmacy": "Pharmacy Manager",
    "nursing": "Nursing Unit Manager",
    "hr": "HR Manager",
    "facilities": "Facilities Manager",
    "quality": "Quality Manager",
}
_DEFAULT_ROLE_FALLBACK = "Department Manager"


class GroundingValidationResult:
    def __init__(self, cleaned_output: dict, hard_violations: list[GroundingViolation],
                 soft_violations: list[GroundingViolation], checked_entities: int,
                 downgraded_status: RootCauseStatus | None):
        self.cleaned_output = cleaned_output
        self.hard_violations = hard_violations
        self.soft_violations = soft_violations
        self.checked_entities = checked_entities
        self.downgraded_status = downgraded_status


_FALLBACK_LINE = "Specific system involved not stated in the observation -- auditor to identify."

# Fields whose HARD violations count as "root_cause / five_why content" (spec 3.1.4) and
# therefore downgrade root_cause_status a tier when hallucinated.
_STATUS_SENSITIVE_FIELDS = {"five_why", "corrective_actions_immediate", "preventive_actions_recurrence"}


def _source_text(finding_text: str, extraction: ExtractionResult) -> str:
    parts = [finding_text, *extraction.stated_facts, *extraction.referenced_records,
             *extraction.named_systems_or_documents]
    parts.extend(f"{s.speaker} {s.claim}" for s in extraction.attributed_statements)
    if extraction.timeframe:
        parts.append(extraction.timeframe)
    if extraction.asset_or_location:
        parts.append(extraction.asset_or_location)
    return " ".join(parts)


def _entities_in(text: str) -> list[str]:
    return _ENTITY_RE.findall(text)


def _numbers_in(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def _quotes_in(text: str) -> list[str]:
    return [a or b for a, b in _QUOTED_RE.findall(text)]


def _check_item(field: str, item: str, finding_text: str, source_text: str) -> GroundingViolation | None:
    """Returns a HARD violation if `item` contains a named entity, number, or quoted
    string that doesn't trace back to the source; otherwise None (may still log a SOFT
    note via the caller)."""
    for entity in _entities_in(item):
        if not entity_is_grounded(entity, source_text):
            return GroundingViolation(field=field, claimed_entity=entity, note="named system/ID not in source")
    for number in _numbers_in(item):
        if number not in finding_text:
            return GroundingViolation(field=field, claimed_entity=number, note="numeric value not in finding text")
    for quote in _quotes_in(item):
        if quote.lower() not in finding_text.lower():
            return GroundingViolation(field=field, claimed_entity=quote, note="quoted text not verbatim in finding")
    return None


def _is_soft_gap_phrase(item: str) -> bool:
    return bool(_SOFT_GAP_PHRASES_RE.search(item))


def _clean_list_field(
    field: str, items: list, finding_text: str, source_text: str,
    hard: list[GroundingViolation], soft: list[GroundingViolation], checked: list[int],
) -> list[str]:
    kept = []
    had_content = bool(items)
    for raw in items:
        item = str(raw)
        checked[0] += len(_entities_in(item)) + len(_numbers_in(item)) + len(_quotes_in(item))
        violation = _check_item(field, item, finding_text, source_text)
        if violation:
            hard.append(violation)
            logger.warning("Grounding HARD violation field=%s entity=%r note=%s", field, violation.claimed_entity, violation.note)
            continue
        if _is_soft_gap_phrase(item):
            soft.append(GroundingViolation(field=field, claimed_entity=item[:60], note="generalized gap phrasing (allowed)"))
        kept.append(item)
    if had_content and not kept:
        kept = [_FALLBACK_LINE]
    return kept


def _check_recall_language(items: list[str], external_impact_stated: bool,
                            hard: list[GroundingViolation]) -> list[str]:
    if external_impact_stated:
        return items
    kept = []
    for item in items:
        if _RECALL_RE.search(item):
            hard.append(GroundingViolation(
                field="corrective_actions_immediate", claimed_entity=item[:60],
                note="recall/quarantine/notify language with no external impact stated",
            ))
            logger.warning("Grounding HARD violation: recall/quarantine/notify language with EXTERNAL_IMPACT_STATED=false: %r", item)
            continue
        kept.append(item)
    return kept


def _coerce_capa_owner(owner: str | None, department: str, hard: list[GroundingViolation]) -> str | None:
    if not owner:
        return owner
    candidate = owner.strip()
    looks_like_a_name = bool(_TWO_CAPITALIZED_WORDS_RE.match(candidate)) and not any(
        kw in candidate.lower() for kw in _ROLE_KEYWORDS
    )
    if not looks_like_a_name:
        return candidate
    hard.append(GroundingViolation(field="recommended_capa_owner", claimed_entity=candidate, note="named individual, not a role"))
    logger.warning("Grounding HARD violation: recommended_capa_owner looked like a person's name: %r", candidate)
    fallback = _ROLE_FALLBACK_BY_DEPARTMENT.get((department or "").strip().lower(), _DEFAULT_ROLE_FALLBACK)
    return fallback


def validate_grounding(
    generation_output: dict,
    finding_text: str,
    extraction: ExtractionResult,
    root_cause_status: RootCauseStatus,
    department: str = "",
) -> GroundingValidationResult:
    hard: list[GroundingViolation] = []
    soft: list[GroundingViolation] = []
    checked = [0]
    source_text = _source_text(finding_text, extraction)

    cleaned = dict(generation_output)

    cleaned["five_why"] = _clean_list_field(
        "five_why", generation_output.get("five_why", []), finding_text, source_text, hard, soft, checked
    )
    cleaned["contributing_factors"] = _clean_list_field(
        "contributing_factors", generation_output.get("contributing_factors", []), finding_text, source_text, hard, soft, checked
    )

    capa = dict(generation_output.get("capa") or {})
    capa["corrective_actions_immediate"] = _clean_list_field(
        "corrective_actions_immediate", capa.get("corrective_actions_immediate", []), finding_text, source_text, hard, soft, checked
    )
    capa["corrective_actions_immediate"] = _check_recall_language(
        capa["corrective_actions_immediate"], extraction.external_impact_stated, hard
    )
    capa["preventive_actions_recurrence"] = _clean_list_field(
        "preventive_actions_recurrence", capa.get("preventive_actions_recurrence", []), finding_text, source_text, hard, soft, checked
    )
    cleaned["capa"] = capa

    cleaned["recommended_capa_owner"] = _coerce_capa_owner(
        generation_output.get("recommended_capa_owner"), department, hard
    )

    status_sensitive_violations = [v for v in hard if v.field in _STATUS_SENSITIVE_FIELDS]
    downgraded_status: RootCauseStatus | None = None
    if status_sensitive_violations:
        if root_cause_status == RootCauseStatus.ESTABLISHED:
            downgraded_status = RootCauseStatus.SELF_REPORTED
        elif root_cause_status == RootCauseStatus.SELF_REPORTED:
            downgraded_status = RootCauseStatus.NOT_ESTABLISHED
        if downgraded_status:
            logger.warning(
                "Downgrading root_cause_status %s -> %s due to grounding violation(s): %s",
                root_cause_status.value, downgraded_status.value,
                [v.claimed_entity for v in status_sensitive_violations],
            )

    return GroundingValidationResult(
        cleaned_output=cleaned,
        hard_violations=hard,
        soft_violations=soft,
        checked_entities=checked[0],
        downgraded_status=downgraded_status,
    )
