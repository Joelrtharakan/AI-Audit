"""Deterministic semantic consistency validator (no LLM call).

Checks that the canonical finding subject established in understand_finding
survives — semantically, not by exact string match — into the downstream
analysis (root cause, 5-Why, impact assessment). This is the code-level
enforcement of "KNOWN ENTITY -> MUST SURVIVE DOWNSTREAM": if
CanonicalFindingState.affected_objects names a real entity, that entity
(or a clear paraphrase of it) must be traceable somewhere in the final
narrative/impact/five-why text. If it is missing everywhere, or the
downstream affected_object contains entities absent from the finding, that's
flagged as a trace warning so an auditor can see the pipeline lost or
invented context — it never silently fixes the content itself (that's what
the individual node-level grounding guards already do).
"""

from __future__ import annotations

from app.models.agent import CanonicalFindingState
from app.services.text_grounding import phrase_is_grounded, significant_words

_PLACEHOLDER_STRINGS = {
    "affected item",
    "record/process item",
    "specific object",
    "operational process",
    "the during the internal audit, it",
    "during the internal audit, it",
    "it",
    "the audit",
}


def _is_placeholder(text: str | None) -> bool:
    if not text:
        return False
    return text.strip().lower() in _PLACEHOLDER_STRINGS


def validate_semantic_consistency(
    canonical: CanonicalFindingState | None,
    state: dict,
) -> list[str]:
    """Returns a list of human-readable warning messages (empty if consistent).
    Never raises, never mutates state — purely diagnostic."""
    warnings: list[str] = []
    if canonical is None:
        return warnings

    known_entities = [
        e for e in (
            list(canonical.affected_objects)
            + list(canonical.affected_equipment)
            + list(canonical.affected_records)
        )
        if e and not e.startswith("UNKNOWN")
    ]

    # Combine every downstream free-text field that should preserve the
    # canonical subject if it's mentioned anywhere in the final analysis.
    downstream_parts: list[str] = []
    root_cause = state.get("root_cause")
    if root_cause:
        if root_cause.narrative:
            downstream_parts.append(root_cause.narrative)
        if root_cause.statement:
            downstream_parts.append(root_cause.statement)
        for h in root_cause.candidate_hypotheses or []:
            downstream_parts.append(h.statement)

    five_why = state.get("five_why")
    if five_why:
        for step in five_why.steps or []:
            if step.question:
                downstream_parts.append(step.question)
            if step.answer:
                downstream_parts.append(step.answer)

    impact = state.get("impact_assessment")
    impact_affected_object = None
    if impact:
        if impact.narrative:
            downstream_parts.append(impact.narrative)
        if impact.affected_object:
            downstream_parts.append(impact.affected_object)
            impact_affected_object = impact.affected_object

    downstream_text = " ".join(p for p in downstream_parts if p)
    downstream_words = significant_words(downstream_text)

    # 1. Entity preservation: every known canonical entity should be
    # semantically traceable somewhere downstream, once any analysis exists.
    if downstream_text:
        for entity in known_entities:
            if not phrase_is_grounded(entity, downstream_words):
                warnings.append(
                    f"Semantic consistency: canonical entity {entity!r} does not appear to "
                    "survive into the downstream analysis (root cause / 5-Why / impact)."
                )

    # 2. Placeholder leakage: the final affected_object should never regress
    # to a generic placeholder when a real canonical entity was available.
    if impact_affected_object and _is_placeholder(impact_affected_object) and known_entities:
        warnings.append(
            f"Semantic consistency: impact_assessment.affected_object regressed to a generic "
            f"placeholder ({impact_affected_object!r}) despite a known canonical entity "
            f"{known_entities[0]!r}."
        )

    # 3. New-entity introduction: the downstream affected_object should be
    # grounded in the finding text or the canonical entities, not invented.
    if impact_affected_object and known_entities:
        source_words = significant_words(canonical.raw_finding) | set(significant_words(" ".join(known_entities)))
        if not phrase_is_grounded(impact_affected_object, source_words):
            warnings.append(
                f"Semantic consistency: impact_assessment.affected_object {impact_affected_object!r} "
                "does not trace back to the finding text or canonical entities."
            )

    return warnings
