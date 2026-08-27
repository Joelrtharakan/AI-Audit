"""Shadow-mode comparison between the existing deterministic pipeline's own
interpretation and the (validated) canonical LLM interpretation.

Per this pass's explicit instruction: the deterministic result remains
AUTHORITATIVE. This module never changes any deterministic output -- it
only records where the two disagree, as a `SemanticDisagreement` list, so
the gap can be observed and closed over time before the canonical path is
ever allowed to become authoritative for semantic interpretation.
"""

from __future__ import annotations

from app.agent.recurrence_guard import detect_recurrence
from app.services.canonical_context_validator import get_affected_object_candidate
from app.services.canonical_semantic_models import CanonicalFindingContext, SemanticDisagreement


def compare_deterministic_vs_canonical(
    finding_text: str,
    deterministic_subject: str | None,
    canonical_context: CanonicalFindingContext | None,
) -> list[SemanticDisagreement]:
    """Compare the deterministic pipeline's own resolved subject/affected
    object and previous-CAPA signal against the canonical (already
    validated/sanitized) context. Returns an empty list if no canonical
    context is available -- shadow comparison is simply skipped, never
    treated as a disagreement."""
    if canonical_context is None:
        return []

    disagreements: list[SemanticDisagreement] = []

    canonical_subject = get_affected_object_candidate(canonical_context)
    det_norm = (deterministic_subject or "").strip().lower()
    can_norm = (canonical_subject or "").strip().lower()
    if det_norm != can_norm:
        disagreements.append(SemanticDisagreement(
            field="affected_object",
            deterministic_value=deterministic_subject,
            canonical_value=canonical_subject,
            disagreement_type="AFFECTED_OBJECT_MISMATCH",
            evidence_ids=[],
            downstream_consequence=(
                "Investigation questions and Five-Why phrasing may reference a different "
                "subject depending on which interpretation is used."
            ),
        ))

    det_capa = detect_recurrence(finding_text).has_previous_capa_reference
    if det_capa != canonical_context.explicit_previous_capa_reference:
        disagreements.append(SemanticDisagreement(
            field="previous_capa_reference",
            deterministic_value=str(det_capa),
            canonical_value=str(canonical_context.explicit_previous_capa_reference),
            disagreement_type="PREVIOUS_CAPA_MISMATCH",
            evidence_ids=list(canonical_context.previous_capa_evidence_ids),
            downstream_consequence="Investigation plan may or may not include previous-CAPA questions depending on which signal is used.",
        ))

    return disagreements
