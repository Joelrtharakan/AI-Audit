"""Deterministic validation/sanitization for the LLM-produced
`CanonicalFindingContext`.

Unlike the financial calculation validator (which accepts/rejects whole
calculation proposals), this validator SANITIZES the canonical context in
place: anything unsupported by real evidence is stripped or forced to its
safe default rather than causing the whole context to be discarded, since
most of a canonical interpretation can be perfectly sound even if one
field is not.

This is where the "never inferred from recurrence alone" and "never a
financial/recovery fact as a cause" rules are enforced independently of
whatever the LLM claims -- the LLM's own `explicit_previous_capa_reference`
and `is_causal` flags are never trusted on their own.
"""

from __future__ import annotations

import re

from app.agent.recurrence_guard import detect_recurrence
from app.models.agent import EvidenceItem
from app.services.canonical_semantic_models import CanonicalFindingContext

_NON_CAUSAL_KINDS = {"CONSEQUENCE", "FINANCIAL_METRIC", "HISTORICAL_CONTEXT", "REMEDIATION", "RECOVERY"}
_ENTITY_LIKE_KINDS = {"ENTITY"}
# A candidate deviation must be grounded in a genuine OCCURRENCE, never a
# bare entity mention, a state, or a financial/historical/recovery/
# remediation fact -- see the primary_deviation grounding check below.
_DEVIATION_GROUNDING_KINDS = {"EVENT"}

# Structural, domain-neutral connective/hedge phrases that assert
# ASSOCIATION or co-occurrence without asserting causation or established
# fact ("associated with", "in connection with", ...) -- the same class of
# fixed, generic structural marker already used elsewhere in this codebase
# (e.g. the historical-framing marker), not a per-finding keyword list.
# When a candidate entity's ONLY supporting evidence is a sentence using
# language like this, the entity is a MENTION inside a hedged/associative
# clause, not an independently established affected object -- generalizes
# to any finding using this class of wording, not any specific phrase.
_ASSOCIATIVE_HEDGE_RE = re.compile(
    r"\b(?:associated\s+with|in\s+connection\s+with|in\s+relation\s+to|"
    r"related\s+to|linked\s+to|in\s+relation\s+with|connected\s+(?:to|with))\b",
    re.IGNORECASE,
)


def _valid_evidence_ids(evidence_count: int) -> set[str]:
    return {f"E{i}" for i in range(evidence_count)}


def validate_canonical_context(
    context: CanonicalFindingContext,
    evidence_ledger: list[EvidenceItem],
    finding_text: str = "",
) -> CanonicalFindingContext:
    """Return a sanitized copy of `context` -- every field either passed
    validation as-is or was stripped/forced to its safe default."""

    valid_ids = _valid_evidence_ids(len(evidence_ledger))
    evidence_text_by_id = {f"E{i}": e.claim for i, e in enumerate(evidence_ledger)}
    sanitized = context.model_copy(deep=True)

    # 1/2. Entities must be evidence-supported; a STATE (or any non-ENTITY
    # kind) is retained for context but is never a valid affected-object
    # candidate -- see get_affected_object_candidate() below, which is the
    # only function downstream modules should call.
    sanitized.entities = [
        e for e in sanitized.entities
        if e.source_evidence_ids and all(eid in valid_ids for eid in e.source_evidence_ids)
    ]

    # PROPERTY 1/3 (financial-consequence firewall): an ENTITY-kind
    # candidate whose ONLY supporting evidence is a sentence using
    # associative/hedging language ("associated with", "linked to", ...)
    # is a MENTION inside a hedged clause, not an independently
    # established affected object -- e.g. "losses associated with the
    # same control failure" never establishes "the same control failure"
    # as a real entity, regardless of what kind the LLM tagged it. An
    # EVENT-kind entity is exempt: an occurrence can legitimately be the
    # thing a hedge clause is ABOUT (see Example B's trailing "resulting
    # in" clause), only a claimed ENTITY mention is this fragile.
    def _is_hedge_only(entity) -> bool:
        if entity.kind != "ENTITY" or not entity.source_evidence_ids:
            return False
        texts = [evidence_text_by_id.get(eid, "") for eid in entity.source_evidence_ids]
        return bool(texts) and all(_ASSOCIATIVE_HEDGE_RE.search(t) for t in texts)

    sanitized.entities = [e for e in sanitized.entities if not _is_hedge_only(e)]

    entity_and_claim_ids = {e.entity_id for e in sanitized.entities}
    entity_kind_by_id = {e.entity_id: e.kind for e in sanitized.entities}
    financial_claim_ids = {c.claim_id for c in sanitized.financial.claims}
    entity_and_claim_ids |= financial_claim_ids

    # 3/6. primary_deviation must reference real evidence via a real claim
    # or entity id, AND that id must be a genuine OCCURRENCE (EVENT-kind),
    # never a bare entity mention, a state, a financial/historical/
    # recovery/remediation fact, or a financial.claims id -- a financial
    # consequence can never itself BE the deviation under investigation
    # (Section 3: priority 1-3 require an explicitly stated deviation/
    # nonconformance before financial consequences are even considered).
    # Falls back to NOT_ESTABLISHED rather than keep an ungrounded string.
    if sanitized.primary_deviation_claim_id:
        _dev_id = sanitized.primary_deviation_claim_id
        _grounded = (
            _dev_id in entity_kind_by_id
            and entity_kind_by_id[_dev_id] in _DEVIATION_GROUNDING_KINDS
        )
        if not _grounded:
            sanitized.primary_deviation = None
            sanitized.primary_deviation_claim_id = None
            sanitized.primary_deviation_confidence = "NOT_ESTABLISHED"
    if not sanitized.primary_deviation:
        sanitized.primary_deviation_confidence = "NOT_ESTABLISHED"

    # 8/9/12. Causal claims: cause_ref/effect_ref must reference real
    # entities/claims, must cite real evidence, and a financial/recovery/
    # remediation/historical fact can never be the CAUSE side of a causal
    # claim regardless of what the LLM's is_causal flag says.
    sanitized_causal = []
    entity_kind_by_id = {e.entity_id: e.kind for e in sanitized.entities}
    for cc in sanitized.causal_claims:
        cc = cc.model_copy(deep=True)
        has_evidence = bool(cc.source_evidence_ids) and all(eid in valid_ids for eid in cc.source_evidence_ids)
        if not has_evidence:
            cc.is_causal = False
        if cc.is_causal and cc.cause_ref:
            cause_kind = entity_kind_by_id.get(cc.cause_ref)
            if cause_kind in _NON_CAUSAL_KINDS:
                cc.is_causal = False
        if cc.is_causal and (not cc.cause_ref or not cc.effect_ref):
            # A causal claim with no identified cause/effect reference is
            # not a usable causal claim.
            cc.is_causal = False
        sanitized_causal.append(cc)
    sanitized.causal_claims = sanitized_causal

    # 14. Previous CAPA: never trusted from the LLM alone -- must ALSO be
    # confirmed by the existing deterministic structural detector
    # (app.agent.recurrence_guard.detect_recurrence), which recognizes
    # explicit prior-CAPA phrasing independent of language-model output.
    # Both must agree; recurrence/historical/repeated wording alone can
    # never satisfy this on its own from either side.
    if sanitized.explicit_previous_capa_reference:
        has_real_evidence = bool(sanitized.previous_capa_evidence_ids) and all(
            eid in valid_ids for eid in sanitized.previous_capa_evidence_ids
        )
        deterministic_confirms = detect_recurrence(finding_text).has_previous_capa_reference
        if not (has_real_evidence and deterministic_confirms):
            sanitized.explicit_previous_capa_reference = False
            sanitized.previous_capa_evidence_ids = []

    return sanitized


def get_affected_object_candidate(context: CanonicalFindingContext | None) -> str | None:
    """The ONLY function downstream modules should use to read an
    affected object off a (validated) canonical context. Returns the
    entity tied to the primary deviation if one exists, else the first
    genuine ENTITY-kind (never STATE/FINANCIAL_METRIC/RECOVERY/...)
    entity, else None. Never returns a state word or a raw clause."""
    if context is None:
        return None
    if context.primary_deviation_claim_id:
        for e in context.entities:
            if e.entity_id == context.primary_deviation_claim_id and e.kind in _ENTITY_LIKE_KINDS:
                return e.name
    for e in context.entities:
        if e.kind in _ENTITY_LIKE_KINDS:
            return e.name
    return None
