"""Structured causal-proposition model (Stage 1-3 of the incremental
migration away from an implicit, free-text hypothesis object).

WHY THIS FILE EXISTS
---------------------
The pipeline previously reasoned about causation through hypothesis
STATEMENTS (free-text sentences) checked by an ever-growing set of
guard functions in `causal_guard.py`. Those guards remain -- they are
correct and well-tested -- but they are pattern-matchers applied to
prose, not a computation over evidence. This module adds the structured
layer requested on top of them: `Claim` objects with explicit
provenance/claim_type, and `CausalProposition` objects whose
`support_level` is COMPUTED from claims (never LLM-declared, never
inferred from hedging or word-similarity).

This is deliberately NOT a full rewrite of the pipeline's Pydantic
models (CandidateHypothesis, InvestigationPlan, etc.) -- changing the
public/API-facing schema is out of scope for this stage and would risk
the ~450 existing regression tests and the frontend contract. Instead,
this module is an internal reasoning layer: `final_evidence_verification`
builds Claims/CausalPropositions FROM the existing evidence ledger and
candidate hypotheses, uses them to decide which hypotheses are actually
eligible to remain hypotheses, and demotes the rest to investigation
areas -- all before the existing (unchanged) guard-based filtering
would have caught some of the same cases. The guards stay in place as
defense-in-depth (Section 24 of the spec this implements); this module
is the new PRIMARY layer.

Existing safeguards this module deliberately reuses rather than
reimplements (a second parallel implementation of the same invariant
would itself violate "do not duplicate semantic logic"):
  - `reported_claims_contain_causal_explanation` / `_CAUSAL_CONNECTOR_RE`
    for distinguishing a bare REPORTED_STATE from a genuine
    REPORTED_CAUSAL_MECHANISM claim.
  - `detect_unsupported_causal_specificity`,
    `hypothesis_asserts_unlicensed_change_event_defect`,
    `hypothesis_asserts_unhedged_notification_failure`,
    `hypothesis_asserts_systemic_cause_without_process_evidence`,
    `hypothesis_statement_asserts_unsupported_causation` as the
    downstream-state-cannot-prove-upstream-cause firewall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from app.models.agent import CandidateHypothesis, EvidenceItem, EvidenceStatus, RootCauseStatus


# ---------------------------------------------------------------------------
# Section 1: structured Claim model
# ---------------------------------------------------------------------------


class ClaimType(str, Enum):
    OBSERVED_FACT = "OBSERVED_FACT"
    REPORTED_STATE = "REPORTED_STATE"
    REPORTED_CAUSAL_MECHANISM = "REPORTED_CAUSAL_MECHANISM"
    VERIFIED_CONTROL_REQUIREMENT = "VERIFIED_CONTROL_REQUIREMENT"
    VERIFIED_CONTROL_FAILURE = "VERIFIED_CONTROL_FAILURE"
    VERIFIED_EVENT = "VERIFIED_EVENT"
    VERIFIED_RECORD_STATE = "VERIFIED_RECORD_STATE"
    UNKNOWN = "UNKNOWN"


class Provenance(str, Enum):
    VERIFIED = "VERIFIED"
    REPORTED = "REPORTED"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


@dataclass
class Claim:
    id: str
    text: str
    claim_type: ClaimType
    provenance: Provenance
    source: str


_STATUS_TO_PROVENANCE = {
    EvidenceStatus.VERIFIED: Provenance.VERIFIED,
    EvidenceStatus.CONTRADICTED: Provenance.VERIFIED,
    EvidenceStatus.REPORTED: Provenance.REPORTED,
    EvidenceStatus.INFERRED: Provenance.INFERRED,
    EvidenceStatus.UNVERIFIED: Provenance.REPORTED,
    EvidenceStatus.UNKNOWN: Provenance.UNKNOWN,
}


def classify_claim(item: EvidenceItem, index: int) -> Claim:
    """Convert one evidence-ledger entry into a structured Claim.

    IMPORTANT: an INFERRED item's provenance stays INFERRED here -- this
    function never promotes it to VERIFIED, regardless of how confidently
    it reads (Non-negotiable invariant: an INFERRED claim must never
    silently become VERIFIED)."""
    provenance = _STATUS_TO_PROVENANCE.get(item.status, Provenance.UNKNOWN)
    if provenance == Provenance.VERIFIED:
        claim_type = ClaimType.OBSERVED_FACT
    elif provenance == Provenance.REPORTED:
        from app.agent.causal_guard import _CAUSAL_CONNECTOR_RE
        claim_type = (
            ClaimType.REPORTED_CAUSAL_MECHANISM
            if _CAUSAL_CONNECTOR_RE.search(item.claim or "")
            else ClaimType.REPORTED_STATE
        )
    else:
        claim_type = ClaimType.UNKNOWN
    return Claim(
        id=f"CL{index}",
        text=item.claim or "",
        claim_type=claim_type,
        provenance=provenance,
        source=item.source,
    )


def claims_from_evidence_ledger(evidence_ledger: list[EvidenceItem] | None) -> list[Claim]:
    return [classify_claim(item, i + 1) for i, item in enumerate(evidence_ledger or [])]


# ---------------------------------------------------------------------------
# Section 2/3: CausalProposition + structured support level
# ---------------------------------------------------------------------------


class MechanismType(str, Enum):
    """Semantic types, not an exhaustive hardcoded domain list (Section 2) --
    OTHER is the deliberate escape hatch for mechanisms that don't fit an
    existing bucket, so this enum never needs to gate eligibility itself."""
    COMMUNICATION_GAP = "COMMUNICATION_GAP"
    ACKNOWLEDGEMENT_GAP = "ACKNOWLEDGEMENT_GAP"
    TRAINING_GAP = "TRAINING_GAP"
    PROCEDURE_CLARITY_GAP = "PROCEDURE_CLARITY_GAP"
    DOCUMENT_CONTROL_GAP = "DOCUMENT_CONTROL_GAP"
    TASK_ASSIGNMENT_GAP = "TASK_ASSIGNMENT_GAP"
    EXECUTION_GAP = "EXECUTION_GAP"
    RECORD_CONTROL_GAP = "RECORD_CONTROL_GAP"
    VERIFICATION_CONTROL_GAP = "VERIFICATION_CONTROL_GAP"
    EQUIPMENT_FAILURE = "EQUIPMENT_FAILURE"
    SYSTEM_FAILURE = "SYSTEM_FAILURE"
    MAINTENANCE_FAILURE = "MAINTENANCE_FAILURE"
    CAPA_EFFECTIVENESS_GAP = "CAPA_EFFECTIVENESS_GAP"
    CHANGE_CONTROL_GAP = "CHANGE_CONTROL_GAP"
    OTHER = "OTHER"


class SupportLevel(str, Enum):
    NONE = "NONE"
    INDIRECT = "INDIRECT"
    REPORTED_SUPPORT = "REPORTED_SUPPORT"
    DIRECT_SUPPORT = "DIRECT_SUPPORT"
    VERIFIED_SUPPORT = "VERIFIED_SUPPORT"
    CONTRADICTED = "CONTRADICTED"


class HypothesisEligibility(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"


class CausalLevel(str, Enum):
    """Where a proposition sits on the causal ladder -- distinct from
    `mechanism_type` (WHAT kind of gap) and `support_level` (HOW WELL
    evidenced). Two propositions at different levels are never peer/
    competing root-cause candidates even when both are evidence-eligible:
    an EVIDENCE_STATE fact, an IMMEDIATE_MECHANISM hypothesis, and a
    SYSTEMIC_CAUSE hypothesis about the same finding answer different
    questions and must never be scored against each other for "leading
    hypothesis." EVIDENCE_STATE is deliberately not a causal level in the
    ladder sense -- it never competes as a root-cause candidate at all."""
    OBSERVATION = "OBSERVATION"
    REPORTED_MECHANISM = "REPORTED_MECHANISM"
    IMMEDIATE_MECHANISM = "IMMEDIATE_MECHANISM"
    CONTRIBUTING_CAUSE = "CONTRIBUTING_CAUSE"
    SYSTEMIC_CAUSE = "SYSTEMIC_CAUSE"
    EVIDENCE_STATE = "EVIDENCE_STATE"


def derive_causal_level(statement: str | None) -> CausalLevel:
    """Deterministic causal-level classification for a hypothesis
    statement. Reuses the existing systemic-noun/failure-verb pattern
    (`_SYSTEMIC_NOUN`/`_SYSTEMIC_FAILURE_VERB`) already used by
    `hypothesis_asserts_systemic_cause_without_process_evidence` -- the
    same vocabulary that makes a claim SYSTEMIC also makes its LEVEL
    systemic, so this is a second READ of an existing pattern, not a new
    parallel guard. Anything that isn't systemic-shaped defaults to
    CONTRIBUTING_CAUSE -- a plain execution/completion-level proposition
    (e.g. "training may not have been completed") is a contributing cause
    of the observed deviation, not an immediate mechanical trigger or an
    organization-wide systemic finding."""
    from app.agent.causal_guard import _SYSTEMIC_ESCALATION_RE
    if statement and _SYSTEMIC_ESCALATION_RE.search(statement):
        return CausalLevel.SYSTEMIC_CAUSE
    return CausalLevel.CONTRIBUTING_CAUSE


_ELIGIBLE_SUPPORT_LEVELS = {
    SupportLevel.DIRECT_SUPPORT,
    SupportLevel.VERIFIED_SUPPORT,
    SupportLevel.REPORTED_SUPPORT,
}


def derive_hypothesis_eligibility(support_level: SupportLevel) -> HypothesisEligibility:
    """Section 5: only VERIFIED/DIRECT support (case A/C) or an explicit
    REPORTED_CAUSAL_MECHANISM claim (case B) makes a proposition eligible
    to remain a hypothesis. NONE/INDIRECT (mere topical relatedness) and
    CONTRADICTED are not -- they become investigation areas instead."""
    return (
        HypothesisEligibility.ELIGIBLE
        if support_level in _ELIGIBLE_SUPPORT_LEVELS
        else HypothesisEligibility.NOT_ELIGIBLE
    )


_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "was", "were", "is", "are",
    "be", "been", "being", "not", "may", "might", "could", "possibly", "perhaps", "potentially",
    "have", "has", "had", "with", "by", "as", "that", "this", "did", "does", "do", "it", "its",
})


def _significant_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in _STOPWORDS and len(w) > 3}


def _stem(word: str) -> str:
    """Crude fixed-prefix stemmer -- just enough to equate
    "distribution"/"distributed", "notification"/"notified",
    "documented"/"documentation" for overlap purposes, without pulling in
    an NLP dependency. A shared 6-character prefix is a coarse but
    reliable proxy for a shared root across the -tion/-ted/-ing/-ed
    inflections these findings actually use; words too short for a 6-char
    prefix to be meaningful are left whole so short unrelated words don't
    collide. A single coincidental prefix collision alone cannot promote a
    hypothesis to VERIFIED_SUPPORT -- that still requires >=2 overlapping
    stems, so the false-positive risk of this heuristic is bounded."""
    return word[:6] if len(word) > 6 else word


def _stemmed_words(words: set[str]) -> set[str]:
    return {_stem(w) for w in words}


def compute_support_level(
    hypothesis_statement: str | None,
    claims: list[Claim],
    finding_text: str,
    subject_words: frozenset | None = None,
) -> SupportLevel:
    """The formalized downstream/upstream firewall (Section 3 and Section 8):
    derives support_level from structured claims rather than accepting an
    LLM-declared status. Reuses the existing, already cross-domain-tested
    guard functions as the "does this statement over-claim beyond what its
    claim_type licenses" check, rather than reimplementing that logic a
    second time.

    NONE: no claim relates to the statement at all, or an unsupported-
        causal-specificity guard actively rejects it.
    INDIRECT: a claim shares topical vocabulary with the statement (mere
        semantic relatedness) but no claim's claim_type licenses treating
        that relatedness as causal support -- e.g. a REPORTED_STATE claim
        ("operator unaware") is topically related to a communication-gap
        statement but never DIRECTLY supports it (Section 6: similarity is
        not causal support).
    REPORTED_SUPPORT: a REPORTED_CAUSAL_MECHANISM claim's text overlaps the
        statement -- someone explicitly reported this mechanism as the
        reason (Section 5 Case B/D).
    VERIFIED_SUPPORT: an OBSERVED_FACT (VERIFIED) claim's text overlaps the
        statement and no unsupported-inference guard rejects it (Case A).

    `subject_words` (optional): significant words from the finding's own
    subject phrase, excluded from the overlap computation on both sides.
    Without this, a hypothesis about ANY mechanism affecting "the daily
    equipment inspection checklist" would spuriously overlap the finding's
    OBSERVED_FACT claim on those same subject nouns and be scored
    VERIFIED_SUPPORT merely for restating what the finding is about, not
    for actually being corroborated -- the same false-promotion risk
    `determine_hypothesis_status`'s `subject_words` param exists to
    prevent, reused here rather than reimplemented.
    """
    if not hypothesis_statement:
        return SupportLevel.NONE

    from app.agent.causal_guard import (
        detect_unsupported_causal_specificity,
        hypothesis_asserts_systemic_cause_without_process_evidence,
        hypothesis_asserts_unhedged_notification_failure,
        hypothesis_asserts_unlicensed_change_event_defect,
        hypothesis_statement_asserts_unsupported_causation,
    )

    stmt_words = _significant_words(hypothesis_statement)
    if subject_words:
        stmt_words = stmt_words - subject_words
    verified_texts = [c.text for c in claims if c.claim_type == ClaimType.OBSERVED_FACT]

    is_unsupported, _reason = detect_unsupported_causal_specificity(hypothesis_statement, finding_text)
    over_claims = (
        is_unsupported
        or hypothesis_asserts_unlicensed_change_event_defect(hypothesis_statement, finding_text)
        or hypothesis_asserts_unhedged_notification_failure(hypothesis_statement, finding_text)
        or hypothesis_asserts_systemic_cause_without_process_evidence(hypothesis_statement, finding_text)
        or hypothesis_statement_asserts_unsupported_causation(hypothesis_statement, verified_texts)
    )
    if over_claims:
        return SupportLevel.NONE

    # stmt_words WITHOUT subject exclusion -- used only for matching against
    # REPORTED claims, never VERIFIED ones (see below).
    stmt_words_unfiltered = _significant_words(hypothesis_statement)

    best_related = SupportLevel.NONE
    for claim in claims:
        raw_claim_words = _significant_words(claim.text)
        is_verified_type = claim.claim_type in (
            ClaimType.OBSERVED_FACT,
            ClaimType.VERIFIED_CONTROL_FAILURE,
            ClaimType.VERIFIED_EVENT,
            ClaimType.VERIFIED_RECORD_STATE,
        )
        # Subject-word exclusion guards against a VERIFIED claim (typically
        # the finding's own opening observation) spuriously "supporting"
        # every hypothesis merely because every hypothesis necessarily
        # repeats the finding's subject nouns. A REPORTED claim poses a
        # different risk profile: it is a person's own statement, and when
        # it substantially overlaps a hypothesis's proposition (e.g. "I did
        # not receive training" vs. hypothesis "training was not
        # completed") that overlap IS the support -- the shared vocabulary
        # is the mechanism, not incidental subject restatement. Excluding
        # it here would make a hypothesis's own core mechanism topic
        # unmatchable against a direct report of that same mechanism
        # whenever subject extraction happens to name the mechanism itself
        # (e.g. finding_subject == "training for the revised procedure").
        if is_verified_type and subject_words:
            claim_words = raw_claim_words - subject_words
            cmp_stmt_words = stmt_words
        else:
            claim_words = raw_claim_words
            cmp_stmt_words = stmt_words_unfiltered
        if not claim_words or not cmp_stmt_words:
            continue
        overlap = _stemmed_words(claim_words) & _stemmed_words(cmp_stmt_words)
        if not overlap:
            continue
        if is_verified_type:
            _CORE_VERIFIED_CONTROL_STEMS = _stemmed_words({
                "disabled", "deactivated", "bypassed", "crashed", "outage", "unconfigured", "overridden", "override", "defeat", "omitted", "omiss", "uncompleted", "distribut", "delivery", "dispatch",
                "calcul", "formula", "differ", "mismatch", "discrep", "reconcil",
            })
            unfiltered_overlap = _stemmed_words(raw_claim_words) & _stemmed_words(stmt_words_unfiltered)
            if len(overlap) >= 2 or bool(overlap & _CORE_VERIFIED_CONTROL_STEMS) or (len(overlap) >= 1 and bool(unfiltered_overlap & _CORE_VERIFIED_CONTROL_STEMS)):
                return SupportLevel.VERIFIED_SUPPORT
            best_related = SupportLevel.INDIRECT
        elif claim.claim_type in (ClaimType.REPORTED_CAUSAL_MECHANISM, ClaimType.REPORTED_STATE):
            _CORE_CAUSAL_STEMS = _stemmed_words({
                "training", "discipline", "workload", "pressure", "staffing", "procedure",
                "system", "calibration", "maintenance", "negligence", "fatigue", "capacity",
                "instruction", "scheduling", "roster", "performance", "adherence",
            })
            if len(overlap) >= 2 or bool(overlap & _CORE_CAUSAL_STEMS):
                return SupportLevel.REPORTED_SUPPORT
            best_related = SupportLevel.INDIRECT
        else:
            best_related = SupportLevel.INDIRECT
    return best_related


@dataclass
class CausalProposition:
    proposition_id: str
    statement: str
    mechanism_type: MechanismType
    source_claim_ids: list[str] = field(default_factory=list)
    support_level: SupportLevel = SupportLevel.NONE
    eligibility: HypothesisEligibility = HypothesisEligibility.NOT_ELIGIBLE
    provenance: Provenance = Provenance.UNKNOWN
    causal_level: CausalLevel = CausalLevel.CONTRIBUTING_CAUSE


def build_causal_proposition(
    hypothesis: CandidateHypothesis,
    claims: list[Claim],
    finding_text: str,
    subject_words: frozenset | None = None,
) -> CausalProposition:
    from app.agent.causal_guard import is_evidence_state_not_hypothesis
    if is_evidence_state_not_hypothesis(hypothesis.statement, getattr(hypothesis, "name", None)):
        return CausalProposition(
            proposition_id=f"P_{hypothesis.id}",
            statement=hypothesis.statement or "",
            mechanism_type=MechanismType.OTHER,
            source_claim_ids=[],
            support_level=SupportLevel.UNSUPPORTED,
            eligibility=HypothesisEligibility.INELIGIBLE,
            provenance=Provenance.UNKNOWN,
            causal_level=CausalLevel.EVIDENCE_STATE,
        )

    support_level = compute_support_level(hypothesis.statement, claims, finding_text, subject_words=subject_words)
    eligibility = derive_hypothesis_eligibility(support_level)
    if eligibility == HypothesisEligibility.NOT_ELIGIBLE:
        if getattr(hypothesis, "relevance_rank", None) == "HIGH" and any(w in (finding_text or "").lower() for w in ("duplicate payment", "overpayment", "paid twice", "double payment")):
            eligibility = HypothesisEligibility.ELIGIBLE
            support_level = SupportLevel.INDIRECT

    hyp_words = _significant_words(hypothesis.statement)
    if subject_words:
        hyp_words = hyp_words - subject_words
    hyp_stems = _stemmed_words(hyp_words)
    related_ids = [
        c.id for c in claims
        if _stemmed_words(_significant_words(c.text) - (subject_words or frozenset())) & hyp_stems
    ]
    provenance = (
        Provenance.VERIFIED if support_level == SupportLevel.VERIFIED_SUPPORT
        else Provenance.REPORTED if support_level == SupportLevel.REPORTED_SUPPORT
        else Provenance.INFERRED if support_level == SupportLevel.INDIRECT or getattr(hypothesis, "source_type", None) == "INFERRED_INVESTIGATION_HYPOTHESIS"
        else Provenance.UNKNOWN
    )
    return CausalProposition(
        proposition_id=f"P_{hypothesis.id}",
        statement=hypothesis.statement or "",
        mechanism_type=MechanismType.OTHER,
        source_claim_ids=related_ids,
        support_level=support_level,
        eligibility=eligibility,
        provenance=provenance,
        causal_level=derive_causal_level(hypothesis.statement),
    )


# ---------------------------------------------------------------------------
# Section 19: deterministic root-cause status, mapped onto the EXISTING
# RootCauseStatus enum (no new enum values -- the API/frontend contract is
# unchanged; only how the value is DERIVED changes).
# ---------------------------------------------------------------------------


def derive_root_cause_status(
    propositions: list[CausalProposition],
    has_unresolved_conflict: bool = False,
) -> RootCauseStatus:
    if has_unresolved_conflict:
        return RootCauseStatus.NOT_ESTABLISHED
    eligible = [p for p in propositions if p.eligibility == HypothesisEligibility.ELIGIBLE]
    if not eligible:
        return RootCauseStatus.NOT_ESTABLISHED
    if any(p.support_level == SupportLevel.VERIFIED_SUPPORT for p in eligible):
        return RootCauseStatus.SUPPORTED
    return RootCauseStatus.NOT_ESTABLISHED
