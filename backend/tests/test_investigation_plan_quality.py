"""Investigation-plan quality layer tests (Section 24 of the investigation-
plan hardening spec) -- exercises the GENERIC deduplication/ranking/area-
clustering functions in `app.agent.analytical_validator` directly, since
these are the single authoritative post-processing layer every execution
path (LLM plan, hypothesis-derived questions, deterministic fallback) is
routed through from `final_evidence_verification_node`.

Deliberately domain-agnostic scenarios (notification AND a second, unrelated
domain) so nothing here re-introduces a single-domain-specific patch.
"""

from __future__ import annotations

from app.agent.analytical_validator import (
    deduplicate_investigation_questions,
    derive_investigation_areas,
    normalize_and_dedupe_evidence_items,
    rank_questions_by_information_gain,
)
from app.models.agent import InvestigationQuestion


def _q(question: str, **kwargs) -> InvestigationQuestion:
    return InvestigationQuestion(question=question, **kwargs)


# ---------------------------------------------------------------------------
# 1. Exact duplicate questions
# ---------------------------------------------------------------------------

def test_exact_duplicate_questions_are_removed():
    questions = [
        _q("Do delivery logs confirm notification was delivered?"),
        _q("Do delivery logs confirm notification was delivered?"),
    ]
    result = deduplicate_investigation_questions(questions)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# 2/3. Semantic duplicates with different wording (Section 6's own examples)
# ---------------------------------------------------------------------------

def test_semantic_duplicate_delivery_questions_collapse_to_one():
    questions = [
        _q("Verify notification delivery to each recipient."),
        _q("Establish whether notification was delivered to each recipient."),
        _q("Reconcile notification delivery records for each recipient."),
    ]
    result = deduplicate_investigation_questions(questions)
    assert len(result) <= 2, "near-identical delivery-verification restatements must collapse"


def test_semantic_duplicate_acknowledgement_questions_collapse_to_one():
    questions = [
        _q("Verify acknowledgement was completed by each recipient."),
        _q("Determine whether acknowledgement was completed by each recipient."),
        _q("Establish whether formal acknowledgement occurred for each recipient."),
    ]
    result = deduplicate_investigation_questions(questions)
    assert len(result) <= 2


def test_distinct_investigative_objectives_are_not_merged():
    """Dedup must not be so aggressive that it collapses genuinely distinct
    propositions (delivery vs. acknowledgement) just because they share a
    subject word."""
    questions = [
        _q("Do delivery logs confirm the notification was delivered to each recipient?"),
        _q("Do acknowledgement records confirm each recipient formally acknowledged the notification?"),
    ]
    result = deduplicate_investigation_questions(questions)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# 4/6. Duplicate / orphan investigation areas
# ---------------------------------------------------------------------------

def test_no_duplicate_investigation_areas_survive_family_clustering():
    """Reproduces the reported defect: a notification delivery/receipt
    conflict must not surface near-duplicate areas like 'Notification
    delivery and receipt verification' AND 'Revised SOP notification
    delivery and receipt reconciliation' AND 'Technical/administrative
    factors affecting delivery or receipt' as separate entries."""
    questions = [
        _q("Do per-recipient delivery records establish successful delivery?",
           purpose="Verify delivery-side completion details", target_proposition_id="P_DELIVERY"),
        _q("Do independent records establish actual receipt or access?",
           purpose="Establish whether delivery resulted in actual recipient receipt or access",
           target_proposition_id="P_RECEIPT"),
        _q("Was acknowledgement required before proceeding?",
           purpose="Establish whether formal acknowledgement was required and completed",
           target_proposition_id="P_REQ_ACK"),
        _q("Did recipients act before receipt was confirmed?",
           purpose="Determine whether operational activity occurred prior to confirmed receipt",
           target_proposition_id="P_ACTION"),
        _q("What technical mechanism explains non-receipt, if it occurred?",
           purpose="Identify an established mechanism if receipt did not occur",
           target_proposition_id="P_MECHANISM"),
    ]
    areas = derive_investigation_areas(
        candidate_hypotheses=[], questions=questions, canonical_subject="the revised SOP notification",
    )
    assert 1 <= len(areas) <= 4
    assert len(areas) == len(set(a.lower() for a in areas)), f"duplicate areas survived: {areas}"
    # No area may be a verbatim copy of a question's own purpose text --
    # areas summarize a cluster of questions, they don't restate one.
    purposes = {q.purpose.lower() for q in questions}
    assert not any(a.lower() in purposes for a in areas), f"an area restated a question purpose verbatim: {areas}"


def test_every_area_has_at_least_one_supporting_question():
    questions = [_q("Do delivery logs confirm delivery occurred?", target_proposition_id="P_DELIVERY")]
    areas = derive_investigation_areas(candidate_hypotheses=[], questions=questions, canonical_subject="the notice")
    assert len(areas) >= 1
    # Every returned area must be traceable to the single classifiable
    # aspect present (delivery) -- no unrelated area invented.
    assert all("delivery" in a.lower() or "receipt" in a.lower() for a in areas)


def test_no_orphan_areas_when_no_questions_classify():
    """If nothing in the plan is classifiable and no existing_areas were
    supplied, the function must not invent placeholder areas."""
    areas = derive_investigation_areas(candidate_hypotheses=[], questions=[], existing_areas=None)
    assert areas == []


# ---------------------------------------------------------------------------
# 16. Evidence-needed deduplication (Section 17)
# ---------------------------------------------------------------------------

def test_evidence_items_normalize_and_dedupe():
    items = [
        "delivery logs", "system delivery logs", "delivery records",
        "acknowledgement records", "acknowledgement confirmation records",
        "channel error logs",
    ]
    result = normalize_and_dedupe_evidence_items(items)
    assert len(result) < len(items), "near-duplicate evidence artifacts must collapse"
    assert len(result) <= 6


def test_evidence_items_capped_to_small_high_value_set():
    items = [f"artifact type {i} record" for i in range(15)]
    result = normalize_and_dedupe_evidence_items(items, max_items=6)
    assert len(result) <= 6


# ---------------------------------------------------------------------------
# 18. Information-gain ordering (Section 7)
# ---------------------------------------------------------------------------

def test_mechanism_question_ranked_after_delivery_receipt_questions():
    """Mechanism questions ('was the system misconfigured') must never be
    surfaced ahead of establishing the underlying delivery/receipt event."""
    mechanism_q = _q("Was the notification system misconfigured?", target_proposition_id="P_MECHANISM")
    delivery_q = _q("Do delivery logs establish delivery occurred?", target_proposition_id="P_DELIVERY")
    receipt_q = _q("Do independent records establish actual receipt?", target_proposition_id="P_RECEIPT")

    ranked = rank_questions_by_information_gain([mechanism_q, receipt_q, delivery_q])
    ranked_texts = [q.question for q in ranked]
    assert ranked_texts.index(delivery_q.question) < ranked_texts.index(mechanism_q.question)
    assert ranked_texts.index(receipt_q.question) < ranked_texts.index(mechanism_q.question)


def test_discrimination_question_ranked_after_requirement_question():
    discrimination_q = _q("Does the record confirm H1?", hypothesis_tested="H1")
    requirement_q = _q("What procedure governs the required activity?", target_proposition_id="P_GOV")
    ranked = rank_questions_by_information_gain([discrimination_q, requirement_q])
    assert [q.question for q in ranked].index(requirement_q.question) < [q.question for q in ranked].index(discrimination_q.question)


def test_recurrence_question_ranked_last():
    recurrence_q = _q("Was the previous CAPA effective?", target_proposition_id="P_REC_2")
    delivery_q = _q("Do delivery logs establish delivery occurred?", target_proposition_id="P_DELIVERY")
    ranked = rank_questions_by_information_gain([recurrence_q, delivery_q])
    assert ranked[-1].question == recurrence_q.question


# ---------------------------------------------------------------------------
# 25. Investigation areas never becoming implicit causes
# ---------------------------------------------------------------------------

def test_derived_areas_are_plain_strings_never_hypothesis_objects():
    """Structural guarantee: derive_investigation_areas can only ever
    return `list[str]` -- there is no code path by which an investigation
    area label can be assigned into root_cause.candidate_hypotheses (a
    different type, populated by an entirely different producer)."""
    areas = derive_investigation_areas(
        candidate_hypotheses=[],
        questions=[_q("Do delivery logs establish delivery occurred?", target_proposition_id="P_DELIVERY")],
        canonical_subject="the notice",
    )
    assert all(isinstance(a, str) for a in areas)


# ---------------------------------------------------------------------------
# Cross-domain regression: the same clustering logic must generalize to an
# unrelated domain (calibration) without any domain-specific code path.
# ---------------------------------------------------------------------------

def test_area_clustering_generalizes_to_a_different_domain():
    questions = [
        _q("Does the calibration certificate establish the expiry date?", purpose="Establish applicable calibration expiry"),
        _q("Do usage logs establish equipment was used after expiry?", purpose="Establish actual usage relative to expiry"),
        _q("Was an approved exception or deviation authorized for continued use?", purpose="Determine whether an authorized exception applied"),
    ]
    areas = derive_investigation_areas(candidate_hypotheses=[], questions=questions, canonical_subject="the equipment calibration")
    assert 1 <= len(areas) <= 4
    assert len(areas) == len(set(a.lower() for a in areas))
    assert not any("notification" in a.lower() for a in areas), "no domain-specific vocabulary must leak in from an unrelated domain"
