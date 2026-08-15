"""Deterministic fallback 5-Why generator (Section 10).

Fired when the 5-Why array is empty or LLM core synthesis fails. Uses the
canonical case model (extracted facts, reported statements, evidence ledger)
to construct an evidence-bound 5-Why chain that NEVER outputs "Analysis unavailable".
"""

from __future__ import annotations

from app.models.agent import EvidenceItem, EvidenceStatus, FiveWhyAnalysis, FiveWhyStep


def build_deterministic_five_why(
    finding_text: str,
    evidence_ledger: list[EvidenceItem],
) -> FiveWhyAnalysis:
    """Build a deterministic, evidence-bound 5-Why chain from the case model.

    This only runs in DEGRADED MODE (the LLM call failed or returned
    unusable JSON) -- it must never fabricate a causal explanation the
    evidence doesn't support. Where an immediate mechanism is available
    (Layer 2, see app/agent/causal_guard.py) it becomes WHY-1's answer and
    WHY-2 asks the genuinely open question (why did that mechanism occur);
    where none is available, the chain honestly stops at WHY-1 rather than
    manufacturing a generic WHY-2 question that isn't grounded in anything."""
    from app.agent.causal_guard import extract_immediate_mechanism, repeats_previous_why_answer
    from app.services.semantic_subject import resolve_deviation

    steps: list[FiveWhyStep] = []

    fact_claims = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.VERIFIED]
    reported_claims = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.REPORTED]
    raw_obs = fact_claims[0] if fact_claims else finding_text.strip()
    resolved = resolve_deviation(finding_text, fact_claims)
    noun_sub = resolved.subject or "UNKNOWN — no affected object could be isolated from the finding text"

    mechanism = extract_immediate_mechanism(reported_claims, fact_claims)

    # WHY 1: the observation itself (Level 1 Verified Fact). If the
    # mechanism was found directly among VERIFIED facts, it already IS the
    # most specific verified statement available, so it becomes the WHY-1
    # answer directly rather than a separate, less specific one.
    steps.append(FiveWhyStep(
        level=1,
        question=f"Why was the {noun_sub.lower()} incomplete or nonconforming?",
        answer=mechanism.statement if mechanism.status == "VERIFIED" else raw_obs,
        status="VERIFIED",
        evidence_reference="Auditor finding text",
    ))

    if mechanism.status == "REPORTED":
        # WHY 2 (only when the mechanism is a distinct fact beyond the raw
        # observation, e.g. a technician's account of *how* it happened):
        # present the mechanism itself as the answer, sourced to the
        # reported statement -- this is Layer 2, not yet an explanation of
        # WHY that mechanism occurred.
        if not repeats_previous_why_answer(raw_obs, mechanism.statement):
            steps.append(FiveWhyStep(
                level=2,
                question=f"Why did the {noun_sub.lower()} condition occur?",
                answer=mechanism.statement,
                status="REPORTED",
                evidence_reference="Reported statement in finding",
            ))
        # The next WHY (why did the mechanism occur) is a genuinely open
        # causal question this fallback cannot answer without the LLM --
        # asking it honestly and stopping is the correct, non-fabricating
        # outcome, not a failure.
        steps.append(FiveWhyStep(
            level=len(steps) + 1,
            question=f"Why did the following occur: {mechanism.statement}?",
            answer="NOT ESTABLISHED FROM AVAILABLE EVIDENCE — DEGRADED MODE, LLM-based causal analysis was unavailable.",
            status="UNKNOWN",
            evidence_reference="Requires auditor investigation",
        ))
    elif mechanism.status == "VERIFIED":
        # WHY-1's answer already IS the mechanism; WHY 2 asks why it
        # occurred and, without the LLM, honestly stops there.
        steps.append(FiveWhyStep(
            level=2,
            question=f"Why did the following occur: {mechanism.statement}?",
            answer="NOT ESTABLISHED FROM AVAILABLE EVIDENCE — DEGRADED MODE, LLM-based causal analysis was unavailable.",
            status="UNKNOWN",
            evidence_reference="Requires auditor investigation",
        ))
    # else: no mechanism (VERIFIED or REPORTED) is available at all -- the
    # chain honestly stops at WHY-1 rather than manufacturing a generic
    # WHY-2 question ("why were the required entries or controls missing")
    # that isn't grounded in anything the evidence actually supports.

    return FiveWhyAnalysis(
        steps=steps,
        is_complete=False,
        status_note="DEGRADED MODE — LLM-based causal analysis was unavailable; chain stopped at evidence boundary. Auditor investigation required.",
    )
