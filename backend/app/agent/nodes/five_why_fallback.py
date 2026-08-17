"""Deterministic fallback 5-Why generator.

Fired when the 5-Why array is empty or LLM core synthesis fails. Uses the
canonical case model (extracted facts, reported statements, evidence conflicts)
to construct an evidence-bound 5-Why chain that NEVER outputs "Analysis unavailable".
"""

from __future__ import annotations

from app.models.agent import EvidenceItem, EvidenceStatus, FiveWhyAnalysis, FiveWhyStep


_DEGRADED_SUBJECTS = {"process compliance", None, ""}


def build_deterministic_five_why(
    finding_text: str,
    evidence_ledger: list[EvidenceItem],
    canonical_subject: str | None = None,
) -> FiveWhyAnalysis:
    """Build a deterministic, evidence-bound 5-Why chain from the case model.

    Enforces deterministic 5-Why stopping policy:
      - Never fabricates causal layers beyond available evidence.
      - Handles conflicting reports with explicit conflict recognition.
      - Stops at the evidence boundary with UNKNOWN status.
    """
    from app.agent.causal_guard import extract_immediate_mechanism, repeats_previous_why_answer
    from app.agent.claim_extractor import detect_evidence_conflicts, extract_claims
    from app.services.semantic_subject import extract_temporal_clause, format_deviation_why_question, resolve_deviation, topic_word

    steps: list[FiveWhyStep] = []

    fact_claims = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.VERIFIED]
    reported_claims = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.REPORTED]
    resolved = resolve_deviation(finding_text, fact_claims)
    if canonical_subject and canonical_subject not in _DEGRADED_SUBJECTS:
        noun_sub = canonical_subject
    else:
        noun_sub = resolved.finding_subject or resolved.subject or "the affected process"
    deviation_desc = resolved.deviation or f"{noun_sub} condition noted in finding"

    claims = extract_claims(finding_text, evidence_ledger)
    conflicts = detect_evidence_conflicts(claims)
    mechanism = extract_immediate_mechanism(reported_claims, fact_claims)

    # 1. Conflicting Evidence Case (e.g. operator vs supervisor reports)
    if conflicts:
        conflict = conflicts[0]
        topic = topic_word(noun_sub)

        # DELIVERY_VS_RECEIPT (Conflict-Center hardening, Section 10/11): the
        # 5-Why chain must NOT choose either conflicting proposition as fact
        # ("Why were the operators not aware..." presupposes non-receipt;
        # "Why was the notification not received..." presupposes delivery
        # succeeded). The question must stay proposition-neutral and stop
        # immediately at the evidence boundary -- never invent a second or
        # third causal layer while the conflict itself remains unresolved.
        if getattr(conflict, "proposition_type", None) == "DELIVERY_VS_RECEIPT":
            steps.append(FiveWhyStep(
                question=(
                    f"Why do system records indicate successful delivery of {noun_sub} while the affected "
                    "personnel report that they did not receive it?"
                ),
                answer=(
                    "Objective verification is required to resolve the conflicting evidence and determine "
                    "whether the discrepancy concerns delivery, receipt, accessibility, acknowledgement, or "
                    "another mechanism."
                ),
                status="UNKNOWN",
            ))
            return FiveWhyAnalysis(
                steps=steps,
                is_complete=False,
                status_note=(
                    "Evidence boundary reached — conflicting delivery/receipt evidence requires objective "
                    "verification before the chain can proceed."
                ),
            )
        # Step 1 content: why was the proposition unconfirmed — grounded in
        # the actual reported claim texts, with speaker attribution when
        # available, instead of a generic "conflicting reports" summary.
        reported = [c for c in claims if getattr(c, "status", None) == EvidenceStatus.REPORTED]
        if len(reported) >= 2:
            first, second = reported[0], reported[1]
            first_desc = f"{first.speaker} stated {first.predicate or first.text}" if first.speaker else first.text
            second_desc = f"{second.speaker} stated {second.predicate or second.text}" if second.speaker else second.text
            conflict_summary = (
                f"Available evidence contains conflicting reported statements: {first_desc}, "
                f"while {second_desc}."
            )
        else:
            conflict_summary = f"Available evidence contains conflicting reports regarding {conflict.proposition}."

        # THE VERIFIED DEVIATION IS NEVER "UNCONFIRMED": if a VERIFIED
        # claim independent of the conflict itself directly states the
        # observed deviation (a finding conventionally opens with this
        # observation), the chain must start from that fact -- never ask
        # whether an already-verified observation is "unconfirmed"; only
        # its CAUSE is in question. Detected structurally: the finding's
        # first claim is VERIFIED and is not one of the two conflicting
        # claims.
        conflict_claim_ids = set(conflict.claims)
        first_claim = claims[0] if claims else None
        has_separate_verified_deviation = bool(
            first_claim is not None
            and getattr(first_claim, "status", None) == EvidenceStatus.VERIFIED
            and first_claim.claim_id not in conflict_claim_ids
        )

        if has_separate_verified_deviation:
            from app.services.semantic_subject import _strip_framing, declarative_to_why_question
            deviation_fact = _strip_framing(first_claim.text).strip()
            # WHY#1 must EXPLAIN the deviation, not restate it as its own
            # answer -- a verified RECORD-level deviation (e.g. a record was
            # incomplete/missing) never by itself establishes what happened
            # at the EVENT level (the underlying activity itself). Naming
            # that unresolved event-vs-record gap is the actual explanation
            # the evidence supports at this step; restating the same fact
            # back as the answer is circular and explains nothing.
            deviation_clause = deviation_fact.rstrip(".")
            if deviation_clause and deviation_clause[0].isupper() and not deviation_clause.split()[0].isupper():
                deviation_clause = deviation_clause[0].lower() + deviation_clause[1:]
            why1_answer = (
                f"The available evidence establishes that {deviation_clause}, but it does not establish "
                "whether the underlying activity did not occur, occurred but was not recorded, or was "
                "affected by another process failure."
            )
            steps.append(FiveWhyStep(
                question=declarative_to_why_question(first_claim.text),
                answer=why1_answer,
                status="VERIFIED",
            ))
            steps.append(FiveWhyStep(
                question=f"Why did this breakdown occur in the process for {noun_sub}?",
                answer=conflict_summary,
                status="MIXED",
            ))
            steps.append(FiveWhyStep(
                question="Why can the actual mechanism not yet be established?",
                answer=(
                    "Objective evidence has not yet established the underlying mechanism from the "
                    "available evidence."
                ),
                status="UNKNOWN",
            ))
            return FiveWhyAnalysis(
                steps=steps,
                is_complete=False,
                status_note="Evidence boundary reached — conflicting reported explanations require objective record verification.",
            )

        # No separately-verified deviation exists -- the observation itself
        # (not just its cause) is what the conflicting reports concern, so
        # the 2-step chain below correctly frames the uncertainty as being
        # about the proposition itself.
        # noun_sub is often already a "<topic> compliance for X" phrase (e.g.
        # "training compliance for the revised procedure") — don't re-prefix
        # it with the topic word a second time.
        subject_phrase = noun_sub if topic in noun_sub.lower() else f"{topic} compliance for {noun_sub}"
        steps.append(FiveWhyStep(
            question=f"Why was {subject_phrase} unconfirmed?",
            answer=conflict_summary,
            status="MIXED",
        ))
        # Step 2: why could completion not be established — evidence boundary.
        steps.append(FiveWhyStep(
            question=f"Why could {topic} completion not be established?",
            answer=f"No objective {topic}-completion record has been verified from the available evidence.",
            status="UNKNOWN",
        ))
        return FiveWhyAnalysis(
            steps=steps,
            is_complete=False,
            status_note="Evidence boundary reached — conflicting reported statements require objective record verification.",
        )

    # 2. Single Reported Mechanism (e.g. Case 1, Case 2, Case 3)
    if mechanism.status == "REPORTED" and mechanism.statement:
        # WHY#1 must be a grammatical question built from subject/condition
        # (never the raw dash-joined "subject — condition" deviation string
        # interpolated into "Why did X occur?"), and its answer should be
        # the finding's own VERIFIED sentence when one exists -- not that
        # same dash-joined fragment repeated back as prose.
        why1_question = format_deviation_why_question(
            resolved.subject or noun_sub, resolved.condition, extract_temporal_clause(finding_text)
        )
        why1_answer = fact_claims[0] if fact_claims else deviation_desc
        steps.append(FiveWhyStep(
            question=why1_question,
            answer=why1_answer,
            status="VERIFIED" if fact_claims else "REPORTED",
        ))
        if not repeats_previous_why_answer(why1_answer, mechanism.statement):
            steps.append(FiveWhyStep(
                question=f"What immediate mechanism explains the nonconformity in {noun_sub}?",
                answer=mechanism.statement,
                status="REPORTED",
            ))
        steps.append(FiveWhyStep(
            question=f"Why did this breakdown occur in the process for {noun_sub}?",
            answer="NOT ESTABLISHED FROM AVAILABLE EVIDENCE — objective verification required to confirm underlying cause.",
            status="UNKNOWN",
        ))
        return FiveWhyAnalysis(
            steps=steps,
            is_complete=False,
            status_note="DEGRADED MODE — CHAIN STOPPED AT EVIDENCE BOUNDARY — Mechanism is reported; objective root cause requires investigation.",
        )

    # 3. Verified Mechanism
    if mechanism.status == "VERIFIED" and mechanism.statement:
        steps.append(FiveWhyStep(
            question=f"Why was {noun_sub} nonconforming?",
            answer=mechanism.statement,
            status="VERIFIED",
        ))
        steps.append(FiveWhyStep(
            question=f"Why did this breakdown occur in the process for {noun_sub}?",
            answer="NOT ESTABLISHED FROM AVAILABLE EVIDENCE — objective verification required to confirm underlying root cause.",
            status="UNKNOWN",
        ))
        return FiveWhyAnalysis(
            steps=steps,
            is_complete=False,
            status_note="DEGRADED MODE — CHAIN STOPPED AT EVIDENCE BOUNDARY — Root cause not established from initial evidence.",
        )

    # 4. General fallback when no mechanism is present
    steps.append(FiveWhyStep(
        question=format_deviation_why_question(
            resolved.subject or noun_sub, resolved.condition, extract_temporal_clause(finding_text)
        ),
        answer=fact_claims[0] if fact_claims else deviation_desc,
        status="REPORTED" if not fact_claims else "VERIFIED",
    ))
    steps.append(FiveWhyStep(
        question=f"Why did the deviation in {noun_sub} occur?",
        answer="NOT ESTABLISHED FROM AVAILABLE EVIDENCE — objective investigation required to confirm underlying cause.",
        status="UNKNOWN",
    ))
    return FiveWhyAnalysis(
        steps=steps,
        is_complete=False,
        status_note="DEGRADED MODE — CHAIN STOPPED AT EVIDENCE BOUNDARY — Root cause not established from initial evidence.",
    )
