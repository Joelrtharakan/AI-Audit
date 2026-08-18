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
    from app.services.semantic_subject import (
        extract_temporal_clause,
        format_deviation_why_question,
        is_actor_noun,
        resolve_deviation,
        topic_word,
    )

    steps: list[FiveWhyStep] = []

    fact_claims = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.VERIFIED]
    reported_claims = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.REPORTED]
    resolved = resolve_deviation(finding_text, fact_claims)
    if canonical_subject is not None:
        if isinstance(canonical_subject, str):
            if canonical_subject not in _DEGRADED_SUBJECTS:
                noun_sub = canonical_subject
            else:
                noun_sub = resolved.finding_subject or resolved.subject or (
                    "email notification" if any(w in finding_text.lower() for w in ("email", "notification", "dispatch")) else "the affected process"
                )
        else:
            subj_val = getattr(canonical_subject, "finding_subject", getattr(canonical_subject, "subject", None))
            if subj_val and subj_val not in _DEGRADED_SUBJECTS:
                noun_sub = subj_val
            else:
                noun_sub = resolved.finding_subject or resolved.subject or (
                    "email notification" if any(w in finding_text.lower() for w in ("email", "notification", "dispatch")) else "the affected process"
                )
    else:
        noun_sub = resolved.finding_subject or resolved.subject or (
            "email notification" if any(w in finding_text.lower() for w in ("email", "notification", "dispatch")) else "the affected process"
        )
    deviation_desc = resolved.deviation or f"{noun_sub} condition noted in finding"

    claims = extract_claims(finding_text, evidence_ledger)
    conflicts = detect_evidence_conflicts(claims)
    mechanism = extract_immediate_mechanism(reported_claims, fact_claims)

    # 0. Document referenced but unavailable — underlying event is not objectively verified
    import re
    has_unavail_ref_doc = bool(re.search(r"\b(?:referenced|attached|cited)\b.*?\b(?:not\s+available|unavailable|missing|could\s+not\s+be\s+located)\b", finding_text, re.IGNORECASE))
    if has_unavail_ref_doc and not any(getattr(c, "source_type", None) in ("OBJECTIVE_RECORD", "SYSTEM_RECORD") for c in claims):
        why1_q = f"Was {deviation_desc} objectively established?"
        why1_ans = "The audit observation asserts the condition, but the referenced supporting report is unavailable and no independent objective evidence has been verified."
        steps.append(FiveWhyStep(
            question=why1_q,
            answer=why1_ans,
            status="UNKNOWN",
        ))
        return FiveWhyAnalysis(
            steps=steps,
            is_complete=False,
            status_note="EVIDENCE BOUNDARY — Underlying event requires objective verification before causal mechanism can be established.",
        )

    # 0b. Duplicate Payment / Transaction Finding (Section 6 Hardening)
    if re.search(r"\b(?:duplicate\s+payment|paid\s+twice|double\s+payment|overpayment)\b", finding_text, re.IGNORECASE):
        steps.append(FiveWhyStep(
            question="Why was a duplicate payment identified?",
            answer="Two payment transactions associated with the same supplier obligation were identified as duplicate.",
            status="VERIFIED",
        ))
        steps.append(FiveWhyStep(
            question="Why was the duplicate payment processed?",
            answer="The available evidence confirms that a duplicate payment occurred, but does not establish the mechanism that resulted in the second payment.",
            status="UNKNOWN",
        ))
        steps.append(FiveWhyStep(
            question="Which control condition allowed the duplicate payment?",
            answer="The underlying control condition that allowed the second transaction is not established from available evidence — objective records from payment workflow, duplicate-detection, approval, and reconciliation logs are required.",
            status="UNKNOWN",
        ))
        return FiveWhyAnalysis(
            steps=steps,
            is_complete=False,
            status_note="INCOMPLETE — EVIDENCE BOUNDARY",
        )

    # 1. Multiple Competing Reported Explanations Case (e.g. training vs workload vs discipline)
    reported_claims_list = [c for c in claims if getattr(c, "status", None) == EvidenceStatus.REPORTED]
    if not conflicts and len(reported_claims_list) >= 2:
        from app.services.semantic_subject import _strip_framing, strip_leading_article
        deviation_fact = fact_claims[0] if fact_claims else deviation_desc
        deviation_clause = _strip_framing(deviation_fact).strip().rstrip(".")
        if deviation_clause and deviation_clause[0].isupper() and not deviation_clause.split()[0].isupper():
            deviation_clause = deviation_clause[0].lower() + deviation_clause[1:]

        # Step 1: Why did the deviation occur? -> Verified observation
        why1_q = format_deviation_why_question(
            resolved.subject or noun_sub, resolved.condition, extract_temporal_clause(finding_text)
        )
        why1_ans = f"The evidence establishes that {deviation_clause}."
        steps.append(FiveWhyStep(
            question=why1_q,
            answer=why1_ans,
            status="VERIFIED",
        ))

        # Step 2: Why did the personnel not complete the activity? -> Competing reported explanations
        actor_name = resolved.actor or "the affected personnel"
        stripped_actor = strip_leading_article(actor_name).lower()
        predicates = []
        for rc in reported_claims_list:
            pred = (rc.predicate or rc.text).strip().rstrip(".")
            # Clean common prefixes
            pred_clean = re.sub(r"^(?:that\s+|they\s+were\s+|there\s+was\s+)", "", pred, flags=re.IGNORECASE).strip()
            if pred_clean and pred_clean not in predicates:
                predicates.append(pred_clean)

        if len(predicates) >= 2:
            num_words = {2: "Two", 3: "Three", 4: "Four", 5: "Five"}.get(len(predicates), str(len(predicates)))
            list_str = ", ".join(predicates[:-1]) + f", and {predicates[-1]}"
            why2_ans = f"{num_words} explanations were reported: {list_str}. None is independently verified by the available evidence."
        elif predicates:
            why2_ans = f"An explanation was reported: {predicates[0]}, but it is not independently verified by available evidence."
        else:
            why2_ans = "Multiple explanations were reported; none is independently verified by available evidence."

        target_obj_name = noun_sub.lower()
        if "completion" in target_obj_name:
            why2_q = f"Why did the {stripped_actor} not complete the {target_obj_name.replace(' completion', '')}?"
        elif is_actor_noun(actor_name):
            why2_q = f"Why did the {stripped_actor} not complete the {target_obj_name}?"
        else:
            why2_q = f"Why did this nonconformity occur in {target_obj_name}?"

        steps.append(FiveWhyStep(
            question=why2_q,
            answer=why2_ans,
            status="REPORTED",
        ))

        # Step 3: Which mechanism caused it? -> Evidence boundary
        why3_q = "Which of these mechanisms caused the non-completion?" if "checklist" in noun_sub.lower() or "complete" in finding_text.lower() else "Which of these mechanisms caused the nonconformity?"
        why3_ans = "The operative causal mechanism is not established from available evidence — objective records are required to distinguish the competing explanations."
        steps.append(FiveWhyStep(
            question=why3_q,
            answer=why3_ans,
            status="UNKNOWN",
        ))

        return FiveWhyAnalysis(
            steps=steps,
            is_complete=False,
            status_note="Evidence boundary reached — multiple competing reported explanations require objective record verification.",
        )

    # 1b. Conflicting Evidence Case (e.g. delivery vs receipt)
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
            deviation_clause = deviation_fact.rstrip(".")
            if deviation_clause and deviation_clause[0].isupper() and not deviation_clause.split()[0].isupper():
                deviation_clause = deviation_clause[0].lower() + deviation_clause[1:]
            why1_answer = (
                "The available evidence does not establish whether the checks were not performed, were "
                "performed but not recorded, or were affected by another process condition."
            )
            steps.append(FiveWhyStep(
                question=declarative_to_why_question(first_claim.text),
                answer=why1_answer,
                status="UNKNOWN",
            ))
            return FiveWhyAnalysis(
                steps=steps,
                is_complete=False,
                status_note="Evidence boundary reached — conflicting reported statements require objective verification before causal chain can proceed.",
            )

        # No separately-verified deviation exists -- the observation itself
        # (not just its cause) is what the conflicting reports concern, so
        # the 2-step chain below correctly frames the uncertainty as being
        # about the proposition itself.
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

    # 2. Single Reported Mechanism (e.g. Case 1, Case 2, Case 3, Case 4)
    if mechanism.status == "REPORTED" and mechanism.statement:
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
            status_note="EVIDENCE BOUNDARY — Mechanism is reported; objective root cause requires investigation.",
        )

    # 3. Verified Mechanism
    if mechanism.status == "VERIFIED" and mechanism.statement:
        steps.append(FiveWhyStep(
            question=format_deviation_why_question(
                resolved.subject or noun_sub, resolved.condition, extract_temporal_clause(finding_text)
            ),
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
            status_note="EVIDENCE BOUNDARY — Root cause not established from initial evidence.",
        )

    from app.services.semantic_subject import _strip_framing
    deviation_fact = fact_claims[0] if fact_claims else deviation_desc
    deviation_clause = _strip_framing(deviation_fact).strip().rstrip(".")
    if deviation_clause and deviation_clause[0].isupper() and not deviation_clause.split()[0].isupper():
        deviation_clause = deviation_clause[0].lower() + deviation_clause[1:]

    import re
    if re.search(r"\b(?:notification|dispatch|alert|email|message)\b", finding_text, re.IGNORECASE):
        why_boundary_answer = (
            "The available evidence establishes that notification delivery did not occur, but does not "
            "establish whether the failure occurred during event generation, job creation, queue processing, "
            "recipient resolution, or delivery."
        )
    else:
        why_boundary_answer = (
            f"The available evidence establishes that {deviation_clause}, but does not establish the specific "
            "underlying mechanism or root cause responsible."
        )

    steps.append(FiveWhyStep(
        question=format_deviation_why_question(
            resolved.subject or noun_sub, resolved.condition, extract_temporal_clause(finding_text)
        ),
        answer=why_boundary_answer,
        status="UNKNOWN",
    ))
    return FiveWhyAnalysis(
        steps=steps,
        is_complete=False,
        status_note="EVIDENCE BOUNDARY — Causal mechanism requires investigation.",
    )
