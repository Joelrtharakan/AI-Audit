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
    canonical_state: Any = None,
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

    # A "mechanism" that is really just the OBSERVATION restated (fact_claims[0]
    # is, by this codebase's convention, the deviation itself) produces a
    # circular Why#1 ("Why was X incomplete?" -> "X was incomplete") --
    # extract_immediate_mechanism has no notion of "which fact is the
    # observation," so it can pick fact_claims[0] merely because its verb
    # shape happens to match a recognized polarity (e.g. "was not
    # performed" matches non_performance just as readily as a genuinely
    # deeper "was never assigned" fact would). When a second, textually
    # DIFFERENT verified fact exists, prefer whichever mechanism comes from
    # excluding the observation fact -- never let the observation explain
    # itself.
    if fact_claims and mechanism.statement and repeats_previous_why_answer(fact_claims[0], mechanism.statement):
        if len(fact_claims) > 1:
            _alt_mechanism = extract_immediate_mechanism(reported_claims, fact_claims[1:])
            if _alt_mechanism.statement:
                mechanism = _alt_mechanism

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

    # 0a1. Recurrence + prior corrective action (Section 8/9): the 5-Why
    # must never turn "a prior corrective action exists" into "the prior
    # action was ineffective" -- that requires objective evidence, not the
    # mere co-occurrence of recurrence and a prior action. Only asks the
    # "recur after prior action" question when BOTH recurrence AND a prior
    # action are structurally established; otherwise falls through to the
    # generic branches below. Generalizes across any domain (manufacturing,
    # finance, document control, ...) via detect_recurrence's own
    # structural detection, not a keyword list for this specific finding.
    from app.agent.recurrence_guard import detect_recurrence as _detect_recurrence_for_5why
    _recurrence_info = _detect_recurrence_for_5why(finding_text)
    if _recurrence_info.is_recurring and _recurrence_info.has_previous_capa_reference:
        _rec_subject = resolved.subject or noun_sub
        why_q = f"Why did the {_rec_subject} recur after the prior corrective action?"
        why_answer = (
            f"The evidence confirms recurrence of the {_rec_subject} and documents a prior corrective "
            "action, but does not establish whether the prior action failed to remain effective, was "
            "incompletely implemented, fell outside its scope, or was unrelated to the current mechanism."
        )
        steps.append(FiveWhyStep(
            question=why_q,
            answer=why_answer,
            status="UNKNOWN",
        ))
        return FiveWhyAnalysis(
            steps=steps,
            is_complete=False,
            status_note="EVIDENCE BOUNDARY — Recurrence and prior corrective action are verified; the relationship between them requires investigation.",
        )

    # 0a-1. Requirement-uncertain finding (Section 12): 5-Why is gated by
    # CAUSAL READINESS -- if the applicable requirement (and therefore
    # whether a deviation even exists) has not been established, the
    # engine must not invent a causal question that presupposes a
    # confirmed deviation. Defer instead of asking "why" at all.
    # Generalizes across any requirement type (specification, procedure,
    # standard, instruction, limit, ...), not one specific domain.
    if (
        resolved.semantic_type == "REQUIREMENT_UNCERTAIN"
        or (resolved.requirement_status == "UNKNOWN" and getattr(resolved, "semantic_type", None) in ("REQUIREMENT_UNCERTAIN", "OBSERVATION_VERIFICATION"))
    ):
        steps.append(FiveWhyStep(
            question="Why-chain deferred pending requirement resolution",
            answer=(
                "Why-chain deferred because the applicable requirement/deviation status has not yet been established."
            ),
            status="UNKNOWN",
        ))
        return FiveWhyAnalysis(
            steps=steps,
            is_complete=False,
            status_note="EVIDENCE BOUNDARY — Applicable requirement is unresolved; causal analysis deferred until compliance status is established.",
        )



    # 0a0. Event-sequence / control-point finding (Section 13): a
    # controlled TRANSITION whose required justification/authorization is
    # missing must ask about the TRANSITION itself, never the generic
    # subject/condition template, and must never speculate about WHY the
    # justification is missing (bypass/misunderstanding/omission are all
    # equally unverified). Generalizes across any transition type
    # (invalidation, override, exception, waiver, ...) via the canonical
    # transition_type field, not domain-specific wording.
    if resolved.semantic_type == "EVENT_SEQUENCE_CONTROL" and resolved.transition_type:
        _transition_label = resolved.transition_type.replace("_", " ").lower()
        if noun_sub and noun_sub.lower() != _transition_label:
            why_q = f"Why was the {noun_sub} {_transition_label} performed without the required justification/evidence?"
            why_answer = (
                f"The available evidence confirms that the {noun_sub} {_transition_label} occurred and that the required "
                "justification is not documented, but does not establish whether the control was bypassed, "
                f"misunderstood, omitted, or otherwise not executed."
            )
        else:
            why_q = f"Why did the {_transition_label} occur without the required justification/evidence?"
            why_answer = (
                f"The available evidence confirms that the {_transition_label} occurred and that the required "
                "justification is not documented, but does not establish whether the control was bypassed, "
                f"misunderstood, omitted, or otherwise not executed."
            )
        downstream_clause = (
            " A downstream action is also reported in the finding; this does not establish that the "
            "downstream action depended on or was improperly affected by the transition."
            if resolved.downstream_action_present else ""
        )
        why_answer = f"{why_answer}{downstream_clause}"
        steps.append(FiveWhyStep(
            question=why_q,
            answer=why_answer,
            status="UNKNOWN",
        ))
        return FiveWhyAnalysis(
            steps=steps,
            is_complete=False,
            status_note="EVIDENCE BOUNDARY — Transition and missing justification are verified; the control mechanism requires investigation.",
        )

    # 0a2. Missing-record / missing-documentation finding (Section 1/2/9):
    # the 5-Why must consume the canonical MISSING_RECORD event (activity/
    # context/downstream) rather than routing through the generic
    # mechanism-status branches below, which would otherwise restate the
    # observation as a "VERIFIED" Why#1 step and then pad a second,
    # near-duplicate step to reach a target chain length. A missing record
    # answers exactly ONE evidence-bound question: is the activity's
    # PERFORMANCE established -- and it is not, by the missing record
    # alone. Generalizes across any domain (inspection, review, approval,
    # verification, calibration check, ...), not specific to any one
    # activity type.
    if resolved.semantic_type == "MISSING_RECORD" and resolved.missing_record_activity:
        activity = resolved.missing_record_activity
        _condition_word = resolved.condition or "not documented"
        why_q = f"Why was the {activity} {_condition_word}?"
        downstream_clause = (
            " A subsequent action is also reported in the finding, but the missing record does not by "
            "itself establish whether that action was appropriately supported."
            if resolved.downstream_action_present else ""
        )
        why_answer = (
            f"The evidence confirms that the required record for the {activity} is missing, but does not "
            "establish whether the underlying activity was performed or why the record is absent."
            f"{downstream_clause}"
        )
        steps.append(FiveWhyStep(
            question=why_q,
            answer=why_answer,
            status="UNKNOWN",
        ))
        return FiveWhyAnalysis(
            steps=steps,
            is_complete=False,
            status_note="EVIDENCE BOUNDARY — Missing record is verified; underlying activity performance and cause require investigation.",
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

    # 0c. Comparison/mismatch finding (Section 1/2/9): the 5-Why for a
    # verified comparison ("recorded X did not match calculated Y") must be
    # built from the canonical comparison event (comparison_type/left/
    # right/measurement), not from a generic subject/condition template --
    # generalizes across any comparison domain (yield, temperature,
    # invoice amount, ...), not specific to any one finding.
    if resolved.semantic_type == "COMPARISON" and resolved.comparison_type and resolved.comparison_left and resolved.comparison_right:
        from app.services.semantic_subject import format_comparison_why_question
        _qualified_left = (
            f"{resolved.comparison_left_qualifier} {resolved.comparison_left}"
            if resolved.comparison_left_qualifier else resolved.comparison_left
        )
        why_q = format_comparison_why_question(
            resolved.comparison_type, resolved.comparison_left, resolved.comparison_right,
            left_qualifier=resolved.comparison_left_qualifier,
        ) or format_deviation_why_question(
            resolved.subject or noun_sub, resolved.condition, extract_temporal_clause(finding_text)
        )
        if resolved.measurement_value is not None:
            qual = f"{resolved.measurement_qualifier} " if resolved.measurement_qualifier else ""
            magnitude_phrase = f"{qual}{resolved.measurement_value}{resolved.measurement_unit or ''} discrepancy"
        else:
            magnitude_phrase = "discrepancy"
        from app.services.semantic_subject import (
            COMPARISON_SUBTYPE_MECHANISM_CATEGORIES,
            _DEFAULT_COMPARISON_MECHANISM_CATEGORIES,
        )
        _mechanism_categories = COMPARISON_SUBTYPE_MECHANISM_CATEGORIES.get(
            resolved.comparison_subtype or "", _DEFAULT_COMPARISON_MECHANISM_CATEGORIES
        )
        why_answer = (
            f"The available evidence confirms {'an' if magnitude_phrase[:1].lower() in 'aeiou' else 'a'} "
            f"{magnitude_phrase} between the {_qualified_left} and the {resolved.comparison_right}, "
            f"but does not establish whether the difference resulted from {_mechanism_categories}, "
            "or another mechanism."
        )
        steps.append(FiveWhyStep(
            question=why_q,
            answer=why_answer,
            status="UNKNOWN",
        ))
        return FiveWhyAnalysis(
            steps=steps,
            is_complete=False,
            status_note="EVIDENCE BOUNDARY — Comparison discrepancy is verified; underlying mechanism requires investigation.",
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
        # deviation_clause is always already a full clause with its own verb
        # (it comes from a verified fact/sentence, e.g. "the required
        # inspection was not completed") -- appending another verb phrase
        # like "was identified during inspection" produces a double-verb
        # grammar defect ("was not completed was identified during
        # inspection"). State it directly instead.
        why1_ans = f"Audit observation confirms that {deviation_clause}."
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
        why3_q = f"Which of these mechanisms caused the deviation in {target_obj_name}?"
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
        from app.services.semantic_subject import _strip_framing
        deviation_fact = fact_claims[0] if fact_claims else deviation_desc
        deviation_clause = _strip_framing(deviation_fact).strip().rstrip(".")
        if deviation_clause and deviation_clause[0].isupper() and not deviation_clause.split()[0].isupper():
            deviation_clause = deviation_clause[0].lower() + deviation_clause[1:]
        why1_question = format_deviation_why_question(
            resolved.subject or noun_sub, resolved.condition, extract_temporal_clause(finding_text)
        )
        # See the identical fix/comment in the reported-explanations branch
        # above: deviation_clause already contains its own verb, so
        # appending "occurred during inspection" produces a double-verb
        # grammar defect.
        why1_answer = f"Audit observation confirms that {deviation_clause}."
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
