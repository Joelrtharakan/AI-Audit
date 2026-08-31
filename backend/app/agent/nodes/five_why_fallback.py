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
    semantic_context: Any = None,
) -> FiveWhyAnalysis:
    """Build a deterministic, evidence-bound 5-Why chain from the case model.

    Enforces deterministic 5-Why stopping policy:
      - Never fabricates causal layers beyond available evidence.
      - Handles conflicting reports with explicit conflict recognition.
      - Stops at the evidence boundary with UNKNOWN status.

    `semantic_context` (a validated `CanonicalFindingContext`) is
    authoritative over `canonical_subject`/`canonical_state`/raw-text
    re-derivation when present -- its `primary_deviation` anchors Why-1,
    and its resolved affected-object entity (never a bare STATE word)
    anchors the subject. When `semantic_context` is None, behavior is
    fully unchanged from before this parameter existed.
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

    # PRECEDENCE + CONVERGENCE (Part 2/7/12): if the finding TEXT explicitly
    # enumerates competing causal mechanisms, that enumeration IS the 5-Why
    # boundary. Source the subject/observation from the CANONICAL state
    # (authoritative, post-merge) -- resolve_deviation() is deferred so a
    # competing-causes finding never triggers a redundant deterministic
    # re-parse.
    # Spec Pass 47 §5/§6: on the canonical-success path the canonical LLM's
    # own `stated_causal_alternatives` is authoritative -- the deterministic
    # raw-text extractor (`_esca0`) is a FALLBACK disjunct that runs ONLY when
    # there is no canonical context.
    from app.agent.causal_guard import extract_stated_causal_alternatives as _esca0
    _early_alts = list(getattr(canonical_state, "stated_causal_alternatives", []) or [])
    if not _early_alts and semantic_context is None:
        _early_alts = _esca0(finding_text)
    if len(_early_alts) >= 2:
        from app.services.semantic_subject import (
            _strip_framing as _sf0,
            format_deviation_why_question as _fdwq0,
        )
        _cf_subj = getattr(canonical_state, "finding_subject", None)
        _cf_cond = getattr(canonical_state, "deviation_condition", None)
        _cf_obs = getattr(canonical_state, "observed_deviation", None)
        if not (_cf_subj and _cf_cond) and semantic_context is None:
            # FALLBACK-ONLY raw-text recovery of subject/condition.
            resolved = resolve_deviation(finding_text, fact_claims)
            _cf_subj = _cf_subj or getattr(resolved, "finding_subject", None)
            _cf_cond = _cf_cond or resolved.condition
            _cf_obs = _cf_obs or resolved.deviation
        _obs = (fact_claims[0] if fact_claims else (_cf_obs or finding_text))
        _obs = _sf0(_obs).strip().rstrip(".")
        if _obs and _obs[0].isupper() and not _obs.split()[0].isupper():
            _obs = _obs[0].lower() + _obs[1:]
        _aj = "; ".join(a.rstrip(". ").strip() for a in _early_alts)
        _subj0 = _cf_subj if _cf_subj else "the affected process"
        return FiveWhyAnalysis(
            steps=[FiveWhyStep(
                question=_fdwq0(_subj0, _cf_cond, None),
                answer=(
                    f"The available evidence establishes that {_obs}, but does not establish "
                    f"which mechanism is responsible. The finding states the plausible "
                    f"mechanisms remaining are: {_aj}. Investigation is required to "
                    f"discriminate between them."
                ),
                status="UNKNOWN",
            )],
            is_complete=False,
            status_note=(
                "EVIDENCE BOUNDARY — Competing causal mechanisms stated by the finding remain "
                "unresolved; investigation must discriminate between them."
            ),
        )

    # CANONICAL CAUSAL STATE IS AUTHORITATIVE (spec §13). When a valid
    # canonical interpretation exists and it did NOT establish a root cause,
    # 5-Why is a STRUCTURED PRESENTATION of that canonical reasoning -- it
    # stops at the evidence boundary and never runs a second deterministic
    # causal inference (resolve_deviation + multi-level chaining). The
    # deterministic chain below is the fail-closed floor for when there is no
    # canonical context at all.
    if semantic_context is not None:
        _rcs = getattr(semantic_context, "root_cause_status", None)
        if _rcs in (None, "NOT_ESTABLISHED", "STATED_UNVERIFIED", "CONTRADICTED"):
            from app.services.canonical_context_validator import get_affected_object_candidate
            _pd = getattr(semantic_context, "primary_deviation", None)
            _cond = getattr(semantic_context, "observed_condition", None)
            _subj = _pd or get_affected_object_candidate(semantic_context) or "the affected process"
            _obs = (fact_claims[0] if fact_claims else (_pd or _cond or "the reported condition"))
            _hyps = [h.statement.strip().rstrip(".") for h in
                     (getattr(semantic_context, "candidate_hypotheses", []) or []) if getattr(h, "statement", None)]
            _gaps = [str(g).strip().rstrip(".") for g in
                     (getattr(semantic_context, "information_gaps", []) or []) if str(g).strip()]
            _ans = (
                f"The available evidence establishes that {str(_obs).strip().rstrip('.')}, but does "
                "not establish why it occurred. Root cause is NOT_ESTABLISHED."
            )
            if _hyps:
                _ans += " Possible explanations that remain unverified: " + "; ".join(_hyps) + "."
            if _gaps:
                _ans += " The following must first be established: " + "; ".join(_gaps) + "."
            _ans += " Investigation is required before a causal conclusion can be drawn."
            _q1 = f"Why did the observed condition affecting {_subj} occur?"
            return FiveWhyAnalysis(
                steps=[FiveWhyStep(question=_q1, answer=_ans, status="UNKNOWN")],
                is_complete=False,
                status_note=(
                    "EVIDENCE BOUNDARY — the canonical interpretation did not establish a root "
                    "cause; investigation is required."
                ),
            )

    # The deterministic 5-Why below needs the resolver's full DeviationInfo
    # (condition, comparison_*, recurrence_*, transition_type, actors, ...) --
    # more than canonical_finding_state exposes -- so it is the fail-closed
    # floor for every non-competing-causes shape (and for a canonical context
    # that DID establish the cause -- it presents the established chain).
    resolved = resolve_deviation(finding_text, fact_claims)

    # Generic degraded-subject fallback -- deliberately domain-agnostic
    # ("the affected process") rather than guessing a specific entity from
    # finding vocabulary. A keyword-triggered fabricated entity (e.g.
    # inferring "email notification" merely because the finding mentions
    # "email"/"dispatch") is exactly the kind of finding-specific guess
    # that must never stand in for genuine extraction: an honest generic
    # placeholder is always safer than a plausible-looking invention.
    _GENERIC_SUBJECT_FALLBACK = "the affected process"
    _semantic_primary_deviation: str | None = None
    if semantic_context is not None:
        # States A/B: the validated LLM canonical context is authoritative
        # when present. An explicit NOT_ESTABLISHED affected-object (state
        # B) still uses the generic placeholder, never a raw-text guess.
        # Five-Why's subject is the DEVIATION under investigation (which
        # may be an EVENT, e.g. "packaging failure" -- not restricted to
        # ENTITY-kind the way the investigation planner's "what controls
        # this entity" questions are), so primary_deviation takes priority
        # over the entity-only affected-object candidate here.
        from app.services.canonical_context_validator import get_affected_object_candidate
        _semantic_primary_deviation = getattr(semantic_context, "primary_deviation", None)
        _canonical_affected = get_affected_object_candidate(semantic_context)
        noun_sub = _semantic_primary_deviation or _canonical_affected or _GENERIC_SUBJECT_FALLBACK
    elif canonical_subject is not None:
        from app.services.semantic_subject import is_established_subject
        if isinstance(canonical_subject, str):
            subj_val = canonical_subject
        else:
            subj_val = getattr(canonical_subject, "finding_subject", getattr(canonical_subject, "subject", None))
        if is_established_subject(subj_val):
            noun_sub = subj_val
        elif is_established_subject(resolved.finding_subject or resolved.subject):
            noun_sub = resolved.finding_subject or resolved.subject
        else:
            noun_sub = _GENERIC_SUBJECT_FALLBACK
    else:
        from app.services.semantic_subject import is_established_subject
        noun_sub = (
            (resolved.finding_subject or resolved.subject)
            if is_established_subject(resolved.finding_subject or resolved.subject)
            else _GENERIC_SUBJECT_FALLBACK
        )
    # The canonical primary_deviation (Why-1's actual subject, per Section
    # 6 of the promotion pass) takes priority over the raw-text resolver's
    # `deviation` whenever a validated semantic context supplied one --
    # this is what stops Why-1 from being anchored on whichever evidence
    # sentence resolve_deviation happened to match (e.g. a recovery or
    # historical statement) instead of the actual deviation under
    # investigation.
    deviation_desc = _semantic_primary_deviation or resolved.deviation or f"{noun_sub} condition noted in finding"
    # When a validated semantic context is present, its resolved subject
    # (noun_sub) must win over the raw-text resolver's own `resolved.
    # subject` in every question-formatting call below -- otherwise a
    # regex fabrication (e.g. "active") would silently override the
    # canonical entity merely because `resolved.subject` happens to be
    # truthy. Legacy behavior (`effective_subject`) is fully
    # preserved when no semantic context was supplied.
    effective_subject = noun_sub if semantic_context is not None else (resolved.subject or noun_sub)

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

    # CAUSAL SAFETY VETO (promotion pass, Section 7/11): extract_immediate_
    # mechanism has no notion of financial/recovery/remediation/historical
    # CONSEQUENCE vs CAUSE -- it will happily select a REPORTED recovery or
    # historical-recurrence sentence as "the mechanism" merely because its
    # verb shape looks like an attributed explanation. When a validated
    # semantic context is present, veto any mechanism whose text matches a
    # claim the canonical context explicitly marked non-causal (a
    # FINANCIAL_METRIC/RECOVERY/REMEDIATION/HISTORICAL_CONTEXT/CONSEQUENCE
    # fact, or any causal_claim with is_causal=False) -- a vetoed mechanism
    # is treated as absent, so the chain falls through to the evidence-
    # boundary branches below instead of presenting a financial consequence
    # as if it explained the deviation.
    if semantic_context is not None and mechanism.statement:
        _non_causal_kinds = {"FINANCIAL_METRIC", "RECOVERY", "REMEDIATION", "PREVENTION", "HISTORICAL_CONTEXT", "CONSEQUENCE"}
        _non_causal_texts: set[str] = set()
        _evidence_by_id = {f"E{i}": e.claim for i, e in enumerate(evidence_ledger)}
        for cc in getattr(semantic_context, "causal_claims", []) or []:
            if not getattr(cc, "is_causal", False):
                for eid in getattr(cc, "source_evidence_ids", []) or []:
                    if eid in _evidence_by_id:
                        _non_causal_texts.add(_evidence_by_id[eid].strip().lower())
        for ent in getattr(semantic_context, "entities", []) or []:
            if getattr(ent, "kind", None) in _non_causal_kinds:
                for eid in getattr(ent, "source_evidence_ids", []) or []:
                    if eid in _evidence_by_id:
                        _non_causal_texts.add(_evidence_by_id[eid].strip().lower())
        _mech_norm = mechanism.statement.strip().lower()
        if any(_mech_norm == t or _mech_norm in t or t in _mech_norm for t in _non_causal_texts):
            from app.agent.causal_guard import MechanismInfo
            mechanism = MechanismInfo()

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
    # As with the investigation planner: when a validated semantic context
    # exists, its explicit_previous_capa_reference (already cross-checked
    # against this same detect_recurrence signal during canonical
    # validation) is authoritative.
    _has_previous_capa = (
        semantic_context.explicit_previous_capa_reference
        if semantic_context is not None
        else _recurrence_info.has_previous_capa_reference
    )
    if _recurrence_info.is_recurring and _has_previous_capa:
        _rec_subject = effective_subject
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

    # 0b. Duplicate Payment / Overpayment Finding (Section 6 Hardening)
    #
    # "Overpayment" and "duplicate payment" are NOT the same financial
    # claim -- a duplicate payment specifically means the same obligation
    # was paid twice, while an overpayment could equally arise from a
    # pricing error, a quantity/tax miscalculation, or a currency
    # conversion mistake, with no second transaction involved at all.
    # Asserting "two payment transactions... identified as duplicate" as
    # VERIFIED whenever the evidence only says "overpayment" fabricates a
    # specific causal mechanism (double payment) the evidence never
    # actually established -- exactly the "financial claim silently
    # becomes a causal hypothesis" failure mode this fallback must avoid.
    _dup_match = re.search(r"\b(?:duplicate\s+payment|paid\s+twice|double\s+payment)\b", finding_text, re.IGNORECASE)
    _overpay_match = re.search(r"\boverpayment\b", finding_text, re.IGNORECASE)
    if _dup_match or _overpay_match:
        if _dup_match:
            term, article = "duplicate payment", "a"
            first_answer = "Two payment transactions associated with the same supplier obligation were identified as duplicate."
            first_status = "VERIFIED"
        else:
            term, article = "overpayment", "an"
            first_answer = "The evidence identifies an overpayment, but does not by itself establish whether it resulted from a duplicate transaction, a pricing or quantity error, or another mechanism."
            first_status = "UNKNOWN"
        steps.append(FiveWhyStep(
            question=f"Why was {article} {term} identified?",
            answer=first_answer,
            status=first_status,
        ))
        steps.append(FiveWhyStep(
            question=f"Why did the {term} occur?",
            answer=f"The available evidence confirms that {article} {term} occurred, but does not establish the mechanism that produced it.",
            status="UNKNOWN",
        ))
        steps.append(FiveWhyStep(
            question=f"Which control condition allowed the {term}?",
            answer=f"The underlying control condition that allowed the {term} is not established from available evidence — objective records from the payment workflow, verification, approval, and reconciliation process are required.",
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
            effective_subject, resolved.condition, extract_temporal_clause(finding_text)
        )
        if resolved.measurement_value is not None:
            qual = f"{resolved.measurement_qualifier} " if resolved.measurement_qualifier else ""
            _u = resolved.measurement_unit or ""
            _unit_txt = _u if _u == "%" else (f" {_u}" if _u else "")
            magnitude_phrase = f"{qual}{resolved.measurement_value:g}{_unit_txt} discrepancy"
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
            effective_subject, resolved.condition, extract_temporal_clause(finding_text)
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
            effective_subject, resolved.condition, extract_temporal_clause(finding_text)
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
                effective_subject, resolved.condition, extract_temporal_clause(finding_text)
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

    # Preserve any causal differential the finding states explicitly -- the
    # boundary answer must NOT collapse "A, B, or C remain unresolved" into a
    # bare "root cause unknown" (spec 8 / 16).
    from app.agent.causal_guard import extract_stated_causal_alternatives
    _stated_alts = (
        list(getattr(canonical_state, "stated_causal_alternatives", []) or [])
        or extract_stated_causal_alternatives(finding_text)
    )
    if len(_stated_alts) >= 2:
        _alts_join = "; ".join(a.rstrip(". ").strip() for a in _stated_alts)
        why_boundary_answer = (
            f"The available evidence establishes that {deviation_clause}, but does not "
            f"establish which mechanism is responsible. The finding states the plausible "
            f"mechanisms remaining are: {_alts_join}. Investigation is required to "
            f"discriminate between them."
        )
        _note = (
            "EVIDENCE BOUNDARY — Competing causal mechanisms stated by the finding remain "
            "unresolved; investigation must discriminate between them."
        )
    else:
        why_boundary_answer = (
            f"The available evidence establishes that {deviation_clause}, but does not establish the specific "
            "underlying mechanism or root cause responsible."
        )
        _note = "EVIDENCE BOUNDARY — Causal mechanism requires investigation."

    steps.append(FiveWhyStep(
        question=format_deviation_why_question(
            effective_subject, resolved.condition, extract_temporal_clause(finding_text)
        ),
        answer=why_boundary_answer,
        status="UNKNOWN",
    ))
    return FiveWhyAnalysis(
        steps=steps,
        is_complete=False,
        status_note=_note,
    )
