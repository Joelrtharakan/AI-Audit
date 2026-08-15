"""Node 5: root_cause_analysis

Performs RCA and 5-Why analysis over the evidence ledger. This is a separate
LLM call that reasons ONLY over classified evidence, not raw finding text,
matching the existing pipeline's separation of extraction from classification.

Enforces:
  - ESTABLISHED status requires at least one VERIFIED evidence item
  - 5-Why SUPPORTED answers require corresponding ledger entries
  - Contributing factors are conditions, not action items
"""

from __future__ import annotations

import json
import logging

from app.agent.causal_guard import (
    MechanismInfo,
    extract_immediate_mechanism,
    hypothesis_contradicts_mechanism,
    is_circular_why_answer,
    is_generic_non_analysis_filler,
    mechanism_already_names_generic_hypothesis,
    question_reopens_mechanism,
    repeats_previous_why_answer,
    restates_observation,
)
from app.agent.grounding_guard import (
    SAFE_RECURRENCE_FALLBACK,
    SAFE_ROOT_CAUSE_FALLBACK,
    build_source_text,
    claims_unsupported_effectiveness,
    clean_structured_leak,
    detect_evidence_contradictions,
    has_causal_language,
    is_placeholder_leak,
    mentions_unsupported_domain,
    ungrounded_entities,
)
from app.agent.state import AgentState
from app.config import get_settings
from app.models.agent import (
    AgentTraceStep,
    ContributingFactor,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    RootCauseAnalysis,
    RootCauseStatus,
)
from app.services.llm_client import LLMError, get_llm_client
from app.services.llm_json import parse_llm_json
from app.services.taxonomy import coerce_category

logger = logging.getLogger(__name__)

_ACTION_PREFIXES = ("review", "verify", "confirm", "check", "pull", "collect", "conduct", "perform")


def _looks_like_action_item(text: str) -> bool:
    return text.strip().lower().split()[0] in _ACTION_PREFIXES if text.strip() else False


def _has_verified_evidence(evidence_ledger: list) -> bool:
    return any(e.status == EvidenceStatus.VERIFIED for e in evidence_ledger)


def _filter_hypotheses(
    raw_hypotheses: list, source_text: str, trace: list, mechanism: MechanismInfo | None = None
) -> list:
    """Apply the same entity/domain/placeholder grounding filters used on the
    main RCA response's hypotheses, factored out so the focused retry below
    can reuse it identically."""
    from app.models.agent import CandidateHypothesis

    mechanism = mechanism or MechanismInfo()
    valid_hyp_statuses = {"POSSIBLE", "SUPPORTED", "REFUTED", "UNVERIFIED"}
    parsed = []
    for ch in raw_hypotheses:
        if isinstance(ch, dict):
            status_raw = str(ch.get("status", "POSSIBLE")).upper()
            parsed.append(CandidateHypothesis(
                id=str(ch.get("id", "H")),
                name=str(ch.get("name", "HYPOTHESIS")),
                statement=clean_structured_leak(ch.get("statement", "")),
                status=status_raw if status_raw in valid_hyp_statuses else "POSSIBLE",
                evidence_needed=clean_structured_leak(ch.get("evidence_needed", "")) or "Investigation required",
                discrimination_evidence=clean_structured_leak(ch.get("discrimination_evidence", "")) or None,
                resolves_investigation=ch.get("resolves_investigation") or None,
                rationale=clean_structured_leak(ch.get("rationale")) or None,
                evidence_against=clean_structured_leak(ch.get("evidence_against")) or None,
                confirms_if=clean_structured_leak(ch.get("confirms_if")) or None,
                refutes_if=clean_structured_leak(ch.get("refutes_if")) or None,
            ))

    kept = []
    for ch in parsed:
        violations = ungrounded_entities(ch.statement, source_text)
        if violations:
            trace.append(AgentTraceStep.warn(
                f"Hypothesis {ch.id} dropped — references ungrounded entity/number: {violations}"
            ))
            continue
        if mentions_unsupported_domain(ch.statement, source_text) or mentions_unsupported_domain(ch.name, source_text):
            trace.append(AgentTraceStep.warn(
                f"Hypothesis {ch.id} dropped — invokes unsupported domain not "
                "supported by this finding"
            ))
            continue

        if is_placeholder_leak(ch.statement):
            trace.append(AgentTraceStep.warn(
                f"Hypothesis {ch.id} dropped — echoed prompt instruction text instead of real analysis"
            ))
            continue

        if hypothesis_contradicts_mechanism(ch.statement, mechanism):
            trace.append(AgentTraceStep.warn(
                f"Hypothesis {ch.id} dropped — contradicts the established mechanism "
                f"({mechanism.statement!r})"
            ))
            continue

        if mechanism_already_names_generic_hypothesis(ch.statement, mechanism):
            trace.append(AgentTraceStep.warn(
                f"Hypothesis {ch.id} dropped — restates the already-established mechanism as an "
                "open possibility instead of reasoning about its cause"
            ))
            continue
        if ch.status == "REFUTED":
            trace.append(AgentTraceStep.warn(
                f"Hypothesis {ch.id} dropped — directly contradicted by evidence in the ledger"
            ))
            continue
        kept.append(ch)
    return kept


async def _retry_hypotheses(
    client, finding_text: str, evidence_ledger: list, source_text: str, trace: list,
    mechanism: MechanismInfo | None = None,
) -> list:
    """When every hypothesis from the main RCA call gets filtered out, that's
    usually not "this finding has no plausible cause" -- it's a weak model
    defaulting to a generic mechanism instead of reasoning about this
    finding's actual subject matter. A smaller, more focused retry call
    (just hypothesis generation, nothing else) gives it a second, narrower
    attempt rather than silently shipping an empty hypothesis list. This is
    a retry with better-targeted instructions, not a hardcoded fallback set."""
    settings = get_settings()
    template = (settings.agent_prompts_dir / "rca_hypotheses_retry.txt").read_text(encoding="utf-8")
    prompt = template.format(
        finding_text=finding_text,
        evidence_ledger_json=json.dumps([e.model_dump() for e in evidence_ledger], default=str),
    )
    try:
        raw = await client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format_json=True,
            max_tokens=768,
        )
        parsed = parse_llm_json(raw)
        candidates = parsed.get("candidate_hypotheses", [])
    except (LLMError, ValueError, KeyError) as exc:
        logger.warning("Hypothesis retry call failed: %s", exc)
        return []

    filtered = _filter_hypotheses(candidates, source_text, trace, mechanism)
    if filtered:
        trace.append(AgentTraceStep.ok(f"Hypothesis retry produced {len(filtered)} grounded hypothesis(es)"))
    return filtered


async def root_cause_node(state: AgentState) -> AgentState:
    """Perform root cause analysis and 5-Why over the evidence ledger."""
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    evidence_ledger = state.get("evidence_ledger", [])
    extraction = state.get("extraction")
    settings = get_settings()
    client = get_llm_client()

    finding_text = state["request"].finding_text
    extra_trusted = []
    if extraction:
        extra_trusted.extend(extraction.stated_facts)
        extra_trusted.extend(extraction.referenced_records)
        extra_trusted.extend(f"{s.speaker} {s.claim}" for s in extraction.attributed_statements)
    source_text = build_source_text(finding_text, evidence_ledger, extra_trusted)

    # Immediate mechanism (Layer 2): prefer the canonical state's already-
    # computed mechanism when available, else derive it directly from the
    # evidence ledger — self-contained so this node works standalone too.
    canonical = state.get("canonical_finding_state")
    if canonical and canonical.immediate_mechanism:
        mechanism = MechanismInfo(
            statement=canonical.immediate_mechanism,
            status=canonical.immediate_mechanism_status,
        )
        from app.agent.causal_guard import classify_mechanism_polarity
        mechanism.polarity = classify_mechanism_polarity(canonical.immediate_mechanism)
    else:
        reported = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.REPORTED]
        verified = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.VERIFIED]
        mechanism = extract_immediate_mechanism(reported, verified)

    # CONTRADICTORY EVIDENCE: detect same-subject evidence items that assert
    # opposite polarity (e.g. "training was completed" vs "no completion
    # record found") before RCA runs, so a conflict can never be silently
    # resolved in favor of one side.
    contradictions = detect_evidence_contradictions(evidence_ledger)
    if contradictions:
        for claim_a, claim_b in contradictions:
            trace.append(AgentTraceStep.warn(
                f"Evidence conflict detected: {claim_a!r} vs {claim_b!r} — cannot be resolved automatically"
            ))

    system_prompt = (settings.agent_prompts_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (settings.agent_prompts_dir / "rca.txt").read_text(encoding="utf-8")

    tools_available = len(state.get("completed_tools", [])) > 0

    prompt = template.format(
        finding_text=state["request"].finding_text,
        extraction_json=extraction.model_dump_json(indent=2) if extraction else "{}",
        evidence_ledger_json=json.dumps(
            [e.model_dump() for e in evidence_ledger], default=str
        ),
        evidence_gaps_json=json.dumps(
            [g.model_dump() for g in state.get("evidence_gaps", [])], default=str
        ),
        tools_were_available="YES" if tools_available else "NO — LQMS not configured",
    )

    # Defaults if LLM fails — deliberately generic/entity-free so a failure never
    # injects another case's facts into this finding's output.
    root_cause = RootCauseAnalysis(
        status="NOT_ESTABLISHED",
        category=None,
        narrative="Root cause analysis could not be completed for this finding. Manual investigation is required.",
        evidence_status=EvidenceStatus.UNKNOWN,
        verification_needed="Full manual investigation required — AI root cause analysis was unavailable for this finding.",
    )
    five_why = FiveWhyAnalysis(
        steps=[],
        is_complete=False,
        status_note="INCOMPLETE — ROOT CAUSE NOT ESTABLISHED",
    )
    contributing_factors: list[ContributingFactor] = []

    try:
        raw = await client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format_json=True,
            max_tokens=2048,
        )
        parsed = parse_llm_json(raw)

        # --- Root cause ---
        rc_raw = parsed.get("root_cause", {})
        status_raw = str(rc_raw.get("status", "NOT_ESTABLISHED")).upper()
        valid_statuses = {"VERIFIED", "SUPPORTED", "STATED_UNVERIFIED", "INFERRED", "NOT_ESTABLISHED", "CONTRADICTED"}
        if status_raw not in valid_statuses:
            status_raw = "NOT_ESTABLISHED"

        # Enforce: VERIFIED/SUPPORTED both require at least one VERIFIED evidence
        # item in the ledger — per rca.txt, SUPPORTED means "strongly supported by
        # verified facts", so it cannot rest on REPORTED-only evidence either.
        if status_raw in ("VERIFIED", "SUPPORTED") and not _has_verified_evidence(evidence_ledger):
            logger.warning(
                "RCA claimed %s but no VERIFIED evidence in ledger — downgrading to STATED_UNVERIFIED",
                status_raw,
            )
            trace.append(AgentTraceStep.warn(
                f"Root cause downgraded: {status_raw} claimed without VERIFIED evidence item"
            ))
            status_raw = "STATED_UNVERIFIED"

        category = coerce_category(rc_raw.get("category")).value if rc_raw.get("category") else None
        ev_status_str = str(rc_raw.get("evidence_status", "UNKNOWN")).upper()
        if ev_status_str not in {s.value for s in EvidenceStatus}:
            ev_status_str = "UNKNOWN"

        confidence_str = str(rc_raw.get("confidence", "LOW")).upper()
        if confidence_str not in ("LOW", "MEDIUM", "HIGH"):
            confidence_str = "LOW"

        raw_narrative = str(rc_raw.get("narrative", "")).strip()
        raw_statement = str(rc_raw.get("statement", "") or "").strip()
        if not raw_narrative or "LLM error" in raw_narrative:
            raw_narrative = (
                "No leading hypothesis established from the available evidence for this finding. "
                "Status: NOT_ESTABLISHED — auditor investigation required."
            )

        # CAUSAL INFERENCE GUARD: definitive causal language ("X and Y caused Z")
        # without VERIFIED evidence must never stand as an established cause —
        # force NOT_ESTABLISHED and neutral narrative when definitive causation is claimed.
        verified_support = _has_verified_evidence(evidence_ledger)
        if status_raw not in ("VERIFIED", "SUPPORTED") and not verified_support and (
            has_causal_language(raw_narrative) or has_causal_language(raw_statement)
        ):
            logger.warning(
                "RCA narrative/statement used causal language without VERIFIED evidence — "
                "forcing NOT_ESTABLISHED."
            )
            trace.append(AgentTraceStep.warn(
                "Root cause forced to NOT_ESTABLISHED: causal claim made from unverified/reported evidence"
            ))
            status_raw = "NOT_ESTABLISHED"
            raw_narrative = SAFE_ROOT_CAUSE_FALLBACK
            raw_statement = ""

        # CONTRADICTORY EVIDENCE GUARD: if the ledger itself contains conflicting
        # claims about the same subject, no status stronger than STATED_UNVERIFIED
        # may stand, and the narrative must say so explicitly rather than silently
        # taking one side.
        if contradictions and status_raw in ("VERIFIED", "SUPPORTED", "STATED_UNVERIFIED"):
            claim_a, claim_b = contradictions[0]
            trace.append(AgentTraceStep.warn(
                "Root cause forced to NOT_ESTABLISHED: conflicting evidence in the ledger"
            ))
            status_raw = "NOT_ESTABLISHED"
            raw_narrative = (
                f"The available evidence is conflicting and cannot currently be resolved: "
                f"{claim_a!r} conflicts with {claim_b!r}. This requires reconciliation before "
                f"a root cause can be established."
            )
            raw_statement = ""

        # RECURRENCE / EFFECTIVENESS GUARD: "previous corrective action was
        # recorded as completed" is not evidence that it was implemented,
        # verified, or effective — those are separate, independently-evidenced
        # claims. Strip any effectiveness/success claim the finding itself
        # never actually makes.
        if claims_unsupported_effectiveness(raw_narrative, finding_text) or claims_unsupported_effectiveness(raw_statement, finding_text):
            trace.append(AgentTraceStep.warn(
                "Root cause claimed previous-CAPA effectiveness not supported by this finding — "
                "downgraded to completion-only"
            ))
            status_raw = "NOT_ESTABLISHED"
            raw_narrative = SAFE_RECURRENCE_FALLBACK
            raw_statement = ""

        cand_hyp_list = _filter_hypotheses(rc_raw.get("candidate_hypotheses", []), source_text, trace, mechanism)

        # RETRY, DON'T JUST GIVE UP: an empty hypothesis list after filtering
        # is usually not "this finding has no plausible cause" -- it's a weak
        # model defaulting to a generic mechanism (training/communication/etc)
        # instead of reasoning about this finding's actual subject matter.
        # One focused retry, scoped to hypothesis generation only, before
        # falling back to a genuinely empty list.
        if not cand_hyp_list and status_raw != "VERIFIED":
            cand_hyp_list = await _retry_hypotheses(client, finding_text, evidence_ledger, source_text, trace, mechanism)

        # No hardcoded fallback hypothesis set: if the retry also returned
        # none, leave the list empty rather than injecting a fixed (and
        # possibly wrong-domain) set — a fabricated hypothesis is worse than
        # no hypothesis.
        if not cand_hyp_list:
            trace.append(AgentTraceStep.warn(
                "No candidate hypotheses returned by the model for this finding"
            ))

        # HARD EVIDENCE GATE: Set leading_hypothesis to None if no hypothesis has stronger evidence support
        leading_hyp = str(rc_raw.get("leading_hypothesis", "") or "").strip()
        if status_raw == "NOT_ESTABLISHED" or not leading_hyp or "not established" in leading_hyp.lower() or "none" in leading_hyp.lower() or "null" in leading_hyp.lower():
            leading_hyp = None
        elif ungrounded_entities(leading_hyp, source_text):
            trace.append(AgentTraceStep.warn("Leading hypothesis dropped — references ungrounded entity/number"))
            leading_hyp = None

        # GROUNDING GUARD on the primary narrative/statement fields: these are the
        # highest-stakes fields (they drive the CA draft), so an ungrounded entity
        # here forces the whole root cause back to NOT_ESTABLISHED rather than
        # just being stripped.
        narrative_violations = ungrounded_entities(raw_narrative, source_text)
        statement_violations = ungrounded_entities(raw_statement, source_text)
        if narrative_violations or statement_violations:
            trace.append(AgentTraceStep.warn(
                f"Root cause narrative/statement referenced ungrounded entity/number "
                f"{narrative_violations or statement_violations} — forcing NOT_ESTABLISHED"
            ))
            logger.warning(
                "RCA narrative/statement grounding violation: %s",
                narrative_violations or statement_violations,
            )
            status_raw = "NOT_ESTABLISHED"
            raw_narrative = SAFE_ROOT_CAUSE_FALLBACK
            raw_statement = ""
            leading_hyp = None
        elif mentions_unsupported_domain(raw_narrative, source_text) or mentions_unsupported_domain(raw_statement, source_text):
            trace.append(AgentTraceStep.warn(
                "Root cause narrative/statement invoked training/authorization domain not "
                "supported by this finding — forcing NOT_ESTABLISHED"
            ))
            status_raw = "NOT_ESTABLISHED"
            raw_narrative = SAFE_ROOT_CAUSE_FALLBACK
            raw_statement = ""
            leading_hyp = None
        elif is_placeholder_leak(raw_narrative) or is_placeholder_leak(raw_statement):
            trace.append(AgentTraceStep.warn(
                "Root cause narrative/statement echoed prompt instruction text instead of real "
                "analysis — forcing NOT_ESTABLISHED"
            ))
            status_raw = "NOT_ESTABLISHED"
            raw_narrative = SAFE_ROOT_CAUSE_FALLBACK
            raw_statement = ""
            leading_hyp = None

        root_cause = RootCauseAnalysis(
            status=status_raw,  # type: ignore[arg-type]
            # A causal category is only meaningful once a cause is at least
            # claimed — if the LLM didn't supply one (or status forced it to
            # NOT_ESTABLISHED), say so explicitly rather than leaving a blank
            # that could be misread as an omission.
            category=(category if (category and status_raw != "NOT_ESTABLISHED") else "TO_BE_CONFIRMED"),
            statement=raw_statement or None,
            leading_hypothesis=leading_hyp,
            candidate_hypotheses=cand_hyp_list,
            supporting_evidence=[str(x) for x in (rc_raw.get("supporting_evidence") or [])],
            contradicting_evidence=[str(x) for x in (rc_raw.get("contradicting_evidence") or [])],
            missing_evidence=[str(x) for x in (rc_raw.get("missing_evidence") or [])],
            confidence=confidence_str,  # type: ignore[arg-type]
            narrative=raw_narrative,
            evidence_status=EvidenceStatus(ev_status_str),
            verification_needed=rc_raw.get("verification_needed") or None,
        )

        # Filter out exact restatements of finding text from root cause statement
        finding_text_clean = state["request"].finding_text.strip().lower()
        if root_cause.statement and root_cause.statement.strip().lower() == finding_text_clean:
            root_cause.statement = None
            if root_cause.status == RootCauseStatus.VERIFIED:
                root_cause.status = RootCauseStatus.STATED_UNVERIFIED

        # --- Contributing factors (conditions, not action items) ---
        contributing_factors = []
        for cf in parsed.get("contributing_factors", []):
            if not isinstance(cf, dict):
                continue
            desc = clean_structured_leak(cf.get("description", ""))
            if _looks_like_action_item(desc):
                continue  # filter action items
            if is_generic_non_analysis_filler(desc):
                continue  # boilerplate "not established" non-answer, not analysis
            cf_status = str(cf.get("evidence_status", "INFERRED")).upper()
            if cf_status not in {s.value for s in EvidenceStatus}:
                cf_status = "INFERRED"
            cf_factor_status = "VERIFIED" if cf_status == "VERIFIED" else "POSSIBLE_UNCONFIRMED"
            contributing_factors.append(ContributingFactor(
                description=desc,
                evidence_status=EvidenceStatus(cf_status),
                status=cf_factor_status,
            ))

        # --- 5-Why ---
        fw_raw = parsed.get("five_why", {})
        steps = []
        for step in fw_raw.get("steps", []):
            if not isinstance(step, dict):
                continue
            s_status = str(step.get("status", "UNKNOWN")).upper()
            valid_why_statuses = ("VERIFIED", "SUPPORTED", "REPORTED_UNVERIFIED", "INFERRED", "UNKNOWN", "REQUIRES_EVIDENCE", "NOT_ESTABLISHED")
            if s_status not in valid_why_statuses:
                s_status = "UNKNOWN"
            question = str(step.get("question", ""))
            answer = clean_structured_leak(step.get("answer")) or None

            # MECHANISM-REOPENING GUARD: a question that re-litigates whether
            # the already-established mechanism occurred (opposite polarity,
            # e.g. asking about documentation once non-performance is
            # confirmed) reopens a resolved distinction — truncate here.
            if question_reopens_mechanism(question, mechanism):
                trace.append(AgentTraceStep.warn(
                    f"5-Why step {len(steps) + 1} question reopened the already-established "
                    f"mechanism ({mechanism.statement!r}) — truncating chain here"
                ))
                break

            # GROUNDING GUARD: an answer that references an entity/number absent
            # from this finding's evidence cannot be trusted as-is — replace it
            # and force the status to UNKNOWN so the evidence-bound stop below
            # truncates the chain right here.
            if answer:
                violations = ungrounded_entities(str(answer), source_text)
                if violations:
                    trace.append(AgentTraceStep.warn(
                        f"5-Why answer referenced ungrounded entity/number {violations} — marked UNKNOWN"
                    ))
                    answer = "The available evidence does not establish this — the prior answer could not be traced to this finding's evidence."
                    s_status = "UNKNOWN"
                elif is_circular_why_answer(question, answer):
                    trace.append(AgentTraceStep.warn(
                        f"5-Why step {len(steps) + 1} answer restated its own question instead of "
                        "explaining it — marked UNKNOWN"
                    ))
                    answer = "The available evidence does not establish this — the prior answer could not be traced to a distinct causal explanation."
                    s_status = "UNKNOWN"
                elif steps and repeats_previous_why_answer(steps[-1].answer, answer):
                    trace.append(AgentTraceStep.warn(
                        f"5-Why step {len(steps) + 1} repeated the previous step's answer instead of "
                        "advancing — marked UNKNOWN"
                    ))
                    answer = "The available evidence does not establish a further cause beyond the preceding step."
                    s_status = "UNKNOWN"
                elif len(steps) >= 1 and canonical and canonical.observed_deviation and restates_observation(answer, canonical.observed_deviation):
                    trace.append(AgentTraceStep.warn(
                        f"5-Why step {len(steps) + 1} answer merely restated the original observation "
                        "instead of explaining it — marked UNKNOWN"
                    ))
                    answer = "The available evidence does not establish a cause beyond the observation itself."
                    s_status = "UNKNOWN"
            steps.append(FiveWhyStep(
                question=question,
                answer=answer,
                status=s_status,  # type: ignore[arg-type]
            ))

        # WHY 1 must restate an established fact from the finding/evidence ledger,
        # per the rca.txt prompt contract — enforce VERIFIED regardless of wording.
        if steps and (steps[0].answer or "").strip():
            steps[0].status = "VERIFIED"

        # Evidence-bound stop: truncate the chain right after the first step whose
        # evidence status is UNKNOWN/NOT_ESTABLISHED — never force speculative
        # steps beyond the point where evidence actually runs out.
        for idx, step in enumerate(steps):
            if step.status in ("UNKNOWN", "NOT_ESTABLISHED"):
                steps = steps[: idx + 1]
                break

        five_why_status_note = str(fw_raw.get("status_note", ""))
        if not five_why_status_note:
            if root_cause.status in ("NOT_ESTABLISHED", "STATED_UNVERIFIED"):
                five_why_status_note = "INCOMPLETE — ROOT CAUSE NOT ESTABLISHED"
            else:
                five_why_status_note = "COMPLETE"

        five_why = FiveWhyAnalysis(
            steps=steps,
            is_complete=bool(fw_raw.get("is_complete", False)),
            status_note=five_why_status_note,
        )

        trace.append(AgentTraceStep.ok(
            f"Root cause analyzed: {root_cause.status.value} — {root_cause.category or 'no category'}"
        ))
        if not five_why.is_complete:
            trace.append(AgentTraceStep.warn(f"5-Why note: {five_why.status_note}"))

    except (LLMError, ValueError, KeyError) as exc:

        logger.warning("RCA node failed: %s", exc)
        trace.append(AgentTraceStep.warn(f"Root cause analysis failed — defaulting to NOT_ESTABLISHED: {exc}"))
        errors.append(f"RCA error: {exc}")

    return {
        **state,
        "root_cause": root_cause,
        "five_why": five_why,
        "contributing_factors": contributing_factors,
        "trace": trace,
        "errors": errors,
    }
