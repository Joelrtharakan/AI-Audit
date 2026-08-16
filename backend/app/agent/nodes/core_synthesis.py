"""Consolidated Node: core_synthesis

Replaces serial rca -> impact -> capa -> ca_draft LLM calls with a single
high-reasoning, structured JSON call. Executes RCA, 5-Why, Impact, CAPA, and
CA Draft simultaneously in ONE LLM round-trip.

Failure path (never skips straight to deterministic synthesis on a mere
output-budget ceiling -- see `_classify_failure` and `_parse_causal_fields`):

    Ollama primary call
      -> parse JSON / schema validate
      -> ACCEPT if valid (regardless of how close to the token ceiling it got)
      -> else: compact JSON-first recovery call (causal fields only)
      -> ACCEPT recovery if valid (analysis_mode stays "LLM")
      -> else: deterministic evidence-grounded synthesis (analysis_mode = "DETERMINISTIC")
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agent.analytical_validator import (
    hypothesis_confidence,
    leading_hypothesis_confidence,
    leading_hypothesis_display,
    leading_hypothesis_status,
)
from app.agent.causal_guard import (
    MechanismInfo,
    answer_asserts_verified_but_is_reported,
    classify_mixed_evidence_answer,
    hypothesis_attacks_statement_credibility,
    hypothesis_contradicts_mechanism,
    hypothesis_contradicts_verified_completion,
    hypothesis_discrimination_cites_wrong_id,
    hypothesis_overclaims_human_error,
    is_circular_why_answer,
    is_evidence_gap_not_hypothesis,
    is_generic_non_analysis_filler,
    is_reporting_why_question,
    mechanism_already_names_generic_hypothesis,
    question_reopens_mechanism,
    repeats_previous_why_answer,
    restates_observation,
)
from app.agent.grounding_guard import (
    build_source_text,
    claims_unsupported_effectiveness,
    clean_structured_leak,
    is_placeholder_leak,
    mentions_unsupported_domain,
    ungrounded_entities,
)
from app.agent.permissions import build_ca_draft
from app.agent.state import AgentState
from app.config import get_settings
from app.models.agent import (
    AgentTraceStep,
    CandidateHypothesis,
    CapaAnalysis,
    CapaStatus,
    ContributingFactor,
    EvidenceStatus,
    FiveWhyAnalysis,
    FiveWhyStep,
    ImpactAssessment,
    ImpactStatus,
    InvestigationQuestion,
    RootCauseAnalysis,
    RootCauseStatus,
)
from app.services.llm_client import LLMError, LLMNetworkError, LLMTimeoutError, get_llm_client
from app.services.llm_json import parse_llm_json
from app.services.taxonomy import coerce_category

logger = logging.getLogger(__name__)


def _assign_claim_ids(evidence_ledger: list) -> list[tuple[str, Any]]:
    """ONE canonical claim-ID assignment (C1, C2, ... in evidence-ledger
    order), reused for both the primary and recovery prompts and for
    validating the LLM's supporting_claim_ids/contradicting_claim_ids
    against. IDs are positional in the ORIGINAL (untrimmed) ledger, so "C3"
    means the same claim whether it appears in the primary call's full
    ledger or the recovery call's trimmed subset -- never re-indexed per
    call, which would let the same ID silently mean two different claims
    across the two prompts."""
    return [(f"C{i + 1}", e) for i, e in enumerate(evidence_ledger)]


def _classify_failure(exc: Exception, ollama_meta: dict) -> str:
    """Map a core_synthesis call failure to one of the specific categories
    from the performance-logging contract, instead of collapsing everything
    into two coarse buckets.

    Crucially: OUTPUT_TRUNCATED is only ever returned when Ollama itself
    reports `done_reason == "length"` (surfaced here as
    `ollama_meta["hit_output_limit"]`) AND parsing/validation actually
    failed. generated_tokens == max_output_tokens alone is never sufficient
    proof of truncation -- a model can legitimately fill the entire output
    budget and still return complete, valid JSON, in which case no exception
    reaches this function at all.
    """
    try:
        from pydantic import ValidationError as _PydanticValidationError
    except ImportError:  # pragma: no cover - pydantic always installed here
        _PydanticValidationError = ()  # type: ignore[assignment]

    if isinstance(exc, LLMTimeoutError):
        return "TIMEOUT"
    if isinstance(exc, LLMNetworkError):
        return "PROVIDER_FAILURE"
    if isinstance(exc, _PydanticValidationError):
        return "SCHEMA_VALIDATION_FAILURE"
    if isinstance(exc, LLMError):
        # Any other provider-level error (HTTP 4xx/5xx, empty completion,
        # unexpected response shape) that isn't a timeout/network error.
        return "PROVIDER_FAILURE"
    if ollama_meta.get("hit_output_limit"):
        return "OUTPUT_TRUNCATED"
    return "INVALID_JSON"


_MAX_LLM_HYPOTHESES = 3


def _parse_causal_fields(
    parsed: dict,
    mechanism: MechanismInfo,
    evidence_ledger: list,
    source_text: str,
    observed_deviation: str,
    finding_text: str,
    trace: list,
    has_unresolved_conflict: bool = False,
    claim_ids: list[tuple[str, Any]] | None = None,
) -> tuple[RootCauseAnalysis, FiveWhyAnalysis, list[ContributingFactor]]:
    """Parse+guard root_cause / five_why / contributing_factors from a
    core_synthesis-shaped JSON object.

    This is the SINGLE canonical parser for these three fields -- used both
    for the primary full-schema response and for the compact recovery
    response (Section 4/19: one canonical representation, not two competing
    implementations of the same guards). All the causal-grounding guards
    (contradiction detection, domain-leakage, evidence-gap-not-hypothesis,
    etc.) apply identically regardless of which call produced the JSON.
    """
    # -----------------------------------------------------------------
    # 1. Root cause + candidate hypotheses
    # -----------------------------------------------------------------
    raw_rc = parsed.get("root_cause", {})
    from app.services.status_normalizer import normalize_root_cause_status
    rc_status = normalize_root_cause_status(raw_rc.get("status"))

    rc_category = coerce_category(raw_rc.get("category"))
    if rc_status == RootCauseStatus.NOT_ESTABLISHED:
        rc_category = "TO_BE_CONFIRMED"

    from app.services.status_normalizer import normalize_hypothesis_status
    verified_facts = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.VERIFIED]
    valid_claim_ids = {cid for cid, _ in (claim_ids or [])}
    claim_text_by_id = {cid: e.claim for cid, e in (claim_ids or [])}
    _raw_hyp_count = len(raw_rc.get("candidate_hypotheses", []))
    cand_hypotheses = []
    for ch in raw_rc.get("candidate_hypotheses", []):
        if isinstance(ch, dict):
            rank_raw = str(ch.get("relevance_rank", "HIGH")).upper()
            rank = rank_raw if rank_raw in ("HIGH", "MEDIUM", "LOW") else "HIGH"
            # Single authoritative status-normalization boundary (shared with
            # root_cause.status above) -- maps LLM synonyms like "CONFIRMED"/
            # "REJECTED"/"TIED" onto their correct canonical status instead of
            # collapsing every unrecognized value to a blunt "POSSIBLE"
            # default, while still never raising on an invalid enum.
            hyp_status = normalize_hypothesis_status(ch.get("status"))
            statement = clean_structured_leak(ch.get("statement", ""))
            # HUMAN-ERROR OVERCLAIMING (structural, not finding-specific):
            # a hypothesis that stops at "human oversight/error" without
            # also asking what process/control allowed the error is
            # premature -- never delete it (human execution error can be
            # a genuine hypothesis), just don't let it outrank a
            # systemic explanation that hasn't been given the same
            # benefit of the doubt.
            if hypothesis_overclaims_human_error(statement) and rank == "HIGH":
                rank = "MEDIUM"
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: hypothesis {ch.get('id', 'H')} attributes the deviation to human "
                    "error/oversight without a process/control framing — relevance downgraded to avoid "
                    "prematurely leading with an individual-blame explanation"
                ))
            if hyp_status == "REFUTED":
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — directly contradicted by evidence"
                ))
                continue
            # CONTRADICTION-AWARE FILTER (structural, not finding-specific):
            # a hypothesis claiming "performed but not recorded" cannot
            # coexist with an established mechanism confirming the
            # activity was not performed at all — regardless of what
            # status the LLM assigned it.
            if hypothesis_contradicts_mechanism(statement, mechanism):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — contradicts the "
                    f"established mechanism ({mechanism.statement!r})"
                ))
                continue
            # REDUNDANCY FILTER: a hedged restatement of the already-established
            # mechanism is not a new hypothesis about its CAUSE.
            if mechanism_already_names_generic_hypothesis(statement, mechanism):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — restates the already-"
                    "established mechanism as an open possibility instead of reasoning about its cause"
                ))
                continue
            # VERIFIED-COMPLETION CONTRADICTION (structural, not tied to any
            # topic word): a hypothesis proposing a deficiency/absence of
            # something a VERIFIED fact already confirms WAS done (e.g.
            # "training deficiency" when the ledger confirms training was
            # completed) is contradicted, not merely unlikely.
            if hypothesis_contradicts_verified_completion(statement, verified_facts):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — contradicts a VERIFIED "
                    "fact that the thing it claims is deficient was actually completed"
                ))
                continue
            # STATEMENT-CREDIBILITY GUARD (structural, not finding-specific):
            # a hypothesis whose mechanism is "the supervisor's/operator's
            # claim was inaccurate/dishonest" attacks the credibility of a
            # REPORTED statement instead of reasoning about the underlying
            # proposition -- a person's statement is evidence, never itself
            # a root-cause mechanism.
            if hypothesis_attacks_statement_credibility(statement):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — attacks the credibility of "
                    "a reported statement instead of reasoning about the underlying proposition"
                ))
                continue
            # HYPOTHESIS CAUSALITY FILTER (structural, not finding-specific):
            # a "hypothesis" that just restates an evidence gap already
            # stated in the finding (e.g. "the certificate was not
            # available") is not a causal explanation for WHY the
            # deviation occurred -- it's a fact the finding already
            # gives, dressed up as a hypothesis. Reject it rather than
            # let it crowd out an actual candidate cause.
            if is_evidence_gap_not_hypothesis(statement, source_text):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — restates an evidence "
                    "gap/fact already stated in the finding rather than proposing a causal explanation"
                ))
                continue
            # PROVENANCE ENFORCEMENT: only applies when the response
            # actually uses the structured-provenance schema (either
            # supporting_claim_ids or contradicting_claim_ids key present)
            # -- a response using an older/different shape that never
            # attempts this field isn't "asserting zero provenance", it's
            # a different (still test-covered) contract, so it is left to
            # the existing statement-level guards above/below instead.
            # Once a response DOES use this schema: an ID the evidence
            # ledger never issued is always rejected (the LLM cannot
            # invent claim IDs), and citing NEITHER a supporting NOR a
            # contradicting claim at all is zero provenance and rejected
            # outright, regardless of how plausible the statement reads.
            raw_supporting_ids = ch.get("supporting_claim_ids")
            raw_contradicting_ids = ch.get("contradicting_claim_ids")
            provenance_schema_used = raw_supporting_ids is not None or raw_contradicting_ids is not None
            supporting_ids: list[str] = []
            contradicting_ids: list[str] = []
            if provenance_schema_used and valid_claim_ids:
                supporting_ids = [c for c in (raw_supporting_ids or []) if isinstance(c, str)]
                contradicting_ids = [c for c in (raw_contradicting_ids or []) if isinstance(c, str)]
                invalid_ids = [c for c in (*supporting_ids, *contradicting_ids) if c not in valid_claim_ids]
                if invalid_ids:
                    trace.append(AgentTraceStep.warn(
                        f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — cited claim id(s) "
                        f"{invalid_ids} that do not exist in the evidence ledger"
                    ))
                    from app.services import llm_metrics as _llm_metrics
                    _llm_metrics.increment("hypotheses_provenance_rejected")
                    continue
                if not supporting_ids and not contradicting_ids:
                    trace.append(AgentTraceStep.warn(
                        f"Core Synthesis: dropped hypothesis {ch.get('id', 'H')} — cited zero supporting "
                        "or contradicting claim provenance"
                    ))
                    from app.services import llm_metrics as _llm_metrics
                    _llm_metrics.increment("hypotheses_provenance_rejected")
                    continue
            new_hyp = CandidateHypothesis(
                id=str(ch.get("id", "H")),
                name=str(ch.get("name", "HYPOTHESIS")),
                statement=statement,
                status=hyp_status,
                evidence_needed=clean_structured_leak(ch.get("evidence_needed", "")) or "Investigation required",
                discrimination_evidence=clean_structured_leak(ch.get("discrimination_evidence")) or None,
                relevance_rank=rank,
                rationale=clean_structured_leak(ch.get("rationale")) or None,
                evidence_against=clean_structured_leak(ch.get("evidence_against")) or None,
                confirms_if=clean_structured_leak(ch.get("confirms_if")) or None,
                refutes_if=clean_structured_leak(ch.get("refutes_if")) or None,
                supporting_evidence=[claim_text_by_id[c] for c in supporting_ids if c in claim_text_by_id],
                contradicting_evidence=[claim_text_by_id[c] for c in contradicting_ids if c in claim_text_by_id],
            )
            # CROSS-HYPOTHESIS SEMANTIC CONSISTENCY: reject a hypothesis
            # whose own discrimination/confirms/refutes text asserts that
            # its evidence "supports"/"confirms" a DIFFERENT hypothesis id
            # -- evidence and discrimination criteria must describe the SAME
            # causal proposition as the hypothesis they're attached to.
            if hypothesis_discrimination_cites_wrong_id(new_hyp):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: dropped hypothesis {new_hyp.id} — its discrimination criterion "
                    "describes evidence that supports a DIFFERENT hypothesis id, not its own"
                ))
                continue
            cand_hypotheses.append(new_hyp)

    # HYPOTHESIS COUNT CAP (Phase 4/9 of the LLM-boundary rebuild): the
    # compact prompt now asks for at most _MAX_LLM_HYPOTHESES, but this is
    # the deterministic enforcement of that limit -- never trust prompt
    # wording alone to bound how many a model actually returns. Highest-
    # ranked (then first-seen) survive; this is independent of, and
    # stricter than, the separate 4-hypothesis cap final_evidence_
    # verification applies later as defense-in-depth.
    if len(cand_hypotheses) > _MAX_LLM_HYPOTHESES:
        _rank_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        dropped = cand_hypotheses[_MAX_LLM_HYPOTHESES:]
        cand_hypotheses = sorted(cand_hypotheses, key=lambda h: _rank_order.get(h.relevance_rank, 1))[:_MAX_LLM_HYPOTHESES]
        trace.append(AgentTraceStep.warn(
            f"Core Synthesis: LLM returned {len(dropped) + len(cand_hypotheses)} hypotheses — capped to "
            f"{_MAX_LLM_HYPOTHESES} highest-ranked"
        ))

    from app.services import llm_metrics as _llm_metrics
    _llm_metrics.increment("hypotheses_generated", len(cand_hypotheses))
    _llm_metrics.increment("hypotheses_rejected", max(_raw_hyp_count - len(cand_hypotheses), 0))

    root_cause = RootCauseAnalysis(
        status=rc_status,
        category=rc_category,
        statement=clean_structured_leak(raw_rc.get("statement")) or None,
        leading_hypothesis=clean_structured_leak(raw_rc.get("leading_hypothesis")) or None,
        candidate_hypotheses=cand_hypotheses,
        risk_of_recurrence=raw_rc.get("risk_of_recurrence", "NOT_ASSESSABLE"),
        narrative=clean_structured_leak(raw_rc.get("narrative")) or "The available evidence establishes the observed condition but does not establish why it occurred.",
        root_cause_basis=clean_structured_leak(raw_rc.get("root_cause_basis")) or None,
        evidence_required=[
            clean_structured_leak(x) for x in raw_rc.get("evidence_required", []) if clean_structured_leak(x)
        ],
        leading_hypothesis_rationale=clean_structured_leak(raw_rc.get("leading_hypothesis_rationale")) or None,
    )

    # -----------------------------------------------------------------
    # 2. 5-Why
    # -----------------------------------------------------------------
    raw_fw = parsed.get("five_why", {})
    fw_steps = []
    reported_facts = [e.claim for e in evidence_ledger if e.status == EvidenceStatus.REPORTED]
    from app.services.status_normalizer import normalize_five_why_status
    for s in raw_fw.get("steps", []):
        if isinstance(s, dict):
            # Single authoritative status-normalization boundary (same
            # module used for root_cause/hypothesis statuses above) --
            # maps LLM synonyms ("CONFIRMED", "CONFLICTING", "STATED", etc.)
            # onto their correct canonical status instead of a blunt
            # substring-only fallback, while never raising on an invalid enum.
            st = normalize_five_why_status(s.get("status"))
            question = clean_structured_leak(s.get("question", ""))
            answer = clean_structured_leak(s.get("answer", ""))

            # MIXED-EVIDENCE GUARD (structural): a compound answer whose
            # clauses carry DIFFERENT evidence provenance (e.g. "X
            # stated Y occurred, but Z was independently observed") must
            # never collapse to one status word -- checked BEFORE the
            # pure-REPORTED check below, since a mixed sentence would
            # otherwise get blanket-downgraded to REPORTED_UNVERIFIED,
            # hiding that part of it (e.g. "Z was not available") is
            # independently a VERIFIED observation, not a reported one.
            mixed = classify_mixed_evidence_answer(answer)
            if mixed:
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} answer combines clauses with "
                    "different evidence provenance (reported + independently observed) — labeled MIXED"
                ))
                st = mixed
            # REPORTED != VERIFIED GUARD (structural): the model
            # labeled this step VERIFIED, but its content traces back to
            # a REPORTED evidence-ledger claim (someone's account), not
            # a VERIFIED one -- never let a status word alone promote a
            # reported statement to established fact.
            elif answer_asserts_verified_but_is_reported(answer, st, reported_facts, verified_facts):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} labeled VERIFIED but its content "
                    "traces to a REPORTED statement, not a VERIFIED fact — downgraded to REPORTED_UNVERIFIED"
                ))
                st = "REPORTED_UNVERIFIED"

            # CONSOLIDATED WHY-QUESTION QUALITY GATE (Phase 5)
            from app.agent.causal_guard import validate_why_question
            prev_ans = fw_steps[-1].answer if fw_steps else None
            is_valid_q, invalid_reason = validate_why_question(
                question=question,
                previous_answer=prev_ans,
                observation=observed_deviation,
                mechanism=mechanism,
                finding_text=finding_text,
            )
            if not is_valid_q:
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} question rejected ({invalid_reason}) — truncating chain here"
                ))
                break

            if is_circular_why_answer(question, answer):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} answer restated its own "
                    "question instead of explaining it — truncating chain here"
                ))
                answer = "The available evidence does not establish this — the prior answer could not be traced to a distinct causal explanation."
                st = "UNKNOWN"
            elif fw_steps and repeats_previous_why_answer(fw_steps[-1].answer, answer):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} repeated the previous "
                    "step's answer instead of advancing — truncating chain here"
                ))
                answer = "The available evidence does not establish a further cause beyond the preceding step."
                st = "UNKNOWN"
            elif len(fw_steps) >= 1 and restates_observation(answer, observed_deviation):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} answer merely restated the "
                    "original observation instead of explaining it — truncating chain here"
                ))
                answer = "The available evidence does not establish a cause beyond the observation itself."
                st = "UNKNOWN"
            elif ungrounded_entities(answer, source_text):
                trace.append(AgentTraceStep.warn(
                    f"Core Synthesis: 5-Why step {len(fw_steps) + 1} answer referenced an "
                    "ungrounded entity/number — truncating chain here"
                ))
                answer = "The available evidence does not establish this — the answer could not be traced to this finding's evidence."
                st = "UNKNOWN"

            fw_steps.append(FiveWhyStep(
                level=int(s.get("level", len(fw_steps) + 1)),
                question=question,
                answer=answer,
                status=st,
                evidence_reference=clean_structured_leak(s.get("evidence_reference")),
            ))
            # Evidence-bound stop: truncate right after the first step
            # whose status is UNKNOWN/NOT_ESTABLISHED (including one this
            # guard just forced) — never fabricate steps past that point.
            if st in ("UNKNOWN", "NOT_ESTABLISHED"):
                break

    five_why = FiveWhyAnalysis(
        steps=fw_steps,
        is_complete=bool(raw_fw.get("is_complete", False)) and not any(s.status in ("UNKNOWN", "NOT_ESTABLISHED") for s in fw_steps),
        status_note=clean_structured_leak(raw_fw.get("status_note")) or "Stopped at evidence boundary",
    )

    # -----------------------------------------------------------------
    # 3. Contributing factors (dynamically derived, not a fixed
    # "not established" filler — an empty list is a correct outcome when
    # the finding supports none).
    # -----------------------------------------------------------------
    contributing_factors: list[ContributingFactor] = []
    valid_cf_status = {"ESTABLISHED", "POTENTIAL", "POSSIBLE_UNCONFIRMED", "VERIFIED"}
    for cf in parsed.get("contributing_factors", []):
        if not isinstance(cf, dict):
            continue
        description = clean_structured_leak(cf.get("description", ""))
        if not description:
            continue
        # A boilerplate "not established" non-answer carries zero
        # analytical content -- an empty contributing_factors list is
        # the correct representation of "nothing supports a factor
        # here", not an entry that just says so in words.
        if is_generic_non_analysis_filler(description):
            trace.append(AgentTraceStep.warn(
                "Core Synthesis: dropped contributing factor — generic non-analysis filler text"
            ))
            continue
        # Same causality discipline applied to hypotheses: a
        # "contributing factor" that's really just a restated evidence
        # gap or the reported mechanism itself (not a CONDITION that
        # made the mechanism more likely) carries no analytical content
        # beyond what's already stated elsewhere in the report.
        if is_evidence_gap_not_hypothesis(description, source_text):
            trace.append(AgentTraceStep.warn(
                "Core Synthesis: dropped contributing factor — restates an evidence gap/fact "
                f"already stated in the finding rather than a condition contributing to it: {description!r}"
            ))
            continue
        cf_evidence_status = str(cf.get("evidence_status", "INFERRED")).upper()
        if cf_evidence_status not in {s.value for s in EvidenceStatus}:
            cf_evidence_status = "INFERRED"
        cf_status = str(cf.get("status", "POTENTIAL")).upper()
        if cf_status not in valid_cf_status:
            cf_status = "POTENTIAL" if cf_evidence_status != "VERIFIED" else "ESTABLISHED"
        contributing_factors.append(ContributingFactor(
            description=description,
            evidence_status=EvidenceStatus(cf_evidence_status),
            status=cf_status,
            rationale=clean_structured_leak(cf.get("rationale")) or None,
            evidence_required=clean_structured_leak(cf.get("evidence_required")) or None,
        ))

    # -----------------------------------------------------------------
    # 4. Production-Grade Post-Processing Guards
    # -----------------------------------------------------------------

    # CERTAINTY MONOTONICITY & CATEGORY LOCK: if root cause is
    # NOT_ESTABLISHED, category MUST be TO_BE_CONFIRMED.
    if root_cause.status == RootCauseStatus.NOT_ESTABLISHED:
        root_cause.category = "TO_BE_CONFIRMED"

    # DOMAIN LEAKAGE & CAUSAL ANCHOR GUARD
    # Drop candidate hypotheses that invoke unsupported domains or lack causal anchors
    kept_hypotheses = []
    for h in root_cause.candidate_hypotheses:
        if mentions_unsupported_domain(h.statement, source_text) or mentions_unsupported_domain(h.name, source_text):
            trace.append(AgentTraceStep.warn(
                f"Core Synthesis: dropped hypothesis mentioning unsupported domain: {h.statement!r}"
            ))
        elif ungrounded_entities(h.statement, source_text):
            trace.append(AgentTraceStep.warn(
                f"Core Synthesis: dropped hypothesis with ungrounded entity: {h.statement!r}"
            ))
        else:
            kept_hypotheses.append(h)

    root_cause.candidate_hypotheses = kept_hypotheses

    # Same grounding/domain discipline applied to contributing factors —
    # they carry the same fabrication risk as hypotheses.
    kept_factors = []
    for cf in contributing_factors:
        if mentions_unsupported_domain(cf.description, source_text):
            trace.append(AgentTraceStep.warn(
                f"Core Synthesis: dropped contributing factor — unsupported domain: {cf.description!r}"
            ))
        elif ungrounded_entities(cf.description, source_text):
            trace.append(AgentTraceStep.warn(
                f"Core Synthesis: dropped contributing factor — ungrounded entity: {cf.description!r}"
            ))
        else:
            kept_factors.append(cf)
    contributing_factors = kept_factors

    # Ensure candidate_hypotheses is NEVER empty
    if not root_cause.candidate_hypotheses:
        from app.agent.nodes.plan_investigation_fallback import build_deterministic_investigation_plan
        fallback_hyps, _ = build_deterministic_investigation_plan(finding_text, evidence_ledger)
        root_cause.candidate_hypotheses = fallback_hyps
        trace.append(AgentTraceStep.warn("Core Synthesis: generated fallback candidate hypotheses from finding text"))

    # Every surviving hypothesis carries its own deterministic confidence
    # grade (never asserted by the LLM) so a report never shows a bare
    # hypothesis with no sense of how strong it is.
    for h in root_cause.candidate_hypotheses:
        h.confidence = hypothesis_confidence(h)

    # LEADING HYPOTHESIS ENFORCEMENT: root_cause.status staying
    # NOT_ESTABLISHED must not mean the report goes silent about which
    # candidate is best-evidenced -- deterministic, not dependent on the
    # LLM remembering to fill leading_hypothesis. select_leading_hypothesis
    # is the single source of truth for this rule (also used by the
    # analytical validator) -- it deliberately returns None when
    # hypotheses are equally plausible rather than forcing a pick. That
    # TIED case must still be visible, not indistinguishable from "no
    # hypotheses at all" -- leading_hypothesis_status/_display make that
    # explicit instead of leaving the field silently blank.
    root_cause.leading_hypothesis_status = leading_hypothesis_status(root_cause.candidate_hypotheses)
    if not root_cause.leading_hypothesis:
        root_cause.leading_hypothesis = leading_hypothesis_display(root_cause.candidate_hypotheses)
        if root_cause.leading_hypothesis_status == "SELECTED":
            root_cause.confidence = leading_hypothesis_confidence(
                root_cause.candidate_hypotheses, root_cause.leading_hypothesis
            )

    # CONFLICT-AWARE TIE OVERRIDE: candidate hypotheses that are competing
    # explanations for the SAME unresolved evidence conflict must never have
    # a single leader promoted merely because of an incidental scoring
    # difference between equally-plausible statements -- see
    # apply_conflict_tie_override's docstring. A no-op unless the scorer
    # actually picked a winner above.
    from app.agent.analytical_validator import apply_conflict_tie_override
    apply_conflict_tie_override(root_cause, has_unresolved_conflict)

    # REPORTED STATEMENT ATTRIBUTION GUARD
    # If narrative claims carelessness/negligence from reported statements, flag & sanitize
    if "careless" in root_cause.narrative.lower() or "negligen" in root_cause.narrative.lower():
        root_cause.narrative = (
            "Personnel reported attribution to human factors/carelessness. "
            "This statement is unverified and does not establish objective root cause."
        )

    return root_cause, five_why, contributing_factors


def _derive_ca_draft_fields(root_cause, impact) -> dict:
    """Derives the 5 CA-draft fields from the already-synthesized
    root_cause/impact objects instead of asking the LLM for a second,
    separate restatement of the same analysis."""
    affected = (impact.affected_object if impact and impact.affected_object else None) or "the affected process/item"
    from app.services.semantic_subject import validate_semantic_subject
    if not validate_semantic_subject(affected):
        affected = "the affected process/record"

    leading_stmt = None
    if root_cause and getattr(root_cause, "leading_hypothesis_status", None) == "SELECTED" and getattr(root_cause, "candidate_hypotheses", None):
        leading_id = (getattr(root_cause, "leading_hypothesis", None) or "").split(" — ", 1)[0].strip()
        match = next((h for h in root_cause.candidate_hypotheses if h.id == leading_id), None)
        if match:
            leading_stmt = match.statement

    is_established = bool(root_cause) and root_cause.status not in (
        RootCauseStatus.NOT_ESTABLISHED, "NOT_ESTABLISHED",
    )
    if is_established:
        root_cause_text = root_cause.narrative
    elif leading_stmt:
        root_cause_text = f"NOT_ESTABLISHED — leading hypothesis: {leading_stmt}"
    elif root_cause:
        root_cause_text = f"NOT_ESTABLISHED — {root_cause.narrative}"
    else:
        root_cause_text = "NOT_ESTABLISHED"

    preventive_action = (
        f"Address the confirmed cause via {leading_stmt[0].lower()}{leading_stmt[1:]}" if leading_stmt
        else "Strengthen the relevant control once root cause is confirmed."
    )

    import re as _re
    # Avoid "Verify the current status of X status" when the affected-object
    # phrase already names a status/qualification (e.g. "Operator training
    # status for the revised procedure") -- say it once, not twice.
    if affected.strip().lower().endswith(("status", "qualification")) or " status for " in affected.lower() or " qualification for " in affected.lower():
        lead_word = affected.split(" ", 1)[0].lower()
        article = "" if lead_word in ("the", "a", "an") else "the "
        immediate_action = f"Verify {article}{affected[0].lower()}{affected[1:]} against authorized records before permitting independent execution or release, where applicable."
    elif _re.search(r"\b(records?|logs?|documentation)\b", affected, _re.IGNORECASE):
        # The affected object IS itself a record/log/documentation artifact
        # (e.g. "Temperature monitoring records for refrigerator
        # QC-REF-02") -- "permitting independent execution" doesn't fit a
        # records-completeness finding; the appropriate immediate action is
        # reviewing the affected records and assessing current status
        # against the applicable procedure, not a personnel authorization
        # decision.
        lead_word = affected.split(" ", 1)[0].lower()
        article = "" if lead_word in ("the", "a", "an") else "the "
        immediate_action = (
            f"Review {article}{affected[0].lower()}{affected[1:]} and assess the current status of the "
            "affected item(s) in accordance with the applicable procedure."
        )
    else:
        immediate_action = (
            f"Verify the current status of {affected} against authorized records before permitting "
            "independent execution or release, where applicable."
        )

    return {
        "immediate_action": immediate_action,
        "root_cause": root_cause_text,
        "root_cause_category": (
            "TO_BE_CONFIRMED" if (not root_cause or not is_established) else root_cause.category
        ),
        "preventive_action": preventive_action,
        "impact_analysis": (impact.narrative if (impact and impact.narrative) else "Impact scope pending auditor verification."),
    }


def _trim_evidence_for_recovery(claim_ids: list[tuple[str, Any]]) -> list[dict]:
    """Build a materially smaller evidence context for the recovery retry.

    Keeps only VERIFIED/REPORTED items (the tiers the causal reasoning
    actually leans on) and caps the count, ranked by relevance -- this is
    what makes the recovery call "compact", not just a shorter output
    budget on the exact same input. Takes the SAME (id, EvidenceItem) pairs
    `_assign_claim_ids` produced for the primary call, so a claim's ID is
    identical across both prompts, and emits the same compact
    {id, claim, status} shape the primary prompt uses -- not a full
    `.model_dump()` (source/source_reference/relevance/notes), which is
    both unnecessary token weight and inconsistent with the primary
    prompt's evidence shape.
    """
    relevance_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    candidates = [
        (cid, e) for cid, e in claim_ids
        if e.status in (EvidenceStatus.VERIFIED, EvidenceStatus.REPORTED)
    ]
    candidates.sort(key=lambda pair: relevance_rank.get(getattr(pair[1], "relevance", "MEDIUM"), 1))
    trimmed = candidates[:6] or claim_ids[:6]
    return [
        {"id": cid, "claim": e.claim, "status": getattr(e.status, "value", str(e.status))}
        for cid, e in trimmed
    ]


def _derive_deterministic_impact(request_finding_text: str, canonical, observed_deviation: str) -> tuple[ImpactAssessment, str, str, str | None]:
    """Shared deterministic impact derivation used both when a compact
    recovery call succeeds (recovery's schema deliberately excludes impact)
    and when full deterministic synthesis runs. Returns
    (impact, clean_noun, topic, actor) so callers can reuse clean_noun/topic
    for CAPA derivation without recomputing them."""
    from app.services.semantic_subject import (
        build_affected_object_phrase,
        extract_temporal_clause,
        resolve_deviation,
        split_topic_and_tail,
        strip_leading_article,
        topic_word,
    )

    fact_claims = []  # populated by caller via evidence_ledger where available; kept for signature symmetry
    resolved = resolve_deviation(request_finding_text, fact_claims)
    clean_noun = resolved.subject or "UNKNOWN — no affected object could be isolated from the finding text"
    topic = topic_word(clean_noun)
    topic_cap = topic[0].upper() + topic[1:]
    temporal_clause = extract_temporal_clause(request_finding_text)
    if temporal_clause:
        degraded_period = temporal_clause[0].upper() + temporal_clause[1:]
    else:
        degraded_period = (
            canonical.affected_period if (canonical and canonical.affected_period != "UNKNOWN") else None
        ) or "UNKNOWN"
    # Strip a stray leading article (e.g. an actor captured mid-sentence as
    # "The operator") before re-embedding it into new sentences, so it never
    # produces a doubled/misplaced article like "The Operator training
    # status" or "the the operator".
    actor = strip_leading_article((canonical.actor if canonical else None) or None)

    if clean_noun and not clean_noun.startswith("UNKNOWN"):
        # tail = the subject phrase with its own leading "<topic> for "
        # prefix stripped (e.g. "training for the revised procedure" ->
        # "the revised procedure"). Only set when the subject actually HAS
        # that "<topic> for <tail>" activity/qualification shape.
        tail = split_topic_and_tail(clean_noun, topic)
        if tail:
            # Activity/qualification-shaped subject (e.g. "training for the
            # revised procedure") -- affected object is a role/qualification
            # status, built from role + topic + tail exactly ONCE each.
            derived_process = f"{topic_cap} and compliance control"
            derived_obj = build_affected_object_phrase(clean_noun, actor)
            actor_word = actor.capitalize() if actor else "Personnel"
            effect_subject = actor_word.lower() if actor_word == "Personnel" else f"the {actor_word.lower()}"
            derived_effect = (
                f"If required {topic} was not completed before the applicable effective date, {effect_subject} "
                f"may have proceeded without confirmed qualification for {tail}."
            )
            derived_evidence_needed = f"Approved {topic} completion and authorization record"
        else:
            # Record/documentation-shaped subject (e.g. "temperature
            # monitoring records for refrigerator QC-REF-02") -- the
            # subject itself IS already a clean, specific affected object;
            # forcing it through the actor/qualification template above
            # would double the whole subject into its own "status for"
            # clause (e.g. "Technician temperature status for temperature
            # monitoring records..."). Use it directly instead.
            derived_obj = clean_noun[0].upper() + clean_noun[1:]
            derived_process = f"{topic_cap} monitoring and record control"
            condition = (
                canonical.deviation_condition if (canonical and canonical.deviation_condition not in (None, "UNKNOWN"))
                else "incomplete"
            )
            derived_effect = (
                f"Potential inability to confirm the required activity was performed: {condition} {topic} "
                "records prevent confirmation that the required activity was performed during the "
                "affected period. Any downstream impact requires assessment of actual conditions and "
                "applicable requirements."
            )
            derived_evidence_needed = f"Independent {topic} verification record (e.g. instrument audit trail, electronic log, or supervisory review)"
    else:
        derived_process = "NOT ESTABLISHED"
        derived_obj = "NOT ESTABLISHED"
        derived_effect = "Scope and downstream consequence require auditor assessment."
        derived_evidence_needed = "Auditor assessment of records relevant to this finding"

    impact = ImpactAssessment(
        status=ImpactStatus.IMPACT_REQUIRES_ASSESSMENT,
        affected_object=derived_obj,
        affected_period=degraded_period,
        process_at_risk=derived_process,
        relevant_change="NOT ESTABLISHED",
        potential_effect=derived_effect,
        evidence_needed=derived_evidence_needed,
        impact_observed=observed_deviation,
        impact_inferred=None,
        impact_unknown="Full scope and downstream consequence require auditor assessment.",
        narrative="Assessment of scope and downstream consequences is required based on objective records.",
    )
    return impact, clean_noun, topic, actor


async def core_synthesis_node(state: AgentState) -> AgentState:
    """Run consolidated RCA, 5-Why, Impact, CAPA, and CA Draft synthesis."""
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))
    request = state["request"]
    evidence_ledger = state.get("evidence_ledger", [])
    quality = state.get("observation_quality")
    canonical = state.get("canonical_finding_state")
    # Computed once, used everywhere a leading hypothesis is derived below
    # (primary path, recovery path, and full deterministic fallback) so the
    # conflict-tie override is applied consistently regardless of which path
    # actually produced the hypotheses.
    has_unresolved_conflict = bool(
        canonical and any(getattr(c, "status", "UNRESOLVED") == "UNRESOLVED" for c in canonical.evidence_conflicts)
    )
    # Set only by the deterministic/recovery fallback path below, which
    # builds its own well-matched investigation questions ALONGSIDE its
    # hypotheses (build_deterministic_investigation_plan) -- these must be
    # propagated into state, otherwise they're silently discarded and
    # final_evidence_verification's generic hypothesis->question fallback
    # (designed for LLM-produced hypotheses) has to re-derive questions from
    # confirms_if/refutes_if text it was never designed to parse, which can
    # mis-render compound clauses. None means "leave state's existing
    # investigation_plan untouched" (the normal LLM-success case).
    investigation_plan_override = None

    settings = get_settings()

    system_prompt = (settings.agent_prompts_dir / "system_prompt.txt").read_text(encoding="utf-8")
    template = (settings.agent_prompts_dir / "core_synthesis.txt").read_text(encoding="utf-8")
    recovery_template = (settings.agent_prompts_dir / "core_synthesis_recovery.txt").read_text(encoding="utf-8")

    observed_deviation = canonical.observed_deviation if canonical else "not extracted"
    mechanism_statement = canonical.immediate_mechanism if canonical else None
    mechanism_status = canonical.immediate_mechanism_status if canonical else "UNKNOWN"
    mechanism = MechanismInfo(
        statement=mechanism_statement,
        status=mechanism_status,
        polarity=None,
    )
    # Recompute polarity locally (cheap, deterministic) so the contradiction
    # guard below works even though CanonicalFindingState only persists the
    # statement/status, not the polarity classification.
    if mechanism_statement:
        from app.agent.causal_guard import classify_mechanism_polarity
        mechanism.polarity = classify_mechanism_polarity(mechanism_statement)

    # ONE canonical representation of the case passed to the LLM: the
    # structured evidence ledger + evidence conflicts + mechanism/deviation
    # already resolved by understand_finding_node, never the raw finding
    # text repeated a second/third time alongside it.
    #
    # Every claim carries its canonical id (C1, C2, ...) so the LLM's
    # hypotheses can cite supporting_claim_ids/contradicting_claim_ids by
    # reference instead of restating claim text -- both more compact and
    # the prerequisite for rejecting a hypothesis with invented provenance
    # (a claim ID the ledger never issued) or zero provenance at all.
    claim_ids = _assign_claim_ids(evidence_ledger)
    valid_claim_ids = {cid for cid, _ in claim_ids}
    compact_ledger = [
        {"id": cid, "claim": e.claim, "status": getattr(e.status, "value", str(e.status))}
        for cid, e in claim_ids[:6]
    ]
    compact_conflicts = [
        {"proposition": getattr(c, "proposition", str(c)), "status": getattr(c, "status", "UNRESOLVED")}
        for c in (canonical.evidence_conflicts if canonical else [])
    ]

    prompt = template.format(
        finding_text=request.finding_text,
        evidence_ledger_json=json.dumps(compact_ledger, default=str),
        observed_deviation=observed_deviation,
        immediate_mechanism=mechanism_statement or "none established",
        immediate_mechanism_status=mechanism_status,
        evidence_conflicts_json=json.dumps(compact_conflicts, default=str) if compact_conflicts else "[]",
    )

    provider_used: str | None = None
    fallback_used = False
    provider_attempts: list[str] = []

    async def _call(prompt_text: str, max_tokens: int, timeout_seconds: float, node: str, ctx_tokens: int | None = None) -> str:
        from app.services.ollama_client import set_current_node
        set_current_node(node)
        call_client = get_llm_client(timeout_seconds=timeout_seconds)
        return await call_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.1,
            response_format_json=True,
            max_tokens=max_tokens,
            num_ctx=ctx_tokens,
        )

    from app.services import llm_metrics
    llm_metrics.increment("llm_primary_attempted")

    try:
        # Section 1/2: compact schema/prompt, sufficient-not-excessive token
        # ceiling. A response that fills the whole budget and still parses
        # as valid, schema-conformant JSON is ACCEPTED below -- reaching the
        # ceiling is not itself treated as failure.
        raw = await _call(
            prompt, settings.ollama_core_synthesis_max_tokens,
            settings.ollama_primary_synthesis_timeout_seconds, "core_synthesis",
            settings.ollama_core_synthesis_num_ctx,
        )
        from app.services.llm_router import get_last_call_metadata
        _router_meta = get_last_call_metadata()
        provider_used = _router_meta.get("provider_used")
        fallback_used = bool(_router_meta.get("fallback_used", False))
        provider_attempts = list(_router_meta.get("provider_attempts", []))
        parsed = parse_llm_json(raw)
        llm_metrics.increment("llm_primary_success")
        source_text = build_source_text(request.finding_text, evidence_ledger)

        root_cause, five_why, contributing_factors = _parse_causal_fields(
            parsed, mechanism, evidence_ledger, source_text, observed_deviation, request.finding_text, trace,
            has_unresolved_conflict, claim_ids,
        )

        # ---------------------------------------------------------------------
        # Impact & CAPA: constructed deterministically from the already-
        # synthesized root_cause/hypotheses, not requested from the LLM.
        # These are the same derivation functions the recovery/fallback paths
        # below already use (one canonical implementation, not a competing
        # LLM-authored one) -- the LLM's job is compact causal reasoning
        # (hypotheses, 5-Why), never CAPA/impact prose, which reduces both
        # the schema the model must fill and the hallucination surface area.
        # ---------------------------------------------------------------------
        impact, clean_noun, topic, actor = _derive_deterministic_impact(
            request.finding_text, canonical, observed_deviation,
        )
        # NOTE: investigation_plan_override is deliberately left unset here
        # (stays None). build_deterministic_investigation_plan() generates
        # its OWN independent hypothesis set (its own H1..H4 with its own
        # statements) purely to derive CAPA potential_areas below -- those
        # IDs are NOT root_cause.candidate_hypotheses (the LLM's actual,
        # validated hypotheses) and must never be surfaced as investigation
        # questions, or the report would reference hypothesis IDs that don't
        # exist in the displayed hypothesis list (hypothesis-ID drift).
        # Investigation questions are derived downstream (final_evidence_
        # verification_node) directly from root_cause.candidate_hypotheses
        # whenever state's own investigation_plan.questions is empty --
        # the single consistent source of hypothesis-bound questions.
        from app.agent.nodes.plan_investigation_fallback import build_conditional_capa_actions
        from app.agent.nodes.plan_investigation_fallback import (
            build_deterministic_investigation_plan as _build_area_plan,
        )
        _, _area_plan = _build_area_plan(request.finding_text, evidence_ledger)
        capa = CapaAnalysis(
            status=CapaStatus.INVESTIGATION_REQUIRED,
            potential_areas=_area_plan.areas,
            recommended_investigation=[
                "Verify the candidate causal hypotheses above through objective record investigation."
            ],
            conditional_actions=build_conditional_capa_actions(root_cause.candidate_hypotheses, clean_noun, topic),
        )

        # CA DRAFT: built deterministically from the already-synthesized
        # root_cause/impact instead of asking the LLM to restate them a
        # second time.
        ca_draft = build_ca_draft(_derive_ca_draft_fields(root_cause, impact))

        analysis_mode = "LLM"
        analysis_engine = "LLM"
        trace.append(AgentTraceStep.ok("Consolidated core synthesis completed and validated against production rules."))

    except Exception as primary_exc:
        from app.services.ollama_client import get_last_call_metadata as get_last_ollama_metadata
        primary_ollama_meta = get_last_ollama_metadata()
        primary_failure_type = _classify_failure(primary_exc, primary_ollama_meta)
        llm_metrics.increment(
            "llm_primary_timeout" if primary_failure_type == "TIMEOUT" else "llm_primary_invalid_json"
        )
        logger.info(
            "node=core_synthesis failure_type=%s hit_output_limit=%s eval_count=%s max_output_tokens=%s",
            primary_failure_type,
            primary_ollama_meta.get("hit_output_limit"),
            primary_ollama_meta.get("eval_count"),
            primary_ollama_meta.get("max_output_tokens"),
        )
        trace.append(AgentTraceStep.warn(
            f"Core synthesis primary call did not produce a usable result ({primary_failure_type}) — "
            "attempting compact JSON-first recovery before deterministic synthesis."
        ))

        from app.services.llm_router import get_last_call_metadata
        _router_meta = get_last_call_metadata()
        provider_used = None
        fallback_used = True
        provider_attempts = list(_router_meta.get("provider_attempts", []))

        source_text = build_source_text(request.finding_text, evidence_ledger)
        recovery_used = False
        root_cause: RootCauseAnalysis | None = None
        five_why: FiveWhyAnalysis | None = None
        contributing_factors: list[ContributingFactor] = []

        # -------------------------------------------------------------
        # Section 4: JSON-first recovery. Do NOT immediately abandon the
        # LLM path and fall to deterministic synthesis -- one compact,
        # causal-fields-only retry against a materially smaller prompt and
        # a smaller (but sufficient) output budget first.
        # -------------------------------------------------------------
        llm_metrics.increment("llm_recovery_attempted")
        try:
            recovery_prompt = recovery_template.format(
                finding_text=request.finding_text,
                evidence_ledger_json=json.dumps(_trim_evidence_for_recovery(claim_ids), default=str),
                observed_deviation=observed_deviation,
                immediate_mechanism=mechanism_statement or "none established",
                immediate_mechanism_status=mechanism_status,
            )
            recovery_raw = await _call(
                recovery_prompt, settings.ollama_recovery_max_tokens,
                settings.ollama_recovery_synthesis_timeout_seconds, "core_synthesis_recovery",
                settings.ollama_recovery_num_ctx,
            )
            recovery_parsed = parse_llm_json(recovery_raw)
            llm_metrics.increment("llm_recovery_success")
            root_cause, five_why, contributing_factors = _parse_causal_fields(
                recovery_parsed, mechanism, evidence_ledger, source_text, observed_deviation, request.finding_text, trace,
                has_unresolved_conflict, claim_ids,
            )
            recovery_used = True
            trace.append(AgentTraceStep.ok(
                "Core Synthesis: compact JSON recovery call produced valid causal-reasoning JSON — "
                "analysis_mode remains LLM (a reduced-scope but genuine LLM analysis, not a provider "
                "failure). Impact/CAPA below use deterministic derivation since the recovery schema is "
                "causal-reasoning only."
            ))
        except Exception as recovery_exc:
            recovery_meta = get_last_ollama_metadata()
            recovery_failure_type = _classify_failure(recovery_exc, recovery_meta)
            llm_metrics.increment(
                "llm_recovery_timeout" if recovery_failure_type == "TIMEOUT" else "llm_recovery_invalid_json"
            )
            logger.info(
                "node=core_synthesis_recovery failure_type=%s recovery=DETERMINISTIC_SYNTHESIS",
                recovery_failure_type,
            )
            trace.append(AgentTraceStep.ok(
                f"Core synthesis recovery call also failed ({recovery_failure_type}) — transitioning to "
                "deterministic evidence-grounded synthesis."
            ))

        # -------------------------------------------------------------
        # Deterministic derivations shared by both outcomes: impact/CAPA
        # are always derived this way after a primary-call failure (the
        # recovery schema deliberately excludes them), and the
        # topic/clean_noun/actor values feed CAPA regardless of whether
        # root_cause/five_why came from recovery or full deterministic
        # synthesis below.
        # -------------------------------------------------------------
        impact, clean_noun, topic, actor = _derive_deterministic_impact(
            request.finding_text, canonical, observed_deviation,
        )

        from app.agent.nodes.plan_investigation_fallback import (
            build_conditional_capa_actions,
            build_deterministic_investigation_plan,
        )
        _, fallback_plan = build_deterministic_investigation_plan(request.finding_text, evidence_ledger)
        investigation_plan_override = fallback_plan

        if recovery_used and root_cause is not None and five_why is not None:
            analysis_mode = "LLM"
            analysis_engine = "LLM"
            capa = CapaAnalysis(
                status=CapaStatus.INVESTIGATION_REQUIRED,
                potential_areas=fallback_plan.areas,
                recommended_investigation=[
                    "Verify the candidate causal hypotheses above through objective record investigation."
                ],
                conditional_actions=build_conditional_capa_actions(root_cause.candidate_hypotheses, clean_noun, topic),
            )
        else:
            # -----------------------------------------------------------
            # Full deterministic evidence-grounded synthesis (Section 16):
            # reached only when BOTH the primary call and the compact
            # recovery retry failed to produce usable, valid JSON -- never
            # merely because a response consumed its configured token
            # budget while still being complete and valid.
            # -----------------------------------------------------------
            analysis_mode = "DETERMINISTIC"
            analysis_engine = "DETERMINISTIC"
            llm_metrics.increment("deterministic_fallback")

            from app.agent.nodes.five_why_fallback import build_deterministic_five_why
            five_why = build_deterministic_five_why(request.finding_text, evidence_ledger)
            fallback_hyps, fallback_plan = build_deterministic_investigation_plan(request.finding_text, evidence_ledger)
            investigation_plan_override = fallback_plan
            contributing_factors = []

            for h in fallback_hyps:
                h.confidence = hypothesis_confidence(h)
            leading_hypothesis = leading_hypothesis_display(fallback_hyps)

            rc_basis = (
                "Available evidence contains conflicting reported statements regarding completion. Objective records are required to determine the actual state."
                if has_unresolved_conflict
                else "Root cause is not established from the available evidence alone; objective records are required to confirm the causal mechanism."
            )

            root_cause = RootCauseAnalysis(
                status=RootCauseStatus.NOT_ESTABLISHED,
                category="TO_BE_CONFIRMED",
                candidate_hypotheses=fallback_hyps,
                leading_hypothesis=leading_hypothesis,
                leading_hypothesis_status=leading_hypothesis_status(fallback_hyps),
                root_cause_basis=rc_basis,
                evidence_required=["Auditor investigation and objective records required to confirm root cause."],
                narrative=(
                    "The available evidence establishes the observed condition but does not establish why it occurred. "
                    "Auditor investigation is required to verify the candidate causal hypotheses."
                ),
            )
            from app.agent.analytical_validator import apply_conflict_tie_override
            apply_conflict_tie_override(root_cause, has_unresolved_conflict)

            # CAPA areas reuse the same dynamically-derived, finding-grounded
            # investigation plan used for hypotheses above rather than a fixed
            # universal category list. conditional_actions map each surviving
            # hypothesis to an organizational corrective action (never an
            # evidence source dressed up as an action) — CAPA stays pending on
            # root cause even in deterministic mode, never a final action.
            capa = CapaAnalysis(
                status=CapaStatus.INVESTIGATION_REQUIRED,
                potential_areas=fallback_plan.areas,
                recommended_investigation=[
                    "Verify the candidate causal hypotheses above through objective record investigation."
                ],
                conditional_actions=build_conditional_capa_actions(fallback_hyps, clean_noun, topic),
            )

        # CA draft reuses the same deterministic derivation the successful-LLM
        # path uses (never a separately hand-authored, less-careful wording) —
        # this is what keeps immediate_action evidence-appropriate ("verify
        # status against the record" rather than presupposing a correction is
        # needed) whether this ran the recovery path or full deterministic
        # synthesis.
        ca_draft = build_ca_draft(_derive_ca_draft_fields(root_cause, impact))

    return {
        **state,
        "root_cause": root_cause,
        "five_why": five_why,
        "impact_assessment": impact,
        "capa_analysis": capa,
        "contributing_factors": contributing_factors,
        "ca_draft": ca_draft,
        "analysis_mode": analysis_mode,
        "analysis_engine": analysis_engine,
        "provider_used": provider_used,
        "fallback_used": fallback_used,
        "provider_attempts": provider_attempts,
        "investigation_plan": investigation_plan_override if investigation_plan_override is not None else state.get("investigation_plan"),
        "trace": trace,
        "errors": errors,
    }
