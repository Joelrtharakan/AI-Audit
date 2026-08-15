"""Analytical validation firewall.

Runs AFTER causal synthesis (core_synthesis_node, which already applies the
causal_guard contradiction/circularity guards inline) and is invoked from
final_evidence_verification_node — the single point through which every
result reaches report_generator, so there is exactly one place these
invariants are enforced, not a second competing implementation.

Every function here is a pure, deterministic, structural check over the
already-synthesized analytical state. None of it generates new causal
content: a violation is handled by REPAIRING STRUCTURE (e.g. inserting a
mechanism step the model skipped, using text that PROVENANCE already gives
us), DOWNGRADING a claim's status, or REMOVING unsupported content — never
by inventing a fact, a mechanism, or a root cause that isn't already present
in the evidence ledger / canonical finding state.
"""

from __future__ import annotations

import logging
import re

from app.agent.causal_guard import MechanismInfo, repeats_previous_why_answer
from app.services.text_grounding import significant_words

logger = logging.getLogger(__name__)

_UNKNOWN_VALUE_RE = re.compile(
    r"^\s*(not\s+established|unknown|to\s+be\s+confirmed|n/?a|none|pending|requires\s+confirmation)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 1. Leading hypothesis selection (single source of truth)
# ---------------------------------------------------------------------------

_RANK_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def select_leading_hypothesis(candidate_hypotheses: list) -> str | None:
    """Deterministic leading-hypothesis rule, used everywhere a leading
    hypothesis needs to be derived (never left to prompt compliance):

    - No hypotheses at all -> None.
    - Exactly one SUPPORTED hypothesis -> that one.
    - Multiple SUPPORTED hypotheses that are equally strong (same
      relevance_rank) -> None (they are competing, not a single leader).
    - No SUPPORTED hypothesis, but the POSSIBLE ones differ in
      relevance_rank -> the single highest-ranked one.
    - No SUPPORTED hypothesis and all POSSIBLE ones share the same
      relevance_rank -> None (equally plausible, no leader to force).
    """
    if not candidate_hypotheses:
        return None

    supported = [h for h in candidate_hypotheses if h.status == "SUPPORTED"]
    pool = supported if supported else [h for h in candidate_hypotheses if h.status != "REFUTED"]
    if not pool:
        return None

    ranks = {_RANK_ORDER.get(h.relevance_rank, 1) for h in pool}
    if len(pool) > 1 and len(ranks) == 1:
        # All tied at the same rank -- no defensible single leader.
        return None

    best = min(pool, key=lambda h: _RANK_ORDER.get(h.relevance_rank, 1))
    return f"{best.id} — {best.statement}"


def leading_hypothesis_confidence(candidate_hypotheses: list, leading_hypothesis: str | None) -> str:
    """LOW/MEDIUM/HIGH confidence for the leading hypothesis (not for the
    root cause itself, which stays whatever root_cause.confidence already
    reflects) -- SUPPORTED gets MEDIUM, a single best-ranked POSSIBLE gets
    LOW, no leader gets LOW."""
    if not leading_hypothesis or not candidate_hypotheses:
        return "LOW"
    leading_id = leading_hypothesis.split(" — ", 1)[0].strip()
    match = next((h for h in candidate_hypotheses if h.id == leading_id), None)
    if match and match.status == "SUPPORTED":
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# 2. Root-cause state validation
# ---------------------------------------------------------------------------

# Statuses that assert a causal relationship has actually been established
# (mapped onto the existing RootCauseStatus vocabulary: VERIFIED and
# SUPPORTED are the "don't claim this without evidence" tier; STATED_UNVERIFIED
# and INFERRED are already self-labeled as unconfirmed; NOT_ESTABLISHED and
# CONTRADICTED never need downgrading).
_ESTABLISHED_LIKE_STATUSES = {"VERIFIED", "SUPPORTED"}


def validate_root_cause_state(root_cause, mechanism: MechanismInfo | None) -> list[str]:
    """Mutates `root_cause` in place if its status claims more certainty
    than the evidence actually supports. Returns a list of human-readable
    warnings for the trace (empty if nothing was downgraded).

    Critical distinction (Case A vs Case C from the hardening request): the
    existence of *some* VERIFIED fact in the ledger is NOT sufficient to
    justify an ESTABLISHED-like root cause -- that VERIFIED fact might just
    be the OBSERVATION itself ("the check was not completed"), not evidence
    of WHY it happened. Only a VERIFIED *mechanism* (an action-level causal
    claim, e.g. "the audit trail confirms the user was never assigned the
    task") justifies root_cause claiming this level of certainty. A REPORTED
    mechanism (someone's account of what happened) never does.
    """
    warnings: list[str] = []
    if root_cause is None:
        return warnings

    status_value = getattr(root_cause.status, "value", root_cause.status)
    if status_value in _ESTABLISHED_LIKE_STATUSES:
        mechanism_is_verified = bool(mechanism and mechanism.status == "VERIFIED")
        if not mechanism_is_verified:
            warnings.append(
                f"Analytical Validator: root_cause.status={status_value} downgraded to STATED_UNVERIFIED "
                "— no VERIFIED causal mechanism (as opposed to a VERIFIED observation or a REPORTED "
                "account) supports this level of certainty."
            )
            root_cause.status = "STATED_UNVERIFIED"  # type: ignore[assignment]

    return warnings


# ---------------------------------------------------------------------------
# 3. 5-Why: detect a skipped, explicitly available causal fact
# ---------------------------------------------------------------------------


def five_why_skips_available_mechanism(five_why_steps: list, mechanism: MechanismInfo) -> bool:
    """True if the finding/evidence establishes an immediate mechanism
    (VERIFIED or REPORTED) but NONE of the 5-Why steps' answers actually
    reflect it -- i.e. the chain stopped (or never engaged) before an
    explicitly available causal fact, which the finding itself already
    resolves."""
    if not mechanism or mechanism.status not in ("VERIFIED", "REPORTED") or not mechanism.statement:
        return False
    if not five_why_steps:
        return True
    mechanism_words = significant_words(mechanism.statement)
    if not mechanism_words:
        return False
    for step in five_why_steps:
        answer_words = significant_words(step.answer or "")
        if not answer_words:
            continue
        overlap = mechanism_words & answer_words
        if len(overlap) >= max(2, len(mechanism_words) // 2):
            return False
    return True


def repair_five_why_with_mechanism(five_why_steps: list, mechanism: MechanismInfo, observed_deviation: str | None):
    """Deterministic structure repair (never invents content): if the chain
    skipped the mechanism entirely, insert it as the step right after the
    observation, using the mechanism's own already-extracted text and
    status -- not a fabricated explanation. Returns a NEW list of FiveWhyStep
    objects; caller is responsible for updating the FiveWhyAnalysis."""
    from app.models.agent import FiveWhyStep

    if not mechanism or not mechanism.statement:
        return five_why_steps

    mechanism_step = FiveWhyStep(
        question=f"Why did {observed_deviation or 'the observed condition'} occur?",
        answer=mechanism.statement,
        status="REPORTED" if mechanism.status == "REPORTED" else "VERIFIED",
    )

    if not five_why_steps:
        return [mechanism_step]

    # Insert right after the first step (the observation) if that step
    # doesn't already carry the mechanism's content, to preserve
    # Answer(N) explains Question(N-1) ordering.
    first = five_why_steps[0]
    if repeats_previous_why_answer(first.answer, mechanism.statement):
        return five_why_steps
    return [five_why_steps[0], mechanism_step, *five_why_steps[1:]]


# ---------------------------------------------------------------------------
# 4. Contributing factors: established vs. potential
# ---------------------------------------------------------------------------


def classify_contributing_factors(factors: list) -> tuple[list, list]:
    """Splits factors into (established, potential) based on their own
    evidence_status/status fields -- never reclassifies content, only
    groups it. A factor is "established" only if it is VERIFIED; everything
    else (including anything the model marked POSSIBLE_UNCONFIRMED) is
    potential, never presented as confirmed."""
    established = [f for f in factors if getattr(f.evidence_status, "value", f.evidence_status) == "VERIFIED" and f.status == "VERIFIED"]
    potential = [f for f in factors if f not in established and f.status != "REJECTED"]
    return established, potential


def classify_contributing_factors_full(factors: list, mechanism: MechanismInfo, verified_facts: list[str]) -> tuple[list, list, list]:
    """Three-way split (established, potential, rejected). A factor is
    REJECTED (mutates its status in place) if it structurally contradicts
    the established mechanism or a VERIFIED completion fact -- reusing the
    same contradiction detectors applied to hypotheses, since a
    contributing factor makes the same kind of causal claim. Never
    reclassifies a factor as MORE certain than it already was."""
    from app.agent.causal_guard import hypothesis_contradicts_mechanism, hypothesis_contradicts_verified_completion

    rejected = []
    survivors = []
    for f in factors:
        contradicts = (
            hypothesis_contradicts_mechanism(f.description, mechanism)
            or hypothesis_contradicts_verified_completion(f.description, verified_facts)
        )
        if contradicts:
            f.status = "REJECTED"  # type: ignore[assignment]
            rejected.append(f)
        else:
            survivors.append(f)

    established, potential = classify_contributing_factors(survivors)
    return established, potential, rejected


# ---------------------------------------------------------------------------
# 5. CAPA causal linkage
# ---------------------------------------------------------------------------


def conditional_action_has_causal_linkage(if_cause_confirmed: str, candidate_hypotheses: list) -> bool:
    """True if a conditional CAPA branch's condition text ('if_cause_confirmed')
    actually references one of the live candidate hypotheses (by id or by
    word overlap with its statement/name) rather than floating free of any
    stated cause."""
    if not candidate_hypotheses:
        # No hypotheses exist to link to -- can't require linkage that has
        # nothing to link to; this is not itself a violation.
        return True
    if not if_cause_confirmed:
        return False
    text_words = significant_words(if_cause_confirmed)
    for h in candidate_hypotheses:
        if h.id and h.id.lower() in if_cause_confirmed.lower():
            return True
        hyp_words = significant_words(h.statement) | significant_words(h.name.replace("_", " "))
        if text_words & hyp_words:
            return True
    return False


def validate_capa_causal_linkage(capa, candidate_hypotheses: list) -> list[str]:
    """Drops conditional CAPA actions that don't trace back to any live
    hypothesis. Mutates `capa` in place. Returns trace warnings."""
    warnings: list[str] = []
    if capa is None or not capa.conditional_actions:
        return warnings
    kept = []
    for action in capa.conditional_actions:
        if conditional_action_has_causal_linkage(action.if_cause_confirmed, candidate_hypotheses):
            kept.append(action)
        else:
            warnings.append(
                f"Analytical Validator: dropped conditional CAPA action — condition "
                f"{action.if_cause_confirmed!r} does not trace back to any candidate hypothesis"
            )
    capa.conditional_actions = kept
    return warnings


# ---------------------------------------------------------------------------
# 6. Analytical quality score (internal only -- never exposed as a claim of
#    truth, purely a signal for detecting weak synthesis / logging).
# ---------------------------------------------------------------------------


def compute_analytical_quality(root_cause, five_why, contributing_factors, capa, mechanism: MechanismInfo) -> dict[str, str]:
    """Returns a dict of HIGH/MEDIUM/LOW internal quality signals. This is
    NOT surfaced to the report as confidence -- it exists so trace/logs can
    flag a weak synthesis for review, without pretending to be a validated
    numeric score."""
    scores: dict[str, str] = {}

    # mechanism_accuracy: did we manage to extract an evidence-backed mechanism at all?
    scores["mechanism_accuracy"] = "HIGH" if mechanism and mechanism.status in ("VERIFIED", "REPORTED") else "LOW"

    # hypothesis_discrimination: do hypotheses carry discrimination_evidence?
    hyps = root_cause.candidate_hypotheses if root_cause else []
    if hyps:
        with_discrimination = sum(1 for h in hyps if h.discrimination_evidence)
        scores["hypothesis_discrimination"] = "HIGH" if with_discrimination == len(hyps) else ("MEDIUM" if with_discrimination else "LOW")
    else:
        scores["hypothesis_discrimination"] = "LOW"

    # five_why_coherence: has at least one step, and doesn't end mid-chain on a VERIFIED/SUPPORTED status
    # without an honest UNKNOWN boundary when the chain is incomplete.
    steps = five_why.steps if five_why else []
    if not steps:
        scores["five_why_coherence"] = "LOW"
    elif steps[-1].status in ("UNKNOWN", "NOT_ESTABLISHED") or (five_why and five_why.is_complete):
        scores["five_why_coherence"] = "HIGH"
    else:
        scores["five_why_coherence"] = "MEDIUM"

    # contributing_factor_quality: potential factors carry rationale/evidence_required
    if contributing_factors:
        well_formed = sum(1 for f in contributing_factors if f.rationale and f.evidence_required)
        scores["contributing_factor_quality"] = "HIGH" if well_formed == len(contributing_factors) else "MEDIUM"
    else:
        scores["contributing_factor_quality"] = "MEDIUM"  # empty is a valid, not a weak, outcome

    # capa_linkage: conditional actions exist and (post-validation) are all linked
    if capa and getattr(capa, "conditional_actions", None):
        scores["capa_linkage"] = "HIGH"
    elif capa and root_cause and root_cause.candidate_hypotheses:
        scores["capa_linkage"] = "LOW"  # hypotheses exist but no conditional actions map to them
    else:
        scores["capa_linkage"] = "MEDIUM"

    # uncertainty_discipline: NOT_ESTABLISHED/STATED_UNVERIFIED root cause with a populated
    # leading_hypothesis is the desired disciplined outcome; an ESTABLISHED-like status is
    # only HIGH-quality if evidence backs it (validate_root_cause_state already enforces this).
    status_value = getattr(root_cause.status, "value", root_cause.status) if root_cause else "NOT_ESTABLISHED"
    if status_value in ("NOT_ESTABLISHED", "STATED_UNVERIFIED", "INFERRED"):
        scores["uncertainty_discipline"] = "HIGH" if (root_cause and getattr(root_cause, "leading_hypothesis", None)) else "MEDIUM"
    else:
        scores["uncertainty_discipline"] = "MEDIUM"

    return scores


# ---------------------------------------------------------------------------
# 7. Impact field EXPLICIT / INFERRED / UNKNOWN classification
# ---------------------------------------------------------------------------

_IMPACT_FIELD_NAMES = ("affected_object", "affected_period", "process_at_risk", "relevant_change", "potential_effect")


def classify_impact_field_basis(value: str | None, finding_text: str) -> str:
    """Deterministic EXPLICIT/INFERRED/UNKNOWN classification for a single
    impact field, computed here (never asserted by the LLM, which can't be
    trusted to grade its own certainty):

    - UNKNOWN: empty, or itself says "not established"/"unknown"/etc.
    - EXPLICIT: most of the field's own vocabulary appears verbatim in the
      finding text -- i.e. it's a close restatement of something the
      finding actually said.
    - INFERRED: grounded (it already passed the entity/domain guards
      upstream) but synthesizes beyond a verbatim restatement -- e.g.
      "temperature monitoring" derived from "temperature log"."""
    if not value or not value.strip():
        return "UNKNOWN"
    if _UNKNOWN_VALUE_RE.match(value.strip()):
        return "UNKNOWN"

    value_words = significant_words(value)
    if not value_words:
        return "UNKNOWN"
    finding_words = significant_words(finding_text)
    overlap = value_words & finding_words
    ratio = len(overlap) / len(value_words)
    return "EXPLICIT" if ratio >= 0.7 else "INFERRED"


def compute_impact_field_basis(impact, finding_text: str) -> dict[str, str]:
    """Returns the {field_name: EXPLICIT|INFERRED|UNKNOWN} mapping for all
    of ImpactAssessment's named structured fields."""
    if impact is None:
        return {name: "UNKNOWN" for name in _IMPACT_FIELD_NAMES}
    return {
        name: classify_impact_field_basis(getattr(impact, name, None), finding_text)
        for name in _IMPACT_FIELD_NAMES
    }


# ---------------------------------------------------------------------------
# 8. Consolidated causal-graph edge validator
# ---------------------------------------------------------------------------
#
# Observation -> Immediate Mechanism -> Candidate Hypothesis -> Evidence ->
# Root Cause -> Corrective Action -> Effectiveness Check
#
# This is an AUDIT over content already produced by the checks above (and a
# couple of new leaf checks) -- it never repairs or invents anything itself;
# it exists so a single call can report every structural violation still
# present after the individual guards have already run, for observability
# and for tests that want one "is this analytically sound" answer.


def validate_causal_graph(root_cause, five_why, capa, mechanism: MechanismInfo | None) -> list[str]:
    """Returns every remaining structural violation across the causal chain.
    Does not mutate anything -- callers that want repairs already applied
    them via the more specific functions above (validate_root_cause_state,
    repair_five_why_with_mechanism, validate_capa_causal_linkage)."""
    violations: list[str] = []

    # Root Cause -> must have supporting evidence if it claims certainty.
    status_value = getattr(root_cause.status, "value", root_cause.status) if root_cause else "NOT_ESTABLISHED"
    if status_value in _ESTABLISHED_LIKE_STATUSES and root_cause and not root_cause.supporting_evidence:
        # Not fatal on its own (narrative may carry the evidence reference),
        # but worth surfacing: an ESTABLISHED-like claim with an empty
        # supporting_evidence list is a weak edge.
        violations.append("Root cause claims established-level certainty but supporting_evidence is empty")

    # Candidate Hypothesis -> Evidence: every hypothesis needs a rationale
    # and something identifying what would confirm/discriminate it.
    for h in (root_cause.candidate_hypotheses if root_cause else []):
        if not h.rationale:
            violations.append(f"Hypothesis {h.id} has no rationale")
        if not h.evidence_needed and not h.discrimination_evidence:
            violations.append(f"Hypothesis {h.id} has no evidence_needed/discrimination_evidence")

    # Corrective Action -> linked cause: every conditional CAPA action must
    # trace back to a hypothesis (already enforced/repaired by
    # validate_capa_causal_linkage upstream; this re-checks post-repair).
    if capa:
        for action in capa.conditional_actions:
            if not conditional_action_has_causal_linkage(action.if_cause_confirmed, root_cause.candidate_hypotheses if root_cause else []):
                violations.append(f"CAPA action {action.if_cause_confirmed!r} has no causal linkage")
            # Systemic action -> effectiveness check: a SYSTEMIC_ACTION
            # branch needs a way to verify it worked.
            if action.action_type == "SYSTEMIC_ACTION" and not action.verification_method:
                violations.append(f"Systemic CAPA action {action.if_cause_confirmed!r} has no verification_method")

    # 5-Why -> mechanism preservation (already repaired upstream; re-check).
    if five_why and mechanism and mechanism.statement:
        if five_why_skips_available_mechanism(five_why.steps, mechanism):
            violations.append("5-Why chain does not reflect the explicitly available mechanism")

    return violations
