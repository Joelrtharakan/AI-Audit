"""Programmatic confidence scoring (spec section 17).

The LLM never decides its own confidence -- that invites it to rationalize a
high score for a weak analysis. Confidence is instead computed here from
objective signals already present in the request/response: how much detail
the auditor provided, whether root cause and CAPA could actually be
established, and how many open questions remain.
"""

from dataclasses import dataclass

from app.models.analysis import CapaStatus, ConfidenceLevel, ObservationQualityStatus, RootCauseStatus

# Word-count proxy for "auditor supplied enough detail to reason about" --
# calibrated against the spec's example observations (a one-line vague finding
# is ~6 words; a well-described one is 20+).
_DETAILED_OBSERVATION_WORD_COUNT = 15

_HIGH_THRESHOLD = 4.0
_MEDIUM_THRESHOLD = 1.5


@dataclass
class ConfidenceInputs:
    finding_text: str
    observation_quality: ObservationQualityStatus
    root_cause_status: RootCauseStatus
    capa_status: CapaStatus
    missing_information_count: int
    open_questions_count: int
    five_why_steps: int


# Root-cause-status contribution: real corroboration scores highest, a bare
# self-reported/asserted cause scores positively but well below corroborated (enough to
# reach MEDIUM together with other signals, never enough alone to reach HIGH -- HIGH is
# reserved for independently corroborated causes), and no causal signal at all is
# penalized same as before.
_ROOT_CAUSE_STATUS_SCORE = {
    RootCauseStatus.ESTABLISHED: 2.0,
    RootCauseStatus.SELF_REPORTED: 0.75,
    RootCauseStatus.NOT_ESTABLISHED: -1.0,
}


def calculate_confidence(inputs: ConfidenceInputs) -> ConfidenceLevel:
    score = 0.0

    # observation_quality and root_cause_status are correlated, not independent: the
    # quality check is a cheap upfront heuristic, while root_cause_status comes from the
    # extraction+classification steps actually finding (or not finding) real signal in
    # the text. When those two disagree -- quality says INSUFFICIENT but classification
    # still found an attributed/corroborated cause -- trust the step that did real work
    # and treat quality as neutral rather than applying the full penalty twice. Only
    # double-penalize when both agree there's nothing here (INSUFFICIENT AND
    # NOT_ESTABLISHED). Without this, a quality-check false negative alone can collapse
    # every case to LOW regardless of how well-supported the actual classification is.
    if inputs.observation_quality == ObservationQualityStatus.SUFFICIENT:
        score += 2.0
    elif inputs.root_cause_status == RootCauseStatus.NOT_ESTABLISHED:
        score -= 1.0
    else:
        score += 0.0

    score += _ROOT_CAUSE_STATUS_SCORE[inputs.root_cause_status]
    score += 1.0 if inputs.capa_status == CapaStatus.AI_SUGGESTED else 0.0

    word_count = len(inputs.finding_text.split())
    score += 1.0 if word_count >= _DETAILED_OBSERVATION_WORD_COUNT else -1.0

    score += 1.0 if inputs.five_why_steps >= 3 else 0.0

    # missing_information_count and open_questions_count are both "thinness" signals
    # from the SAME underlying situation the quality-check/root-cause-status disagreement
    # above already accounts for: they're near-guaranteed to be non-zero for
    # SELF_REPORTED (the generation prompt requires a verification checklist there) and
    # are often stale leftovers from a quality check that classification then overrode.
    # Only apply them when root cause is genuinely NOT_ESTABLISHED -- that's the one case
    # where "lots of open questions" really does mean "this is thin," rather than
    # penalizing the very verification checklist a SELF_REPORTED/ESTABLISHED result is
    # supposed to produce.
    if inputs.root_cause_status == RootCauseStatus.NOT_ESTABLISHED:
        score -= min(inputs.missing_information_count, 3) * 0.5
        score -= min(inputs.open_questions_count, 5) * 0.2

    if score >= _HIGH_THRESHOLD:
        level = ConfidenceLevel.HIGH
    elif score >= _MEDIUM_THRESHOLD:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW

    # HIGH is reserved for a truly ESTABLISHED (corroborated) root cause -- a
    # self-reported/asserted-but-unverified cause can score well (MEDIUM) but must never
    # round up to HIGH just by accumulating enough of the other signals.
    if level == ConfidenceLevel.HIGH and inputs.root_cause_status != RootCauseStatus.ESTABLISHED:
        level = ConfidenceLevel.MEDIUM

    return level
