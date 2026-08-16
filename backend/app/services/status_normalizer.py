"""Centralized status normalization boundary.

Guarantees that invalid or variant status strings produced by LLMs, heuristics,
or fallback paths are safely normalized before reaching Pydantic models.
Prevents any ValidationError or HTTP 500 crashes.
"""

from typing import Literal
from app.models.agent import RootCauseStatus, EvidenceStatus, CapaStatus, ImpactStatus

VALID_FIVE_WHY_STATUSES = {
    "VERIFIED",
    "SUPPORTED",
    "REPORTED",
    "REPORTED_STATEMENT",
    "REPORTED_UNVERIFIED",
    "MIXED",
    "INFERRED",
    "UNKNOWN",
    "REQUIRES_EVIDENCE",
    "NOT_ESTABLISHED",
}

VALID_HYPOTHESIS_STATUSES = {
    "POSSIBLE",
    "SUPPORTED",
    "REFUTED",
    "UNRESOLVED",
    "UNVERIFIED",
}


def normalize_five_why_status(
    raw_status: str | None,
) -> Literal[
    "VERIFIED",
    "SUPPORTED",
    "REPORTED",
    "REPORTED_STATEMENT",
    "REPORTED_UNVERIFIED",
    "MIXED",
    "INFERRED",
    "UNKNOWN",
    "REQUIRES_EVIDENCE",
    "NOT_ESTABLISHED",
]:
    """Normalize any raw status string into a valid FiveWhyStep.status literal."""
    if not raw_status:
        return "UNKNOWN"
    s = str(raw_status).strip().upper()
    if s in VALID_FIVE_WHY_STATUSES:
        return s  # type: ignore[return-value]
    # Mappings
    if s in ("CONFLICTING", "CONFLICT", "DISPUTED", "CONTRADICTORY", "CONFLICTING_REPORTS"):
        return "MIXED"
    if s in ("UNRESOLVED", "NOT_CONFIRMED", "UNCONFIRMED"):
        return "UNKNOWN"
    if s in ("FACT", "CONFIRMED", "OBSERVED"):
        return "VERIFIED"
    if s in ("STATED", "CLAIMED", "ALLEGED", "REPORTED_UNVERIFIED"):
        return "REPORTED"
    if s in ("PLAUSIBLE", "HYPOTHESIS", "PROBABLE"):
        return "SUPPORTED"
    if s in ("UNSUPPORTED", "MISSING_EVIDENCE", "NEEDS_EVIDENCE"):
        return "REQUIRES_EVIDENCE"
    return "UNKNOWN"


def normalize_hypothesis_status(
    raw_status: str | None,
) -> Literal["POSSIBLE", "SUPPORTED", "REFUTED", "UNRESOLVED", "UNVERIFIED"]:
    """Normalize any raw status string into a valid CandidateHypothesis.status literal."""
    if not raw_status:
        return "POSSIBLE"
    s = str(raw_status).strip().upper()
    if s in VALID_HYPOTHESIS_STATUSES:
        return s  # type: ignore[return-value]
    if s in ("DISPROVEN", "CONTRADICTED", "REJECTED", "FALSE"):
        return "REFUTED"
    if s in ("CONFIRMED", "PROVEN", "ESTABLISHED", "VERIFIED"):
        return "SUPPORTED"
    if s in ("CONFLICTING", "DISPUTED", "TIED"):
        return "UNRESOLVED"
    if s in ("UNCONFIRMED", "UNTESTED", "STATED_UNVERIFIED"):
        return "UNVERIFIED"
    return "POSSIBLE"


def normalize_root_cause_status(raw_status: str | None) -> RootCauseStatus:
    """Normalize any raw status into a valid RootCauseStatus enum."""
    if not raw_status:
        return RootCauseStatus.NOT_ESTABLISHED
    s = str(raw_status).strip().upper()
    try:
        return RootCauseStatus(s)
    except ValueError:
        pass
    if s in ("CONFIRMED", "PROVEN", "ESTABLISHED"):
        return RootCauseStatus.VERIFIED
    if s in ("PLAUSIBLE", "LIKELY", "PROBABLE"):
        return RootCauseStatus.SUPPORTED
    if s in ("DISPROVEN", "REFUTED"):
        return RootCauseStatus.CONTRADICTED
    if s in ("REPORTED", "CLAIMED", "STATED"):
        return RootCauseStatus.STATED_UNVERIFIED
    return RootCauseStatus.NOT_ESTABLISHED
