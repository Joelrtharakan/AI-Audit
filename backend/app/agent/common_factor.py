"""Common-factor / pattern-based hypothesis leads.

A common factor shared across multiple observations (same system, process,
procedure, equipment, location, role, vendor, control, department, time
period, or failure mode) does NOT prove causality -- it creates a candidate
INVESTIGATION LEAD: a hypothesis that must stay at status=POSSIBLE,
evidence_strength=INDICATIVE until objective evidence establishes the
mechanism. Deliberately conservative: only fires when the evidence contains
BOTH a multi-department/multi-location pattern AND an explicit statement
that those locations share a system/process/vendor/control -- a single
signal alone is not a strong enough investigation lead to manufacture a
hypothesis from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_DEPARTMENT_RE = re.compile(
    r"\b(?:production|warehouse|quality\s+control|qc|packaging|maintenance|"
    r"engineering|laboratory|lab|shipping|receiving|purchasing|logistics|"
    r"assembly|distribution|manufacturing)\b",
    re.IGNORECASE,
)

# "all/both/each ... use/share/rely on the same <factor>" -- the explicit
# common-factor assertion (Section 1: "same system/process/equipment/vendor/
# control"). Deliberately requires the explicit "all X use/share the same Y"
# sharing-verb shape rather than a bare "same <noun>" match anywhere in the
# text: a finding's own subject is often independently described with "the
# same <document/procedure>" (e.g. "outdated versions of the SAME
# controlled procedure were found...") without that being the CAUSAL common
# factor -- only an explicit shared-usage assertion is.
_SHARED_FACTOR_RE = re.compile(
    r"\b(?:all|both|each)\b.{0,60}?\b(?:use|uses|share|shares|rely\s+on|relies\s+on|"
    r"utilize|utilizes|are\s+served\s+by|is\s+served\s+by|follow|follows|"
    r"are\s+(?:supervised|managed|overseen|controlled)\s+by|"
    r"is\s+(?:supervised|managed|overseen|controlled)\s+by|"
    r"report\s+to|reports\s+to|are\s+supplied\s+by|is\s+supplied\s+by)\b.{0,20}?"
    r"\bsame\s+(?P<factor>[a-z][a-z\s-]*?)\s*(?:[.,;]|$)",
    re.IGNORECASE,
)


@dataclass
class CommonFactorResult:
    detected: bool = False
    factor: str | None = None  # e.g. "document-control system"
    departments: list[str] = field(default_factory=list)
    multi_department: bool = False
    supporting_claim_ids: list[str] = field(default_factory=list)
    supporting_texts: list[str] = field(default_factory=list)


def detect_common_factor(evidence_ledger: list[Any] | None) -> CommonFactorResult:
    """Deterministic, conservative common-factor pattern detection across
    the VERIFIED claims in the evidence ledger. Claim IDs follow the same
    positional "C<index+1>" convention as
    app.agent.causal_model.claims_from_evidence_ledger (EvidenceItem itself
    carries no claim_id field -- ids are assigned by ledger position)."""
    result = CommonFactorResult()
    indexed_claims = [
        (i + 1, e) for i, e in enumerate(evidence_ledger or [])
        if str(getattr(e, "status", "")).upper().endswith("VERIFIED") and getattr(e, "claim", None)
    ]
    if len(indexed_claims) < 2:
        return result

    claims = [e for _, e in indexed_claims]
    full_text = " ".join(c.claim for c in claims)

    departments = sorted({m.group(0).strip().title() for m in _DEPARTMENT_RE.finditer(full_text)})
    if len(departments) < 2:
        return result

    factor_match = _SHARED_FACTOR_RE.search(full_text)
    if not factor_match:
        return result

    result.detected = True
    result.factor = re.sub(r"\s+", " ", factor_match.group("factor").strip())
    result.departments = departments
    result.multi_department = True
    result.supporting_claim_ids = [f"C{i}" for i, _ in indexed_claims]
    result.supporting_texts = [c.claim for c in claims]
    return result


def build_common_factor_hypothesis(result: CommonFactorResult, hyp_id: str = "H1"):
    """Build the POSSIBLE/UNVERIFIED, evidence_strength=INDICATIVE candidate
    hypothesis from a detected common factor. Never SUPPORTED/ESTABLISHED --
    that requires objective evidence of the actual mechanism, which a common
    factor alone never provides (Section 1/2: pattern != cause)."""
    from app.models.agent import CandidateHypothesis

    factor = result.factor or "shared system"
    scope = ", ".join(result.departments[:-1]) + f", and {result.departments[-1]}" if len(result.departments) > 1 else (result.departments[0] if result.departments else "the affected locations")

    return CandidateHypothesis(
        id=hyp_id,
        name="SHARED_SYSTEM_COMMON_FACTOR",
        statement=(
            f"Failure in the shared {factor}'s distribution or obsolete-copy withdrawal control "
            f"may have allowed the outdated version to remain in use across {scope}."
        ),
        status="POSSIBLE",
        evidence_needed=f"{factor.capitalize()} distribution logs, revision/version-control records, and obsolete-copy withdrawal records",
        confirms_if=f"Distribution or withdrawal-control records confirm a failure in the shared {factor}",
        refutes_if=f"Distribution and withdrawal-control records confirm the {factor} operated correctly across all locations",
        discrimination_evidence=f"H{hyp_id[1:]} strengthens if the {factor}'s distribution or withdrawal-control records show a failure common to all affected locations.",
        supporting_evidence=result.supporting_texts,
        supporting_claim_ids=result.supporting_claim_ids,
        evidence_strength="INDICATIVE",
        relevance_rank="HIGH",
        causal_role="PRIMARY_CAUSE",
        rationale=(
            f"A common factor (shared {factor}) across {len(result.departments)} independently-affected "
            "locations is a strong investigation lead, but a common factor alone does not prove causality — "
            "objective evidence of the actual mechanism is required before this can be promoted beyond POSSIBLE."
        ),
    )
