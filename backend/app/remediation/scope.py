"""Deterministic, finding-aware remediation SCOPE derivation.

This module answers question (A) from the remediation design principle --
"what activities may be required to address the finding" -- WITHOUT touching
pricing (B) or arithmetic (C). It is used:

  * as the fail-closed fallback when the LLM semantic interpreter is
    unavailable / returned nothing usable, and
  * to replace an LLM scope that is empty or is merely the prompt schema's
    example text echoed back by a weak model.

It is DOMAIN-AGNOSTIC and keyword-free:

  * It keys only on STRUCTURAL signals already established by the canonical
    semantic pipeline -- the finding's `semantic_type` (a closed enum), the
    grammatical class of the observed condition (omission vs. state, via the
    same deficiency regex the subject resolver uses), whether the finding is
    a recurrence, and whether the root cause is established.
  * Every activity is built from ROLE SLOTS (subject / condition / process),
    never from finding-specific vocabulary.
  * The five activity KINDS below apply to every audit finding; only the
    IMMEDIATE_CORRECTION wording and the presence of a retrospective-review
    activity vary with `semantic_type`.

Nothing here produces a number, a rate, a quantity, or a currency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# The observed condition denotes an OMITTED / OUTSTANDING deliverable that
# can, in principle, be produced ("not documented", "overdue", "incomplete",
# "not performed", "not evaluated", "missing", "not updated", ...). The verb
# set is the generic audit-deficiency grammar already used by the semantic
# subject resolver -- imported, not re-listed, so it never drifts.
try:  # pragma: no cover - import shape guard
    from app.services.semantic_subject import _DEFICIENCY_VERB as _DEF_VERB
except Exception:  # pragma: no cover
    _DEF_VERB = (
        r"document|record|log|eviden|complet|perform|conduct|execut|approv|"
        r"review|verif|authoriz|assess|updat|renew|calibrat|sign|obtain|"
        r"establish|maintain|retain|escalat|reconcil|validat|qualif"
    )
_OMISSION_CONDITION_RE = re.compile(
    rf"\b(?:not|never|no|without|yet\s+to\s+be|outstanding|overdue|pending|"
    rf"missing|absent|incomplete|lapsed|expired)\b"
    rf"|\b(?:{_DEF_VERB})\w*\b\s*$"
    rf"|^\s*(?:not\s+|un)(?:{_DEF_VERB})\w*",
    re.IGNORECASE,
)

# semantic_type values (from CanonicalFindingState / DeviationInfo) that mean
# "a required record / activity / control was not satisfied".
_OMISSION_TYPES = {"MISSING_RECORD", "OBSERVATION_VERIFICATION"}
_RECORD_TYPES = {"RECORD", "MISSING_RECORD", "DOCUMENT"}
_CONTROL_TYPES = {"CONTROL", "EVENT_SEQUENCE_CONTROL"}


@dataclass
class ScopeActivity:
    description: str
    kind: str            # IMMEDIATE_CORRECTION | SCOPE_ASSESSMENT | CAUSAL_INVESTIGATION | SYSTEMIC_STRENGTHENING | EFFECTIVENESS_VERIFICATION
    conditionality: str  # CONFIRMED | CONDITIONAL | INVESTIGATION_REQUIRED
    pricing_evidence_needed: str = ""


@dataclass
class RemediationScope:
    approach: str = ""
    activities: list[ScopeActivity] = field(default_factory=list)

    @property
    def activity_descriptions(self) -> list[str]:
        return [a.description for a in self.activities]

    @property
    def conditional_activity_descriptions(self) -> list[str]:
        """Activities whose necessity is contingent on confirming the cause."""
        return [a.description for a in self.activities if a.conditionality == "CONDITIONAL"]

    @property
    def evidence_needed(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for a in self.activities:
            e = a.pricing_evidence_needed.strip()
            if e and e.lower() not in seen:
                seen.add(e.lower())
                out.append(e)
        return out


def _cond_is_omission(condition: str | None, semantic_type: str | None) -> bool:
    if (semantic_type or "").upper() in _OMISSION_TYPES:
        return True
    if not condition:
        return False
    c = condition.strip().lower()
    if c in ("status unconfirmed", "condition unconfirmed", "unknown", ""):
        return False
    return bool(_OMISSION_CONDITION_RE.search(c))


# Finding types that denote a PHYSICAL asset / equipment (as opposed to a
# record, control, activity or process). Structural enum values, not
# vocabulary.
_PHYSICAL_ASSET_TYPES = {"EQUIPMENT", "PARAMETER"}
# Activity kinds that are ALWAYS internal analysis / review work -- their
# cost driver is personnel effort, so the missing basis is a rate + effort.
_ANALYSIS_KINDS = {"SCOPE_ASSESSMENT", "CAUSAL_INVESTIGATION", "EFFECTIVENESS_VERIFICATION"}


def _pricing_need(activity_desc: str, kind: str, semantic_type: str | None) -> str:
    """The pricing evidence that would establish THIS activity's monetary
    basis, chosen from the activity's STRUCTURAL role (its `kind` and the
    finding's `semantic_type`) -- never from finding vocabulary."""
    st = (semantic_type or "").upper()
    tail = f" — to cost “{activity_desc}”"
    if kind in _ANALYSIS_KINDS:
        return (
            "an approved internal rate for the personnel who would perform this review, "
            "together with an estimate of the effort required (hours or days)" + tail
        )
    if kind == "IMMEDIATE_CORRECTION" and st in _PHYSICAL_ASSET_TYPES:
        return (
            "a service or contractor quotation for the outstanding work, plus a "
            "materials / component quotation if any replacement is involved" + tail
        )
    if kind == "IMMEDIATE_CORRECTION" and st in _RECORD_TYPES:
        return (
            "an approved internal rate and an effort estimate for the personnel who "
            "would reconstruct or complete the record" + tail
        )
    if kind == "SYSTEMIC_STRENGTHENING":
        return (
            "the specific control change identified once the cause is confirmed, then "
            "a quotation or an effort estimate for that change (for example a procedure "
            "revision, a system change, an additional check, a training programme, "
            "tooling, or a supplier action)" + tail
        )
    # IMMEDIATE_CORRECTION of a control / process / activity, or unknown type.
    return (
        "the corrective work the assessment identifies, and for it either an approved "
        "internal rate with an effort estimate, or a supplier / service quotation if it "
        "would be outsourced" + tail
    )


def derive_remediation_scope(
    *,
    subject: str | None,
    condition: str | None,
    semantic_type: str | None,
    affected_process: str | None,
    root_cause_established: bool,
    is_recurring: bool = False,
    immediate_correction_hint: str | None = None,
) -> RemediationScope:
    """Build a finding-specific remediation scope from semantic roles."""
    _subj = (subject or "").strip()
    if not _subj or _subj.upper().startswith(("UNKNOWN", "UNRESOLVED", "NOT ESTABLISHED")):
        # No substantive subject -> no defensible finding-specific scope.
        return RemediationScope()
    subj = _subj[0].lower() + _subj[1:] if _subj[:2].isupper() is False else _subj
    _raw_cond = (condition or "").strip().lower()
    if _raw_cond in ("status unconfirmed", "condition unconfirmed", "unknown", ""):
        cond = "the observed deviation"
    elif re.match(r"^(?:not|never|no)\b", _raw_cond) or re.search(r"\w+(?:ed|ing)\b", _raw_cond):
        cond = _raw_cond  # verb / participial phrase reads naturally after "why ... was"
    else:
        cond = f"the {_raw_cond} condition"  # bare state adjective -> nominalise
    proc = (affected_process or "").strip()
    if not proc or proc.upper() in ("UNKNOWN", "NOT ESTABLISHED", ""):
        proc = f"the process and control governing {subj}"
    else:
        proc = proc[0].lower() + proc[1:]

    st = (semantic_type or "").upper()
    omission = _cond_is_omission(condition, semantic_type)

    def _mk(desc: str, kind: str, cond: str) -> ScopeActivity:
        return ScopeActivity(desc, kind, cond, _pricing_need(desc, kind, semantic_type))

    acts: list[ScopeActivity] = []

    _is_recurrence = is_recurring or st == "RECURRENCE"

    # 1. IMMEDIATE CORRECTION -- always CONFIRMED (independent of root cause).
    if immediate_correction_hint:
        _imm = immediate_correction_hint.strip()
    elif _is_recurrence:
        _imm = (
            f"Perform a retrospective review of {subj} across the affected population and "
            "determine the trending / escalation status"
        )
    elif omission and st in _RECORD_TYPES:
        _imm = (
            f"Reconstruct or complete {subj} for the affected period where this can be "
            "legitimately supported by objective evidence, or formally record why it cannot be reconstructed"
        )
    elif omission:
        _imm = (
            f"Carry out the outstanding {subj} now and confirm the current state "
            "against the applicable requirement"
        )
    elif st == "COMPARISON":
        _imm = (
            f"Independently re-verify {subj} against its reference value and correct the "
            "discrepant record"
        )
    else:
        _imm = (
            f"Assess {subj} against the applicable requirement and apply whatever correction "
            "that assessment shows to be necessary"
        )
    acts.append(_mk(_imm, "IMMEDIATE_CORRECTION", "CONFIRMED"))

    # 2. SCOPE ASSESSMENT -- how far the deviation extends.
    _scope = (
        f"Determine the extent of {cond} — other periods, records, units, transactions "
        f"or personnel potentially affected within {proc}"
    )
    acts.append(_mk(_scope, "SCOPE_ASSESSMENT", "CONFIRMED"))

    # 3. CAUSAL INVESTIGATION -- only when the cause is not yet established.
    if not root_cause_established:
        _inv = f"Investigate why {cond} occurred within {proc}"
        acts.append(_mk(_inv, "CAUSAL_INVESTIGATION", "INVESTIGATION_REQUIRED"))

    # 4. SYSTEMIC STRENGTHENING -- CONDITIONAL on confirming the cause.
    if root_cause_established:
        _sys = f"Strengthen {proc} so that {cond} is prevented or reliably detected"
        acts.append(_mk(_sys, "SYSTEMIC_STRENGTHENING", "CONFIRMED"))
    else:
        _sys = (
            f"Subject to confirming the underlying cause, strengthen {proc} so that {cond} "
            "is prevented or reliably detected"
        )
        acts.append(_mk(_sys, "SYSTEMIC_STRENGTHENING", "CONDITIONAL"))

    # 5. EFFECTIVENESS VERIFICATION -- always.
    _eff = "Verify that the completed remediation is effective before the finding is closed"
    acts.append(_mk(_eff, "EFFECTIVENESS_VERIFICATION", "CONFIRMED"))

    # Approach sentence -- finding-specific, causally cautious.
    if root_cause_established:
        approach = (
            f"Address {cond} affecting {subj}: correct the specific instance, establish its "
            f"extent, and strengthen {proc}."
        )
    else:
        approach = (
            f"Potential implementation approach, subject to confirming the underlying cause: "
            f"correct the specific instance of {cond} affecting {subj}, establish its extent, "
            f"investigate the cause, and — once the cause is confirmed — strengthen {proc}."
        )

    return RemediationScope(approach=approach, activities=acts)


# --- prompt-echo detection -------------------------------------------------

# Distinctive prose from the schema example the prompt used to carry. A weak
# model that parrots the example instead of reasoning reproduces these
# near-verbatim. This is NOT a finding blacklist -- it guards against the
# model copying OUR OWN prompt. The generic "labour rate / hours" pair is
# deliberately excluded: that can be a legitimate answer for a labour
# activity, so it alone never marks an interpretation as an echo.
_PROMPT_ECHO_MARKERS = (
    "define and perform the missing verification",
    "procedure definition + execution",
    "draft the procedure",
    "procedure drafting effort",
)


def looks_like_prompt_echo(
    strategy_summary: str | None,
    activity_descriptions: list[str] | None,
    evidence_improves: list[str] | None,
) -> bool:
    blob = " ".join(
        [strategy_summary or ""]
        + list(activity_descriptions or [])
        + list(evidence_improves or [])
    ).lower().strip()
    if not blob:
        return True
    return any(m in blob for m in _PROMPT_ECHO_MARKERS)
