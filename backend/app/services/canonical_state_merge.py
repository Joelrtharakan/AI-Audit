"""Merge the validated LLM `CanonicalFindingContext` into the deterministic
`CanonicalFindingState` -- making the LLM the PRIMARY semantic interpreter
while `resolve_deviation` stays the fail-closed FLOOR.

Contract (spec Phase 4/5):
  1. LLM value present AND passes validation  -> use the LLM value.
  2. LLM value present but fails validation    -> reject it, keep deterministic.
  3. LLM omitted the field                     -> keep the deterministic value.
  4. LLM + deterministic conflict on a value   -> keep deterministic, record it.
  5. Deterministic extracted material info the LLM dropped
     (measurement / comparison / recurrence / stated alternatives)
     -> NEVER let the omission erase it; the deterministic value survives.
  6. `semantic_context is None`                -> pure no-op (the entire
     deterministic test baseline runs through here unchanged).

This module does DATA MERGING ONLY -- no LLM calls, no arithmetic, no
re-parsing of raw text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.canonical_semantic_models import CanonicalFindingContext

# Evidence-source / reporting nouns that must never be the finding subject
# (structural class, mirrors semantic_subject._EVIDENCE_SOURCE_NOUNS intent).
_EVIDENCE_SOURCE_HEAD_RE = re.compile(
    r"\b(?:records?|logs?|log|documentation|documents?|register|registers?|"
    r"audit|audits?|report|reports?|review|reviews?|inspection|inspections?|"
    r"assessment|assessments?|evidence|data|trail|history|file|files?|"
    r"observation|observations?|finding|findings?|walkthrough|survey|surveys?|"
    r"reconciliation)\s*$",
    re.IGNORECASE,
)
# A bare state/condition word is not a subject.
_BARE_STATE_RE = re.compile(
    r"^(?:excessive|excess|inadequate|insufficient|incomplete|missing|absent|"
    r"unavailable|overdue|expired|lapsed|weak|poor|inconsistent|non-?compliant|"
    r"unauthori[sz]ed|invalid|unverified|unknown|unclear|status|condition)$",
    re.IGNORECASE,
)
# Causal / analytical role phrases that must never be the subject.
_CAUSAL_ROLE_RE = re.compile(
    r"\b(?:root\s+cause|assignable\s+cause|underlying\s+cause|the\s+cause|"
    r"a\s+cause|causes?|mechanism|failure\s+mode|reason|contributing\s+factor|"
    r"provisioning\s+error|data-?entry\s+error|human\s+error)\b",
    re.IGNORECASE,
)

_DIR_TO_COMPARISON_TYPE = {
    "ABOVE": "EXCEEDED",
    "BELOW": "BELOW",
    "MISMATCH": "MISMATCH",
    "UNKNOWN": "MISMATCH",
}


@dataclass
class MergeOutcome:
    fields_from_llm: list[str]
    fields_rejected: list[str]
    fields_conserved: list[str]   # deterministic value kept because LLM dropped material info
    disagreements: list[str]


def _sig(s: str | None) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", (s or "").lower())}


def _subject_is_safe(subject: str | None, ctx: CanonicalFindingContext) -> bool:
    """Reject an LLM subject that is a causal mechanism, an evidence source,
    a bare state word, or one of the finding's own stated alternatives."""
    s = (subject or "").strip()
    if not s or len(s) < 3:
        return False
    low = s.lower().rstrip(".")
    if _BARE_STATE_RE.match(low) or _CAUSAL_ROLE_RE.search(low):
        return False
    # A candidate that is a CLAUSE / predication ("<entity> remains open",
    # "<entity> lacks X", "<reported speech>") -- not an entity noun phrase.
    # The one authoritative structural gate, shared pipeline-wide.
    try:
        from app.services.semantic_subject import reject_subject_if_clause
        if reject_subject_if_clause(s):
            return False
    except Exception:  # pragma: no cover - never let the gate crash the merge
        pass
    if _EVIDENCE_SOURCE_HEAD_RE.search(low) and len(low.split()) <= 2:
        return False
    # never one of the enumerated causes
    _sw = _sig(low)
    for alt in ctx.stated_causal_alternatives or []:
        at = _sig(alt)
        if at and at <= _sw:
            return False
    # the LLM's own kind classification
    for ent in ctx.entities or []:
        if _sig(ent.name) == _sw and ent.kind in ("CAUSE", "HYPOTHESIS", "STATE"):
            return False
    return True


def _has_det_comparison(cs) -> bool:
    return bool(getattr(cs, "comparison_type", None) or getattr(cs, "comparison_left", None))


def _has_det_recurrence(cs) -> bool:
    return getattr(cs, "recurrence_count", None) is not None


def _has_det_measurement(cs) -> bool:
    return getattr(cs, "measurement", None) is not None


def merge_semantic_context_into_canonical(canonical_state, semantic_context):
    """Return (canonical_state, MergeOutcome). `canonical_state` is mutated in
    place. No-op when `semantic_context` is None."""
    out = MergeOutcome([], [], [], [])
    ctx = semantic_context
    cs = canonical_state
    if ctx is None or cs is None:
        return cs, out

    # ---- SUBJECT -------------------------------------------------------
    # LLM-PRIMARY: a SAFE LLM subject wins. The deterministic value is used
    # only when the LLM omitted it or returned an unsafe one.
    llm_subj = (getattr(ctx, "finding_subject", None) or "").strip() or None
    det_subj = getattr(cs, "finding_subject", None) or ""
    _llm_safe = bool(llm_subj and _subject_is_safe(llm_subj, ctx))
    if _llm_safe:
        if _sig(llm_subj) != _sig(det_subj):
            if _is_established(det_subj):
                out.disagreements.append(f"finding_subject: det={det_subj!r} llm={llm_subj!r}")
            cs.finding_subject = llm_subj
            cs.affected_object = llm_subj[:1].upper() + llm_subj[1:]
            if getattr(cs, "subject_unresolved", False):
                cs.subject_unresolved = False
            out.fields_from_llm.append("finding_subject")
    else:
        if llm_subj:
            out.fields_rejected.append("finding_subject")
        # Neither the LLM nor the floor gave a SAFE subject -> fail closed
        # rather than keep a causal/evidence-source phrase as the subject.
        if det_subj and not _subject_is_safe(det_subj, ctx):
            from app.services.semantic_subject import UNRESOLVED_SUBJECT_DISPLAY
            cs.finding_subject = UNRESOLVED_SUBJECT_DISPLAY
            cs.affected_object = UNRESOLVED_SUBJECT_DISPLAY
            cs.subject_unresolved = True
            out.fields_rejected.append("finding_subject(floor)")

    # ---- EVIDENCE SOURCE / REPORTED OBSERVATION ----------------------
    if getattr(ctx, "evidence_source", None) and not getattr(cs, "evidence_source_phrase", None):
        # informational only -- CanonicalFindingState has no dedicated field;
        # keep it on the context (already there) for downstream provenance.
        pass

    # ---- OBSERVED CONDITION -----------------------------------------
    llm_cond = (getattr(ctx, "observed_condition", None) or "").strip() or None
    _det_cond = (getattr(cs, "deviation_condition", None) or "").strip().lower()
    if llm_cond and _det_cond in ("", "status unconfirmed", "condition unconfirmed", "unknown", "not noted"):
        cs.deviation_condition = llm_cond
        out.fields_from_llm.append("deviation_condition")

    # ---- COMPARISON ------------------------------------------------
    cmp_ = getattr(ctx, "comparison", None)
    if cmp_ is not None and (cmp_.magnitude is not None or cmp_.left or cmp_.right):
        if not _has_det_comparison(cs):
            cs.comparison_type = _DIR_TO_COMPARISON_TYPE.get(cmp_.direction or "UNKNOWN", "MISMATCH")
            cs.comparison_left = cmp_.left or getattr(cs, "finding_subject", None)
            cs.comparison_right = cmp_.right or cmp_.reference
            cs.comparison_basis = cmp_.reference or cmp_.right
            if cs.semantic_type in (None, "OBJECT", "ACTIVITY", "NON_ACTIONABLE"):
                cs.semantic_type = "COMPARISON"
            if cmp_.magnitude is not None and not _has_det_measurement(cs):
                from app.models.agent import SemanticMeasurement
                cs.measurement = SemanticMeasurement(
                    value=float(cmp_.magnitude), unit=cmp_.unit,
                    role="OBSERVED_DISCREPANCY", evidence_status="UNKNOWN",
                )
            # the deviation text must READ as a comparison (INV-SEMANTIC-002)
            _ref = cs.comparison_right or cs.comparison_basis or "the reference value"
            _mag = ""
            if cmp_.magnitude is not None:
                _u = "%" if cmp_.unit == "%" else (f" {cmp_.unit}" if cmp_.unit else "")
                _mag = f"{cmp_.magnitude:g}{_u}"
            _dir = {"BELOW": "below", "EXCEEDED": "above"}.get(cs.comparison_type, "")
            if _dir and _mag:
                _cond = f"{'shortfall' if _dir == 'below' else 'excess'} of {_mag} {_dir} {_ref}"
            elif _mag:
                _cond = f"differed from {_ref} by {_mag}"
            else:
                _cond = f"did not match {_ref}"
            cs.deviation_condition = _cond
            _subj = getattr(cs, "finding_subject", None) or "the finding subject"
            cs.observed_deviation = f"{_subj} — {_cond}"
            cs.deviation = cs.observed_deviation
            out.fields_from_llm.append("comparison")
        elif cmp_.magnitude is not None and not _has_det_measurement(cs):
            from app.models.agent import SemanticMeasurement
            cs.measurement = SemanticMeasurement(
                value=float(cmp_.magnitude), unit=cmp_.unit,
                role="OBSERVED_DISCREPANCY", evidence_status="UNKNOWN",
            )
            out.fields_from_llm.append("comparison.magnitude")
    elif _has_det_comparison(cs):
        out.fields_conserved.append("comparison")

    # ---- RECURRENCE ---------------------------------------------
    rec = getattr(ctx, "recurrence", None)
    if rec is not None and rec.count is not None:
        if not _has_det_recurrence(cs):
            cs.recurrence_count = int(rec.count)
            cs.recurrence_event = rec.event
            cs.recurrence_period = rec.period
            if not getattr(cs, "occurrence_population", None) and rec.event:
                cs.occurrence_population = f"{rec.count} {rec.event}"
            if rec.period and getattr(cs, "affected_period", "UNKNOWN") in (None, "", "UNKNOWN"):
                cs.affected_period = rec.period
            if cs.semantic_type in (None, "OBJECT", "ACTIVITY", "NON_ACTIONABLE"):
                cs.semantic_type = "RECURRENCE"
            out.fields_from_llm.append("recurrence")
    elif _has_det_recurrence(cs):
        out.fields_conserved.append("recurrence")

    # ---- STATED CAUSAL ALTERNATIVES (never erased -- union) ---------
    det_alts = list(getattr(cs, "stated_causal_alternatives", []) or [])
    llm_alts = list(getattr(ctx, "stated_causal_alternatives", []) or [])
    merged_alts: list[str] = []
    _seen: set[str] = set()
    for a in det_alts + llm_alts:
        k = re.sub(r"\s+", " ", a.strip().lower())
        if k and k not in _seen:
            _seen.add(k)
            merged_alts.append(a.strip())
    if merged_alts and merged_alts != det_alts:
        cs.stated_causal_alternatives = merged_alts
        out.fields_from_llm.append("stated_causal_alternatives")
    if merged_alts:
        cs.causal_alternatives_unresolved = bool(
            getattr(cs, "causal_alternatives_unresolved", False)
            or getattr(ctx, "causal_alternatives_unresolved", False)
        )
    if det_alts and len(llm_alts) < len(det_alts):
        out.fields_conserved.append("stated_causal_alternatives")

    # ---- MISSING RECORD STATUS ------------------------------------
    mrs = getattr(ctx, "missing_record_status", None)
    if mrs and mrs not in ("UNKNOWN", "RECORD_EXISTS"):
        # ACTIVITY_NOT_PERFORMED requires explicit support -- the validator
        # already downgrades an unsupported one; trust the validated value.
        cs.missing_record_status = mrs
        cs.activity_performance_ambiguity = bool(
            getattr(ctx, "activity_performance_ambiguity", False)
        )
        if cs.semantic_type in (None, "OBJECT", "ACTIVITY", "NON_ACTIONABLE", "RECORD"):
            cs.semantic_type = "MISSING_RECORD"
        out.fields_from_llm.append("missing_record_status")
    elif getattr(ctx, "activity_performance_ambiguity", False):
        cs.activity_performance_ambiguity = True

    # ---- EVIDENCE-PROPOSITION PROVENANCE --------------------------
    if getattr(ctx, "evidence_source", None) and not getattr(cs, "evidence_source", None):
        cs.evidence_source = ctx.evidence_source
        out.fields_from_llm.append("evidence_source")
    if getattr(ctx, "reported_observation", None) and not getattr(cs, "reported_observation", None):
        cs.reported_observation = ctx.reported_observation
        out.fields_from_llm.append("reported_observation")
    if getattr(ctx, "epistemic_status", None) and not getattr(cs, "finding_epistemic_status", None):
        cs.finding_epistemic_status = ctx.epistemic_status
        out.fields_from_llm.append("finding_epistemic_status")

    # ---- AFFECTED PERIOD / SCOPE --------------------------------
    if getattr(ctx, "affected_period", None) and getattr(cs, "affected_period", "UNKNOWN") in (None, "", "UNKNOWN"):
        cs.affected_period = ctx.affected_period
        out.fields_from_llm.append("affected_period")
    if getattr(ctx, "scope", None) and not getattr(cs, "occurrence_population", None):
        cs.occurrence_population = ctx.scope
        out.fields_from_llm.append("scope")

    # ---- AFFECTED PROCESS (spec §6/§15) ------------------------
    # The canonical LLM is authoritative for whether a process is established.
    # If it named one, use it; if it did NOT, the deterministic resolver's
    # generic "<subject> operational process" guess must NOT stand -- clear it
    # so downstream renders NOT_ESTABLISHED rather than a fabrication.
    _llm_proc = (getattr(ctx, "affected_process", None) or "").strip()
    if _llm_proc:
        cs.affected_process = _llm_proc
        out.fields_from_llm.append("affected_process")
    elif getattr(cs, "affected_process", None) not in (None, "", "UNKNOWN", "NOT_ESTABLISHED", "NOT ESTABLISHED"):
        out.fields_rejected.append("affected_process(deterministic-guess)")
        cs.affected_process = "UNKNOWN"

    return cs, out


def _is_established(subject: str | None) -> bool:
    try:
        from app.services.semantic_subject import is_established_subject
        return bool(is_established_subject(subject))
    except Exception:  # pragma: no cover
        s = (subject or "").strip()
        return bool(s) and not s.upper().startswith(("UNKNOWN", "UNRESOLVED", "NOT ESTABLISHED"))
