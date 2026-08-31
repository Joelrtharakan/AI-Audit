"""Project the validated LLM `CanonicalFindingContext` into the shapes the
`CanonicalFindingState` constructor consumes -- the canonical-SUCCESS
semantic source.

Post-Pass-41/42/43/44 this module is NOT a "merge". On canonical success the
semantic state is built ENTIRELY from the LLM context via:
  * `deviation_info_from_canonical(ctx)`   -> DeviationInfo (subject / condition
                                             / process / period / scope /
                                             entities / active comparison /
                                             recurrence)
  * `recurrence_info_from_canonical(ctx)`  -> RecurrenceInfo (count / event /
                                             period / explicit_previous_capa)
  * `mechanism_info_from_canonical(ctx)`   -> MechanismInfo (a causal_claim the
                                             LLM marked is_causal, else UNKNOWN)
  * `apply_canonical_provenance_fields(cs, ctx)` -> copies the remaining
                                             context-only provenance fields
                                             (evidence_source /
                                             reported_observation /
                                             finding_epistemic_status /
                                             affected_period / scope) + a
                                             structural subject-safety re-check.

Every one of these is a straight structural projection: NO regex, NO keyword
logic, NO raw-finding-text inspection, NO deterministic semantic inference.
An LLM field the model did not establish stays at its non-activating
sentinel. On the FALLBACK path (ctx is None) the projections return None and
`apply_canonical_provenance_fields` is a no-op -- the deterministic resolver
owns semantics there.
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


def deviation_info_from_canonical(ctx):
    """Spec Pass 41 §3/§7: build a `DeviationInfo`-shaped object PURELY from the
    validated canonical LLM context, so the canonical-SUCCESS path in
    `understand_finding_node` never calls `resolve_deviation`. Every field the
    LLM did not establish stays at its `DeviationInfo` default (the
    non-activating sentinel). No raw-finding-text inspection, no keyword logic,
    no regex -- a straight structural projection of the LLM's own fields.
    Returns None if `ctx` is None (caller then uses the deterministic
    fallback)."""
    if ctx is None:
        return None
    from app.services.canonical_semantic_models import comparison_is_active
    from app.services.semantic_subject import DeviationInfo

    subj = (getattr(ctx, "finding_subject", None) or "").strip() or None
    cond = (getattr(ctx, "observed_condition", None) or "").strip() or None
    proc = (getattr(ctx, "affected_process", None) or "").strip() or None
    period = (getattr(ctx, "affected_period", None) or "").strip() or None
    scope = (getattr(ctx, "scope", None) or "").strip() or None
    ents = [e.name for e in (getattr(ctx, "entities", None) or []) if getattr(e, "name", None)]

    di = DeviationInfo(
        subject=subj,
        finding_subject=subj,
        affected_object=subj,
        affected_process=proc,
        condition=cond,
        deviation=cond,
        date=period,
        entities=ents,
        matched=subj is not None,
        # Spec Pass 42 §11: no implicit semantic classification. `semantic_type`
        # is set below ONLY when the LLM explicitly establishes a COMPARISON or
        # RECURRENCE; otherwise it stays None (NOT_ESTABLISHED). "OBJECT" was a
        # silent default and is gone.
        semantic_type=None,
        occurrence_population=scope,
        extraction_confidence="RESOLVED" if subj else "UNRESOLVED",
        subject_unresolved=subj is None,
        requirement_status="UNKNOWN",
    )

    # RECURRENCE -- only what the LLM stated.
    rec = getattr(ctx, "recurrence", None)
    if rec is not None and getattr(rec, "count", None) is not None:
        di.recurrence_count = int(rec.count)
        di.recurrence_event = getattr(rec, "event", None)
        di.recurrence_period = getattr(rec, "period", None)
        di.semantic_type = "RECURRENCE"

    # COMPARISON -- only when the LLM explicitly ACTIVATED it.
    cmp_ = getattr(ctx, "comparison", None)
    if comparison_is_active(cmp_):
        di.comparison_type = _DIR_TO_COMPARISON_TYPE.get(
            getattr(cmp_, "direction", None) or "UNKNOWN", "MISMATCH"
        )
        di.comparison_left = getattr(cmp_, "left", None) or subj
        di.comparison_right = getattr(cmp_, "right", None) or getattr(cmp_, "reference", None)
        di.comparison_basis = getattr(cmp_, "reference", None) or getattr(cmp_, "right", None)
        _mag = getattr(cmp_, "magnitude", None)
        if _mag is not None:
            di.measurement_value = float(_mag)
            di.measurement_unit = getattr(cmp_, "unit", None)
        di.semantic_type = "COMPARISON"

    return di


def recurrence_info_from_canonical(ctx):
    """Spec Pass 42 §4/§5: build a `RecurrenceInfo`-shaped object PURELY from
    the validated canonical LLM context, so the canonical-SUCCESS path never
    calls `recurrence_guard.detect_recurrence(finding_text)`. Recurrence and
    previous-CAPA reference are LLM-owned: `ctx.recurrence` (explicit stated
    count) + `ctx.explicit_previous_capa_reference`. `previous_capa_status` /
    `previous_capa_effectiveness` have no generic canonical representation ->
    they stay at the schema default (NOT_ESTABLISHED). No raw-text inspection,
    no regex, no keyword list. Returns None when `ctx` is None."""
    if ctx is None:
        return None
    from app.agent.recurrence_guard import RecurrenceInfo

    rec = getattr(ctx, "recurrence", None)
    count = getattr(rec, "count", None) if rec is not None else None
    prev_capa = bool(getattr(ctx, "explicit_previous_capa_reference", False))
    is_recurring = bool(count is not None or prev_capa)
    return RecurrenceInfo(
        is_recurring=is_recurring,
        has_previous_capa_reference=prev_capa,
        previous_capa_status=None,
        previous_capa_effectiveness="NOT_VERIFIED",
        rationale=(
            "The canonical interpretation established a repeated occurrence."
            if count is not None else
            "The canonical interpretation established a reference to a previous corrective action."
            if prev_capa else None
        ),
        recurrence_count=int(count) if count is not None else None,
        recurrence_event=getattr(rec, "event", None) if rec is not None else None,
        recurrence_period=getattr(rec, "period", None) if rec is not None else None,
    )


def mechanism_info_from_canonical(ctx):
    """Spec Pass 43 §8: build a `MechanismInfo`-shaped object PURELY from the
    validated canonical LLM context, so the canonical-SUCCESS path never runs
    `extract_immediate_mechanism` (deterministic causal-signal detection over
    evidence claims). The immediate mechanism is LLM-owned: a `causal_claim`
    the LLM marked `is_causal`, with its own `evidence_status`. If the LLM
    established none, the mechanism is UNKNOWN (NOT_ESTABLISHED). No regex, no
    keyword list, no evidence re-interpretation. Returns None when `ctx` is
    None."""
    if ctx is None:
        return None
    from app.agent.causal_guard import MechanismInfo

    causal = [
        c for c in (getattr(ctx, "causal_claims", None) or [])
        if getattr(c, "is_causal", False) and (getattr(c, "statement", None) or "").strip()
    ]
    if not causal:
        return MechanismInfo(statement=None, status="UNKNOWN", polarity=None, source_claim=None)
    # Prefer a VERIFIED causal claim; else the first stated one (REPORTED).
    verified = next((c for c in causal if getattr(c, "evidence_status", "") == "VERIFIED"), None)
    chosen = verified or causal[0]
    _st = getattr(chosen, "evidence_status", "UNVERIFIED")
    status = "VERIFIED" if _st == "VERIFIED" else ("REPORTED" if _st == "REPORTED" else "UNKNOWN")
    return MechanismInfo(
        statement=chosen.statement.strip(),
        status=status,
        polarity=None,
        source_claim=chosen.statement.strip(),
    )


def apply_canonical_provenance_fields(canonical_state, semantic_context):
    """PROVENANCE ONLY -- NOT a semantic merge (spec Pass 44 §3). The
    canonical-SUCCESS `canonical_state` is already built entirely from THIS
    `semantic_context` (via `deviation_info_from_canonical` /
    `recurrence_info_from_canonical` / `mechanism_info_from_canonical`). This
    function only copies the remaining context-only provenance fields the
    builders don't carry -- evidence_source / reported_observation /
    finding_epistemic_status / affected_period / scope -- and runs a structural
    subject-safety re-check. It NEVER recovers a semantic value from
    deterministic state (there is none on this path). No-op when
    `semantic_context` is None (fallback path). Returns (canonical_state,
    MergeOutcome) for call-site compatibility."""
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
    # LLM-PRIMARY (spec Pass 38 §1/§11): a condition the canonical LLM stated
    # wins outright; the deterministic floor's condition is used only when the
    # LLM omitted one.
    llm_cond = (getattr(ctx, "observed_condition", None) or "").strip() or None
    if llm_cond:
        cs.deviation_condition = llm_cond
        out.fields_from_llm.append("deviation_condition")

    # ---- COMPARISON ------------------------------------------------
    # Spec Pass 36 §B/§C: ONLY an LLM-declared ACTIVE comparison (explicit
    # ACTUAL_CONFLICT / UNRESOLVED_COMPARISON status + a stated why_comparable)
    # may contribute semantic_type=COMPARISON, a comparison-derived
    # measurement, or the comparison deviation phrasing. An inactive /
    # unclassified comparison object stays on `ctx` for provenance but has NO
    # effect on the canonical finding state. Decided by the structured status,
    # never by inspecting the numbers or the prose.
    # Only an LLM-declared ACTIVE comparison (explicit ACTUAL_CONFLICT /
    # UNRESOLVED_COMPARISON status + a stated why_comparable) contributes any
    # comparison semantics. Pass 40: the floor comparison fields were already
    # wiped by `reset_llm_owned_semantic_fields`, so there is nothing to clear
    # here -- only populate.
    from app.services.canonical_semantic_models import comparison_is_active
    cmp_ = getattr(ctx, "comparison", None)
    if not comparison_is_active(cmp_):
        cmp_ = None

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
    # LLM-PRIMARY (spec Pass 38 §1/§15): a recurrence the canonical LLM stated
    # wins outright (count + event + period), even when the deterministic floor
    # also extracted one.
    rec = getattr(ctx, "recurrence", None)
    if rec is not None and rec.count is not None:
        cs.recurrence_count = int(rec.count)
        cs.recurrence_event = rec.event
        cs.recurrence_period = rec.period
        if rec.event and not getattr(cs, "occurrence_population", None):
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

    # Pass 40: the historical "SEMANTIC AUTHORITY SWEEP" (Pass 37-39) that reset
    # stale `resolve_deviation` floor values is GONE -- `reset_llm_owned_
    # semantic_fields` already wiped them all before this function ran, so every
    # LLM-owned semantic field is either populated above from the canonical
    # context or sitting at its non-activating sentinel. Nothing to sweep.

    return cs, out


def _is_established(subject: str | None) -> bool:
    try:
        from app.services.semantic_subject import is_established_subject
        return bool(is_established_subject(subject))
    except Exception:  # pragma: no cover
        s = (subject or "").strip()
        return bool(s) and not s.upper().startswith(("UNKNOWN", "UNRESOLVED", "NOT ESTABLISHED"))


# Backward-compatible alias: the name predates Pass 43/44 (when this became
# provenance-only). Retained so existing importers (incl. a test module that
# must not be modified) keep working; production code uses
# `apply_canonical_provenance_fields`.
merge_semantic_context_into_canonical = apply_canonical_provenance_fields
