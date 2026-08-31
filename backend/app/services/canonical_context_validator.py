"""Deterministic validation/sanitization for the LLM-produced
`CanonicalFindingContext`.

Unlike the financial calculation validator (which accepts/rejects whole
calculation proposals), this validator SANITIZES the canonical context in
place: anything unsupported by real evidence is stripped or forced to its
safe default rather than causing the whole context to be discarded, since
most of a canonical interpretation can be perfectly sound even if one
field is not.

This is where the "never inferred from recurrence alone" and "never a
financial/recovery fact as a cause" rules are enforced independently of
whatever the LLM claims -- the LLM's own `explicit_previous_capa_reference`
and `is_causal` flags are never trusted on their own.
"""

from __future__ import annotations

import re

from app.models.agent import EvidenceItem
from app.services.canonical_semantic_models import CanonicalFindingContext

_NON_CAUSAL_KINDS = {"CONSEQUENCE", "FINANCIAL_METRIC", "HISTORICAL_CONTEXT", "REMEDIATION", "RECOVERY"}
_ENTITY_LIKE_KINDS = {"ENTITY"}
# A candidate deviation must be grounded in a genuine OCCURRENCE, never a
# bare entity mention, a state, or a financial/historical/recovery/
# remediation fact -- see the primary_deviation grounding check below.
_DEVIATION_GROUNDING_KINDS = {"EVENT"}

# Structural, domain-neutral connective/hedge phrases that assert
# ASSOCIATION or co-occurrence without asserting causation or established
# fact ("associated with", "in connection with", ...) -- the same class of
# fixed, generic structural marker already used elsewhere in this codebase
# (e.g. the historical-framing marker), not a per-finding keyword list.
# When a candidate entity's ONLY supporting evidence is a sentence using
# language like this, the entity is a MENTION inside a hedged/associative
# clause, not an independently established affected object -- generalizes
# to any finding using this class of wording, not any specific phrase.
_ASSOCIATIVE_HEDGE_RE = re.compile(
    r"\b(?:associated\s+with|in\s+connection\s+with|in\s+relation\s+to|"
    r"related\s+to|linked\s+to|in\s+relation\s+with|connected\s+(?:to|with))\b",
    re.IGNORECASE,
)


def _valid_evidence_ids(evidence_count: int) -> set[str]:
    return {f"E{i}" for i in range(evidence_count)}


def validate_canonical_context(
    context: CanonicalFindingContext,
    evidence_ledger: list[EvidenceItem],
    finding_text: str = "",
) -> CanonicalFindingContext:
    """Return a sanitized copy of `context` -- every field either passed
    validation as-is or was stripped/forced to its safe default."""

    valid_ids = _valid_evidence_ids(len(evidence_ledger))
    evidence_text_by_id = {f"E{i}": e.claim for i, e in enumerate(evidence_ledger)}
    sanitized = context.model_copy(deep=True)

    # 1/2. Entities must be evidence-supported; a STATE (or any non-ENTITY
    # kind) is retained for context but is never a valid affected-object
    # candidate -- see get_affected_object_candidate() below, which is the
    # only function downstream modules should call.
    sanitized.entities = [
        e for e in sanitized.entities
        if e.source_evidence_ids and all(eid in valid_ids for eid in e.source_evidence_ids)
    ]

    # PROPERTY 1/3 (financial-consequence firewall): an ENTITY-kind
    # candidate whose ONLY supporting evidence is a sentence using
    # associative/hedging language ("associated with", "linked to", ...)
    # is a MENTION inside a hedged clause, not an independently
    # established affected object -- e.g. "losses associated with the
    # same control failure" never establishes "the same control failure"
    # as a real entity, regardless of what kind the LLM tagged it. An
    # EVENT-kind entity is exempt: an occurrence can legitimately be the
    # thing a hedge clause is ABOUT (see Example B's trailing "resulting
    # in" clause), only a claimed ENTITY mention is this fragile.
    def _is_hedge_only(entity) -> bool:
        if entity.kind != "ENTITY" or not entity.source_evidence_ids:
            return False
        texts = [evidence_text_by_id.get(eid, "") for eid in entity.source_evidence_ids]
        return bool(texts) and all(_ASSOCIATIVE_HEDGE_RE.search(t) for t in texts)

    sanitized.entities = [e for e in sanitized.entities if not _is_hedge_only(e)]

    entity_and_claim_ids = {e.entity_id for e in sanitized.entities}
    entity_kind_by_id = {e.entity_id: e.kind for e in sanitized.entities}
    financial_claim_ids = {c.claim_id for c in sanitized.financial.claims}
    entity_and_claim_ids |= financial_claim_ids

    # 3/6. primary_deviation must reference real evidence via a real claim
    # or entity id, AND that id must be a genuine OCCURRENCE (EVENT-kind),
    # never a bare entity mention, a state, a financial/historical/
    # recovery/remediation fact, or a financial.claims id -- a financial
    # consequence can never itself BE the deviation under investigation
    # (Section 3: priority 1-3 require an explicitly stated deviation/
    # nonconformance before financial consequences are even considered).
    # Falls back to NOT_ESTABLISHED rather than keep an ungrounded string.
    if sanitized.primary_deviation_claim_id:
        _dev_id = sanitized.primary_deviation_claim_id
        _grounded = (
            _dev_id in entity_kind_by_id
            and entity_kind_by_id[_dev_id] in _DEVIATION_GROUNDING_KINDS
        )
        if not _grounded:
            sanitized.primary_deviation = None
            sanitized.primary_deviation_claim_id = None
            sanitized.primary_deviation_confidence = "NOT_ESTABLISHED"
    if not sanitized.primary_deviation:
        sanitized.primary_deviation_confidence = "NOT_ESTABLISHED"

    # 8/9/12. Causal claims: cause_ref/effect_ref must reference real
    # entities/claims, must cite real evidence, and a financial/recovery/
    # remediation/historical fact can never be the CAUSE side of a causal
    # claim regardless of what the LLM's is_causal flag says.
    sanitized_causal = []
    entity_kind_by_id = {e.entity_id: e.kind for e in sanitized.entities}
    for cc in sanitized.causal_claims:
        cc = cc.model_copy(deep=True)
        has_evidence = bool(cc.source_evidence_ids) and all(eid in valid_ids for eid in cc.source_evidence_ids)
        if not has_evidence:
            cc.is_causal = False
        if cc.is_causal and cc.cause_ref:
            cause_kind = entity_kind_by_id.get(cc.cause_ref)
            if cause_kind in _NON_CAUSAL_KINDS:
                cc.is_causal = False
        if cc.is_causal and (not cc.cause_ref or not cc.effect_ref):
            # A causal claim with no identified cause/effect reference is
            # not a usable causal claim.
            cc.is_causal = False
        sanitized_causal.append(cc)
    sanitized.causal_claims = sanitized_causal

    # 14. Previous CAPA (spec Pass 42 §5/§15/§29): LLM-owned, validated
    # STRUCTURALLY -- an `explicit_previous_capa_reference` claim must cite at
    # least one evidence id that actually resolves. The former raw-finding-text
    # `detect_recurrence` deterministic-agreement re-check is removed: on the
    # canonical-success path semantic recurrence comes only from the LLM, and a
    # deterministic re-interpretation of the finding prose would violate the
    # authority boundary. A bare claim with no resolvable evidence id is
    # dropped (fail-closed).
    if sanitized.explicit_previous_capa_reference:
        has_real_evidence = bool(sanitized.previous_capa_evidence_ids) and all(
            eid in valid_ids for eid in sanitized.previous_capa_evidence_ids
        )
        if not has_real_evidence:
            sanitized.explicit_previous_capa_reference = False
            sanitized.previous_capa_evidence_ids = []

    _validate_llm_primary_fields(sanitized, finding_text)
    _validate_llm_reasoning_fields(sanitized, evidence_ledger, finding_text)
    return sanitized


# --- LLM-PRIMARY field safety (spec Phase 5) --------------------------------

_LLM_CAUSAL_ROLE_RE = re.compile(
    r"\b(?:root\s+cause|assignable\s+cause|underlying\s+cause|the\s+cause\b|"
    r"a\s+cause\b|causes?\b|mechanism|failure\s+mode|reason\b|contributing\s+factor)\b",
    re.IGNORECASE,
)
_LLM_EVIDENCE_SOURCE_RE = re.compile(
    r"^(?:the\s+|a\s+|an\s+)?(?:records?|logs?|documentation|documents?|register|"
    r"audit|report|review|inspection|assessment|evidence|data|audit\s+trail|"
    r"history|file|observation|survey|walkthrough|reconciliation)\s*$",
    re.IGNORECASE,
)
_LLM_NONPERFORMANCE_RE = re.compile(
    r"\b(?:was|were)\s+(?:never|not)\s+(?:performed|carried\s+out|conducted|done|"
    r"completed|executed|undertaken)\b|"
    r"\bdid\s+not\s+(?:occur|happen|take\s+place|perform|conduct|complete)\b|"
    r"\bnever\s+(?:performed|conducted|done|occurred|took\s+place)\b|"
    r"\bfailed\s+to\s+(?:perform|conduct|complete|carry\s+out)\b|"
    r"\b(?:activity|check|verification|inspection|review|step)\s+(?:was\s+)?"
    r"(?:not\s+performed|omitted|skipped|missed)\b",
    re.IGNORECASE,
)
# "unclear/not established WHETHER ... performed" -> the ambiguity is
# explicit; non-performance is NOT supported.
_LLM_PERFORMANCE_AMBIGUOUS_RE = re.compile(
    r"\b(?:unclear|not\s+(?:clear|established|determined|known)|could\s+not\s+"
    r"(?:be\s+)?(?:established|determined|confirmed)|uncertain)\b[\w\s,'-]{0,40}?"
    r"\bwhether\b",
    re.IGNORECASE,
)
_LLM_DIRECTION_WORD_RE = re.compile(
    r"\b(?:above|below|over|under|exceed\w*|short(?:fall|age)?|surplus|excess|"
    r"deficit|in\s+excess\s+of|greater\s+than|less\s+than|higher\s+than|lower\s+than|"
    r"more\s+than|fewer\s+than|fell\s+(?:short|below))\b",
    re.IGNORECASE,
)


def _sig_words(s: str | None) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", (s or "").lower())}


def _validate_llm_primary_fields(ctx: CanonicalFindingContext, finding_text: str) -> None:
    """Independently constrain the structured LLM-primary fields. Mutates
    `ctx` in place -- an unsafe value is nulled / downgraded, never kept."""
    ft = finding_text or ""
    ft_words = _sig_words(ft)

    # SUBJECT: never a causal role, an evidence-source noun, or a bare
    # enumerated cause.
    subj = (ctx.finding_subject or "").strip()
    if subj:
        low = subj.lower().rstrip(".")
        _bad = (
            _LLM_CAUSAL_ROLE_RE.search(low)
            or _LLM_EVIDENCE_SOURCE_RE.match(low)
            or any(_sig_words(a) and _sig_words(a) <= _sig_words(low)
                   for a in (ctx.stated_causal_alternatives or []))
        )
        if not _bad:
            # A subject must be an ENTITY noun phrase, not a clause / predication
            # ("<entity> remains open", "<entity> lacks X", reported speech).
            # Structural grammatical gate, shared pipeline-wide.
            try:
                from app.services.semantic_subject import reject_subject_if_clause
                _bad = reject_subject_if_clause(subj)
            except Exception:  # pragma: no cover
                pass
        if _bad:
            ctx.finding_subject = None
            ctx.subject_kind = None

    # STATED ALTERNATIVES: each must be grounded in the finding text (no
    # fabrication). Keep order, drop unsupported ones.
    if ctx.stated_causal_alternatives:
        kept = [
            a for a in ctx.stated_causal_alternatives
            if _sig_words(a) and len(_sig_words(a) & ft_words) >= 1
        ]
        ctx.stated_causal_alternatives = kept
        if len(kept) < 2:
            ctx.causal_alternatives_unresolved = ctx.causal_alternatives_unresolved and bool(kept)

    # MISSING RECORD: ACTIVITY_NOT_PERFORMED requires explicit non-performance
    # wording in the finding; otherwise downgrade and mark the ambiguity.
    if ctx.missing_record_status == "ACTIVITY_NOT_PERFORMED" and (
        _LLM_PERFORMANCE_AMBIGUOUS_RE.search(ft) or not _LLM_NONPERFORMANCE_RE.search(ft)
    ):
        ctx.missing_record_status = "ACTIVITY_NOT_RECORDED"
        ctx.activity_performance_ambiguity = True

    # COMPARISON REALITY (spec Pass 34 §9/§11/§36). Two separate ideas:
    #  (a) A comparison the LLM explicitly classified as NOT a discrepancy
    #      (NONE / legitimate multiple prices / subtotal-total) is cleared
    #      entirely -- it must never reach a downstream layer.
    #  (b) Otherwise the object is KEPT for structural preservation of the
    #      stated values (magnitude / direction / measurement), but it is an
    #      ACTIVE semantic comparison -- one that drives a reconciliation
    #      obligation or a comparability investigation question -- ONLY when the
    #      LLM explicitly classified it ACTUAL_CONFLICT / UNRESOLVED_COMPARISON
    #      AND stated why the two values belong together. The fail-closed
    #      NOT_ESTABLISHED default is non-activating. `comparison_is_active`
    #      carries this one decision to every consumer; no consumer re-derives
    #      it, and nothing inspects finding prose.
    if ctx.comparison is not None and getattr(ctx.comparison, "status", "NOT_ESTABLISHED") in (
        "NONE", "LEGITIMATE_MULTIPLE_PRICES", "SUBTOTAL_TOTAL_RELATIONSHIP",
    ):
        ctx.comparison = None

    # COMPARISON DIRECTION: a stated ABOVE/BELOW needs a directional word in
    # the finding; otherwise it is a bare "differed" -> MISMATCH.
    if ctx.comparison is not None and ctx.comparison.direction in ("ABOVE", "BELOW"):
        if not _LLM_DIRECTION_WORD_RE.search(ft):
            ctx.comparison.direction = "MISMATCH"

    # NO MANUFACTURED MAGNITUDE: a comparison magnitude the finding text does
    # not contain as a number is dropped.
    if ctx.comparison is not None and ctx.comparison.magnitude is not None:
        _mag = ctx.comparison.magnitude
        _mag_strs = {str(int(_mag)) if float(_mag).is_integer() else str(_mag), f"{_mag:g}"}
        if not any(m in ft for m in _mag_strs):
            ctx.comparison.magnitude = None
            ctx.comparison.unit = None

    # NO MANUFACTURED RECURRENCE COUNT.
    if ctx.recurrence is not None and ctx.recurrence.count is not None:
        _c = ctx.recurrence.count
        _words = {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
        _c_ok = str(_c) in ft or any(
            w in ft.lower() and {2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                                 7: "seven", 8: "eight", 9: "nine", 10: "ten"}.get(_c) == w
            for w in _words
        )
        if not _c_ok:
            ctx.recurrence.count = None


# --- LLM-owned investigative / remediation reasoning safety (spec §6-§10) ---

# An assertion that a cause IS established as fact (as opposed to naming it as
# an open question). Structural, domain-neutral.
_CAUSE_ASSERTED_RE = re.compile(
    r"\b(?:was|were|is|are|has\s+been|have\s+been)\s+caused\s+by\b|"
    r"\b(?:because\s+of|due\s+to|as\s+a\s+result\s+of|attributable\s+to|"
    r"stems?\s+from|resulted\s+from|the\s+root\s+cause\s+(?:is|was)\b)\b",
    re.IGNORECASE,
)
# Interrogative / uncertainty framing that makes a causal mention an OPEN
# question rather than an assertion.
_UNCERTAIN_FRAME_RE = re.compile(
    r"\b(?:whether|if|which|what|why|how|unknown|unclear|not\s+(?:known|established|"
    r"determined|confirmed)|possible|may|might|could|suspected|to\s+be\s+determined|"
    r"extent|degree|the\s+cause\s+of)\b",
    re.IGNORECASE,
)
# A statement that LEADS with an investigative/methodological verb is an
# EVIDENCE REQUEST (a proposal for obtaining evidence), never an assertion
# about reality -- so it is never held to the evidence-grounding bar that
# assertions are. Generic method vocabulary, not a domain or finding list.
_EVIDENCE_REQUEST_LEAD_RE = re.compile(
    r"^\s*(?:to\s+)?(?:review|examine|obtain|gather|collect|request|determine|confirm|"
    r"verify|validate|assess|evaluate|inspect|interview|reconcile|compare|establish|"
    r"check|identify|quantify|trace|analy[sz]e|audit|test|sample|observe|walk\s*through|"
    r"clarify|ascertain|corroborate|cross-?check)\b",
    re.IGNORECASE,
)
# Bare declarative attribution of failure / blame with NO hedge -- an
# assertion about reality that must be evidence-supported (spec §2A / §3).
_BLAME_ASSERTION_RE = re.compile(
    r"\b(?:failed\s+to|did\s+not|neglected\s+to|was\s+negligent|"
    r"was\s+responsible\s+for|deliberately|intentionally|"
    r"(?:was|were|is|are)\s+(?:at\s+fault|to\s+blame|non-?compliant|in\s+violation))\b",
    re.IGNORECASE,
)


def _asserts_as_fact(text: str | None) -> bool:
    """True when `text` states a CAUSE or an act of failure/blame as an
    established fact, with no interrogative/uncertainty framing and not
    phrased as an evidence request. Role check (spec §2/§3): only assertions
    about reality are gated -- questions, requests and hypotheses are not."""
    t = (text or "").strip()
    if not t or _EVIDENCE_REQUEST_LEAD_RE.match(t) or _UNCERTAIN_FRAME_RE.search(t):
        return False
    return bool(
        _CAUSE_ASSERTED_RE.search(t)
        or _BLAME_ASSERTION_RE.search(t)
        or _LLM_CAUSAL_ROLE_RE.search(t)
    )


def _validate_llm_reasoning_fields(
    ctx: CanonicalFindingContext,
    evidence_ledger: list[EvidenceItem],
    finding_text: str,
) -> None:
    """Constrain the LLM's proposed hypotheses / investigation plan /
    remediation reasoning by SEMANTIC ROLE, not vocabulary overlap
    (spec §2/§3):

      - Assertions about reality  -> must be evidence-supported.
      - Investigation questions / evidence requests -> the requested evidence
        need NOT appear in the finding; only rejected if they assert an
        unverified cause/blame as fact.
      - Hypotheses -> stay POSSIBLE unless a VERIFIED causal claim backs them;
        never dropped for lacking finding tokens.
      - Remediation activities -> proposals; kept unless they are an
        unsupported concrete systemic prescription while the cause is
        unconfirmed (then forced CONDITIONAL, not dropped).

    Mutates `ctx` in place. Never strengthens epistemic status."""

    # Did the LLM opt into the investigative/remediation reasoning contract
    # at all? If it produced none of it, we do NOT synthesize any -- the
    # deterministic path (which already preserves stated alternatives) runs
    # unchanged. Only when the LLM actually reasoned do we normalize +
    # guard that reasoning.
    _llm_reasoned = bool(
        ctx.candidate_hypotheses or ctx.investigation_plan
        or ctx.remediation_activities or ctx.investigation_activities
        or ctx.information_gaps
    )

    # ---- ROOT CAUSE: never stronger than a VERIFIED causal claim ----------
    has_verified_cause = any(
        cc.is_causal and cc.evidence_status == "VERIFIED" and cc.cause_ref
        for cc in ctx.causal_claims
    )
    if ctx.causal_alternatives_unresolved or not has_verified_cause:
        if ctx.root_cause_status == "ESTABLISHED":
            ctx.root_cause_status = "NOT_ESTABLISHED"
        ctx.leading_hypothesis_id = None

    # ---- CANDIDATE HYPOTHESES ------------------------------------------
    # Role = HYPOTHESIS: a possible explanation of WHY the condition occurred.
    # PRIMARY gate (spec §3): the LLM's OWN `semantic_role` -- keep only
    # CAUSAL_MECHANISM (a finding-enumerated alternative is always kept, §13).
    # A statement the model itself labels OBSERVATION_RESTATEMENT / CONSEQUENCE
    # / OTHER_NON_CAUSAL_STATEMENT names no mechanism -> dropped.
    # SAFETY net, ONLY when the model left `semantic_role` unset: the structural
    # set-difference below (a paraphrase introduces no vocabulary of its own).
    _OBS_FUNCTION_WORDS = {
        "was", "were", "not", "been", "has", "have", "had", "did", "does", "the",
        "and", "but", "that", "this", "with", "for", "are", "its", "will", "then",
        "which", "such", "any", "all", "due", "yet",
    }
    _obs_vocab = (
        _sig_words(ctx.finding_subject) | _sig_words(ctx.observed_condition)
        | _sig_words(getattr(ctx, "primary_deviation", None))
    )
    for _e in evidence_ledger:  # the raw evidence text is also "the observation"
        _obs_vocab |= _sig_words(getattr(_e, "claim", ""))

    def _restates_observation(statement: str) -> bool:
        w = _sig_words(statement) - _OBS_FUNCTION_WORDS
        if not w:
            return True
        _novel = w - _obs_vocab
        # a genuine causal mechanism contributes >=2 concepts of its own
        return len(_novel) < 2

    kept_hyps = []
    seen_ids: set[str] = set()
    seen_stmts: set[frozenset] = set()
    _stated_alt_vocab = [
        _sig_words(a) for a in (ctx.stated_causal_alternatives or []) if _sig_words(a)
    ]
    for h in ctx.candidate_hypotheses:
        if h.hypothesis_id in seen_ids or not (h.statement or "").strip():
            continue
        _sk = frozenset(_sig_words(h.statement))
        if _sk and _sk in seen_stmts:
            continue
        # keep finding-enumerated alternatives; drop non-causal statements.
        _is_stated_alt = bool(h.from_finding_text) or any(
            av and av <= _sig_words(h.statement) for av in _stated_alt_vocab
        )
        if not _is_stated_alt:
            _role = getattr(h, "semantic_role", None)
            if _role in ("OBSERVATION_RESTATEMENT", "CONSEQUENCE", "OTHER_NON_CAUSAL_STATEMENT"):
                continue  # the model's own classification: not a causal mechanism
            if _role is None and _restates_observation(h.statement):
                continue  # safety net only when the model left the role unset
        seen_ids.add(h.hypothesis_id)
        if _sk:
            seen_stmts.add(_sk)
        if h.epistemic == "SUPPORTED" and not (
            has_verified_cause and h.hypothesis_id == ctx.leading_hypothesis_id
        ):
            h.epistemic = "POSSIBLE"
        h.source_evidence_ids = [e for e in h.source_evidence_ids if e in _valid_evidence_ids(len(evidence_ledger))]
        kept_hyps.append(h)
    if _llm_reasoned:
        # The LLM engaged the reasoning contract -> every finding-enumerated
        # alternative MUST be represented (never silently dropped, §5).
        for i, alt in enumerate(ctx.stated_causal_alternatives or []):
            alt_w = _sig_words(alt)
            _match = next(
                (h for h in kept_hyps if alt_w and alt_w <= _sig_words(h.statement)), None
            )
            if _match is not None:
                _match.from_finding_text = True
                if _match.epistemic == "SUPPORTED":
                    _match.epistemic = "POSSIBLE"
                continue
            from app.services.canonical_semantic_models import SemHypothesis
            kept_hyps.append(SemHypothesis(
                hypothesis_id=f"HALT{i + 1}", statement=alt.strip(),
                epistemic="POSSIBLE", from_finding_text=True,
            ))
    ctx.candidate_hypotheses = kept_hyps
    if ctx.leading_hypothesis_id and ctx.leading_hypothesis_id not in {h.hypothesis_id for h in kept_hyps}:
        ctx.leading_hypothesis_id = None
    # If the finding enumerates alternatives, none of them leads.
    if ctx.causal_alternatives_unresolved:
        ctx.leading_hypothesis_id = None

    # ---- INVESTIGATION PLAN ------------------------------------------
    # Role = INVESTIGATION REQUEST: a proposal for obtaining evidence. The
    # evidence it names need NOT be in the finding. Rejected ONLY when the
    # `unknown` presupposes an unverified cause / act of blame as fact
    # (spec §4 / §2B). Duplicate-unknown collapse only.
    kept_steps = []
    seen_unknowns: set[frozenset] = set()
    valid_hyp_ids = {x.hypothesis_id for x in ctx.candidate_hypotheses}
    for s in ctx.investigation_plan:
        u = (s.unknown or "").strip()
        if not u or _asserts_as_fact(u):
            continue
        _uk = frozenset(_sig_words(u))
        if _uk and _uk in seen_unknowns:
            continue
        if _uk:
            seen_unknowns.add(_uk)
        s.related_hypothesis_ids = [h for h in s.related_hypothesis_ids if h in valid_hyp_ids]
        kept_steps.append(s)
    ctx.investigation_plan = kept_steps

    # Role = UNKNOWN: a statement of what is not established. Kept unless it
    # actually asserts a cause/blame as fact.
    ctx.information_gaps = [
        g for g in ctx.information_gaps if (g or "").strip() and not _asserts_as_fact(g)
    ]

    # ---- REMEDIATION OBLIGATION ----------------------------------
    # Consistency between the LLM's own top-level semantic decisions (spec
    # §"REMEDIATION OBLIGATION" / §"CURRENT DEMONSTRATED CASE"). All triggers
    # here are the model's OWN flags -- comparison / causal_alternatives_
    # unresolved / root_cause_status / causal_claims -- never a verb or
    # keyword rule.
    _has_verified_corrective_evidence = any(
        cc.is_causal and cc.evidence_status == "VERIFIED" and cc.cause_ref
        for cc in ctx.causal_claims
    )
    _cause_or_condition_established = (
        ctx.root_cause_status == "ESTABLISHED" or _has_verified_corrective_evidence
    )
    # DIRECT REMEDIATION (spec §2/§3/§5): an activity the model classified as
    # IMMEDIATE_CORRECTION / CONTAINMENT addresses the ESTABLISHED OBSERVED
    # CONDITION -- it is valid independently of whether the CAUSE is confirmed.
    # An unknown root cause must NOT strip this work. (Reads the model's own
    # `disposition`, not the wording.)
    _DIRECT_DISPOSITIONS = {"IMMEDIATE_CORRECTION", "CONTAINMENT"}
    _has_direct_remediation = any(
        (getattr(a, "disposition", "") or "") in _DIRECT_DISPOSITIONS
        for a in ctx.remediation_activities
    )
    # An UNRESOLVED COMPARISON whose competing mechanisms are not resolved and
    # whose root cause is not established -- and where the model did NOT
    # establish any direct-correction work -- is a discrepancy to RECONCILE
    # first, not a corrective obligation.
    from app.services.canonical_semantic_models import comparison_is_active
    _cmp_active = comparison_is_active(ctx.comparison)
    _unresolved_discrepancy = bool(
        _cmp_active
        and ctx.causal_alternatives_unresolved
        and not _cause_or_condition_established
        and not _has_direct_remediation
    )

    # NORMALISE the obligation against the model's OWN higher-level signals
    # (spec §26 -- a downstream layer may downgrade / make explicit, never
    # escalate). No verbs, no keywords.
    if _unresolved_discrepancy:
        ctx.remediation_obligation = "RECONCILIATION_REQUIRED"
    elif ctx.remediation_obligation == "NOT_DETERMINED":
        if _cause_or_condition_established:
            ctx.remediation_obligation = "ESTABLISHED_CORRECTIVE_OBLIGATION"
        elif _has_direct_remediation:
            # The model established DIRECT remediation of the observed condition
            # even though the cause is unconfirmed (spec §3/§5/§18).
            ctx.remediation_obligation = "IMMEDIATE_CORRECTION_ONLY"
        else:
            ctx.remediation_obligation = "INVESTIGATION_REQUIRED"
    elif (
        ctx.remediation_obligation == "ESTABLISHED_CORRECTIVE_OBLIGATION"
        and ctx.causal_alternatives_unresolved
        and not _cause_or_condition_established
    ):
        # A SYSTEMIC corrective obligation cannot be established while the
        # competing mechanisms are unresolved -- but any DIRECT correction the
        # model established of the observed condition still stands.
        ctx.remediation_obligation = (
            "IMMEDIATE_CORRECTION_ONLY" if _has_direct_remediation
            else "RECONCILIATION_REQUIRED" if _cmp_active
            else "INVESTIGATION_REQUIRED"
        )

    # No remediation is established ONLY when the obligation is
    # investigation/reconciliation-only AND the model established no direct
    # correction of the observed condition (spec §5: root-cause uncertainty is
    # NOT a remediation gate). When it holds, activities the model put under
    # remediation are preserved as investigation_activities, never dropped.
    _remediation_not_established = (
        ctx.remediation_obligation in ("RECONCILIATION_REQUIRED", "INVESTIGATION_REQUIRED")
        and not _has_direct_remediation
    )
    _no_systemic_obligation = ctx.remediation_obligation in (
        "RECONCILIATION_REQUIRED", "INVESTIGATION_REQUIRED", "NO_SYSTEMIC_REMEDIATION_JUSTIFIED",
    )

    # ---- ACTIVITIES: partition INVESTIGATION from REMEDIATION -----------
    # Spec §2/§14: the two are DIFFERENT layers. The partition is on the LLM's
    # own declared `disposition` -- the deterministic layer never re-classifies
    # an activity by its verb (§2/§17). Deterministic constraints are epistemic
    # only:
    #  - CORRECTIVE_ACTION addresses a CONFIRMED cause -> valid only when root
    #    cause is ESTABLISHED, else it is cause-dependent (conditional).
    #  - a concrete systemic prescription / an activity asserting the cause as
    #    fact cannot be CONFIRMED while the cause is unconfirmed -> CONDITIONAL
    #    (never deleted).
    #  - when the evidence has NOT established a systemic obligation, a
    #    systemic/corrective activity stays CONDITIONAL; investigation,
    #    immediate correction, containment, effectiveness checks are untouched.
    from app.remediation.activities import is_unsupported_concrete_intervention

    contingent = ctx.root_cause_status != "ESTABLISHED"
    seen_acts: set[frozenset] = set()
    _investigation: list = []
    _remediation: list = []
    # LLM-declared investigation activities first (force disposition), then the
    # activities it put in remediation_activities -- partitioned by disposition.
    _tagged = [(a, True) for a in ctx.investigation_activities] + [
        (a, False) for a in ctx.remediation_activities
    ]
    for a, _declared_investigation in _tagged:
        act = (a.activity or "").strip()
        if not act:
            continue
        _ak = frozenset(_sig_words(act))
        if _ak and _ak in seen_acts:
            continue
        if _ak:
            seen_acts.add(_ak)
        if _declared_investigation:
            a.disposition = "INVESTIGATION"
        if a.disposition == "INVESTIGATION":
            # Investigation/verification work -- NOT remediation, never
            # conditional (its dependency is on nothing; it IS the enquiry).
            a.depends_on_root_cause = False
            _investigation.append(a)
            continue
        # DIRECT REMEDIATION (spec §4/§6/§10/§23): the model classified this as
        # a direct correction of the ESTABLISHED observed condition. Its
        # necessity does NOT depend on the cause -- it is NOT conditionalised by
        # root-cause uncertainty, and "replace/restore the damaged X" is the
        # correction, not an "unsupported concrete intervention". The only
        # epistemic guard that still applies: it must not assert an unverified
        # cause as fact in its own wording.
        if a.disposition in ("IMMEDIATE_CORRECTION", "CONTAINMENT"):
            if contingent and _asserts_as_fact(act):
                a.disposition = "CONDITIONAL_SYSTEMIC"
                a.depends_on_root_cause = True
            _remediation.append(a)
            continue
        # SYSTEMIC / CORRECTIVE (cause-dependent): CORRECTIVE_ACTION needs an
        # established cause; a concrete new-resource prescription or a cause
        # asserted as fact while the cause is unconfirmed -> CONDITIONAL.
        if contingent and a.disposition == "CORRECTIVE_ACTION":
            a.disposition = "CONDITIONAL_SYSTEMIC"
            a.depends_on_root_cause = True
        if contingent and (
            a.disposition == "CONDITIONAL_SYSTEMIC"
            or a.depends_on_root_cause
            or is_unsupported_concrete_intervention(act)
            or _asserts_as_fact(act)
            or _no_systemic_obligation
        ):
            a.disposition = "CONDITIONAL_SYSTEMIC"
            a.depends_on_root_cause = True
        _remediation.append(a)

    # Model's own obligation says only investigation/reconciliation is required
    # (or this is an unresolved discrepancy): no remediation activity is
    # established -> every listed remediation activity is premature and is
    # preserved as investigation, never dropped, never priced, never conditional
    # remediation.
    if _remediation_not_established and _remediation:
        for a in _remediation:
            a.disposition = "INVESTIGATION"
            a.depends_on_root_cause = False
            if not any(frozenset(_sig_words(a.activity)) == frozenset(_sig_words(x.activity)) for x in _investigation):
                _investigation.append(a)
        _remediation = []

    ctx.investigation_activities = _investigation
    ctx.remediation_activities = _remediation  # MAY be empty -- valid (§9)

    # Keep the flat projections consistent with dispositions.
    ctx.immediate_actions = [
        a.activity for a in _remediation
        if a.disposition in ("IMMEDIATE_CORRECTION", "CONTAINMENT")
    ]
    ctx.conditional_actions = [
        a.activity for a in _remediation if a.disposition == "CONDITIONAL_SYSTEMIC"
    ]

    # ---- PRICING: downstream of remediation (§10/§11/§12) --------------
    # Every priceable item must map to a genuine remediation activity. An item
    # that points at investigation work, or at nothing while remediation
    # activities exist, is dropped -- pricing is never manufactured ahead of an
    # established remediation scope. A pure "observed value in the finding"
    # note is kept (it is the §12 firewall, not a price).
    ft = finding_text or ""
    _rem_ids = {a.action_id for a in _remediation}
    kept_pricing = []
    for p in ctx.pricing_information:
        if p.pricing_basis and any(ch.isdigit() for ch in p.pricing_basis):
            p.pricing_basis = None  # a basis is a category, never a figure
        _is_observed_note = bool(p.observed_value_in_finding)
        if _is_observed_note:
            digits = re.findall(r"\d[\d,.]*", p.observed_value_in_finding)
            if digits and not any(d.rstrip(".,") in ft for d in digits):
                p.observed_value_in_finding = None
                _is_observed_note = False
        # An observed value in the finding is NEVER a remediation cost here --
        # only the finding text can establish that, and this field is the §12
        # firewall, not a price.
        p.observed_value_is_remediation_cost = False
        # Pricing is strictly downstream of an established remediation (§8/§11):
        # keep an item ONLY if it maps to a genuine remediation activity, or it
        # is purely a §12 observed-value note. Everything else -- an item that
        # points at investigation work, at nothing, or exists while there is no
        # remediation activity at all -- is dropped.
        if p.action_id in _rem_ids or _is_observed_note:
            kept_pricing.append(p)
    ctx.pricing_information = kept_pricing


def get_affected_object_candidate(context: CanonicalFindingContext | None) -> str | None:
    """The ONLY function downstream modules should use to read an
    affected object off a (validated) canonical context. Returns the
    entity tied to the primary deviation if one exists, else the first
    genuine ENTITY-kind (never STATE/FINANCIAL_METRIC/RECOVERY/...)
    entity, else None. Never returns a state word or a raw clause."""
    if context is None:
        return None
    if context.primary_deviation_claim_id:
        for e in context.entities:
            if e.entity_id == context.primary_deviation_claim_id and e.kind in _ENTITY_LIKE_KINDS:
                return e.name
    for e in context.entities:
        if e.kind in _ENTITY_LIKE_KINDS:
            return e.name
    return None
