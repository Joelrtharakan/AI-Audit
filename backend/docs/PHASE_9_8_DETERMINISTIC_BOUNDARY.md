# Phase 9.8 — Deterministic boundary audit

Categories: **A** structural, **B** arithmetic/dimensional, **C** provenance,
**D** rendering safety, **E** security/workflow, **F** semantic interpretation.
Goal: no new category F. Existing F sites are listed honestly below.

## Added / modified in Phase 9.8

| Item | Cat. | Why |
|---|---|---|
| `canonical_consistency_review.review_canonical_consistency` | A, C | Compares the model's own structured fields to each other and checks evidence-id resolution. Imports no `re`; reads no prose (test: prose-invariance). Reports issues, never rewrites. |
| `_enforce_action_status_and_impact_provenance` | C, E | Demotes an AUTHORIZED/COMPLETED action (or OBSERVED impact) that cites no resolvable evidence id to the safe state. Looks only at ids. |
| `SemRemediationAction.action_status`, `SemImpact`, `SemSelfReview` | schema | New semantic concepts the model owns; default is the safe state (PROPOSED / NOT_ESTABLISHED / no admissions). |
| Opt-in regeneration (`canonical_semantic_consistency_regeneration`, default OFF) | A | Feeds issue *codes* back to the same provider once; adopts the retry only if it has fewer issues. |
| Prompt: reasoning procedure, action status, impact, recurrence kinds, untrusted content | prompt | Semantic improvements live in the model's reasoning procedure. |

## Pre-existing category F sites (NOT introduced here, NOT removed here)

These are regex/vocabulary heuristics. They are retained because they sit in
the fail-closed deterministic floor or act as vetoes on LLM output, and ~400
locked tests depend on them. They are the remaining redesign backlog:

- `canonical_context_validator.py`: `_LLM_NONPERFORMANCE_RE`, `_LLM_PERFORMANCE_AMBIGUOUS_RE`,
  `_LLM_DIRECTION_WORD_RE`, `_CAUSE_ASSERTED_RE`, `_BLAME_ASSERTION_RE`, `_UNCERTAIN_FRAME_RE`,
  `_ASSOCIATIVE_HEDGE_RE`, `_restates_observation` (vocabulary overlap), `_is_pricing_gap` `_AGG_META`.
- `canonical_state_merge.py`: `_EVIDENCE_SOURCE_HEAD_RE`, `_BARE_STATE_RE`, `_CAUSAL_ROLE_RE`.
- `remediation/activities.py`: verb/noun regex family used to normalize action wording.
- `financial/extractor.py`: regex extractor (the deterministic floor when the LLM is unavailable).
- `agent/nodes/*_fallback.py`, `final_evidence_verification.py`: keyword fallbacks
  (frozen in `tests/_raw_text_semantic_authority_baseline.txt`; CI fails if the set grows).

No category F classifier was added in this phase.
