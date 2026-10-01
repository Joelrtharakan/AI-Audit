# Phase 9.9 — Semantic heuristic inventory (generated + hand-classified)

Source: `python scripts/audit_semantic_heuristics.py` (AST scan of `app/`): **460 heuristic sites in 56 modules**
(module-level `re.compile` constants / vocabulary sets with >=4 string literals / inline `re.*` calls with a literal pattern).
Regenerate per-site JSON with `--json`. Counts are `regex/vocab/inline`; *Tests* = number of test files importing the module.

Categories: **A** structural, **B** arithmetic, **C** provenance, **D** presentation, **E** security, **F** semantic interpretation,
**G** legacy compatibility. Action: KEEP / MIGRATE (to a canonical LLM-owned field) / REVIEW (needs per-rule decision) / SPLIT.

> The deterministic semantic boundary is **NOT clean**. Category F/G rows below are real, active, and mostly test-locked.
> Nothing here was deleted in Phase 9.9: removal needs per-rule migration to LLM-owned canonical fields with live validation.
> The raw-text semantic-authority guard (`tests/_raw_text_semantic_authority_baseline.txt`) still freezes the set of
> raw-finding-text keyword sites so it cannot grow. Unclassified modules: none.

## Live-path exposure (what actually runs on every request)

* `app/financial/extractor.py` — runs on **every** request (not only when the LLM is unavailable) to project `cost_impact` /
  `financial_amount` into the Risk & Impact section (`report_generator.py` -> `final_evidence_verification.py`). It is labelled
  `reasoning_source=DETERMINISTIC_REGEX`; Phase 9.9 makes that label raise the review reason `FINANCIAL_SEMANTICS_NOT_LLM_VERIFIED`.
  The auditor-facing cost section is the LLM remediation pipeline, not this.
* `app/agent/nodes/final_evidence_verification.py` — 77 inline regexes run on every request: grounding strip (C, keep) mixed with
  keyword-derived impact/process wording (F, migrate).
* `app/remediation/activities.py`, `canonical_context_validator.py`, `canonical_state_merge.py`, `causal_guard.py` — regex vetoes
  on LLM output. They can only demote/clear an LLM claim (fail-closed); they do not originate meaning. Still category F by the
  Phase 9.9 definition and listed for review.

## Per-module inventory

| Module | regex/vocab/inline | Tests | Category | Path | Action | Purpose / canonical replacement / migration risk |
|---|---|---|---|---|---|---|
| `app/services/semantic_subject.py` | 58/28/98 | 39 | F (structural/grammatical gate + deterministic floor) | floor + sanctioned subject gate | **KEEP gate / REVIEW floor** | Canonical `finding_subject`/`observed_condition` (LLM-primary, flag `canonical_semantic_llm_primary`). Floor is the fail-closed fallback when the LLM is unavailable; ~400 tests lock it. Migration risk HIGH. |
| `app/agent/causal_guard.py` | 42/3/21 | 26 | G/F (verb-shape mechanism extraction + vetoes on LLM output) | live (vetoes) + floor | **REVIEW** | Canonical `causal_claims`, `stated_causal_alternatives`, `root_cause_status`. Vetoes protect against over-claiming; keep until LLM output has own provenance. Risk HIGH. |
| `app/agent/claim_extractor.py` | 15/5/2 | 17 | F (attribution/claim decomposition of raw text) | floor/degraded | **MIGRATE** | `semantic_evidence_interpreter` / canonical evidence claims. Risk MEDIUM. |
| `app/financial/extractor.py` | 19/1/2 | 14 | B (number/currency/unit parsing) + F (cost-ROLE regexes: remediation/recovery/rework/scrap/penalty/historical) | LIVE every request (internal projection, labelled DETERMINISTIC_REGEX) | **SPLIT: KEEP parsing, MIGRATE role classes** | LLM `pricing_information` + remediation cost components own cost role; review state now emits FINANCIAL_SEMANTICS_NOT_LLM_VERIFIED. Risk HIGH (financial tests). |
| `app/agent/invariants.py` | 9/1/13 | 57 | E/A (invariant registry validating pipeline OUTPUT; some regexes read generated prose) | live (validator) | **KEEP; REVIEW prose-reading checks** | Invariants are the safety net (INV-*); prose-reading ones should move to structured fields as canonical state grows. Risk HIGH. |
| `app/remediation/activities.py` | 15/1/3 | 4 | D (action wording normalisation) + F (disposition/scope regexes over LLM output) | live (post-LLM) | **REVIEW** | LLM `disposition`/`depends_on_root_cause` are authoritative; regexes duplicate them. Risk HIGH. |
| `app/services/cost_analysis.py` | 13/1/7 | 5 | G/F (legacy deterministic cost analysis; last-resort fallback) | fallback | **MIGRATE/RETIRE** | Superseded by `FinancialAnalysisResult` + remediation pipeline. Risk MEDIUM. |
| `app/agent/grounding_guard.py` | 9/5/5 | 4 | C/E (hallucination/contamination guard on generated text) | live | **KEEP** | Provenance/containment; REVIEW the few regex checks that read meaning. |
| `app/agent/recurrence_guard.py` | 13/0/1 | 5 | F (raw-text recurrence / previous-CAPA detection) | floor | **MIGRATE** | Canonical `recurrence` + `explicit_previous_capa_reference` (LLM-primary, evidence-id gated). Risk MEDIUM. |
| `app/agent/analytical_validator.py` | 4/4/9 | 11 | A/C (structural repairs of structured outputs; word-overlap helpers) | live | **KEEP; REVIEW overlap helpers** | Phase 9.9 fixed a status-promotion defect here (REPORTED -> SUPPORTED). Risk MEDIUM. |
| `app/services/canonical_context_validator.py` | 10/1/2 | 9 | C + F (regex vetoes on LLM strings; `_restates_observation` word overlap; `_AGG_META`) | live | **REVIEW** | Vetoes can only DEMOTE LLM claims (fail-closed). Phase 9.9 REMOVED the `_asserts_as_fact` veto on `information_gaps` / plan `unknown` (live run: it deleted the model's only gap `the reason for identical invoices`; no existing test depended on it). The veto remains on remediation activity text (lines ~821/836). Replace the rest with structured self_review/consistency review. Risk HIGH. |
| `app/services/epistemic_modality.py` | 9/3/1 | 1 | F (grammatical stance/modality classification of raw clauses) | floor | **MIGRATE** | Canonical `epistemic_status` (LLM-primary). Risk HIGH. |
| `app/services/instruction_detector.py` | 10/1/3 | 4 | E (prompt-injection detection over untrusted text) | live (security) | **KEEP** | Security boundary, not meaning classification; prompt also instructs the model to treat content as data. |
| `app/services/attribution_extraction.py` | 8/1/0 | 2 | F (degraded-mode attribution extraction) | degraded only | **REVIEW** | Used only when LLM extraction unavailable. Risk LOW. |
| `app/services/llm/json_parser.py` | 6/0/2 | 2 | A (defensive JSON extraction) | live | **KEEP** | - |
| `app/agent/proposition_engine.py` | 1/1/17 | 10 | F (causal-ladder / investigation-mode classification) | live | **REVIEW** | Canonical `causal_claims`/`root_cause_status`; risk HIGH. |
| `app/services/grounding_validator.py` | 6/1/0 | 1 | C (quote-in-source grounding) | live | **KEEP** | - |
| `app/agent/nodes/plan_investigation_fallback.py` | 2/1/28 | 27 | F (keyword investigation plan) | fallback | **MIGRATE** | Canonical `investigation_plan`/`information_gaps` (LLM). In frozen baseline guard. |
| `app/agent/nodes/core_synthesis.py` | 3/0/5 | 77 | F (post-LLM status guards; a few keyword defaults) | live | **REVIEW** | Reported-vs-verified guard is epistemic safety (keep). |
| `app/remediation/validator.py` | 1/4/0 | 4 | A/C (structural validation; small occurrence-noun vocabulary) | live | **KEEP; REVIEW vocab** | Structural; the vocabulary is a presentation/occurrence guard. Risk LOW. |
| `app/services/canonical_state_merge.py` | 3/0/2 | 1 | F (evidence-source/bare-state/causal-role regex vetoes) | live | **REVIEW** | Same as above. |
| `app/agent/nodes/final_evidence_verification.py` | 1/0/77 | 56 | F+C (77 inline regexes: strips ungrounded policy/SOP claims AND keyword-derived impact text) | LIVE every request | **SPLIT: KEEP grounding strip, MIGRATE keyword impact/process text** | Canonical `affected_process`, `impact`. Largest single backlog item; in frozen baseline guard. Risk HIGH. |
| `app/agent/causal_graph_traversal.py` | 0/1/3 | 9 | F/C (5-Why grounding to graph) | live | **REVIEW** | - |
| `app/agent/common_factor.py` | 2/0/1 | 2 | F (common-factor lead detection) | live | **REVIEW** | Lead hypotheses must stay POSSIBLE; canonical hypotheses. |
| `app/agent/nodes/ca_draft_generator.py` | 1/0/4 | 4 | G (LEGACY, not in live graph) | dead code | **RETIRE after test migration** | Only unit-test guard checks reference it. |
| `app/agent/permissions.py` | 0/2/2 | 2 | E (write-permission boundary) | live (security) | **KEEP** | - |
| `app/remediation/engine.py` | 0/1/5 | 28 | A (orchestration; small constant sets) | live | **KEEP** | - |
| `app/remediation/provider_normalization.py` | 0/3/0 | 1 | A (schema normalisation) | live | **KEEP** | - |
| `app/remediation/scope.py` | 1/1/2 | 2 | F (deterministic finding-aware remediation scope) | floor | **MIGRATE** | Canonical `remediation_activities` + `remediation_obligation`. Risk MEDIUM. |
| `app/services/text_grounding.py` | 0/1/2 | 1 | C (substring/fuzzy grounding primitives) | live | **KEEP** | - |
| `app/agent/causal_graph.py` | 0/1/9 | 30 | F (typed causal graph eligibility, some wording regexes) | live | **REVIEW** | Canonical `causal_claims`. Risk HIGH. |
| `app/agent/causal_model.py` | 0/1/1 | 2 | A (structured causal model) | live | **KEEP** | - |
| `app/agent/evidence_interpreter.py` | 1/1/0 | 8 | A/C (LLM evidence->claim bridge; small regex) | live | **KEEP** | - |
| `app/agent/nodes/understanding.py` | 1/0/1 | 77 | F (small keyword gate for short findings) | live | **REVIEW** | In frozen baseline guard. |
| `app/agent/output_quality_scorer.py` | 0/2/0 | 2 | D/A (structural scoring) | live | **KEEP** | - |
| `app/services/finding_analysis_service.py` | 0/2/0 | 1 | A (constants) | live | **KEEP** | - |
| `app/services/llm/providers/_m365_copilot_litellm_handler.py` | 2/0/0 | 1 | A/E (provider response parsing) | live | **KEEP** | - |
| `app/services/status_normalizer.py` | 0/2/0 | 0 | A (status enum normalisation) | live | **KEEP** | - |
| `app/agent/nodes/five_why_fallback.py` | 0/0/4 | 21 | F (keyword 5-Why) | fallback | **MIGRATE** | LLM 5-Why from canonical state. In frozen baseline guard. |
| `app/agent/nodes/rca.py` | 0/1/0 | 7 | A (constants) | live | **KEEP** | - |
| `app/agent/nodes/report_generator.py` | 0/1/0 | 46 | D (constants) | live | **KEEP** | - |
| `app/agent/review_state.py` | 0/1/0 | 3 | A (review reasons from structured fields) | live | **KEEP** | Phase 9.9 adds SEMANTIC_CONSISTENCY_ISSUES / FINANCIAL_SEMANTICS_NOT_LLM_VERIFIED. |
| `app/agent/semantic_validator.py` | 0/1/0 | 1 | A/C (structural) | live | **KEEP** | - |
| `app/agent/tools/registry.py` | 0/1/0 | 1 | A (tool registry constants) | live | **KEEP** | - |
| `app/auth/microsoft_entra.py` | 0/1/0 | 1 | E (auth constants) | live (security) | **KEEP** | - |
| `app/config.py` | 0/1/0 | 31 | A (constants) | live | **KEEP** | - |
| `app/financial/provider_normalization.py` | 0/1/0 | 1 | A (schema normalisation) | live | **KEEP** | - |
| `app/financial/relationship_validator.py` | 0/1/0 | 10 | B/C (structural validation of LLM financial relationships) | live | **KEEP** | - |
| `app/remediation/__init__.py` | 0/1/0 | 0 | A (exports) | live | **KEEP** | - |
| `app/remediation/calculator.py` | 0/1/0 | 3 | B (arithmetic) | live | **KEEP** | Phase 9.9: no invented most-likely for alternatives. |
| `app/routers/health.py` | 0/1/0 | 0 | A (constants) | live | **KEEP** | - |
| `app/routers/investigate.py` | 0/1/0 | 1 | E/A (constants) | live | **KEEP** | - |
| `app/services/evidence_ids.py` | 3/0/0 | 1 | C (evidence-label id SYNTAX: `C2:` prefix shape; no meaning) | live | **KEEP** | Phase 9.9 provenance registry; resolves/display-maps ids, never guesses; unresolvable -> None. |
| `app/services/canonical_consistency_review.py` | 0/1/0 | 1 | A/C (structural; no prose) | live | **KEEP** | Phase 9.8/9.9. Imports no `re`. |
| `app/services/canonical_finding_interpreter.py` | 0/1/0 | 12 | A (schema hint constant) | live | **KEEP** | - |
| `app/services/llm_metrics.py` | 0/1/0 | 3 | A (metric labels) | live | **KEEP** | - |
| `app/startup_checks.py` | 0/1/0 | 1 | A (constants) | live | **KEEP** | - |

## Per-rule documentation for Category F/G

For each F/G module the table gives purpose, canonical replacement, test dependence and risk. Rule-level (per-regex) detail is
the machine inventory (`--json`: file, constant name, line, kind, number of test files referencing the constant). A per-rule
migration plan should be written when the owning module is migrated; this phase did not migrate any F rule.
