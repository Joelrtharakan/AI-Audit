# LQMS Corrective Action Investigation Agent

An autonomous LangGraph agent that turns a raw audit finding into an
auditor-facing **Investigation Report** and a 5-field **CA (Corrective
Action) draft**. It never writes to the production LQMS — every output is a
suggestion that a human auditor must review and approve.

Entry point: `POST /api/v1/investigate` ([backend/app/routers/investigate.py](routers/investigate.py))
Graph definition: [`app/agent/graph.py`](graph.py)
State shape: [`app/agent/state.py`](state.py)
Output schema: [`app/models/agent.py`](../models/agent.py)

---

## Quick Start (Running Locally)

### 1. Backend (FastAPI on Port 8010)
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8010
```
- **Backend API**: `http://localhost:8010`
- **Interactive OpenAPI Docs**: `http://localhost:8010/docs`

### 2. Frontend (Dev Server on Port 5510)
```bash
cd frontend
python3 dev_server.py
```
- **Frontend App**: `http://localhost:5510/index.html`

---

## 1. Design principles

These rules are enforced **in code**, not just in prompts, because prompts
can be ignored by a model but code cannot:

1. **Evidence-gated, not vibes-gated.** Every claim the agent makes must
   trace back to something in the evidence ledger. A hypothesis with no
   supporting/contradicting claim ID is rejected outright
   ([`_parse_causal_fields`](nodes/core_synthesis.py) — "PROVENANCE
   ENFORCEMENT").
2. **The agent cannot invent facts to fill gaps.** When the LLM fails, the
   fallback is a *deterministic*, evidence-grounded generator — never a
   second LLM call dressed up as ground truth, and never silence.
3. **Read-only tool access.** The agent can only call a fixed allowlist of
   read GET endpoints against the LQMS ([`APPROVED_TOOLS`](tools/registry.py)).
   It cannot write to LQMS records at all — the ASP.NET client only exposes
   GET methods.
4. **A hard write-permission boundary on the CA draft.** The agent is only
   ever allowed to write 5 named fields
   ([`AI_WRITABLE_FIELDS`](permissions.py)); anything else raises
   `PermissionError` before it can reach the response, regardless of what
   the LLM returned.
5. **`human_review_required` is always `True`.** Enforced by a Pydantic
   validator on `InvestigationReport`, not just a default value — it is
   impossible to construct a report object with this set to `False`.
6. **Nodes never raise.** Every node degrades gracefully and records what
   happened in `trace`/`errors`; the graph always reaches `END` and returns
   a usable (if degraded) response instead of a 500.
7. **Certainty is monotonic and structurally provable, not asserted.** A
   root cause the LLM merely *states* as "VERIFIED" is downgraded unless a
   VERIFIED evidence-ledger claim actually overlaps it. Status/confidence/
   category fields are re-derived deterministically from the evidence, not
   trusted verbatim from the model's JSON.

---

## 2. High-level pipeline

```
POST /api/v1/investigate
        │
        ▼
┌───────────────────┐
│  understand_finding │  deterministic obs. quality + LLM extraction +
│                    │  claim decomposition + conflict detection +
│                    │  recurrence detection → CanonicalFindingState
└─────────┬──────────┘
          ▼
┌───────────────────┐
│ plan_investigation │  LLM decides which read-only LQMS tools to call
│                    │  (skipped entirely if no tool endpoints configured)
└─────────┬──────────┘
          │
      needs tools? ──── no ───────────────────────────┐
          │ yes                                        │
          ▼                                            │
┌───────────────────┐   ┌────────────────────┐         │
│   execute_tool     │──▶│  record_evidence   │         │
│ (allowlisted only) │   │ classify results   │         │
└─────────▲──────────┘   │ into ledger/gaps   │         │
          │               └─────────┬──────────┘         │
          └──── more tools? ──yes───┘                    │
                          │ no                            │
                          ▼                                ▼
                 ┌──────────────────────────────────────────┐
                 │              core_synthesis               │
                 │  ONE LLM call: RCA + 5-Why + hypotheses;   │
                 │  Impact/CAPA/CA-draft derived deterministically
                 │  from that result. Falls back to a compact │
                 │  recovery call, then full deterministic     │
                 │  synthesis, if the primary call fails.      │
                 └───────────────────┬────────────────────────┘
                                     ▼
                            ┌────────────────┐
                            │     critic      │  0ms deterministic pre-gate;
                            │                 │  LLM self-review only runs
                            │                 │  if something looks ungrounded
                            └───────┬─────────┘
                          send back? │ approved/max-iter
                          (loops to execute_tool)
                                     ▼
                            ┌────────────────────┐
                            │  generate_report     │  deterministic assembly,
                            │                      │  no LLM call
                            └──────────┬───────────┘
                                       ▼
                            ┌────────────────────────────┐
                            │ final_evidence_verification │  the analytical
                            │                              │  firewall — last
                            │                              │  stop before the
                            │                              │  auditor sees it
                            └──────────────┬───────────────┘
                                           ▼
                                          END
                            → InvestigateResponse (report, ca_draft, trace)
```

The graph is a LangGraph `StateGraph` compiled once and reused
([`get_agent_graph()`](graph.py)). `AgentState` (a single `TypedDict`) is
the only thing that flows between nodes — nodes read from it and return a
merged copy; LangGraph persists it across the conditional loops.

### Control-flow / conditional edges

| Edge function | Decides |
|---|---|
| `should_investigate` | After planning: go to `execute_tool` or skip straight to `core_synthesis`, gated on `needs_investigation`, a non-empty `planned_tools`, and the iteration cap. |
| `more_tools_needed` | After recording evidence: loop back to `execute_tool` for the next planned tool, or move on, gated on `agent_max_tool_calls` / `agent_max_iterations`. |
| `critic_decision` | After the critic: send back to `execute_tool` for another investigation round (capped by `agent_max_critic_iterations`), or proceed to `generate_report`. |

---

## 3. Node-by-node detail

### 3.1 `understand_finding` — [nodes/understanding.py](nodes/understanding.py)

The intake node. Nothing downstream re-interprets the raw finding text —
everything below reasons over what this node produces.

- **Observation quality** — deterministic 0ms check (`len(words) < 8` →
  `INSUFFICIENT`), later upgraded to `SUFFICIENT` once a concrete affected
  object and deviation are resolved. Root-cause uncertainty can never
  downgrade this field.
- **Structured extraction** — one LLM call (`extract_finding`) split into
  `stated_facts` (→ `EvidenceStatus.VERIFIED`) and `attributed_statements`
  (→ `EvidenceStatus.REPORTED`), seeding the evidence ledger. If the LLM
  call fails, a deterministic sentence-splitter + attribution-pattern
  fallback (`_fallback_extraction_result`) still recovers facts vs.
  reported statements instead of leaving the ledger empty.
- **Semantic subject resolution** — `resolve_deviation()` is the sole
  authoritative producer of `finding_subject`/`affected_object`; the LLM's
  own subject guess is used only as a last resort (a real production bug
  came from trusting the LLM's subject whenever it merely shared
  vocabulary with the finding).
- **Immediate mechanism extraction** — `causal_guard.extract_immediate_mechanism`
  detects whether the finding *already states* how the deviation happened
  (structural detection, not keyword lists), tagged VERIFIED/REPORTED/UNKNOWN.
- **Claim decomposition & conflict detection** — `claim_extractor.extract_claims`
  breaks the finding into individually-attributed `EvidenceClaim`s;
  `detect_evidence_conflicts` finds pairs of claims that contradict each
  other (e.g. "delivered" vs. "not received").
- **Referenced-but-unavailable documents** — `semantic_subject.resolve_referenced_documents`
  flags documents the finding *cites* but that were never actually
  inspected, so their content is never later treated as evidence
  (`ReferencedDocumentInfo`).
- **Recurrence detection** — `recurrence_guard.detect_recurrence` flags
  whether this finding recurs a previous one, and whether a previous CAPA's
  *completion* is being conflated with its *effectiveness* (it never is).
- **Instruction/prompt-injection stripping** — `is_instruction()` drops any
  "fact" that reads like an instruction to the model before it ever reaches
  the ledger.

All of the above is assembled into **`CanonicalFindingState`** — the single
intermediate representation every downstream node consumes.

### 3.2 `plan_investigation` — [nodes/investigation_planner.py](nodes/investigation_planner.py)

Asks the LLM which read-only LQMS tools to call (only from
`APPROVED_TOOLS`) and produces the initial `InvestigationPlan`
(`areas`, `questions`, `evidence_to_collect`).

- **Fast path**: if no ASP.NET tool endpoint is configured, this node
  costs 0ms and sets `needs_investigation=False`.
- **Domain guard**: drops any planned area/question/evidence item that
  invokes a domain (e.g. training/authorization) this finding never
  mentions (`grounding_guard.mentions_unsupported_domain`).
- **Question quality guard**: `analytical_validator.validate_investigation_question`
  rejects malformed or non-causal questions.
- **Question ↔ evidence consistency check**: flags a question with no
  matching evidence item in the plan.
- If the plan ends up with zero questions (LLM failure or all filtered
  out), a deterministic fallback plan is generated
  ([`plan_investigation_fallback.py`](nodes/plan_investigation_fallback.py)).

### 3.3 `execute_tool` — [nodes/tool_executor.py](nodes/tool_executor.py)

Executes one planned tool per pass (the graph loops back for the next).

- Re-validates the tool name against `APPROVED_TOOLS` in code — a
  `PermissionError` here is a security-boundary breach, logged as an error.
- Per-tool timeout (`agent_tool_timeout_seconds`) and a hard cap on total
  tool calls (`agent_max_tool_calls`).
- Never raises: permission errors, timeouts, and arbitrary exceptions are
  all caught, logged to `trace`/`errors`, and the tool is marked complete
  so the loop always terminates.

### 3.4 `record_evidence` — [nodes/evidence_recorder.py](nodes/evidence_recorder.py)

The **hallucination firewall for tool results**. An LLM call classifies
each piece of returned data into `EvidenceItem`s with an `EvidenceStatus` —
it is instructed to *classify*, never fabricate.

- If a tool returned no data, an `EvidenceGap` is recorded instead of
  inventing evidence.
- Any LLM failure here degrades to a logged gap, never a crash.

### 3.5 `core_synthesis` — [nodes/core_synthesis.py](nodes/core_synthesis.py)

**The single authoritative implementation of RCA, 5-Why, contributing
factors, impact, CAPA, and CA-draft generation.** (Older per-stage node
files — `rca.py`, `impact.py`, `capa.py`, `ca_draft_generator.py` — still
exist for isolated unit tests but are **not** part of the live graph.)

**Call chain on failure** (never skips straight to deterministic synthesis
just because the model filled its whole token budget):

```
Primary LLM call (full schema: RCA + hypotheses + 5-Why + contributing factors)
  → parse/validate JSON
  → ACCEPT if valid                                    (analysis_mode="LLM")
  → else: compact recovery call (causal fields only, smaller prompt)
      → ACCEPT if valid                                (analysis_mode="LLM", source="RECOVERY_LLM")
      → else: full deterministic evidence-grounded synthesis
                                                          (analysis_mode="DETERMINISTIC")
```

Impact, CAPA, and the CA draft are **always derived deterministically**
from whichever root-cause/hypothesis result won above — the LLM is never
asked to restate the same analysis a second time in a different field.

**What gets checked/rejected on every candidate hypothesis** (in
[`_parse_causal_fields`](nodes/core_synthesis.py)), each backed by a
function in [`causal_guard.py`](causal_guard.py):

| Guard | Rejects a hypothesis that... |
|---|---|
| `hypothesis_overclaims_human_error` | blames "human error/oversight" without a process/control framing → demoted from HIGH to MEDIUM relevance, not dropped |
| status `REFUTED` | the model itself marked it refuted |
| `hypothesis_contradicts_mechanism` | conflicts with the already-established immediate mechanism |
| `mechanism_already_names_generic_hypothesis` | just restates the established mechanism as a hedge |
| `hypothesis_contradicts_verified_completion` | claims something is deficient that a VERIFIED fact confirms was done |
| `hypothesis_attacks_statement_credibility` | attacks a reported statement's honesty instead of reasoning about the underlying fact |
| `is_evidence_gap_not_hypothesis` | just restates an evidence gap, dressed as a cause |
| **provenance enforcement** | cites zero claim IDs, or cites a claim ID the ledger never issued |
| `hypothesis_discrimination_cites_wrong_id` | its own discrimination/confirms/refutes text describes evidence for a *different* hypothesis |
| `evaluate_causal_eligibility` ([causal_graph.py](causal_graph.py)) | fails the formal causal-eligibility check (see §4) |

Surviving hypotheses are capped at **`_MAX_LLM_HYPOTHESES = 3`** per call
(ranked by relevance), then promoted/scored again via
`causal_graph.evaluate_root_cause_eligibility`, and a single **leading
hypothesis** is selected by `causal_graph.select_authoritative_leading_hypothesis`
(deterministic — never trusts the LLM's own pick).

5-Why steps go through an equivalent gauntlet (`is_circular_why_answer`,
`repeats_previous_why_answer`, `restates_observation`,
`ungrounded_entities`, `classify_mixed_evidence_answer`,
`answer_asserts_verified_but_is_reported`) and the chain is **truncated**
(not padded) the moment a step can't be validated — an incomplete,
evidence-bounded 5-Why is preferred over a fabricated complete one.

### 3.6 `critic` — [nodes/critic.py](nodes/critic.py)

A **deterministic pre-gate** first: if `analysis_mode != "DEGRADED"` and no
structural concern is found (ungrounded entity/domain in the narrative or
any hypothesis, empty hypothesis list, empty 5-Why), the critic is skipped
entirely (`critic_status = "SKIPPED"`, 0ms). It only spends an LLM call
when something a keyword/regex guard *could* have missed genuinely needs
semantic judgment (e.g. a hallucinated mechanism using only ordinary,
already-grounded vocabulary).

When it does run, it can:
- **Approve** the analysis.
- **Drop specific hypotheses** it judges unsupported (`unsupported_hypothesis_ids`).
- **Replace the root-cause narrative** with `SAFE_ROOT_CAUSE_FALLBACK` if it
  judges the narrative itself unsupported, forcing `status = NOT_ESTABLISHED`.
- **Send the case back** for another investigation round (capped by
  `agent_max_critic_iterations`).

A critic LLM failure never demotes `analysis_mode` — the critic is a
secondary check on an already-successful synthesis, not the source of
truth (`critic_status = "UNAVAILABLE"`).

### 3.7 `generate_report` — [nodes/report_generator.py](nodes/report_generator.py)

Pure assembly, no LLM call. Computes:
- `observation_quality`, `observation_confidence`, `root_cause_confidence`,
  `overall_confidence` (root cause `NOT_ESTABLISHED`/`STATED_UNVERIFIED`/
  `CONTRADICTED` forces confidence to `LOW`, never higher — see
  [§5.4](#54-confidence-fields)).
- `investigation_required` (`YES`/`NO`/`LIMITED`).
- `final_state` (the closed `AgentFinalState` enum — see §5.7).
- A **final grounding sweep** (`_final_grounding_sweep`) — defense-in-depth
  that re-checks every field against the finding text one more time, in
  case any earlier guard was bypassed; violations are logged as an error
  (`"caught a violation that should have been caught upstream"`) since they
  should be structurally impossible.

### 3.8 `final_evidence_verification` — [nodes/final_evidence_verification.py](nodes/final_evidence_verification.py)

Runs **last**, after `generate_report`, and mutates the report's embedded
objects in place. This is the analytical validation firewall — the last
stop before an auditor sees the output. Highlights:

- **Sanitizes ungrounded specifics**: strips invented SOP identifiers,
  fabricated revision claims, ungrounded patient/customer/product-safety
  language, ungrounded severe actions (recall/quarantine/halt), ungrounded
  training recommendations, ungrounded population expansion ("all
  personnel"), verbatim finding-text leakage into templated sentences —
  all conditioned on whether the finding text actually mentions the
  underlying concept.
- **Derives investigation questions from hypotheses** when the plan (built
  before `core_synthesis` ran) has none — `analytical_validator.derive_investigation_questions`.
- **Deduplicates and caps questions**: max 3 current-event + max 1
  recurrence question (`analytical_validator.deduplicate_investigation_questions`).
- **LOW-specificity gate**: if the finding has no entity/date/period, no
  reported statement, and no established mechanism, *every* non-recurrence
  hypothesis is stripped, root cause is forced to `NOT_ESTABLISHED`/`LOW`
  confidence, and the investigation plan is replaced with 5 foundational
  (non-presupposing) questions — generating a specific hypothesis here
  would misrepresent a data-empty allegation as causally analyzed.
- **Per-hypothesis firewall** (re-applies and extends the core_synthesis
  guards, since this runs after the critic may have modified things):
  drops hypotheses that assert content of a referenced-but-unavailable
  document, narrate their own "supporting evidence" inline, escalate to a
  systemic/process claim with no process evidence, use a generic causal
  bucket phrase, assert unhedged unverified completion, assert unhedged
  notification failure, or assert an unlicensed change-event defect.
- **Deterministic status/strength assignment**: `causal_guard.determine_hypothesis_status`
  recomputes each hypothesis's status/evidence_strength from the ledger —
  never trusts the LLM's self-reported status.
- **Traceability firewall**: a hypothesis can never end up `SUPPORTED`
  without a citable overlapping VERIFIED fact; if none can be found it's
  downgraded to `POSSIBLE` even if every earlier check passed.
- **Cap of 4 current-event hypotheses** (recurrence hypotheses exempt),
  keeping the highest-ranked/highest-scoring.
- **Evidence-grounded backfill**: if every hypothesis for this call got
  rejected but the finding has real specificity, the deterministic
  generator backfills hypotheses rather than shipping an empty list.
- **Hypothesis-ID consistency firewall**: any investigation question or
  CAPA conditional action referencing a hypothesis ID that didn't survive
  filtering is dropped (never invents a replacement).
- **Causal-proposition eligibility layer** (`causal_model.py`) —
  independently recomputes each hypothesis's `support_level` from the
  structured claim ledger; a hypothesis with only `NONE`/`INDIRECT` support
  (topical relatedness, not causal support) is **demoted to an
  investigation area** rather than asserted as a candidate cause.
- **Semantic consistency validator** (`semantic_validator.py`) — confirms
  the canonical finding subject actually survived into the final analysis
  and impact's `affected_object` didn't regress to a placeholder.

---

## 4. The causal reasoning engine

Three modules formalize "how sure can the agent actually be about this?" so
that certainty is *computed*, never merely asserted by an LLM.

### 4.1 [`causal_guard.py`](causal_guard.py) — structural, sentence-level guards

Pattern/structure-based checks (not hardcoded domain vocabulary) used
throughout §3.5–3.8: mechanism polarity classification, immediate-mechanism
extraction, contradiction detection, circular/repeating/restating-answer
detection, unsupported-causal-specificity detection, self-referential
evidence detection, systemic-escalation detection, notification/change-
event-inversion detection, and the deterministic `determine_hypothesis_status`
policy.

### 4.2 [`causal_graph.py`](causal_graph.py) — formal root-cause eligibility engine

Implements the evidence-backed progression:

```
OBSERVATION → REPORTED_MECHANISM → POSSIBLE_HYPOTHESIS → SUPPORTED_HYPOTHESIS → ESTABLISHED_ROOT_CAUSE
```

- `evaluate_root_cause_eligibility()` — deterministic promotion criteria
  from evidence/conflicts/referenced-docs to a `SupportLevel` + `CausalLevel`.
- `select_authoritative_leading_hypothesis()` — deterministic selection: 0
  candidates → `NONE`, a genuine tie → `NONE`/`TIED` (never force-picks a
  winner), a clearly best one → `SELECTED`.
- `generate_structured_conflict_text()` — precise, neutral descriptions of
  detected evidence conflicts (delivery-vs-receipt, training, SOP/checklist).
- Enforces strict state separation: `DELIVERY != RECEIPT != ACKNOWLEDGEMENT != ACTION`.

### 4.3 [`causal_model.py`](causal_model.py) — structured claim/proposition model

- `Claim` / `ClaimType` (`OBSERVED_FACT`, `REPORTED_CAUSAL_MECHANISM`,
  `REPORTED_STATE`, `VERIFIED_CONTROL_FAILURE`, `VERIFIED_EVENT`,
  `VERIFIED_RECORD_STATE`) — classifies each evidence-ledger item.
- `compute_support_level()` — the formalized "does this claim actually
  support this hypothesis" computation: `NONE` → `INDIRECT` (topical
  overlap only) → `REPORTED_SUPPORT` → `VERIFIED_SUPPORT`, with explicit
  subject-word exclusion so a hypothesis doesn't get credit merely for
  repeating the finding's own subject nouns.
- `derive_hypothesis_eligibility()` — only `DIRECT_SUPPORT`/`VERIFIED_SUPPORT`/
  `REPORTED_SUPPORT` make a proposition eligible to remain a hypothesis;
  everything else becomes an investigation area instead.
- `derive_causal_level()` / `derive_root_cause_status()` — where a
  proposition sits on the causal ladder, so an `IMMEDIATE_MECHANISM` claim
  and a `SYSTEMIC_CAUSE` claim about the same finding are never scored
  against each other as competing "leading hypothesis" candidates.

### 4.4 Supporting modules

| Module | Role |
|---|---|
| [`claim_extractor.py`](claim_extractor.py) | Breaks finding text into individually-attributed `EvidenceClaim`s; detects conflicts (`DELIVERY_VS_RECEIPT`, `RECORD_VS_STATEMENT`, `SYSTEM_RECORD_VS_HUMAN_REPORT`, etc.) between claims about the same proposition. |
| [`proposition_engine.py`](proposition_engine.py) | Classifies `InvestigationMode` (NORMAL/CONFLICT/LOW_SPECIFICITY/DOCUMENT_UNAVAILABLE/...) and `EvidenceCompleteness`; builds the canonical `Proposition` list from the ledger. |
| [`grounding_guard.py`](grounding_guard.py) | Post-generation hallucination guard: `ungrounded_entities()` (numbers/proper nouns not in source text), `mentions_unsupported_domain()` (fixed list of QMS domains a finding must actually invoke), `build_source_text()` (the canonical "what's actually grounded" corpus every guard checks against). |
| [`analytical_validator.py`](analytical_validator.py) | Leading-hypothesis selection/scoring, investigation-question derivation/dedup, per-hypothesis confidence grading (`hypothesis_confidence` — never LLM-asserted), CAPA↔hypothesis causal-linkage validation, 5-Why/mechanism-skip repair. |
| [`semantic_validator.py`](semantic_validator.py) | Deterministic (no LLM) check that the canonical finding subject survived into the final report and wasn't replaced by a placeholder. |
| [`recurrence_guard.py`](recurrence_guard.py) | Detects recurring findings and previous-CAPA references; the key invariant enforced is that a previous CAPA being marked "COMPLETED" is **never** treated as proof it was *effective*. |

---

## 5. Output schema — what the auditor actually sees

Everything below lives in [`app/models/agent.py`](../models/agent.py) and is
returned by `POST /api/v1/investigate` as `InvestigateResponse`:

```
InvestigateResponse
├── final_state        AgentFinalState enum         (§5.7)
├── report              InvestigationReport | null   (§5.1–5.6)
├── ca_draft            CADraft | null                (§5.8)
├── trace                list[AgentTraceStep]          (§5.9)
└── ai_metadata          model, prompt_version, generated_at, suggestion_id
```

### 5.1 Root cause — `report.root_cause` (`RootCauseAnalysis`)

| Field | What it shows |
|---|---|
| `status` | One of `RootCauseStatus`: `VERIFIED`, `SUPPORTED`, `ESTABLISHED`, `STATED_UNVERIFIED`, `INFERRED`, `NOT_ESTABLISHED`, `CONTRADICTED`, `CONFLICTED`, `INSUFFICIENT_EVIDENCE`. |
| `category` | 6M-taxonomy value, forced to `TO_BE_CONFIRMED` whenever status is `NOT_ESTABLISHED` (certainty-monotonicity lock). |
| `leading_hypothesis` / `leading_hypothesis_status` | Which hypothesis (by ID) leads, and *why* it's blank when it is — `SELECTED` (a clear winner), `TIED` (candidates are genuinely equally plausible — distinct from "none generated"), or `NONE`. |
| `candidate_hypotheses` | list of `CandidateHypothesis` — see below. |
| `narrative` | The prose RCA summary, sanitized by every guard in §3.8. |
| `root_cause_basis` | *Why* the status is what it is — separate from the narrative. |
| `evidence_required` | What would need to be verified to establish the cause, populated regardless of current status. |
| `confidence` | `LOW`/`MEDIUM`/`HIGH` — forced `LOW` whenever status is `NOT_ESTABLISHED`/`STATED_UNVERIFIED`/`CONTRADICTED`. |

Each **`CandidateHypothesis`** the auditor sees carries:
`id`, `name`, `statement`, `status` (`POSSIBLE`/`SUPPORTED`/`REFUTED`/
`UNRESOLVED`/`UNVERIFIED` — deterministically computed, never LLM-trusted),
`relevance_rank` (`HIGH`/`MEDIUM`/`LOW`), `rationale`, `supporting_evidence`
/ `contradicting_evidence` (actual claim text, traceable), `confirms_if` /
`refutes_if` (what would resolve it), `evidence_strength`
(`NONE`/`REPORTED`/`CORROBORATED`/`VERIFIED`/`CONFLICTING`), and a
deterministic per-hypothesis **`confidence`** grade the LLM never sets
directly.

### 5.2 5-Why — `report.five_why` (`FiveWhyAnalysis`)

Each `FiveWhyStep` has `question`, `answer`, and a `status` drawn from a
closed set: `VERIFIED`, `SUPPORTED`, `REPORTED`, `REPORTED_STATEMENT`,
`REPORTED_UNVERIFIED`, `MIXED` (evidence genuinely conflicts on this
proposition — never a synonym for "uncertain"), `CONFLICTING`,
`CONFLICTING_REPORTS`, `INFERRED`, `UNKNOWN`, `REQUIRES_EVIDENCE`,
`NOT_ESTABLISHED`. The chain **truncates** at the first step that can't be
validated rather than padding to 5 with invented steps —
`is_complete=False` and `status_note` explain why.

### 5.3 Impact — `report.impact_assessment` (`ImpactAssessment`)

`status` (`IMPACT_VERIFIED`/`IMPACT_NOT_IDENTIFIED`/`IMPACT_POSSIBLE`/
`IMPACT_REQUIRES_ASSESSMENT`), plus structured fields
(`affected_object`, `affected_people`, `affected_period`,
`process_at_risk`, `relevant_change`, `potential_effect`,
`evidence_needed`) all derived deterministically
(`_derive_deterministic_impact`) from the canonical finding state and the
already-synthesized root cause — never a second free-form LLM narrative.

### 5.4 Confidence fields

`observation_confidence`, `root_cause_confidence`, `overall_confidence`,
and the legacy `confidence` alias are all computed in
[`report_generator.py`](nodes/report_generator.py) from status values, not
copied from the LLM's own confidence claim.

### 5.5 CAPA — `report.capa` (`CapaAnalysis`)

`status` (`CAPA_RECOMMENDED`/`CAPA_DRAFT_POSSIBLE`/`INVESTIGATION_REQUIRED`/
`INSUFFICIENT_EVIDENCE`/`NO_CAPA_RECOMMENDATION_YET` — in this build almost
always `INVESTIGATION_REQUIRED`, since CAPA stays pending on root cause).
`potential_areas` are derived from the surviving hypothesis set, never
independently invented. `conditional_actions` are evidence-gated branches:
each `ConditionalCapaAction` reads "*If [cause] is confirmed → [specific
action]*" rather than asserting a single unconditional action.

### 5.6 Evidence & provenance — visible on the report for auditability

- `report.evidence` — the full `EvidenceItem` ledger (`claim`, `source`,
  `status`, `relevance`).
- `report.evidence_claims` — claim-level decomposition with attribution
  (`ClaimAttribution`: `AUDITOR_OBSERVED`/`SUPERVISOR_REPORTED`/`AI_INFERENCE`/etc.)
- `report.evidence_conflicts` — detected contradictions between claims,
  each with `conflict_type`, `status` (`UNRESOLVED`/`RESOLVED_FOR`/
  `RESOLVED_AGAINST`), and `severity`.
- `report.evidence_gaps` — what's missing, distinct from what's contradicted.
- `report.propositions` — the canonical `Proposition` graph nodes, each
  tagged with `causal_level` (`L0_OBSERVATION` … `L5_SYSTEMIC_CAUSE`) and
  `support_level`.
- `report.referenced_documents` — documents the finding *cited* but that
  were never inspected (content always `UNKNOWN`, never treated as evidence).
- `report.investigation_mode` — `NORMAL`/`CONFLICT`/`LOW_SPECIFICITY`/
  `DOCUMENT_UNAVAILABLE`/`RECORD_UNAVAILABLE`/`REPORTED_MECHANISM`/
  `TEMPORAL_DEVIATION`/`COMBINED` — tells the auditor at a glance what kind
  of finding this was to reason about.
- `report.analysis_mode` — **`LLM`** (normal), **`DEGRADED`** (primary LLM
  call failed and a deterministic fallback ran — never mistaken for a
  normal result), or **`DETERMINISTIC`** (both primary and recovery calls
  failed).
- `report.provider_used` / `fallback_used` / `provider_attempts` — pure
  observability into which LLM provider actually answered the call (see §6).
- `report.critic_status` — `SKIPPED`/`OK`/`UNAVAILABLE`.

### 5.7 `final_state` (`AgentFinalState`) — the top-level outcome banner

A **closed enum** — the agent cannot invent a new outcome:

| Value | Meaning |
|---|---|
| `READY_FOR_HUMAN_REVIEW` | Normal successful path. |
| `INVESTIGATION_REQUIRED` | Root cause not yet established. |
| `INSUFFICIENT_EVIDENCE` | CAPA status flags insufficient evidence. |
| `CONTRADICTORY_EVIDENCE` | Unresolved evidence conflicts block conclusions. |
| `REQUIRES_HUMAN_INPUT` | Agent needs clarification it can't infer. |
| `TOOL_FAILURE` | A tool/permission error occurred during investigation. |
| `MAX_ITERATIONS_REACHED` | The iteration safety cap was hit. |

### 5.8 CA Draft — `ca_draft` (`CADraft`)

**Exactly 5 fields, enforced with `model_config = {"extra": "forbid"}`** —
this is the only thing the agent is allowed to write into the LQMS
corrective-action form:
`immediate_action`, `root_cause`, `root_cause_category`,
`preventive_action`, `impact_analysis`. Built by
[`permissions.build_ca_draft()`](permissions.py), which validates the
field set in code (`validate_ai_output_fields`) before construction — a
model returning `{"ca_status": "Closed"}` would raise `PermissionError`
rather than reach the response.

### 5.9 Trace — `trace` (`list[AgentTraceStep]`)

A running, timestamped, icon-tagged log of every decision the agent made —
`✓` (ok), `⚠` (warn — something was dropped/downgraded/rejected and why),
`✗` (error). This is what a curious auditor (or a developer) reads to see
*why* the report looks the way it does: every guard in §3–§4 that fires
appends a human-readable line here (e.g. *"Core Synthesis: dropped
hypothesis H2 — contradicts the established mechanism..."*).

---

## 6. Pluggable LLM Provider Architecture & Resilience

The LLM layer is fully decoupled from the LangGraph orchestration and deterministic causal engine:

```
┌─────────────────────────────────────────────────────────┐
│                       LangGraph                         │
│                 SAME FOR ALL PROVIDERS                  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ LLMProvider Interface │
                 │ (app/services/llm)    │
                 └───────────┬───────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
      ┌───────────────────┐     ┌───────────────────┐
      │  Ollama Provider  │     │ Copilot Provider  │
      │ (Local Inference) │     │(github-copilot-sdk│
      └─────────┬─────────┘     └─────────┬─────────┘
                │                         │
                └────────────┬────────────┘
                             ▼
                    Normalized LLMResponse
                             │
                             ▼
                    SAME ANALYSIS STATE
                             │
                             ▼
                     SAME CAUSAL ENGINE
                             │
                             ▼
                    SAME FINAL VALIDATOR
                             │
                             ▼
                      SAME LQMS REPORT
```

### 6.1 Switching Providers (Zero Code Changes)

Provider selection is purely configuration-driven via the `LLM_PROVIDER` environment variable:

#### Development / Testing (Ollama + Qwen3:8b)
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
```

#### Production (Official GitHub Copilot Python SDK)
```bash
LLM_PROVIDER=copilot
COPILOT_MODEL=auto
COPILOT_GITHUB_TOKEN=<securely-injected-token>
COPILOT_TIMEOUT_SECONDS=30.0
```

### 6.2 Provider Setup & Authentication

1. **Ollama Setup (Development):**
   - Install and start Ollama locally (`ollama serve`).
   - Pull the required model: `ollama pull qwen3:8b` (or `qwen2.5:7b-instruct-q4_K_M`).
   - Verify health check at `GET /api/v1/health/llm`.

2. **GitHub Copilot SDK Setup (Production):**
   - The official Python SDK `github-copilot-sdk` is installed via `requirements.txt`.
   - Provide a valid token in `COPILOT_GITHUB_TOKEN` (or `GITHUB_TOKEN` / `GH_TOKEN`).
   - The provider initializes `CopilotClient`, creates dedicated isolated sessions per investigation to eliminate cross-finding context contamination, normalizes responses to `LLMResponse`, and safely cleans up session handles.
   - For live Copilot CI/CD testing, run: `pytest -m live_copilot`.

### 6.3 Deterministic Safety Principle

Regardless of whether Ollama or GitHub Copilot is active:
- **LLM proposes** → **Deterministic engine evaluates** → **Evidence & causal rules decide** → **Final validator approves/rejects**.
- If any provider fails, times out, or returns malformed JSON, the engine enters the deterministic fallback path (`analysis_mode = DETERMINISTIC`).
- `analysis_mode = LLM` is reported only when the LLM successfully completes valid causal analysis.

### 6.4 Multi-Provider Failover Router (Optional Cloud Fallback)

[`app/services/llm_router.py`](../services/llm_router.py) provides optional multi-provider cloud failover when `LLM_PROVIDER` is set to `groq`, `openrouter`, or `gemini`:
- Fixed failover order: **Groq → OpenRouter → Gemini**.
- Per-provider circuit breaker (`ProviderCircuit`, `CircuitState`) with classified cooldowns.
- Surfaced on the report as `provider_used`/`fallback_used`/`provider_attempts` (§5.6).

`app/services/llm_metrics.py` tracks structured validation events (bounded
ring buffers), running averages, and execution telemetry — every guard
rejection/repair increments a counter keyed by `reason` and `node`
(`llm_metrics.record_validation_rejection` / `record_validation_repair`),
which is what `synthesis_execution.validation_rejections`/`validation_repairs`
report for each call.

---

## 7. Security boundaries

| Boundary | Enforced by |
|---|---|
| Tool allowlist | [`tools/registry.py`](tools/registry.py) — `APPROVED_TOOLS`, checked in code both in the planner and again in `execute_tool` before any call. |
| Read-only LQMS access | The ASP.NET client (`aspnet_lqms_client.py`) only exposes GET methods — writes are structurally impossible, not just policy. |
| CA-draft field boundary | [`permissions.py`](permissions.py) — `AI_WRITABLE_FIELDS` (5 fields) vs. `AI_FORBIDDEN_FIELDS` (status, approvals, attachments, etc.); any other field raises `PermissionError`. |
| Prompt injection | `is_instruction()` strips instruction-like text from the finding before it enters the evidence ledger (`understanding.py`). |
| `human_review_required` | Pydantic field validator on `InvestigationReport` — cannot be constructed as `False`. |
| API auth | `X-Internal-Api-Key` via `require_internal_api_key` on the router. |

---

## 8. Tests

Representative behaviors verified in [`backend/tests/`](../../tests/):

- `test_master_causal_architecture.py` — end-to-end causal-architecture assertions across the full pipeline.
- `test_causal_graph_engine.py` — `causal_graph.py`'s eligibility/promotion/leading-hypothesis-selection logic.
- `test_causal_ab_matrix.py` — matrix of evidence-shape scenarios against expected causal outcomes.
- `test_causal_promotion_positive.py` / `test_causal_promotion_negative.py` — cases that should/shouldn't promote a hypothesis to SUPPORTED.
- `test_semantic_ownership_invariants.py` — subject/entity resolution never regresses to a placeholder or drifts to the wrong entity.
- `test_adversarial_llm_outputs.py` — the guards in §3.5/§3.8 actually catch adversarial/malformed LLM JSON.
- `test_audit_observation_evidence_boundary.py` — the Referenced-Evidence Boundary (a cited document is never treated as inspected content).

Run with `pytest` from `backend/`.
