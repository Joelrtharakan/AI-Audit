# LQMS AI Gateway (Backend)

FastAPI service providing an **LLM-only AI intelligence layer** (finding analysis) around
the existing ASP.NET WebForms LQMS. This service is a **system of intelligence**, not a
system of record: it never writes to the production LQMS database, never changes CAPA/audit
workflow state, and never auto-saves anything. Every AI output is a suggestion for a human
auditor to review and edit.

## 0. Implementation status

**LLM-only AI-assisted audit analysis.** The single workflow (`POST /api/v1/analyze-finding`)
reasons about a brand-new auditor observation using the LLM alone. There is no RAG, no
vector database, no embeddings, no document ingestion, no historical-finding similarity
search, and no synthetic knowledge base anywhere in this codebase — all of that was
deliberately removed, not just disabled. The LLM identifies what's known, what's missing,
and what the auditor should investigate, and only claims a root cause or CAPA when the
observation itself actually supports one. It never fabricates organizational facts (SOP
text, policy requirements, prior findings, regulatory clauses) because none are ever given
to it — there is no mechanism in this codebase that could supply them.

**Built and verified (backend running, tests passing):**

- FastAPI core: config, `X-Internal-Api-Key` auth (constant-time compare, fails closed),
  CORS, structured logging (`app/main.py`, `app/config.py`, `app/auth.py`).
- **Two interchangeable LLM providers** behind one interface: OpenRouter (default/
  production) and a local Ollama server (fast dev iteration, no API key, no network).
  Provider is a single env var (`LLM_PROVIDER`) — see section 6b.
  (`app/services/llm_client.py`, `app/services/openrouter_client.py`,
  `app/services/ollama_client.py`).
- Finding analysis + `POST /api/v1/analyze-finding`, decomposed into three narrower,
  independently-calibrated LLM calls — extraction → causation classification →
  generation — plus programmatic enforcement (never trusts the LLM's own status claims
  blindly, actually re-checks Step 2 against Step 1) and programmatic confidence scoring.
  See section 8. (`app/services/finding_analysis_service.py`,
  `app/services/observation_quality.py`, `app/services/extraction.py`,
  `app/services/causation_classifier.py`, `app/services/confidence.py`,
  `app/models/analysis.py`, `app/routers/analyze.py`).
- 6M root-cause taxonomy with LLM-output coercion (`app/services/taxonomy.py`).
- Retry/backoff/429-handling clients plus versioned, anti-hallucination system prompts and
  4 calibration few-shot examples (`app/services/prompt_builder.py`, `app/prompts/*.txt`).
- A golden-set evaluation harness (18 hand-written LQMS findings, `tests/golden/
  findings.jsonl`) that measures root-cause-status and confidence accuracy against the
  live endpoint instead of eyeballing individual responses — `scripts/run_golden_eval.py`,
  `scripts/eval.sh`. See section 15.
- 24 unit/integration tests (finding-analysis pipeline incl. extraction/classification
  unit tests and the 3 calibration scenarios, taxonomy, auth, and explicit checks that no
  RAG code/routes exist) — all passing, with zero third-party runtime dependency beyond
  FastAPI/httpx/pydantic/tenacity (verified by uninstalling chromadb/sentence-transformers/
  pypdf/python-docx/beautifulsoup4/markdown/pyyaml from the venv and re-running the suite).
- Frontend: a single **Analyze Finding** button (`frontend/assets/js/lqms_ai.js`) calling
  this one endpoint. See section 10.

**Out of scope / TODO:**

- The ASP.NET `.ashx` proxy that keeps `INTERNAL_API_KEY` out of the browser in production —
  see `TODO(prod-security)` in section 12 and in `frontend/config.js`.
- RAG, document retrieval, and historical-finding similarity are not planned — this system
  is intentionally LLM-only. Ollama is a second *inference* provider for local speed, not a
  retrieval layer -- it changes nothing about that. If organizational documents ever become
  available and grounding is wanted again, that would be a new, separate design effort, not
  a re-enable of a flag.
- Golden-set accuracy against local `qwen2.5:3b` is currently **72.2%** (13/18), below the
  85% target threshold — see section 15 for the two calibration bugs found and fixed along
  the way (44% → 72.2%) and why the remaining gap is a small-model reasoning limit, not a
  known bug. An `openrouter` comparison run wasn't completed this session (stopped early;
  free-tier rate limits make a full run slow) — run `./scripts/eval.sh` to get that number
  before deciding whether 3B-local calibration is trustworthy beyond fast dev iteration.

## 1. Project purpose

An auditor writes an observation in the existing LQMS form; `POST /api/v1/analyze-finding`
reasons about it directly with the LLM (no organizational documents, no retrieval of any
kind) and returns a root-cause analysis, investigation guidance, and — only when the
observation actually supports it — a CAPA suggestion, for the auditor to review before
anything reaches the real LQMS workflow.

## 2. Architecture

```
Existing LQMS (ASP.NET WebForms / jQuery)
        │ AJAX (X-Internal-Api-Key)
        ▼
   FastAPI (this service)
        │
   POST /api/v1/analyze-finding
        │
   Observation Quality Check (LLM call 1)
        │
   Step 1: Extraction (LLM call 2)         -- facts/claims/records only, no judgment
        │
   Step 2: Causation Classification (LLM call 3)  -- reasons over Step 1's output only
        │
   Step 3: Generation (LLM call 4)         -- five-why/investigation/capa/impact
        │
   Programmatic enforcement (validates Step 2 against Step 1) + confidence scoring
        │
        ▼
   get_llm_client()  ->  OpenRouterClient  or  OllamaClient
```

There is no vector store, no embedding model, and no document/finding corpus anywhere in
this service — every pipeline step talks only to whichever `LLMClient` `get_llm_client()`
hands it (`app/services/llm_client.py`). `tests/test_no_rag.py` asserts this stays true (no
RAG modules on disk, no RAG imports in the analysis pipeline, no RAG routes registered) --
Ollama is a second inference backend, not retrieval, so this guarantee is unaffected by it.

## 3. Frontend setup

The frontend (`../frontend`) is a reference/dev copy of the real ASP.NET WebForms page. It
calls this API from `frontend/assets/js/lqms_ai.js` and is configured via `frontend/config.js`
(`apiBaseUrl`, `internalApiKey`). Serve it with `frontend/dev_server.py` (a thin wrapper
around Python's `http.server` that also shims the handful of ASP.NET PageMethod/`.axd`
requests this legacy page fires on load, which a plain static server can't answer), and make
sure `ALLOWED_ORIGINS` below matches that origin:

```bash
cd frontend
python3 dev_server.py 5500
```

## 4. Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY and INTERNAL_API_KEY
uvicorn app.main:app --reload --port 8000
```

Health check: `GET /health` (no auth required). `POST /api/v1/analyze-finding` requires
header `X-Internal-Api-Key: <INTERNAL_API_KEY>`. These are the only two routes the service
exposes.

## 5. Environment variables

See `.env.example`. Never commit `.env`.

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `openrouter` (default) or `ollama`. See section 6b. |
| `OPENROUTER_API_KEY` | OpenRouter key. Never sent to the browser. |
| `OPENROUTER_MODEL` | Model id, e.g. `openrouter/auto` or a specific model — change without touching code. |
| `OLLAMA_MODEL` | Local model tag (default `qwen2.5:7b-instruct-q4_K_M`). Only used when `LLM_PROVIDER=ollama`. |
| `OLLAMA_BASE_URL` | Local Ollama OpenAI-compatible base URL (default `http://localhost:11434/v1`). |
| `INTERNAL_API_KEY` | Shared secret the frontend sends as `X-Internal-Api-Key`. |
| `ALLOWED_ORIGINS` | CORS allow-list, comma-separated. |
| `LOG_LEVEL` | Python logging level. |
| `ANALYSIS_PROMPT_VERSION` | Stamped onto every AI response for traceability. |

## 6. OpenRouter configuration

`app/services/openrouter_client.py` calls the OpenAI-compatible chat completions endpoint
with retry + exponential backoff on `429`. Swapping models is a config change
(`OPENROUTER_MODEL`), never a code change — and, per section 6b, swapping the *provider*
entirely (OpenRouter vs. local Ollama) is the same kind of config change, not a code change.

## 6b. Local development with Ollama

For fast local iteration without burning OpenRouter credits or waiting on network latency,
point the same pipeline at a local Ollama server instead:

```bash
ollama serve                                   # if not already running
ollama pull qwen2.5:7b-instruct-q4_K_M         # or any instruction-tuned model you have
```

```env
# .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_K_M
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Restart `uvicorn` and every LLM call in the pipeline (observation quality, extraction,
classification, generation) now goes to `app/services/ollama_client.py` instead of
OpenRouter — no code path changes, since every step type-hints against the shared
`LLMClient` protocol and gets its client from `get_llm_client()`
(`app/services/llm_client.py`), which is the *only* place that branches on
`LLM_PROVIDER`. No API key is sent (Ollama is local and unauthenticated).

On startup, if `LLM_PROVIDER=ollama`, the app does a quick `GET` against the Ollama base
URL and logs a clear warning — not a crash — if nothing answers:

```
LLM_PROVIDER=ollama but http://localhost:11434 is not reachable. Is `ollama serve` running?
```

**Ollama is for local dev only, never production.** Keep `LLM_PROVIDER=openrouter` (the
default) for anything deployed. Model quality/calibration also differs between providers —
use `./scripts/eval.sh` (section 15) to check the local model's golden-set accuracy before
trusting it for anything beyond fast iteration; a small local model may need `openrouter` as
a sanity check before shipping a prompt change.

This machine has `qwen2.5:3b` and `qwen3:8b` pulled, not the documented default
`qwen2.5:7b-instruct-q4_K_M` — `.env` here is set to `OLLAMA_MODEL=qwen2.5:3b` for speed.
Pull the 7b model and update `.env` if you want the documented default; either way it's a
one-line config change.

## 7. Root-cause taxonomy

`app/services/taxonomy.py` defines a fixed 6M taxonomy (MAN, METHOD, MACHINE, MATERIAL,
MEASUREMENT, ENVIRONMENT_MGMT, OTHER). The Step 2 classifier is instructed to use it, and
`coerce_category()` forces any out-of-taxonomy value the LLM might still produce back to
`OTHER` rather than trusting it — see `causation_classifier.py::_to_classification`.

## 8. Finding analysis pipeline

```bash
POST /api/v1/analyze-finding
{"finding_text": "...", "department": "...", "branch": "...", "standard": "...",
 "clause": "...", "finding_type": "...", "nature_of_nc": "...",
 "risk_severity": "...", "risk_likelihood": "...", "risk_result": "..."}
```

Only `finding_text` is required — every other field is optional, matching what the existing
frontend actually has available per finding (there's no reliable per-finding "branch" field
in `frontend/index.html`, for example, so the frontend always sends `""` for it).

Pipeline (`app/services/finding_analysis_service.py::FindingAnalysisService.analyze`) —
**four sequential LLM calls**, each narrower and more reliably calibrated than one big
"analyze everything" call:

0. `observation_quality.check_observation_quality()` asks the LLM whether the observation
   has enough detail for meaningful analysis (`SUFFICIENT` / `INSUFFICIENT` +
   `missing_information`). Fails *safe* to `INSUFFICIENT` if this call itself fails.
1. **Extraction** (`extraction.py::extract_finding`) — pulls `stated_facts`,
   `attributed_statements` (`{speaker, claim}`), `referenced_records`, `timeframe`, and
   `asset_or_location` out of the raw text. No causal judgment at all -- purely "what does
   the text say and who said what." Validated with a lightweight grounding check (do the
   extracted phrases' significant words actually appear in the source text?) and retried
   once if the LLM introduces content absent from the observation; raises cleanly if still
   ungrounded after the retry.
2. **Causation classification** (`causation_classifier.py::classify_causation`) — reasons
   ONLY over the Step 1 `ExtractionResult` (never the raw text again), calibrated against 4
   worked examples in `app/prompts/causation_classification_fewshot.txt`. Produces
   `root_cause_status` (`ESTABLISHED` / `SELF_REPORTED` / `NOT_ESTABLISHED`) + `category` +
   one-line `reasoning`:
   - `ESTABLISHED` — at least one `referenced_records` entry corroborates the cause.
   - `SELF_REPORTED` — at least one `attributed_statements` entry names a specific cause,
     but nothing in `referenced_records` corroborates it. Surfaced as a labeled hypothesis
     for the auditor to verify — deliberately *not* collapsed into `NOT_ESTABLISHED` (which
     would under-credit real information the observation gave) and deliberately *not*
     promoted to `ESTABLISHED` (which would treat an unverified claim as confirmed).
   - `NOT_ESTABLISHED` — neither of the above.
3. **Generation** (`prompt_builder.build_finding_analysis_messages` + the main LLM call) —
   takes finding_text + Step 1 extraction + Step 2 classification as fixed input and
   produces `five_why`, `investigation`, `capa`, `impact_analysis`,
   `possible_contributing_factors`. **It does not decide root_cause.status itself** — the
   schema no longer even has a `root_cause` field, closing the gap where status and content
   could drift apart. Invalid/incomplete responses get one stricter retry, then a clean
   `LLMError`.

**Programmatic enforcement, not just prompting** — and it now does real validation, not
just post-hoc stripping:

- `_build_root_cause()` re-checks Step 2's classification against Step 1's extraction and
  *downgrades* it if unsupported: a claimed `ESTABLISHED` with no `referenced_records` in
  the extraction is downgraded to `SELF_REPORTED`; a claimed `SELF_REPORTED` with no
  `attributed_statements` is downgraded to `NOT_ESTABLISHED`. The classifier's own claim is
  never trusted at face value.
- `_build_capa()` enforces that `capa.status` can only be `AI_SUGGESTED` when
  `root_cause.status` is truly `ESTABLISHED` (after the above downgrade) — if the
  generation step tries to auto-suggest a CAPA off a merely `SELF_REPORTED` cause, the code
  forces it back to `INVESTIGATION_REQUIRED` and blanks `corrective_action`/
  `preventive_action` regardless of what the LLM returned.
- `RootCause.is_hypothesis` (`true` whenever status != `ESTABLISHED`) and
  `RootCause.verification_needed` (`true` only for `SELF_REPORTED`) are computed fields, so
  frontend/report logic that only cares "is this confirmed" doesn't have to re-derive it
  from the status string.

`confidence.calculate_confidence()` computes `HIGH`/`MEDIUM`/`LOW` from objective signals
(observation quality, root-cause status, whether CAPA could be established, finding-text
word count, five-why depth, count of open investigation questions) — **the LLM never sets
its own confidence**. `SELF_REPORTED` contributes positively (enough to typically reach
`MEDIUM`) but `HIGH` is hard-capped to require `root_cause.status == ESTABLISHED`, no matter
how the other signals stack up.

Response always includes static `ai_limitations` text making clear this is "AI-assisted
audit analysis," not an evidence-grounded organizational CAPA.

Never writes to the LQMS. On any LLM failure, returns a clean `502` with existing form data
untouched — see `app/routers/analyze.py`.

## 9. HTML output validation

Every free-text field the LLM returns is passed through `strip_html()`
(`app/services/llm_json.py`) before it's allowed anywhere near the frontend's form fields.

## 10. Frontend integration

`frontend/assets/js/lqms_ai.js` wires the single **Analyze Finding** button
(`frontend/index.html`) to `/api/v1/analyze-finding`. It reads `#txtFindingObsn` /
`#txtFindingOrObsn`, `#lblAuditCriteria`, `#lblFindingType`, `#lblNatureOfNC`,
`#lblDepartments`, `#lblClauseNo`, `#lblRiskS`, `#lblRiskL`, `#lblRiskResult`; renders the
full structured result into `#lqmsAiResultsPanel` (observation quality, confidence, root
cause status — including a distinct "STATED — UNVERIFIED" badge for
`SELF_REPORTED` — possible factors, investigation areas/questions/evidence, 5-why,
CAPA or investigation guidance, impact assessment, AI limitations warning); and
**conditionally populates form fields** — `#txtCorrectiveAction` + the root-cause dropdown
only when `root_cause.status === "ESTABLISHED"`, and `#txtRootCause` /
`#txtPreventiveAction` only when `capa.status === "AI_SUGGESTED"`. For both
`SELF_REPORTED` and `NOT_ESTABLISHED`, those fields are deliberately left untouched
and the panel explains why instead — an unverified, self-reported cause is surfaced as a
hypothesis in the panel, never written into the LQMS form as if confirmed. All
AI-populated fields get an "AI Suggested — Review Required" badge that disappears the
instant the auditor edits that field. Nothing here triggers a WebForms postback or
auto-saves; the AJAX call sends `X-Internal-Api-Key` from `config.js`.

## 11. Running tests

```bash
python -m pytest -q
```

24 tests covering: extraction unit tests (structured output, fabricated-content rejection
+ retry, clean failure if still fabricated), causation-classifier unit tests (all 3
statuses), the 3 end-to-end calibration scenarios (established+corroborated → HIGH,
self-reported+uncorroborated → MEDIUM, no-causal-signal → LOW), 2 tests that the
classification-vs-extraction enforcement actually downgrades an unsupported claimed
status, LLM-unavailable → clean error, taxonomy coercion, auth enforcement, and explicit
structural guarantees that RAG is fully gone (`tests/test_no_rag.py`: no RAG modules on
disk, no RAG imports in the analysis pipeline, no RAG routes registered) — all using a
fake LLM client, so they run in well under a second with no network access or model
download required.

## 12. Security

**TODO(prod-security)**: In production, the browser must never hold `INTERNAL_API_KEY`
(see `frontend/config.js`). Route the call through a thin ASP.NET `.ashx` proxy that injects
`X-Internal-Api-Key` server-side. `INTERNAL_API_KEY` is validated with a constant-time
comparison (`hmac.compare_digest`) in `app/auth.py`; a server missing the env var fails
closed (`503`), not open.

## 13. AI traceability

Every `AnalyzeFindingResponse.ai_metadata` includes `suggestion_id`, `model` (the actual
provider/model that ran, e.g. `qwen2.5:3b` under Ollama or `openai/gpt-oss-20b:free` under
OpenRouter), `generated_at`, and `prompt_version`. Every `RootCause` includes
`classification_reasoning` (the Step 2 classifier's one-line explanation) so a
`SELF_REPORTED` or `ESTABLISHED` call can always be traced back to *why*. The full
`ExtractionResult` is also returned (`AnalyzeFindingResponse.extraction`) so an auditor (or
the golden-eval harness) can see exactly what the pipeline extracted from the raw text
before any classification happened.

## 14. Production architecture

```
Browser → ASP.NET → ASP.NET .ashx proxy → FastAPI → OpenRouter
```

The proxy step (not yet built) is what keeps `INTERNAL_API_KEY` out of the browser in
production; local dev talks to FastAPI directly per `frontend/config.js`. Production always
uses `LLM_PROVIDER=openrouter` -- Ollama never appears in this path (section 6b).

## 15. Golden-set evaluation harness

Prompt and enforcement-logic changes are measured against `tests/golden/findings.jsonl` (18
hand-written LQMS findings spanning: no causal claim, self-reported single/conflicting
causes, corroborated cause, implied-but-unstated cause, garbled/short text, no department
given, multi-day recurring issues, a one-off equipment failure, mixed technical jargon, and
department/standard variety across Laboratory/Quality/Facilities/Pharmacy/HR/Nursing) --
instead of eyeballing individual responses.

```bash
# Run against whichever provider the backend currently has configured:
python scripts/run_golden_eval.py --base-url http://localhost:8000 --api-key "$INTERNAL_API_KEY"

# Run against both providers and diff the reports side by side (starts Ollama and the
# backend itself, with each provider in turn):
./scripts/eval.sh
```

`run_golden_eval.py` compares each result's `root_cause.status` against `expected_status`
and `analysis.confidence` against a minimum bar (`expected_confidence_min`, ordinal
LOW < MEDIUM < HIGH), then prints: overall accuracy, a table of mismatches (with the
classifier's `reasoning` excerpt so you can see *why* it decided what it decided), and a
confidence-level distribution histogram with a warning if more than 70% of results land in
one bucket -- the exact "confidence collapsed to LOW/MEDIUM for everything" symptom this
three-step decomposition and the `confidence.py` scoring fix (see below) were built to
catch. Exits non-zero if accuracy is below `--threshold` (default `0.85`), so it can gate CI.

**Real calibration bugs this harness caught during development**, against `qwen2.5:3b`
locally (this machine's pulled model -- see section 6b):

- **Run 1: 44% accuracy, confidence 100% collapsed into `LOW`** -- even for findings the
  classifier correctly marked `ESTABLISHED` or `SELF_REPORTED`. Cause:
  `observation_quality` (a cheap upfront heuristic) and `root_cause_status` (the result of
  the extraction+classification steps actually working the text) are correlated, not
  independent signals, but the confidence formula penalized `INSUFFICIENT` at full weight
  even when classification had already found and corroborated a real cause -- effectively
  penalizing the same weak signal twice.
- **Fix 1** (`confidence.py`): when the two disagree, trust the step that did real work
  (classification) and don't apply the quality penalty a second time; only penalize when
  both agree there's nothing there. Rerun: 44% → still ~44%, but confidence spread slightly
  (a second, related bug remained).
- **Fix 2** (`confidence.py`): `missing_information_count` and `open_questions_count` were
  *also* penalizing the same redundant signal a third time -- and open questions are
  near-guaranteed to be non-zero for `SELF_REPORTED` by design (the generation prompt
  requires a verification checklist there), so this was punishing the pipeline for doing
  exactly what it was told to do. Fixed to only apply those two penalties when
  `root_cause_status == NOT_ESTABLISHED`, where "lots of open questions" genuinely does
  mean "this is thin."
- **Result after both fixes: 72.2% accuracy (13/18), confidence spread 56% LOW / 37.5%
  MEDIUM / 6.2% HIGH** -- no more single-bucket collapse. Still below the 85% threshold;
  the remaining 5 mismatches are genuine reasoning misses from a small (3B, CPU) local
  model on nuanced cases (conflicting self-reported causes, a corroborating record phrased
  indirectly) plus two malformed-JSON failures after retry -- not further formula bugs.
  This is exactly the "good enough for fast dev iteration, verify against a cloud model
  before trusting it" signal `./scripts/eval.sh` and section 6b are for; a larger local
  model (e.g. the documented `qwen2.5:7b-instruct-q4_K_M` default) or `openrouter` should
  be checked before treating 3B-local calibration as final. (An `openrouter` comparison
  run was started during development and intentionally stopped before completion --
  `openai/gpt-oss-20b:free`'s rate limits made a full 18-item run slow; rerun
  `./scripts/eval.sh` when you want that number.)

This progression -- real bugs found, fixed, and *measured*, not eyeballed -- is the actual
point of this harness.
