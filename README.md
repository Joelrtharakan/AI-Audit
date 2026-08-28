# LQMS Corrective Action Investigation Agent & Financial Exposure Engine

An autonomous, evidence-grounded LangGraph agent that transforms raw audit findings into auditor-facing **Investigation Reports**, **5-field Corrective Action (CA) drafts**, and rigorous **Cost & Financial Exposure Analyses**.

It adheres to strict epistemic boundaries: **the AI never writes to the production LQMS, never invents financial figures, and every output is a human-gated draft.**

- **Entry Point**: `POST /api/v1/investigate` ([backend/app/routers/investigate.py](backend/app/routers/investigate.py))
- **Financial Analysis API**: `POST /api/v1/financial/analyze` ([backend/app/routers/financial.py](backend/app/routers/financial.py))
- **Authentication & Security**: Microsoft Entra ID delegated OAuth 2.0 + optional tenant gate ([backend/app/auth/microsoft_entra.py](backend/app/auth/microsoft_entra.py))
- **Graph Orchestration**: [`app/agent/graph.py`](backend/app/agent/graph.py)
- **Financial & Cost Factor Engine**: [`app/financial/`](backend/app/financial/)

---

## Quick Start (Running Locally)

### 1. Backend Server (FastAPI on Port 8010)
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8010
```
- **API Base**: `http://localhost:8010`
- **Interactive OpenAPI Docs**: `http://localhost:8010/docs`
- **LLM Provider Health**: `GET http://localhost:8010/api/v1/health/llm`

### 2. Frontend Application (Dev Server on Port 5510)
```bash
cd frontend
python3 dev_server.py
```
- **Application Dashboard**: `http://localhost:5510/index.html`
- **Microsoft Entra Sign-in**: `http://localhost:5510/login.html` (or `http://localhost:8010/api/auth/microsoft/login`)

---

## 1. End-to-End System Architecture

```
                                  USER BROWSER / AUDITOR WORKFLOW
                  ┌─────────────────────────────────────────────────────────────┐
                  │ • Legacy ASP.NET Master UI + Modern Responsive Tailwind CSS │
                  │ • Microsoft Entra Sign-in Button & Tenant Access Boundary   │
                  │ • Live Investigation Dashboard, RCA Visualizer, CAPA Draft  │
                  │ • Cost Factor & Financial Exposure Breakdown (PAF/Scenarios)│
                  └──────────────────────────────┬──────────────────────────────┘
                                                 │ HTTP / JSON API
                                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FASTAPI BACKEND RUNTIME (:8010)                                 │
│                                                                                                  │
│ ┌────────────────────────────────────┐         ┌───────────────────────────────────────────────┐ │
│ │  AUTH & SECURITY BOUNDARIES        │         │  AUDIT INVESTIGATION ORCHESTRATION (LangGraph)│ │
│ │  • Microsoft Entra ID (delegated)  │         │  1. understand_finding (Intake & Claims)      │ │
│ │  • Optional Tenant Allow-list Gate │         │  2. plan_investigation (Read-only tools)      │ │
│ │  • HMAC Signed Nonce Cookies       │         │  3. execute_tool (Allowlist GET queries)      │ │
│ │  • X-Internal-API-Key Gateway      │         │  4. record_evidence (Hallucination Firewall)  │ │
│ └────────────────────────────────────┘         │  5. core_synthesis (Unified RCA + 5-Why)     │ │
│                                                │  6. critic (Deterministic Pre-Gate & Guard)  │ │
│ ┌────────────────────────────────────┐         │  7. generate_report (Prose Assembly)         │ │
│ │  EVIDENCE & CAUSAL ENGINE          │         │  8. final_evidence_verification (Firewall)    │ │
│ │  • Claim Extraction & Attribution  │         └───────────────────────┬───────────────────────┘ │
│ │  • Evidence Ledger & Conflicts     │                                 │                         │
│ │  • Monotonic Certainty Progression │                                 ▼                         │
│ │  • Non-Causal Proposition Demotion │         ┌───────────────────────────────────────────────┐ │
│ └────────────────────────────────────┘         │  COST FACTOR & FINANCIAL EXPOSURE ENGINE      │ │
│                                                │  • Dual-Engine: Semantic LLM + Regex Fallback │ │
│ ┌────────────────────────────────────┐         │  • Grounded Cost Factor Classifier (PAF)      │ │
│ │  PLUGGABLE LLM PROVIDER ADAPTER    │         │  • Epistemic Separation (Verified vs Reported)│ │
│ │  • Ollama (local / fallback)       │         │  • Recovery Safety & Scenario Analysis        │ │
│ │  • Microsoft 365 Copilot (LiteLLM) │         │  • Deterministic Arithmetic (No LLM Math)     │ │
│ │  • Multi-Provider Cloud Failover   │         │  • CAPA Payback & Return on Investment        │ │
│ └────────────────────────────────────┘         └───────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed LangGraph Pipeline Architecture

```
POST /api/v1/investigate
        │
        ▼
┌───────────────────┐
│  understand_finding │  Deterministic obs. quality + LLM extraction +
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
                            │  generate_report     │  Deterministic assembly,
                            │                      │  no LLM call
                            └──────────┬───────────┘
                                       ▼
                            ┌────────────────────────────┐
                            │ final_evidence_verification │  The analytical
                            │                              │  firewall — last
                            │                              │  stop before auditor
                            └──────────────┬───────────────┘
                                           ▼
                                          END
                            → InvestigateResponse (report, ca_draft, trace)
```

---

## 3. Cost Factor & Financial Exposure Engine

The system features an autonomous **Evidence-Grounded Cost & Financial Exposure Engine** ([`app/financial/`](backend/app/financial/)). It computes comprehensive economic exposure while enforcing a strict boundary: **the LLM is only permitted to perform semantic extraction and propose structural relationships; all calculations, conversions, and annualizations are executed deterministically in code.**

### 3.1 Cost Factor Analysis Architecture

```
                             AUDIT FINDING & EVIDENCE LEDGER
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │                                             │
                     ▼ (Enabled)                                   ▼ (Disabled / Fallback)
         ┌───────────────────────┐                     ┌───────────────────────┐
         │  LLM Semantic Engine  │                     │  Regex/Pattern Engine │
         │ (semantic_engine.py)  │                     │     (extractor.py)    │
         └───────────┬───────────┘                     └───────────┬───────────┘
                     │                                             │
                     ▼                                             ▼
          Structured Interpretation                     Regex-Extracted Tokens
          (Claims, Relationships,                       (Amounts, Currencies,
           Calculations, Cost Factor)                    Periods, Quantities)
                     │                                             │
                     ▼                                             ▼
         ┌────────────────────────┐                    ┌───────────────────────┐
         │ Relationship Validator │                    │ Currency & Population │
         │(relationship_validator)│                    │       Isolation       │
         └───────────┬────────────┘                    └───────────┬───────────┘
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
                              Validated FinancialObservation
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │      DETERMINISTIC FINANCIAL CALCULATOR       │
                    │                 (calculator.py)               │
                    ├───────────────────────────────────────────────┤
                    │ 1. Confirmed Financial Impact                 │
                    │    - Verified Gross vs. Reported Exposure     │
                    │    - Verified Recovery vs. Net Loss           │
                    │ 2. Cost Factor Classification (PAF Taxonomy)  │
                    │ 3. Potential Exposure & Range Bounds          │
                    │ 4. Annualized & Historical Recurrence Model   │
                    │ 5. Scenario Modeling (Conservative/Exp/High)  │
                    │ 6. CAPA Economic Payback & Avoided Loss       │
                    │ 7. Multi-Dimensional Confidence Grading       │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                                 FinancialAnalysisResult
```

### 3.2 Standard Cost Factor Taxonomy (PAF Model)

Costs are automatically categorized into the classic **Prevention, Appraisal, and Failure (PAF)** quality costing framework:

| PAF Category | Cost Factor (`FinancialAmountType`) | Description & Trigger Conditions |
|---|---|---|
| **Internal Failure** | `REWORK_COST` | Cost of labor, machine time, or materials required to reprocess defective output before release. |
| **Internal Failure** | `SCRAP_COST` | Direct financial loss of discarded materials, unrecoverable product, or expired inventory. |
| **Internal Failure** | `DOWNTIME_COST` | Line stoppage, idle machine hours, or facility delay costs resulting from the failure. |
| **External Failure** | `CUSTOMER_COMPENSATION` | Direct restitution, refunds, customer credits, or recall costs. |
| **External Failure** | `PENALTY` | Contractual penalties, regulatory fines, or non-conformance fees assessed by external bodies. |
| **External Failure** | `REVENUE_IMPACT` | Unrecovered billing, lost sales, or uncollectible receivables. |
| **Prevention & Remediation**| `REMEDIATION_COST` | Immediate containment, rework tooling, or corrective software patch deployment costs. |
| **Prevention & Remediation**| `PREVENTION_COST` | Long-term CAPA investment, new poka-yoke controls, staff retraining, or QMS redesign. |
| **Direct Accounting**| `DIRECT_LOSS` / `OVERPAYMENT` / `DUPLICATE_PAYMENT` | Concrete ledger over-disbursements, unrecorded debit notes, or duplicate billing. |

### 3.3 Epistemic Safety Invariants for Financials

1. **Epistemic Separation**: `VERIFIED GROSS EXPOSURE != REPORTED FINANCIAL EXPOSURE != CONFIRMED NET LOSS`.
2. **Monotonic Status Propagation**: `REPORTED` or `UNVERIFIED` claims never combine with verified numbers to assert a "Verified" total.
3. **Recovery Safety Barrier**: Never assume recovery is `0` merely because it is not mentioned; never calculate `confirmed_net_loss` unless **both** the gross exposure and the recovery amount are independently `VERIFIED`.
4. **Historical vs. Current Isolation**: Past finding amounts (historical context) are excluded from the current finding's immediate net loss and feed only recurrence/annualization projection models.
5. **Deterministic Arithmetic**: The LLM never computes products or sums. It emits `{operation: MULTIPLY, left: C1, right: C2}`; the backend validates units/currencies and runs the math.

---

## 4. Node-by-Node Pipeline Detail

### 4.1 `understand_finding` — [nodes/understanding.py](backend/app/agent/nodes/understanding.py)
The intake firewall. Breaks finding text into individually attributed `EvidenceClaim`s, detects contradictory claims (`detect_evidence_conflicts`), strips prompt injections (`is_instruction`), and produces the immutable `CanonicalFindingState`.

### 4.2 `plan_investigation` — [nodes/investigation_planner.py](backend/app/agent/nodes/investigation_planner.py)
Determines which read-only LQMS tools to query from `APPROVED_TOOLS`. Bounded by domain guards to prevent invoking unrelated QMS topics.

### 4.3 `execute_tool` — [nodes/tool_executor.py](backend/app/agent/nodes/tool_executor.py)
Executes tool calls against the read-only ASP.NET mock/live client with per-tool timeouts and hard tool caps. Never raises unhandled exceptions.

### 4.4 `record_evidence` — [nodes/evidence_recorder.py](backend/app/agent/nodes/evidence_recorder.py)
The tool hallucination firewall. Classifies raw tool outputs into structured `EvidenceItem` records tagged with `EvidenceStatus` (`VERIFIED` / `REPORTED` / `UNVERIFIED`).

### 4.5 `core_synthesis` — [nodes/core_synthesis.py](backend/app/agent/nodes/core_synthesis.py)
Single authoritative RCA, 5-Why, hypothesis, and CAPA derivation node.
- Validates candidate hypotheses against 10+ causal guards (refutes, mechanism conflicts, human error overclaims, circular 5-Why answers).
- Caps surviving hypotheses to a maximum of 3 candidates.
- Derives CAPA and CA draft fields deterministically from the surviving root cause.

### 4.6 `critic` — [nodes/critic.py](backend/app/agent/nodes/critic.py)
Deterministic 0ms pre-gate. Only invokes an LLM self-review when structural flags or potential ungrounded entities are detected.

### 4.7 `generate_report` — [nodes/report_generator.py](backend/app/agent/nodes/report_generator.py)
Pure deterministic assembly. Derives multi-dimensional confidence scores and closes the `AgentFinalState` outcome enum.

### 4.8 `final_evidence_verification` — [nodes/final_evidence_verification.py](backend/app/agent/nodes/final_evidence_verification.py)
The final analytical firewall before the auditor views the report:
- Sanitizes ungrounded SOP codes, numbers, and dates.
- Truncates unverified 5-Why steps.
- Demotes ungrounded hypotheses to investigation areas.

---

## 5. Security & Write-Permission Boundaries

| Boundary | Enforcement Mechanism |
|---|---|
| **Read-Only LQMS Interface** | ASP.NET client only implements GET calls; write operations are structurally impossible. |
| **Tool Allowlist** | `APPROVED_TOOLS` in [`tools/registry.py`](backend/app/agent/tools/registry.py) re-checked at runtime before execution. |
| **Strict 5-Field CA Draft** | `AI_WRITABLE_FIELDS` in [`permissions.py`](backend/app/agent/permissions.py) forbids AI writes to approval, status, or assignment fields. |
| **Human Review Lock** | Pydantic validator enforces `human_review_required = True` on all reports. |
| **Microsoft Entra ID sign-in** | Delegated OAuth 2.0 authorization-code flow (HMAC-signed state) + optional single-tenant enforcement. Delegated Graph token encrypted at rest. |
| **Internal API Gateway** | `X-Internal-Api-Key` enforced on all programmatic endpoints. |

---

## 6. Pluggable LLM Provider Decoupling

The platform supports local-first development and enterprise production inference with zero changes to the LangGraph core. Production inference is **Microsoft 365 Copilot**, reached through **LiteLLM** so future provider swaps are configuration-only.

```
┌─────────────────────────────────────────────────────────┐
│                       LangGraph                          │
│      nodes call get_llm_client().chat_completion(...)    │
└────────────────────────────┬────────────────────────────┘
                             ▼
                 ┌───────────────────────┐
                 │ LLMProvider Interface │   app/services/llm/base.py
                 └───────────┬───────────┘
                ┌────────────┴───────────────────────┐
                ▼                                    ▼
      ┌───────────────────┐        ┌──────────────────────────────────┐
      │  Ollama Provider  │        │   MicrosoftCopilotProvider       │
      │ (local / fallback)│        │              │                   │
      └─────────┬─────────┘        │              ▼                   │
                │                  │   LiteLLM  (custom_provider_map)  │
                │                  │              │                   │
                │                  │              ▼                   │
                │                  │  M365 Copilot Chat handler       │
                │                  │  (_m365_copilot_litellm_handler) │
                │                  │              │                   │
                │                  │              ▼                   │
                │                  │  Microsoft Graph /beta/copilot   │
                │                  │  Microsoft Entra ID (delegated)  │
                │                  └──────────────┬───────────────────┘
                └────────────┬────────────────────┘
                             ▼
                    Normalized LLMResponse  →  SAME analysis state,
                    causal engine, validators, and LQMS report
```

### Configuration (`backend/.env`)
```bash
# Local development / degraded fallback (Ollama + Qwen3:8b)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b

# Production (Microsoft 365 Copilot via LiteLLM)
LLM_PROVIDER=microsoft_copilot
MICROSOFT_TENANT_ID=<tenant-guid>
MICROSOFT_CLIENT_ID=<app-registration-client-id>
MICROSOFT_CLIENT_SECRET=<app-registration-secret>
MICROSOFT_REDIRECT_URI=http://localhost:8010/api/auth/microsoft/callback
# Local-dev only: paste a delegated Graph token to skip the browser sign-in
MICROSOFT_COPILOT_ACCESS_TOKEN=
```

### 6.1 Microsoft 365 Copilot integration

**Access mechanism.** The organization's Microsoft 365 Copilot add-on license is consumed
through the **Microsoft 365 Copilot Chat API** on Microsoft Graph:

| Step | Request |
|---|---|
| Create conversation | `POST https://graph.microsoft.com/beta/copilot/conversations` (empty body) |
| Send prompt | `POST https://graph.microsoft.com/beta/copilot/conversations/{id}/chat` |
| Cleanup | `DELETE https://graph.microsoft.com/beta/copilot/conversations/{id}` (best effort) |

There is **no generic model endpoint** and no model identifier — Copilot selects the
underlying model internally. `LLMResponse.model` is the fixed label `m365-chat`.

**Authentication — delegated only.** The Chat API supports **delegated permissions only**
(no application/daemon flow, no personal accounts). A work/school user signs in via
Microsoft Entra ID (MSAL authorization-code flow, `app/auth/microsoft_entra.py`); the
resulting delegated Graph token is encrypted into the server-side session
(`app/auth/session.py`) and passed per-request to the provider.

**Required delegated Microsoft Graph permissions** (all are `.All` scopes → **tenant admin
consent required**):

```
Sites.Read.All   Mail.Read   People.Read.All   OnlineMeetingTranscript.Read.All
Chat.Read   ChannelMessage.Read.All   ExternalItem.Read.All
```

plus `offline_access` (refresh token) and the OIDC scopes.

**Entra app registration.**
1. entra.microsoft.com → App registrations → New registration.
2. Redirect URI (platform **Web**): value of `MICROSOFT_REDIRECT_URI`
   (`.../api/auth/microsoft/callback` — add the production URL for deployment).
3. Certificates & secrets → new client secret → `MICROSOFT_CLIENT_SECRET`.
4. API permissions → Microsoft Graph → **Delegated** → add the seven scopes above →
   **Grant admin consent**.
5. Every user of the app must hold a **Microsoft 365 Copilot** add-on license.

**Capabilities & limitations (verified against Microsoft Learn, 2026):**

| Capability | Status | Handling |
|---|---|---|
| Streaming | Supported (SSE) | Not wired — nodes use non-streaming synthesis |
| Structured output / JSON mode | **Not supported** | System instructions + a "return only JSON" directive are folded into `message.text`; parsed by the existing `parse_llm_json` salvage path; unparseable output **fails closed** |
| Tool / function calling | **Not supported** | Not used by any node |
| Temperature / max tokens | **Not supported** | Accepted and ignored |
| System-prompt field | **Not supported** | Folded into `message.text` |
| Token usage metrics | Not returned | `input_tokens` / `output_tokens` = `None` |
| API maturity | **`/beta`** — "not supported for production" per Microsoft | Ollama remains the configured degraded fallback; every failure mode fails closed, never fabricates a result |
| Long prompts | Prone to Graph gateway timeouts | `MICROSOFT_COPILOT_TIMEOUT_SECONDS` default 90s; timeout → `LLMTimeoutError` |

**Guardrails unchanged.** The provider is a pure pass-through: it returns the model's text
(minus Copilot inline markup such as `<Person>` / `[^1^]`) and performs **no** numeric,
status, evidence, or relationship interpretation. All deterministic values continue to come
from `app/financial/*`, `app/services/confidence.py`, the 6M taxonomy, rule engines, and
invariant/grounding validators.

**Local development without a browser.** Set `MICROSOFT_COPILOT_ACCESS_TOKEN` to a delegated
Graph token (e.g. from Graph Explorer consented to the seven scopes) and run with
`LLM_PROVIDER=microsoft_copilot`.

---

## 7. Verification & Automated Test Suite

Run the full test suite from `backend/`:

```bash
pytest
```

Key test coverage areas:
- `tests/test_master_causal_architecture.py`: Full pipeline causal invariants.
- `tests/test_causal_graph_engine.py`: Hypothesis promotion and score gating.
- `tests/test_financial_semantic_engine.py`: Cost factor extraction, relationship validation, and fail-closed math.
- `tests/test_financial_calculator.py`: PAF breakdown, recovery deduction, scenario analysis, and annualization.
- `tests/test_adversarial_llm_outputs.py`: Hallucination and injection protection.
