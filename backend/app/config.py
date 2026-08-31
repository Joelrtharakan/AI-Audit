from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM semantic financial-evidence interpretation (app.financial.
    # semantic_engine): when enabled, financial analysis first asks the
    # LLM to structurally interpret the evidence (claims/relationships/
    # calculation proposals), independently validates and calculates the
    # result deterministically, and only falls back to the pure
    # regex-extraction engine (app.financial.engine.analyze_financial_
    # exposure) when the LLM path is unavailable or produces nothing
    # validatable -- every failure mode (no provider configured, network
    # error, invalid JSON, empty validated result) fails closed to the
    # regex fallback, never fabricates a result.
    #
    # Defaults to ON: verified against a real, locally-running Ollama
    # provider (qwen3:8b) -- the semantic interpreter correctly identifies
    # quantity/rate claims, the RATE_APPLIES_TO_QUANTITY relationship, the
    # MULTIPLY calculation, and a grounded cost factor (e.g. DOWNTIME_COST)
    # from unseen finding text, taking ~20s per call. In an environment
    # with NO reachable LLM provider, attempting the call does not fail
    # fast (measured: an unreachable local Ollama endpoint does not refuse
    # the connection, it hangs until `financial_semantic_reasoning_
    # timeout_seconds`) -- but every failure mode still fails closed to
    # the regex-extraction fallback (see analyze_financial_exposure_
    # semantic's docstring), so the cost of leaving this on with no
    # provider configured is added latency per investigation, never a
    # wrong or fabricated result. Set to False to force pure-regex
    # behavior regardless of provider availability (e.g. to avoid that
    # latency in a deployment with no LLM provider reachable).
    financial_semantic_reasoning_enabled: bool = True
    # Measured real qwen3:8b latency for this prompt/schema: ~20-40s on a
    # lightly-loaded box, but a genuinely complete multi-claim
    # interpretation (5+ claims, relationships, several calculation
    # proposals) is ~1300 output tokens and can take 90s+ under load or on
    # a slower host. 60s was leaving real, well-formed findings timing out
    # (observed: TIMEOUT at 60050ms) -> LLM_UNAVAILABLE -> the whole
    # financial section collapsing to NOT_ESTABLISHED. 120s gives the
    # honest interpretation room to finish; a genuinely unreachable
    # provider still fails closed (just later).
    financial_semantic_reasoning_timeout_seconds: float = 120.0

    # Canonical semantic finding context (app.services.canonical_finding_
    # interpreter): a broader LLM interpretation (primary deviation,
    # entity/state separation, causal claims, previous-CAPA reference)
    # computed alongside the existing deterministic pipeline in SHADOW
    # MODE ONLY -- the deterministic result remains authoritative;
    # disagreements are recorded on the report, never used to override
    # any output. Same off-by-default rationale as financial_semantic_
    # reasoning_enabled (an unreachable provider does not fail fast).
    canonical_semantic_shadow_enabled: bool = False
    canonical_semantic_shadow_timeout_seconds: float = 8.0

    # LLM-PRIMARY canonical interpretation call budget. This is the largest
    # single structured LLM response in the pipeline. `num_ctx` MUST exceed
    # (system prompt + schema hint + finding + evidence) tokens PLUS max_tokens
    # -- a compacted ~11KB system prompt + ~5KB schema + a typical finding is
    # ~4.5-5.5K prompt tokens, so 12288 leaves real headroom for a 2200-token
    # structured response. An under-budgeted num_ctx makes the provider thrash
    # / truncate and the call times out, silently dropping the whole pipeline
    # back to the deterministic floor. A genuinely unreachable provider still
    # fails closed. `agent_overall_timeout_seconds` (see below) is sized so the
    # deterministic fallback path (core_synthesis + concurrent financial /
    # remediation LLM calls) still completes after a canonical timeout.
    # §7 fast-fail: after the prompt/schema compaction + dropping the unused
    # `financial` sub-object, the canonical call is MEASURED at ~58-60s warm
    # (isolated) on qwen3:8b -- was ~110s / timing out. 100s covers warm +
    # cold-start + a large finding with real margin and is still below the
    # old 110s (NOT raised to improve success rate). Pass 29 removes the
    # redundant core_synthesis call (~20-150s), which is the real latency
    # win; on a genuine canonical timeout the request proceeds on
    # DETERMINISTIC_FALLBACK.
    canonical_semantic_primary_timeout_seconds: float = 100.0
    # §9 / Pass 36 M5: generation time is the dominant canonical latency cost on
    # qwen3:8b, so this is the primary lever. The Pass-36 compact-output
    # contract (per-field word caps, omit optional fields, reference evidence by
    # E-id, no prose) puts a COMPLETE structured response well under 1.2K
    # tokens; 1400 (was 1600, was 2200) is a safe ceiling with headroom for a
    # large multi-hypothesis finding. Truncation is still salvaged field-by-
    # field. The `LLM RESPONSE ... finish_reason=length` log line surfaces any
    # real truncation so this can be re-tuned from data, never guessed.
    # Spec Pass 49 §18: per-stage model override on the SAME configured
    # provider. Empty -> the global model (`ollama_model` / `llm_model`). Set
    # e.g. CANONICAL_SEMANTIC_MODEL=qwen3:14b to run the primary semantic
    # interpretation on a stronger local model when qwen3:8b's structured
    # accuracy on complex multi-component findings is insufficient -- WITHOUT
    # a code change and WITHOUT switching providers. Never an automatic switch.
    canonical_semantic_model: str = ""
    remediation_cost_model: str = ""
    canonical_semantic_max_tokens: int = 1400
    # §8: right-sized for the compacted prompt (~3K sys+schema tok + finding +
    # evidence, typically ~4-4.5K input) + output. num_ctx is NOT reduced
    # (Pass 36 M6): the finding+evidence input is variable and an
    # under-budgeted window makes the provider thrash. 8192 covers a large
    # finding + full output with headroom.
    canonical_semantic_num_ctx: int = 8192

    # LLM-PRIMARY canonical semantic interpretation: when True, the LLM
    # canonical interpretation runs in understand_finding_node (once, reused
    # downstream) and its VALIDATED structured fields are merged into
    # canonical_finding_state -- the LLM becomes the primary semantic
    # interpreter and resolve_deviation() becomes the fail-closed floor. The
    # LLM also owns the investigation/remediation/pricing distinction, which
    # the downstream remediation-cost engine consumes rather than
    # re-deriving from the raw finding.
    #
    # ON by default: the deterministic path is a SAFETY FLOOR, not the
    # primary intelligence layer. When the LLM provider is unreachable or
    # returns None/invalid, `interpret_finding_canonically` returns None, the
    # merge is a pure no-op, and every downstream consumer falls back to the
    # unchanged deterministic behaviour -- so an unreachable provider costs
    # only added latency, never a wrong result. The test suite pins this
    # False (see tests/conftest.py) so the regression baseline stays
    # deterministic and fast; dedicated LLM-primary tests enable it
    # explicitly.
    canonical_semantic_llm_primary: bool = True

    # Remediation Cost Estimation (app.remediation): a SEPARATE semantic
    # analysis from financial exposure -- "what will it cost to correct/
    # prevent the finding?" rather than "what did the finding cost?". The
    # LLM infers the remediation strategy, implementation activities, and
    # cost drivers from the finding context; deterministic code validates
    # structure and executes arithmetic; a single canonical
    # RemediationCostResult is attached to the report.
    #
    # Same off-by-failure rationale as financial_semantic_reasoning_enabled:
    # an unreachable LLM provider does not fail fast, and every failure mode
    # fails closed to a professional NOT_ASSESSABLE result -- never a
    # fabricated number, never an internal diagnostic string. The cost of
    # leaving this on with no provider reachable is added latency per
    # investigation, never a wrong result. Set to False to skip the section
    # entirely in a deployment with no LLM provider.
    remediation_cost_estimation_enabled: bool = True
    # Pass 52: 75s was too tight. MEASURED on the full compiled graph, qwen3:8b,
    # multi-component finding (8 panels: materials + derived labour + fixed
    # inspection) -- the remediation call takes >75s (prompt-eval on the fuller
    # graph context + ~22 tok/s generation of a 3-component plan) and TIMED OUT,
    # dropping the whole estimate to NOT_ASSESSABLE. This is a model/hardware
    # throughput limit, not a prompt defect. Give it real headroom (parallels
    # canonical primary 100s and OLLAMA_PRIMARY_SYNTHESIS_TIMEOUT_SECONDS=110).
    # A stronger/faster configured model finishes well inside this.
    remediation_cost_estimation_timeout_seconds: float = 150.0
    # A compact remediation interpretation for a typical finding (2-4 cost
    # components + activities, optional/null fields omitted per the prompt) is
    # ~700-1100 output tokens against real qwen3:8b. 1400 covers that with
    # headroom; a genuinely large multi-component finding that truncates is
    # Pass 38: MEASURED against real qwen3:8b + Ollama native timing -- a
    # 5-asset finding (5 inspections + 5 installs + recurring verification +
    # activities + pricing_information + calc plans + auditor inputs) produced
    # eval_count=1200 with done_reason=length, i.e. the Pass-36 1200 ceiling
    # TRUNCATED it. Per the "if finish_reason=length, compact the contract, do
    # not lower the limit -- then set it above observed legitimate size" rule,
    # this is 1800: above the observed truncation, still salvage-protected.
    # Was 2200 -> 1400 -> 1200 (too low) -> 1800. `LLM RESPONSE
    # finish_reason=length` remains the re-tuning signal.
    remediation_cost_max_tokens: int = 1800
    # Input for this call is the compacted pricing system prompt (~2.3K tok) +
    # schema hint (~0.6K tok) + the structured context block (canonical
    # remediation state + evidence, ~1-2K tok). 4096 could NOT hold that: the
    # provider context-shifted mid-generation and the call took ~62s in
    # production. 8192 fits input + a 1400-token response with headroom and
    # removes the thrash (measured ~62s -> ~18-25s). Still well below
    # ollama's 40960 native ceiling for qwen3:8b.
    remediation_cost_num_ctx: int = 8192

    # -------------------------------------------------------------------------
    # LLM execution route (app.services.llm.execution / .providers.litellm_provider)
    # ONE provider + ONE model, resolved once per investigation request, routed
    # through LiteLLM as the single inference boundary. LLM_MODEL is
    # provider-neutral (the LiteLLM identifier is derived only at the adapter
    # boundary). LLM_FALLBACK_ENABLED=false means a failed call surfaces the
    # app's degraded / fail-closed behavior -- never a silent switch to another
    # provider. When true, retries stay against the SAME provider and model.
    # -------------------------------------------------------------------------
    llm_provider: str = "ollama"
    # Empty -> each provider's own default (ollama_model / "m365-chat" /
    # copilot_model / groq_model / ...). Set explicitly (e.g. "qwen3:8b") to pin
    # the model for the configured provider.
    llm_model: str = ""
    llm_fallback_enabled: bool = False

    # Local Ollama inference server (base URL + default model when LLM_MODEL unset).
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_timeout_seconds: float = 25.0
    # 10 minutes so the model stays resident across the extraction ->
    # core_synthesis -> critic sequence of a single investigation (and across
    # back-to-back investigations) instead of being evicted and reloaded --
    # a reload shows up as a large `load_ms` on the very next call.
    ollama_keep_alive: str = "10m"
    # qwen3:8b supports up to 40960 natively; 8192 comfortably covers a
    # typical finding + evidence ledger + full core_synthesis output with
    # headroom, without paying for a much larger KV cache than needed. This
    # remains the default/fallback context; per-call overrides below use a
    # smaller window for the lighter-weight nodes (extraction, critic) so
    # prompt evaluation isn't paying for KV cache it never uses.
    ollama_num_ctx: int = 4096
    ollama_temperature: float = 0.1
    # `ollama_thinking=False` (default) means: for a selected model that matches
    # one of `ollama_thinking_model_markers`, thinking is DISABLED at the Ollama
    # level (LiteLLM `reasoning_effort="disable"` -> `"think": false` in the
    # /api/chat body) so no <think>...</think> tokens are generated for our
    # structured-JSON extraction -- pure latency/token saving. Set True to let
    # such models reason normally.
    ollama_thinking: bool = False
    # Comma-separated, case-insensitive substrings identifying Ollama reasoning
    # models. A selected model whose name contains any marker gets the
    # thinking-disable treatment above. Generic -- "qwen3" covers qwen3:8b,
    # qwen3:14b, qwen3-coder, hf.co/*/Qwen3-32B, etc. Add e.g. "deepseek-r1"
    # if you run other reasoning models. Never a hardcoded single model id.
    ollama_thinking_model_markers: str = "qwen3"
    ollama_max_retries: int = 1

    # Operation-specific timeouts & token budgets
    ollama_extraction_timeout_seconds: float = 15.0
    ollama_extraction_max_tokens: int = 300
    ollama_extraction_num_ctx: int = 4096
    ollama_primary_synthesis_timeout_seconds: float = 45.0
    ollama_core_synthesis_max_tokens: int = 650
    ollama_core_synthesis_num_ctx: int = 4096
    ollama_recovery_synthesis_timeout_seconds: float = 25.0
    ollama_recovery_max_tokens: int = 450
    ollama_recovery_num_ctx: int = 4096
    ollama_critic_timeout_seconds: float = 15.0
    ollama_critic_max_tokens: int = 250
    ollama_critic_num_ctx: int = 4096
    # A complete multi-claim financial interpretation (claims +
    # relationships + several calculation proposals + cost_factor +
    # quantification) measured at ~1300 output tokens for a 5-claim
    # finding against real qwen3:8b. 900 was truncating the JSON mid-
    # structure -> unparseable -> LLM_INVALID -> the entire financial
    # analysis discarded, even when the model's semantics were correct.
    # 2000 covers the observed complete output (~1300 for 5 claims) with
    # ~54% headroom; the parser additionally salvages a near-complete
    # truncated response (see json_parser.extract_json_str) rather than
    # failing all-or-nothing.
    ollama_financial_semantic_max_tokens: int = 2000
    # 6144 (was 8192): the financial system prompt + schema hint is ~3200
    # tokens and the evidence context is typically <500; 6144 covers input +
    # a 2000-token response while cutting KV-cache allocation ~25%. A very
    # large evidence ledger that would exceed this shifts the oldest context
    # (system prompt is preserved) -- a rare, graceful degradation, not a
    # correctness break, and salvage still handles a truncated response.
    ollama_financial_semantic_num_ctx: int = 6144

    # Optional provider timeouts & keys (for fallback router tests)
    google_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"
    groq_timeout_seconds: float = 15.0
    openrouter_timeout_seconds: float = 20.0
    gemini_timeout_seconds: float = 25.0

    # -------------------------------------------------------------------------
    # Microsoft 365 Copilot (Production) -- accessed through the Microsoft
    # Graph beta Copilot Chat API (POST /beta/copilot/conversations then
    # .../{id}/chat). Reached via LiteLLM through a registered custom
    # provider ("microsoft_copilot"). See
    # app/services/llm/providers/microsoft_copilot_provider.py and
    # app/services/llm/providers/_m365_copilot_litellm_handler.py.
    #
    # The Chat API is DELEGATED-ONLY (no application/daemon auth), requires
    # a per-user Microsoft 365 Copilot add-on license, exposes no model
    # selection, and returns free text only (no native JSON mode / tool
    # calling / temperature). It is currently a /beta API.
    # -------------------------------------------------------------------------
    microsoft_copilot_enabled: bool = True
    # Dev/testing bypass: a delegated Graph access token pasted directly so
    # the provider can run without the interactive Entra sign-in flow. In
    # production this is populated per-request from the authenticated user
    # session (routers/investigate.py etc.), never set statically.
    microsoft_copilot_access_token: str = ""
    # The Chat API is prone to gateway timeouts on long prompts; give the
    # honest synthesis room to finish. A genuinely unreachable Graph
    # endpoint still fails closed (just later).
    microsoft_copilot_timeout_seconds: float = 90.0
    microsoft_copilot_max_retries: int = 2
    # locationHint.timeZone is a required Chat API request parameter.
    microsoft_copilot_timezone: str = "UTC"
    # Audit reasoning should not pull public web content into grounded
    # answers; enterprise search grounding is always on regardless.
    microsoft_copilot_web_grounding: bool = False
    # Graph base URL (national-cloud override point, e.g. US Gov L4/L5).
    microsoft_graph_base_url: str = "https://graph.microsoft.com"

    # Microsoft Entra ID (delegated OAuth 2.0 authorization-code flow) --
    # replaces the former GitHub OAuth sign-in. See app/auth/microsoft_entra.py.
    microsoft_tenant_id: str = ""
    microsoft_client_id: str = ""
    microsoft_client_secret: str = ""
    microsoft_redirect_uri: str = "http://localhost:8010/api/auth/microsoft/callback"
    # Optional tenant allow-list gate (parallels the former github_allowed_org):
    # when set, only users whose token tenant id (tid claim) matches are admitted.
    microsoft_allowed_tenant_id: str = ""

    # -------------------------------------------------------------------------
    # GitHub OAuth + GitHub Copilot -- an ALTERNATIVE sign-in to Microsoft
    # Entra / M365 Copilot. Both coexist: whichever provider a user signs in
    # with determines which Copilot backend their investigations use for that
    # session (see app/routers/auth.py::apply_user_copilot_token). See
    # app/auth/github_oauth.py and
    # app/services/llm/providers/github_copilot_provider.py.
    # -------------------------------------------------------------------------
    github_client_id: str = ""
    github_client_secret: str = ""
    github_redirect_uri: str = "http://localhost:8010/api/auth/github/callback"
    # Optional org allow-list gate; empty = any GitHub account with a Copilot
    # subscription is admitted.
    github_allowed_org: str = ""
    # GitHub Copilot SDK model ("auto" lets Copilot choose). Dev bypass token is
    # populated per-request from the authenticated session in production.
    copilot_model: str = "auto"
    copilot_github_token: str = ""
    copilot_timeout_seconds: float = 90.0
    copilot_log_level: str = "info"

    session_secret: str = "dev-secret-key-change-in-production-min-32-chars"
    session_cookie_name: str = "lqms_session"
    session_expiry_hours: int = 24
    frontend_dashboard_url: str = "http://localhost:5510/index.html"
    frontend_login_url: str = "http://localhost:5510/login.html"

    # -------------------------------------------------------------------------
    # LLM provider router (app/services/llm_router.py): circuit-breaker
    # cooldowns and bounded per-provider concurrency.
    # -------------------------------------------------------------------------
    # Cooldown applied when a provider fails with something other than a
    # 429 with a Retry-After header (timeout, 5xx, network error, etc).
    llm_router_default_cooldown_seconds: float = 30.0
    # Upper bound on how long a provider's circuit stays OPEN, regardless of
    # what Retry-After says -- a provider reporting a multi-hour Retry-After
    # must not be skipped for the rest of the day; it just means "try again
    # in this cap, not sooner."
    llm_router_max_cooldown_seconds: float = 120.0
    # Bounded concurrency per provider so a burst of parallel graph nodes
    # (e.g. the understanding node's asyncio.gather) can't fan out into a
    # flood of simultaneous requests against one provider.
    llm_router_max_concurrency_per_provider: int = 6

    # Internal auth
    internal_api_key: str = ""

    # CORS
    allowed_origins: str = "http://localhost:5500,http://localhost:5501,http://localhost:5510"

    # Logging
    log_level: str = "INFO"

    # Prompt version, stamped onto every AI response for traceability
    analysis_prompt_version: str = "1.0"

    # -------------------------------------------------------------------------
    # ASP.NET LQMS integration
    # When empty, every tool call returns an empty-but-valid result and the
    # agent continues, recording the gap in the evidence ledger.
    # -------------------------------------------------------------------------
    lqms_aspnet_base_url: str = ""
    lqms_aspnet_api_key: str = ""  # bearer token / API key for the ASP.NET API
    lqms_mode: str = "production"  # "production" (real HTTP) or "mock" (safe demo records tagged DEMO DATA)

    # -------------------------------------------------------------------------
    # Agent limits
    # -------------------------------------------------------------------------
    agent_max_iterations: int = 10
    agent_max_tool_calls: int = 15
    agent_tool_timeout_seconds: float = 10.0
    # Worst realistic path on a local model: canonical interpretation up to its
    # ~110s timeout, then (on failure) core_synthesis (~70s) with financial +
    # remediation-cost interpretation running CONCURRENTLY (~90s). 240s could
    # not fit that; 330s does, with margin for the deterministic stages.
    agent_overall_timeout_seconds: float = 330.0
    agent_max_critic_iterations: int = 2

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def prompts_dir(self) -> Path:
        return BACKEND_ROOT / "app" / "prompts"

    @property
    def agent_prompts_dir(self) -> Path:
        return BACKEND_ROOT / "app" / "prompts" / "agent"


@lru_cache
def get_settings() -> Settings:
    return Settings()
