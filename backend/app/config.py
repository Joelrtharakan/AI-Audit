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

    # Production Local Ollama Inference Server
    llm_provider: str = "ollama"
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
    ollama_thinking: bool = False
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
    # 2200 covers the observed complete output with headroom; the parser
    # additionally salvages a near-complete truncated response (see
    # json_parser.extract_json_str) rather than failing all-or-nothing.
    ollama_financial_semantic_max_tokens: int = 2200
    ollama_financial_semantic_num_ctx: int = 8192

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

    # GitHub Copilot SDK (Production)
    copilot_model: str = "auto"
    copilot_github_token: str = ""
    copilot_timeout_seconds: float = 30.0
    copilot_log_level: str = "info"

    # GitHub OAuth & Enterprise Configuration
    github_client_id: str = ""
    github_client_secret: str = ""
    github_redirect_uri: str = "http://localhost:8010/api/auth/github/callback"
    github_allowed_org: str = ""  # Configurable: e.g. "my-company"
    github_enterprise: str = ""   # Optional GitHub Enterprise slug/domain
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
    agent_overall_timeout_seconds: float = 240.0
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
