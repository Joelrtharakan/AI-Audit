from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.1
    ollama_thinking: bool = False
    ollama_max_retries: int = 1

    # Operation-specific timeouts. A single global timeout meant a slow but
    # non-critical call (extraction, critic) could eat the same budget as
    # the primary synthesis call the whole investigation depends on.
    #
    # Generation throughput on this deployment's qwen3:8b (CPU) is ~20-24
    # tokens/sec regardless of context size, so latency scales almost
    # entirely with how much output is requested -- NOT with a larger
    # ceiling. The schema/prompt were made deliberately compact (1-sentence
    # fields, no narrative outside JSON) so a normal core_synthesis response
    # is ~300-700 tokens; the 1400-token ceiling below is a safety margin,
    # not a target, and should rarely be reached by a well-formed response.
    ollama_extraction_timeout_seconds: float = 12.0
    ollama_extraction_max_tokens: int = 350
    # Extraction's schema/prompt is small and self-contained -- it never
    # needs the full 8192-token synthesis window.
    ollama_extraction_num_ctx: int = 4096
    ollama_primary_synthesis_timeout_seconds: float = 25.0
    # Compact-schema ceiling (Section 1 fix): large enough for the full
    # core_synthesis JSON object (root cause, hypotheses, 5-Why, contributing
    # factors, CAPA, impact) at the prompt's 1-sentence-per-field discipline,
    # without inviting the model to fill unused budget with prose.
    ollama_core_synthesis_max_tokens: int = 1400
    ollama_recovery_synthesis_timeout_seconds: float = 15.0
    # Recovery is causal-reasoning fields only (no impact/CAPA) -- 900 tokens
    # is comfortably above the ~400-550 tokens that schema measures at, while
    # still being a materially smaller ceiling than the primary call's.
    ollama_recovery_max_tokens: int = 900
    ollama_critic_timeout_seconds: float = 10.0
    # The critic returns a handful of structured validation fields, not
    # prose -- 250 tokens is generous headroom over its typical ~60-100.
    ollama_critic_max_tokens: int = 250
    ollama_critic_num_ctx: int = 4096

    # Optional provider timeouts (for fallback router tests)
    groq_timeout_seconds: float = 15.0
    openrouter_timeout_seconds: float = 20.0
    gemini_timeout_seconds: float = 25.0

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
