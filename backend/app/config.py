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
    ollama_timeout_seconds: float = 60.0
    ollama_keep_alive: str = "5m"
    ollama_num_ctx: int = 4096
    ollama_temperature: float = 0.1
    ollama_thinking: bool = False
    ollama_max_retries: int = 1

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
