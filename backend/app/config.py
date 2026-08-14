from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider selection -- "openrouter" (default/production) or "ollama" (fast local
    # dev iteration only). See README section 6b.
    llm_provider: str = "openrouter"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/auto"
    openrouter_site_url: str = "http://localhost:5500"
    openrouter_app_name: str = "LQMS-AI"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Ollama (local, unauthenticated -- never used in production)
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434/v1"

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
