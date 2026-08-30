"""Per-investigation LLM execution configuration.

ONE selected provider + ONE selected model = ONE immutable execution
configuration for the entire investigation request. Resolved ONCE at the start
of the request (routers/investigate.py, routers/analyze.py) and threaded through
every LLM stage via a ContextVar so no node can pick a different provider/model.

Nothing here performs inference or provider discovery. `resolve_execution_config`
is pure: it turns a (provider, model) pair into the LiteLLM model identifier and
the provider-specific transport parameters that
`app.services.llm.providers.litellm_provider.LiteLLMProvider` will use.

Business logic stays provider-neutral: LiteLLM model identifiers
("ollama_chat/qwen3:8b", "microsoft_copilot/m365-chat", ...) are produced ONLY
here, at the adapter boundary.
"""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings

# Canonical provider names + accepted aliases -> canonical.
_PROVIDER_ALIASES: dict[str, str] = {
    "ollama": "ollama",
    "microsoft_copilot": "microsoft_copilot",
    "microsoft-copilot": "microsoft_copilot",
    "m365_copilot": "microsoft_copilot",
    "m365-copilot": "microsoft_copilot",
    "copilot": "microsoft_copilot",  # legacy bare alias
    "github_copilot": "github_copilot",
    "github-copilot": "github_copilot",
    "groq": "groq",
    "gemini": "gemini",
    "openrouter": "openrouter",
}

# The custom-handler prefix for the GitHub Copilot adapter -- deliberately NOT
# LiteLLM's native "github_copilot/" route, which runs its own interactive
# device-auth flow and would conflict with our GitHub OAuth / session tokens.
GITHUB_LITELLM_PREFIX = "github_copilot_session"
MICROSOFT_LITELLM_PREFIX = "microsoft_copilot"


@dataclass(frozen=True)
class LLMExecutionConfig:
    """Immutable. The single authoritative route for one investigation."""

    provider: str                 # canonical provider name
    model: str                    # bare model, provider-neutral, e.g. "qwen3:8b"
    litellm_model: str            # adapter-boundary identifier, e.g. "ollama_chat/qwen3:8b"
    api_base: str | None = None
    api_key: str | None = None    # secret -- never logged, never serialized
    extra_params: dict[str, Any] = field(default_factory=dict)
    fallback_enabled: bool = False

    def public_dict(self) -> dict[str, Any]:
        """Safe view for logs / API -- NO api_key."""
        return {
            "provider": self.provider,
            "model": self.model,
            "litellm_model": self.litellm_model,
            "fallback_enabled": self.fallback_enabled,
        }


class UnknownProviderError(ValueError):
    """Raised for a provider name that is not supported."""


def _is_reasoning_model(model: str, settings: Any) -> bool:
    """True when the selected Ollama model name matches a configured
    reasoning-model marker (OLLAMA_THINKING_MODEL_MARKERS, comma-separated,
    case-insensitive substring match). Generic -- e.g. "qwen3" matches
    qwen3:8b / qwen3:14b / qwen3-coder / hf.co/.../Qwen3-32B-GGUF, etc."""
    markers = [
        m.strip().lower()
        for m in (getattr(settings, "ollama_thinking_model_markers", "") or "").split(",")
        if m.strip()
    ]
    name = (model or "").lower()
    return any(marker in name for marker in markers)


def _canonical_provider(name: str | None) -> str:
    key = (name or "").strip().lower()
    if not key:
        return _canonical_provider(get_settings().llm_provider)
    canonical = _PROVIDER_ALIASES.get(key)
    if canonical is None:
        raise UnknownProviderError(
            f"Unsupported LLM provider '{name}'. Supported: ollama, microsoft_copilot, "
            "github_copilot, groq, gemini, openrouter."
        )
    return canonical


def resolve_execution_config(
    provider: str | None = None,
    model: str | None = None,
) -> LLMExecutionConfig:
    """Pure resolution. (provider, model) -> immutable LLMExecutionConfig.

    Missing provider/model fall back to settings (LLM_PROVIDER / LLM_MODEL). No
    network, no probing, no side effects.
    """
    settings = get_settings()
    canonical = _canonical_provider(provider)
    bare_model = (model or "").strip()
    # LLM_MODEL is the model for the CONFIGURED provider. When the request keeps
    # the configured provider and states no model, LLM_MODEL wins; when it
    # overrides the provider without a model, the provider's own default applies.
    if not bare_model and canonical == _canonical_provider(settings.llm_provider):
        bare_model = (settings.llm_model or "").strip()

    if canonical == "ollama":
        bare_model = bare_model or settings.ollama_model
        ollama_params: dict[str, Any] = {
            "keep_alive": settings.ollama_keep_alive,
            "num_ctx": settings.ollama_num_ctx,
        }
        # Qwen3 (and other Ollama reasoning models) emit <think>...</think>
        # tokens by default -- pure latency + token cost for our structured
        # JSON extraction. When the model is a configured reasoning model and
        # thinking is not explicitly wanted, disable it AT THE OLLAMA LEVEL via
        # LiteLLM's supported `reasoning_effort` param, which LiteLLM's
        # ollama_chat transform maps to `"think": false` in the /api/chat body.
        # Non-reasoning models get no `think` field at all (avoids a 400 from
        # Ollama for models that do not support thinking). Generic -- driven by
        # OLLAMA_THINKING_MODEL_MARKERS, never a hardcoded model id.
        if _is_reasoning_model(bare_model, settings) and not settings.ollama_thinking:
            ollama_params["reasoning_effort"] = "disable"
        return LLMExecutionConfig(
            provider="ollama",
            model=bare_model,
            litellm_model=f"ollama_chat/{bare_model}",
            api_base=settings.ollama_base_url,
            extra_params=ollama_params,
            fallback_enabled=settings.llm_fallback_enabled,
        )

    if canonical == "microsoft_copilot":
        bare_model = bare_model or "m365-chat"
        return LLMExecutionConfig(
            provider="microsoft_copilot",
            model=bare_model,
            litellm_model=f"{MICROSOFT_LITELLM_PREFIX}/{bare_model}",
            api_key=settings.microsoft_copilot_access_token or None,
            extra_params={
                "graph_base_url": settings.microsoft_graph_base_url,
                "timezone": settings.microsoft_copilot_timezone,
                "web_grounding": settings.microsoft_copilot_web_grounding,
                "max_retries": settings.microsoft_copilot_max_retries,
            },
            fallback_enabled=settings.llm_fallback_enabled,
        )

    if canonical == "github_copilot":
        bare_model = bare_model or settings.copilot_model or "auto"
        return LLMExecutionConfig(
            provider="github_copilot",
            model=bare_model,
            litellm_model=f"{GITHUB_LITELLM_PREFIX}/{bare_model}",
            api_key=settings.copilot_github_token or None,
            extra_params={"log_level": settings.copilot_log_level},
            fallback_enabled=settings.llm_fallback_enabled,
        )

    # Native LiteLLM cloud providers -- single route, no failover.
    if canonical == "groq":
        bare_model = bare_model or settings.groq_model
        return LLMExecutionConfig(
            provider="groq", model=bare_model, litellm_model=f"groq/{bare_model}",
            api_key=settings.groq_api_key or None, fallback_enabled=settings.llm_fallback_enabled,
        )
    if canonical == "gemini":
        bare_model = bare_model or settings.gemini_model
        return LLMExecutionConfig(
            provider="gemini", model=bare_model, litellm_model=f"gemini/{bare_model}",
            api_key=settings.google_api_key or None, fallback_enabled=settings.llm_fallback_enabled,
        )
    if canonical == "openrouter":
        bare_model = bare_model or settings.openrouter_model
        return LLMExecutionConfig(
            provider="openrouter", model=bare_model, litellm_model=f"openrouter/{bare_model}",
            api_key=settings.openrouter_api_key or None, fallback_enabled=settings.llm_fallback_enabled,
        )

    raise UnknownProviderError(canonical)  # unreachable


# --------------------------------------------------------------------------
# Request-scoped context (frozen for the whole investigation)
# --------------------------------------------------------------------------

_execution_config: contextvars.ContextVar[LLMExecutionConfig | None] = contextvars.ContextVar(
    "llm_execution_config", default=None
)
_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_request_id", default=None
)


def begin_request(config: LLMExecutionConfig, request_id: str | None = None) -> str:
    """Freeze the LLM execution config + correlation id for this request's
    async context. Called ONCE by the investigation/analysis entry point."""
    _execution_config.set(config)
    rid = request_id or uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def current_execution_config() -> LLMExecutionConfig | None:
    return _execution_config.get()


def current_request_id() -> str | None:
    return _request_id.get()


def effective_execution_config() -> LLMExecutionConfig:
    """The frozen request config if one was set, else resolved from settings
    (direct unit-test / non-request callers)."""
    return current_execution_config() or resolve_execution_config()
