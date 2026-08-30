"""Ollama support module.

Direct-HTTP INFERENCE through this provider is DISABLED: all model inference now
goes through the single application boundary
`app.services.llm.providers.litellm_provider.LiteLLMProvider` (LiteLLM native
`ollama_chat/<model>` route). `OllamaProvider.generate` raises rather than
opening a second, parallel inference path.

What remains here is the non-inference Ollama readiness/metadata probe
(`check_health` -> GET /api/tags) and the shared connection pool helper, still
used by the health endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Any
import httpx

from app.config import get_settings
from app.services.llm.base import LLMProvider, LLMResponse
from app.services.llm.exceptions import LLMConfigurationError

logger = logging.getLogger(__name__)

# One persistent, connection-pooling AsyncClient per running event loop,
# reused across every node/call instead of opening a fresh TCP connection per request.
_client_lock = asyncio.Lock()
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = weakref.WeakKeyDictionary()


async def _get_shared_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is not None and not client.is_closed:
        return client
    async with _client_lock:
        client = _clients.get(loop)
        if client is not None and not client.is_closed:
            return client
        client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
        _clients[loop] = client
        return client


class OllamaProvider(LLMProvider):
    """Ollama implementation of LLMProvider for local inference."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        self._base_url = base_url or settings.ollama_base_url
        self._model = model or settings.ollama_model
        self._timeout = timeout_seconds or settings.ollama_timeout_seconds

    def _get_native_url(self, endpoint: str) -> str:
        base = self._base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/{endpoint.lstrip('/')}"

    async def check_health(self) -> dict[str, Any]:
        """Check server reachability and model availability."""
        url = self._get_native_url("/api/tags")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    models = [m.get("name") for m in resp.json().get("models", [])]
                    target_model = self._model
                    is_available = target_model in models or any(m.startswith(target_model) for m in models)
                    return {
                        "available": True,
                        "provider": "ollama",
                        "model": target_model,
                        "model_installed": is_available,
                        "installed_models": models,
                    }
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)
        return {
            "available": False,
            "provider": "ollama",
            "model": self._model,
            "model_installed": False,
            "installed_models": [],
        }

    async def generate(self, *, node: str, prompt: str, **kwargs: Any) -> LLMResponse:
        raise LLMConfigurationError(
            "Direct Ollama inference is disabled. All model inference is routed through "
            "app.services.llm.providers.litellm_provider.LiteLLMProvider (LiteLLM "
            "'ollama_chat/<model>'). This provider now serves only the non-inference "
            "readiness probe (check_health)."
        )

    # NOTE: the former direct-httpx /api/chat implementation lived here; it has
    # been removed. Ollama inference now goes exclusively through
    # LiteLLMProvider (LiteLLM 'ollama_chat/<model>').
