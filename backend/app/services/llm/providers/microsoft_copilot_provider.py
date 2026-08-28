"""Production Microsoft 365 Copilot provider for the LQMS Audit Investigation Engine.

Implements the provider-neutral :class:`LLMProvider` contract on top of the
Microsoft 365 Copilot Chat API, routed through LiteLLM's custom-provider mechanism
(see :mod:`app.services.llm.providers._m365_copilot_litellm_handler`).

Every LangGraph node and pipeline service continues to type-hint against
``LLMProvider`` and call ``generate()`` / ``chat_completion()`` -- they never learn
that the backend is Microsoft Graph rather than the former GitHub Copilot SDK.

Capability notes (Chat API, /beta): no model selection, no native JSON mode, no
tool calling, no temperature control, text responses only. Unsupported knobs are
accepted and ignored for interface compatibility. All numbers / statuses /
evidence remain owned by the deterministic layer; unparseable model text fails
closed in the existing ``parse_llm_json`` path.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

from app.config import get_settings
from app.services.llm.base import LLMProvider, LLMResponse
from app.services.llm.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

_LITELLM_MODEL = "microsoft_copilot/m365-chat"


def reset_m365_copilot_clients() -> None:
    """Compatibility hook (parallels ``reset_copilot_clients``).

    The Chat API adapter holds no cached authenticated client -- the delegated
    bearer token is passed per request -- so there is nothing to tear down. Kept
    so routers can call it unconditionally after swapping the per-user token.
    """
    return None


def _register_handler() -> None:
    settings = get_settings()
    from app.services.llm.providers._m365_copilot_litellm_handler import register

    register(
        graph_base_url=settings.microsoft_graph_base_url,
        timezone=settings.microsoft_copilot_timezone,
        web_grounding=settings.microsoft_copilot_web_grounding,
    )


class MicrosoftCopilotProvider(LLMProvider):
    """Microsoft 365 Copilot Chat API implementation of :class:`LLMProvider`."""

    def __init__(
        self,
        model: str | None = None,
        ms_access_token: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        # The Chat API exposes no model identifier; this is a fixed label used
        # only for logging / cache-key namespacing / LLMResponse.model.
        self._model = model or "m365-chat"
        self._access_token = (
            ms_access_token
            or settings.microsoft_copilot_access_token
            or os.getenv("MICROSOFT_COPILOT_ACCESS_TOKEN")
            or ""
        ).strip()
        self._timeout = timeout_seconds or settings.microsoft_copilot_timeout_seconds
        self._max_retries = settings.microsoft_copilot_max_retries
        _register_handler()

    # -- health ----------------------------------------------------------

    async def check_health(self) -> dict[str, Any]:
        token = self._access_token or os.getenv("MICROSOFT_COPILOT_ACCESS_TOKEN") or ""
        if not token.strip():
            return {
                "available": False,
                "provider": "microsoft_copilot",
                "model": self._model,
                "error": "No delegated Microsoft Graph access token configured",
                "details": (
                    "Sign in via /api/auth/microsoft/login, or set MICROSOFT_COPILOT_ACCESS_TOKEN "
                    "for local development."
                ),
            }
        return {
            "available": True,
            "provider": "microsoft_copilot",
            "model": self._model,
            "details": "Delegated Graph token present (Chat API reachability verified on first call).",
        }

    # -- generation ------------------------------------------------------

    async def generate(
        self,
        *,
        node: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        response_format: str | None = None,
        num_ctx: int | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        import litellm

        request_id = uuid.uuid4().hex[:8]
        effective_timeout = timeout_seconds or self._timeout
        token = (kwargs.get("user_token") or self._access_token or "").strip()

        if not token:
            raise LLMAuthenticationError(
                "Microsoft 365 Copilot: no delegated Graph access token available. Sign in with a "
                "work account (Microsoft 365 Copilot licensed) or set MICROSOFT_COPILOT_ACCESS_TOKEN."
            )

        _register_handler()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if temperature not in (0.0, None) or max_output_tokens or num_ctx:
            logger.debug(
                "microsoft_copilot: ignoring unsupported params (temperature/max_output_tokens/num_ctx) node=%s",
                node,
            )

        prompt_chars = len(prompt) + (len(system_prompt) if system_prompt else 0)
        logger.info(
            "LLM REQUEST provider=microsoft_copilot model=%s node=%s request_id=%s prompt_chars=%d "
            "timeout_s=%.1f format=%s",
            self._model, node, request_id, prompt_chars, effective_timeout, response_format or "text",
        )

        t_start = time.monotonic()
        try:
            completion = await litellm.acompletion(
                model=_LITELLM_MODEL,
                messages=messages,
                api_key=token,
                timeout=effective_timeout,
                num_retries=self._max_retries,
                response_format={"type": "json_object"} if response_format == "json" else None,
            )
        except litellm.AuthenticationError as exc:
            raise LLMAuthenticationError(f"Microsoft 365 Copilot authentication failed: {exc}") from exc
        except litellm.RateLimitError as exc:
            raise LLMRateLimitError(f"Microsoft 365 Copilot rate limited: {exc}") from exc
        except litellm.Timeout as exc:
            raise LLMTimeoutError(
                f"Microsoft 365 Copilot request timed out after {effective_timeout}s."
            ) from exc
        except (litellm.ServiceUnavailableError, litellm.InternalServerError, litellm.BadGatewayError) as exc:
            raise LLMUnavailableError(f"Microsoft 365 Copilot is currently unavailable: {exc}") from exc
        except litellm.APIConnectionError as exc:
            raise LLMConnectionError(f"Microsoft 365 Copilot connection failure: {exc}") from exc
        except litellm.APIError as exc:
            raise LLMProviderError(f"Microsoft 365 Copilot execution failed: {exc}") from exc
        except LLMProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize anything else
            raise LLMProviderError(f"Microsoft 365 Copilot unexpected failure: {exc}") from exc

        elapsed_ms = int((time.monotonic() - t_start) * 1000)

        try:
            content = completion.choices[0].message.content or ""
        except (AttributeError, IndexError):
            content = ""
        if not content.strip():
            raise LLMInvalidResponseError("Microsoft 365 Copilot returned empty completion content.")

        hidden = getattr(completion, "_hidden_params", {}) or {}
        graph_request_id = hidden.get("graph_request_id", "")

        logger.info(
            "LLM RESPONSE provider=microsoft_copilot model=%s node=%s request_id=%s "
            "graph_request_id=%s elapsed_ms=%d success=true",
            self._model, node, request_id, graph_request_id, elapsed_ms,
        )

        return LLMResponse(
            content=content,
            provider="microsoft_copilot",
            model=self._model,
            latency_ms=elapsed_ms,
            input_tokens=None,
            output_tokens=None,
            finish_reason="stop",
            raw_metadata={
                "node": node,
                "request_id": request_id,
                "graph_request_id": graph_request_id,
                "elapsed_ms": elapsed_ms,
                "attributions": hidden.get("attributions", []),
                "sensitivity_label": hidden.get("sensitivity_label"),
            },
        )
