"""Production GitHub Copilot Provider for LQMS Audit Investigation Engine.

Uses the official `github-copilot-sdk` (CopilotClient) to execute LLM reasoning
sessions in production environments.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
import weakref
from typing import Any

from app.config import get_settings
from app.services.llm.base import LLMProvider, LLMResponse
from app.services.llm.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

_client_lock = asyncio.Lock()
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any]" = weakref.WeakKeyDictionary()
_client_tokens: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, str]" = weakref.WeakKeyDictionary()


def reset_copilot_clients() -> None:
    """Clear cached Copilot clients across event loops (e.g. after updating token)."""
    _clients.clear()
    _client_tokens.clear()


async def _get_shared_copilot_client(github_token: str | None = None, log_level: str = "info") -> Any:
    """Obtain or initialize a started CopilotClient tied to the running event loop."""
    try:
        from copilot import CopilotClient
    except ImportError as exc:
        raise LLMUnavailableError(
            "github-copilot-sdk is not installed. Install via `pip install github-copilot-sdk`."
        ) from exc

    loop = asyncio.get_running_loop()
    token = github_token or os.getenv("COPILOT_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    clean_token = token.strip() if token else ""

    client = _clients.get(loop)
    cached_token = _client_tokens.get(loop, "")
    if client is not None and cached_token == clean_token and clean_token:
        return client

    async with _client_lock:
        client = _clients.get(loop)
        cached_token = _client_tokens.get(loop, "")
        if client is not None and cached_token == clean_token and clean_token:
            return client

        if not clean_token:
            raise LLMAuthenticationError(
                "GitHub Copilot authentication token missing. Please paste your GitHub Token in the frontend "
                "or add COPILOT_GITHUB_TOKEN=ghp_... to backend/.env."
            )

        clean_log_level = (log_level or "info").strip().lower()
        if clean_log_level not in ("none", "error", "warning", "info", "debug", "all", "default"):
            clean_log_level = "info"

        client_kwargs: dict[str, Any] = {"log_level": clean_log_level, "github_token": clean_token}

        copilot_client = CopilotClient(**client_kwargs)
        try:
            await asyncio.wait_for(copilot_client.start(), timeout=2.5)
        except asyncio.TimeoutError as exc:
            raise LLMConnectionError("GitHub Copilot runtime initialization timed out.") from exc
        except Exception as exc:
            err_msg = str(exc).lower()
            if "auth" in err_msg or "token" in err_msg or "unauthorized" in err_msg:
                raise LLMAuthenticationError(f"GitHub Copilot authentication failed: {exc}") from exc
            if "connect" in err_msg or "socket" in err_msg or "runtime" in err_msg or "spawn" in err_msg or "enoent" in err_msg:
                raise LLMConnectionError(f"GitHub Copilot runtime connection failed: {exc}") from exc
            raise LLMProviderError(f"Failed to start GitHub Copilot client: {exc}") from exc

        _clients[loop] = copilot_client
        _client_tokens[loop] = clean_token
        return copilot_client


class GitHubCopilotProvider(LLMProvider):
    """GitHub Copilot SDK implementation of LLMProvider for production inference."""

    def __init__(
        self,
        model: str | None = None,
        github_token: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        self._model = model or settings.copilot_model or "auto"
        self._github_token = github_token or settings.copilot_github_token or None
        self._timeout = timeout_seconds or settings.copilot_timeout_seconds

    async def check_health(self) -> dict[str, Any]:
        """Check Copilot client readiness and auth status."""
        token = self._github_token or os.getenv("COPILOT_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        has_token = bool(token and token.strip())
        
        # Fast path if token is missing
        if not has_token:
            return {
                "available": False,
                "provider": "copilot",
                "model": self._model,
                "error": "COPILOT_GITHUB_TOKEN is not configured",
                "details": "Set COPILOT_GITHUB_TOKEN in .env or environment for production mode.",
            }

        try:
            client = await asyncio.wait_for(
                _get_shared_copilot_client(
                    github_token=token,
                    log_level=self._settings.copilot_log_level,
                ),
                timeout=2.0,
            )
            auth_status = None
            if hasattr(client, "get_auth_status"):
                auth_status = await asyncio.wait_for(client.get_auth_status(), timeout=2.0)
            return {
                "available": True,
                "provider": "copilot",
                "model": self._model,
                "auth_status": auth_status,
            }
        except (LLMAuthenticationError, asyncio.TimeoutError) as exc:
            return {
                "available": False,
                "provider": "copilot",
                "model": self._model,
                "error": "Client not ready or authentication failed",
                "details": str(exc),
            }
        except Exception as exc:
            return {
                "available": False,
                "provider": "copilot",
                "model": self._model,
                "error": str(exc),
            }

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
        try:
            from copilot.session import PermissionHandler
        except ImportError:
            PermissionHandler = None

        request_id = uuid.uuid4().hex[:8]
        effective_timeout = timeout_seconds or self._timeout
        effective_model = self._model if self._model != "auto" else None

        prompt_chars = len(prompt) + (len(system_prompt) if system_prompt else 0)
        estimated_input_tokens = prompt_chars // 4

        logger.info(
            "LLM REQUEST provider=copilot model=%s node=%s request_id=%s prompt_chars=%d "
            "estimated_input_tokens=%d timeout_s=%.1f format=%s",
            self._model,
            node,
            request_id,
            prompt_chars,
            estimated_input_tokens,
            effective_timeout,
            response_format or "text",
        )

        t_start = time.monotonic()
        effective_token = kwargs.get("user_token") or self._github_token
        effective_user_id = str(kwargs.get("user_id") or "user")

        client = await _get_shared_copilot_client(
            github_token=effective_token,
            log_level=self._settings.copilot_log_level,
        )

        # Isolated session per request to prevent cross-finding conversation contamination
        session_opts: dict[str, Any] = {}
        if effective_model:
            session_opts["model"] = effective_model
        if PermissionHandler and hasattr(PermissionHandler, "approve_all"):
            session_opts["on_permission_request"] = PermissionHandler.approve_all

        # Prepare system prompt instructions
        if system_prompt:
            session_opts["system_message"] = {"mode": "replace", "content": system_prompt}

        session = None
        try:
            session = await client.create_session(**session_opts)
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            err_msg = str(exc).lower()
            logger.error(
                "LLM REQUEST FAILED provider=copilot node=%s model=%s request_id=%s "
                "failure_type=SESSION_CREATION_FAILED elapsed_ms=%d err=%s",
                node,
                self._model,
                request_id,
                elapsed_ms,
                exc,
            )
            if "auth" in err_msg or "token" in err_msg or "unauthorized" in err_msg:
                raise LLMAuthenticationError(f"Copilot session creation unauthorized: {exc}") from exc
            raise LLMProviderError(f"Copilot session creation failed: {exc}") from exc

        content = ""
        try:
            event = await session.send_and_wait(prompt, timeout=effective_timeout)
            if event is None or getattr(event, "data", None) is None:
                raise LLMInvalidResponseError("Copilot returned an empty event response.")

            event_data = event.data
            if hasattr(event_data, "content") and event_data.content is not None:
                content = str(event_data.content)
            elif hasattr(event_data, "message") and event_data.message is not None:
                content = str(event_data.message)
            else:
                content = str(event_data)

            if not content or not content.strip():
                raise LLMInvalidResponseError("Copilot returned empty completion content.")
        except (TimeoutError, asyncio.TimeoutError) as exc:
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            logger.error(
                "LLM REQUEST FAILED provider=copilot node=%s model=%s request_id=%s failure_type=TIMEOUT elapsed_ms=%d",
                node,
                self._model,
                request_id,
                elapsed_ms,
            )
            raise LLMTimeoutError(f"Copilot request timed out after {effective_timeout}s.") from exc
        except (LLMProviderError, LLMInvalidResponseError):
            raise
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            err_msg = str(exc).lower()
            logger.error(
                "LLM REQUEST FAILED provider=copilot node=%s model=%s request_id=%s failure_type=PROVIDER_ERROR elapsed_ms=%d err=%s",
                node,
                self._model,
                request_id,
                elapsed_ms,
                exc,
            )
            if "auth" in err_msg or "token" in err_msg:
                raise LLMAuthenticationError(f"Copilot authentication error: {exc}") from exc
            if "connection" in err_msg or "network" in err_msg or "offline" in err_msg:
                raise LLMConnectionError(f"Copilot connection failure: {exc}") from exc
            raise LLMProviderError(f"Copilot execution failed: {exc}") from exc
        finally:
            if session is not None:
                try:
                    if hasattr(session, "delete"):
                        await session.delete()
                    elif hasattr(session, "close"):
                        await session.close()
                except Exception:
                    pass

        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        output_tokens = len(content) // 4

        logger.info(
            "LLM RESPONSE provider=copilot model=%s node=%s request_id=%s elapsed_ms=%d "
            "estimated_output_tokens=%d success=true",
            self._model,
            node,
            request_id,
            elapsed_ms,
            output_tokens,
        )

        return LLMResponse(
            content=content,
            provider="copilot",
            model=self._model,
            latency_ms=elapsed_ms,
            input_tokens=estimated_input_tokens,
            output_tokens=output_tokens,
            finish_reason="stop",
            raw_metadata={
                "node": node,
                "request_id": request_id,
                "elapsed_ms": elapsed_ms,
            },
        )
