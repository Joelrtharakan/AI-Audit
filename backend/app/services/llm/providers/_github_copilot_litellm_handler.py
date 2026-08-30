"""LiteLLM custom-provider handler for GitHub Copilot -- using OUR GitHub OAuth
/ session authentication, NOT LiteLLM's native `github_copilot/` route (which
launches its own interactive device-auth flow and cannot consume our session
tokens).

Registered under the provider name ``github_copilot_session``. Reached through
the same LiteLLM boundary as every other provider::

    await litellm.acompletion(
        model="github_copilot_session/auto",
        messages=[...],
        api_key="<GitHub user OAuth token from the authenticated session>",
    )

The handler delegates to the existing, tested
``app.services.llm.providers.github_copilot_provider.GitHubCopilotProvider``
(the ``github-copilot-sdk`` ``CopilotClient``) and wraps its normalized
``LLMResponse`` into a ``litellm.ModelResponse``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PROVIDER_NAME = "github_copilot_session"
_registered = False


def register() -> None:
    """Idempotently register the handler in ``litellm.custom_provider_map``."""
    global _registered
    if _registered:
        return
    import litellm

    existing = list(getattr(litellm, "custom_provider_map", []) or [])
    if not any(e.get("provider") == _PROVIDER_NAME for e in existing):
        existing.append({"provider": _PROVIDER_NAME, "custom_handler": GitHubCopilotSessionLLM()})
        litellm.custom_provider_map = existing
    _registered = True


class GitHubCopilotSessionLLM:
    """``litellm.CustomLLM``-compatible handler delegating to GitHubCopilotProvider."""

    def completion(self, *args: Any, **kwargs: Any):  # pragma: no cover - sync unused
        raise NotImplementedError("github_copilot_session supports async completion only.")

    def streaming(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("github_copilot_session streaming is not wired.")

    async def astreaming(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("github_copilot_session streaming is not wired.")

    async def acompletion(self, *args: Any, **kwargs: Any):
        import litellm
        from litellm.types.utils import ModelResponse

        from app.services.llm.providers.github_copilot_provider import GitHubCopilotProvider

        messages: list[dict[str, Any]] = kwargs.get("messages") or []
        model_id: str = kwargs.get("model") or "auto"
        # LiteLLM strips the "github_copilot_session/" prefix before calling us;
        # `model_id` is the bare model ("auto", "gpt-4o", ...).
        api_key: str | None = kwargs.get("api_key")
        optional_params: dict[str, Any] = kwargs.get("optional_params") or {}
        timeout = kwargs.get("timeout") or 90.0

        if not api_key:
            raise litellm.AuthenticationError(
                message="GitHub Copilot: no GitHub OAuth token supplied (api_key).",
                llm_provider=_PROVIDER_NAME,
                model=model_id,
            )

        system_parts = [m["content"] for m in messages if m.get("role") in ("system", "developer") and m.get("content")]
        user_parts = [m["content"] for m in messages if m.get("role") not in ("system", "developer") and m.get("content")]
        system_prompt = "\n\n".join(system_parts) or None
        prompt = "\n\n".join(user_parts)

        rf = optional_params.get("response_format") or kwargs.get("response_format")
        want_json = bool(rf) and (rf == "json" or (isinstance(rf, dict) and "json" in str(rf.get("type", ""))))

        provider = GitHubCopilotProvider(
            model=(None if model_id in ("auto", "") else model_id),
            github_token=api_key,
            timeout_seconds=float(timeout),
        )
        try:
            resp = await provider.generate(
                node=optional_params.get("node") or "github_copilot",
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=float(optional_params.get("temperature", 0.0) or 0.0),
                max_output_tokens=optional_params.get("max_tokens"),
                response_format="json" if want_json else None,
                timeout_seconds=float(timeout),
            )
        except Exception as exc:  # normalize into a LiteLLM exception
            _norm = _to_litellm_error(exc, model_id)
            raise _norm from exc

        mr = ModelResponse()
        mr.choices[0].message.content = resp.content  # type: ignore[union-attr]
        mr.choices[0].finish_reason = resp.finish_reason or "stop"
        mr.model = f"{_PROVIDER_NAME}/{model_id}"
        try:
            mr.usage = litellm.Usage(  # type: ignore[attr-defined]
                prompt_tokens=resp.input_tokens or 0,
                completion_tokens=resp.output_tokens or 0,
                total_tokens=(resp.input_tokens or 0) + (resp.output_tokens or 0),
            )
        except Exception:  # pragma: no cover
            pass
        mr._hidden_params = {"latency_ms": resp.latency_ms, **(resp.raw_metadata or {})}
        return mr


def _to_litellm_error(exc: Exception, model_id: str):
    import litellm

    from app.services.llm.exceptions import (
        LLMAuthenticationError,
        LLMConnectionError,
        LLMRateLimitError,
        LLMTimeoutError,
        LLMUnavailableError,
    )

    if isinstance(exc, LLMAuthenticationError):
        return litellm.AuthenticationError(message=str(exc), llm_provider=_PROVIDER_NAME, model=model_id)
    if isinstance(exc, LLMRateLimitError):
        return litellm.RateLimitError(message=str(exc), llm_provider=_PROVIDER_NAME, model=model_id)
    if isinstance(exc, LLMTimeoutError):
        return litellm.Timeout(message=str(exc), llm_provider=_PROVIDER_NAME, model=model_id)
    if isinstance(exc, (LLMUnavailableError,)):
        return litellm.ServiceUnavailableError(message=str(exc), llm_provider=_PROVIDER_NAME, model=model_id)
    if isinstance(exc, LLMConnectionError):
        return litellm.APIConnectionError(message=str(exc), llm_provider=_PROVIDER_NAME, model=model_id)
    return litellm.APIError(status_code=502, message=str(exc), llm_provider=_PROVIDER_NAME, model=model_id)
