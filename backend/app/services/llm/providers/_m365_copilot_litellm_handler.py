"""LiteLLM custom-provider handler for the Microsoft 365 Copilot Chat API.

LiteLLM has no native Microsoft 365 Copilot provider, so this module registers a
`CustomLLM` handler under the provider name ``microsoft_copilot``. Once registered
(see :func:`register`), callers reach it through the standard LiteLLM entrypoint::

    await litellm.acompletion(
        model="microsoft_copilot/m365-chat",
        messages=[...],
        api_key="<delegated Microsoft Graph access token>",
    )

The handler translates the OpenAI-style request into the Graph beta Copilot Chat
API sequence:

    POST {graph}/beta/copilot/conversations            -> conversationId
    POST {graph}/beta/copilot/conversations/{id}/chat  -> synthesized text answer
    DELETE {graph}/beta/copilot/conversations/{id}      -> best-effort cleanup

Constraints of the Chat API that shape this adapter:
  * Delegated bearer token only; no model selection; text responses only.
  * No system-prompt field -> system/developer messages are folded into message.text.
  * No native JSON mode -> when a JSON response_format is requested we append an
    explicit "return only JSON" instruction; parsing/validation stays the caller's job.
  * locationHint.timeZone is required on every chat call.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import httpx

# Keep LiteLLM fully offline: never fetch its remote model-cost map (the Chat API
# reports no token usage and we set usage to zero anyway).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

logger = logging.getLogger(__name__)

_MARKUP_TOKEN_RE = re.compile(r"</?(?:Person|Event|File|Citation|Attachment)>", re.IGNORECASE)
_CITATION_MARKER_RE = re.compile(r"\[\^\d+\^\]")

_PROVIDER_NAME = "microsoft_copilot"
_handler: "M365CopilotLLM | None" = None


def register(
    *,
    graph_base_url: str = "https://graph.microsoft.com",
    timezone: str = "UTC",
    web_grounding: bool = False,
) -> "M365CopilotLLM":
    """Idempotently register this handler in ``litellm.custom_provider_map``.

    Safe to call repeatedly; updates the (non-secret, tenant-global) request
    configuration on the shared handler each time. The per-user bearer token is
    never stored here -- it is passed per call via ``api_key``.
    """
    global _handler
    import litellm

    if _handler is None:
        _handler = M365CopilotLLM()
        existing = list(getattr(litellm, "custom_provider_map", []) or [])
        if not any(entry.get("provider") == _PROVIDER_NAME for entry in existing):
            existing.append({"provider": _PROVIDER_NAME, "custom_handler": _handler})
            litellm.custom_provider_map = existing

    _handler.graph_base_url = graph_base_url.rstrip("/")
    _handler.timezone = timezone
    _handler.web_grounding = web_grounding
    return _handler


def _strip_copilot_markup(text: str) -> str:
    text = _MARKUP_TOKEN_RE.sub("", text)
    text = _CITATION_MARKER_RE.sub("", text)
    return text.strip()


def _flatten_messages(messages: list[dict[str, Any]], *, want_json: bool) -> tuple[str, list[str]]:
    """Fold an OpenAI message list into (message_text, additional_context_excerpts)."""
    system_parts: list[str] = []
    convo_parts: list[str] = []
    for msg in messages or []:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content")
        if isinstance(content, list):  # multimodal content blocks -> text only
            content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
        content = (content or "").strip()
        if not content:
            continue
        if role in ("system", "developer"):
            system_parts.append(content)
        elif role == "assistant":
            convo_parts.append(f"[assistant previously said]\n{content}")
        else:
            convo_parts.append(content)

    header = "\n\n".join(system_parts)
    if want_json:
        header = (header + "\n\n" if header else "") + (
            "Respond with ONLY a single valid JSON object. Do not include prose, "
            "explanations, markdown code fences, or citations."
        )
    body = "\n\n".join(convo_parts)
    message_text = (header + "\n\n" + body).strip() if header else body
    return message_text, []


class M365CopilotLLM:
    """``litellm.CustomLLM`` implementation backed by the Graph Copilot Chat API."""

    # NOTE: litellm.CustomLLM is imported lazily so this module stays importable
    # without litellm at collection time; register() binds the real base.
    def __init__(self) -> None:
        self.graph_base_url: str = "https://graph.microsoft.com"
        self.timezone: str = "UTC"
        self.web_grounding: bool = False

    # -- litellm entrypoints -------------------------------------------------

    def completion(self, *args: Any, **kwargs: Any):  # pragma: no cover - sync unused
        raise NotImplementedError("microsoft_copilot supports async completion only (use acompletion).")

    def streaming(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("microsoft_copilot streaming is not wired.")

    async def astreaming(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("microsoft_copilot streaming is not wired.")

    async def acompletion(self, *args: Any, **kwargs: Any):
        import litellm
        from litellm.types.utils import ModelResponse

        messages: list[dict[str, Any]] = kwargs.get("messages") or []
        model: str = kwargs.get("model") or "m365-chat"
        api_key: str | None = kwargs.get("api_key")
        optional_params: dict[str, Any] = kwargs.get("optional_params") or {}
        litellm_params: dict[str, Any] = kwargs.get("litellm_params") or {}
        timeout = kwargs.get("timeout") or litellm_params.get("timeout") or 90.0

        if not api_key:
            raise litellm.AuthenticationError(
                message="Microsoft 365 Copilot: no delegated Graph access token supplied (api_key).",
                llm_provider=_PROVIDER_NAME,
                model=model,
            )

        response_format = optional_params.get("response_format") or kwargs.get("response_format")
        want_json = bool(response_format) and (
            response_format == "json"
            or (isinstance(response_format, dict) and "json" in str(response_format.get("type", "")))
        )
        message_text, _extra = _flatten_messages(messages, want_json=want_json)

        graph_base = self.graph_base_url.rstrip("/")
        timezone = self.timezone
        web_enabled = self.web_grounding

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        t0 = time.monotonic()

        async with httpx.AsyncClient(timeout=float(timeout)) as client:
            conv_id, _ = await self._request(
                client, "POST", f"{graph_base}/beta/copilot/conversations", headers, {}, model
            )
            try:
                chat_body: dict[str, Any] = {
                    "message": {"text": message_text},
                    "locationHint": {"timeZone": timezone},
                }
                if not web_enabled:
                    chat_body["contextualResources"] = {"webContext": {"isWebEnabled": False}}

                payload, resp_headers = await self._request(
                    client,
                    "POST",
                    f"{graph_base}/beta/copilot/conversations/{conv_id}/chat",
                    headers,
                    chat_body,
                    model,
                    expect_id=False,
                )
            finally:
                try:
                    await client.request(
                        "DELETE",
                        f"{graph_base}/beta/copilot/conversations/{conv_id}",
                        headers=headers,
                    )
                except Exception:  # pragma: no cover - best effort
                    pass

        text = self._extract_answer(payload)
        if not text:
            raise litellm.APIError(
                status_code=502,
                message="Microsoft 365 Copilot returned an empty or unparseable answer.",
                llm_provider=_PROVIDER_NAME,
                model=model,
            )

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        graph_request_id = (
            resp_headers.get("request-id")
            or resp_headers.get("client-request-id")
            or resp_headers.get("x-ms-ags-diagnostic")
            or ""
        )

        model_response = ModelResponse()
        model_response.choices[0].message.content = text  # type: ignore[union-attr]
        model_response.choices[0].finish_reason = "stop"
        model_response.model = f"{_PROVIDER_NAME}/{model}"
        try:
            model_response.usage = litellm.Usage(  # type: ignore[attr-defined]
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            )
        except Exception:  # pragma: no cover
            pass
        model_response._hidden_params = {
            "graph_request_id": graph_request_id,
            "elapsed_ms": elapsed_ms,
            "attributions": self._collect_attributions(payload),
            "sensitivity_label": self._collect_sensitivity_label(payload),
        }
        return model_response

    # -- Graph plumbing --------------------------------------------------

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        model: str,
        *,
        expect_id: bool = True,
    ) -> tuple[Any, httpx.Headers]:
        import litellm

        try:
            resp = await client.request(method, url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise litellm.Timeout(
                message=f"Microsoft 365 Copilot request timed out: {exc}",
                llm_provider=_PROVIDER_NAME,
                model=model,
            ) from exc
        except httpx.HTTPError as exc:
            raise litellm.APIConnectionError(
                message=f"Microsoft 365 Copilot connection error: {exc}",
                llm_provider=_PROVIDER_NAME,
                model=model,
            ) from exc

        if resp.status_code in (401, 403):
            raise litellm.AuthenticationError(
                message=f"Microsoft 365 Copilot auth failed (HTTP {resp.status_code}): {resp.text[:400]}",
                llm_provider=_PROVIDER_NAME,
                model=model,
            )
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            raise litellm.RateLimitError(
                message=f"Microsoft 365 Copilot rate limited (Retry-After={retry_after}).",
                llm_provider=_PROVIDER_NAME,
                model=model,
            )
        if resp.status_code in (500, 502, 503, 504):
            raise litellm.ServiceUnavailableError(
                message=f"Microsoft 365 Copilot unavailable (HTTP {resp.status_code}).",
                llm_provider=_PROVIDER_NAME,
                model=model,
            )
        if resp.status_code >= 400:
            raise litellm.APIError(
                status_code=resp.status_code,
                message=f"Microsoft 365 Copilot error (HTTP {resp.status_code}): {resp.text[:400]}",
                llm_provider=_PROVIDER_NAME,
                model=model,
            )

        try:
            payload = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise litellm.APIError(
                status_code=resp.status_code,
                message=f"Microsoft 365 Copilot returned a non-JSON body: {exc}",
                llm_provider=_PROVIDER_NAME,
                model=model,
            ) from exc

        if expect_id:
            conv_id = payload.get("id")
            if not conv_id:
                raise litellm.APIError(
                    status_code=resp.status_code,
                    message="Microsoft 365 Copilot conversation creation returned no id.",
                    llm_provider=_PROVIDER_NAME,
                    model=model,
                )
            return conv_id, resp.headers
        return payload, resp.headers

    @staticmethod
    def _response_messages(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        return [m for m in (payload.get("messages") or []) if isinstance(m, dict)]

    def _extract_answer(self, payload: Any) -> str:
        # The assistant answer is the last response message that is NOT the echoed
        # user prompt. The echoed prompt has empty attributions/adaptiveCards and
        # matches the request text; the model answer carries adaptiveCards/attributions
        # or is simply the final message. Take the last message with non-empty text
        # that isn't the first (prompt) echo.
        messages = self._response_messages(payload)
        texts = [str(m.get("text") or "").strip() for m in messages]
        if not texts:
            return ""
        prompt_echo = texts[0]
        # The model answer is the last non-empty message that is not the echoed
        # user prompt (the Chat API always echoes the prompt as the first message).
        for candidate in reversed(texts[1:]):
            if candidate and candidate != prompt_echo:
                return _strip_copilot_markup(candidate)
        # Single-message conversations: only an answer if it isn't the echo.
        if len(texts) == 1 and texts[0]:
            return _strip_copilot_markup(texts[0])
        return ""

    def _collect_attributions(self, payload: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in self._response_messages(payload):
            for a in m.get("attributions") or []:
                if isinstance(a, dict):
                    out.append(
                        {
                            "type": a.get("attributionType"),
                            "source": a.get("attributionSource"),
                            "provider": a.get("providerDisplayName"),
                            "url": a.get("seeMoreWebUrl"),
                        }
                    )
        return out

    def _collect_sensitivity_label(self, payload: Any) -> dict[str, Any] | None:
        for m in reversed(self._response_messages(payload)):
            label = m.get("sensitivityLabel")
            if isinstance(label, dict) and label.get("displayName"):
                return label
        return None
