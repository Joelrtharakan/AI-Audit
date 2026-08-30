"""Provider Contract Tests for Ollama and Microsoft 365 Copilot.

Validates that all LLM providers conform to the standardized LLMProvider contract:
1. Provider initialization & factory selection
2. Successful generation & normalized LLMResponse
3. Timeout error mapping to LLMTimeoutError
4. Connection error mapping to LLMConnectionError
5. Authentication error mapping to LLMAuthenticationError
6. Invalid/empty response mapping to LLMInvalidResponseError
7. Malformed JSON handling & schema validation
8. Metadata normalization
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.config import Settings
from app.services.llm.base import LLMProvider, LLMResponse
from app.services.llm.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMTimeoutError,
    UnsupportedLLMProviderError,
)
from app.services.llm.factory import get_llm_provider
from app.services.llm.json_parser import (
    extract_json_str,
    normalize_llm_output,
    parse_llm_json,
    validate_llm_schema,
)
from app.services.llm.providers.litellm_provider import LiteLLMProvider
from app.services.llm.providers.microsoft_copilot_provider import MicrosoftCopilotProvider

import httpx
import respx


def _fake_completion(content: str, finish_reason: str = "stop"):
    from litellm.types.utils import ModelResponse, Usage
    mr = ModelResponse()
    mr.choices[0].message.content = content
    mr.choices[0].finish_reason = finish_reason
    mr.usage = Usage(prompt_tokens=128, completion_tokens=42, total_tokens=170)
    return mr


class TestProviderFactory:
    """The factory returns exactly ONE boundary -- LiteLLMProvider -- bound to
    one resolved (provider, model)."""

    def _prov(self, *, llm_provider, **settings_kw):
        with patch("app.services.llm.execution.get_settings") as ms:
            ms.return_value = Settings(llm_provider=llm_provider, **settings_kw)
            # explicit provider_name so this never reads a leaked request ContextVar
            return get_llm_provider(provider_name=llm_provider, model=settings_kw.get("llm_model") or None)

    def test_factory_returns_litellm_provider_for_ollama(self):
        p = self._prov(llm_provider="ollama", llm_model="qwen3:8b")
        assert isinstance(p, LiteLLMProvider)
        assert p.config.provider == "ollama"
        assert p.config.litellm_model == "ollama_chat/qwen3:8b"

    def test_factory_ollama_default_model_when_llm_model_unset(self):
        p = self._prov(llm_provider="ollama", llm_model="", ollama_model="qwen3:8b")
        assert p.config.litellm_model == "ollama_chat/qwen3:8b"

    def test_factory_returns_litellm_provider_for_microsoft(self):
        p = self._prov(llm_provider="microsoft_copilot", microsoft_copilot_access_token="mock_token")
        assert isinstance(p, LiteLLMProvider)
        assert p.config.provider == "microsoft_copilot"
        assert p.config.litellm_model == "microsoft_copilot/m365-chat"

    def test_factory_returns_litellm_provider_for_github(self):
        p = self._prov(llm_provider="github_copilot", copilot_github_token="ghp_x")
        assert isinstance(p, LiteLLMProvider)
        assert p.config.provider == "github_copilot"
        assert p.config.litellm_model.startswith("github_copilot_session/")

    def test_factory_copilot_alias_maps_to_microsoft(self):
        assert self._prov(llm_provider="copilot").config.provider == "microsoft_copilot"

    def test_factory_rejects_unsupported_provider(self):
        with patch("app.services.llm.execution.get_settings") as ms:
            ms.return_value = Settings(llm_provider="unsupported_provider_xyz")
            with pytest.raises(UnsupportedLLMProviderError):
                get_llm_provider("unsupported_provider_xyz")


class TestLiteLLMBoundaryContract:
    """LiteLLMProvider: ONE call, normalized LLMResponse, mapped errors, no fan-out."""

    def _provider(self, provider="ollama", model="qwen3:8b", **kw):
        from app.services.llm.execution import resolve_execution_config, begin_request
        with patch("app.services.llm.execution.get_settings") as ms:
            ms.return_value = Settings(llm_provider=provider, llm_model=model, **kw)
            cfg = resolve_execution_config(provider, model)
        begin_request(cfg, request_id="TESTREQ")
        return LiteLLMProvider(cfg)

    @pytest.mark.asyncio
    async def test_single_acompletion_call_and_normalization(self):
        p = self._provider()
        with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_completion('{"ok": true}'))) as mock_ac:
            resp = await p.generate(node="core_synthesis", prompt="x", system_prompt="s", response_format="json")
        assert mock_ac.await_count == 1                       # exactly one call
        assert mock_ac.await_args.kwargs["model"] == "ollama_chat/qwen3:8b"
        assert mock_ac.await_args.kwargs["num_retries"] == 0  # no fallback
        assert isinstance(resp, LLMResponse)
        assert resp.provider == "ollama" and resp.model == "qwen3:8b"
        assert resp.content == '{"ok": true}'
        from app.services.llm.call_metadata import get_last_call_metadata
        meta = get_last_call_metadata()
        assert meta["provider_used"] == "ollama" and meta["fallback_used"] is False
        assert meta["provider_attempts"] == ["ollama"] and meta["request_id"] == "TESTREQ"

    @pytest.mark.asyncio
    async def test_timeout_maps_to_LLMTimeoutError(self):
        import litellm
        p = self._provider()
        with patch("litellm.acompletion", new=AsyncMock(side_effect=litellm.Timeout("slow", model="m", llm_provider="ollama"))):
            with pytest.raises(LLMTimeoutError):
                await p.generate(node="critic", prompt="x")

    @pytest.mark.asyncio
    async def test_connection_error_maps_to_LLMConnectionError(self):
        import litellm
        p = self._provider()
        with patch("litellm.acompletion", new=AsyncMock(side_effect=litellm.APIConnectionError(message="refused", model="m", llm_provider="ollama"))):
            with pytest.raises(LLMConnectionError):
                await p.generate(node="extraction", prompt="x")

    @pytest.mark.asyncio
    async def test_auth_error_maps_to_LLMAuthenticationError(self):
        import litellm
        p = self._provider(provider="microsoft_copilot", model="m365-chat", microsoft_copilot_access_token="bad")
        with patch("litellm.acompletion", new=AsyncMock(side_effect=litellm.AuthenticationError(message="401", model="m", llm_provider="microsoft_copilot"))):
            with pytest.raises(LLMAuthenticationError):
                await p.generate(node="core_synthesis", prompt="x")

    @pytest.mark.asyncio
    async def test_empty_content_maps_to_LLMInvalidResponseError(self):
        p = self._provider()
        with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_completion("   "))):
            with pytest.raises(LLMInvalidResponseError):
                await p.generate(node="critic", prompt="x")

    @pytest.mark.asyncio
    async def test_no_fallback_even_when_first_provider_fails(self):
        """A failure NEVER results in a second provider being attempted."""
        import litellm
        p = self._provider()
        with patch("litellm.acompletion", new=AsyncMock(side_effect=litellm.ServiceUnavailableError(message="503", model="m", llm_provider="ollama"))) as mock_ac:
            with pytest.raises(LLMProviderError):
                await p.generate(node="core_synthesis", prompt="x")
        assert mock_ac.await_count == 1  # one provider, one attempt, then fail-closed


def _conv_route(rsx, *, chat_status=200, chat_json=None, conv_status=201):
    rsx.post("https://graph.microsoft.com/beta/copilot/conversations").mock(
        return_value=httpx.Response(conv_status, json={"id": "c1", "status": "active"}, headers={"request-id": "r1"})
    )
    rsx.post(url__regex=r".*/conversations/c1/chat").mock(
        return_value=httpx.Response(chat_status, json=chat_json or {}, headers={"request-id": "r2"})
    )
    rsx.delete(url__regex=r".*/conversations/c1").mock(return_value=httpx.Response(204))


class TestMicrosoftCopilotProviderContract:
    """Test MicrosoftCopilotProvider behavior and error mapping (Graph Chat API mocked)."""

    @pytest.mark.asyncio
    async def test_successful_generation_normalization(self):
        provider = MicrosoftCopilotProvider(ms_access_token="mock_token", timeout_seconds=10.0)
        chat = {
            "messages": [
                {"text": "Analyze finding", "attributions": []},
                {
                    "text": '{"root_cause_status": "NOT_ESTABLISHED", "category": "OTHER"} [^1^]',
                    "attributions": [{"attributionType": "citation", "attributionSource": "grounding"}],
                },
            ]
        }
        with respx.mock(assert_all_called=False) as rsx:
            _conv_route(rsx, chat_json=chat)
            response = await provider.generate(
                node="core_synthesis",
                prompt="Analyze finding",
                system_prompt="System prompt",
                response_format="json",
            )
        assert isinstance(response, LLMResponse)
        assert response.provider == "microsoft_copilot"
        assert response.model == "m365-chat"
        assert response.content == '{"root_cause_status": "NOT_ESTABLISHED", "category": "OTHER"}'
        assert response.finish_reason == "stop"
        assert response.raw_metadata["node"] == "core_synthesis"
        assert response.raw_metadata["graph_request_id"] == "r2"

    @pytest.mark.asyncio
    async def test_missing_token_raises_auth_error(self):
        provider = MicrosoftCopilotProvider(ms_access_token="")
        with pytest.raises(LLMAuthenticationError):
            await provider.generate(node="extraction", prompt="Extract finding")

    @pytest.mark.asyncio
    async def test_auth_error_mapping(self):
        provider = MicrosoftCopilotProvider(ms_access_token="invalid")
        with respx.mock(assert_all_called=False) as rsx:
            rsx.post("https://graph.microsoft.com/beta/copilot/conversations").mock(
                return_value=httpx.Response(401, json={"error": "unauthorized"})
            )
            with pytest.raises(LLMAuthenticationError):
                await provider.generate(node="extraction", prompt="Extract finding")

    @pytest.mark.asyncio
    async def test_rate_limit_error_mapping(self):
        provider = MicrosoftCopilotProvider(ms_access_token="tok")
        with respx.mock(assert_all_called=False) as rsx:
            rsx.post("https://graph.microsoft.com/beta/copilot/conversations").mock(
                return_value=httpx.Response(429, headers={"Retry-After": "5"}, json={})
            )
            from app.services.llm.exceptions import LLMRateLimitError

            with pytest.raises(LLMRateLimitError):
                await provider.generate(node="rca", prompt="analyze")

    @pytest.mark.asyncio
    async def test_service_unavailable_error_mapping(self):
        provider = MicrosoftCopilotProvider(ms_access_token="tok")
        with respx.mock(assert_all_called=False) as rsx:
            rsx.post("https://graph.microsoft.com/beta/copilot/conversations").mock(
                return_value=httpx.Response(504, json={})
            )
            from app.services.llm.exceptions import LLMUnavailableError

            with pytest.raises(LLMUnavailableError):
                await provider.generate(node="rca", prompt="analyze")

    @pytest.mark.asyncio
    async def test_empty_response_error_mapping(self):
        provider = MicrosoftCopilotProvider(ms_access_token="tok")
        with respx.mock(assert_all_called=False) as rsx:
            _conv_route(rsx, chat_json={"messages": [{"text": "analyze", "attributions": []}, {"text": "   "}]})
            with pytest.raises((LLMInvalidResponseError, LLMProviderError)):
                await provider.generate(node="critic", prompt="Critic review")

    @pytest.mark.asyncio
    async def test_unsupported_params_accepted_and_ignored(self):
        provider = MicrosoftCopilotProvider(ms_access_token="tok")
        chat = {"messages": [{"text": "p"}, {"text": '{"ok": true}'}]}
        with respx.mock(assert_all_called=False) as rsx:
            _conv_route(rsx, chat_json=chat)
            response = await provider.generate(
                node="n", prompt="p", temperature=0.9, max_output_tokens=999, num_ctx=8192
            )
        assert response.content == '{"ok": true}'


class TestProviderNeutralJsonParser:
    """Test robust JSON parser and recovery mechanisms across all providers."""

    def test_parses_plain_json(self):
        raw = '{"status": "OK", "count": 5}'
        assert parse_llm_json(raw) == {"status": "OK", "count": 5}

    def test_strips_markdown_code_fences(self):
        raw = "```json\n{\n  \"status\": \"OK\",\n  \"value\": 10\n}\n```"
        assert parse_llm_json(raw) == {"status": "OK", "value": 10}

    def test_strips_think_tags(self):
        raw = "<think>Analyzing root cause step by step...</think>\n{\"root_cause\": \"CALIBRATION\"}"
        assert parse_llm_json(raw) == {"root_cause": "CALIBRATION"}

    def test_removes_trailing_commas(self):
        raw = '{"items": ["a", "b",], "done": true,}'
        assert parse_llm_json(raw) == {"items": ["a", "b"], "done": True}

    def test_extracts_embedded_json_object(self):
        raw = 'Here is the analysis:\n{"hypothesis": "H1", "support": "VERIFIED"}\nHope this helps!'
        assert parse_llm_json(raw) == {"hypothesis": "H1", "support": "VERIFIED"}

    def test_validates_schema_keys(self):
        data = {"a": 1, "b": 2}
        assert validate_llm_schema(data, ["a", "b"]) == []
        assert validate_llm_schema(data, ["a", "b", "c"]) == ["c"]

    def test_empty_input_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_llm_json("")


@pytest.mark.live_copilot
@pytest.mark.skipif(
    not __import__("os").getenv("MICROSOFT_COPILOT_ACCESS_TOKEN"),
    reason="requires a delegated Microsoft Graph token in MICROSOFT_COPILOT_ACCESS_TOKEN",
)
@pytest.mark.asyncio
async def test_live_microsoft_copilot_integration():
    """Live integration test against the Microsoft 365 Copilot Chat API.

    Runs only when explicitly invoked with a delegated token available
    (MICROSOFT_COPILOT_ACCESS_TOKEN) and:
        pytest -m live_copilot
    """
    provider = MicrosoftCopilotProvider()
    response = await provider.generate(
        node="test_live",
        prompt='Respond with valid JSON: {"status": "copilot_live_ready"}',
        response_format="json",
    )
    assert response.provider == "microsoft_copilot"
    parsed = parse_llm_json(response.content)
    assert parsed.get("status") == "copilot_live_ready"
