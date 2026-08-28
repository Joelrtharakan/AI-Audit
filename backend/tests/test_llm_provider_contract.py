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
from app.services.llm.providers.microsoft_copilot_provider import MicrosoftCopilotProvider
from app.services.llm.providers.ollama_provider import OllamaProvider

import httpx
import respx


class TestProviderFactory:
    """Test single authoritative provider factory selection."""

    def test_factory_returns_ollama_provider(self):
        with patch("app.services.llm.factory.get_settings") as mock_settings:
            mock_settings.return_value = Settings(llm_provider="ollama", ollama_model="qwen3:8b")
            provider = get_llm_provider()
            assert isinstance(provider, OllamaProvider)
            assert provider._model == "qwen3:8b"

    def test_factory_returns_microsoft_copilot_provider(self):
        with patch("app.services.llm.factory.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                llm_provider="microsoft_copilot", microsoft_copilot_access_token="mock_token"
            )
            provider = get_llm_provider()
            assert isinstance(provider, MicrosoftCopilotProvider)
            assert provider._model == "m365-chat"

    def test_factory_copilot_alias_maps_to_microsoft(self):
        with patch("app.services.llm.factory.get_settings") as mock_settings:
            mock_settings.return_value = Settings(llm_provider="copilot")
            assert isinstance(get_llm_provider(), MicrosoftCopilotProvider)

    def test_factory_rejects_unsupported_provider(self):
        with patch("app.services.llm.factory.get_settings") as mock_settings:
            mock_settings.return_value = Settings(llm_provider="unsupported_provider_xyz")
            with pytest.raises(UnsupportedLLMProviderError):
                get_llm_provider()


class TestOllamaProviderContract:
    """Test OllamaProvider behavior and error mapping."""

    @pytest.mark.asyncio
    async def test_ollama_successful_generation_normalization(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen3:8b")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": '{"root_cause_status": "NOT_ESTABLISHED"}'},
            "done_reason": "stop",
            "eval_count": 42,
            "prompt_eval_count": 128,
            "eval_duration": 1_000_000_000,
            "prompt_eval_duration": 500_000_000,
            "load_duration": 10_000_000,
        }

        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response

        with patch("app.services.llm.providers.ollama_provider._get_shared_client", return_value=mock_http_client):
            response = await provider.generate(
                node="core_synthesis",
                prompt="Analyze the finding",
                system_prompt="System instructions",
                response_format="json",
            )

            assert isinstance(response, LLMResponse)
            assert response.provider == "ollama"
            assert response.model == "qwen3:8b"
            assert response.content == '{"root_cause_status": "NOT_ESTABLISHED"}'
            assert response.finish_reason == "stop"
            assert response.output_tokens == 42
            assert response.input_tokens == 128
            assert response.raw_metadata["hit_output_limit"] is False

    @pytest.mark.asyncio
    async def test_ollama_timeout_error_mapping(self):
        import httpx

        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen3:8b", timeout_seconds=1.0)
        mock_http_client = AsyncMock()
        mock_http_client.post.side_effect = httpx.TimeoutException("Read timed out")

        with patch("app.services.llm.providers.ollama_provider._get_shared_client", return_value=mock_http_client):
            with pytest.raises(LLMTimeoutError):
                await provider.generate(node="extraction", prompt="Extract finding")

    @pytest.mark.asyncio
    async def test_ollama_connection_error_mapping(self):
        import httpx

        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen3:8b")
        mock_http_client = AsyncMock()
        mock_http_client.post.side_effect = httpx.ConnectError("Connection refused")

        with patch("app.services.llm.providers.ollama_provider._get_shared_client", return_value=mock_http_client):
            with pytest.raises(LLMConnectionError):
                await provider.generate(node="extraction", prompt="Extract finding")

    @pytest.mark.asyncio
    async def test_ollama_empty_response_error_mapping(self):
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen3:8b")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "   "},
            "done_reason": "stop",
        }

        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response

        with patch("app.services.llm.providers.ollama_provider._get_shared_client", return_value=mock_http_client):
            with pytest.raises(LLMInvalidResponseError):
                await provider.generate(node="critic", prompt="Critic review")


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
