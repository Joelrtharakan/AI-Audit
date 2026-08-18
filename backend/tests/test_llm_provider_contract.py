"""Provider Contract Tests for Ollama and GitHub Copilot SDK.

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
from app.services.llm.providers.github_copilot_provider import GitHubCopilotProvider
from app.services.llm.providers.ollama_provider import OllamaProvider


class TestProviderFactory:
    """Test single authoritative provider factory selection."""

    def test_factory_returns_ollama_provider(self):
        with patch("app.services.llm.factory.get_settings") as mock_settings:
            mock_settings.return_value = Settings(llm_provider="ollama", ollama_model="qwen3:8b")
            provider = get_llm_provider()
            assert isinstance(provider, OllamaProvider)
            assert provider._model == "qwen3:8b"

    def test_factory_returns_copilot_provider(self):
        with patch("app.services.llm.factory.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                llm_provider="copilot", copilot_model="auto", copilot_github_token="mock_token"
            )
            provider = get_llm_provider()
            assert isinstance(provider, GitHubCopilotProvider)
            assert provider._model == "auto"

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


class TestGitHubCopilotProviderContract:
    """Test GitHubCopilotProvider behavior and error mapping."""

    @pytest.mark.asyncio
    async def test_copilot_successful_generation_normalization(self):
        provider = GitHubCopilotProvider(model="auto", github_token="mock_gh_token", timeout_seconds=10.0)

        mock_event_data = MagicMock()
        mock_event_data.content = '{"root_cause_status": "NOT_ESTABLISHED", "category": "OTHER"}'
        mock_event = MagicMock()
        mock_event.data = mock_event_data

        mock_session = AsyncMock()
        mock_session.send_and_wait.return_value = mock_event

        mock_client = AsyncMock()
        mock_client.create_session.return_value = mock_session

        with patch(
            "app.services.llm.providers.github_copilot_provider._get_shared_copilot_client",
            return_value=mock_client,
        ):
            response = await provider.generate(
                node="core_synthesis",
                prompt="Analyze finding",
                system_prompt="System prompt",
                response_format="json",
            )

            assert isinstance(response, LLMResponse)
            assert response.provider == "copilot"
            assert response.model == "auto"
            assert response.content == '{"root_cause_status": "NOT_ESTABLISHED", "category": "OTHER"}'
            assert response.finish_reason == "stop"
            assert response.raw_metadata["node"] == "core_synthesis"

    @pytest.mark.asyncio
    async def test_copilot_timeout_error_mapping(self):
        provider = GitHubCopilotProvider(model="auto", timeout_seconds=2.0)

        mock_session = AsyncMock()
        mock_session.send_and_wait.side_effect = asyncio.TimeoutError("Timeout waiting for idle")

        mock_client = AsyncMock()
        mock_client.create_session.return_value = mock_session

        with patch(
            "app.services.llm.providers.github_copilot_provider._get_shared_copilot_client",
            return_value=mock_client,
        ):
            with pytest.raises(LLMTimeoutError):
                await provider.generate(node="core_synthesis", prompt="Analyze finding")

    @pytest.mark.asyncio
    async def test_copilot_auth_error_mapping(self):
        provider = GitHubCopilotProvider(model="auto", github_token="invalid_token")

        mock_client = AsyncMock()
        mock_client.create_session.side_effect = Exception("Unauthorized: invalid GitHub token")

        with patch(
            "app.services.llm.providers.github_copilot_provider._get_shared_copilot_client",
            return_value=mock_client,
        ):
            with pytest.raises(LLMAuthenticationError):
                await provider.generate(node="extraction", prompt="Extract finding")

    @pytest.mark.asyncio
    async def test_copilot_empty_response_error_mapping(self):
        provider = GitHubCopilotProvider(model="auto")

        mock_event_data = MagicMock()
        mock_event_data.content = ""
        mock_event = MagicMock()
        mock_event.data = mock_event_data

        mock_session = AsyncMock()
        mock_session.send_and_wait.return_value = mock_event

        mock_client = AsyncMock()
        mock_client.create_session.return_value = mock_session

        with patch(
            "app.services.llm.providers.github_copilot_provider._get_shared_copilot_client",
            return_value=mock_client,
        ):
            with pytest.raises(LLMInvalidResponseError):
                await provider.generate(node="critic", prompt="Critic review")


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
@pytest.mark.asyncio
async def test_live_github_copilot_integration():
    """Live integration test against GitHub Copilot.

    Runs only when explicitly invoked and authenticated:
        pytest -m live_copilot
    """
    provider = GitHubCopilotProvider(model="auto")
    response = await provider.generate(
        node="test_live",
        prompt="Respond with valid JSON: {\"status\": \"copilot_live_ready\"}",
        response_format="json",
    )
    assert response.provider == "copilot"
    parsed = parse_llm_json(response.content)
    assert parsed.get("status") == "copilot_live_ready"
