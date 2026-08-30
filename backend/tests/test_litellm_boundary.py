"""LiteLLM is the single inference boundary: ONE provider + ONE model, frozen
once per request, used by every LLM stage. No fan-out, no fallback, no duplicate
provider paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.config import Settings
from app.services.llm.call_metadata import get_last_call_metadata
from app.services.llm.execution import (
    begin_request,
    current_execution_config,
    current_request_id,
    resolve_execution_config,
)


def _fake_completion(content='{"ok": true}'):
    from litellm.types.utils import ModelResponse, Usage
    mr = ModelResponse()
    mr.choices[0].message.content = content
    mr.choices[0].finish_reason = "stop"
    mr.usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return mr


@pytest.fixture
def _ollama_request():
    with patch("app.services.llm.execution.get_settings") as ms:
        ms.return_value = Settings(llm_provider="ollama", llm_model="qwen3:8b")
        cfg = resolve_execution_config("ollama", "qwen3:8b")
    rid = begin_request(cfg, request_id="RID-1")
    yield rid


@pytest.mark.asyncio
async def test_every_stage_of_one_request_uses_the_same_frozen_route(_ollama_request):
    from app.services.llm_client import get_llm_client

    seen_models: list[str] = []
    seen_retries: list[int] = []

    async def _spy(*args, **kwargs):
        seen_models.append(kwargs["model"])
        seen_retries.append(kwargs.get("num_retries"))
        return _fake_completion()

    with patch("litellm.acompletion", new=AsyncMock(side_effect=_spy)):
        # Simulate the intentional LLM stages of one investigation.
        for node in ("understanding", "core_synthesis", "critic", "remediation_cost_interpretation"):
            client = get_llm_client()
            await client.chat_completion([{"role": "user", "content": "x"}], node=node)

    # One provider, one model -- for EVERY stage. Nothing else executed.
    assert set(seen_models) == {"ollama_chat/qwen3:8b"}
    assert set(seen_retries) == {0}                       # no automatic fallback
    assert len(seen_models) == 4                          # one call per intentional stage, no duplicates

    assert current_execution_config().provider == "ollama"
    assert current_request_id() == "RID-1"
    meta = get_last_call_metadata()
    assert meta["provider_used"] == "ollama"
    assert meta["fallback_used"] is False
    assert meta["provider_attempts"] == ["ollama"]


@pytest.mark.asyncio
async def test_switching_provider_does_not_bleed_across_requests():
    """A github_copilot request must not carry over an ollama route, and vice
    versa. The frozen config is per async-context."""
    from app.services.llm.execution import begin_request, resolve_execution_config

    with patch("app.services.llm.execution.get_settings") as ms:
        ms.return_value = Settings(llm_provider="ollama", llm_model="qwen3:8b")
        c_ollama = resolve_execution_config("ollama", "qwen3:8b")
        ms.return_value = Settings(llm_provider="github_copilot", copilot_github_token="ghp_x")
        c_gh = resolve_execution_config("github_copilot", None)

    begin_request(c_ollama, "R-A")
    assert current_execution_config().litellm_model == "ollama_chat/qwen3:8b"
    begin_request(c_gh, "R-B")
    assert current_execution_config().litellm_model.startswith("github_copilot_session/")
    assert current_execution_config().provider == "github_copilot"


@pytest.mark.asyncio
async def test_qwen3_thinking_disabled_via_reasoning_effort():
    """A Qwen3/Ollama route freezes reasoning_effort='disable' and forwards it
    through the SINGLE litellm.acompletion call -> LiteLLM maps it to
    'think': false in the Ollama /api/chat body. Non-reasoning models get
    nothing (no 400 risk). Microsoft/GitHub unaffected."""
    from app.services.llm.execution import resolve_execution_config
    from app.services.llm.providers.litellm_provider import LiteLLMProvider

    def _route(provider, model, **kw):
        with patch("app.services.llm.execution.get_settings") as ms:
            ms.return_value = Settings(llm_provider=provider, llm_model=model, **kw)
            return resolve_execution_config(provider, model)

    seen = {}

    async def _spy(*a, **kw):
        seen.clear(); seen.update(kw)
        return _fake_completion()

    # Qwen3 -> reasoning_effort disabled
    with patch("litellm.acompletion", new=AsyncMock(side_effect=_spy)):
        await LiteLLMProvider(_route("ollama", "qwen3:14b")).generate(node="core_synthesis", prompt="x", response_format="json")
    assert seen["model"] == "ollama_chat/qwen3:14b"
    assert seen["reasoning_effort"] == "disable"
    assert seen["num_retries"] == 0

    # Non-reasoning Ollama model -> NO reasoning_effort key at all
    with patch("litellm.acompletion", new=AsyncMock(side_effect=_spy)):
        await LiteLLMProvider(_route("ollama", "llama3.1:8b")).generate(node="critic", prompt="x")
    assert "reasoning_effort" not in seen

    # ollama_thinking=True -> Qwen3 reasons normally (no disable)
    with patch("litellm.acompletion", new=AsyncMock(side_effect=_spy)):
        await LiteLLMProvider(_route("ollama", "qwen3:8b", ollama_thinking=True)).generate(node="critic", prompt="x")
    assert "reasoning_effort" not in seen

    # Microsoft route unaffected
    with patch("litellm.acompletion", new=AsyncMock(side_effect=_spy)):
        await LiteLLMProvider(_route("microsoft_copilot", "m365-chat", microsoft_copilot_access_token="t")).generate(node="critic", prompt="x")
    assert seen["model"] == "microsoft_copilot/m365-chat"
    assert "reasoning_effort" not in seen


@pytest.mark.asyncio
async def test_no_fan_out_on_failure(_ollama_request):
    """A provider failure NEVER triggers a call to a different provider/model."""
    import litellm
    from app.services.llm_client import get_llm_client
    from app.services.llm.exceptions import LLMProviderError

    call_count = {"n": 0}

    async def _boom(*args, **kwargs):
        call_count["n"] += 1
        raise litellm.ServiceUnavailableError(message="503", model=kwargs["model"], llm_provider="ollama")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=_boom)):
        with pytest.raises(LLMProviderError):
            await get_llm_client().chat_completion([{"role": "user", "content": "x"}], node="core_synthesis")

    assert call_count["n"] == 1  # exactly one provider attempted, then fail-closed
