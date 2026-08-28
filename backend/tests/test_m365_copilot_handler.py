"""Tests for the LiteLLM custom-provider handler backing Microsoft 365 Copilot."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

import litellm
from app.services.llm.providers import _m365_copilot_litellm_handler as h
from app.services.llm.providers.microsoft_copilot_provider import MicrosoftCopilotProvider


def test_handler_registered_in_custom_provider_map():
    h.register()
    providers = [e.get("provider") for e in litellm.custom_provider_map]
    assert "microsoft_copilot" in providers


def test_flatten_messages_folds_system_and_adds_json_directive():
    text, extra = h._flatten_messages(
        [
            {"role": "system", "content": "You are an auditor."},
            {"role": "user", "content": "Assess finding F-1."},
        ],
        want_json=True,
    )
    assert "You are an auditor." in text
    assert "ONLY a single valid JSON object" in text
    assert "Assess finding F-1." in text
    assert text.index("auditor") < text.index("Assess finding F-1.")


def test_strip_copilot_markup():
    assert h._strip_copilot_markup("<Person>Amy</Person> did it [^1^]") == "Amy did it"


@pytest.mark.asyncio
async def test_conversation_lifecycle_and_web_grounding_toggle():
    calls: list[tuple[str, str]] = []

    def _record(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path.endswith("/conversations"):
            return httpx.Response(201, json={"id": "c9", "status": "active"})
        if request.url.path.endswith("/chat"):
            body = json.loads(request.content)
            assert body["locationHint"]["timeZone"] == "UTC"
            # web grounding disabled by default -> explicit opt-out in the body
            assert body["contextualResources"]["webContext"]["isWebEnabled"] is False
            return httpx.Response(
                200,
                json={"messages": [{"text": "prompt echo"}, {"text": '{"verdict": "ok"}'}]},
                headers={"request-id": "gr-9"},
            )
        return httpx.Response(204)

    with respx.mock(assert_all_called=False) as rsx:
        rsx.route(host="graph.microsoft.com").mock(side_effect=_record)
        provider = MicrosoftCopilotProvider(ms_access_token="tok")
        resp = await provider.generate(node="critic", prompt="review", response_format="json")

    assert resp.content == '{"verdict": "ok"}'
    assert resp.raw_metadata["graph_request_id"] == "gr-9"
    methods = [m for m, _ in calls]
    assert methods.count("POST") == 2  # create + chat
    assert "DELETE" in methods  # best-effort cleanup
