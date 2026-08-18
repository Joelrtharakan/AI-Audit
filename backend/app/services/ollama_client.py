"""Legacy compatibility wrapper for OllamaClient.

Delegates to `OllamaProvider` in `app.services.llm.providers.ollama_provider`.
"""

from __future__ import annotations

import contextvars
from typing import Any

from app.services.llm.exceptions import LLMError
from app.services.llm.providers.ollama_provider import OllamaProvider

_current_node: contextvars.ContextVar[str] = contextvars.ContextVar(
    "ollama_current_node", default="unknown"
)
_last_call_metadata: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "ollama_last_call_metadata", default={}
)


def set_current_node(node: str) -> None:
    _current_node.set(node)


def get_last_call_metadata() -> dict:
    return _last_call_metadata.get()


class OllamaError(LLMError):
    pass


class OllamaClient(OllamaProvider):
    """Compatibility subclass of OllamaProvider."""

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format_json: bool = False,
        max_tokens: int | None = None,
        num_ctx: int | None = None,
        node: str = "unknown",
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> str:
        effective_node = node if node != "unknown" else _current_node.get()
        system_parts: list[str] = []
        user_parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        prompt = "\n\n".join(user_parts) if user_parts else ""

        response = await self.generate(
            node=effective_node,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_format="json" if response_format_json else None,
            num_ctx=num_ctx,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        _last_call_metadata.set(response.raw_metadata or {})
        return response.content


__all__ = [
    "OllamaClient",
    "OllamaProvider",
    "OllamaError",
    "set_current_node",
    "get_last_call_metadata",
]
