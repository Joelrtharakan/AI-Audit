"""Provider-agnostic LLM client contract.

Every step of the pipeline (observation quality, extraction, classification,
generation) type-hints against `LLMClient` and catches `LLMError`, never a
concrete provider class -- this is what lets `get_llm_client()` in
finding_analysis_service.py be the only place that knows which provider is
configured. See openrouter_client.py and ollama_client.py for the two
implementations; both raise (subclasses of) the exceptions defined here.
"""

from typing import Protocol


class LLMError(RuntimeError):
    pass


class LLMRateLimitedError(LLMError):
    pass


class LLMClient(Protocol):
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        response_format_json: bool = False,
        max_tokens: int | None = None,
    ) -> str: ...


def get_llm_client() -> LLMClient:
    """The one place that branches on LLM_PROVIDER. Every pipeline step (observation
    quality, extraction, classification, generation) calls this instead of
    constructing a provider class directly, so the provider is a config change, not a
    code change -- see README section 6."""
    from app.config import get_settings

    settings = get_settings()
    if settings.llm_provider == "ollama":
        from app.services.ollama_client import OllamaClient

        return OllamaClient()

    from app.services.openrouter_client import OpenRouterClient

    return OpenRouterClient()
