"""Provider selection.

**There is deliberately no fallback chain.** ``LLM_PROVIDER`` selects exactly
one provider for the lifetime of the process. If that provider is unreachable
the request fails with HTTP 503 and an actionable message -- the application
never silently reaches for a cloud endpoint to rescue a failed local call.
That property is what lets the assessment demo claim, truthfully, that every
answer it produced came from the local Ollama runtime.
"""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.ollama import OllamaProvider
from app.llm.openai_compat import OpenAICompatProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    OllamaProvider.name: OllamaProvider,
    OpenAICompatProvider.name: OpenAICompatProvider,
}


def build_provider(settings: Settings) -> LLMProvider:
    """Construct the single configured provider."""
    try:
        provider_cls = _PROVIDERS[settings.llm_provider]
    except KeyError:  # pragma: no cover - pydantic Literal already rejects this
        raise ValueError(
            f"Unknown LLM_PROVIDER {settings.llm_provider!r}. "
            f"Supported: {', '.join(sorted(_PROVIDERS))}."
        ) from None

    return provider_cls(
        base_url=settings.resolved_base_url,
        model=settings.resolved_model,
        timeout=settings.llm_timeout_seconds,
        max_output_tokens=settings.llm_max_output_tokens,
        api_key=settings.llm_api_key,
    )
