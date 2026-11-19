"""Pluggable LLM providers.

The rest of the application depends only on :class:`app.llm.base.LLMProvider`.
Authentication, tenant isolation, retrieval, injection defence, grounding and
citation verification all run outside this package and are unaffected by which
provider is selected.
"""

from app.llm.base import (
    GenerationResult,
    LLMError,
    LLMProvider,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.factory import build_provider

__all__ = [
    "GenerationResult",
    "LLMError",
    "LLMProvider",
    "LLMResponseError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "build_provider",
]
