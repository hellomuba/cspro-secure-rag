"""OpenAI-compatible HTTP provider -- development convenience only.

Use this to iterate quickly when no local runtime is installed. Inference runs
on a remote service, so ``execution`` is ``"remote"`` and ``/health`` and
``/ask`` say so in their responses. The README and the demo must not present a
run made with this provider as local inference.

The API key is read from ``LLM_API_KEY`` and is never logged or echoed.
"""

from __future__ import annotations

import time

import httpx

from app.llm.base import (
    GenerationResult,
    HealthReport,
    LLMProvider,
    LLMResponseError,
    LLMUnavailableError,
)


class OpenAICompatProvider(LLMProvider):
    name = "openai"
    execution = "remote"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        if not self._api_key:
            raise LLMUnavailableError(
                "LLM_PROVIDER=openai requires LLM_API_KEY to be set."
            )

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def generate(self, *, system: str, user: str) -> GenerationResult:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = self._post("/chat/completions", payload, headers=self._headers)
        choices = data.get("choices") or []
        text = (choices[0].get("message") or {}).get("content") if choices else None
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError("openai-compatible endpoint returned no completion")
        return GenerationResult(
            text=text.strip(),
            provider=self.name,
            model=self.model,
            execution=self.execution,
            latency_ms=self._elapsed_ms(started),
        )

    def health(self) -> HealthReport:
        try:
            response = self.http.get(
                f"{self.base_url}/models", headers=self._headers, timeout=5.0
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return HealthReport(
                provider=self.name,
                model=self.model,
                endpoint=self.safe_endpoint,
                execution=self.execution,
                reachable=False,
                model_available=False,
                detail=f"Remote endpoint unreachable or rejected ({type(exc).__name__}).",
            )

        ids = {str(m.get("id", "")) for m in payload.get("data", [])}
        return HealthReport(
            provider=self.name,
            model=self.model,
            endpoint=self.safe_endpoint,
            execution=self.execution,
            reachable=True,
            # Some compatible gateways do not list every served model.
            model_available=True if self.model in ids else None,
            detail=(
                None
                if self.model in ids
                else "Endpoint reachable; model listing did not confirm the model id."
            ),
        )
