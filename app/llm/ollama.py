"""Local Ollama provider -- the runtime used for the CSPRO assessment.

Inference happens in the Ollama server process on this machine. No request
leaves the host, and this class never falls back to a remote endpoint: if the
local runtime is down, :meth:`generate` raises and the API answers 503.
"""

from __future__ import annotations

import time

import httpx

from app.llm.base import (
    GenerationResult,
    HealthReport,
    LLMProvider,
    LLMResponseError,
)


class OllamaProvider(LLMProvider):
    name = "ollama"
    execution = "local"

    def _payload(self, system: str, user: str, *, think: bool | None) -> dict:
        payload: dict = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                # Greedy decoding keeps the demo reproducible run-to-run.
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "num_predict": self.max_output_tokens,
            },
        }
        if think is not None:
            payload["think"] = think
        return payload

    def generate(self, *, system: str, user: str) -> GenerationResult:
        """Run one local completion.

        Reasoning models (qwen3, deepseek-r1, ...) spend part of the
        ``num_predict`` budget on a hidden chain of thought that Ollama returns
        in a separate ``thinking`` field. On a long RAG context that reasoning
        can consume the entire budget and leave ``content`` empty -- which would
        surface to the caller as a 503 even though the runtime is perfectly
        healthy.

        So: if the model produced thinking but no answer, retry once with
        thinking disabled. The retry is conditional on evidence that this model
        actually thinks, which keeps the request unchanged for every model that
        does not support the ``think`` flag.
        """
        started = time.perf_counter()
        data = self._post("/api/chat", self._payload(system, user, think=None))
        message = data.get("message") or {}
        text = message.get("content")

        if (not isinstance(text, str) or not text.strip()) and (message.get("thinking") or "").strip():
            data = self._post("/api/chat", self._payload(system, user, think=False))
            message = data.get("message") or {}
            text = message.get("content")

        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError(
                f"ollama returned an empty completion for model '{self.model}'. "
                "If this is a reasoning model, raise LLM_MAX_OUTPUT_TOKENS so the "
                "answer still fits after the hidden reasoning."
            )
        return GenerationResult(
            text=text.strip(),
            provider=self.name,
            model=self.model,
            execution=self.execution,
            latency_ms=self._elapsed_ms(started),
        )

    def health(self) -> HealthReport:
        try:
            response = self.http.get(f"{self.base_url}/api/tags", timeout=3.0)
            response.raise_for_status()
            tags = response.json().get("models", [])
        except (httpx.HTTPError, ValueError) as exc:
            return HealthReport(
                provider=self.name,
                model=self.model,
                endpoint=self.safe_endpoint,
                execution=self.execution,
                reachable=False,
                model_available=False,
                detail=(
                    "Local Ollama runtime unreachable. Start it with 'ollama serve' "
                    f"and pull the model with 'ollama pull {self.model}'. "
                    f"({type(exc).__name__})"
                ),
            )

        names = {str(entry.get("name", "")) for entry in tags}
        available = self.model in names or any(
            n.split(":")[0] == self.model.split(":")[0] and n == self.model for n in names
        )
        return HealthReport(
            provider=self.name,
            model=self.model,
            endpoint=self.safe_endpoint,
            execution=self.execution,
            reachable=True,
            model_available=available,
            detail=(
                None
                if available
                else f"Runtime is up but model '{self.model}' is not pulled."
            ),
        )
