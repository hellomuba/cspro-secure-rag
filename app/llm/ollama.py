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

    def generate(self, *, system: str, user: str) -> GenerationResult:
        started = time.perf_counter()
        payload = {
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
        data = self._post("/api/chat", payload)
        text = (data.get("message") or {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError("ollama returned an empty completion")
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
