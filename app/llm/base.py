"""Provider-agnostic LLM interface.

Contract for every implementation:

* ``generate`` returns plain text or raises an :class:`LLMError` subclass. It
  never returns a fabricated answer and never silently switches provider.
* ``health`` is cheap, side-effect free and must not raise.
* ``execution`` states truthfully where inference ran: ``"local"`` only when
  the model executes on this machine.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Literal

import httpx

Execution = Literal["local", "remote"]


class LLMError(RuntimeError):
    """Base class for every provider failure."""


class LLMUnavailableError(LLMError):
    """The configured runtime could not be reached. Surfaces as HTTP 503."""


class LLMTimeoutError(LLMUnavailableError):
    """The runtime accepted the request but did not answer in time (503)."""


class LLMResponseError(LLMError):
    """The runtime answered, but not with a usable completion (503)."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    provider: str
    model: str
    execution: Execution
    latency_ms: int


@dataclass(frozen=True, slots=True)
class HealthReport:
    provider: str
    model: str
    endpoint: str
    execution: Execution
    reachable: bool
    model_available: bool | None = None
    detail: str | None = None


class LLMProvider(abc.ABC):
    """Base class for all providers."""

    name: str
    execution: Execution

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float,
        max_output_tokens: int,
        api_key: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self._api_key = api_key
        # Built lazily: constructing an httpx.Client materialises an SSL
        # context, which costs over a second on some Windows machines. A
        # provider that is never called should not pay for it.
        self._client: httpx.Client | None = None

    # ------------------------------------------------------------------ API
    @abc.abstractmethod
    def generate(self, *, system: str, user: str) -> GenerationResult:
        """Run one completion. Raise an :class:`LLMError` on any failure."""

    @abc.abstractmethod
    def health(self) -> HealthReport:
        """Probe the runtime. Must not raise."""

    @property
    def http(self) -> httpx.Client:
        """The shared HTTP client, created on first use."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -------------------------------------------------------------- helpers
    @property
    def safe_endpoint(self) -> str:
        """Endpoint string suitable for logs and API responses (no credentials)."""
        try:
            parsed = httpx.URL(self.base_url)
            return f"{parsed.scheme}://{parsed.host}:{parsed.port or ''}".rstrip(":")
        except (httpx.InvalidURL, ValueError):  # pragma: no cover - defensive
            return "<unparsable-endpoint>"

    def _post(self, path: str, payload: dict, headers: dict[str, str] | None = None) -> dict:
        url = f"{self.base_url}{path}"
        try:
            response = self.http.post(url, json=payload, headers=headers or {})
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"{self.name} timed out after {self.timeout:.0f}s"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMUnavailableError(
                f"{self.name} runtime at {self.safe_endpoint} is unreachable"
            ) from exc

        if response.status_code >= 400:
            raise LLMResponseError(
                f"{self.name} returned HTTP {response.status_code} for {path}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise LLMResponseError(f"{self.name} returned a non-JSON body") from exc

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)
