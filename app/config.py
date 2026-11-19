"""Environment-driven configuration.

Every security-relevant knob is read from the environment exactly once, at
application start-up, and exposed as an immutable settings object. Nothing in
the request path may mutate it.

No real secret is ever committed: ``.env.example`` carries placeholder values
and ``.env`` is git-ignored.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["ollama", "openai"]

#: Synthetic leak markers. These are *tripwires*, not the authorization
#: boundary (see ``app/grounding.py``). The boundary is the tenant filter
#: applied in ``app/store.py`` before ranking or inference.
CANARY_MARKERS: tuple[str, ...] = ("CANARY-TENANT-B-DO-NOT-DISCLOSE",)


class Settings(BaseSettings):
    """Application settings, loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Tenant credentials -------------------------------------------------
    # Development defaults only. They are deliberately obvious placeholders so
    # that a misconfigured deployment is visible rather than silently insecure.
    tenant_a_api_key: str = Field(default="dev-a-change-me")
    tenant_b_api_key: str = Field(default="dev-b-change-me")

    # --- LLM provider abstraction ------------------------------------------
    # ``ollama`` is the provider used for the CSPRO assessment. ``openai`` is an
    # OpenAI-compatible HTTP endpoint used only for rapid development. There is
    # no automatic failover between them: see app/llm/factory.py.
    llm_provider: LLMProviderName = Field(default="ollama")
    llm_base_url: str = Field(default="")
    llm_model: str = Field(default="")
    llm_api_key: str = Field(default="")
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    llm_max_output_tokens: int = Field(default=400, ge=16, le=4096)

    # --- Retrieval ----------------------------------------------------------
    db_path: str = Field(default="data/aegisrag.sqlite3")
    records_seed_path: str = Field(default="data/synthetic_records.json")
    retrieval_min_score: float = Field(default=0.35, ge=0.0)
    retrieval_max_top_k: int = Field(default=10, ge=1, le=50)
    excerpt_max_chars: int = Field(default=320, ge=40, le=2000)

    # Minimum fraction of an answer's content words that must also occur in the
    # authorised context. Below this the answer is downgraded to an abstention.
    # Heuristic, not a proof of grounding -- see README "Known limitations".
    grounding_min_support: float = Field(default=0.35, ge=0.0, le=1.0)

    # --- Behaviour ----------------------------------------------------------
    quarantine_injected_records: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    @field_validator("tenant_a_api_key", "tenant_b_api_key")
    @classmethod
    def _non_empty_key(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tenant API keys must not be empty")
        return value

    # --- Derived ------------------------------------------------------------
    @property
    def resolved_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url.rstrip("/")
        return {
            "ollama": "http://localhost:11434",
            "openai": "https://api.openai.com/v1",
        }[self.llm_provider]

    @property
    def resolved_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return {
            "ollama": "qwen3:0.6b",
            "openai": "gpt-4o-mini",
        }[self.llm_provider]

    @property
    def tenant_key_hashes(self) -> dict[str, str]:
        """``sha256(api_key) -> tenant_id``.

        Raw keys are never stored on the request path and never logged.
        """
        return {
            hash_api_key(self.tenant_a_api_key): "tenant-a",
            hash_api_key(self.tenant_b_api_key): "tenant-b",
        }

    @property
    def secret_values(self) -> tuple[str, ...]:
        """Values that must never appear in a log line or an HTTP response."""
        return tuple(
            v
            for v in (self.tenant_a_api_key, self.tenant_b_api_key, self.llm_api_key)
            if v
        )


def hash_api_key(raw: str) -> str:
    """Hash a presented API key for constant-time comparison."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (overridable in tests via DI)."""
    return Settings()
