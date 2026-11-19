"""Shared fixtures.

The suite is deliberately split into two kinds of test:

* **Unit/API tests** (default) run against a :class:`FakeLLM` injected through
  FastAPI's ``dependency_overrides``. No network, no model, fully
  deterministic. These prove the security and grounding logic.
* **Integration tests** (``-m integration``) talk to a real provider. They are
  skipped, never silently passed, when the runtime is not available -- so a
  green default run never implies live inference happened.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_llm
from app.llm.base import GenerationResult, HealthReport, LLMProvider
from app.main import create_app

TENANT_A_KEY = "test-key-tenant-a-0000000000"
TENANT_B_KEY = "test-key-tenant-b-1111111111"
CANARY = "CANARY-TENANT-B-DO-NOT-DISCLOSE"

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = REPO_ROOT / "data" / "synthetic_records.json"


class FakeLLM(LLMProvider):
    """Scripted provider. Records every prompt so tests can assert on it."""

    name = "fake"
    execution = "local"

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        error: Exception | None = None,
        reachable: bool = True,
    ) -> None:
        super().__init__(
            base_url="http://fake.invalid",
            model="fake-model",
            timeout=1.0,
            max_output_tokens=256,
        )
        self.responses = list(responses or [])
        self.error = error
        self.reachable = reachable
        self.calls: list[tuple[str, str]] = []

    @property
    def last_user_prompt(self) -> str:
        assert self.calls, "the model was never called"
        return self.calls[-1][1]

    @property
    def last_system_prompt(self) -> str:
        assert self.calls, "the model was never called"
        return self.calls[-1][0]

    def generate(self, *, system: str, user: str) -> GenerationResult:
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        text = self.responses.pop(0) if self.responses else "See the retrieved policy [DOC1]."
        return GenerationResult(
            text=text,
            provider=self.name,
            model=self.model,
            execution=self.execution,
            latency_ms=1,
        )

    def health(self) -> HealthReport:
        return HealthReport(
            provider=self.name,
            model=self.model,
            endpoint="http://fake.invalid",
            execution=self.execution,
            reachable=self.reachable,
            model_available=self.reachable,
            detail=None if self.reachable else "fake provider marked unreachable",
        )


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's real environment from steering the unit tests.

    Without this, an exported ``LLM_PROVIDER`` or ``TENANT_A_API_KEY`` on the
    machine running the suite would silently change what is under test.
    """
    for name in (
        "LLM_PROVIDER",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_API_KEY",
        "TENANT_A_API_KEY",
        "TENANT_B_API_KEY",
        "DB_PATH",
        "RECORDS_SEED_PATH",
        "QUARANTINE_INJECTED_RECORDS",
        "RETRIEVAL_MIN_SCORE",
        "GROUNDING_MIN_SUPPORT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings: temp database, known keys, real synthetic corpus."""
    return Settings(
        tenant_a_api_key=TENANT_A_KEY,
        tenant_b_api_key=TENANT_B_KEY,
        db_path=str(tmp_path / "test.sqlite3"),
        records_seed_path=str(SEED_PATH),
        llm_provider="ollama",
        log_level="WARNING",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(settings: Settings, fake_llm: FakeLLM) -> Iterator[TestClient]:
    """A TestClient whose model calls are served by ``fake_llm``."""
    app = create_app(settings)
    app.dependency_overrides[get_llm] = lambda: fake_llm
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def make_client(settings: Settings):
    """Factory for clients with a custom provider and/or settings overrides."""

    def _make(
        provider: LLMProvider,
        *,
        raise_server_exceptions: bool = True,
        **setting_overrides: object,
    ) -> TestClient:
        effective = (
            settings.model_copy(update=setting_overrides) if setting_overrides else settings
        )
        app = create_app(effective)
        app.dependency_overrides[get_llm] = lambda: provider
        # `raise_server_exceptions=False` lets a test observe the sanitized 500
        # body the real server would send, instead of the raw exception.
        return TestClient(app, raise_server_exceptions=raise_server_exceptions)

    return _make


@pytest.fixture
def auth_a() -> dict[str, str]:
    return {"X-API-Key": TENANT_A_KEY}


@pytest.fixture
def auth_b() -> dict[str, str]:
    return {"X-API-Key": TENANT_B_KEY}


@pytest.fixture(scope="session")
def synthetic_records() -> list[dict]:
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))
