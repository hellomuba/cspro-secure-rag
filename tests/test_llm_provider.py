"""The provider abstraction: selection, isolation from security logic, failures.

All of these use mocked transports. Tests that talk to a real runtime live in
``tests/test_integration_*.py`` and are marked ``integration``.
"""

from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.llm.base import (
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.factory import build_provider
from app.llm.ollama import OllamaProvider
from app.llm.openai_compat import OpenAICompatProvider
from tests.conftest import CANARY, FakeLLM

ASK = {"question": "What is the current import retry policy?"}


def make_settings(**overrides) -> Settings:
    base = {
        "tenant_a_api_key": "a-key",
        "tenant_b_api_key": "b-key",
        "_env_file": None,
    }
    return Settings(**{**base, **overrides})  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def test_default_provider_is_local_ollama_with_the_assessment_model():
    provider = build_provider(make_settings())
    assert isinstance(provider, OllamaProvider)
    assert provider.name == "ollama"
    assert provider.execution == "local"
    assert provider.model == "qwen3:0.6b"
    assert provider.base_url == "http://localhost:11434"
    provider.close()


def test_openai_provider_is_selected_by_env():
    provider = build_provider(
        make_settings(
            llm_provider="openai",
            llm_base_url="https://example.invalid/v1",
            llm_model="gpt-4o-mini",
            llm_api_key="sk-test-not-a-real-key",
        )
    )
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.execution == "remote", "a cloud call is never 'local inference'"
    provider.close()


def test_env_vars_override_every_provider_setting(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("LLM_MODEL", "local-gguf-model")
    monkeypatch.setenv("LLM_API_KEY", "sk-env-value")
    settings = Settings(tenant_a_api_key="a", tenant_b_api_key="b", _env_file=None)  # type: ignore[call-arg]

    assert settings.llm_provider == "openai"
    assert settings.resolved_base_url == "http://127.0.0.1:1234/v1"
    assert settings.resolved_model == "local-gguf-model"
    provider = build_provider(settings)
    assert provider.model == "local-gguf-model"
    provider.close()


def test_openai_provider_requires_an_api_key():
    with pytest.raises(LLMUnavailableError, match="LLM_API_KEY"):
        build_provider(make_settings(llm_provider="openai", llm_api_key=""))


def test_unknown_provider_is_rejected_at_configuration_time():
    with pytest.raises(ValueError):
        Settings(
            tenant_a_api_key="a",
            tenant_b_api_key="b",
            llm_provider="definitely-not-a-provider",  # type: ignore[arg-type]
            _env_file=None,  # type: ignore[call-arg]
        )


def test_there_is_no_automatic_cloud_fallback():
    """Selecting Ollama must never construct a cloud client, even on failure."""
    settings = make_settings(llm_provider="ollama", llm_api_key="sk-present-but-unused")
    provider = build_provider(settings)
    assert isinstance(provider, OllamaProvider)
    assert not isinstance(provider, OpenAICompatProvider)

    # A failing local call raises; it does not return a remote result.
    provider._client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused"))
        )
    )
    with pytest.raises(LLMUnavailableError):
        provider.generate(system="s", user="u")
    provider.close()


def test_provider_endpoint_string_carries_no_credentials():
    provider = build_provider(
        make_settings(
            llm_provider="openai",
            llm_base_url="https://example.invalid/v1",
            llm_api_key="sk-super-secret-value",
        )
    )
    assert "sk-super-secret-value" not in provider.safe_endpoint
    provider.close()


# --------------------------------------------------------------------------
# Ollama wire behaviour (mocked transport)
# --------------------------------------------------------------------------
def ollama_with(handler) -> OllamaProvider:
    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="qwen2.5:1.5b-instruct",
        timeout=5.0,
        max_output_tokens=128,
    )
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    return provider


def test_ollama_generate_parses_a_chat_response():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"content": "  grounded answer  "}})

    provider = ollama_with(handler)
    result = provider.generate(system="SYS", user="USR")
    assert result.text == "grounded answer"
    assert result.execution == "local"
    assert result.provider == "ollama"
    assert captured["options"]["temperature"] == 0.0, "deterministic decoding"
    assert captured["stream"] is False
    assert [m["role"] for m in captured["messages"]] == ["system", "user"]
    provider.close()


def test_ollama_connection_refused_raises_unavailable():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    provider = ollama_with(handler)
    with pytest.raises(LLMUnavailableError, match="unreachable"):
        provider.generate(system="s", user="u")
    provider.close()


def test_ollama_timeout_raises_timeout_error():
    def handler(request):
        raise httpx.ReadTimeout("too slow")

    provider = ollama_with(handler)
    with pytest.raises(LLMTimeoutError, match="timed out"):
        provider.generate(system="s", user="u")
    provider.close()


def test_ollama_http_error_raises_response_error():
    provider = ollama_with(lambda r: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(LLMResponseError, match="HTTP 500"):
        provider.generate(system="s", user="u")
    provider.close()


def test_ollama_empty_completion_is_an_error_not_an_answer():
    provider = ollama_with(lambda r: httpx.Response(200, json={"message": {"content": "  "}}))
    with pytest.raises(LLMResponseError, match="empty"):
        provider.generate(system="s", user="u")
    provider.close()


def test_ollama_health_detects_a_missing_model():
    provider = ollama_with(
        lambda r: httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})
    )
    report = provider.health()
    assert report.reachable is True
    assert report.model_available is False
    assert "not pulled" in report.detail
    provider.close()


def test_ollama_health_does_not_raise_when_down():
    def handler(request):
        raise httpx.ConnectError("refused")

    provider = ollama_with(handler)
    report = provider.health()
    assert report.reachable is False
    assert "ollama serve" in report.detail
    provider.close()


# --------------------------------------------------------------------------
# OpenAI-compatible wire behaviour (mocked transport)
# --------------------------------------------------------------------------
def test_openai_generate_parses_a_chat_completion():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["path"] = request.url.path
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "remote answer"}}]}
        )

    provider = OpenAICompatProvider(
        base_url="https://example.invalid/v1",
        model="gpt-4o-mini",
        timeout=5.0,
        max_output_tokens=128,
        api_key="sk-test",
    )
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    result = provider.generate(system="s", user="u")

    assert result.text == "remote answer"
    assert result.execution == "remote"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["path"].endswith("/chat/completions")
    provider.close()


# --------------------------------------------------------------------------
# Failure surfaces through the API as 503
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "error,fragment",
    [
        (LLMUnavailableError("ollama runtime at http://localhost:11434 is unreachable"), "unreachable"),
        (LLMTimeoutError("ollama timed out after 60s"), "timed out"),
        (LLMResponseError("ollama returned HTTP 500 for /api/chat"), "HTTP 500"),
    ],
)
def test_model_failure_is_503_not_a_fabricated_answer(make_client, auth_a, error, fragment):
    with make_client(FakeLLM(error=error)) as client:
        response = client.post("/ask", json=ASK, headers=auth_a)

    assert response.status_code == 503
    body = response.json()
    assert fragment in body["detail"]
    assert body["audit_id"].startswith("aud_")
    assert "answer" not in body, "a failed generation never becomes an answer"


def test_503_message_does_not_offer_a_cloud_fallback(make_client, auth_a):
    with make_client(FakeLLM(error=LLMUnavailableError("ollama runtime is unreachable"))) as client:
        detail = client.post("/ask", json=ASK, headers=auth_a).json()["detail"]
    assert "openai" not in detail.lower()


def test_model_failure_never_leaks_records(make_client, auth_a):
    with make_client(FakeLLM(error=LLMUnavailableError("down"))) as client:
        response = client.post("/ask", json=ASK, headers=auth_a)
    assert CANARY not in response.text
    assert "idempotency" not in response.text.lower()


# --------------------------------------------------------------------------
# Provider independence
# --------------------------------------------------------------------------
def test_security_behaviour_is_identical_across_providers(make_client, auth_a):
    """Swapping the provider must not change any authorization outcome."""
    results = []
    for execution in ("local", "remote"):
        llm = FakeLLM([f"The marker is {CANARY}."])
        llm.execution = execution  # type: ignore[assignment]
        with make_client(llm) as client:
            body = client.post("/ask", json=ASK, headers=auth_a).json()
        results.append((body["status"], body["sources"], body["security_flags"]))

    assert results[0] == results[1]
    assert results[0][0] == "blocked_policy_violation"
