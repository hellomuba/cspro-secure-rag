"""Audit logging hygiene and sanitized error handling."""

from __future__ import annotations

import json
import logging

import pytest

from app.audit import audit, new_audit_id
from app.dependencies import get_store
from app.store import TenantScopeError
from tests.conftest import CANARY, TENANT_A_KEY, FakeLLM

ASK = {"question": "What is the current import retry policy?"}


# --------------------------------------------------------------------------
# Audit logging
# --------------------------------------------------------------------------
def test_audit_lines_are_single_line_json(settings, caplog):
    with caplog.at_level(logging.INFO, logger="aegisrag.audit"):
        audit(settings, "ask", audit_id="aud_1", tenant_id="tenant-a", citation_ids=["a-1"])
    payload = json.loads(caplog.records[-1].message)
    assert payload["event"] == "ask"
    assert payload["citation_ids"] == ["a-1"]


def test_audit_never_writes_an_api_key_even_if_passed(settings, caplog):
    with caplog.at_level(logging.INFO, logger="aegisrag.audit"):
        audit(settings, "ask", leaked=TENANT_A_KEY)
    assert TENANT_A_KEY not in caplog.text
    assert "***REDACTED***" in caplog.text


def test_audit_omits_forbidden_fields_by_name(settings, caplog):
    with caplog.at_level(logging.INFO, logger="aegisrag.audit"):
        audit(settings, "ask", question="secret question", answer="secret answer", body="x")
    payload = json.loads(caplog.records[-1].message)
    assert payload["question"] == "<omitted-by-policy>"
    assert payload["answer"] == "<omitted-by-policy>"
    assert "secret question" not in caplog.text


def test_ask_audit_records_citations_without_record_text(make_client, auth_a, caplog):
    with caplog.at_level(logging.INFO, logger="aegisrag.audit"):
        with make_client(FakeLLM(["Idempotency keys are required [DOC1]."])) as client:
            client.post("/ask", json=ASK, headers=auth_a)

    ask_lines = [
        json.loads(r.message)
        for r in caplog.records
        if r.name == "aegisrag.audit" and '"event": "ask"' in r.message
    ]
    assert ask_lines, "an /ask call must produce an audit record"
    entry = ask_lines[-1]
    assert entry["tenant_id"] == "tenant-a"
    assert entry["citation_ids"] == ["a-policy-2"]
    assert entry["status"] == "answered"
    assert isinstance(entry["latency_ms"], int)
    assert "idempotency" not in caplog.text.lower(), "record text is never logged"


def test_no_log_line_anywhere_contains_the_canary(make_client, auth_b, caplog):
    with caplog.at_level(logging.DEBUG):
        with make_client(FakeLLM(["Seven days [DOC1]."])) as client:
            client.post(
                "/ask",
                json={"question": "How long are demo records retained?"},
                headers=auth_b,
            )
    assert CANARY not in caplog.text


def test_short_configured_values_do_not_shred_log_lines():
    """Regression: a 1-character test key once matched inside every word."""
    from app.sanitize import redact_secrets

    assert redact_secrets("a database of records", ("a", "b")) == "a database of records"
    assert "***REDACTED***" in redact_secrets("key=sk-abcdefgh12345", ("sk-abcdefgh12345",))


def test_audit_ids_are_unique():
    ids = {new_audit_id() for _ in range(500)}
    assert len(ids) == 500
    assert all(i.startswith("aud_") for i in ids)


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------
def test_unexpected_error_returns_a_sanitized_500(settings, make_client, auth_a):
    """No stack trace, no exception text, no record data -- just an audit id."""

    class Exploding:
        def list_for_tenant(self, *_args, **_kwargs):
            raise RuntimeError(f"boom: internal detail containing {CANARY}")

        def count_by_tenant(self):
            return {"total": 0}

    client = make_client(FakeLLM(), raise_server_exceptions=False)
    with client:
        client.app.dependency_overrides[get_store] = lambda: Exploding()
        response = client.post("/ask", json=ASK, headers=auth_a, follow_redirects=False)

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error."
    assert body["audit_id"].startswith("aud_")
    assert CANARY not in response.text
    assert "boom" not in response.text
    assert "Traceback" not in response.text


def test_tenant_scope_violation_is_blocked_with_a_safe_message(make_client, auth_a):
    class Escaping:
        def list_for_tenant(self, *_args, **_kwargs):
            raise TenantScopeError("record 'b-policy' escaped tenant scope 'tenant-a'")

        def count_by_tenant(self):
            return {"total": 0}

    client = make_client(FakeLLM(), raise_server_exceptions=False)
    with client:
        client.app.dependency_overrides[get_store] = lambda: Escaping()
        response = client.post("/ask", json=ASK, headers=auth_a)

    assert response.status_code == 500
    assert response.json()["detail"] == "Request blocked by an internal safety check."
    assert "b-policy" not in response.text


@pytest.mark.parametrize("method,path", [("get", "/does-not-exist"), ("post", "/ingest/extra")])
def test_unknown_routes_are_404(client, method, path):
    assert getattr(client, method)(path).status_code == 404


def test_wrong_method_is_405(client):
    assert client.get("/ask").status_code == 405


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {"/health", "/ingest", "/ask"}
