"""Authentication and server-side tenant resolution."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.auth import resolve_principal
from tests.conftest import TENANT_A_KEY, TENANT_B_KEY

ASK_BODY = {"question": "What is the current import retry policy?"}
INGEST_BODY = {"records": [{"id": "x1", "title": "T", "body": "B", "topic": "imports"}]}


@pytest.mark.parametrize("path,body", [("/ask", ASK_BODY), ("/ingest", INGEST_BODY)])
def test_missing_api_key_is_401(client, path, body):
    response = client.post(path, json=body)
    assert response.status_code == 401
    assert "detail" in response.json()


@pytest.mark.parametrize("path,body", [("/ask", ASK_BODY), ("/ingest", INGEST_BODY)])
def test_wrong_api_key_is_401(client, path, body):
    response = client.post(path, json=body, headers={"X-API-Key": "not-a-real-key"})
    assert response.status_code == 401


@pytest.mark.parametrize("bad", ["", "   ", "null", TENANT_A_KEY + "x", TENANT_A_KEY[:-1]])
def test_malformed_keys_are_rejected(client, bad):
    response = client.post("/ask", json=ASK_BODY, headers={"X-API-Key": bad})
    assert response.status_code == 401


def test_401_response_never_echoes_the_presented_key(client):
    secret = "super-secret-presented-value"
    response = client.post("/ask", json=ASK_BODY, headers={"X-API-Key": secret})
    assert response.status_code == 401
    assert secret not in response.text


def test_valid_keys_resolve_to_distinct_tenants(settings):
    assert resolve_principal(TENANT_A_KEY, settings).tenant_id == "tenant-a"
    assert resolve_principal(TENANT_B_KEY, settings).tenant_id == "tenant-b"


def test_key_fingerprint_is_not_the_key(settings):
    principal = resolve_principal(TENANT_A_KEY, settings)
    assert principal.key_fingerprint not in TENANT_A_KEY
    assert len(principal.key_fingerprint) == 8


def test_unknown_key_raises_401_without_leaking_which_tenant(settings):
    with pytest.raises(HTTPException) as excinfo:
        resolve_principal("totally-unknown", settings)
    assert excinfo.value.status_code == 401
    assert "tenant" not in str(excinfo.value.detail).lower()


def test_tenant_id_in_body_is_rejected_not_silently_ignored(client, auth_a):
    """The only way to set a tenant is to hold its key."""
    response = client.post(
        "/ingest",
        json={
            "records": [
                {
                    "id": "spoof-1",
                    "title": "T",
                    "body": "B",
                    "topic": "imports",
                    "tenant_id": "tenant-b",
                }
            ]
        },
        headers=auth_a,
    )
    assert response.status_code == 422


def test_ask_body_cannot_carry_a_tenant_field(client, auth_a):
    response = client.post(
        "/ask", json={**ASK_BODY, "tenant_id": "tenant-b"}, headers=auth_a
    )
    assert response.status_code == 422
