"""Authentication and server-side tenant resolution.

This module is the *only* place a tenant identity is created. Every downstream
component receives a :class:`Principal` that was produced here; none of them
accept a tenant string from a request body, query string or document.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, hash_api_key
from app.dependencies import get_app_settings


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller. ``tenant_id`` is authoritative."""

    tenant_id: str
    key_fingerprint: str  # first 8 hex chars of sha256(key); safe to log


_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or missing API key.",
    headers={"WWW-Authenticate": "ApiKey"},
)


def resolve_principal(presented_key: str | None, settings: Settings) -> Principal:
    """Map a presented API key to a tenant, in constant time.

    Raises :class:`HTTPException` 401 for a missing, malformed or unknown key.
    The error message never echoes the presented value.
    """
    if not presented_key:
        raise _UNAUTHORIZED

    presented_hash = hash_api_key(presented_key)
    matched: str | None = None
    # Compare against every configured key so that timing does not reveal
    # which tenant (if any) a prefix matched.
    for known_hash, tenant_id in settings.tenant_key_hashes.items():
        if hmac.compare_digest(presented_hash, known_hash):
            matched = tenant_id

    if matched is None:
        raise _UNAUTHORIZED

    return Principal(tenant_id=matched, key_fingerprint=presented_hash[:8])


def require_tenant(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_app_settings),
) -> Principal:
    """FastAPI dependency: authenticate the caller and resolve its tenant."""
    return resolve_principal(x_api_key, settings)
