"""FastAPI dependency providers.

Components are built once during the lifespan event and stashed on
``app.state``. Reading them through dependencies (rather than importing module
globals) is what lets the test-suite swap in a fake LLM provider or a temporary
database without monkey-patching application code.

This module must not import ``app.auth`` or ``app.service``: those import it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.config import Settings
    from app.llm.base import LLMProvider
    from app.store import RecordStore


def get_app_settings(request: Request) -> "Settings":
    return request.app.state.settings


def get_store(request: Request) -> "RecordStore":
    return request.app.state.store


def get_llm(request: Request) -> "LLMProvider":
    return request.app.state.llm
