"""Public request/response contracts.

Design rule: **no request model carries a tenant identifier**. Tenant identity
is derived server-side from the authenticated API key only. Input models use
``extra="forbid"`` so a client that tries to smuggle ``tenant_id`` into a body
receives HTTP 422 instead of being silently ignored.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
class RecordIn(StrictModel):
    """A single record submitted for ingestion.

    Deliberately has no ``tenant_id`` field.
    """

    id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._\-]+$")
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=10_000)
    topic: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._\- ]+$")
    version: int = Field(default=1, ge=1, le=10_000)
    is_current: bool = True


class IngestRequest(StrictModel):
    records: list[RecordIn] = Field(min_length=1, max_length=100)


class IngestStatus(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


class IngestResult(BaseModel):
    """Per-record outcome. ``unchanged`` is what an idempotent retry yields."""

    id: str
    status: IngestStatus
    quarantined: bool = False
    quarantine_reason: str | None = None


class IngestResponse(BaseModel):
    tenant_id: str
    accepted: int
    created: int
    updated: int
    unchanged: int
    quarantined: int
    results: list[IngestResult]
    audit_id: str


# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------
class AskRequest(StrictModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED_NO_EVIDENCE = "abstained_no_evidence"
    ABSTAINED_CONFLICT = "abstained_conflict"
    ABSTAINED_UNGROUNDED = "abstained_ungrounded"
    BLOCKED_POLICY_VIOLATION = "blocked_policy_violation"


class SourceOut(BaseModel):
    """A citation. Only ever constructed by the backend from an authorized,
    tenant-filtered record -- never parsed out of model output."""

    id: str
    tenant_id: str
    title: str
    topic: str
    version: int
    is_current: bool
    score: float
    excerpt: str


class SupersededRef(BaseModel):
    id: str
    title: str
    version: int
    superseded_by: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    conflict: bool
    status: AnswerStatus
    sources: list[SourceOut]
    superseded_excluded: list[SupersededRef] = Field(default_factory=list)
    security_flags: list[str] = Field(default_factory=list)
    audit_id: str
    provider: str
    model: str
    #: "local" when the answer came from on-machine inference, "remote"
    #: otherwise. Never inferred -- reported by the provider that ran.
    execution: Literal["local", "remote", "none"]


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
class ProviderHealth(BaseModel):
    provider: str
    model: str
    endpoint: str
    #: "local" only when inference runs on this machine (Ollama).
    execution: Literal["local", "remote"]
    reachable: bool
    model_available: bool | None = None
    detail: str | None = None

    # ``model_`` is a protected namespace in pydantic v2; opt out explicitly.
    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api: Literal["ok"]
    llm: ProviderHealth
    records: dict[str, int]
    version: str


class ErrorResponse(BaseModel):
    detail: str
    audit_id: str | None = None
