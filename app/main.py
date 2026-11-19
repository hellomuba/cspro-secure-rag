"""FastAPI application: wiring, routes and sanitized error handling.

Run locally (PowerShell):

    .\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000

Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.audit import audit, configure_logging, new_audit_id
from app.auth import Principal, require_tenant
from app.config import Settings, get_settings
from app.dependencies import get_app_settings, get_llm, get_store
from app.llm.base import LLMError, LLMProvider
from app.llm.factory import build_provider
from app.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ProviderHealth,
)
from app.service import AskService, DuplicateRecordIdError, IngestService
from app.store import RecordStore, TenantScopeError

VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)

    store = RecordStore(settings.db_path)
    if store.count_by_tenant().get("total", 0) == 0:
        try:
            loaded = store.seed_from_json(settings.records_seed_path)
            audit(settings, "seed", records=loaded, source=settings.records_seed_path)
        except FileNotFoundError:
            audit(settings, "seed", records=0, outcome="seed_file_missing")

    app.state.store = store
    app.state.llm = build_provider(settings)
    audit(
        settings,
        "startup",
        provider=app.state.llm.name,
        model=app.state.llm.model,
        execution=app.state.llm.execution,
        endpoint=app.state.llm.safe_endpoint,
    )
    try:
        yield
    finally:
        app.state.llm.close()
        store.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="AegisRAG / TenantShield RAG",
        version=VERSION,
        description=(
            "Secure multi-tenant retrieval-augmented generation over synthetic "
            "records. Tenant identity is derived from the X-API-Key header; it "
            "is never accepted from a request body."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings or get_settings()
    _register_routes(app)
    _register_error_handlers(app)
    return app


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(
        store: Annotated[RecordStore, Depends(get_store)],
        llm: Annotated[LLMProvider, Depends(get_llm)],
    ) -> HealthResponse:
        """Report API liveness separately from model availability.

        A reachable API with an unreachable model reports ``degraded``: the
        endpoint never claims the model is usable without probing it.
        """
        report = llm.health()
        usable = report.reachable and report.model_available is not False
        return HealthResponse(
            status="ok" if usable else "degraded",
            api="ok",
            llm=ProviderHealth(
                provider=report.provider,
                model=report.model,
                endpoint=report.endpoint,
                execution=report.execution,
                reachable=report.reachable,
                model_available=report.model_available,
                detail=report.detail,
            ),
            records=store.count_by_tenant(),
            version=VERSION,
        )

    @app.post(
        "/ingest",
        response_model=IngestResponse,
        tags=["records"],
        responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    def ingest(
        request: IngestRequest,
        principal: Annotated[Principal, Depends(require_tenant)],
        store: Annotated[RecordStore, Depends(get_store)],
        settings: Annotated[Settings, Depends(get_app_settings)],
    ) -> IngestResponse:
        """Idempotently upsert records into the authenticated tenant's namespace."""
        return IngestService(store=store, settings=settings).ingest(principal, request)

    @app.post(
        "/ask",
        response_model=AskResponse,
        tags=["rag"],
        responses={
            401: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def ask(
        request: AskRequest,
        principal: Annotated[Principal, Depends(require_tenant)],
        store: Annotated[RecordStore, Depends(get_store)],
        llm: Annotated[LLMProvider, Depends(get_llm)],
        settings: Annotated[Settings, Depends(get_app_settings)],
    ) -> AskResponse:
        """Answer from the authenticated tenant's records, or abstain."""
        return AskService(store=store, llm=llm, settings=settings).ask(principal, request)


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------
def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DuplicateRecordIdError)
    async def _duplicate(request: Request, exc: DuplicateRecordIdError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": f"Duplicate record id in batch: {exc.record_id}"},
        )

    @app.exception_handler(LLMError)
    async def _llm_unavailable(request: Request, exc: LLMError) -> JSONResponse:
        audit_id = new_audit_id()
        settings: Settings = request.app.state.settings
        audit(
            settings,
            "llm_failure",
            audit_id=audit_id,
            error_class=type(exc).__name__,
            provider=settings.llm_provider,
            outcome="unavailable",
        )
        # The message names the configured provider and what to do about it.
        # It never falls back to another provider.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc), "audit_id": audit_id},
        )

    @app.exception_handler(TenantScopeError)
    async def _tenant_scope(request: Request, exc: TenantScopeError) -> JSONResponse:
        audit_id = new_audit_id()
        audit(
            request.app.state.settings,
            "tenant_scope_violation",
            audit_id=audit_id,
            error_class=type(exc).__name__,
            outcome="blocked",
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Request blocked by an internal safety check.", "audit_id": audit_id},
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        audit_id = new_audit_id()
        audit(
            request.app.state.settings,
            "unhandled_error",
            audit_id=audit_id,
            path=request.url.path,
            error_class=type(exc).__name__,
            outcome="error",
        )
        # Deliberately opaque: no exception text, no stack trace, no record data.
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error.", "audit_id": audit_id},
        )


app = create_app()
