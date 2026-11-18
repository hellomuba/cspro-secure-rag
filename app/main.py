"""AegisRAG schema-first FastAPI scaffold. See docs/PRD.md for pending work."""
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="AegisRAG", version="0.1.0")


class Record(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=10000)
    topic: str = Field(min_length=1, max_length=100)
    version: int = Field(default=1, ge=1)
    is_current: bool = True


class IngestRequest(BaseModel):
    records: list[Record] = Field(min_length=1, max_length=100)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health():
    return {
        "status": "scaffold",
        "message": "Security, retrieval and LLM integration incomplete; not submission-ready",
    }


@app.post("/ingest", status_code=501)
def ingest(request: IngestRequest):
    return {"detail": "Authenticated ingestion not yet implemented"}


@app.post("/ask", status_code=501)
def ask(request: AskRequest):
    return {"detail": "Tenant-scoped RAG and local LLM not yet implemented"}
