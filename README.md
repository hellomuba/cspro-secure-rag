# AegisRAG | Secure Multi-Tenant RAG Assessment

A local-first security demonstration for the **CSPRO Senior AI Developer / Open Source LLM Engineer** technical assessment. Uses **100% synthetic** ticket and policy records.

> **Status: INITIAL SCAFFOLD / NOT SUBMISSION-READY.** The APIs and model integration are not yet implemented or validated. Do not present this repository as a completed assessment until implementation, execution, tests and a demonstration video are complete.

## The incident

After a batch-import retry, a document belonging to Tenant B contaminates Tenant A's RAG context and includes a forged SYSTEM instruction. AegisRAG is designed to prevent unauthorized retrieval, citation and answer leakage **through application-level authorization**, independent of model behavior.

## Scope and stack

- Python 3.11+, FastAPI, Pydantic and pytest
- Tenant-scoped authenticated ingestion and retrieval
- Ollama with the local open-weight model `qwen2.5:0.5b-instruct`
- Verifiable source IDs and excerpts, abstention and conflict detection
- Automated cross-tenant leakage and prompt-injection regression tests
- Reproducible synthetic LoRA/QLoRA experiment and baseline evaluation

Full requirements, API contracts, acceptance criteria and delivery priorities: **[docs/PRD.md](docs/PRD.md)**.

## Models and development hardware

**Default local model:** `qwen2.5:0.5b-instruct` (approximately 398 MB, Q4 quantization). This is a lightweight CPU-friendly development choice, but it may be less reliable than larger models on grounding and instruction following. **Optional comparison model:** `qwen2.5:1.5b-instruct` (approximately 986 MB); download it separately and set `OLLAMA_MODEL` accordingly.

**Hardware reference retained:** HP ZBook G5, the machine previously discussed for this assessment. **Current machine being used for the work:** a different Windows x64 PC with an 11th Gen Intel Core i7-1165G7 and 32 GB RAM. Its GPU and exact model are not confirmed. Do not assume the current PC is the ZBook, or that it supports CUDA/QLoRA. Only use organization-managed hardware if permitted.

The final demo must test the **actual installed model**; do not claim live Ollama inference until it is executed and verified.

## Quick start (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:0.5b-instruct
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
.\.venv\Scripts\python.exe -m pytest -q
```

API docs: http://127.0.0.1:8000/docs

**Important:** At this stage, `/health` identifies the application as a scaffold and `/ingest` and `/ask` explicitly report that the implementation is pending. The commands above do not turn this into functioning RAG.

## Intended API

| Method | Route | Intended behavior |
| --- | --- | --- |
| GET | `/health` | API and local-model runtime health |
| POST | `/ingest` | Authenticated tenant-owned synthetic records, idempotent upsert |
| POST | `/ask` | Authorized retrieval and cited, grounded local-model answer |

Tenant identity must be derived from `X-API-Key` rather than a client-supplied tenant identifier. No source ID or excerpt may be accepted merely because the LLM generated it.

## Data security

All nine sample records are entirely fictional and contain no real candidate, patient, client, employer or production information. Never commit credentials, real identity data or proprietary documents. Respect any employer-managed-device policy.

## Assessment tracking

- [x] PRD and architecture specification
- [x] Synthetic two-tenant incident dataset
- [x] Initial API schemas and scaffold tests
- [ ] Authenticated ingestion and tenant-scoped retrieval
- [ ] Local Ollama generation with citations and abstention
- [ ] Prompt-injection and cross-tenant security tests
- [ ] Reproducible LoRA/QLoRA scripts and evaluation
- [ ] Verified local run, test evidence and 3–5-minute demo

**Fine-tuning execution status:** UNEXECUTED. No performance metrics are claimed.

## AI assistance disclosure

The PRD and initial scaffold were prepared with assistance from ChatGPT (OpenAI). The candidate must independently review, execute, debug and explain all code and results before submission.

## Naming and license

**AegisRAG is a working project name**, not a claim of exclusive trademark rights. The GitHub repository slug remains `cspro-secure-rag` and can stay unchanged if the display name changes later. Assessment code is not licensed for third-party reuse by default.
