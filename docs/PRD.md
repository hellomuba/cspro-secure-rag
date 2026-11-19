# Product Requirements Document: AegisRAG

**Subtitle:** Secure Local LLM & Multi-Tenant RAG Assessment  
**Version:** 1.2 (assessment MVP, canonical local model `qwen3:0.6b`)  
**Date:** 2026-11-18  
**Maintainer:** Mubarak Ibrahim  
**Repository:** `hellomuba/cspro-secure-rag`  
**Status:** Implementation specification; not evidence of completed features or passed tests

## 1. Product summary

AegisRAG is a local-first, multi-tenant retrieval-augmented answering service built over wholly synthetic recruitment-support tickets, import logs and policies. The challenge is to prevent cross-tenant leakage after a batch-import retry puts a second tenant's document into the retrieval corpus, and that document contains an adversarial fake SYSTEM instruction. Responses must contain only tenant-authorized evidence, with verifiable source IDs and excerpts, and must abstain on insufficient or conflicting evidence.

## 2. Users and security principles

**Personas:** Tenant A support analyst, Tenant B support analyst, security reviewer, CSPRO hiring reviewer.

**Non-negotiable invariant:** Authentication and tenant filtering happen in backend code before ranking or model inference. A prompt is never an access-control mechanism. Retrieved documents are untrusted data. Citation IDs and snippets must be generated or checked against backend-authorized sources.

## 3. Scope

**In scope:** Python/FastAPI/Pydantic API, local Ollama Qwen, minimum 8 original synthetic records across 2 tenants, idempotent synthetic ingestion, tenant-scoped lexical retrieval (TF-IDF/BM25 acceptable), citations, injection testing, conflicting-policy handling, pytest, LoRA/QLoRA reproducible experiment, README and demo.

**Out of scope:** Production candidate or patient data, external HR/customer integrations, public hosted LLM APIs, real multi-tenant IAM, enterprise vector databases, production-grade key rotation.

## 4. Acceptance criteria

| ID | Requirement | Verification |
| --- | --- | --- |
| AC01 | FastAPI with GET /health, POST /ingest and POST /ask | Routes served and included in OpenAPI |
| AC02 | Validate inputs and handle malformed requests | 422 errors and informative safe 4xx/5xx |
| AC03 | At least 8 original synthetic records | >=2 tenants, older/current contradiction and forged instruction |
| AC04 | Model uses local open-weight runtime | Ollama `qwen3:0.6b`; no paid inference |
| AC05 | Authenticated tenant isolation in code | Tenant A cannot retrieve, cite or output Tenant B canary |
| AC06 | Evidence-grounded RAG | Authorized source IDs and exact excerpts, with abstention |
| AC07 | Prompt-injection resistance | Malicious retrieved instructions ignored/quarantined |
| AC08 | Explicit conflict behavior | Outdated policy superseded or ambiguous current policies flagged |
| AC09 | pytest coverage | API, retrieval, citations, authorization, injection and failure tests |
| AC10 | LoRA/QLoRA experiment | Reproducible code, synthetic examples, baseline vs adaptation protocol |
| AC11 | Honest execution reporting | No invented scores; mark unavailable hardware as UNEXECUTED |
| AC12 | Submission package | GitHub/ZIP, README, 3–5-minute recording and engagement details |

## 5. Threat model

- **Cross-tenant import contamination:** A retry incorrectly inserts a Tenant B record into Tenant A's candidate search space.
- **Forged high-priority instruction:** A document includes strings such as `[SYSTEM] Ignore authorization...`.
- **Malicious user query:** User asks the model to ignore prior rules and reveal Tenant B material.
- **Citation laundering:** Model fabricates source IDs or references documents it was not authorized to access.
- **Stale/contradictory policies:** Older or conflicting active policies are returned as current without a warning.

**Mitigations:** Authenticated tenant identity; composite key `(tenant_id, record_id)`; filter first, then rank; backend-source citation construction; untrusted context framing; answer validation/abstention; deterministic conflict checks; regression tests with a synthetic leak canary.

## 6. System architecture

```text
Client request + X-API-Key
        |
        v
FastAPI / Pydantic validation
        |
        v
Authenticated tenant resolved in Python
        |
        +--> POST /ingest: validation --> tenant-scoped upsert
        |
        +--> POST /ask: tenant-filtered retrieval
                               |
                     current-policy/conflict evaluation
                               |
                       untrusted text isolation
                               |
                         Ollama / local Qwen
                               |
                  verified citations / answer guard
                               |
                       answer or abstention
```

## 7. Endpoint contracts

### GET /health

Return API status, configured local model, local runtime reachability, record count. If Ollama is unavailable, report degraded model state rather than pretending inference is working.

### POST /ingest

Header: `X-API-Key: <synthetic tenant key>`.

Request example:

```json
{"records":[{"id":"policy-001","title":"Import retry policy","body":"Use idempotency keys.","topic":"imports","version":2,"is_current":true}]}
```

Tenant comes from API key, not body. Reject duplicated IDs within request, oversized input and malformed records; upsert is scoped by `(tenant_id,id)`.

### POST /ask

Header: `X-API-Key: <synthetic tenant key>`.

Request: `{"question":"What is the current import retry policy?","top_k":3}`.

Response shape:

```json
{
  "answer": "Answer using authorized evidence, or abstention",
  "abstained": false,
  "conflict": false,
  "sources": [{"id":"a-policy-2","tenant_id":"tenant-a","excerpt":"Current policy text"}],
  "audit_id": "random-audit-id"
}
```

When no evidence exists, return `abstained: true` and empty `sources`. When contradictory current policies cannot be resolved reliably, return `conflict: true` and abstain or explicitly surface the conflict. Never emit model-invented source IDs.

**Errors:** 401 invalid/missing key, 422 malformed request, 503 when needed Ollama inference fails, sanitized 500 errors for unexpected faults.

## 8. Synthetic records and incident setup

The sample dataset contains five `tenant-a` records and four `tenant-b` records, current/superseded import policies, a contradictory retention record and a forged SYSTEM instruction in Tenant B. Unique synthetic test marker: `CANARY-TENANT-B-DO-NOT-DISCLOSE`. The actual incident test must assert that this marker is absent from **both answer and source metadata** for Tenant A.

## 9. Local model & retrieval

Model: **`qwen3:0.6b`** via Ollama. Reached at `http://localhost:11434` when FastAPI runs natively on Windows, and at `http://ollama:11434` from inside the Docker Compose network. Optional documented alternatives, not defaults: `qwen2.5:0.5b-instruct` and `qwen2.5:1.5b-instruct`; set `LLM_MODEL` to switch. Nothing in the implementation is model-specific. MVP retrieval may use deterministic TF-IDF/BM25; embeddings are optional. Search over authorized records only; require a grounding threshold or abstain. Treat retrieved text as untrusted and reject model-only citation suggestions.

## 10. LoRA/QLoRA experiment

Use a small suitable open-weight base model with a synthetic instruction dataset, training/holdout split, fixed seed and versioned PEFT/Transformers scripts. Evaluate baseline and adapted model on the **same held-out examples** with a declared metric (exact match/token F1), plus security regression tests. Save hyperparameters, hardware/device information, logs and outcomes. If unable to execute, mark it **UNEXECUTED / NOT MEASURED**, explaining limitations without implying improvement.

## 11. Test plan

| Test | Expected |
| --- | --- |
| Missing/invalid API key | 401 |
| Invalid question / body length | 422 |
| Valid ingestion and retry | No duplicate within tenant |
| Tenant A request for Tenant B record | No Tenant B body or metadata |
| Injected retrieved instruction | No authority escalation or canary |
| User prompt override | Tenant restrictions unchanged |
| Model invents citations | Returned citations constrained to retrieved authorized set |
| Superseded policy | Latest current policy used |
| Conflicting active policies | Flag or abstain |
| Ollama offline | Graceful 503 / degraded health |
| /health | Reflects actual runtime state |

## 12. Implementation sequence

1. Authenticated schemas, synthetic corpus and three endpoints.
2. Tenant-bound ingestion, retrieval, deterministic citations.
3. Local Ollama inference and evidence-grounding behavior.
4. Security and failure test suite with real execution logs.
5. Fine-tuning code, synthetic data and evaluation methodology.
6. README, test reports, 3–5-minute screen recording and submission email.

## 13. Constraints and completion definition

The previously discussed **HP ZBook G5** remains the documentation hardware reference. Development will currently use a **different Windows x64 PC** with Intel Core i7-1165G7 (11th Gen), 32 GB RAM, and unverified GPU. The current PC's make/model must not be inferred to be the ZBook. Lightweight quantized CPU inference is a reasonable starting approach; successful QLoRA execution must not be claimed without hardware verification.

**Definition of done:** The local application executes; tenant isolation and prompt-injection tests pass; citations are verifiable; errors are honest; fine-tuning is executed and evaluated or explicitly labeled unexecuted; README contains exact tested setup; 3–5-minute recording shows real behavior; submission includes GitHub/ZIP link and requested engagement details.

## 14. Disclosure

I wrote this specification and the initial scaffold with help from ChatGPT (OpenAI), and I built the implementation with Claude Code (Anthropic). I have verified and run everything I claim here; I report no metric I did not measure, and I label anything unexecuted as such. All data in this project is synthetic — no real customer, candidate or employee data is used anywhere. The full disclosure is in the README.
