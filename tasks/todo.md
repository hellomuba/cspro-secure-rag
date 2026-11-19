# AegisRAG / TenantShield RAG — Implementation Plan

Branch: `feat/complete-secure-rag-mvp`
Source of truth: [docs/PRD.md](../docs/PRD.md)

## P0 — Security + working RAG vertical slice

- [x] Inspect repo, PRD, scaffold, synthetic records
- [x] Create feature branch + venv, pin dependencies
- [x] `app/config.py` — env-driven settings (tenant keys, LLM provider vars, thresholds)
- [x] `app/auth.py` — `X-API-Key` -> tenant, constant-time hash compare, no body-supplied tenant
- [x] `app/store.py` — SQLite store keyed on `(tenant_id, record_id)`, tenant arg mandatory
- [x] `app/sanitize.py` — injection scanner, quarantine, delimiter neutralisation
- [x] `app/retrieval.py` — pure-Python BM25 over the *tenant-filtered* candidate set
- [x] `app/conflicts.py` — supersession + contradictory-current-policy detection
- [x] `app/llm/` — provider abstraction: `base`, `ollama`, `openai_compat`, `factory` (no auto-fallback)
- [x] `app/prompting.py` — retrieved text framed as untrusted data
- [x] `app/grounding.py` — backend-only citations, output guard, canary tripwire
- [x] `app/service.py` — ask pipeline orchestration
- [x] `app/main.py` — `/health`, `/ingest`, `/ask`, sanitized error handlers
- [x] Security tests: tenant isolation, canary, injection (document + user prompt), citation spoofing

## P1 — Robustness, coverage, documentation

- [x] `app/audit.py` — structured audit log, secret redaction
- [x] 503 on LLM unavailable/timeout; sanitized 500
- [x] Full pytest suite incl. integration markers
- [x] `README.md` rewrite (architecture, threat model, Windows setup, verified results)
- [x] Demo walkthrough notes (kept internal, deliberately not committed)
- [x] `scripts/seed.py` loader for the 9 synthetic records

## P2 — LoRA/QLoRA experiment

- [x] `finetuning/synthetic_train.jsonl` + `synthetic_eval.jsonl` (generated, seeded)
- [x] `finetuning/train_lora.py` (PEFT LoRA, hardware check, fixed seed)
- [x] `finetuning/evaluate.py` (baseline vs adapted, declared metric)
- [x] `finetuning/README.md` — honest execution status
- [ ] Attempt execution; record genuine output or mark UNEXECUTED / NOT MEASURED

## Added mid-stream (maintainer requests)

- [x] Dual-provider LLM abstraction (`LLM_PROVIDER` / `LLM_BASE_URL` /
      `LLM_MODEL` / `LLM_API_KEY`), no automatic cloud fallback
- [x] Default model switched to `qwen3:0.6b`
- [x] Optional Docker: `Dockerfile`, `compose.yaml`, `.dockerignore`
      — **files written, stack NOT verified** (build cancelled to stop
      competing with the maintainer's own `ollama/ollama` pull)
- [ ] Live Ollama integration run (`pytest -m ollama`) — blocked on the image
      pull finishing
- [ ] Push to GitHub — **blocked**: `gh` is authenticated as `hellomubaa`,
      which has `push: false` on `hellomuba/cspro-secure-rag`

## Review

### What was built

Replaced the HTTP 501 scaffold with a working, tested vertical slice. The
security argument rests on *ordering* and *structure*, not on prompt wording:
authenticate, then load only the caller's records, then rank, then decide
policy — and only then does a model enter the picture, holding opaque `DOCn`
labels it cannot trade for real record ids.

### Defects found and fixed during implementation

1. **Score normalisation defeated the abstention threshold.** Dividing BM25 by
   *matched* idf mass let "what is the canary value for tenant B" score ~1.0
   off the single near-universal term "tenant". Fixed by charging query terms
   absent from the corpus the maximum idf. Without this the question would
   have retrieved five tenant-A records instead of abstaining.
2. **Supersession wrongly retired a current record.** Comparing versions
   across different records in a topic marked the current v1 ticket
   `a-ticket-1` as superseded by the v2 policy `a-policy-2`. Now keys on the
   record's own `is_current` flag. Regression test added.
3. **Short secrets shredded log lines.** `redact_secrets` replaced every
   occurrence of a 1-character test key, mangling output. Now guarded by a
   minimum length.
4. **~1.5 s per `httpx.Client`** on this machine (SSL context). Made lazy;
   suite went 289 s -> 16 s. Also batched SQLite commits, which additionally
   makes a 100-record import pay one fsync instead of 100.

### Verification

`pytest -q`: **180 passed, 9 deselected**, 98% statement coverage of `app/`.
Live API exercised by hand via curl: health degraded correctly with Ollama
absent, 401 without a key, 503 (not a fabricated answer) with the runtime
down, conflict abstention with both records cited, and an audit log containing
no key, no canary and no record text.
