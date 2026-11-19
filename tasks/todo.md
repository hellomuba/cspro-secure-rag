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
- [x] `docs/DEMO_SCRIPT.md`
- [x] `scripts/seed.py` loader for the 9 synthetic records

## P2 — LoRA/QLoRA experiment

- [x] `finetuning/synthetic_train.jsonl` + `synthetic_eval.jsonl` (generated, seeded)
- [x] `finetuning/train_lora.py` (PEFT LoRA, hardware check, fixed seed)
- [x] `finetuning/evaluate.py` (baseline vs adapted, declared metric)
- [x] `finetuning/README.md` — honest execution status
- [ ] Attempt execution; record genuine output or mark UNEXECUTED / NOT MEASURED

## Review

See the "Review" section appended at the bottom of this file after implementation.
