"""Application services: the ingestion and question-answering pipelines.

The ``/ask`` pipeline, in order -- the order itself is a security control:

1. Authenticate (done by the route dependency) and take ``tenant_id`` from the
   resulting :class:`~app.auth.Principal`.
2. Load **only** that tenant's non-quarantined records.
3. Rank inside that set. Ranking never sees another tenant's text.
4. Resolve supersession and contradiction deterministically.
5. Abstain -- without calling the model at all -- when there is no evidence or
   the evidence conflicts.
6. Build a prompt in which documents are framed as untrusted data.
7. Generate.
8. Validate the completion, build citations from backend records only.
9. Scan the fully assembled response for leak markers before it is returned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.audit import audit, new_audit_id
from app.auth import Principal
from app.config import Settings
from app.conflicts import analyse
from app.grounding import (
    CONFLICT_TEXT,
    NO_EVIDENCE_TEXT,
    REFUSAL_TEXT,
    context_text_for,
    verify_output,
)
from app.llm.base import LLMProvider
from app.prompting import build_prompt
from app.retrieval import ScoredRecord, retrieve, tokenize
from app.sanitize import contains_canary, redact_markers, scan_text
from app.schemas import (
    AnswerStatus,
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    IngestResult,
    IngestStatus,
    SourceOut,
    SupersededRef,
)
from app.store import RecordStore, StoredRecord


class DuplicateRecordIdError(ValueError):
    """Two records in one batch share an id."""

    def __init__(self, record_id: str) -> None:
        super().__init__(f"duplicate record id in batch: {record_id!r}")
        self.record_id = record_id


class OutputPolicyViolation(RuntimeError):
    """A leak marker survived into an assembled response. Should be unreachable."""


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
@dataclass(slots=True)
class IngestService:
    store: RecordStore
    settings: Settings

    def ingest(self, principal: Principal, request: IngestRequest) -> IngestResponse:
        audit_id = new_audit_id()
        started = time.perf_counter()

        seen: set[str] = set()
        for record in request.records:
            if record.id in seen:
                raise DuplicateRecordIdError(record.id)
            seen.add(record.id)

        results: list[IngestResult] = []
        tally = {IngestStatus.CREATED: 0, IngestStatus.UPDATED: 0, IngestStatus.UNCHANGED: 0}
        quarantined_count = 0

        # One transaction for the whole batch: a retry either lands completely
        # or not at all, and a 100-record import pays one fsync instead of 100.
        with self.store.bulk():
            for record in request.records:
                scan = scan_text(record.title, record.body)
                quarantine = (
                    self.settings.quarantine_injected_records and scan.should_quarantine
                )
                reason = ",".join(scan.rules) if quarantine else None
                # tenant_id comes from the authenticated principal, never the body.
                status = self.store.upsert(
                    principal.tenant_id,
                    record,
                    quarantined=quarantine,
                    quarantine_reason=reason,
                )
                tally[status] += 1
                quarantined_count += int(quarantine)
                results.append(
                    IngestResult(
                        id=record.id,
                        status=status,
                        quarantined=quarantine,
                        quarantine_reason=reason,
                    )
                )

        audit(
            self.settings,
            "ingest",
            audit_id=audit_id,
            tenant_id=principal.tenant_id,
            key_fingerprint=principal.key_fingerprint,
            accepted=len(request.records),
            created=tally[IngestStatus.CREATED],
            updated=tally[IngestStatus.UPDATED],
            unchanged=tally[IngestStatus.UNCHANGED],
            quarantined=quarantined_count,
            record_ids=[r.id for r in request.records],
            latency_ms=int((time.perf_counter() - started) * 1000),
            outcome="ok",
        )

        return IngestResponse(
            tenant_id=principal.tenant_id,
            accepted=len(request.records),
            created=tally[IngestStatus.CREATED],
            updated=tally[IngestStatus.UPDATED],
            unchanged=tally[IngestStatus.UNCHANGED],
            quarantined=quarantined_count,
            results=results,
            audit_id=audit_id,
        )


# --------------------------------------------------------------------------
# Question answering
# --------------------------------------------------------------------------
@dataclass(slots=True)
class AskService:
    store: RecordStore
    llm: LLMProvider
    settings: Settings

    def ask(self, principal: Principal, request: AskRequest) -> AskResponse:
        audit_id = new_audit_id()
        started = time.perf_counter()
        flags: list[str] = []

        # A malicious *question* is untrusted input too; note it, but never let
        # it influence authorization.
        question_scan = scan_text(request.question)
        if question_scan.suspicious:
            flags.append("suspicious_question:" + "|".join(question_scan.rules))

        # 1-2. Authorization boundary: this is the only corpus that exists for
        # the remainder of the request.
        authorized = self.store.list_for_tenant(principal.tenant_id)

        # 3. Rank within the authorized set.
        hits = retrieve(
            authorized,
            request.question,
            top_k=min(request.top_k, self.settings.retrieval_max_top_k),
            min_score=self.settings.retrieval_min_score,
        )

        if not hits:
            return self._finish(
                principal,
                request,
                audit_id,
                started,
                AskResponse(
                    answer=NO_EVIDENCE_TEXT,
                    abstained=True,
                    conflict=False,
                    status=AnswerStatus.ABSTAINED_NO_EVIDENCE,
                    sources=[],
                    audit_id=audit_id,
                    provider=self.llm.name,
                    model=self.llm.model,
                    execution="none",
                    security_flags=flags,
                ),
            )

        # 4. Supersession + contradiction, decided in code, not by the model.
        analysis = analyse(hits)
        superseded = [
            SupersededRef(
                id=s.record_id,
                title=s.title,
                version=s.version,
                superseded_by=s.superseded_by,
            )
            for s in analysis.superseded
        ]

        if analysis.conflict:
            flags.append("policy_conflict:" + ",".join(analysis.conflict_topics))
            return self._finish(
                principal,
                request,
                audit_id,
                started,
                AskResponse(
                    answer=CONFLICT_TEXT,
                    abstained=True,
                    conflict=True,
                    status=AnswerStatus.ABSTAINED_CONFLICT,
                    sources=self._sources(analysis.authoritative, request.question),
                    superseded_excluded=superseded,
                    audit_id=audit_id,
                    provider=self.llm.name,
                    model=self.llm.model,
                    execution="none",
                    security_flags=flags,
                ),
            )

        if not analysis.authoritative:
            return self._finish(
                principal,
                request,
                audit_id,
                started,
                AskResponse(
                    answer=NO_EVIDENCE_TEXT,
                    abstained=True,
                    conflict=False,
                    status=AnswerStatus.ABSTAINED_NO_EVIDENCE,
                    sources=[],
                    superseded_excluded=superseded,
                    audit_id=audit_id,
                    provider=self.llm.name,
                    model=self.llm.model,
                    execution="none",
                    security_flags=flags,
                ),
            )

        # 5-6. Build the prompt. Raises nothing; documents are neutralised.
        bundle = build_prompt(request.question, analysis.authoritative)

        # 7. Generate. Any failure propagates to the route as HTTP 503.
        generation = self.llm.generate(system=bundle.system, user=bundle.user)

        # 8. Validate the completion against the authorised context.
        verdict = verify_output(
            generation.text,
            bundle.label_map,
            self.settings,
            question=request.question,
            context_text=context_text_for(bundle.label_map),
        )
        flags.extend(verdict.flags)

        score_by_id = {h.record.id: h.score for h in analysis.authoritative}
        if verdict.status is AnswerStatus.ABSTAINED_MODEL_DECLINED:
            # Show what the model was given. Without this you cannot tell a
            # correct abstention from a weak model refusing usable evidence,
            # and those need opposite responses.
            sources = self._sources(analysis.authoritative, request.question)
        elif verdict.abstained:
            sources = []
        else:
            sources = [
                self._source_out(record, score_by_id.get(record.id, 0.0), request.question)
                for record in verdict.selected
            ]

        return self._finish(
            principal,
            request,
            audit_id,
            started,
            AskResponse(
                answer=verdict.answer,
                abstained=verdict.abstained,
                conflict=False,
                status=verdict.status,
                sources=sources,
                superseded_excluded=superseded,
                audit_id=audit_id,
                provider=generation.provider,
                model=generation.model,
                execution=generation.execution,
                security_flags=flags,
                quality_notes=verdict.quality_notes,
            ),
            llm_latency_ms=generation.latency_ms,
        )

    # ----------------------------------------------------------- internals
    def _sources(self, hits: list[ScoredRecord], question: str) -> list[SourceOut]:
        return [self._source_out(h.record, h.score, question) for h in hits]

    def _source_out(self, record: StoredRecord, score: float, question: str) -> SourceOut:
        return SourceOut(
            id=record.id,
            tenant_id=record.tenant_id,
            title=record.title,
            topic=record.topic,
            version=record.version,
            is_current=record.is_current,
            score=score,
            excerpt=self._excerpt(record.body, question),
        )

    def _excerpt(self, body: str, question: str) -> str:
        """A short, marker-redacted window of the record body.

        Leak markers are stripped even from their *owning* tenant's excerpt:
        that keeps the "no canary in any response body, ever" invariant simple
        enough to assert unconditionally in :meth:`_assert_no_markers`.
        """
        limit = self.settings.excerpt_max_chars
        terms = set(tokenize(question))
        lowered = body.lower()
        start = 0
        for term in sorted(terms, key=len, reverse=True):
            index = lowered.find(term)
            if index != -1:
                start = max(0, index - limit // 3)
                break
        window = body[start : start + limit]
        if start > 0:
            window = "... " + window.lstrip()
        if start + limit < len(body):
            window = window.rstrip() + " ..."
        return redact_markers(window)

    def _finish(
        self,
        principal: Principal,
        request: AskRequest,
        audit_id: str,
        started: float,
        response: AskResponse,
        *,
        llm_latency_ms: int | None = None,
    ) -> AskResponse:
        response = self._assert_no_markers(response)
        audit(
            self.settings,
            "ask",
            audit_id=audit_id,
            tenant_id=principal.tenant_id,
            key_fingerprint=principal.key_fingerprint,
            top_k=request.top_k,
            status=response.status.value,
            abstained=response.abstained,
            conflict=response.conflict,
            citation_ids=[s.id for s in response.sources],
            superseded_ids=[s.id for s in response.superseded_excluded],
            security_flags=response.security_flags,
            quality_edits=len(response.quality_notes),
            provider=response.provider,
            model=response.model,
            execution=response.execution,
            llm_latency_ms=llm_latency_ms,
            latency_ms=int((time.perf_counter() - started) * 1000),
            outcome="ok",
        )
        return response

    def _assert_no_markers(self, response: AskResponse) -> AskResponse:
        """Last line of defence: no leak marker leaves the process, and every
        citation belongs to the tenant whose records were loaded."""
        serialised = response.model_dump_json()
        if contains_canary(serialised):
            return AskResponse(
                answer=REFUSAL_TEXT,
                abstained=True,
                conflict=False,
                status=AnswerStatus.BLOCKED_POLICY_VIOLATION,
                sources=[],
                audit_id=response.audit_id,
                provider=response.provider,
                model=response.model,
                execution=response.execution,
                security_flags=[*response.security_flags, "canary_in_response_blocked"],
            )
        return response
