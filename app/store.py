"""Tenant-scoped persistence (SQLite).

Security invariants enforced here:

1. The primary key is the composite ``(tenant_id, record_id)``. A record id is
   only ever unique *within* a tenant, so one tenant can never overwrite or
   address another tenant's row.
2. Every read method takes ``tenant_id`` as a required positional argument and
   applies it as a SQL ``WHERE`` predicate. There is no "fetch all records"
   method on the retrieval path.
3. Rows are re-checked against the requested tenant after they come back from
   SQLite, so a future query bug cannot silently widen the scope.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.schemas import IngestStatus, RecordIn

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    tenant_id         TEXT    NOT NULL,
    record_id         TEXT    NOT NULL,
    title             TEXT    NOT NULL,
    body              TEXT    NOT NULL,
    topic             TEXT    NOT NULL,
    version           INTEGER NOT NULL,
    is_current        INTEGER NOT NULL,
    content_hash      TEXT    NOT NULL,
    quarantined       INTEGER NOT NULL DEFAULT 0,
    quarantine_reason TEXT,
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL,
    PRIMARY KEY (tenant_id, record_id)
);
CREATE INDEX IF NOT EXISTS idx_records_tenant_topic ON records (tenant_id, topic);
"""


class TenantScopeError(RuntimeError):
    """Raised when a row escapes its tenant scope. Should be unreachable."""


@dataclass(frozen=True, slots=True)
class StoredRecord:
    tenant_id: str
    id: str
    title: str
    body: str
    topic: str
    version: int
    is_current: bool
    quarantined: bool
    quarantine_reason: str | None


def content_hash(record: RecordIn) -> str:
    payload = json.dumps(
        {
            "title": record.title,
            "body": record.body,
            "topic": record.topic,
            "version": record.version,
            "is_current": record.is_current,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RecordStore:
    """A small, thread-safe, tenant-scoped SQLite record store."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        # Re-entrant: `bulk()` holds the lock for the whole batch and the
        # `upsert` calls inside it re-acquire on the same thread.
        self._lock = threading.RLock()
        self._bulk_depth = 0
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        # WAL + NORMAL is the standard durable-enough pairing: a crash can lose
        # the last transaction but cannot corrupt the database. For a synthetic
        # demo corpus that trade is correct, and it removes an fsync per record.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def bulk(self) -> Iterator[None]:
        """Group many ``upsert`` calls into one transaction and one commit.

        A 100-record batch import otherwise pays one fsync per record.

        The lock is held for the whole block, which serialises concurrent
        batches. That is deliberate: ``_bulk_depth`` is plain integer state,
        and FastAPI runs sync endpoints in a threadpool, so two overlapping
        ingests could otherwise interleave their increments and leave the
        depth above zero -- at which point no commit would ever fire and
        writes would sit unflushed. Serialising is the cheap, obviously
        correct option at this scale.

        An exception inside the block rolls the whole batch back, so a failed
        import cannot leave half its records committed.
        """
        with self._lock:
            self._bulk_depth += 1
            try:
                yield
            except BaseException:
                self._bulk_depth -= 1
                if self._bulk_depth == 0:
                    self._conn.rollback()
                raise
            else:
                self._bulk_depth -= 1
                if self._bulk_depth == 0:
                    self._conn.commit()

    def _maybe_commit(self) -> None:
        """Commit unless a :meth:`bulk` block will do it later. Caller holds the lock."""
        if self._bulk_depth == 0:
            self._conn.commit()

    # ---------------------------------------------------------------- writes
    def upsert(
        self,
        tenant_id: str,
        record: RecordIn,
        *,
        quarantined: bool = False,
        quarantine_reason: str | None = None,
    ) -> IngestStatus:
        """Insert or update a record inside ``tenant_id``'s namespace.

        Returns ``UNCHANGED`` when the submitted content is byte-identical to
        what is already stored -- this is what makes a batch-import retry
        idempotent rather than duplicating or churning rows.
        """
        _require_tenant(tenant_id)
        digest = content_hash(record)
        now = datetime.now(UTC).isoformat()

        with self._lock:
            cur = self._conn.execute(
                "SELECT content_hash FROM records WHERE tenant_id = ? AND record_id = ?",
                (tenant_id, record.id),
            )
            existing = cur.fetchone()

            if existing is not None and existing["content_hash"] == digest:
                # Idempotent retry: refresh quarantine verdict only.
                self._conn.execute(
                    "UPDATE records SET quarantined = ?, quarantine_reason = ? "
                    "WHERE tenant_id = ? AND record_id = ?",
                    (int(quarantined), quarantine_reason, tenant_id, record.id),
                )
                self._maybe_commit()
                return IngestStatus.UNCHANGED

            self._conn.execute(
                """
                INSERT INTO records (tenant_id, record_id, title, body, topic, version,
                                     is_current, content_hash, quarantined,
                                     quarantine_reason, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, record_id) DO UPDATE SET
                    title = excluded.title,
                    body = excluded.body,
                    topic = excluded.topic,
                    version = excluded.version,
                    is_current = excluded.is_current,
                    content_hash = excluded.content_hash,
                    quarantined = excluded.quarantined,
                    quarantine_reason = excluded.quarantine_reason,
                    updated_at = excluded.updated_at
                """,
                (
                    tenant_id,
                    record.id,
                    record.title,
                    record.body,
                    record.topic,
                    record.version,
                    int(record.is_current),
                    digest,
                    int(quarantined),
                    quarantine_reason,
                    now,
                    now,
                ),
            )
            self._maybe_commit()

        return IngestStatus.CREATED if existing is None else IngestStatus.UPDATED

    # ----------------------------------------------------------------- reads
    def list_for_tenant(
        self, tenant_id: str, *, include_quarantined: bool = False
    ) -> list[StoredRecord]:
        """Return every record visible to ``tenant_id`` and nothing else.

        This is the only method the retrieval path may call.
        """
        _require_tenant(tenant_id)
        sql = "SELECT * FROM records WHERE tenant_id = ?"
        params: list[object] = [tenant_id]
        if not include_quarantined:
            sql += " AND quarantined = 0"
        sql += " ORDER BY record_id"

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        records = [_row_to_record(row) for row in rows]
        # Defence in depth: the query above is tenant-scoped, but verify the
        # result set too so a future refactor cannot widen it unnoticed.
        for record in records:
            if record.tenant_id != tenant_id:
                raise TenantScopeError(
                    f"record {record.id!r} escaped tenant scope {tenant_id!r}"
                )
        return records

    def get(self, tenant_id: str, record_id: str) -> StoredRecord | None:
        _require_tenant(tenant_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM records WHERE tenant_id = ? AND record_id = ?",
                (tenant_id, record_id),
            ).fetchone()
        if row is None:
            return None
        record = _row_to_record(row)
        if record.tenant_id != tenant_id:  # pragma: no cover - unreachable
            raise TenantScopeError("tenant scope violation in get()")
        return record

    def count_by_tenant(self) -> dict[str, int]:
        """Aggregate counts for ``/health``. Returns counts only, never bodies."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT tenant_id, COUNT(*) AS n FROM records GROUP BY tenant_id"
            ).fetchall()
        counts = {row["tenant_id"]: int(row["n"]) for row in rows}
        counts["total"] = sum(counts.values())
        return counts

    # ------------------------------------------------------------- seeding
    def seed_from_json(self, path: str | Path, *, quarantine: bool = True) -> int:
        """Load the synthetic corpus from disk (trusted admin path).

        Unlike ``POST /ingest`` the seed file *does* carry ``tenant_id``: it is
        an operator-controlled fixture, not a client request. The injection
        scanner is still applied so the forged-SYSTEM record is quarantined
        exactly as it would be on the HTTP path.
        """
        from app.sanitize import scan_text  # local import: avoids import cycle

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        loaded = 0
        with self.bulk():
            for raw in data:
                tenant_id = raw["tenant_id"]
                record = RecordIn(
                    id=raw["id"],
                    title=raw["title"],
                    body=raw["body"],
                    topic=raw["topic"],
                    version=int(raw.get("version", 1)),
                    is_current=bool(raw.get("is_current", True)),
                )
                scan = scan_text(record.title, record.body)
                should_quarantine = quarantine and scan.should_quarantine
                self.upsert(
                    tenant_id,
                    record,
                    quarantined=should_quarantine,
                    quarantine_reason=",".join(scan.rules) if should_quarantine else None,
                )
                loaded += 1
        return loaded


def _row_to_record(row: sqlite3.Row) -> StoredRecord:
    return StoredRecord(
        tenant_id=row["tenant_id"],
        id=row["record_id"],
        title=row["title"],
        body=row["body"],
        topic=row["topic"],
        version=int(row["version"]),
        is_current=bool(row["is_current"]),
        quarantined=bool(row["quarantined"]),
        quarantine_reason=row["quarantine_reason"],
    )


def _require_tenant(tenant_id: str) -> None:
    if not tenant_id or not isinstance(tenant_id, str):
        raise TenantScopeError("a non-empty tenant_id is required for every query")
