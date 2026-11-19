"""Structured audit logging.

What is logged: audit id, tenant id, operation, outcome, the *ids* of cited
records, security flags, latency and the class name of any error.

What is never logged: API keys, the LLM API key, record bodies, excerpts,
question text or model output. Every emitted line additionally passes through
:func:`app.sanitize.redact_secrets` as a backstop, so a future code change that
accidentally puts a key into a field still cannot write it to disk.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from app.config import Settings
from app.sanitize import redact_secrets

logger = logging.getLogger("aegisrag.audit")

#: Fields that may never be emitted, regardless of caller.
_FORBIDDEN_FIELDS = frozenset(
    {"api_key", "x_api_key", "question", "answer", "body", "excerpt", "prompt"}
)


def new_audit_id() -> str:
    """A per-request correlation id, safe to return to the client."""
    return f"aud_{uuid.uuid4().hex[:16]}"


def configure_logging(level: str = "INFO") -> None:
    """Emit one JSON object per line on stderr."""
    root = logging.getLogger()
    if not any(getattr(h, "_aegisrag", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._aegisrag = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def audit(settings: Settings, event: str, **fields: Any) -> None:
    """Write one audit record."""
    payload: dict[str, Any] = {"event": event}
    for key, value in fields.items():
        if key in _FORBIDDEN_FIELDS:
            payload[key] = "<omitted-by-policy>"
            continue
        payload[key] = value

    line = json.dumps(payload, default=str, sort_keys=True, ensure_ascii=False)
    logger.info(redact_secrets(line, settings.secret_values))
