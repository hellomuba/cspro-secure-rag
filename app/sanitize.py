"""Untrusted-text handling: injection detection, quarantine and neutralisation.

Threat model note
-----------------
Nothing in this module is an authorization boundary. Pattern matching on
adversarial natural language is inherently incomplete -- an attacker can always
paraphrase. These controls exist to (a) *quarantine* obvious forged-instruction
payloads at ingestion time and (b) make it structurally harder for document
text to be read as a control instruction.

The authorization boundary is the tenant filter in :mod:`app.store`, which runs
before ranking and before any text reaches the model, plus the backend-only
citation construction in :mod:`app.grounding`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.config import CANARY_MARKERS

# --------------------------------------------------------------------------
# Detection rules
# --------------------------------------------------------------------------
# Each rule is (rule_id, compiled pattern, weight). Weight >= 3 on its own is
# enough to quarantine a record.
_RULES: list[tuple[str, re.Pattern[str], int]] = [
    (
        "forged_system_block",
        re.compile(r"\[\s*/?\s*system\s*\]|<\s*/?\s*system\s*>", re.IGNORECASE),
        3,
    ),
    (
        "chat_template_token",
        re.compile(r"<\|\s*im_(start|end)\s*\|>|<\|\s*(system|assistant|user)\s*\|>", re.IGNORECASE),
        3,
    ),
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
            r"\b(all|any|previous|prior|above|earlier|the)\b[^.\n]{0,40}?"
            r"\b(instruction|instructions|prompt|prompts|rule|rules|polic\w+|restriction\w*|guardrail\w*)\b",
            re.IGNORECASE,
        ),
        3,
    ),
    (
        "role_reassignment",
        re.compile(
            r"\byou\s+are\s+now\b|\bact\s+as\s+(?:a\s+|an\s+)?(?:system|admin|developer)\b"
            r"|\bnew\s+(?:system\s+)?instructions\s*:",
            re.IGNORECASE,
        ),
        3,
    ),
    (
        "exfiltration_request",
        re.compile(
            r"\b(print|reveal|disclose|output|dump|leak|show)\b[^.\n]{0,60}?"
            r"\b(canary|secret|api[\s_-]?key|password|other\s+tenant\w*|tenant\s+[a-z]\b)",
            re.IGNORECASE,
        ),
        3,
    ),
    (
        "cross_tenant_citation_request",
        re.compile(r"\bcite\b[^.\n]{0,40}\btenant\s+[a-z]\b[^.\n]{0,20}\bfor\b", re.IGNORECASE),
        2,
    ),
    (
        "canary_marker",
        re.compile("|".join(re.escape(m) for m in CANARY_MARKERS), re.IGNORECASE),
        2,
    ),
]

QUARANTINE_THRESHOLD = 3

# Characters that would let document text break out of the prompt's data frame.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FENCE_CHARS = re.compile(r"[<>\[\]{}|`]")

#: Zero-width / invisible code points an attacker can use to break up a
#: matched phrase. Listed as explicit code points so they stay reviewable
#: in a diff instead of being invisible in the source.
_ZERO_WIDTH = (
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2060",  # word joiner
    "\ufeff",  # zero width no-break space / BOM
    "\u00ad",  # soft hyphen
)

REDACTED_MARKER = "[REDACTED-MARKER]"


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Outcome of scanning a piece of untrusted text."""

    rules: tuple[str, ...] = field(default=())
    score: int = 0

    @property
    def suspicious(self) -> bool:
        return self.score > 0

    @property
    def should_quarantine(self) -> bool:
        return self.score >= QUARANTINE_THRESHOLD


def scan_text(*parts: str) -> ScanResult:
    """Scan one or more untrusted strings for forged-instruction patterns."""
    haystack = "\n".join(p for p in parts if p)
    normalised = normalise_unicode(haystack)
    hits: list[str] = []
    score = 0
    for rule_id, pattern, weight in _RULES:
        if pattern.search(normalised):
            hits.append(rule_id)
            score += weight
    return ScanResult(rules=tuple(hits), score=score)


def normalise_unicode(text: str) -> str:
    """NFKC-normalise and strip zero-width characters.

    Defeats the cheapest homoglyph / zero-width-space evasions against the
    pattern rules above. It does not claim to defeat all of them.
    """
    text = unicodedata.normalize("NFKC", text)
    for zero_width in _ZERO_WIDTH:
        text = text.replace(zero_width, "")
    return text


def neutralise_for_prompt(text: str, *, max_chars: int = 2000) -> str:
    """Render untrusted text safe(r) to embed inside a prompt data block.

    - Normalises unicode and removes control characters.
    - Replaces fence/markup characters so the text cannot terminate the data
      delimiter or forge a chat-template role token.
    - Redacts leak markers, so neither a retrieved document *nor a crafted
      question* can prime the model with the literal canary string.
    - Truncates, bounding how much attacker-controlled text reaches the model.
    """
    cleaned = normalise_unicode(text)
    cleaned = _CONTROL_CHARS.sub(" ", cleaned)
    cleaned = redact_markers(cleaned)
    cleaned = _FENCE_CHARS.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip() + " ..."
    return cleaned


def contains_canary(text: str) -> bool:
    """True if any synthetic leak marker appears in ``text``.

    Case-insensitive and applied after unicode normalisation, so lower-casing,
    full-width characters and zero-width padding do not slip a marker through.
    """
    haystack = normalise_unicode(text).upper()
    return any(marker.upper() in haystack for marker in CANARY_MARKERS)


def redact_markers(text: str) -> str:
    """Replace any leak marker with a fixed placeholder, case-insensitively."""
    normalised = normalise_unicode(text)
    for marker in CANARY_MARKERS:
        normalised = re.sub(
            re.escape(marker), REDACTED_MARKER, normalised, flags=re.IGNORECASE
        )
    return normalised


#: Secrets shorter than this are not substituted. A one- or two-character
#: "secret" (only ever seen in throwaway test configs) would otherwise match
#: inside ordinary words and shred every log line it touched.
MIN_REDACTABLE_SECRET_LEN = 8


def redact_secrets(text: str, secrets: tuple[str, ...]) -> str:
    """Replace any configured secret value with a fixed placeholder."""
    for secret in secrets:
        if len(secret) >= MIN_REDACTABLE_SECRET_LEN and secret in text:
            text = text.replace(secret, "***REDACTED***")
    return text
