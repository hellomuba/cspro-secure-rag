"""Deterministic validation of model output, performed after generation.

The model is treated as an untrusted component. Nothing it returns is passed
through unchecked:

* **Citations are selected, never supplied.** The model can only reference the
  opaque ``[DOCn]`` labels it was given. Labels are resolved back to records
  through the backend's own label map, so a citation can only ever name a
  record that the tenant filter already authorised. Unknown labels are dropped
  and flagged. The model has no ability to *add* a source.
* **Marker tripwire.** If a synthetic canary or a configured secret appears in
  the completion, the completion is discarded entirely and replaced with a
  refusal.
* **Support check.** The answer's content words are compared against the
  authorised context; an answer that is mostly unsupported text is downgraded
  to an abstention rather than returned as fact.

Limitations are stated plainly in the README: these checks reduce the blast
radius of a successful injection, they do not prove one is impossible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import Settings
from app.prompting import ABSTAIN_TOKEN
from app.retrieval import tokenize
from app.sanitize import contains_canary, normalise_unicode
from app.schemas import AnswerStatus
from app.store import StoredRecord

_LABEL_RE = re.compile(r"\[\s*(DOC\d{1,3})\s*\]", re.IGNORECASE)

REFUSAL_TEXT = (
    "The request was blocked by an output policy check. No answer is returned "
    "because the generated text contained content that is not authorised for "
    "this tenant."
)

UNGROUNDED_TEXT = (
    "I don't have enough supporting evidence in this tenant's authorised "
    "records to answer that reliably."
)

NO_EVIDENCE_TEXT = (
    "No authorised records for this tenant match the question, so no answer is "
    "given."
)

CONFLICT_TEXT = (
    "This tenant's current records disagree with each other on this topic, so "
    "no single answer is authoritative. The conflicting evidence is listed in "
    "the sources."
)


@dataclass(slots=True)
class OutputVerdict:
    answer: str
    status: AnswerStatus
    abstained: bool
    selected: list[StoredRecord] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def verify_output(
    raw_answer: str,
    label_map: dict[str, StoredRecord],
    settings: Settings,
    *,
    question: str,
    context_text: str,
) -> OutputVerdict:
    """Validate a completion against the authorised context. Never raises."""
    flags: list[str] = []
    answer = normalise_unicode(raw_answer).strip()

    # 1. Hard tripwires -----------------------------------------------------
    if contains_canary(answer):
        flags.append("canary_in_model_output")
        return OutputVerdict(REFUSAL_TEXT, AnswerStatus.BLOCKED_POLICY_VIOLATION, True, [], flags)

    for secret in settings.secret_values:
        if secret and secret in answer:
            flags.append("credential_in_model_output")
            return OutputVerdict(
                REFUSAL_TEXT, AnswerStatus.BLOCKED_POLICY_VIOLATION, True, [], flags
            )

    # 2. Explicit abstention requested by the model -------------------------
    if ABSTAIN_TOKEN in answer.upper():
        return OutputVerdict(
            NO_EVIDENCE_TEXT, AnswerStatus.ABSTAINED_NO_EVIDENCE, True, [], flags
        )

    # 3. Citation verification ---------------------------------------------
    cited = [m.upper() for m in _LABEL_RE.findall(answer)]
    valid = [label for label in cited if label in label_map]
    invalid = [label for label in cited if label not in label_map]
    if invalid:
        flags.append("fabricated_citation_dropped")

    # Resolve labels through the backend map. Order follows the label map so a
    # repeated or reordered citation cannot change which records are emitted.
    selected_ids = {label for label in valid}
    selected = [rec for label, rec in label_map.items() if label in selected_ids]
    if not selected:
        # The model cited nothing usable; fall back to the full authorised hit
        # set rather than returning an answer with no traceable evidence.
        if cited:
            flags.append("no_valid_citation_in_output")
        selected = list(label_map.values())

    # 4. Support check ------------------------------------------------------
    support = _support_ratio(answer, context_text=context_text, question=question)
    if support is not None and support < settings.grounding_min_support:
        flags.append(f"low_context_support:{support:.2f}")
        return OutputVerdict(
            UNGROUNDED_TEXT, AnswerStatus.ABSTAINED_UNGROUNDED, True, [], flags
        )

    # 5. Rewrite labels to verified record ids ------------------------------
    rendered = _render_citations(answer, label_map)
    if not rendered.strip():
        return OutputVerdict(
            UNGROUNDED_TEXT, AnswerStatus.ABSTAINED_UNGROUNDED, True, [], flags
        )

    return OutputVerdict(rendered, AnswerStatus.ANSWERED, False, selected, flags)


def _render_citations(answer: str, label_map: dict[str, StoredRecord]) -> str:
    """Replace valid ``[DOCn]`` labels with verified record ids; drop the rest."""

    def replace(match: re.Match[str]) -> str:
        label = match.group(1).upper()
        record = label_map.get(label)
        return f"[{record.id}]" if record else ""

    return re.sub(r"\s{2,}", " ", _LABEL_RE.sub(replace, answer)).strip()


def _support_ratio(answer: str, *, context_text: str, question: str) -> float | None:
    """Fraction of the answer's content words that appear in the context.

    Returns ``None`` when the answer is too short for the ratio to mean
    anything, so a terse-but-correct answer is not punished.
    """
    answer_terms = [t for t in tokenize(_LABEL_RE.sub(" ", answer)) if len(t) >= 4]
    if len(answer_terms) < 5:
        return None
    allowed = set(tokenize(context_text)) | set(tokenize(question))
    supported = sum(1 for t in answer_terms if t in allowed)
    return supported / len(answer_terms)


def context_text_for(label_map: dict[str, StoredRecord]) -> str:
    """Flatten the authorised context for the support check."""
    return "\n".join(f"{r.title} {r.topic} {r.body}" for r in label_map.values())
