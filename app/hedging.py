"""Removal of self-contradictory trailing hedges from grounded answers.

The problem this solves, observed with a real `qwen3:0.6b` run:

    "The current import retry policy requires idempotency keys ([a-policy-2])
     and avoids duplicating candidate records ([a-ticket-1]). Insufficient
     information to determine the exact policy details."

The first two clauses are correct and cited. The third contradicts them, cites
nothing, and tells the reader nothing. Small instruct models append this kind
of filler routinely.

The rule applied here is deliberately narrow, because over-reaching would be
worse than the hedge:

* Only **trailing** sentences are considered. A caveat in the middle of an
  answer is part of the argument and is left alone.
* Only sentences that make a **global claim of insufficiency** qualify --
  "insufficient information", "unable to determine the exact details". A
  *specific* limitation such as "the records do not give a retention period"
  is real information and is kept.
* A hedge is only removed when the answer **is otherwise grounded**: it must
  carry at least one verified citation and at least one surviving sentence.
  An answer that is nothing but a hedge is an abstention, not an answer, and
  is reported as one.
* Nothing is removed silently. Every removed sentence is returned to the
  caller and surfaced in `quality_notes` on the response.
"""

from __future__ import annotations

import re

#: Split on sentence terminators, keeping it simple and deterministic.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: A blanket assertion that the available information is inadequate.
_GLOBAL_INSUFFICIENCY = re.compile(
    r"\b(?:insufficient|inadequate|not\s+enough|insufficiently)\s+"
    r"(?:information|evidence|data|detail|details|context|records?)\b"
    r"|\bno\s+(?:further|additional|more|other|sufficient|enough)\s+"
    r"(?:information|evidence|data|detail|details|context)\b"
    r"|\binsufficient_evidence\b",
    re.IGNORECASE,
)

#: A generic inability to produce detail that was not asked for.
_VAGUE_INABILITY = re.compile(
    r"\b(?:unable|cannot|can't|could\s+not|couldn't|not\s+possible)\b"
    r"[^.!?]{0,60}?"
    r"\b(?:determine|provide|confirm|establish|specify|ascertain)\b"
    r"[^.!?]{0,60}?"
    r"\b(?:exact|precise|specific|further|additional|full|complete|detailed)\b",
    re.IGNORECASE,
)

#: Citation shapes that make a sentence load-bearing rather than filler.
_CITATION = re.compile(r"\[[^\]]+\]")

#: Tokens that mean the sentence carries actual content.
_WORD = re.compile(r"[A-Za-z0-9]+")


def is_contradictory_hedge(sentence: str) -> bool:
    """True when a sentence is a blanket insufficiency claim and nothing more."""
    if _CITATION.search(sentence):
        # It points at evidence, so it is part of the answer.
        return False
    return bool(_GLOBAL_INSUFFICIENCY.search(sentence) or _VAGUE_INABILITY.search(sentence))


def split_sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENTENCE_SPLIT.split(text.strip())) if s]


def has_substance(text: str) -> bool:
    """Does anything remain that could be read as an answer?"""
    return len(_WORD.findall(text)) >= 3


def strip_trailing_hedges(answer: str, *, grounded: bool) -> tuple[str, list[str]]:
    """Remove self-contradictory trailing hedges from an otherwise good answer.

    ``grounded`` must be True only when the answer carries at least one
    verified citation. Returns ``(cleaned_answer, removed_sentences)``.

    When the answer consists *entirely* of hedging, nothing is removed -- the
    caller is expected to treat that as an abstention rather than an answer, so
    silently deleting the text would destroy the signal.
    """
    sentences = split_sentences(answer)
    if not grounded or len(sentences) < 2:
        return answer, []

    removed: list[str] = []
    while len(sentences) > 1 and is_contradictory_hedge(sentences[-1]):
        removed.append(sentences.pop())

    if not removed:
        return answer, []

    cleaned = " ".join(sentences).strip()
    if not has_substance(cleaned):
        # Refuse to turn an abstention into an answer by deletion.
        return answer, []

    removed.reverse()
    return cleaned, removed


#: The abstain token wherever it appears, including trailing punctuation.
_ABSTAIN_INLINE = re.compile(
    r"\bINSUFFICIENT[_\s]?EVIDENCE\b\s*[.!?]?",
    re.IGNORECASE,
)


def remove_abstain_token(text: str) -> tuple[str, bool]:
    """Strip the abstain token from a completion. Returns (text, changed)."""
    cleaned = _ABSTAIN_INLINE.sub(" ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, cleaned != text.strip()


def has_answer_content(text: str) -> bool:
    """Real words outside citation brackets.

    ``"[DOC1]"`` is a citation with nothing attached, not an answer. Used to
    decide whether anything survives once the abstain token is removed.
    """
    return len(_WORD.findall(_CITATION.sub(" ", text))) >= 2
