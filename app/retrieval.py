"""Deterministic lexical retrieval (Okapi BM25), implemented in pure Python.

Why BM25 by hand rather than scikit-learn / a vector DB:

* **Determinism.** The same query over the same corpus always produces the same
  ranking and the same scores, which is what makes the security tests
  meaningful assertions rather than flaky approximations.
* **Auditability.** Every number in a citation can be traced to a formula on
  this page during a live code walkthrough.
* **Footprint.** The assessment corpus is nine records; a 120-line index beats
  a 400 MB dependency tree.

The index is *always* built from a candidate list that the caller has already
tenant-filtered (see :meth:`app.store.RecordStore.list_for_tenant`). Ranking
never sees another tenant's text.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.store import StoredRecord

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately small: enough to stop function words dominating idf mass on a
# nine-record corpus, small enough to read in full during a walkthrough.
_STOPWORDS = frozenset(
    """a an and are as at be but by for from had has have how i if in into is it its
    may must no not of on or our shall should so than that the their them then there
    these they this to was we were what when where which who why will with you your
    do does did can could would about please tell me my""".split()
)

K1 = 1.5
B = 0.75


@dataclass(frozen=True, slots=True)
class ScoredRecord:
    record: StoredRecord
    score: float
    matched_terms: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    """Lower-case, split on non-alphanumerics, drop stopwords, naive de-plural."""
    tokens = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in _STOPWORDS or len(raw) < 2:
            continue
        # Crude but deterministic stemming: "policies" -> "policie" -> "policy"
        # is not attempted; we only fold a trailing plural "s".
        if len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
            raw = raw[:-1]
        tokens.append(raw)
    return tokens


class BM25Index:
    """An Okapi BM25 index over an already-authorized set of records."""

    def __init__(self, records: list[StoredRecord], *, k1: float = K1, b: float = B) -> None:
        self._records = records
        self._k1 = k1
        self._b = b
        # Title text is weighted by repeating it: titles are short and highly
        # discriminative on this corpus.
        self._docs: list[Counter[str]] = [
            Counter(tokenize(r.title) * 2 + tokenize(r.topic) * 2 + tokenize(r.body))
            for r in records
        ]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._doc_freq: Counter[str] = Counter()
        for doc in self._docs:
            self._doc_freq.update(doc.keys())
        self._n = len(records)

    def idf(self, term: str) -> float:
        """Robertson/Sparck-Jones idf with +1 smoothing so it stays positive."""
        df = self._doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def query_idf(self, term: str) -> float:
        """idf used for *normalisation*, including terms absent from the corpus.

        A query word that occurs nowhere still represents informative content
        the corpus failed to cover, so it is charged the maximum idf. Without
        this, a question made almost entirely of unknown words (``"what is the
        canary value for tenant B"``) would score ~1.0 on the strength of one
        near-universal term like ``tenant`` and defeat the abstention
        threshold.
        """
        df = self._doc_freq.get(term, 0)
        if df == 0:
            return math.log(1.0 + (self._n + 0.5) / 0.5)
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[ScoredRecord]:
        """Rank the authorized corpus against ``query``.

        Scores are normalised by the total idf mass of the query terms that
        exist in this corpus, so a score is roughly "what fraction of the
        question's informative content this document covers". That makes the
        abstention threshold in ``RETRIEVAL_MIN_SCORE`` portable across
        questions of different lengths.
        """
        if not self._records:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        idf_mass = sum(self.query_idf(t) for t in set(query_terms))
        if idf_mass <= 0.0:  # pragma: no cover - query_idf is always positive
            return []

        results: list[ScoredRecord] = []
        for record, doc, length in zip(self._records, self._docs, self._lengths, strict=True):
            raw = 0.0
            matched: list[str] = []
            for term in set(query_terms):
                freq = doc.get(term, 0)
                if freq == 0:
                    continue
                matched.append(term)
                denom = freq + self._k1 * (
                    1 - self._b + self._b * (length / self._avg_len if self._avg_len else 1.0)
                )
                raw += self.idf(term) * (freq * (self._k1 + 1)) / denom
            if raw <= 0.0:
                continue
            results.append(
                ScoredRecord(
                    record=record,
                    score=round(raw / idf_mass, 6),
                    matched_terms=tuple(sorted(matched)),
                )
            )

        # Deterministic tie-break: score desc, then record id asc.
        results.sort(key=lambda s: (-s.score, s.record.id))
        return results[:top_k]


def retrieve(
    records: list[StoredRecord],
    query: str,
    *,
    top_k: int,
    min_score: float,
) -> list[ScoredRecord]:
    """Build an index over the authorized records and return the hits above
    ``min_score``. An empty result is the signal to abstain."""
    index = BM25Index(records)
    return [hit for hit in index.search(query, top_k) if hit.score >= min_score]
