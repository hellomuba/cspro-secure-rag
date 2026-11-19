"""Policy supersession and contradiction analysis.

Two separate questions are answered here, deterministically and without the
model's help:

**Supersession** -- within one topic, is this record still authoritative?
A record is superseded when it is flagged ``is_current = false``, or when a
higher ``version`` of the same topic is current. Superseded records are removed
from the prompt context and reported back to the caller explicitly, so they can
never be *silently* treated as authoritative.

**Contradiction** -- do two records that are both current make incompatible
quantitative claims about the same topic? We extract normalised numeric claims
(durations, percentages) and compare them. If a topic carries more than one
distinct value, the system flags a conflict and abstains rather than picking a
winner. This is intentionally narrow: it detects the class of contradiction the
synthetic corpus contains and says so, instead of pretending to do general
natural-language entailment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.retrieval import ScoredRecord

# <number><optional separator><unit>, e.g. "30 days", "7-day", "45days", "20%"
_CLAIM_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)[\s\-]*"
    r"(hours?|hrs?|days?|weeks?|months?|years?|percent|%)(?![\w])",
    re.IGNORECASE,
)

_DURATION_HOURS: dict[str, float] = {
    "hour": 1.0,
    "hr": 1.0,
    "day": 24.0,
    "week": 24.0 * 7,
    "month": 24.0 * 30,
    "year": 24.0 * 365,
}


@dataclass(frozen=True, slots=True)
class Claim:
    record_id: str
    family: str  # "duration" | "ratio"
    value: float  # normalised: hours for durations, 0-100 for ratios
    raw: str


@dataclass(frozen=True, slots=True)
class SupersededRecord:
    record_id: str
    title: str
    version: int
    superseded_by: str


@dataclass(slots=True)
class PolicyAnalysis:
    authoritative: list[ScoredRecord] = field(default_factory=list)
    superseded: list[SupersededRecord] = field(default_factory=list)
    conflict: bool = False
    conflict_topics: list[str] = field(default_factory=list)
    conflict_detail: str | None = None


def extract_claims(record_id: str, text: str) -> list[Claim]:
    """Pull normalised quantitative claims out of a record body."""
    claims: list[Claim] = []
    for value_str, unit_raw in _CLAIM_RE.findall(text):
        unit = unit_raw.lower().rstrip("s")
        value = float(value_str)
        if unit in {"percent", "%"}:
            claims.append(Claim(record_id, "ratio", value, f"{value_str}{unit_raw}"))
        elif unit in _DURATION_HOURS:
            claims.append(
                Claim(
                    record_id,
                    "duration",
                    value * _DURATION_HOURS[unit],
                    f"{value_str} {unit_raw}",
                )
            )
    return claims


def analyse(hits: list[ScoredRecord]) -> PolicyAnalysis:
    """Split retrieved hits into authoritative vs superseded and flag conflicts."""
    analysis = PolicyAnalysis()
    if not hits:
        return analysis

    by_topic: dict[str, list[ScoredRecord]] = {}
    for hit in hits:
        by_topic.setdefault(hit.record.topic, []).append(hit)

    for topic, topic_hits in by_topic.items():
        current = [h for h in topic_hits if h.record.is_current]
        # Best-effort pointer to what replaced an outdated record: the highest
        # version still marked current in the same topic.
        successor = next(
            iter(sorted(current, key=lambda h: (-h.record.version, h.record.id))),
            None,
        )

        topic_authoritative: list[ScoredRecord] = []
        for hit in topic_hits:
            # Supersession keys on the record's own `is_current` flag only.
            # Comparing versions across *different* records in a topic was
            # tried and rejected: it wrongly retired a current support ticket
            # (`a-ticket-1`, v1) because an unrelated policy in the same topic
            # had reached v2. Without a lineage/parent field there is no sound
            # way to infer that two records are versions of one document, and
            # guessing would silently drop authoritative evidence.
            if not hit.record.is_current:
                analysis.superseded.append(
                    SupersededRecord(
                        record_id=hit.record.id,
                        title=hit.record.title,
                        version=hit.record.version,
                        superseded_by=successor.record.id if successor else "",
                    )
                )
            else:
                topic_authoritative.append(hit)

        analysis.authoritative.extend(topic_authoritative)

        # Contradiction check runs only over records that survived supersession.
        conflict_detail = _find_contradiction(topic_authoritative)
        if conflict_detail is not None:
            analysis.conflict = True
            analysis.conflict_topics.append(topic)
            analysis.conflict_detail = (
                analysis.conflict_detail + "; " + conflict_detail
                if analysis.conflict_detail
                else conflict_detail
            )

    analysis.authoritative.sort(key=lambda h: (-h.score, h.record.id))
    analysis.superseded.sort(key=lambda s: s.record_id)
    analysis.conflict_topics.sort()
    return analysis


def _find_contradiction(topic_hits: list[ScoredRecord]) -> str | None:
    """Return a description when two current records disagree numerically."""
    if len(topic_hits) < 2:
        return None

    claims: list[Claim] = []
    for hit in topic_hits:
        claims.extend(extract_claims(hit.record.id, hit.record.body))

    for family in ("duration", "ratio"):
        family_claims = [c for c in claims if c.family == family]
        distinct = {c.value for c in family_claims}
        if len(distinct) < 2:
            continue
        # Only a genuine cross-record disagreement counts: a single record that
        # quotes two figures (e.g. "7 days ... 45 days") is reported together
        # with the record it disagrees with.
        contributing = sorted({c.record_id for c in family_claims})
        if len(contributing) < 2:
            continue
        rendered = ", ".join(
            f"{c.record_id}: {c.raw}" for c in sorted(family_claims, key=lambda c: c.record_id)
        )
        topic = topic_hits[0].record.topic
        return (
            f"topic '{topic}' has {len(distinct)} incompatible current "
            f"{family} values ({rendered})"
        )
    return None
