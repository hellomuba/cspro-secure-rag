"""Response quality: removing self-contradictory hedges without losing real doubt.

The behaviour under test comes from an actual `qwen3:0.6b` completion:

    "The current import retry policy requires idempotency keys ([a-policy-2])
     and avoids duplicating candidate records ([a-ticket-1]). Insufficient
     information to determine the exact policy details."

The first sentence is correct and cited; the second contradicts it and cites
nothing. The hard part is removing the second without ever removing a *genuine*
limitation, so the preservation tests below matter at least as much as the
removal ones.
"""

from __future__ import annotations

import pytest

from app.hedging import is_contradictory_hedge, strip_trailing_hedges
from app.schemas import AnswerStatus
from tests.conftest import FakeLLM

CURRENT_POLICY_Q = "What is the current import retry policy?"


def ask(client, headers, question=CURRENT_POLICY_Q, **extra):
    return client.post("/ask", json={"question": question, **extra}, headers=headers)


# --------------------------------------------------------------------------
# Unit: classification
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sentence",
    [
        "Insufficient information to determine the exact policy details.",
        "There is insufficient evidence to say more.",
        "Not enough information is available.",
        "No further details are provided.",
        "I am unable to determine the precise retention period.",
        "It is not possible to confirm the specific timeout.",
        "INSUFFICIENT_EVIDENCE",
    ],
)
def test_blanket_insufficiency_claims_are_hedges(sentence):
    assert is_contradictory_hedge(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        # Specific, informative limitations -- these are real answers.
        "The records do not give a deadline for this step.",
        "The policy does not state how long keys remain valid.",
        "Retention is 30 days, but the records do not cover archived tickets.",
        "Two current records disagree on the retention period.",
        # Ordinary answer content.
        "The current import retry policy requires idempotency keys.",
        "Retrying a failed batch must not duplicate candidate records.",
        # A hedge that cites evidence is load-bearing, not filler.
        "Further detail is not available in [a-policy-2].",
    ],
)
def test_specific_limitations_and_answers_are_not_hedges(sentence):
    assert not is_contradictory_hedge(sentence)


# --------------------------------------------------------------------------
# Unit: stripping
# --------------------------------------------------------------------------
def test_the_observed_qwen3_answer_is_cleaned():
    answer = (
        "The current import retry policy requires idempotency keys ([a-policy-2]) "
        "and avoids duplicating candidate records ([a-ticket-1]). "
        "Insufficient information to determine the exact policy details."
    )
    cleaned, removed = strip_trailing_hedges(answer, grounded=True)

    assert cleaned.endswith("([a-ticket-1]).")
    assert "Insufficient information" not in cleaned
    assert removed == ["Insufficient information to determine the exact policy details."]


def test_multiple_trailing_hedges_are_all_removed():
    answer = (
        "Retention is 30 days [DOC1]. "
        "Insufficient information to determine more. "
        "I cannot provide further specific details."
    )
    cleaned, removed = strip_trailing_hedges(answer, grounded=True)
    assert cleaned == "Retention is 30 days [DOC1]."
    assert len(removed) == 2


def test_a_hedge_in_the_middle_is_left_alone():
    """Only trailing filler is removed; mid-answer caveats are part of the argument."""
    answer = (
        "Retention is 30 days [DOC1]. "
        "There is insufficient information about archives. "
        "Exports additionally require approval [DOC2]."
    )
    cleaned, removed = strip_trailing_hedges(answer, grounded=True)
    assert cleaned == answer
    assert removed == []


def test_nothing_is_stripped_from_an_ungrounded_answer():
    """Without a verified citation the answer is not trustworthy enough to tidy."""
    answer = "Retention might be 30 days. Insufficient information to confirm."
    cleaned, removed = strip_trailing_hedges(answer, grounded=False)
    assert cleaned == answer and removed == []


def test_an_answer_that_is_only_hedging_is_never_emptied():
    """Deleting the whole completion would turn a refusal into a fake answer."""
    answer = "Insufficient information to determine the policy."
    cleaned, removed = strip_trailing_hedges(answer, grounded=True)
    assert cleaned == answer and removed == []


def test_specific_limitation_survives_stripping():
    answer = "Retention is 30 days [DOC1]. The records do not cover archived tickets."
    cleaned, removed = strip_trailing_hedges(answer, grounded=True)
    assert cleaned == answer and removed == []


# --------------------------------------------------------------------------
# API level
# --------------------------------------------------------------------------
def test_grounded_answer_with_trailing_hedge_is_answered_and_cleaned(make_client, auth_a):
    llm = FakeLLM(
        [
            "The current import retry policy requires idempotency keys [DOC1]. "
            "Insufficient information to determine the exact policy details."
        ]
    )
    with make_client(llm) as client:
        body = ask(client, auth_a).json()

    assert body["status"] == AnswerStatus.ANSWERED
    assert body["abstained"] is False
    assert "Insufficient information" not in body["answer"]
    assert "[a-policy-2]" in body["answer"], "the real citation survives"
    assert {s["id"] for s in body["sources"]} == {"a-policy-2"}
    # The edit is reported, never silent.
    assert body["quality_notes"], "the removed text is disclosed"
    assert "Insufficient information" in body["quality_notes"][0]
    assert "contradictory_hedge_removed" in body["security_flags"]


def test_trailing_abstain_token_no_longer_discards_a_good_answer(make_client, auth_a):
    """Regression: the token stapled to a cited answer caused a full abstention."""
    llm = FakeLLM(["Idempotency keys are required [DOC1]. INSUFFICIENT_EVIDENCE"])
    with make_client(llm) as client:
        body = ask(client, auth_a).json()

    assert body["status"] == AnswerStatus.ANSWERED
    assert "INSUFFICIENT_EVIDENCE" not in body["answer"]
    assert {s["id"] for s in body["sources"]} == {"a-policy-2"}
    assert body["quality_notes"]


def test_citation_plus_refusal_in_one_sentence_is_an_abstention(make_client, auth_a):
    """Regression from a live qwen3 run: `[DOC1] INSUFFICIENT_EVIDENCE.`

    Sentence-level stripping cannot see this -- the token shares a sentence
    with the citation -- and it was being returned as an answer whose entire
    content was a refusal.
    """
    with make_client(FakeLLM(["[DOC1] INSUFFICIENT_EVIDENCE."])) as client:
        body = ask(client, auth_a).json()

    assert body["abstained"] is True
    assert body["status"] == AnswerStatus.ABSTAINED_MODEL_DECLINED
    assert "INSUFFICIENT_EVIDENCE" not in body["answer"]
    assert body["sources"], "the evidence the model was shown is still reported"


def test_inline_token_beside_real_content_is_stripped_not_discarded(make_client, auth_a):
    """The mirror case: there IS an answer, so keep it and drop the token."""
    llm = FakeLLM(["Idempotency keys are required [DOC1] INSUFFICIENT_EVIDENCE"])
    with make_client(llm) as client:
        body = ask(client, auth_a).json()

    assert body["status"] == AnswerStatus.ANSWERED
    assert "INSUFFICIENT_EVIDENCE" not in body["answer"]
    assert "idempotency keys are required" in body["answer"].lower()
    assert body["quality_notes"]


def test_a_real_abstention_still_abstains(make_client, auth_a):
    with make_client(FakeLLM(["INSUFFICIENT_EVIDENCE"])) as client:
        body = ask(client, auth_a).json()
    assert body["status"] == AnswerStatus.ABSTAINED_MODEL_DECLINED
    assert body["abstained"] is True
    assert body["quality_notes"] == []


def test_a_prose_only_refusal_still_abstains(make_client, auth_a):
    """No citation, nothing but hedging -- that is a refusal, not an answer."""
    with make_client(FakeLLM(["Insufficient information to answer this question."])) as client:
        body = ask(client, auth_a).json()
    assert body["abstained"] is True
    assert body["status"] == AnswerStatus.ABSTAINED_MODEL_DECLINED


def test_genuine_partial_uncertainty_is_preserved_end_to_end(make_client, auth_a):
    """A specific limitation is information. It must reach the caller intact."""
    llm = FakeLLM(
        [
            "Import retries require idempotency keys [DOC1]. "
            "The records do not give a deadline for completing the retry."
        ]
    )
    with make_client(llm) as client:
        body = ask(client, auth_a).json()

    assert body["status"] == AnswerStatus.ANSWERED
    assert "do not give a deadline" in body["answer"], "real uncertainty survives"
    assert body["quality_notes"] == []
    assert "contradictory_hedge_removed" not in body["security_flags"]


def test_hedge_removal_cannot_introduce_a_foreign_citation(make_client, auth_a):
    """Tidying output must not touch which records are cited."""
    llm = FakeLLM(["Keys are required [DOC1]. Insufficient information to say more."])
    with make_client(llm) as client:
        body = ask(client, auth_a).json()
    assert all(s["tenant_id"] == "tenant-a" for s in body["sources"])
    assert all(s["id"].startswith("a-") for s in body["sources"])


def test_hedge_removal_runs_after_the_canary_tripwire(make_client, auth_a):
    """Security checks must not be bypassable by appending a hedge."""
    from tests.conftest import CANARY

    llm = FakeLLM([f"The marker is {CANARY} [DOC1]. Insufficient information to say more."])
    with make_client(llm) as client:
        response = ask(client, auth_a)
    assert CANARY not in response.text
    assert response.json()["status"] == AnswerStatus.BLOCKED_POLICY_VIOLATION
