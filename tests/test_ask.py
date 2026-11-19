"""POST /ask -- retrieval relevance, grounding, citations, abstention, conflicts."""

from __future__ import annotations

import pytest

from app.schemas import AnswerStatus
from tests.conftest import FakeLLM

CURRENT_POLICY_Q = "What is the current import retry policy?"
RETENTION_Q = "What is the retention policy for demo records?"


def ask(client, headers, question, **extra):
    return client.post("/ask", json={"question": question, **extra}, headers=headers)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {"question": "x"},
        {"question": ""},
        {"question": "a" * 2001},
        {"question": "valid question", "top_k": 0},
        {"question": "valid question", "top_k": 11},
        {"question": "valid question", "top_k": "three"},
        {"top_k": 3},
        {},
    ],
)
def test_invalid_ask_payload_is_422(client, auth_a, payload):
    assert client.post("/ask", json=payload, headers=auth_a).status_code == 422


# --------------------------------------------------------------------------
# Grounded answers and citations
# --------------------------------------------------------------------------
def test_answer_cites_only_verified_backend_records(make_client, auth_a):
    llm = FakeLLM(["The current policy requires idempotency keys for retries [DOC1]."])
    with make_client(llm) as client:
        body = ask(client, auth_a, CURRENT_POLICY_Q).json()

    assert body["status"] == AnswerStatus.ANSWERED
    assert body["abstained"] is False
    assert [s["id"] for s in body["sources"]] == ["a-policy-2"]
    assert all(s["tenant_id"] == "tenant-a" for s in body["sources"])
    # The opaque DOC label is rewritten to the verified record id.
    assert "[a-policy-2]" in body["answer"]
    assert "DOC1" not in body["answer"]


def test_every_citation_traces_to_a_real_authorized_record(
    make_client, auth_a, synthetic_records
):
    """A cited id must exist in the corpus *and* belong to the caller's tenant."""
    tenant_a_ids = {r["id"] for r in synthetic_records if r["tenant_id"] == "tenant-a"}
    llm = FakeLLM(["Idempotency keys are required for import retries [DOC1][DOC2]."])
    with make_client(llm) as client:
        body = ask(client, auth_a, CURRENT_POLICY_Q).json()

    assert body["sources"], "an answered response must cite something"
    for source in body["sources"]:
        assert source["id"] in tenant_a_ids
        assert source["tenant_id"] == "tenant-a"
        assert source["excerpt"].strip(), "every citation carries an excerpt"


def test_sources_carry_provenance_metadata(make_client, auth_a):
    with make_client(FakeLLM(["Use idempotency keys [DOC1]."])) as client:
        source = ask(client, auth_a, CURRENT_POLICY_Q).json()["sources"][0]
    assert source["title"] == "Import policy v2"
    assert source["topic"] == "imports"
    assert source["version"] == 2
    assert source["is_current"] is True
    assert source["score"] > 0


def test_relevant_document_outranks_unrelated_ones(make_client, auth_a):
    llm = FakeLLM(["Retention is thirty days for demo records [DOC1]."])
    with make_client(llm) as client:
        body = ask(client, auth_a, RETENTION_Q, top_k=5).json()
    assert body["sources"][0]["id"] == "a-policy-retain"


def test_response_states_where_inference_ran(make_client, auth_a):
    with make_client(FakeLLM(["Use idempotency keys [DOC1]."])) as client:
        body = ask(client, auth_a, CURRENT_POLICY_Q).json()
    assert body["provider"] == "fake"
    assert body["execution"] in {"local", "remote"}
    assert body["model"] == "fake-model"


def test_audit_id_is_returned_and_unique(make_client, auth_a):
    with make_client(FakeLLM(["a [DOC1]", "b [DOC1]"])) as client:
        first = ask(client, auth_a, CURRENT_POLICY_Q).json()["audit_id"]
        second = ask(client, auth_a, CURRENT_POLICY_Q).json()["audit_id"]
    assert first.startswith("aud_") and second.startswith("aud_")
    assert first != second


# --------------------------------------------------------------------------
# Abstention
# --------------------------------------------------------------------------
def test_no_matching_records_abstains_without_calling_the_model(make_client, auth_a):
    llm = FakeLLM(["this answer must never be produced"])
    with make_client(llm) as client:
        body = ask(client, auth_a, "How do I bake sourdough bread at high altitude?").json()

    assert body["abstained"] is True
    assert body["status"] == AnswerStatus.ABSTAINED_NO_EVIDENCE
    assert body["sources"] == []
    assert body["execution"] == "none"
    assert llm.calls == [], "no evidence means no inference at all"


def test_model_abstention_is_reported_separately_from_a_retrieval_miss(
    make_client, auth_a
):
    """A weak model refusing usable evidence must not look like an empty corpus.

    Both are abstentions, but "no record matched" and "the model would not use
    the records it was given" have opposite fixes, so they get distinct
    statuses. Qwen3 0.6B does exactly this in practice.
    """
    with make_client(FakeLLM(["INSUFFICIENT_EVIDENCE"])) as client:
        body = ask(client, auth_a, CURRENT_POLICY_Q).json()

    assert body["abstained"] is True
    assert body["status"] == AnswerStatus.ABSTAINED_MODEL_DECLINED
    assert body["status"] != AnswerStatus.ABSTAINED_NO_EVIDENCE
    assert body["execution"] == "local", "the model really was called"
    # The evidence the model was shown is returned, so a wrong refusal is visible.
    assert {s["id"] for s in body["sources"]} == {"a-policy-2", "a-ticket-1"}
    assert all(s["tenant_id"] == "tenant-a" for s in body["sources"])


def test_empty_retrieval_is_a_different_status_and_skips_the_model(make_client, auth_a):
    llm = FakeLLM(["never reached"])
    with make_client(llm) as client:
        body = ask(client, auth_a, "How do I bake sourdough bread at high altitude?").json()

    assert body["status"] == AnswerStatus.ABSTAINED_NO_EVIDENCE
    assert body["execution"] == "none"
    assert body["sources"] == []
    assert llm.calls == []


def test_unsupported_answer_is_downgraded_to_abstention(make_client, auth_a):
    """An answer made of words absent from the context is not returned as fact."""
    hallucination = (
        "Quarterly dividends settle through the Luxembourg custodian before "
        "escrow reconciliation completes."
    )
    with make_client(FakeLLM([hallucination])) as client:
        body = ask(client, auth_a, CURRENT_POLICY_Q).json()

    assert body["abstained"] is True
    assert body["status"] == AnswerStatus.ABSTAINED_UNGROUNDED
    assert body["sources"] == []
    assert any(f.startswith("low_context_support") for f in body["security_flags"])


# --------------------------------------------------------------------------
# Supersession and conflict
# --------------------------------------------------------------------------
def test_superseded_policy_is_excluded_and_reported(make_client, auth_a):
    llm = FakeLLM(["Retries require idempotency keys [DOC1]."])
    with make_client(llm) as client:
        body = ask(client, auth_a, CURRENT_POLICY_Q, top_k=5).json()

    superseded_ids = {s["id"] for s in body["superseded_excluded"]}
    assert "a-policy-1" in superseded_ids, "v1 is flagged, not silently used"
    assert body["superseded_excluded"][0]["superseded_by"] == "a-policy-2"
    assert "a-policy-1" not in {s["id"] for s in body["sources"]}
    # The superseded record's *text* must not reach the model either.
    assert "older nightly import policy" not in llm.last_user_prompt
    assert "idempotency keys" in llm.last_user_prompt, "the current policy does"


def test_current_version_wins(make_client, auth_a):
    with make_client(FakeLLM(["Idempotency keys are required [DOC1]."])) as client:
        body = ask(client, auth_a, CURRENT_POLICY_Q, top_k=5).json()
    assert body["sources"][0]["id"] == "a-policy-2"
    assert body["sources"][0]["version"] == 2


def test_contradictory_current_policies_flag_conflict_and_abstain(make_client, auth_b):
    """Tenant B holds two *current* retention records claiming 7 and 45 days."""
    llm = FakeLLM(["Records are kept for 7 days [DOC1]."])
    with make_client(llm) as client:
        body = ask(client, auth_b, RETENTION_Q, top_k=5).json()

    assert body["conflict"] is True
    assert body["abstained"] is True
    assert body["status"] == AnswerStatus.ABSTAINED_CONFLICT
    assert llm.calls == [], "a conflict is resolved in code, not by asking the model"
    cited = {s["id"] for s in body["sources"]}
    assert cited == {"b-policy", "b-contradiction"}, "conflicting evidence is supplied"
    assert any(f.startswith("policy_conflict:retention") for f in body["security_flags"])


def test_when_every_match_is_superseded_the_system_abstains(make_client, auth_a):
    """Outdated evidence is not a weaker answer -- it is no answer."""
    llm = FakeLLM(["this must never be produced"])
    with make_client(llm) as client:
        client.post(
            "/ingest",
            json={
                "records": [
                    {
                        "id": "a-widget-old",
                        "title": "Widget calibration policy",
                        "body": "Fictional Tenant A: widget calibration ran nightly.",
                        "topic": "widgets",
                        "version": 1,
                        "is_current": False,
                    }
                ]
            },
            headers=auth_a,
        )
        body = ask(client, auth_a, "What is the widget calibration policy?", top_k=5).json()

    assert body["abstained"] is True
    assert body["status"] == AnswerStatus.ABSTAINED_NO_EVIDENCE
    assert body["sources"] == []
    assert [s["id"] for s in body["superseded_excluded"]] == ["a-widget-old"]
    assert llm.calls == [], "no authoritative evidence means no inference"


def test_long_record_excerpts_are_windowed_around_the_match(make_client, auth_a):
    filler = "Background narrative that is not relevant to the question. " * 12
    body_text = (
        f"{filler}Fictional Tenant A: sprocket inspections require two signatures. {filler}"
    )
    with make_client(FakeLLM(["Two signatures are required [DOC1]."])) as client:
        client.post(
            "/ingest",
            json={
                "records": [
                    {
                        "id": "a-sprocket",
                        "title": "Sprocket inspection policy",
                        "body": body_text,
                        "topic": "sprockets",
                    }
                ]
            },
            headers=auth_a,
        )
        body = ask(client, auth_a, "What do sprocket inspections require?", top_k=5).json()

    excerpt = body["sources"][0]["excerpt"]
    assert len(excerpt) < len(body_text), "the excerpt is a window, not the whole record"
    assert excerpt.startswith("... ") and excerpt.endswith(" ...")


def test_no_false_conflict_for_a_single_current_policy(make_client, auth_a):
    with make_client(FakeLLM(["Thirty days [DOC1]."])) as client:
        body = ask(client, auth_a, RETENTION_Q, top_k=5).json()
    assert body["conflict"] is False


# --------------------------------------------------------------------------
# top_k behaviour
# --------------------------------------------------------------------------
def test_top_k_bounds_the_number_of_sources(make_client, auth_a):
    with make_client(FakeLLM(["Policy text [DOC1]."])) as client:
        body = ask(client, auth_a, CURRENT_POLICY_Q, top_k=1).json()
    assert len(body["sources"]) <= 1
