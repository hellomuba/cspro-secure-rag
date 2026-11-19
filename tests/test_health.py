"""GET /health -- API liveness must be reported separately from model readiness."""

from __future__ import annotations

from tests.conftest import FakeLLM


def test_health_reports_ok_when_model_reachable(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["api"] == "ok"
    assert body["llm"]["reachable"] is True
    assert body["llm"]["model"] == "fake-model"


def test_health_reports_record_counts(client):
    body = client.get("/health").json()
    # The nine synthetic records are seeded on first start-up.
    assert body["records"]["total"] == 9
    assert body["records"]["tenant-a"] == 5
    assert body["records"]["tenant-b"] == 4


def test_health_degrades_without_crashing_when_model_unavailable(make_client):
    with make_client(FakeLLM(reachable=False)) as client:
        response = client.get("/health")

    assert response.status_code == 200, "the API itself is still alive"
    body = response.json()
    assert body["status"] == "degraded"
    assert body["api"] == "ok"
    assert body["llm"]["reachable"] is False
    assert body["llm"]["model_available"] is False
    assert body["llm"]["detail"]


def test_health_never_claims_availability_without_probing(make_client):
    """`model_available` must come from an actual probe, not a default."""
    probe = FakeLLM(reachable=False)
    with make_client(probe) as client:
        assert client.get("/health").json()["llm"]["model_available"] is False
    probe.reachable = True
    with make_client(probe) as client:
        assert client.get("/health").json()["llm"]["model_available"] is True


def test_health_requires_no_credentials(client):
    assert client.get("/health").status_code == 200


def test_health_reports_execution_location(client):
    """A response must state where inference runs, never imply it."""
    assert client.get("/health").json()["llm"]["execution"] in {"local", "remote"}
