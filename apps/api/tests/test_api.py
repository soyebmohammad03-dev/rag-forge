from fastapi.testclient import TestClient


def test_health_reports_components_honestly(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    states = {c["name"]: c["state"] for c in body["components"]}
    assert states["api"] == "ok"
    assert states["vector_index"] == "not_configured"
    assert states["llm_provider"] == "not_configured"


def test_experiment_rejects_unknown_corpus(client: TestClient) -> None:
    c = client
    assert c.post("/api/v1/experiments", json={"name": "x", "corpus_id": "nope"}).status_code == 422
    created = c.post("/api/v1/experiments", json={"name": "baseline"})
    assert created.status_code == 201
    assert created.json()["status"] == "draft"


def test_unimplemented_capabilities_return_501_not_fake_data(client: TestClient) -> None:
    cid = client.post("/api/v1/corpora", json={"name": "c"}).json()["corpus"]["id"]
    body = {"corpus_id": cid, "query": "q", "router": {"policy": "learned"}}
    r = client.post("/api/v1/router/decide", json=body)
    assert r.status_code == 501
    assert r.json()["detail"]["capability"] == "Router"


def test_evaluation_endpoint_measures(client: TestClient) -> None:
    r = client.post(
        "/api/v1/evaluation/retrieval",
        json={"ranked_ids": ["a", "b", "c"], "relevance": {"b": 1}, "ks": [1, 3]},
    )
    metrics = {(m["name"], m["k"]): m for m in r.json()["metrics"]}
    assert metrics[("mrr", None)]["value"] == 0.5
    assert metrics[("recall", 1)]["value"] == 0.0
    assert metrics[("recall", 3)]["value"] == 1.0
    assert all(m["origin"] == "measured" for m in metrics.values())


def test_evaluation_rejects_empty_judgements(client: TestClient) -> None:
    r = client.post("/api/v1/evaluation/retrieval", json={"ranked_ids": ["a"], "relevance": {}})
    assert r.status_code == 422


def test_environment_snapshot(client: TestClient) -> None:
    env = client.get("/api/v1/provenance/environment").json()["environment"]
    assert env["rag_forge_version"] == "0.1.0"
    assert "fastapi" in env["packages"]
