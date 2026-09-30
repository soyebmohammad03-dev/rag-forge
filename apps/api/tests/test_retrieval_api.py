from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import make_pdf


def test_retrieve_end_to_end(client: TestClient) -> None:
    cid = client.post("/api/v1/corpora", json={"name": "api"}).json()["corpus"]["id"]
    client.post(
        f"/api/v1/corpora/{cid}/documents",
        files=[
            (
                "files",
                (
                    "bm25.md",
                    b"# BM25\n\nBM25 ranks by term frequency and inverse document frequency.",
                ),
            ),
            ("files", ("pdf.pdf", make_pdf(["Dense vectors live on page one."]))),
        ],
    )
    r = client.post(f"/api/v1/corpora/{cid}/retrieve", json={"query": "dense vectors", "top_k": 5})
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    hit = body["hits"][0]
    assert hit["filename"] == "pdf.pdf"
    assert hit["chunk"]["metadata"]["page_start"] == 1
    assert hit["result"]["rank"] == 1 and hit["result"]["score"] > 0
    assert hit["result"]["document_version_id"] == hit["chunk"]["document_version_id"]
    assert set(body["provenance"]) >= {
        "corpus_id", "corpus_version", "chunking_hash", "retriever", "retriever_config",
        "retriever_config_hash", "query_terms", "statistics", "environment", "elapsed_ms",
    }  # fmt: skip
    assert body["provenance"]["corpus_version"] == 1
    # database internals stay private
    assert "doc_key" not in r.text and "term_id" not in r.text


def test_retrieve_errors(client: TestClient) -> None:
    cid = client.post("/api/v1/corpora", json={"name": "e"}).json()["corpus"]["id"]
    url = f"/api/v1/corpora/{cid}/retrieve"
    assert client.post("/api/v1/corpora/nope/retrieve", json={"query": "x"}).status_code == 404
    assert client.post(url, json={"query": "x", "version": 5}).status_code == 404
    assert client.post(url, json={"query": ""}).status_code == 422
    assert client.post(url, json={"query": "   "}).status_code == 422
    assert client.post(url, json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post(url, json={"query": "x", "top_k": 101}).status_code == 422
    assert client.post(url, json={"query": "x", "bm25": {"b": 2}}).status_code == 422
    dense = client.post(url, json={"query": "x", "strategy": "dense"})
    assert dense.status_code == 501
    assert dense.json()["detail"]["capability"] == "Retrieval strategy"
    empty = client.post(url, json={"query": "anything"})
    assert empty.status_code == 200 and empty.json()["hits"] == []
