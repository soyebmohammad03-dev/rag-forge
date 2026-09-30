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
    graph = client.post(url, json={"query": "x", "strategy": "graph"})
    assert graph.status_code == 501
    assert graph.json()["detail"]["capability"] == "Retrieval strategy"
    empty = client.post(url, json={"query": "anything"})
    assert empty.status_code == 200 and empty.json()["hits"] == []


def test_dense_index_lifecycle_and_strategy_parity(client: TestClient) -> None:
    cid = client.post("/api/v1/corpora", json={"name": "dense"}).json()["corpus"]["id"]
    client.post(
        f"/api/v1/corpora/{cid}/documents",
        files=[
            ("files", ("plants.txt", b"Photosynthesis converts light into chemical energy.")),
            ("files", ("cars.txt", b"Automobiles need regular oil changes.")),
        ],
    )
    status_url = f"/api/v1/corpora/{cid}/dense-index"
    missing = client.get(status_url).json()
    assert missing["state"] == "missing" and missing["index"] is None
    assert missing["embedder"]["model"] == "BAAI/bge-small-en-v1.5"

    url = f"/api/v1/corpora/{cid}/retrieve"
    blocked = client.post(url, json={"query": "plants", "strategy": "dense"})
    assert blocked.status_code == 409 and blocked.json()["detail"]["state"] == "missing"

    started = client.post(status_url, json={})
    assert started.status_code == 202  # TestClient runs the background build before returning
    ready = client.get(status_url).json()
    assert ready["state"] == "ready" and ready["index"]["chunk_count"] == 2
    assert client.post(status_url, json={}).status_code == 200  # idempotent

    dense = client.post(url, json={"query": "how do green plants make food", "strategy": "dense"})
    sparse = client.post(url, json={"query": "oil changes", "strategy": "bm25"})  # alias
    assert dense.status_code == sparse.status_code == 200
    d, s = dense.json(), sparse.json()
    assert d.keys() == s.keys() and d["hits"][0].keys() == s["hits"][0].keys()
    assert d["provenance"].keys() == s["provenance"].keys()
    assert d["hits"][0]["filename"] == "plants.txt"
    assert d["provenance"]["strategy"] == "dense" and s["provenance"]["strategy"] == "sparse"
    assert (
        d["provenance"]["index_id"] == ready["index"]["id"] and s["provenance"]["index_id"] is None
    )

    client.post(f"/api/v1/corpora/{cid}/documents", files=[("files", ("new.txt", b"New text."))])
    stale = client.post(url, json={"query": "plants", "strategy": "dense"})
    assert stale.status_code == 409 and stale.json()["detail"]["state"] == "stale"
    assert client.get(status_url, params={"version": 1}).json()["state"] == "ready"
    assert client.get(status_url, params={"version": 9}).status_code == 404


def test_unavailable_embedder_is_503(tmp_path: Any) -> None:
    from rag_forge.domain.models import EmbedderSpec
    from rag_forge.main import create_app
    from rag_forge.retrieval.embedding import OnnxSentenceEmbedder

    bad = OnnxSentenceEmbedder(EmbedderSpec(revision="0" * 40), cache_dir=tmp_path / "hf")
    with TestClient(create_app(tmp_path, embedder=bad)) as c:
        cid = c.post("/api/v1/corpora", json={"name": "x"}).json()["corpus"]["id"]
        r = c.post(f"/api/v1/corpora/{cid}/dense-index", json={})
        assert r.status_code == 503 and "unavailable" in r.json()["detail"]
        assert c.get(f"/api/v1/corpora/{cid}/dense-index").json()["state"] == "missing"
