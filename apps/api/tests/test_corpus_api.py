from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import make_pdf


def create(client: TestClient, name: str = "papers", **extra: Any) -> dict[str, Any]:
    r = client.post("/api/v1/corpora", json={"name": name, **extra})
    assert r.status_code == 201, r.text
    body: dict[str, Any] = r.json()
    return body


def upload(client: TestClient, corpus_id: str, *files: tuple[str, bytes]) -> dict[str, Any]:
    r = client.post(
        f"/api/v1/corpora/{corpus_id}/documents",
        files=[("files", (name, data)) for name, data in files],
    )
    assert r.status_code == 200, r.text
    body: dict[str, Any] = r.json()
    return body


def test_corpus_create_list_inspect(client: TestClient) -> None:
    body = create(
        client,
        description="d",
        metadata={"domain": "biomed"},
        chunking={"strategy": "fixed", "chunk_size": 200, "chunk_overlap": 0},
    )
    corpus, stats = body["corpus"], body["stats"]
    assert corpus["version"] == 0 and corpus["metadata"] == {"domain": "biomed"}
    assert corpus["chunking"]["strategy"] == "fixed"
    assert stats == {
        "document_count": 0,
        "chunk_count": 0,
        "total_bytes": 0,
        "total_chars": 0,
        "last_ingestion": None,
    }
    assert [c["corpus"]["id"] for c in client.get("/api/v1/corpora").json()] == [corpus["id"]]
    assert client.get(f"/api/v1/corpora/{corpus['id']}").json()["corpus"]["name"] == "papers"


def test_corpus_validation(client: TestClient) -> None:
    create(client)
    assert client.post("/api/v1/corpora", json={"name": "papers"}).status_code == 409
    bad = {"name": "x", "chunking": {"chunk_size": 100, "chunk_overlap": 100}}
    assert client.post("/api/v1/corpora", json=bad).status_code == 422
    assert client.get("/api/v1/corpora/cor_missing").status_code == 404


def test_upload_version_and_inspect(client: TestClient) -> None:
    cid = create(client)["corpus"]["id"]
    rec = upload(
        client,
        cid,
        ("notes.md", b"# Notes\nbody"),
        ("paper.pdf", make_pdf(["pdf text"])),
        ("bad.docx", b"PK"),
    )
    assert rec["status"] == "completed" and rec["version_after"] == 1
    by_name = {f["filename"]: f for f in rec["files"]}
    assert by_name["bad.docx"]["outcome"] == "rejected"
    assert by_name["paper.pdf"]["media_type"] == "application/pdf"

    summary = client.get(f"/api/v1/corpora/{cid}").json()
    assert summary["corpus"]["version"] == 1
    assert summary["stats"]["document_count"] == 2
    assert summary["stats"]["last_ingestion"]["id"] == rec["id"]

    docs = client.get(f"/api/v1/corpora/{cid}/documents").json()
    assert [d["filename"] for d in docs] == ["notes.md", "paper.pdf"]
    doc_id, dv_id = docs[0]["document_id"], docs[0]["current"]["id"]

    detail = client.get(f"/api/v1/corpora/{cid}/documents/{doc_id}").json()
    assert detail["current_version_id"] == dv_id
    assert detail["versions"][0]["metadata"]["title"] == "Notes"

    chunks = client.get(f"/api/v1/document-versions/{dv_id}/chunks").json()
    assert chunks["total"] == 1 and chunks["items"][0]["text"] == "# Notes\nbody"

    fmt = client.get("/api/v1/ingestion/formats").json()
    assert {f["media_type"] for f in fmt} == {"text/plain", "text/markdown", "application/pdf"}


def test_versions_changes_and_removal(client: TestClient) -> None:
    cid = create(client)["corpus"]["id"]
    upload(client, cid, ("a.txt", b"one"), ("b.txt", b"two"))
    upload(client, cid, ("a.txt", b"one, revised"))
    doc_b = next(
        d for d in client.get(f"/api/v1/corpora/{cid}/documents").json() if d["filename"] == "b.txt"
    )["document_id"]
    removal = client.delete(f"/api/v1/corpora/{cid}/documents/{doc_b}").json()
    assert removal["version_after"] == 3
    assert client.delete(f"/api/v1/corpora/{cid}/documents/{doc_b}").status_code == 404

    versions = client.get(f"/api/v1/corpora/{cid}/versions").json()
    assert [(v["version"], v["added"], v["modified"], v["removed"]) for v in versions] == [
        (3, 0, 0, 1),
        (2, 0, 1, 0),
        (1, 2, 0, 0),
        (0, 0, 0, 0),
    ]
    changes = client.get(f"/api/v1/corpora/{cid}/versions/2/changes").json()
    assert {c["filename"]: c["change"] for c in changes} == {
        "a.txt": "modified",
        "b.txt": "unchanged",
    }
    assert client.get(f"/api/v1/corpora/{cid}/versions/9/changes").status_code == 404

    # historical snapshot still readable
    v1_docs = client.get(f"/api/v1/corpora/{cid}/documents", params={"version": 1}).json()
    assert sorted(d["filename"] for d in v1_docs) == ["a.txt", "b.txt"]
    assert client.get(f"/api/v1/corpora/{cid}/documents", params={"version": 7}).status_code == 404

    detail = client.get(f"/api/v1/corpora/{cid}/documents/{doc_b}").json()
    assert detail["current_version_id"] is None and len(detail["versions"]) == 1
    assert len(client.get(f"/api/v1/corpora/{cid}/ingestions").json()) == 3


def test_upload_to_missing_corpus(client: TestClient) -> None:
    r = client.post("/api/v1/corpora/nope/documents", files=[("files", ("a.txt", b"x"))])
    assert r.status_code == 404


def test_data_survives_app_restart(tmp_path: Any) -> None:
    from rag_forge.main import create_app

    with TestClient(create_app(tmp_path)) as c:
        cid = create(c)["corpus"]["id"]
        upload(c, cid, ("a.txt", b"durable"))
    with TestClient(create_app(tmp_path)) as c:
        assert c.get(f"/api/v1/corpora/{cid}").json()["stats"]["document_count"] == 1
