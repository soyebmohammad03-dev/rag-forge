import sqlite3
from pathlib import Path

import pytest

from rag_forge.domain.models import (
    ChunkingConfig,
    Corpus,
    ExtractionStatus,
    FileOutcome,
    IngestionStatus,
)
from rag_forge.ingestion.service import DocumentNotInCorpusError, IngestionService
from rag_forge.storage.blobs import BlobStore, sha256_hex
from rag_forge.storage.sqlite import SqliteStore
from tests.conftest import make_pdf


def outcomes(record) -> dict[str, FileOutcome]:  # type: ignore[no-untyped-def]
    return {f.filename: f.outcome for f in record.files}


@pytest.fixture
def corpus(store: SqliteStore) -> Corpus:
    return store.add_corpus(
        Corpus(name="c", chunking=ChunkingConfig(chunk_size=100, chunk_overlap=20))
    )


def test_first_ingestion_creates_version_one(
    service: IngestionService, store: SqliteStore, corpus: Corpus
) -> None:
    rec = service.ingest(
        corpus.id,
        [("a.txt", b"alpha " * 40), ("dir/b.md", b"# B\nbeta"), ("c.pdf", make_pdf(["gamma"]))],
    )
    assert rec.status is IngestionStatus.COMPLETED
    assert (rec.version_before, rec.version_after) == (0, 1)
    assert outcomes(rec) == {"a.txt": "added", "b.md": "added", "c.pdf": "added"}  # dir stripped
    a = rec.files[0]
    assert a.content_sha256 == sha256_hex(b"alpha " * 40)
    assert a.chunk_count and a.chunk_count > 1
    assert a.parser == "plaintext@1"
    assert rec.chunking_hash == corpus.chunking.config_hash()
    assert rec.environment.rag_forge_version
    stats = store.corpus_stats(store.get_corpus(corpus.id))  # type: ignore[arg-type]
    assert (stats.document_count, stats.chunk_count) == (
        3,
        sum(f.chunk_count or 0 for f in rec.files),
    )
    v1 = store.list_versions(corpus.id)[0]
    assert (v1.version, v1.added, v1.document_count) == (1, 3, 3)


def test_reupload_identical_is_unchanged_and_creates_no_version(
    service: IngestionService, store: SqliteStore, corpus: Corpus
) -> None:
    service.ingest(corpus.id, [("a.txt", b"alpha")])
    rec = service.ingest(corpus.id, [("a.txt", b"alpha")])
    assert rec.status is IngestionStatus.NO_CHANGE
    assert outcomes(rec) == {"a.txt": "unchanged"}
    assert rec.version_after == 1
    assert [v.version for v in store.list_versions(corpus.id)] == [1, 0]
    assert len(store.list_ingestions(corpus.id)) == 2  # the no-op is still recorded


def test_duplicate_content_under_another_name_is_not_ingested(
    service: IngestionService, corpus: Corpus
) -> None:
    first = service.ingest(corpus.id, [("a.txt", b"same bytes")])
    rec = service.ingest(corpus.id, [("copy.txt", b"same bytes"), ("x.txt", b"x"), ("y.txt", b"x")])
    got = {f.filename: f for f in rec.files}
    assert got["copy.txt"].outcome is FileOutcome.DUPLICATE
    assert got["copy.txt"].duplicate_of == first.files[0].document_id
    assert got["copy.txt"].duplicate_of_filename == "a.txt"
    assert got["x.txt"].outcome is FileOutcome.ADDED
    assert got["y.txt"].outcome is FileOutcome.DUPLICATE  # duplicates within one batch too


def test_modification_preserves_history(
    service: IngestionService, store: SqliteStore, corpus: Corpus
) -> None:
    v1 = service.ingest(corpus.id, [("a.txt", b"original text"), ("b.txt", b"stays")])
    v2 = service.ingest(corpus.id, [("a.txt", b"edited text")])
    assert outcomes(v2) == {"a.txt": "modified"}
    doc_id = v1.files[0].document_id
    assert v2.files[0].document_id == doc_id  # same logical document
    versions = store.document_versions(doc_id)  # type: ignore[arg-type]
    assert [(v.version, v.content_sha256) for v in versions] == [
        (1, sha256_hex(b"original text")),
        (2, sha256_hex(b"edited text")),
    ]
    # old corpus version still points at the old document version and its chunks
    assert store.members(corpus.id, 1)[doc_id].id == v1.files[0].document_version_id  # type: ignore[index]
    assert store.members(corpus.id, 2)[doc_id].id == v2.files[0].document_version_id  # type: ignore[index]
    old_chunks, _ = store.chunks(v1.files[0].document_version_id, corpus.chunking.config_hash())
    assert old_chunks[0].text == "original text"
    assert store.document_text(v1.files[0].document_version_id) == "original text"
    changes = {c.filename: c.change for c in store.version_changes(corpus.id, 2)}
    assert changes == {"a.txt": "modified", "b.txt": "unchanged"}
    cv2 = store.list_versions(corpus.id)[0]
    assert (cv2.added, cv2.modified, cv2.unchanged, cv2.removed) == (0, 1, 1, 0)


def test_remove_and_re_add(service: IngestionService, store: SqliteStore, corpus: Corpus) -> None:
    rec = service.ingest(corpus.id, [("a.txt", b"one"), ("b.txt", b"two")])
    doc_a = rec.files[0].document_id
    removal = service.remove(corpus.id, doc_a)  # type: ignore[arg-type]
    assert removal.version_after == 2
    assert outcomes(removal) == {"a.txt": "removed"}
    assert set(store.members(corpus.id, 2)) == {rec.files[1].document_id}
    assert doc_a in store.members(corpus.id, 1)  # history intact
    assert {c.filename: c.change for c in store.version_changes(corpus.id, 2)}["a.txt"] == "removed"
    with pytest.raises(DocumentNotInCorpusError):
        service.remove(corpus.id, doc_a)
    back = service.ingest(corpus.id, [("a.txt", b"one")])
    assert outcomes(back) == {"a.txt": "added"}
    assert back.files[0].document_id == doc_a  # identity survives removal
    assert len(store.document_versions(doc_a)) == 2


def test_rejections_are_reported_and_do_not_block_valid_files(
    service: IngestionService, corpus: Corpus
) -> None:
    rec = service.ingest(
        corpus.id,
        [
            ("good.txt", b"fine"),
            ("empty.txt", b""),
            ("big.txt", b"x" * 10_001),
            ("bad.pdf", b"%PDF-1.4 broken"),
            ("photo.png", b"\x89PNG"),
            ("good.txt", b"again"),
        ],
    )
    errors = {(f.filename, f.outcome.value): f.error for f in rec.files}
    assert errors[("good.txt", "added")] is None
    assert errors[("empty.txt", "rejected")] == "empty file"
    assert "limit" in errors[("big.txt", "rejected")]  # type: ignore[operator]
    assert "PDF" in errors[("bad.pdf", "rejected")]  # type: ignore[operator]
    assert "unsupported" in errors[("photo.png", "rejected")]  # type: ignore[operator]
    assert "more than once" in errors[("good.txt", "rejected")]  # type: ignore[operator]
    assert rec.status is IngestionStatus.COMPLETED


def test_all_rejected_is_failed_without_new_version(
    service: IngestionService, corpus: Corpus
) -> None:
    rec = service.ingest(corpus.id, [("a.exe", b"MZ")])
    assert rec.status is IngestionStatus.FAILED
    assert rec.version_after == 0


def test_partial_pdf_extraction_is_flagged(
    service: IngestionService, store: SqliteStore, corpus: Corpus
) -> None:
    rec = service.ingest(corpus.id, [("p.pdf", make_pdf(["has text", ""]))])
    dv = store.get_document_version(rec.files[0].document_version_id)  # type: ignore[arg-type]
    assert dv is not None
    assert dv.extraction_status is ExtractionStatus.PARTIAL
    assert dv.metadata["pages"] == 2


def test_persistence_across_restarts(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "db.sqlite3")
    corpus = store.add_corpus(Corpus(name="persist"))
    IngestionService(store, BlobStore(tmp_path / "b")).ingest(corpus.id, [("a.txt", b"kept")])
    reopened = SqliteStore(tmp_path / "db.sqlite3")
    again = reopened.get_corpus(corpus.id)
    assert again is not None and again.version == 1
    assert [d.filename for d in reopened.members(corpus.id, 1).values()] == ["a.txt"]
    assert BlobStore(tmp_path / "b").get(sha256_hex(b"kept")) == b"kept"


def test_history_tables_are_append_only(
    service: IngestionService, store: SqliteStore, corpus: Corpus
) -> None:
    service.ingest(corpus.id, [("a.txt", b"immutable")])
    db = sqlite3.connect(store.path)
    for sql in (
        "UPDATE document_versions SET content_sha256 = 'x'",
        "DELETE FROM chunks",
        "DELETE FROM corpus_version_members",
        "UPDATE ingestions SET data = '{}'",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute(sql)
