"""Dense retrieval against the real pinned embedding model (downloaded once into the HF cache)."""

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from rag_forge.domain.models import (
    Corpus,
    DenseIndexState,
    DenseIndexStatus,
    EmbedderInfo,
    EmbedderSpec,
    RetrievalRequest,
    RetrievalStrategy,
)
from rag_forge.ingestion.service import IngestionService
from rag_forge.retrieval.dense import DenseIndexNotReadyError, DenseIndexService, DenseRetriever
from rag_forge.retrieval.embedding import EmbedderUnavailableError, OnnxSentenceEmbedder
from rag_forge.retrieval.lexical import Bm25Retriever
from rag_forge.retrieval.service import RetrievalService
from rag_forge.storage.blobs import BlobStore
from rag_forge.storage.lexical_index import SqliteLexicalIndex
from rag_forge.storage.sqlite import SqliteStore
from rag_forge.storage.vector_index import IndexIntegrityError, SqliteVectorIndex

DOCS = [
    ("plants.txt", b"Photosynthesis converts light energy into chemical energy stored in glucose."),
    ("cars.txt", b"Automobiles need regular oil changes and tyre rotations to stay reliable."),
    ("markets.txt", b"Equity prices fell sharply after the central bank raised interest rates."),
    ("sparse.txt", b"BM25 scores documents by exact term overlap with the query."),
]


@pytest.fixture
def dense(store: SqliteStore, embedder: OnnxSentenceEmbedder) -> DenseIndexService:
    return DenseIndexService(SqliteVectorIndex(store), embedder)


@pytest.fixture
def corpus(store: SqliteStore, service: IngestionService) -> Corpus:
    c = store.add_corpus(Corpus(name="mixed"))
    service.ingest(c.id, DOCS)
    return store.get_corpus(c.id)  # type: ignore[return-value]


def build(dense: DenseIndexService, corpus: Corpus, version: int | None = None):  # type: ignore[no-untyped-def]
    index, needs = dense.start(corpus, corpus.version if version is None else version)
    return dense.build(index, corpus) if needs else index


def retrieval(store: SqliteStore, dense: DenseIndexService) -> RetrievalService:
    lexical = SqliteLexicalIndex(store)
    return RetrievalService(
        store,
        {
            RetrievalStrategy.SPARSE: lambda r: Bm25Retriever(lexical, r.bm25),
            RetrievalStrategy.DENSE: lambda r: DenseRetriever(dense),
        },
    )


def ask(svc: RetrievalService, corpus_id: str, query: str, **kw: object):  # type: ignore[no-untyped-def]
    return svc.retrieve(corpus_id, RetrievalRequest.model_validate({"query": query, **kw}))


# --- embeddings -------------------------------------------------------------------------


def test_embeddings_are_unit_length_deterministic_and_semantic(
    embedder: OnnxSentenceEmbedder,
) -> None:
    texts = [t.decode() for _, t in DOCS]
    a, b = embedder.embed_documents(texts), embedder.embed_documents(texts)
    assert a.shape == (4, 384) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-5)
    assert np.array_equal(a, b)  # bit-identical across calls
    assert np.array_equal(a[1], embedder.embed_documents([texts[1]])[0])  # batch-independent
    q = embedder.embed_query("how do green plants make their food")
    assert int(np.argmax(a @ q)) == 0
    assert not np.array_equal(
        q, embedder.embed_documents(["how do green plants make their food"])[0]
    )
    info = embedder.info()
    assert (info.dimension, info.pooling, info.normalize, info.max_seq_length) == (
        384,
        "cls",
        True,
        512,
    )
    assert len(info.weights_sha256) == 64 and info.config_hash == EmbedderSpec().config_hash()
    assert embedder.embed_documents([]).shape == (0, 384)


def test_unavailable_model_is_an_explicit_error(tmp_path: Path) -> None:
    bad = OnnxSentenceEmbedder(EmbedderSpec(revision="0" * 40), cache_dir=tmp_path)
    with pytest.raises(EmbedderUnavailableError, match="is unavailable"):
        bad.info()


# --- index lifecycle ----------------------------------------------------------------------


def test_build_records_identity_and_is_idempotent(dense: DenseIndexService, corpus: Corpus) -> None:
    assert dense.state(corpus, 1) == (DenseIndexState.MISSING, None)
    index = build(dense, corpus)
    assert index.status is DenseIndexStatus.READY
    assert (index.corpus_id, index.corpus_version, index.chunk_count) == (corpus.id, 1, 4)
    assert (index.embedded, index.reused) == (4, 0)
    assert index.embedder.spec.model == "BAAI/bge-small-en-v1.5" and index.similarity == "cosine"
    assert index.content_hash and index.finished_at and index.error is None
    assert dense.state(corpus, 1) == (DenseIndexState.READY, index)
    again, needs = dense.start(corpus, 1)
    assert (again.id, needs) == (index.id, False)


def test_new_version_reuses_vectors_and_isolates(
    store: SqliteStore, service: IngestionService, dense: DenseIndexService, corpus: Corpus
) -> None:
    v1 = build(dense, corpus)
    service.ingest(corpus.id, [("cars.txt", b"Electric cars need far less maintenance.")])
    corpus = store.get_corpus(corpus.id)  # type: ignore[assignment]
    assert dense.state(corpus, 2)[0] is DenseIndexState.STALE  # v1 is indexed, v2 is not
    with pytest.raises(DenseIndexNotReadyError, match="version 2"):
        dense.ready_index(corpus, 2)
    v2 = build(dense, corpus)
    assert (v2.chunk_count, v2.reused, v2.embedded) == (4, 3, 4)  # only the edited chunk embedded
    assert dense.vectors.content_hash(v1) == v1.content_hash  # v1 untouched by the v2 build
    svc = retrieval(store, dense)
    old = ask(svc, corpus.id, "vehicle upkeep", strategy="dense", version=1).hits
    new = ask(svc, corpus.id, "vehicle upkeep", strategy="dense").hits
    assert old[0].chunk.text.startswith("Automobiles") and old[0].result.document_version_id != (
        new[0].result.document_version_id
    )
    assert new[0].chunk.text.startswith("Electric cars")


def test_same_texts_embed_identically_in_separate_databases(
    tmp_path: Path, embedder: OnnxSentenceEmbedder
) -> None:
    matrices = []
    for name in ("a", "b"):
        store = SqliteStore(tmp_path / f"{name}.sqlite3")
        c = store.add_corpus(Corpus(name=name))
        IngestionService(store, BlobStore(tmp_path / name)).ingest(c.id, DOCS)
        c = store.get_corpus(c.id)  # type: ignore[assignment]
        vectors = SqliteVectorIndex(store)
        index = build(DenseIndexService(vectors, embedder), c)
        ids, matrix = vectors._load_matrix(index)
        texts = dict(vectors.member_chunks(c, 1))
        order = np.argsort([texts[i] for i in ids])
        matrices.append(matrix[order])
    assert np.array_equal(matrices[0], matrices[1])


def test_failed_and_interrupted_builds_are_visible(
    store: SqliteStore, corpus: Corpus, embedder: OnnxSentenceEmbedder
) -> None:
    class Broken:
        spec = embedder.spec

        def info(self) -> EmbedderInfo:
            return embedder.info()

        def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
            raise RuntimeError("out of memory")

        def embed_query(self, text: str) -> np.ndarray:
            raise AssertionError

    broken = DenseIndexService(SqliteVectorIndex(store), Broken())
    failed = build(broken, corpus)
    assert (failed.status, failed.error) == (DenseIndexStatus.FAILED, "out of memory")
    assert broken.state(corpus, 1)[0] is DenseIndexState.FAILED
    healthy = DenseIndexService(SqliteVectorIndex(store), embedder)
    assert build(healthy, corpus).status is DenseIndexStatus.READY  # a failed index can be rebuilt

    other = store.add_corpus(Corpus(name="interrupted"))
    started, _ = healthy.start(other, 0)  # never built: simulates a crash mid-build
    DenseIndexService(SqliteVectorIndex(store), embedder)  # restart
    state, index = healthy.state(other, 0)
    assert state is DenseIndexState.FAILED and index is not None and "interrupted" in index.error  # type: ignore[operator]
    assert index.id == started.id


def test_corrupt_or_missing_vectors_are_detected(
    store: SqliteStore, dense: DenseIndexService, corpus: Corpus
) -> None:
    index = build(dense, corpus)
    db = sqlite3.connect(store.path)
    victim = db.execute("SELECT chunk_id FROM dense_vectors LIMIT 1").fetchone()[0]
    db.execute("DELETE FROM dense_vectors WHERE chunk_id = ?", (victim,))
    db.commit()
    with pytest.raises(IndexIntegrityError, match="1 chunks have no vector"):
        SqliteVectorIndex(store).search(index, np.zeros(384, np.float32), 3)
    db.execute(
        "INSERT INTO dense_vectors VALUES (?, ?, ?)",
        (index.embedder.config_hash, victim, b"\0" * 8),
    )
    db.commit()
    with pytest.raises(IndexIntegrityError, match="not 384-dimensional"):
        SqliteVectorIndex(store).search(index, np.zeros(384, np.float32), 3)
    with pytest.raises(IndexIntegrityError, match="query has shape"):
        dense.vectors.search(index, np.zeros(8, np.float32), 3)


# --- retrieval ------------------------------------------------------------------------------


def test_dense_retrieval_end_to_end(
    store: SqliteStore, dense: DenseIndexService, corpus: Corpus
) -> None:
    index = build(dense, corpus)
    svc = retrieval(store, dense)
    # no word in common with plants.txt: lexical retrieval cannot find it, dense does
    r = ask(svc, corpus.id, "how do green plants make their food", strategy="dense", top_k=2)
    top = r.hits[0]
    expected = dense.vectors.member_chunks(corpus, 1)
    assert top.filename == "plants.txt"
    assert top.result.chunk_id in {cid for cid, _ in expected}
    assert top.result.strategy is RetrievalStrategy.DENSE and top.result.retriever == "dense"
    assert len(r.hits) == 2 and 0 < r.hits[1].result.score < top.result.score <= 1
    assert top.matched_terms == []
    p = r.provenance
    assert p.index_id == index.id and p.retriever_config["model"] == "BAAI/bge-small-en-v1.5"
    assert p.retriever_config["similarity"] == "cosine" and p.retriever_config["dimension"] == 384
    assert p.retriever_config["revision"] == EmbedderSpec().revision
    assert p.statistics["candidate_chunks"] == 4 and p.environment.rag_forge_version
    lexical = ask(svc, corpus.id, "how do green plants make their food", strategy="bm25")
    assert "plants.txt" not in [h.filename for h in lexical.hits]  # BM25 unchanged (regression)


def test_ordering_is_deterministic_with_ties(
    store: SqliteStore, service: IngestionService, dense: DenseIndexService
) -> None:
    c = store.add_corpus(Corpus(name="ties"))
    service.ingest(
        c.id, [("a.txt", b"identical text"), ("b.txt", b"identical text\n"), ("c.txt", b"x")]
    )
    c = store.get_corpus(c.id)  # type: ignore[assignment]
    build(dense, c)
    svc = retrieval(store, dense)
    runs = [
        [
            (h.result.chunk_id, h.result.score)
            for h in ask(svc, c.id, "identical", strategy="dense").hits
        ]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]
    assert runs[0][0][1] == runs[0][1][1] and runs[0][0][0] < runs[0][1][0]  # tie -> chunk id
    assert len(ask(svc, c.id, "identical", strategy="dense", top_k=1).hits) == 1


def test_empty_version_and_unindexed_corpus(store: SqliteStore, dense: DenseIndexService) -> None:
    c = store.add_corpus(Corpus(name="empty"))
    svc = retrieval(store, dense)
    with pytest.raises(DenseIndexNotReadyError, match="build it first"):
        ask(svc, c.id, "anything", strategy="dense")
    build(dense, c)
    r = ask(svc, c.id, "anything", strategy="dense")
    assert r.hits == [] and r.warnings == ["the dense index for this version is empty"]
