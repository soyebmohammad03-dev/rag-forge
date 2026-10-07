import math
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from rag_forge.domain.models import (
    Bm25Params,
    ChunkingConfig,
    Corpus,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStrategy,
)
from rag_forge.ingestion.service import IngestionService
from rag_forge.retrieval import bm25
from rag_forge.retrieval.analysis import ANALYZER, analyze
from rag_forge.retrieval.lexical import Bm25Retriever
from rag_forge.retrieval.service import (
    CorpusVersionNotFoundError,
    RetrievalService,
    StrategyNotAvailableError,
)
from rag_forge.storage.blobs import BlobStore
from rag_forge.storage.lexical_index import SqliteLexicalIndex
from rag_forge.storage.sqlite import SCHEMA_VERSION, SqliteStore

DOCS = {
    "sparse.txt": b"BM25 is a sparse lexical retrieval function. It rewards exact term matching "
    b"and normalises for document length.",
    "dense.txt": b"Dense retrieval embeds queries and passages as vectors and compares them with "
    b"cosine similarity.",
    "graph.md": b"# Graph retrieval\n\nGraph retrieval follows entity links between passages to "
    b"answer multi-hop questions.",
    "cooking.txt": b"Simmer the tomatoes with garlic and basil for twenty minutes.",
}


def make_service(store: SqliteStore) -> RetrievalService:
    index = SqliteLexicalIndex(store)
    return RetrievalService(
        store, {RetrievalStrategy.SPARSE: lambda r: Bm25Retriever(index, r.bm25)}
    )


@pytest.fixture
def retrieval(store: SqliteStore) -> RetrievalService:
    return make_service(store)


@pytest.fixture
def corpus(store: SqliteStore, service: IngestionService) -> Corpus:
    c = store.add_corpus(
        Corpus(name="ir", chunking=ChunkingConfig(chunk_size=500, chunk_overlap=0))
    )
    service.ingest(c.id, list(DOCS.items()))
    return c


def ask(svc: RetrievalService, corpus_id: str, query: str, **kw: object) -> RetrievalResponse:
    return svc.retrieve(corpus_id, RetrievalRequest.model_validate({"query": query, **kw}))


def files(r: RetrievalResponse) -> list[str]:
    return [h.filename for h in r.hits]


# --- analysis and scoring ---------------------------------------------------------------


def test_analyzer_normalises_and_drops_stopwords() -> None:
    assert analyze("The Dense-Retrieval of CAFÉ ﬁles, and it's 2026!") == [
        "dense", "retrieval", "café", "files", "s", "2026",
    ]  # fmt: skip


def test_bm25_matches_hand_computation() -> None:
    postings = [bm25.Posting("c1", "vector", 2, 10), bm25.Posting("c2", "vector", 1, 5)]
    params = Bm25Params(k1=1.2, b=0.75)
    ranked = bm25.rank(postings, Counter({"vector": 1}), n_docs=4, avg_length=7.5, params=params,
                       top_k=10)  # fmt: skip
    idf = math.log(1 + (4 - 2 + 0.5) / (2 + 0.5))
    s1 = idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 10 / 7.5))
    s2 = idf * 1 * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 5 / 7.5))
    assert [r.chunk_id for r in ranked] == ["c1", "c2"]
    assert [r.score for r in ranked] == pytest.approx([s1, s2])


# --- indexing ---------------------------------------------------------------------------


def test_index_is_built_once_per_chunk(store: SqliteStore, corpus: Corpus) -> None:
    index = SqliteLexicalIndex(store)
    assert index.ensure_indexed(corpus, 1) == 4  # one chunk per short document
    assert index.ensure_indexed(corpus, 1) == 0  # idempotent
    with store.transaction() as db:
        rows = db.execute(
            "SELECT analyzer, COUNT(*) FROM lexical_docs GROUP BY analyzer"
        ).fetchall()
        # only terms and counts are stored, never chunk text
        cols = [r[1] for r in db.execute("PRAGMA table_info(lexical_postings)")]
    assert rows == [(ANALYZER, 4)]
    assert cols == ["term_id", "doc_key", "tf"]
    assert index.stats(corpus, 1).chunks == 4


# --- ranking ------------------------------------------------------------------------------


def test_relevant_document_ranks_first(retrieval: RetrievalService, corpus: Corpus) -> None:
    assert files(ask(retrieval, corpus.id, "exact term matching with BM25"))[0] == "sparse.txt"
    assert files(ask(retrieval, corpus.id, "cosine similarity of vectors"))[0] == "dense.txt"
    assert files(ask(retrieval, corpus.id, "multi-hop entity links"))[0] == "graph.md"


def test_top_k_and_scores_descend(retrieval: RetrievalService, corpus: Corpus) -> None:
    r = ask(retrieval, corpus.id, "retrieval passages", top_k=2)
    assert len(r.hits) == 2
    assert [h.result.rank for h in r.hits] == [1, 2]
    assert r.hits[0].result.score >= r.hits[1].result.score > 0
    assert len(ask(retrieval, corpus.id, "retrieval passages", top_k=100).hits) == 3


def test_no_match_and_stopword_only_queries(retrieval: RetrievalService, corpus: Corpus) -> None:
    none = ask(retrieval, corpus.id, "quantum chromodynamics")
    assert none.hits == [] and none.warnings == []
    assert none.provenance.query_terms == ["chromodynamics", "quantum"]
    stop = ask(retrieval, corpus.id, "the and of it")
    assert stop.hits == []
    assert stop.warnings == ["query has no searchable terms (only stopwords or punctuation)"]
    with pytest.raises(ValueError, match="blank"):
        RetrievalRequest(query="   ")


def test_ordering_is_deterministic_with_id_tiebreak(
    store: SqliteStore, service: IngestionService, retrieval: RetrievalService
) -> None:
    c = store.add_corpus(Corpus(name="ties"))
    # different bytes, identical chunk text after trimming -> identical scores
    service.ingest(c.id, [("a.txt", b"shared words here"), ("b.txt", b"shared words here\n")])
    runs = [ask(retrieval, c.id, "shared words") for _ in range(3)]
    ids = [[h.result.chunk_id for h in r.hits] for r in runs]
    assert ids[0] == ids[1] == ids[2] == sorted(ids[0])
    assert runs[0].hits[0].result.score == runs[0].hits[1].result.score


# --- isolation and reproducibility ----------------------------------------------------------


def test_corpus_isolation(
    store: SqliteStore, service: IngestionService, retrieval: RetrievalService, corpus: Corpus
) -> None:
    other = store.add_corpus(Corpus(name="other"))
    service.ingest(other.id, [("tomato.txt", b"Tomatoes and garlic, simmered slowly.")])
    hits = ask(retrieval, other.id, "tomatoes garlic basil simmer").hits
    assert [h.filename for h in hits] == ["tomato.txt"]  # cooking.txt is in the other corpus


def test_version_isolation(
    store: SqliteStore, service: IngestionService, retrieval: RetrievalService
) -> None:
    c = store.add_corpus(Corpus(name="versions"))
    v1 = service.ingest(c.id, [("notes.txt", b"alpha protocol description")])
    v2 = service.ingest(c.id, [("notes.txt", b"beta protocol description")])
    old = ask(retrieval, c.id, "alpha", version=1)
    assert [h.result.document_version_id for h in old.hits] == [v1.files[0].document_version_id]
    assert old.provenance.corpus_version == 1
    assert ask(retrieval, c.id, "alpha").hits == []  # current (v2) no longer contains it
    current = ask(retrieval, c.id, "beta")
    assert [h.result.document_version_id for h in current.hits] == [v2.files[0].document_version_id]
    assert ask(retrieval, c.id, "beta", version=1).hits == []
    assert ask(retrieval, c.id, "alpha", version=0).hits == []  # the empty initial version
    with pytest.raises(CorpusVersionNotFoundError):
        ask(retrieval, c.id, "alpha", version=3)


def test_scores_depend_only_on_the_searched_version(
    store: SqliteStore, service: IngestionService, retrieval: RetrievalService, corpus: Corpus
) -> None:
    before = ask(retrieval, corpus.id, "dense retrieval vectors", version=1)
    other = store.add_corpus(Corpus(name="noise"))
    service.ingest(other.id, [(f"n{i}.txt", f"dense vectors {i}".encode()) for i in range(20)])
    service.ingest(corpus.id, [("more.txt", b"Retrieval of dense vectors, again and again.")])
    after = ask(retrieval, corpus.id, "dense retrieval vectors", version=1)
    assert [(h.result.chunk_id, h.result.score) for h in after.hits] == [
        (h.result.chunk_id, h.result.score) for h in before.hits
    ]
    assert after.provenance.statistics["candidate_chunks"] == 4


# --- provenance and errors ------------------------------------------------------------------


def test_hit_and_provenance_metadata(retrieval: RetrievalService, corpus: Corpus) -> None:
    r = ask(retrieval, corpus.id, "dense vectors", top_k=1, bm25={"k1": 0.9, "b": 0.4})
    hit = r.hits[0]
    assert hit.result.query_id == r.query.id
    assert hit.result.retriever == "bm25" and hit.result.strategy is RetrievalStrategy.SPARSE
    assert hit.result.origin == "retrieved"
    assert hit.filename == "dense.txt" and hit.document_version == 1
    assert hit.media_type == "text/plain"
    assert hit.chunk.id == hit.result.chunk_id and "vectors" in hit.chunk.text
    assert hit.matched_terms == ["dense", "vectors"]
    p = r.provenance
    assert (p.corpus_id, p.corpus_version) == (corpus.id, 1)
    assert p.chunking_hash == corpus.chunking.config_hash()
    assert p.retriever_config == {"k1": 0.9, "b": 0.4, "analyzer": ANALYZER}
    default = ask(retrieval, corpus.id, "dense vectors").provenance.retriever_config_hash
    assert p.retriever_config_hash != default
    assert p.statistics["candidate_chunks"] == 4 and p.statistics["matched_chunks"] == 1
    assert p.environment.rag_forge_version and p.elapsed_ms >= 0


def test_unimplemented_strategy(retrieval: RetrievalService, corpus: Corpus) -> None:
    with pytest.raises(StrategyNotAvailableError, match="'graph' is not implemented"):
        ask(retrieval, corpus.id, "vectors", strategy="graph")


def test_v1_database_migrates_and_backfills_index(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite3"
    store = SqliteStore(path)
    c = store.add_corpus(Corpus(name="legacy"))
    IngestionService(store, BlobStore(tmp_path / "b")).ingest(c.id, [("a.txt", b"legacy text")])
    db = sqlite3.connect(path)  # rewind to a Phase 1 (schema v1) database
    db.executescript(
        "DROP TABLE dense_vectors; DROP TABLE dense_indexes;"
        "DROP TABLE lexical_postings; DROP TABLE lexical_docs; DROP TABLE lexical_terms;"
        "DROP TABLE arena_replays; DROP TABLE arena_artifacts; DROP TABLE arena_run_cases;"
        "DROP TABLE arena_runs;"
        "DROP TABLE arena_experiments; DROP TABLE benchmark_datasets;"
        "PRAGMA user_version=1;"
    )
    db.close()

    reopened = SqliteStore(path)
    with reopened.transaction() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 5
    r = ask(make_service(reopened), c.id, "legacy")
    assert files(r) == ["a.txt"]
    assert r.provenance.statistics["indexed_now"] == 1
