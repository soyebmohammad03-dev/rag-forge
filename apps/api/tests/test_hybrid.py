"""Hybrid retrieval over a real corpus: real BM25, real dense index (pinned model), real fusion."""

from typing import Any

import pytest

from rag_forge.domain.models import (
    Corpus,
    FusionMethod,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStrategy,
)
from rag_forge.ingestion.service import IngestionService
from rag_forge.retrieval.base import Candidate, RetrieverOutput
from rag_forge.retrieval.dense import DenseIndexNotReadyError, DenseIndexService, DenseRetriever
from rag_forge.retrieval.embedding import OnnxSentenceEmbedder
from rag_forge.retrieval.fusion import ReciprocalRankFusion
from rag_forge.retrieval.hybrid import ComponentMismatchError, HybridRetriever, hybrid_factory
from rag_forge.retrieval.lexical import Bm25Retriever
from rag_forge.retrieval.service import ForeignResultError, RetrievalService, RetrieverFactory
from rag_forge.storage.lexical_index import SqliteLexicalIndex
from rag_forge.storage.sqlite import SqliteStore
from rag_forge.storage.vector_index import SqliteVectorIndex

S, D, H = RetrievalStrategy.SPARSE, RetrievalStrategy.DENSE, RetrievalStrategy.HYBRID

DOCS = [
    ("plants.txt", b"Photosynthesis converts light energy into chemical energy stored in glucose."),
    ("green.txt", b"Green plants make food from sunlight, water and carbon dioxide."),
    ("cars.txt", b"Automobiles need regular oil changes and tyre rotations to stay reliable."),
    ("markets.txt", b"Equity prices fell sharply after the central bank raised interest rates."),
    ("bm25.txt", b"BM25 scores documents by exact term overlap with the query."),
]


def make(
    store: SqliteStore, embedder: OnnxSentenceEmbedder
) -> tuple[RetrievalService, DenseIndexService]:
    lexical = SqliteLexicalIndex(store)
    dense = DenseIndexService(SqliteVectorIndex(store), embedder)
    single: dict[RetrievalStrategy, RetrieverFactory] = {
        S: lambda r: Bm25Retriever(lexical, r.bm25),
        D: lambda r: DenseRetriever(dense),
    }
    return RetrievalService(store, {**single, H: hybrid_factory(single)}, embedder.spec), dense


def build(dense: DenseIndexService, corpus: Corpus, version: int) -> None:
    index, needs = dense.start(corpus, version)
    if needs:
        dense.build(index, corpus)


def ask(svc: RetrievalService, corpus_id: str, query: str, **kw: Any) -> RetrievalResponse:
    return svc.retrieve(corpus_id, RetrievalRequest.model_validate({"query": query, **kw}))


@pytest.fixture
def setup(
    store: SqliteStore, service: IngestionService, embedder: OnnxSentenceEmbedder
) -> tuple[RetrievalService, DenseIndexService, Corpus]:
    c = store.add_corpus(Corpus(name="hybrid"))
    service.ingest(c.id, DOCS)
    corpus = store.get_corpus(c.id)
    assert corpus is not None
    svc, dense = make(store, embedder)
    build(dense, corpus, 1)
    return svc, dense, corpus


def test_rrf_integration_matches_independent_component_runs(setup: Any) -> None:
    svc, dense, corpus = setup
    query = "how do plants make food from light"
    r = ask(
        svc, corpus.id, query, strategy="hybrid", top_k=3, hybrid={"candidate_k": 5, "rrf_k": 60}
    )
    bm25 = ask(svc, corpus.id, query, strategy="sparse", top_k=5)
    vec = ask(svc, corpus.id, query, strategy="dense", top_k=5)
    rank = {
        s: {h.result.chunk_id: h.result.rank for h in run.hits} for s, run in ((S, bm25), (D, vec))
    }
    raw = {
        s: {h.result.chunk_id: h.result.score for h in run.hits} for s, run in ((S, bm25), (D, vec))
    }

    # recompute RRF from the separate runs and compare with what hybrid returned
    union = set(rank[S]) | set(rank[D])
    expected = {c: sum(1 / (60 + rank[s][c]) for s in (S, D) if c in rank[s]) for c in union}
    best3 = sorted(expected, key=lambda c: (-round(expected[c], 12), c))[:3]
    assert [h.result.chunk_id for h in r.hits] == best3
    assert {h.filename for h in r.hits[:2]} == {"plants.txt", "green.txt"}

    members = {cid for cid, _ in dense.vectors.member_chunks(corpus, 1)}
    for i, hit in enumerate(r.hits, 1):
        f = hit.fusion
        assert f is not None and f.method is FusionMethod.RRF
        assert (
            hit.result.rank == i
            and hit.result.strategy is H
            and hit.result.retriever == "hybrid-rrf"
        )
        assert hit.result.score == f.score == pytest.approx(expected[hit.result.chunk_id])
        assert hit.result.chunk_id in members and hit.chunk.id == hit.result.chunk_id
        for comp in f.components:  # original ranks and raw scores kept, never overwritten
            assert comp.rank == rank[comp.strategy].get(hit.result.chunk_id)
            assert comp.score == raw[comp.strategy].get(hit.result.chunk_id)
            assert comp.contribution == pytest.approx(1 / (60 + comp.rank) if comp.rank else 0.0)

    p = r.provenance
    assert p.retriever == "hybrid-rrf" and p.index_id is not None
    assert p.retriever_config["fusion"] == {"method": "rrf", "k": 60}
    assert set(p.retriever_config["components"]) == {"sparse", "dense"}
    assert p.statistics["sparse.candidate_chunks"] == p.statistics["dense.candidate_chunks"] == 5
    assert p.query_terms  # from the BM25 component
    c = p.configuration
    assert (c.strategy, c.corpus_version, c.top_k) == (H, 1, 3)
    assert c.bm25 is not None and c.embedder is not None and c.hybrid is not None
    assert c.hybrid.weights == {}  # ignored by RRF, so not part of the configuration


def test_weighted_hybrid(setup: Any) -> None:
    svc, _, corpus = setup
    weighted = {"fusion": "weighted", "weights": {"sparse": 0.3, "dense": 0.7}, "candidate_k": 5}
    r = ask(svc, corpus.id, "plants sunlight", strategy="hybrid", top_k=5, hybrid=weighted)
    top = r.hits[0].fusion
    assert top is not None and top.method is FusionMethod.WEIGHTED
    assert r.hits[0].filename == "green.txt"
    for comp in top.components:
        weight = {S: 0.3, D: 0.7}[comp.strategy]
        assert comp.contribution == pytest.approx(weight * (comp.normalized_score or 0))
    assert r.provenance.retriever_config["fusion"]["normalization"] == "min-max per list"
    assert r.provenance.configuration.hybrid.rrf_k == 60  # type: ignore[union-attr]


def test_deterministic_and_top_k(setup: Any) -> None:
    svc, _, corpus = setup
    runs = [ask(svc, corpus.id, "energy", strategy="hybrid", top_k=4) for _ in range(3)]
    assert len({tuple((h.result.chunk_id, h.result.score) for h in r.hits) for r in runs}) == 1
    assert len({r.provenance.configuration_hash for r in runs}) == 1
    assert len(runs[0].hits) == 4
    other = ask(svc, corpus.id, "energy", strategy="hybrid", top_k=4, hybrid={"rrf_k": 10})
    assert other.provenance.configuration_hash != runs[0].provenance.configuration_hash


def test_missing_dense_index_fails_instead_of_falling_back(
    store: SqliteStore, service: IngestionService, embedder: OnnxSentenceEmbedder
) -> None:
    c = store.add_corpus(Corpus(name="no-dense"))
    service.ingest(c.id, DOCS)
    svc, _ = make(store, embedder)
    with pytest.raises(DenseIndexNotReadyError, match="build it first"):
        ask(svc, c.id, "plants", strategy="hybrid")
    assert ask(svc, c.id, "plants", strategy="sparse").hits  # BM25 alone still works


def test_bm25_index_cannot_be_missing_it_is_built_on_demand(setup: Any) -> None:
    svc, _, corpus = setup
    r = ask(svc, corpus.id, "plants", strategy="hybrid")
    assert (
        r.provenance.statistics["sparse.indexed_now"] == 5
    )  # first lexical use indexed the version
    assert (
        ask(svc, corpus.id, "plants", strategy="hybrid").provenance.statistics["sparse.indexed_now"]
        == 0
    )


def test_corpus_and_version_isolation(
    setup: Any, store: SqliteStore, service: IngestionService, embedder: OnnxSentenceEmbedder
) -> None:
    svc, dense, corpus = setup
    other = store.add_corpus(Corpus(name="other"))
    service.ingest(
        other.id, [("leaves.txt", b"Leaves capture light for photosynthesis in plants.")]
    )
    other_corpus = store.get_corpus(other.id)
    assert other_corpus is not None
    build(dense, other_corpus, 1)
    assert {h.filename for h in ask(svc, other.id, "plants", strategy="hybrid").hits} == {
        "leaves.txt"
    }

    service.ingest(corpus.id, [("plants.txt", b"Chlorophyll absorbs red and blue light.")])
    v1 = ask(svc, corpus.id, "photosynthesis glucose", strategy="hybrid", version=1)
    assert v1.provenance.corpus_version == 1
    assert v1.hits[0].chunk.text.startswith("Photosynthesis")  # the v1 text, not the edit
    with pytest.raises(DenseIndexNotReadyError, match="version 2"):
        ask(svc, corpus.id, "photosynthesis", strategy="hybrid")  # v2 has no dense index yet


def test_components_over_different_chunk_sets_are_refused(setup: Any, store: SqliteStore) -> None:
    _, _, corpus = setup

    class Partial:
        name = "partial"
        strategy = D

        def config(self) -> dict[str, Any]:
            return {}

        def retrieve(self, corpus: Corpus, version: int, query: str, top_k: int) -> RetrieverOutput:
            return RetrieverOutput([], [], {"candidate_chunks": 2.0})

    hybrid = HybridRetriever(
        {
            S: Bm25Retriever(SqliteLexicalIndex(store), RetrievalRequest(query="x").bm25),
            D: Partial(),
        },
        ReciprocalRankFusion(),
        10,
    )
    with pytest.raises(ComponentMismatchError, match="different chunk sets"):
        hybrid.retrieve(corpus, 1, "plants", 5)


def test_service_refuses_chunks_from_another_corpus_version(
    setup: Any, store: SqliteStore, service: IngestionService
) -> None:
    _, _, corpus = setup
    other = store.add_corpus(Corpus(name="elsewhere"))
    rec = service.ingest(other.id, [("x.txt", b"unrelated text")])
    dv_id = rec.files[0].document_version_id
    assert dv_id is not None
    foreign_chunk = store.chunks(dv_id, other.chunking.config_hash())[0][0]

    class Rogue:
        name = "rogue"
        strategy = S

        def config(self) -> dict[str, Any]:
            return {}

        def retrieve(self, corpus: Corpus, version: int, query: str, top_k: int) -> RetrieverOutput:
            return RetrieverOutput([Candidate(foreign_chunk.id, 1.0)], [])

    svc = RetrievalService(store, {S: lambda r: Rogue()})
    with pytest.raises(ForeignResultError, match="outside corpus version 1"):
        ask(svc, corpus.id, "anything")
