"""Cross-encoder reranking: movement semantics, service integration, API, and the real model."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_forge.domain.models import (
    Corpus,
    RankMovement,
    RerankerInfo,
    RerankerSpec,
    RetrievalConfiguration,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStrategy,
    canonical_hash,
)
from rag_forge.ingestion.service import IngestionService
from rag_forge.main import create_app
from rag_forge.retrieval.base import Candidate
from rag_forge.retrieval.dense import DenseIndexService, DenseRetriever
from rag_forge.retrieval.embedding import OnnxSentenceEmbedder
from rag_forge.retrieval.hybrid import hybrid_factory
from rag_forge.retrieval.lexical import Bm25Retriever
from rag_forge.retrieval.rerank import (
    OnnxCrossEncoder,
    Reranker,
    RerankerUnavailableError,
    rank_movement,
)
from rag_forge.retrieval.service import (
    RerankerNotAvailableError,
    RetrievalService,
    RetrieverFactory,
)
from rag_forge.storage.lexical_index import SqliteLexicalIndex
from rag_forge.storage.sqlite import SqliteStore
from rag_forge.storage.vector_index import SqliteVectorIndex

S, D, H = RetrievalStrategy.SPARSE, RetrievalStrategy.DENSE, RetrievalStrategy.HYBRID
MODEL = RerankerSpec().model

# BM25 favours the lexical decoys ("make", "food", "need"); only one passage answers the query.
DOCS = [
    ("trucks.txt", b"Food trucks need permits to make and sell food, and food safety checks."),
    ("recipes.txt", b"To make bread you need flour, water, yeast and salt."),
    ("photo.txt", b"Through photosynthesis, plants use sunlight, water and CO2 to produce food."),
    ("cars.txt", b"Automobiles need regular oil changes and tyre rotations to stay reliable."),
    ("markets.txt", b"Equity prices fell sharply after the central bank raised interest rates."),
]
QUERY = "what do plants need to make food"


class KeywordReranker:
    """Deterministic test double: scores a passage by occurrences of `word` (ties are common)."""

    name = "keyword"

    def __init__(self, word: str = "plants", fail: bool = False) -> None:
        self.word = word
        self.fail = fail
        self.spec = RerankerSpec(provider="test", model=MODEL, revision="test")
        self.calls: list[int] = []

    def info(self) -> RerankerInfo:
        if self.fail:
            raise RerankerUnavailableError("reranker model test is unavailable: offline")
        return RerankerInfo(
            spec=self.spec,
            config_hash=self.spec.config_hash(),
            scoring="keyword count",
            activation="identity",
            max_seq_length=0,
            truncation="none",
            weights_sha256="0" * 64,
        )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append(len(passages))  # one batch call per request
        return [float(p.lower().count(self.word)) for p in passages]


def make(
    store: SqliteStore, embedder: OnnxSentenceEmbedder, reranker: Reranker
) -> tuple[RetrievalService, DenseIndexService]:
    lexical = SqliteLexicalIndex(store)
    dense = DenseIndexService(SqliteVectorIndex(store), embedder)
    single: dict[RetrievalStrategy, RetrieverFactory] = {
        S: lambda r: Bm25Retriever(lexical, r.bm25),
        D: lambda r: DenseRetriever(dense),
    }
    svc = RetrievalService(
        store, {**single, H: hybrid_factory(single)}, embedder.spec, {MODEL: reranker}
    )
    return svc, dense


def ask(svc: RetrievalService, corpus_id: str, **kw: Any) -> RetrievalResponse:
    return svc.retrieve(corpus_id, RetrievalRequest.model_validate({"query": QUERY, **kw}))


@pytest.fixture
def corpus(store: SqliteStore, service: IngestionService) -> Corpus:
    c = store.add_corpus(Corpus(name="rerank"))
    service.ingest(c.id, DOCS)
    found = store.get_corpus(c.id)
    assert found is not None
    return found


# --- movement semantics (pure) ------------------------------------------------------------


def cands(*scores: float) -> list[Candidate]:
    return [Candidate(f"c{i}", s) for i, s in enumerate(scores, 1)]


def test_rank_movement_definitions() -> None:
    pool = cands(9.0, 8.0, 7.0, 6.0)  # upstream ranks 1..4
    ranked = rank_movement(pool, [0.1, 0.9, 0.5, 0.7], final_top_k=2)
    assert [cid for cid, _ in ranked] == ["c2", "c4", "c3", "c1"]  # best reranker score first
    moved = dict(ranked)
    assert [moved[f"c{i}"].final_rank for i in range(1, 5)] == [4, 1, 3, 2]
    c1, c2, c3, c4 = (moved[f"c{i}"] for i in range(1, 5))
    assert (c2.original_rank, c2.final_rank, c2.rank_delta) == (2, 1, 1)
    assert c2.movement is RankMovement.PROMOTED and not c2.entered_top_k
    assert c1.movement is RankMovement.DEMOTED and c1.rank_delta == -3 and c1.left_top_k
    assert c3.movement is RankMovement.UNCHANGED and c3.rank_delta == 0
    assert c4.entered_top_k and not c4.left_top_k  # upstream #4 -> final #2 with k = 2
    assert (c4.original_score, c4.reranker_score) == (6.0, 0.7)


def test_rank_movement_ties_keep_upstream_order_and_is_deterministic() -> None:
    pool = cands(5.0, 4.0, 3.0, 2.0)
    first = rank_movement(pool, [1.0, 2.0, 1.0, 2.0], final_top_k=4)
    assert [cid for cid, _ in first] == ["c2", "c4", "c1", "c3"]
    assert all(rank_movement(pool, [1.0, 2.0, 1.0, 2.0], 4) == first for _ in range(5))
    assert rank_movement([], [], 3) == []


def test_rank_movement_rejects_bad_scores() -> None:
    with pytest.raises(RerankerUnavailableError, match="2 scores for 3"):
        rank_movement(cands(1, 2, 3), [0.1, 0.2], 1)
    with pytest.raises(RerankerUnavailableError, match="non-finite"):
        rank_movement(cands(1), [float("nan")], 1)


# --- service integration (deterministic reranker, real BM25) -----------------------------


def test_candidate_k_and_final_top_k(
    store: SqliteStore, corpus: Corpus, embedder: OnnxSentenceEmbedder
) -> None:
    reranker = KeywordReranker()
    svc, _ = make(store, embedder, reranker)
    upstream = ask(svc, corpus.id, top_k=4)
    r = ask(svc, corpus.id, top_k=2, rerank={"enabled": True, "candidate_k": 4})
    assert reranker.calls == [4]  # the whole pool, in one batched call
    assert len(r.hits) == 2 and r.reranking is not None
    pool = r.reranking.candidates
    assert [c.chunk_id for c in sorted(pool, key=lambda c: c.rerank.original_rank)] == [
        h.result.chunk_id for h in upstream.hits
    ]  # original ranks are exactly the unreranked ranking
    assert [c.rerank.final_rank for c in pool] == [1, 2, 3, 4]
    for hit, cand in zip(r.hits, pool, strict=False):
        assert hit.result.chunk_id == cand.chunk_id and hit.rerank == cand.rerank
        assert hit.result.rank == cand.rerank.final_rank
        assert hit.result.score == cand.rerank.reranker_score
        up = next(h for h in upstream.hits if h.result.chunk_id == hit.result.chunk_id)
        assert hit.rerank is not None and hit.rerank.original_score == up.result.score
        assert hit.matched_terms == up.matched_terms  # upstream evidence is preserved
    assert r.hits[0].filename == "photo.txt"  # the only passage mentioning plants
    prov = r.provenance.reranking
    assert prov is not None
    assert (prov.candidate_k, prov.final_top_k, prov.candidates_scored) == (4, 2, 4)
    stats = prov.statistics
    assert stats["promoted"] + stats["demoted"] + stats["unchanged"] == 4
    assert stats["entered_top_k"] == stats["left_top_k"]  # top-k membership is conserved


def test_small_pool_and_unreranked_requests_are_unchanged(
    store: SqliteStore, corpus: Corpus, embedder: OnnxSentenceEmbedder
) -> None:
    reranker = KeywordReranker()
    svc, _ = make(store, embedder, reranker)
    plain = ask(svc, corpus.id, top_k=3)
    assert plain.reranking is None and plain.provenance.reranking is None
    assert all(h.rerank is None for h in plain.hits) and reranker.calls == []
    config = plain.provenance.configuration
    assert config.rerank is None
    legacy = canonical_hash(config.model_dump(mode="json", exclude={"rerank", "routing"}))
    assert plain.provenance.configuration_hash == legacy  # Phase 4 hashes still hold
    disabled = ask(svc, corpus.id, top_k=3, rerank={"enabled": False, "candidate_k": 7})
    assert disabled.provenance.configuration_hash == plain.provenance.configuration_hash

    # BM25 matches only some chunks; asking for 50 candidates scores just those
    small = ask(svc, corpus.id, top_k=10, rerank={"enabled": True, "candidate_k": 50})
    assert small.reranking is not None and small.provenance.reranking is not None
    matched = small.provenance.statistics["matched_chunks"]
    assert small.provenance.reranking.candidates_scored == len(small.hits) == matched < len(DOCS)


def test_provenance_is_complete_and_hashes_are_deterministic(
    store: SqliteStore, corpus: Corpus, embedder: OnnxSentenceEmbedder
) -> None:
    svc, _ = make(store, embedder, KeywordReranker())
    kw: dict[str, Any] = {"top_k": 2, "rerank": {"enabled": True, "candidate_k": 4}}
    a, b = ask(svc, corpus.id, **kw), ask(svc, corpus.id, **kw)
    pa, pb = a.provenance, b.provenance
    assert pa.configuration_hash == pb.configuration_hash
    assert pa.configuration.rerank is not None and pa.configuration.rerank.candidate_k == 4
    assert pa.configuration.rerank.reranker.model == MODEL
    rp = pa.reranking
    assert rp is not None and pb.reranking is not None
    assert rp.reranker_config_hash == pb.reranking.reranker_config_hash
    assert rp.reranker == "keyword" and rp.info.weights_sha256 == "0" * 64
    assert rp.latency_ms >= 0 and pa.corpus_version == rp.upstream_configuration.corpus_version
    # the upstream configuration is exactly the unreranked retrieval of the pool
    upstream = ask(svc, corpus.id, top_k=4)
    assert rp.upstream_configuration_hash == upstream.provenance.configuration_hash
    assert pa.configuration_hash != upstream.provenance.configuration_hash
    other = ask(svc, corpus.id, top_k=2, rerank={"enabled": True, "candidate_k": 3})
    assert other.provenance.configuration_hash != pa.configuration_hash


def test_reranking_respects_corpus_version(
    store: SqliteStore,
    corpus: Corpus,
    service: IngestionService,
    embedder: OnnxSentenceEmbedder,
) -> None:
    svc, _ = make(store, embedder, KeywordReranker())
    service.ingest(corpus.id, [("v2.txt", b"Plants plants plants need food to make more plants.")])
    rerank = {"enabled": True, "candidate_k": 10}
    v1 = ask(svc, corpus.id, top_k=5, version=1, rerank=rerank)
    v2 = ask(svc, corpus.id, top_k=5, rerank=rerank)
    assert "v2.txt" not in {h.filename for h in v1.hits}
    assert v1.reranking is not None and "v2.txt" not in {
        c.filename for c in v1.reranking.candidates
    }
    assert v2.hits[0].filename == "v2.txt"  # the new chunk wins only in the version that has it
    assert v1.provenance.configuration_hash != v2.provenance.configuration_hash


def test_reranker_failures_are_explicit(
    store: SqliteStore, corpus: Corpus, embedder: OnnxSentenceEmbedder
) -> None:
    svc, _ = make(store, embedder, KeywordReranker(fail=True))
    with pytest.raises(RerankerUnavailableError, match="unavailable"):
        ask(svc, corpus.id, rerank={"enabled": True})
    assert ask(svc, corpus.id).hits  # unreranked retrieval does not touch the reranker
    with pytest.raises(RerankerNotAvailableError, match="'other/model' is not registered"):
        ask(svc, corpus.id, rerank={"enabled": True, "model": "other/model"})


def test_unavailable_model_is_an_explicit_error(tmp_path: Path) -> None:
    bad = OnnxCrossEncoder(RerankerSpec(revision="0" * 40), cache_dir=tmp_path)
    with pytest.raises(RerankerUnavailableError, match="is unavailable"):
        bad.info()


# --- API ------------------------------------------------------------------------------------


def upload(c: TestClient) -> str:
    cid: str = c.post("/api/v1/corpora", json={"name": "rr"}).json()["corpus"]["id"]
    c.post(f"/api/v1/corpora/{cid}/documents", files=[("files", d) for d in DOCS])
    return cid


def test_api_validation_and_failures(tmp_path: Path, embedder: OnnxSentenceEmbedder) -> None:
    with TestClient(create_app(tmp_path, embedder, KeywordReranker(fail=True))) as c:
        url = f"/api/v1/corpora/{upload(c)}/retrieve"
        on = {"enabled": True}
        for bad in (
            {"top_k": 5, "rerank": {**on, "candidate_k": 4}},  # pool smaller than final top-k
            {"rerank": {**on, "candidate_k": 0}},
            {"rerank": {**on, "candidate_k": 201}},
            {"strategy": "hybrid", "rerank": {**on, "candidate_k": 60}},  # hybrid pool is 50
            {"rerank": {**on, "top_k": 3}},  # unknown field
        ):
            r = c.post(url, json={"query": QUERY, **bad})
            assert r.status_code == 422, bad
        down = c.post(url, json={"query": QUERY, "rerank": on})
        assert down.status_code == 503 and down.json()["detail"]["component"] == "reranker"
        unknown = c.post(url, json={"query": QUERY, "rerank": {**on, "model": "x/y"}})
        assert unknown.status_code == 501 and unknown.json()["detail"]["capability"] == "Reranker"
        plain = c.post(url, json={"query": QUERY})
        assert plain.status_code == 200 and plain.json()["reranking"] is None
        health = {x["name"]: x for x in c.get("/api/v1/health").json()["components"]}
        assert health["reranker"]["state"] == "ok" and MODEL in health["reranker"]["detail"]


# --- the real cross-encoder (not mocked) -----------------------------------------------------


def test_real_cross_encoder_scores_pairs(cross_encoder: OnnxCrossEncoder) -> None:
    info = cross_encoder.info()
    assert info.spec.model == MODEL and info.spec.revision == RerankerSpec().revision
    assert info.activation == "identity" and info.max_seq_length == 512
    assert len(info.weights_sha256) == 64
    passages = [d.decode() for _, d in DOCS]
    scores = cross_encoder.score(QUERY, passages)
    assert scores == cross_encoder.score(QUERY, passages)  # deterministic
    best = max(range(len(scores)), key=scores.__getitem__)
    assert DOCS[best][0] == "photo.txt"
    # batch composition does not change a pair's score
    small = OnnxCrossEncoder(RerankerSpec(batch_size=2))
    assert small.score(QUERY, passages) == pytest.approx(scores, abs=1e-4)


def test_real_reranking_over_bm25_dense_and_hybrid(
    tmp_path: Path, embedder: OnnxSentenceEmbedder, cross_encoder: OnnxCrossEncoder
) -> None:
    with TestClient(create_app(tmp_path, embedder, cross_encoder)) as c:
        cid = upload(c)
        url = f"/api/v1/corpora/{cid}/retrieve"
        assert c.post(f"/api/v1/corpora/{cid}/dense-index", json={}).status_code == 202
        rerank = {"enabled": True, "candidate_k": 5}
        bm25 = c.post(url, json={"query": QUERY, "top_k": 5}).json()
        assert bm25["hits"][0]["filename"] != "photo.txt"  # the lexical decoy wins upstream
        for strategy in ("sparse", "dense", "hybrid"):
            r = c.post(
                url,
                json={"query": QUERY, "strategy": strategy, "top_k": 2, "rerank": rerank},
            )
            assert r.status_code == 200, (strategy, r.text)
            body = r.json()
            top = body["hits"][0]
            assert top["filename"] == "photo.txt", strategy  # the answer, by the real model
            assert top["result"]["strategy"] == strategy  # upstream identity is kept
            assert top["result"]["score"] == top["rerank"]["reranker_score"]
            if strategy == "hybrid":
                assert top["fusion"] is not None  # fusion detail survives reranking
            prov = body["provenance"]["reranking"]
            assert prov["info"]["spec"]["revision"] == RerankerSpec().revision
            assert prov["reranker"] == "cross-encoder" and prov["candidates_scored"] >= 2
            config = RetrievalConfiguration.model_validate(prov["upstream_configuration"])
            assert config.strategy == strategy and config.top_k == 5 and config.rerank is None
        moved = c.post(url, json={"query": QUERY, "top_k": 2, "rerank": rerank}).json()
        photo = next(x for x in moved["reranking"]["candidates"] if x["filename"] == "photo.txt")
        assert photo["rerank"]["movement"] == "promoted" and photo["rerank"]["rank_delta"] > 0
