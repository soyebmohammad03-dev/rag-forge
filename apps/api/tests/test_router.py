"""Query intelligence and adaptive routing: features, rules, decisions, and the real stack."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rag_forge.domain.models import (
    Complexity,
    Corpus,
    CorpusQuerySignals,
    EvidenceNeed,
    QueryAnalysis,
    QueryClass,
    QuerySignal,
    QuestionType,
    RetrievalMode,
    RetrievalRequest,
    RetrievalStrategy,
    RouteOption,
    canonical_hash,
)
from rag_forge.ingestion.service import IngestionService
from rag_forge.main import create_app, make_router
from rag_forge.retrieval.dense import DenseIndexNotReadyError, DenseIndexService, DenseRetriever
from rag_forge.retrieval.embedding import OnnxSentenceEmbedder
from rag_forge.retrieval.hybrid import hybrid_factory
from rag_forge.retrieval.lexical import Bm25Retriever
from rag_forge.retrieval.rerank import OnnxCrossEncoder
from rag_forge.retrieval.service import RetrievalService, RetrieverFactory
from rag_forge.router import analyzer as qa
from rag_forge.router.analyzer import HeuristicQueryAnalyzer, extract_features, label, normalize
from rag_forge.router.policy import RouterContext, RulePolicy, dense_weight
from rag_forge.router.service import AdaptiveRouter, RouterComponentNotAvailableError
from rag_forge.storage.lexical_index import SqliteLexicalIndex
from rag_forge.storage.sqlite import SqliteStore
from rag_forge.storage.vector_index import SqliteVectorIndex
from tests.test_rerank import KeywordReranker

ANALYZER = HeuristicQueryAnalyzer()
POLICY = RulePolicy()
S, D, H = RetrievalStrategy.SPARSE, RetrievalStrategy.DENSE, RetrievalStrategy.HYBRID


def analyze(q: str, corpus: CorpusQuerySignals | None = None) -> QueryAnalysis:
    return ANALYZER.analyze(q, corpus)


# --- query intelligence -------------------------------------------------------------------


def test_normalization_and_determinism() -> None:
    a = analyze("  What  is \u201cRRF\u201d\uff1f ")  # curly quotes, full-width question mark
    b = analyze('What is "RRF"?')
    assert a.normalized_query == b.normalized_query == 'What is "RRF"?'
    assert a.query != b.query and a.analysis_hash == b.analysis_hash  # same analysis
    assert analyze("BM25 k1") == analyze("BM25 k1")  # bit-identical, no timestamps
    assert analyze("BM25 k1").analysis_hash != analyze("BM25 k2").analysis_hash


def test_feature_extraction() -> None:
    f = extract_features(
        normalize(
            'Does BM25 in docs/retrieval.md use k1=1.2 and "length normalisation", unlike '
            "Dense Retrieval or snake_case_names? Lucene doesn't, e.g. since 2019."
        )
    )
    assert f.identifiers == ["BM25", "docs/retrieval.md", "k1=1.2", "snake_case_names"]
    assert "e.g" not in f.identifiers and "2019" not in f.identifiers
    assert f.numbers == ["2019"]
    assert f.quoted_phrases == ["length normalisation"]
    assert f.capitalized_terms == ["Dense Retrieval"]  # "Does" and "Lucene" start sentences
    assert f.entities[:2] == ["length normalisation", "BM25"]
    assert f.question_word == "does" and f.is_question
    assert "2019" in f.temporal_markers and "since" in f.temporal_markers
    assert f.negation_markers == ["doesn't"]
    segments = extract_features("dense retrieval and BM25, or reranking").concept_segments
    assert segments == ["dense retrieval", "bm25", "reranking"]  # split on and / , / or


def test_terms_and_function_words() -> None:
    f = extract_features("what do plants need to make food")
    assert f.bm25_terms == ["what", "do", "plants", "need", "make", "food"]  # BM25 keeps these
    assert f.key_terms == ["plants", "need", "make", "food"]
    assert f.function_word_ratio == round(3 / 7, 4)  # what, do, to
    empty = extract_features("")
    assert empty.token_count == 0 and empty.function_word_ratio == 0 and not empty.is_question


@pytest.mark.parametrize(
    ("query", "kind"),
    [
        ("What is RRF?", QuestionType.DEFINITION),
        ("define reciprocal rank fusion", QuestionType.DEFINITION),
        ("What are the limits of hybrid retrieval?", QuestionType.LIST),
        ("examples of lexical queries", QuestionType.LIST),
        ("how to build a dense index", QuestionType.PROCEDURAL),
        ("How do I rebuild the index?", QuestionType.PROCEDURAL),
        ("why does hybrid never fall back", QuestionType.EXPLANATORY),
        ("how does reranking work", QuestionType.EXPLANATORY),
        ("BM25 vs dense", QuestionType.COMPARISON),
        ("difference between RRF and weighted fusion", QuestionType.COMPARISON),
        ("is BM25 deterministic", QuestionType.BOOLEAN),
        ("how many chunks per version", QuestionType.FACTOID),
        ("who wrote the BM25 paper", QuestionType.FACTOID),
        ("BM25 k1 saturation", QuestionType.KEYWORD),
    ],
)
def test_question_types(query: str, kind: QuestionType) -> None:
    assert extract_features(normalize(query)).question_type is kind


def test_signals_are_exact_sums_of_their_contributions() -> None:
    for name, weights in qa.CONFIG["signals"].items():
        assert sum(weights.values()) == pytest.approx(1.0), name
    for q in ("BM25 k1", "why does dense retrieval need an index", 'compare "RRF" and BM25'):
        for s in analyze(q).signals:
            total = round(min(1.0, max(0.0, sum(c.contribution for c in s.contributions))), 4)
            assert s.score == total and 0 <= s.score <= 1
            assert all(c.contribution == round(c.value * c.weight, 4) for c in s.contributions)


def _signals(lex: float, sem: float, cx: float) -> list[QuerySignal]:
    return [QuerySignal(name=n, score=v, contributions=[]) for n, v in
            (("lexical", lex), ("semantic", sem), ("complexity", cx))]  # fmt: skip


def test_label_boundaries() -> None:
    f = extract_features("plants food")
    assert label(f, _signals(0.65, 0.5, 0)).query_class is QueryClass.LEXICAL  # margin = 0.15
    assert label(f, _signals(0.6499, 0.5, 0)).query_class is QueryClass.MIXED
    assert label(f, _signals(0.35, 0.5, 0)).query_class is QueryClass.SEMANTIC  # margin = -0.15
    assert label(f, _signals(0, 0, 0.2499)).complexity is Complexity.SIMPLE
    assert label(f, _signals(0, 0, 0.25)).complexity is Complexity.MODERATE
    assert label(f, _signals(0, 0, 0.5)).complexity is Complexity.COMPLEX
    assert label(f, _signals(0, 0, 0.5)).evidence_need is EvidenceNeed.MULTIPLE_PASSAGES


def test_labels_on_real_queries() -> None:
    lookup = analyze("BM25 k1 parameter").labels
    assert lookup.query_class is QueryClass.LEXICAL and lookup.complexity is Complexity.SIMPLE
    assert not lookup.ambiguous and lookup.evidence_need is EvidenceNeed.SINGLE_PASSAGE
    semantic = analyze("what do plants need to make food").labels
    assert semantic.query_class is QueryClass.SEMANTIC
    hop = analyze(
        "Compare dense retrieval and BM25 for rare identifiers, and how does reranking affect both?"
    ).labels
    assert hop.multi_hop_likely and hop.complexity is Complexity.COMPLEX
    assert hop.evidence_need is EvidenceNeed.MULTIPLE_PASSAGES
    assert analyze("how does it work").labels.ambiguous  # vague referent, no entity
    assert not analyze("RAG_FORGE_DATA_DIR").labels.ambiguous  # one term, but an exact anchor


def test_config_hash_tracks_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    before = ANALYZER.config_hash()
    monkeypatch.setitem(qa.CONFIG, "class_margin", 0.2)
    assert ANALYZER.config_hash() != before


# --- policy -------------------------------------------------------------------------------


def decide(q: str, corpus: CorpusQuerySignals | None = None, **ctx: Any) -> Any:
    return POLICY.decide(analyze(q, corpus), RouterContext(**{"top_k": 10, "reranker": "m", **ctx}))


def test_policy_routes_by_measured_class() -> None:
    lookup = decide("BM25 k1 parameter")
    assert lookup.option is RouteOption.SPARSE and not lookup.rerank.enabled
    assert [r.rule for r in lookup.rules] == [
        "no-searchable-terms",
        "vocabulary-mismatch",
        "exact-lookup",
    ]
    assert [r.matched for r in lookup.rules] == [False, False, True]  # stops at the first match

    dense = decide("what do plants need to make food")
    assert dense.option is RouteOption.DENSE and dense.hybrid is None
    assert dense.rerank.enabled and dense.rerank.candidate_k == 20  # simple → 20 candidates

    anchored = decide("Why does hybrid retrieval never fall back to BM25 alone?")
    assert anchored.option is RouteOption.HYBRID_WEIGHTED and anchored.hybrid is not None
    w = anchored.hybrid.weights
    assert w[D] > w[S] and w[D] + w[S] == pytest.approx(1)

    lexical_complex = decide('"BM25" k1 "saturation" b normalisation and RRF k parameter tuning')
    assert lexical_complex.option is RouteOption.HYBRID_WEIGHTED
    assert lexical_complex.hybrid.weights[S] > lexical_complex.hybrid.weights[D]

    assert decide("the of and").option is RouteOption.DENSE  # nothing BM25 can search
    assert decide("the of and").rules[0].matched


def test_policy_mixed_routes_to_rrf() -> None:
    q = "BM25 k1 for long documents?"  # an exact anchor, phrased as a question
    labels = analyze(q).labels
    assert labels.query_class is QueryClass.MIXED and abs(labels.class_margin) < 0.15
    plan = decide(q)
    assert plan.option is RouteOption.HYBRID_RRF and plan.rules[-1].rule == "balanced"
    assert plan.margin == pytest.approx(round(0.15 - abs(labels.class_margin), 4))
    assert plan.hybrid is not None and plan.hybrid.weights  # carried but unused by RRF


def test_vocabulary_mismatch_boundary() -> None:
    def signals(coverage: float) -> CorpusQuerySignals:
        return CorpusQuerySignals(
            corpus_id="c", corpus_version=1, analyzer="a", chunk_count=10,
            terms=[], coverage=coverage, missing_terms=["x"], mean_idf=None,
        )  # fmt: skip

    q = "BM25 k1 parameter"
    base = analyze(q)
    for coverage, mismatch in ((0.49, True), (0.5, False)):
        c = CorpusQuerySignals.model_validate(
            {**signals(coverage).model_dump(), "terms": [
                {"term": t, "document_frequency": 1, "idf": 1.0} for t in base.features.bm25_terms
            ]}
        )  # fmt: skip
        plan = decide(q, c)
        assert (plan.option is RouteOption.DENSE) is mismatch, coverage
        assert plan.rules[1].inputs["coverage"] == coverage
        assert plan.rules[1].margin == pytest.approx(abs(coverage - 0.5))


def test_unavailable_options_are_constrained_explicitly() -> None:
    reason = "the dense index for v1 is missing"
    plan = decide(
        "what do plants need to make food",
        unavailable={o: reason for o in RouteOption if o is not RouteOption.SPARSE},
    )
    assert plan.preferred is RouteOption.DENSE and plan.option is RouteOption.SPARSE
    assert (
        plan.rules[-1].rule == "constraint-unavailable"
        and plan.rules[-1].inputs["reason"] == reason
    )
    assert any("dense preferred but" in r for r in plan.rationale)
    none = decide("what do plants need to make food", reranker=None)
    assert not none.rerank.enabled and none.rerank_rules[0].rule == "rerank-unavailable"


def test_pool_sizes_respect_top_k() -> None:
    plan = decide("Why does hybrid retrieval never fall back to BM25 alone?", top_k=80)
    assert plan.rerank.enabled and plan.rerank.candidate_k == 80  # table says 20; top_k wins
    assert plan.hybrid is not None and plan.hybrid.candidate_k >= plan.rerank.candidate_k


def test_dense_weight_grid_and_clamp() -> None:
    assert dense_weight(0.0) == 0.5
    assert dense_weight(-0.27) == 0.75  # 0.77 on the 0.05 grid
    assert dense_weight(-1.0) == 0.8 and dense_weight(1.0) == 0.2


def test_policy_is_deterministic_and_hash_tracks_rules() -> None:
    q = "Why does hybrid retrieval never fall back to BM25 alone?"
    assert decide(q) == decide(q)
    assert POLICY.config_hash() == RulePolicy().config_hash()


# --- through the real retrieval stack -----------------------------------------------------

DOCS = [
    ("trucks.txt", b"Food trucks need permits to make and sell food, and food safety checks."),
    ("photo.txt", b"Through photosynthesis, plants use sunlight, water and CO2 to produce food."),
    ("bm25.txt", b"BM25 k1 controls term frequency saturation; b controls length normalisation."),
    ("cars.txt", b"Automobiles need regular oil changes and tyre rotations to stay reliable."),
]


def make(
    store: SqliteStore, embedder: OnnxSentenceEmbedder, router: AdaptiveRouter | None = None
) -> tuple[RetrievalService, DenseIndexService]:
    lexical = SqliteLexicalIndex(store)
    dense = DenseIndexService(SqliteVectorIndex(store), embedder)
    single: dict[RetrievalStrategy, RetrieverFactory] = {
        S: lambda r: Bm25Retriever(lexical, r.bm25),
        D: lambda r: DenseRetriever(dense),
    }
    reranker = KeywordReranker()
    svc = RetrievalService(
        store,
        {**single, H: hybrid_factory(single)},
        embedder.spec,
        {reranker.spec.model: reranker},
        router or make_router(lexical, dense, [reranker.spec.model]),
    )
    return svc, dense


@pytest.fixture
def corpus(store: SqliteStore, service: IngestionService) -> Corpus:
    c = store.add_corpus(Corpus(name="router"))
    service.ingest(c.id, DOCS)
    found = store.get_corpus(c.id)
    assert found is not None
    return found


def ask(svc: RetrievalService, corpus_id: str, query: str, **kw: Any) -> Any:
    return svc.retrieve(corpus_id, RetrievalRequest.model_validate({"query": query, **kw}))


def test_adaptive_equals_the_manual_run_it_selected(
    store: SqliteStore, corpus: Corpus, embedder: OnnxSentenceEmbedder
) -> None:
    svc, dense = make(store, embedder)
    index, _ = dense.start(corpus, 1)
    dense.build(index, corpus)
    for q in ("BM25 k1 parameter", "what do plants need to make food",
              "Why does BM25 controls saturation matter for food trucks?"):  # fmt: skip
        adaptive = ask(svc, corpus.id, q, mode="adaptive", top_k=3)
        routing = adaptive.provenance.routing
        assert routing is not None
        _, routed, _ = svc.route(
            corpus.id, RetrievalRequest(query=q, top_k=3, mode=RetrievalMode.ADAPTIVE)
        )
        manual = svc.retrieve(corpus.id, routed)
        assert routed.mode is RetrievalMode.MANUAL and manual.provenance.routing is None
        assert [(h.result.chunk_id, h.result.score) for h in adaptive.hits] == [
            (h.result.chunk_id, h.result.score) for h in manual.hits
        ]
        decision = routing.decision
        assert decision.configuration_hash == manual.provenance.configuration_hash
        assert adaptive.provenance.configuration_hash != manual.provenance.configuration_hash
        assert adaptive.provenance.configuration.routing == routing.routing
        assert decision.query_id == adaptive.query.id and decision.option is decision.preferred
        assert decision.strategy is adaptive.provenance.strategy
        assert (adaptive.provenance.reranking is not None) is decision.rerank.enabled
    lookup = ask(svc, corpus.id, "BM25 k1 parameter", mode="adaptive").provenance.routing.decision
    assert lookup.option is RouteOption.SPARSE  # dense is ready, so no constraint was involved
    semantic = ask(svc, corpus.id, "what do plants need to make food", mode="adaptive")
    assert semantic.provenance.routing.decision.option is RouteOption.DENSE
    assert semantic.provenance.strategy is D and semantic.hits[0].filename == "photo.txt"


def test_routing_provenance_is_complete_and_deterministic(
    store: SqliteStore, corpus: Corpus, embedder: OnnxSentenceEmbedder
) -> None:
    svc, _ = make(store, embedder)  # no dense index: every dense preference is constrained
    a = ask(svc, corpus.id, "what do plants need to make food", mode="adaptive")
    b = ask(svc, corpus.id, "what do plants need to make food", mode="adaptive")
    ra, rb = a.provenance.routing, b.provenance.routing
    assert ra is not None and rb is not None
    assert ra.analysis == rb.analysis and ra.routing_hash == rb.routing_hash
    assert ra.decision.decision_hash == rb.decision.decision_hash
    assert (
        ra.decision.id != rb.decision.id
        and a.provenance.configuration_hash == b.provenance.configuration_hash
    )
    assert ra.routing.analyzer_version == ANALYZER.version
    assert ra.routing.policy_version == POLICY.version
    assert ra.routing.policy_config_hash == POLICY.config_hash()
    assert ra.decision.analysis_hash == ra.analysis.analysis_hash
    assert ra.decision.preferred is RouteOption.DENSE and ra.decision.option is RouteOption.SPARSE
    alts = {x.option: x for x in ra.decision.alternatives}
    assert alts[RouteOption.SPARSE].selected and alts[RouteOption.SPARSE].available
    assert not alts[RouteOption.DENSE].available and "missing" in (
        alts[RouteOption.DENSE].unavailable_reason or ""
    )
    assert alts[RouteOption.DENSE].rules == [
        "no-searchable-terms",
        "vocabulary-mismatch",
        "semantic-unanchored",
    ]
    corpus_signals = ra.analysis.corpus
    assert corpus_signals is not None and corpus_signals.corpus_version == 1
    assert corpus_signals.chunk_count == len(DOCS) and corpus_signals.coverage == 1.0
    assert ra.analysis_ms >= 0 and ra.decision_ms >= 0
    rerank = a.provenance.reranking  # the router enabled reranking; its provenance is attached
    assert rerank is not None and rerank.upstream_configuration.routing is None


def test_manual_requests_are_unchanged(
    store: SqliteStore, corpus: Corpus, embedder: OnnxSentenceEmbedder
) -> None:
    svc, _ = make(store, embedder)
    plain = ask(svc, corpus.id, "food")
    assert plain.provenance.routing is None and plain.provenance.configuration.routing is None
    legacy = canonical_hash(
        plain.provenance.configuration.model_dump(mode="json", exclude={"rerank", "routing"})
    )
    assert plain.provenance.configuration_hash == legacy
    explicit = ask(svc, corpus.id, "food", mode="manual", router={"policy": "nope"})
    assert explicit.provenance.configuration_hash == legacy  # router params unused in manual mode


def test_corpus_signals_are_version_scoped(
    store: SqliteStore, corpus: Corpus, service: IngestionService, embedder: OnnxSentenceEmbedder
) -> None:
    svc, _ = make(store, embedder)
    service.ingest(corpus.id, [("zebra.txt", b"Zebra migration patterns follow seasonal rains.")])
    q = "zebra migration patterns"
    v1 = ask(svc, corpus.id, q, mode="adaptive", version=1).provenance.routing
    v2 = ask(svc, corpus.id, q, mode="adaptive", version=2).provenance.routing
    assert v1.analysis.corpus.coverage == 0.0 and v1.analysis.corpus.missing_terms == q.split()
    assert v2.analysis.corpus.coverage == 1.0 and v2.analysis.corpus.chunk_count == len(DOCS) + 1
    assert v1.decision.preferred is RouteOption.DENSE  # vocabulary-mismatch fires on v1 only
    assert v1.decision.rules[1].matched and not v2.decision.rules[1].matched
    assert v2.decision.preferred is RouteOption.SPARSE
    assert v1.analysis.analysis_hash != v2.analysis.analysis_hash


def test_router_failures_are_explicit(
    store: SqliteStore, corpus: Corpus, embedder: OnnxSentenceEmbedder
) -> None:
    svc, dense = make(store, embedder)
    with pytest.raises(RouterComponentNotAvailableError, match="router policy 'nope'"):
        ask(svc, corpus.id, "food", mode="adaptive", router={"policy": "nope"})
    with pytest.raises(RouterComponentNotAvailableError, match="query analyzer 'nope'"):
        ask(svc, corpus.id, "food", mode="adaptive", router={"analyzer": "nope"})
    no_router = RetrievalService(store, svc.retrievers, embedder.spec)
    with pytest.raises(RouterComponentNotAvailableError, match="router"):
        ask(no_router, corpus.id, "food", mode="adaptive")

    # availability that claims a ready dense index: the routed run fails, it does not degrade
    lexical = SqliteLexicalIndex(store)
    lying = make_router(lexical, dense, [])
    lying.availability = lambda corpus, version: {}
    svc2, _ = make(store, embedder, lying)
    with pytest.raises(DenseIndexNotReadyError):
        ask(svc2, corpus.id, "what do plants need to make food", mode="adaptive")

    class BrokenPolicy(RulePolicy):  # selects a pool smaller than the final top-k
        def decide(self, analysis: QueryAnalysis, context: RouterContext) -> Any:
            plan = super().decide(analysis, context)
            bad = plan.rerank.model_copy(update={"enabled": True, "candidate_k": 1})
            return type(plan)(**{**plan.__dict__, "rerank": bad})

    broken = make_router(lexical, dense, ["cross-encoder/ms-marco-MiniLM-L-6-v2"])
    broken.policies["rules-baseline"] = BrokenPolicy()
    svc3, _ = make(store, embedder, broken)
    with pytest.raises(ValidationError, match="candidate_k must be at least top_k"):
        ask(svc3, corpus.id, "food", mode="adaptive", top_k=5)


# --- API ----------------------------------------------------------------------------------


def test_router_api(
    tmp_path: Path, embedder: OnnxSentenceEmbedder, cross_encoder: OnnxCrossEncoder
) -> None:
    with TestClient(create_app(tmp_path, embedder, cross_encoder)) as c:
        cid = c.post("/api/v1/corpora", json={"name": "api"}).json()["corpus"]["id"]
        c.post(f"/api/v1/corpora/{cid}/documents", files=[("files", d) for d in DOCS])
        assert c.post(f"/api/v1/corpora/{cid}/dense-index", json={}).status_code == 202
        url = f"/api/v1/corpora/{cid}/retrieve"
        q = "what do plants need to make food"

        decided = c.post("/api/v1/router/decide", json={"corpus_id": cid, "query": q, "top_k": 2})
        assert decided.status_code == 200, decided.text
        body = decided.json()
        decision = body["routing"]["decision"]
        assert decision["option"] == "dense" and decision["rerank"]["enabled"]
        assert body["request"]["mode"] == "manual" and body["request"]["strategy"] == "dense"

        adaptive = c.post(url, json={"query": q, "mode": "adaptive", "top_k": 2}).json()
        replay = c.post(url, json=body["request"]).json()  # the decide output, sent as-is
        assert [h["result"]["chunk_id"] for h in adaptive["hits"]] == [
            h["result"]["chunk_id"] for h in replay["hits"]
        ]
        assert adaptive["hits"][0]["filename"] == "photo.txt"  # real dense + real cross-encoder
        assert adaptive["provenance"]["reranking"]["reranker"] == "cross-encoder"
        assert replay["provenance"]["configuration_hash"] == decision["configuration_hash"]

        for bad, code in (
            ({"query": q, "mode": "routed"}, 422),
            ({"query": q, "mode": "adaptive", "router": {"policy": "nope"}}, 501),
            ({"query": q, "mode": "adaptive", "router": {"analyzer": "nope"}}, 501),
            ({"query": q, "mode": "adaptive", "router": {"extra": 1}}, 422),
            ({"query": q, "mode": "adaptive", "version": 7}, 404),
        ):
            r = c.post(url, json=bad)
            assert r.status_code == code, (bad, r.text)
            if code == 501:
                assert r.json()["detail"]["capability"] == "Router"
        decide_url = "/api/v1/router/decide"
        assert c.post(decide_url, json={"corpus_id": "nope", "query": q}).status_code == 404
        assert c.post(decide_url, json={"corpus_id": cid, "query": "  "}).status_code == 422
        assert (
            c.post(decide_url, json={"corpus_id": cid, "query": q, "version": 9}).status_code == 404
        )
        unknown = c.post(decide_url, json={"corpus_id": cid, "query": q, "router": {"policy": "x"}})
        assert unknown.status_code == 501
