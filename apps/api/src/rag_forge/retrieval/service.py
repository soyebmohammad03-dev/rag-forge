"""Query -> corpus version -> (router) -> retriever -> (reranker) -> hits + provenance."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from rag_forge.domain.models import (
    Chunk,
    Corpus,
    DocumentVersion,
    EmbedderSpec,
    Query,
    RankMovement,
    RerankCandidate,
    RerankConfiguration,
    RerankDetail,
    RerankProvenance,
    RerankReport,
    RetrievalConfiguration,
    RetrievalHit,
    RetrievalMode,
    RetrievalProvenance,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
    RetrievalStrategy,
    RoutingConfiguration,
    RoutingProvenance,
    canonical_hash,
)
from rag_forge.ingestion.service import CorpusNotFoundError
from rag_forge.provenance.environment import capture_environment
from rag_forge.retrieval.base import Candidate, Retriever, RetrieverOutput
from rag_forge.retrieval.rerank import Reranker, rank_movement
from rag_forge.router.service import AdaptiveRouter, RouterComponentNotAvailableError
from rag_forge.storage.base import CorpusStore

RetrieverFactory = Callable[[RetrievalRequest], Retriever]


class CorpusVersionNotFoundError(LookupError):
    pass


class ForeignResultError(RuntimeError):
    """A retriever returned chunks that are not in the requested corpus version."""


class StrategyNotAvailableError(LookupError):
    def __init__(self, strategy: RetrievalStrategy, available: list[RetrievalStrategy]) -> None:
        names = ", ".join(sorted(available))
        super().__init__(
            f"retrieval strategy '{strategy}' is not implemented yet (available: {names})"
        )


class RerankerNotAvailableError(LookupError):
    def __init__(self, model: str, available: list[str]) -> None:
        names = ", ".join(sorted(available)) or "none"
        super().__init__(f"reranker '{model}' is not registered (available: {names})")


class RetrievalService:
    def __init__(
        self,
        store: CorpusStore,
        retrievers: Mapping[RetrievalStrategy, RetrieverFactory],
        embedder: EmbedderSpec | None = None,
        rerankers: Mapping[str, Reranker] | None = None,
        router: AdaptiveRouter | None = None,
    ) -> None:
        self.store = store
        self.retrievers = retrievers
        self.embedder = embedder  # recorded in configurations that involve dense retrieval
        self.rerankers = dict(rerankers or {})  # keyed by model id; instances hold loaded models
        self.router = router

    def _resolve(self, corpus_id: str, request: RetrievalRequest) -> tuple[Corpus, int]:
        corpus = self.store.get_corpus(corpus_id)
        if corpus is None:
            raise CorpusNotFoundError(corpus_id)
        version = corpus.version if request.version is None else request.version
        if version > corpus.version:
            raise CorpusVersionNotFoundError(f"corpus has no version {version}")
        return corpus, version

    def route(
        self, corpus_id: str, request: RetrievalRequest, query: Query | None = None
    ) -> tuple[Query, RetrievalRequest, RoutingProvenance]:
        """The router's decision for a request, without retrieving anything."""
        corpus, version = self._resolve(corpus_id, request)
        query = query or Query(text=request.query)
        if self.router is None:
            raise RouterComponentNotAvailableError("router", "adaptive", [])
        routed, routing = self.router.route(
            corpus, version, request, query.id, lambda r: self._configuration(r, corpus, version)
        )
        return query, routed, routing

    def retrieve(self, corpus_id: str, request: RetrievalRequest) -> RetrievalResponse:
        started = time.perf_counter()
        corpus, version = self._resolve(corpus_id, request)
        query = Query(text=request.query)
        routing: RoutingProvenance | None = None
        if request.mode is RetrievalMode.ADAPTIVE:  # from here on, an ordinary manual request
            _, request, routing = self.route(corpus_id, request, query)
        factory = self.retrievers.get(request.strategy)
        if factory is None:
            raise StrategyNotAvailableError(request.strategy, list(self.retrievers))
        reranker = None
        if request.rerank.enabled:  # resolved before retrieval: never retrieve and then degrade
            reranker = self.rerankers.get(request.rerank.model)
            if reranker is None:
                raise RerankerNotAvailableError(request.rerank.model, list(self.rerankers))

        retriever = factory(request)
        output = retriever.retrieve(corpus, version, request.query, request.pool_k)
        chunks = self.store.get_chunks([c.chunk_id for c in output.candidates])
        members = {dv.id for dv in self.store.members(corpus.id, version).values()}
        chash = corpus.chunking.config_hash()
        foreign = [
            c.chunk_id
            for c in output.candidates
            if c.chunk_id not in chunks
            or chunks[c.chunk_id].document_version_id not in members
            or chunks[c.chunk_id].chunking_hash != chash
        ]
        if foreign:  # a retriever returned something outside the requested version: never serve it
            raise ForeignResultError(
                f"{retriever.name} returned {len(foreign)} chunk(s) outside corpus version "
                f"{version} (e.g. {foreign[0]})"
            )
        versions: dict[str, DocumentVersion] = {}

        def version_of(chunk_id: str) -> DocumentVersion:
            dv_id = chunks[chunk_id].document_version_id
            if dv_id not in versions:
                found = self.store.get_document_version(dv_id)
                assert found is not None, f"chunk {chunk_id} references missing version {dv_id}"
                versions[dv_id] = found
            return versions[dv_id]

        configuration = self._configuration(
            request, corpus, version, routing.routing if routing else None
        )
        ranked: list[tuple[Candidate, float, RerankDetail | None]] = [
            (c, c.score, None) for c in output.candidates
        ]
        report = rerank_provenance = None
        if reranker is not None:
            ranked, report, rerank_provenance = self._rerank(
                reranker, request, configuration, output, chunks, version_of
            )

        hits = []
        for rank, (candidate, score, detail) in enumerate(ranked, 1):
            chunk = chunks[candidate.chunk_id]
            dv = version_of(chunk.id)
            hits.append(
                RetrievalHit(
                    result=RetrievalResult(
                        query_id=query.id,
                        strategy=retriever.strategy,
                        retriever=retriever.name,
                        chunk_id=chunk.id,
                        document_id=dv.document_id,
                        document_version_id=dv.id,
                        rank=rank,
                        score=score,
                    ),
                    chunk=chunk,
                    filename=dv.filename,
                    media_type=dv.media_type,
                    document_version=dv.version,
                    matched_terms=list(candidate.matched_terms),
                    fusion=candidate.fusion,
                    rerank=detail,
                )
            )

        config = retriever.config()
        provenance = RetrievalProvenance(
            corpus_id=corpus.id,
            corpus_version=version,
            chunking_hash=corpus.chunking.config_hash(),
            strategy=retriever.strategy,
            retriever=retriever.name,
            retriever_config=config,
            retriever_config_hash=canonical_hash({"retriever": retriever.name, **config}),
            configuration=configuration,
            configuration_hash=configuration.config_hash(),
            index_id=output.index_id,
            query_terms=output.query_terms,
            statistics=output.statistics,
            environment=capture_environment(),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            reranking=rerank_provenance,
            routing=routing,
        )
        return RetrievalResponse(
            query=query,
            hits=hits,
            provenance=provenance,
            warnings=output.warnings,
            reranking=report,
        )

    def _rerank(
        self,
        reranker: Reranker,
        request: RetrievalRequest,
        configuration: RetrievalConfiguration,
        output: RetrieverOutput,
        chunks: Mapping[str, Chunk],
        version_of: Callable[[str], DocumentVersion],
    ) -> tuple[list[tuple[Candidate, float, RerankDetail | None]], RerankReport, RerankProvenance]:
        """Score the whole upstream pool, then keep the final top-k. Failures propagate."""
        info = reranker.info()  # loads the model once; its failure fails the request
        pool = output.candidates
        started = time.perf_counter()
        scores = (
            reranker.score(request.query, [chunks[c.chunk_id].text for c in pool]) if pool else []
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        moved = rank_movement(pool, scores, request.top_k)
        by_id = {c.chunk_id: c for c in pool}
        ranked: list[tuple[Candidate, float, RerankDetail | None]] = [
            (by_id[cid], d.reranker_score, d) for cid, d in moved[: request.top_k]
        ]
        report = RerankReport(
            candidates=[
                RerankCandidate(
                    chunk_id=cid,
                    document_id=version_of(cid).document_id,
                    filename=version_of(cid).filename,
                    chunk_ordinal=chunks[cid].ordinal,
                    rerank=d,
                )
                for cid, d in moved
            ]
        )
        details = [d for _, d in moved]
        upstream = configuration.model_copy(
            update={"rerank": None, "routing": None, "top_k": request.pool_k}
        )
        provenance = RerankProvenance(
            reranker=reranker.name,
            info=info,
            reranker_config_hash=canonical_hash(
                {"reranker": reranker.name, **info.model_dump(mode="json")}
            ),
            candidate_k=request.rerank.candidate_k,
            final_top_k=request.top_k,
            candidates_scored=len(pool),
            upstream_configuration=upstream,
            upstream_configuration_hash=upstream.config_hash(),
            latency_ms=latency_ms,
            statistics={
                **{m.value: float(sum(d.movement is m for d in details)) for m in RankMovement},
                "entered_top_k": float(sum(d.entered_top_k for d in details)),
                "left_top_k": float(sum(d.left_top_k for d in details)),
            },
        )
        return ranked, report, provenance

    def _configuration(
        self,
        request: RetrievalRequest,
        corpus: Corpus,
        version: int,
        routing: RoutingConfiguration | None = None,
    ) -> RetrievalConfiguration:
        """The resolved configuration, holding only parameters that affect this strategy."""
        hybrid = request.strategy is RetrievalStrategy.HYBRID
        used = set(request.hybrid.retrievers) if hybrid else {request.strategy}
        return RetrievalConfiguration(
            corpus_id=corpus.id,
            corpus_version=version,
            chunking_hash=corpus.chunking.config_hash(),
            strategy=request.strategy,
            top_k=request.top_k,
            bm25=request.bm25 if RetrievalStrategy.SPARSE in used else None,
            embedder=self.embedder if RetrievalStrategy.DENSE in used else None,
            hybrid=request.hybrid.effective() if hybrid else None,
            rerank=(
                RerankConfiguration(
                    reranker=self.rerankers[request.rerank.model].spec,
                    candidate_k=request.rerank.candidate_k,
                )
                if request.rerank.enabled
                else None
            ),
            routing=routing,
        )
