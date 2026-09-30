"""Query -> corpus/version resolution -> retriever -> normalised hits + provenance."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from rag_forge.domain.models import (
    DocumentVersion,
    Query,
    RetrievalHit,
    RetrievalProvenance,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
    RetrievalStrategy,
    canonical_hash,
)
from rag_forge.ingestion.service import CorpusNotFoundError
from rag_forge.provenance.environment import capture_environment
from rag_forge.retrieval.base import Retriever
from rag_forge.storage.base import CorpusStore

RetrieverFactory = Callable[[RetrievalRequest], Retriever]


class CorpusVersionNotFoundError(LookupError):
    pass


class StrategyNotAvailableError(LookupError):
    def __init__(self, strategy: RetrievalStrategy, available: list[RetrievalStrategy]) -> None:
        names = ", ".join(sorted(available))
        super().__init__(
            f"retrieval strategy '{strategy}' is not implemented yet (available: {names})"
        )


class RetrievalService:
    def __init__(
        self, store: CorpusStore, retrievers: Mapping[RetrievalStrategy, RetrieverFactory]
    ) -> None:
        self.store = store
        self.retrievers = retrievers

    def retrieve(self, corpus_id: str, request: RetrievalRequest) -> RetrievalResponse:
        started = time.perf_counter()
        corpus = self.store.get_corpus(corpus_id)
        if corpus is None:
            raise CorpusNotFoundError(corpus_id)
        version = corpus.version if request.version is None else request.version
        if version > corpus.version:
            raise CorpusVersionNotFoundError(f"corpus has no version {version}")
        factory = self.retrievers.get(request.strategy)
        if factory is None:
            raise StrategyNotAvailableError(request.strategy, list(self.retrievers))

        retriever = factory(request)
        output = retriever.retrieve(corpus, version, request.query, request.top_k)
        query = Query(text=request.query)
        chunks = self.store.get_chunks([c.chunk_id for c in output.candidates])
        versions: dict[str, DocumentVersion] = {}
        hits = []
        for rank, candidate in enumerate(output.candidates, 1):
            chunk = chunks[candidate.chunk_id]
            dv_id = chunk.document_version_id
            if dv_id not in versions:
                found = self.store.get_document_version(dv_id)
                assert found is not None, f"chunk {chunk.id} references missing version {dv_id}"
                versions[dv_id] = found
            dv = versions[dv_id]
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
                        score=candidate.score,
                    ),
                    chunk=chunk,
                    filename=dv.filename,
                    media_type=dv.media_type,
                    document_version=dv.version,
                    matched_terms=list(candidate.matched_terms),
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
            index_id=output.index_id,
            query_terms=output.query_terms,
            statistics=output.statistics,
            environment=capture_environment(),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return RetrievalResponse(
            query=query, hits=hits, provenance=provenance, warnings=output.warnings
        )
