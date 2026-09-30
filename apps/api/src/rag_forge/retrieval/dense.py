"""Dense retrieval: index lifecycle for a corpus version, and the retriever that searches it.

Indexes are built explicitly (they cost real compute), never lazily during a query, and a query
against a version without a ready index fails with that index's state. There is no fallback to
BM25: a dense result is always a dense result.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from rag_forge.domain.models import (
    Corpus,
    DenseIndex,
    DenseIndexState,
    DenseIndexStatus,
    RetrievalStrategy,
    utcnow,
)
from rag_forge.retrieval.base import Candidate, RetrieverOutput
from rag_forge.retrieval.embedding import Embedder, Vectors
from rag_forge.storage.vector_index import SqliteVectorIndex

BUILD_BATCH = 64


class DenseIndexNotReadyError(LookupError):
    def __init__(self, state: DenseIndexState, version: int) -> None:
        self.state = state
        hint = {
            DenseIndexState.BUILDING: "it is still building",
            DenseIndexState.FAILED: "its last build failed; rebuild it",
            DenseIndexState.STALE: "only other versions or embedder configs are indexed; build it",
            DenseIndexState.MISSING: "build it first",
        }[state]
        super().__init__(f"no ready dense index for corpus version {version}: {hint}")


def _unit(vectors: Vectors) -> Vectors:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)


class DenseIndexService:
    def __init__(self, vectors: SqliteVectorIndex, embedder: Embedder) -> None:
        self.vectors = vectors
        self.embedder = embedder
        self.embedder_hash = embedder.spec.config_hash()
        # ponytail: one build at a time per process; a job queue when builds get long or many.
        self._lock = threading.Lock()
        vectors.fail_interrupted()

    def state(self, corpus: Corpus, version: int) -> tuple[DenseIndexState, DenseIndex | None]:
        latest = self.vectors.latest(corpus.id, version, self.embedder_hash)
        if latest is not None and latest.status is DenseIndexStatus.READY:
            return DenseIndexState.READY, latest
        if latest is not None and latest.status is DenseIndexStatus.BUILDING:
            return DenseIndexState.BUILDING, latest
        if latest is not None:
            return DenseIndexState.FAILED, latest
        if self.vectors.has_other_ready(corpus.id, version, self.embedder_hash):
            return DenseIndexState.STALE, None
        return DenseIndexState.MISSING, None

    def ready_index(self, corpus: Corpus, version: int) -> DenseIndex:
        state, index = self.state(corpus, version)
        if index is None or state is not DenseIndexState.READY:
            raise DenseIndexNotReadyError(state, version)
        return index

    def start(self, corpus: Corpus, version: int) -> tuple[DenseIndex, bool]:
        """Create a build record, or return the ready/building one. (index, needs_build)."""
        with self._lock:
            state, current = self.state(corpus, version)
            if current is not None and state in (DenseIndexState.READY, DenseIndexState.BUILDING):
                return current, False
            info = self.embedder.info()  # EmbedderUnavailableError propagates: nothing to record
            members = self.vectors.member_chunks(corpus, version)
            index = DenseIndex(
                corpus_id=corpus.id,
                corpus_version=version,
                chunking_hash=corpus.chunking.config_hash(),
                embedder=info,
                chunk_count=len(members),
            )
            self.vectors.save(index)
            return index, True

    def build(self, index: DenseIndex, corpus: Corpus) -> DenseIndex:
        """Embed missing chunks in chunk-id order, then seal the index. Records failure."""
        try:
            members = self.vectors.member_chunks(corpus, index.corpus_version)
            ids = [cid for cid, _ in members]
            have = self.vectors.existing(self.embedder_hash, ids)
            todo = [(cid, text) for cid, text in members if cid not in have]
            index = index.model_copy(update={"reused": len(have), "embedded": len(have)})
            self.vectors.save(index)
            for start in range(0, len(todo), BUILD_BATCH):
                batch = todo[start : start + BUILD_BATCH]
                vecs = _unit(self.embedder.embed_documents([text for _, text in batch]))
                self.vectors.put(self.embedder_hash, [cid for cid, _ in batch], vecs)
                index = index.model_copy(update={"embedded": index.embedded + len(batch)})
                self.vectors.save(index)
            index = index.model_copy(
                update={
                    "status": DenseIndexStatus.READY,
                    "content_hash": self.vectors.content_hash(index),
                    "finished_at": utcnow(),
                }
            )
        except Exception as exc:  # recorded on the index, surfaced as state "failed"
            index = index.model_copy(
                update={
                    "status": DenseIndexStatus.FAILED,
                    "error": str(exc),
                    "finished_at": utcnow(),
                }
            )
        self.vectors.save(index)
        return index


class DenseRetriever:
    name = "dense"
    strategy = RetrievalStrategy.DENSE

    def __init__(self, indexes: DenseIndexService) -> None:
        self.indexes = indexes

    def config(self) -> dict[str, Any]:
        info = self.indexes.embedder.info()
        return {
            "model": info.spec.model,
            "revision": info.spec.revision,
            "provider": info.spec.provider,
            "dimension": info.dimension,
            "pooling": info.pooling,
            "normalize": info.normalize,
            "max_seq_length": info.max_seq_length,
            "query_prefix": info.spec.query_prefix,
            "weights_sha256": info.weights_sha256,
            "embedder_hash": info.config_hash,
            "similarity": "cosine",
        }

    def retrieve(self, corpus: Corpus, version: int, query: str, top_k: int) -> RetrieverOutput:
        index = self.indexes.ready_index(corpus, version)
        started = time.perf_counter()
        vector = _unit(self.indexes.embedder.embed_query(query))
        embed_ms = (time.perf_counter() - started) * 1000
        ranked = self.indexes.vectors.search(index, vector, top_k)
        statistics = {
            "candidate_chunks": float(index.chunk_count),
            "query_embedding_ms": round(embed_ms, 3),
        }
        warnings = ["the dense index for this version is empty"] if index.chunk_count == 0 else []
        return RetrieverOutput(
            [Candidate(cid, score) for cid, score in ranked], [], statistics, warnings, index.id
        )
