from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_forge import __version__
from rag_forge.api import corpus, routes
from rag_forge.domain.models import EmbedderSpec, RerankerSpec, RetrievalStrategy
from rag_forge.ingestion.service import IngestionService
from rag_forge.retrieval.dense import DenseIndexService, DenseRetriever
from rag_forge.retrieval.embedding import Embedder, OnnxSentenceEmbedder
from rag_forge.retrieval.hybrid import hybrid_factory
from rag_forge.retrieval.lexical import Bm25Retriever
from rag_forge.retrieval.rerank import OnnxCrossEncoder, Reranker
from rag_forge.retrieval.service import RetrievalService, RetrieverFactory
from rag_forge.storage.blobs import BlobStore
from rag_forge.storage.lexical_index import SqliteLexicalIndex
from rag_forge.storage.sqlite import SqliteStore
from rag_forge.storage.vector_index import SqliteVectorIndex


def create_app(
    data_dir: Path | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    """`data_dir` holds the SQLite database and content-addressed blobs.

    Defaults to $RAG_FORGE_DATA_DIR, else ./data relative to the working directory. The embedder
    defaults to the pinned local ONNX model; $RAG_FORGE_EMBEDDER (an EmbedderSpec as JSON)
    selects another sentence-transformers-layout model. The reranker defaults to the pinned local
    ONNX cross-encoder; $RAG_FORGE_RERANKER (a RerankerSpec as JSON) selects another. Models load
    lazily on first use and stay loaded for the life of the process.
    """
    spec_json = os.environ.get("RAG_FORGE_EMBEDDER")
    embedder = embedder or OnnxSentenceEmbedder(
        EmbedderSpec.model_validate_json(spec_json) if spec_json else EmbedderSpec()
    )
    rerank_json = os.environ.get("RAG_FORGE_RERANKER")
    reranker = reranker or OnnxCrossEncoder(
        RerankerSpec.model_validate_json(rerank_json) if rerank_json else RerankerSpec()
    )
    data_dir = data_dir or Path(os.environ.get("RAG_FORGE_DATA_DIR", "data"))
    app = FastAPI(title="RAG FORGE API", version=__version__)
    store = SqliteStore(data_dir / "rag_forge.sqlite3")
    lexical = SqliteLexicalIndex(store)
    app.state.store = store
    app.state.ingestion = IngestionService(store, BlobStore(data_dir / "blobs"))
    dense = DenseIndexService(SqliteVectorIndex(store), embedder)
    app.state.dense = dense
    # New strategies register here; the service, API and UI stay unchanged.
    single: dict[RetrievalStrategy, RetrieverFactory] = {
        RetrievalStrategy.SPARSE: lambda req: Bm25Retriever(lexical, req.bm25),
        RetrievalStrategy.DENSE: lambda req: DenseRetriever(dense),
    }
    app.state.retrieval = RetrievalService(
        store,
        {**single, RetrievalStrategy.HYBRID: hybrid_factory(single)},
        embedder=embedder.spec,
        rerankers={reranker.spec.model: reranker},
    )
    app.state.reranker = reranker
    app.state.started_at = datetime.now(UTC)
    origins = os.environ.get("RAG_FORGE_CORS_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
    )
    app.include_router(routes.router)
    app.include_router(corpus.router)
    return app
