from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_forge import __version__
from rag_forge.api import arena, corpus, routes
from rag_forge.arena.engine import ArenaEngine
from rag_forge.arena.replay import ReplayService
from rag_forge.domain.models import (
    Corpus,
    DenseIndexState,
    EmbedderSpec,
    GeneratorSpec,
    RerankerSpec,
    RetrievalStrategy,
    RouteOption,
)
from rag_forge.ingestion.service import IngestionService
from rag_forge.rag.generation import (
    ExtractiveGenerator,
    Generator,
    OnnxCausalLM,
    OpenAICompatibleGenerator,
)
from rag_forge.rag.grounding import LexicalSemanticVerifier
from rag_forge.rag.service import RagService
from rag_forge.retrieval.dense import DenseIndexService, DenseRetriever
from rag_forge.retrieval.embedding import Embedder, OnnxSentenceEmbedder
from rag_forge.retrieval.hybrid import hybrid_factory
from rag_forge.retrieval.lexical import Bm25Retriever
from rag_forge.retrieval.rerank import OnnxCrossEncoder, Reranker
from rag_forge.retrieval.service import RetrievalService, RetrieverFactory
from rag_forge.router.analyzer import HeuristicQueryAnalyzer
from rag_forge.router.policy import RulePolicy
from rag_forge.router.service import AdaptiveRouter
from rag_forge.storage.arena import ArenaStore
from rag_forge.storage.blobs import BlobStore
from rag_forge.storage.lexical_index import SqliteLexicalIndex
from rag_forge.storage.sqlite import SqliteStore
from rag_forge.storage.vector_index import SqliteVectorIndex


def make_router(
    lexical: SqliteLexicalIndex, dense: DenseIndexService, rerankers: list[str]
) -> AdaptiveRouter:
    """The adaptive router over this app's indexes. Analyzers and policies register here."""

    def term_statistics(
        corpus: Corpus, version: int, terms: list[str]
    ) -> tuple[int, dict[str, int]]:
        lexical.ensure_indexed(corpus, version)
        n = lexical.stats(corpus, version).chunks
        return n, lexical.document_frequencies(corpus, version, terms)

    def availability(corpus: Corpus, version: int) -> dict[RouteOption, str]:
        state, _ = dense.state(corpus, version)
        if state is DenseIndexState.READY:
            return {}
        reason = f"the dense index for v{version} is {state.value}"
        return {o: reason for o in RouteOption if o is not RouteOption.SPARSE}

    return AdaptiveRouter(
        {"heuristic": HeuristicQueryAnalyzer()},
        {"rules-baseline": RulePolicy()},
        term_statistics,
        availability,
        rerankers,
    )


def make_generators(default: Generator) -> dict[str, Generator]:
    """The default local model, the extractive baseline, and an optional HTTP endpoint.

    $RAG_FORGE_OPENAI_BASE_URL and $RAG_FORGE_OPENAI_MODEL register an OpenAI-compatible
    `/chat/completions` endpoint (e.g. a local llama.cpp or Ollama server) as "openai-compatible";
    $RAG_FORGE_OPENAI_API_KEY is sent as a bearer token if set and is never recorded.
    """
    generators: list[Generator] = [default, ExtractiveGenerator()]
    base_url, model = (
        os.environ.get("RAG_FORGE_OPENAI_BASE_URL"),
        os.environ.get("RAG_FORGE_OPENAI_MODEL"),
    )
    if base_url and model:
        generators.append(
            OpenAICompatibleGenerator(base_url, model, os.environ.get("RAG_FORGE_OPENAI_API_KEY"))
        )
    return {g.name: g for g in generators}


def create_app(
    data_dir: Path | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    generator: Generator | None = None,
) -> FastAPI:
    """`data_dir` holds the SQLite database and content-addressed blobs.

    Defaults to $RAG_FORGE_DATA_DIR, else ./data relative to the working directory. The embedder
    defaults to the pinned local ONNX model; $RAG_FORGE_EMBEDDER (an EmbedderSpec as JSON)
    selects another sentence-transformers-layout model. The reranker defaults to the pinned local
    ONNX cross-encoder; $RAG_FORGE_RERANKER (a RerankerSpec as JSON) selects another. The
    generator defaults to the pinned local ONNX chat model; $RAG_FORGE_GENERATOR (a GeneratorSpec
    as JSON) selects another. Models load lazily on first use and stay loaded for the life of the
    process.
    """
    spec_json = os.environ.get("RAG_FORGE_EMBEDDER")
    embedder = embedder or OnnxSentenceEmbedder(
        EmbedderSpec.model_validate_json(spec_json) if spec_json else EmbedderSpec()
    )
    rerank_json = os.environ.get("RAG_FORGE_RERANKER")
    reranker = reranker or OnnxCrossEncoder(
        RerankerSpec.model_validate_json(rerank_json) if rerank_json else RerankerSpec()
    )
    generator_json = os.environ.get("RAG_FORGE_GENERATOR")
    generator = generator or OnnxCausalLM(
        GeneratorSpec.model_validate_json(generator_json) if generator_json else GeneratorSpec()
    )
    data_dir = data_dir or Path(os.environ.get("RAG_FORGE_DATA_DIR", "data"))
    app = FastAPI(title="RAG FORGE API", version=__version__)
    store = SqliteStore(data_dir / "rag_forge.sqlite3")
    lexical = SqliteLexicalIndex(store)
    app.state.store = store
    blobs = BlobStore(data_dir / "blobs")
    app.state.ingestion = IngestionService(store, blobs)
    dense = DenseIndexService(SqliteVectorIndex(store), embedder)
    app.state.dense = dense
    # New strategies register here; the service, API and UI stay unchanged.
    single: dict[RetrievalStrategy, RetrieverFactory] = {
        RetrievalStrategy.SPARSE: lambda req: Bm25Retriever(lexical, req.bm25),
        RetrievalStrategy.DENSE: lambda req: DenseRetriever(dense),
    }
    router = make_router(lexical, dense, [reranker.spec.model])
    app.state.retrieval = RetrievalService(
        store,
        {**single, RetrievalStrategy.HYBRID: hybrid_factory(single)},
        embedder=embedder.spec,
        rerankers={reranker.spec.model: reranker},
        router=router,
    )
    app.state.reranker = reranker
    verifier = LexicalSemanticVerifier(embedder)  # shares the loaded embedding model
    app.state.rag = RagService(
        app.state.retrieval,
        make_generators(generator),
        {verifier.name: verifier},
        default_generator=generator.name,
    )
    app.state.arena = ArenaEngine(
        store,
        ArenaStore(store, blobs),
        app.state.ingestion,
        app.state.retrieval,
        app.state.rag,
        dense,
    )
    app.state.replay = ReplayService(app.state.arena)
    app.state.started_at = datetime.now(UTC)
    origins = os.environ.get("RAG_FORGE_CORS_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
    )
    app.include_router(routes.router)
    app.include_router(corpus.router)
    app.include_router(arena.router)
    return app
