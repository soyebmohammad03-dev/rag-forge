from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_forge import __version__
from rag_forge.api import corpus, routes
from rag_forge.ingestion.service import IngestionService
from rag_forge.storage.blobs import BlobStore
from rag_forge.storage.sqlite import SqliteStore


def create_app(data_dir: Path | None = None) -> FastAPI:
    """`data_dir` holds the SQLite database and content-addressed blobs.

    Defaults to $RAG_FORGE_DATA_DIR, else ./data relative to the working directory.
    """
    data_dir = data_dir or Path(os.environ.get("RAG_FORGE_DATA_DIR", "data"))
    app = FastAPI(title="RAG FORGE API", version=__version__)
    app.state.store = SqliteStore(data_dir / "rag_forge.sqlite3")
    app.state.ingestion = IngestionService(app.state.store, BlobStore(data_dir / "blobs"))
    app.state.started_at = datetime.now(UTC)
    origins = os.environ.get("RAG_FORGE_CORS_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
    )
    app.include_router(routes.router)
    app.include_router(corpus.router)
    return app
