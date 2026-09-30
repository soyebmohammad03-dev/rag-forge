from __future__ import annotations

import os
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_forge import __version__
from rag_forge.api.routes import router
from rag_forge.storage.memory import InMemoryStore


def create_app() -> FastAPI:
    app = FastAPI(title="RAG FORGE API", version=__version__)
    app.state.store = InMemoryStore()
    app.state.started_at = datetime.now(UTC)
    origins = os.environ.get("RAG_FORGE_CORS_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
    )
    app.include_router(router)
    return app


app = create_app()
