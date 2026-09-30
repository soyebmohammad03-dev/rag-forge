"""Request/response contracts. The web client's types are generated from these via OpenAPI."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from rag_forge.domain.models import (
    Chunk,
    ChunkingConfig,
    Corpus,
    DenseIndex,
    DenseIndexState,
    Document,
    DocumentVersion,
    EmbedderSpec,
    EnvironmentSnapshot,
    Metric,
    Query,
    RouterDecision,
)
from rag_forge.storage.base import CorpusStats


class ComponentState(StrEnum):
    OK = "ok"
    NOT_CONFIGURED = "not_configured"
    ERROR = "error"


class ComponentHealth(BaseModel):
    name: str
    state: ComponentState
    detail: str


class HealthStatus(BaseModel):
    status: str  # "ok" when every configured component is healthy
    version: str
    started_at: datetime
    uptime_seconds: float
    components: list[ComponentHealth]


class NotImplementedDetail(BaseModel):
    capability: str
    message: str


class CorpusCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


class CorpusSummary(BaseModel):
    corpus: Corpus
    stats: CorpusStats


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    version_count: int
    current: DocumentVersion


class DocumentDetail(BaseModel):
    document: Document
    versions: list[DocumentVersion] = Field(description="All versions, oldest first")
    current_version_id: str | None = Field(
        description="Version in the corpus's current version; null if the document was removed"
    )


class ChunkPage(BaseModel):
    document_version_id: str
    chunking_hash: str
    total: int
    offset: int
    items: list[Chunk]


class DenseIndexView(BaseModel):
    """Whether a corpus version has a usable dense index under the configured embedder."""

    corpus_id: str
    version: int
    state: DenseIndexState
    index: DenseIndex | None = Field(description="Latest build for this version, if any")
    embedder: EmbedderSpec
    embedder_hash: str


class DenseIndexBuild(BaseModel):
    version: int | None = Field(default=None, ge=0, description="Defaults to the current version")


class SupportedFormat(BaseModel):
    media_type: str
    extensions: list[str]
    parser: str


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    hypothesis: str = ""
    corpus_id: str | None = None


class RouterRequest(BaseModel):
    corpus_id: str
    query: str = Field(min_length=1)
    policy: str


class RouterResponse(BaseModel):
    query: Query
    decision: RouterDecision


class RetrievalEvaluationRequest(BaseModel):
    ranked_ids: list[str]
    relevance: dict[str, float] = Field(
        description="Graded relevance judgements; grade > 0 counts as relevant."
    )
    ks: list[int] = Field(default=[1, 5, 10], min_length=1)


class RetrievalEvaluationResponse(BaseModel):
    metrics: list[Metric]


class EnvironmentResponse(BaseModel):
    environment: EnvironmentSnapshot
