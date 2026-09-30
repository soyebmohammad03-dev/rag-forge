"""Request/response contracts. The web client's types are generated from these via OpenAPI."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from rag_forge.domain.models import (
    EnvironmentSnapshot,
    Metric,
    Query,
    RetrievalResult,
    RetrievalStrategy,
    RouterDecision,
)


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


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    hypothesis: str = ""
    corpus_id: str | None = None


class DocumentIngestRequest(BaseModel):
    source_uri: str
    media_type: str | None = None


class RetrievalRequest(BaseModel):
    corpus_id: str
    query: str = Field(min_length=1)
    strategies: list[RetrievalStrategy] = Field(min_length=1)
    top_k: int = Field(default=10, gt=0, le=1000)


class RetrievalResponse(BaseModel):
    query: Query
    results: list[RetrievalResult]


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
