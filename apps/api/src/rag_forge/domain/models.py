"""Foundational RAG FORGE domain models.

These are deliberately thin: they fix names, identities and provenance-relevant
fields so later phases (ingestion, retrieval, router, arena) share one vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Model(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", json_schema_serialization_defaults_required=True
    )


class ContentOrigin(StrEnum):
    """Where a piece of information came from. Every value shown to a user carries one."""

    RETRIEVED = "retrieved"  # verbatim from the corpus
    INFERRED = "inferred"  # derived by deterministic processing (e.g. parsing, entity linking)
    GENERATED = "generated"  # produced by a language model
    MEASURED = "measured"  # computed by an evaluator on real outputs
    SIMULATED = "simulated"  # estimated, synthetic or preview data; never a result


# --- Corpus -------------------------------------------------------------------


class Corpus(Model):
    id: str = Field(default_factory=lambda: new_id("cor"))
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    version: int = 0
    document_count: int = 0
    created_at: datetime = Field(default_factory=utcnow)


class Document(Model):
    id: str = Field(default_factory=lambda: new_id("doc"))
    corpus_id: str
    title: str
    source_uri: str
    media_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class DocumentVersion(Model):
    id: str = Field(default_factory=lambda: new_id("dv"))
    document_id: str
    version: int = Field(ge=1)
    content_sha256: str
    byte_size: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utcnow)


class Chunk(Model):
    id: str = Field(default_factory=lambda: new_id("chk"))
    document_version_id: str
    ordinal: int = Field(ge=0)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    chunker: str  # chunking strategy identifier + params hash
    metadata: dict[str, Any] = Field(default_factory=dict)


class Embedding(Model):
    chunk_id: str
    model: str
    model_version: str
    dimensions: int = Field(gt=0)
    vector: list[float] = Field(repr=False)


# --- Retrieval ----------------------------------------------------------------


class Query(Model):
    id: str = Field(default_factory=lambda: new_id("qry"))
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class RetrievalStrategy(StrEnum):
    SPARSE = "sparse"
    DENSE = "dense"
    HYBRID = "hybrid"
    METADATA = "metadata"
    GRAPH = "graph"
    MULTI_HOP = "multi_hop"
    DECOMPOSITION = "decomposition"
    MULTIMODAL = "multimodal"


class RetrievalResult(Model):
    query_id: str
    strategy: RetrievalStrategy
    retriever: str  # concrete retriever identifier, e.g. "bm25:k1=1.2,b=0.75"
    chunk_id: str
    document_id: str
    rank: int = Field(ge=1)
    score: float
    origin: ContentOrigin = ContentOrigin.RETRIEVED


class Evidence(Model):
    id: str = Field(default_factory=lambda: new_id("evd"))
    chunk_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
    origin: ContentOrigin = ContentOrigin.RETRIEVED


class ClaimSupport(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"


class Claim(Model):
    id: str = Field(default_factory=lambda: new_id("clm"))
    text: str
    origin: ContentOrigin = ContentOrigin.GENERATED
    evidence_ids: list[str] = Field(default_factory=list)
    support: ClaimSupport = ClaimSupport.UNVERIFIED


# --- Configuration & routing -------------------------------------------------


class RAGConfiguration(Model):
    """Everything that determines a run's behaviour. Its hash identifies it."""

    id: str = Field(default_factory=lambda: new_id("cfg"))
    name: str
    chunking: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
    strategies: list[RetrievalStrategy] = Field(min_length=1)
    top_k: int = Field(default=10, gt=0)
    reranker: str | None = None
    router_policy: str | None = None
    generation_model: str | None = None
    prompt_template: str | None = None

    def config_hash(self) -> str:
        """Stable hash of behaviour-relevant fields (excludes id and name)."""
        payload = self.model_dump(mode="json", exclude={"id", "name"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class RouterDecision(Model):
    id: str = Field(default_factory=lambda: new_id("rtd"))
    query_id: str
    policy: str
    query_features: dict[str, Any] = Field(default_factory=dict)
    selected_strategies: list[RetrievalStrategy]
    rationale: str
    created_at: datetime = Field(default_factory=utcnow)


# --- Experiments & evaluation ------------------------------------------------


class ExperimentStatus(StrEnum):
    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Experiment(Model):
    id: str = Field(default_factory=lambda: new_id("exp"))
    name: str = Field(min_length=1, max_length=200)
    hypothesis: str = ""
    corpus_id: str | None = None
    configuration_ids: list[str] = Field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = Field(default_factory=utcnow)


class ExperimentRun(Model):
    id: str = Field(default_factory=lambda: new_id("run"))
    experiment_id: str
    configuration_id: str
    config_hash: str
    status: ExperimentStatus = ExperimentStatus.QUEUED
    started_at: datetime | None = None
    finished_at: datetime | None = None
    provenance_id: str | None = None


class Metric(Model):
    name: str
    value: float
    k: int | None = None
    origin: ContentOrigin = ContentOrigin.MEASURED


class EvaluationResult(Model):
    run_id: str
    metrics: list[Metric]
    evaluator: str
    evaluated_at: datetime = Field(default_factory=utcnow)


class Artifact(Model):
    id: str = Field(default_factory=lambda: new_id("art"))
    run_id: str
    kind: str  # e.g. "retrieval_log", "answers", "index_manifest"
    uri: str
    media_type: str
    sha256: str


class EnvironmentSnapshot(Model):
    rag_forge_version: str
    python_version: str
    platform: str
    git_commit: str | None
    packages: dict[str, str]


class ProvenanceRecord(Model):
    id: str = Field(default_factory=lambda: new_id("prv"))
    run_id: str
    config_hash: str
    corpus_id: str | None = None
    corpus_version: int | None = None
    dataset_version: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    environment: EnvironmentSnapshot
    created_at: datetime = Field(default_factory=utcnow)
