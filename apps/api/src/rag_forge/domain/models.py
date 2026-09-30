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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


def canonical_hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


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


# Bump whenever chunk boundaries can change for the same config. It is part of the config hash,
# so chunks produced by different algorithm versions are never mistaken for each other.
CHUNKER_VERSION = 2


class ChunkingStrategy(StrEnum):
    RECURSIVE = "recursive"  # split on paragraph > line > sentence > word boundaries, then merge
    FIXED = "fixed"  # fixed-size character windows; the naive baseline


class ChunkingConfig(Model):
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = Field(default=1000, ge=50, le=20_000, description="Max characters")
    chunk_overlap: int = Field(default=150, ge=0, description="Characters shared with previous")

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def config_hash(self) -> str:
        return canonical_hash({**self.model_dump(mode="json"), "chunker_version": CHUNKER_VERSION})


class Corpus(Model):
    id: str = Field(default_factory=lambda: new_id("cor"))
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    version: int = Field(default=0, ge=0, description="0 = empty; +1 per content change")
    created_at: datetime = Field(default_factory=utcnow)


class CorpusVersion(Model):
    """An immutable snapshot: which document versions the corpus contained."""

    corpus_id: str
    version: int = Field(ge=0)
    ingestion_id: str | None
    document_count: int
    added: int = 0
    modified: int = 0
    removed: int = 0
    unchanged: int = 0
    created_at: datetime = Field(default_factory=utcnow)


class Document(Model):
    """A logical document, identified within its corpus by filename. Content lives in versions."""

    id: str = Field(default_factory=lambda: new_id("doc"))
    corpus_id: str
    filename: str
    created_at: datetime = Field(default_factory=utcnow)


class ExtractionStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"  # some parts (e.g. PDF pages) failed; see warnings


class DocumentVersion(Model):
    id: str = Field(default_factory=lambda: new_id("dv"))
    document_id: str
    version: int = Field(ge=1)
    filename: str
    media_type: str
    content_sha256: str
    byte_size: int = Field(ge=0)
    parser: str  # extractor name@version
    extraction_status: ExtractionStatus
    extraction_warnings: list[str] = Field(default_factory=list)
    text_chars: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class Chunk(Model):
    id: str
    document_version_id: str
    chunking_hash: str
    ordinal: int = Field(ge=0)
    text: str
    char_start: int = Field(ge=0, description="Offset into the extracted text (inclusive)")
    char_end: int = Field(ge=0, description="Offset into the extracted text (exclusive)")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Embedding(Model):
    chunk_id: str
    model: str
    model_version: str
    dimensions: int = Field(gt=0)
    vector: list[float] = Field(repr=False)


# --- Ingestion provenance ----------------------------------------------------


class FileOutcome(StrEnum):
    ADDED = "added"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"  # same filename, same content hash
    DUPLICATE = "duplicate"  # same content as another document; not ingested
    REJECTED = "rejected"  # unsupported, corrupt, empty or too large; not ingested
    REMOVED = "removed"


class IngestionFileResult(Model):
    filename: str
    outcome: FileOutcome
    byte_size: int | None = None
    content_sha256: str | None = None
    media_type: str | None = None
    parser: str | None = None
    document_id: str | None = None
    document_version_id: str | None = None
    duplicate_of: str | None = None
    duplicate_of_filename: str | None = None
    chunk_count: int | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class IngestionStatus(StrEnum):
    COMPLETED = "completed"  # produced a new corpus version
    NO_CHANGE = "no_change"  # every file unchanged, duplicate or rejected
    FAILED = "failed"  # nothing usable: every file rejected


class IngestionRecord(Model):
    """Provenance of one ingestion operation. Immutable once written."""

    id: str = Field(default_factory=lambda: new_id("ing"))
    corpus_id: str
    status: IngestionStatus
    version_before: int
    version_after: int
    files: list[IngestionFileResult]
    chunking: ChunkingConfig
    chunking_hash: str
    environment: EnvironmentSnapshot
    started_at: datetime
    finished_at: datetime


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
    retriever: str  # concrete retriever identifier, e.g. "bm25"
    chunk_id: str
    document_id: str
    document_version_id: str
    rank: int = Field(ge=1)
    score: float
    origin: ContentOrigin = ContentOrigin.RETRIEVED


class Bm25Params(Model):
    k1: float = Field(default=1.2, ge=0, le=3, description="Term-frequency saturation")
    b: float = Field(default=0.75, ge=0, le=1, description="Chunk-length normalisation")


class EmbedderSpec(Model):
    """What determines an embedding. Its hash keys every stored vector."""

    provider: str = "onnx-sentence-transformers"
    model: str = "BAAI/bge-small-en-v1.5"
    revision: str = Field(
        default="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", description="Pinned model commit"
    )
    query_prefix: str = Field(
        default="Represent this sentence for searching relevant passages: ",
        description="Instruction prepended to queries (not documents), per the model card",
    )
    max_seq_length: int | None = Field(default=None, description="None = the model's own limit")
    batch_size: int = Field(default=32, ge=1, le=512)

    def config_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class EmbedderInfo(Model):
    """The spec plus facts read from the loaded model files."""

    spec: EmbedderSpec
    config_hash: str
    dimension: int
    pooling: str
    normalize: bool
    max_seq_length: int  # effective limit (spec override or the model's own)
    weights_sha256: str


class DenseIndexStatus(StrEnum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class DenseIndex(Model):
    """One build of a dense index for a corpus version under one embedder config."""

    id: str = Field(default_factory=lambda: new_id("dix"))
    corpus_id: str
    corpus_version: int
    chunking_hash: str
    embedder: EmbedderInfo
    similarity: str = "cosine"
    status: DenseIndexStatus = DenseIndexStatus.BUILDING
    chunk_count: int = Field(ge=0, description="Chunks in the corpus version")
    embedded: int = Field(default=0, ge=0, description="Progress: chunks with a vector so far")
    reused: int = Field(default=0, ge=0, description="Vectors reused from earlier builds")
    content_hash: str | None = Field(
        default=None, description="SHA-256 over chunk ids and exact vector bytes; set when ready"
    )
    error: str | None = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


class DenseIndexState(StrEnum):
    """What the UI may promise for a corpus version under the configured embedder."""

    READY = "ready"
    BUILDING = "building"
    FAILED = "failed"  # the latest build for this version failed
    STALE = "stale"  # no index here, but one exists for another version or embedder config
    MISSING = "missing"


class FusionMethod(StrEnum):
    RRF = "rrf"  # reciprocal rank fusion: uses ranks only (primary baseline)
    WEIGHTED = "weighted"  # weighted sum of per-list min-max normalised scores


FUSABLE = (RetrievalStrategy.SPARSE, RetrievalStrategy.DENSE)


class HybridParams(Model):
    retrievers: list[RetrievalStrategy] = Field(
        default_factory=lambda: list(FUSABLE), description="Component strategies to fuse"
    )
    fusion: FusionMethod = FusionMethod.RRF
    rrf_k: int = Field(default=60, ge=1, le=1000, description="RRF smoothing constant")
    weights: dict[RetrievalStrategy, float] = Field(
        default_factory=lambda: {s: 0.5 for s in FUSABLE},
        description="Weighted fusion only: one weight per component, each 0..1, summing to 1",
    )
    candidate_k: int = Field(
        default=50, ge=1, le=200, description="Candidates requested from each component"
    )

    @model_validator(mode="after")
    def _valid(self) -> HybridParams:
        if len(set(self.retrievers)) != len(self.retrievers) or len(self.retrievers) < 2:
            raise ValueError("hybrid needs at least two distinct component retrievers")
        if not set(self.retrievers) <= set(FUSABLE):
            raise ValueError(f"hybrid components must be among {[s.value for s in FUSABLE]}")
        if self.fusion is FusionMethod.WEIGHTED:
            if set(self.weights) != set(self.retrievers):
                raise ValueError("weights must name exactly the component retrievers")
            if any(not 0 <= w <= 1 for w in self.weights.values()):
                raise ValueError("each weight must be between 0 and 1")
            if abs(sum(self.weights.values()) - 1) > 1e-9:
                raise ValueError("weights must sum to 1")
        return self

    def effective(self) -> HybridParams:
        """Blank out parameters the chosen method ignores, so they cannot change config hashes."""
        if self.fusion is FusionMethod.RRF:
            return self.model_copy(update={"weights": {}})
        return self.model_copy(update={"rrf_k": HybridParams.model_fields["rrf_k"].default})


class ComponentScore(Model):
    """How one component retriever saw a fused chunk. Raw scores are never rescaled here."""

    strategy: RetrievalStrategy
    rank: int | None = Field(description="Rank in that retriever's candidates; null if absent")
    score: float | None = Field(description="That retriever's own raw score")
    normalized_score: float | None = Field(
        default=None, description="Weighted fusion only: min-max normalised score in 0..1"
    )
    contribution: float = Field(description="This component's share of the fused score")


class FusionDetail(Model):
    method: FusionMethod
    score: float = Field(description="Final fused score (the hit's result.score)")
    components: list[ComponentScore]


class RetrievalRequest(Model):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)
    version: int | None = Field(default=None, ge=0, description="Corpus version; default current")
    strategy: RetrievalStrategy = RetrievalStrategy.SPARSE
    bm25: Bm25Params = Field(default_factory=Bm25Params)
    hybrid: HybridParams = Field(default_factory=HybridParams)

    @field_validator("strategy", mode="before")
    @classmethod
    def _strategy_alias(cls, value: object) -> object:
        return "sparse" if value == "bm25" else value  # "bm25" names the sparse baseline

    @model_validator(mode="after")
    def _query_not_blank(self) -> RetrievalRequest:
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if self.strategy is RetrievalStrategy.HYBRID and self.hybrid.candidate_k < self.top_k:
            raise ValueError("hybrid.candidate_k must be at least top_k")
        return self


class RetrievalConfiguration(Model):
    """Everything that determines a ranked result set, resolved. The unit an experiment varies."""

    corpus_id: str
    corpus_version: int
    chunking_hash: str
    strategy: RetrievalStrategy
    top_k: int
    bm25: Bm25Params | None = None
    embedder: EmbedderSpec | None = None
    hybrid: HybridParams | None = None

    def config_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class RetrievalHit(Model):
    """One retrieved chunk with everything needed to display and trace it."""

    result: RetrievalResult
    chunk: Chunk
    filename: str
    media_type: str
    document_version: int
    matched_terms: list[str]
    fusion: FusionDetail | None = Field(default=None, description="Hybrid only")


class RetrievalProvenance(Model):
    corpus_id: str
    corpus_version: int
    chunking_hash: str
    strategy: RetrievalStrategy
    retriever: str
    retriever_config: dict[str, Any]
    retriever_config_hash: str
    configuration: RetrievalConfiguration
    configuration_hash: str
    index_id: str | None = Field(default=None, description="Dense index used, if any")
    query_terms: list[str]
    statistics: dict[str, float] = Field(
        description="Retriever-specific, e.g. candidate_chunks, avg_chunk_length, indexed_now"
    )
    environment: EnvironmentSnapshot
    elapsed_ms: float
    retrieved_at: datetime = Field(default_factory=utcnow)


class RetrievalResponse(Model):
    query: Query
    hits: list[RetrievalHit]
    provenance: RetrievalProvenance
    warnings: list[str] = Field(default_factory=list)


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
        return canonical_hash(self.model_dump(mode="json", exclude={"id", "name"}))


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
