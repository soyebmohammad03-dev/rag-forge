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
    score: float = Field(description="Fused score (the hit's result.score unless reranked)")
    components: list[ComponentScore]


DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerSpec(Model):
    """What determines a reranker's scores. Its hash is part of every reranked configuration."""

    provider: str = "onnx-cross-encoder"
    model: str = DEFAULT_RERANKER
    revision: str = Field(
        default="233902d25c440f23af6f7d6e94d2946bac0bee0a", description="Pinned model commit"
    )
    max_seq_length: int | None = Field(default=None, description="None = the model's own limit")
    batch_size: int = Field(default=32, ge=1, le=512)

    def config_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class RerankerInfo(Model):
    """The spec plus facts read from the loaded model files."""

    spec: RerankerSpec
    config_hash: str
    scoring: str = Field(description="How a (query, passage) pair becomes a score")
    activation: str = Field(description="Applied to the model's single logit: identity or sigmoid")
    max_seq_length: int  # effective pair limit (spec override or the model's own)
    truncation: str
    weights_sha256: str


class RerankParams(Model):
    enabled: bool = False
    model: str = Field(default=DEFAULT_RERANKER, description="A registered reranker model")
    candidate_k: int = Field(
        default=50, ge=1, le=200, description="Upstream candidates the reranker scores"
    )


class RerankConfiguration(Model):
    """The resolved reranking step: which model scored how many upstream candidates."""

    reranker: RerankerSpec
    candidate_k: int


class RankMovement(StrEnum):
    PROMOTED = "promoted"  # final rank better (smaller) than the upstream rank
    DEMOTED = "demoted"
    UNCHANGED = "unchanged"


class RerankDetail(Model):
    """How the reranker moved one candidate. Ranks are 1-based positions in the candidate pool."""

    original_rank: int = Field(ge=1, description="Rank in the upstream retrieval")
    original_score: float = Field(description="The upstream retriever's own score")
    reranker_score: float
    final_rank: int = Field(ge=1, description="Rank after reranking the whole candidate pool")
    rank_delta: int = Field(description="original_rank - final_rank; positive = moved up")
    movement: RankMovement
    entered_top_k: bool = Field(description="In the final top-k but not the upstream top-k")
    left_top_k: bool = Field(description="In the upstream top-k but not the final top-k")


class RerankCandidate(Model):
    """One scored candidate, including those that did not make the final top-k."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_ordinal: int
    rerank: RerankDetail


class RerankReport(Model):
    candidates: list[RerankCandidate] = Field(description="The whole scored pool, by final rank")


# --- Query intelligence & routing --------------------------------------------


class QuestionType(StrEnum):
    DEFINITION = "definition"  # what is X, define X, meaning of X
    PROCEDURAL = "procedural"  # how to, how do I, steps to
    EXPLANATORY = "explanatory"  # why, how does X work, explain
    COMPARISON = "comparison"  # compare, difference between, X vs Y
    LIST = "list"  # list, examples of, what are the
    BOOLEAN = "boolean"  # is/are/does/can ... (yes/no)
    FACTOID = "factoid"  # who, when, where, which, how many, other what
    KEYWORD = "keyword"  # no question form


class QueryClass(StrEnum):
    LEXICAL = "lexical"  # exact anchors dominate: identifiers, quoted phrases, keyword form
    SEMANTIC = "semantic"  # natural-language need without exact anchors
    MIXED = "mixed"


class Complexity(StrEnum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class EvidenceNeed(StrEnum):
    SINGLE_PASSAGE = "single_passage"
    MULTIPLE_PASSAGES = "multiple_passages"


class QueryFeatures(Model):
    """Measured properties of the query text. Every list keeps first-occurrence order."""

    char_count: int
    token_count: int = Field(description="Word tokens (Unicode word characters)")
    bm25_terms: list[str] = Field(description="Distinct terms BM25 searches (its analyzer)")
    key_terms: list[str] = Field(description="bm25_terms that are not function words")
    function_word_ratio: float = Field(
        description="Share of word tokens that are function words (stopwords, interrogatives, "
        "auxiliaries, pronouns, prepositions)"
    )
    is_question: bool = Field(description="Ends with '?' or starts with a question/request word")
    question_word: str | None = Field(description="The leading question or request word")
    question_type: QuestionType
    quoted_phrases: list[str]
    identifiers: list[str] = Field(
        description="Tokens with digits, inner capitals, underscores, dots or 2+ capitals"
    )
    capitalized_terms: list[str] = Field(description="Capitalised words not at sentence start")
    numbers: list[str]
    entities: list[str] = Field(description="Quoted phrases, identifiers and capitalised terms")
    concept_segments: list[str] = Field(
        description="Parts split on and/or/vs/commas/semicolons that contain a key term"
    )
    comparison_markers: list[str]
    multi_hop_markers: list[str]
    temporal_markers: list[str]
    negation_markers: list[str]
    ambiguity_markers: list[str]


class SignalContribution(Model):
    feature: str
    value: float = Field(description="The feature's value, scaled to 0..1")
    weight: float
    contribution: float = Field(description="value * weight")


class QuerySignal(Model):
    """A 0..1 score that is exactly the clipped sum of its listed contributions."""

    name: str
    score: float
    contributions: list[SignalContribution]


class QueryLabels(Model):
    query_class: QueryClass
    class_margin: float = Field(
        description="lexical - semantic score; |margin| is the class confidence, not a probability"
    )
    complexity: Complexity
    multi_hop_likely: bool
    ambiguous: bool
    evidence_need: EvidenceNeed


class TermStatistic(Model):
    term: str
    document_frequency: int = Field(description="Chunks in the corpus version containing it")
    idf: float = Field(description="BM25 idf in this corpus version")


class CorpusQuerySignals(Model):
    """How the query's terms occur in one corpus version (BM25 analyzer, BM25 statistics)."""

    corpus_id: str
    corpus_version: int
    analyzer: str
    chunk_count: int
    terms: list[TermStatistic]
    coverage: float = Field(
        description="Share of key terms (bm25_terms if there are none) found in the version"
    )
    missing_terms: list[str] = Field(description="The terms coverage counts that are absent")
    mean_idf: float | None = Field(description="Mean idf of the terms that are present")


class QueryAnalysis(Model):
    analyzer: str
    analyzer_version: str
    config_hash: str = Field(description="Hash of the analyzer's weights and thresholds")
    query: str
    normalized_query: str
    features: QueryFeatures
    signals: list[QuerySignal]
    labels: QueryLabels
    corpus: CorpusQuerySignals | None = None
    analysis_hash: str = Field(description="Hash of everything above; equal inputs, equal hash")


class RetrievalMode(StrEnum):
    MANUAL = "manual"  # the request's strategy, hybrid and rerank parameters are used as given
    ADAPTIVE = "adaptive"  # the router chooses strategy, hybrid and rerank parameters


class RouteOption(StrEnum):
    SPARSE = "sparse"
    DENSE = "dense"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_WEIGHTED = "hybrid_weighted"


class RouterParams(Model):
    analyzer: str = Field(default="heuristic", description="A registered query analyzer")
    policy: str = Field(default="rules-baseline", description="A registered router policy")


class RuleEvaluation(Model):
    """One policy rule as evaluated for this query: what it read and whether it fired."""

    rule: str
    description: str = Field(description="The rule as written in the policy")
    matched: bool
    inputs: dict[str, float | int | str | bool | None]
    margin: float | None = Field(
        default=None, description="Distance of the inputs from the nearest threshold of the rule"
    )
    outcome: str | None = Field(default=None, description="What the rule selects when it fires")


class RouteAlternative(Model):
    option: RouteOption
    selected: bool
    available: bool
    unavailable_reason: str | None = None
    rules: list[str] = Field(description="Policy rules that select this option")


class RoutingConfiguration(Model):
    """The identity of the routing step: which analyzer and policy, at which versions."""

    analyzer: str
    analyzer_version: str
    analyzer_config_hash: str
    policy: str
    policy_version: str
    policy_config_hash: str

    def config_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class RouterDecision(Model):
    id: str = Field(default_factory=lambda: new_id("rtd"))
    query_id: str
    policy: str
    policy_version: str
    policy_config_hash: str
    analysis_hash: str
    option: RouteOption = Field(description="The selected retrieval option")
    preferred: RouteOption = Field(description="The policy's choice before availability")
    strategy: RetrievalStrategy
    hybrid: HybridParams | None
    rerank: RerankParams
    rules: list[RuleEvaluation] = Field(description="Strategy rules in order, to the first match")
    rerank_rules: list[RuleEvaluation] = Field(description="Rerank rules, to the first match")
    alternatives: list[RouteAlternative]
    margin: float | None = Field(description="The deciding rule's margin; small = near a boundary")
    rationale: list[str] = Field(description="The fired rules, filled in with measured values")
    configuration_hash: str = Field(
        description="Hash of the selected configuration, as the same manual request reports it"
    )
    decision_hash: str = Field(description="Hash of the decision, excluding ids and timestamps")
    created_at: datetime = Field(default_factory=utcnow)


class RoutingProvenance(Model):
    routing: RoutingConfiguration
    routing_hash: str
    analysis: QueryAnalysis
    decision: RouterDecision
    analysis_ms: float = Field(description="Query analysis, including corpus term statistics")
    decision_ms: float


class RetrievalRequest(Model):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)
    version: int | None = Field(default=None, ge=0, description="Corpus version; default current")
    strategy: RetrievalStrategy = RetrievalStrategy.SPARSE
    bm25: Bm25Params = Field(default_factory=Bm25Params)
    hybrid: HybridParams = Field(default_factory=HybridParams)
    rerank: RerankParams = Field(default_factory=RerankParams)
    mode: RetrievalMode = Field(
        default=RetrievalMode.MANUAL,
        description="adaptive: the router replaces strategy, hybrid and rerank",
    )
    router: RouterParams = Field(default_factory=RouterParams)

    @field_validator("strategy", mode="before")
    @classmethod
    def _strategy_alias(cls, value: object) -> object:
        return "sparse" if value == "bm25" else value  # "bm25" names the sparse baseline

    @property
    def pool_k(self) -> int:
        """How many ranked candidates the upstream retriever must return."""
        return self.rerank.candidate_k if self.rerank.enabled else self.top_k

    @model_validator(mode="after")
    def _query_not_blank(self) -> RetrievalRequest:
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if self.rerank.enabled and self.rerank.candidate_k < self.top_k:
            raise ValueError("rerank.candidate_k must be at least top_k")
        if self.strategy is RetrievalStrategy.HYBRID and self.hybrid.candidate_k < self.pool_k:
            what = "rerank.candidate_k" if self.rerank.enabled else "top_k"
            raise ValueError(f"hybrid.candidate_k must be at least {what}")
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
    rerank: RerankConfiguration | None = None
    routing: RoutingConfiguration | None = None

    def config_hash(self) -> str:
        # Steps that did not run are left out, so hashes from before they existed still hold.
        exclude = {k for k in ("rerank", "routing") if getattr(self, k) is None}
        return canonical_hash(self.model_dump(mode="json", exclude=exclude))


class RetrievalHit(Model):
    """One retrieved chunk with everything needed to display and trace it."""

    result: RetrievalResult
    chunk: Chunk
    filename: str
    media_type: str
    document_version: int
    matched_terms: list[str]
    fusion: FusionDetail | None = Field(default=None, description="Hybrid only")
    rerank: RerankDetail | None = Field(default=None, description="Reranked requests only")


class RerankProvenance(Model):
    reranker: str
    info: RerankerInfo
    reranker_config_hash: str
    candidate_k: int
    final_top_k: int
    candidates_scored: int = Field(description="May be below candidate_k on small corpora")
    upstream_configuration: RetrievalConfiguration = Field(
        description="The unreranked retrieval that produced the pool (top_k = candidate_k)"
    )
    upstream_configuration_hash: str
    latency_ms: float = Field(description="Scoring time only; excludes the one-time model load")
    statistics: dict[str, float] = Field(
        description="promoted, demoted, unchanged, entered_top_k, left_top_k over the whole pool"
    )


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
    reranking: RerankProvenance | None = Field(default=None, description="Reranked requests only")
    routing: RoutingProvenance | None = Field(default=None, description="Adaptive requests only")


class RetrievalResponse(Model):
    query: Query
    hits: list[RetrievalHit] = Field(
        description="Final ranking; when reranked, result.rank and result.score are the reranker's"
    )
    provenance: RetrievalProvenance
    warnings: list[str] = Field(default_factory=list)
    reranking: RerankReport | None = Field(default=None, description="Reranked requests only")


# --- Evidence, generation & grounding -----------------------------------------


class EvidenceParams(Model):
    """How the evidence stage picks passages from the final ranking. Deterministic."""

    max_items: int = Field(default=5, ge=1, le=20, description="Evidence budget in passages")
    max_context_tokens: int = Field(
        default=1500, ge=64, le=16_000, description="Evidence budget in generator tokens"
    )
    max_per_document: int | None = Field(
        default=2, ge=1, le=20, description="Diversity cap per document version; null = no cap"
    )
    near_duplicate_threshold: float | None = Field(
        default=0.8,
        gt=0,
        le=1,
        description="Skip a passage whose term-set Jaccard with a selected one reaches this; "
        "null = keep near-duplicates",
    )
    min_score: float | None = Field(
        default=None, description="Skip passages whose final ranking score is below this"
    )

    def config_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class SelectionOutcome(StrEnum):
    SELECTED = "selected"
    NEAR_DUPLICATE = "near_duplicate"  # too similar to an already selected passage
    DOCUMENT_CAP = "document_cap"  # its document already supplied max_per_document passages
    BELOW_MIN_SCORE = "below_min_score"
    OVER_TOKEN_BUDGET = "over_token_budget"  # would not fit in the remaining context budget
    OVER_ITEM_BUDGET = "over_item_budget"  # max_items already selected


class EvidenceRetrieval(Model):
    """How retrieval and reranking saw a passage. Copied from the hit, never recomputed."""

    strategy: RetrievalStrategy
    retriever: str
    rank: int = Field(ge=1, description="Final rank (after reranking, if any)")
    score: float = Field(description="Final score (the reranker's, if reranked)")
    upstream_rank: int | None = Field(description="Rank before reranking; null if not reranked")
    upstream_score: float | None = Field(description="Retriever score before reranking")
    reranker_score: float | None
    configuration_hash: str = Field(description="The retrieval configuration that ranked it")


class Evidence(Model):
    """A passage selected to support an answer, pinned to the exact corpus version it came from."""

    id: str = Field(
        description="Stable: a hash of corpus version, chunk and exact span, so equal evidence "
        "has an equal id across runs"
    )
    citation: str = Field(description="The label the generator cites, e.g. E1")
    corpus_id: str
    corpus_version: int
    chunking_hash: str
    document_id: str
    document_version_id: str
    document_version: int
    filename: str
    media_type: str
    chunk_id: str
    chunk_ordinal: int
    char_start: int = Field(ge=0, description="Span in the extracted document text (inclusive)")
    char_end: int = Field(ge=0, description="Span in the extracted document text (exclusive)")
    text: str = Field(description="Exactly the text placed in the generation context")
    text_sha256: str
    token_count: int = Field(description="Generator tokens this passage uses in the context")
    retrieval: EvidenceRetrieval
    selection_rank: int = Field(ge=1, description="Order of selection, 1 = first selected")
    selection_score: float = Field(description="The final ranking score selection ordered by")
    selection_reason: str = Field(description="Why it was selected, with measured values")
    origin: ContentOrigin = ContentOrigin.RETRIEVED


class SelectionDecision(Model):
    """One ranked candidate as the evidence stage judged it, selected or not."""

    chunk_id: str
    document_id: str
    filename: str
    rank: int
    score: float
    outcome: SelectionOutcome
    detail: str
    token_count: int | None = Field(description="Generator tokens; null if never measured")
    evidence_id: str | None = Field(description="Set when selected")
    similar_to: str | None = Field(default=None, description="Near duplicates: the evidence id")
    similarity: float | None = Field(default=None, description="Near duplicates: the Jaccard")


class EvidenceSelection(Model):
    params: EvidenceParams
    params_hash: str
    selector: str
    candidates: int = Field(description="Ranked hits the selector considered")
    selected: list[Evidence]
    decisions: list[SelectionDecision] = Field(description="Every candidate, in ranking order")
    tokens_used: int
    tokenizer: str = Field(description="The tokenizer that measured the budget")
    selection_hash: str = Field(description="Hash of params and selected evidence ids, in order")
    latency_ms: float


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(Model):
    role: ChatRole
    content: str


class ContextBlock(Model):
    citation: str
    evidence_id: str
    header: str = Field(description="The source line shown to the generator")
    text: str
    token_count: int


class GenerationContext(Model):
    """Exactly what the generator saw: the evidence blocks and the rendered prompt."""

    corpus_id: str
    corpus_version: int
    prompt_template: str = Field(description="name@version of the grounded prompt contract")
    blocks: list[ContextBlock]
    evidence_text: str = Field(description="The rendered evidence section, blocks in order")
    messages: list[ChatMessage]
    context_tokens: int = Field(description="Generator tokens of the evidence section")
    max_context_tokens: int
    tokenizer: str
    context_hash: str = Field(
        description="Hash of corpus version, template, evidence ids and exact evidence text"
    )
    prompt_hash: str = Field(description="Hash of the rendered messages")


DEFAULT_GENERATOR = "onnx-community/Qwen2.5-0.5B-Instruct"


class GeneratorSpec(Model):
    """What determines a local generator's output. Its hash is part of every answer's provenance."""

    provider: str = "onnx-causal-lm"
    model: str = DEFAULT_GENERATOR
    revision: str = Field(
        default="cc5cc01a65cc3ff17bdb73a7de33d879f62599b0", description="Pinned model commit"
    )
    weights_file: str = Field(
        default="onnx/model_q4.onnx", description="ONNX export to load (4-bit weights)"
    )
    chat_template: str = Field(default="chatml", description="How messages become a prompt")

    def config_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class GeneratorInfo(Model):
    """The spec plus facts read from the loaded model (or the remote endpoint's identity)."""

    name: str = Field(description="The registered generator name requests use")
    provider: str
    model: str
    revision: str | None
    config_hash: str
    local: bool = Field(description="Runs in this process; false = an HTTP endpoint")
    deterministic_at_zero_temperature: bool
    max_context_tokens: int | None
    tokenizer: str
    weights_sha256: str | None
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class GeneratorDescriptor(Model):
    """A registered generator, described without loading it."""

    name: str
    provider: str
    model: str
    revision: str | None
    config_hash: str = Field(description="Equals GeneratorInfo.config_hash once loaded")
    local: bool
    loaded: bool


class GenerationParams(Model):
    generator: str | None = Field(
        default=None, description="A registered generator; null = the configured default"
    )
    max_new_tokens: int = Field(default=200, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0, le=2, description="0 = greedy, reproducible")
    top_p: float = Field(default=1.0, gt=0, le=1, description="Sampling only")
    seed: int = Field(default=0, ge=0, description="Sampling only; recorded so samples replay")

    def effective(self) -> GenerationParams:
        """Blank out sampling parameters greedy decoding ignores, so they cannot change hashes."""
        if self.temperature == 0:
            return self.model_copy(update={"top_p": 1.0, "seed": 0})
        return self


class FinishReason(StrEnum):
    STOP = "stop"  # the model ended its answer
    LENGTH = "length"  # max_new_tokens reached; the answer may be cut off


class GenerationRecord(Model):
    generator: GeneratorInfo
    params: GenerationParams = Field(description="Effective parameters")
    params_hash: str
    prompt_hash: str
    raw_text: str = Field(description="The generator's output, verbatim")
    answer_hash: str
    finish_reason: FinishReason
    prompt_tokens: int | None
    completion_tokens: int | None
    deterministic: bool = Field(description="Greedy decoding of a local model: replays exactly")
    load_ms: float = Field(description="Model load paid by this request (0 when already loaded)")
    latency_ms: float = Field(description="Generation time, excluding the model load")
    origin: ContentOrigin = ContentOrigin.GENERATED


class ClaimKind(StrEnum):
    FACTUAL = "factual"  # asserts something checkable against evidence
    ABSTENTION = "abstention"  # says the evidence is insufficient
    NON_ASSERTIVE = "non_assertive"  # no content terms to check (e.g. "Here is the answer:")


class SupportStatus(StrEnum):
    SUPPORTED = "supported"
    WEAKLY_SUPPORTED = "weakly_supported"
    UNSUPPORTED = "unsupported"  # no evidence was measured to support it (not "false")
    CONTRADICTED = "contradicted"  # only verifiers that can measure contradiction emit this
    NOT_APPLICABLE = "not_applicable"  # abstentions and non-assertive sentences


class Citation(Model):
    label: str = Field(description="As written, e.g. E2")
    evidence_id: str | None = Field(description="Null when the label names no supplied evidence")
    valid: bool
    char_start: int = Field(description="Offset of the marker in the raw answer")
    char_end: int


class EvidenceSupport(Model):
    """How well one evidence passage supports one claim, as measured by the verifier."""

    evidence_id: str
    citation: str
    cited: bool = Field(description="The generator cited this passage for the claim")
    lexical_coverage: float = Field(description="Share of the claim's content terms in the passage")
    semantic_similarity: float = Field(description="Best cosine of claim vs passage sentences")
    best_sentence: str = Field(description="The passage sentence most similar to the claim")
    score: float
    status: SupportStatus


class Claim(Model):
    id: str = Field(description="Stable: hash of the answer hash and the claim's position")
    index: int
    text: str = Field(description="The sentence with citation markers removed")
    raw_text: str = Field(description="The sentence as generated, markers included")
    char_start: int
    char_end: int
    kind: ClaimKind
    content_terms: list[str] = Field(description="The terms the verifier checks")
    citations: list[Citation]
    cited_evidence_ids: list[str] = Field(description="What the generator cited (generated)")
    supporting_evidence_ids: list[str] = Field(
        description="What the verifier measured as supporting (measured)"
    )
    support: SupportStatus
    support_score: float | None = Field(description="Null when not applicable")
    missing_terms: list[str] = Field(description="Content terms absent from the supporting text")
    unmatched_numbers: list[str] = Field(description="Numbers absent from the supporting text")
    evidence: list[EvidenceSupport] = Field(description="Every evidence passage, best first")
    flags: list[str] = Field(
        description="e.g. uncited, invalid_citation, cited_not_supporting, number_mismatch, "
        "negation_mismatch, multi_passage"
    )
    rationale: str
    origin: ContentOrigin = ContentOrigin.GENERATED


class GroundingParams(Model):
    verifier: str = Field(default="lexical-semantic", description="A registered verifier")


class AnswerGrounding(StrEnum):
    GROUNDED = "grounded"  # every factual claim supported
    PARTIALLY_GROUNDED = "partially_grounded"  # some factual claims weak or unsupported
    UNGROUNDED = "ungrounded"  # no factual claim supported or weakly supported
    ABSTAINED = "abstained"  # the answer only says the evidence is insufficient
    NO_CLAIMS = "no_claims"


class GroundingReport(Model):
    verifier: str
    verifier_version: str
    config_hash: str
    thresholds: dict[str, float]
    detects_contradiction: bool = Field(
        description="False: 'contradicted' is never emitted, absence of support is 'unsupported'"
    )
    status: AnswerGrounding
    claims: int
    factual_claims: int
    supported: int
    weakly_supported: int
    unsupported: int
    contradicted: int
    abstentions: int
    non_assertive: int
    grounding_score: float | None = Field(
        description="(supported + 0.5 x weakly supported) / factual claims; null if none"
    )
    evidence_coverage: float | None = Field(
        description="Share of selected evidence that supports at least one claim"
    )
    citation_coverage: float | None = Field(
        description="Share of factual claims with at least one valid citation"
    )
    citation_precision: float | None = Field(
        description="Share of valid citations whose passage the verifier found supporting"
    )
    invalid_citations: int = Field(description="Citations naming evidence that was not supplied")
    grounding_hash: str = Field(description="Hash of every claim's measured support")
    load_ms: float = Field(description="Verifier model load paid by this request (0 if loaded)")
    latency_ms: float = Field(description="Claim extraction and verification, excluding the load")
    origin: ContentOrigin = ContentOrigin.MEASURED


class AnswerStatus(StrEnum):
    ANSWERED = "answered"  # generated an answer with at least one factual claim
    ABSTAINED = "abstained"  # the generator said the evidence is insufficient
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # no usable evidence; generator not called
    NOT_GENERATED = "not_generated"  # the request stopped after context assembly


class RagRequest(Model):
    retrieval: RetrievalRequest
    evidence: EvidenceParams = Field(default_factory=EvidenceParams)
    generation: GenerationParams | None = Field(
        default_factory=GenerationParams,
        description="null = stop after evidence selection and context assembly",
    )
    grounding: GroundingParams = Field(default_factory=GroundingParams)


class PipelineStage(Model):
    """One link of the provenance chain: what the stage was, its identity and its cost."""

    stage: str
    hash: str | None = Field(description="Identity of the stage's output")
    config_hash: str | None = Field(description="Identity of the stage's configuration")
    latency_ms: float | None
    origin: ContentOrigin
    deterministic: bool = Field(description="Equal inputs and configuration give equal output")
    detail: str


class RagConfiguration(Model):
    """Everything that determines an answer, resolved. Its hash identifies the pipeline."""

    retrieval: RetrievalConfiguration
    evidence: EvidenceParams
    prompt_template: str | None
    generator: str | None
    generator_config_hash: str | None
    generation: GenerationParams | None
    verifier: str | None
    verifier_config_hash: str | None

    def config_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class RagProvenance(Model):
    configuration: RagConfiguration
    configuration_hash: str
    chain: list[PipelineStage]
    environment: EnvironmentSnapshot
    elapsed_ms: float
    answered_at: datetime = Field(default_factory=utcnow)


class GeneratedAnswer(Model):
    text: str = Field(description="The raw generated answer, citation markers included")
    generation: GenerationRecord


class RagResponse(Model):
    query: Query
    status: AnswerStatus
    retrieval: RetrievalResponse
    evidence: EvidenceSelection
    context: GenerationContext | None = Field(description="Null when no evidence was usable")
    answer: GeneratedAnswer | None
    claims: list[Claim]
    grounding: GroundingReport | None
    provenance: RagProvenance
    warnings: list[str] = Field(default_factory=list)


# --- Evaluation ----------------------------------------------------------------


class Metric(Model):
    name: str
    value: float
    k: int | None = None
    origin: ContentOrigin = ContentOrigin.MEASURED


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
