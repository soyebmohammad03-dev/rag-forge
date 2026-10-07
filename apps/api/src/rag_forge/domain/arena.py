"""Benchmarking and experiment models: datasets, configurations, runs, metrics, comparisons.

Every number the Arena shows is a `MetricValue` on a `RunCase`, or an aggregate of them, and every
`RunCase` names its run, arm, case, configuration snapshot hash and full trace artifact. A metric
whose required annotations are absent is `skipped` with a reason, never 0.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from rag_forge.domain.models import (
    AnswerGrounding,
    AnswerStatus,
    Bm25Params,
    ContentOrigin,
    EmbedderSpec,
    EnvironmentSnapshot,
    EvidenceParams,
    GenerationParams,
    GroundingParams,
    HybridParams,
    Model,
    RerankerSpec,
    RerankParams,
    RetrievalConfiguration,
    RetrievalMode,
    RetrievalRequest,
    RetrievalStrategy,
    RouterParams,
    canonical_hash,
    new_id,
    utcnow,
)

SLUG = r"^[a-z0-9][a-z0-9_-]{0,47}$"
CASE_ID = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"

# --- Benchmark datasets ------------------------------------------------------------------


class DatasetSource(StrEnum):
    DEVELOPMENT = "development"  # bundled with RAG FORGE to exercise the pipeline, not to rank it
    USER = "user"  # supplied through the API by a researcher
    EXTERNAL = "external"  # reserved for imported public benchmarks (no importer yet)


class Annotation(StrEnum):
    RELEVANCE = "relevance"  # graded relevant documents or chunks
    REFERENCE_ANSWER = "reference_answer"
    ANSWERABLE = "answerable"  # whether the corpus can answer the query at all
    EXPECTED_EVIDENCE = "expected_evidence"  # documents the evidence should include


class BenchmarkCase(Model):
    """One query with whatever ground truth exists for it. Every annotation is optional."""

    id: str = Field(pattern=CASE_ID)
    query: str = Field(min_length=1, max_length=2000)
    relevant_documents: dict[str, float] = Field(
        default_factory=dict,
        description="Filename -> graded relevance (> 0 relevant, 0 judged non-relevant)",
    )
    relevant_chunks: dict[str, float] = Field(
        default_factory=dict, description="Chunk id -> graded relevance; takes precedence"
    )
    reference_answer: str | None = Field(default=None, max_length=4000)
    answerable: bool | None = Field(
        default=None, description="null = not judged; false = the corpus cannot answer it"
    )
    expected_evidence: list[str] = Field(
        default_factory=list, description="Filenames the selected evidence should include"
    )
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> BenchmarkCase:
        if not self.query.strip():
            raise ValueError("query must not be blank")
        grades = [*self.relevant_documents.values(), *self.relevant_chunks.values()]
        if any(g < 0 or g > 10 for g in grades):
            raise ValueError("relevance grades must be between 0 and 10")
        if self.answerable is False and (any(g > 0 for g in grades) or self.reference_answer):
            raise ValueError("an unanswerable case cannot have relevant items or a reference")
        return self

    def annotations(self) -> list[Annotation]:
        out = []
        if any(g > 0 for g in (*self.relevant_documents.values(), *self.relevant_chunks.values())):
            out.append(Annotation.RELEVANCE)
        if self.reference_answer:
            out.append(Annotation.REFERENCE_ANSWER)
        if self.answerable is not None:
            out.append(Annotation.ANSWERABLE)
        if self.expected_evidence:
            out.append(Annotation.EXPECTED_EVIDENCE)
        return out


class BenchmarkDatasetCreate(Model):
    name: str = Field(pattern=SLUG)
    description: str = ""
    corpus_id: str
    corpus_version: int | None = Field(default=None, ge=1, description="Default: current")
    annotation_notes: str = Field(default="", description="How the ground truth was established")
    cases: list[BenchmarkCase] = Field(min_length=1, max_length=2000)

    @field_validator("cases")
    @classmethod
    def _unique_ids(cls, cases: list[BenchmarkCase]) -> list[BenchmarkCase]:
        ids = [c.id for c in cases]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate case ids: {', '.join(dupes)}")
        return cases


class BenchmarkDataset(Model):
    """An immutable dataset version, pinned to one corpus version."""

    id: str = Field(default_factory=lambda: new_id("bds"))
    name: str
    version: int = Field(ge=1, description="+1 whenever the content of a dataset name changes")
    source: DatasetSource
    description: str
    annotation_notes: str
    corpus_id: str
    corpus_version: int
    chunking_hash: str
    cases: list[BenchmarkCase]
    annotation_counts: dict[Annotation, int]
    content_hash: str = Field(
        description="Hash of corpus version and cases: equal content, equal hash"
    )
    created_at: datetime = Field(default_factory=utcnow)

    @staticmethod
    def hash_content(corpus_id: str, corpus_version: int, cases: list[BenchmarkCase]) -> str:
        return canonical_hash(
            {
                "corpus_id": corpus_id,
                "corpus_version": corpus_version,
                "cases": [c.model_dump(mode="json") for c in cases],
            }
        )


class DatasetSummary(Model):
    id: str
    name: str
    version: int
    source: DatasetSource
    description: str
    corpus_id: str
    corpus_version: int
    case_count: int
    annotation_counts: dict[Annotation, int]
    content_hash: str
    created_at: datetime


# --- Metrics ---------------------------------------------------------------------------------


class MetricFamily(StrEnum):
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    EVIDENCE = "evidence"  # measured against the supplied evidence (grounding)
    GENERATION = "generation"  # automatic proxies against annotations; not answer quality
    OPERATIONAL = "operational"


class MetricDefinition(Model):
    name: str = Field(description="Base name; k-parameterised metrics report name@k")
    family: MetricFamily
    version: int
    description: str
    requires: list[Annotation] = Field(description="Annotations without which it is skipped")
    requires_output: list[str] = Field(
        description="Pipeline outputs it needs, e.g. reranking, generation"
    )
    per_k: bool
    unit: str
    aggregation: str = "mean over the cases where it is defined"
    higher_is_better: bool | None = Field(description="null for descriptive metrics")


class MetricValue(Model):
    metric: str = Field(description="e.g. recall@5")
    family: MetricFamily
    version: int
    value: float | None = Field(description="null when skipped")
    skipped: str | None = Field(default=None, description="Why it could not be computed")
    detail: str | None = Field(default=None, description="e.g. relevance unit: document")
    origin: ContentOrigin = ContentOrigin.MEASURED


class MetricSettings(Model):
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10], min_length=1, max_length=6)
    metrics: list[str] | None = Field(
        default=None, description="Base metric names to compute; null = every registered metric"
    )

    @field_validator("ks")
    @classmethod
    def _ks(cls, ks: list[int]) -> list[int]:
        if any(k < 1 or k > 100 for k in ks):
            raise ValueError("each k must be between 1 and 100")
        return sorted(set(ks))


# --- Configurations ----------------------------------------------------------------------------


class PipelineKind(StrEnum):
    RETRIEVAL = "retrieval"  # retrieval (and reranking) only
    RAG = "rag"  # retrieval -> evidence -> (generation -> claims -> grounding)


class RetrievalTemplate(Model):
    """A RetrievalRequest without the query and corpus version, which each case supplies."""

    top_k: int = Field(default=10, ge=1, le=100)
    strategy: RetrievalStrategy = RetrievalStrategy.SPARSE
    bm25: Bm25Params = Field(default_factory=Bm25Params)
    hybrid: HybridParams = Field(default_factory=HybridParams)
    rerank: RerankParams = Field(default_factory=RerankParams)
    mode: RetrievalMode = RetrievalMode.MANUAL
    router: RouterParams = Field(default_factory=RouterParams)

    def request(self, query: str, version: int) -> RetrievalRequest:
        return RetrievalRequest(query=query, version=version, **self.model_dump())

    @model_validator(mode="after")
    def _valid(self) -> RetrievalTemplate:
        self.request("validation", 0)  # the same rules as a request, minus the query
        return self


class Arm(Model):
    """One configuration under test."""

    name: str = Field(pattern=SLUG)
    label: str = ""
    pipeline: PipelineKind = PipelineKind.RETRIEVAL
    retrieval: RetrievalTemplate = Field(default_factory=RetrievalTemplate)
    evidence: EvidenceParams | None = None
    generation: GenerationParams | None = None
    grounding: GroundingParams = Field(default_factory=GroundingParams)

    @model_validator(mode="after")
    def _pipeline(self) -> Arm:
        if self.pipeline is PipelineKind.RETRIEVAL and (self.evidence or self.generation):
            raise ValueError(f"arm {self.name}: a retrieval arm has no evidence or generation")
        return self

    def evidence_params(self) -> EvidenceParams | None:
        """RAG arms without explicit evidence parameters use the defaults."""
        if self.pipeline is PipelineKind.RAG:
            return self.evidence or EvidenceParams()
        return None


class GeneratorIdentity(Model):
    name: str
    provider: str
    model: str
    revision: str | None
    config_hash: str


class ConfigurationSnapshot(Model):
    """Everything that determines an arm's results, resolved at experiment creation."""

    arm: str
    pipeline: PipelineKind
    dataset_id: str
    dataset_name: str
    dataset_version: int
    dataset_hash: str
    corpus_id: str
    corpus_version: int
    chunking_hash: str
    retrieval_mode: RetrievalMode
    retrieval_template: RetrievalTemplate
    retrieval: RetrievalConfiguration | None = Field(
        description="Manual arms: the resolved retrieval configuration. Adaptive arms choose per "
        "case; each RunCase records the configuration actually used"
    )
    query_analyzer: str | None = Field(description="name@version, adaptive arms only")
    router_policy: str | None = Field(description="name@version, adaptive arms only")
    routing_hash: str | None
    embedder: EmbedderSpec | None
    reranker: RerankerSpec | None
    evidence: EvidenceParams | None
    prompt_template: str | None
    generator: GeneratorIdentity | None
    generation: GenerationParams | None = Field(description="Effective parameters")
    verifier: str | None
    verifier_config_hash: str | None
    metrics: MetricSettings
    metric_versions: dict[str, int]
    engine_version: str

    def config_hash(self) -> str:
        """Identity of the configuration, excluding the arm's name."""
        return canonical_hash(self.model_dump(mode="json", exclude={"arm"}))


class ConfigChange(Model):
    path: str = Field(description="Dotted path in the configuration snapshot")
    baseline: Any
    variant: Any


class AblationSpec(Model):
    baseline: str = Field(description="Arm name")
    variant: str = Field(description="Arm name")
    factor: str = Field(min_length=1, max_length=120, description="What the ablation varies")
    note: str = ""


class Ablation(Model):
    baseline: str
    variant: str
    factor: str
    note: str
    factors: list[str] = Field(description="Arm settings that differ, e.g. retrieval.rerank")
    single_factor: bool = Field(description="Exactly one arm setting differs")
    changes: list[ConfigChange] = Field(description="Every resolved snapshot field that differs")


class RunLimits(Model):
    max_cases: int | None = Field(default=None, ge=1, le=2000, description="First N cases")
    concurrency: int = Field(default=1, ge=1, le=4, description="Cases evaluated in parallel")


class ExperimentCreate(Model):
    name: str = Field(min_length=1, max_length=200)
    hypothesis: str = ""
    dataset_id: str
    arms: list[Arm] = Field(min_length=1, max_length=10)
    ablations: list[AblationSpec] = Field(default_factory=list)
    metrics: MetricSettings = Field(default_factory=MetricSettings)
    limits: RunLimits = Field(default_factory=RunLimits)

    @model_validator(mode="after")
    def _arms(self) -> ExperimentCreate:
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique")
        for a in self.ablations:
            for side in (a.baseline, a.variant):
                if side not in names:
                    raise ValueError(f"ablation references unknown arm {side!r}")
            if a.baseline == a.variant:
                raise ValueError("an ablation compares two different arms")
        return self


class Experiment(Model):
    id: str = Field(default_factory=lambda: new_id("exp"))
    name: str
    hypothesis: str
    dataset_id: str
    dataset_name: str
    dataset_version: int
    dataset_source: DatasetSource
    corpus_id: str
    corpus_version: int
    arms: list[Arm]
    snapshots: list[ConfigurationSnapshot]
    snapshot_hashes: dict[str, str] = Field(description="Arm -> configuration hash")
    ablations: list[Ablation]
    metrics: MetricSettings
    limits: RunLimits
    created_at: datetime = Field(default_factory=utcnow)


# --- Runs ----------------------------------------------------------------------------------------


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"  # every case succeeded in every arm
    PARTIAL = "partial"  # finished; some cases failed and are recorded as failures
    FAILED = "failed"  # the run itself could not proceed


class CaseStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"


class RankedItem(Model):
    rank: int
    chunk_id: str
    document_id: str
    filename: str
    score: float
    upstream_rank: int | None = Field(description="Rank before reranking, if reranked")
    relevance: float | None = Field(description="The case's grade for it; null if not judged")


class GroundingSnapshot(Model):
    status: AnswerGrounding
    claims: int
    factual_claims: int
    supported: int
    weakly_supported: int
    unsupported: int
    grounding_score: float | None
    citation_coverage: float | None
    citation_precision: float | None
    evidence_coverage: float | None


class RunCase(Model):
    """One case evaluated under one arm: what ran, what came out, and its metrics."""

    run_id: str
    experiment_id: str
    arm: str
    case_id: str
    config_hash: str = Field(description="The arm's configuration snapshot hash")
    retrieval_configuration_hash: str | None = Field(
        description="The retrieval configuration that actually ran (adaptive arms vary)"
    )
    status: CaseStatus
    error_type: str | None = None
    error: str | None = None
    route: str | None = Field(
        default=None, description="Adaptive arms: the option the router chose"
    )
    ranking: list[RankedItem] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list, description="Selected evidence filenames")
    answer_status: AnswerStatus | None = None
    answer: str | None = None
    grounding: GroundingSnapshot | None = None
    metrics: list[MetricValue] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    tokens: dict[str, int] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict, description="Component -> model@revision")
    artifact_id: str | None = Field(default=None, description="Full pipeline trace")
    started_at: datetime
    finished_at: datetime


class MetricSummary(Model):
    metric: str
    family: MetricFamily
    version: int
    higher_is_better: bool | None
    n: int = Field(description="Cases where the metric is defined")
    skipped: int = Field(description="Cases where it was skipped (annotations or output absent)")
    mean: float | None
    median: float | None
    std: float | None
    min: float | None
    max: float | None
    ci_low: float | None
    ci_high: float | None
    ci_method: str | None


class ArmSummary(Model):
    arm: str
    label: str
    pipeline: PipelineKind
    config_hash: str
    cases: int
    succeeded: int
    failed: int
    failure_types: dict[str, int]
    metrics: list[MetricSummary]
    skipped_reasons: dict[str, str] = Field(description="Metric -> why some cases skipped it")


class ExperimentRun(Model):
    id: str = Field(default_factory=lambda: new_id("run"))
    experiment_id: str
    status: RunStatus = RunStatus.QUEUED
    dataset_id: str
    dataset_version: int
    dataset_hash: str
    corpus_id: str
    corpus_version: int
    case_ids: list[str]
    arms: list[str]
    snapshot_hashes: dict[str, str]
    limits: RunLimits
    total: int = Field(description="cases x arms")
    completed: int = 0
    failed: int = 0
    metric_registry_version: str
    stats_method: str
    environment: EnvironmentSnapshot
    summaries: list[ArmSummary] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class Artifact(Model):
    id: str = Field(default_factory=lambda: new_id("art"))
    run_id: str
    arm: str
    case_id: str
    kind: str = Field(description="rag_trace | retrieval_trace")
    media_type: str = "application/json"
    sha256: str
    bytes: int
    uri: str = Field(description="Content-addressed blob")


# --- Comparisons -------------------------------------------------------------------------------


class StatsMethod(Model):
    name: str
    version: int
    confidence: float
    bootstrap_resamples: int
    seed: int
    min_cases_for_conclusion: int
    multiple_comparisons: str


class CaseDifference(Model):
    case_id: str
    baseline: float
    variant: float
    difference: float = Field(description="variant - baseline")


class PairedComparison(Model):
    metric: str
    family: MetricFamily
    higher_is_better: bool | None
    n_pairs: int = Field(description="Cases where both arms define the metric")
    baseline_mean: float | None
    variant_mean: float | None
    mean_difference: float | None
    median_difference: float | None
    std_difference: float | None
    ci_low: float | None
    ci_high: float | None
    effect_size_dz: float | None = Field(description="mean / sd of paired differences")
    wins: int = Field(description="Cases where the variant is better")
    losses: int
    ties: int
    sign_test_p: float | None = Field(description="Exact two-sided sign test, ties dropped")
    holm_p: float | None = Field(description="Holm-adjusted across the metrics compared")
    conclusion: str = Field(
        description="insufficient_cases | no_detectable_difference | variant_higher | "
        "variant_lower | descriptive_only"
    )
    differences: list[CaseDifference]


class ArmRef(Model):
    run_id: str
    arm: str
    config_hash: str
    label: str


class Comparison(Model):
    baseline: ArmRef
    variant: ArmRef
    comparable: bool = Field(description="False: no paired statistics are computed")
    issues: list[str] = Field(description="Why the arms cannot be compared")
    warnings: list[str]
    changes: list[ConfigChange]
    factor: str | None = Field(description="The declared ablation factor, if this pair is one")
    method: StatsMethod
    metrics: list[PairedComparison]


class LeaderboardRow(Model):
    position: int | None = Field(description="null when the metric is undefined for the arm")
    arm: str
    label: str
    pipeline: PipelineKind
    config_hash: str
    summary: MetricSummary | None
    ci_overlaps_leader: bool | None
    failed: int
    cases: int
    mean_latency_ms: float | None


class Leaderboard(Model):
    run_id: str
    metric: str
    higher_is_better: bool | None
    rows: list[LeaderboardRow]
    note: str


class CaseView(Model):
    """One benchmark case across every arm of a run."""

    run_id: str
    case: BenchmarkCase
    results: list[RunCase]
