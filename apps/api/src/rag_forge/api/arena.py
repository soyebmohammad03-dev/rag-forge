"""Arena routes: benchmark datasets, experiments, runs, results, leaderboards and comparisons."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from rag_forge.api.schemas import NotImplementedDetail
from rag_forge.arena import metrics, presets, stats
from rag_forge.arena.config import factors, snapshot_diff
from rag_forge.arena.datasets import DatasetValidationError
from rag_forge.arena.engine import ArenaEngine, ArenaNotFoundError, RunNotFinishedError
from rag_forge.domain.arena import (
    AblationSpec,
    Arm,
    BenchmarkDataset,
    BenchmarkDatasetCreate,
    CaseStatus,
    CaseView,
    Comparison,
    ConfigChange,
    ConfigurationSnapshot,
    DatasetSource,
    DatasetSummary,
    Experiment,
    ExperimentCreate,
    ExperimentRun,
    Leaderboard,
    MetricDefinition,
    MetricSettings,
    RunCase,
    RunStatus,
    StatsMethod,
)
from rag_forge.domain.models import RagResponse, RetrievalResponse
from rag_forge.ingestion.service import CorpusNotFoundError
from rag_forge.rag.service import RagComponentNotAvailableError
from rag_forge.retrieval.service import RerankerNotAvailableError, StrategyNotAvailableError
from rag_forge.router.service import RouterComponentNotAvailableError

router = APIRouter(prefix="/api/v1", tags=["arena"])
NOT_IMPLEMENTED: dict[int | str, dict[str, object]] = {501: {"model": NotImplementedDetail}}


def _engine(request: Request) -> ArenaEngine:
    engine: ArenaEngine = request.app.state.arena
    return engine


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except (ArenaNotFoundError, CorpusNotFoundError) as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DatasetValidationError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "the dataset does not match the corpus version",
                "problems": exc.problems,
            },
        ) from exc
    except (
        RouterComponentNotAvailableError,
        RerankerNotAvailableError,
        StrategyNotAvailableError,
        RagComponentNotAvailableError,
    ) as exc:
        detail = NotImplementedDetail(capability="Arena configuration", message=str(exc))
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, detail=detail.model_dump()) from exc
    except RunNotFinishedError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


def _summary(ds: BenchmarkDataset) -> DatasetSummary:
    return DatasetSummary(
        id=ds.id,
        name=ds.name,
        version=ds.version,
        source=ds.source,
        description=ds.description,
        corpus_id=ds.corpus_id,
        corpus_version=ds.corpus_version,
        case_count=len(ds.cases),
        annotation_counts=ds.annotation_counts,
        content_hash=ds.content_hash,
        created_at=ds.created_at,
    )


# --- overview, definitions, presets -----------------------------------------------------------


class RunDigest(BaseModel):
    run_id: str
    experiment_id: str
    experiment_name: str
    status: RunStatus
    dataset_name: str
    dataset_version: int
    dataset_source: DatasetSource
    arms: list[str]
    total: int
    completed: int
    failed: int
    created_at: datetime
    finished_at: datetime | None


class ArenaOverview(BaseModel):
    datasets: list[DatasetSummary]
    experiments: int
    runs: list[RunDigest] = Field(description="Most recent first")
    metric_registry: str
    stats_method: StatsMethod


class ArenaPresets(BaseModel):
    arms: list[Arm]
    ablations: list[AblationSpec]
    metrics: MetricSettings


def _digest(run: ExperimentRun, e: Experiment) -> RunDigest:
    return RunDigest(
        run_id=run.id,
        experiment_id=e.id,
        experiment_name=e.name,
        status=run.status,
        dataset_name=e.dataset_name,
        dataset_version=e.dataset_version,
        dataset_source=e.dataset_source,
        arms=run.arms,
        total=run.total,
        completed=run.completed,
        failed=run.failed,
        created_at=run.created_at,
        finished_at=run.finished_at,
    )


@router.get("/arena/overview", response_model=ArenaOverview)
def overview(request: Request) -> ArenaOverview:
    engine = _engine(request)
    experiments = {e.id: e for e in engine.arena.list_experiments()}
    runs = [_digest(r, experiments[r.experiment_id]) for r in engine.arena.list_runs()[:20]]
    return ArenaOverview(
        datasets=[_summary(d) for d in engine.arena.list_datasets()],
        experiments=len(experiments),
        runs=runs,
        metric_registry=metrics.REGISTRY_VERSION,
        stats_method=stats.METHOD,
    )


@router.get("/arena/metrics", response_model=list[MetricDefinition])
def metric_definitions() -> list[MetricDefinition]:
    """Every metric: family, version, required annotations and outputs, unit, direction."""
    return metrics.definitions()


@router.get("/arena/presets", response_model=ArenaPresets)
def arena_presets() -> ArenaPresets:
    """Arm templates and suggested ablations. Templates, not results: nothing runs by default."""
    return ArenaPresets(arms=presets.ARMS, ablations=presets.ABLATIONS, metrics=MetricSettings())


# --- datasets -------------------------------------------------------------------------------


@router.get("/benchmarks", response_model=list[DatasetSummary])
def list_datasets(request: Request) -> list[DatasetSummary]:
    return [_summary(d) for d in _engine(request).arena.list_datasets()]


@router.post("/benchmarks", response_model=BenchmarkDataset, status_code=status.HTTP_201_CREATED)
def create_dataset(body: BenchmarkDatasetCreate, request: Request) -> BenchmarkDataset:
    """Register a user benchmark. Annotations are checked against the pinned corpus version.
    Unchanged content returns the existing version; changed content creates the next one."""
    with _errors():
        return _engine(request).create_dataset(body)


@router.post(
    "/benchmarks/development",
    response_model=BenchmarkDataset,
    status_code=status.HTTP_201_CREATED,
)
def install_development(request: Request) -> BenchmarkDataset:
    """Create the bundled development corpus (and its dense index) and register its dataset.
    Idempotent. The development set validates the pipeline; it does not rank methods."""
    with _errors():
        return _engine(request).install_development()


@router.get("/benchmarks/{dataset_id}", response_model=BenchmarkDataset)
def get_dataset(dataset_id: str, request: Request) -> BenchmarkDataset:
    with _errors():
        return _engine(request).dataset(dataset_id)


# --- configurations and experiments -----------------------------------------------------------


class ResolveRequest(BaseModel):
    dataset_id: str
    arms: list[Arm] = Field(min_length=1, max_length=10)
    metrics: MetricSettings = Field(default_factory=MetricSettings)


class ResolvedArm(BaseModel):
    snapshot: ConfigurationSnapshot
    config_hash: str


class PairDiff(BaseModel):
    baseline: str
    variant: str
    factors: list[str]
    changes: list[ConfigChange]


class ResolveResponse(BaseModel):
    arms: list[ResolvedArm]
    diffs: list[PairDiff] = Field(description="Each arm against the first")
    dense_index_ready: bool


@router.post(
    "/arena/configurations/resolve", response_model=ResolveResponse, responses=NOT_IMPLEMENTED
)
def resolve_configurations(body: ResolveRequest, request: Request) -> ResolveResponse:
    """Snapshots, hashes and differences for a candidate matrix, without creating anything."""
    engine = _engine(request)
    with _errors():
        spec = ExperimentCreate(
            name="preview", dataset_id=body.dataset_id, arms=body.arms, metrics=body.metrics
        )
        snaps = engine.resolve(body.dataset_id, body.arms, spec)
        ds = engine.dataset(body.dataset_id)
        ready = engine.dense_ready(ds.corpus_id, ds.corpus_version)
    first = body.arms[0]
    return ResolveResponse(
        arms=[ResolvedArm(snapshot=s, config_hash=s.config_hash()) for s in snaps],
        diffs=[
            PairDiff(
                baseline=first.name,
                variant=a.name,
                factors=factors(first, a),
                changes=snapshot_diff(snaps[0], s),
            )
            for a, s in zip(body.arms[1:], snaps[1:], strict=True)
        ],
        dense_index_ready=ready,
    )


@router.get("/experiments", response_model=list[Experiment])
def list_experiments(request: Request) -> list[Experiment]:
    return _engine(request).arena.list_experiments()


@router.post(
    "/experiments",
    response_model=Experiment,
    status_code=status.HTTP_201_CREATED,
    responses=NOT_IMPLEMENTED,
)
def create_experiment(body: ExperimentCreate, request: Request) -> Experiment:
    """Resolve every arm into an immutable configuration snapshot and validate the ablations."""
    with _errors():
        return _engine(request).create_experiment(body)


@router.get("/experiments/{experiment_id}", response_model=Experiment)
def get_experiment(experiment_id: str, request: Request) -> Experiment:
    with _errors():
        return _engine(request).experiment(experiment_id)


@router.post(
    "/experiments/{experiment_id}/runs",
    response_model=ExperimentRun,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_run(experiment_id: str, request: Request, background: BackgroundTasks) -> ExperimentRun:
    """Queue a run of every arm over the dataset's cases; poll GET /runs/{id} for progress."""
    engine = _engine(request)
    with _errors():
        run = engine.create_run(experiment_id)
    background.add_task(engine.run, run.id)
    return run


@router.get("/experiments/{experiment_id}/runs", response_model=list[ExperimentRun])
def experiment_runs(experiment_id: str, request: Request) -> list[ExperimentRun]:
    with _errors():
        _engine(request).experiment(experiment_id)
    return _engine(request).arena.list_runs(experiment_id)


# --- runs and results -------------------------------------------------------------------------


@router.get("/runs", response_model=list[RunDigest])
def list_runs(request: Request) -> list[RunDigest]:
    engine = _engine(request)
    experiments = {e.id: e for e in engine.arena.list_experiments()}
    return [_digest(r, experiments[r.experiment_id]) for r in engine.arena.list_runs()]


@router.get("/runs/{run_id}", response_model=ExperimentRun)
def get_run(run_id: str, request: Request) -> ExperimentRun:
    with _errors():
        return _engine(request).get_run(run_id)


@router.get("/runs/{run_id}/cases", response_model=list[RunCase])
def run_cases(
    run_id: str,
    request: Request,
    arm: Annotated[str | None, Query()] = None,
    status_: Annotated[CaseStatus | None, Query(alias="status")] = None,
) -> list[RunCase]:
    """Per-case results, in dataset order then arm order. Failed cases are included."""
    engine = _engine(request)
    with _errors():
        run = engine.get_run(run_id)
    order = {c: i for i, c in enumerate(run.case_ids)}
    arms = {a: i for i, a in enumerate(run.arms)}
    cases = [
        c for c in engine.arena.run_cases(run_id, arm) if status_ is None or c.status is status_
    ]
    return sorted(cases, key=lambda c: (order.get(c.case_id, 0), arms.get(c.arm, 0)))


@router.get("/runs/{run_id}/cases/{case_id}", response_model=CaseView)
def case_view(run_id: str, case_id: str, request: Request) -> CaseView:
    """One benchmark case, with its annotations, across every arm of the run."""
    with _errors():
        return _engine(request).case_view(run_id, case_id)


@router.get("/runs/{run_id}/leaderboard", response_model=Leaderboard)
def leaderboard(
    run_id: str, request: Request, metric: Annotated[str, Query()] = "ndcg@10"
) -> Leaderboard:
    with _errors():
        return _engine(request).leaderboard(run_id, metric)


@router.get("/runs/{run_id}/compare", response_model=Comparison)
def compare_arms(
    run_id: str,
    request: Request,
    baseline: Annotated[str, Query()],
    variant: Annotated[str, Query()],
    metric: Annotated[list[str] | None, Query()] = None,
) -> Comparison:
    """Paired comparison of two arms of one run over the cases both evaluated."""
    with _errors():
        return _engine(request).compare(run_id, baseline, run_id, variant, metric)


@router.get("/arena/compare", response_model=Comparison)
def compare_runs(
    request: Request,
    baseline_run: Annotated[str, Query()],
    baseline_arm: Annotated[str, Query()],
    variant_run: Annotated[str, Query()],
    variant_arm: Annotated[str, Query()],
    metric: Annotated[list[str] | None, Query()] = None,
) -> Comparison:
    """Paired comparison across runs. Refused (comparable: false) when the datasets, corpus
    versions or metric definitions differ."""
    with _errors():
        return _engine(request).compare(
            baseline_run, baseline_arm, variant_run, variant_arm, metric
        )


class ArtifactTrace(BaseModel):
    artifact_id: str
    kind: str
    rag: RagResponse | None = Field(description="rag_trace artifacts")
    retrieval: RetrievalResponse | None = Field(description="retrieval_trace artifacts")


@router.get("/artifacts/{artifact_id}", response_model=ArtifactTrace)
def artifact(artifact_id: str, request: Request, response: Response) -> ArtifactTrace:
    """The full pipeline trace recorded for one case of one arm."""
    with _errors():
        kind, body = _engine(request).trace(artifact_id)
    response.headers["Cache-Control"] = "private, max-age=31536000, immutable"
    if kind == "rag_trace":
        return ArtifactTrace(
            artifact_id=artifact_id, kind=kind, rag=RagResponse.model_validate(body), retrieval=None
        )
    return ArtifactTrace(
        artifact_id=artifact_id,
        kind=kind,
        rag=None,
        retrieval=RetrievalResponse.model_validate(body),
    )
