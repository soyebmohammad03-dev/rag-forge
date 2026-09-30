"""HTTP routes, grouped by capability. Split into modules once a group outgrows a screen."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, NoReturn

from fastapi import APIRouter, HTTPException, Request, status

from rag_forge import __version__
from rag_forge.api.schemas import (
    ComponentHealth,
    ComponentState,
    EnvironmentResponse,
    ExperimentCreate,
    HealthStatus,
    NotImplementedDetail,
    RetrievalEvaluationRequest,
    RetrievalEvaluationResponse,
    RetrievalRequest,
    RetrievalResponse,
    RouterRequest,
    RouterResponse,
)
from rag_forge.domain.models import Experiment, ExperimentRun, Metric
from rag_forge.evaluation import retrieval_metrics as rm
from rag_forge.provenance.environment import capture_environment
from rag_forge.storage.base import CorpusStore

router = APIRouter(prefix="/api/v1")

NOT_IMPLEMENTED: dict[int | str, dict[str, Any]] = {501: {"model": NotImplementedDetail}}


def _store(request: Request) -> CorpusStore:
    store: CorpusStore = request.app.state.store
    return store


def _not_implemented(capability: str) -> NoReturn:
    raise HTTPException(
        status.HTTP_501_NOT_IMPLEMENTED,
        detail=NotImplementedDetail(
            capability=capability,
            message=f"{capability} is defined by contract but not implemented yet.",
        ).model_dump(),
    )


# --- System -------------------------------------------------------------------


@router.get("/health", response_model=HealthStatus, tags=["system"])
def health(request: Request) -> HealthStatus:
    started_at: datetime = request.app.state.started_at
    store = _store(request)
    components = [
        ComponentHealth(name="api", state=ComponentState.OK, detail=f"rag-forge {__version__}"),
        ComponentHealth(
            name="metadata_store",
            state=ComponentState.OK,
            detail=f"{store.kind}: {getattr(store, 'path', '')}",
        ),
        ComponentHealth(
            name="vector_index", state=ComponentState.NOT_CONFIGURED, detail="no index backend"
        ),
        ComponentHealth(
            name="llm_provider", state=ComponentState.NOT_CONFIGURED, detail="no provider"
        ),
        ComponentHealth(name="reranker", state=ComponentState.NOT_CONFIGURED, detail="no reranker"),
    ]
    healthy = all(c.state != ComponentState.ERROR for c in components)
    return HealthStatus(
        status="ok" if healthy else "degraded",
        version=__version__,
        started_at=started_at,
        uptime_seconds=(datetime.now(UTC) - started_at).total_seconds(),
        components=components,
    )


@router.get("/provenance/environment", response_model=EnvironmentResponse, tags=["provenance"])
def environment() -> EnvironmentResponse:
    return EnvironmentResponse(environment=capture_environment())


# --- Retrieval & routing -------------------------------------------------------


@router.post(
    "/retrieval/search",
    response_model=RetrievalResponse,
    tags=["retrieval"],
    responses=NOT_IMPLEMENTED,
)
def search(body: RetrievalRequest) -> RetrievalResponse:
    _not_implemented("Retrieval")


@router.post(
    "/router/decide", response_model=RouterResponse, tags=["router"], responses=NOT_IMPLEMENTED
)
def decide(body: RouterRequest) -> RouterResponse:
    _not_implemented("Router policy")


# --- Experiments --------------------------------------------------------------


@router.get("/experiments", response_model=list[Experiment], tags=["experiments"])
def list_experiments(request: Request) -> list[Experiment]:
    return _store(request).list_experiments()


@router.post(
    "/experiments",
    response_model=Experiment,
    status_code=status.HTTP_201_CREATED,
    tags=["experiments"],
)
def create_experiment(body: ExperimentCreate, request: Request) -> Experiment:
    store = _store(request)
    if body.corpus_id is not None and store.get_corpus(body.corpus_id) is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail="unknown corpus_id")
    return store.add_experiment(Experiment(**body.model_dump()))


@router.get("/runs", response_model=list[ExperimentRun], tags=["experiments"])
def list_runs(request: Request) -> list[ExperimentRun]:
    return _store(request).list_runs()


# --- Evaluation ---------------------------------------------------------------


@router.post(
    "/evaluation/retrieval", response_model=RetrievalEvaluationResponse, tags=["evaluation"]
)
def evaluate_retrieval(body: RetrievalEvaluationRequest) -> RetrievalEvaluationResponse:
    ranked, rel = body.ranked_ids, body.relevance
    try:
        metrics = [Metric(name="mrr", value=rm.reciprocal_rank(ranked, rel))]
        for k in sorted(set(body.ks)):
            metrics += [
                Metric(name="recall", k=k, value=rm.recall_at_k(ranked, rel, k)),
                Metric(name="precision", k=k, value=rm.precision_at_k(ranked, rel, k)),
                Metric(name="ndcg", k=k, value=rm.ndcg_at_k(ranked, rel, k)),
            ]
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    return RetrievalEvaluationResponse(metrics=metrics)
