"""Arm templates and suggested ablations for the experiment builder. Nothing runs by default:
a researcher picks, edits and combines these into an experiment's matrix."""

from __future__ import annotations

from rag_forge.domain.arena import AblationSpec, Arm, PipelineKind, RetrievalTemplate
from rag_forge.domain.models import (
    FusionMethod,
    GenerationParams,
    HybridParams,
    RerankParams,
    RetrievalMode,
    RetrievalStrategy,
)

S, D, H = RetrievalStrategy.SPARSE, RetrievalStrategy.DENSE, RetrievalStrategy.HYBRID
RERANK = RerankParams(enabled=True, candidate_k=50)
WEIGHTED = HybridParams(fusion=FusionMethod.WEIGHTED)


def _t(strategy: RetrievalStrategy, **kw: object) -> RetrievalTemplate:
    return RetrievalTemplate.model_validate({"strategy": strategy, **kw})


ARMS: list[Arm] = [
    Arm(name="bm25", label="BM25", retrieval=_t(S)),
    Arm(name="dense", label="Dense", retrieval=_t(D)),
    Arm(name="hybrid-rrf", label="Hybrid RRF", retrieval=_t(H)),
    Arm(name="hybrid-weighted", label="Hybrid weighted 0.5/0.5", retrieval=_t(H, hybrid=WEIGHTED)),
    Arm(
        name="adaptive",
        label="Adaptive router",
        retrieval=RetrievalTemplate(mode=RetrievalMode.ADAPTIVE),
    ),
    Arm(name="bm25-rerank", label="BM25 + cross-encoder", retrieval=_t(S, rerank=RERANK)),
    Arm(name="dense-rerank", label="Dense + cross-encoder", retrieval=_t(D, rerank=RERANK)),
    Arm(
        name="hybrid-rrf-rerank", label="Hybrid RRF + cross-encoder", retrieval=_t(H, rerank=RERANK)
    ),
    Arm(
        name="rag-extractive",
        label="Grounded RAG: hybrid + rerank, extractive baseline",
        pipeline=PipelineKind.RAG,
        retrieval=_t(H, rerank=RERANK),
        generation=GenerationParams(generator="extractive-baseline"),
    ),
    Arm(
        name="rag-local",
        label="Grounded RAG: hybrid + rerank, local model",
        pipeline=PipelineKind.RAG,
        retrieval=_t(H, rerank=RERANK),
        generation=GenerationParams(),  # the configured default generator
    ),
]

ABLATIONS: list[AblationSpec] = [
    AblationSpec(baseline="bm25", variant="dense", factor="retrieval strategy"),
    AblationSpec(baseline="bm25", variant="hybrid-rrf", factor="retrieval strategy"),
    AblationSpec(baseline="hybrid-rrf", variant="hybrid-weighted", factor="fusion method"),
    AblationSpec(baseline="bm25", variant="bm25-rerank", factor="reranking"),
    AblationSpec(baseline="hybrid-rrf", variant="hybrid-rrf-rerank", factor="reranking"),
    AblationSpec(baseline="hybrid-rrf-rerank", variant="adaptive", factor="fixed vs adaptive"),
    AblationSpec(baseline="rag-extractive", variant="rag-local", factor="generator"),
]
