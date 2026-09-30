"""Hybrid retrieval: run component retrievers on the same corpus version, then fuse.

A fixed strategy, not a router: the components and the fusion method come from the request.
If a component cannot run (for example, no ready dense index), the whole request fails with
that component's error. There is never a silent single-retriever fallback.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from rag_forge.domain.models import (
    Corpus,
    FusionMethod,
    HybridParams,
    RetrievalRequest,
    RetrievalStrategy,
)
from rag_forge.retrieval.base import Candidate, Retriever, RetrieverOutput
from rag_forge.retrieval.fusion import FusionStrategy, ReciprocalRankFusion, WeightedScoreFusion

RetrieverFactory = Callable[[RetrievalRequest], Retriever]


class ComponentMismatchError(RuntimeError):
    """Components searched different chunk sets, so their rankings are not comparable."""


def make_fusion(params: HybridParams) -> FusionStrategy:
    if params.fusion is FusionMethod.RRF:
        return ReciprocalRankFusion(params.rrf_k)
    return WeightedScoreFusion(params.weights)


class HybridRetriever:
    strategy = RetrievalStrategy.HYBRID

    def __init__(
        self,
        components: Mapping[RetrievalStrategy, Retriever],
        fusion: FusionStrategy,
        candidate_k: int,
    ) -> None:
        self.components = dict(sorted(components.items()))
        self.fusion = fusion
        self.candidate_k = candidate_k
        self.name = f"hybrid-{fusion.method.value}"

    def config(self) -> dict[str, Any]:
        return {
            "fusion": self.fusion.config(),
            "candidate_k": self.candidate_k,
            "components": {
                s.value: {"retriever": r.name, **r.config()} for s, r in self.components.items()
            },
        }

    def retrieve(self, corpus: Corpus, version: int, query: str, top_k: int) -> RetrieverOutput:
        outputs = {
            s: r.retrieve(corpus, version, query, self.candidate_k)
            for s, r in self.components.items()
        }
        searched = {s.value: o.statistics.get("candidate_chunks") for s, o in outputs.items()}
        if len(set(searched.values())) != 1:
            raise ComponentMismatchError(
                f"component retrievers searched different chunk sets for version {version}: "
                f"{searched}; rebuild the dense index for this version"
            )
        fused = self.fusion.fuse({s: o.candidates for s, o in outputs.items()}, top_k)
        matched = {
            c.chunk_id: c.matched_terms
            for o in outputs.values()
            for c in o.candidates
            if c.matched_terms
        }
        statistics = {
            f"{s.value}.{key}": value
            for s, o in outputs.items()
            for key, value in o.statistics.items()
        }
        statistics["candidate_chunks"] = float(next(iter(searched.values())) or 0)
        statistics["fused_candidates"] = float(
            len({c.chunk_id for o in outputs.values() for c in o.candidates})
        )
        sparse = outputs.get(RetrievalStrategy.SPARSE)
        return RetrieverOutput(
            candidates=[
                Candidate(f.chunk_id, f.detail.score, matched.get(f.chunk_id, ()), f.detail)
                for f in fused
            ],
            query_terms=sparse.query_terms if sparse else [],
            statistics=statistics,
            warnings=[f"{s.value}: {w}" for s, o in outputs.items() for w in o.warnings],
            index_id=next((o.index_id for o in outputs.values() if o.index_id), None),
        )


def hybrid_factory(components: Mapping[RetrievalStrategy, RetrieverFactory]) -> RetrieverFactory:
    """Factory for the service's registry: builds components per request from their factories."""

    def factory(request: RetrievalRequest) -> Retriever:
        params = request.hybrid
        return HybridRetriever(
            {s: components[s](request) for s in params.retrievers},
            make_fusion(params),
            params.candidate_k,
        )

    return factory
