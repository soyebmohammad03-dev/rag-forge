"""Fusion: combine ranked lists from several retrievers into one ranking.

Fusion only sees ranked candidates per strategy; it knows nothing about how they were produced.
Every fused result keeps each component's rank and raw score, plus that component's
contribution to the fused score, so fusion is fully inspectable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rag_forge.domain.models import ComponentScore, FusionDetail, FusionMethod, RetrievalStrategy
from rag_forge.retrieval.base import Candidate

RankedLists = Mapping[RetrievalStrategy, Sequence[Candidate]]

# Fused scores are rounded before ranking so that float summation order (1/61 + 1/62 vs
# 1/62 + 1/61) can never break a genuine tie differently; ties then go to chunk id.
_ROUND = 12


@dataclass(frozen=True)
class FusedCandidate:
    chunk_id: str
    detail: FusionDetail


class FusionStrategy(Protocol):
    method: FusionMethod

    def config(self) -> dict[str, Any]: ...
    def fuse(self, lists: RankedLists, top_k: int) -> list[FusedCandidate]: ...


def _positions(lists: RankedLists) -> dict[str, dict[RetrievalStrategy, tuple[int, float]]]:
    """chunk -> strategy -> (1-based rank, raw score). Duplicates within a list keep the first."""
    seen: dict[str, dict[RetrievalStrategy, tuple[int, float]]] = {}
    for strategy in sorted(lists):
        for rank, c in enumerate(lists[strategy], 1):
            seen.setdefault(c.chunk_id, {}).setdefault(strategy, (rank, c.score))
    return seen


def _rank(
    scored: dict[str, tuple[float, list[ComponentScore]]], method: FusionMethod, top_k: int
) -> list[FusedCandidate]:
    ordered = sorted(scored.items(), key=lambda kv: (-round(kv[1][0], _ROUND), kv[0]))
    return [
        FusedCandidate(cid, FusionDetail(method=method, score=score, components=components))
        for cid, (score, components) in ordered[:top_k]
    ]


class ReciprocalRankFusion:
    """RRF(d) = sum over lists containing d of 1 / (k + rank(d)). Scores are ignored entirely."""

    method = FusionMethod.RRF

    def __init__(self, k: int = 60) -> None:
        if k < 1:
            raise ValueError("RRF k must be at least 1")
        self.k = k

    def config(self) -> dict[str, Any]:
        return {"method": self.method.value, "k": self.k}

    def fuse(self, lists: RankedLists, top_k: int) -> list[FusedCandidate]:
        scored = {}
        for chunk_id, hits in _positions(lists).items():
            components = []
            for strategy in sorted(lists):
                rank, raw = hits.get(strategy, (None, None))
                share = 1 / (self.k + rank) if rank is not None else 0.0
                components.append(
                    ComponentScore(strategy=strategy, rank=rank, score=raw, contribution=share)
                )
            scored[chunk_id] = (sum(c.contribution for c in components), components)
        return _rank(scored, self.method, top_k)


def min_max(candidates: Sequence[Candidate]) -> dict[str, float]:
    """Rescale one list's scores to 0..1 (best = 1). A constant list maps to 1.0 throughout.

    Scale-free and order-preserving, so BM25 (unbounded) and cosine (bounded) become
    combinable. It is relative to the retrieved candidates, so it depends on candidate_k.
    """
    if not candidates:
        return {}
    lo = min(c.score for c in candidates)
    hi = max(c.score for c in candidates)
    span = hi - lo
    return {c.chunk_id: (c.score - lo) / span if span else 1.0 for c in candidates}


class WeightedScoreFusion:
    """sum over components of weight * min_max(score); a chunk absent from a list gets 0 there."""

    method = FusionMethod.WEIGHTED

    def __init__(self, weights: Mapping[RetrievalStrategy, float]) -> None:
        if any(not 0 <= w <= 1 for w in weights.values()) or abs(sum(weights.values()) - 1) > 1e-9:
            raise ValueError("weights must each be in 0..1 and sum to 1")
        self.weights = dict(weights)

    def config(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "normalization": "min-max per list",
            "weights": {s.value: w for s, w in sorted(self.weights.items())},
        }

    def fuse(self, lists: RankedLists, top_k: int) -> list[FusedCandidate]:
        if set(lists) != set(self.weights):
            raise ValueError("weighted fusion needs exactly one list per weighted strategy")
        normalized = {s: min_max(lists[s]) for s in lists}
        scored = {}
        for chunk_id, hits in _positions(lists).items():
            components = []
            for strategy in sorted(lists):
                rank, raw = hits.get(strategy, (None, None))
                norm = normalized[strategy].get(chunk_id)
                share = self.weights[strategy] * norm if norm is not None else 0.0
                components.append(
                    ComponentScore(
                        strategy=strategy,
                        rank=rank,
                        score=raw,
                        normalized_score=norm,
                        contribution=share,
                    )
                )
            scored[chunk_id] = (sum(c.contribution for c in components), components)
        return _rank(scored, self.method, top_k)
