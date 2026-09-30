"""Standard ranked-retrieval metrics for a single query.

`ranked` is the retriever's output order (best first). `relevance` maps ids to graded
relevance; any id with grade > 0 counts as relevant for binary metrics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _relevant(relevance: Mapping[str, float]) -> set[str]:
    rel = {doc_id for doc_id, grade in relevance.items() if grade > 0}
    if not rel:
        raise ValueError("at least one relevant id is required")
    return rel


def recall_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    rel = _relevant(relevance)
    return len(rel.intersection(ranked[:k])) / len(rel)


def precision_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _relevant(relevance)
    return len(rel.intersection(ranked[:k])) / k


def reciprocal_rank(ranked: Sequence[str], relevance: Mapping[str, float]) -> float:
    rel = _relevant(relevance)
    return next((1.0 / i for i, doc_id in enumerate(ranked, 1) if doc_id in rel), 0.0)


def ndcg_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    _relevant(relevance)

    def dcg(grades: Sequence[float]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))

    ideal = dcg(sorted(relevance.values(), reverse=True))
    return dcg([relevance.get(doc_id, 0.0) for doc_id in ranked]) / ideal
