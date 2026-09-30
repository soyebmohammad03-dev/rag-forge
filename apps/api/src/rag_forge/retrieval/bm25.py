"""Okapi BM25 over caller-supplied statistics, so they can be scoped to one corpus version."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from rag_forge.domain.models import Bm25Params


@dataclass(frozen=True)
class Posting:
    chunk_id: str
    term: str
    tf: int
    length: int  # analysed tokens in the chunk


@dataclass(frozen=True)
class Scored:
    chunk_id: str
    score: float
    matched_terms: tuple[str, ...]


def idf(n_docs: int, df: int) -> float:
    """Lucene's BM25 IDF: always positive, even for terms in more than half the chunks."""
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def rank(
    postings: Iterable[Posting],
    query_terms: Counter[str],
    n_docs: int,
    avg_length: float,
    params: Bm25Params,
    top_k: int,
) -> list[Scored]:
    """Score every chunk with at least one query term; return the top_k.

    Deterministic: contributions are summed in (chunk, term) order and ties are broken by
    chunk id, so identical inputs always give identical scores and order.
    """
    by_chunk: dict[str, list[Posting]] = defaultdict(list)
    for p in postings:
        by_chunk[p.chunk_id].append(p)
    df = Counter(p.term for ps in by_chunk.values() for p in ps)
    k1, b = params.k1, params.b
    scored = []
    for chunk_id in sorted(by_chunk):
        score = 0.0
        terms = sorted(by_chunk[chunk_id], key=lambda p: p.term)
        for p in terms:
            norm = 1 - b + b * (p.length / avg_length if avg_length else 0)
            score += (
                query_terms[p.term] * idf(n_docs, df[p.term]) * p.tf * (k1 + 1) / (p.tf + k1 * norm)
            )
        scored.append(Scored(chunk_id, score, tuple(p.term for p in terms)))
    scored.sort(key=lambda s: (-s.score, s.chunk_id))
    return scored[:top_k]
