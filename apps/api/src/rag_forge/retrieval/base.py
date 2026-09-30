"""The contract every retrieval strategy implements.

A retriever searches exactly one corpus version and returns ranked chunk ids with scores and
its own statistics. Everything else (resolving the corpus, loading chunks, provenance, the HTTP
shape) is the service's job, so a dense or hybrid retriever only has to implement `retrieve`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_forge.domain.models import Corpus, FusionDetail, RetrievalStrategy


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    score: float
    matched_terms: tuple[str, ...] = ()
    fusion: FusionDetail | None = None  # set by hybrid retrievers


@dataclass(frozen=True)
class RetrieverOutput:
    candidates: list[Candidate]  # ranked best first, at most top_k, ties broken deterministically
    query_terms: list[str]
    statistics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    index_id: str | None = None  # the concrete index searched, when the retriever has one


class Retriever(Protocol):
    name: str
    strategy: RetrievalStrategy

    def config(self) -> dict[str, Any]:
        """Everything that affects results; hashed into provenance."""
        ...

    def retrieve(self, corpus: Corpus, version: int, query: str, top_k: int) -> RetrieverOutput: ...
