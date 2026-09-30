"""BM25 lexical retriever: the baseline every other strategy is measured against."""

from __future__ import annotations

from collections import Counter
from typing import Any

from rag_forge.domain.models import Bm25Params, Corpus, RetrievalStrategy
from rag_forge.retrieval import bm25
from rag_forge.retrieval.analysis import analyze
from rag_forge.retrieval.base import Candidate, RetrieverOutput
from rag_forge.storage.lexical_index import SqliteLexicalIndex


class Bm25Retriever:
    name = "bm25"
    strategy = RetrievalStrategy.SPARSE

    def __init__(self, index: SqliteLexicalIndex, params: Bm25Params) -> None:
        self.index = index
        self.params = params

    def config(self) -> dict[str, Any]:
        return {"k1": self.params.k1, "b": self.params.b, "analyzer": self.index.analyzer}

    def retrieve(self, corpus: Corpus, version: int, query: str, top_k: int) -> RetrieverOutput:
        indexed_now = self.index.ensure_indexed(corpus, version)
        stats = self.index.stats(corpus, version)
        query_terms = Counter(analyze(query))
        statistics = {
            "candidate_chunks": float(stats.chunks),
            "avg_chunk_length": round(stats.avg_length, 4),
            "indexed_now": float(indexed_now),
        }
        if not query_terms:
            return RetrieverOutput(
                [],
                [],
                statistics,
                ["query has no searchable terms (only stopwords or punctuation)"],
            )
        terms = sorted(query_terms)
        postings = self.index.postings(corpus, version, terms)
        scored = bm25.rank(
            postings, query_terms, stats.chunks, stats.avg_length, self.params, top_k
        )
        statistics["matched_chunks"] = float(len({p.chunk_id for p in postings}))
        return RetrieverOutput(
            [Candidate(s.chunk_id, s.score, s.matched_terms) for s in scored], terms, statistics
        )
