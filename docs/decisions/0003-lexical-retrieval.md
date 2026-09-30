# 0003: Retrieval contract and lexical baseline

Date: 2026-09-30 · Status: accepted

## Decisions

1. **One `Retriever` protocol, registered per strategy.** The service owns resolution, hydration
   and provenance; a retriever only ranks chunk ids for one corpus version. New strategies are a
   new class plus one registration line in `create_app()`.
2. **BM25 is the baseline.** Transparent, cheap, and the standard reference point; later
   strategies are judged against it on identical corpus versions.
3. **Own inverted index in SQLite instead of FTS5.** FTS5's `bm25()` uses table-wide statistics,
   so scores for one corpus version would drift as other corpora or versions are indexed. The
   custom index computes N, average length and document frequency over exactly the searched
   version.
4. **Index per chunk, not per corpus version.** Chunks are immutable and shared across versions;
   indexing each once and joining through membership avoids duplicating the index per version.
5. **Lazy, idempotent indexing on first query.** No indexing coupling in ingestion, automatic
   backfill of Phase 1 data, and the cost is reported in `indexed_now`.
6. **Analyzer identity is versioned** (`…@1`) and part of every index row and retriever config
   hash, like `CHUNKER_VERSION` for chunks.
7. **No stemming or stop-list tuning.** A transparent, reproducible baseline matters more than a
   few points of recall; stemmed or tuned variants can be separate analyzers later.
8. **Forward-only, transactional migrations** keyed by `PRAGMA user_version`.
9. **Unregistered strategies return 501** with `NotImplementedDetail`, consistent with 0001.
10. **Retrieval responses are not persisted yet.** They carry full provenance; persisting runs
    belongs with experiments.
