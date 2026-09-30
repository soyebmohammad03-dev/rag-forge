# Retrieval

Code: `apps/api/src/rag_forge/retrieval/` and `apps/api/src/rag_forge/storage/lexical_index.py`.

## Flow

```
POST /api/v1/corpora/{id}/retrieve  { query, top_k, version?, strategy, bm25 }
        │
RetrievalService
  1. resolve corpus (404) and version (default current; > current → 404)
  2. pick the retriever registered for `strategy` (unregistered → 501)
  3. retriever.retrieve(corpus, version, query, top_k) → ranked chunk ids + scores + statistics
  4. load chunks and document versions; build hits (rank, score, chunk, filename, doc version)
  5. attach provenance (corpus version, chunking hash, retriever config + hash, query terms,
     statistics, environment, elapsed time)
        │
RetrievalResponse { query, hits[], provenance, warnings[] }
```

## Retrieval contract

`retrieval/base.py` defines `Retriever`: a `name`, a `strategy`, `config()` (everything that
affects results) and `retrieve(corpus, version, query, top_k) -> RetrieverOutput`. A retriever:

- searches exactly one corpus version, under the corpus's chunking config;
- returns at most `top_k` candidates, best first, ties broken deterministically;
- reports its own statistics and warnings.

Strategies are registered in `create_app()` as `{RetrievalStrategy: factory(request) -> Retriever}`.
Adding dense retrieval means implementing `Retriever` and registering it for
`RetrievalStrategy.DENSE`; the service, API, response shape and UI do not change. A hybrid
retriever can wrap other retrievers and fuse their outputs behind the same contract.

## Lexical baseline: BM25

Why lexical first: it needs no model, no GPU and no external service; it is transparent (every
score decomposes into term statistics); it is strong on identifiers, rare terms and exact phrases;
and it is the standard baseline in IR evaluation (e.g. BEIR). Every later strategy has to beat it
on the same corpus version to justify its cost.

- **Analyzer** (`retrieval/analysis.py`, `nfkc-casefold-word-lucene33@1`): NFKC normalisation,
  casefolding, Unicode word tokens, Lucene's 33-word English stop set. No stemming, so `fuse` does
  not match `fuses`; that is a deliberate property of the baseline, not a bug.
- **Scoring** (`retrieval/bm25.py`): Okapi BM25 with Lucene's IDF `ln(1 + (N − df + 0.5)/(df + 0.5))`.
  Defaults `k1 = 1.2`, `b = 0.75`, both settable per request.
- **Determinism**: contributions are summed in (chunk, term) order; ties are broken by chunk id.

## Index design

```
lexical_terms     (id, term UNIQUE)
lexical_docs      (doc_key INTEGER PK, analyzer, chunk_id → chunks.id, length, UNIQUE(analyzer, chunk_id))
lexical_postings  (term_id, doc_key, tf)  PRIMARY KEY (term_id, doc_key), WITHOUT ROWID
```

- The index is keyed by **chunk and analyzer**, not by corpus. Chunks are immutable and shared by
  every corpus version containing them, so each chunk is indexed once.
- A corpus version is searched by joining postings through `corpus_version_members` and
  `chunks.chunking_hash`. Chunks from other corpora, other versions or other chunking configs can
  never appear.
- **Statistics are scoped to the searched version**: `N`, average chunk length and document
  frequency come from that version's chunks only. Ingesting into another corpus, or creating a
  later version, never changes the scores of an existing version (tested).
- Only terms and counts are stored; chunk text stays in `chunks`.
- Indexing is lazy and idempotent: the first query on a version indexes any of its chunks not yet
  indexed (`statistics.indexed_now` reports how many). The index is derived data and can be
  rebuilt from chunks at any time.
- `doc_key` is an explicit `INTEGER PRIMARY KEY` so it survives `VACUUM` (implicit rowids do not).

### Why not SQLite FTS5

FTS5's `bm25()` uses statistics over the whole FTS table. With several corpora and versions in one
table, a version's scores would change whenever anything else was indexed, which breaks
reproducible comparison. A per-version FTS table would avoid that but duplicate the index for
every version. The small inverted index above gives version-scoped statistics with each chunk
indexed once.

## Provenance

Every response carries: corpus id and version, chunking hash, strategy and retriever name,
retriever config (`k1`, `b`, analyzer) and its hash, the analysed query terms, statistics
(`candidate_chunks`, `avg_chunk_length`, `matched_chunks`, `indexed_now`), the environment
snapshot (rag-forge version, git commit, Python, packages), elapsed time and timestamp. Each hit
carries a `RetrievalResult` (query id, strategy, retriever, chunk, document and document-version
ids, rank, score, `origin = retrieved`).

Responses are not persisted yet; recording retrieval runs belongs to the experiment layer.

## Schema migration

`SqliteStore` applies forward-only migrations by `PRAGMA user_version`. Migration 2 adds the three
lexical tables. Each migration runs in one transaction. Phase 1 databases upgrade in place on
startup and are indexed on first query (tested).

## Known limits

- No stemming, synonyms, phrase or proximity scoring (baseline by design).
- Postings for very common terms are filtered through a membership join per query; fine at
  research scale, and the obvious place to optimise if corpora grow large.
- The first query on a large version pays the indexing cost.
- Scores are only comparable within one query and corpus version.
