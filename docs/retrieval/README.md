# Retrieval

Code: `apps/api/src/rag_forge/retrieval/`, `apps/api/src/rag_forge/storage/lexical_index.py` and
`apps/api/src/rag_forge/storage/vector_index.py`.

Three strategies run behind one contract: **`sparse`** (BM25, also accepted as `"bm25"`),
**`dense`** (embedding similarity) and **`hybrid`** (fusion of the two). All return the same
`RetrievalResponse` shape; hybrid hits add a `fusion` block, which is `null` for the others.
Any of them can be followed by an optional **cross-encoder reranking** stage (see
[Reranking](#reranking)), which adds a `rerank` block per hit and a `reranking` report.

## Flow

```
POST /api/v1/corpora/{id}/retrieve  { query, top_k, version?, strategy, bm25, hybrid, rerank }
        │
RetrievalService
  1. resolve corpus (404) and version (default current; > current → 404)
  2. pick the retriever registered for `strategy` (unregistered → 501; dense without a ready
     index → 409 with the index state; embedding model unavailable → 503)
  3. if rerank.enabled, resolve the reranker for rerank.model (unregistered → 501)
  4. retriever.retrieve(corpus, version, query, top_k, or rerank.candidate_k when reranking)
     → ranked chunk ids + scores + statistics
  5. if reranking: score every (query, chunk) pair in batches, reorder the pool, keep top_k
     (model unavailable → 503 with component "reranker"; never an unreranked fallback)
  6. load chunks and document versions; build hits (rank, score, chunk, filename, doc version)
  7. attach provenance (corpus version, chunking hash, retriever config + hash, query terms,
     statistics, environment, elapsed time; reranking provenance when reranked)
        │
RetrievalResponse { query, hits[], provenance, warnings[], reranking? }
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

## Dense retrieval

### Model

`BAAI/bge-small-en-v1.5`, pinned to revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (MIT).
384 dimensions, CLS pooling, L2-normalised, 512-token limit, and the model card's query
instruction (`"Represent this sentence for searching relevant passages: "`) prepended to queries
only. Chosen because it is among the strongest small retrieval models on BEIR/MTEB, runs on a
laptop CPU in milliseconds, and ships an official ONNX export.

### Embedding runtime

`OnnxSentenceEmbedder` (`retrieval/embedding.py`) runs any Hugging Face model in the
sentence-transformers layout that includes `onnx/model.onnx`, using `onnxruntime` + `tokenizers`
(about 85 MB installed, versus well over 1 GB for PyTorch + sentence-transformers). Pooling and
normalisation are read from the model's `1_Pooling/config.json` and `modules.json`. Files are
downloaded at the pinned revision into the Hugging Face cache (`~/.cache/huggingface`), never into
the repository. Parity with the `sentence-transformers` package was checked on this model: maximum
absolute difference 1.8e-7 (float32 precision).

The provider is behind an `Embedder` protocol (`spec`, `info()`, `embed_documents`,
`embed_query`). `$RAG_FORGE_EMBEDDER` (an `EmbedderSpec` as JSON) selects another model without
code changes. `EmbedderSpec` (provider, model, revision, query prefix, max length, batch size)
is hashed into `embedder_hash`, which keys every stored vector.

### Vector storage

```
dense_vectors  (embedder_hash, chunk_id → chunks.id, vector BLOB float32-LE)  PK (embedder_hash, chunk_id)
dense_indexes  (id, corpus_id, version, embedder_hash, status, started_at, data JSON)
```

- Vectors are keyed by embedder config and chunk, like the lexical index: each immutable chunk is
  embedded once per model config and shared by every corpus version that contains it. Building
  v2 after editing one document re-embeds only that document's chunks (`reused` counts the rest).
- An index record describes one build of one **(corpus, version, embedder)**: chunk count,
  progress, reused vectors, embedder info (model, revision, dimension, pooling, weights SHA-256),
  similarity, timestamps, error, and a `content_hash` = SHA-256 over the embedder hash, the chunk
  ids and the exact vector bytes. Two indexes with equal content hashes search identical data.
- Search loads the matrix of exactly that version's chunks (membership join, chunk-id order),
  checks every vector exists and has the right dimension, and computes cosine similarity as a dot
  product of unit vectors in float64. Ready indexes are immutable, so matrices are cached by index.
- The storage class is the replacement point for FAISS, pgvector or an on-disk ANN index.

### Similarity and ordering

Cosine similarity of L2-normalised vectors (range −1…1, in practice 0.3…0.9 for this model).
Scores are rounded to 6 decimals before ranking and ties are broken by chunk id, so last-bit
floating-point noise cannot reorder results. Embeddings are bit-identical across calls and
independent of batch composition (tested).

### Index lifecycle

| State | Meaning | Dense queries |
|---|---|---|
| `missing` | No build for this version and embedder, nothing else indexed | 409 |
| `stale` | Not built for this version, but another version or embedder config is | 409 |
| `building` | Build in progress; `embedded / chunk_count` shows progress | 409 |
| `failed` | Latest build failed (error recorded), or the API stopped mid-build | 409 |
| `ready` | Built and sealed with a content hash | 200 |

`POST /api/v1/corpora/{id}/dense-index {version?}` starts a build in the background (202) from the
chunks already stored; nothing is re-ingested. It is idempotent: a ready or building index is
returned as-is (200). A failed or stale state is rebuilt by posting again. On startup, builds left
`building` by a previous process are marked failed. `GET` on the same path reports the state.
Dense retrieval never falls back to BM25.

Dense indexes are built explicitly, unlike the lexical index, because embedding costs real
compute and the UI must not suggest dense retrieval is available before it is.

### Provenance

Dense responses record `index_id`, and `retriever_config` holds model, revision, provider,
dimension, pooling, normalisation, max length, query prefix, weights SHA-256, embedder hash and
similarity metric; statistics include `candidate_chunks` and `query_embedding_ms`. With the same
corpus version, embedder spec and index content hash, ordering is reproducible.

## BM25 vs dense

| | BM25 | Dense |
|---|---|---|
| Matches | exact analysed terms | meaning, paraphrase, no shared words needed |
| Fails on | synonyms, paraphrase, vocabulary mismatch | rare identifiers, exact codes, out-of-domain jargon |
| Scores | unbounded, query-relative | cosine, bounded, compressed range |
| Index | lazy, per chunk, free | explicit build, model compute |
| Explainability | per-term contributions | opaque vector geometry |

The Retrieval Lab's **Compare** mode runs both on the same query, corpus and version and shows
shared and unique chunks, rank movement, score distributions (on their own scales) and latency.
It is inspection only; measured quality comparisons belong to the Arena.

## Hybrid retrieval

Code: `retrieval/fusion.py`, `retrieval/hybrid.py`.

```
query ──▶ HybridRetriever
            ├─ BM25 retriever  ── top candidate_k ─┐   (same corpus version, same chunking)
            └─ Dense retriever ── top candidate_k ─┤
                                                   ▼
               check: both searched the same chunk set (else 409 "mismatch")
                                                   ▼
               FusionStrategy.fuse(lists, top_k) ──▶ fused candidates + per-component detail
```

Hybrid is a **fixed strategy**: the request names the components and the fusion method. It does
not look at the query to decide anything; adaptive routing is a later phase.

### Fusion contract

`FusionStrategy` (`method`, `config()`, `fuse(lists, top_k)`) receives `{strategy: ranked
candidates}` and returns fused candidates. Each carries a `FusionDetail`: the method, the final
score, and one `ComponentScore` per component with its **original rank, original raw score,
normalised score (weighted only) and contribution**. Raw component scores are never overwritten;
the fused score is `result.score` and `fusion.score`. Duplicate chunks are merged (a chunk
appears once, with one entry per component). Ordering is by fused score rounded to 12 decimals,
then chunk id, so floating-point summation order cannot decide a genuine tie. Adding a method
(e.g. CombSUM, Borda, learned fusion) is one class and one branch in `make_fusion`.

### Reciprocal Rank Fusion (primary baseline)

`RRF(d) = Σ_lists 1 / (k + rank_list(d))`, default `k = 60` (Cormack et al., 2009), configurable
1–1000. A chunk missing from a list contributes 0 there.

Why RRF is the baseline: it uses **ranks only**, so it needs no assumptions about how BM25 scores
(unbounded, query-dependent) relate to cosine similarities (bounded, compressed). It has one
parameter, is robust across collections, and is the standard hybrid baseline in IR. `k` controls
how much the top ranks dominate: small `k` rewards being first in one list; large `k` rewards
appearing in several lists.

### Weighted fusion and normalisation

`score(d) = Σ_lists w_list · minmax_list(d)`, where `minmax` rescales one list's raw scores to
0..1 over the retrieved candidates (`(s − min) / (max − min)`; a constant list maps to 1). A chunk
missing from a list contributes 0. Weights are one per component, each in 0..1, summing to 1
(validated; 422 otherwise).

Raw BM25 and cosine scores are **never** combined directly: a BM25 score of 5 and a cosine of 0.7
are not on a common scale, so a raw sum would be dominated by BM25 regardless of intent. Min-max
is simple, deterministic and scale-free, but relative: it depends on the candidate set (and so on
`candidate_k`) and is sensitive to outliers. That is why RRF stays the primary baseline.

### Parameters and validation

`hybrid`: `retrievers` (≥ 2 distinct of `sparse`, `dense`), `fusion` (`rrf` | `weighted`),
`rrf_k`, `weights`, `candidate_k` (1–200, must be ≥ `top_k`). `top_k` is the final list length.
Invalid combinations return 422 with the reason.

### Compatibility and failure behaviour

- Every component searches the requested corpus version under the corpus's chunking config.
  The hybrid retriever refuses to fuse unless all components report the same number of searched
  chunks (409, `state: "mismatch"`).
- The service checks every returned chunk belongs to the requested version and chunking before
  serving it (for all strategies); anything else is a 500, never a silent result.
- No ready dense index: 409 with `component: "dense"` and the index state. Hybrid never degrades
  to BM25-only.
- The BM25 index cannot be missing: it is built lazily per chunk on first use
  (`sparse.indexed_now` reports it).

### Provenance

`retriever_config` holds the fusion config (method, `k` or weights and normalisation),
`candidate_k` and each component's full config; `statistics` are prefixed by component
(`sparse.matched_chunks`, `dense.query_embedding_ms`, …) plus `fused_candidates`; `index_id` is
the dense index used; `query_terms` come from BM25.

## Reranking

A local cross-encoder can rescore the candidates of any strategy: see [reranking.md](reranking.md).

## Retrieval configuration

`RetrievalConfiguration` (in every response's provenance, with `configuration_hash`) is the
complete, resolved description of a ranked result set: corpus id, version and chunking hash,
strategy, `top_k`, BM25 parameters (if BM25 participates), embedder spec (if dense participates)
hybrid parameters (if hybrid) and the reranking step (if reranked; omitted from the hash otherwise,
so configuration hashes from before reranking existed are unchanged). Parameters a strategy ignores are omitted or blanked
(`HybridParams.effective()`), so two behaviourally identical runs share a hash. This is the unit
the Arena will vary and compare.

## Adaptive routing

With `mode: "adaptive"` the router chooses the strategy, fusion and reranking for each query and
hands the service an ordinary manual request; retrievers, fusion and reranking are unchanged. See
[router.md](../routing/README.md).

## Known limits

- No stemming, synonyms, phrase or proximity scoring (baseline by design).
- Postings for very common terms are filtered through a membership join per query; fine at
  research scale, and the obvious place to optimise if corpora grow large.
- The first query on a large version pays the indexing cost.
- Scores are only comparable within one query and corpus version; fused scores are not comparable
  with component scores.
- Hybrid runs its components sequentially; fusion is over each component's top `candidate_k`, so
  a chunk outside both candidate lists cannot be retrieved by hybrid.
- Dense search is exact and loads a version's whole matrix (fine to ~10⁶ chunks); dense indexes
  build one at a time per process, in the API process.
- Long chunks are truncated at the model's 512-token limit when embedded.
- Reranking cost grows with `candidate_k` and pair length (batches pad to their longest pair).
  Measured on an Apple Silicon CPU in fp32: ≈ 0.9 s for 30 pairs of ~170–340 tokens, ≈ 45 ms for
  30 short pairs. The first reranked request also loads the model. Quantised exports are not
  used: they would change scores.
- The cross-encoder sees at most 512 tokens per (query, chunk) pair; longer chunks are truncated.
  It is an English MS MARCO model, and reranking cannot recover chunks outside the upstream pool.
- Reranking runs in the request thread; requests are not batched across users.
- Vectors are computed on CPU; results are reproducible on one machine, and ordering is protected
  against tiny cross-machine float differences by rounding, but not guaranteed bit-identical
  across CPU architectures.
