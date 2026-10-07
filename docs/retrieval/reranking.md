# Cross-encoder reranking

Code: `apps/api/src/rag_forge/retrieval/rerank.py`, applied by `RetrievalService` after any strategy. Part of [retrieval](README.md).

## Reranking

A second stage after any strategy:

```
query → BM25 | dense | hybrid → candidate pool (rerank.candidate_k)
      → cross-encoder scores each (query, chunk) pair → reorder pool → final top_k
```

Code: `retrieval/rerank.py` (contract, `OnnxCrossEncoder`, `rank_movement`) and
`RetrievalService._rerank`. A `Reranker` has a `name`, a `spec`, `info()` and
`score(query, passages) -> list[float]`; it sees passages, not the corpus, so retrievers know
nothing about it. Rerankers are registered in `create_app()` by model id, like strategies.

### Model

`cross-encoder/ms-marco-MiniLM-L-6-v2` at revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`
(`RerankerSpec`), a 6-layer MiniLM trained on MS MARCO passage ranking: ~91 MB fp32 ONNX export
(`onnx/model.onnx`), 512-token pair limit. It runs on onnxruntime's CPU provider (Apple Silicon
and x86, no GPU), with the same tokenizers dependency as the embedder. Files are fetched once into
the Hugging Face cache at the pinned revision and loaded lazily, once per process, on the first
reranked request. `$RAG_FORGE_RERANKER` (a `RerankerSpec` as JSON) selects another cross-encoder
in the same layout (single-logit sequence classifier with an ONNX export).

The score is the model's single logit with the activation from its own config
(`sbert_ce_default_activation_function`: identity for this model, so scores are unbounded
logits, typically −11…+10). Pairs are truncated `longest_first` at 512 tokens, the
sentence-transformers default. Scores are rounded to 6 decimals.

### Parameters and validation

`rerank`: `enabled` (default `false`), `model` (a registered reranker; default the one above),
`candidate_k` (1–200, the upstream pool size, must be ≥ `top_k`). `top_k` is the **final** top-k
in every request; when reranking, the upstream retriever is asked for `candidate_k` results
instead. For hybrid, `hybrid.candidate_k` (per component) must be ≥ `rerank.candidate_k`.
Invalid combinations return 422.

### Rank movement

Each scored candidate gets a `RerankDetail`:

| Field | Meaning |
|---|---|
| `original_rank`, `original_score` | position and score in the upstream ranking (1-based) |
| `reranker_score`, `final_rank` | cross-encoder score; position after sorting the whole pool |
| `rank_delta` | `original_rank − final_rank`; positive = moved up |
| `movement` | `promoted` (Δ > 0), `demoted` (Δ < 0), `unchanged` (Δ = 0) |
| `entered_top_k` | `final_rank ≤ top_k < original_rank` |
| `left_top_k` | `original_rank ≤ top_k < final_rank` |

The pool is sorted by reranker score, best first; equal scores keep their upstream order, so the
ranking is deterministic and a reranker that scores everything equally changes nothing. Hits are
the final top-k: `result.rank` and `result.score` are the final rank and the reranker score, while
`result.strategy`, `matched_terms` and `fusion` still describe the upstream retrieval. The
response's `reranking.candidates` lists the whole pool by final rank, including chunks that left
the top-k.

For any candidate, `rank_delta` equals the number of candidates it overtook minus the number that
overtook it. The Retrieval Lab explains movement in exactly those terms (the swaps and both
scores), never with generated prose.

### Provenance

`provenance.reranking` holds the reranker name, `RerankerInfo` (spec with model and revision,
scoring and activation, effective max length, truncation, SHA-256 of the ONNX weights),
`reranker_config_hash`, `candidate_k`, `final_top_k`, `candidates_scored` (below `candidate_k`
when fewer chunks match), scoring latency (excludes the one-time model load), movement counts,
and the **upstream configuration**: the unreranked retrieval that produced the pool, whose hash is
exactly what that retrieval returns on its own with `top_k = candidate_k`.
`configuration.rerank` (model spec + `candidate_k`) makes reranked configurations hash
differently from unreranked ones.

### Failure behaviour

- Unregistered `rerank.model`: 501 (`capability: "Reranker"`), checked before retrieval runs.
- Model cannot be fetched or loaded, or returns a wrong number of or non-finite scores: 503 with
  `component: "reranker"`. Unreranked results are never returned in place of reranked ones.
- Unreranked requests never touch the reranker; the health endpoint reports the configured model
  without loading it.
