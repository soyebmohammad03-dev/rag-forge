# 0004: Dense retrieval

Date: 2026-10-01 · Status: accepted

## Decisions

1. **`BAAI/bge-small-en-v1.5` at a pinned revision.** Strong small retrieval model, MIT, 384-d,
   CPU-fast. The revision, weights SHA-256 and full embedder spec are recorded in provenance.
2. **ONNX Runtime instead of PyTorch/sentence-transformers.** Same outputs (verified to 1.8e-7)
   from the model's official ONNX export, at about 85 MB of dependencies instead of over 1 GB,
   with pinned files and deterministic CPU inference. Pooling and normalisation come from the
   model's sentence-transformers config, so other models in that layout work unchanged.
3. **`Embedder` protocol, configurable by `$RAG_FORGE_EMBEDDER`.** No provider is hard-coded in
   services, API or UI; a hosted or GPU embedder is another implementation.
4. **Vectors in SQLite, keyed by (embedder hash, chunk id).** One store, transactional with the
   metadata, and vectors are reused across corpus versions. Exact NumPy search over the
   version's matrix is the baseline; FAISS/pgvector/ANN is a replacement of one class.
5. **Explicit, recorded index builds with a lifecycle state.** Builds run in the background,
   are idempotent, record failures, and are sealed with a content hash over chunk ids and vector
   bytes. Interrupted builds are marked failed at startup.
6. **No fallback.** A dense request without a ready index fails with 409 and the index state;
   an unavailable model is 503.
7. **Cosine similarity on unit vectors, rounded to 6 decimals, ties by chunk id.**
8. **`"bm25"` accepted as an alias of `"sparse"`** so both spellings work without changing the
   Phase 2 contract.
9. **Comparison is client-side inspection.** BM25 vs dense overlap, rank movement, score
   distributions and latency are computed from two ordinary retrieval responses; no metrics are
   invented before the Arena exists.
