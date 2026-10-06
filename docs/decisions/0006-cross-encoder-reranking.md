# 0006: Cross-encoder reranking

Date: 2026-10-06 · Status: accepted

## Decisions

1. **Reranking is a stage, not a strategy.** `Reranker` scores (query, passage) pairs and never
   sees the corpus. It runs after BM25, dense or hybrid without any retriever knowing about it,
   and new rerankers register by model id in `create_app()`.
2. **`cross-encoder/ms-marco-MiniLM-L-6-v2`, pinned, local, fp32 ONNX on CPU.** Small enough
   for a laptop and a standard MS MARCO baseline. It uses the same onnxruntime + tokenizers stack as the
   embedder (no PyTorch, no hosted API). The activation comes from the model's own config, and the
   weights' SHA-256 is recorded.
3. **`top_k` stays the final list length.** Reranking adds `rerank.candidate_k`, the upstream
   pool size (≥ `top_k`). A separate `final_top_k` field would have given one quantity two names.
4. **Movement is positional and exact.** `rank_delta = original_rank − final_rank` over the whole
   pool. Ties keep upstream order. `entered_top_k` / `left_top_k` compare both ranks with `top_k`.
   The UI explains movement only through measured swaps and scores.
5. **Upstream information is kept.** Every hit keeps its upstream strategy, matched terms and
   fusion detail next to its original rank and score. The response lists the whole scored pool,
   including chunks that left the top-k.
6. **No silent degradation.** An unregistered model is 501 and an unloadable one is 503, both
   before any result is served. Unreranked requests never load the model.
7. **Backward-compatible hashes.** `RetrievalConfiguration.rerank` is excluded from the hash when
   absent, so unreranked configuration hashes are unchanged. The reranking provenance records the
   upstream configuration hash, which equals that of the standalone unreranked run of the pool.
8. **Reranking analysis runs on one strategy at a time in the UI.** The four-way comparison stays
   unreranked. Comparing reranked variants across strategies belongs in the Arena.
