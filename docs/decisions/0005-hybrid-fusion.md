# 0005: Hybrid retrieval and fusion

Date: 2026-10-01 · Status: accepted

## Decisions

1. **Fusion is its own abstraction.** `FusionStrategy` sees only ranked lists per strategy, never
   retrievers. `HybridRetriever` is a `Retriever` that composes components and a fusion method, so
   the service, API shape and UI treat hybrid like any other strategy.
2. **RRF (k = 60) is the primary hybrid baseline.** Rank-based, parameter-light, and free of any
   assumption about how BM25 and cosine scores relate.
3. **Weighted fusion uses explicit min-max normalisation per list.** Raw scores are never summed
   across strategies. Weights are validated (each 0..1, sum 1).
4. **Nothing is overwritten.** Every fused hit keeps each component's original rank and raw score
   next to its normalised score and contribution; the fused score is reported separately.
5. **Deterministic ordering.** Fused scores are rounded to 12 decimals before sorting and ties go
   to chunk id, so summation order cannot break genuine ties.
6. **No partial hybrid.** If a component cannot run (no ready dense index), the request fails with
   the component and its state. Components that searched different chunk sets are refused, and
   the service rejects any chunk outside the requested version for every strategy.
7. **`RetrievalConfiguration` in every response.** The resolved, hashable description of a result
   set, with parameters the strategy ignores removed from the hash. It is the unit the Arena will
   vary.
8. **Hybrid is fixed, not adaptive.** Choosing a configuration per query is the Router's job and
   is deferred.
9. **Comparison stays inspection.** The Retrieval Lab shows overlap, rank movement, fusion effects
   (promoted/dropped), per-run score scales and latency, and names no winner: there is no
   ground-truth dataset yet.
