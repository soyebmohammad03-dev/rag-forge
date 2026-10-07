# 0009: Arena, benchmarking and experiments

Date: 2026-10-07 · Status: accepted

## Decisions

1. **Datasets are versioned and pinned to a corpus version.** Annotations are validated against
   that version at registration; identical content reuses the version, changed content creates the
   next one. Sources distinguish the bundled development set from user and (future) external data.
2. **Annotations are optional; missing ground truth is skipped.** Each metric declares the
   annotations and pipeline outputs it needs. Without them it reports `null` with a reason, never
   0, and aggregates count defined and skipped cases separately.
3. **The development benchmark is honest about its role.** Twelve purpose-written documents and
   fourteen cases with relevance by construction. It exercises every metric family, including an
   unanswerable case. It saturates (retrieval arms reach nDCG@10 ≥ 0.97), so every comparison on
   it is labelled as pipeline validation, not evidence about methods.
4. **One metric registry, versioned.** Retrieval, reranking, evidence/grounding, automatic
   generation proxies and operational metrics share one definition format (family, version,
   requirements, k, unit, aggregation, direction). Grounding metrics are support by evidence, not
   truth; no metric judges answer quality, and there is no LLM-as-judge.
5. **Configurations are immutable snapshots.** Each arm is resolved at experiment creation into
   every component identity and parameter that affects results; its hash excludes the arm's name.
   Unregistered components fail creation, not the run.
6. **Ablations state what actually changed.** The engine diffs the two arms and the two
   snapshots. Multi-factor pairs are kept and flagged, identical pairs rejected.
7. **The engine is reused, not reimplemented.** Arms call the existing retrieval and RAG services
   with the models already loaded, so Arena numbers equal the labs' numbers for the same request.
8. **Failures are results.** A failing case is stored with its error and counted in the failure
   rate; the run continues and ends `partial`. A run that cannot proceed ends `failed`.
9. **Paired statistics, conservative conclusions.** Seeded percentile bootstrap CIs, per-case
   differences, effect size, direction-aware wins/losses, exact sign test with Holm adjustment.
   No conclusion below 10 pairs; a conclusion is about the dataset only. The method is versioned
   (`paired-bootstrap-sign@1`) and recorded on every run.
10. **Incomparable runs are refused, not compared.** Differing dataset content, corpus versions,
    metric versions or k values make a comparison `comparable: false` with reasons. Budget
    differences and partial failures are warnings.
11. **Laptop-scale execution.** In-process background runs, a bounded thread pool (1–4), case
    limits, progress saved per case. A queue and resumable runs are deferred.
12. **Scope.** Replay, final reproducibility tooling, report export and presentation are Phase 9.
