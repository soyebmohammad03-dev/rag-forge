# Research methodology

## Question

Can adaptive retrieval policies select and compose strategies according to query characteristics
and evidence requirements more effectively than fixed pipelines?

"More effectively" is judged on several axes at once. A router that improves nDCG by doubling
latency has not won; it has moved along a trade-off curve. RAG FORGE therefore reports quality,
grounding and cost (latency, tokens, failures) side by side, per case.

## Controlled comparison

For each comparison, hold fixed: corpus version, chunking, embedding model, query set, relevance
judgements, generator and prompt. Vary one thing (retrieval strategy, fusion, reranker, router
policy, generator). The Arena records, for every declared ablation, which arm settings actually
differ and flags pairs where more than one does; it refuses comparisons across dataset content,
corpus versions or metric definitions ([experiment configuration](../experiments/configuration.md)).

Configurations available as presets: BM25, dense, hybrid RRF, hybrid weighted, each with and
without cross-encoder reranking, the adaptive router, and grounded RAG with the extractive baseline
or the local generator. Graph and metadata retrieval are not implemented.

## Metrics

| Family | Metrics | Status |
|---|---|---|
| Retrieval | Recall@K, Precision@K, HitRate@K, MRR, nDCG@K | Implemented |
| Reranking | MRR / nDCG change over the same pool, recall preservation, rank movement | Implemented |
| Evidence and grounding | Evidence recall, grounding score, supported / unsupported claim rate, citation coverage and precision, evidence coverage | Implemented; support by evidence as measured by the verifier `lexical-semantic@1`, not truth |
| Generation | Token F1 against a reference, abstention accuracy | Implemented as automatic proxies |
| Efficiency | Latency per stage, prompt and completion tokens, failure rate | Implemented; no monetary cost model |
| Robustness | Performance under query perturbation and distractor documents | Not implemented |

Definitions, requirements and skip semantics: [metrics](../experiments/metrics.md). nDCG uses the
exponential gain `2^rel − 1` with `log2(rank + 1)` discount. A metric without its annotations is
skipped with a recorded reason, never scored 0.

Answer correctness and faithfulness judgements (human or LLM-as-judge) are not part of RAG FORGE.
Grounding measures whether claims are supported by the passages the generator saw.

## Statistics

`paired-bootstrap-sign@1` ([statistical methodology](../experiments/statistics.md)):

- per-case scores are stored and shown, not only means;
- paired differences over the cases both arms define, seeded percentile bootstrap CIs (95%,
  5000 resamples), effect size `dz`, direction-aware wins/losses/ties;
- exact sign test with Holm adjustment across the metrics of a comparison;
- no conclusion below 10 paired cases; conclusions describe the dataset only.

Not implemented: stratified reporting by query type (tags are stored per case and exported, so it
can be done on the CSV export), and multiple-comparison correction across a whole ablation grid
rather than per comparison.

## Datasets

Present: the bundled development benchmark (12 documents, 14 cases), which validates the pipeline
and is too small and too easy to rank methods, and user datasets registered through
`POST /api/v1/benchmarks` with annotations checked against the corpus version. Suitable public
benchmarks for real comparisons include BEIR subsets (SciFact, FiQA, NFCorpus) for retrieval and
HotpotQA or 2WikiMultiHopQA for multi-hop; RAG FORGE has no importer for them, so they must be
converted to the dataset schema.

## Integrity rules

- No result is shown unless it was produced by a recorded run.
- Every value carries its origin (`retrieved`, `inferred`, `generated`, `measured`); simulated
  values are labelled and never aggregated with measured ones.
- Every number traces to a configuration snapshot, a dataset version and a stored trace, and the
  run can be replayed ([reproducibility](../reproducibility/README.md)).
