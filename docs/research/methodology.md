# Research methodology (planned)

## Question

Can adaptive retrieval policies select and compose strategies according to query characteristics
and evidence requirements more effectively than fixed pipelines?

"More effectively" is judged on several axes at once. A router that improves nDCG by doubling
latency and cost has not won; it has moved along a trade-off curve.

## Controlled comparison

For each comparison, hold fixed: corpus version, chunking, embedding model, query set, relevance
judgements, generator and prompt. Vary one thing (retrieval strategy, reranker, or router policy).
The Arena enforces this by comparing runs whose configuration hashes differ only in the
declared variable.

Systems compared: naive (dense top-k), sparse (BM25), dense, hybrid (fusion), hybrid + rerank,
graph-aware, and routed (router policy selecting among the above).

## Metrics

| Family | Metrics | Status |
|---|---|---|
| Retrieval | Recall@K, Precision@K, MRR, nDCG@K | Implemented (`evaluation/retrieval_metrics.py`) |
| Context | Context precision, context recall | Planned |
| Generation | Answer correctness, faithfulness, citation accuracy, hallucination rate | Planned |
| Efficiency | Latency (p50/p95), tokens, compute/cost proxies | Planned |
| Robustness | Performance under query perturbation and distractor documents | Planned |

Metric definitions follow standard IR usage. nDCG uses the exponential gain `2^rel − 1` with
`log2(rank + 1)` discount. Metrics raise on undefined inputs (e.g. no relevant documents)
instead of returning a silent 0.

## Statistics

- Report per-query scores, not only means.
- Paired tests across the same queries (paired bootstrap or permutation test); report
  confidence intervals and effect sizes.
- Stratify by query type (single-hop, multi-hop, comparison, temporal, lexical-heavy), because
  that stratification is where routing is supposed to help.
- Correct for multiple comparisons when running ablation grids.

## Datasets (candidates)

BEIR subsets (SciFact, FiQA, NFCorpus) for retrieval; HotpotQA and 2WikiMultiHopQA for
multi-hop; a small hand-judged internal set for faithfulness and citation checks.

## Integrity rules

- No result is shown unless it was produced by a recorded run.
- Simulated or estimated values carry `ContentOrigin.SIMULATED` and are never aggregated with
  measured ones.
- Any LLM-as-judge evaluator records its model, version and prompt in the run's provenance.
