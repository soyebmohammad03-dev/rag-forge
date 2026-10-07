# Research methodology

## Question

Can adaptive retrieval policies select and compose strategies according to query characteristics
and evidence requirements more effectively than fixed pipelines?

"More effectively" is judged on several axes at once. A router that improves nDCG by doubling
latency and cost has not won; it has moved along a trade-off curve.

## Controlled comparison

For each comparison, hold fixed: corpus version, chunking, embedding model, query set, relevance
judgements, generator and prompt. Vary one thing (retrieval strategy, reranker, or router policy).
The Arena records, for every declared ablation, which arm settings actually differ and flags
pairs where more than one does; it refuses comparisons across dataset content, corpus versions
or metric definitions (see [../arena.md](../arena.md)).

Systems compared: naive (dense top-k), sparse (BM25), dense, hybrid (fusion), hybrid + rerank,
graph-aware, and routed (router policy selecting among the above).

## Metrics

| Family | Metrics | Status |
|---|---|---|
| Retrieval | Recall@K, Precision@K, HitRate@K, MRR, nDCG@K | Implemented (`evaluation/retrieval_metrics.py`, Arena registry) |
| Reranking | MRR / nDCG change over the same pool, recall preservation, rank movement | Implemented (Arena) |
| Evidence and grounding | Evidence recall, grounding score, supported / unsupported claim rate, citation coverage and precision, evidence coverage | Implemented (Arena; support by evidence as measured by the Phase 7 verifier, not truth) |
| Generation | Token F1 against a reference, abstention accuracy | Implemented as automatic proxies; answer correctness and faithfulness judgements are not |
| Efficiency | Latency per stage, prompt and completion tokens, failure rate | Implemented (Arena); cost proxies planned |
| Robustness | Performance under query perturbation and distractor documents | Planned |

Metric definitions follow standard IR usage. nDCG uses the exponential gain `2^rel − 1` with
`log2(rank + 1)` discount. The evaluation endpoint raises on undefined inputs; in the Arena a
metric without its annotations is skipped with a recorded reason, never scored 0.

## Statistics

Implemented in the Arena as `paired-bootstrap-sign@1`:

- per-case scores are stored and shown, not only means;
- paired differences over the cases both arms define, seeded percentile bootstrap CIs (95%,
  5000 resamples), effect size `dz`, direction-aware wins/losses/ties;
- exact sign test with Holm adjustment across the metrics of a comparison;
- no conclusion below 10 paired cases; conclusions describe the dataset only.

Still to do: stratified reporting by query type (single-hop, multi-hop, comparison, temporal,
lexical-heavy), which is where routing is supposed to help, and corrections across whole
ablation grids rather than per comparison.

## Datasets (candidates)

Present: the bundled development benchmark (12 documents, 14 cases), which validates the
pipeline and is too small and too easy to rank methods, and user datasets registered through
the API. Candidates for real comparisons: BEIR subsets (SciFact, FiQA, NFCorpus) for retrieval;
HotpotQA and 2WikiMultiHopQA for multi-hop; a hand-judged set for faithfulness and citation
checks. No importer exists yet.

## Integrity rules

- No result is shown unless it was produced by a recorded run.
- Simulated or estimated values carry `ContentOrigin.SIMULATED` and are never aggregated with
  measured ones.
- Any LLM-as-judge evaluator records its model, version and prompt in the run's provenance.
