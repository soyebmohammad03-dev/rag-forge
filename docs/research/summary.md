# Research summary

RAG FORGE is built to answer questions about retrieval-augmented generation with controlled,
replayable experiments rather than anecdotes. This page lists the questions it can investigate,
how, and what the bundled data does and does not show.

## Questions the platform supports

| Question | How to study it in RAG FORGE |
|---|---|
| How does lexical retrieval compare with dense retrieval? | Arena ablation `bm25 → dense` (single factor: `retrieval.strategy`); per-query view in the Retrieval Lab's four-way comparison |
| When does hybrid retrieval help? | `bm25 → hybrid-rrf`, `hybrid-rrf → hybrid-weighted`; case explorer filtered by tags (lexical, semantic, identifier, …); per-component ranks in the fusion explanation |
| Does cross-encoder reranking improve ranking quality? | `hybrid-rrf → hybrid-rrf-rerank`; reranking metrics over the same candidate pool (MRR/nDCG change, recall preservation, rank movement) |
| Can query characteristics drive retrieval selection? | `hybrid-rrf-rerank → adaptive`; the router's recorded analysis and rule trace per case; Router page comparison with a fixed baseline |
| How does retrieval quality affect evidence grounding? | Grounded RAG arms with different retrieval settings; evidence recall and grounding metrics next to retrieval metrics, case by case |
| What is the latency/quality trade-off? | Every arm reports stage latencies, tokens and failure rate beside quality; paired comparisons include operational metrics |
| How reproducible are RAG experiments under controlled configurations? | Replay: per-stage output hashes compared with the recording; manifests with model file hashes, lockfiles and seeds |

Every answer is a statement about a dataset. Conclusions need at least 10 paired cases, and the
confidence interval of the paired difference must exclude 0.

## Example: the development benchmark (measured)

Run `run_25ad15b603e04574`, recorded with RAG FORGE 1.0.0 on the pre-release working tree
(commit `8a63bd3` plus the uncommitted 1.0.0 changes, recorded as such in its manifest), on an
Apple Silicon laptop (CPU only), 7 configurations × 14 cases = 98 evaluations, 0 failed:

| Configuration | nDCG@10 | MRR | Grounding score | Abstention accuracy | Token F1 | Mean latency |
|---|---|---|---|---|---|---|
| BM25 | 0.972 | 0.962 | — | — | — | 10.0 ms |
| Dense | 0.994 | 1.000 | — | — | — | 33.5 ms |
| Hybrid RRF | 1.000 | 1.000 | — | — | — | 35.7 ms |
| Hybrid RRF + cross-encoder | 1.000 | 1.000 | — | — | — | 248 ms |
| Adaptive router | 1.000 | 1.000 | — | — | — | 223 ms |
| Grounded RAG, extractive baseline | 1.000 | 1.000 | 1.000 (n = 13) | 1.000 | 0.272 | 499 ms |
| Grounded RAG, Qwen2.5-0.5B (local, greedy) | 1.000 | 1.000 | 0.750 (n = 4) | 0.643 | 0.067 | 5089 ms |

Retrieval metrics are over the 13 cases with relevance judgements (the unanswerable case is
skipped). Grounding is defined only for answers with factual claims.

What the paired comparisons support **on this dataset**:

- Retrieval quality: no detectable difference between any of the declared pairs (BM25 vs dense,
  hybrid vs hybrid + reranking, hybrid + reranking vs adaptive). The set is saturated: most arms
  are at the ceiling.
- Latency: dense is slower than BM25 (+23.5 ms, 95% CI [+16.6, +33.0]); reranking adds +212 ms
  ([+201, +223]); the adaptive router was faster than always reranking (−24.7 ms, [−37.2, −14.4])
  because it does not always rerank.
- Generator: the local 0.5B model abstained wrongly more often than the extractive baseline
  (abstention accuracy −0.357, [−0.643, −0.143]), had lower token F1 (−0.205, [−0.333, −0.071])
  and lower evidence coverage (−0.243, [−0.329, −0.157]), and took +4590 ms per case. Its
  grounding-score difference had only 4 pairs: no conclusion.
- Replay: all 98 evaluations reproduced every stage hash exactly on the same machine, generation
  included.

## What this does not show

- That any retrieval method is better in general. Fourteen purpose-written cases on a 12-document
  corpus cannot rank methods; the development benchmark exists to validate the pipeline.
- That the router is better than a fixed pipeline. Its rules are a transparent heuristic baseline,
  not a trained policy, and it was not compared on data where retrieval methods differ.
- Anything about answer quality. Grounding is support by the supplied passages; token F1 and
  abstention accuracy are lexical proxies against short reference spans.
- Reproducibility across machines. Replays were measured on the recording machine only.

Answering the research question needs a judged dataset where methods actually differ (for example
a BEIR subset converted to the dataset schema) and enough cases per query type.
