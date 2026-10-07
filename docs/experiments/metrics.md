# Metrics

Code: `apps/api/src/rag_forge/arena/metrics.py` (registry `arena-metrics@1`). Part of the [Arena](README.md).

## Metrics

`GET /api/v1/arena/metrics` returns each `MetricDefinition`: family, version, description,
required annotations, required pipeline outputs, whether it is reported per k, unit, aggregation
and direction (`higher_is_better`, `null` for descriptive metrics). Registry: `arena-metrics@1`.

| Family | Metric | Needs | Notes |
|---|---|---|---|
| retrieval | `recall@k`, `precision@k`, `hit_rate@k`, `mrr`, `ndcg@k` | relevance | Document level unless chunks are judged; chunk rankings collapse to documents at their first rank. nDCG uses gain `2^grade − 1`, discount `log2(rank + 1)` |
| reranking | `rerank_mrr_delta`, `rerank_ndcg_delta@k` | relevance, reranking | Same candidate pool before and after the cross-encoder |
| reranking | `recall_preservation@k` | relevance, reranking | Share of relevant pool items the reranker keeps in the final top k |
| reranking | `rerank_promoted_share`, `rerank_mean_shift` | reranking | Rank movement; descriptive |
| evidence | `evidence_recall` | expected evidence | Share of expected documents kept by evidence selection |
| evidence | `grounding_score`, `supported_claim_rate`, `unsupported_claim_rate`, `citation_coverage`, `citation_precision`, `evidence_coverage` | generation | As measured by the grounding verifier (`lexical-semantic@1`): support by the supplied evidence, not truth |
| generation | `answer_token_f1` | reference answer | SQuAD-normalised token F1: a lexical proxy |
| generation | `abstention_accuracy` | answerable | Answered an answerable case / declined an unanswerable one |
| operational | `latency_ms`, `retrieval_ms`, `generation_ms`, `grounding_ms` | — | Wall time on this machine; generation and grounding exclude model load |
| operational | `prompt_tokens`, `completion_tokens` | generation | Descriptive |
| operational | `failed` | — | 1 if the case failed; its mean is the failure rate |

**Missing ground truth is skipped, never zero.** A metric whose annotations or outputs are
absent returns `value: null` with a `skipped` reason ("no relevance judgements for this case",
"retrieval-only arm", "the arm does not rerank", …). Aggregates report `n` (defined) and
`skipped` separately, and each arm summary keeps one reason per skipped metric. A failed case
carries only `failed = 1`; it is not scored 0 on quality metrics.

**Measured versus subjective quality.** No metric here judges whether an answer is good.
Grounding metrics measure support by the evidence the generator saw; `answer_token_f1` and
`abstention_accuracy` are automatic proxies against annotations. There is no LLM-as-judge.
