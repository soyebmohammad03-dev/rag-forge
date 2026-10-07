# RAG Arena: benchmarking and experiments

Code: `apps/api/src/rag_forge/arena/` (`datasets.py`, `metrics.py`, `stats.py`, `config.py`,
`presets.py`, `engine.py`), models in `domain/arena.py`, persistence in `storage/arena.py`
(migration 4 in `storage/sqlite.py`), HTTP in `api/arena.py`. UI: `apps/web/src/features/arena/`
(`/arena`, `/experiments`, `/results`).

```
versioned dataset ─▶ arms ─▶ configuration snapshots (hashed) ─▶ experiment
                                                                    │
                     run: every case × every arm, partial failures kept
                                                                    │
       RunCase (ranking, evidence, answer, grounding, timings, models, artifact, metrics)
                                                                    │
         arm summaries (mean, median, std, bootstrap CI, skipped, failures)
                                                                    │
           leaderboard (descriptive) · paired comparison (per-case differences)
```

The Arena reuses the engine unchanged: retrieval arms call `RetrievalService.retrieve`, RAG arms
call `RagService.answer`, with the same models the API already holds in memory. Nothing in the
Arena reimplements retrieval, generation or grounding, so a number in the Arena is the number
the Retrieval Lab or Evidence Lab would show for the same request.

## Datasets

A `BenchmarkDataset` is an immutable version pinned to one corpus version (and its chunking
hash). Every annotation on a `BenchmarkCase` is optional:

| Field | Meaning | Used by |
|---|---|---|
| `query` | The question | every arm |
| `relevant_documents` | filename → graded relevance (> 0 relevant, 0 judged non-relevant) | retrieval and reranking metrics |
| `relevant_chunks` | chunk id → grade; takes precedence over documents when present | retrieval and reranking metrics |
| `reference_answer` | A short reference span | `answer_token_f1` |
| `answerable` | `false`: the corpus cannot answer it; `null`: not judged | `abstention_accuracy` |
| `expected_evidence` | filenames the selected evidence should include | `evidence_recall` |
| `tags`, `metadata` | Free stratification labels | display, filtering |

Registration (`POST /api/v1/benchmarks`) checks every judged filename and chunk id against the
pinned corpus version and returns the full list of problems (422) instead of the first.
Identical content (same corpus version, same cases) returns the existing version; changed
content under the same name creates version + 1. `content_hash` identifies the content.

Sources: `development` (bundled), `user` (registered through the API), `external` (reserved for
imported public benchmarks; there is no importer yet).

### The development benchmark

`POST /api/v1/benchmarks/development` creates a 12-document corpus written for the purpose
(photosynthesis, food-truck permits, bread, car maintenance, interest rates, BM25, dense
retrieval, RRF, cross-encoders, tomato sauce, error codes, Vesuvius), builds its dense index and
registers 14 cases: lexical, semantic/paraphrase, identifier, numeric, explanatory, a two-document
comparison (no reference answer), and one unanswerable question. Relevance holds by
construction (each document was written to answer its query); reference answers are spans
copied from the relevant document. The annotation notes say so on the dataset itself.

**It validates the pipeline. It does not rank methods.** It is small and easy: in local runs
every retrieval arm reaches nDCG@10 ≥ 0.97, so differences are mostly ceiling effects. Every
comparison on it carries a warning saying exactly this.

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
| evidence | `grounding_score`, `supported_claim_rate`, `unsupported_claim_rate`, `citation_coverage`, `citation_precision`, `evidence_coverage` | generation | As measured by the Phase 7 verifier: support by the supplied evidence, not truth |
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

## Configurations

An `Arm` is one configuration under test: a name, a pipeline (`retrieval` or `rag`), a
`RetrievalTemplate` (a retrieval request without query and corpus version: strategy, top-k, BM25,
hybrid, rerank, manual or adaptive mode, router), and for RAG arms evidence, generation and
grounding parameters.

At experiment creation each arm is resolved into an immutable `ConfigurationSnapshot`: dataset
id, version and hash; corpus version and chunking hash; the resolved retrieval configuration
(manual arms) or analyzer and router policy versions plus routing hash (adaptive arms); embedder
and reranker specs with revisions; evidence parameters; prompt template; generator identity
(name, provider, model, revision, config hash) and effective generation parameters; verifier
and its config hash; metric settings and metric versions; engine version (`arena-engine@1`).
Components that are not registered fail creation with 501 rather than at run time.

`config_hash` is the SHA-256 of the canonical snapshot excluding the arm's name, so two arms
with different names and identical behaviour have the same hash. Every `RunCase` stores it.

`GET /api/v1/arena/presets` lists templates (BM25, Dense, Hybrid RRF, Hybrid weighted, Adaptive,
three reranked arms, grounded RAG with the extractive baseline or the local model) and suggested
ablations. They are templates: nothing runs by default.

## Ablations

An `AblationSpec` names a baseline arm, a variant arm and the factor the researcher intends to
vary. The engine records:

- `factors`: the arm settings that actually differ (`retrieval.strategy`, `retrieval.rerank`,
  `generation`, `pipeline`, …);
- `single_factor`: whether exactly one differs;
- `changes`: every resolved snapshot field that differs, with both values.

A multi-factor ablation is kept and flagged as confounded, never silently relabelled. Identical
arms are rejected ("nothing varies").

## Runs

`POST /api/v1/experiments/{id}/runs` queues a run (202) and executes it in the background:

- the first `limits.max_cases` cases (default: all), every arm, `limits.concurrency` (1–4) cases
  in parallel, the loaded models reused throughout;
- one `RunCase` per case × arm: status, ranking with relevance grades and upstream ranks, route
  (adaptive), selected evidence, answer status and text, grounding counts, timings, tokens, model
  identities, a full trace artifact (content-addressed blob) and its metrics;
- a case that raises is stored as `failed` with its error type and message; the run continues;
- progress is persisted after each case; final status is `completed` (no failures), `partial`
  (some failures, all recorded) or `failed` (the run itself could not proceed).

Each run records dataset hash, corpus version, case ids, arms, snapshot hashes, limits, the
metric registry version, the statistics method and an `EnvironmentSnapshot`.

## Statistics (`paired-bootstrap-sign@1`)

Per arm and metric: n, skipped, mean, median, sample std, min, max and a 95% percentile bootstrap
CI of the mean (5000 resamples, seed 20261007, so recomputing gives identical intervals).

Paired comparison of two arms (`/runs/{id}/compare`, or `/arena/compare` across runs) uses only
cases where both arms define the metric:

- per-case differences (variant − baseline), mean, median, std, bootstrap CI of the mean
  difference, effect size `dz` = mean / sd of differences;
- wins / losses / ties oriented by the metric's direction;
- exact two-sided sign test (ties dropped), Holm-adjusted across the metrics of the comparison;
- a conclusion: `insufficient_cases` below 10 pairs, `descriptive_only` for directionless
  metrics, otherwise `variant_higher` / `variant_lower` when the CI excludes 0, else
  `no_detectable_difference`.

"Higher" describes the value; the UI resolves whether that is better from the metric's direction.
A conclusion is a statement about this dataset, never a general claim.

Incomparable arms are refused (`comparable: false`, with reasons) when a run is unfinished, or
the dataset content, corpus version, metric versions or k values differ, or no case was
evaluated by both. Warnings are added for differing case sets, failed cases, differing top-k or
evidence budgets (budget confounds), identical configurations, and the development dataset.

The leaderboard (`/runs/{id}/leaderboard?metric=`) orders arms by mean, gives tied means the same
position, gives no position to descriptive or undefined metrics, and marks arms whose CI overlaps
the leader's. It is descriptive; decisions use paired comparisons.

## Provenance: what produced this number?

Every aggregate is a mean over `RunCase.metrics`. Each `RunCase` names its run, arm, case,
configuration snapshot hash and retrieval configuration hash; `GET /api/v1/artifacts/{id}`
returns the full `RagResponse` or `RetrievalResponse` it came from. The snapshot pins the dataset
version, corpus version, models and parameters; the run pins the metric registry, statistics
method and environment.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/arena/overview` | Datasets, experiment count, recent runs, registry and statistics method |
| GET | `/api/v1/arena/metrics` | Metric definitions |
| GET | `/api/v1/arena/presets` | Arm templates, suggested ablations, default metric settings |
| GET, POST | `/api/v1/benchmarks` | List / register datasets |
| POST | `/api/v1/benchmarks/development` | Install the development benchmark (idempotent) |
| GET | `/api/v1/benchmarks/{id}` | A dataset version with its cases |
| POST | `/api/v1/arena/configurations/resolve` | Snapshots, hashes and diffs for a candidate matrix, without creating anything |
| GET, POST | `/api/v1/experiments` | List / create experiments |
| GET | `/api/v1/experiments/{id}` | Experiment with snapshots and ablations |
| GET, POST | `/api/v1/experiments/{id}/runs` | List runs / queue a run |
| GET | `/api/v1/runs`, `/api/v1/runs/{id}` | Run digests / one run with summaries |
| GET | `/api/v1/runs/{id}/cases[?arm=&status=]` | Per-case results, failures included |
| GET | `/api/v1/runs/{id}/cases/{case_id}` | One case across every arm |
| GET | `/api/v1/runs/{id}/leaderboard?metric=` | Descriptive ordering |
| GET | `/api/v1/runs/{id}/compare?baseline=&variant=` | Paired comparison within a run |
| GET | `/api/v1/arena/compare` | Paired comparison across runs |
| GET | `/api/v1/artifacts/{id}` | The full trace behind one case of one arm |

The retrieval, router and answer endpoints are unchanged. The earlier placeholder
`GET/POST /api/v1/experiments` and `GET /api/v1/runs` (storage-only records around a
`RAGConfiguration`, never executed) are replaced by the routes above, and the placeholder
`RAGConfiguration`, `Experiment`, `ExperimentRun`, `EvaluationResult` and `Artifact` domain
models by their Arena counterparts.

## UI

`/arena` (also `/experiments` opening the builder and `/results` opening results):

- **Overview**: method, recent runs, registry and statistics method.
- **Datasets**: versions, sources, annotation coverage, cases with their ground truth.
- **Experiment builder**: dataset, arms (with top-k), suggested and custom ablations, ks, case
  limit, concurrency; preview of snapshots, hashes and differing factors before creating.
- **Run monitor**: live progress and failures.
- **Results**: metric cards, leaderboard with interval plot, statistical comparison (changes,
  per-metric paired statistics, per-case difference plot), case explorer (case × arm grid, then
  annotations, rankings with relevance and rank movement, evidence, answers, grounding, metrics,
  skips, timings and the trace link for two arms side by side), failures and skip reasons,
  latency distributions (p50/p95 by stage), configuration snapshots and ablation factors.

Every chart draws recorded values only; empty states say nothing has been measured.

## Limits and what is not here

- Runs execute in-process (FastAPI background tasks, a small thread pool). A restart interrupts a
  running run; it stays `running` and is not resumed.
- Latency includes first-use model loading on the first case and is machine-specific.
- The development benchmark is too small and too easy to rank methods; there is no importer for
  public benchmarks yet (`external` is reserved).
- No LLM-as-judge, no human evaluation interface, no cost model.
- **Phase 9** handles replay of past runs from their provenance, final reproducibility tooling,
  report and publication export, and final presentation. None of that is implemented here.
