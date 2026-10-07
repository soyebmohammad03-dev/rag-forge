# Query intelligence and the adaptive router

Code: `apps/api/src/rag_forge/router/` (`analyzer.py`, `policy.py`, `service.py`), wired in
`create_app()` / `make_router()`; HTTP in `api/corpus.py` (retrieve) and `api/routes.py`
(`/router/decide`).

```
query ─▶ QueryAnalyzer ─▶ QueryAnalysis ─┐
          (+ corpus term statistics)     ├─▶ RouterPolicy ─▶ RouterDecision ─▶ manual RetrievalRequest
corpus version ─▶ availability ──────────┘                                  ─▶ BM25 | dense | hybrid
                                                                             ─▶ (reranker) ─▶ hits
```

The router's output is an ordinary **manual** `RetrievalRequest`. A routed query runs through
exactly the same retrieval and reranking code as a fixed one, so its results equal those of the
fixed configuration it chose (tested), and `decision.configuration_hash` is the hash that fixed
request reports on its own.

## Query intelligence

`QueryAnalyzer` is a contract: `name`, `version`, `config_hash()` and
`analyze(query, corpus_signals) -> QueryAnalysis`. `HeuristicQueryAnalyzer`
(`heuristic-query-analyzer@1`) is a transparent baseline, **not a trained model**: every feature
is a pattern match or a count, every score is a clipped sum of listed contributions, every label
is a threshold. A learned classifier can replace it by implementing the contract and registering
under another name; requests choose it with `router.analyzer`.

### Normalisation

NFKC, curly quotes to straight quotes, whitespace collapsed, trimmed. Case is kept, because
capitalisation is a feature. Queries that normalise identically produce the same analysis and
`analysis_hash` (the raw text is recorded but not hashed).

### Features (`QueryFeatures`)

| Feature | Definition |
|---|---|
| `char_count`, `token_count` | characters of the normalised query; Unicode word tokens |
| `bm25_terms` | distinct terms the BM25 analyzer produces, exactly what BM25 searches |
| `key_terms` | `bm25_terms` minus function words |
| `function_word_ratio` | share of tokens that are function words (Lucene stopwords, interrogatives, auxiliaries, pronouns, prepositions) |
| `is_question`, `question_word` | ends with `?`, or starts with an interrogative, auxiliary or request verb |
| `question_type` | first match of: comparison, definition, list, factoid (how many/much…), procedural, explanatory, boolean, factoid, keyword |
| `quoted_phrases` | text in double quotes |
| `identifiers` | tokens with a digit, `_`, inner capital, ≥ 2 capitals, or `.`/`/` between word characters (not `e.g.`, not plain numbers) |
| `capitalized_terms` | runs of capitalised words that do not start a sentence |
| `numbers` | integers, decimals, percentages |
| `entities` | quoted phrases ∪ identifiers ∪ capitalised terms (case-insensitive dedupe) |
| `concept_segments` | parts split on `and`, `or`, `vs`, `versus`, `compared to/with`, `as well as`, commas, semicolons that contain a key term |
| `comparison_markers`, `multi_hop_markers`, `temporal_markers`, `negation_markers`, `ambiguity_markers` | fixed phrase lists matched on word boundaries (years count as temporal; `n't` as negation) |

### Signals

Each signal is `clip(Σ value × weight, 0, 1)`, and its contributions are returned so the score can
be read term by term. The weights of each signal sum to 1 and are part of the analyzer's
`config_hash`.

| Signal | Contributions (value → weight) |
|---|---|
| lexical | identifiers (≥ 1) 0.35 · quoted phrases 0.20 · numbers 0.10 · keyword form (not a question) 0.20 · term density (1 − function-word ratio) 0.15 |
| semantic | question form 0.25 · function words (ratio / 0.5, capped) 0.20 · length ((tokens − 3) / 9, capped) 0.20 · descriptive need (definition, procedural, explanatory, comparison, list) 0.20 · no exact anchor (no identifier or quote) 0.15 |
| complexity | key terms ((n − 2) / 8) 0.30 · concepts ((segments − 1) / 2) 0.25 · multi-hop markers (n / 2) 0.25 · comparison 0.10 · constraints ((temporal + negation) / 2) 0.10 |

### Labels

- `query_class`: lexical if `lexical − semantic ≥ 0.15`, semantic if `≤ −0.15`, otherwise mixed.
  `class_margin` is that difference; its magnitude is the only "confidence", and it is not a
  probability.
- `complexity`: simple below 0.25, moderate below 0.5, complex from 0.5.
- `multi_hop_likely`: ≥ 2 multi-hop markers, or 1 marker and ≥ 2 concept segments.
- `ambiguous`: no entity, and at most one key term or a vague referent (`it`, `this`, `things`…).
- `evidence_need`: multiple passages if multi-hop, comparison, list or complex; otherwise single.

### Corpus signals

Before analysing, the router looks up each BM25 term's document frequency in the requested corpus
version (the same version-scoped lexical index BM25 uses) and records `TermStatistic`s (df, BM25
idf), `coverage` (share of key terms present; all BM25 terms if there are no key terms),
`missing_terms` and `mean_idf`. These are the only retrieval-difficulty signals that are measured
rather than guessed: a query whose key terms are absent cannot be matched by BM25.

## Router policy `rules-baseline@1`

`RouterPolicy` is a contract (`name`, `version`, `config_hash()`, `decide(analysis, context)`).
`RulePolicy` evaluates ordered rules; the first match decides, and every rule evaluated up to it
is recorded with the exact inputs it read, its outcome and its margin (distance of those inputs
from the rule's nearest threshold).

Strategy rules:

| # | Rule | Condition | Option |
|---|---|---|---|
| 1 | `no-searchable-terms` | no BM25 terms (stopwords only) | dense |
| 2 | `vocabulary-mismatch` | corpus coverage < 0.5 | dense |
| 3 | `exact-lookup` | lexical class and simple complexity | BM25 |
| 4 | `anchored-complex` | lexical class (not simple) | hybrid weighted, BM25-leaning |
| 5 | `semantic-unanchored` | semantic class, no entity | dense |
| 6 | `semantic-anchored` | semantic class with entities | hybrid weighted, dense-leaning |
| 7 | `balanced` | mixed class | hybrid RRF (k = 60) |

Weighted fusion uses `dense = clamp(0.5 − class_margin, 0.2, 0.8)` on a 0.05 grid and
`BM25 = 1 − dense`; components fetch `max(50, pool)` candidates.

Rerank rules (when a reranker is registered; otherwise `rerank-unavailable`):
`rerank-exact-lookup` (lexical and simple → no rerank), `rerank-precision` (semantic ≥ 0.4, or
not simple, or likely multi-hop → rerank with a pool of 20 / 30 / 50 for simple / moderate /
complex, at least `top_k`), `rerank-default` (no rerank).

**Availability.** Before deciding, the router measures what the corpus version can serve: without
a ready dense index, dense and both hybrids are unavailable. If the policy prefers one of them,
the decision keeps `preferred`, adds a matched `constraint-unavailable` rule with the reason, and
runs BM25 (`option`). This is recorded, never silent, and the UI marks it. If availability is
wrong (the index disappears between decision and retrieval), retrieval fails with its usual 409;
there is no second fallback.

**Rationale.** `rationale` is the fired rules' templates filled with measured values, for
example `semantic-unanchored: semantic query (margin -0.57) with no entity to match exactly →
dense`. Nothing in it is generated or inferred beyond those values.

**Alternatives.** For each of the four options the decision lists whether it was selected,
whether the version can serve it (and why not), and which policy rules select it.

## Manual and adaptive retrieval

`POST /api/v1/corpora/{id}/retrieve` takes `mode`:

- `manual` (default): unchanged. `strategy`, `hybrid` and `rerank` are used as given; `router`
  is ignored. Existing requests and configuration hashes are unaffected.
- `adaptive`: the router replaces `strategy`, `hybrid` and `rerank`; `query`, `top_k`, `version`
  and `bm25` are honoured. `router: {analyzer, policy}` selects registered components (unknown →
  501, `capability: "Router"`).

`POST /api/v1/router/decide` (`corpus_id`, `query`, `version`, `top_k`, `router`) returns the
analysis and decision without retrieving, plus `request`: the manual request the router selected.
Sending that request to `/retrieve` reproduces the adaptive run.

The Retrieval Lab has a Manual / Adaptive switch (and `/router` opens it in adaptive mode). In
adaptive mode it shows the query intelligence panel (labels, lexical–semantic balance, each
signal's contributions, features, corpus term statistics), the decision panel (selected option
and parameters, rationale, full rule traces, alternatives, hashes), a latency breakdown
(analysis, decision, retrieval, rerank), the adaptive run next to a fixed baseline the user picks
(overlap, rank alignment, latency), and a button that runs all four options with the router's
rerank setting. Manual mode, the four-way comparison and the reranking analysis are unchanged.

## Provenance

An adaptive response's `provenance.routing` holds:

- `routing`: the routing identity (analyzer + version + config hash, policy + version + config
  hash) and `routing_hash`;
- `analysis`: the complete `QueryAnalysis`, including corpus term statistics, and its
  `analysis_hash`;
- `decision`: the complete `RouterDecision` (option, preferred, parameters, rule traces,
  alternatives, margin, rationale, `configuration_hash` of the selected manual configuration,
  `decision_hash` over everything but ids and timestamps);
- `analysis_ms` and `decision_ms`.

`configuration.routing` carries the routing identity, so an adaptive configuration hashes
differently from the fixed configuration it selected, while `decision.configuration_hash` links
the two. Retrieval and reranking provenance are recorded as for any request; the reranking
step's upstream configuration excludes routing. Given the same corpus version, query, analyzer
and policy versions, the analysis and decision hashes are reproduced exactly.

## Limits

- The analyzer is English-centric phrase lists and weights chosen by hand, not learned or tuned.
  Its labels are descriptions of the text, not judgements of what the user needs.
- Thresholds (0.15, 0.25 / 0.5, 0.5 coverage, 0.4 rerank) are priors. They have not been
  calibrated, and queries near a threshold (small `margin`) can flip with small edits.
- The policy routes among four retrieval options and one reranker; it does not decompose queries,
  run multiple hops or verify evidence.
- Corpus coverage is lexical: a term present in the corpus with another meaning still counts.
- The router never reads retrieval results; it decides before retrieving.

## Why there are no claims of superiority

Nothing here shows that adaptive routing retrieves better than any fixed pipeline. The lab shows
agreement between rankings, which is not quality. The Arena measures routing against fixed
strategies on judged datasets (`hybrid-rrf-rerank → adaptive` is a preset ablation), with the
routing identity hash as the unit that is varied. On the bundled development benchmark the
comparison shows no detectable quality difference, because that set is saturated
([research summary](../research/summary.md)); a dataset where methods differ is needed.
