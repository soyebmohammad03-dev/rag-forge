# 0007: Query intelligence and adaptive routing

Date: 2026-10-06 · Status: accepted

## Decisions

1. **Analyzer and policy are separate contracts.** `QueryAnalyzer` turns text (plus corpus term
   statistics) into a `QueryAnalysis`. `RouterPolicy` turns that, plus what the corpus version can
   serve, into a decision. Both register by name, so a learned classifier or a learned policy
   can replace either one without touching retrieval.
2. **The baseline is honest about being heuristic.** Features are pattern matches and counts.
   Scores are clipped weighted sums that return their contributions. Labels are thresholds. The
   only confidence reported is a margin to a threshold, and it is never called a probability.
3. **Corpus vocabulary coverage is the one measured difficulty signal.** It comes from the same
   version-scoped lexical index BM25 uses, so it is exact for that corpus version.
4. **Routing produces a manual request.** Adaptive runs reuse the fixed pipeline unchanged. Their
   results equal those of the configuration they selected, and `/router/decide` returns that
   request so anyone can replay it.
5. **Decisions are fully traced.** Every evaluated rule is recorded with the inputs it read, its
   outcome and its margin. The rationale is those rules filled in with values. Alternatives
   list availability and the rules that would select each one.
6. **Availability is an explicit, recorded constraint.** Without a ready dense index the router
   keeps its preference, records `constraint-unavailable` with the reason, and runs BM25. If the
   routed run itself fails, the request fails.
7. **Hashes separate "how it was routed" from "what ran".** `configuration.routing` makes
   adaptive configurations distinct. `decision.configuration_hash` is the selected fixed
   configuration's hash. `routing_hash` identifies analyzer and policy versions for the Arena.
8. **No claims of superiority.** The lab shows decisions, agreement and latency. Whether routing
   beats fixed pipelines is measured in Phase 8 against judged queries.
