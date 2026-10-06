"""Adaptive routing: analyse the query, measure what the corpus version can serve, decide.

The result is an ordinary manual `RetrievalRequest`, so a routed query runs through exactly the
same retrieval and reranking code as a fixed one, and its results can be compared one-to-one
with the fixed configuration the router chose.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from rag_forge.domain.models import (
    Corpus,
    CorpusQuerySignals,
    RerankParams,
    RetrievalConfiguration,
    RetrievalMode,
    RetrievalRequest,
    RouteAlternative,
    RouteOption,
    RouterDecision,
    RoutingConfiguration,
    RoutingProvenance,
    TermStatistic,
    canonical_hash,
)
from rag_forge.retrieval import bm25
from rag_forge.retrieval.analysis import ANALYZER
from rag_forge.router.analyzer import QueryAnalyzer
from rag_forge.router.policy import STRATEGY, RouterContext, RouterPolicy

# (corpus, version, terms) -> (chunk count, document frequency per term)
TermStatistics = Callable[[Corpus, int, list[str]], tuple[int, dict[str, int]]]
# (corpus, version) -> options that cannot run, with the reason
Availability = Callable[[Corpus, int], Mapping[RouteOption, str]]


class RouterComponentNotAvailableError(LookupError):
    def __init__(self, kind: str, name: str, available: list[str]) -> None:
        self.kind = kind
        names = ", ".join(sorted(available)) or "none"
        super().__init__(f"{kind} '{name}' is not registered (available: {names})")


class AdaptiveRouter:
    def __init__(
        self,
        analyzers: Mapping[str, QueryAnalyzer],
        policies: Mapping[str, RouterPolicy],
        term_statistics: TermStatistics,
        availability: Availability,
        rerankers: list[str],
    ) -> None:
        self.analyzers = dict(analyzers)
        self.policies = dict(policies)
        self.term_statistics = term_statistics
        self.availability = availability
        self.rerankers = rerankers

    def route(
        self,
        corpus: Corpus,
        version: int,
        request: RetrievalRequest,
        query_id: str,
        configure: Callable[[RetrievalRequest], RetrievalConfiguration],
    ) -> tuple[RetrievalRequest, RoutingProvenance]:
        """The manual request the policy selects, and the full record of why."""
        params = request.router
        analyzer = self.analyzers.get(params.analyzer)
        if analyzer is None:
            raise RouterComponentNotAvailableError(
                "query analyzer", params.analyzer, list(self.analyzers)
            )
        policy = self.policies.get(params.policy)
        if policy is None:
            raise RouterComponentNotAvailableError(
                "router policy", params.policy, list(self.policies)
            )

        started = time.perf_counter()
        bare = analyzer.analyze(request.query)  # terms first, then their corpus statistics
        f = bare.features
        analysis = analyzer.analyze(
            request.query,
            self._corpus_signals(corpus, version, f.bm25_terms, f.key_terms or f.bm25_terms),
        )
        analysis_ms = round((time.perf_counter() - started) * 1000, 3)

        started = time.perf_counter()
        default = RerankParams().model
        reranker = (
            default if default in self.rerankers else next(iter(sorted(self.rerankers)), None)
        )
        unavailable = dict(self.availability(corpus, version))
        plan = policy.decide(
            analysis, RouterContext(request.top_k, unavailable=unavailable, reranker=reranker)
        )
        routed = request.model_copy(
            update={
                "mode": RetrievalMode.MANUAL,
                "strategy": STRATEGY[plan.option],
                "rerank": plan.rerank,
                **({"hybrid": plan.hybrid} if plan.hybrid is not None else {}),
            }
        )
        RetrievalRequest.model_validate(routed.model_dump())  # a policy bug fails here, loudly
        body = {
            "query_id": query_id,
            "policy": policy.name,
            "policy_version": policy.version,
            "policy_config_hash": policy.config_hash(),
            "analysis_hash": analysis.analysis_hash,
            "option": plan.option,
            "preferred": plan.preferred,
            "strategy": STRATEGY[plan.option],
            "hybrid": plan.hybrid,
            "rerank": plan.rerank,
            "rules": plan.rules,
            "rerank_rules": plan.rerank_rules,
            "alternatives": [
                RouteAlternative(
                    option=o,
                    selected=o is plan.option,
                    available=o not in unavailable,
                    unavailable_reason=unavailable.get(o),
                    rules=plan.rule_options[o],
                )
                for o in RouteOption
            ],
            "margin": plan.margin,
            "rationale": plan.rationale,
            "configuration_hash": configure(routed).config_hash(),
        }
        decision_hash = canonical_hash(
            RouterDecision(**body, decision_hash="").model_dump(
                mode="json", exclude={"id", "query_id", "created_at", "decision_hash"}
            )
        )
        decision = RouterDecision(**body, decision_hash=decision_hash)
        decision_ms = round((time.perf_counter() - started) * 1000, 3)

        identity = RoutingConfiguration(
            analyzer=analyzer.name,
            analyzer_version=analyzer.version,
            analyzer_config_hash=analyzer.config_hash(),
            policy=policy.name,
            policy_version=policy.version,
            policy_config_hash=policy.config_hash(),
        )
        provenance = RoutingProvenance(
            routing=identity,
            routing_hash=identity.config_hash(),
            analysis=analysis,
            decision=decision,
            analysis_ms=analysis_ms,
            decision_ms=decision_ms,
        )
        return routed, provenance

    def _corpus_signals(
        self, corpus: Corpus, version: int, terms: list[str], counted: list[str]
    ) -> CorpusQuerySignals:
        """Statistics for every BM25 term; coverage over the terms that carry the query."""
        n, df = self.term_statistics(corpus, version, terms) if terms else (0, {})
        stats = [
            TermStatistic(term=t, document_frequency=df[t], idf=round(bm25.idf(n, df[t]), 6))
            for t in terms
        ]
        present = [s for s in stats if s.document_frequency > 0]
        missing = [t for t in counted if df.get(t, 0) == 0]
        return CorpusQuerySignals(
            corpus_id=corpus.id,
            corpus_version=version,
            analyzer=ANALYZER,
            chunk_count=n,
            terms=stats,
            coverage=round(1 - len(missing) / len(counted), 4) if counted else 0.0,
            missing_terms=missing,
            mean_idf=round(sum(s.idf for s in present) / len(present), 6) if present else None,
        )
