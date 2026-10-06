"""Router policies: QueryAnalysis + what can run -> one retrieval configuration, with its reasons.

`RulePolicy` is the deterministic baseline. Strategy rules are evaluated in order and the first
match decides; reranking has its own ordered rules. Every evaluated rule is recorded with the
exact inputs it read and its margin to the nearest threshold, and the rationale is those rules
filled in with measured values, so a decision explains itself without any generated text.

An option the corpus version cannot serve (no ready dense index) is never chosen silently: the
policy's preference is kept, a constraint rule records why it was replaced, and the decision
says which option actually ran.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_forge.domain.models import (
    Complexity,
    FusionMethod,
    HybridParams,
    QueryAnalysis,
    QueryClass,
    RerankParams,
    RetrievalStrategy,
    RouteOption,
    RuleEvaluation,
    canonical_hash,
)

STRATEGY = {
    RouteOption.SPARSE: RetrievalStrategy.SPARSE,
    RouteOption.DENSE: RetrievalStrategy.DENSE,
    RouteOption.HYBRID_RRF: RetrievalStrategy.HYBRID,
    RouteOption.HYBRID_WEIGHTED: RetrievalStrategy.HYBRID,
}
LABEL = {
    RouteOption.SPARSE: "BM25",
    RouteOption.DENSE: "dense",
    RouteOption.HYBRID_RRF: "hybrid RRF",
    RouteOption.HYBRID_WEIGHTED: "hybrid weighted",
}


@dataclass(frozen=True)
class RouterContext:
    """What the requested corpus version can serve, measured before deciding."""

    top_k: int
    unavailable: Mapping[RouteOption, str] = field(default_factory=dict)  # option -> reason
    reranker: str | None = None  # the registered reranker model, if any


@dataclass(frozen=True)
class RoutePlan:
    option: RouteOption
    preferred: RouteOption
    hybrid: HybridParams | None
    rerank: RerankParams
    rules: list[RuleEvaluation]
    rerank_rules: list[RuleEvaluation]
    margin: float | None
    rationale: list[str]
    rule_options: dict[RouteOption, list[str]]  # which rules select each option


class RouterPolicy(Protocol):
    name: str
    version: str

    def config_hash(self) -> str: ...

    def decide(self, analysis: QueryAnalysis, context: RouterContext) -> RoutePlan: ...


CONFIG: dict[str, Any] = {
    "min_coverage": 0.5,  # below this share of query terms in the corpus, BM25 cannot anchor
    "class_margin": 0.15,  # mirrors the analyzer's label threshold (recorded, not re-derived)
    "moderate_complexity": 0.25,
    "rerank_min_semantic": 0.4,
    "rerank_candidates": {"simple": 20, "moderate": 30, "complex": 50},
    "hybrid_candidates": 50,
    "rrf_k": 60,
    "dense_weight": {"base": 0.5, "min": 0.2, "max": 0.8, "step": 0.05},
}

Outcome = tuple[bool, dict[str, Any], float | None, str]  # matched, inputs, margin, rationale


@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    evaluate: Callable[[QueryAnalysis], Outcome]
    option: RouteOption | None = None  # strategy rules
    rerank: bool | None = None  # rerank rules


def _r(x: float) -> float:
    return round(x, 4)


def _no_terms(a: QueryAnalysis) -> Outcome:
    n = len(a.features.bm25_terms)
    return n == 0, {"bm25_terms": n}, None, "no BM25-searchable terms (stopwords only) → dense"


def _vocabulary(a: QueryAnalysis) -> Outcome:
    c = a.corpus
    if c is None or not c.terms:
        return False, {"coverage": None}, None, ""
    t = CONFIG["min_coverage"]
    return (
        c.coverage < t,
        {"coverage": c.coverage, "missing_terms": len(c.missing_terms)},
        _r(abs(c.coverage - t)),
        f"only {c.coverage:.0%} of query terms occur in corpus v{c.corpus_version} "
        f"(< {t:.0%}; missing: {', '.join(c.missing_terms[:5])}) → dense",
    )


def _class_inputs(a: QueryAnalysis) -> dict[str, Any]:
    complexity = next(s.score for s in a.signals if s.name == "complexity")
    return {
        "query_class": a.labels.query_class.value,
        "class_margin": a.labels.class_margin,
        "complexity": complexity,
        "entities": len(a.features.entities),
    }


def _class_margin(a: QueryAnalysis) -> float:
    return _r(abs(abs(a.labels.class_margin) - CONFIG["class_margin"]))


def _exact_lookup(a: QueryAnalysis) -> Outcome:
    i = _class_inputs(a)
    hit = a.labels.query_class is QueryClass.LEXICAL and a.labels.complexity is Complexity.SIMPLE
    margin = min(_class_margin(a), _r(abs(i["complexity"] - CONFIG["moderate_complexity"])))
    return (
        hit,
        i,
        margin,
        f"lexical query (margin {i['class_margin']:+.2f}) of simple complexity "
        f"({i['complexity']:.2f} < {CONFIG['moderate_complexity']}) → BM25",
    )


def _anchored_complex(a: QueryAnalysis) -> Outcome:
    i = _class_inputs(a)
    return (
        a.labels.query_class is QueryClass.LEXICAL,
        i,
        _class_margin(a),
        f"lexical query (margin {i['class_margin']:+.2f}) that is not simple "
        f"(complexity {i['complexity']:.2f}) → hybrid weighted toward BM25",
    )


def _semantic_unanchored(a: QueryAnalysis) -> Outcome:
    i = _class_inputs(a)
    return (
        a.labels.query_class is QueryClass.SEMANTIC and not a.features.entities,
        i,
        _class_margin(a),
        f"semantic query (margin {i['class_margin']:+.2f}) with no entity to match exactly → dense",
    )


def _semantic_anchored(a: QueryAnalysis) -> Outcome:
    i = _class_inputs(a)
    return (
        a.labels.query_class is QueryClass.SEMANTIC,
        i,
        _class_margin(a),
        f"semantic query (margin {i['class_margin']:+.2f}) with {i['entities']} "
        f"entit{'y' if i['entities'] == 1 else 'ies'} ({', '.join(a.features.entities[:3])}) "
        "→ hybrid weighted toward dense",
    )


def _balanced(a: QueryAnalysis) -> Outcome:
    i = _class_inputs(a)
    return (
        a.labels.query_class is QueryClass.MIXED,
        i,
        _class_margin(a),
        f"mixed query: lexical and semantic scores within {CONFIG['class_margin']} "
        f"(margin {i['class_margin']:+.2f}) → hybrid RRF",
    )


def _rerank_inputs(a: QueryAnalysis) -> dict[str, Any]:
    semantic = next(s.score for s in a.signals if s.name == "semantic")
    return {
        "query_class": a.labels.query_class.value,
        "complexity": a.labels.complexity.value,
        "semantic": semantic,
        "multi_hop_likely": a.labels.multi_hop_likely,
    }


def _rerank_exact(a: QueryAnalysis) -> Outcome:
    hit = a.labels.query_class is QueryClass.LEXICAL and a.labels.complexity is Complexity.SIMPLE
    return hit, _rerank_inputs(a), None, "simple exact-term lookup: upstream order kept, no rerank"


def _rerank_precision(a: QueryAnalysis) -> Outcome:
    i = _rerank_inputs(a)
    t = CONFIG["rerank_min_semantic"]
    hit = (
        i["semantic"] >= t or a.labels.complexity is not Complexity.SIMPLE or i["multi_hop_likely"]
    )
    why = []
    if i["semantic"] >= t:
        why.append(f"semantic score {i['semantic']:.2f} ≥ {t}")
    if a.labels.complexity is not Complexity.SIMPLE:
        why.append(f"{i['complexity']} complexity")
    if i["multi_hop_likely"]:
        why.append("likely multi-hop")
    return hit, i, _r(abs(i["semantic"] - t)), f"{' and '.join(why)} → rerank"


def _rerank_default(a: QueryAnalysis) -> Outcome:
    return True, _rerank_inputs(a), None, "no rerank rule applies → no rerank"


STRATEGY_RULES = (
    Rule(
        "no-searchable-terms",
        "The query has no BM25-searchable terms: route to dense.",
        _no_terms,
        RouteOption.DENSE,
    ),
    Rule(
        "vocabulary-mismatch",
        "Under half of the query's terms occur in the corpus version: route to dense.",
        _vocabulary,
        RouteOption.DENSE,
    ),
    Rule(
        "exact-lookup",
        "Lexical class and simple complexity: route to BM25.",
        _exact_lookup,
        RouteOption.SPARSE,
    ),
    Rule(
        "anchored-complex",
        "Lexical class, not simple: hybrid weighted fusion leaning toward BM25.",
        _anchored_complex,
        RouteOption.HYBRID_WEIGHTED,
    ),
    Rule(
        "semantic-unanchored",
        "Semantic class with no entity: route to dense.",
        _semantic_unanchored,
        RouteOption.DENSE,
    ),
    Rule(
        "semantic-anchored",
        "Semantic class with entities: hybrid weighted fusion leaning toward dense.",
        _semantic_anchored,
        RouteOption.HYBRID_WEIGHTED,
    ),
    Rule(
        "balanced",
        "Mixed class: hybrid reciprocal rank fusion.",
        _balanced,
        RouteOption.HYBRID_RRF,
    ),
)
RERANK_RULES = (
    Rule(
        "rerank-exact-lookup",
        "Simple exact-term lookups keep the upstream order.",
        _rerank_exact,
        rerank=False,
    ),
    Rule(
        "rerank-precision",
        "Rerank when the semantic score is high, the query is not simple, or likely multi-hop.",
        _rerank_precision,
        rerank=True,
    ),
    Rule("rerank-default", "Otherwise, no rerank.", _rerank_default, rerank=False),
)


def _evaluate(rule: Rule, a: QueryAnalysis) -> tuple[RuleEvaluation, str]:
    matched, inputs, margin, rationale = rule.evaluate(a)
    outcome = rule.option.value if rule.option else ("rerank" if rule.rerank else "no rerank")
    ev = RuleEvaluation(
        rule=rule.id,
        description=rule.description,
        matched=matched,
        inputs=inputs,
        margin=margin,
        outcome=outcome,
    )
    return ev, f"{rule.id}: {rationale}"


def dense_weight(class_margin: float) -> float:
    """Lean toward the stronger side by the class margin, clamped and on a 0.05 grid."""
    w = CONFIG["dense_weight"]
    raw = min(w["max"], max(w["min"], w["base"] - class_margin))
    return float(round(round(raw / w["step"]) * w["step"], 2))


class RulePolicy:
    name = "rules-baseline"
    version = "rules-baseline@1"

    def config_hash(self) -> str:
        rules = [(r.id, r.description) for r in (*STRATEGY_RULES, *RERANK_RULES)]
        return canonical_hash({"version": self.version, "rules": rules, **CONFIG})

    def decide(self, analysis: QueryAnalysis, context: RouterContext) -> RoutePlan:
        rules: list[RuleEvaluation] = []
        rationale: list[str] = []
        preferred = RouteOption.HYBRID_RRF  # unreachable default: "balanced" catches the rest
        margin = None
        for rule in STRATEGY_RULES:
            ev, why = _evaluate(rule, analysis)
            rules.append(ev)
            if ev.matched:
                assert rule.option is not None
                preferred, margin = rule.option, ev.margin
                rationale.append(why)
                break

        option = preferred
        if preferred in context.unavailable:
            option = RouteOption.SPARSE  # BM25 indexes lazily, so it can always run
            reason = context.unavailable[preferred]
            rules.append(
                RuleEvaluation(
                    rule="constraint-unavailable",
                    description="A preferred option the corpus version cannot serve runs as BM25.",
                    matched=True,
                    inputs={"preferred": preferred.value, "reason": reason},
                    outcome=option.value,
                )
            )
            rationale.append(
                f"constraint-unavailable: {LABEL[preferred]} preferred but {reason} → BM25"
            )

        hybrid = None
        if STRATEGY[option] is RetrievalStrategy.HYBRID:
            weighted = option is RouteOption.HYBRID_WEIGHTED
            wd = dense_weight(analysis.labels.class_margin)
            hybrid = HybridParams(
                fusion=FusionMethod.WEIGHTED if weighted else FusionMethod.RRF,
                rrf_k=CONFIG["rrf_k"],
                candidate_k=CONFIG["hybrid_candidates"],
                weights={
                    RetrievalStrategy.SPARSE: round(1 - wd, 2),
                    RetrievalStrategy.DENSE: wd,
                },
            )
            if weighted:
                rationale.append(
                    f"weights: dense {wd} = clamp({CONFIG['dense_weight']['base']} - class margin "
                    f"{analysis.labels.class_margin:+.2f}), BM25 {round(1 - wd, 2)}"
                )

        rerank_rules: list[RuleEvaluation] = []
        rerank = RerankParams()
        if context.reranker is None:
            rerank_rules.append(
                RuleEvaluation(
                    rule="rerank-unavailable",
                    description="No reranker is registered.",
                    matched=True,
                    inputs={"reranker": None},
                    outcome="no rerank",
                )
            )
            rationale.append("rerank-unavailable: no reranker registered → no rerank")
        else:
            for rule in RERANK_RULES:
                ev, why = _evaluate(rule, analysis)
                rerank_rules.append(ev)
                if ev.matched:
                    rationale.append(why)
                    if rule.rerank:
                        table = CONFIG["rerank_candidates"]
                        k = max(table[analysis.labels.complexity.value], context.top_k)
                        rerank = RerankParams(enabled=True, model=context.reranker, candidate_k=k)
                    break

        if hybrid is not None:  # components must supply at least the pool the next step needs
            pool = rerank.candidate_k if rerank.enabled else context.top_k
            hybrid = hybrid.model_copy(update={"candidate_k": max(hybrid.candidate_k, pool)})

        rule_options: dict[RouteOption, list[str]] = {o: [] for o in RouteOption}
        for r in STRATEGY_RULES:
            assert r.option is not None
            rule_options[r.option].append(r.id)
        return RoutePlan(
            option=option,
            preferred=preferred,
            hybrid=hybrid,
            rerank=rerank,
            rules=rules,
            rerank_rules=rerank_rules,
            margin=margin,
            rationale=rationale,
            rule_options=rule_options,
        )
