"""The metric registry. Each metric declares its family, version, the annotations and pipeline
outputs it needs, whether it is evaluated at k, and its unit. `compute` returns a value or a
skip reason for every requested metric. A metric whose inputs are missing is never 0.

Relevance unit: chunk ids if the case annotates chunks, otherwise documents (filenames), with the
chunk ranking collapsed to a document ranking at each document's best rank.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from rag_forge.domain.arena import (
    Annotation,
    BenchmarkCase,
    MetricDefinition,
    MetricFamily,
    MetricSettings,
    MetricValue,
    PipelineKind,
    RankedItem,
)
from rag_forge.domain.models import AnswerStatus, RagResponse
from rag_forge.evaluation import retrieval_metrics as rm

REGISTRY_VERSION = "arena-metrics@1"


class Skip(Exception):
    """The metric is undefined for this case; the message says why."""


@dataclass(frozen=True)
class MetricContext:
    case: BenchmarkCase
    pipeline: PipelineKind
    ranking: list[RankedItem]  # final ranking, best first
    pool_upstream: list[RankedItem] | None = None  # reranked runs: the whole pool, upstream order
    pool_final: list[RankedItem] | None = None  # the whole pool, reranked order
    rag: RagResponse | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)


def judgements(case: BenchmarkCase) -> tuple[str, dict[str, float]]:
    if case.relevant_chunks:
        return "chunk", dict(case.relevant_chunks)
    return "document", dict(case.relevant_documents)


def ranked_ids(items: list[RankedItem], unit: str) -> list[str]:
    """Chunk ids, or filenames in order of each document's best-ranked chunk."""
    if unit == "chunk":
        return [i.chunk_id for i in items]
    return list(dict.fromkeys(i.filename for i in items))


def _relevance(ctx: MetricContext) -> tuple[str, dict[str, float]]:
    unit, rel = judgements(ctx.case)
    if not any(g > 0 for g in rel.values()):
        raise Skip("no relevance judgements for this case")
    return unit, rel


def _pools(ctx: MetricContext) -> tuple[list[RankedItem], list[RankedItem]]:
    if ctx.pool_upstream is None or ctx.pool_final is None:
        raise Skip("the arm does not rerank")
    return ctx.pool_upstream, ctx.pool_final


def _grounding(ctx: MetricContext, attr: str) -> float:
    if ctx.rag is None:
        raise Skip("retrieval-only arm")
    g = ctx.rag.grounding
    if g is None:
        raise Skip(f"no grounding ({ctx.rag.status.value})")
    value = getattr(g, attr)
    if value is None:
        raise Skip(f"{attr} undefined: no factual claims or citations")
    return float(value)


def _rate(ctx: MetricContext, count: str) -> float:
    if ctx.rag is None:
        raise Skip("retrieval-only arm")
    g = ctx.rag.grounding
    if g is None:
        raise Skip(f"no grounding ({ctx.rag.status.value})")
    if g.factual_claims == 0:
        raise Skip("no factual claims")
    return float(getattr(g, count)) / g.factual_claims


_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _normalize(text: str) -> list[str]:
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    return _ARTICLES.sub(" ", text).split()


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 after lowercasing and removing punctuation and articles."""
    p, r = _normalize(prediction), _normalize(reference)
    common = sum((Counter(p) & Counter(r)).values())
    if not p or not r or common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def _answer_f1(ctx: MetricContext) -> float:
    if not ctx.case.reference_answer:
        raise Skip("no reference answer")
    if ctx.rag is None:
        raise Skip("retrieval-only arm")
    if ctx.rag.answer is None:
        raise Skip(f"no generated answer ({ctx.rag.status.value})")
    return token_f1(ctx.rag.answer.text, ctx.case.reference_answer)


def _abstention(ctx: MetricContext) -> float:
    if ctx.case.answerable is None:
        raise Skip("answerability not annotated")
    if ctx.rag is None:
        raise Skip("retrieval-only arm")
    if ctx.rag.status is AnswerStatus.NOT_GENERATED:
        raise Skip("the arm does not generate")
    declined = ctx.rag.status in (AnswerStatus.ABSTAINED, AnswerStatus.INSUFFICIENT_EVIDENCE)
    return float(declined != ctx.case.answerable)


def _evidence_recall(ctx: MetricContext) -> float:
    if not ctx.case.expected_evidence:
        raise Skip("no expected evidence annotated")
    if ctx.rag is None:
        raise Skip("retrieval-only arm")
    selected = {e.filename for e in ctx.rag.evidence.selected}
    expected = set(ctx.case.expected_evidence)
    return len(expected & selected) / len(expected)


def _timing(name: str) -> Callable[[MetricContext, int | None], float]:
    def fn(ctx: MetricContext, k: int | None) -> float:
        if name not in ctx.timings_ms:
            raise Skip(f"no {name} stage in this arm")
        return ctx.timings_ms[name]

    return fn


def _tokens(name: str) -> Callable[[MetricContext, int | None], float]:
    def fn(ctx: MetricContext, k: int | None) -> float:
        if name not in ctx.tokens:
            raise Skip("the generator does not report tokens" if ctx.rag else "no generation")
        return float(ctx.tokens[name])

    return fn


def _retrieval(fn: Callable[[list[str], dict[str, float], int], float]) -> Callable[..., float]:
    def compute(ctx: MetricContext, k: int | None) -> float:
        unit, rel = _relevance(ctx)
        assert k is not None
        return fn(ranked_ids(ctx.ranking, unit), rel, k)

    return compute


def _hit(ranked: list[str], rel: dict[str, float], k: int) -> float:
    return float(any(rel.get(i, 0) > 0 for i in ranked[:k]))


def _mrr(ctx: MetricContext, k: int | None) -> float:
    unit, rel = _relevance(ctx)
    return rm.reciprocal_rank(ranked_ids(ctx.ranking, unit), rel)


def _rerank_mrr_delta(ctx: MetricContext, k: int | None) -> float:
    unit, rel = _relevance(ctx)
    up, final = _pools(ctx)
    return rm.reciprocal_rank(ranked_ids(final, unit), rel) - rm.reciprocal_rank(
        ranked_ids(up, unit), rel
    )


def _rerank_ndcg_delta(ctx: MetricContext, k: int | None) -> float:
    unit, rel = _relevance(ctx)
    up, final = _pools(ctx)
    assert k is not None
    return rm.ndcg_at_k(ranked_ids(final, unit), rel, k) - rm.ndcg_at_k(
        ranked_ids(up, unit), rel, k
    )


def _preservation(ctx: MetricContext, k: int | None) -> float:
    unit, rel = _relevance(ctx)
    up, final = _pools(ctx)
    assert k is not None
    reachable = {i for i in ranked_ids(up, unit) if rel.get(i, 0) > 0}
    if not reachable:
        raise Skip("no relevant item in the candidate pool")
    return len(reachable & set(ranked_ids(final, unit)[:k])) / len(reachable)


def _shifts(ctx: MetricContext) -> list[int]:
    """upstream rank - final rank for every pooled candidate (positive = promoted)."""
    _, final = _pools(ctx)
    shifts = [i.upstream_rank - i.rank for i in final if i.upstream_rank is not None]
    if not shifts:
        raise Skip("the candidate pool is empty")
    return shifts


def _promoted(ctx: MetricContext, k: int | None) -> float:
    shifts = _shifts(ctx)
    return sum(d > 0 for d in shifts) / len(shifts)


def _shift(ctx: MetricContext, k: int | None) -> float:
    shifts = _shifts(ctx)
    return sum(abs(d) for d in shifts) / len(shifts)


@dataclass(frozen=True)
class Entry:
    definition: MetricDefinition
    fn: Callable[[MetricContext, int | None], float]


def _d(
    name: str,
    family: MetricFamily,
    description: str,
    fn: Callable[[MetricContext, int | None], float],
    requires: list[Annotation] | None = None,
    outputs: list[str] | None = None,
    per_k: bool = False,
    unit: str = "score",
    higher: bool | None = True,
) -> Entry:
    return Entry(
        MetricDefinition(
            name=name,
            family=family,
            version=1,
            description=description,
            requires=requires or [],
            requires_output=outputs or [],
            per_k=per_k,
            unit=unit,
            higher_is_better=higher,
        ),
        fn,
    )


R, RR, E, G, OP = (
    MetricFamily.RETRIEVAL,
    MetricFamily.RERANKING,
    MetricFamily.EVIDENCE,
    MetricFamily.GENERATION,
    MetricFamily.OPERATIONAL,
)
REL = [Annotation.RELEVANCE]

ENTRIES: list[Entry] = [
    _d("recall", R, "Share of relevant items in the top k.", _retrieval(rm.recall_at_k), REL,
       per_k=True, unit="fraction"),
    _d("precision", R, "Share of the top k that is relevant (divides by k).",
       _retrieval(rm.precision_at_k), REL, per_k=True, unit="fraction"),
    _d("hit_rate", R, "1 if any relevant item is in the top k, else 0.", _retrieval(_hit), REL,
       per_k=True, unit="fraction"),
    _d("mrr", R, "1 / rank of the first relevant item; 0 if none is retrieved.", _mrr, REL,
       unit="fraction"),
    _d("ndcg", R, "nDCG@k with gain 2^grade - 1 and log2(rank + 1) discount.",
       _retrieval(rm.ndcg_at_k), REL, per_k=True, unit="fraction"),
    _d("rerank_mrr_delta", RR, "MRR of the reranked pool minus MRR of the same pool upstream.",
       _rerank_mrr_delta, REL, ["reranking"], unit="difference"),
    _d("rerank_ndcg_delta", RR, "nDCG@k reranked minus nDCG@k upstream, over the same pool.",
       _rerank_ndcg_delta, REL, ["reranking"], per_k=True, unit="difference"),
    _d("recall_preservation", RR, "Share of the relevant items in the candidate pool that the "
       "reranker keeps in the final top k.", _preservation, REL, ["reranking"], per_k=True,
       unit="fraction"),
    _d("rerank_promoted_share", RR, "Share of the pool moved up by the reranker (descriptive).",
       _promoted, outputs=["reranking"], unit="fraction", higher=None),
    _d("rerank_mean_shift", RR, "Mean absolute rank change across the pool (descriptive).",
       _shift, outputs=["reranking"], unit="ranks", higher=None),
    _d("evidence_recall", E, "Share of the expected evidence documents that evidence selection "
       "kept.", lambda c, k: _evidence_recall(c), [Annotation.EXPECTED_EVIDENCE], ["evidence"],
       unit="fraction"),
    _d("grounding_score", E, "(supported + 0.5 x weakly supported) / factual claims, as measured "
       "by the grounding verifier.", lambda c, k: _grounding(c, "grounding_score"),
       outputs=["generation"], unit="fraction"),
    _d("supported_claim_rate", E, "Supported factual claims / factual claims.",
       lambda c, k: _rate(c, "supported"), outputs=["generation"], unit="fraction"),
    _d("unsupported_claim_rate", E, "Unsupported factual claims / factual claims.",
       lambda c, k: _rate(c, "unsupported"), outputs=["generation"], unit="fraction",
       higher=False),
    _d("citation_coverage", E, "Factual claims with a valid citation / factual claims.",
       lambda c, k: _grounding(c, "citation_coverage"), outputs=["generation"], unit="fraction"),
    _d("citation_precision", E, "Valid citations to a passage measured as supporting the claim "
       "/ valid citations.", lambda c, k: _grounding(c, "citation_precision"),
       outputs=["generation"], unit="fraction"),
    _d("evidence_coverage", E, "Selected evidence that supports at least one claim / selected "
       "evidence.", lambda c, k: _grounding(c, "evidence_coverage"), outputs=["generation"],
       unit="fraction"),
    _d("answer_token_f1", G, "Token F1 of the answer against the reference answer (SQuAD "
       "normalisation). A lexical proxy, not a judgement of answer quality.",
       lambda c, k: _answer_f1(c), [Annotation.REFERENCE_ANSWER], ["generation"],
       unit="fraction"),
    _d("abstention_accuracy", G, "1 if the arm answered an answerable case or declined an "
       "unanswerable one, else 0.", lambda c, k: _abstention(c), [Annotation.ANSWERABLE],
       ["generation"], unit="fraction"),
    _d("latency_ms", OP, "End-to-end wall time for the case.", _timing("total"), unit="ms",
       higher=False),
    _d("retrieval_ms", OP, "Retrieval, including routing and reranking.", _timing("retrieval"),
       unit="ms", higher=False),
    _d("generation_ms", OP, "Generation time, excluding model load.", _timing("generation"),
       outputs=["generation"], unit="ms", higher=False),
    _d("grounding_ms", OP, "Claim extraction and verification, excluding model load.",
       _timing("grounding"), outputs=["generation"], unit="ms", higher=False),
    _d("prompt_tokens", OP, "Prompt tokens reported by the generator (descriptive).",
       _tokens("prompt"), outputs=["generation"], unit="tokens", higher=None),
    _d("completion_tokens", OP, "Completion tokens reported by the generator (descriptive).",
       _tokens("completion"), outputs=["generation"], unit="tokens", higher=None),
]  # fmt: skip
REGISTRY = {e.definition.name: e for e in ENTRIES}

FAILED = MetricDefinition(
    name="failed",
    family=MetricFamily.OPERATIONAL,
    version=1,
    description="1 if the case failed in this arm, else 0; its mean is the failure rate.",
    requires=[],
    requires_output=[],
    per_k=False,
    unit="fraction",
    higher_is_better=False,
)


def definitions() -> list[MetricDefinition]:
    return [e.definition for e in ENTRIES] + [FAILED]


def selected(settings: MetricSettings) -> list[Entry]:
    if settings.metrics is None:
        return ENTRIES
    unknown = sorted(set(settings.metrics) - set(REGISTRY))
    if unknown:
        raise ValueError(f"unknown metrics: {', '.join(unknown)}")
    return [REGISTRY[n] for n in settings.metrics]


def versions(settings: MetricSettings) -> dict[str, int]:
    return {e.definition.name: e.definition.version for e in selected(settings)} | {"failed": 1}


def metric_names(entry: Entry, ks: list[int]) -> list[tuple[str, int | None]]:
    if entry.definition.per_k:
        return [(f"{entry.definition.name}@{k}", k) for k in ks]
    return [(entry.definition.name, None)]


def compute(ctx: MetricContext, settings: MetricSettings) -> list[MetricValue]:
    unit, _ = judgements(ctx.case)
    out = [MetricValue(metric="failed", family=FAILED.family, version=1, value=0.0)]
    for entry in selected(settings):
        d = entry.definition
        for name, k in metric_names(entry, settings.ks):
            try:
                value: float | None = round(float(entry.fn(ctx, k)), 6)
                skipped = None
            except Skip as why:
                value, skipped = None, str(why)
            detail = f"relevance unit: {unit}" if Annotation.RELEVANCE in d.requires else None
            out.append(
                MetricValue(
                    metric=name,
                    family=d.family,
                    version=d.version,
                    value=value,
                    skipped=skipped,
                    detail=detail if value is not None else None,
                )
            )
    return out


def failure_metrics() -> list[MetricValue]:
    """A failed case carries only `failed = 1`; every other metric is absent, not zero."""
    return [MetricValue(metric="failed", family=FAILED.family, version=1, value=1.0)]
