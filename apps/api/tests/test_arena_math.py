"""Arena metrics and statistics on small hand-computed fixtures."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from rag_forge.arena import metrics, stats
from rag_forge.arena.metrics import MetricContext, compute, ranked_ids, token_f1
from rag_forge.domain.arena import (
    BenchmarkCase,
    MetricFamily,
    MetricSettings,
    PipelineKind,
    RankedItem,
)


def item(rank: int, filename: str, chunk: str | None = None, up: int | None = None) -> RankedItem:
    return RankedItem(
        rank=rank,
        chunk_id=chunk or f"chk-{filename}-{rank}",
        document_id=f"doc-{filename}",
        filename=filename,
        score=1.0 / rank,
        upstream_rank=up,
        relevance=None,
    )


def values(ctx: MetricContext, ks: list[int] | None = None) -> dict[str, float | None]:
    return {m.metric: m.value for m in compute(ctx, MetricSettings(ks=ks or [1, 3]))}


def skips(ctx: MetricContext) -> dict[str, str | None]:
    return {m.metric: m.skipped for m in compute(ctx, MetricSettings(ks=[1, 3]))}


CASE = BenchmarkCase(id="c", query="q", relevant_documents={"a.txt": 2, "c.txt": 1, "z.txt": 0})
# chunks of b, a, a (second chunk of a), c: document ranking collapses to b, a, c
RANKING = [item(1, "b.txt"), item(2, "a.txt"), item(3, "a.txt"), item(4, "c.txt")]


def test_document_level_retrieval_metrics_match_hand_computation() -> None:
    ctx = MetricContext(case=CASE, pipeline=PipelineKind.RETRIEVAL, ranking=RANKING)
    assert ranked_ids(RANKING, "document") == ["b.txt", "a.txt", "c.txt"]
    v = values(ctx)
    assert v["recall@1"] == 0.0 and v["recall@3"] == 1.0  # {a, c} both in the top 3 documents
    assert v["precision@1"] == 0.0 and v["precision@3"] == pytest.approx(2 / 3, abs=1e-6)
    assert v["hit_rate@1"] == 0.0 and v["hit_rate@3"] == 1.0
    assert v["mrr"] == 0.5
    dcg = (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)
    idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert v["ndcg@3"] == pytest.approx(dcg / idcg, abs=1e-6)
    assert v["failed"] == 0.0


def test_chunk_level_judgements_take_precedence() -> None:
    case = BenchmarkCase(
        id="c", query="q", relevant_chunks={"x": 1}, relevant_documents={"b.txt": 1}
    )
    ranking = [item(1, "b.txt", "y"), item(2, "a.txt", "x")]
    out = compute(
        MetricContext(case=case, pipeline=PipelineKind.RETRIEVAL, ranking=ranking),
        MetricSettings(ks=[1]),
    )
    by = {m.metric: m for m in out}
    assert by["mrr"].value == 0.5 and by["mrr"].detail == "relevance unit: chunk"


def test_missing_annotations_skip_instead_of_scoring_zero() -> None:
    bare = BenchmarkCase(id="c", query="q")
    ctx = MetricContext(case=bare, pipeline=PipelineKind.RETRIEVAL, ranking=RANKING)
    s, v = skips(ctx), values(ctx)
    for name in ("recall@3", "precision@1", "mrr", "ndcg@3", "hit_rate@1"):
        assert v[name] is None and s[name] == "no relevance judgements for this case"
    assert s["rerank_mrr_delta"] == "no relevance judgements for this case"
    assert s["rerank_promoted_share"] == "the arm does not rerank"
    assert s["answer_token_f1"] == "no reference answer"
    assert s["abstention_accuracy"] == "answerability not annotated"
    assert s["evidence_recall"] == "no expected evidence annotated"
    assert s["grounding_score"] == "retrieval-only arm"
    assert s["latency_ms"] == "no total stage in this arm"
    judged_irrelevant = BenchmarkCase(id="c", query="q", relevant_documents={"a.txt": 0})
    ctx = MetricContext(case=judged_irrelevant, pipeline=PipelineKind.RETRIEVAL, ranking=RANKING)
    assert values(ctx)["mrr"] is None  # grade 0 everywhere: nothing relevant to find


def test_rerank_metrics_over_the_same_pool() -> None:
    # upstream order: c, b, a ; reranked order: a, b, c (k = 1)
    final = [item(1, "a.txt", up=3), item(2, "b.txt", up=2), item(3, "c.txt", up=1)]
    upstream = sorted(
        (i.model_copy(update={"rank": i.upstream_rank}) for i in final), key=lambda i: i.rank
    )
    case = BenchmarkCase(id="c", query="q", relevant_documents={"a.txt": 1})
    ctx = MetricContext(
        case=case,
        pipeline=PipelineKind.RETRIEVAL,
        ranking=final[:1],
        pool_upstream=upstream,
        pool_final=final,
    )
    v = values(ctx, [1])
    assert v["rerank_mrr_delta"] == pytest.approx(1 - 1 / 3, abs=1e-6)
    assert v["recall_preservation@1"] == 1.0  # the reachable relevant item ends in the top 1
    assert v["rerank_promoted_share"] == pytest.approx(1 / 3, abs=1e-6)  # a moved up
    assert v["rerank_mean_shift"] == pytest.approx(4 / 3, abs=1e-6)  # |3-1| + 0 + |1-3| over 3
    empty = MetricContext(
        case=case, pipeline=PipelineKind.RETRIEVAL, ranking=[], pool_upstream=[], pool_final=[]
    )
    s = skips(empty)
    assert s["rerank_promoted_share"] == "the candidate pool is empty"
    assert s["recall_preservation@1"] == "no relevant item in the candidate pool"


def test_token_f1() -> None:
    assert (
        token_f1("The upload exceeded the 25 MB limit.", "the upload exceeded the 25 MB limit")
        == 1.0
    )
    assert token_f1("carbon dioxide gas", "carbon dioxide") == pytest.approx(0.8)
    assert token_f1("nothing alike", "carbon dioxide") == 0.0
    assert token_f1("", "x") == 0.0


def test_metric_registry_declares_requirements() -> None:
    defs = {d.name: d for d in metrics.definitions()}
    assert defs["ndcg"].per_k and defs["ndcg"].requires == ["relevance"]
    assert defs["recall_preservation"].requires_output == ["reranking"]
    assert defs["answer_token_f1"].family is MetricFamily.GENERATION
    assert defs["latency_ms"].higher_is_better is False
    assert defs["rerank_mean_shift"].higher_is_better is None
    assert all(d.version >= 1 and d.aggregation for d in defs.values())
    assert metrics.versions(MetricSettings(metrics=["mrr"])) == {"mrr": 1, "failed": 1}
    with pytest.raises(ValueError, match="unknown metrics: bleu"):
        metrics.selected(MetricSettings(metrics=["bleu"]))
    with pytest.raises(ValidationError):
        MetricSettings(ks=[0])
    assert MetricSettings(ks=[10, 1, 10]).ks == [1, 10]


# --- statistics -----------------------------------------------------------------------


def test_describe_and_deterministic_bootstrap() -> None:
    d = stats.describe([1.0, 2.0, 3.0, 4.0])
    assert (d["mean"], d["median"], d["min"], d["max"]) == (2.5, 2.5, 1.0, 4.0)
    assert d["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1), abs=1e-6)
    assert d["ci_low"] is not None and d["ci_high"] is not None
    assert 1.0 <= d["ci_low"] <= 2.5 <= d["ci_high"] <= 4.0
    assert stats.describe([1.0, 2.0, 3.0, 4.0]) == d  # seeded: identical every time
    one = stats.describe([0.7])
    assert one["mean"] == 0.7 and one["std"] is None and one["ci_low"] is None
    assert stats.describe([])["mean"] is None


def test_exact_sign_test_and_holm() -> None:
    assert stats.sign_test(9, 1) == pytest.approx(2 * 11 / 1024)
    assert stats.sign_test(5, 5) == 1.0
    assert stats.sign_test(0, 0) is None
    assert stats.holm([0.01, 0.04, None, 0.03]) == [
        pytest.approx(0.03),
        pytest.approx(0.06),
        None,
        pytest.approx(0.06),
    ]


def test_paired_comparison() -> None:
    base = {f"c{i:02d}": 0.5 for i in range(12)}
    better = {k: 0.8 if i < 10 else 0.4 for i, k in enumerate(base)}
    p = stats.paired("ndcg@10", MetricFamily.RETRIEVAL, True, base, better)
    assert p.n_pairs == 12 and (p.wins, p.losses, p.ties) == (10, 2, 0)
    assert p.mean_difference == pytest.approx((10 * 0.3 - 2 * 0.1) / 12, abs=1e-6)
    diffs = np.array([0.3] * 10 + [-0.1] * 2)
    assert p.effect_size_dz == pytest.approx(diffs.mean() / diffs.std(ddof=1), abs=1e-5)
    assert p.sign_test_p == pytest.approx(2 * (1 + 12 + 66) / 4096, abs=1e-6)
    assert p.ci_low is not None and p.ci_low > 0 and p.conclusion == "variant_higher"
    assert [d.case_id for d in p.differences] == sorted(base)

    lower_better = stats.paired("latency_ms", MetricFamily.OPERATIONAL, False, base, better)
    assert (lower_better.wins, lower_better.losses) == (2, 10)  # oriented by direction
    assert lower_better.conclusion == "variant_higher"  # the value is higher; here that is worse

    few = stats.paired(
        "mrr", MetricFamily.RETRIEVAL, True, {"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 1.0}
    )
    assert few.conclusion == "insufficient_cases"  # below MIN_CASES_FOR_CONCLUSION
    same = stats.paired("mrr", MetricFamily.RETRIEVAL, True, base, dict(base))
    assert same.conclusion == "no_detectable_difference" and same.effect_size_dz is None
    descriptive = stats.paired("rerank_mean_shift", MetricFamily.RERANKING, None, base, better)
    assert descriptive.conclusion == "descriptive_only" and descriptive.sign_test_p is None
    disjoint = stats.paired("mrr", MetricFamily.RETRIEVAL, True, {"a": 1.0}, {"b": 1.0})
    assert disjoint.n_pairs == 0 and disjoint.conclusion == "insufficient_cases"
    adjusted = stats.adjust([p, lower_better, descriptive])
    assert adjusted[0].holm_p == pytest.approx(min(1.0, 2 * p.sign_test_p), abs=1e-6)  # type: ignore[operator]
    assert adjusted[2].holm_p is None
