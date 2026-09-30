"""Fusion arithmetic, checked by hand against the formulas."""

import pytest
from pydantic import ValidationError

from rag_forge.domain.models import (
    FusionMethod,
    HybridParams,
    RetrievalRequest,
    RetrievalStrategy,
)
from rag_forge.retrieval.base import Candidate
from rag_forge.retrieval.fusion import ReciprocalRankFusion, WeightedScoreFusion, min_max

S, D = RetrievalStrategy.SPARSE, RetrievalStrategy.DENSE


def ranked(*pairs: tuple[str, float]) -> list[Candidate]:
    return [Candidate(cid, score) for cid, score in pairs]


def ids(fused: list) -> list[str]:  # type: ignore[type-arg]
    return [f.chunk_id for f in fused]


def test_rrf_scores_merges_duplicates_and_keeps_raw_values() -> None:
    lists = {
        S: ranked(("a", 9.1), ("b", 4.0), ("c", 1.5)),
        D: ranked(("c", 0.83), ("a", 0.80), ("d", 0.7)),
    }
    fused = ReciprocalRankFusion(k=60).fuse(lists, top_k=10)
    assert ids(fused) == ["a", "c", "b", "d"]  # a: 1/61+1/62 > c: 1/63+1/61
    a = fused[0].detail
    assert a.method is FusionMethod.RRF
    assert a.score == pytest.approx(1 / 61 + 1 / 62)
    comps = {c.strategy: c for c in a.components}
    assert (comps[S].rank, comps[S].score) == (1, 9.1)
    assert comps[S].contribution == pytest.approx(1 / 61)
    assert (comps[D].rank, comps[D].score) == (2, 0.80)
    assert comps[D].contribution == pytest.approx(1 / 62)
    b = {c.strategy: c for c in fused[2].detail.components}
    assert (b[D].rank, b[D].score, b[D].contribution) == (None, None, 0.0)  # absent from dense
    assert len(fused) == 4  # each chunk once, even when both lists contain it


def test_rrf_ignores_duplicate_within_a_list_and_truncates() -> None:
    fused = ReciprocalRankFusion().fuse(
        {S: ranked(("a", 3), ("a", 2), ("b", 1)), D: ranked(("b", 1))}, 1
    )
    assert ids(fused) == ["b"]  # b: 1/62 + 1/61 beats a: 1/61
    sparse = next(c for c in fused[0].detail.components if c.strategy is S)
    assert sparse.rank == 3  # original position kept: the retriever ranked b third


def test_rrf_k_changes_the_ranking_and_ties_break_by_chunk_id() -> None:
    lists = {S: ranked(("x", 5), ("p", 4), ("y", 3)), D: ranked(("q", 0.9), ("r", 0.8), ("y", 0.7))}
    assert ids(ReciprocalRankFusion(k=60).fuse(lists, 2)) == ["y", "q"]  # 2/63 beats 1/61
    # k=1: x = 1/2, q = 1/2, y = 1/4 + 1/4 = 1/2 -> a three-way tie, resolved by id
    assert ids(ReciprocalRankFusion(k=1).fuse(lists, 3)) == ["q", "x", "y"]
    with pytest.raises(ValueError, match="at least 1"):
        ReciprocalRankFusion(k=0)


def test_rrf_tie_is_not_broken_by_float_summation_order() -> None:
    # a = 1/61 + 1/62 and b = 1/62 + 1/61: equal in exact arithmetic
    fused = ReciprocalRankFusion().fuse(
        {S: ranked(("b", 2), ("a", 1)), D: ranked(("a", 2), ("b", 1))}, 2
    )
    assert ids(fused) == ["a", "b"]
    assert [ids(ReciprocalRankFusion().fuse({D: ranked(("z", 1)), S: ranked(("m", 1))}, 2))] == [
        ["m", "z"]
    ]


def test_min_max_normalisation() -> None:
    assert min_max(ranked(("a", 10.0), ("b", 5.0), ("c", 0.0))) == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert min_max(ranked(("a", 0.42), ("b", 0.42))) == {"a": 1.0, "b": 1.0}  # constant list
    assert min_max([]) == {}


def test_weighted_fusion_uses_normalised_scores_and_keeps_raw_ones() -> None:
    lists = {S: ranked(("a", 10.0), ("b", 0.0)), D: ranked(("b", 0.9), ("c", 0.5))}
    fused = WeightedScoreFusion({S: 0.7, D: 0.3}).fuse(lists, 10)
    assert ids(fused) == ["a", "b", "c"]
    scores = {f.chunk_id: f.detail.score for f in fused}
    assert scores == {"a": pytest.approx(0.7), "b": pytest.approx(0.3), "c": pytest.approx(0.0)}
    b = {c.strategy: c for c in fused[1].detail.components}
    assert (b[S].score, b[S].normalized_score, b[S].contribution) == (0.0, 0.0, 0.0)
    assert (b[D].score, b[D].normalized_score) == (0.9, 1.0)
    assert b[D].contribution == pytest.approx(0.3)
    # the same inputs with the weights flipped reorder the result
    assert ids(WeightedScoreFusion({S: 0.2, D: 0.8}).fuse(lists, 10)) == ["b", "a", "c"]


@pytest.mark.parametrize("weights", [{S: 0.6, D: 0.6}, {S: -0.1, D: 1.1}, {S: 1.2, D: -0.2}])
def test_weight_validation(weights: dict[RetrievalStrategy, float]) -> None:
    with pytest.raises(ValueError, match="weights"):
        WeightedScoreFusion(weights)
    with pytest.raises(ValidationError, match="weight"):
        HybridParams(fusion=FusionMethod.WEIGHTED, weights=weights)


def test_hybrid_params_validation() -> None:
    with pytest.raises(ValidationError, match="two distinct"):
        HybridParams(retrievers=[S, S])
    with pytest.raises(ValidationError, match="must be among"):
        HybridParams(retrievers=[S, RetrievalStrategy.GRAPH])
    with pytest.raises(ValidationError, match="exactly the component"):
        HybridParams(fusion=FusionMethod.WEIGHTED, weights={S: 1.0})
    with pytest.raises(ValidationError, match="candidate_k must be at least top_k"):
        RetrievalRequest(query="q", strategy=RetrievalStrategy.HYBRID, top_k=20,
                         hybrid=HybridParams(candidate_k=10))  # fmt: skip
    RetrievalRequest(query="q", top_k=20, hybrid=HybridParams(candidate_k=10))  # sparse: irrelevant


def test_ignored_parameters_do_not_change_the_effective_configuration() -> None:
    a = HybridParams(weights={S: 0.9, D: 0.1}).effective()
    b = HybridParams().effective()
    assert a == b and a.weights == {}
    w1 = HybridParams(fusion=FusionMethod.WEIGHTED, rrf_k=5).effective()
    w2 = HybridParams(fusion=FusionMethod.WEIGHTED).effective()
    assert w1 == w2
