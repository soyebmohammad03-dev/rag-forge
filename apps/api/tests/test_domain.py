import math

import pytest
from pydantic import ValidationError

from rag_forge.domain.models import ContentOrigin, Metric
from rag_forge.evaluation import retrieval_metrics as rm


def test_models_are_immutable_and_strict() -> None:
    m = Metric(name="recall", value=1.0, k=5)
    assert m.origin is ContentOrigin.MEASURED
    with pytest.raises(ValidationError):
        m.value = 0.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Metric(name="x", value=1.0, unexpected=True)  # type: ignore[call-arg]


RANKED = ["d1", "d2", "d3", "d4"]


def test_binary_metrics() -> None:
    rel = {"d2": 1.0, "d4": 1.0, "d9": 1.0}
    assert rm.recall_at_k(RANKED, rel, 2) == pytest.approx(1 / 3)
    assert rm.precision_at_k(RANKED, rel, 4) == 0.5
    assert rm.reciprocal_rank(RANKED, rel) == 0.5
    assert rm.reciprocal_rank(RANKED, {"zz": 1.0}) == 0.0


def test_ndcg_matches_hand_computation() -> None:
    rel = {"d1": 1.0, "d3": 2.0}
    dcg = 1 / math.log2(2) + 3 / math.log2(4)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert rm.ndcg_at_k(RANKED, rel, 3) == pytest.approx(dcg / idcg)
    assert rm.ndcg_at_k(["d3", "d1"], rel, 2) == pytest.approx(1.0)


def test_metrics_reject_undefined_inputs() -> None:
    with pytest.raises(ValueError):
        rm.recall_at_k(RANKED, {}, 5)
    with pytest.raises(ValueError):
        rm.precision_at_k(RANKED, {"d1": 1.0}, 0)
