"""The experiment engine end to end: datasets, snapshots, runs, failures, aggregation,
comparisons, provenance and the HTTP API. Real BM25, embedder, cross-encoder and local generator;
the engine itself is never mocked."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_forge.arena import config
from rag_forge.arena.datasets import DEV_CASES, DEV_DOCUMENTS
from rag_forge.arena.metrics import ranked_ids
from rag_forge.domain.arena import (
    Arm,
    CaseView,
    Comparison,
    Experiment,
    ExperimentRun,
    Leaderboard,
    RunCase,
)
from rag_forge.domain.models import GeneratorSpec, RagResponse
from rag_forge.evaluation import retrieval_metrics as rm
from rag_forge.main import create_app
from rag_forge.rag.generation import OnnxCausalLM
from rag_forge.retrieval.embedding import OnnxSentenceEmbedder
from rag_forge.retrieval.rerank import OnnxCrossEncoder

API = "/api/v1"


@pytest.fixture(scope="module")
def lab(
    tmp_path_factory: pytest.TempPathFactory,
    embedder: OnnxSentenceEmbedder,
    cross_encoder: OnnxCrossEncoder,
    generator: OnnxCausalLM,
) -> Iterator[tuple[TestClient, dict[str, Any], dict[str, dict[str, Any]]]]:
    app = create_app(tmp_path_factory.mktemp("arena"), embedder, cross_encoder, generator)
    with TestClient(app) as c:
        r = c.post(f"{API}/benchmarks/development")
        assert r.status_code == 201, r.text
        presets = {a["name"]: a for a in c.get(f"{API}/arena/presets").json()["arms"]}
        yield c, r.json(), presets


def experiment(c: TestClient, body: dict[str, Any]) -> Experiment:
    r = c.post(f"{API}/experiments", json=body)
    assert r.status_code == 201, r.text
    return Experiment.model_validate(r.json())


def run(c: TestClient, experiment_id: str) -> ExperimentRun:
    r = c.post(f"{API}/experiments/{experiment_id}/runs")
    assert r.status_code == 202 and r.json()["status"] == "queued"
    return ExperimentRun.model_validate(c.get(f"{API}/runs/{r.json()['id']}").json())


def cases(c: TestClient, run_id: str, **params: str) -> list[RunCase]:
    r = c.get(f"{API}/runs/{run_id}/cases", params=params)
    return [RunCase.model_validate(x) for x in r.json()]


# --- datasets -----------------------------------------------------------------------------


def test_development_dataset_is_versioned_and_honest(lab: Any) -> None:
    c, ds, _ = lab
    assert ds["source"] == "development" and ds["version"] == 1 and len(ds["cases"]) == 14
    assert {d["name"] for d in c.get(f"{API}/benchmarks").json()} == {"rag-forge-dev"}
    assert c.post(f"{API}/benchmarks/development").json()["id"] == ds["id"]  # idempotent
    unanswerable = next(x for x in ds["cases"] if x["answerable"] is False)
    assert unanswerable["relevant_documents"] == {} and unanswerable["reference_answer"] is None
    comparison = next(x for x in ds["cases"] if "comparison" in x["tags"])
    assert comparison["reference_answer"] is None  # no ground truth is invented
    assert ds["annotation_counts"] == {
        "relevance": 13,
        "reference_answer": 12,
        "answerable": 14,
        "expected_evidence": 13,
    }
    files = {f for f, _ in DEV_DOCUMENTS}
    assert all(set(x.relevant_documents) <= files for x in DEV_CASES)


def test_user_datasets_are_validated_and_versioned(lab: Any) -> None:
    c, dev, _ = lab
    cid = dev["corpus_id"]
    case = {
        "id": "q1",
        "query": "What does error E4021 mean?",
        "relevant_documents": {"error-codes.txt": 1},
    }
    body = {"name": "my-set", "corpus_id": cid, "cases": [case]}
    v1 = c.post(f"{API}/benchmarks", json=body)
    assert v1.status_code == 201 and v1.json()["version"] == 1 and v1.json()["source"] == "user"
    assert c.post(f"{API}/benchmarks", json=body).json()["id"] == v1.json()["id"]  # unchanged
    changed = {**body, "cases": [{**case, "reference_answer": "upload over 25 MB"}]}
    v2 = c.post(f"{API}/benchmarks", json=changed).json()
    assert v2["version"] == 2 and v2["content_hash"] != v1.json()["content_hash"]

    bad = {
        **body,
        "cases": [
            {**case, "relevant_documents": {"missing.txt": 1}, "relevant_chunks": {"chk_nope": 1}}
        ],
    }
    r = c.post(f"{API}/benchmarks", json=bad)
    assert r.status_code == 422
    problems = r.json()["detail"]["problems"]
    assert any("missing.txt" in p for p in problems) and any("chk_nope" in p for p in problems)
    for invalid in (
        {**body, "cases": [case, case]},  # duplicate ids
        {**body, "cases": [{**case, "answerable": False}]},  # unanswerable yet relevant
        {**body, "cases": [{**case, "relevant_documents": {"error-codes.txt": -1}}]},
        {**body, "cases": [{**case, "query": "  "}]},
        {**body, "name": "Not A Slug"},
    ):
        assert c.post(f"{API}/benchmarks", json=invalid).status_code == 422, invalid
    assert c.post(f"{API}/benchmarks", json={**body, "corpus_id": "nope"}).status_code == 404
    taken = {**body, "name": "rag-forge-dev"}
    assert c.post(f"{API}/benchmarks", json=taken).status_code == 422  # a development name


# --- configurations and ablations -----------------------------------------------------------


def test_snapshots_hash_configuration_not_names(lab: Any) -> None:
    c, ds, presets = lab
    bm25 = presets["bm25"]
    renamed = {**bm25, "name": "lexical", "label": "Lexical baseline"}
    deeper = {**bm25, "name": "bm25-k20", "retrieval": {**bm25["retrieval"], "top_k": 20}}
    r = c.post(
        f"{API}/arena/configurations/resolve",
        json={
            "dataset_id": ds["id"],
            "arms": [bm25, renamed, deeper, presets["rag-local"], presets["adaptive"]],
        },
    )
    assert r.status_code == 200, r.text
    arms = r.json()["arms"]
    assert arms[0]["config_hash"] == arms[1]["config_hash"]  # names are not configuration
    assert arms[0]["config_hash"] != arms[2]["config_hash"]
    diff = r.json()["diffs"][1]
    assert diff["factors"] == ["retrieval.top_k"]
    assert {ch["path"] for ch in diff["changes"]} == {"retrieval.top_k", "retrieval_template.top_k"}
    s0, rag, adaptive = arms[0]["snapshot"], arms[3]["snapshot"], arms[4]["snapshot"]
    assert (s0["dataset_version"], s0["corpus_version"]) == (ds["version"], ds["corpus_version"])
    assert s0["retrieval"]["bm25"] == {"k1": 1.2, "b": 0.75} and s0["embedder"] is None
    assert rag["generator"]["revision"] == GeneratorSpec().revision
    assert rag["generator"]["config_hash"] == GeneratorSpec().config_hash()
    assert rag["verifier"] == "lexical-semantic@1" and rag["prompt_template"] == "grounded-qa@1"
    assert rag["reranker"]["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert adaptive["retrieval"] is None and adaptive["router_policy"] == "rules-baseline@1"
    assert adaptive["query_analyzer"] == "heuristic-query-analyzer@1"
    assert s0["metric_versions"]["ndcg"] == 1 and s0["engine_version"] == config.ENGINE_VERSION


def test_experiment_validation(lab: Any) -> None:
    c, ds, presets = lab
    bm25, dense = presets["bm25"], presets["dense"]
    base = {"name": "v", "dataset_id": ds["id"], "arms": [bm25, dense]}
    identical = {
        **base,
        "arms": [bm25, {**bm25, "name": "bm25-again"}],
        "ablations": [{"baseline": "bm25", "variant": "bm25-again", "factor": "nothing"}],
    }
    r = c.post(f"{API}/experiments", json=identical)
    assert r.status_code == 422 and "nothing varies" in r.json()["detail"]
    for invalid in (
        {**base, "arms": [bm25, bm25]},
        {**base, "ablations": [{"baseline": "bm25", "variant": "ghost", "factor": "x"}]},
        {**base, "ablations": [{"baseline": "bm25", "variant": "bm25", "factor": "x"}]},
        {**base, "arms": []},
        {**base, "metrics": {"ks": [0]}},
        {**base, "metrics": {"metrics": ["bleu"]}},
        {**base, "limits": {"concurrency": 9}},
        {
            **base,
            "arms": [
                {
                    "name": "x",
                    "retrieval": {"top_k": 20, "rerank": {"enabled": True, "candidate_k": 5}},
                }
            ],
        },
        {**base, "arms": [{"name": "x", "pipeline": "retrieval", "generation": {}}]},
    ):
        assert c.post(f"{API}/experiments", json=invalid).status_code == 422, invalid
    ghost = {**presets["rag-local"], "generation": {"generator": "gpt-imaginary"}}
    r = c.post(f"{API}/experiments", json={**base, "arms": [ghost]})
    assert r.status_code == 501 and "extractive-baseline" in r.json()["detail"]["message"]
    r = c.post(
        f"{API}/experiments", json={**base, "arms": [{**bm25, "retrieval": {"strategy": "graph"}}]}
    )
    assert r.status_code == 501
    assert c.post(f"{API}/experiments", json={**base, "dataset_id": "bds_nope"}).status_code == 404
    e = experiment(
        c, {**base, "ablations": [{"baseline": "bm25", "variant": "dense", "factor": "strategy"}]}
    )
    (ab,) = e.ablations
    assert ab.factors == ["retrieval.strategy"] and ab.single_factor
    assert any(ch.path == "retrieval.strategy" for ch in ab.changes)
    multi = experiment(
        c,
        {
            **base,
            "arms": [presets["hybrid-rrf-rerank"], presets["adaptive"]],
            "ablations": [
                {"baseline": "hybrid-rrf-rerank", "variant": "adaptive", "factor": "routing"}
            ],
        },
    )
    (m,) = multi.ablations
    assert not m.single_factor  # stated, not hidden
    assert m.factors == ["retrieval.mode", "retrieval.rerank", "retrieval.strategy"]


# --- runs ------------------------------------------------------------------------------


def test_partial_failures_are_recorded_not_dropped(
    tmp_path: Path, cross_encoder: OnnxCrossEncoder
) -> None:
    """Dense arms on a corpus without a dense index fail every case: recorded, typed, counted."""
    with TestClient(create_app(tmp_path, reranker=cross_encoder)) as c:
        cid = c.post(f"{API}/corpora", json={"name": "nodense"}).json()["corpus"]["id"]
        c.post(
            f"{API}/corpora/{cid}/documents",
            files=[("files", (f, t.encode())) for f, t in DEV_DOCUMENTS],
        )
        ds = c.post(
            f"{API}/benchmarks",
            json={
                "name": "small",
                "corpus_id": cid,
                "cases": [x.model_dump(mode="json") for x in DEV_CASES[:4]],
            },
        ).json()
        e = experiment(
            c,
            {
                "name": "fail",
                "dataset_id": ds["id"],
                "arms": [{"name": "bm25"}, {"name": "dense", "retrieval": {"strategy": "dense"}}],
            },
        )
        r = run(c, e.id)
        assert r.status.value == "partial" and (r.total, r.completed, r.failed) == (8, 8, 4)
        failed = cases(c, r.id, status="failed")
        assert len(failed) == 4 and {x.arm for x in failed} == {"dense"}
        assert all(x.error_type == "DenseIndexNotReadyError" and x.error for x in failed)
        assert all([m.metric for m in x.metrics] == ["failed"] for x in failed)  # absent, not 0
        dense = next(s for s in r.summaries if s.arm == "dense")
        assert (dense.cases, dense.succeeded, dense.failed) == (4, 0, 4)
        assert dense.failure_types == {"DenseIndexNotReadyError": 4}
        by = {m.metric: m for m in dense.metrics}
        assert by["failed"].mean == 1.0 and "ndcg@10" not in by
        bm25 = next(s for s in r.summaries if s.arm == "bm25")
        assert (
            bm25.failed == 0 and next(m for m in bm25.metrics if m.metric == "failed").mean == 0.0
        )
        cmp = Comparison.model_validate(
            c.get(
                f"{API}/runs/{r.id}/compare", params={"baseline": "bm25", "variant": "dense"}
            ).json()
        )
        assert cmp.comparable and any("failed in at least one arm" in w for w in cmp.warnings)
        rate = next(m for m in cmp.metrics if m.metric == "failed")
        assert rate.n_pairs == 4 and rate.mean_difference == 1.0
        assert all(m.n_pairs == 0 for m in cmp.metrics if m.metric == "mrr")


def test_run_lifecycle_aggregation_and_provenance(lab: Any) -> None:
    c, ds, p = lab
    body = {
        "name": "lifecycle",
        "hypothesis": "reranking changes rankings on this set",
        "dataset_id": ds["id"],
        "arms": [p["bm25"], p["bm25-rerank"], p["rag-extractive"]],
        "ablations": [{"baseline": "bm25", "variant": "bm25-rerank", "factor": "reranking"}],
        "limits": {"concurrency": 2},
    }
    e = experiment(c, body)
    r = run(c, e.id)
    assert r.status.value == "completed" and (r.total, r.completed, r.failed) == (42, 42, 0)
    assert r.dataset_hash == ds["content_hash"] and r.snapshot_hashes == e.snapshot_hashes
    assert r.metric_registry_version == "arena-metrics@1"
    assert r.stats_method == "paired-bootstrap-sign@1" and r.environment.python_version
    assert [x["id"] for x in c.get(f"{API}/experiments/{e.id}/runs").json()] == [r.id]
    all_cases = cases(c, r.id)
    assert len(all_cases) == 42 and [x.case_id for x in all_cases[:3]] == [DEV_CASES[0].id] * 3

    # every stored number recomputes from the stored ranking
    for rc in cases(c, r.id, arm="bm25"):
        case = next(x for x in DEV_CASES if x.id == rc.case_id)
        mrr = next(m for m in rc.metrics if m.metric == "mrr")
        if case.relevant_documents and any(g > 0 for g in case.relevant_documents.values()):
            assert mrr.value == round(
                rm.reciprocal_rank(ranked_ids(rc.ranking, "document"), case.relevant_documents), 6
            )
        else:
            assert mrr.value is None and mrr.skipped
        assert rc.config_hash == e.snapshot_hashes["bm25"] and rc.retrieval_configuration_hash
        assert rc.artifact_id and rc.timings_ms["total"] > 0

    # aggregates equal the mean of the per-case values, over the cases that define them
    bm25 = next(s for s in r.summaries if s.arm == "bm25")
    vals = [
        m.value
        for x in cases(c, r.id, arm="bm25")
        for m in x.metrics
        if m.metric == "ndcg@10" and m.value is not None
    ]
    nd = next(m for m in bm25.metrics if m.metric == "ndcg@10")
    assert (
        nd.n == len(vals) == 13
        and nd.skipped == 1
        and nd.mean == pytest.approx(sum(vals) / len(vals), abs=1e-6)
    )
    assert nd.mean is not None and nd.ci_low is not None and nd.ci_high is not None
    assert nd.ci_low <= nd.mean <= nd.ci_high
    assert bm25.skipped_reasons["ndcg@10"] == "no relevance judgements for this case"

    # reranked arm: models and rank movement recorded
    rr = cases(c, r.id, arm="bm25-rerank")
    assert all(x.models["reranker"].startswith("cross-encoder/ms-marco-MiniLM-L-6-v2@") for x in rr)
    assert any(i.upstream_rank is not None for x in rr for i in x.ranking)

    # grounded RAG arm: evidence, answer, grounding and abstention measured
    rag = {x.case_id: x for x in cases(c, r.id, arm="rag-extractive")}
    wc = rag["world-cup-2018"]
    assert wc.answer_status is not None and wc.answer_status.value in (
        "abstained",
        "insufficient_evidence",
    )
    assert next(m for m in wc.metrics if m.metric == "abstention_accuracy").value == 1.0
    assert rag["error-e4021"].evidence and rag["error-e4021"].grounding is not None
    trace = c.get(f"{API}/artifacts/{rag['error-e4021'].artifact_id}").json()
    assert trace["kind"] == "rag_trace"
    full = RagResponse.model_validate(trace["rag"])
    assert full.answer is not None and full.answer.text == rag["error-e4021"].answer
    assert full.provenance.configuration.retrieval.corpus_version == ds["corpus_version"]

    # case explorer, leaderboard and comparison
    view = CaseView.model_validate(c.get(f"{API}/runs/{r.id}/cases/error-e4021").json())
    assert [x.arm for x in view.results] == ["bm25", "bm25-rerank", "rag-extractive"]
    assert view.case.relevant_documents == {"error-codes.txt": 2}
    lb = Leaderboard.model_validate(
        c.get(f"{API}/runs/{r.id}/leaderboard", params={"metric": "mrr"}).json()
    )
    assert {row.arm for row in lb.rows} == {"bm25", "bm25-rerank", "rag-extractive"}
    means = [row.summary.mean or 0.0 for row in lb.rows if row.summary]
    assert means == sorted(means, reverse=True)
    lat = Leaderboard.model_validate(
        c.get(f"{API}/runs/{r.id}/leaderboard", params={"metric": "latency_ms"}).json()
    )
    latencies = [row.summary.mean or 0.0 for row in lat.rows if row.summary]
    assert latencies == sorted(latencies)  # lower is better: fastest first
    desc = Leaderboard.model_validate(
        c.get(f"{API}/runs/{r.id}/leaderboard", params={"metric": "prompt_tokens"}).json()
    )
    assert all(row.position is None for row in desc.rows)  # descriptive: listed, not ranked
    cmp = Comparison.model_validate(
        c.get(
            f"{API}/runs/{r.id}/compare", params={"baseline": "bm25", "variant": "bm25-rerank"}
        ).json()
    )
    assert (
        cmp.comparable and cmp.factor == "reranking" and cmp.method.name == "paired-bootstrap-sign"
    )
    assert any("development benchmark" in w for w in cmp.warnings)
    assert {ch.path.split(".")[0] for ch in cmp.changes} >= {"retrieval", "reranker"}
    ndcg = next(m for m in cmp.metrics if m.metric == "ndcg@10")
    assert ndcg.n_pairs == 13 and len(ndcg.differences) == 13
    assert ndcg.mean_difference == pytest.approx(
        sum(d.difference for d in ndcg.differences) / 13, abs=1e-6
    )
    assert ndcg.conclusion in ("no_detectable_difference", "variant_higher", "variant_lower")
    for err, params in (
        (404, {"baseline": "bm25", "variant": "ghost"}),
        (422, {"baseline": "bm25"}),
    ):
        assert c.get(f"{API}/runs/{r.id}/compare", params=params).status_code == err
    assert c.get(f"{API}/runs/{r.id}/leaderboard", params={"metric": "bleu"}).status_code == 404
    assert c.get(f"{API}/runs/{r.id}/cases/nope").status_code == 404
    assert c.get(f"{API}/artifacts/art_nope").status_code == 404


def test_incomparable_runs_are_refused(lab: Any) -> None:
    c, ds, p = lab
    other = c.post(
        f"{API}/benchmarks",
        json={
            "name": "two-cases",
            "corpus_id": ds["corpus_id"],
            "cases": [x.model_dump(mode="json") for x in DEV_CASES[:2]],
        },
    ).json()
    a = run(
        c,
        experiment(
            c,
            {"name": "a", "dataset_id": ds["id"], "arms": [p["bm25"]], "limits": {"max_cases": 3}},
        ).id,
    )
    b = run(c, experiment(c, {"name": "b", "dataset_id": other["id"], "arms": [p["bm25"]]}).id)
    assert a.case_ids == [x.id for x in DEV_CASES[:3]]  # max_cases takes the first N, in order
    cmp = Comparison.model_validate(
        c.get(
            f"{API}/arena/compare",
            params={
                "baseline_run": a.id,
                "baseline_arm": "bm25",
                "variant_run": b.id,
                "variant_arm": "bm25",
            },
        ).json()
    )
    assert not cmp.comparable and cmp.metrics == []
    assert "different benchmark datasets or dataset versions" in cmp.issues
    k5 = experiment(
        c,
        {
            "name": "k",
            "dataset_id": ds["id"],
            "arms": [p["bm25"]],
            "metrics": {"ks": [5]},
            "limits": {"max_cases": 3},
        },
    )
    k = run(c, k5.id)
    cmp = Comparison.model_validate(
        c.get(
            f"{API}/arena/compare",
            params={
                "baseline_run": a.id,
                "baseline_arm": "bm25",
                "variant_run": k.id,
                "variant_arm": "bm25",
            },
        ).json()
    )
    assert not cmp.comparable and "different metric definitions or k values" in cmp.issues
    same = Comparison.model_validate(
        c.get(
            f"{API}/arena/compare",
            params={
                "baseline_run": a.id,
                "baseline_arm": "bm25",
                "variant_run": a.id,
                "variant_arm": "bm25",
            },
        ).json()
    )
    assert same.comparable and "the two configurations are identical" in same.warnings
    assert all(
        m.mean_difference in (None, 0.0) or m.metric.endswith("_ms") or m.metric == "latency_ms"
        for m in same.metrics
    )


def test_unfinished_runs_and_overview(lab: Any) -> None:
    c, ds, p = lab
    e = experiment(c, {"name": "queued", "dataset_id": ds["id"], "arms": [p["bm25"]]})
    queued = c.app.state.arena.create_run(e.id)
    assert c.get(f"{API}/runs/{queued.id}/leaderboard").status_code == 409
    cmp = c.get(
        f"{API}/runs/{queued.id}/compare", params={"baseline": "bm25", "variant": "bm25"}
    ).json()
    assert not cmp["comparable"] and f"run {queued.id} is queued" in cmp["issues"]
    ov = c.get(f"{API}/arena/overview").json()
    assert ov["experiments"] >= 1 and ov["stats_method"]["name"] == "paired-bootstrap-sign"
    assert any(r["status"] == "queued" for r in ov["runs"])
    assert {d["name"] for d in c.get(f"{API}/arena/metrics").json()} >= {"recall", "ndcg", "failed"}


def test_integration_real_configurations(lab: Any) -> None:
    """Development benchmark -> six real configurations (fixed, adaptive, reranked, grounded
    RAG with the local model) -> stored run -> metrics -> paired comparison."""
    c, ds, p = lab
    arms = [
        p[n] for n in ("bm25", "dense", "hybrid-rrf", "adaptive", "hybrid-rrf-rerank", "rag-local")
    ]
    e = experiment(
        c,
        {
            "name": "integration",
            "dataset_id": ds["id"],
            "arms": arms,
            "ablations": [
                {"baseline": "hybrid-rrf", "variant": "hybrid-rrf-rerank", "factor": "reranking"},
                {
                    "baseline": "hybrid-rrf-rerank",
                    "variant": "adaptive",
                    "factor": "fixed vs adaptive",
                },
            ],
            "limits": {"max_cases": 6},
        },
    )
    r = run(c, e.id)
    assert r.status.value == "completed" and r.completed == r.total == 36
    by_arm = {s.arm: s for s in r.summaries}
    for name in ("bm25", "dense", "hybrid-rrf", "adaptive", "hybrid-rrf-rerank"):
        m = {x.metric: x for x in by_arm[name].metrics}
        assert m["recall@10"].n == 6 and m["mrr"].mean is not None and 0 <= m["mrr"].mean <= 1
    adaptive = cases(c, r.id, arm="adaptive")
    assert all(x.route in ("sparse", "dense", "hybrid_rrf", "hybrid_weighted") for x in adaptive)
    rag = cases(c, r.id, arm="rag-local")
    assert all(x.models["generator"].startswith(f"{GeneratorSpec().model}@") for x in rag)
    assert all("generation" in x.timings_ms and x.tokens.get("prompt", 0) > 0 for x in rag)
    f1 = next(m for m in by_arm["rag-local"].metrics if m.metric == "answer_token_f1")
    assert f1.n + f1.skipped == 6
    cmp = Comparison.model_validate(
        c.get(
            f"{API}/runs/{r.id}/compare",
            params={"baseline": "hybrid-rrf", "variant": "hybrid-rrf-rerank"},
        ).json()
    )
    assert cmp.comparable and cmp.factor == "reranking"
    ndcg = next(m for m in cmp.metrics if m.metric == "ndcg@10")
    assert ndcg.n_pairs == 6 and ndcg.conclusion == "insufficient_cases"  # 6 < 10: no claim made
    snap = next(s for s in e.snapshots if s.arm == "rag-local")
    assert snap.generator is not None and snap.generator.revision == GeneratorSpec().revision
    assert Arm.model_validate(p["rag-local"]).pipeline.value == "rag"
