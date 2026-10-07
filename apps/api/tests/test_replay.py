"""Replay, reproducibility manifests and research exports, on real runs with the real models.
The engine and the replay service are never mocked."""

import csv
import io
import json
import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_forge.arena.replay import classify, compare_stages
from rag_forge.domain.arena import ExperimentRun
from rag_forge.domain.models import ContentOrigin, PipelineStage, RuntimeSnapshot
from rag_forge.domain.replay import Replay, ReplayOutcome, ReproducibilityManifest
from rag_forge.main import create_app
from rag_forge.provenance.runtime import settings
from rag_forge.rag.generation import OnnxCausalLM
from rag_forge.retrieval.embedding import OnnxSentenceEmbedder
from rag_forge.retrieval.rerank import OnnxCrossEncoder

API = "/api/v1"
SECRET = "sk-test-not-a-real-key-0123456789"


def stage(name: str, h: str, deterministic: bool = True) -> PipelineStage:
    return PipelineStage(
        stage=name,
        hash=h,
        config_hash=None,
        latency_ms=None,
        origin=ContentOrigin.MEASURED,
        deterministic=deterministic,
        detail="",
    )


def test_stage_comparison_and_classification() -> None:
    rec = [stage("retrieval", "a"), stage("generation", "g", False), stage("grounding", "x")]
    same = compare_stages(rec, list(rec))
    assert classify(same) == (ReplayOutcome.EXACT, None, "every recorded stage hash matched")

    generated = [stage("retrieval", "a"), stage("generation", "h", False), stage("grounding", "y")]
    outcome, first, reason = classify(compare_stages(rec, generated))
    assert outcome is ReplayOutcome.EQUIVALENT and first == "generation"
    assert "not deterministic" in reason

    moved = [stage("retrieval", "b"), stage("generation", "g", False), stage("grounding", "x")]
    outcome, first, _ = classify(compare_stages(rec, moved))
    assert outcome is ReplayOutcome.DIVERGED and first == "retrieval"

    missing = compare_stages(rec, rec[:2])
    assert missing[-1].match is None
    assert classify(missing)[0] is ReplayOutcome.DIVERGED


def test_settings_never_contain_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_FORGE_OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("RAG_FORGE_OPENAI_BASE_URL", "http://localhost:9")
    s = settings()
    assert s["RAG_FORGE_OPENAI_API_KEY"] == "<redacted>"
    assert s["RAG_FORGE_OPENAI_BASE_URL"] == "http://localhost:9"
    assert SECRET not in json.dumps(s)


@pytest.fixture(scope="module")
def replayed(
    tmp_path_factory: pytest.TempPathFactory,
    embedder: OnnxSentenceEmbedder,
    cross_encoder: OnnxCrossEncoder,
    generator: OnnxCausalLM,
) -> Iterator[tuple[TestClient, ExperimentRun, Replay]]:
    """A real run (BM25, adaptive router, grounded RAG with the local model, extractive RAG) over
    three development cases, recorded with a secret in the environment, then replayed."""
    mp = pytest.MonkeyPatch()
    mp.setenv("RAG_FORGE_OPENAI_API_KEY", SECRET)  # present, never registered, never recorded
    app = create_app(tmp_path_factory.mktemp("replay"), embedder, cross_encoder, generator)
    with TestClient(app) as c:
        ds = c.post(f"{API}/benchmarks/development").json()
        presets = {a["name"]: a for a in c.get(f"{API}/arena/presets").json()["arms"]}
        body = {
            "name": "replay check",
            "dataset_id": ds["id"],
            "arms": [presets[n] for n in ("bm25", "adaptive", "rag-local", "rag-extractive")],
            "ablations": [
                {"baseline": "rag-extractive", "variant": "rag-local", "factor": "generator"}
            ],
            "limits": {"max_cases": 3},
        }
        exp = c.post(f"{API}/experiments", json=body).json()
        run_id = c.post(f"{API}/experiments/{exp['id']}/runs").json()["id"]
        run = ExperimentRun.model_validate(c.get(f"{API}/runs/{run_id}").json())
        assert run.status == "completed", run.error
        r = c.post(f"{API}/runs/{run_id}/replays", json={})
        assert r.status_code == 202, r.text
        replay = Replay.model_validate(c.get(f"{API}/replays/{r.json()['id']}").json())
        yield c, run, replay
    mp.undo()


def test_replay_reproduces_every_stage_of_a_real_run(replayed: Any) -> None:
    c, run, replay = replayed
    assert replay.status == "completed" and replay.total == replay.completed == 12
    assert all(ch.replayable and ch.changes == [] for ch in replay.checks)
    # same process, same models, greedy decoding: every stage of every case reproduces
    assert replay.outcomes == {ReplayOutcome.EXACT: 12}, [
        (x.arm, x.case_id, x.reason) for x in replay.cases if x.outcome != "exact"
    ]
    by = {(x.arm, x.case_id): x for x in replay.cases}
    rag = by[("rag-local", run.case_ids[0])]
    assert [s.stage for s in rag.stages] == [
        "query", "retrieval", "reranking", "evidence_selection", "context", "generation",
        "claims", "grounding",
    ]  # fmt: skip
    assert all(s.match for s in rag.stages) and rag.metrics_compared > 10
    adaptive = by[("adaptive", run.case_ids[0])]
    assert {"query_analysis", "router_decision"} <= {s.stage for s in adaptive.stages}
    # the replay's own traces are stored and name the replay
    trace = c.get(f"{API}/artifacts/{rag.replayed_artifact_id}").json()
    assert trace["replay_id"] == replay.id and trace["run_id"] == run.id
    assert trace["rag"]["provenance"]["chain"][5]["stage"] == "generation"
    assert rag.replayed_artifact_id != rag.recorded_artifact_id
    checks = {x.arm: x for x in replay.checks}
    assert "deterministic" in (checks["rag-extractive"].generation or "")
    assert "greedy decoding" in (checks["rag-local"].generation or "")
    assert checks["bm25"].generation is None
    assert replay.id in [r["id"] for r in c.get(f"{API}/runs/{run.id}/replays").json()]


def test_unavailable_components_make_cases_not_replayable(replayed: Any) -> None:
    c, run, _ = replayed
    rag = c.app.state.rag
    removed = rag.generators.pop("extractive-baseline")
    try:
        checks = {x["arm"]: x for x in c.get(f"{API}/runs/{run.id}/replayability").json()}
        assert checks["rag-extractive"]["replayable"] is False
        assert "extractive-baseline" in checks["rag-extractive"]["reason"]
        assert checks["bm25"]["replayable"] is True
        r = c.post(f"{API}/runs/{run.id}/replays", json={"arms": ["rag-extractive", "bm25"]})
        replay = Replay.model_validate(c.get(f"{API}/replays/{r.json()['id']}").json())
    finally:
        rag.generators["extractive-baseline"] = removed
    outcomes = {(x.arm, x.outcome) for x in replay.cases}
    assert outcomes == {("rag-extractive", "not_replayable"), ("bm25", "exact")}
    assert all(x.replayed_artifact_id is None for x in replay.cases if x.arm == "rag-extractive")


def test_replay_requests_are_validated(replayed: Any) -> None:
    c, run, _ = replayed
    r = c.post(f"{API}/runs/{run.id}/replays", json={"case_ids": ["nope"]})
    assert r.status_code == 422 and "nope" in r.text
    queued = c.app.state.arena.create_run(run.experiment_id)
    assert c.post(f"{API}/runs/{queued.id}/replays", json={}).status_code == 409
    assert c.get(f"{API}/runs/{queued.id}/report.md").status_code == 409
    assert c.get(f"{API}/replays/rpl_missing").status_code == 404


def test_manifest_records_what_produced_the_run(replayed: Any) -> None:
    c, run, replay = replayed
    raw = c.get(f"{API}/runs/{run.id}/manifest").text
    assert SECRET not in raw
    m = ReproducibilityManifest.model_validate_json(raw)
    assert run.runtime is not None and m.runtime is not None and m.runtime == run.runtime
    assert m.runtime.settings["RAG_FORGE_OPENAI_API_KEY"] == "<redacted>"
    assert "apps/api/uv.lock" in m.runtime.lockfiles and m.runtime.uv_version
    assert {x.role for x in m.models} == {"embedder", "reranker", "generator", "verifier-embedder"}
    qwen = next(x for x in m.models if x.model == "onnx-community/Qwen2.5-0.5B-Instruct")
    weights = next(f for f in qwen.files if f.path.endswith(".onnx"))
    assert weights.source == "lfs-blob-id" and re.fullmatch(r"[0-9a-f]{64}", weights.sha256 or "")
    assert qwen.revision == "cc5cc01a65cc3ff17bdb73a7de33d879f62599b0"
    extractive = next(x for x in m.models if x.provider == "extractive")
    assert extractive.files == []  # nothing to hash: no model files
    arms = {a.arm: a for a in m.arms}
    assert arms["rag-local"].generation_deterministic is True
    assert arms["rag-local"].context_hashes == 3 and arms["rag-local"].temperature == 0
    assert arms["adaptive"].routing_hash and arms["bm25"].retrieval_configuration_hash
    assert m.seeds["bootstrap"] == 20261007
    assert len(m.artifacts) == 12  # the run's traces; replay traces are not the run's
    assert replay.id in {r["id"] for r in m.replays} and m.current["differences"] == []
    md = c.get(f"{API}/runs/{run.id}/manifest.md")
    assert md.headers["content-type"].startswith("text/markdown")
    assert "## Models" in md.text and SECRET not in md.text


def test_exports(replayed: Any) -> None:
    c, run, _ = replayed
    metrics = list(
        csv.DictReader(io.StringIO(c.get(f"{API}/runs/{run.id}/export/metrics.csv").text))
    )
    assert {r["arm"] for r in metrics} == set(run.arms)
    assert len(metrics) == sum(len(s.metrics) for s in run.summaries)
    ndcg = next(r for r in metrics if r["arm"] == "bm25" and r["metric"] == "ndcg@10")
    assert float(ndcg["mean"]) == next(
        m.mean for m in run.summaries[0].metrics if m.metric == "ndcg@10"
    )
    rows = list(csv.DictReader(io.StringIO(c.get(f"{API}/runs/{run.id}/export/cases.csv").text)))
    assert {(r["arm"], r["case_id"]) for r in rows} == {
        (a, cid) for a in run.arms for cid in run.case_ids
    }
    skipped = [r for r in rows if r["skipped"]]
    assert skipped and all(r["value"] == "" for r in skipped)  # skipped is never 0

    bundle = c.get(f"{API}/runs/{run.id}/export.json")
    assert "attachment" in bundle.headers["content-disposition"]
    data = bundle.json()
    assert data["format"] == "rag-forge-export@1" and len(data["cases"]) == 12
    assert data["comparisons"][0]["baseline"]["arm"] == "rag-extractive"
    assert SECRET not in bundle.text

    report = c.get(f"{API}/runs/{run.id}/report.md").text
    for heading in (
        "## Configurations",
        "## Retrieval (measured)",
        "## Generation (automatic proxies)",
        "## Latency and failures",
        "## Statistical comparisons",
        "### rag-extractive → rag-local (generator)",
        "## Per-case results",
        "## Interpretation (restated from the recorded conclusions)",
        "### Qualitative interpretation",
        "## Limitations",
        "## Reproducibility",
    ):
        assert heading in report, heading
    assert "Development benchmark" in report and "too few paired cases" in report
    assert "higher mean" not in report  # three cases: no conclusion may be drawn
    assert run.snapshot_hashes["rag-local"][:16] in report and SECRET not in report


def test_records_from_before_runtime_fields_still_load() -> None:
    """Fields added to stored models must default, or older rows stop loading."""
    old: dict[str, object] = {
        "node_version": None,
        "uv_version": None,
        "git_dirty": None,
        "lockfiles": {},
        "packages": {},
        "settings": {},
        "models": [],
    }
    assert RuntimeSnapshot.model_validate(old).git_commit is None
    run_fields = set(ExperimentRun.model_fields)
    assert ExperimentRun.model_fields["runtime"].default is None and "runtime" in run_fields
