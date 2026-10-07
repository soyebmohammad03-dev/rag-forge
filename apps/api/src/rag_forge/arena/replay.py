"""Replay recorded Arena runs and say exactly how far each case reproduced.

For every requested case x arm the replay re-executes the run's own arm on the pinned corpus
version (the same `ArenaEngine.execute` the run used), stores the new trace, and compares the
output hash of every pipeline stage (query, query analysis, router decision, retrieval,
reranking, evidence selection, context, generation, claims, grounding) and every quality metric
with the recorded ones.

Before executing, each arm is resolved again. If a component is no longer registered or the
resolved configuration differs from the recorded snapshot (another model revision, a changed
registry), the arm's cases are `not_replayable` and the differences are listed: replaying them
would test another configuration.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from rag_forge.arena import config
from rag_forge.arena.engine import ArenaEngine, ArenaNotFoundError, RunNotFinishedError
from rag_forge.domain.arena import (
    Arm,
    BenchmarkCase,
    CaseStatus,
    Experiment,
    ExperimentRun,
    MetricFamily,
    PipelineKind,
    RunCase,
    RunStatus,
)
from rag_forge.domain.models import PipelineStage, RagResponse, RetrievalResponse, utcnow
from rag_forge.domain.replay import (
    ArmReplayability,
    MetricDifference,
    Replay,
    ReplayCase,
    ReplayOutcome,
    ReplayRequest,
    ReplayStatus,
    StageComparison,
)
from rag_forge.ingestion.service import CorpusNotFoundError
from rag_forge.provenance.environment import capture_environment
from rag_forge.provenance.runtime import capture_runtime
from rag_forge.rag.generation import GeneratorUnavailableError
from rag_forge.rag.grounding import GroundingUnavailableError
from rag_forge.rag.service import RagComponentNotAvailableError, retrieval_stages
from rag_forge.retrieval.dense import DenseIndexNotReadyError
from rag_forge.retrieval.embedding import EmbedderUnavailableError
from rag_forge.retrieval.rerank import RerankerUnavailableError
from rag_forge.retrieval.service import (
    CorpusVersionNotFoundError,
    RerankerNotAvailableError,
    StrategyNotAvailableError,
)
from rag_forge.router.service import RouterComponentNotAvailableError

# A case that raises one of these could not be re-executed as recorded: the environment lacks a
# component, a model or an index. That is "not replayable", not a divergence.
UNAVAILABLE: tuple[type[Exception], ...] = (
    CorpusNotFoundError,
    CorpusVersionNotFoundError,
    DenseIndexNotReadyError,
    EmbedderUnavailableError,
    GeneratorUnavailableError,
    GroundingUnavailableError,
    RagComponentNotAvailableError,
    RerankerNotAvailableError,
    RerankerUnavailableError,
    RouterComponentNotAvailableError,
    StrategyNotAvailableError,
)
TOLERANCE = 1e-9


def stages_of(trace: RagResponse | RetrievalResponse) -> list[PipelineStage]:
    """The recorded provenance chain of a trace (computed the same way for retrieval traces)."""
    if isinstance(trace, RagResponse):
        return trace.provenance.chain
    return retrieval_stages(trace)


def compare_stages(
    recorded: list[PipelineStage], replayed: list[PipelineStage]
) -> list[StageComparison]:
    rec = {s.stage: s for s in recorded}
    rep = {s.stage: s for s in replayed}
    order = [s.stage for s in recorded] + [s.stage for s in replayed if s.stage not in rec]
    out = []
    for name in order:
        a, b = rec.get(name), rep.get(name)
        either = a or b
        out.append(
            StageComparison(
                stage=name,
                deterministic=either.deterministic if either else True,
                recorded=a.hash if a else None,
                replayed=b.hash if b else None,
                match=a.hash == b.hash if a and b else None,
            )
        )
    return out


def classify(stages: list[StageComparison]) -> tuple[ReplayOutcome, str | None, str]:
    """Outcome, first differing stage and the reason, from the stage comparisons in order."""
    first = next((s for s in stages if s.match is not True), None)
    if first is None:
        return ReplayOutcome.EXACT, None, "every recorded stage hash matched"
    if first.match is None:
        side = "replay" if first.replayed is None else "recording"
        return (
            ReplayOutcome.DIVERGED,
            first.stage,
            f"stage {first.stage} is missing from the {side}",
        )
    if first.deterministic:
        return (
            ReplayOutcome.DIVERGED,
            first.stage,
            f"deterministic stage {first.stage} produced a different output",
        )
    return (
        ReplayOutcome.EQUIVALENT,
        first.stage,
        f"every deterministic stage before {first.stage} matched; {first.stage} is not "
        "deterministic, so its output (and the claims and grounding derived from it) may differ",
    )


def compare_metrics(recorded: RunCase, replayed: RunCase) -> tuple[int, list[MetricDifference]]:
    rec = {m.metric: m for m in recorded.metrics if m.family is not MetricFamily.OPERATIONAL}
    rep = {m.metric: m for m in replayed.metrics if m.family is not MetricFamily.OPERATIONAL}
    diffs = []
    for name in sorted(set(rec) | set(rep)):
        a = rec[name].value if name in rec else None
        b = rep[name].value if name in rep else None
        same = (a is None and b is None) or (
            a is not None and b is not None and abs(a - b) <= TOLERANCE
        )
        if not same:
            diffs.append(MetricDifference(metric=name, recorded=a, replayed=b))
    return len(set(rec) | set(rep)), diffs


def generation_note(engine: ArenaEngine, arm: Arm) -> str | None:
    if arm.pipeline is not PipelineKind.RAG or arm.generation is None:
        return None
    name = arm.generation.generator or engine.rag.default_generator
    gen = engine.rag.generators.get(name)
    if gen is None:
        return f"generator {name} is not registered"
    d = gen.describe()
    if not d.local:
        return (
            f"{d.model} is a remote endpoint: outputs are not guaranteed to repeat, so replay "
            "can at best be configuration-equivalent"
        )
    if arm.generation.temperature == 0:
        if name == "extractive-baseline":
            return "extractive baseline: deterministic"
        return (
            f"greedy decoding with {d.model}: deterministic on the same runtime and hardware; "
            "other CPUs or onnxruntime builds may change the text"
        )
    return (
        f"sampled (temperature {arm.generation.temperature}, seed {arm.generation.seed}) with "
        f"{d.model}: repeats only where the seed fully determines sampling"
    )


class ReplayService:
    def __init__(self, engine: ArenaEngine) -> None:
        self.engine = engine
        self._progress = threading.Lock()

    def checks(self, run: ExperimentRun) -> list[ArmReplayability]:
        """Re-resolve every arm now and compare with the recorded snapshot. Executes nothing."""
        e = self.engine.experiment(run.experiment_id)
        recorded = {s.arm: s for s in e.snapshots}
        out = []
        for arm in e.arms:
            snap = recorded[arm.name]
            note = generation_note(self.engine, arm)
            try:
                ds = self.engine.dataset(e.dataset_id)
                corpus = self.engine._corpus(ds.corpus_id)
                current = config.snapshot(
                    arm, ds, corpus, self.engine.retrieval, self.engine.rag, e.metrics
                )
            except Exception as exc:  # unregistered component, missing corpus, ...
                out.append(
                    ArmReplayability(
                        arm=arm.name,
                        recorded_hash=run.snapshot_hashes[arm.name],
                        current_hash=None,
                        replayable=False,
                        reason=f"{type(exc).__name__}: {exc}",
                        changes=[],
                        generation=note,
                    )
                )
                continue
            changes = config.snapshot_diff(snap, current)
            same = current.config_hash() == run.snapshot_hashes[arm.name]
            out.append(
                ArmReplayability(
                    arm=arm.name,
                    recorded_hash=run.snapshot_hashes[arm.name],
                    current_hash=current.config_hash(),
                    replayable=same,
                    reason=None
                    if same
                    else "the arm resolves to a different configuration in this environment",
                    changes=changes,
                    generation=note,
                )
            )
        return out

    def create(self, run_id: str, request: ReplayRequest) -> Replay:
        engine = self.engine
        run = engine.get_run(run_id)
        if run.status not in (RunStatus.COMPLETED, RunStatus.PARTIAL):
            raise RunNotFinishedError(f"run {run_id} is {run.status.value}; replay needs a result")
        arms = request.arms or run.arms
        cases = request.case_ids or run.case_ids
        unknown = sorted(set(arms) - set(run.arms)) + sorted(set(cases) - set(run.case_ids))
        if unknown:
            raise ValueError(f"not part of run {run_id}: {', '.join(unknown)}")
        e = engine.experiment(run.experiment_id)
        return engine.arena.save_replay(
            Replay(
                run_id=run.id,
                experiment_id=run.experiment_id,
                arms=[a for a in run.arms if a in arms],
                case_ids=[c for c in run.case_ids if c in cases],
                total=len(arms) * len(cases),
                checks=self.checks(run),
                environment=capture_environment(),
                runtime=capture_runtime(engine.model_records(e.snapshots)),
            )
        )

    def get(self, replay_id: str) -> Replay:
        replay = self.engine.arena.get_replay(replay_id)
        if replay is None:
            raise ArenaNotFoundError(f"no replay {replay_id}")
        return replay

    def run(self, replay_id: str) -> Replay:
        engine = self.engine
        replay = self.get(replay_id)
        try:
            run = engine.get_run(replay.run_id)
            e = engine.experiment(run.experiment_id)
            ds = engine.dataset(run.dataset_id)
            cases = {c.id: c for c in ds.cases}
            arms = {a.name: a for a in e.arms}
            checks = {c.arm: c for c in replay.checks}
            recorded = {(c.arm, c.case_id): c for c in engine.arena.run_cases(run.id)}
            state = {"replay": engine.arena.save_replay(
                replay.model_copy(update={"status": ReplayStatus.RUNNING})
            )}  # fmt: skip

            def one(task: tuple[str, str]) -> None:
                arm, case_id = task
                result = self._case(
                    replay.id, run, e, arms[arm], cases[case_id], recorded[(arm, case_id)],
                    checks[arm],
                )  # fmt: skip
                with self._progress:
                    r = state["replay"]
                    state["replay"] = engine.arena.save_replay(
                        r.model_copy(
                            update={"completed": r.completed + 1, "cases": [*r.cases, result]}
                        )
                    )

            tasks = [(a, c) for a in replay.arms for c in replay.case_ids]
            with ThreadPoolExecutor(max_workers=e.limits.concurrency) as pool:
                list(pool.map(one, tasks))
            r = state["replay"]
            order = {t: i for i, t in enumerate(tasks)}
            done = sorted(r.cases, key=lambda c: order[(c.arm, c.case_id)])
            counts = {o: sum(c.outcome is o for c in done) for o in ReplayOutcome}
            return engine.arena.save_replay(
                r.model_copy(
                    update={
                        "status": ReplayStatus.COMPLETED,
                        "cases": done,
                        "outcomes": {o: n for o, n in counts.items() if n},
                        "finished_at": utcnow(),
                    }
                )
            )
        except Exception as exc:  # the replay itself could not proceed; record why
            r = self.get(replay_id)
            return engine.arena.save_replay(
                r.model_copy(
                    update={
                        "status": ReplayStatus.FAILED,
                        "error": f"{type(exc).__name__}: {exc}",
                        "finished_at": utcnow(),
                    }
                )
            )

    def _recorded_trace(self, case: RunCase) -> RagResponse | RetrievalResponse | None:
        if case.artifact_id is None:
            return None
        found = self.engine.arena.get_artifact(case.artifact_id)
        if found is None:
            return None
        art, body = found
        if art.kind == "rag_trace":
            return RagResponse.model_validate_json(body)
        return RetrievalResponse.model_validate_json(body)

    def _case(
        self,
        replay_id: str,
        run: ExperimentRun,
        e: Experiment,
        arm: Arm,
        case: BenchmarkCase,
        recorded: RunCase,
        check: ArmReplayability,
    ) -> ReplayCase:
        base: dict[str, object] = {
            "arm": arm.name,
            "case_id": case.id,
            "recorded_status": recorded.status.value,
            "recorded_artifact_id": recorded.artifact_id,
            "replayed_status": None,
            "replayed_artifact_id": None,
            "metrics_compared": 0,
            "metric_differences": [],
            "latency_ms": None,
            "stages": [],
            "first_divergence": None,
        }
        if not check.replayable:
            return ReplayCase.model_validate(
                base
                | {
                    "outcome": ReplayOutcome.NOT_REPLAYABLE,
                    "reason": check.reason or "the arm cannot be resolved as recorded",
                }
            )
        replayed = self.engine._evaluate(run, e, arm, case, replay_id)
        compared, diffs = compare_metrics(recorded, replayed)
        common = base | {
            "replayed_status": replayed.status.value,
            "replayed_artifact_id": replayed.artifact_id,
            "metrics_compared": compared,
            "metric_differences": diffs,
            "latency_ms": replayed.timings_ms.get("total"),
        }
        if replayed.status is CaseStatus.FAILED or recorded.status is CaseStatus.FAILED:
            return self._failure(recorded, replayed, common)
        before = self._recorded_trace(recorded)
        found = self.engine.arena.get_artifact(replayed.artifact_id or "")
        if before is None or found is None:
            return ReplayCase.model_validate(
                common
                | {
                    "outcome": ReplayOutcome.NOT_REPLAYABLE,
                    "reason": "the recorded trace artifact is missing",
                }
            )
        art, body = found
        after: RagResponse | RetrievalResponse = (
            RagResponse.model_validate_json(body)
            if art.kind == "rag_trace"
            else RetrievalResponse.model_validate_json(body)
        )
        stages = compare_stages(stages_of(before), stages_of(after))
        outcome, first, reason = classify(stages)
        if outcome is ReplayOutcome.EXACT and diffs:
            outcome, reason = ReplayOutcome.DIVERGED, "stage hashes matched but metrics differ"
        return ReplayCase.model_validate(
            common
            | {"outcome": outcome, "reason": reason, "stages": stages, "first_divergence": first}
        )

    def _failure(
        self, recorded: RunCase, replayed: RunCase, common: dict[str, object]
    ) -> ReplayCase:
        if replayed.status is CaseStatus.FAILED and replayed.error_type in {
            t.__name__ for t in UNAVAILABLE
        }:
            outcome = (
                ReplayOutcome.EXACT
                if recorded.error_type == replayed.error_type
                else ReplayOutcome.NOT_REPLAYABLE
            )
            reason = (
                f"the recorded failure reproduced: {replayed.error_type}"
                if outcome is ReplayOutcome.EXACT
                else f"a component is unavailable: {replayed.error_type}: {replayed.error}"
            )
        elif recorded.status is CaseStatus.FAILED and replayed.status is CaseStatus.FAILED:
            same = recorded.error_type == replayed.error_type
            outcome = ReplayOutcome.EXACT if same else ReplayOutcome.DIVERGED
            reason = (
                f"the recorded failure reproduced: {replayed.error_type}"
                if same
                else f"failed differently: {recorded.error_type} -> {replayed.error_type}"
            )
        elif recorded.status is CaseStatus.FAILED:
            outcome = ReplayOutcome.DIVERGED
            reason = f"the recorded failure ({recorded.error_type}) did not reproduce"
        else:
            outcome = ReplayOutcome.DIVERGED
            reason = f"the replay failed: {replayed.error_type}: {replayed.error}"
        return ReplayCase.model_validate(common | {"outcome": outcome, "reason": reason})
