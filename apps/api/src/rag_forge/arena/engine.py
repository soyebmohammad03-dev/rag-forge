"""The experiment engine: datasets -> experiments (arms + snapshots) -> runs -> per-case results
-> aggregates -> leaderboards and paired comparisons.

Runs execute in process, on the already-loaded models, case by case (optionally a few cases in
parallel). A case that raises is stored as a failed `RunCase` with its error; it is never
dropped, and its metrics are absent rather than zero. Results are stored as they complete, so a
run's progress is observable and partial results survive.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from rag_forge.arena import config, metrics, stats
from rag_forge.arena.datasets import (
    DEV_CASES,
    DEV_CORPUS,
    DEV_DESCRIPTION,
    DEV_DOCUMENTS,
    DEV_NAME,
    DEV_NOTES,
    build_dataset,
    validate_against_corpus,
)
from rag_forge.arena.metrics import judgements
from rag_forge.domain.arena import (
    Arm,
    ArmRef,
    ArmSummary,
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkDatasetCreate,
    CaseStatus,
    CaseView,
    Comparison,
    ConfigurationSnapshot,
    DatasetSource,
    Experiment,
    ExperimentCreate,
    ExperimentRun,
    GroundingSnapshot,
    Leaderboard,
    LeaderboardRow,
    MetricSettings,
    MetricSummary,
    PipelineKind,
    RankedItem,
    RunCase,
    RunStatus,
)
from rag_forge.domain.models import (
    ChunkingConfig,
    Corpus,
    DenseIndexState,
    RagRequest,
    RagResponse,
    RetrievalResponse,
    utcnow,
)
from rag_forge.ingestion.service import CorpusNotFoundError, IngestionService
from rag_forge.provenance.environment import capture_environment
from rag_forge.rag.service import RagService
from rag_forge.retrieval.dense import DenseIndexService
from rag_forge.retrieval.service import RetrievalService
from rag_forge.storage.arena import ArenaStore
from rag_forge.storage.base import CorpusStore

DEFINITIONS = {d.name: d for d in metrics.definitions()}


class ArenaNotFoundError(LookupError):
    pass


class RunNotFinishedError(RuntimeError):
    pass


def definition_of(metric: str) -> tuple[str, bool | None]:
    d = DEFINITIONS[metric.split("@")[0]]
    return d.family, d.higher_is_better


class ArenaEngine:
    def __init__(
        self,
        store: CorpusStore,
        arena: ArenaStore,
        ingestion: IngestionService,
        retrieval: RetrievalService,
        rag: RagService,
        dense: DenseIndexService,
    ) -> None:
        self.store = store
        self.arena = arena
        self.ingestion = ingestion
        self.retrieval = retrieval
        self.rag = rag
        self.dense = dense
        self._progress = threading.Lock()

    # --- datasets ---------------------------------------------------------------------

    def _corpus(self, corpus_id: str) -> Corpus:
        corpus = self.store.get_corpus(corpus_id)
        if corpus is None:
            raise CorpusNotFoundError(corpus_id)
        return corpus

    def create_dataset(
        self, spec: BenchmarkDatasetCreate, source: DatasetSource = DatasetSource.USER
    ) -> BenchmarkDataset:
        corpus = self._corpus(spec.corpus_id)
        version = corpus.version if spec.corpus_version is None else spec.corpus_version
        validate_against_corpus(self.store, corpus, version, spec.cases)
        previous = self.arena.dataset_versions(spec.name)
        if previous and previous[-1].source is not source:
            raise ValueError(f"dataset name {spec.name!r} is a {previous[-1].source} dataset")
        ds = build_dataset(spec, corpus, version, source, previous)
        return ds if any(p.id == ds.id for p in previous) else self.arena.add_dataset(ds)

    def install_development(self, build_dense_index: bool = True) -> BenchmarkDataset:
        """Create (or reuse) the development corpus and register the development dataset."""
        corpus = next((c for c in self.store.list_corpora() if c.name == DEV_CORPUS), None)
        if corpus is None:
            corpus = self.store.add_corpus(
                Corpus(
                    name=DEV_CORPUS,
                    description="Documents of the bundled development benchmark (Arena).",
                    chunking=ChunkingConfig(),
                )
            )
        self.ingestion.ingest(corpus.id, [(f, t.encode()) for f, t in DEV_DOCUMENTS])
        corpus = self._corpus(corpus.id)
        if build_dense_index:
            index, needs_build = self.dense.start(corpus, corpus.version)
            if needs_build:
                self.dense.build(index, corpus)
        spec = BenchmarkDatasetCreate(
            name=DEV_NAME,
            description=DEV_DESCRIPTION,
            corpus_id=corpus.id,
            corpus_version=corpus.version,
            annotation_notes=DEV_NOTES,
            cases=DEV_CASES,
        )
        return self.create_dataset(spec, DatasetSource.DEVELOPMENT)

    def dataset(self, dataset_id: str) -> BenchmarkDataset:
        ds = self.arena.get_dataset(dataset_id)
        if ds is None:
            raise ArenaNotFoundError(f"no dataset {dataset_id}")
        return ds

    # --- experiments ------------------------------------------------------------------

    def resolve(
        self, dataset_id: str, arms: list[Arm], settings: ExperimentCreate | None = None
    ) -> list[ConfigurationSnapshot]:
        ds = self.dataset(dataset_id)
        corpus = self._corpus(ds.corpus_id)
        ms = settings.metrics if settings else MetricSettings()
        metrics.selected(ms)  # unknown metric names fail here
        return [config.snapshot(a, ds, corpus, self.retrieval, self.rag, ms) for a in arms]

    def create_experiment(self, spec: ExperimentCreate) -> Experiment:
        ds = self.dataset(spec.dataset_id)
        snapshots = self.resolve(spec.dataset_id, spec.arms, spec)
        by_name = {s.arm: s for s in snapshots}
        arms = {a.name: a for a in spec.arms}
        return self.arena.add_experiment(
            Experiment(
                name=spec.name,
                hypothesis=spec.hypothesis,
                dataset_id=ds.id,
                dataset_name=ds.name,
                dataset_version=ds.version,
                dataset_source=ds.source,
                corpus_id=ds.corpus_id,
                corpus_version=ds.corpus_version,
                arms=spec.arms,
                snapshots=snapshots,
                snapshot_hashes={s.arm: s.config_hash() for s in snapshots},
                ablations=[config.ablation(a, arms, by_name) for a in spec.ablations],
                metrics=spec.metrics,
                limits=spec.limits,
            )
        )

    def experiment(self, experiment_id: str) -> Experiment:
        e = self.arena.get_experiment(experiment_id)
        if e is None:
            raise ArenaNotFoundError(f"no experiment {experiment_id}")
        return e

    # --- runs -------------------------------------------------------------------------

    def create_run(self, experiment_id: str) -> ExperimentRun:
        e = self.experiment(experiment_id)
        ds = self.dataset(e.dataset_id)
        cases = ds.cases[: e.limits.max_cases] if e.limits.max_cases else ds.cases
        return self.arena.save_run(
            ExperimentRun(
                experiment_id=e.id,
                dataset_id=ds.id,
                dataset_version=ds.version,
                dataset_hash=ds.content_hash,
                corpus_id=ds.corpus_id,
                corpus_version=ds.corpus_version,
                case_ids=[c.id for c in cases],
                arms=[a.name for a in e.arms],
                snapshot_hashes=e.snapshot_hashes,
                limits=e.limits,
                total=len(cases) * len(e.arms),
                metric_registry_version=metrics.REGISTRY_VERSION,
                stats_method=stats.METHOD_ID,
                environment=capture_environment(),
            )
        )

    def run(self, run_id: str) -> ExperimentRun:
        run = self.get_run(run_id)
        try:
            e = self.experiment(run.experiment_id)
            ds = self.dataset(run.dataset_id)
            cases = {c.id: c for c in ds.cases}
            run = self.arena.save_run(
                run.model_copy(update={"status": RunStatus.RUNNING, "started_at": utcnow()})
            )
            tasks = [(arm, cases[cid]) for arm in e.arms for cid in run.case_ids]
            state = {"run": run}

            def one(task: tuple[Arm, BenchmarkCase]) -> None:
                arm, case = task
                result = self._evaluate(state["run"], e, arm, case)
                self.arena.add_case(result)
                with self._progress:
                    r = state["run"]
                    state["run"] = self.arena.save_run(
                        r.model_copy(
                            update={
                                "completed": r.completed + 1,
                                "failed": r.failed + (result.status is CaseStatus.FAILED),
                            }
                        )
                    )

            with ThreadPoolExecutor(max_workers=e.limits.concurrency) as pool:
                list(pool.map(one, tasks))  # re-raises storage errors, not case errors
            run = state["run"]
            summaries = [self._summarize(run, e, arm) for arm in e.arms]
            status = RunStatus.PARTIAL if run.failed else RunStatus.COMPLETED
            return self.arena.save_run(
                run.model_copy(
                    update={"status": status, "summaries": summaries, "finished_at": utcnow()}
                )
            )
        except Exception as exc:  # the run itself could not proceed; record why
            run = self.get_run(run_id)
            return self.arena.save_run(
                run.model_copy(
                    update={
                        "status": RunStatus.FAILED,
                        "error": f"{type(exc).__name__}: {exc}",
                        "finished_at": utcnow(),
                    }
                )
            )

    def get_run(self, run_id: str) -> ExperimentRun:
        run = self.arena.get_run(run_id)
        if run is None:
            raise ArenaNotFoundError(f"no run {run_id}")
        return run

    def _evaluate(
        self, run: ExperimentRun, e: Experiment, arm: Arm, case: BenchmarkCase
    ) -> RunCase:
        started_at, started = utcnow(), time.perf_counter()
        base = {
            "run_id": run.id,
            "experiment_id": e.id,
            "arm": arm.name,
            "case_id": case.id,
            "config_hash": e.snapshot_hashes[arm.name],
            "started_at": started_at,
        }
        try:
            request = arm.retrieval.request(case.query, e.corpus_version)
            rag: RagResponse | None = None
            if arm.pipeline is PipelineKind.RAG:
                rag = self.rag.answer(
                    e.corpus_id,
                    RagRequest(
                        retrieval=request,
                        evidence=arm.evidence_params(),
                        generation=arm.generation,
                        grounding=arm.grounding,
                    ),
                )
                response = rag.retrieval
            else:
                response = self.retrieval.retrieve(e.corpus_id, request)
            total = round((time.perf_counter() - started) * 1000, 3)
            return self._record(base, e, arm, case, response, rag, total)
        except Exception as exc:  # a failed case is a result: kept, typed and explained
            return RunCase(
                **base,
                retrieval_configuration_hash=None,
                status=CaseStatus.FAILED,
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
                metrics=metrics.failure_metrics(),
                timings_ms={"total": round((time.perf_counter() - started) * 1000, 3)},
                finished_at=utcnow(),
            )

    def _record(
        self,
        base: dict[str, object],
        e: Experiment,
        arm: Arm,
        case: BenchmarkCase,
        response: RetrievalResponse,
        rag: RagResponse | None,
        total: float,
    ) -> RunCase:
        unit, rel = judgements(case)

        def grade(chunk_id: str, filename: str) -> float | None:
            return rel.get(chunk_id if unit == "chunk" else filename)

        ranking = [
            RankedItem(
                rank=h.result.rank,
                chunk_id=h.chunk.id,
                document_id=h.result.document_id,
                filename=h.filename,
                score=h.result.score,
                upstream_rank=h.rerank.original_rank if h.rerank else None,
                relevance=grade(h.chunk.id, h.filename),
            )
            for h in response.hits
        ]
        pool_up = pool_final = None
        if response.reranking is not None:
            pool = response.reranking.candidates
            pool_final = [
                RankedItem(
                    rank=c.rerank.final_rank,
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    filename=c.filename,
                    score=c.rerank.reranker_score,
                    upstream_rank=c.rerank.original_rank,
                    relevance=grade(c.chunk_id, c.filename),
                )
                for c in sorted(pool, key=lambda c: c.rerank.final_rank)
            ]
            pool_up = [
                i.model_copy(update={"rank": i.upstream_rank})
                for i in sorted(pool_final, key=lambda i: i.upstream_rank or 0)
            ]
        prov = response.provenance
        timings = {"total": total, "retrieval": prov.elapsed_ms}
        if prov.reranking:
            timings["rerank"] = prov.reranking.latency_ms
        if prov.routing:
            timings["routing"] = round(prov.routing.analysis_ms + prov.routing.decision_ms, 3)
        models = {}
        if prov.configuration.embedder:
            emb = prov.configuration.embedder
            models["embedder"] = f"{emb.model}@{emb.revision[:8]}"
        if prov.reranking:
            spec = prov.reranking.info.spec
            models["reranker"] = f"{spec.model}@{spec.revision[:8]}"
        tokens: dict[str, int] = {}
        grounding = None
        if rag is not None:
            if rag.answer is not None:
                g = rag.answer.generation
                timings["generation"] = g.latency_ms
                timings["generator_load"] = g.load_ms
                rev = f"@{g.generator.revision[:8]}" if g.generator.revision else ""
                models["generator"] = f"{g.generator.model}{rev}"
                if g.prompt_tokens is not None:
                    tokens["prompt"] = g.prompt_tokens
                if g.completion_tokens is not None:
                    tokens["completion"] = g.completion_tokens
            if rag.grounding is not None:
                gr = rag.grounding
                timings["grounding"] = gr.latency_ms
                timings["verifier_load"] = gr.load_ms
                grounding = GroundingSnapshot(
                    status=gr.status,
                    claims=gr.claims,
                    factual_claims=gr.factual_claims,
                    supported=gr.supported,
                    weakly_supported=gr.weakly_supported,
                    unsupported=gr.unsupported,
                    grounding_score=gr.grounding_score,
                    citation_coverage=gr.citation_coverage,
                    citation_precision=gr.citation_precision,
                    evidence_coverage=gr.evidence_coverage,
                )
        ctx = metrics.MetricContext(
            case=case,
            pipeline=arm.pipeline,
            ranking=ranking,
            pool_upstream=pool_up,
            pool_final=pool_final,
            rag=rag,
            timings_ms=timings,
            tokens=tokens,
        )
        body = (rag or response).model_dump_json().encode()
        artifact = self.arena.add_artifact(
            str(base["run_id"]), arm.name, case.id, "rag_trace" if rag else "retrieval_trace", body
        )
        return RunCase(
            **base,
            retrieval_configuration_hash=prov.configuration_hash,
            status=CaseStatus.OK,
            route=prov.routing.decision.option.value if prov.routing else None,
            ranking=ranking,
            evidence=[x.filename for x in rag.evidence.selected] if rag else [],
            answer_status=rag.status if rag else None,
            answer=rag.answer.text if rag and rag.answer else None,
            grounding=grounding,
            metrics=metrics.compute(ctx, e.metrics),
            timings_ms=timings,
            tokens=tokens,
            models=models,
            artifact_id=artifact.id,
            finished_at=utcnow(),
        )

    # --- aggregation ------------------------------------------------------------------

    def _summarize(self, run: ExperimentRun, e: Experiment, arm: Arm) -> ArmSummary:
        cases = self.arena.run_cases(run.id, arm.name)
        values: dict[str, list[float]] = {}
        skipped: dict[str, Counter[str]] = {}
        order: list[str] = []
        for c in sorted(cases, key=lambda c: c.case_id):
            for m in c.metrics:
                if m.metric not in values:
                    values[m.metric], skipped[m.metric] = [], Counter()
                    order.append(m.metric)
                if m.value is not None:
                    values[m.metric].append(m.value)
                elif m.skipped:
                    skipped[m.metric][m.skipped] += 1
        summaries = []
        for name in order:
            d = DEFINITIONS[name.split("@")[0]]
            s = stats.describe(values[name])
            summaries.append(
                MetricSummary(
                    metric=name,
                    family=d.family,
                    version=d.version,
                    higher_is_better=d.higher_is_better,
                    n=len(values[name]),
                    skipped=sum(skipped[name].values()),
                    ci_method=f"percentile bootstrap {stats.CONFIDENCE:.0%}"
                    if s["ci_low"] is not None
                    else None,
                    **s,
                )
            )
        failed = [c for c in cases if c.status is CaseStatus.FAILED]
        return ArmSummary(
            arm=arm.name,
            label=arm.label or arm.name,
            pipeline=arm.pipeline,
            config_hash=e.snapshot_hashes[arm.name],
            cases=len(cases),
            succeeded=len(cases) - len(failed),
            failed=len(failed),
            failure_types=dict(Counter(c.error_type or "unknown" for c in failed)),
            metrics=summaries,
            skipped_reasons={k: v.most_common(1)[0][0] for k, v in skipped.items() if v},
        )

    # --- reading results --------------------------------------------------------------

    def case_view(self, run_id: str, case_id: str) -> CaseView:
        run = self.get_run(run_id)
        ds = self.dataset(run.dataset_id)
        case = next((c for c in ds.cases if c.id == case_id), None)
        if case is None or case_id not in run.case_ids:
            raise ArenaNotFoundError(f"run {run_id} has no case {case_id}")
        results = self.arena.run_cases(run_id, case_id=case_id)
        order = {a: i for i, a in enumerate(run.arms)}
        return CaseView(
            run_id=run_id, case=case, results=sorted(results, key=lambda r: order[r.arm])
        )

    def leaderboard(self, run_id: str, metric: str) -> Leaderboard:
        run = self.get_run(run_id)
        if run.status not in (RunStatus.COMPLETED, RunStatus.PARTIAL):
            raise RunNotFinishedError(f"run {run_id} is {run.status.value}")
        if metric.split("@")[0] not in DEFINITIONS:
            raise ArenaNotFoundError(f"unknown metric {metric}")
        _, higher = definition_of(metric)

        def mean_of(summary: ArmSummary, name: str) -> MetricSummary | None:
            return next((x for x in summary.metrics if x.metric == name and x.n > 0), None)

        defined = [(s, m) for s in run.summaries if (m := mean_of(s, metric)) is not None]
        sign = -1 if higher is not False else 1
        defined.sort(key=lambda r: (sign * (r[1].mean or 0.0), r[0].arm))
        leader = defined[0][1] if defined and higher is not None else None
        out = []
        position = 0
        for i, (s, m) in enumerate(defined, 1):
            if i == 1 or m.mean != defined[i - 2][1].mean:
                position = i  # ties share a position (competition ranking)
            overlaps = None
            if leader is not None and m.ci_low is not None and leader.ci_low is not None:
                overlaps = not (
                    (m.ci_high or 0) < leader.ci_low or m.ci_low > (leader.ci_high or 0)
                )
            out.append(self._row(position if higher is not None else None, s, m, overlaps))
        ranked = {s.arm for s, _ in defined}
        out += [self._row(None, s, None, None) for s in run.summaries if s.arm not in ranked]
        note = (
            "Ordered by mean; tied means share a position, and overlapping confidence intervals "
            "mean the order is not established. "
            "Use a paired comparison for a decision."
            if higher is not None
            else "A descriptive metric: rows are listed, not ranked."
        )
        return Leaderboard(
            run_id=run_id, metric=metric, higher_is_better=higher, rows=out, note=note
        )

    @staticmethod
    def _row(
        pos: int | None,
        s: ArmSummary,
        m: MetricSummary | None,
        overlaps: bool | None,
    ) -> LeaderboardRow:
        lat = next((x.mean for x in s.metrics if x.metric == "latency_ms"), None)
        return LeaderboardRow(
            position=pos,
            arm=s.arm,
            label=s.label,
            pipeline=s.pipeline,
            config_hash=s.config_hash,
            summary=m,
            ci_overlaps_leader=overlaps,
            failed=s.failed,
            cases=s.cases,
            mean_latency_ms=lat,
        )

    def compare(
        self,
        baseline_run: str,
        baseline_arm: str,
        variant_run: str,
        variant_arm: str,
        metric_names: list[str] | None = None,
    ) -> Comparison:
        runs = {r: self.get_run(r) for r in {baseline_run, variant_run}}
        rb, rv = runs[baseline_run], runs[variant_run]
        for r, a in ((rb, baseline_arm), (rv, variant_arm)):
            if a not in r.arms:
                raise ArenaNotFoundError(f"run {r.id} has no arm {a}")
        eb, ev = self.experiment(rb.experiment_id), self.experiment(rv.experiment_id)
        sb = next(s for s in eb.snapshots if s.arm == baseline_arm)
        sv = next(s for s in ev.snapshots if s.arm == variant_arm)
        issues, warnings = [], []
        for r in runs.values():
            if r.status not in (RunStatus.COMPLETED, RunStatus.PARTIAL):
                issues.append(f"run {r.id} is {r.status.value}")
        if rb.dataset_hash != rv.dataset_hash:
            issues.append("different benchmark datasets or dataset versions")
        if (rb.corpus_id, rb.corpus_version) != (rv.corpus_id, rv.corpus_version):
            issues.append("different corpus versions")
        if sb.metric_versions != sv.metric_versions or sb.metrics.ks != sv.metrics.ks:
            issues.append("different metric definitions or k values")
        cb = {c.case_id: c for c in self.arena.run_cases(rb.id, baseline_arm)}
        cv = {c.case_id: c for c in self.arena.run_cases(rv.id, variant_arm)}
        shared = sorted(set(cb) & set(cv))
        if not shared and not issues:
            issues.append("no benchmark case was evaluated by both arms")
        if set(cb) != set(cv):
            warnings.append(f"only the {len(shared)} cases evaluated by both arms are paired")
        failed = [c for c in shared if CaseStatus.FAILED in (cb[c].status, cv[c].status)]
        if failed:
            warnings.append(
                f"{len(failed)} case(s) failed in at least one arm; quality metrics pair only "
                "the cases both completed, failure rate pairs all"
            )
        if sb.retrieval_template.top_k != sv.retrieval_template.top_k:
            warnings.append(
                f"retrieval budgets differ: top_k {sb.retrieval_template.top_k} vs "
                f"{sv.retrieval_template.top_k}"
            )
        if sb.evidence and sv.evidence and sb.evidence != sv.evidence:
            warnings.append("evidence budgets differ")
        if sb.config_hash() == sv.config_hash():
            warnings.append("the two configurations are identical")
        if eb.dataset_source is DatasetSource.DEVELOPMENT:
            warnings.append(
                "development benchmark: results validate the pipeline and are not evidence "
                "about the methods in general"
            )
        factor = None
        if rb.id == rv.id:
            factor = next(
                (
                    a.factor
                    for a in eb.ablations
                    if (a.baseline, a.variant) == (baseline_arm, variant_arm)
                ),
                None,
            )
        compared = []
        if not issues:
            names = metric_names or self._metric_order(rb, baseline_arm)
            for name in names:
                if name.split("@")[0] not in DEFINITIONS:
                    raise ArenaNotFoundError(f"unknown metric {name}")
                family, higher = definition_of(name)
                bv, vv = self._values(cb, shared, name), self._values(cv, shared, name)
                compared.append(stats.paired(name, family, higher, bv, vv))  # type: ignore[arg-type]
            compared = stats.adjust(compared)
        return Comparison(
            baseline=ArmRef(
                run_id=rb.id,
                arm=baseline_arm,
                config_hash=sb.config_hash(),
                label=self._label(eb, baseline_arm),
            ),
            variant=ArmRef(
                run_id=rv.id,
                arm=variant_arm,
                config_hash=sv.config_hash(),
                label=self._label(ev, variant_arm),
            ),
            comparable=not issues,
            issues=issues,
            warnings=warnings,
            changes=config.snapshot_diff(sb, sv),
            factor=factor,
            method=stats.METHOD,
            metrics=compared,
        )

    @staticmethod
    def _label(e: Experiment, arm: str) -> str:
        a = next(x for x in e.arms if x.name == arm)
        return a.label or a.name

    @staticmethod
    def _metric_order(run: ExperimentRun, arm: str) -> list[str]:
        s = next((s for s in run.summaries if s.arm == arm), None)
        return [m.metric for m in s.metrics] if s else []

    @staticmethod
    def _values(cases: dict[str, RunCase], shared: list[str], metric: str) -> dict[str, float]:
        out = {}
        for cid in shared:
            for m in cases[cid].metrics:
                if m.metric == metric and m.value is not None:
                    out[cid] = m.value
        return out

    def trace(self, artifact_id: str) -> tuple[str, dict[str, object]]:
        found = self.arena.get_artifact(artifact_id)
        if found is None:
            raise ArenaNotFoundError(f"no artifact {artifact_id}")
        art, body = found
        return art.kind, json.loads(body)

    def dense_ready(self, corpus_id: str, version: int) -> bool:
        state, _ = self.dense.state(self._corpus(corpus_id), version)
        return state is DenseIndexState.READY
