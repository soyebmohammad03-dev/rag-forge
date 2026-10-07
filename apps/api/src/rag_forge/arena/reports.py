"""Reproducibility manifests and research exports (JSON, CSV, Markdown) for recorded runs.

Everything here is read from what a run recorded. Nothing is recomputed into a new claim: the
report's interpretation section only restates the paired comparisons' recorded conclusions, and
it says so. Qualitative judgement is left to the researcher.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from rag_forge.arena import stats
from rag_forge.arena.engine import ArenaEngine, RunNotFinishedError
from rag_forge.domain.arena import (
    CaseStatus,
    Comparison,
    DatasetSource,
    Experiment,
    ExperimentRun,
    MetricFamily,
    RunCase,
    RunStatus,
)
from rag_forge.domain.models import RagResponse, canonical_hash
from rag_forge.domain.replay import ManifestArm, ManifestArtifact, ReproducibilityManifest
from rag_forge.provenance.environment import capture_environment
from rag_forge.provenance.runtime import capture_runtime

EXPORT_FORMAT = "rag-forge-export@1"
FAMILY_TITLE = {
    MetricFamily.RETRIEVAL: "Retrieval",
    MetricFamily.RERANKING: "Reranking",
    MetricFamily.EVIDENCE: "Evidence and grounding",
    MetricFamily.GENERATION: "Generation",
    MetricFamily.OPERATIONAL: "Operational",
}
PRIMARY = ("ndcg@10", "ndcg@5", "mrr", "grounding_score", "latency_ms")


def fmt(value: float | None, metric: str) -> str:
    """Milliseconds, token counts or a 3-decimal score, as the UI shows them."""
    if value is None:
        return "—"
    if metric.endswith("_ms"):
        return f"{value:.1f} ms" if value < 100 else f"{round(value)} ms"
    if metric.endswith("_tokens"):
        return f"{value:.0f}"
    return f"{value:.3f}"


def signed(value: float | None, metric: str) -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else "-" if value < 0 else "±"
    return sign + fmt(abs(value), metric)


def _finished(run: ExperimentRun) -> None:
    if run.status not in (RunStatus.COMPLETED, RunStatus.PARTIAL):
        raise RunNotFinishedError(f"run {run.id} is {run.status.value}; nothing to export yet")


def _traces(engine: ArenaEngine, run: ExperimentRun) -> dict[tuple[str, str], RagResponse]:
    """The recorded RAG traces of a run, by (arm, case)."""
    out = {}
    for art in engine.arena.run_artifacts(run.id):
        if art.replay_id is None and art.kind == "rag_trace":
            found = engine.arena.get_artifact(art.id)
            if found:
                out[(art.arm, art.case_id)] = RagResponse.model_validate_json(found[1])
    return out


# --- manifest -------------------------------------------------------------------------------


def manifest(engine: ArenaEngine, run_id: str) -> ReproducibilityManifest:
    run = engine.get_run(run_id)
    e = engine.experiment(run.experiment_id)
    ds = engine.dataset(run.dataset_id)
    traces = _traces(engine, run)
    arms = []
    for s in e.snapshots:
        mine = [t for (arm, _), t in traces.items() if arm == s.arm]
        contexts = {c.hash for t in mine for c in t.provenance.chain if c.stage == "context"}
        gens = [t.answer.generation.deterministic for t in mine if t.answer is not None]
        arms.append(
            ManifestArm(
                arm=s.arm,
                pipeline=s.pipeline.value,
                config_hash=run.snapshot_hashes[s.arm],
                retrieval_mode=s.retrieval_mode.value,
                retrieval_configuration_hash=s.retrieval.config_hash() if s.retrieval else None,
                routing_hash=s.routing_hash,
                query_analyzer=s.query_analyzer,
                router_policy=s.router_policy,
                evidence_params_hash=s.evidence.config_hash() if s.evidence else None,
                prompt_template=s.prompt_template,
                prompt_template_hash=canonical_hash(s.prompt_template)
                if s.prompt_template
                else None,
                generator=s.generator.name if s.generator else None,
                generator_config_hash=s.generator.config_hash if s.generator else None,
                temperature=s.generation.temperature if s.generation else None,
                seed=s.generation.seed if s.generation else None,
                verifier=s.verifier,
                verifier_config_hash=s.verifier_config_hash,
                context_hashes=len(contexts - {None}),
                generation_deterministic=all(gens) if gens else None,
            )
        )
    now_env = capture_environment()
    now_rt = capture_runtime(engine.model_records(e.snapshots))
    differences = []
    recorded_commit = run.runtime.git_commit if run.runtime else run.environment.git_commit
    if now_rt.git_commit != recorded_commit:
        differences.append(f"git commit: recorded {recorded_commit}, now {now_rt.git_commit}")
    if now_env.python_version != run.environment.python_version:
        differences.append(
            f"python: recorded {run.environment.python_version}, now {now_env.python_version}"
        )
    if now_env.packages != run.environment.packages:
        differences.append("tracked package versions differ")
    if run.runtime is not None:
        if run.runtime.lockfiles != now_rt.lockfiles:
            differences.append("lockfile hashes differ")
        rec_files = {(m.model, f.path): f.sha256 for m in run.runtime.models for f in m.files}
        now_files = {(m.model, f.path): f.sha256 for m in now_rt.models for f in m.files}
        changed = sorted(
            f"{m}/{p}" for (m, p), h in rec_files.items() if now_files.get((m, p)) != h
        )
        if changed:
            differences.append(f"model files differ or are not cached: {', '.join(changed)}")
    notes = [
        "Secrets are never recorded: settings whose names contain KEY, TOKEN, SECRET, PASSWORD "
        "or CREDENTIAL are redacted.",
        "Timings are machine-specific; replays compare stage output hashes and quality "
        "metrics, not latency.",
        "Retrieval, reranking, evidence selection, context assembly, claim extraction and "
        "grounding have no randomness. Generation is deterministic only for greedy local "
        "decoding (see generation_deterministic per arm).",
    ]
    if run.runtime is None:
        notes.append(
            "This run was recorded before runtime capture existed: toolchain versions, lockfile "
            "hashes and model file hashes are not available for it ('runtime' is null)."
        )
    if ds.source is DatasetSource.DEVELOPMENT:
        notes.append(
            "Development benchmark: validates the pipeline; too small and too easy to rank methods."
        )
    replays = [
        {
            "id": r.id,
            "status": r.status.value,
            "created_at": r.created_at.isoformat(),
            "cases": r.total,
            "outcomes": {o.value: n for o, n in r.outcomes.items()},
        }
        for r in engine.arena.list_replays(run.id)
    ]
    return ReproducibilityManifest(
        run={
            "id": run.id,
            "experiment_id": e.id,
            "experiment": e.name,
            "hypothesis": e.hypothesis,
            "status": run.status.value,
            "created_at": run.created_at.isoformat(),
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            "evaluations": run.total,
            "completed": run.completed,
            "failed": run.failed,
            "case_ids": run.case_ids,
            "arms": run.arms,
            "limits": run.limits.model_dump(mode="json"),
        },
        dataset={
            "id": ds.id,
            "name": ds.name,
            "version": ds.version,
            "source": ds.source.value,
            "content_hash": ds.content_hash,
            "corpus_id": ds.corpus_id,
            "corpus_version": ds.corpus_version,
            "chunking_hash": ds.chunking_hash,
            "cases": len(ds.cases),
            "annotation_notes": ds.annotation_notes,
        },
        arms=arms,
        models=run.runtime.models if run.runtime else [],
        seeds={
            "bootstrap": stats.SEED,
            "generation": {a.arm: a.seed for a in arms if a.generator},
            "temperature": {a.arm: a.temperature for a in arms if a.generator},
        },
        metrics={
            "registry": run.metric_registry_version,
            "ks": e.metrics.ks,
            "selected": e.metrics.metrics or "all registered",
            "versions": e.snapshots[0].metric_versions if e.snapshots else {},
        },
        statistics=stats.METHOD.model_dump(),
        environment=run.environment,
        runtime=run.runtime,
        current={
            "environment": now_env.model_dump(),
            "runtime": now_rt.model_dump(mode="json"),
            "differences": differences,
        },
        artifacts=[
            ManifestArtifact(
                arm=a.arm,
                case_id=a.case_id,
                artifact_id=a.id,
                kind=a.kind,
                sha256=a.sha256,
                bytes=a.bytes,
            )
            for a in sorted(engine.arena.run_artifacts(run.id), key=lambda a: (a.arm, a.case_id))
            if a.replay_id is None
        ],
        replays=replays,
        notes=notes,
    )


def manifest_markdown(m: ReproducibilityManifest) -> str:
    rt = m.runtime
    lines = [
        f"# Reproducibility manifest: {m.run['experiment']}",
        "",
        f"`{m.manifest_version}` · generated {m.generated_at.isoformat()} · run `{m.run['id']}`",
        "",
        "## Run",
        "",
        f"- Status: {m.run['status']} ({m.run['completed']}/{m.run['evaluations']} evaluated, "
        f"{m.run['failed']} failed)",
        f"- Created {m.run['created_at']}, finished {m.run['finished_at']}",
        f"- Dataset: {m.dataset['name']} v{m.dataset['version']} ({m.dataset['source']}), "
        f"content hash `{m.dataset['content_hash']}`",
        f"- Corpus `{m.dataset['corpus_id']}` v{m.dataset['corpus_version']}, chunking hash "
        f"`{m.dataset['chunking_hash']}`",
        "",
        "## Environment (recorded at run creation)",
        "",
        f"- Git commit: `{rt.git_commit if rt else m.environment.git_commit}`"
        + (f" (uncommitted changes: {'yes' if rt.git_dirty else 'no'})" if rt else ""),
        f"- RAG FORGE {m.environment.rag_forge_version}, Python {m.environment.python_version}, "
        f"{m.environment.platform}",
    ]
    if rt:
        lines += [
            f"- Node {rt.node_version or 'not found'}, uv {rt.uv_version or 'not found'}",
            "- Packages: " + ", ".join(f"{k} {v}" for k, v in rt.packages.items()),
            "- Lockfiles: "
            + ", ".join(f"`{k}` sha256 `{v[:16]}…`" for k, v in rt.lockfiles.items()),
            "- Settings: "
            + (", ".join(f"`{k}={v}`" for k, v in rt.settings.items()) or "defaults"),
        ]
    else:
        lines.append("- Runtime: not recorded for this run (recorded before runtime capture).")
    lines += ["", "## Models", ""]
    if m.models:
        lines += [
            "| Role | Model | Revision | File | sha256 | Bytes |",
            "|---|---|---|---|---|---|",
        ]
        for mr in m.models:
            for f in mr.files:
                sha = f"`{f.sha256[:16]}…`" if f.sha256 else f.source
                lines.append(
                    f"| {mr.role} | {mr.model} | `{(mr.revision or '—')[:12]}` | {f.path} | {sha} "
                    f"| {f.bytes if f.bytes is not None else '—'} |"
                )
    else:
        lines.append("Model files were not recorded for this run.")
    lines += [
        "",
        "## Configurations",
        "",
        "| Arm | Pipeline | Config hash | Retrieval / routing hash | Generator | T | Seed "
        "| Generation deterministic |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in m.arms:
        rh = a.retrieval_configuration_hash or a.routing_hash or "—"
        det = "—" if a.generation_deterministic is None else str(a.generation_deterministic).lower()
        lines.append(
            f"| {a.arm} | {a.pipeline} | `{a.config_hash[:16]}` | `{rh[:16]}` | "
            f"{a.generator or '—'} | {a.temperature if a.temperature is not None else '—'} | "
            f"{a.seed if a.seed is not None else '—'} | {det} |"
        )
    lines += [
        "",
        "## Metrics and statistics",
        "",
        f"- Registry `{m.metrics['registry']}`, ks {m.metrics['ks']}",
        f"- Statistics `{m.statistics['name']}@{m.statistics['version']}`: "
        f"{m.statistics['bootstrap_resamples']} bootstrap resamples, seed {m.statistics['seed']}, "
        f"{float(str(m.statistics['confidence'])):.0%} intervals, conclusions from "
        f"{m.statistics['min_cases_for_conclusion']} pairs",
        "",
        "## Artifacts",
        "",
        f"{len(m.artifacts)} trace artifacts (content-addressed, sha256), "
        f"{sum(a.bytes for a in m.artifacts)} bytes.",
        "",
        "## Replays",
        "",
    ]
    lines += [
        f"- `{r['id']}` {r['status']}, {r['cases']} cases: {r['outcomes']}" for r in m.replays
    ] or ["None yet."]
    raw = m.current.get("differences")
    diffs = [str(d) for d in raw] if isinstance(raw, list) else []
    lines += ["", "## This environment compared with the recording", ""]
    lines += [f"- {d}" for d in diffs] if diffs else ["No differences detected."]
    lines += ["", "## Notes", ""] + [f"- {n}" for n in m.notes]
    return "\n".join(lines) + "\n"


# --- exports --------------------------------------------------------------------------------


def comparisons(engine: ArenaEngine, run: ExperimentRun, e: Experiment) -> list[Comparison]:
    """The declared ablations, or each arm against the first when none were declared."""
    pairs = [(a.baseline, a.variant) for a in e.ablations] or [
        (run.arms[0], arm) for arm in run.arms[1:]
    ]
    return [engine.compare(run.id, b, run.id, v) for b, v in pairs]


def export_json(engine: ArenaEngine, run_id: str) -> dict[str, Any]:
    run = engine.get_run(run_id)
    _finished(run)
    e = engine.experiment(run.experiment_id)
    return {
        "format": EXPORT_FORMAT,
        "experiment": e.model_dump(mode="json"),
        "run": run.model_dump(mode="json"),
        "dataset": engine.dataset(run.dataset_id).model_dump(mode="json"),
        "cases": [c.model_dump(mode="json") for c in _ordered_cases(engine, run)],
        "comparisons": [c.model_dump(mode="json") for c in comparisons(engine, run, e)],
        "manifest": manifest(engine, run_id).model_dump(mode="json"),
    }


def _ordered_cases(engine: ArenaEngine, run: ExperimentRun) -> list[RunCase]:
    order = {c: i for i, c in enumerate(run.case_ids)}
    arms = {a: i for i, a in enumerate(run.arms)}
    return sorted(
        engine.arena.run_cases(run.id), key=lambda c: (order.get(c.case_id, 0), arms.get(c.arm, 0))
    )


def _csv(header: list[str], rows: list[list[Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def metrics_csv(engine: ArenaEngine, run_id: str) -> str:
    run = engine.get_run(run_id)
    _finished(run)
    header = ["run_id", "arm", "config_hash", "metric", "family", "version", "higher_is_better",
              "n", "skipped", "mean", "median", "std", "min", "max", "ci_low", "ci_high",
              "ci_method"]  # fmt: skip
    rows = [
        [run.id, s.arm, s.config_hash, m.metric, m.family.value, m.version, m.higher_is_better,
         m.n, m.skipped, m.mean, m.median, m.std, m.min, m.max, m.ci_low, m.ci_high, m.ci_method]
        for s in run.summaries
        for m in s.metrics
    ]  # fmt: skip
    return _csv(header, rows)


def cases_csv(engine: ArenaEngine, run_id: str) -> str:
    """One row per case x arm x metric (long format), failures and skips included."""
    run = engine.get_run(run_id)
    _finished(run)
    header = ["run_id", "arm", "case_id", "status", "config_hash", "retrieval_configuration_hash",
              "route", "answer_status", "error_type", "artifact_id", "metric", "family", "value",
              "skipped"]  # fmt: skip
    rows = []
    for c in _ordered_cases(engine, run):
        head = [run.id, c.arm, c.case_id, c.status.value, c.config_hash,
                c.retrieval_configuration_hash, c.route,
                c.answer_status.value if c.answer_status else None, c.error_type,
                c.artifact_id]  # fmt: skip
        for m in c.metrics:
            rows.append([*head, m.metric, m.family.value, m.value, m.skipped])
    return _csv(header, rows)


def report_markdown(engine: ArenaEngine, run_id: str) -> str:
    run = engine.get_run(run_id)
    _finished(run)
    e = engine.experiment(run.experiment_id)
    ds = engine.dataset(run.dataset_id)
    m = manifest(engine, run_id)
    cmps = comparisons(engine, run, e)
    cases = _ordered_cases(engine, run)
    dev = ds.source is DatasetSource.DEVELOPMENT
    L: list[str] = [
        f"# {e.name}",
        "",
        f"Research report for run `{run.id}` of experiment `{e.id}`, generated by RAG FORGE "
        f"from the recorded results. Status **{run.status.value}**: {run.completed} of "
        f"{run.total} evaluations, {run.failed} failed.",
        "",
        "> How to read this report. **Measured** sections contain values computed by this run "
        "from its recorded outputs. **Automatic proxy** metrics are computed against annotations "
        "and are not judgements of answer quality. The **interpretation** section only restates "
        "the recorded statistical conclusions; no qualitative judgement is generated.",
        "",
    ]
    if dev:
        L += [
            "> **Development benchmark.** This dataset is small and saturated. It validates the "
            "pipeline; it is not evidence that one method is better than another.",
            "",
        ]
    L += ["## Hypothesis", "", e.hypothesis or "None was recorded.", ""]
    L += [
        "## Dataset",
        "",
        f"- {ds.name} v{ds.version} ({ds.source.value}), {len(run.case_ids)} of {len(ds.cases)} "
        f"cases evaluated, content hash `{ds.content_hash}`",
        f"- Corpus `{ds.corpus_id}` v{ds.corpus_version}",
        "- Annotations: " + ", ".join(f"{k.value} {v}" for k, v in ds.annotation_counts.items()),
        f"- Ground truth: {ds.annotation_notes or 'not described'}",
        "",
        "## Configurations",
        "",
        "| Arm | Label | Pipeline | Retrieval | Models | Config hash |",
        "|---|---|---|---|---|---|",
    ]
    labels = {a.name: a.label for a in e.arms}
    for s in e.snapshots:
        t = s.retrieval_template
        retr = (
            f"adaptive ({s.router_policy})"
            if s.retrieval_mode.value == "adaptive"
            else f"{t.strategy.value}, top_k {t.top_k}" + (" + rerank" if t.rerank.enabled else "")
        )
        models = ", ".join(
            x
            for x in (
                s.embedder.model if s.embedder else None,
                s.reranker.model if s.reranker else None,
                s.generator.model if s.generator else None,
            )
            if x
        )
        L.append(
            f"| {s.arm} | {labels.get(s.arm, '')} | {s.pipeline.value} | {retr} | "
            f"{models or 'lexical only'} | `{run.snapshot_hashes[s.arm][:16]}` |"
        )
    if e.ablations:
        L += ["", "Declared ablations:", ""]
        for a in e.ablations:
            conf = "single factor" if a.single_factor else f"confounded: {', '.join(a.factors)}"
            L.append(f"- {a.baseline} → {a.variant}: {a.factor} ({conf})")
    for fam in MetricFamily:
        names = [
            x.metric
            for x in (run.summaries[0].metrics if run.summaries else [])
            if x.family is fam and any(_summary(run, arm, x.metric) for arm in run.arms)
        ]
        names = [n for n in names if any((_summary(run, a, n) or {}).get("n") for a in run.arms)]
        if not names or (fam is MetricFamily.OPERATIONAL):
            continue
        kind = "automatic proxies" if fam is MetricFamily.GENERATION else "measured"
        L += [
            "",
            f"## {FAMILY_TITLE[fam]} ({kind})",
            "",
            "Mean [95% bootstrap CI] (n defined / skipped).",
            "",
            "| Metric | " + " | ".join(run.arms) + " |",
            "|---" * (len(run.arms) + 1) + "|",
        ]
        for n in names:
            cells = []
            for arm in run.arms:
                sm = _summary(run, arm, n)
                if not sm or not sm["n"]:
                    cells.append("—")
                    continue
                ci = (
                    f" [{fmt(sm['ci_low'], n)}, {fmt(sm['ci_high'], n)}]"
                    if sm["ci_low"] is not None
                    else ""
                )
                cells.append(f"{fmt(sm['mean'], n)}{ci} ({sm['n']}/{sm['skipped']})")
            L.append(f"| {n} | " + " | ".join(cells) + " |")
    L += [
        "",
        "## Latency and failures (measured, this machine)",
        "",
        "| Arm | Mean total | p50 | p95 | Failed | Failure types |",
        "|---|---|---|---|---|---|",
    ]
    for sm_arm in run.summaries:
        lat = sorted(
            c.timings_ms["total"]
            for c in cases
            if c.arm == sm_arm.arm and c.status is CaseStatus.OK and "total" in c.timings_ms
        )
        mean = sum(lat) / len(lat) if lat else None
        types = ", ".join(f"{k} x {v}" for k, v in sm_arm.failure_types.items()) or "—"
        L.append(
            f"| {sm_arm.arm} | {fmt(mean, 'latency_ms')} | {fmt(_pct(lat, 0.5), 'latency_ms')} | "
            f"{fmt(_pct(lat, 0.95), 'latency_ms')} | {sm_arm.failed}/{sm_arm.cases} | {types} |"
        )
    L += ["", "## Statistical comparisons (measured)", ""]
    L.append(
        f"Method `{stats.METHOD_ID}`: paired differences over the cases both arms define, "
        f"{stats.RESAMPLES} bootstrap resamples (seed {stats.SEED}), exact sign test with Holm "
        f"adjustment across the metrics of each comparison, no conclusion below "
        f"{stats.MIN_CASES_FOR_CONCLUSION} pairs."
    )
    for c in cmps:
        L += [
            "",
            f"### {c.baseline.arm} → {c.variant.arm}" + (f" ({c.factor})" if c.factor else ""),
        ]
        L += [""] + [f"- Not comparable: {i}" for i in c.issues]
        L += [f"- Note: {w}" for w in c.warnings]
        if not c.comparable:
            continue
        L += [
            "",
            "| Metric | n | Baseline | Variant | Δ mean [95% CI] | dz | W/L/T | sign p (Holm) | "
            "Conclusion |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for p in c.metrics:
            if p.n_pairs == 0:
                continue
            ci = (
                f" [{signed(p.ci_low, p.metric)}, {signed(p.ci_high, p.metric)}]"
                if p.ci_low is not None
                else ""
            )
            wlt = "—" if p.higher_is_better is None else f"{p.wins}/{p.losses}/{p.ties}"
            sp = "—" if p.sign_test_p is None else f"{p.sign_test_p:.3f} ({p.holm_p:.3f})"
            dz = "—" if p.effect_size_dz is None else f"{p.effect_size_dz:.2f}"
            L.append(
                f"| {p.metric} | {p.n_pairs} | {fmt(p.baseline_mean, p.metric)} | "
                f"{fmt(p.variant_mean, p.metric)} | {signed(p.mean_difference, p.metric)}{ci} | "
                f"{dz} | {wlt} | {sp} | {p.conclusion} |"
            )
    primary = next(
        (n for n in PRIMARY if any((_summary(run, a, n) or {}).get("n") for a in run.arms)), None
    )
    if primary:
        L += [
            "",
            f"## Per-case results: {primary} (measured)",
            "",
            "| Case | " + " | ".join(run.arms) + " |",
            "|---" * (len(run.arms) + 1) + "|",
        ]
        grid = {(c.case_id, c.arm): c for c in cases}
        for cid in run.case_ids:
            cells = []
            for arm in run.arms:
                rc = grid.get((cid, arm))
                if rc is None:
                    cells.append("—")
                elif rc.status is CaseStatus.FAILED:
                    cells.append(f"failed ({rc.error_type})")
                else:
                    v = next((x for x in rc.metrics if x.metric == primary), None)
                    cells.append(fmt(v.value, primary) if v and v.value is not None else "skipped")
            L.append(f"| {cid} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("Every metric of every case is in the CSV and JSON exports.")
    L += ["", "## Skipped metrics", ""]
    skips = [
        f"- {s.arm}: {name} skipped where {reason}" for s in run.summaries
        for name, reason in s.skipped_reasons.items()
    ]  # fmt: skip
    L += skips[:60] or ["None."]
    if len(skips) > 60:
        L.append(f"- … {len(skips) - 60} more in the JSON export")
    L += [
        "",
        "## Interpretation (restated from the recorded conclusions)",
        "",
    ]
    statements = []
    for c in cmps:
        if not c.comparable:
            statements.append(f"- {c.baseline.arm} → {c.variant.arm}: not comparable.")
            continue
        found = [p for p in c.metrics if p.conclusion in ("variant_higher", "variant_lower")]
        for p in found:
            higher = p.conclusion == "variant_higher"
            better = p.higher_is_better is higher
            statements.append(
                f"- On this dataset, {c.variant.arm} has a {'higher' if higher else 'lower'} mean "
                f"{p.metric} than {c.baseline.arm} ({signed(p.mean_difference, p.metric)}, 95% CI "
                f"[{signed(p.ci_low, p.metric)}, {signed(p.ci_high, p.metric)}], "
                f"{p.n_pairs} pairs): "
                f"{'better' if better else 'worse'} by the metric's direction."
            )
        flat = sum(p.conclusion == "no_detectable_difference" for p in c.metrics)
        few = sum(p.conclusion == "insufficient_cases" and p.n_pairs > 0 for p in c.metrics)
        if flat:
            statements.append(
                f"- {c.baseline.arm} → {c.variant.arm}: no detectable difference on "
                f"{flat} metric(s)."
            )
        if few:
            statements.append(
                f"- {c.baseline.arm} → {c.variant.arm}: {few} metric(s) had too few paired cases "
                "for a conclusion."
            )
    L += statements or ["- No comparison was possible."]
    L += [
        "",
        "These statements describe this dataset and these configurations only.",
        "",
        "### Qualitative interpretation",
        "",
        "None generated. RAG FORGE does not judge answer quality; add the researcher's reading "
        "here.",
        "",
        "## Limitations",
        "",
    ]
    lims = []
    if dev:
        lims.append(
            "The development benchmark is small and saturated: most retrieval arms reach the "
            "ceiling, so differences are mostly not measurable."
        )
    if len(run.case_ids) < stats.MIN_CASES_FOR_CONCLUSION:
        lims.append(f"Only {len(run.case_ids)} cases: below the threshold for any conclusion.")
    if run.failed:
        lims.append(f"{run.failed} evaluations failed; quality metrics pair only completed cases.")
    lims += [
        "Grounding metrics measure support by the supplied evidence (lexical-semantic "
        "verifier), not factual truth; the verifier cannot detect contradiction.",
        "Token F1 and abstention accuracy are automatic proxies against annotations.",
        "Latency is wall time on the recording machine and includes first-use model loading.",
    ]
    L += [f"- {x}" for x in lims]
    rt = m.runtime
    L += [
        "",
        "## Reproducibility",
        "",
        f"- Git commit `{rt.git_commit if rt else m.environment.git_commit}`"
        + (f", uncommitted changes: {'yes' if rt.git_dirty else 'no'}" if rt else ""),
        f"- Python {m.environment.python_version} on {m.environment.platform}"
        + (f"; Node {rt.node_version}; uv {rt.uv_version}" if rt else ""),
        f"- Metric registry `{run.metric_registry_version}`, statistics `{run.stats_method}`",
        f"- Seeds: bootstrap {stats.SEED}; generation " + (
            ", ".join(f"{a.arm} {a.seed} (T={a.temperature})" for a in m.arms if a.generator)
            or "none"
        ),
    ]  # fmt: skip
    for mr in m.models:
        weights = next((f for f in mr.files if f.path.endswith(".onnx")), None)
        sha = f", weights sha256 `{weights.sha256[:16]}…`" if weights and weights.sha256 else ""
        L.append(f"- {mr.role}: {mr.model}@{(mr.revision or '—')[:12]}{sha}")
    L += [
        f"- {len(m.artifacts)} content-addressed trace artifacts; full manifest at "
        f"`GET /api/v1/runs/{run.id}/manifest`",
        f"- Replays: {len(m.replays)}"
        + (
            " (latest: " + ", ".join(f"{k} {v}" for k, v in m.replays[0]["outcomes"].items()) + ")"  # type: ignore[attr-defined]
            if m.replays
            else ""
        ),
        "",
    ]
    return "\n".join(L)


def _summary(run: ExperimentRun, arm: str, metric: str) -> dict[str, Any] | None:
    s = next((s for s in run.summaries if s.arm == arm), None)
    m = next((x for x in s.metrics if x.metric == metric), None) if s else None
    return m.model_dump() if m else None


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    i = (len(values) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


__all__ = [
    "cases_csv",
    "export_json",
    "manifest",
    "manifest_markdown",
    "metrics_csv",
    "report_markdown",
]
