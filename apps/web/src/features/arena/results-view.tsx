"use client";

import type { ArmSummary, CaseView, Comparison, Experiment, ExperimentRun, Leaderboard, MetricFamily, RunCase } from "@rag-forge/shared";
import { AlertTriangle, ExternalLink, FlaskConical } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { Badge, OriginBadge } from "@/components/ui/badge";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { StatusIndicator } from "@/components/ui/status";
import { Tabs } from "@/components/ui/tabs";
import { API_URL, api } from "@/lib/api";
import { shortHash } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { DiffPlot, IntervalPlot, StripPlot } from "./charts";
import { FAMILIES, FAMILY_LABEL, RUN_STATUS, armColor, domain, fmt, metricsByFamily, percentile, signed, verdict } from "./format";

type View = "leaderboard" | "compare" | "cases" | "failures" | "latency" | "configs";
const VIEWS: { id: View; label: string }[] = [
  { id: "leaderboard", label: "Leaderboard" },
  { id: "compare", label: "Statistical comparison" },
  { id: "cases", label: "Case explorer" },
  { id: "failures", label: "Failures & skips" },
  { id: "latency", label: "Latency" },
  { id: "configs", label: "Configurations" },
];

const finished = (r: ExperimentRun) => r.status === "completed" || r.status === "partial";

export function ResultsView({ runId }: { runId: string }) {
  const fetchRun = useCallback(() => api.GET("/api/v1/runs/{run_id}", { params: { path: { run_id: runId } } }), [runId]);
  const pollRun = useCallback((r: ExperimentRun | undefined) => (r && (r.status === "queued" || r.status === "running") ? 1500 : undefined), []);
  const run = useApi(fetchRun, pollRun);
  if (!run.data) return <Panel>{run.error ? <ErrorState title="Could not load the run">{run.error}</ErrorState> : <LoadingState rows={6} label="Loading the run" />}</Panel>;
  return <RunResults key={run.data.experiment_id} run={run.data} />;
}

function RunResults({ run }: { run: ExperimentRun }) {
  const fetchExp = useCallback(() => api.GET("/api/v1/experiments/{experiment_id}", { params: { path: { experiment_id: run.experiment_id } } }), [run.experiment_id]);
  const exp = useApi(fetchExp);
  const [view, setView] = useState<View>("leaderboard");
  if (!exp.data) return <Panel>{exp.error ? <ErrorState title="Could not load the experiment">{exp.error}</ErrorState> : <LoadingState rows={4} label="Loading the experiment" />}</Panel>;
  const e = exp.data;
  return (
    <div className="space-y-5">
      <RunHeader run={run} exp={e} />
      {!finished(run) ? (
        <Panel>
          <EmptyState icon={FlaskConical} title={run.status === "failed" ? "The run could not proceed" : "Nothing measured yet"}>
            {run.status === "failed" ? run.error ?? "The run failed before producing results." : `The run is ${run.status}: results appear when every case of every arm has been evaluated.`}
          </EmptyState>
        </Panel>
      ) : (
        <>
          <Tabs tabs={VIEWS} value={view} onChange={(v) => setView(v as View)} />
          {view === "leaderboard" && <LeaderboardView run={run} />}
          {view === "compare" && <CompareView run={run} exp={e} />}
          {view === "cases" && <CasesView run={run} />}
          {view === "failures" && <FailuresView run={run} />}
          {view === "latency" && <LatencyView run={run} />}
          {view === "configs" && <ConfigsView run={run} exp={e} />}
        </>
      )}
    </div>
  );
}

function RunHeader({ run, exp }: { run: ExperimentRun; exp: Experiment }) {
  const pct = (run.completed / Math.max(1, run.total)) * 100;
  return (
    <Panel>
      <PanelHeader
        eyebrow={`Run ${run.id}`}
        title={exp.name}
        actions={<><OriginBadge origin="measured" /><StatusIndicator status={RUN_STATUS[run.status]} label={run.status} /></>}
      />
      <div className="grid gap-4 p-4 lg:grid-cols-[1fr_auto]">
        <div className="space-y-2 text-[12px] leading-relaxed text-fg-muted">
          {exp.hypothesis ? <p><span className="text-fg">Hypothesis:</span> {exp.hypothesis}</p> : <p className="text-fg-subtle">No hypothesis was recorded.</p>}
          <p className="font-mono text-[11px] text-fg-subtle">
            {exp.dataset_name} v{run.dataset_version} ({shortHash(run.dataset_hash, 8)}) · corpus {run.corpus_id.slice(0, 12)} v{run.corpus_version} · {run.case_ids.length} cases × {run.arms.length} arms · {run.metric_registry_version} · {run.stats_method}
          </p>
          {exp.dataset_source === "development" && (
            <p className="flex items-start gap-1.5 text-signal"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" /> Development benchmark: small and easy. These numbers show the pipeline works; they are not evidence that one method is better.</p>
          )}
        </div>
        <div className="w-56">
          <div className="flex h-1.5 overflow-hidden rounded-full bg-surface-3">
            <span className="bg-ok" style={{ width: `${((run.completed - run.failed) / Math.max(1, run.total)) * 100}%` }} />
            <span className="bg-err" style={{ width: `${(run.failed / Math.max(1, run.total)) * 100}%` }} />
          </div>
          <div className="num mt-1 font-mono text-[10px] text-fg-subtle">{run.completed}/{run.total} evaluated ({pct.toFixed(0)}%) · {run.failed} failed</div>
        </div>
      </div>
    </Panel>
  );
}

function MetricPicker({ run, value, onChange }: { run: ExperimentRun; value: string; onChange: (m: string) => void }) {
  const groups = metricsByFamily(run.summaries.map((s) => s.metrics));
  return (
    <select aria-label="Metric" value={value} onChange={(e) => onChange(e.target.value)} className="h-7 rounded-md border border-line-strong bg-surface px-2 font-mono text-[11px]">
      {FAMILIES.filter((f) => groups[f].length).map((f) => (
        <optgroup key={f} label={FAMILY_LABEL[f]}>
          {groups[f].map((m) => <option key={m} value={m}>{m}</option>)}
        </optgroup>
      ))}
    </select>
  );
}

/** The first defined metric among sensible defaults, so the first view never shows an empty metric. */
function defaultMetric(run: ExperimentRun): string {
  const groups = metricsByFamily(run.summaries.map((s) => s.metrics));
  const all = FAMILIES.flatMap((f) => groups[f]);
  return ["ndcg@10", "mrr", "grounding_score", "latency_ms"].find((m) => all.includes(m)) ?? all[0] ?? "latency_ms";
}

const armIndex = (run: ExperimentRun, arm: string) => run.arms.indexOf(arm);

// --- leaderboard --------------------------------------------------------------------------------

function LeaderboardView({ run }: { run: ExperimentRun }) {
  const [metric, setMetric] = useState(() => defaultMetric(run));
  const fetcher = useCallback(() => api.GET("/api/v1/runs/{run_id}/leaderboard", { params: { path: { run_id: run.id }, query: { metric } } }), [run.id, metric]);
  const lb = useApi(fetcher);
  return (
    <div className="space-y-5">
      <MetricCards run={run} />
      <Panel>
        <PanelHeader eyebrow="Leaderboard" title="Mean with 95% bootstrap interval, per arm" actions={<MetricPicker run={run} value={metric} onChange={setMetric} />} />
        {!lb.data ? (
          lb.error ? <ErrorState title="Could not build the leaderboard">{lb.error}</ErrorState> : <LoadingState rows={4} label="Ranking" />
        ) : (
          <LeaderboardTable run={run} lb={lb.data} />
        )}
      </Panel>
    </div>
  );
}

function LeaderboardTable({ run, lb }: { run: ExperimentRun; lb: Leaderboard }) {
  const dom = domain(lb.rows.flatMap((r) => [r.summary?.ci_low, r.summary?.ci_high, r.summary?.mean]));
  return (
    <div>
      <div className="grid gap-4 p-4 lg:grid-cols-[1fr_380px]">
        <table className="w-full text-left text-[12px]">
          <thead className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
            <tr className="border-b border-line">
              <th className="py-2 font-normal">#</th>
              <th className="font-normal">arm</th>
              <th className="text-right font-normal">mean</th>
              <th className="text-right font-normal">95% CI</th>
              <th className="text-right font-normal">n / skipped</th>
              <th className="text-right font-normal">failed</th>
              <th className="text-right font-normal">mean latency</th>
            </tr>
          </thead>
          <tbody>
            {lb.rows.map((r) => (
              <tr key={r.arm} className="border-b border-line/60 last:border-0">
                <td className="py-2 font-mono">{r.position ?? "—"}</td>
                <td>
                  <span className="flex items-center gap-2">
                    <span className="size-2 rounded-full" style={{ background: armColor(armIndex(run, r.arm)) }} />
                    <span>{r.label || r.arm}</span>
                    {r.ci_overlaps_leader && r.position !== 1 && <Badge title="Its confidence interval overlaps the leader's">overlaps #1</Badge>}
                  </span>
                </td>
                <td className="num text-right">{fmt(r.summary?.mean, lb.metric)}</td>
                <td className="num text-right text-fg-muted">{r.summary?.ci_low != null ? `${fmt(r.summary.ci_low, lb.metric)} – ${fmt(r.summary.ci_high, lb.metric)}` : "—"}</td>
                <td className="num text-right text-fg-muted">{r.summary ? `${r.summary.n} / ${r.summary.skipped}` : "—"}</td>
                <td className={`num text-right ${r.failed ? "text-err" : "text-fg-muted"}`}>{r.failed}/{r.cases}</td>
                <td className="num text-right text-fg-muted">{fmt(r.mean_latency_ms, "latency_ms")}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <IntervalPlot
          label={`${lb.metric} by arm`}
          domain={dom}
          format={(v) => fmt(v, lb.metric)}
          rows={lb.rows.map((r) => ({ key: r.arm, label: r.label || r.arm, color: armColor(armIndex(run, r.arm)), mean: r.summary?.mean ?? null, low: r.summary?.ci_low ?? null, high: r.summary?.ci_high ?? null }))}
        />
      </div>
      <p className="border-t border-line px-4 py-2.5 text-[11px] leading-relaxed text-fg-subtle">{lb.note}</p>
    </div>
  );
}

/** Headline measured values per arm for a few metrics that are defined in this run. */
function MetricCards({ run }: { run: ExperimentRun }) {
  const groups = metricsByFamily(run.summaries.map((s) => s.metrics));
  const all = FAMILIES.flatMap((f) => groups[f]);
  const picks = ["ndcg@10", "mrr", "recall@5", "grounding_score", "abstention_accuracy", "latency_ms"].filter((m) => all.includes(m)).slice(0, 4);
  if (!picks.length) return null;
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {picks.map((m) => (
        <Panel key={m} className="p-4">
          <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-fg-muted">{m}</div>
          <ul className="mt-2 space-y-1">
            {run.summaries.map((s, i) => {
              const v = s.metrics.find((x) => x.metric === m);
              return (
                <li key={s.arm} className="flex items-center justify-between gap-2 text-[12px]">
                  <span className="flex min-w-0 items-center gap-1.5"><span className="size-1.5 shrink-0 rounded-full" style={{ background: armColor(i) }} /><span className="truncate text-fg-muted">{s.arm}</span></span>
                  <span className="num">{v && v.n ? fmt(v.mean, m) : <span className="text-fg-subtle" title={s.skipped_reasons[m] ?? "not defined for this arm"}>n/a</span>}</span>
                </li>
              );
            })}
          </ul>
        </Panel>
      ))}
    </div>
  );
}

// --- comparison ---------------------------------------------------------------------------------

function CompareView({ run, exp }: { run: ExperimentRun; exp: Experiment }) {
  const first = exp.ablations[0];
  const [pair, setPair] = useState<{ baseline: string; variant: string }>(first ? { baseline: first.baseline, variant: first.variant } : { baseline: run.arms[0], variant: run.arms[1] ?? run.arms[0] });
  const same = pair.baseline === pair.variant;
  if (run.arms.length < 2)
    return <Panel><EmptyState icon={FlaskConical} title="One arm only">A paired comparison needs two arms of the same run.</EmptyState></Panel>;
  return (
    <div className="space-y-5">
      <Panel>
        <PanelHeader eyebrow="Pair" title="Variant against baseline, over the cases both evaluated" />
        <div className="flex flex-wrap items-center gap-3 p-4">
          {exp.ablations.map((a) => (
            <button key={`${a.baseline}>${a.variant}`} type="button" aria-pressed={pair.baseline === a.baseline && pair.variant === a.variant}
              onClick={() => setPair({ baseline: a.baseline, variant: a.variant })}
              className={`rounded-md px-2.5 py-1 font-mono text-[11px] ring-1 ring-inset ${pair.baseline === a.baseline && pair.variant === a.variant ? "bg-surface-3 text-fg ring-line-strong" : "text-fg-subtle ring-line hover:text-fg-muted"}`}>
              {a.baseline} → {a.variant} <span className="text-fg-subtle">· {a.factor}</span>
            </button>
          ))}
          <span className="flex items-center gap-1.5 font-mono text-[11px]">
            {(["baseline", "variant"] as const).map((side) => (
              <select key={side} aria-label={side} value={pair[side]} onChange={(e) => setPair((p) => ({ ...p, [side]: e.target.value }))} className="h-7 rounded-md border border-line-strong bg-surface px-2">
                {run.arms.map((a) => <option key={a} value={a}>{a}</option>)}
              </select>
            ))}
          </span>
        </div>
      </Panel>
      {same ? <Panel><p className="p-4 text-[12px] text-fg-subtle">Choose two different arms.</p></Panel> : <ComparisonResult key={`${pair.baseline}>${pair.variant}`} runId={run.id} baseline={pair.baseline} variant={pair.variant} />}
    </div>
  );
}

function ComparisonResult({ runId, baseline, variant }: { runId: string; baseline: string; variant: string }) {
  const fetcher = useCallback(
    () => api.GET("/api/v1/runs/{run_id}/compare", { params: { path: { run_id: runId }, query: { baseline, variant } } }),
    [runId, baseline, variant],
  );
  const cmp = useApi(fetcher);
  const [focus, setFocus] = useState<string | null>(null);
  if (!cmp.data) return <Panel>{cmp.error ? <ErrorState title="Could not compare">{cmp.error}</ErrorState> : <LoadingState rows={6} label="Bootstrapping" />}</Panel>;
  const c: Comparison = cmp.data;
  const focused = c.metrics.find((m) => m.metric === focus) ?? c.metrics.find((m) => m.n_pairs > 0 && m.higher_is_better !== null) ?? c.metrics[0];
  return (
    <div className="space-y-5">
      {(c.issues.length > 0 || c.warnings.length > 0) && (
        <Panel className="space-y-1.5 p-4 text-[12px]">
          {c.issues.map((i) => <p key={i} className="flex gap-1.5 text-err"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" />{i}</p>)}
          {c.warnings.map((w) => <p key={w} className="flex gap-1.5 text-signal"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" />{w}</p>)}
        </Panel>
      )}
      <Panel>
        <PanelHeader eyebrow={c.factor ? `Declared factor: ${c.factor}` : "Undeclared pair"} title={`What differs between ${c.baseline.arm} and ${c.variant.arm}`} />
        {c.changes.length ? (
          <div className="max-h-60 overflow-auto">
            <table className="w-full text-left font-mono text-[11px]">
              <tbody>
                {c.changes.map((ch) => (
                  <tr key={ch.path} className="border-b border-line/60 last:border-0">
                    <td className="px-4 py-1.5 text-fg-muted">{ch.path}</td>
                    <td className="py-1.5 pr-3 text-fg-subtle">{JSON.stringify(ch.baseline)}</td>
                    <td className="py-1.5 pr-4 text-fg">{JSON.stringify(ch.variant)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="p-4 text-[12px] text-fg-subtle">The two snapshots are identical apart from the arm name.</p>
        )}
      </Panel>
      {!c.comparable ? (
        <Panel><EmptyState icon={AlertTriangle} title="Not comparable">No paired statistics are computed for these arms.</EmptyState></Panel>
      ) : (
        <div className="grid items-start gap-5 xl:grid-cols-[1fr_440px]">
          <Panel>
            <PanelHeader eyebrow={`${c.method.name}@${c.method.version} · ${Math.round(c.method.confidence * 100)}% · ${c.method.bootstrap_resamples} resamples`} title="Paired differences (variant − baseline)" />
            <div className="overflow-x-auto">
              <table className="w-full text-left text-[12px]">
                <thead className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
                  <tr className="border-b border-line">
                    <th className="px-4 py-2 font-normal">metric</th>
                    <th className="text-right font-normal">n</th>
                    <th className="text-right font-normal">baseline</th>
                    <th className="text-right font-normal">variant</th>
                    <th className="text-right font-normal">Δ mean [CI]</th>
                    <th className="text-right font-normal">dz</th>
                    <th className="text-right font-normal">W/L/T</th>
                    <th className="text-right font-normal">sign p (Holm)</th>
                    <th className="px-4 font-normal">reading</th>
                  </tr>
                </thead>
                <tbody>
                  {c.metrics.map((m) => {
                    const v = verdict(m, c.method.min_cases_for_conclusion);
                    return (
                      <tr key={m.metric} onClick={() => setFocus(m.metric)} className={`cursor-pointer border-b border-line/60 last:border-0 hover:bg-surface-2 ${focused?.metric === m.metric ? "bg-surface-2" : ""}`}>
                        <td className="px-4 py-1.5 font-mono text-[11px]">{m.metric}</td>
                        <td className="num text-right">{m.n_pairs}</td>
                        <td className="num text-right text-fg-muted">{fmt(m.baseline_mean, m.metric)}</td>
                        <td className="num text-right text-fg-muted">{fmt(m.variant_mean, m.metric)}</td>
                        <td className="num whitespace-nowrap text-right">{signed(m.mean_difference, m.metric)}{m.ci_low != null && <span className="text-fg-subtle"> [{signed(m.ci_low, m.metric)}, {signed(m.ci_high, m.metric)}]</span>}</td>
                        <td className="num text-right text-fg-muted">{m.effect_size_dz != null ? m.effect_size_dz.toFixed(2) : "—"}</td>
                        <td className="num text-right text-fg-muted">{m.higher_is_better === null ? "—" : `${m.wins}/${m.losses}/${m.ties}`}</td>
                        <td className="num whitespace-nowrap text-right text-fg-muted">{m.sign_test_p != null ? `${m.sign_test_p.toFixed(3)} (${m.holm_p?.toFixed(3) ?? "—"})` : "—"}</td>
                        <td className="px-4"><Badge tone={v.tone} title={v.help}>{v.label}</Badge></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="border-t border-line px-4 py-2.5 text-[11px] leading-relaxed text-fg-subtle">
              W/L/T count cases where the variant is better / worse / equal, by the metric&apos;s direction. Conclusions need at least {c.method.min_cases_for_conclusion} pairs;
              &ldquo;higher&rdquo; describes the value, and the CI excluding 0 is a statement about this dataset only. {c.method.multiple_comparisons}
            </p>
          </Panel>
          {focused && (
            <Panel>
              <PanelHeader eyebrow="Per case" title={`${focused.metric}: ${focused.n_pairs} paired cases`} />
              <div className="p-4">
                {focused.differences.length ? (
                  <DiffPlot differences={focused.differences} higherIsBetter={focused.higher_is_better} format={(v) => fmt(v, focused.metric)} />
                ) : (
                  <p className="text-[12px] text-fg-subtle">No case defines this metric in both arms.</p>
                )}
                <p className="mt-2 text-[11px] text-fg-subtle">Sorted differences; green where the variant is better by the metric&apos;s direction. Hover for the case.</p>
              </div>
            </Panel>
          )}
        </div>
      )}
    </div>
  );
}

// --- cases --------------------------------------------------------------------------------------

function useRunCases(runId: string) {
  const fetcher = useCallback(() => api.GET("/api/v1/runs/{run_id}/cases", { params: { path: { run_id: runId }, query: {} } }), [runId]);
  return useApi(fetcher);
}

const valueOf = (c: RunCase, metric: string) => c.metrics.find((m) => m.metric === metric);

function CasesView({ run }: { run: ExperimentRun }) {
  const cases = useRunCases(run.id);
  const [metric, setMetric] = useState(() => defaultMetric(run));
  const [open, setOpen] = useState<string | null>(null);
  const grid = useMemo(() => {
    const m = new Map<string, RunCase>();
    for (const c of cases.data ?? []) m.set(`${c.case_id}|${c.arm}`, c);
    return m;
  }, [cases.data]);
  if (!cases.data) return <Panel>{cases.error ? <ErrorState title="Could not load cases">{cases.error}</ErrorState> : <LoadingState rows={8} label="Loading cases" />}</Panel>;
  return (
    <div className="grid items-start gap-5 2xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <Panel>
        <PanelHeader eyebrow="Case × arm" title="Per-case values (click a row)" actions={<MetricPicker run={run} value={metric} onChange={setMetric} />} />
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[12px]">
            <thead className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
              <tr className="border-b border-line">
                <th className="px-4 py-2 font-normal">case</th>
                {run.arms.map((a, i) => <th key={a} className="px-2 text-right font-normal"><span className="inline-flex items-center gap-1"><span className="size-1.5 rounded-full" style={{ background: armColor(i) }} />{a}</span></th>)}
              </tr>
            </thead>
            <tbody>
              {run.case_ids.map((id) => (
                <tr key={id} tabIndex={0} onClick={() => setOpen(id)} onKeyDown={(e) => e.key === "Enter" && setOpen(id)} className={`cursor-pointer border-b border-line/60 last:border-0 hover:bg-surface-2 ${open === id ? "bg-surface-2" : ""}`}>
                  <td className="px-4 py-1.5 font-mono text-[11px]">{id}</td>
                  {run.arms.map((a) => {
                    const c = grid.get(`${id}|${a}`);
                    const v = c && valueOf(c, metric);
                    return (
                      <td key={a} className="num px-2 text-right">
                        {!c ? <span className="text-fg-subtle">—</span> : c.status === "failed" ? <span className="text-err" title={c.error ?? ""}>failed</span> : v?.value != null ? fmt(v.value, metric) : <span className="text-fg-subtle" title={v?.skipped ?? "not computed"}>skip</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      {open ? <CaseDetail key={open} run={run} caseId={open} /> : <Panel><p className="p-4 text-[12px] text-fg-subtle">Select a case to see its annotations and every arm&apos;s ranking, evidence, answer and metrics side by side.</p></Panel>}
    </div>
  );
}

function CaseDetail({ run, caseId }: { run: ExperimentRun; caseId: string }) {
  const fetcher = useCallback(() => api.GET("/api/v1/runs/{run_id}/cases/{case_id}", { params: { path: { run_id: run.id, case_id: caseId } } }), [run.id, caseId]);
  const view = useApi(fetcher);
  const [arms, setArms] = useState<[string, string]>([run.arms[0], run.arms[1] ?? run.arms[0]]);
  if (!view.data) return <Panel>{view.error ? <ErrorState title="Could not load the case">{view.error}</ErrorState> : <LoadingState rows={6} label="Loading the case" />}</Panel>;
  const v: CaseView = view.data;
  const c = v.case;
  const shown = [...new Set(arms)].map((a) => v.results.find((r) => r.arm === a)).filter((r): r is RunCase => Boolean(r));
  return (
    <Panel>
      <PanelHeader eyebrow={`Case ${c.id}`} title={c.query} />
      <div className="space-y-1 border-b border-line px-4 py-3 text-[12px] text-fg-muted">
        <p><span className="text-fg-subtle">Relevant:</span> {c.answerable === false ? "unanswerable from the corpus" : Object.entries({ ...c.relevant_documents, ...c.relevant_chunks }).map(([k, g]) => `${k} (${g})`).join(", ") || "not judged"}</p>
        <p><span className="text-fg-subtle">Reference answer:</span> {c.reference_answer ?? "none"}</p>
        {c.expected_evidence.length > 0 && <p><span className="text-fg-subtle">Expected evidence:</span> {c.expected_evidence.join(", ")}</p>}
      </div>
      <div className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-2 font-mono text-[11px] text-fg-subtle">
        compare
        {[0, 1].map((i) => (
          <select key={i} aria-label={`Arm ${i + 1}`} value={arms[i]} onChange={(e) => setArms((p) => (i === 0 ? [e.target.value, p[1]] : [p[0], e.target.value]))} className="h-7 rounded-md border border-line-strong bg-surface px-2">
            {run.arms.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        ))}
      </div>
      <div className={`grid divide-line ${shown.length > 1 ? "md:grid-cols-2 md:divide-x" : ""}`}>
        {shown.map((r) => <ArmResult key={r.arm} r={r} color={armColor(armIndex(run, r.arm))} />)}
      </div>
    </Panel>
  );
}

function ArmResult({ r, color }: { r: RunCase; color: string }) {
  const measured = r.metrics.filter((m) => m.value !== null && m.metric !== "failed");
  const skipped = r.metrics.filter((m) => m.skipped);
  return (
    <div className="min-w-0 space-y-3 p-4 text-[12px]">
      <div className="flex items-center gap-2">
        <span className="size-2 rounded-full" style={{ background: color }} />
        <span className="font-mono text-fg">{r.arm}</span>
        {r.route && <Badge tone="trace" title="The option the router chose">{r.route}</Badge>}
        <span className="ml-auto font-mono text-[10px] text-fg-subtle" title={r.config_hash}>{shortHash(r.config_hash, 8)}</span>
      </div>
      {r.status === "failed" ? (
        <p role="alert" className="text-err"><span className="font-mono">{r.error_type}</span>: {r.error}</p>
      ) : (
        <>
          <ol className="space-y-0.5 font-mono text-[11px]">
            {r.ranking.slice(0, 10).map((h) => (
              <li key={h.chunk_id} className="flex items-center gap-2">
                <span className="w-5 text-right text-fg-subtle">{h.rank}</span>
                <span className={`min-w-0 flex-1 truncate ${h.relevance ? "text-ok" : h.relevance === 0 ? "text-fg-subtle line-through" : "text-fg-muted"}`} title={h.chunk_id}>{h.filename}</span>
                {h.upstream_rank != null && h.upstream_rank !== h.rank && <span className={h.upstream_rank > h.rank ? "text-ok" : "text-err"} title={`rank ${h.upstream_rank} before reranking`}>{h.upstream_rank > h.rank ? "▲" : "▼"}{Math.abs(h.upstream_rank - h.rank)}</span>}
                {h.relevance != null && <span className="text-fg-subtle">·{h.relevance}</span>}
              </li>
            ))}
            {!r.ranking.length && <li className="text-fg-subtle">no results</li>}
          </ol>
          {r.answer_status && (
            <div className="space-y-1 rounded-md bg-surface-2 p-2.5">
              <div className="flex flex-wrap items-center gap-1.5"><Badge>{r.answer_status}</Badge>{r.grounding && <Badge tone={r.grounding.status === "grounded" ? "ok" : "neutral"}>{r.grounding.status}</Badge>}</div>
              {r.answer && <p className="whitespace-pre-wrap leading-relaxed text-fg">{r.answer}</p>}
              {r.evidence.length > 0 && <p className="font-mono text-[10px] text-fg-subtle">evidence: {r.evidence.join(", ")}</p>}
              {r.grounding && <p className="font-mono text-[10px] text-fg-subtle">claims {r.grounding.factual_claims} factual · {r.grounding.supported} supported · {r.grounding.weakly_supported} weak · {r.grounding.unsupported} unsupported</p>}
            </div>
          )}
        </>
      )}
      <dl className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-[11px]">
        {measured.map((m) => (
          <div key={m.metric} className="flex justify-between gap-2"><dt className="truncate text-fg-subtle">{m.metric}</dt><dd className="num">{fmt(m.value, m.metric)}</dd></div>
        ))}
      </dl>
      {skipped.length > 0 && (
        <details className="text-[11px] text-fg-subtle">
          <summary className="cursor-pointer">{skipped.length} metrics skipped</summary>
          <ul className="mt-1 space-y-0.5 font-mono">{skipped.map((m) => <li key={m.metric}>{m.metric}: {m.skipped}</li>)}</ul>
        </details>
      )}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] text-fg-subtle">
        {Object.entries(r.timings_ms).map(([k, ms]) => <span key={k}>{k} {fmt(ms, "x_ms")}</span>)}
        {Object.entries(r.models).map(([k, m]) => <span key={k} title={m}>{k}: {m.split("@")[0]}</span>)}
      </div>
      {r.artifact_id && (
        <a href={`${API_URL}/api/v1/artifacts/${r.artifact_id}`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 font-mono text-[10px] text-trace hover:underline">
          full trace {r.artifact_id} <ExternalLink className="size-3" />
        </a>
      )}
    </div>
  );
}

// --- failures, latency, configurations ----------------------------------------------------------

function FailuresView({ run }: { run: ExperimentRun }) {
  const fetcher = useCallback(() => api.GET("/api/v1/runs/{run_id}/cases", { params: { path: { run_id: run.id }, query: { status: "failed" } } }), [run.id]);
  const failed = useApi(fetcher);
  return (
    <div className="grid items-start gap-5 xl:grid-cols-2">
      <Panel>
        <PanelHeader eyebrow="Failures" title={`${run.failed} of ${run.total} evaluations failed`} />
        {run.failed === 0 ? (
          <p className="p-4 text-[12px] text-fg-subtle">Every case of every arm completed.</p>
        ) : !failed.data ? (
          failed.error ? <ErrorState title="Could not load failures">{failed.error}</ErrorState> : <LoadingState rows={4} label="Loading failures" />
        ) : (
          <div className="divide-y divide-line">
            {run.summaries.filter((s) => s.failed).map((s) => (
              <div key={s.arm} className="px-4 py-2.5 text-[12px]">
                <div className="flex items-center gap-2"><span className="font-mono">{s.arm}</span><span className="text-err">{s.failed} failed</span>{Object.entries(s.failure_types).map(([t, n]) => <Badge key={t} tone="err">{t} × {n}</Badge>)}</div>
              </div>
            ))}
            <ul className="divide-y divide-line/60">
              {failed.data.map((c) => (
                <li key={`${c.arm}-${c.case_id}`} className="px-4 py-2 text-[11px]">
                  <span className="font-mono text-fg">{c.case_id}</span> <span className="font-mono text-fg-subtle">· {c.arm}</span>
                  <p className="mt-0.5 text-err"><span className="font-mono">{c.error_type}</span>: {c.error}</p>
                </li>
              ))}
            </ul>
          </div>
        )}
      </Panel>
      <Panel>
        <PanelHeader eyebrow="Skipped metrics" title="Why a metric was not computed" />
        <SkipTable summaries={run.summaries} />
      </Panel>
    </div>
  );
}

function SkipTable({ summaries }: { summaries: ArmSummary[] }) {
  const rows = summaries.flatMap((s) => s.metrics.filter((m) => m.skipped > 0).map((m) => ({ arm: s.arm, metric: m.metric, skipped: m.skipped, n: m.n, reason: s.skipped_reasons[m.metric] })));
  if (!rows.length) return <p className="p-4 text-[12px] text-fg-subtle">No metric was skipped.</p>;
  return (
    <div className="max-h-[520px] overflow-auto">
      <table className="w-full text-left text-[11px]">
        <thead className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle"><tr className="border-b border-line"><th className="px-4 py-2 font-normal">arm</th><th className="font-normal">metric</th><th className="text-right font-normal">skipped / defined</th><th className="px-4 font-normal">reason</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.arm}-${r.metric}`} className="border-b border-line/60 last:border-0">
              <td className="px-4 py-1 font-mono">{r.arm}</td><td className="font-mono">{r.metric}</td><td className="num text-right">{r.skipped} / {r.n}</td><td className="px-4 text-fg-subtle">{r.reason ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const STAGES = ["total", "retrieval", "rerank", "routing", "generation", "grounding"];

function LatencyView({ run }: { run: ExperimentRun }) {
  const cases = useRunCases(run.id);
  const [stage, setStage] = useState("total");
  if (!cases.data) return <Panel>{cases.error ? <ErrorState title="Could not load cases">{cases.error}</ErrorState> : <LoadingState rows={4} label="Loading timings" />}</Panel>;
  const groups = run.arms.map((a, i) => ({
    key: a, label: a, color: armColor(i),
    values: cases.data!.filter((c) => c.arm === a && c.status === "ok" && c.timings_ms[stage] !== undefined).map((c) => c.timings_ms[stage]),
  }));
  return (
    <Panel>
      <PanelHeader eyebrow="Latency" title={`${stage} time per case, measured on this machine`}
        actions={<select aria-label="Stage" value={stage} onChange={(e) => setStage(e.target.value)} className="h-7 rounded-md border border-line-strong bg-surface px-2 font-mono text-[11px]">{STAGES.map((s) => <option key={s}>{s}</option>)}</select>} />
      <div className="grid gap-4 p-4 lg:grid-cols-[1fr_320px]">
        <StripPlot groups={groups.filter((g) => g.values.length)} label={`${stage} latency by arm`} format={(v) => fmt(v, "x_ms")} />
        <table className="w-full text-left text-[12px]">
          <thead className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle"><tr className="border-b border-line"><th className="py-1.5 font-normal">arm</th><th className="text-right font-normal">n</th><th className="text-right font-normal">p50</th><th className="text-right font-normal">p95</th></tr></thead>
          <tbody>
            {groups.map((g) => (
              <tr key={g.key} className="border-b border-line/60 last:border-0">
                <td className="py-1.5 font-mono text-[11px]">{g.label}</td>
                <td className="num text-right">{g.values.length}</td>
                <td className="num text-right">{fmt(percentile(g.values, 0.5), "x_ms")}</td>
                <td className="num text-right">{fmt(percentile(g.values, 0.95), "x_ms")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="border-t border-line px-4 py-2.5 text-[11px] text-fg-subtle">Wall-clock times include model loading on the first case; concurrency was {run.limits.concurrency}. Not comparable across machines.</p>
    </Panel>
  );
}

function ConfigsView({ run, exp }: { run: ExperimentRun; exp: Experiment }) {
  const families: MetricFamily[] = FAMILIES;
  return (
    <div className="space-y-5">
      <Panel>
        <PanelHeader eyebrow="Configuration snapshots" title="What exactly produced these numbers" />
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[11px]">
            <thead className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
              <tr className="border-b border-line">
                <th className="px-4 py-2 font-normal">arm</th><th className="font-normal">hash</th><th className="font-normal">retrieval</th><th className="font-normal">models</th><th className="font-normal">evidence / generation</th><th className="px-4 font-normal">engine</th>
              </tr>
            </thead>
            <tbody>
              {exp.snapshots.map((s, i) => (
                <tr key={s.arm} className="border-b border-line/60 align-top last:border-0">
                  <td className="px-4 py-2"><span className="flex items-center gap-1.5 font-mono"><span className="size-1.5 rounded-full" style={{ background: armColor(i) }} />{s.arm}</span></td>
                  <td className="py-2 font-mono text-fg-subtle" title={run.snapshot_hashes[s.arm]}>{shortHash(run.snapshot_hashes[s.arm] ?? "", 12)}</td>
                  <td className="py-2 pr-3 font-mono">
                    {s.retrieval_mode === "adaptive" ? `adaptive · ${s.query_analyzer} · ${s.router_policy}` : `${s.retrieval?.strategy} · top_k ${s.retrieval_template.top_k}`}
                    {s.retrieval_template.rerank.enabled && ` · rerank pool ${s.retrieval_template.rerank.candidate_k}`}
                  </td>
                  <td className="py-2 pr-3 font-mono text-fg-muted">
                    {s.embedder && <div>{s.embedder.model}@{s.embedder.revision.slice(0, 8)}</div>}
                    {s.reranker && <div>{s.reranker.model}@{s.reranker.revision.slice(0, 8)}</div>}
                    {s.generator && <div>{s.generator.model}{s.generator.revision ? `@${s.generator.revision.slice(0, 8)}` : ""}</div>}
                    {s.verifier && <div>{s.verifier}</div>}
                    {!s.embedder && !s.reranker && !s.generator && <span className="text-fg-subtle">lexical only</span>}
                  </td>
                  <td className="py-2 pr-3 font-mono text-fg-muted">
                    {s.evidence ? `≤${s.evidence.max_items} passages · ${s.evidence.max_context_tokens} tok` : "—"}
                    {s.generation && <div>{s.generator?.name} · T={s.generation.temperature} · max {s.generation.max_new_tokens}</div>}
                  </td>
                  <td className="px-4 py-2 font-mono text-fg-subtle">{s.engine_version}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="border-t border-line px-4 py-2.5 text-[11px] text-fg-subtle">
          Metrics: ks {exp.metrics.ks.join(", ")} · registry {run.metric_registry_version} · families {families.map((f) => FAMILY_LABEL[f]).join(", ")} · environment: Python {run.environment.python_version} on {run.environment.platform}.
        </p>
      </Panel>
      <Panel>
        <PanelHeader eyebrow="Ablations" title={exp.ablations.length ? "Declared pairs and the factors that actually differ" : "No ablations declared"} />
        {exp.ablations.length ? (
          <ul className="divide-y divide-line">
            {exp.ablations.map((a) => (
              <li key={`${a.baseline}>${a.variant}`} className="flex flex-wrap items-center gap-2 px-4 py-2.5 text-[12px]">
                <span className="font-mono">{a.baseline} → {a.variant}</span>
                <Badge>{a.factor}</Badge>
                {a.single_factor ? <Badge tone="ok">single factor</Badge> : <Badge tone="signal" title="More than one arm setting differs, so the factor is confounded">{a.factors.length} factors differ</Badge>}
                <span className="font-mono text-[10px] text-fg-subtle">{a.factors.join(", ")}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="p-4 text-[12px] text-fg-subtle">Any two arms can still be compared in the statistical comparison view.</p>
        )}
      </Panel>
    </div>
  );
}
