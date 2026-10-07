"use client";

import type { ExperimentRun, RunDigest } from "@rag-forge/shared";
import { Download, FileText, History } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback } from "react";
import { Badge, OriginBadge } from "@/components/ui/badge";
import { type Column, DataTable } from "@/components/ui/data-table";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { StatusIndicator } from "@/components/ui/status";
import { Tabs } from "@/components/ui/tabs";
import { API_URL, api } from "@/lib/api";
import { shortHash } from "@/lib/format";
import { fetchRuns, useApi } from "@/lib/use-api";
import { RUN_STATUS } from "@/features/arena/format";
import { ResultsView } from "@/features/arena/results-view";
import { PipelineView } from "./pipeline-view";
import { ReplayView } from "./replay-view";
import { ReproducibilityView } from "./reproducibility-view";

type View = "run" | "pipeline" | "reproducibility" | "replay" | "results";
const VIEWS: { id: View; label: string }[] = [
  { id: "run", label: "Run" },
  { id: "pipeline", label: "Pipeline" },
  { id: "reproducibility", label: "Reproducibility" },
  { id: "replay", label: "Replay" },
  { id: "results", label: "Results" },
];

export const EXPORTS = [
  { path: "report.md", label: "Research report", ext: "md" },
  { path: "export.json", label: "Full results", ext: "json" },
  { path: "export/metrics.csv", label: "Metric summaries", ext: "csv" },
  { path: "export/cases.csv", label: "Per-case metrics", ext: "csv" },
  { path: "manifest", label: "Manifest", ext: "json" },
  { path: "manifest.md", label: "Manifest", ext: "md" },
];

const finished = (s: RunDigest["status"]) => s === "completed" || s === "partial";

export function ReplayPage() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const runId = params.get("run");
  const view = (params.get("view") as View | null) ?? "run";
  const go = useCallback(
    (next: { run?: string | null; view?: View }) => {
      const q = new URLSearchParams();
      const r = next.run === undefined ? runId : next.run;
      if (r) q.set("run", r);
      q.set("view", next.view ?? (next.run !== undefined ? "run" : view));
      router.replace(`${pathname}?${q.toString()}`);
    },
    [pathname, router, runId, view],
  );
  return (
    <div className="mx-auto max-w-[1600px] space-y-5">
      <header>
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-signal">Evaluate</p>
        <h1 className="mt-1.5 flex items-center gap-2 text-2xl font-medium tracking-tight"><History className="size-5 text-signal" /> Replay &amp; run inspector</h1>
        <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-fg-muted">
          A recorded run as an auditable artifact: what produced it, every stage of every case with its hashes, the
          environment it ran in, and a replay that re-executes it and reports, stage by stage, what reproduced.
        </p>
      </header>
      {runId ? <RunInspector key={runId} runId={runId} view={view} onView={(v) => go({ view: v })} onClose={() => go({ run: null })} /> : <RunPicker onPick={(id) => go({ run: id })} />}
    </div>
  );
}

function RunPicker({ onPick }: { onPick: (id: string) => void }) {
  const runs = useApi(fetchRuns);
  const columns: Column<RunDigest>[] = [
    { key: "exp", header: "Experiment", cell: (r) => <div className="min-w-0"><div className="truncate text-[13px] text-fg">{r.experiment_name}</div><div className="font-mono text-[10px] text-fg-subtle">{r.run_id}</div></div> },
    { key: "ds", header: "Dataset", cell: (r) => <span className="flex items-center gap-1.5 font-mono">{r.dataset_name} v{r.dataset_version}{r.dataset_source === "development" && <Badge tone="signal">dev</Badge>}</span> },
    { key: "arms", header: "Arms", align: "right", cell: (r) => <span className="num">{r.arms.length}</span> },
    { key: "cases", header: "Evaluations", align: "right", cell: (r) => <span className="num">{r.completed}/{r.total}</span> },
    { key: "status", header: "Status", cell: (r) => <StatusIndicator status={RUN_STATUS[r.status]} label={r.status} /> },
    { key: "when", header: "Recorded", align: "right", cell: (r) => new Date(r.created_at).toLocaleString(), className: "hidden md:table-cell" },
  ];
  const rows = runs.data?.filter((r) => finished(r.status)) ?? [];
  return (
    <Panel>
      <PanelHeader eyebrow="Recorded runs" title="Choose a finished run to inspect" actions={runs.data && <OriginBadge origin="measured" />} />
      {!runs.data ? (
        runs.error ? <ErrorState title="Could not load runs">{runs.error}</ErrorState> : <LoadingState rows={4} label="Loading runs" />
      ) : rows.length ? (
        <DataTable caption="Finished runs" columns={columns} rows={rows} rowKey={(r) => r.run_id} onRowClick={(r) => onPick(r.run_id)} />
      ) : (
        <EmptyState icon={History} title="No finished runs to inspect" action={<a href="/experiments" className="text-xs text-trace hover:underline">Build and run an experiment</a>}>
          Replay and inspection need a run that has finished. Nothing has been recorded yet.
        </EmptyState>
      )}
    </Panel>
  );
}

function RunInspector({ runId, view, onView, onClose }: { runId: string; view: View; onView: (v: View) => void; onClose: () => void }) {
  const fetchRun = useCallback(() => api.GET("/api/v1/runs/{run_id}", { params: { path: { run_id: runId } } }), [runId]);
  const run = useApi(fetchRun);
  if (!run.data) return <Panel>{run.error ? <ErrorState title="Could not load the run">{run.error}</ErrorState> : <LoadingState rows={5} label="Loading the run" />}</Panel>;
  const r = run.data;
  if (!finished(r.status))
    return <Panel><EmptyState icon={History} title={`This run is ${r.status}`} action={<button type="button" onClick={onClose} className="text-xs text-trace hover:underline">Choose another run</button>}>Only finished runs can be inspected and replayed.</EmptyState></Panel>;
  return (
    <div className="space-y-5">
      <RunHeader run={r} onClose={onClose} />
      <Tabs tabs={VIEWS} value={view} onChange={(v) => onView(v as View)} />
      {view === "run" && <ReproducibilityView run={r} part="run" />}
      {view === "pipeline" && <PipelineView run={r} />}
      {view === "reproducibility" && <ReproducibilityView run={r} part="reproducibility" />}
      {view === "replay" && <ReplayView run={r} />}
      {view === "results" && <ResultsView runId={r.id} />}
    </div>
  );
}

function RunHeader({ run, onClose }: { run: ExperimentRun; onClose: () => void }) {
  const commit = run.runtime?.git_commit ?? run.environment.git_commit;
  return (
    <Panel>
      <PanelHeader
        eyebrow={`Run ${run.id}`}
        title={<span className="flex items-center gap-2">{run.arms.length} configurations × {run.case_ids.length} cases <StatusIndicator status={RUN_STATUS[run.status]} label={run.status} /></span>}
        actions={<><OriginBadge origin="measured" /><button type="button" onClick={onClose} className="text-xs text-fg-subtle hover:text-fg">Change run</button></>}
      />
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
        <p className="font-mono text-[11px] text-fg-subtle">
          commit {commit ? shortHash(commit, 10) : "not recorded"}{run.runtime?.git_dirty ? " (uncommitted changes)" : ""} · dataset {shortHash(run.dataset_hash, 10)} · corpus v{run.corpus_version} · {run.metric_registry_version} · {run.stats_method}
        </p>
        <div className="flex flex-wrap items-center gap-1.5" aria-label="Exports">
          {EXPORTS.map((x) => (
            <a key={x.path} href={`${API_URL}/api/v1/runs/${run.id}/${x.path}`} download className="inline-flex h-7 items-center gap-1.5 rounded-md px-2.5 text-xs text-fg-muted ring-1 ring-inset ring-line-strong transition-colors hover:bg-surface-3 hover:text-fg">
              {x.ext === "md" ? <FileText className="size-3.5" /> : <Download className="size-3.5" />}
              {x.label} <span className="font-mono text-[10px] text-fg-subtle">.{x.ext}</span>
            </a>
          ))}
        </div>
      </div>
    </Panel>
  );
}
