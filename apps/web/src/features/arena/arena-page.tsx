"use client";

import type { ArenaOverview, RunDigest } from "@rag-forge/shared";
import { FlaskConical, Swords } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback } from "react";
import { Badge, OriginBadge } from "@/components/ui/badge";
import { type Column, DataTable } from "@/components/ui/data-table";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { PipelineSteps } from "@/components/ui/pipeline";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { StatusIndicator } from "@/components/ui/status";
import { Tabs } from "@/components/ui/tabs";
import { api } from "@/lib/api";
import { useApi } from "@/lib/use-api";
import { BuilderTab } from "./builder-tab";
import { DatasetsTab } from "./datasets-tab";
import { RUN_STATUS } from "./format";
import { ResultsView } from "./results-view";

export type ArenaTab = "overview" | "datasets" | "builder" | "runs" | "results";

const TABS: { id: ArenaTab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "datasets", label: "Datasets" },
  { id: "builder", label: "Experiment builder" },
  { id: "runs", label: "Run monitor" },
  { id: "results", label: "Results" },
];

const fetchOverview = () => api.GET("/api/v1/arena/overview");
const pollOverview = (d: ArenaOverview | undefined) => (d?.runs.some((r) => r.status === "queued" || r.status === "running") ? 1500 : undefined);

export function ArenaPage({ initialTab = "overview" }: { initialTab?: ArenaTab }) {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const tab = (params.get("tab") as ArenaTab | null) ?? (params.get("run") ? "results" : initialTab);
  const runId = params.get("run");
  const overview = useApi(fetchOverview, pollOverview);
  const go = useCallback(
    (next: ArenaTab, run?: string | null) => {
      const q = new URLSearchParams();
      q.set("tab", next);
      const r = run === undefined ? runId : run;
      if (r) q.set("run", r);
      router.replace(`${pathname}?${q.toString()}`);
    },
    [pathname, router, runId],
  );
  const openRun = (id: string) => go("results", id);

  return (
    <div className="mx-auto max-w-[1600px] space-y-5">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-signal">Evaluate</p>
          <h1 className="mt-1.5 flex items-center gap-2 text-2xl font-medium tracking-tight"><Swords className="size-5 text-signal" /> RAG Arena</h1>
          <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-fg-muted">
            Controlled experiments: the same benchmark cases, corpus version and metric definitions for every configuration,
            paired statistics per case, and the provenance of every number. Only recorded runs are shown; a measured
            benchmark score is not a judgement of answer quality.
          </p>
        </div>
        <Tabs tabs={TABS} value={tab} onChange={(t) => go(t as ArenaTab)} />
      </header>
      {overview.error && !overview.data ? (
        <Panel><ErrorState title="Could not reach the API">{overview.error}</ErrorState></Panel>
      ) : !overview.data ? (
        <Panel><LoadingState rows={5} label="Loading the Arena" /></Panel>
      ) : tab === "overview" ? (
        <OverviewTab overview={overview.data} onOpenRun={openRun} onGo={go} />
      ) : tab === "datasets" ? (
        <DatasetsTab onChanged={overview.reload} onBuild={() => go("builder")} />
      ) : tab === "builder" ? (
        <BuilderTab datasets={overview.data.datasets} onStarted={(id) => { overview.reload(); go("runs", id); }} onGoDatasets={() => go("datasets")} />
      ) : tab === "runs" ? (
        <RunsTab runs={overview.data.runs} onOpenRun={openRun} onGo={go} />
      ) : runId ? (
        <ResultsView key={runId} runId={runId} />
      ) : (
        <RunsTab runs={overview.data.runs} onOpenRun={openRun} onGo={go} pick />
      )}
    </div>
  );
}

function OverviewTab({ overview, onOpenRun, onGo }: { overview: ArenaOverview; onOpenRun: (id: string) => void; onGo: (t: ArenaTab) => void }) {
  const finished = overview.runs.filter((r) => r.status === "completed" || r.status === "partial");
  return (
    <div className="grid items-start gap-5 xl:grid-cols-[1fr_420px]">
      <div className="space-y-5">
        <Panel>
          <PanelHeader eyebrow="Method" title="How a comparison is produced" />
          <div className="space-y-4 p-4">
            <PipelineSteps
              steps={[
                { id: "dataset", label: "Versioned dataset" },
                { id: "arms", label: "Configuration snapshots" },
                { id: "run", label: "Controlled run" },
                { id: "metrics", label: "Per-case metrics" },
                { id: "stats", label: "Paired statistics" },
              ]}
            />
            <ul className="grid gap-2 text-[12px] leading-relaxed text-fg-muted sm:grid-cols-2">
              <li><span className="text-fg">Same cases, same corpus version.</span> Every arm of a run sees exactly the dataset version&apos;s cases, pinned to one corpus version.</li>
              <li><span className="text-fg">Hashed configurations.</span> Each arm is resolved into an immutable snapshot whose hash is stored on every result.</li>
              <li><span className="text-fg">Missing ground truth is skipped.</span> A metric without its annotations is reported as skipped with a reason, never as 0.</li>
              <li><span className="text-fg">Failures are results.</span> A case that fails is stored with its error and counted in the failure rate.</li>
              <li><span className="text-fg">Paired comparisons.</span> Differences per case, bootstrap confidence intervals, effect size and an exact sign test with Holm adjustment.</li>
              <li><span className="text-fg">No conclusion below {overview.stats_method.min_cases_for_conclusion} cases.</span> Small or development datasets validate the pipeline; they do not rank methods.</li>
            </ul>
          </div>
        </Panel>
        <Panel>
          <PanelHeader eyebrow="Runs" title="Recent experiment runs" actions={<OriginBadge origin="measured" />} />
          {overview.runs.length ? (
            <RunTable runs={overview.runs.slice(0, 8)} onOpenRun={onOpenRun} />
          ) : (
            <EmptyState icon={FlaskConical} title="No measured results yet" action={<button type="button" onClick={() => onGo("builder")} className="text-xs text-trace hover:underline">Open the experiment builder</button>}>
              Nothing has been run. Register a dataset (the development benchmark is one click away), choose configurations and run them.
            </EmptyState>
          )}
        </Panel>
      </div>
      <div className="space-y-5">
        <Panel>
          <PanelHeader eyebrow="State" title="Arena" />
          <dl className="grid grid-cols-3 divide-x divide-line">
            {[
              ["Datasets", overview.datasets.length],
              ["Experiments", overview.experiments],
              ["Finished runs", finished.length],
            ].map(([k, v]) => (
              <div key={k} className="p-4">
                <dt className="font-mono text-[10px] uppercase tracking-[0.12em] text-fg-subtle">{k}</dt>
                <dd className="num mt-1 text-2xl">{v}</dd>
              </div>
            ))}
          </dl>
        </Panel>
        <Panel>
          <PanelHeader eyebrow="Definitions" title="Recorded with every run" />
          <dl className="divide-y divide-line text-xs">
            <div className="flex justify-between px-4 py-2.5"><dt className="text-fg-subtle">Metric registry</dt><dd className="font-mono">{overview.metric_registry}</dd></div>
            <div className="flex justify-between px-4 py-2.5"><dt className="text-fg-subtle">Statistics</dt><dd className="font-mono">{overview.stats_method.name}@{overview.stats_method.version}</dd></div>
            <div className="flex justify-between px-4 py-2.5"><dt className="text-fg-subtle">Bootstrap</dt><dd className="font-mono">{overview.stats_method.bootstrap_resamples} resamples · {Math.round(overview.stats_method.confidence * 100)}% · seed {overview.stats_method.seed}</dd></div>
            <div className="px-4 py-2.5"><dt className="text-fg-subtle">Multiple comparisons</dt><dd className="mt-0.5 text-fg-muted">{overview.stats_method.multiple_comparisons}</dd></div>
          </dl>
        </Panel>
      </div>
    </div>
  );
}

function RunTable({ runs, onOpenRun }: { runs: RunDigest[]; onOpenRun: (id: string) => void }) {
  const columns: Column<RunDigest>[] = [
    {
      key: "exp",
      header: "Experiment",
      cell: (r) => (
        <div className="min-w-0">
          <div className="truncate text-[13px] text-fg">{r.experiment_name}</div>
          <div className="font-mono text-[10px] text-fg-subtle">{r.run_id}</div>
        </div>
      ),
    },
    {
      key: "ds",
      header: "Dataset",
      cell: (r) => (
        <span className="flex items-center gap-1.5 font-mono">
          {r.dataset_name} v{r.dataset_version}
          {r.dataset_source === "development" && <Badge tone="signal">dev</Badge>}
        </span>
      ),
    },
    { key: "arms", header: "Arms", align: "right", cell: (r) => <span className="num">{r.arms.length}</span> },
    {
      key: "progress",
      header: "Progress",
      cell: (r) => (
        <div className="w-36">
          <div className="flex h-1.5 overflow-hidden rounded-full bg-surface-3">
            <span className="bg-ok" style={{ width: `${((r.completed - r.failed) / Math.max(1, r.total)) * 100}%` }} />
            <span className="bg-err" style={{ width: `${(r.failed / Math.max(1, r.total)) * 100}%` }} />
          </div>
          <div className="num mt-1 font-mono text-[10px] text-fg-subtle">{r.completed}/{r.total}{r.failed ? ` · ${r.failed} failed` : ""}</div>
        </div>
      ),
    },
    { key: "status", header: "Status", cell: (r) => <StatusIndicator status={RUN_STATUS[r.status]} label={r.status} /> },
    { key: "when", header: "Started", align: "right", cell: (r) => new Date(r.created_at).toLocaleString(), className: "hidden md:table-cell" },
  ];
  return <DataTable caption="Experiment runs" columns={columns} rows={runs} rowKey={(r) => r.run_id} onRowClick={(r) => onOpenRun(r.run_id)} />;
}

function RunsTab({ runs, onOpenRun, onGo, pick }: { runs: RunDigest[]; onOpenRun: (id: string) => void; onGo: (t: ArenaTab) => void; pick?: boolean }) {
  return (
    <Panel>
      <PanelHeader eyebrow="Run monitor" title={pick ? "Choose a run to inspect" : "Runs, newest first (live while running)"} />
      {runs.length ? (
        <RunTable runs={runs} onOpenRun={onOpenRun} />
      ) : (
        <EmptyState icon={FlaskConical} title="No runs yet" action={<button type="button" onClick={() => onGo("builder")} className="text-xs text-trace hover:underline">Build an experiment</button>}>
          Runs appear here as soon as they are queued, with their progress and failures.
        </EmptyState>
      )}
    </Panel>
  );
}
