"use client";

import type { RunDigest } from "@rag-forge/shared";
import { FlaskConical } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { OriginBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { STRATEGY_COLOR } from "@/components/ui/chart";
import { type Column, DataTable } from "@/components/ui/data-table";
import { Dialog } from "@/components/ui/dialog";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { type Status, StatusIndicator } from "@/components/ui/status";
import { Tabs } from "@/components/ui/tabs";
import { fetchRuns, useApi } from "@/lib/use-api";
import { SAMPLE_ORIGIN, type SampleExperiment, type SampleRunStatus, sampleExperiments } from "@/sample/preview-data";

const RUN_STATUS: Record<SampleRunStatus, Status> = {
  completed: "ok",
  running: "loading",
  failed: "err",
  queued: "idle",
};

const sampleColumns: Column<SampleExperiment>[] = [
  {
    key: "name",
    header: "Experiment",
    cell: (e) => (
      <div className="min-w-0">
        <div className="truncate text-[13px] text-fg">{e.name}</div>
        <div className="font-mono text-[10px] text-fg-subtle">{e.id}</div>
      </div>
    ),
  },
  { key: "corpus", header: "Corpus", cell: (e) => <span className="font-mono">{e.corpus}</span>, className: "hidden md:table-cell" },
  { key: "strategies", header: "Strategies", cell: (e) => <Strategies keys={e.strategies} />, className: "hidden lg:table-cell" },
  { key: "status", header: "Status", cell: (e) => <StatusIndicator status={RUN_STATUS[e.status]} label={e.status} /> },
  { key: "ndcg", header: "nDCG@10", align: "right", cell: (e) => <span className="num text-fg">{e.ndcg?.toFixed(3) ?? "—"}</span> },
  { key: "started", header: "Started", align: "right", cell: (e) => e.started, className: "hidden sm:table-cell" },
];

const LIVE_STATUS: Record<RunDigest["status"], Status> = {
  completed: "ok",
  partial: "warn",
  running: "loading",
  queued: "idle",
  failed: "err",
};

const liveColumns: Column<RunDigest>[] = [
  {
    key: "name",
    header: "Experiment",
    cell: (r) => (
      <div className="min-w-0">
        <div className="truncate text-[13px] text-fg">{r.experiment_name}</div>
        <div className="font-mono text-[10px] text-fg-subtle">{r.run_id}</div>
      </div>
    ),
  },
  { key: "dataset", header: "Dataset", cell: (r) => <span className="font-mono">{r.dataset_name} v{r.dataset_version}</span>, className: "hidden md:table-cell" },
  { key: "arms", header: "Arms", align: "right", cell: (r) => <span className="num">{r.arms.length}</span> },
  { key: "status", header: "Status", cell: (r) => <StatusIndicator status={LIVE_STATUS[r.status]} label={`${r.status} · ${r.completed}/${r.total}${r.failed ? ` · ${r.failed} failed` : ""}`} /> },
  { key: "created", header: "Started", align: "right", cell: (r) => new Date(r.created_at).toLocaleString(), className: "hidden sm:table-cell" },
];

function Strategies({ keys }: { keys: string[] }) {
  return (
    <div className="flex gap-1">
      {keys.map((k) => (
        <span key={k} className="flex items-center gap-1 rounded bg-surface-3 px-1.5 py-0.5 font-mono text-[10px]">
          <span className="size-1.5 rounded-full" style={{ background: STRATEGY_COLOR[k] }} />
          {k}
        </span>
      ))}
    </div>
  );
}

export function RecentExperiments({ className }: { className?: string }) {
  const [tab, setTab] = useState("recorded");
  const [open, setOpen] = useState<SampleExperiment | null>(null);
  const live = useApi(fetchRuns);
  const router = useRouter();
  const count = live.data?.length;

  return (
    <Panel className={className}>
      <PanelHeader
        eyebrow="Experiments"
        title="Recent runs"
        actions={
          <Tabs
            value={tab}
            onChange={setTab}
            tabs={[
              { id: "recorded", label: <>Recorded{count !== undefined && <span className="num ml-1.5 text-fg-subtle">{count}</span>}</> },
              { id: "preview", label: "Preview" },
            ]}
          />
        }
      />
      {tab === "recorded" ? (
        live.loading ? (
          <LoadingState rows={3} label="Loading runs" />
        ) : live.error ? (
          <ErrorState title="Could not load runs">{live.error}</ErrorState>
        ) : count ? (
          <DataTable
            caption="Recorded experiment runs"
            columns={liveColumns}
            rows={live.data!.slice(0, 8)}
            rowKey={(r) => r.run_id}
            onRowClick={(r) => router.push(`/arena?run=${encodeURIComponent(r.run_id)}`)}
          />
        ) : (
          <EmptyState
            icon={FlaskConical}
            title="No experiment runs recorded yet"
            action={<Button size="sm" onClick={() => setTab("preview")}>View sample preview</Button>}
          >
            Runs appear here once an experiment is run in the Arena. Results are only ever shown from recorded runs.
          </EmptyState>
        )
      ) : (
        <>
          <div className="flex items-center gap-2 border-b border-line bg-surface-2/50 px-4 py-2 text-[11px] text-fg-muted">
            <OriginBadge origin={SAMPLE_ORIGIN} />
            Illustrative rows for interface review. None of these runs happened.
          </div>
          <DataTable
            caption="Sample experiments (not real)"
            columns={sampleColumns}
            rows={sampleExperiments}
            rowKey={(e) => e.id}
            onRowClick={setOpen}
          />
        </>
      )}

      <Dialog
        variant="drawer"
        open={open !== null}
        onClose={() => setOpen(null)}
        title={open?.name}
        description={open && <span className="font-mono">{open.id}</span>}
      >
        {open && <ExperimentDetail e={open} />}
      </Dialog>
    </Panel>
  );
}

function ExperimentDetail({ e }: { e: SampleExperiment }) {
  const rows: [string, React.ReactNode][] = [
    ["Status", <StatusIndicator key="s" status={RUN_STATUS[e.status]} label={e.status} />],
    ["Corpus", <span key="c" className="font-mono">{e.corpus}</span>],
    ["Strategies", <Strategies key="st" keys={e.strategies} />],
    ["Config hash", <span key="h" className="font-mono text-trace">{e.configHash}</span>],
    ["nDCG@10", <span key="n" className="num">{e.ndcg?.toFixed(3) ?? "—"}</span>],
  ];
  return (
    <div className="space-y-6">
      <OriginBadge origin={SAMPLE_ORIGIN} />
      <section>
        <h3 className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-subtle">Hypothesis</h3>
        <p className="text-[13px] leading-relaxed text-fg">{e.hypothesis}</p>
      </section>
      <dl className="divide-y divide-line rounded-lg border border-line text-xs">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-center justify-between gap-4 px-3 py-2.5">
            <dt className="text-fg-subtle">{k}</dt>
            <dd className="text-fg-muted">{v}</dd>
          </div>
        ))}
      </dl>
      <section className="rounded-lg border border-dashed border-line-strong p-3 text-xs leading-relaxed text-fg-muted">
        <h3 className="mb-1 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-subtle">Provenance</h3>
        Sample rows carry no provenance. Recorded runs will show dataset and corpus versions, the full configuration,
        retrieved chunk ids and the environment snapshot needed to replay them.
      </section>
    </div>
  );
}
