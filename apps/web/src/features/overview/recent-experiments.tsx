"use client";

import type { RunDigest } from "@rag-forge/shared";
import { FlaskConical } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { OriginBadge } from "@/components/ui/badge";
import { type Column, DataTable } from "@/components/ui/data-table";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { type Status, StatusIndicator } from "@/components/ui/status";
import { fetchRuns, useApi } from "@/lib/use-api";

const LIVE_STATUS: Record<RunDigest["status"], Status> = {
  completed: "ok",
  partial: "warn",
  running: "loading",
  queued: "idle",
  failed: "err",
};

const columns: Column<RunDigest>[] = [
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

export function RecentExperiments({ className }: { className?: string }) {
  const runs = useApi(fetchRuns);
  const router = useRouter();
  return (
    <Panel className={className}>
      <PanelHeader eyebrow="Experiments" title="Recent runs" actions={runs.data && <OriginBadge origin="measured" />} />
      {runs.loading && !runs.data ? (
        <LoadingState rows={3} label="Loading runs" />
      ) : runs.error && !runs.data ? (
        <ErrorState title="Could not load runs">{runs.error}</ErrorState>
      ) : runs.data?.length ? (
        <DataTable
          caption="Recorded experiment runs"
          columns={columns}
          rows={runs.data.slice(0, 6)}
          rowKey={(r) => r.run_id}
          onRowClick={(r) => router.push(`/replay?run=${encodeURIComponent(r.run_id)}`)}
        />
      ) : (
        <EmptyState icon={FlaskConical} title="No experiment runs recorded yet" action={<Link href="/experiments" className="text-xs text-trace hover:underline">Build an experiment</Link>}>
          Runs appear here once an experiment has been run. Results are only ever shown from recorded runs.
        </EmptyState>
      )}
    </Panel>
  );
}
