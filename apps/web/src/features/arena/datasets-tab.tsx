"use client";

import type { BenchmarkDataset, DatasetSummary } from "@rag-forge/shared";
import { Database, Download } from "lucide-react";
import { useCallback, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { api } from "@/lib/api";
import { shortHash } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { errorMessage } from "@/features/retrieval/errors";

const fetchDatasets = () => api.GET("/api/v1/benchmarks");

export const SOURCE_HELP: Record<DatasetSummary["source"], string> = {
  development: "Bundled to validate the pipeline. Too small and too easy to rank methods.",
  user: "Registered through the API; annotations checked against the pinned corpus version.",
  external: "Reserved for imported public benchmarks (no importer yet).",
};

export function DatasetsTab({ onChanged, onBuild }: { onChanged: () => void; onBuild: () => void }) {
  const list = useApi(fetchDatasets);
  const [selected, setSelected] = useState<string | null>(null);
  const [installing, setInstalling] = useState<{ busy: boolean; error?: string }>({ busy: false });
  const install = async () => {
    setInstalling({ busy: true });
    const { data, error, response } = await api.POST("/api/v1/benchmarks/development");
    if (!data) return setInstalling({ busy: false, error: errorMessage(error, response.status) });
    setInstalling({ busy: false });
    setSelected(data.id);
    list.reload();
    onChanged();
  };
  const datasets = list.data ?? [];
  const current = selected ?? datasets[0]?.id ?? null;
  const hasDev = datasets.some((d) => d.source === "development");

  return (
    <div className="grid items-start gap-5 xl:grid-cols-[420px_1fr]">
      <div className="space-y-5">
        <Panel>
          <PanelHeader
            eyebrow="Benchmark datasets"
            title="Versioned, pinned to a corpus version"
            actions={
              !hasDev && datasets.length > 0 && (
                <Button size="sm" onClick={install} disabled={installing.busy}>
                  <Download className="size-3.5" /> {installing.busy ? "Installing…" : "Install development benchmark"}
                </Button>
              )
            }
          />
          {installing.error && <p role="alert" className="border-b border-line px-4 py-2 text-xs text-err">{installing.error}</p>}
          {list.loading && !list.data ? (
            <LoadingState rows={3} label="Loading datasets" />
          ) : list.error ? (
            <ErrorState title="Could not load datasets">{list.error}</ErrorState>
          ) : datasets.length === 0 ? (
            <EmptyState icon={Database} title="No benchmark datasets" action={<Button size="sm" variant="primary" onClick={install} disabled={installing.busy}>{installing.busy ? "Installing…" : "Install development benchmark"}</Button>}>
              The development benchmark adds a 12-document corpus with 14 annotated cases (and builds its dense index).
              Your own datasets are registered with <span className="font-mono">POST /api/v1/benchmarks</span>.
            </EmptyState>
          ) : (
            <ul className="divide-y divide-line">
              {datasets.map((d) => (
                <li key={d.id}>
                  <button type="button" onClick={() => setSelected(d.id)} aria-pressed={current === d.id} className={`w-full px-4 py-3 text-left transition-colors hover:bg-surface-2 ${current === d.id ? "bg-surface-2 shadow-[inset_2px_0_0_var(--color-signal)]" : ""}`}>
                    <div className="flex items-center gap-2">
                      <span className="text-[13px] font-medium">{d.name}</span>
                      <Badge>v{d.version}</Badge>
                      <Badge tone={d.source === "development" ? "signal" : d.source === "user" ? "trace" : "neutral"} title={SOURCE_HELP[d.source]}>{d.source}</Badge>
                    </div>
                    <div className="mt-1 font-mono text-[10px] text-fg-subtle">
                      {d.case_count} cases · corpus {d.corpus_id.slice(0, 12)} v{d.corpus_version} · {shortHash(d.content_hash, 8)}
                    </div>
                    <AnnotationBar counts={d.annotation_counts} total={d.case_count} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
        {datasets.length > 0 && (
          <p className="px-1 text-[11px] leading-relaxed text-fg-subtle">
            Annotations are optional per case. A metric that needs an annotation a case lacks is skipped for that case and
            reported as skipped. <button type="button" onClick={onBuild} className="text-trace hover:underline">Build an experiment →</button>
          </p>
        )}
      </div>
      {current && <DatasetDetail key={current} id={current} />}
    </div>
  );
}

function AnnotationBar({ counts, total }: { counts: Record<string, number>; total: number }) {
  return (
    <div className="mt-2 grid grid-cols-4 gap-2">
      {Object.entries(counts).map(([k, v]) => (
        <div key={k} title={`${v} of ${total} cases annotated with ${k}`}>
          <div className="h-1 overflow-hidden rounded-full bg-surface-3"><div className="h-full bg-trace" style={{ width: `${(v / Math.max(1, total)) * 100}%` }} /></div>
          <div className="mt-0.5 truncate font-mono text-[9px] text-fg-subtle">{k.replace("_", " ")} {v}</div>
        </div>
      ))}
    </div>
  );
}

function DatasetDetail({ id }: { id: string }) {
  const fetcher = useCallback(() => api.GET("/api/v1/benchmarks/{dataset_id}", { params: { path: { dataset_id: id } } }), [id]);
  const ds = useApi(fetcher);
  if (!ds.data) return <Panel>{ds.error ? <ErrorState title="Could not load the dataset">{ds.error}</ErrorState> : <LoadingState rows={6} label="Loading cases" />}</Panel>;
  return <CaseTable ds={ds.data} />;
}

function CaseTable({ ds }: { ds: BenchmarkDataset }) {
  return (
    <Panel>
      <PanelHeader eyebrow={`${ds.name} v${ds.version} · ${ds.source}`} title={`${ds.cases.length} benchmark cases`} />
      <div className="space-y-1 border-b border-line px-4 py-3 text-[12px] leading-relaxed text-fg-muted">
        <p>{ds.description}</p>
        {ds.annotation_notes && <p className="text-fg-subtle"><span className="text-fg-muted">Ground truth:</span> {ds.annotation_notes}</p>}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-[12px]">
          <thead className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
            <tr className="border-b border-line">
              <th className="px-4 py-2 font-normal">case</th>
              <th className="font-normal">query</th>
              <th className="font-normal">relevant</th>
              <th className="font-normal">reference answer</th>
              <th className="pr-4 font-normal">tags</th>
            </tr>
          </thead>
          <tbody>
            {ds.cases.map((c) => (
              <tr key={c.id} className="border-b border-line/60 align-top last:border-0">
                <td className="px-4 py-2 font-mono text-[11px] text-fg-subtle">{c.id}</td>
                <td className="py-2 pr-3 text-fg">{c.query}</td>
                <td className="py-2 pr-3 font-mono text-[11px]">
                  {c.answerable === false ? <Badge tone="signal">unanswerable</Badge> : Object.entries({ ...c.relevant_documents, ...c.relevant_chunks }).map(([k, g]) => <div key={k}>{k} <span className="text-fg-subtle">·{g}</span></div>)}
                  {c.answerable !== false && !Object.keys({ ...c.relevant_documents, ...c.relevant_chunks }).length && <span className="text-fg-subtle">not judged</span>}
                </td>
                <td className="py-2 pr-3 text-fg-muted">{c.reference_answer ?? <span className="text-fg-subtle">none</span>}</td>
                <td className="py-2 pr-4"><div className="flex flex-wrap gap-1">{c.tags.map((t) => <Badge key={t}>{t}</Badge>)}</div></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
