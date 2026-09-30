"use client";

import { Database } from "lucide-react";
import Link from "next/link";
import { Badge, OriginBadge } from "@/components/ui/badge";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { compact, formatBytes } from "@/lib/format";
import { fetchCorpora, useApi } from "@/lib/use-api";

export function CorpusOverview({ className }: { className?: string }) {
  const { data, error, loading } = useApi(fetchCorpora);
  const chunks = data?.reduce((s, c) => s + c.stats.chunk_count, 0) ?? 0;
  const docs = data?.reduce((s, c) => s + c.stats.document_count, 0) ?? 0;
  return (
    <Panel className={className}>
      <PanelHeader
        eyebrow="Corpus"
        title="Collections"
        actions={data && <OriginBadge origin="live" />}
      />
      {loading && !data && <LoadingState rows={4} label="Loading corpora" />}
      {error && !data && <ErrorState title="Could not load corpora">{error}</ErrorState>}
      {data?.length === 0 && (
        <EmptyState icon={Database} title="No corpora yet" action={<Link href="/corpus" className="text-xs text-trace underline-offset-4 hover:underline">Create one</Link>}>
          Corpora you create and ingest appear here.
        </EmptyState>
      )}
      {!!data?.length && (
        <>
          <div className="grid grid-cols-3 gap-px border-b border-line bg-line">
            {[
              ["corpora", data.length],
              ["documents", docs],
              ["chunks", chunks],
            ].map(([k, v]) => (
              <div key={k} className="bg-surface px-4 py-3">
                <div className="num text-xl text-fg">{compact.format(Number(v))}</div>
                <div className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">{k}</div>
              </div>
            ))}
          </div>
          <ul className="divide-y divide-line/60">
            {data.slice(0, 6).map(({ corpus, stats }) => (
              <li key={corpus.id}>
                <Link href={`/corpus/${corpus.id}`} className="flex items-center justify-between gap-3 px-4 py-2.5 transition-colors hover:bg-surface-2">
                  <span className="min-w-0 truncate font-mono text-xs text-fg">{corpus.name}</span>
                  <span className="flex shrink-0 items-center gap-3 text-[11px] text-fg-subtle">
                    <span className="num">{stats.document_count} docs · {formatBytes(stats.total_bytes)}</span>
                    <Badge tone="trace">v{corpus.version}</Badge>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </>
      )}
    </Panel>
  );
}
