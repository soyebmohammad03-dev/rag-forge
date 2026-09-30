"use client";

import type { CorpusSummary } from "@rag-forge/shared";
import { Database, Plus } from "lucide-react";
import { motion } from "motion/react";
import Link from "next/link";
import { useState } from "react";
import { Badge, OriginBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Panel } from "@/components/ui/panel";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { compact, formatBytes, timeAgo } from "@/lib/format";
import { fetchCorpora, useApi } from "@/lib/use-api";
import { ChunkingSpec, IngestionStatusBadge } from "./bits";
import { CreateCorpusDialog } from "./create-corpus-dialog";

export function CorpusListPage() {
  const { data, error, loading, reload } = useApi(fetchCorpora);
  const [creating, setCreating] = useState(false);

  return (
    <div className="mx-auto max-w-[1440px] space-y-5">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-signal">Data</p>
          <h1 className="mt-1.5 text-2xl font-medium tracking-tight">Corpus</h1>
          <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-fg-muted">
            Every change to a corpus creates an immutable version. Experiments pin a version, so results stay
            reproducible after documents change.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {data && <OriginBadge origin="live" />}
          <Button variant="primary" onClick={() => setCreating(true)}>
            <Plus className="size-4" /> New corpus
          </Button>
        </div>
      </header>

      {loading && !data && (
        <Panel>
          <LoadingState rows={4} label="Loading corpora" />
        </Panel>
      )}
      {error && !data && (
        <Panel>
          <ErrorState title="Could not load corpora" action={<Button size="sm" onClick={reload}>Retry</Button>}>
            {error}
          </ErrorState>
        </Panel>
      )}
      {data?.length === 0 && (
        <Panel>
          <EmptyState
            icon={Database}
            title="No corpora yet"
            action={<Button variant="primary" size="sm" onClick={() => setCreating(true)}><Plus className="size-3.5" /> New corpus</Button>}
          >
            Create a corpus, then upload .txt, .md or .pdf files. Each upload becomes a new corpus version.
          </EmptyState>
        </Panel>
      )}
      {!!data?.length && (
        <ul className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {data.map((c, i) => (
            <motion.li
              key={c.corpus.id}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i, 8) * 0.03, duration: 0.2 }}
            >
              <CorpusCard summary={c} />
            </motion.li>
          ))}
        </ul>
      )}

      <CreateCorpusDialog open={creating} onClose={() => setCreating(false)} />
    </div>
  );
}

function CorpusCard({ summary: { corpus, stats } }: { summary: CorpusSummary }) {
  const last = stats.last_ingestion;
  return (
    <Link
      href={`/corpus/${corpus.id}`}
      className="group block h-full rounded-[var(--radius-panel)] focus-visible:outline-offset-4"
    >
      <Panel className="h-full p-4 transition-shadow group-hover:shadow-[0_0_0_1px_var(--color-line-strong),0_12px_32px_-16px_rgb(0_0_0/0.6)]">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="truncate font-mono text-sm text-fg group-hover:text-signal">{corpus.name}</h2>
            <p className="mt-0.5 line-clamp-1 text-xs text-fg-subtle">{corpus.description || "No description"}</p>
          </div>
          <Badge tone="trace">v{corpus.version}</Badge>
        </div>
        <dl className="mt-4 grid grid-cols-3 gap-2">
          {[
            ["docs", compact.format(stats.document_count)],
            ["chunks", compact.format(stats.chunk_count)],
            ["size", formatBytes(stats.total_bytes)],
          ].map(([k, v]) => (
            <div key={k}>
              <dd className="num text-base text-fg">{v}</dd>
              <dt className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">{k}</dt>
            </div>
          ))}
        </dl>
        <div className="mt-4 flex flex-wrap items-center justify-between gap-x-3 gap-y-2 border-t border-line pt-3">
          <ChunkingSpec config={corpus.chunking} />
          {last ? (
            <span className="flex items-center gap-2 whitespace-nowrap text-[11px] text-fg-subtle">
              {timeAgo(last.finished_at)} <IngestionStatusBadge status={last.status} />
            </span>
          ) : (
            <span className="text-[11px] text-fg-subtle">never ingested</span>
          )}
        </div>
      </Panel>
    </Link>
  );
}
