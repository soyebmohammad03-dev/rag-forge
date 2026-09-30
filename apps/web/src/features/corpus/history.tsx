"use client";

import type { CorpusVersion, IngestionRecord } from "@rag-forge/shared";
import { ChevronRight, GitCommitVertical, History } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatBytes, shortHash, timeAgo } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { ChunkingSpec, Field, IngestionStatusBadge, OUTCOME, OutcomeBadge } from "./bits";

const expand = {
  initial: { height: 0, opacity: 0 },
  animate: { height: "auto", opacity: 1 },
  exit: { height: 0, opacity: 0 },
  transition: { duration: 0.2, ease: "easeOut" },
} as const;

function Counts({ v }: { v: Pick<CorpusVersion, "added" | "modified" | "removed" | "unchanged"> }) {
  const items = (["added", "modified", "removed", "unchanged"] as const).filter((k) => v[k] > 0);
  return (
    <span className="flex flex-wrap gap-1">
      {items.map((k) => (
        <Badge key={k} tone={OUTCOME[k].tone} title={k}>
          {OUTCOME[k].sign}
          {v[k]}
        </Badge>
      ))}
    </span>
  );
}

/** Corpus versions as a timeline; each expands to the per-document diff against its parent. */
export function VersionTimeline({ corpusId, rev }: { corpusId: string; rev: number }) {
  const fetchVersions = useCallback(
    () => api.GET("/api/v1/corpora/{corpus_id}/versions", { params: { path: { corpus_id: corpusId } } }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- rev forces a refetch after ingestion
    [corpusId, rev],
  );
  const { data, error, loading } = useApi(fetchVersions);
  const [open, setOpen] = useState<number | null>(null);

  if (loading && !data) return <LoadingState rows={4} label="Loading versions" />;
  if (error && !data) return <ErrorState title="Could not load versions">{error}</ErrorState>;
  return (
    <ol className="relative p-4" aria-label="Corpus versions">
      <span className="absolute bottom-6 left-[27px] top-6 w-px bg-line-strong" aria-hidden />
      {data?.map((v, i) => {
        const isOpen = open === v.version;
        return (
          <motion.li
            key={v.version}
            layout="position"
            initial={i === 0 ? { opacity: 0, y: -6 } : false}
            animate={{ opacity: 1, y: 0 }}
            className="relative pl-9"
          >
            <span
              className={cn(
                "absolute left-[7px] top-3 grid size-[15px] place-items-center rounded-full border bg-surface",
                i === 0 ? "border-signal text-signal" : "border-line-strong text-fg-subtle",
              )}
              aria-hidden
            >
              <GitCommitVertical className="size-2.5" />
            </span>
            <button
              type="button"
              disabled={v.version === 0}
              aria-expanded={v.version === 0 ? undefined : isOpen}
              onClick={() => setOpen(isOpen ? null : v.version)}
              className="flex w-full items-center gap-3 rounded-md px-2 py-2 text-left transition-colors hover:bg-surface-2 disabled:hover:bg-transparent"
            >
              <span className={cn("num w-10 font-mono text-sm", i === 0 ? "text-signal" : "text-fg")}>v{v.version}</span>
              <span className="num w-20 text-[11px] text-fg-muted">{v.document_count} docs</span>
              {v.version === 0 ? <span className="text-[11px] text-fg-subtle">created, empty</span> : <Counts v={v} />}
              <span className="ml-auto whitespace-nowrap text-[11px] text-fg-subtle">{timeAgo(v.created_at)}</span>
              {v.version > 0 && <ChevronRight className={cn("size-3.5 text-fg-subtle transition-transform", isOpen && "rotate-90")} />}
            </button>
            <AnimatePresence initial={false}>
              {isOpen && (
                <motion.div {...expand} className="overflow-hidden">
                  <VersionChanges corpusId={corpusId} version={v.version} />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.li>
        );
      })}
    </ol>
  );
}

function VersionChanges({ corpusId, version }: { corpusId: string; version: number }) {
  const fetchChanges = useCallback(
    () =>
      api.GET("/api/v1/corpora/{corpus_id}/versions/{version}/changes", {
        params: { path: { corpus_id: corpusId, version } },
      }),
    [corpusId, version],
  );
  const { data, error } = useApi(fetchChanges);
  if (error) return <p className="px-2 pb-2 text-[11px] text-err">{error}</p>;
  if (!data) return <LoadingState rows={2} label="Loading changes" />;
  // unchanged documents are implied; list only what moved
  const moved = data.filter((c) => c.change !== "unchanged");
  return (
    <ul className="mb-2 ml-2 space-y-0.5 border-l border-line pl-3">
      {moved.map((c) => (
        <li key={c.document_id} className="flex items-center justify-between gap-3 py-1">
          <span className="truncate font-mono text-[11px] text-fg-muted">{c.filename}</span>
          <OutcomeBadge outcome={c.change} />
        </li>
      ))}
      {data.length > moved.length && (
        <li className="py-1 text-[11px] text-fg-subtle">+ {data.length - moved.length} unchanged</li>
      )}
    </ul>
  );
}

/** Ingestion provenance: every batch, including the ones that changed nothing. */
export function IngestionLog({ corpusId, rev }: { corpusId: string; rev: number }) {
  const fetchLog = useCallback(
    () => api.GET("/api/v1/corpora/{corpus_id}/ingestions", { params: { path: { corpus_id: corpusId } } }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- rev forces a refetch after ingestion
    [corpusId, rev],
  );
  const { data, error, loading } = useApi(fetchLog);
  const [open, setOpen] = useState<string | null>(null);

  if (loading && !data) return <LoadingState rows={4} label="Loading ingestions" />;
  if (error && !data) return <ErrorState title="Could not load ingestions">{error}</ErrorState>;
  if (!data?.length)
    return (
      <EmptyState icon={History} title="No ingestions yet">
        Each upload or removal is recorded here with its parser, chunking configuration, content hashes and
        environment.
      </EmptyState>
    );
  return (
    <ul className="divide-y divide-line/60">
      {data.map((r) => {
        const isOpen = open === r.id;
        const counts = r.files.reduce<Record<string, number>>((m, f) => ({ ...m, [f.outcome]: (m[f.outcome] ?? 0) + 1 }), {});
        return (
          <li key={r.id}>
            <button
              type="button"
              aria-expanded={isOpen}
              onClick={() => setOpen(isOpen ? null : r.id)}
              className="flex w-full flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2.5 text-left transition-colors hover:bg-surface-2"
            >
              <IngestionStatusBadge status={r.status} />
              <span className="num font-mono text-xs text-fg">
                v{r.version_before}
                {r.version_after !== r.version_before && <span className="text-signal"> → v{r.version_after}</span>}
              </span>
              <span className="flex flex-wrap gap-1">
                {Object.entries(counts).map(([k, n]) => (
                  <Badge key={k} tone={OUTCOME[k as keyof typeof OUTCOME].tone}>{n} {k}</Badge>
                ))}
              </span>
              <span className="ml-auto whitespace-nowrap text-[11px] text-fg-subtle">{timeAgo(r.started_at)}</span>
              <ChevronRight className={cn("size-3.5 text-fg-subtle transition-transform", isOpen && "rotate-90")} />
            </button>
            <AnimatePresence initial={false}>
              {isOpen && (
                <motion.div {...expand} className="overflow-hidden">
                  <Provenance record={r} />
                </motion.div>
              )}
            </AnimatePresence>
          </li>
        );
      })}
    </ul>
  );
}

function Provenance({ record: r }: { record: IngestionRecord }) {
  const ms = new Date(r.finished_at).getTime() - new Date(r.started_at).getTime();
  return (
    <div className="grid gap-4 bg-surface-2/40 px-4 py-3 lg:grid-cols-[1fr_300px]">
      <table className="w-full text-left text-[11px]">
        <caption className="sr-only">Files in this ingestion</caption>
        <thead className="font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
          <tr>
            <th className="py-1 font-normal">File</th>
            <th className="py-1 font-normal">SHA-256</th>
            <th className="hidden py-1 font-normal sm:table-cell">Parser</th>
            <th className="py-1 text-right font-normal">Result</th>
          </tr>
        </thead>
        <tbody>
          {r.files.map((f, i) => (
            <tr key={`${f.filename}-${i}`} className="border-t border-line/60 align-top">
              <td className="py-1.5 pr-3">
                <div className="font-mono text-fg">{f.filename}</div>
                <div className="text-fg-subtle">
                  {f.byte_size != null && formatBytes(f.byte_size)}
                  {f.chunk_count != null && ` · ${f.chunk_count} chunks`}
                </div>
                {f.error && <div className="text-err">{f.error}</div>}
              </td>
              <td className="py-1.5 pr-3 font-mono text-fg-subtle">{f.content_sha256 ? shortHash(f.content_sha256, 12) : "—"}</td>
              <td className="hidden py-1.5 pr-3 font-mono text-fg-subtle sm:table-cell">{f.parser ?? "—"}</td>
              <td className="py-1.5 text-right"><OutcomeBadge outcome={f.outcome} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <dl className="h-fit divide-y divide-line rounded-lg border border-line bg-surface">
        <Field label="Record"><span className="font-mono">{r.id}</span></Field>
        <Field label="Chunking"><ChunkingSpec config={r.chunking} /></Field>
        <Field label="Config hash"><span className="font-mono text-trace">{shortHash(r.chunking_hash, 12)}</span></Field>
        <Field label="Duration"><span className="num">{ms} ms</span></Field>
        <Field label="rag-forge"><span className="font-mono">{r.environment.rag_forge_version}</span></Field>
        <Field label="Git commit"><span className="font-mono">{r.environment.git_commit ? shortHash(r.environment.git_commit, 12) : "unavailable"}</span></Field>
        <Field label="Python"><span className="font-mono">{r.environment.python_version}</span></Field>
      </dl>
    </div>
  );
}
