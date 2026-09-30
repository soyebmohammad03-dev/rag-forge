"use client";

import type { DocumentSummary } from "@rag-forge/shared";
import { ArrowLeft, FileText, ScanSearch } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import Link from "next/link";
import { useCallback, useState } from "react";
import { Badge, OriginBadge } from "@/components/ui/badge";
import { type Column, DataTable } from "@/components/ui/data-table";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { Tabs } from "@/components/ui/tabs";
import { api } from "@/lib/api";
import { compact, formatBytes, shortHash, timeAgo } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { ChunkingSpec, Field, Stat } from "./bits";
import { DenseIndexPanel, useDenseIndex } from "@/features/retrieval/dense-index";
import { DocumentDrawer } from "./document-drawer";
import { IngestionLog, VersionTimeline } from "./history";
import { Uploader } from "./uploader";

const TYPE_LABEL: Record<string, string> = {
  "text/plain": "TXT",
  "text/markdown": "MD",
  "application/pdf": "PDF",
};

const columns: Column<DocumentSummary>[] = [
  {
    key: "file",
    header: "Document",
    cell: (d) => (
      <div className="flex min-w-0 items-center gap-2.5">
        <span className="grid h-6 w-9 shrink-0 place-items-center rounded border border-line-strong font-mono text-[9px] text-fg-muted">
          {TYPE_LABEL[d.current.media_type] ?? "?"}
        </span>
        <span className="truncate font-mono text-[12px] text-fg" title={d.filename}>{d.filename}</span>
      </div>
    ),
  },
  { key: "size", header: "Size", align: "right", cell: (d) => <span className="num">{formatBytes(d.current.byte_size)}</span>, className: "hidden sm:table-cell" },
  { key: "chars", header: "Text", align: "right", cell: (d) => <span className="num">{compact.format(d.current.text_chars)}</span>, className: "hidden md:table-cell" },
  {
    key: "version",
    header: "Version",
    cell: (d) => (
      <span className="num font-mono text-[11px]">
        v{d.current.version}
        {d.version_count > 1 && <span className="text-fg-subtle"> / {d.version_count}</span>}
      </span>
    ),
  },
  {
    key: "extraction",
    header: "Extraction",
    cell: (d) => <Badge tone={d.current.extraction_status === "complete" ? "ok" : "signal"}>{d.current.extraction_status}</Badge>,
    className: "hidden lg:table-cell",
  },
  { key: "hash", header: "SHA-256", cell: (d) => <span className="font-mono text-[10px] text-fg-subtle">{shortHash(d.current.content_sha256)}</span>, className: "hidden xl:table-cell" },
  { key: "when", header: "Ingested", align: "right", cell: (d) => <span className="whitespace-nowrap">{timeAgo(d.current.created_at)}</span>, className: "hidden sm:table-cell" },
];

export function CorpusDetailPage({ corpusId }: { corpusId: string }) {
  const [rev, setRev] = useState(0);
  const bump = useCallback(() => setRev((r) => r + 1), []);
  const [tab, setTab] = useState("documents");
  const [openDoc, setOpenDoc] = useState<string | null>(null);

  const fetchCorpus = useCallback(
    () => api.GET("/api/v1/corpora/{corpus_id}", { params: { path: { corpus_id: corpusId } } }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- rev forces a refetch after ingestion
    [corpusId, rev],
  );
  const summary = useApi(fetchCorpus);
  const dense = useDenseIndex(corpusId, null, rev);

  if (summary.error && !summary.data)
    return (
      <Panel className="mx-auto max-w-3xl">
        <ErrorState title="Corpus unavailable">
          {summary.error}. <Link href="/corpus" className="text-trace underline-offset-4 hover:underline">Back to corpora</Link>
        </ErrorState>
      </Panel>
    );
  if (!summary.data)
    return (
      <Panel className="mx-auto max-w-[1440px]">
        <LoadingState rows={6} label="Loading corpus" />
      </Panel>
    );

  const { corpus, stats } = summary.data;
  return (
    <div className="mx-auto max-w-[1440px] space-y-5">
      <Link href="/corpus" className="inline-flex items-center gap-1.5 font-mono text-[11px] text-fg-subtle transition-colors hover:text-fg">
        <ArrowLeft className="size-3.5" /> All corpora
      </Link>

      <header className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <h1 className="truncate font-mono text-2xl tracking-tight">{corpus.name}</h1>
            <AnimatePresence mode="popLayout" initial={false}>
              <motion.span
                key={corpus.version}
                initial={{ y: 10, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                exit={{ y: -10, opacity: 0 }}
                transition={{ type: "spring", stiffness: 400, damping: 30 }}
              >
                <Badge tone="signal" className="h-6 text-[11px]">v{corpus.version}</Badge>
              </motion.span>
            </AnimatePresence>
            <OriginBadge origin="live" />
          </div>
          <p className="mt-1 max-w-2xl text-[13px] text-fg-muted">{corpus.description || "No description."}</p>
        </div>
        <div className="flex shrink-0 items-center gap-4">
          <ChunkingSpec config={corpus.chunking} />
          <Link
            href={`/retrieval?corpus=${corpus.id}`}
            className="inline-flex h-8 items-center gap-2 rounded-md bg-surface-3 px-3 text-[13px] text-fg shadow-[0_0_0_1px_var(--color-line-strong)] transition-colors hover:bg-[#222731]"
          >
            <ScanSearch className="size-4 text-signal" /> Search corpus
          </Link>
        </div>
      </header>

      <section aria-label="Corpus statistics" className="grid grid-cols-2 gap-px overflow-hidden rounded-[var(--radius-panel)] bg-line shadow-[var(--shadow-panel)] sm:grid-cols-5">
        <Stat label="version" value={`v${corpus.version}`} />
        <Stat label="documents" value={compact.format(stats.document_count)} />
        <Stat label="chunks" value={compact.format(stats.chunk_count)} />
        <Stat label="source size" value={formatBytes(stats.total_bytes)} />
        <Stat label="extracted chars" value={compact.format(stats.total_chars)} className="col-span-2 sm:col-span-1" />
      </section>

      <div className="grid items-start gap-5 xl:grid-cols-3">
        <Panel className="min-w-0 xl:col-span-2">
          <PanelHeader
            eyebrow={`Corpus v${corpus.version}`}
            title={tab === "documents" ? "Documents" : tab === "versions" ? "Version history" : "Ingestion provenance"}
            actions={
              <Tabs
                value={tab}
                onChange={setTab}
                tabs={[
                  { id: "documents", label: <>Documents <span className="num ml-1 text-fg-subtle">{stats.document_count}</span></> },
                  { id: "versions", label: <>Versions <span className="num ml-1 text-fg-subtle">{corpus.version + 1}</span></> },
                  { id: "ingestions", label: "Ingestions" },
                ]}
              />
            }
          />
          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={tab} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.12 }}>
              {tab === "documents" && (
                <Documents corpusId={corpus.id} currentVersion={corpus.version} rev={rev} onOpen={setOpenDoc} />
              )}
              {tab === "versions" && <VersionTimeline corpusId={corpus.id} rev={rev} />}
              {tab === "ingestions" && <IngestionLog corpusId={corpus.id} rev={rev} />}
            </motion.div>
          </AnimatePresence>
        </Panel>

        <div className="space-y-5">
          <Panel>
            <PanelHeader eyebrow="Ingest" title="Add or update documents" />
            <div className="p-4">
              <Uploader corpusId={corpus.id} onIngested={bump} />
              <p className="mt-3 text-[11px] leading-relaxed text-fg-subtle">
                Same filename with new content creates a new document version. Identical content is detected by
                SHA-256 and skipped. Unsupported or unreadable files are rejected with a reason.
              </p>
            </div>
          </Panel>
          <DenseIndexPanel handle={dense} />
          <Panel>
            <PanelHeader eyebrow="Identity" title="Corpus record" />
            <dl className="divide-y divide-line">
              <Field label="ID"><span className="font-mono">{corpus.id}</span></Field>
              <Field label="Created">{new Date(corpus.created_at).toLocaleString()}</Field>
              <Field label="Strategy"><span className="font-mono">{corpus.chunking.strategy}</span></Field>
              <Field label="Last ingestion">{stats.last_ingestion ? timeAgo(stats.last_ingestion.finished_at) : "never"}</Field>
              {Object.entries(corpus.metadata).map(([k, v]) => (
                <Field key={k} label={k}><span className="font-mono">{String(v)}</span></Field>
              ))}
            </dl>
          </Panel>
        </div>
      </div>

      {openDoc && (
        <DocumentDrawer
          key={openDoc}
          corpusId={corpus.id}
          documentId={openDoc}
          corpusVersion={corpus.version}
          onClose={() => setOpenDoc(null)}
          onChanged={bump}
        />
      )}
    </div>
  );
}

function Documents({
  corpusId,
  currentVersion,
  rev,
  onOpen,
}: {
  corpusId: string;
  currentVersion: number;
  rev: number;
  onOpen: (id: string) => void;
}) {
  const [asOf, setAsOf] = useState<number | null>(null);
  const version = asOf ?? currentVersion;
  const fetchDocs = useCallback(
    () =>
      api.GET("/api/v1/corpora/{corpus_id}/documents", {
        params: { path: { corpus_id: corpusId }, query: { version } },
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- rev forces a refetch after ingestion
    [corpusId, version, rev],
  );
  const { data, error, loading } = useApi(fetchDocs);

  return (
    <>
      {currentVersion > 0 && (
        <div className="flex items-center justify-between gap-3 border-b border-line bg-surface-2/40 px-4 py-2 text-[11px] text-fg-muted">
          <label className="flex items-center gap-2">
            View as of
            <select
              value={version}
              onChange={(e) => setAsOf(Number(e.target.value) === currentVersion ? null : Number(e.target.value))}
              className="rounded border border-line-strong bg-surface px-1.5 py-0.5 font-mono text-[11px] text-fg"
            >
              {Array.from({ length: currentVersion + 1 }, (_, i) => currentVersion - i).map((v) => (
                <option key={v} value={v}>
                  v{v}
                  {v === currentVersion ? " (current)" : ""}
                </option>
              ))}
            </select>
          </label>
          {asOf !== null && <Badge tone="signal">historical snapshot</Badge>}
        </div>
      )}
      {loading && !data ? (
        <LoadingState rows={5} label="Loading documents" />
      ) : error ? (
        <ErrorState title="Could not load documents">{error}</ErrorState>
      ) : data?.length ? (
        <DataTable caption="Documents" columns={columns} rows={data} rowKey={(d) => d.document_id} onRowClick={(d) => onOpen(d.document_id)} />
      ) : (
        <EmptyState icon={FileText} title={version === 0 ? "No documents yet" : "No documents in this version"}>
          {version === 0 ? "Upload files to create the first version of this corpus." : "Every document was removed in this version."}
        </EmptyState>
      )}
    </>
  );
}
