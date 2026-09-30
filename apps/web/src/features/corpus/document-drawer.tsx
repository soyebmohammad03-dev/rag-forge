"use client";

import type { Chunk, DocumentVersion } from "@rag-forge/shared";
import { Trash2 } from "lucide-react";
import { useCallback, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { ErrorState, LoadingState } from "@/components/ui/states";
import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import { formatBytes, shortHash, timeAgo } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { Field } from "./bits";
import { ChunkList } from "./chunk-preview";

const PAGE = 50;

export function DocumentDrawer({
  corpusId,
  documentId,
  corpusVersion,
  onClose,
  onChanged,
}: {
  corpusId: string;
  documentId: string;
  corpusVersion: number;
  onClose: () => void;
  onChanged: () => void;
}) {
  const fetchDetail = useCallback(
    () =>
      api.GET("/api/v1/corpora/{corpus_id}/documents/{document_id}", {
        params: { path: { corpus_id: corpusId, document_id: documentId } },
      }),
    [corpusId, documentId],
  );
  const detail = useApi(fetchDetail);
  const doc = detail.data;
  const [selected, setSelected] = useState<string | null>(null);
  const shown = selected ?? doc?.current_version_id;
  const version = doc?.versions.find((v) => v.id === shown) ?? doc?.versions.at(-1);

  return (
    <Dialog
      variant="drawer"
      open
      onClose={onClose}
      title={<span className="font-mono">{doc?.document.filename ?? "Document"}</span>}
      description={doc && <span className="font-mono">{doc.document.id}</span>}
    >
      {!doc && detail.loading && <LoadingState rows={6} label="Loading document" />}
      {!doc && detail.error && <ErrorState title="Could not load document">{detail.error}</ErrorState>}
      {doc && version && (
        <div className="space-y-6">
          <section>
            <SectionTitle>Version history</SectionTitle>
            <ol className="space-y-1" aria-label="Document versions">
              {[...doc.versions].reverse().map((v) => (
                <li key={v.id}>
                  <button
                    type="button"
                    onClick={() => setSelected(v.id)}
                    aria-pressed={v.id === version.id}
                    className={cn(
                      "flex w-full items-center gap-3 rounded-md border px-3 py-2 text-left text-xs transition-colors",
                      v.id === version.id ? "border-line-strong bg-surface-3" : "border-transparent hover:bg-surface",
                    )}
                  >
                    <span className="num w-7 font-mono text-fg">v{v.version}</span>
                    <span className="font-mono text-[10px] text-fg-subtle">{shortHash(v.content_sha256)}</span>
                    <span className="num text-fg-subtle">{formatBytes(v.byte_size)}</span>
                    <span className="ml-auto text-fg-subtle">{timeAgo(v.created_at)}</span>
                    {v.id === doc.current_version_id && <Badge tone="ok">in v{corpusVersion}</Badge>}
                  </button>
                </li>
              ))}
            </ol>
            {doc.current_version_id === null && (
              <p className="mt-2 text-[11px] text-fg-subtle">Removed from the current corpus version; history is kept.</p>
            )}
          </section>

          <VersionMetadata v={version} />

          <section>
            <SectionTitle>Chunks · v{version.version}</SectionTitle>
            <Chunks key={version.id} versionId={version.id} textChars={version.text_chars} />
          </section>

          {version.id === doc.current_version_id && (
            <RemoveButton corpusId={corpusId} documentId={doc.document.id} corpusVersion={corpusVersion} onRemoved={() => { onChanged(); onClose(); }} />
          )}
        </div>
      )}
    </Dialog>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <h3 className="mb-2 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-subtle">{children}</h3>;
}

function VersionMetadata({ v }: { v: DocumentVersion }) {
  const meta = Object.entries(v.metadata).filter(([, val]) => val !== null && val !== undefined);
  return (
    <section>
      <SectionTitle>Metadata</SectionTitle>
      <dl className="divide-y divide-line rounded-lg border border-line">
        <Field label="Type">{v.media_type}</Field>
        <Field label="Size">{formatBytes(v.byte_size)} · {v.text_chars.toLocaleString()} chars extracted</Field>
        <Field label="SHA-256"><span className="font-mono text-[10px]" title={v.content_sha256}>{v.content_sha256}</span></Field>
        <Field label="Parser"><span className="font-mono">{v.parser}</span></Field>
        <Field label="Extraction">
          <Badge tone={v.extraction_status === "complete" ? "ok" : "signal"}>{v.extraction_status}</Badge>
        </Field>
        <Field label="Ingested">{new Date(v.created_at).toLocaleString()}</Field>
        {meta.map(([k, val]) => (
          <Field key={k} label={k}><span className="font-mono">{String(val)}</span></Field>
        ))}
      </dl>
      {v.extraction_warnings.length > 0 && (
        <ul className="mt-2 space-y-1 text-[11px] text-signal">
          {v.extraction_warnings.map((w) => <li key={w}>⚠ {w}</li>)}
        </ul>
      )}
    </section>
  );
}

function Chunks({ versionId, textChars }: { versionId: string; textChars: number }) {
  const fetchPage = useCallback(
    (offset = 0) =>
      api.GET("/api/v1/document-versions/{version_id}/chunks", {
        params: { path: { version_id: versionId }, query: { offset, limit: PAGE } },
      }),
    [versionId],
  );
  const first = useApi(fetchPage);
  const [more, setMore] = useState<Chunk[]>([]);
  const [moreError, setMoreError] = useState<string | null>(null);

  if (first.error) return <ErrorState title="Could not load chunks">{first.error}</ErrorState>;
  if (!first.data) return <LoadingState rows={4} label="Loading chunks" />;
  const chunks = [...first.data.items, ...more];
  const total = first.data.total;
  const loadMore = async () => {
    const { data, response } = await fetchPage(chunks.length);
    if (data) setMore((m) => [...m, ...data.items]);
    else setMoreError(`API responded ${response.status}`);
  };
  return (
    <div className="space-y-3">
      <p className="num text-[11px] text-fg-subtle">
        {total} chunk{total === 1 ? "" : "s"}
        {chunks.length < total && ` · showing ${chunks.length}`}
      </p>
      <ChunkList chunks={chunks} textChars={textChars} />
      {chunks.length < total && (
        <Button size="sm" onClick={loadMore}>Load {Math.min(PAGE, total - chunks.length)} more</Button>
      )}
      {moreError && <p role="alert" className="text-[11px] text-err">{moreError}</p>}
    </div>
  );
}

function RemoveButton({
  corpusId,
  documentId,
  corpusVersion,
  onRemoved,
}: {
  corpusId: string;
  documentId: string;
  corpusVersion: number;
  onRemoved: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const remove = async () => {
    const { response } = await api.DELETE("/api/v1/corpora/{corpus_id}/documents/{document_id}", {
      params: { path: { corpus_id: corpusId, document_id: documentId } },
    });
    if (response.ok) onRemoved();
    else setError(`API responded ${response.status}`);
  };
  return (
    <section className="rounded-lg border border-line p-3">
      <p className="text-xs text-fg-muted">
        Removing creates corpus <span className="num text-fg">v{corpusVersion + 1}</span> without this document.
        Earlier versions and their chunks are kept.
      </p>
      <div className="mt-3 flex gap-2">
        {confirming ? (
          <>
            <Button size="sm" className="text-err" onClick={remove}><Trash2 className="size-3.5" /> Confirm removal</Button>
            <Button size="sm" variant="ghost" onClick={() => setConfirming(false)}>Cancel</Button>
          </>
        ) : (
          <Button size="sm" onClick={() => setConfirming(true)}><Trash2 className="size-3.5" /> Remove from corpus</Button>
        )}
      </div>
      {error && <p role="alert" className="mt-2 text-[11px] text-err">{error}</p>}
    </section>
  );
}
