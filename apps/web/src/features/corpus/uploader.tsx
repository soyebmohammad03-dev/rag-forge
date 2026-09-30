"use client";

import type { IngestionRecord } from "@rag-forge/shared";
import { FileUp, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { type DragEvent, useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cn";
import { formatBytes, shortHash } from "@/lib/format";
import { type UploadPhase, uploadDocuments } from "@/lib/upload";
import { fetchFormats, useApi } from "@/lib/use-api";
import { IngestionStatusBadge, OutcomeBadge } from "./bits";

type State =
  | { kind: "idle" }
  | { kind: "busy"; files: File[]; phase: UploadPhase }
  | { kind: "done"; record: IngestionRecord }
  | { kind: "error"; message: string };

export function Uploader({ corpusId, onIngested }: { corpusId: string; onIngested: () => void }) {
  const formats = useApi(fetchFormats);
  const [state, setState] = useState<State>({ kind: "idle" });
  const [dragging, setDragging] = useState(false);
  const accept = formats.data?.flatMap((f) => f.extensions).join(",");

  const send = async (list: FileList | null) => {
    const files = Array.from(list ?? []);
    if (!files.length || state.kind === "busy") return;
    setState({ kind: "busy", files, phase: { phase: "uploading", fraction: 0 } });
    try {
      const record = await uploadDocuments(corpusId, files, (phase) =>
        setState((s) => (s.kind === "busy" ? { ...s, phase } : s)),
      );
      setState({ kind: "done", record });
      onIngested();
    } catch (e) {
      setState({ kind: "error", message: (e as Error).message });
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    send(e.dataTransfer.files);
  };

  const busy = state.kind === "busy";
  const totalBytes = busy ? state.files.reduce((s, f) => s + f.size, 0) : 0;

  return (
    <div className="space-y-3">
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={cn(
          "relative flex cursor-pointer flex-col items-center justify-center overflow-hidden rounded-lg border border-dashed px-6 py-7 text-center transition-colors has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-trace",
          dragging ? "border-signal bg-signal-dim/30" : "border-line-strong bg-surface-2/40 hover:border-fg-subtle",
          busy && "pointer-events-none",
        )}
      >
        <input
          type="file"
          multiple
          accept={accept}
          className="sr-only"
          disabled={busy}
          onChange={(e) => {
            send(e.target.files);
            e.target.value = "";
          }}
        />
        <AnimatePresence mode="wait" initial={false}>
          {busy ? (
            <motion.div key="busy" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="w-full max-w-md" role="status" aria-live="polite">
              <div className="flex justify-between font-mono text-[11px] text-fg-muted">
                <span>
                  {state.phase.phase === "uploading"
                    ? `Uploading ${state.files.length} file${state.files.length > 1 ? "s" : ""}`
                    : "Extracting, hashing and chunking on the server"}
                </span>
                <span className="num">
                  {state.phase.phase === "uploading" ? `${Math.round(state.phase.fraction * 100)}% of ${formatBytes(totalBytes)}` : "…"}
                </span>
              </div>
              <div className="mt-2 h-1 overflow-hidden rounded-full bg-surface-3">
                {state.phase.phase === "uploading" ? (
                  <motion.div className="h-full rounded-full bg-signal" animate={{ width: `${state.phase.fraction * 100}%` }} transition={{ ease: "easeOut" }} />
                ) : (
                  <motion.div
                    className="h-full w-1/3 rounded-full bg-trace"
                    animate={{ x: ["-100%", "300%"] }}
                    transition={{ repeat: Infinity, duration: 1.1, ease: "easeInOut" }}
                  />
                )}
              </div>
            </motion.div>
          ) : (
            <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center">
              <FileUp className={cn("size-5 transition-colors", dragging ? "text-signal" : "text-fg-subtle")} />
              <p className="mt-2 text-[13px] text-fg">
                Drop files or <span className="text-signal underline-offset-4 hover:underline">browse</span>
              </p>
              <p className="mt-1 text-balance font-mono text-[10px] uppercase tracking-wider text-fg-subtle">
                {accept ? accept.replaceAll(",", " ") : "…"} · max 25 MiB each
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </label>

      <AnimatePresence>
        {state.kind === "error" && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} role="alert" className="flex items-center justify-between rounded-md border border-err/25 bg-err/5 px-3 py-2 text-xs text-err">
            Upload failed: {state.message}
            <Button variant="ghost" size="icon" aria-label="Dismiss" onClick={() => setState({ kind: "idle" })}><X className="size-3.5" /></Button>
          </motion.div>
        )}
        {state.kind === "done" && <IngestionResult record={state.record} onDismiss={() => setState({ kind: "idle" })} />}
      </AnimatePresence>
    </div>
  );
}

function IngestionResult({ record, onDismiss }: { record: IngestionRecord; onDismiss: () => void }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, height: 0 }}
      className="overflow-hidden rounded-lg border border-line bg-surface-2/50"
      aria-live="polite"
    >
      <header className="flex items-center justify-between gap-3 border-b border-line px-3 py-2">
        <div className="flex items-center gap-2 text-xs">
          <IngestionStatusBadge status={record.status} />
          <span className="text-fg-muted">
            {record.status === "completed" ? (
              <>corpus <span className="num text-fg">v{record.version_before}</span> → <span className="num text-signal">v{record.version_after}</span></>
            ) : (
              <>corpus stays at <span className="num text-fg">v{record.version_after}</span></>
            )}
          </span>
        </div>
        <Button variant="ghost" size="icon" aria-label="Dismiss result" onClick={onDismiss}><X className="size-3.5" /></Button>
      </header>
      <ul className="divide-y divide-line/60">
        {record.files.map((f, i) => (
          <motion.li
            key={`${f.filename}-${i}`}
            initial={{ opacity: 0, x: -4 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: Math.min(i, 12) * 0.035 }}
            className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-0.5 px-3 py-2 sm:grid-cols-[minmax(0,1fr)_7rem_6rem_auto]"
          >
            <span className="truncate font-mono text-xs text-fg" title={f.filename}>{f.filename}</span>
            <span className="hidden font-mono text-[10px] text-fg-subtle sm:block">{f.content_sha256 ? shortHash(f.content_sha256) : "—"}</span>
            <span className="num hidden text-right text-[11px] text-fg-subtle sm:block">
              {f.chunk_count != null ? `${f.chunk_count} chunks` : f.byte_size != null ? formatBytes(f.byte_size) : ""}
            </span>
            <span className="justify-self-end"><OutcomeBadge outcome={f.outcome} /></span>
            {(f.error || f.warnings.length > 0 || f.duplicate_of) && (
              <span className={cn("col-span-full text-[11px]", f.error ? "text-err" : "text-fg-subtle")}>
                {f.error ?? (f.duplicate_of ? `same content as ${f.duplicate_of_filename ?? f.duplicate_of}` : f.warnings.join("; "))}
              </span>
            )}
          </motion.li>
        ))}
      </ul>
    </motion.section>
  );
}
