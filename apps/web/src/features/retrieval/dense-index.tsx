"use client";

import type { DenseIndexState, DenseIndexView } from "@rag-forge/shared";
import { Boxes, RefreshCw } from "lucide-react";
import { motion } from "motion/react";
import { useCallback, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Panel, PanelHeader } from "@/components/ui/panel";
import { ErrorState, LoadingState } from "@/components/ui/states";
import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import { shortHash, timeAgo } from "@/lib/format";
import { useApi } from "@/lib/use-api";
import { errorMessage } from "./errors";
import { Field } from "@/features/corpus/bits";

export const DENSE_STATE: Record<DenseIndexState, { tone: "ok" | "signal" | "neutral" | "trace" | "err"; label: string; help: string }> = {
  ready: { tone: "ok", label: "ready", help: "Dense retrieval is available for this version." },
  building: { tone: "signal", label: "building", help: "Embedding chunks. Dense retrieval becomes available when it finishes." },
  missing: { tone: "neutral", label: "missing", help: "No dense index for this version yet. Build one from the stored chunks." },
  stale: { tone: "trace", label: "stale", help: "Other versions (or another embedding model) are indexed, but not this version." },
  failed: { tone: "err", label: "failed", help: "The last build failed. Rebuild to retry." },
};

// Poll only while a build is running.
const pollWhileBuilding = (v: DenseIndexView | undefined) => (v?.state === "building" ? 800 : undefined);

export function useDenseIndex(corpusId: string, version: number | null, rev = 0) {
  const fetcher = useCallback(
    () =>
      api.GET("/api/v1/corpora/{corpus_id}/dense-index", {
        params: { path: { corpus_id: corpusId }, query: version === null ? {} : { version } },
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- rev forces a refetch after ingestion
    [corpusId, version, rev],
  );
  const status = useApi(fetcher, pollWhileBuilding);
  const [buildError, setBuildError] = useState<string | null>(null);
  const build = async () => {
    setBuildError(null);
    try {
      const { error, response } = await api.POST("/api/v1/corpora/{corpus_id}/dense-index", {
        params: { path: { corpus_id: corpusId } },
        body: { version },
      });
      if (!response.ok) setBuildError(errorMessage(error, response.status));
    } catch {
      setBuildError("API unreachable");
    }
    status.reload();
  };
  return { ...status, build, buildError };
}

export type DenseIndexHandle = ReturnType<typeof useDenseIndex>;

export function DenseStateBadge({ state }: { state: DenseIndexState }) {
  const s = DENSE_STATE[state];
  return (
    <Badge tone={s.tone} title={s.help}>
      {state === "building" && <span className="size-1.5 animate-pulse-soft rounded-full bg-signal" />}
      dense {s.label}
    </Badge>
  );
}

/** Build progress, identity and actions for the dense index of one corpus version. */
export function DenseIndexPanel({ handle, className }: { handle: DenseIndexHandle; className?: string }) {
  const { data, error, loading, build, buildError } = handle;
  return (
    <Panel className={className}>
      <PanelHeader
        eyebrow="Dense index"
        title={data ? `Corpus v${data.version}` : "Semantic retrieval"}
        actions={data && <DenseStateBadge state={data.state} />}
      />
      {loading && !data && <LoadingState rows={3} label="Loading dense index status" />}
      {error && !data && <ErrorState title="Status unavailable">{error}</ErrorState>}
      {data && <DenseIndexBody view={data} build={build} buildError={buildError} />}
    </Panel>
  );
}

function DenseIndexBody({ view, build, buildError }: { view: DenseIndexView; build: () => void; buildError: string | null }) {
  const { state, index, embedder } = view;
  const progress = index && index.chunk_count ? index.embedded / index.chunk_count : 0;
  const canBuild = state === "missing" || state === "stale" || state === "failed";
  return (
    <div>
      <p className="px-4 pt-3 text-xs leading-relaxed text-fg-muted">{DENSE_STATE[state].help}</p>
      {state === "building" && index && (
        <div className="px-4 pt-3" role="status" aria-label="Build progress">
          <div className="flex justify-between font-mono text-[11px] text-fg-muted">
            <span>embedding chunks</span>
            <span className="num">
              {index.embedded} / {index.chunk_count}
            </span>
          </div>
          <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-surface-3">
            <motion.div className="h-full rounded-full bg-s-dense" animate={{ width: `${progress * 100}%` }} transition={{ ease: "easeOut" }} />
          </div>
        </div>
      )}
      {state === "failed" && index?.error && <p className="px-4 pt-2 font-mono text-[11px] text-err">{index.error}</p>}
      <dl className="mt-3 divide-y divide-line border-t border-line">
        <Field label="Model"><span className="font-mono" title={`${embedder.model}@${embedder.revision}`}>{embedder.model}</span></Field>
        <Field label="Revision"><span className="font-mono">{shortHash(embedder.revision, 10)}</span></Field>
        {index && state === "ready" && (
          <>
            <Field label="Chunks"><span className="num">{index.chunk_count} · {index.embedder.dimension}d · {index.similarity}</span></Field>
            <Field label="Reused vectors"><span className="num">{index.reused}</span></Field>
            <Field label="Index"><span className="font-mono" title={index.id}>{index.id}</span></Field>
            <Field label="Content hash"><span className="font-mono text-trace" title={index.content_hash ?? ""}>{shortHash(index.content_hash ?? "", 12)}</span></Field>
            <Field label="Built">{index.finished_at ? timeAgo(index.finished_at) : "—"}</Field>
          </>
        )}
      </dl>
      {(canBuild || buildError) && (
        <div className="border-t border-line p-3">
          {canBuild && (
            <Button size="sm" variant={state === "failed" ? "secondary" : "primary"} onClick={build}>
              {state === "failed" ? <RefreshCw className="size-3.5" /> : <Boxes className="size-3.5" />}
              {state === "failed" ? "Rebuild dense index" : "Build dense index"}
            </Button>
          )}
          {buildError && <p role="alert" className={cn("text-[11px] text-err", canBuild && "mt-2")}>{buildError}</p>}
        </div>
      )}
    </div>
  );
}
