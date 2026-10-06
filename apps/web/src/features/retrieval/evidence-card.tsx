"use client";

import type { RerankCandidate, RetrievalHit } from "@rag-forge/shared";
import { ChevronDown } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import Link from "next/link";
import { type ReactNode, useState } from "react";
import { OriginBadge } from "@/components/ui/badge";
import { STRATEGY_COLOR } from "@/components/ui/chart";
import { cn } from "@/lib/cn";
import { FusionExplain, byDisplayOrder } from "./fusion-explain";
import { MovementBadge, MovementExplain } from "./rerank-analysis";

const norm = (w: string) => w.normalize("NFKC").toLocaleLowerCase();

/**
 * Mark words the retriever matched. Display-only approximation of the server analyzer
 * (NFKC + casefold over word characters); the authoritative list is `matched_terms`.
 */
export function highlight(text: string, terms: string[]): ReactNode[] {
  if (!terms.length) return [text];
  const set = new Set(terms);
  return text.split(/([\p{L}\p{N}_]+)/u).map((part, i) =>
    i % 2 && set.has(norm(part)) ? (
      <mark key={i} className="rounded-sm bg-signal/20 px-px text-fg">
        {part}
      </mark>
    ) : (
      part
    ),
  );
}

const SCORE_TITLE: Record<string, string> = {
  sparse: "BM25 score (unbounded; only comparable within this query)",
  dense: "Cosine similarity to the query embedding",
  hybrid: "Fused score (see the fusion explanation)",
};
const UPSTREAM: Record<string, string> = { sparse: "BM25", dense: "Dense", hybrid: "Hybrid" };
const RERANK_COLOR = STRATEGY_COLOR.rerank;

export function EvidenceCard({
  hit,
  topScore,
  corpusId,
  index,
  rrfK,
  pool,
  scoreRange,
}: {
  hit: RetrievalHit;
  topScore: number;
  corpusId: string;
  index: number;
  rrfK?: number;
  pool?: RerankCandidate[]; // the scored candidate pool, when the request was reranked
  scoreRange?: [number, number]; // reranker scores can be negative, so bars use the pool's range
}) {
  const { result, chunk, fusion, rerank } = hit;
  const [open, setOpen] = useState<"fusion" | "rerank" | null>(null);
  const toggle = (what: "fusion" | "rerank") => setOpen((o) => (o === what ? null : what));
  const pages = chunk.metadata.page_start as number | undefined;
  const pageEnd = chunk.metadata.page_end as number | undefined;
  const relative =
    rerank && scoreRange
      ? scoreRange[1] > scoreRange[0]
        ? (result.score - scoreRange[0]) / (scoreRange[1] - scoreRange[0])
        : 1
      : topScore > 0
        ? result.score / topScore
        : 0;
  const color = STRATEGY_COLOR[result.strategy] ?? "var(--color-fg-muted)";
  const scoreColor = rerank ? RERANK_COLOR : color;
  const upstream = UPSTREAM[result.strategy] ?? result.strategy;
  return (
    <motion.article
      layout="position"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(index, 10) * 0.035, duration: 0.2 }}
      aria-label={`Rank ${result.rank}: ${hit.filename}`}
      className="relative overflow-hidden rounded-lg border border-line bg-surface pl-4"
    >
      <span className="absolute inset-y-0 left-0 w-1 opacity-70" style={{ background: color }} aria-hidden />
      <header className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-line/70 py-2.5 pr-3">
        <span className="num w-7 font-mono text-sm" style={{ color }}>#{result.rank}</span>
        <Link
          href={`/corpus/${corpusId}`}
          className="min-w-0 truncate font-mono text-xs text-fg underline-offset-4 hover:underline"
          title={`${hit.filename} (${result.document_id})`}
        >
          {hit.filename}
        </Link>
        <span className="font-mono text-[10px] text-fg-subtle">
          doc v{hit.document_version} · chunk #{chunk.ordinal} · {chunk.char_start}–{chunk.char_end}
          {pages !== undefined && ` · p.${pages}${pageEnd !== pages ? `–${pageEnd}` : ""}`}
        </span>
        <span className="ml-auto flex items-center gap-2">
          {rerank && <MovementBadge delta={rerank.rank_delta} entered={rerank.entered_top_k} />}
          <span className="relative h-1 w-16 overflow-hidden rounded-full bg-surface-3" aria-hidden>
            <motion.span
              className="absolute inset-y-0 left-0 rounded-full"
              style={{ background: scoreColor }}
              initial={{ width: 0 }}
              animate={{ width: `${relative * 100}%` }}
              transition={{ delay: Math.min(index, 10) * 0.035 + 0.1, duration: 0.4, ease: "easeOut" }}
            />
          </span>
          <span
            className="num font-mono text-xs text-fg"
            title={rerank ? "Cross-encoder relevance score (reranker logit; only comparable within this query)" : SCORE_TITLE[result.strategy]}
          >
            {result.score.toFixed(!rerank && fusion?.method === "rrf" ? 5 : 3)}
          </span>
          <OriginBadge origin="retrieved" />
        </span>
      </header>
      <blockquote className="whitespace-pre-wrap break-words py-3 pr-4 text-[13px] leading-relaxed text-fg-muted">
        {highlight(chunk.text, hit.matched_terms)}
      </blockquote>
      <footer className="flex flex-wrap items-center gap-1.5 pb-2.5 pr-3 font-mono text-[10px] text-fg-subtle">
        {rerank && (
          <>
            <span className="rounded bg-surface-3 px-1.5 py-0.5" style={{ color }} title="Upstream rank and score, before reranking">
              {upstream} #{rerank.original_rank} · {rerank.original_score.toFixed(fusion?.method === "rrf" ? 5 : 3)}
            </span>
            <span className="rounded bg-surface-3 px-1.5 py-0.5" style={{ color: RERANK_COLOR }}>
              rerank #{rerank.final_rank} · {rerank.reranker_score.toFixed(3)}
            </span>
            {pool && (
              <button
                type="button"
                aria-expanded={open === "rerank"}
                onClick={() => toggle("rerank")}
                className="ml-1 mr-2 inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-fg-muted ring-1 ring-inset ring-line-strong transition-colors hover:text-fg"
              >
                explain movement <ChevronDown className={cn("size-3 transition-transform", open === "rerank" && "rotate-180")} />
              </button>
            )}
          </>
        )}
        {fusion ? (
          <>
            {byDisplayOrder(fusion.components).map((c) => (
              <span key={c.strategy} className="rounded bg-surface-3 px-1.5 py-0.5" style={{ color: STRATEGY_COLOR[c.strategy] }}>
                {c.strategy === "sparse" ? "BM25" : "Dense"} {c.rank ? `#${c.rank}` : "—"}
              </span>
            ))}
            <button
              type="button"
              aria-expanded={open === "fusion"}
              onClick={() => toggle("fusion")}
              className="ml-1 inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-fg-muted ring-1 ring-inset ring-line-strong transition-colors hover:text-fg"
            >
              explain fusion <ChevronDown className={cn("size-3 transition-transform", open === "fusion" && "rotate-180")} />
            </button>
          </>
        ) : result.strategy === "dense" ? (
          !rerank && <span>semantic match · cosine similarity, no term overlap required</span>
        ) : (
          <>
            matched:
            {hit.matched_terms.map((t) => (
              <span key={t} className="rounded bg-surface-3 px-1.5 text-fg-muted">{t}</span>
            ))}
          </>
        )}
        <span className="ml-auto" title={result.chunk_id}>{result.chunk_id}</span>
      </footer>
      <AnimatePresence initial={false}>
        {open && (open === "fusion" ? fusion : rerank && pool) && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="overflow-hidden border-t border-line/70"
          >
            <div className="py-3 pr-4">
              {open === "fusion" && fusion ? (
                <FusionExplain fusion={fusion} rank={rerank?.original_rank ?? result.rank} rrfK={rrfK} />
              ) : (
                pool && <MovementExplain pool={pool} chunkId={result.chunk_id} upstream={upstream} />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.article>
  );
}
