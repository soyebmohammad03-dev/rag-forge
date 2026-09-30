"use client";

import type { ComponentScore, FusionDetail } from "@rag-forge/shared";
import { motion } from "motion/react";
import { STRATEGY_COLOR } from "@/components/ui/chart";

const NAME: Record<string, string> = { sparse: "BM25", dense: "Dense" };
const ORDER = ["sparse", "dense"];

/** Components in display order: BM25 first, then dense (the API orders them by name). */
export const byDisplayOrder = (components: ComponentScore[]) =>
  [...components].sort((a, b) => ORDER.indexOf(a.strategy) - ORDER.indexOf(b.strategy));

/** How one hybrid result's score was assembled from its components. */
export function FusionExplain({ fusion, rank, rrfK }: { fusion: FusionDetail; rank: number; rrfK?: number }) {
  const total = fusion.score || 1;
  const rrf = fusion.method === "rrf";
  const components = byDisplayOrder(fusion.components);
  return (
    <div className="space-y-3" aria-label={`Fusion explanation for result ${rank}`}>
      <div className="flex items-baseline justify-between font-mono text-[11px]">
        <span className="text-fg-muted">
          Result #{rank} · {rrf ? `RRF (k = ${rrfK ?? "?"})` : "weighted, min-max normalised"}
        </span>
        <span className="num text-fg">
          final <span className="text-s-hybrid">{fusion.score.toFixed(rrf ? 5 : 4)}</span>
        </span>
      </div>

      {/* stacked contribution bar: each component's share of the final score */}
      <div className="flex h-2 overflow-hidden rounded-full bg-surface-3" role="img" aria-label="Contribution by component">
        {components.map((c) => (
          <motion.span
            key={c.strategy}
            className="h-full"
            style={{ background: STRATEGY_COLOR[c.strategy] }}
            initial={{ width: 0 }}
            animate={{ width: `${(c.contribution / total) * 100}%` }}
            transition={{ duration: 0.45, ease: "easeOut" }}
          />
        ))}
      </div>

      <div className="grid gap-2 sm:grid-cols-2">
        {components.map((c) => (
          <ComponentCard key={c.strategy} c={c} rrf={rrf} rrfK={rrfK} share={c.contribution / total} />
        ))}
      </div>
      <p className="font-mono text-[10px] leading-relaxed text-fg-subtle">
        {rrf
          ? "RRF adds 1 / (k + rank) for every list that contains the chunk. Raw scores are shown for reference only; RRF never uses them."
          : "Each list's scores are rescaled to 0..1 (best = 1) before weighting, because BM25 and cosine scores are on different scales. A chunk missing from a list contributes 0 there."}
      </p>
    </div>
  );
}

function ComponentCard({ c, rrf, rrfK, share }: { c: ComponentScore; rrf: boolean; rrfK?: number; share: number }) {
  const absent = c.rank === null || c.rank === undefined;
  return (
    <div className="rounded-md border border-line bg-surface-2/60 px-3 py-2 font-mono text-[11px]">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5" style={{ color: STRATEGY_COLOR[c.strategy] }}>
          <span className="size-2 rounded-sm" style={{ background: STRATEGY_COLOR[c.strategy] }} />
          {NAME[c.strategy] ?? c.strategy}
        </span>
        <span className="num text-fg-muted">{absent ? "not retrieved" : `${Math.round(share * 100)}% of score`}</span>
      </div>
      <dl className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-fg-subtle">
        <dt>rank</dt>
        <dd className="num text-right text-fg">{absent ? "—" : c.rank}</dd>
        <dt>raw score</dt>
        <dd className="num text-right text-fg">{c.score === null || c.score === undefined ? "—" : c.score.toFixed(4)}</dd>
        {!rrf && (
          <>
            <dt>normalised</dt>
            <dd className="num text-right text-fg">{c.normalized_score === null || c.normalized_score === undefined ? "—" : c.normalized_score.toFixed(4)}</dd>
          </>
        )}
        <dt>contribution</dt>
        <dd className="num text-right text-fg">
          {rrf && !absent && rrfK !== undefined && <span className="text-fg-subtle">1/({rrfK}+{c.rank}) = </span>}
          {c.contribution.toFixed(5)}
        </dd>
      </dl>
    </div>
  );
}
