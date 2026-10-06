import type { RerankCandidate, RerankDetail } from "@rag-forge/shared";

export const MOVEMENT_COLOR: Record<RerankDetail["movement"], string> = {
  promoted: "var(--color-ok)",
  demoted: "var(--color-err)",
  unchanged: "var(--color-fg-subtle)",
};

/** file:chunk-ordinal; "#" is reserved for ranks in the reranking views. */
export const candidateLabel = (c: RerankCandidate) => `${c.filename}:${c.chunk_ordinal}`;

export const formatMs = (ms: number) => `${ms < 100 ? ms.toFixed(1) : Math.round(ms)} ms`;

export interface Crossings {
  overtook: RerankCandidate[]; // ranked above this candidate upstream, below it after reranking
  overtakenBy: RerankCandidate[]; // ranked below upstream, above after reranking
}

/**
 * Which candidates a result swapped places with. Purely positional, so it always holds that
 * rank_delta = overtook.length - overtakenBy.length; each swap is explained by the two scores.
 */
export function crossings(pool: RerankCandidate[], chunkId: string): Crossings {
  const me = pool.find((c) => c.chunk_id === chunkId);
  if (!me) return { overtook: [], overtakenBy: [] };
  const { original_rank: o, final_rank: f } = me.rerank;
  return {
    overtook: pool.filter((c) => c.rerank.original_rank < o && c.rerank.final_rank > f),
    overtakenBy: pool.filter((c) => c.rerank.original_rank > o && c.rerank.final_rank < f),
  };
}

export interface RerankSummary {
  pool: number;
  kept: number; // in both the upstream and the final top-k
  entered: number;
  left: number;
  promoted: number;
  demoted: number;
  unchanged: number;
  displacement: number; // mean |rank_delta| over the final top-k
}

export function summarize(pool: RerankCandidate[], k: number): RerankSummary {
  const top = pool.filter((c) => c.rerank.final_rank <= k);
  const count = (m: RerankDetail["movement"]) => top.filter((c) => c.rerank.movement === m).length;
  return {
    pool: pool.length,
    kept: top.filter((c) => !c.rerank.entered_top_k).length,
    entered: top.filter((c) => c.rerank.entered_top_k).length,
    left: pool.filter((c) => c.rerank.left_top_k).length,
    promoted: count("promoted"),
    demoted: count("demoted"),
    unchanged: count("unchanged"),
    displacement: top.length ? top.reduce((s, c) => s + Math.abs(c.rerank.rank_delta), 0) / top.length : 0,
  };
}

/** Candidates worth drawing: in the upstream top-k, the final top-k, or both. */
export const visible = (pool: RerankCandidate[], k: number) =>
  pool.filter((c) => c.rerank.original_rank <= k || c.rerank.final_rank <= k);
