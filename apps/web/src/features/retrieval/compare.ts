import type { RetrievalHit, RetrievalResponse } from "@rag-forge/shared";

/** The four retrieval runs the lab can compare. Hybrid runs keep their fusion method in the key. */
export type RunKey = "sparse" | "dense" | "rrf" | "weighted";
export type Runs = Partial<Record<RunKey, RetrievalResponse>>;

export const RUN_ORDER: RunKey[] = ["sparse", "rrf", "weighted", "dense"]; // hybrids sit between their components
export const RUN_LABEL: Record<RunKey, string> = { sparse: "BM25", dense: "Dense", rrf: "Hybrid RRF", weighted: "Hybrid weighted" };
export const RUN_COLOR: Record<RunKey, string> = {
  sparse: "var(--color-s-sparse)",
  dense: "var(--color-s-dense)",
  rrf: "var(--color-s-hybrid)",
  weighted: "var(--color-s-rerank)",
};

export interface AlignedChunk {
  chunkId: string;
  hit: RetrievalHit; // any run's hit for this chunk (chunk identity is the same across runs)
  ranks: Partial<Record<RunKey, number>>;
  scores: Partial<Record<RunKey, number>>;
}

/** Every chunk retrieved by any of `keys`, with its rank and raw score in each run. */
export function alignRuns(runs: Runs, keys: RunKey[] = RUN_ORDER): AlignedChunk[] {
  const byId = new Map<string, AlignedChunk>();
  for (const key of keys) {
    for (const hit of runs[key]?.hits ?? []) {
      const id = hit.result.chunk_id;
      const row = byId.get(id) ?? { chunkId: id, hit, ranks: {}, scores: {} };
      row.ranks[key] = hit.result.rank;
      row.scores[key] = hit.result.score;
      byId.set(id, row);
    }
  }
  const best = (c: AlignedChunk) => Math.min(...keys.map((k) => c.ranks[k] ?? Infinity));
  return [...byId.values()].sort((a, b) => best(a) - best(b) || (a.chunkId < b.chunkId ? -1 : 1));
}

export interface Overlap {
  aOnly: number;
  bOnly: number;
  shared: number;
  sameRank: number; // shared chunks that kept their rank
}

export function overlap(rows: AlignedChunk[], a: RunKey, b: RunKey): Overlap {
  const inA = (r: AlignedChunk) => r.ranks[a] !== undefined;
  const inB = (r: AlignedChunk) => r.ranks[b] !== undefined;
  return {
    aOnly: rows.filter((r) => inA(r) && !inB(r)).length,
    bOnly: rows.filter((r) => inB(r) && !inA(r)).length,
    shared: rows.filter((r) => inA(r) && inB(r)).length,
    sameRank: rows.filter((r) => inA(r) && inB(r) && r.ranks[a] === r.ranks[b]).length,
  };
}

export type FusionEffect = "promoted" | "dropped" | null;

/**
 * What fusion did to a chunk relative to the single-strategy top-k lists:
 * promoted = in the hybrid top-k but in neither single top-k (it ranked lower in each alone);
 * dropped = in some single top-k but not in the hybrid top-k.
 */
export function fusionEffect(row: AlignedChunk, hybrid: "rrf" | "weighted"): FusionEffect {
  const inSingle = row.ranks.sparse !== undefined || row.ranks.dense !== undefined;
  const inHybrid = row.ranks[hybrid] !== undefined;
  if (inHybrid && !inSingle) return "promoted";
  if (!inHybrid && inSingle) return "dropped";
  return null;
}
