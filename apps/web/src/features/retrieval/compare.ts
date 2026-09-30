import type { RetrievalHit, RetrievalResponse } from "@rag-forge/shared";

export type Membership = "both" | "sparse" | "dense";

export interface ComparedChunk {
  chunkId: string;
  hit: RetrievalHit; // from whichever run ranked it (identical chunk either way)
  sparseRank: number | null;
  denseRank: number | null;
  kind: Membership;
}

export interface Comparison {
  chunks: ComparedChunk[]; // union, ordered by best rank in either run, then chunk id
  shared: number;
  sparseOnly: number;
  denseOnly: number;
}

/** Set and rank relationship between two retrieval runs of the same query and corpus version. */
export function compareRuns(sparse: RetrievalResponse, dense: RetrievalResponse): Comparison {
  const byId = new Map<string, ComparedChunk>();
  for (const [hits, key] of [
    [sparse.hits, "sparseRank"],
    [dense.hits, "denseRank"],
  ] as const) {
    for (const hit of hits) {
      const id = hit.result.chunk_id;
      const row = byId.get(id) ?? { chunkId: id, hit, sparseRank: null, denseRank: null, kind: "both" as Membership };
      row[key] = hit.result.rank;
      byId.set(id, row);
    }
  }
  const chunks = [...byId.values()].map((c) => ({
    ...c,
    kind: (c.sparseRank && c.denseRank ? "both" : c.sparseRank ? "sparse" : "dense") as Membership,
  }));
  const best = (c: ComparedChunk) => Math.min(c.sparseRank ?? Infinity, c.denseRank ?? Infinity);
  chunks.sort((a, b) => best(a) - best(b) || (a.chunkId < b.chunkId ? -1 : 1));
  const count = (k: Membership) => chunks.filter((c) => c.kind === k).length;
  return { chunks, shared: count("both"), sparseOnly: count("sparse"), denseOnly: count("dense") };
}
