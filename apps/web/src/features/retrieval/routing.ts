import type { RetrievalResponse, RouteOption } from "@rag-forge/shared";
import type { RunKey } from "./compare";

/** Router options map onto the lab's four fixed runs. */
export const OPTION_RUN: Record<RouteOption, RunKey> = {
  sparse: "sparse",
  dense: "dense",
  hybrid_rrf: "rrf",
  hybrid_weighted: "weighted",
};
export const OPTIONS: RouteOption[] = ["sparse", "dense", "hybrid_rrf", "hybrid_weighted"];

export interface TopKOverlap {
  shared: number;
  aOnly: number;
  bOnly: number;
  sameRank: number;
  jaccard: number; // |A ∩ B| / |A ∪ B| over the two top-k chunk sets
}

export function topKOverlap(a: RetrievalResponse, b: RetrievalResponse): TopKOverlap {
  const ra = new Map(a.hits.map((h) => [h.result.chunk_id, h.result.rank]));
  const rb = new Map(b.hits.map((h) => [h.result.chunk_id, h.result.rank]));
  const shared = [...ra.keys()].filter((id) => rb.has(id));
  const union = new Set([...ra.keys(), ...rb.keys()]).size;
  return {
    shared: shared.length,
    aOnly: ra.size - shared.length,
    bOnly: rb.size - shared.length,
    sameRank: shared.filter((id) => ra.get(id) === rb.get(id)).length,
    jaccard: union ? shared.length / union : 1,
  };
}

export interface LatencySegment {
  label: string;
  ms: number;
}

/**
 * Server-side time of one request, split into the stages it measured. "retrieval" is the
 * remainder of the total, so the segments always add up to elapsed_ms.
 */
export function latencyBreakdown(r: RetrievalResponse): LatencySegment[] {
  const p = r.provenance;
  const analysis = p.routing?.analysis_ms ?? 0;
  const decision = p.routing?.decision_ms ?? 0;
  const rerank = p.reranking?.latency_ms ?? 0;
  const rest = Math.max(0, p.elapsed_ms - analysis - decision - rerank);
  const segments: LatencySegment[] = [];
  if (p.routing) segments.push({ label: "analysis", ms: analysis }, { label: "decision", ms: decision });
  segments.push({ label: "retrieval", ms: rest });
  if (p.reranking) segments.push({ label: "rerank", ms: rerank });
  return segments;
}

/** Render a rule input for display; numbers keep up to 4 decimals. */
export function formatInput(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Math.round(v * 1e4) / 1e4);
  return String(v);
}
