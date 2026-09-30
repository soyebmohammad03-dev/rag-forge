/**
 * SAMPLE DATA: interface preview only.
 *
 * Nothing here was measured. It exists so the shell can be designed and reviewed
 * before the engine produces real runs. Every component that renders it must show
 * <OriginBadge origin="simulated" />. Never import this from API code, tests of
 * evaluation logic, or anything that writes results.
 */

export const SAMPLE_ORIGIN = "simulated" as const;

// Deterministic pseudo-noise so server and client render identical markup.
const wave = (i: number, seed: number) =>
  Math.sin(i * 0.9 + seed) * 0.5 + Math.sin(i * 0.37 + seed * 2) * 0.5;

const series = (n: number, base: number, amp: number, drift: number, seed: number) =>
  Array.from({ length: n }, (_, i) => +(base + drift * i + amp * wave(i, seed)).toFixed(3));

export interface SampleMetric {
  id: string;
  label: string;
  value: number;
  unit?: string;
  delta: number;
  higherIsBetter: boolean;
  definition: string;
  series: number[];
}

export const sampleMetrics: SampleMetric[] = [
  {
    id: "ndcg",
    label: "nDCG@10",
    value: 0.612,
    delta: 0.041,
    higherIsBetter: true,
    definition: "Normalised discounted cumulative gain over the top 10 retrieved chunks.",
    series: series(16, 0.55, 0.018, 0.004, 1),
  },
  {
    id: "recall",
    label: "Recall@20",
    value: 0.843,
    delta: 0.022,
    higherIsBetter: true,
    definition: "Share of judged-relevant chunks that appear in the top 20.",
    series: series(16, 0.8, 0.015, 0.0028, 2),
  },
  {
    id: "faithfulness",
    label: "Faithfulness",
    value: 0.781,
    delta: -0.012,
    higherIsBetter: true,
    definition: "Share of generated claims supported by retrieved evidence.",
    series: series(16, 0.79, 0.02, -0.0008, 3),
  },
  {
    id: "latency",
    label: "p95 latency",
    value: 1.84,
    unit: "s",
    delta: -0.21,
    higherIsBetter: false,
    definition: "95th percentile end-to-end latency, retrieval through generation.",
    series: series(16, 2.1, 0.08, -0.016, 4),
  },
];

export const STRATEGY_KEYS = ["sparse", "dense", "hybrid", "graph"] as const;
export type StrategyKey = (typeof STRATEGY_KEYS)[number];

export const sampleActivity = Array.from({ length: 24 }, (_, h) => {
  const load = 1 + 0.8 * Math.sin(((h - 6) / 24) * Math.PI * 2);
  const row: Record<string, number | string> = { hour: `${String(h).padStart(2, "0")}:00` };
  STRATEGY_KEYS.forEach((k, s) => {
    const share = [22, 34, 30, 9][s];
    row[k] = Math.max(0, Math.round(share * load * (1 + 0.25 * wave(h, s + 1))));
  });
  return row;
});

export type SampleRunStatus = "completed" | "running" | "failed" | "queued";

export interface SampleExperiment {
  id: string;
  name: string;
  hypothesis: string;
  corpus: string;
  strategies: StrategyKey[];
  status: SampleRunStatus;
  ndcg: number | null;
  configHash: string;
  started: string;
}

export const sampleExperiments: SampleExperiment[] = [
  {
    id: "exp_preview_014",
    name: "Router v0 vs fixed hybrid",
    hypothesis: "A feature-based router beats fixed hybrid on multi-hop questions without hurting single-hop.",
    corpus: "hotpotqa-dev",
    strategies: ["sparse", "dense", "graph"],
    status: "running",
    ndcg: null,
    configHash: "9f3c1e07a2",
    started: "12 min ago",
  },
  {
    id: "exp_preview_013",
    name: "Reranker ablation",
    hypothesis: "Cross-encoder reranking recovers most of the dense-only recall gap at k=10.",
    corpus: "scifact",
    strategies: ["dense", "hybrid"],
    status: "completed",
    ndcg: 0.612,
    configHash: "41be8d2c90",
    started: "3 h ago",
  },
  {
    id: "exp_preview_012",
    name: "Chunk size sweep 128–1024",
    hypothesis: "Retrieval quality peaks at mid-size chunks; generation faithfulness does not.",
    corpus: "fiqa-2018",
    strategies: ["dense"],
    status: "completed",
    ndcg: 0.447,
    configHash: "c07a55f1e3",
    started: "yesterday",
  },
  {
    id: "exp_preview_011",
    name: "BM25 parameter grid",
    hypothesis: "Default k1/b are near-optimal for scientific claims.",
    corpus: "scifact",
    strategies: ["sparse"],
    status: "failed",
    ndcg: null,
    configHash: "e2d9b6014f",
    started: "2 d ago",
  },
];
