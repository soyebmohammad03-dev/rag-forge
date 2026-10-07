import type { MetricFamily, MetricSummary, PairedComparison, RunStatus } from "@rag-forge/shared";
import type { Status } from "@/components/ui/status";

/** One stable colour per arm position, from the strategy palette, so an arm reads the same everywhere. */
const PALETTE = [
  "var(--color-s-sparse)",
  "var(--color-s-dense)",
  "var(--color-s-hybrid)",
  "var(--color-s-metadata)",
  "var(--color-s-graph)",
  "var(--color-s-rerank)",
  "var(--color-trace)",
  "var(--color-fg-muted)",
];
export const armColor = (index: number) => PALETTE[index % PALETTE.length];

export const FAMILY_LABEL: Record<MetricFamily, string> = {
  retrieval: "Retrieval",
  reranking: "Reranking",
  evidence: "Evidence & grounding",
  generation: "Generation (automatic)",
  operational: "Operational",
};
export const FAMILIES: MetricFamily[] = ["retrieval", "reranking", "evidence", "generation", "operational"];

export const RUN_STATUS: Record<RunStatus, Status> = {
  queued: "idle",
  running: "loading",
  completed: "ok",
  partial: "warn",
  failed: "err",
};

export const isMs = (metric: string) => metric.endsWith("_ms");
export const isCount = (metric: string) => metric.endsWith("_tokens");

/** Format a metric value: milliseconds, token counts or a 3-decimal score. */
export function fmt(value: number | null | undefined, metric: string): string {
  if (value === null || value === undefined) return "—";
  if (isMs(metric)) return `${value < 100 ? value.toFixed(1) : Math.round(value)} ms`;
  if (isCount(metric)) return value.toFixed(0);
  return value.toFixed(3);
}

export function signed(value: number | null | undefined, metric: string): string {
  if (value === null || value === undefined) return "—";
  return `${value > 0 ? "+" : value < 0 ? "−" : "±"}${fmt(Math.abs(value), metric)}`;
}

export interface Verdict {
  label: string;
  tone: "ok" | "err" | "neutral" | "signal";
  help: string;
}

/**
 * What a paired comparison supports, in words. "Higher" is about the value; whether that is
 * better depends on the metric's direction, which this resolves.
 */
export function verdict(p: Pick<PairedComparison, "conclusion" | "higher_is_better" | "n_pairs">, minCases: number): Verdict {
  switch (p.conclusion) {
    case "variant_higher":
    case "variant_lower": {
      const higher = p.conclusion === "variant_higher";
      const better = p.higher_is_better === higher;
      return {
        label: `variant ${higher ? "higher" : "lower"} (${better ? "better" : "worse"})`,
        tone: better ? "ok" : "err",
        help: "The bootstrap confidence interval of the paired mean difference excludes 0 on this dataset.",
      };
    }
    case "no_detectable_difference":
      return { label: "no detectable difference", tone: "neutral", help: "The confidence interval includes 0." };
    case "descriptive_only":
      return { label: "descriptive", tone: "neutral", help: "This metric has no better direction; no conclusion is drawn." };
    default:
      return {
        label: p.n_pairs ? `too few cases (${p.n_pairs} < ${minCases})` : "no paired cases",
        tone: "signal",
        help: `Conclusions need at least ${minCases} paired cases.`,
      };
  }
}

export function percentile(values: number[], p: number): number | null {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  const i = (s.length - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}

/** Metric names with at least one defined value, grouped by family, in registry order. */
export function metricsByFamily(summaries: MetricSummary[][]): Record<MetricFamily, string[]> {
  const out = Object.fromEntries(FAMILIES.map((f) => [f, [] as string[]])) as Record<MetricFamily, string[]>;
  const seen = new Set<string>();
  for (const list of summaries)
    for (const m of list)
      if (m.n > 0 && !seen.has(m.metric)) {
        seen.add(m.metric);
        out[m.family].push(m.metric);
      }
  return out;
}

/** A shared [min, max] axis for intervals, padded, and always containing `include` if given. */
export function domain(values: (number | null | undefined)[], include?: number): [number, number] {
  const v = values.filter((x): x is number => x !== null && x !== undefined && Number.isFinite(x));
  if (include !== undefined) v.push(include);
  if (!v.length) return [0, 1];
  let lo = Math.min(...v);
  let hi = Math.max(...v);
  if (lo === hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const pad = (hi - lo) * 0.06;
  return [lo - pad, hi + pad];
}
