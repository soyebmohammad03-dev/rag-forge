import type { Arm, ArenaOverview, Comparison, Experiment, ExperimentRun, Leaderboard, MetricSummary } from "@rag-forge/shared";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ArenaPage } from "./arena-page";
import { needsDense, parseKs } from "./builder-tab";
import { domain, fmt, metricsByFamily, percentile, signed, verdict } from "./format";

let search = "";
const replace = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace, push: vi.fn() }),
  usePathname: () => "/arena",
  useSearchParams: () => new URLSearchParams(search),
}));

afterEach(() => {
  vi.unstubAllGlobals();
  replace.mockReset();
  search = "";
});

const METHOD = {
  name: "paired-bootstrap-sign", version: 1, confidence: 0.95, bootstrap_resamples: 5000, seed: 20261007,
  min_cases_for_conclusion: 10, multiple_comparisons: "Holm across the metrics compared",
};

const overview = (patch: Partial<ArenaOverview> = {}): ArenaOverview => ({
  datasets: [], experiments: 0, runs: [], metric_registry: "arena-metrics@1", stats_method: METHOD, ...patch,
});

const summary = (metric: string, mean: number | null, patch: Partial<MetricSummary> = {}): MetricSummary => ({
  metric, family: metric.endsWith("_ms") ? "operational" : "retrieval", version: 1, higher_is_better: !metric.endsWith("_ms"),
  n: mean === null ? 0 : 3, skipped: mean === null ? 3 : 0, mean, median: mean, std: null, min: mean, max: mean,
  ci_low: mean === null ? null : mean - 0.1, ci_high: mean === null ? null : mean + 0.1, ci_method: "percentile bootstrap", ...patch,
});

describe("format helpers", () => {
  it("formats by unit and signs differences", () => {
    expect(fmt(null, "mrr")).toBe("—");
    expect(fmt(0.5, "mrr")).toBe("0.500");
    expect(fmt(12.34, "latency_ms")).toBe("12.3 ms");
    expect(fmt(1234.5, "latency_ms")).toBe("1235 ms");
    expect(fmt(41, "prompt_tokens")).toBe("41");
    expect(signed(-0.25, "mrr")).toBe("−0.250");
    expect(signed(0, "mrr")).toBe("±0.000");
  });

  it("reads conclusions by the metric's direction and refuses small samples", () => {
    expect(verdict({ conclusion: "variant_higher", higher_is_better: false, n_pairs: 20 }, 10)).toMatchObject({ label: "variant higher (worse)", tone: "err" });
    expect(verdict({ conclusion: "variant_lower", higher_is_better: false, n_pairs: 20 }, 10)).toMatchObject({ label: "variant lower (better)", tone: "ok" });
    expect(verdict({ conclusion: "insufficient_cases", higher_is_better: true, n_pairs: 4 }, 10).label).toBe("too few cases (4 < 10)");
    expect(verdict({ conclusion: "no_detectable_difference", higher_is_better: true, n_pairs: 20 }, 10).label).toBe("no detectable difference");
  });

  it("computes percentiles, axes and the metrics present", () => {
    expect(percentile([], 0.5)).toBeNull();
    expect(percentile([1, 2, 3, 4], 0.5)).toBe(2.5);
    expect(percentile([10, 20], 0.95)).toBeCloseTo(19.5);
    expect(domain([])).toEqual([0, 1]);
    const [lo, hi] = domain([0.2, 0.8], 0);
    expect(lo).toBeLessThan(0);
    expect(hi).toBeGreaterThan(0.8);
    const groups = metricsByFamily([[summary("mrr", 0.5), summary("ndcg@10", null)], [summary("latency_ms", 12)]]);
    expect(groups.retrieval).toEqual(["mrr"]); // never-defined metrics are not offered
    expect(groups.operational).toEqual(["latency_ms"]);
  });

  it("parses cutoffs and knows which arms need a dense index", () => {
    expect(parseKs("10, 1,3, 3")).toEqual([1, 3, 10]);
    expect(parseKs("0")).toBeNull();
    expect(parseKs("a")).toBeNull();
    expect(parseKs("")).toBeNull();
    const arm = (strategy: string, mode = "manual") => ({ retrieval: { strategy, mode } }) as unknown as Arm;
    expect(needsDense(arm("sparse"))).toBe(false);
    expect(needsDense(arm("hybrid"))).toBe(true);
    expect(needsDense(arm("sparse", "adaptive"))).toBe(true);
  });
});

describe("Arena page", () => {
  it("explains that nothing has been measured yet", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json(overview())));
    render(<ArenaPage />);
    expect(await screen.findByText("No measured results yet")).toBeInTheDocument();
    expect(screen.getByText(/No conclusion below 10 cases/)).toBeInTheDocument();
    expect(screen.getByText("paired-bootstrap-sign@1")).toBeInTheDocument();
  });

  it("sends the builder to the datasets when none is registered", async () => {
    vi.stubGlobal("fetch", vi.fn(async (req: Request) => Response.json(req.url.includes("/presets") ? { arms: [], ablations: [], metrics: { ks: [1], metrics: null } } : overview())));
    render(<ArenaPage initialTab="builder" />);
    expect(await screen.findByText("No benchmark dataset to run against")).toBeInTheDocument();
    await userEvent.setup().click(screen.getByRole("button", { name: "Open datasets" }));
    expect(replace).toHaveBeenCalledWith("/arena?tab=datasets");
  });

  it("creates an experiment from the selected arms and starts its run", async () => {
    const ds = { id: "bds_1", name: "rag-forge-dev", version: 1, source: "development", description: "", corpus_id: "cor_dev000000001", corpus_version: 1, case_count: 14, annotation_counts: {}, content_hash: "h", created_at: "2026-10-07T00:00:00Z" };
    const t = (strategy: string) => ({ top_k: 10, strategy, bm25: {}, hybrid: {}, rerank: { enabled: false, candidate_k: 50 }, mode: "manual", router: {} });
    const arms = [
      { name: "bm25", label: "BM25", pipeline: "retrieval", retrieval: t("sparse"), evidence: null, generation: null, grounding: {} },
      { name: "dense", label: "Dense", pipeline: "retrieval", retrieval: t("dense"), evidence: null, generation: null, grounding: {} },
    ];
    const posts: Record<string, unknown> = {};
    vi.stubGlobal(
      "fetch",
      vi.fn(async (req: Request) => {
        if (req.method === "POST") {
          posts[new URL(req.url).pathname] = await req.clone().json().catch(() => null);
          if (req.url.endsWith("/experiments")) return Response.json({ id: "exp_1" }, { status: 201 });
          return Response.json({ id: "run_1" }, { status: 202 });
        }
        if (req.url.includes("/presets")) return Response.json({ arms, ablations: [{ baseline: "bm25", variant: "dense", factor: "retrieval strategy", note: "" }], metrics: { ks: [1, 3, 5, 10], metrics: null } });
        return Response.json(overview({ datasets: [ds] as ArenaOverview["datasets"] }));
      }),
    );
    render(<ArenaPage initialTab="builder" />);
    const user = userEvent.setup();
    expect(await screen.findByText(/Development set: results validate the pipeline/)).toBeInTheDocument();
    const run = screen.getByRole("button", { name: /Create & run/ });
    expect(run).toBeDisabled(); // a name is required
    await user.type(screen.getByLabelText("Name"), "strategy check");
    await user.click(screen.getByRole("checkbox", { name: /bm25 → dense/ }));
    await user.clear(screen.getByLabelText("Max cases"));
    await user.type(screen.getByLabelText("Max cases"), "5");
    expect(screen.getByText(/2 arms × 5 cases = 10 evaluations · 1 ablation/)).toBeInTheDocument();
    await user.click(run);
    expect(replace).toHaveBeenCalledWith("/arena?tab=runs&run=run_1");
    const body = posts["/api/v1/experiments"] as { arms: { name: string }[]; ablations: unknown[]; limits: unknown; dataset_id: string };
    expect(body.dataset_id).toBe("bds_1");
    expect(body.arms.map((a) => a.name)).toEqual(["bm25", "dense"]);
    expect(body.ablations).toHaveLength(1);
    expect(body.limits).toEqual({ max_cases: 5, concurrency: 1 });
    expect(posts).toHaveProperty("/api/v1/experiments/exp_1/runs");
  });

  it("renders a finished run's leaderboard and paired comparison from the API", async () => {
    search = "run=run_1";
    const run = {
      id: "run_1", experiment_id: "exp_1", status: "partial", dataset_id: "bds_1", dataset_version: 1, dataset_hash: "dshash", corpus_id: "cor_1", corpus_version: 1,
      case_ids: ["a", "b", "c"], arms: ["bm25", "dense"], snapshot_hashes: { bm25: "h1", dense: "h2" }, limits: { max_cases: null, concurrency: 1 },
      total: 6, completed: 6, failed: 1, metric_registry_version: "arena-metrics@1", stats_method: "paired-bootstrap-sign@1",
      environment: { rag_forge_version: "0", python_version: "3.12", platform: "test", git_commit: null, packages: {} },
      summaries: [
        { arm: "bm25", label: "BM25", pipeline: "retrieval", config_hash: "h1", cases: 3, succeeded: 3, failed: 0, failure_types: {}, metrics: [summary("ndcg@10", 0.9)], skipped_reasons: {} },
        { arm: "dense", label: "Dense", pipeline: "retrieval", config_hash: "h2", cases: 3, succeeded: 2, failed: 1, failure_types: { DenseIndexNotReadyError: 1 }, metrics: [summary("ndcg@10", 0.8)], skipped_reasons: {} },
      ],
      error: null, created_at: "2026-10-07T00:00:00Z", started_at: null, finished_at: null,
    } as unknown as ExperimentRun;
    const exp = { id: "exp_1", name: "strategy check", hypothesis: "dense helps paraphrases", dataset_name: "rag-forge-dev", dataset_source: "development", ablations: [{ baseline: "bm25", variant: "dense", factor: "retrieval strategy", note: "", factors: ["retrieval.strategy"], single_factor: true, changes: [] }], snapshots: [], metrics: { ks: [10], metrics: null } } as unknown as Experiment;
    const lb: Leaderboard = {
      run_id: "run_1", metric: "ndcg@10", higher_is_better: true, note: "Positions on 3 cases.",
      rows: [
        { position: 1, arm: "bm25", label: "BM25", pipeline: "retrieval", config_hash: "h1", summary: summary("ndcg@10", 0.9), ci_overlaps_leader: true, failed: 0, cases: 3, mean_latency_ms: 2 },
        { position: 2, arm: "dense", label: "Dense", pipeline: "retrieval", config_hash: "h2", summary: summary("ndcg@10", 0.8), ci_overlaps_leader: true, failed: 1, cases: 3, mean_latency_ms: 9 },
      ],
    };
    const cmp = {
      baseline: { run_id: "run_1", arm: "bm25", config_hash: "h1", label: "BM25" }, variant: { run_id: "run_1", arm: "dense", config_hash: "h2", label: "Dense" },
      comparable: true, issues: [], warnings: ["development benchmark"], changes: [{ path: "retrieval.strategy", baseline: "sparse", variant: "dense" }], factor: "retrieval strategy", method: METHOD,
      metrics: [{ metric: "ndcg@10", family: "retrieval", higher_is_better: true, n_pairs: 2, baseline_mean: 0.9, variant_mean: 0.8, mean_difference: -0.1, median_difference: -0.1, std_difference: 0, ci_low: null, ci_high: null, effect_size_dz: null, wins: 0, losses: 1, ties: 1, sign_test_p: 1, holm_p: 1, conclusion: "insufficient_cases", differences: [{ case_id: "a", baseline: 1, variant: 0.8, difference: -0.2 }, { case_id: "b", baseline: 0.8, variant: 0.8, difference: 0 }] }],
    } as unknown as Comparison;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (req: Request) => {
        const path = new URL(req.url).pathname;
        if (path.endsWith("/leaderboard")) return Response.json(lb);
        if (path.endsWith("/compare")) return Response.json(cmp);
        if (path === "/api/v1/runs/run_1") return Response.json(run);
        if (path === "/api/v1/experiments/exp_1") return Response.json(exp);
        return Response.json(overview());
      }),
    );
    render(<ArenaPage />);
    const user = userEvent.setup();
    expect(await screen.findByText("Positions on 3 cases.")).toBeInTheDocument();
    expect(screen.getByText(/Development benchmark: small and easy/)).toBeInTheDocument();
    expect(screen.getByText("1/3")).toBeInTheDocument(); // dense failures shown on the leaderboard
    await user.click(screen.getByRole("tab", { name: "Statistical comparison" }));
    expect(await screen.findByText("too few cases (2 < 10)")).toBeInTheDocument();
    const changes = screen.getByText("retrieval.strategy").closest("tr")!;
    expect(within(changes).getByText('"dense"')).toBeInTheDocument();
    expect(screen.getByText("development benchmark")).toBeInTheDocument();
  });
});
