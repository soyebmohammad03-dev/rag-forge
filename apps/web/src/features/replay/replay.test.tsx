import type { ExperimentRun, Replay, ReproducibilityManifest } from "@rag-forge/shared";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { OverviewPage } from "@/features/overview/overview-page";
import { ReplayPage } from "./replay-page";

let search = "";
const replace = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace, push: vi.fn() }),
  usePathname: () => "/replay",
  useSearchParams: () => new URLSearchParams(search),
}));

afterEach(() => {
  vi.unstubAllGlobals();
  replace.mockReset();
  search = "";
});

const ENV = { rag_forge_version: "1.0.0", python_version: "3.14.0", platform: "test", git_commit: "abcdef1234567890", packages: {} };
const RUNTIME = {
  git_commit: "abcdef1234567890", node_version: "v24.0.0", uv_version: "0.9.0", git_dirty: false,
  lockfiles: { "apps/api/uv.lock": "f".repeat(64) }, packages: { onnxruntime: "1.23" },
  settings: { RAG_FORGE_OPENAI_API_KEY: "<redacted>" }, models: [], captured_at: "2026-10-07T00:00:00Z",
};

const run = {
  id: "run_1", experiment_id: "exp_1", status: "completed", dataset_id: "bds_1", dataset_version: 1, dataset_hash: "d".repeat(64),
  corpus_id: "cor_1", corpus_version: 1, case_ids: ["a", "b"], arms: ["bm25", "rag-local"], snapshot_hashes: { bm25: "1".repeat(64), "rag-local": "2".repeat(64) },
  limits: { max_cases: null, concurrency: 1 }, total: 4, completed: 4, failed: 0, metric_registry_version: "arena-metrics@1", stats_method: "paired-bootstrap-sign@1",
  environment: ENV, runtime: RUNTIME, summaries: [], error: null, created_at: "2026-10-07T00:00:00Z", started_at: null, finished_at: null,
} as unknown as ExperimentRun;

const manifest = {
  manifest_version: "rag-forge-manifest@1", generated_at: "2026-10-07T00:00:00Z",
  run: { id: "run_1", experiment: "replay check", hypothesis: "", status: "completed", completed: 4, evaluations: 4, failed: 0, created_at: "t0", finished_at: "t1" },
  dataset: { name: "rag-forge-dev", version: 1, source: "development", content_hash: "d".repeat(64), corpus_id: "cor_1", corpus_version: 1, chunking_hash: "c", cases: 14 },
  arms: [
    { arm: "bm25", pipeline: "retrieval", config_hash: "1".repeat(64), retrieval_mode: "manual", retrieval_configuration_hash: "r".repeat(64), routing_hash: null, generator: null, generation_deterministic: null, temperature: null, seed: null, prompt_template: null, verifier: null },
    { arm: "rag-local", pipeline: "rag", config_hash: "2".repeat(64), retrieval_mode: "manual", retrieval_configuration_hash: "s".repeat(64), routing_hash: null, generator: "qwen", generation_deterministic: true, temperature: 0, seed: 0, prompt_template: "grounded-qa@1", verifier: "lexical-semantic@1" },
  ],
  models: [{ role: "generator", provider: "onnx-causal-lm", model: "onnx-community/Qwen2.5-0.5B-Instruct", revision: "cc5cc01a65cc", config_hash: "x", files: [{ path: "onnx/model_q4.onnx", sha256: "9".repeat(64), bytes: 786000000, source: "lfs-blob-id" }] }],
  seeds: { bootstrap: 20261007 }, metrics: {}, statistics: {}, environment: ENV, runtime: RUNTIME,
  current: { differences: ["git commit: recorded abc, now def"] }, artifacts: [], replays: [], notes: ["Secrets are never recorded."],
} as unknown as ReproducibilityManifest;

const replay = (outcomes: Record<string, number>): Replay => ({
  id: "rpl_1", run_id: "run_1", experiment_id: "exp_1", status: "completed", arms: run.arms, case_ids: run.case_ids, total: 4, completed: 4,
  checks: [], outcomes, environment: ENV, runtime: RUNTIME, error: null, created_at: "2026-10-07T00:00:00Z", finished_at: "2026-10-07T00:00:01Z",
  cases: [
    { arm: "bm25", case_id: "a", outcome: "exact", reason: "every recorded stage hash matched", recorded_status: "ok", replayed_status: "ok", first_divergence: null, recorded_artifact_id: "art_1", replayed_artifact_id: "art_2", metrics_compared: 20, metric_differences: [], latency_ms: 3,
      stages: [{ stage: "retrieval", deterministic: true, recorded: "aaaa", replayed: "aaaa", match: true }] },
    { arm: "rag-local", case_id: "a", outcome: "equivalent", reason: "generation is not deterministic", recorded_status: "ok", replayed_status: "ok", first_divergence: "generation", recorded_artifact_id: "art_3", replayed_artifact_id: "art_4", metrics_compared: 25, metric_differences: [{ metric: "answer_token_f1", recorded: 0.5, replayed: 0.4 }], latency_ms: 900,
      stages: [{ stage: "context", deterministic: true, recorded: "c1", replayed: "c1", match: true }, { stage: "generation", deterministic: false, recorded: "g1", replayed: "g2", match: false }] },
  ],
}) as unknown as Replay;

describe("Replay page", () => {
  it("explains that there is nothing to inspect without a finished run", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json([{ run_id: "run_q", status: "running" }])));
    render(<ReplayPage />);
    expect(await screen.findByText("No finished runs to inspect")).toBeInTheDocument();
  });

  it("shows provenance, exports, determinism and replay availability of a recorded run", async () => {
    search = "run=run_1&view=reproducibility";
    vi.stubGlobal("fetch", vi.fn(async (req: Request) => {
      const p = new URL(req.url).pathname;
      if (p.endsWith("/manifest")) return Response.json(manifest);
      if (p.endsWith("/replayability")) return Response.json([
        { arm: "bm25", recorded_hash: "1", current_hash: "1", replayable: true, reason: null, changes: [], generation: null },
        { arm: "rag-local", recorded_hash: "2", current_hash: null, replayable: false, reason: "RagComponentNotAvailableError: generator 'qwen' is not registered", changes: [], generation: null },
      ]);
      return Response.json(run);
    }));
    render(<ReplayPage />);
    expect(await screen.findByText("not replayable")).toBeInTheDocument();
    expect(screen.getByText(/generator 'qwen' is not registered/)).toBeInTheDocument();
    expect(screen.getByText("onnx/model_q4.onnx")).toBeInTheDocument();
    expect(screen.getByText("git commit: recorded abc, now def")).toBeInTheDocument();
    expect(screen.getByText(/RAG_FORGE_OPENAI_API_KEY=<redacted>/)).toBeInTheDocument();
    expect(screen.getByText("rag-local: deterministic")).toBeInTheDocument();
    const exports = screen.getByLabelText("Exports");
    expect(within(exports).getByText("Research report").closest("a")).toHaveAttribute("href", "http://localhost:8000/api/v1/runs/run_1/report.md");
    expect(within(exports).getAllByRole("link")).toHaveLength(6);
  });

  it("starts a replay and reports each case's outcome and differing stage", async () => {
    search = "run=run_1&view=replay";
    let posted = false;
    vi.stubGlobal("fetch", vi.fn(async (req: Request) => {
      const p = new URL(req.url).pathname;
      if (req.method === "POST") {
        posted = true;
        return Response.json(replay({}), { status: 202 });
      }
      if (p === "/api/v1/runs/run_1/replays") return Response.json(posted ? [replay({ exact: 1, equivalent: 1 })] : []);
      if (p === "/api/v1/replays/rpl_1") return Response.json(replay({ exact: 1, equivalent: 1 }));
      return Response.json(run);
    }));
    render(<ReplayPage />);
    const user = userEvent.setup();
    expect(await screen.findByText("This run has not been replayed yet.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Replay all 4 evaluations/ }));
    expect(posted).toBe(true);
    expect(await screen.findByText("4 evaluations replayed")).toBeInTheDocument();
    const row = screen.getByText("generation is not deterministic").closest("tr")!;
    expect(within(row).getByText("equivalent")).toBeInTheDocument();
    expect(within(row).getByText("1/25")).toBeInTheDocument();
    await user.click(row);
    expect(await screen.findByText("Recorded vs replayed output hash, per stage")).toBeInTheDocument();
    expect(screen.getByText("differs")).toBeInTheDocument();
    expect(screen.getByText("answer_token_f1: 0.5 → 0.4")).toBeInTheDocument();
  });
});

describe("Overview", () => {
  it("links every implemented stage to its area and shows no planned work", async () => {
    vi.stubGlobal("fetch", vi.fn(async (req: Request) => {
      const p = new URL(req.url).pathname;
      if (p.endsWith("/health")) return Response.json({ status: "ok", version: "1.0.0", started_at: "", uptime_seconds: 1, components: [{ name: "embedder", state: "ok", detail: "BAAI/bge-small-en-v1.5@5c38ec7c" }] });
      if (p.endsWith("/arena/overview")) return Response.json({ datasets: [], experiments: 0, runs: [], metric_registry: "arena-metrics@1", stats_method: { name: "paired-bootstrap-sign", version: 1 } });
      if (p.endsWith("/provenance/environment")) return Response.json({ environment: ENV });
      return Response.json([]);
    }));
    render(<OverviewPage />);
    const pipeline = await screen.findByRole("list", { name: "Implemented pipeline" });
    const links = within(pipeline).getAllByRole("link");
    const labels = ["Ingest", "Retrieve", "Fuse", "Rerank", "Analyze", "Route", "Evidence", "Generate", "Ground", "Benchmark", "Experiment", "Replay"];
    expect(links).toHaveLength(labels.length);
    labels.forEach((label, i) => expect(links[i].textContent).toContain(label));
    expect(links.map((l) => l.getAttribute("href"))).toEqual([
      "/corpus", "/retrieval", "/retrieval", "/retrieval", "/router", "/router", "/evidence", "/evidence", "/evidence", "/arena?tab=datasets", "/experiments", "/replay",
    ]);
    expect(links.at(-1)).toHaveAttribute("href", "/replay");
    expect(await screen.findByText("BAAI/bge-small-en-v1.5")).toBeInTheDocument();
    expect(screen.queryByText(/planned|sample/i)).not.toBeInTheDocument();
    expect(screen.getByText("No replays yet")).toBeInTheDocument();
  });
});
