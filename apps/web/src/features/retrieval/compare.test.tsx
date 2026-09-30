import type { DenseIndexView, FusionDetail, RetrievalHit, RetrievalResponse } from "@rag-forge/shared";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { type RunKey, alignRuns, fusionEffect, overlap } from "./compare";
import { CompareView } from "./compare-view";
import { DenseIndexPanel } from "./dense-index";
import { EvidenceCard } from "./evidence-card";

const STRATEGY: Record<RunKey, "sparse" | "dense" | "hybrid"> = { sparse: "sparse", dense: "dense", rrf: "hybrid", weighted: "hybrid" };

const hit = (id: string, rank: number, score: number, key: RunKey, fusion: FusionDetail | null = null): RetrievalHit => ({
  result: { query_id: "q", strategy: STRATEGY[key], retriever: key, chunk_id: id, document_id: "d", document_version_id: "dv", rank, score, origin: "retrieved" },
  chunk: { id, document_version_id: "dv", chunking_hash: "h", ordinal: Number(id.slice(-1)), text: `text ${id}`, char_start: 0, char_end: 5, metadata: {} },
  filename: `${id}.txt`,
  media_type: "text/plain",
  document_version: 1,
  matched_terms: [],
  fusion,
});

const run = (key: RunKey, ids: string[], ms: number): RetrievalResponse =>
  ({
    query: { id: `q-${key}`, text: "q", metadata: {}, created_at: "" },
    hits: ids.map((id, i) => hit(id, i + 1, 1 / (i + 1), key)),
    warnings: [],
    provenance: { strategy: STRATEGY[key], corpus_version: 2, elapsed_ms: ms, statistics: {}, configuration: { hybrid: null } },
  }) as unknown as RetrievalResponse;

// c5 is in the RRF top-k but in neither single top-k (promoted); c3 is in BM25's but not RRF's (dropped)
const runs = {
  sparse: run("sparse", ["c1", "c2", "c3"], 12),
  dense: run("dense", ["c2", "c4", "c1"], 30),
  rrf: run("rrf", ["c2", "c1", "c5"], 41),
  weighted: run("weighted", ["c1", "c2", "c4"], 42),
};

describe("alignRuns / overlap / fusionEffect", () => {
  it("aligns ranks per run, ordered by best rank then id", () => {
    const rows = alignRuns(runs);
    expect(rows.map((r) => [r.chunkId, r.ranks])).toEqual([
      ["c1", { sparse: 1, rrf: 2, weighted: 1, dense: 3 }],
      ["c2", { sparse: 2, rrf: 1, weighted: 2, dense: 1 }],
      ["c4", { weighted: 3, dense: 2 }],
      ["c3", { sparse: 3 }],
      ["c5", { rrf: 3 }],
    ]);
  });

  it("counts pairwise overlap and fusion effects", () => {
    const rows = alignRuns(runs);
    expect(overlap(rows, "sparse", "dense")).toEqual({ aOnly: 1, bOnly: 1, shared: 2, sameRank: 0 });
    expect(overlap(rows, "sparse", "rrf")).toEqual({ aOnly: 1, bOnly: 1, shared: 2, sameRank: 0 });
    const effects = Object.fromEntries(rows.map((r) => [r.chunkId, fusionEffect(r, "rrf")]));
    expect(effects).toEqual({ c1: null, c2: null, c4: "dropped", c3: "dropped", c5: "promoted" });
  });

  it("handles empty runs", () => {
    expect(alignRuns({ sparse: run("sparse", [], 1), dense: run("dense", [], 1) })).toEqual([]);
  });
});

describe("CompareView", () => {
  it("shows all four runs, then a chosen pair", async () => {
    const { container } = render(<CompareView runs={runs} />);
    expect(screen.getByText("+1 / −2")).toBeInTheDocument(); // RRF promoted / dropped
    expect(screen.getByRole("img", { name: /Rank comparison across BM25, Hybrid RRF, Hybrid weighted, Dense/ })).toBeInTheDocument();
    const matrix = screen.getByRole("table");
    expect(within(matrix).getByText("c5.txt #5").closest("tr")).toHaveTextContent("RRF promoted");
    expect(screen.getByText("41.0 ms")).toBeInTheDocument();

    await userEvent.setup().click(screen.getByRole("tab", { name: "BM25 vs Dense" }));
    expect(screen.getByRole("img", { name: "1 BM25 only, 2 shared, 1 Dense only" })).toBeInTheDocument();
    expect(container.querySelectorAll('svg[aria-label^="Rank comparison"] path')).toHaveLength(2);
  });
});

describe("EvidenceCard fusion explanation", () => {
  it("reveals per-component rank, raw score and contribution", async () => {
    const fusion: FusionDetail = {
      method: "rrf",
      score: 1 / 62 + 1 / 65,
      components: [
        { strategy: "dense", rank: 5, score: 0.71, normalized_score: null, contribution: 1 / 65 },
        { strategy: "sparse", rank: 2, score: 3.2, normalized_score: null, contribution: 1 / 62 },
      ],
    };
    render(<EvidenceCard hit={hit("c1", 1, fusion.score, "rrf", fusion)} topScore={fusion.score} corpusId="c" index={0} rrfK={60} />);
    expect(screen.queryByLabelText("Fusion explanation for result 1")).not.toBeInTheDocument();
    await userEvent.setup().click(screen.getByRole("button", { name: /explain fusion/ }));
    const panel = screen.getByLabelText("Fusion explanation for result 1");
    expect(panel).toHaveTextContent("RRF (k = 60)");
    expect(panel).toHaveTextContent("1/(60+2) = 0.01613");
    expect(panel).toHaveTextContent("1/(60+5) = 0.01538");
    expect(panel).toHaveTextContent("3.2000"); // raw BM25 score kept, not overwritten
  });
});

describe("DenseIndexPanel", () => {
  it("shows build progress while building and never offers a build then", () => {
    const view = {
      corpus_id: "c", version: 2, state: "building", embedder_hash: "e",
      embedder: { model: "BAAI/bge-small-en-v1.5", revision: "5c38ec7c405e" },
      index: { id: "dix_1", chunk_count: 200, embedded: 64, reused: 0, embedder: { dimension: 384 }, similarity: "cosine" },
    } as unknown as DenseIndexView;
    const handle = { data: view, loading: false, reload: () => {}, build: async () => {}, buildError: null };
    render(<DenseIndexPanel handle={handle} />);
    expect(screen.getByText("dense building")).toBeInTheDocument();
    expect(screen.getByRole("status", { name: "Build progress" })).toHaveTextContent("64 / 200");
    expect(screen.queryByRole("button", { name: /Build dense index/ })).not.toBeInTheDocument();
  });
});
