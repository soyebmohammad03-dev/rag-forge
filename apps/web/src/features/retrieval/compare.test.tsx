import type { DenseIndexView, RetrievalHit, RetrievalResponse } from "@rag-forge/shared";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { compareRuns } from "./compare";
import { CompareView } from "./compare-view";
import { DenseIndexPanel } from "./dense-index";

const hit = (id: string, rank: number, score: number, strategy: "sparse" | "dense"): RetrievalHit =>
  ({
    result: { query_id: "q", strategy, retriever: strategy === "sparse" ? "bm25" : "dense", chunk_id: id, document_id: "d", document_version_id: "dv", rank, score, origin: "retrieved" },
    chunk: { id, document_version_id: "dv", chunking_hash: "h", ordinal: Number(id.slice(-1)), text: `text ${id}`, char_start: 0, char_end: 5, metadata: {} },
    filename: `${id}.txt`, media_type: "text/plain", document_version: 1, matched_terms: [],
  }) as RetrievalHit;

const run = (strategy: "sparse" | "dense", hits: RetrievalHit[], ms: number) =>
  ({
    query: { id: `q-${strategy}`, text: "q", metadata: {}, created_at: "" },
    hits, warnings: [],
    provenance: { strategy, corpus_version: 2, elapsed_ms: ms, statistics: strategy === "dense" ? { query_embedding_ms: 5 } : { indexed_now: 0 } },
  }) as unknown as RetrievalResponse;

const sparse = run("sparse", [hit("c1", 1, 4.2, "sparse"), hit("c2", 2, 2.1, "sparse"), hit("c3", 3, 0.5, "sparse")], 12);
const dense = run("dense", [hit("c2", 1, 0.81, "dense"), hit("c4", 2, 0.77, "dense"), hit("c1", 3, 0.7, "dense")], 30);

describe("compareRuns", () => {
  it("classifies overlap and keeps both ranks", () => {
    const cmp = compareRuns(sparse, dense);
    expect([cmp.shared, cmp.sparseOnly, cmp.denseOnly]).toEqual([2, 1, 1]);
    expect(cmp.chunks.map((c) => [c.chunkId, c.kind, c.sparseRank, c.denseRank])).toEqual([
      ["c1", "both", 1, 3],
      ["c2", "both", 2, 1],
      ["c4", "dense", null, 2],
      ["c3", "sparse", 3, null],
    ]);
  });

  it("handles empty runs", () => {
    expect(compareRuns(run("sparse", [], 1), run("dense", [], 1))).toEqual({ chunks: [], shared: 0, sparseOnly: 0, denseOnly: 0 });
  });
});

describe("CompareView", () => {
  it("renders overlap, rank links and latency", () => {
    const { container } = render(<CompareView sparse={sparse} dense={dense} />);
    expect(screen.getByRole("img", { name: "1 BM25 only, 2 shared, 1 dense only" })).toBeInTheDocument();
    expect(container.querySelectorAll('svg[aria-label^="Rank comparison"] line')).toHaveLength(2);
    expect(screen.getByText("0/2")).toBeInTheDocument(); // no shared chunk kept its rank
    expect(screen.getByText("30.0 ms")).toBeInTheDocument();
    expect(screen.getAllByText("both · #3").length).toBe(1);
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
