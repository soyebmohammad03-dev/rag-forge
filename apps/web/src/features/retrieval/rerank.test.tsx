import type { RerankCandidate, RerankDetail, RetrievalResponse } from "@rag-forge/shared";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { crossings, summarize, visible } from "./rerank";
import { RetrievalPage } from "./retrieval-page";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  usePathname: () => "/retrieval",
  useSearchParams: () => new URLSearchParams("corpus=cor_1"),
}));

afterEach(() => vi.unstubAllGlobals());

/** A pool as the API returns it: upstream ranks 1..n, reranked by `scores` (ties keep upstream order). */
function pool(scores: number[], k: number): RerankCandidate[] {
  const order = scores.map((s, i) => [s, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  return order.map(([score, i], f) => {
    const original = i + 1;
    const final = f + 1;
    const delta = original - final;
    const rerank: RerankDetail = {
      original_rank: original, original_score: 10 - i, reranker_score: score, final_rank: final, rank_delta: delta,
      movement: delta > 0 ? "promoted" : delta < 0 ? "demoted" : "unchanged",
      entered_top_k: final <= k && original > k, left_top_k: original <= k && final > k,
    };
    return { chunk_id: `c${original}`, document_id: "d", filename: `f${original}.txt`, chunk_ordinal: 0, rerank };
  });
}

describe("rank movement helpers", () => {
  const p = pool([0.1, 0.9, 0.5, 0.7], 2); // final order c2, c4, c3, c1

  it("summarises the final top-k", () => {
    expect(summarize(p, 2)).toEqual({ pool: 4, kept: 1, entered: 1, left: 1, promoted: 2, demoted: 0, unchanged: 0, displacement: 1.5 });
    expect(visible(p, 2).map((c) => c.chunk_id).sort()).toEqual(["c1", "c2", "c4"]);
  });

  it("explains every rank delta by the candidates swapped with", () => {
    for (const c of p) {
      const { overtook, overtakenBy } = crossings(p, c.chunk_id);
      expect(overtook.length - overtakenBy.length).toBe(c.rerank.rank_delta);
    }
    expect(crossings(p, "c4").overtook.map((c) => c.chunk_id).sort()).toEqual(["c1", "c3"]);
    expect(crossings(p, "c1").overtakenBy.map((c) => c.chunk_id).sort()).toEqual(["c2", "c3", "c4"]);
  });
});

const corpora = [
  {
    corpus: { id: "cor_1", name: "papers", version: 1, chunking: { strategy: "recursive", chunk_size: 400, chunk_overlap: 80 } },
    stats: { document_count: 4, chunk_count: 4, total_bytes: 10, total_chars: 10, last_ingestion: null },
  },
];

const denseView = {
  corpus_id: "cor_1", version: 1, state: "missing", index: null, embedder_hash: "e",
  embedder: { provider: "onnx-sentence-transformers", model: "BAAI/bge-small-en-v1.5", revision: "5c38ec7c405e", query_prefix: "", max_seq_length: null, batch_size: 32 },
};

function reranked(): RetrievalResponse {
  const candidates = pool([-9.3, 2.4, -4.8, 0.5], 2);
  const hits = candidates.slice(0, 2).map((c) => ({
    result: { query_id: "q", strategy: "sparse", retriever: "bm25", chunk_id: c.chunk_id, document_id: "d", document_version_id: "dv", rank: c.rerank.final_rank, score: c.rerank.reranker_score, origin: "retrieved" },
    chunk: { id: c.chunk_id, document_version_id: "dv", chunking_hash: "h", ordinal: 0, text: `text of ${c.chunk_id}`, char_start: 0, char_end: 9, metadata: {} },
    filename: c.filename, media_type: "text/plain", document_version: 1, matched_terms: ["food"], fusion: null, rerank: c.rerank,
  }));
  const spec = { provider: "onnx-cross-encoder", model: "cross-encoder/ms-marco-MiniLM-L-6-v2", revision: "233902d25c440f23", max_seq_length: null, batch_size: 32 };
  return {
    query: { id: "q", text: "food", metadata: {}, created_at: "" },
    warnings: [],
    hits,
    reranking: { candidates },
    provenance: {
      corpus_id: "cor_1", corpus_version: 1, chunking_hash: "h", strategy: "sparse", retriever: "bm25",
      retriever_config: { k1: 1.2, b: 0.75, analyzer: "a" }, retriever_config_hash: "r",
      configuration: { strategy: "sparse", hybrid: null }, configuration_hash: "c0ffee1234567890",
      query_terms: ["food"], statistics: { candidate_chunks: 4, matched_chunks: 4 },
      environment: { git_commit: null }, elapsed_ms: 12.5,
      reranking: {
        reranker: "cross-encoder", info: { spec, config_hash: "x", scoring: "cross-encoder pair logit (identity)", activation: "identity", max_seq_length: 512, truncation: "longest_first", weights_sha256: "5d3e70fd0c9ff14b" },
        reranker_config_hash: "rrrr", candidate_k: 4, final_top_k: 2, candidates_scored: 4,
        upstream_configuration: {}, upstream_configuration_hash: "u9u9u9u9u9u9u9", latency_ms: 7.25, statistics: {},
      },
    },
  } as unknown as RetrievalResponse;
}

describe("Retrieval Lab reranking", () => {
  it("sends the rerank configuration and shows movement with measured scores", async () => {
    const fetchMock = vi.fn(async (req: Request) => {
      if (req.url.endsWith("/retrieve")) return Response.json(reranked());
      if (req.url.includes("/dense-index")) return Response.json(denseView);
      return Response.json(corpora);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<RetrievalPage />);
    const user = userEvent.setup();
    await user.click(await screen.findByLabelText("Cross-encoder"));
    fireEvent.change(screen.getByLabelText("Candidate pool"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("Top k"), { target: { value: "2" } });
    await user.type(screen.getByLabelText("Query"), "food{Enter}");

    const top = await screen.findByRole("article", { name: "Rank 1: f2.txt" });
    expect(within(top).getByText("promoted 1")).toBeInTheDocument();
    expect(within(top).getByText("BM25 #2 · 9.000")).toBeInTheDocument();
    expect(within(top).getByText("rerank #1 · 2.400")).toBeInTheDocument();
    const second = screen.getByRole("article", { name: "Rank 2: f4.txt" });
    expect(within(second).getByText("entered top-k")).toBeInTheDocument();

    await user.click(within(second).getByRole("button", { name: /explain movement/ }));
    const explain = within(second).getByLabelText("Rank movement for result 2");
    expect(explain).toHaveTextContent("overtaken"); // f4 overtook f1 and f3
    expect(within(explain).getByText("f1.txt:0")).toBeInTheDocument();

    expect(screen.getByText("Rank movement")).toBeInTheDocument();
    expect(screen.getByText(/Left the top-k/)).toBeInTheDocument();
    expect(screen.getAllByText("7.3 ms")).toHaveLength(2); // scoring latency: metric and provenance
    expect(screen.getByText("u9u9u9u9u9u9")).toBeInTheDocument(); // upstream configuration hash

    const req = fetchMock.mock.calls.map((c) => c[0] as Request).find((r) => r.url.endsWith("/retrieve"))!;
    expect(await req.json()).toMatchObject({
      top_k: 2, strategy: "sparse", rerank: { enabled: true, model: "cross-encoder/ms-marco-MiniLM-L-6-v2", candidate_k: 4 },
    });
  });

  it("blocks a pool smaller than the final top k and keeps compare unreranked", async () => {
    const fetchMock = vi.fn(async (req: Request) =>
      req.url.includes("/dense-index") ? Response.json(denseView) : Response.json(corpora),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<RetrievalPage />);
    const user = userEvent.setup();
    await user.click(await screen.findByLabelText("Cross-encoder"));
    fireEvent.change(screen.getByLabelText("Candidate pool"), { target: { value: "5" } });
    await user.type(screen.getByLabelText("Query"), "food");
    expect(screen.getByText("The candidate pool must be at least the final top k.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Retrieve/ })).toBeDisabled();
    await user.click(screen.getByLabelText("Compare"));
    expect(screen.getByText(/runs on one strategy at a time/)).toBeInTheDocument();
  });
});
