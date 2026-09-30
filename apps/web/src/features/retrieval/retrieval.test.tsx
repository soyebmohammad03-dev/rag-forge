import type { RetrievalResponse } from "@rag-forge/shared";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { highlight } from "./evidence-card";
import { errorMessage } from "./errors";
import { RetrievalPage } from "./retrieval-page";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  usePathname: () => "/retrieval",
  useSearchParams: () => new URLSearchParams("corpus=cor_1"),
}));

afterEach(() => vi.unstubAllGlobals());

const corpora = [
  {
    corpus: { id: "cor_1", name: "papers", version: 2, chunking: { strategy: "recursive", chunk_size: 400, chunk_overlap: 80 } },
    stats: { document_count: 3, chunk_count: 6, total_bytes: 10, total_chars: 10, last_ingestion: null },
  },
];

const response = {
  query: { id: "qry_1", text: "dense vectors", metadata: {}, created_at: "2026-01-01T00:00:00Z" },
  warnings: [],
  hits: [
    {
      result: { query_id: "qry_1", strategy: "sparse", retriever: "bm25", chunk_id: "chk_a", document_id: "doc_1", document_version_id: "dv_1", rank: 1, score: 2.5, origin: "retrieved" },
      chunk: { id: "chk_a", document_version_id: "dv_1", chunking_hash: "h", ordinal: 0, text: "Dense Vectors compare passages.", char_start: 0, char_end: 31, metadata: { words: 4, page_start: 2, page_end: 2 } },
      filename: "paper.pdf",
      media_type: "application/pdf",
      document_version: 1,
      matched_terms: ["dense", "vectors"],
    },
  ],
  provenance: {
    corpus_id: "cor_1", corpus_version: 2, chunking_hash: "abcdef1234567890", strategy: "sparse", retriever: "bm25",
    retriever_config: { k1: 1.2, b: 0.75, analyzer: "nfkc-casefold-word-lucene33@1" }, retriever_config_hash: "9af8d48f39981234",
    configuration: { corpus_id: "cor_1", corpus_version: 2, chunking_hash: "abcdef", strategy: "sparse", top_k: 10, bm25: { k1: 1.2, b: 0.75 }, embedder: null, hybrid: null },
    configuration_hash: "c0ffee1234567890",
    query_terms: ["dense", "vectors"], statistics: { candidate_chunks: 6, matched_chunks: 1, avg_chunk_length: 30, indexed_now: 0 },
    environment: { rag_forge_version: "0.1.0", python_version: "3.13", platform: "x", git_commit: "733c0aa1234567", packages: {} },
    elapsed_ms: 3.2, retrieved_at: "2026-01-01T00:00:00Z",
  },
} as unknown as RetrievalResponse;

const denseView = (state: string, index: object | null = null) => ({
  corpus_id: "cor_1", version: 2, state, index, embedder_hash: "e",
  embedder: { provider: "onnx-sentence-transformers", model: "BAAI/bge-small-en-v1.5", revision: "5c38ec7c405e", query_prefix: "", max_seq_length: null, batch_size: 32 },
});

/** Minimal API: corpora, dense-index status, and a canned retrieve response. */
function route(req: Request, retrieve: RetrievalResponse | Response) {
  if (req.url.endsWith("/retrieve")) return retrieve instanceof Response ? retrieve : Response.json(retrieve);
  if (req.url.includes("/dense-index")) return Response.json(denseView("missing"));
  return Response.json(corpora);
}

describe("RetrievalPage", () => {
  it("sends a typed request and renders evidence with provenance", async () => {
    const fetchMock = vi.fn(async (req: Request) => route(req, response));
    vi.stubGlobal("fetch", fetchMock);
    render(<RetrievalPage />);
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText("Query"), "dense vectors{Enter}");

    const card = await screen.findByRole("article", { name: "Rank 1: paper.pdf" });
    expect(within(card).getByText("2.500")).toBeInTheDocument();
    expect(within(card).getByText(/doc v1 · chunk #0 · 0–31 · p\.2/)).toBeInTheDocument();
    expect(within(card).getAllByText("retrieved").length).toBeGreaterThan(0);
    expect(screen.getByText("c0ffee123456")).toBeInTheDocument(); // configuration hash in provenance

    const req = fetchMock.mock.calls.map((c) => c[0] as Request).find((r) => r.url.endsWith("/retrieve"))!;
    expect(req.url).toBe("http://localhost:8000/api/v1/corpora/cor_1/retrieve");
    expect(await req.json()).toMatchObject({ query: "dense vectors", top_k: 10, version: null, strategy: "sparse", bm25: { k1: 1.2, b: 0.75 } });
  });

  it("shows API errors instead of results", async () => {
    vi.stubGlobal("fetch", vi.fn(async (req: Request) =>
      route(req, Response.json({ detail: { capability: "Retrieval strategy", message: "not implemented yet" } }, { status: 501 })),
    ));
    render(<RetrievalPage />);
    await userEvent.setup().type(await screen.findByLabelText("Query"), "x{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("not implemented yet");
  });
});

describe("helpers", () => {
  it("highlights matched terms case-insensitively, whole words only", () => {
    const { container } = render(<p>{highlight("Dense densely DENSE vectors.", ["dense"])}</p>);
    expect([...container.querySelectorAll("mark")].map((m) => m.textContent)).toEqual(["Dense", "DENSE"]);
  });

  it("formats FastAPI error bodies", () => {
    expect(errorMessage({ detail: "corpus has no version 9" }, 404)).toBe("corpus has no version 9");
    expect(errorMessage({ detail: [{ msg: "query must not be blank" }] }, 422)).toBe("query must not be blank");
    expect(errorMessage(undefined, 500)).toBe("API responded 500");
  });
});

describe("dense gating", () => {
  it("disables dense retrieval until the version has a ready index", async () => {
    const fetchMock = vi.fn(async (req: Request) => route(req, response));
    vi.stubGlobal("fetch", fetchMock);
    render(<RetrievalPage />);
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText("Query"), "semantic");
    await user.click(screen.getByLabelText("Dense"));
    expect(await screen.findByText(/Dense retrieval is unavailable for this version/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Retrieve/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Build dense index" })).toBeInTheDocument();
    await user.click(screen.getByLabelText("BM25"));
    expect(screen.getByRole("button", { name: /Retrieve/ })).toBeEnabled();
    expect(fetchMock.mock.calls.some((c) => (c[0] as Request).url.endsWith("/retrieve"))).toBe(false);
  });
});
