import type { RetrievalResponse } from "@rag-forge/shared";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RetrievalPage } from "./retrieval-page";
import { formatInput, latencyBreakdown, topKOverlap } from "./routing";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  usePathname: () => "/router",
  useSearchParams: () => new URLSearchParams("corpus=cor_1"),
}));

afterEach(() => vi.unstubAllGlobals());

const hit = (id: string, rank: number, strategy = "dense") => ({
  result: { query_id: "q", strategy, retriever: strategy, chunk_id: id, document_id: "d", document_version_id: "dv", rank, score: 1 / rank, origin: "retrieved" },
  chunk: { id, document_version_id: "dv", chunking_hash: "h", ordinal: rank, text: `text ${id}`, char_start: 0, char_end: 6, metadata: {} },
  filename: `${id}.txt`, media_type: "text/plain", document_version: 1, matched_terms: [], fusion: null, rerank: null,
});

const signal = (name: string, score: number) => ({
  name, score, contributions: [{ feature: `${name}_feature`, value: 1, weight: score, contribution: score }],
});

const decision = {
  id: "rtd_1", query_id: "q", policy: "rules-baseline", policy_version: "rules-baseline@1", policy_config_hash: "pc0ffee1234567",
  analysis_hash: "an1234567890", option: "dense", preferred: "dense", strategy: "dense", hybrid: null,
  rerank: { enabled: false, model: "cross-encoder/ms-marco-MiniLM-L-6-v2", candidate_k: 50 },
  rules: [
    { rule: "no-searchable-terms", description: "No terms.", matched: false, inputs: { bm25_terms: 6 }, margin: null, outcome: "dense" },
    { rule: "semantic-unanchored", description: "Semantic class with no entity: route to dense.", matched: true, inputs: { query_class: "semantic", class_margin: -0.5743 }, margin: 0.4243, outcome: "dense" },
  ],
  rerank_rules: [{ rule: "rerank-default", description: "Otherwise, no rerank.", matched: true, inputs: {}, margin: null, outcome: "no rerank" }],
  alternatives: [
    { option: "sparse", selected: false, available: true, unavailable_reason: null, rules: ["exact-lookup"] },
    { option: "dense", selected: true, available: true, unavailable_reason: null, rules: ["semantic-unanchored"] },
    { option: "hybrid_rrf", selected: false, available: true, unavailable_reason: null, rules: ["balanced"] },
    { option: "hybrid_weighted", selected: false, available: true, unavailable_reason: null, rules: ["semantic-anchored"] },
  ],
  margin: 0.4243,
  rationale: ["semantic-unanchored: semantic query (margin -0.57) with no entity to match exactly → dense"],
  configuration_hash: "sel1234567890ab", decision_hash: "dh1234567890", created_at: "",
};

function adaptiveResponse(): RetrievalResponse {
  return {
    query: { id: "q", text: "what do plants need", metadata: {}, created_at: "" },
    warnings: [],
    reranking: null,
    hits: [hit("photo", 1), hit("trucks", 2)],
    provenance: {
      corpus_id: "cor_1", corpus_version: 1, chunking_hash: "h", strategy: "dense", retriever: "dense",
      retriever_config: { model: "m", revision: "r", dimension: 384, similarity: "cosine", pooling: "cls" }, retriever_config_hash: "x",
      configuration: { strategy: "dense", hybrid: null }, configuration_hash: "adaptive1234567",
      query_terms: [], statistics: { candidate_chunks: 4, query_embedding_ms: 3 }, environment: { git_commit: null }, elapsed_ms: 20,
      reranking: null,
      routing: {
        routing: { analyzer: "heuristic", analyzer_version: "heuristic-query-analyzer@1", analyzer_config_hash: "ac", policy: "rules-baseline", policy_version: "rules-baseline@1", policy_config_hash: "pc" },
        routing_hash: "rh1234567890ab",
        analysis_ms: 4, decision_ms: 1,
        decision,
        analysis: {
          analyzer: "heuristic", analyzer_version: "heuristic-query-analyzer@1", config_hash: "ac", query: "what do plants need",
          normalized_query: "what do plants need", analysis_hash: "an1234567890",
          features: {
            char_count: 19, token_count: 4, bm25_terms: ["what", "do", "plants", "need"], key_terms: ["plants", "need"], function_word_ratio: 0.5,
            is_question: true, question_word: "what", question_type: "factoid", quoted_phrases: [], identifiers: [], capitalized_terms: [], numbers: [],
            entities: [], concept_segments: ["what do plants need"], comparison_markers: [], multi_hop_markers: [], temporal_markers: [], negation_markers: [], ambiguity_markers: [],
          },
          signals: [signal("lexical", 0.08), signal("semantic", 0.66), signal("complexity", 0.05)],
          labels: { query_class: "semantic", class_margin: -0.58, complexity: "simple", multi_hop_likely: false, ambiguous: false, evidence_need: "single_passage" },
          corpus: {
            corpus_id: "cor_1", corpus_version: 1, analyzer: "a", chunk_count: 4, coverage: 0.5, missing_terms: ["need"], mean_idf: 1.2,
            terms: [{ term: "plants", document_frequency: 1, idf: 1.2 }, { term: "need", document_frequency: 0, idf: 2.2 }],
          },
        },
      },
    },
  } as unknown as RetrievalResponse;
}

const fixedResponse = (strategy: string, ids: string[]) =>
  ({
    query: { id: `q-${strategy}`, text: "q", metadata: {}, created_at: "" },
    warnings: [], reranking: null,
    hits: ids.map((id, i) => hit(id, i + 1, strategy)),
    provenance: { strategy, corpus_version: 1, elapsed_ms: 5, statistics: { candidate_chunks: 4, matched_chunks: 2 }, query_terms: [], retriever_config: {}, configuration: { hybrid: null }, configuration_hash: `fixed-${strategy}`, chunking_hash: "h", environment: { git_commit: null }, reranking: null, routing: null },
  }) as unknown as RetrievalResponse;

describe("routing helpers", () => {
  it("measures top-k overlap and splits latency into stages that sum to the total", () => {
    const o = topKOverlap(fixedResponse("sparse", ["a", "b", "c"]), fixedResponse("dense", ["b", "a", "d"]));
    expect(o).toEqual({ shared: 2, aOnly: 1, bOnly: 1, sameRank: 0, jaccard: 0.5 });
    const parts = latencyBreakdown(adaptiveResponse());
    expect(parts.map((p) => p.label)).toEqual(["analysis", "decision", "retrieval"]);
    expect(parts.reduce((s, p) => s + p.ms, 0)).toBe(20);
    expect(formatInput(-0.57431)).toBe("-0.5743");
    expect(formatInput(null)).toBe("—");
  });
});

const corpora = [
  {
    corpus: { id: "cor_1", name: "papers", version: 1, chunking: { strategy: "recursive", chunk_size: 400, chunk_overlap: 80 } },
    stats: { document_count: 2, chunk_count: 4, total_bytes: 10, total_chars: 10, last_ingestion: null },
  },
];
const denseView = {
  corpus_id: "cor_1", version: 1, state: "ready", index: null, embedder_hash: "e",
  embedder: { provider: "onnx-sentence-transformers", model: "BAAI/bge-small-en-v1.5", revision: "5c38ec7c405e", query_prefix: "", max_seq_length: null, batch_size: 32 },
};

describe("Router mode in the Retrieval Lab", () => {
  it("shows query intelligence, the rule trace, fixed vs adaptive and every alternative", async () => {
    const bodies: Record<string, unknown>[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (req: Request) => {
        if (req.url.endsWith("/retrieve")) {
          const body = await req.clone().json();
          bodies.push(body);
          if (body.mode === "adaptive") return Response.json(adaptiveResponse());
          return Response.json(fixedResponse(body.strategy, body.strategy === "sparse" ? ["trucks", "cars"] : ["photo", "trucks"]));
        }
        if (req.url.includes("/dense-index")) return Response.json(denseView);
        return Response.json(corpora);
      }),
    );
    render(<RetrievalPage initialRouting="adaptive" />);
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText("Query"), "what do plants need{Enter}");

    expect(await screen.findByText("What the analyzer measured")).toBeInTheDocument();
    expect(screen.getByText("Selected retrieval configuration")).toBeInTheDocument();
    expect(screen.getByText(/semantic-unanchored: semantic query \(margin -0.57\)/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Class margin -0.58" })).toBeInTheDocument();
    expect(screen.getByText("need", { selector: "span.line-through" })).toBeInTheDocument(); // missing in corpus

    // the adaptive request and the fixed BM25 baseline, side by side
    expect(bodies.map((b) => b.mode)).toEqual(["adaptive", "manual"]);
    expect(bodies[0]).toMatchObject({ query: "what do plants need", top_k: 10, mode: "adaptive" });
    expect(bodies[1]).toMatchObject({ mode: "manual", strategy: "sparse" });
    expect(screen.getByText(/Router \(Dense\) vs fixed BM25/)).toBeInTheDocument();
    expect(screen.getByLabelText("adaptive ranking")).toHaveTextContent("photo.txt:1");
    expect(within(screen.getByLabelText("fixed BM25 ranking")).getByText("trucks.txt:1")).toBeInTheDocument();

    // what each option would have returned, with the router's rerank setting
    await user.click(screen.getByRole("button", { name: /Run all options/ }));
    expect(await screen.findByRole("button", { name: /Run again/ })).toBeInTheDocument();
    const alt = bodies.slice(2);
    expect(alt.map((b) => b.strategy)).toEqual(["sparse", "dense", "hybrid", "hybrid"]);
    expect(alt.every((b) => (b.rerank as { enabled: boolean }).enabled === false && b.mode === "manual")).toBe(true);
    expect((alt[3].hybrid as { fusion: string }).fusion).toBe("weighted");

    // provenance records the routing identity and hashes
    expect(screen.getByText("rules-baseline@1", { selector: "span.font-mono" })).toBeInTheDocument();
    expect(screen.getByText("rh1234567890")).toBeInTheDocument();
  });

  it("keeps manual mode as before and hides compare in adaptive mode", async () => {
    vi.stubGlobal("fetch", vi.fn(async (req: Request) => (req.url.includes("/dense-index") ? Response.json(denseView) : Response.json(corpora))));
    render(<RetrievalPage />);
    const user = userEvent.setup();
    expect(await screen.findByLabelText("Compare")).toBeInTheDocument();
    await user.click(screen.getByLabelText("Adaptive"));
    expect(screen.queryByLabelText("Compare")).not.toBeInTheDocument();
    expect(screen.getByText("Fixed baseline")).toBeInTheDocument();
    expect(screen.getByLabelText("Run this fixed configuration alongside the router")).toBeChecked();
  });
});
