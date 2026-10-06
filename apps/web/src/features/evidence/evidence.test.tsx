import type { Claim, RagResponse } from "@rag-forge/shared";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EvidencePage } from "./evidence-page";
import { answerSegments } from "./grounding";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  usePathname: () => "/evidence",
  useSearchParams: () => new URLSearchParams("corpus=cor_1"),
}));

afterEach(() => vi.unstubAllGlobals());

const ANSWER = "Plants use sunlight and water to make food [E1]. They also need gold [E9].\nThe evidence is insufficient.";

const support = (id: string, citation: string, lexical: number, semantic: number, status: string, cited = false) => ({
  evidence_id: id, citation, cited, lexical_coverage: lexical, semantic_similarity: semantic,
  best_sentence: citation === "E1" ? "Plants use sunlight, water and CO2 to produce food." : "Food trucks need permits.",
  score: (lexical + semantic) / 2, status,
});

const claims = [
  {
    id: "clm_a", index: 0, text: "Plants use sunlight and water to make food.", raw_text: "Plants use sunlight and water to make food [E1].",
    char_start: 0, char_end: 48, kind: "factual", content_terms: ["plants", "sunlight", "water", "make", "food"],
    citations: [{ label: "E1", evidence_id: "evd_1", valid: true, char_start: 43, char_end: 47 }],
    cited_evidence_ids: ["evd_1"], supporting_evidence_ids: ["evd_1"], support: "supported", support_score: 0.86,
    missing_terms: ["make"], unmatched_numbers: [], flags: [],
    evidence: [support("evd_1", "E1", 0.8, 0.93, "supported", true), support("evd_2", "E2", 0.2, 0.4, "unsupported")],
    rationale: "best passage E1: lexical 0.80, semantic 0.93 -> supported", origin: "generated",
  },
  {
    id: "clm_b", index: 1, text: "They also need gold.", raw_text: "They also need gold [E9].",
    char_start: 49, char_end: 74, kind: "factual", content_terms: ["also", "need", "gold"],
    citations: [{ label: "E9", evidence_id: null, valid: false, char_start: 69, char_end: 73 }],
    cited_evidence_ids: [], supporting_evidence_ids: [], support: "unsupported", support_score: 0.2,
    missing_terms: ["also", "gold"], unmatched_numbers: [], flags: ["invalid_citation"],
    evidence: [support("evd_1", "E1", 0.33, 0.5, "unsupported"), support("evd_2", "E2", 0.33, 0.3, "unsupported")],
    rationale: "best passage E1: lexical 0.33, semantic 0.50 -> unsupported", origin: "generated",
  },
  {
    id: "clm_c", index: 2, text: "The evidence is insufficient.", raw_text: "The evidence is insufficient.",
    char_start: 75, char_end: 104, kind: "abstention", content_terms: [], citations: [], cited_evidence_ids: [],
    supporting_evidence_ids: [], support: "not_applicable", support_score: null, missing_terms: [], unmatched_numbers: [],
    flags: [], evidence: [], rationale: "abstention: not checked against evidence", origin: "generated",
  },
] as unknown as Claim[];

const evidence = (id: string, citation: string, file: string, rank: number) => ({
  id, citation, corpus_id: "cor_1", corpus_version: 1, chunking_hash: "ch", document_id: `doc_${file}`, document_version_id: `dv_${file}`,
  document_version: 1, filename: file, media_type: "text/plain", chunk_id: `chk_${file}`, chunk_ordinal: 0, char_start: 0, char_end: 70,
  text: citation === "E1" ? "Through photosynthesis, plants use sunlight, water and CO2 to produce food." : "Food trucks need permits.",
  text_sha256: "abc123", token_count: 20,
  retrieval: { strategy: "sparse", retriever: "bm25", rank, score: 2 / rank, upstream_rank: null, upstream_score: null, reranker_score: null, configuration_hash: "cfg" },
  selection_rank: rank, selection_score: 2 / rank, selection_reason: `final rank ${rank}`, origin: "retrieved",
});

const stage = (name: string, origin: string, ms: number | null, deterministic = true) => ({
  stage: name, hash: `${name}hash0000`, config_hash: null, latency_ms: ms, origin, deterministic, detail: `${name} detail`,
});

function ragResponse(): RagResponse {
  const ev = [evidence("evd_1", "E1", "photo.txt", 1), evidence("evd_2", "E2", "trucks.txt", 2)];
  return {
    query: { id: "q", text: "what do plants need?", metadata: {}, created_at: "" },
    status: "answered",
    retrieval: {
      query: { id: "q", text: "what do plants need?", metadata: {}, created_at: "" }, hits: [], warnings: [], reranking: null,
      provenance: { corpus_version: 1, strategy: "sparse", configuration_hash: "retrievalcfg1234", reranking: null, routing: null },
    },
    evidence: {
      params: { max_items: 5, max_context_tokens: 1500, max_per_document: 2, near_duplicate_threshold: 0.8, min_score: null },
      params_hash: "ph", selector: "ranked-greedy-diverse@1", candidates: 3, selected: ev, tokens_used: 40,
      tokenizer: "onnx-community/Qwen2.5-0.5B-Instruct@cc5cc01a", selection_hash: "selhash12345678", latency_ms: 0.2,
      decisions: [
        { chunk_id: "chk_photo.txt", document_id: "d1", filename: "photo.txt", rank: 1, score: 2, outcome: "selected", detail: "final rank 1, score 2.0000", token_count: 20, evidence_id: "evd_1", similar_to: null, similarity: null },
        { chunk_id: "chk_trucks.txt", document_id: "d2", filename: "trucks.txt", rank: 2, score: 1, outcome: "selected", detail: "final rank 2, score 1.0000", token_count: 20, evidence_id: "evd_2", similar_to: null, similarity: null },
        { chunk_id: "chk_copy.txt", document_id: "d3", filename: "photo-copy.txt", rank: 3, score: 0.9, outcome: "near_duplicate", detail: "term Jaccard 0.920 >= 0.8 with evd_1", token_count: null, evidence_id: null, similar_to: "evd_1", similarity: 0.92 },
      ],
    },
    context: {
      corpus_id: "cor_1", corpus_version: 1, prompt_template: "grounded-qa@1", blocks: [], evidence_text: "", messages: [],
      context_tokens: 41, max_context_tokens: 1500, tokenizer: "t", context_hash: "ctxhash123456789", prompt_hash: "prompthash1234",
    },
    answer: {
      text: ANSWER,
      generation: {
        generator: { name: "onnx-community/Qwen2.5-0.5B-Instruct", provider: "onnx-causal-lm", model: "onnx-community/Qwen2.5-0.5B-Instruct", revision: "cc5cc01a65cc", config_hash: "gencfg", local: true, deterministic_at_zero_temperature: true, max_context_tokens: 32768, tokenizer: "t", weights_sha256: "w123", details: {} },
        params: { generator: "onnx-community/Qwen2.5-0.5B-Instruct", max_new_tokens: 200, temperature: 0, top_p: 1, seed: 0 },
        params_hash: "pp", prompt_hash: "prompthash1234", raw_text: ANSWER, answer_hash: "answerhash1234", finish_reason: "stop",
        prompt_tokens: 180, completion_tokens: 24, deterministic: true, load_ms: 0, latency_ms: 900, origin: "generated",
      },
    },
    claims,
    grounding: {
      verifier: "lexical-semantic", verifier_version: "1", config_hash: "groundcfg", detects_contradiction: false,
      thresholds: { supported_lexical: 0.7, supported_semantic: 0.85 }, status: "partially_grounded",
      claims: 3, factual_claims: 2, supported: 1, weakly_supported: 0, unsupported: 1, contradicted: 0, abstentions: 1, non_assertive: 0,
      grounding_score: 0.5, evidence_coverage: 0.5, citation_coverage: 0.5, citation_precision: 1, invalid_citations: 1,
      grounding_hash: "gh", load_ms: 1200, latency_ms: 20, origin: "measured",
    },
    provenance: {
      configuration: { retrieval: { strategy: "sparse" } }, configuration_hash: "pipelinecfg12345",
      chain: [stage("query", "retrieved", null), stage("retrieval", "retrieved", 5), stage("evidence_selection", "retrieved", 0.2), stage("context", "retrieved", 0.1), stage("generation", "generated", 900), stage("claims", "inferred", null), stage("grounding", "measured", 20)],
      environment: { git_commit: "abcdef1234567" }, elapsed_ms: 930, answered_at: "",
    },
    warnings: ["the answer cites 1 passage id(s) that were not supplied"],
  } as unknown as RagResponse;
}

const corpora = [
  {
    corpus: { id: "cor_1", name: "papers", version: 1, chunking: { strategy: "recursive", chunk_size: 400, chunk_overlap: 80 } },
    stats: { document_count: 3, chunk_count: 3, total_bytes: 10, total_chars: 10, last_ingestion: null },
  },
];
const components = {
  generators: [
    { name: "onnx-community/Qwen2.5-0.5B-Instruct", provider: "onnx-causal-lm", model: "onnx-community/Qwen2.5-0.5B-Instruct", revision: "cc5c", local: true, loaded: false },
    { name: "extractive-baseline", provider: "extractive", model: "extractive-baseline@1", revision: null, local: true, loaded: true },
  ],
  default_generator: "onnx-community/Qwen2.5-0.5B-Instruct",
  verifiers: [{ name: "lexical-semantic", version: 1, detects_contradiction: false, thresholds: {}, config_hash: "v" }],
  prompt_template: "grounded-qa@1",
  evidence_defaults: { max_items: 5, max_context_tokens: 1500, max_per_document: 2, near_duplicate_threshold: 0.8, min_score: null },
  generation_defaults: { generator: "onnx-community/Qwen2.5-0.5B-Instruct", max_new_tokens: 200, temperature: 0, top_p: 1, seed: 0 },
};
const denseView = { corpus_id: "cor_1", version: 1, state: "missing", index: null, embedder_hash: "e", embedder: { model: "m" } };

function stubApi(answer: () => unknown, bodies: Record<string, unknown>[] = []) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (req: Request) => {
      if (req.url.endsWith("/answer")) {
        bodies.push(await req.clone().json());
        return Response.json(answer());
      }
      if (req.url.endsWith("/rag/components")) return Response.json(components);
      if (req.url.includes("/dense-index")) return Response.json(denseView);
      return Response.json(corpora);
    }),
  );
}

describe("answerSegments", () => {
  it("keeps the raw answer intact and separates claims from model-written citations", () => {
    const segs = answerSegments(ANSWER, claims);
    expect(segs.map((s) => s.text).join("")).toBe(ANSWER); // nothing rewritten or dropped
    const cites = segs.filter((s) => s.kind === "citation");
    expect(cites.map((c) => c.kind === "citation" && c.labels)).toEqual([
      [{ label: "E1", evidenceId: "evd_1" }],
      [{ label: "E9", evidenceId: null }],
    ]);
    expect(segs.filter((s) => s.kind === "claim").map((s) => s.text)).toEqual([
      "Plants use sunlight and water to make food ",
      ".",
      "They also need gold ",
      ".",
      "The evidence is insufficient.",
    ]);
    expect(segs.filter((s) => s.kind === "text").map((s) => s.text)).toEqual([" ", "\n"]);
  });

  it("groups several ids written in one bracket into one marker", () => {
    const text = "A fact [E1, E2].";
    const claim = { ...claims[0], char_start: 0, char_end: 16, citations: [
      { label: "E1", evidence_id: "evd_1", valid: true, char_start: 7, char_end: 15 },
      { label: "E2", evidence_id: "evd_2", valid: true, char_start: 7, char_end: 15 },
    ] } as Claim;
    const cites = answerSegments(text, [claim]).filter((s) => s.kind === "citation");
    expect(cites).toHaveLength(1);
    expect(cites[0].text).toBe("[E1, E2]");
    expect(cites[0].kind === "citation" && cites[0].labels.map((l) => l.label)).toEqual(["E1", "E2"]);
  });
});

describe("Evidence Lab", () => {
  it("runs the pipeline and shows answer, grounding, claims, evidence, selection and provenance", async () => {
    const bodies: Record<string, unknown>[] = [];
    stubApi(ragResponse, bodies);
    render(<EvidencePage />);
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText("Question"), "what do plants need?{Enter}");

    // the request carries routing, evidence and generation configuration
    expect(await screen.findByLabelText("Generated answer")).toHaveTextContent("Plants use sunlight and water");
    expect(bodies[0]).toMatchObject({
      retrieval: { query: "what do plants need?", mode: "adaptive" },
      evidence: { max_items: 5, max_context_tokens: 1500, max_per_document: 2, near_duplicate_threshold: 0.8, min_score: null },
      generation: { generator: "onnx-community/Qwen2.5-0.5B-Instruct", temperature: 0, max_new_tokens: 200 },
    });

    // answer: measured support underlines, valid and invalid model citations
    const answer = screen.getByLabelText("Generated answer");
    expect(answer.querySelector('[data-support="supported"]')).toHaveTextContent("Plants use sunlight");
    expect(answer.querySelector('[data-support="unsupported"]')).toHaveTextContent("They also need gold");
    expect(within(answer).getByRole("button", { name: "E9" })).toHaveClass("line-through");
    expect(screen.getByText(/cites 1 passage id\(s\) that were not supplied/)).toBeInTheDocument();

    // grounding: measured counts and coverage, with the honesty caveat
    expect(screen.getByText("partially grounded")).toBeInTheDocument();
    expect(screen.getByText("0.50")).toBeInTheDocument();
    expect(screen.getByText(/cannot detect contradiction/)).toBeInTheDocument();
    expect(screen.getByText(/not answer quality or truth/)).toBeInTheDocument();
    expect(screen.getByText(/20\.0 ms \+ 1200 ms model load/)).toBeInTheDocument(); // load kept apart

    // evidence and every selection decision
    expect(screen.getByText("2 passages selected from 3 ranked")).toBeInTheDocument();
    expect(screen.getByText("near duplicate")).toBeInTheDocument();
    expect(screen.getByText(/term Jaccard 0.920/)).toBeInTheDocument();
    expect(screen.getByText(/41 \/ 1500 tokens/)).toBeInTheDocument();

    // claim -> evidence relationship expands to per-passage measurements
    await user.click(screen.getByRole("button", { name: /They also need gold\./ }));
    const row = document.querySelector('[data-claim="clm_b"]') as HTMLElement;
    expect(within(row).getByText("invalid_citation")).toBeInTheDocument();
    expect(within(row).getByText(/lexical 0.33, semantic 0.50 -> unsupported/)).toBeInTheDocument();
    expect(within(row).getByRole("table")).toBeInTheDocument();

    // provenance chain and identities
    const chain = screen.getByLabelText("Pipeline stages");
    expect(within(chain).getAllByRole("listitem").map((li) => li.querySelector("div > div")?.textContent)).toEqual([
      "Query", "Retrieval", "Evidence", "Context", "Generation", "Claims", "Grounding",
    ]);
    expect(screen.getByText("ctxhash12345")).toBeInTheDocument();
    expect(screen.getByText("pipelinecfg1")).toBeInTheDocument();
    expect(screen.getByText("yes (greedy, local)")).toBeInTheDocument();
  });

  it("shows an explicit insufficient-evidence result and supports context-only runs", async () => {
    const bodies: Record<string, unknown>[] = [];
    stubApi(() => ({
      ...ragResponse(),
      status: "insufficient_evidence",
      context: null,
      answer: null,
      claims: [],
      grounding: null,
      evidence: { ...ragResponse().evidence, selected: [], decisions: [], candidates: 0, tokens_used: 0 },
      warnings: ["retrieval returned no passages: the generator was not called"],
    }), bodies);
    render(<EvidencePage />);
    const user = userEvent.setup();
    await user.selectOptions(await screen.findByLabelText("Generator"), "none: stop after context");
    expect(screen.queryByLabelText("Temp")).not.toBeInTheDocument();
    await user.type(screen.getByLabelText("Question"), "capital of France{Enter}");

    expect(await screen.findByText(/the generator was not called/)).toBeInTheDocument();
    expect(screen.getAllByText("insufficient evidence").length).toBeGreaterThan(0);
    expect(screen.queryByLabelText("Generated answer")).not.toBeInTheDocument();
    expect(bodies[0].generation).toBeNull();
  });
});
