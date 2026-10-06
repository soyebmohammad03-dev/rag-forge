# Evidence, grounding and generation

Code: `apps/api/src/rag_forge/rag/` (`evidence.py`, `prompt.py`, `generation.py`, `claims.py`,
`grounding.py`, `service.py`, `text.py`), wired in `create_app()` / `make_generators()`; HTTP in
`api/corpus.py` (`POST /api/v1/corpora/{id}/answer`) and `api/routes.py` (`GET /api/v1/rag/components`).
UI: `apps/web/src/features/evidence/` (the Evidence Lab, `/evidence`).

```
query ─▶ (query intelligence ─▶ router) ─▶ retrieval ─▶ (reranking)        unchanged RetrievalService
      ─▶ evidence selection ─▶ context assembly ─▶ generation                 retrieved → generated
      ─▶ claim extraction ─▶ grounding analysis ─▶ cited answer + evidence    generated → measured
```

Retrieval is the unchanged `RetrievalService`, so `/answer` runs exactly the ranking `/retrieve`
returns for the same request (tested), manual or adaptive. Everything after it consumes that
response. Generation, grounding and presentation are separate, so a stronger verifier or another
generator plugs in without touching the others.

## Evidence

An `Evidence` is a passage selected to support an answer. It references a chunk rather than
copying a document, and holds exactly the text placed in the context:

| Field | Meaning |
|---|---|
| `id` | `evd_` + hash of corpus id, corpus version, chunk id, character span and text SHA-256. Equal evidence has an equal id across runs; the same chunk in another corpus version is different evidence (tested). |
| `citation` | The label the generator sees and cites: `E1`, `E2`, … in selection order |
| `corpus_id`, `corpus_version`, `chunking_hash` | The exact corpus version retrieval searched |
| `document_id`, `document_version_id`, `document_version`, `filename`, `media_type` | Source document identity |
| `chunk_id`, `chunk_ordinal`, `char_start`, `char_end` | The exact span in the extracted document text |
| `text`, `text_sha256`, `token_count` | The text used and its cost in generator tokens |
| `retrieval` | Strategy, retriever, final rank and score, upstream rank and score and reranker score when reranked, retrieval configuration hash. Copied from the hit, never recomputed. |
| `selection_rank`, `selection_score`, `selection_reason` | Why and in which order it was selected, with measured values |
| `origin` | Always `retrieved` |

### Selection (`ranked-greedy-diverse@1`)

The selector walks the final ranking (after reranking, if any), best first. Each candidate gets
exactly one recorded `SelectionDecision`, in ranking order, with a detail string:

| Outcome | Rule (checked in this order) |
|---|---|
| `over_item_budget` | `max_items` passages already selected |
| `below_min_score` | final ranking score `< min_score` (if set) |
| `document_cap` | its document version already supplied `max_per_document` passages (if set) |
| `near_duplicate` | term-set Jaccard (BM25 analyzer) with an already selected passage `≥ near_duplicate_threshold` (if set); the decision names that passage and the similarity |
| `over_token_budget` | its block would exceed `max_context_tokens` in the remaining budget |
| `selected` | otherwise |

Defaults: `max_items` 5, `max_context_tokens` 1500, `max_per_document` 2,
`near_duplicate_threshold` 0.8, `min_score` none. `min_score` has no default, because scores are
not comparable across strategies (BM25, cosine, RRF and cross-encoder logits differ in scale).
Token budgets are measured with the requested generator's own tokenizer
(`EvidenceSelection.tokenizer`), and loading that tokenizer never loads the model weights. The
selection is deterministic: equal ranking and parameters give equal decisions and an equal
`selection_hash`.

If nothing is selected (no hits, or every candidate rejected), the response is
`status: "insufficient_evidence"`, the generator is **not called**, and a warning counts the
rejection reasons (e.g. `2 over_token_budget`). Passages are never truncated to fit.

## Context assembly

`build_context` renders only the selected evidence, in selection order, one block per passage:

```
[E1] (photo.txt)
Through photosynthesis, plants use sunlight, water and CO2 to produce food.

[E2] (trucks.txt)
...
```

It refuses evidence from more than one corpus version, and raises `ContextBudgetError` (HTTP 422)
if the rendered evidence exceeds `max_context_tokens`. The local generator raises the same error
if prompt plus `max_new_tokens` would exceed the model's context window. The
`GenerationContext` records the blocks, the rendered evidence text, the exact chat messages, the
token count and two hashes:

- `context_hash`: corpus version, template id, and each block's evidence id, label, header and
  exact text. Equal context hashes mean the answer was generated from identical evidence.
- `prompt_hash`: the rendered messages (so it also changes with the question).

## Prompt contract (`grounded-qa@1`)

System: answer only from the passages; no outside knowledge; cite the passage id in square
brackets after each sentence, only ids that appear; say which part is uncertain if the passages
only partly answer; otherwise say "The evidence is insufficient." User: the passages, the
question, and the abstention instruction again, ending with `Answer:`. The template is versioned
and recorded in every context.

The wording was chosen empirically on the default model (below), by trying variants on
answerable, partly answerable and out-of-corpus questions. Long rule lists made the 0.5B model
abstain on answerable questions. One-shot examples made it copy the example, including a citation
to a passage that did not exist. Prefilling `[` produced bracketed nonsense. The chosen template
keeps answers inside the evidence and abstains on unanswerable questions, but **the model rarely
writes citations**. That is measured (citation coverage) and shown, not patched: citations are
never added after the fact.

## Generators

`Generator` is a protocol: `describe()` (never loads), `info()` (loads; model identity),
`count_tokens()`, `tokenizer_id`, `generate(GenerationInput) -> GenerationOutput` (text, finish
reason, prompt and completion tokens, load and generation latency). Requests choose one by name
(`generation.generator`; `null` means the configured default). `GET /api/v1/rag/components`
lists them without loading anything.

| Name | Provider | Notes |
|---|---|---|
| `onnx-community/Qwen2.5-0.5B-Instruct` (default) | `onnx-causal-lm` | Local, CPU, onnxruntime, no PyTorch. 4-bit weights (`onnx/model_q4.onnx`, 786 MB) at pinned revision `cc5cc01a`, fetched into the Hugging Face cache on first use, never into the repository. ChatML template, KV cache, greedy by default. |
| `extractive-baseline` | `extractive` | No model. From each passage in evidence order, copies the sentence sharing most query content terms, cited. Deterministic, instant. A floor that shows what the evidence alone says. |
| `openai-compatible` | `openai-compatible` | Registered only if `RAG_FORGE_OPENAI_BASE_URL` and `RAG_FORGE_OPENAI_MODEL` are set: any `/chat/completions` endpoint (a local llama.cpp or Ollama server, or a hosted API). The optional `RAG_FORGE_OPENAI_API_KEY` is sent as a bearer token and never recorded. Never marked deterministic. |

`RAG_FORGE_GENERATOR` (a `GeneratorSpec` as JSON) selects another ONNX causal-LM export with a
ChatML template, e.g. SmolLM2. Loading checks the chat template and the export's KV-cache layout and
fails explicitly otherwise.

### Choosing the default

Measured on an Apple M3 laptop (8 GB, CPU only), same prompt contract:

| Candidate | Size | 1,222-token prompt | Behaviour |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct, 4-bit (`model_q4`) | 786 MB | 4.2 s for 15 tokens | Correct short answers; abstains when the evidence lacks the answer |
| Qwen2.5-0.5B-Instruct, fp16 | 997 MB | 14.4 s for 15 tokens (fp16 attention is slow on CPU) | Similar answers |
| Qwen2.5-0.5B-Instruct, int8 | 512 MB | 2.7 s | Faster but less coherent: echoed the question, used outside knowledge ("Paris") |
| Qwen2.5-1.5B-Instruct, int8 | 1.6 GB | ~16 s | Degenerate repetition |

Short prompts (three passages, ~180 tokens) take 0.6–1.0 s with the default. The first answer
also pays the model load (download once, then ~1–20 s including hashing the weights), which is
reported separately as `load_ms`. Generation is serialised per model instance.

## Claims

`extract_claims` (`sentence-citation@1`) splits the raw answer into sentences (after `.`, `!`,
`?` plus whitespace, and on newlines; not after `e.g.`, `i.e.`, `Dr.`, single initials, …).
Each sentence is a claim with its exact character span in the raw answer and a deterministic id
(hash of answer hash, index and span).

- **Citations** are `[E<n>]` or several ids in one bracket (`[E1, E3]`). Markers that open a
  sentence are attached to the previous one ("fact. [E1]"). A citation is valid only if its label
  names supplied evidence. Invalid citations are kept, counted and shown struck through. They are
  never dropped or remapped.
- **Kinds**: `abstention` (says the evidence is insufficient: fixed patterns), `non_assertive` (no
  content terms, or a lead-in ending in `:`), otherwise `factual`.
- `cited_evidence_ids` is what the generator wrote (`generated`). `supporting_evidence_ids` is
  what the verifier measured (`measured`). They are separate fields and never merged.

## Grounding

`GroundingVerifier` is a contract (`name`, `version`, `detects_contradiction`, `thresholds()`,
`config_hash()`, `verify()`). The baseline `lexical-semantic@1` is local and deterministic. For
each factual claim and each passage:

- **lexical coverage**: share of the claim's content terms (BM25 analyzer, minus function words)
  found in the passage;
- **semantic similarity**: highest cosine between the claim and the passage or any of its
  sentences, using the platform's pinned embedding model (`bge-small-en-v1.5`, already loaded for
  dense retrieval);
- **status**: `supported` if lexical ≥ 0.7 and semantic ≥ 0.85; `weakly_supported` if ≥ 0.5 and
  ≥ 0.75; otherwise `unsupported`. A number in the claim that is absent from the passage caps it
  at weak. A negation on only one side (claim vs closest sentence) lowers it one level;
- **pooling**: if no single passage supports the claim, passages with semantic ≥ 0.6 that share
  a term are pooled and tested together (flag `multi_passage`), so a sentence combining two
  passages can be supported.

A claim takes its best result. It also lists every passage's scores and closest sentence, the
missing terms, unmatched numbers and flags (`uncited`, `invalid_citation`, `cited_not_supporting`,
`number_mismatch`, `negation_mismatch`, `multi_passage`). Abstentions and non-assertive sentences
are `not_applicable` and are not scored.

### Calibration

Thresholds were set on hand-built pairs, not learned. They are recorded in every report:

| Claim vs passage | Lexical | Semantic | Result |
|---|---|---|---|
| verbatim | 1.00 | 0.96 | supported |
| paraphrase ("need … make" vs "use … produce") | 0.71 | 0.93 | supported |
| one word wrong ("sugar" for "salt") | 0.67 | 0.93 | weak |
| wrong number (300 vs 30 billion) | 0.80 | 0.89 | weak (number cap) |
| reversed facts ("rose after a cut" vs "fell after a rise") | 0.71 | 0.81 | weak ⚠ |
| negated ("do not need water") | 0.60 | 0.78 | unsupported (negation) |
| partial ("Plants need light") | 0.33 | 0.80 | unsupported |
| unrelated ("capital of France") | 0.00 | 0.43 | unsupported |

### Report (`GroundingReport`, origin `measured`)

| Metric | Definition |
|---|---|
| `claims`, `factual_claims`, `supported`, `weakly_supported`, `unsupported`, `contradicted`, `abstentions`, `non_assertive` | Counts |
| `grounding_score` | (supported + 0.5 × weakly supported) / factual claims; null if none |
| `evidence_coverage` | share of selected evidence that supports at least one claim |
| `citation_coverage` | share of factual claims with at least one valid citation |
| `citation_precision` | share of valid citations whose passage the verifier measured as supporting that claim; null if no citations |
| `invalid_citations` | citations naming evidence that was not supplied |
| `status` | `grounded` (all factual supported), `partially_grounded`, `ungrounded` (none supported or weak), `abstained` (only abstentions), `no_claims` |
| `grounding_hash` | hash of every claim's kind, support, score, supporting ids and flags |
| `load_ms`, `latency_ms` | the verifier's one-time model load, kept apart from claim extraction and scoring time |

### Semantics and limits

- **`unsupported` means no support was measured, not that the claim is false.**
- **`contradicted` is never emitted by this verifier** (`detects_contradiction: false`). Term and
  embedding overlap cannot establish contradiction. The reversed-facts row above is only *weak*.
  The status exists for verifiers that can measure it (e.g. an NLI model).
- Unsupported claims stay unsupported. Nothing is rewritten, dropped or re-labelled, and the
  answer text is shown verbatim.
- **Grounding is not answer quality.** A sentence copied from an irrelevant passage is perfectly
  grounded. The extractive baseline is "grounded" by construction, and a short fragment such as
  "food trucks" can be supported without answering anything. Relevance, completeness and
  correctness need judged data. Phase 8 measures them; this phase does not claim them.

## Failure behaviour

Components are resolved before retrieval runs. Every failure is an error response, and no answer
is ever returned with its grounding missing:

| Situation | Response |
|---|---|
| Unknown generator or verifier | 501 `NotImplementedDetail` naming the registered ones |
| Generator cannot load or run (not cached and offline, bad revision, endpoint down) | 503 `{component: "generator"}` |
| Grounding model unavailable | 503 `{component: "grounding"}`, even though generation succeeded |
| Evidence over `max_context_tokens`, or prompt beyond the model's context | 422 `{component: "context"}` |
| No usable evidence | 200, `status: "insufficient_evidence"`, generator not called, reasons in `warnings` |
| Generator says the evidence is insufficient | 200, `status: "abstained"` |
| Answer cut at `max_new_tokens` | 200, `finish_reason: "length"` and a warning |
| Retrieval failures (dense index not ready, reranker unavailable, …) | exactly as `/retrieve` |

## Provenance

`RagProvenance.chain` lists, in order, the stages that ran: `query`, `query_analysis` and
`router_decision` (adaptive only), `retrieval`, `reranking` (if any), `evidence_selection`,
`context`, `generation`, `claims`, `grounding`. Each stage carries the hash of its output, the hash of
its configuration, its latency, its origin and whether it is deterministic.
`RagConfiguration` (retrieval configuration, evidence parameters, prompt template, generator and
its config hash, effective generation parameters, verifier and its config hash) hashes to
`configuration_hash`. `GenerationRecord` adds the model's revision, weights SHA-256, decoding
parameters, token counts and the raw output.

## Reproducibility boundaries

- Deterministic for equal inputs and configuration: retrieval, reranking, routing, evidence ids
  and selection, context and prompt, claims (for equal answer text) and grounding. The
  end-to-end test runs the full pipeline twice and compares every stage hash.
- Local generation is reproducible at temperature 0 (greedy) with the same model revision,
  weights, onnxruntime version and CPU kind. Floating-point differences across machines can
  change a greedy token in rare cases. `deterministic` says what the record can promise.
- Sampling (temperature > 0) is reproducible with the same seed and a local model. Sampling
  parameters are blanked out of the effective configuration at temperature 0, so they cannot
  change hashes.
- Remote endpoints are never marked deterministic: the model behind them is not pinned here.

## Limitations

- The default model is small. It answers short factual questions from clean passages well,
  struggles with Markdown-heavy or long contexts, rarely writes citations, and sometimes abstains
  when the answer is present.
- The verifier is lexical-semantic, not entailment: no contradiction detection, sensitivity to
  paraphrase, and heuristic number and negation checks.
- Sentence splitting is rule-based, and claims are sentences, not atomic facts.
- Generation is serialised per model instance. No streaming.

All ONNX models are imported through `rag_forge/onnx_runtime.py`, which disables onnxruntime's
telemetry. A local research tool has no reason to send usage events. Also, the uploader thread
could still be handling a response when the process exited, which intermittently aborted test
runs on macOS (`recursive_mutex lock failed`, traced to that thread).
