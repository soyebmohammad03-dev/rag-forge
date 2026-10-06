# 0008: Evidence, grounding and generation

Date: 2026-10-07 · Status: accepted

## Decisions

1. **Evidence is first-class and version-pinned.** Selected passages become `Evidence` with a
   content-derived id (corpus version, chunk, span, text hash), the retrieval and reranking ranks
   copied from the hit, and a recorded selection reason. The same chunk in another corpus version
   is different evidence.
2. **Selection is an explicit, logged stage.** A deterministic greedy walk over the final ranking
   with item, token, per-document, near-duplicate and score rules. Every candidate's outcome is
   recorded. No passage is truncated to fit, and an empty selection never calls the generator.
3. **Budgets use the generator's tokenizer.** Token counts differ between models. Loading the
   tokenizer does not load the weights.
4. **Generation is a protocol with a local default.** `Qwen2.5-0.5B-Instruct` as a 4-bit ONNX
   export on onnxruntime: no PyTorch, no GPU, no paid API, weights pinned by revision and kept in
   the Hugging Face cache. Chosen over fp16 (3.4× slower on long contexts on CPU), int8 (less
   coherent) and a 1.5B int8 export (degenerate). An extractive baseline and an optional
   OpenAI-compatible endpoint share the same protocol.
5. **The prompt contract is versioned and chosen by measurement.** `grounded-qa@1` was picked
   among variants on answerable and unanswerable questions. Its known weakness (few citations from
   the 0.5B model) is measured and shown, not repaired after the fact.
6. **What was cited and what supports are separate facts.** `cited_evidence_ids` are generated;
   `supporting_evidence_ids` are measured. Citations naming unknown evidence are kept and counted.
7. **The baseline verifier is honest about its reach.** Lexical coverage plus embedding similarity
   with recorded thresholds, number and negation checks, and pooled multi-passage support. It
   cannot detect contradiction, so it never emits `contradicted`, and `unsupported` never means
   "false".
8. **Failures are explicit.** Missing components are 501, unavailable models 503 (including
   grounding after a successful generation), over-budget context 422. An answer is never
   returned without its grounding.
9. **One provenance chain.** Every stage from query to grounding records output hash, config
   hash, latency, origin and determinism. Retrieval inside `/answer` is the unchanged retrieval
   service, so its ranking equals `/retrieve`'s.
10. **Grounding is not quality.** The lab reports measured support only. Answer quality,
    relevance and faithfulness against judged data are Phase 8.
