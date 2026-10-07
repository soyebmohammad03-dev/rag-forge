# Generation

Code: `apps/api/src/rag_forge/rag/prompt.py`, `rag/generation.py`. Generation consumes the context assembled from selected evidence ([evidence and grounding](README.md)).

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
