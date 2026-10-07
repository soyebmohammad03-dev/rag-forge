# Known limitations

RAG FORGE 1.0 is a complete, laptop-scale research platform. These are its boundaries, stated so
that no result is read as more than it is.

## Data and evaluation

- **The bundled benchmark is a development benchmark.** Twelve documents and fourteen cases,
  written for the purpose and saturated (most retrieval configurations reach nDCG@10 = 1.0). It
  validates the pipeline; it cannot rank methods. Real comparisons need a judged dataset where
  methods differ, registered through `POST /api/v1/benchmarks`.
- **No importer for public benchmarks** (BEIR, HotpotQA, …). They must be converted to the dataset
  schema. The `external` dataset source is reserved for them.
- **No answer-quality judgement.** There is no human evaluation interface and no LLM-as-judge.
  Grounding measures support by the passages the generator saw; token F1 and abstention accuracy
  are lexical proxies.
- **Statistics are per comparison.** Holm adjustment covers the metrics of one comparison, not a
  whole ablation grid; results are not stratified by query type in the UI (tags are exported).

## Retrieval and routing

- Dense search is exact (whole matrix in memory per index); fine to roughly 10^5–10^6 chunks, not
  beyond without an ANN index behind the same interface.
- No metadata filters, graph retrieval or multi-hop retrieval.
- The router is a transparent rule baseline over heuristic query features, not a trained policy.
  Its phrase lists, weights and thresholds are hand-chosen priors, not calibrated, and it has not
  been shown to beat fixed pipelines.
- The cross-encoder and embedder are small English models (`ms-marco-MiniLM-L-6-v2`,
  `bge-small-en-v1.5`).

## Generation and grounding

- The default generator is a 0.5B model chosen to run on a laptop CPU. It cites sparsely, sometimes
  answers with a fragment, and abstains wrongly more often than the extractive baseline (measured
  in [research/summary.md](research/summary.md)).
- The grounding verifier (`lexical-semantic@1`) is a lexical and embedding baseline. It cannot
  detect contradiction, so it never reports `contradicted`; `unsupported` never means "false".
- Long contexts on the local model take seconds per answer; there is no streaming.

## Reproducibility

- Replay compares stage hashes in the installation that runs it. Exact replay was measured on one
  machine; cross-hardware reproducibility of floating-point stages (reranking, embeddings,
  generation) has not been measured. Greedy local generation is deterministic on the same
  runtime; sampling and remote endpoints are not, and are reported as such.
- Runs recorded before runtime capture (pre-1.0 databases) have no toolchain, lockfile or model
  file hashes in their manifest.
- Runs and replays execute in the API process. A restart interrupts them; they are not resumed.

## Operations

- Single-user, local: no authentication, no multi-tenant isolation. Do not expose the API publicly.
- SQLite and local files; the store contract allows Postgres, which is not implemented.
- Ingestion supports TXT, Markdown and text-based PDF (no OCR).
