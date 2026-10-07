# Architecture

## Boundaries

```
web (Next.js) ──typed HTTP (openapi-fetch)──▶ api (FastAPI)
     │                                            │
     └── @rag-forge/shared ◀── generated from ────┘ OpenAPI (Pydantic models)
```

The web app never imports backend code. Its only knowledge of the backend is the generated
`packages/shared/src/api.ts`. Changing a Pydantic schema and forgetting `npm run contracts`
fails CI.

## Backend layers

Packages exist only when they contain code. Current and planned:

| Package | Responsibility | State |
|---|---|---|
| `domain` | Models shared by every layer; no I/O | Present |
| `api` | HTTP schemas and routes; thin, delegates to services | Present |
| `evaluation` | Metrics and evaluators; pure functions over recorded outputs | Retrieval metrics present |
| `provenance` | Environment capture, provenance records | Environment capture present |
| `storage` | `CorpusStore` contract, SQLite store, content-addressed blob store | Present |
| `ingestion` | Extractors, chunkers, ingestion service, versioning | Present (see [ingestion.md](ingestion.md)) |
| `retrieval` | `Retriever` contract, BM25, `Embedder` + ONNX embedder, dense index + retriever, `FusionStrategy` (RRF, weighted), `HybridRetriever`, `Reranker` + ONNX cross-encoder | BM25, dense, hybrid and reranking present (see [retrieval.md](retrieval.md)) |
| `arena` | Benchmark datasets, metric registry, statistics, configuration snapshots, presets, `ArenaEngine` (experiments, runs, leaderboards, comparisons) | Present (see [arena.md](arena.md)) |
| `router` | `QueryAnalyzer` + heuristic analyzer, `RouterPolicy` + rule baseline, `AdaptiveRouter` | Present (see [router.md](router.md)) |
| `rag` | Evidence selection, context assembly, prompt contract, `Generator` + ONNX causal LM / extractive / OpenAI-compatible, claim extraction, `GroundingVerifier` + lexical-semantic baseline, `RagService` | Present (see [rag.md](rag.md)) |
| `knowledge` | Entities, relations, graph retrieval support | Later |
| `models` | Shared provider adapters (embeddings, reranking and generation currently live in `retrieval/` and `rag/`) | When a second consumer appears |

Dependency direction: `api → services → domain`. `domain` imports nothing from the project.

## Storage plan

| Concern | Now | Next |
|---|---|---|
| Relational metadata (corpora, documents, versions, chunks, ingestions, datasets, experiments, runs, run cases, artifacts) | SQLite (`storage/sqlite.py`) | Postgres behind the same `CorpusStore` contract |
| Lexical index | SQLite inverted index (`storage/lexical_index.py`), version-scoped BM25 statistics | Same contract on Postgres |
| Vectors | SQLite BLOBs keyed by (embedder hash, chunk), exact cosine search (`storage/vector_index.py`) | FAISS / pgvector / ANN behind the same class |
| Original document bytes | Content-addressed files `data/blobs/ab/abcd…` | Object storage with the same addressing |
| Run traces | Content-addressed blobs referenced by `arena_artifacts` | Object storage |
| Provenance | Per run: snapshots, environment, registry and statistics versions | Replay (Phase 9) |

## Identity and reproducibility

- `ConfigurationSnapshot.config_hash()` is a SHA-256 of every behaviour-relevant field of an
  Arena arm (not its name), resolved at experiment creation. Two arms with the same hash, corpus
  version and dataset version must be comparable; the engine refuses comparisons that are not.
- `ProvenanceRecord` holds the config hash, corpus and dataset versions, retrieved chunk ids and
  an `EnvironmentSnapshot` (package versions, Python, platform, git commit).
- `RetrievalConfiguration.config_hash()` identifies a complete retrieval setup (corpus version,
  strategy, top-k, BM25 params, embedder spec, hybrid params); it is in every retrieval response.
- All domain models are frozen and reject unknown fields.

## Frontend structure

- `app/`: routing only. Built areas have their own routes (`/corpus`, `/corpus/[corpusId]`, `/retrieval`, `/router`, `/evidence`,
  `/arena`, `/experiments`, `/results`, `/system`); planned areas share one dynamic route (`[area]`) driven by `lib/areas.ts`.
- `components/ui/`: the design system (panel, button, badge, tabs, tooltip, dialog/drawer,
  data table, metric card, sparkline, chart theme, graph node, pipeline steps, states).
- `components/shell/`: sidebar, top bar, command palette (⌘K), shortcuts (?).
- `features/<area>/`: screens and their components.
- `sample/`: preview data. Every consumer shows `<OriginBadge origin="simulated" />`.

Visual language: graphite surfaces, phosphor-amber signal (`--color-signal`), cyan trace
(`--color-trace`), and a fixed colour per retrieval strategy (`--color-s-*`) used by every chart,
diagram and table. Tokens live in `apps/web/src/app/globals.css`.
