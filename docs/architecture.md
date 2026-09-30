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
| `retrieval` | `Retriever` contract, analyzer, BM25, retrieval service; later dense, fusion, reranking | BM25 present (see [retrieval.md](retrieval.md)) |
| `experiments` | Run orchestration, ablation grids | Phase 3 |
| `router` | Query analysis, policies, `RouterDecision` recording | Phase 4 |
| `knowledge` | Entities, relations, graph retrieval support | Later |
| `models` | Provider adapters for embeddings, rerankers and LLMs | With the first dependency |

Dependency direction: `api → services → domain`. `domain` imports nothing from the project.

## Storage plan

| Concern | Now | Next |
|---|---|---|
| Relational metadata (corpora, documents, versions, chunks, ingestions, experiments) | SQLite (`storage/sqlite.py`) | Postgres behind the same `CorpusStore` contract |
| Lexical index | SQLite inverted index (`storage/lexical_index.py`), version-scoped BM25 statistics | Same contract on Postgres |
| Vectors | none | A dense `Retriever` with an embedded index; backend swappable |
| Original document bytes | Content-addressed files `data/blobs/ab/abcd…` | Object storage with the same addressing |
| Provenance | Model defined | Stored with each run, immutable |

## Identity and reproducibility

- `RAGConfiguration.config_hash()` is a SHA-256 of all behaviour-relevant fields (not `id` or
  `name`). Two runs with the same hash, corpus version and dataset version must be comparable.
- `ProvenanceRecord` holds the config hash, corpus and dataset versions, retrieved chunk ids and
  an `EnvironmentSnapshot` (package versions, Python, platform, git commit).
- All domain models are frozen and reject unknown fields.

## Frontend structure

- `app/`: routing only. Built areas have their own routes (`/corpus`, `/corpus/[corpusId]`, `/retrieval`,
  `/system`); planned areas share one dynamic route (`[area]`) driven by `lib/areas.ts`.
- `components/ui/`: the design system (panel, button, badge, tabs, tooltip, dialog/drawer,
  data table, metric card, sparkline, chart theme, graph node, pipeline steps, states).
- `components/shell/`: sidebar, top bar, command palette (⌘K), shortcuts (?).
- `features/<area>/`: screens and their components.
- `sample/`: preview data. Every consumer shows `<OriginBadge origin="simulated" />`.

Visual language: graphite surfaces, phosphor-amber signal (`--color-signal`), cyan trace
(`--color-trace`), and a fixed colour per retrieval strategy (`--color-s-*`) used by every chart,
diagram and table. Tokens live in `apps/web/src/app/globals.css`.
