# RAG FORGE

A research laboratory for building, debugging and comparing retrieval-augmented generation systems.

RAG FORGE is not a chatbot. Its research question is:

> **Can adaptive retrieval policies select and compose retrieval strategies according to query
> characteristics and evidence requirements more effectively than fixed retrieval pipelines?**

Answering that needs three things that most RAG projects don't have together: a configurable
engine, a router whose decisions you can inspect, and an arena where comparisons are controlled
and reproducible. RAG FORGE provides all three, plus an interface that shows how each part works.

## Three systems, one loop

```
            ┌──────────────────────── RAG ARENA ─────────────────────────┐
            │  datasets · query sets · configurations · runs · metrics    │
            │  ablations · significance · replay                          │
            └──────────────▲───────────────────────────────┬──────────────┘
                           │ measured results               │ configurations
┌─────────────── RAG FORGE ENGINE ──────────────┐   ┌───────▼──────── RAG ROUTER ─────────────┐
│ ingest → parse → chunk → embed → index        │   │ query → analyze → classify → select      │
│ sparse · dense · hybrid · metadata · graph    │◀──│ strategies → retrieve → fuse → rerank →  │
│ rerank · generate · cite · provenance         │   │ verify evidence (↺ multi-hop) → generate │
└───────────────────────────────────────────────┘   └──────────────────────────────────────────┘
```

- **Engine** is the set of interchangeable components: ingestion, chunkers, retrievers,
  rerankers, generators, and the provenance recorded for each.
- **Router** decides how a query should be answered: which strategies, in what combination,
  and whether the evidence is enough or another hop is needed. Every choice is recorded as a
  `RouterDecision` with its features and rationale.
- **Arena** runs configurations (naive, sparse, dense, hybrid, reranked, graph, routed) against
  the same data and scores them with the same evaluators.

## Status

**Phase 7, evidence, grounding and generation.** What exists and works today:

| Area | State |
|---|---|
| Persistent storage (SQLite + content-addressed blobs) | Implemented, append-only history enforced by DB triggers |
| Corpus management: create, list, inspect, stats, metadata | Implemented |
| Ingestion of TXT, Markdown, PDF with per-file outcomes and reasons | Implemented |
| Corpus versioning: added / modified / removed / unchanged, duplicates by SHA-256 | Implemented, historical versions readable |
| Chunking: `recursive` (boundary-aware) and `fixed` baseline, exact offsets | Implemented, configurable per corpus |
| Ingestion provenance (parser, chunking hash, hashes, git commit, timings) | Implemented, one immutable record per operation |
| Corpus workspace UI (list, create, upload, documents, versions, chunks, provenance) | Implemented against the real API |
| Retrieval contract (`Retriever`) with a strategy registry | Implemented |
| BM25 lexical retrieval with version-scoped statistics, `POST /api/v1/corpora/{id}/retrieve` | Implemented, deterministic, reproducible per corpus version |
| Dense retrieval: `BAAI/bge-small-en-v1.5` (pinned, local ONNX), SQLite vector store, exact cosine search | Implemented; explicit index builds with ready/building/missing/stale/failed states |
| Hybrid retrieval: RRF (primary) and weighted min-max fusion, per-component ranks/scores/contributions | Implemented; fixed strategy, no silent fallback |
| `RetrievalConfiguration` + hash in every response | Implemented (the unit the Arena will vary) |
| Cross-encoder reranking after any strategy: `cross-encoder/ms-marco-MiniLM-L-6-v2` (pinned, local ONNX, CPU), configurable candidate pool and final top-k, per-candidate rank movement | Implemented; no unreranked fallback, full reranking provenance |
| Query intelligence: deterministic heuristic analyzer (features, lexical/semantic/complexity signals with contributions, labels, corpus term coverage) | Implemented; a baseline, not a trained model |
| Adaptive router: rule policy choosing BM25, dense, hybrid RRF or weighted and reranking, with rule traces, margins, alternatives and availability constraints; `mode: "adaptive"`, `POST /api/v1/router/decide` | Implemented; full routing provenance, no superiority claims |
| Retrieval Lab UI: BM25, Dense, Hybrid RRF, Hybrid weighted, four-way comparison, fusion explanation, reranking analysis, adaptive routing (query intelligence, decision trace, fixed vs adaptive, alternatives) | Implemented against the real API |
| Evidence selection: version-pinned `Evidence` with stable ids, item/token/per-document/near-duplicate/score budgets, every candidate's outcome recorded | Implemented; deterministic, never truncates, no evidence → generator not called |
| Grounded generation: provider-agnostic `Generator`; default `Qwen2.5-0.5B-Instruct` (pinned, 4-bit ONNX, local CPU), extractive baseline, optional OpenAI-compatible endpoint; versioned prompt contract; `POST /api/v1/corpora/{id}/answer` | Implemented; greedy and reproducible locally, raw output and citations retained |
| Claims and grounding: sentence claims, parsed citations (invalid ones kept), lexical-semantic verifier with recorded thresholds, supported / weak / unsupported, grounding score, evidence and citation coverage, citation precision | Implemented; measured support only, never `contradicted`, not a quality score |
| Evidence Lab UI: answer with measured-support underlines and model citations, grounding metrics, claim → evidence table, selection decisions, context budget, provenance chain, query intelligence and router | Implemented against the real API |
| Retrieval metrics: Recall@K, Precision@K, MRR, nDCG@K | Implemented, `POST /api/v1/evaluation/retrieval` |
| Arena and experiments, replay | Not built; unregistered components return `501` |
| Other product areas (Forge, Arena, Experiments, …) | Scoped, not built |

Numbers on the Overview marked **SAMPLE** (hatched badge) are preview data from
`apps/web/src/sample/`. They were not measured. Everything marked **LIVE** comes from the API.

## Principles

1. **Never fabricate results.** Unimplemented capabilities return `501`, not plausible output.
   Preview data is isolated and visibly labelled.
2. **Every value has an origin.** `ContentOrigin` is one of `retrieved`, `inferred`, `generated`,
   `measured`, `simulated`. The UI shows it.
3. **Every run is reproducible.** A run is identified by its configuration hash, and its
   `ProvenanceRecord` pins corpus and dataset versions, retrieved chunk ids and the environment.
4. **Contracts before implementations.** The web client's types are generated from the API's
   OpenAPI schema, and CI fails if they drift.
5. **No lock-in.** No vector database or LLM provider is assumed by the domain model.

## Architecture

```
rag-forge/
├── apps/
│   ├── api/                  FastAPI + Pydantic (Python ≥3.12, uv)
│   │   ├── src/rag_forge/
│   │   │   ├── domain/       foundational models
│   │   │   ├── ingestion/    extraction, chunking, ingestion service
│   │   │   ├── retrieval/    retriever contract, BM25, embeddings, dense, fusion, hybrid, rerank, service
│   │   │   ├── router/       query analyzer, router policy, adaptive router
│   │   │   ├── rag/          evidence selection, context, prompt, generators, claims, grounding, service
│   │   │   ├── evaluation/   retrieval metrics
│   │   │   ├── provenance/   environment capture
│   │   │   ├── storage/      store contract, SQLite + migrations, lexical index, vector index, blobs
│   │   │   ├── api/          HTTP schemas + routes (system, router, rag, corpus)
│   │   │   ├── onnx_runtime.py  single onnxruntime import (telemetry off)
│   │   │   └── main.py       app factory
│   │   └── tests/
│   └── web/                  Next.js 16 (App Router) + TypeScript + Tailwind v4
│       └── src/
│           ├── app/          routes only; thin
│           ├── components/   ui/ (design system) · shell/ (sidebar, command palette)
│           ├── features/     overview/ · corpus/ · retrieval/ · evidence/ · system/ · planned/
│           ├── lib/          typed API client, area registry
│           └── sample/       preview data, isolated
├── packages/shared/          API contracts generated from OpenAPI
├── scripts/gen-contracts.sh  regenerates packages/shared
├── docs/                     architecture, research methodology, decisions
├── data/  experiments/       local corpora and run outputs (git-ignored)
└── .github/workflows/ci.yml
```

See [docs/architecture.md](docs/architecture.md) for layering,
[docs/ingestion.md](docs/ingestion.md) for ingestion and versioning,
[docs/retrieval.md](docs/retrieval.md) for the retrieval contract, BM25, dense and hybrid retrieval and reranking,
[docs/router.md](docs/router.md) for query intelligence and adaptive routing,
[docs/rag.md](docs/rag.md) for evidence, generation, claims and grounding, and
[docs/research/methodology.md](docs/research/methodology.md) for how comparisons will be run.

## Technology

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI, Pydantic v2, uvicorn | Typed contracts that generate OpenAPI |
| Storage | SQLite (stdlib `sqlite3`, WAL), content-addressed files | Zero-ops locally; the store contract allows Postgres later |
| Extraction | pypdf (PDF); stdlib for text/Markdown | Pure Python, no system dependencies |
| Embeddings, reranking, generation | onnxruntime + tokenizers, model files from the Hugging Face cache | Local, pinned, CPU-only, ~85 MB of deps instead of PyTorch |
| Python tooling | uv, ruff, mypy (strict), pytest | Fast, lockfile-reproducible |
| Web | Next.js 16, React 19, TypeScript, Tailwind v4 | App Router, static where possible |
| Motion & charts | Motion, Recharts, SVG | Motion shows state and data flow |
| Interaction | cmdk (command palette), native `<dialog>` | Accessible without a component framework |
| Contracts | openapi-typescript + openapi-fetch | One source of truth: the Python models |
| Tests | Vitest + Testing Library; pytest + TestClient | |

## Development

Requires Node ≥ 20, npm, [uv](https://docs.astral.sh/uv/).

```bash
npm install
(cd apps/api && uv sync)
```

Run the API (http://localhost:8000, docs at `/docs`) and the web app (http://localhost:3000):

```bash
npm run dev:api
```

```bash
npm run dev:web
```

Checks:

```bash
npm run check                                   # web: lint, typecheck, test, build
(cd apps/api && uv run ruff check src tests && uv run mypy src tests && uv run pytest)
npm run contracts                               # after changing API schemas
```

Configuration: `NEXT_PUBLIC_API_URL` (web, default `http://localhost:8000`),
`RAG_FORGE_CORS_ORIGINS` (API, comma-separated, default `http://localhost:3000`),
`RAG_FORGE_DATA_DIR` (API, SQLite database and blobs; `npm run dev:api` uses the repo's `data/`),
`RAG_FORGE_EMBEDDER` (API, an `EmbedderSpec` as JSON to use another embedding model),
`RAG_FORGE_RERANKER` (a `RerankerSpec`), `RAG_FORGE_GENERATOR` (a `GeneratorSpec`),
`RAG_FORGE_OPENAI_BASE_URL` + `RAG_FORGE_OPENAI_MODEL` (+ optional `RAG_FORGE_OPENAI_API_KEY`) to register an
OpenAI-compatible endpoint. Models download once into `~/.cache/huggingface` on first use: the embedder (~133 MB)
at the first dense index build, the reranker (~90 MB) at the first reranked query, the generator (~790 MB) at the
first answer. No model weights are stored in the repository.

## Roadmap

1. ~~**Ingestion and corpus versioning.**~~ Done (Phase 1).
2. ~~**Lexical retrieval.**~~ Done (Phase 2): BM25 behind the `Retriever` contract; Retrieval Lab.
3. ~~**Dense retrieval.**~~ Done (Phase 3).
4. ~~**Hybrid fusion.**~~ Done (Phase 4): RRF and weighted fusion.
5. ~~**Cross-encoder reranking.**~~ Done (Phase 5).
6. ~~**Query intelligence and adaptive routing.**~~ Done (Phase 6): heuristic analyzer, rule policy, recorded decisions.
7. ~~**Evidence, grounding and generation.**~~ Done (Phase 7): evidence selection, local grounded generation, claims, measured support, Evidence Lab.
8. **Arena and experiment engine.** Judged datasets, runs, metrics and ablations; routed vs fixed pipelines measured.
9. **Replay and the research platform.** Reconstructing past runs from provenance; final presentation.
