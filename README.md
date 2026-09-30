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

**Phase 0, foundation.** What exists and works today:

| Area | State |
|---|---|
| Domain models (Corpus → ProvenanceRecord: 18 models, 4 enums) | Implemented, tested |
| Retrieval metrics: Recall@K, Precision@K, MRR, nDCG@K | Implemented, tested, exposed at `POST /api/v1/evaluation/retrieval` |
| Configuration hashing (identity of a run's behaviour) | Implemented, tested |
| Environment snapshot for provenance | Implemented, `GET /api/v1/provenance/environment` |
| Health, corpora, experiments, runs endpoints | Implemented (in-memory store) |
| Ingestion, retrieval, router endpoints | Contract only; return `501` with a structured body |
| Web shell, design system, Overview and System screens | Implemented |
| Other product areas (Forge, Router, Retrieval Lab, Evidence, Arena, …) | Scoped, contracts linked, not built |

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
│   │   │   ├── evaluation/   retrieval metrics
│   │   │   ├── provenance/   environment capture
│   │   │   ├── storage/      metadata store (in-memory for now)
│   │   │   ├── api/          HTTP schemas + routes
│   │   │   └── main.py       app factory
│   │   └── tests/
│   └── web/                  Next.js 16 (App Router) + TypeScript + Tailwind v4
│       └── src/
│           ├── app/          routes only; thin
│           ├── components/   ui/ (design system) · shell/ (sidebar, command palette)
│           ├── features/     overview/ · system/ · planned/
│           ├── lib/          typed API client, area registry
│           └── sample/       preview data, isolated
├── packages/shared/          API contracts generated from OpenAPI
├── scripts/gen-contracts.sh  regenerates packages/shared
├── docs/                     architecture, research methodology, decisions
├── data/  experiments/       local corpora and run outputs (git-ignored)
└── .github/workflows/ci.yml
```

See [docs/architecture.md](docs/architecture.md) for layering, and
[docs/research/methodology.md](docs/research/methodology.md) for how comparisons will be run.

## Technology

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI, Pydantic v2, uvicorn | Typed contracts that generate OpenAPI |
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
`RAG_FORGE_CORS_ORIGINS` (API, comma-separated, default `http://localhost:3000`).

## Roadmap

1. **Ingestion and corpus versioning.** Persistent store, document versions, a first chunker.
2. **Baseline retrievers.** BM25 and one dense retriever behind a common interface; Retrieval Lab.
3. **Arena v1.** Standard benchmark loaders (e.g. BEIR subsets), runs, measured metrics, Results.
4. **Router v0.** Rule-based policy over query features, recorded decisions, head-to-head against fixed pipelines.
5. **Generation and evidence.** Cited answers, claim-level verification, faithfulness.
