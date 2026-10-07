# Contributing

Issues and pull requests are welcome. RAG FORGE is a research platform, so changes are held to
two rules beyond working code:

1. **No unmeasured claims.** Results shown in the UI, docs or reports must come from recorded
   runs. Do not add sample numbers, and do not describe a method as better without a paired
   comparison on a dataset that supports it.
2. **Provenance stays complete.** A new stage, model or parameter must appear in the relevant
   configuration hash and provenance chain, so runs that use it stay comparable and replayable.

## Setup

```bash
npm install
(cd apps/api && uv sync)
npm run dev:api   # http://localhost:8000
npm run dev:web   # http://localhost:3000
```

## Before opening a pull request

```bash
(cd apps/api && uv run ruff format --check src tests && uv run ruff check src tests && uv run mypy src tests && uv run pytest -q)
npm run contracts && git diff --exit-code packages/shared   # after any API schema change
npm run check                                               # web lint, typecheck, tests, build
```

CI runs the same commands. Backend tests use the real pinned models (downloaded once into the
Hugging Face cache, about 1 GB); nothing in the engine is mocked.

## Conventions

- Python: strict mypy, frozen Pydantic models that reject unknown fields, forward-only SQLite
  migrations (add one; never edit a released one). New fields on stored models need defaults.
- TypeScript: types come from `packages/shared` (generated); do not hand-write API types.
- Record a significant design choice as a decision in `docs/decisions/`.
- Commit messages follow Conventional Commits (`feat(scope): …`, `fix(scope): …`).
