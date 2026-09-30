# 0001: Foundations

Date: 2026-09-30 · Status: accepted

## Decisions

1. **Monorepo with npm workspaces + a uv-managed Python app.** No Turborepo/Nx yet: two apps
   and one package don't need a task graph.
2. **API contracts are generated, not handwritten.** Pydantic → OpenAPI → `openapi-typescript`.
   The Python models are the single source of truth.
3. **Unimplemented endpoints return `501` with `NotImplementedDetail`.** They exist so the
   contract is visible and typed; they never return plausible fake output.
4. **In-memory metadata store** for Phase 0. It will be replaced by SQLite/Postgres when
   ingestion lands, keeping the store's method names.
5. **No component framework.** A small in-repo design system on Tailwind tokens; native
   `<dialog>` for dialogs and drawers; cmdk for the command palette. Radix or similar can be
   added when a primitive (e.g. combobox, popover positioning) is needed.
6. **No client data-fetching library yet.** A 40-line `useApi` hook. Adopt a query library
   when screens share and mutate server state.
7. **Dark-first only.** A light theme is deferred; tokens are centralised so it can be added in
   one place.

## Consequences

Data is lost on API restart until persistence lands. Tooltip positioning is CSS-only and will
need collision handling for dense screens.
