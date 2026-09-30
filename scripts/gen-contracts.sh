#!/usr/bin/env bash
# Regenerate TypeScript API contracts from the FastAPI OpenAPI schema.
# CI runs this and fails if packages/shared changes, so contracts never drift.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
(cd "$root/apps/api" && uv run python -c \
  "import json; from rag_forge.main import app; print(json.dumps(app.openapi(), indent=2))") \
  > "$root/packages/shared/openapi.json"
npx --prefix "$root/apps/web" openapi-typescript "$root/packages/shared/openapi.json" \
  -o "$root/packages/shared/src/api.ts"
