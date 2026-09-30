#!/usr/bin/env bash
# Regenerate TypeScript API contracts from the FastAPI OpenAPI schema.
# CI runs this and fails if packages/shared changes, so contracts never drift.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
(cd "$root/apps/api" && uv run python -c \
  "import json, pathlib, tempfile; from rag_forge.main import create_app; app = create_app(pathlib.Path(tempfile.mkdtemp())); print(json.dumps(app.openapi(), indent=2))") \
  > "$root/packages/shared/openapi.json"
npx --prefix "$root/apps/web" openapi-typescript "$root/packages/shared/openapi.json" \
  -o "$root/packages/shared/src/api.ts"
