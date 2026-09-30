import createClient from "openapi-fetch";
import type { paths } from "@rag-forge/shared";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Typed client generated from the FastAPI OpenAPI schema (see scripts/gen-contracts.sh). */
export const api = createClient<paths>({
  baseUrl: API_URL,
  // Resolve fetch per call rather than capturing it at import, so it can be instrumented or stubbed.
  fetch: (request) => globalThis.fetch(request),
});
