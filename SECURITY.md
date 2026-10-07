# Security

RAG FORGE is a single-user research tool meant to run locally. The API has no authentication;
do not expose it on a public network.

## Secrets

- The only secret it reads is the optional `RAG_FORGE_OPENAI_API_KEY` for an OpenAI-compatible
  endpoint. It is sent as a bearer token and never stored.
- Reproducibility manifests, reports and exports record settings with the values of any variable
  whose name contains KEY, TOKEN, SECRET, PASSWORD or CREDENTIAL replaced by `<redacted>`.
- Keep `.env` files out of version control (they are git-ignored).

## Reporting a vulnerability

Please report security issues privately through GitHub's
[security advisories](https://github.com/soyebmohammad03-dev/rag-forge/security/advisories/new)
rather than a public issue. Include steps to reproduce and the affected commit.
