# 0002: Ingestion, storage and corpus versioning

Date: 2026-09-30 · Status: accepted · Supersedes 0001 item 4 (in-memory store)

## Decisions

1. **SQLite through stdlib `sqlite3`, no ORM.** Tables hold lookup columns plus the full Pydantic
   model as JSON, so models can evolve without migrations for every field. A `CorpusStore`
   protocol is the contract services use; Postgres will be a second implementation of it.
   `PRAGMA user_version` tracks the schema version.
2. **History is append-only and the database enforces it.** Triggers abort updates and deletes
   on document versions, texts, chunks, corpus versions, membership and ingestion records. The
   only mutable rows are a corpus's current version pointer and document identities.
3. **Corpus version = membership snapshot.** Each version stores the full
   `document → document version` map. Diffs are derived by comparing adjacent snapshots.
   Simpler and more robust than storing deltas; revisit at ~10⁵ documents.
4. **Filename is document identity within a corpus.** A re-upload with the same name and new
   content is a new version of the same document; identical content under another name is a
   duplicate and is not ingested, so evaluation never double-counts a passage.
5. **Keep original bytes, content-addressed.** Blobs live under `data/blobs/` keyed by SHA-256,
   so extraction can be re-run with better parsers without re-uploading.
6. **Store extracted text per document version.** Re-chunking experiments can run without
   re-parsing.
7. **Chunking config is fixed per corpus and hashed with an algorithm version.** A corpus version
   therefore identifies its chunks exactly. Comparing chunkers means comparing corpora (or, later,
   re-chunking into a new corpus), which is explicit and reproducible.
8. **Reject, never guess.** Unsupported, empty, oversize, non-UTF-8, encrypted, corrupt or
   text-less files are rejected with a reason recorded in provenance. Partially extractable PDFs
   are accepted but flagged `partial` with per-page warnings.
9. **pypdf for PDF.** Pure Python, no system packages, adequate text extraction. Layout-aware or
   OCR extraction is a future extractor, not a replacement.
10. **Synchronous ingestion.** One request = one batch = at most one version. A background job
    queue is deferred until batch sizes demand it.
11. **Upload progress via XHR in the web client.** `fetch` cannot report upload progress; this one
    call uses `XMLHttpRequest` but still returns the generated `IngestionRecord` type.

## Consequences

- Data now survives restarts (`RAG_FORGE_DATA_DIR`, default `./data`).
- The app is created with `create_app()` (`uvicorn --factory`); there is no module-level `app`.
- Changing chunker behaviour requires bumping `CHUNKER_VERSION`; existing corpora then show no
  chunks under the new hash until a re-chunk operation exists.
