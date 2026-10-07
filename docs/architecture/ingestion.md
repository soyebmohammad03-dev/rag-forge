# Ingestion and corpus versioning

Code: `apps/api/src/rag_forge/ingestion/` and `apps/api/src/rag_forge/storage/`.

## Flow

```
upload batch ──▶ per file: name → SHA-256 → size/empty checks
                           │
                           ├─ same name, same hash as current version ─▶ unchanged
                           ├─ same hash as another current document ──▶ duplicate (not ingested)
                           ├─ extractor refuses (type, corrupt, empty) ─▶ rejected + reason
                           └─ extract text → chunk → DocumentVersion ──▶ added / modified
                                                     │
          any added/modified? ── yes ──▶ one transaction: documents, document versions,
                  │                      extracted text, chunks, corpus version, membership,
                  │                      ingestion record, corpus.version += 1
                  └── no ──▶ ingestion record only (status no_change / failed)
```

Removing a document is the same operation with a single `removed` outcome.

## Identity

| Thing | Identity | Notes |
|---|---|---|
| Document | `(corpus, filename)` → stable `doc_…` id | Directories in upload names are stripped. Re-adding a removed filename resumes the same document. |
| Document version | `dv_…`, numbered 1, 2, … per document | Holds SHA-256 of the original bytes, size, media type, parser, extraction status/warnings, extracted-text length, parser metadata. |
| Original bytes | SHA-256 | Stored once under `data/blobs/<first 2 hex>/<hash>`; written via temp file + rename. |
| Chunk | `chk_` + SHA-256(`version id | chunking hash | ordinal`) | Deterministic: same version and config always yield the same ids. |
| Corpus version | `(corpus, n)` | v0 is the empty corpus at creation. Stores counts of added/modified/removed/unchanged. |
| Ingestion | `ing_…` | Immutable provenance record per operation, including no-ops and failures. |

## Versioning rules

- A corpus version is the set of `(document → document version)` pairs in `corpus_version_members`.
  Diffs between versions are derived from membership, never stored separately.
- `document_versions`, `document_texts`, `chunks`, `corpus_versions`, `corpus_version_members` and
  `ingestions` are append-only. SQLite triggers abort any `UPDATE` or `DELETE` on them.
- A batch produces at most one new corpus version. If nothing was added, modified or removed, no
  version is created, but the ingestion is still recorded.
- Commits are atomic and use an optimistic check on `corpus.version`; within one process a lock
  serialises ingestion per API instance.

## Extraction

| Format | Extensions | Extractor | Behaviour |
|---|---|---|---|
| Plain text | `.txt`, `.text` | `plaintext@1` | Strict UTF-8 (BOM allowed); NUL bytes → rejected as binary; newlines normalised to `\n`. |
| Markdown | `.md`, `.markdown` | `markdown@1` | Source kept verbatim; records first `#` title and heading count. |
| PDF | `.pdf` | `pypdf@<version>` | Requires `%PDF-` header; encrypted/corrupt → rejected. Pages joined with blank lines; page start offsets kept so chunks record `page_start`/`page_end`. Pages without text → `partial` with a warning; no text at all → rejected (OCR not supported). |

Also rejected: empty files, files over 25 MiB, unknown extensions, and a filename repeated within
one batch. Every rejection carries a human-readable reason in the ingestion record.

Adding a format means adding one class with `name`, `version`, `media_type`, `extensions` and
`extract(bytes) -> Extraction` to `EXTRACTORS`. `Extraction` already carries page offsets,
warnings and metadata, which is what DOCX, HTML, OCR and table extractors will need.

## Chunking

Chunking is configured per corpus (`ChunkingConfig`: `strategy`, `chunk_size`, `chunk_overlap`,
in characters) and fixed for the corpus's lifetime, so every version is chunked the same way.

- **`recursive`** (default baseline): each chunk extends up to `chunk_size` and ends at the
  coarsest boundary that keeps it at least half full: paragraph, line, sentence (`. ? !`),
  clause (`; ,`), word, else a hard cut. A chunk never ends on a Markdown heading line; the
  heading moves to the next chunk with its section. The next chunk starts inside the previous
  one's last `chunk_overlap` characters, at the first sentence or word boundary, so overlap never
  starts mid-word.
- **`fixed`**: fixed character windows with step `chunk_size − chunk_overlap`. The naive baseline
  for chunking ablations.

Invariants (tested): `text[char_start:char_end] == chunk.text`; chunks are non-empty, trimmed,
at most `chunk_size`; every non-whitespace character is covered; starts strictly increase.

`chunking_hash` = SHA-256 of the config plus `CHUNKER_VERSION`. Changing chunk boundaries for an
existing config requires bumping `CHUNKER_VERSION`, so old and new chunks can never be confused.
Token counts are not recorded yet (no tokenizer is chosen); chunks record word counts.

## Provenance

Each `IngestionRecord` stores: corpus, versions before/after, status, every file's outcome
(filename, size, SHA-256, media type, parser, document and version ids, chunk count, error,
warnings, duplicate-of), the chunking config and hash, timestamps, and the environment snapshot
(rag-forge version, git commit, Python, platform, package versions).

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/ingestion/formats` | Supported extractors |
| GET, POST | `/api/v1/corpora` | List (with stats) / create (409 on duplicate name) |
| GET | `/api/v1/corpora/{id}` | Corpus + stats + last ingestion |
| GET | `/api/v1/corpora/{id}/documents?version=` | Documents in the current or a historical version |
| POST | `/api/v1/corpora/{id}/documents` | Multipart upload (`files`), returns the ingestion record |
| GET | `/api/v1/corpora/{id}/documents/{doc}` | Document with all its versions |
| DELETE | `/api/v1/corpora/{id}/documents/{doc}` | Remove from the next corpus version |
| GET | `/api/v1/document-versions/{dv}/chunks?offset&limit` | Chunks under the corpus's chunking config |
| GET | `/api/v1/corpora/{id}/versions` | Corpus versions, newest first |
| GET | `/api/v1/corpora/{id}/versions/{n}/changes` | Per-document diff against version n − 1 |
| GET | `/api/v1/corpora/{id}/ingestions` | Ingestion provenance, newest first |

## Known limits

- Ingestion runs inside the request. Large batches block that request; background jobs come
  when corpora get large.
- Membership is copied in full per version (fine to ~10⁵ documents × tens of versions).
- No re-chunking of existing versions under a new config or `CHUNKER_VERSION`. A corpus's
  config is fixed at creation.
- No corpus deletion or renaming.
