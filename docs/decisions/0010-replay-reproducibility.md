# 0010: Replay, reproducibility and research exports

Date: 2026-10-07 · Status: accepted

## Decisions

1. **Replay re-executes; it does not re-read.** A replay runs each recorded case again through
   the same `ArenaEngine.execute` on the pinned corpus version and compares the result with the
   recording. Showing a stored trace is inspection, not replay.
2. **Stage hashes are the unit of comparison.** Every stage already records its output hash and
   whether it is deterministic. Replay compares them in order and reports the first difference;
   retrieval-only traces compute their chain with the same function the RAG chain uses.
3. **Four honest outcomes.** `exact` (every stage, generation included), `equivalent` (only a
   non-deterministic stage and what depends on it differ), `diverged` (a deterministic stage
   differs) and `not_replayable` (a component is unavailable or the arm resolves to a different
   configuration). Replaying a changed configuration would test something else, so it is refused
   per arm with the differing snapshot fields.
4. **Generation reproducibility is stated, not assumed.** Greedy local decoding is expected to be
   exact on the same runtime and hardware; sampling and remote endpoints are never treated as
   deterministic.
5. **Runs record their runtime.** Git commit (read at run creation, with a dirty flag), Node and uv
   versions, lockfile hashes, runtime packages, settings and the exact model files. Model file
   hashes come from the Hugging Face cache's LFS blob ids, so nothing is downloaded or re-hashed.
   New stored fields always have defaults, so older rows keep loading.
6. **Secrets are never recorded.** Settings whose names contain KEY, TOKEN, SECRET, PASSWORD or
   CREDENTIAL are stored as `<redacted>`; tests assert a key in the environment appears in no
   manifest, report or export.
7. **Exports restate, they do not interpret.** JSON, CSV and Markdown are rendered from recorded
   values. The report labels measured, automatic-proxy and interpretive content; its
   interpretation repeats the recorded paired conclusions only, and it generates no qualitative
   judgement.
8. **No planned surfaces in the product.** Areas without an implementation (Forge, Knowledge) and
   the sample preview data were removed rather than shown as placeholders; the overview shows the
   implemented pipeline only.
9. **Version 1.0.0** marks the platform complete for its scope; limitations are documented in
   `docs/limitations.md`.
