# Replay

Code: `apps/api/src/rag_forge/arena/replay.py`, models in `domain/replay.py`.
UI: `/replay?run=…` → Replay.

A replay re-executes recorded cases of a finished run and reports, for every case × arm, how far
the result reproduced. It uses the same `ArenaEngine.execute` the run used, on the pinned corpus
version, with the models loaded in this process.

## Steps

1. **Check each arm** (`GET /runs/{id}/replayability`, also stored on the replay). The arm is
   resolved again in this environment and its snapshot hash compared with the recorded one. A
   component that is no longer registered (a generator, a reranker, the router), or a configuration
   that resolves differently (another model revision, a changed metric registry), makes the arm
   **not replayable**, with the reason and the differing snapshot fields. Replaying it anyway would
   test a different configuration.
2. **Re-execute** each requested case of each replayable arm (`POST /runs/{id}/replays`, body
   `{"arms": [...], "case_ids": [...]}`, both optional). The new trace is stored as an artifact that
   names the replay.
3. **Compare** the recorded and replayed provenance chains stage by stage (output hashes of query,
   query analysis, router decision, retrieval, reranking, evidence selection, context, generation,
   claims, grounding) and every non-operational metric value. Timings are not compared.
4. **Classify** each case:

| Outcome | Meaning |
|---|---|
| `exact` | Every recorded stage hash matched, generation included, and every quality metric is equal |
| `equivalent` | Same configuration; every deterministic stage matched; the first difference is a stage that is not deterministic (generation under sampling or a remote endpoint), so claims and grounding derived from its text may differ too |
| `diverged` | Same configuration, but a deterministic stage produced a different output (or a stage is missing, or metrics differ while hashes match). The first differing stage is reported |
| `not_replayable` | A component, model, corpus version or dense index is unavailable here, or the configuration resolves differently |

Recorded failures count too: a failure that reproduces with the same error type is `exact`; one
that does not reproduce is `diverged`.

Replays never claim more than they measured. Greedy local generation is expected to reproduce
exactly on the same runtime and hardware; the UI and the replayability check say that other CPUs
or onnxruntime builds may change the text. A remote OpenAI-compatible endpoint is never treated as
deterministic.

## Measured

Replays of real runs on the machine that recorded them (Apple Silicon, onnxruntime CPU), the
same API process:

| Run | Arms | Evaluations | Outcome |
|---|---|---|---|
| `run_25ad15b603e04574` (v1.0.0 release check) | BM25, dense, hybrid RRF, hybrid RRF + rerank, adaptive, grounded RAG (extractive), grounded RAG (Qwen2.5-0.5B, greedy) | 98 | 98 exact |
| test suite (`tests/test_replay.py`, every CI run) | BM25, adaptive, grounded RAG (local model), grounded RAG (extractive) | 12 | 12 exact (asserted) |

These show that the pipeline is deterministic where it claims to be, on one machine. They do not
show bit-identical output across hardware, which has not been measured.

## API

| Method | Path | |
|---|---|---|
| GET | `/runs/{id}/replayability` | Per-arm check; executes nothing |
| POST | `/runs/{id}/replays` | Queue a replay (202); `409` for unfinished runs, `422` for unknown arms or cases |
| GET | `/runs/{id}/replays`, `/replays`, `/replays/{id}` | Replays with per-case stage comparisons |
| GET | `/artifacts/{id}` | A recorded or replayed trace, with its computed stage chain |
