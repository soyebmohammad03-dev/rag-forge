# experiments/

Reserved for exported run outputs. Git-ignored.

Arena runs are stored by the API itself: experiments, runs, per-case results and metrics in the
SQLite database, full per-case traces as content-addressed blobs (both under `RAG_FORGE_DATA_DIR`).
Each run is identified by its `ExperimentRun.id` and each arm by its configuration snapshot hash;
see [../docs/arena.md](../docs/arena.md). Replay and export are Phase 9.
