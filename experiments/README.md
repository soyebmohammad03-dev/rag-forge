# experiments/

A place for files you export from runs (`report.md`, `export.json`, CSVs, manifests). Git-ignored.

Arena runs are stored by the API itself: experiments, runs, per-case results, metrics and replays
in the SQLite database, full per-case traces as content-addressed blobs (both under
`RAG_FORGE_DATA_DIR`). Each run is identified by its `ExperimentRun.id` and each arm by its
configuration snapshot hash; see [the Arena](../docs/experiments/README.md) and
[reproducibility](../docs/reproducibility/README.md).
