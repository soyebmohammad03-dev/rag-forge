"""Persistence for the Arena, in the same SQLite database as corpora (schema migrations 4-5).

Rows hold the models as JSON. Full per-case traces are content-addressed blobs referenced by
`Artifact`s, so identical traces are stored once.
"""

from __future__ import annotations

from rag_forge.domain.arena import (
    Artifact,
    BenchmarkDataset,
    Experiment,
    ExperimentRun,
    RunCase,
)
from rag_forge.domain.replay import Replay
from rag_forge.storage.blobs import BlobStore
from rag_forge.storage.sqlite import SqliteStore


class ArenaStore:
    def __init__(self, db: SqliteStore, blobs: BlobStore) -> None:
        self.db = db
        self.blobs = blobs

    # --- datasets ---------------------------------------------------------------------

    def add_dataset(self, ds: BenchmarkDataset) -> BenchmarkDataset:
        with self.db.transaction() as db:
            db.execute(
                "INSERT INTO benchmark_datasets VALUES (?, ?, ?, ?, ?)",
                (ds.id, ds.name, ds.version, ds.created_at.isoformat(), ds.model_dump_json()),
            )
        return ds

    def get_dataset(self, dataset_id: str) -> BenchmarkDataset | None:
        with self.db.transaction() as db:
            row = db.execute(
                "SELECT data FROM benchmark_datasets WHERE id = ?", (dataset_id,)
            ).fetchone()
        return BenchmarkDataset.model_validate_json(row[0]) if row else None

    def dataset_versions(self, name: str) -> list[BenchmarkDataset]:
        with self.db.transaction() as db:
            rows = db.execute(
                "SELECT data FROM benchmark_datasets WHERE name = ? ORDER BY version", (name,)
            ).fetchall()
        return [BenchmarkDataset.model_validate_json(r[0]) for r in rows]

    def list_datasets(self) -> list[BenchmarkDataset]:
        with self.db.transaction() as db:
            rows = db.execute(
                "SELECT data FROM benchmark_datasets ORDER BY name, version DESC"
            ).fetchall()
        return [BenchmarkDataset.model_validate_json(r[0]) for r in rows]

    # --- experiments ------------------------------------------------------------------

    def add_experiment(self, e: Experiment) -> Experiment:
        with self.db.transaction() as db:
            db.execute(
                "INSERT INTO arena_experiments VALUES (?, ?, ?)",
                (e.id, e.created_at.isoformat(), e.model_dump_json()),
            )
        return e

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        with self.db.transaction() as db:
            row = db.execute(
                "SELECT data FROM arena_experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        return Experiment.model_validate_json(row[0]) if row else None

    def list_experiments(self) -> list[Experiment]:
        with self.db.transaction() as db:
            rows = db.execute(
                "SELECT data FROM arena_experiments ORDER BY created_at DESC"
            ).fetchall()
        return [Experiment.model_validate_json(r[0]) for r in rows]

    # --- runs -------------------------------------------------------------------------

    def save_run(self, run: ExperimentRun) -> ExperimentRun:
        with self.db.transaction() as db:
            db.execute(
                "INSERT INTO arena_runs VALUES (?, ?, ?, ?) "
                "ON CONFLICT (id) DO UPDATE SET data = excluded.data",
                (run.id, run.experiment_id, run.created_at.isoformat(), run.model_dump_json()),
            )
        return run

    def get_run(self, run_id: str) -> ExperimentRun | None:
        with self.db.transaction() as db:
            row = db.execute("SELECT data FROM arena_runs WHERE id = ?", (run_id,)).fetchone()
        return ExperimentRun.model_validate_json(row[0]) if row else None

    def list_runs(self, experiment_id: str | None = None) -> list[ExperimentRun]:
        with self.db.transaction() as db:
            if experiment_id is None:
                rows = db.execute("SELECT data FROM arena_runs ORDER BY created_at DESC")
            else:
                rows = db.execute(
                    "SELECT data FROM arena_runs WHERE experiment_id = ? ORDER BY created_at DESC",
                    (experiment_id,),
                )
            return [ExperimentRun.model_validate_json(r[0]) for r in rows.fetchall()]

    def add_case(self, case: RunCase) -> None:
        with self.db.transaction() as db:
            db.execute(
                "INSERT INTO arena_run_cases VALUES (?, ?, ?, ?)",
                (case.run_id, case.arm, case.case_id, case.model_dump_json()),
            )

    def run_cases(
        self, run_id: str, arm: str | None = None, case_id: str | None = None
    ) -> list[RunCase]:
        sql, args = "SELECT data FROM arena_run_cases WHERE run_id = ?", [run_id]
        if arm is not None:
            sql, args = sql + " AND arm = ?", [*args, arm]
        if case_id is not None:
            sql, args = sql + " AND case_id = ?", [*args, case_id]
        with self.db.transaction() as db:
            rows = db.execute(sql, args).fetchall()
        return [RunCase.model_validate_json(r[0]) for r in rows]

    # --- artifacts --------------------------------------------------------------------

    def add_artifact(
        self,
        run_id: str,
        arm: str,
        case_id: str,
        kind: str,
        body: bytes,
        replay_id: str | None = None,
    ) -> Artifact:
        digest = self.blobs.put(body)
        art = Artifact(
            run_id=run_id,
            arm=arm,
            case_id=case_id,
            kind=kind,
            replay_id=replay_id,
            sha256=digest,
            bytes=len(body),
            uri=f"blob:sha256:{digest}",
        )
        with self.db.transaction() as db:
            db.execute(
                "INSERT INTO arena_artifacts VALUES (?, ?, ?)",
                (art.id, run_id, art.model_dump_json()),
            )
        return art

    def get_artifact(self, artifact_id: str) -> tuple[Artifact, bytes] | None:
        with self.db.transaction() as db:
            row = db.execute(
                "SELECT data FROM arena_artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        if row is None:
            return None
        art = Artifact.model_validate_json(row[0])
        return art, self.blobs.get(art.sha256)

    def run_artifacts(self, run_id: str) -> list[Artifact]:
        """Every trace recorded for a run, including those recorded by its replays."""
        with self.db.transaction() as db:
            rows = db.execute(
                "SELECT data FROM arena_artifacts WHERE run_id = ?", (run_id,)
            ).fetchall()
        return [Artifact.model_validate_json(r[0]) for r in rows]

    # --- replays ----------------------------------------------------------------------

    def save_replay(self, replay: Replay) -> Replay:
        with self.db.transaction() as db:
            db.execute(
                "INSERT INTO arena_replays VALUES (?, ?, ?, ?) "
                "ON CONFLICT (id) DO UPDATE SET data = excluded.data",
                (replay.id, replay.run_id, replay.created_at.isoformat(), replay.model_dump_json()),
            )
        return replay

    def get_replay(self, replay_id: str) -> Replay | None:
        with self.db.transaction() as db:
            row = db.execute("SELECT data FROM arena_replays WHERE id = ?", (replay_id,)).fetchone()
        return Replay.model_validate_json(row[0]) if row else None

    def list_replays(self, run_id: str | None = None) -> list[Replay]:
        with self.db.transaction() as db:
            if run_id is None:
                rows = db.execute("SELECT data FROM arena_replays ORDER BY created_at DESC")
            else:
                rows = db.execute(
                    "SELECT data FROM arena_replays WHERE run_id = ? ORDER BY created_at DESC",
                    (run_id,),
                )
            return [Replay.model_validate_json(r[0]) for r in rows.fetchall()]
