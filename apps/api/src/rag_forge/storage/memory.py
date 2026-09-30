"""Process-local metadata store.

ponytail: in-memory dicts, data is lost on restart. Replace with a SQL-backed store
(SQLite locally, Postgres for shared experimentation) when ingestion lands; keep the
method names so routes don't change.
"""

from __future__ import annotations

from rag_forge.domain.models import Corpus, Experiment, ExperimentRun


class InMemoryStore:
    kind = "in-memory"

    def __init__(self) -> None:
        self.corpora: dict[str, Corpus] = {}
        self.experiments: dict[str, Experiment] = {}
        self.runs: dict[str, ExperimentRun] = {}

    def add_corpus(self, corpus: Corpus) -> Corpus:
        self.corpora[corpus.id] = corpus
        return corpus

    def add_experiment(self, experiment: Experiment) -> Experiment:
        self.experiments[experiment.id] = experiment
        return experiment
