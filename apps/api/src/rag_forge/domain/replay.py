"""Replay and reproducibility models.

A replay re-executes recorded Arena cases with the run's own arms against the pinned corpus
version, then compares every pipeline stage's output hash with the recorded trace. Deterministic
stages must match; generation is reported for what it is (deterministic only under greedy local
decoding). Nothing here claims a stage is reproducible without having re-executed and compared it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from rag_forge.domain.arena import ConfigChange
from rag_forge.domain.models import (
    EnvironmentSnapshot,
    Model,
    ModelRecord,
    RuntimeSnapshot,
    new_id,
    utcnow,
)

# --- replay -----------------------------------------------------------------------------------


class ReplayStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"  # the replay itself could not proceed


class ReplayOutcome(StrEnum):
    EXACT = "exact"  # every recorded stage hash, generation included, matched
    EQUIVALENT = "equivalent"  # same configuration, deterministic stages matched; a
    # non-deterministic stage (and what depends on it) differed
    DIVERGED = "diverged"  # same configuration, but a deterministic stage differed
    NOT_REPLAYABLE = "not_replayable"  # a component is unavailable or the configuration changed


class StageComparison(Model):
    stage: str
    deterministic: bool
    recorded: str | None
    replayed: str | None
    match: bool | None = Field(description="null when the stage is absent on one side")


class MetricDifference(Model):
    metric: str
    recorded: float | None
    replayed: float | None


class ArmReplayability(Model):
    arm: str
    recorded_hash: str
    current_hash: str | None = Field(description="The arm resolved now; null if it cannot be")
    replayable: bool
    reason: str | None
    changes: list[ConfigChange] = Field(description="Recorded vs current snapshot differences")
    generation: str | None = Field(
        description="How reproducible generation is for this arm, in words; null without one"
    )


class ReplayCase(Model):
    arm: str
    case_id: str
    outcome: ReplayOutcome
    reason: str
    recorded_status: str
    replayed_status: str | None
    stages: list[StageComparison]
    first_divergence: str | None
    recorded_artifact_id: str | None
    replayed_artifact_id: str | None
    metrics_compared: int = Field(description="Quality metrics compared (timings excluded)")
    metric_differences: list[MetricDifference]
    latency_ms: float | None


class ReplayRequest(Model):
    arms: list[str] | None = Field(default=None, description="Default: every arm")
    case_ids: list[str] | None = Field(default=None, description="Default: every case")


class Replay(Model):
    id: str = Field(default_factory=lambda: new_id("rpl"))
    run_id: str
    experiment_id: str
    status: ReplayStatus = ReplayStatus.QUEUED
    arms: list[str]
    case_ids: list[str]
    total: int
    completed: int = 0
    checks: list[ArmReplayability]
    cases: list[ReplayCase] = Field(default_factory=list)
    outcomes: dict[ReplayOutcome, int] = Field(default_factory=dict)
    environment: EnvironmentSnapshot
    runtime: RuntimeSnapshot
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


# --- manifest ---------------------------------------------------------------------------------


class ManifestArm(Model):
    arm: str
    pipeline: str
    config_hash: str
    retrieval_mode: str
    retrieval_configuration_hash: str | None = Field(description="Manual arms")
    routing_hash: str | None = Field(description="Adaptive arms")
    query_analyzer: str | None
    router_policy: str | None
    evidence_params_hash: str | None
    prompt_template: str | None
    prompt_template_hash: str | None
    generator: str | None
    generator_config_hash: str | None
    temperature: float | None
    seed: int | None
    verifier: str | None
    verifier_config_hash: str | None
    context_hashes: int = Field(description="Distinct context (prompt) hashes recorded")
    generation_deterministic: bool | None = Field(
        description="As recorded by the generator; null without generation"
    )


class ManifestArtifact(Model):
    arm: str
    case_id: str
    artifact_id: str
    kind: str
    sha256: str
    bytes: int


class ReproducibilityManifest(Model):
    manifest_version: str = "rag-forge-manifest@1"
    generated_at: datetime = Field(default_factory=utcnow)
    run: dict[str, object]
    dataset: dict[str, object]
    arms: list[ManifestArm]
    models: list[ModelRecord]
    seeds: dict[str, object]
    metrics: dict[str, object]
    statistics: dict[str, object]
    environment: EnvironmentSnapshot = Field(description="Recorded when the run was created")
    runtime: RuntimeSnapshot | None = Field(
        description="Recorded when the run was created; null for runs recorded before it existed"
    )
    current: dict[str, object] = Field(description="This process now, for comparison")
    artifacts: list[ManifestArtifact]
    replays: list[dict[str, object]]
    notes: list[str]
