"""Resolve arms into immutable configuration snapshots, and describe how two arms differ.

A snapshot is resolved when the experiment is created: every component an arm names must be
registered (otherwise the experiment is rejected), and models are identified by spec, revision
and config hash without loading them.
"""

from __future__ import annotations

from typing import Any

from rag_forge.arena import metrics
from rag_forge.domain.arena import (
    Ablation,
    AblationSpec,
    Arm,
    BenchmarkDataset,
    ConfigChange,
    ConfigurationSnapshot,
    GeneratorIdentity,
    MetricSettings,
    PipelineKind,
)
from rag_forge.domain.models import Corpus, RetrievalMode
from rag_forge.rag.service import RagComponentNotAvailableError, RagService
from rag_forge.retrieval.service import (
    RerankerNotAvailableError,
    RetrievalService,
    StrategyNotAvailableError,
)
from rag_forge.router.service import RouterComponentNotAvailableError

ENGINE_VERSION = "arena-engine@1"


def snapshot(
    arm: Arm,
    dataset: BenchmarkDataset,
    corpus: Corpus,
    retrieval: RetrievalService,
    rag: RagService,
    settings: MetricSettings,
) -> ConfigurationSnapshot:
    t = arm.retrieval
    request = t.request("snapshot", dataset.corpus_version)
    adaptive = t.mode is RetrievalMode.ADAPTIVE
    analyzer = policy = routing_hash = None
    resolved = None
    if adaptive:
        if retrieval.router is None:
            raise RouterComponentNotAvailableError("router", "adaptive", [])
        identity = retrieval.router.identity(t.router)
        analyzer = identity.analyzer_version  # versions are already name@n
        policy = identity.policy_version
        routing_hash = identity.config_hash()
        # the router may choose dense retrieval and the default reranker for any case
        embedder = retrieval.embedder
        reranker = next(iter(retrieval.rerankers.values())).spec if retrieval.rerankers else None
    else:
        if t.strategy not in retrieval.retrievers:
            raise StrategyNotAvailableError(t.strategy, list(retrieval.retrievers))
        if t.rerank.enabled and t.rerank.model not in retrieval.rerankers:
            raise RerankerNotAvailableError(t.rerank.model, list(retrieval.rerankers))
        resolved = retrieval._configuration(request, corpus, dataset.corpus_version)
        embedder = resolved.embedder
        reranker = resolved.rerank.reranker if resolved.rerank else None

    generator = generation = verifier = verifier_hash = template = None
    evidence = arm.evidence_params()
    if arm.pipeline is PipelineKind.RAG:
        template = rag.template.id
        name = (arm.generation.generator if arm.generation else None) or rag.default_generator
        gen = rag.generators.get(name)
        if gen is None:
            raise RagComponentNotAvailableError("generator", name, list(rag.generators))
        d = gen.describe()  # the generator whose tokenizer measures the evidence budget
        generator = GeneratorIdentity(
            name=d.name,
            provider=d.provider,
            model=d.model,
            revision=d.revision,
            config_hash=d.config_hash,
        )
        if arm.generation is not None:
            generation = arm.generation.effective().model_copy(update={"generator": name})
            v = rag.verifiers.get(arm.grounding.verifier)
            if v is None:
                raise RagComponentNotAvailableError(
                    "verifier", arm.grounding.verifier, list(rag.verifiers)
                )
            verifier, verifier_hash = f"{v.name}@{v.version}", v.config_hash()

    return ConfigurationSnapshot(
        arm=arm.name,
        pipeline=arm.pipeline,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_hash=dataset.content_hash,
        corpus_id=dataset.corpus_id,
        corpus_version=dataset.corpus_version,
        chunking_hash=dataset.chunking_hash,
        retrieval_mode=t.mode,
        retrieval_template=t,
        retrieval=resolved,
        query_analyzer=analyzer,
        router_policy=policy,
        routing_hash=routing_hash,
        embedder=embedder,
        reranker=reranker,
        evidence=evidence,
        prompt_template=template,
        generator=generator,
        generation=generation,
        verifier=verifier,
        verifier_config_hash=verifier_hash,
        metrics=settings,
        metric_versions=metrics.versions(settings),
        engine_version=ENGINE_VERSION,
    )


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        out: dict[str, Any] = {}
        for k in sorted(value):
            out |= _flatten(value[k], f"{prefix}.{k}" if prefix else str(k))
        return out
    return {prefix: value}


def diff(a: Any, b: Any) -> list[ConfigChange]:
    fa, fb = _flatten(a), _flatten(b)
    return [
        ConfigChange(path=p, baseline=fa.get(p), variant=fb.get(p))
        for p in sorted(set(fa) | set(fb))
        if fa.get(p) != fb.get(p)
    ]


def snapshot_diff(a: ConfigurationSnapshot, b: ConfigurationSnapshot) -> list[ConfigChange]:
    return diff(
        a.model_dump(mode="json", exclude={"arm"}), b.model_dump(mode="json", exclude={"arm"})
    )


def factors(a: Arm, b: Arm) -> list[str]:
    """The arm settings that differ, at section level (retrieval.<field>, evidence, ...)."""
    da = a.model_dump(mode="json", exclude={"name", "label"})
    db = b.model_dump(mode="json", exclude={"name", "label"})
    out = []
    for key in sorted(set(da) | set(db)):
        x, y = da.get(key), db.get(key)
        if key == "retrieval" and isinstance(x, dict) and isinstance(y, dict):
            out += [f"retrieval.{k}" for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)]
        elif x != y:
            out.append(key)
    return out


def ablation(
    spec: AblationSpec, arms: dict[str, Arm], snapshots: dict[str, ConfigurationSnapshot]
) -> Ablation:
    changed = factors(arms[spec.baseline], arms[spec.variant])
    if not changed:
        raise ValueError(
            f"ablation {spec.baseline} vs {spec.variant}: the arms are identical, nothing varies"
        )
    return Ablation(
        baseline=spec.baseline,
        variant=spec.variant,
        factor=spec.factor,
        note=spec.note,
        factors=changed,
        single_factor=len(changed) == 1,
        changes=snapshot_diff(snapshots[spec.baseline], snapshots[spec.variant]),
    )
