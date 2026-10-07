# Experiment configuration

Code: `apps/api/src/rag_forge/arena/config.py`, `arena/presets.py`, models in `domain/arena.py`. Part of the [Arena](README.md).

## Configurations

An `Arm` is one configuration under test: a name, a pipeline (`retrieval` or `rag`), a
`RetrievalTemplate` (a retrieval request without query and corpus version: strategy, top-k, BM25,
hybrid, rerank, manual or adaptive mode, router), and for RAG arms evidence, generation and
grounding parameters.

At experiment creation each arm is resolved into an immutable `ConfigurationSnapshot`: dataset
id, version and hash; corpus version and chunking hash; the resolved retrieval configuration
(manual arms) or analyzer and router policy versions plus routing hash (adaptive arms); embedder
and reranker specs with revisions; evidence parameters; prompt template; generator identity
(name, provider, model, revision, config hash) and effective generation parameters; verifier
and its config hash; metric settings and metric versions; engine version (`arena-engine@1`).
Components that are not registered fail creation with 501 rather than at run time.

`config_hash` is the SHA-256 of the canonical snapshot excluding the arm's name, so two arms
with different names and identical behaviour have the same hash. Every `RunCase` stores it.

`GET /api/v1/arena/presets` lists templates (BM25, Dense, Hybrid RRF, Hybrid weighted, Adaptive,
three reranked arms, grounded RAG with the extractive baseline or the local model) and suggested
ablations. They are templates: nothing runs by default.

## Ablations

An `AblationSpec` names a baseline arm, a variant arm and the factor the researcher intends to
vary. The engine records:

- `factors`: the arm settings that actually differ (`retrieval.strategy`, `retrieval.rerank`,
  `generation`, `pipeline`, …);
- `single_factor`: whether exactly one differs;
- `changes`: every resolved snapshot field that differs, with both values.

A multi-factor ablation is kept and flagged as confounded, never silently relabelled. Identical
arms are rejected ("nothing varies").
