"""Second-stage reranking behind a small contract, with one local cross-encoder implementation.

A reranker re-scores an upstream candidate pool: it sees (query, passage) pairs, not the corpus,
so it works after any retriever without the retrievers knowing it exists. `rank_movement` turns
its scores into a final ranking plus per-candidate movement; it is pure, so the semantics are
testable without a model.

`OnnxCrossEncoder` runs a Hugging Face sequence-classification cross-encoder with a single logit
from its ONNX export (`onnx/model.onnx`), on CPU, at a pinned revision fetched into the Hugging
Face cache (never into the repository). The activation is read from the model's own config.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from rag_forge.domain.models import RankMovement, RerankDetail, RerankerInfo, RerankerSpec
from rag_forge.onnx_runtime import ort
from rag_forge.retrieval.base import Candidate


class RerankerUnavailableError(RuntimeError):
    """The model cannot be loaded or run (not cached and no network, bad revision, bad layout)."""


class Reranker(Protocol):
    name: str
    spec: RerankerSpec

    def info(self) -> RerankerInfo: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """One relevance score per passage, in input order; higher is more relevant. Batched."""
        ...


def rank_movement(
    candidates: Sequence[Candidate], scores: Sequence[float], final_top_k: int
) -> list[tuple[str, RerankDetail]]:
    """Rerank the whole pool by score, best first; ties keep their upstream order.

    original_rank is the candidate's 1-based position upstream, final_rank its position after
    reranking, and rank_delta = original_rank - final_rank (positive = promoted). Entering or
    leaving the top-k compares both ranks against `final_top_k`.
    """
    if len(scores) != len(candidates):
        raise RerankerUnavailableError(
            f"reranker returned {len(scores)} scores for {len(candidates)} candidates"
        )
    if any(not np.isfinite(s) for s in scores):
        raise RerankerUnavailableError("reranker returned a non-finite score")
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    out = []
    for final, i in enumerate(order, 1):
        original = i + 1
        delta = original - final
        movement = (
            RankMovement.PROMOTED
            if delta > 0
            else RankMovement.DEMOTED
            if delta < 0
            else RankMovement.UNCHANGED
        )
        detail = RerankDetail(
            original_rank=original,
            original_score=candidates[i].score,
            reranker_score=float(scores[i]),
            final_rank=final,
            rank_delta=delta,
            movement=movement,
            entered_top_k=final <= final_top_k < original,
            left_top_k=original <= final_top_k < final,
        )
        out.append((candidates[i].chunk_id, detail))
    return out


@dataclass(frozen=True)
class _Loaded:
    session: ort.InferenceSession
    tokenizer: Tokenizer
    inputs: frozenset[str]
    sigmoid: bool
    info: RerankerInfo


_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "onnx/model.onnx",
)
_TRUNCATION = "longest_first"  # sentence-transformers' CrossEncoder default


class OnnxCrossEncoder:
    name = "cross-encoder"

    def __init__(self, spec: RerankerSpec, cache_dir: Path | None = None) -> None:
        self.spec = spec
        self.cache_dir = cache_dir
        self._loaded: _Loaded | None = None
        self._lock = threading.Lock()

    def _load(self) -> _Loaded:
        with self._lock:  # load once per process; every later query reuses the session
            if self._loaded is None:
                self._loaded = self._load_files()
            return self._loaded

    def _load_files(self) -> _Loaded:
        spec = self.spec
        try:
            paths = {
                f: Path(
                    hf_hub_download(spec.model, f, revision=spec.revision, cache_dir=self.cache_dir)
                )
                for f in _FILES
            }
        except Exception as exc:  # network, auth, missing file or revision
            raise RerankerUnavailableError(
                f"reranker model {spec.model}@{spec.revision[:8]} is unavailable: {exc}"
            ) from exc

        def read(name: str) -> dict:  # type: ignore[type-arg]
            return json.loads(paths[name].read_text())  # type: ignore[no-any-return]

        config = read("config.json")
        if len(config.get("id2label", {"0": ""})) != 1:
            raise RerankerUnavailableError(f"{spec.model}: expected a single relevance logit")
        activation_fn = config.get("sbert_ce_default_activation_function") or ""
        if activation_fn.endswith("Identity"):
            activation = "identity"
        elif activation_fn.endswith("Sigmoid") or not activation_fn:  # sigmoid is the CE default
            activation = "sigmoid"
        else:
            raise RerankerUnavailableError(f"{spec.model}: unsupported activation {activation_fn}")
        model_max = min(
            int(read("tokenizer_config.json").get("model_max_length", 512)),
            int(config.get("max_position_embeddings", 512)),
        )
        max_len = spec.max_seq_length or model_max

        tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
        pad = read("special_tokens_map.json")["pad_token"]
        pad = pad["content"] if isinstance(pad, dict) else pad
        tokenizer.enable_truncation(max_len, strategy=_TRUNCATION)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id(pad), pad_token=pad)

        try:
            session = ort.InferenceSession(
                str(paths["onnx/model.onnx"]), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise RerankerUnavailableError(f"{spec.model}: cannot load ONNX model: {exc}") from exc
        info = RerankerInfo(
            spec=spec,
            config_hash=spec.config_hash(),
            scoring=f"cross-encoder pair logit ({activation})",
            activation=activation,
            max_seq_length=max_len,
            truncation=_TRUNCATION,
            weights_sha256=hashlib.sha256(paths["onnx/model.onnx"].read_bytes()).hexdigest(),
        )
        inputs = frozenset(i.name for i in session.get_inputs())
        return _Loaded(session, tokenizer, inputs, activation == "sigmoid", info)

    def info(self) -> RerankerInfo:
        return self._load().info

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        m = self._load()
        out: list[float] = []
        for start in range(0, len(passages), self.spec.batch_size):
            batch = [(query, p) for p in passages[start : start + self.spec.batch_size]]
            enc = m.tokenizer.encode_batch(batch)
            feeds = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
            }
            try:
                logits = m.session.run(None, {k: v for k, v in feeds.items() if k in m.inputs})[0]
            except Exception as exc:
                raise RerankerUnavailableError(f"{self.spec.model}: scoring failed: {exc}") from exc
            scores = np.asarray(logits, dtype=np.float64).reshape(-1)
            if m.sigmoid:
                scores = 1 / (1 + np.exp(-scores))
            # rounded like dense scores, so last-bit noise (e.g. padding) cannot reorder results
            out.extend(round(float(s), 6) for s in scores)
        return out
