"""Text embeddings behind a small contract, with one local implementation.

`OnnxSentenceEmbedder` runs any Hugging Face model published in the sentence-transformers
layout that ships an ONNX export (`onnx/model.onnx`). Pooling and normalisation are read from
the model's own `modules.json` and `1_Pooling/config.json`, so nothing model-specific is hard-coded.
Files are fetched at a pinned revision into the Hugging Face cache (never into the repository).
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
from numpy.typing import NDArray
from tokenizers import Tokenizer

from rag_forge.domain.models import EmbedderInfo, EmbedderSpec
from rag_forge.onnx_runtime import ort

Vectors = NDArray[np.float32]


class EmbedderUnavailableError(RuntimeError):
    """The model cannot be loaded (not cached and no network, bad revision, unsupported layout)."""


class Embedder(Protocol):
    spec: EmbedderSpec

    def info(self) -> EmbedderInfo: ...
    def embed_documents(self, texts: Sequence[str]) -> Vectors: ...
    def embed_query(self, text: str) -> Vectors: ...


@dataclass(frozen=True)
class _Loaded:
    session: ort.InferenceSession
    tokenizer: Tokenizer
    inputs: frozenset[str]
    info: EmbedderInfo


_FILES = (
    "modules.json",
    "1_Pooling/config.json",
    "sentence_bert_config.json",
    "config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "onnx/model.onnx",
)


class OnnxSentenceEmbedder:
    def __init__(self, spec: EmbedderSpec, cache_dir: Path | None = None) -> None:
        self.spec = spec
        self.cache_dir = cache_dir
        self._loaded: _Loaded | None = None
        self._lock = threading.Lock()

    def _load(self) -> _Loaded:
        with self._lock:
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
            raise EmbedderUnavailableError(
                f"embedding model {spec.model}@{spec.revision[:8]} is unavailable: {exc}"
            ) from exc

        def read(name: str) -> dict:  # type: ignore[type-arg]
            return json.loads(paths[name].read_text())  # type: ignore[no-any-return]

        pooling_cfg = read("1_Pooling/config.json")
        if pooling_cfg.get("pooling_mode_cls_token"):
            pooling = "cls"
        elif pooling_cfg.get("pooling_mode_mean_tokens"):
            pooling = "mean"
        else:
            raise EmbedderUnavailableError(f"{spec.model}: unsupported pooling {pooling_cfg}")
        normalize = any(m["type"].endswith(".Normalize") for m in read("modules.json"))
        max_len = spec.max_seq_length or read("sentence_bert_config.json")["max_seq_length"]

        tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
        pad = read("special_tokens_map.json")["pad_token"]
        pad = pad["content"] if isinstance(pad, dict) else pad
        tokenizer.enable_truncation(max_len)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id(pad), pad_token=pad)

        session = ort.InferenceSession(
            str(paths["onnx/model.onnx"]), providers=["CPUExecutionProvider"]
        )
        info = EmbedderInfo(
            spec=spec,
            config_hash=spec.config_hash(),
            dimension=read("config.json")["hidden_size"],
            pooling=pooling,
            normalize=normalize,
            max_seq_length=max_len,
            weights_sha256=hashlib.sha256(paths["onnx/model.onnx"].read_bytes()).hexdigest(),
        )
        return _Loaded(session, tokenizer, frozenset(i.name for i in session.get_inputs()), info)

    def info(self) -> EmbedderInfo:
        return self._load().info

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        m = self._load()
        dim = m.info.dimension
        out = [np.empty((0, dim), dtype=np.float32)]
        for start in range(0, len(texts), self.spec.batch_size):
            enc = m.tokenizer.encode_batch(list(texts[start : start + self.spec.batch_size]))
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            hidden = m.session.run(None, {k: v for k, v in feeds.items() if k in m.inputs})[0]
            if m.info.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                weights = mask[..., None].astype(np.float32)
                pooled = (hidden * weights).sum(1) / np.clip(weights.sum(1), 1e-9, None)
            if m.info.normalize:
                pooled = pooled / np.clip(
                    np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None
                )
            out.append(pooled.astype(np.float32))
        return np.vstack(out)

    def embed_query(self, text: str) -> Vectors:
        return np.asarray(self.embed_documents([self.spec.query_prefix + text])[0], np.float32)
