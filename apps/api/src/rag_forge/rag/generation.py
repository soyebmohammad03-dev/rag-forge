"""Answer generation behind a provider-agnostic contract, with three implementations.

- `OnnxCausalLM` (default): a decoder-only chat model's ONNX export run on CPU with onnxruntime,
  with a KV cache and greedy or seeded sampling. Files are fetched at a pinned revision into
  the Hugging Face cache (never into the repository) and loaded once per process.
- `ExtractiveGenerator`: a model-free baseline that copies the evidence sentences sharing the
  most query terms, each cited. Deterministic and instant; a floor that any model should beat.
- `OpenAICompatibleGenerator`: any `/chat/completions` endpoint (a local llama.cpp or Ollama
  server, or a hosted API). Optional, registered only when configured; never required.

The pipeline only sees `Generator`: `info()`, `count_tokens()` and `generate()`.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from tokenizers import Tokenizer

from rag_forge.domain.models import (
    ChatMessage,
    ContextBlock,
    FinishReason,
    GenerationParams,
    GeneratorDescriptor,
    GeneratorInfo,
    GeneratorSpec,
)
from rag_forge.onnx_runtime import ort
from rag_forge.rag.evidence import ContextBudgetError
from rag_forge.rag.prompt import INSUFFICIENT
from rag_forge.rag.text import content_terms, sentence_spans, terms


class GeneratorUnavailableError(RuntimeError):
    """The generator cannot load or run (not cached and offline, bad layout, endpoint down)."""


@dataclass(frozen=True)
class GenerationInput:
    messages: list[ChatMessage]
    question: str
    blocks: list[ContextBlock]
    params: GenerationParams


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    finish_reason: FinishReason
    prompt_tokens: int | None
    completion_tokens: int | None
    load_ms: float
    latency_ms: float


class Generator(Protocol):
    name: str

    @property
    def tokenizer_id(self) -> str:
        """Identifies how `count_tokens` counts; recorded with every token budget."""
        ...

    def describe(self) -> GeneratorDescriptor:
        """Never loads a model."""
        ...

    def info(self) -> GeneratorInfo:
        """Loads the model if needed. Raises GeneratorUnavailableError."""
        ...

    def count_tokens(self, text: str) -> int: ...

    def generate(self, inp: GenerationInput) -> GenerationOutput: ...


# --- Local ONNX causal LM ------------------------------------------------------------------


def render_chatml(messages: Sequence[ChatMessage]) -> str:
    """ChatML (Qwen2/2.5, SmolLM2 instruct models), ending with an open assistant turn."""
    turns = "".join(f"<|im_start|>{m.role.value}\n{m.content}<|im_end|>\n" for m in messages)
    return turns + "<|im_start|>assistant\n"


CHAT_TEMPLATES = {"chatml": (render_chatml, "<|im_start|>", "<|im_end|>")}


@dataclass(frozen=True)
class _LoadedLM:
    session: ort.InferenceSession
    inputs: frozenset[str]
    outputs: list[str]
    kv_dtype: type[np.floating[Any]]
    layers: int
    kv_heads: int
    head_dim: int
    eos: frozenset[int]
    info: GeneratorInfo


_LM_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


def _sample(logits: NDArray[np.float32], params: GenerationParams, rng: np.random.Generator) -> int:
    if params.temperature == 0:
        return int(np.argmax(logits))
    z = logits.astype(np.float64) / params.temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    order = np.argsort(-p, kind="stable")
    keep = order[: int(np.searchsorted(np.cumsum(p[order]), params.top_p) + 1)]
    q = p[keep] / p[keep].sum()
    return int(keep[rng.choice(len(keep), p=q)])


class OnnxCausalLM:
    """Greedy (temperature 0) or seeded nucleus sampling over an ONNX decoder with a KV cache.

    Expects the transformers.js / optimum export layout: inputs `input_ids`, `attention_mask`,
    optional `position_ids` and `past_key_values.{i}.key|value`; outputs `logits` and
    `present.{i}.key|value`. Generation is serialised per instance (one model, one session).
    """

    def __init__(self, spec: GeneratorSpec, cache_dir: Path | None = None) -> None:
        if spec.chat_template not in CHAT_TEMPLATES:
            raise ValueError(f"unsupported chat template {spec.chat_template!r}")
        self.spec = spec
        self.name = spec.model
        self.cache_dir = cache_dir
        self._tokenizer: Tokenizer | None = None
        self._loaded: _LoadedLM | None = None
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def tokenizer_id(self) -> str:
        return f"{self.spec.model}@{self.spec.revision[:8]}"

    def _fetch(self, filename: str) -> Path:
        spec = self.spec
        try:
            return Path(
                hf_hub_download(
                    spec.model, filename, revision=spec.revision, cache_dir=self.cache_dir
                )
            )
        except Exception as exc:  # network, auth, missing file or revision
            raise GeneratorUnavailableError(
                f"generator model {spec.model}@{spec.revision[:8]} is unavailable: {exc}"
            ) from exc

    def tokenizer(self) -> Tokenizer:
        """Only the tokenizer (a few MB): token budgets never load the model weights."""
        with self._lock:
            if self._tokenizer is None:
                self._tokenizer = Tokenizer.from_file(str(self._fetch("tokenizer.json")))
            return self._tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer().encode(text, add_special_tokens=False).ids)

    def _load(self) -> tuple[_LoadedLM, float]:
        tok = self.tokenizer()
        with self._lock:
            if self._loaded is not None:
                return self._loaded, 0.0
            started = time.perf_counter()
            self._loaded = self._load_files(tok)
            return self._loaded, round((time.perf_counter() - started) * 1000, 3)

    def _load_files(self, tok: Tokenizer) -> _LoadedLM:
        spec = self.spec
        paths = {f: self._fetch(f) for f in (*_LM_FILES, spec.weights_file)}
        config = json.loads(paths["config.json"].read_text())
        template = json.loads(paths["tokenizer_config.json"].read_text()).get("chat_template", "")
        _, start_marker, end_marker = CHAT_TEMPLATES[spec.chat_template]
        if start_marker not in str(template) or tok.token_to_id(end_marker) is None:
            raise GeneratorUnavailableError(
                f"{spec.model}: its chat template is not {spec.chat_template}"
            )
        try:
            session = ort.InferenceSession(
                str(paths[spec.weights_file]), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise GeneratorUnavailableError(f"{spec.model}: cannot load ONNX model: {exc}") from exc
        inputs = {i.name: i for i in session.get_inputs()}
        outputs = [o.name for o in session.get_outputs()]
        if "input_ids" not in inputs or not outputs or outputs[0] != "logits":
            raise GeneratorUnavailableError(f"{spec.model}: not a causal-LM ONNX export")
        layers = int(config["num_hidden_layers"])
        heads = int(config["num_attention_heads"])
        kv_heads = int(config.get("num_key_value_heads", heads))
        head_dim = int(config.get("head_dim") or config["hidden_size"] // heads)
        kv_type = inputs.get("past_key_values.0.key")
        if kv_type is None:
            raise GeneratorUnavailableError(f"{spec.model}: export has no KV-cache inputs")
        kv_dtype: type[np.floating[Any]] = np.float16 if "float16" in kv_type.type else np.float32
        eos_cfg = config.get("eos_token_id")
        eos = set(
            eos_cfg if isinstance(eos_cfg, list) else [eos_cfg] if eos_cfg is not None else []
        )
        eos.add(tok.token_to_id(end_marker))
        max_ctx = int(config.get("max_position_embeddings", 2048))
        info = GeneratorInfo(
            name=self.name,
            provider=spec.provider,
            model=spec.model,
            revision=spec.revision,
            config_hash=spec.config_hash(),
            local=True,
            deterministic_at_zero_temperature=True,
            max_context_tokens=max_ctx,
            tokenizer=self.tokenizer_id,
            weights_sha256=hashlib.sha256(paths[spec.weights_file].read_bytes()).hexdigest(),
            details={
                "weights_file": spec.weights_file,
                "chat_template": spec.chat_template,
                "kv_dtype": np.dtype(kv_dtype).name,
                "layers": layers,
                "execution_provider": "CPUExecutionProvider",
                "onnxruntime": ort.__version__,
            },
        )
        return _LoadedLM(
            session=session,
            inputs=frozenset(inputs),
            outputs=outputs,
            kv_dtype=kv_dtype,
            layers=layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
            eos=frozenset(int(e) for e in eos),
            info=info,
        )

    def describe(self) -> GeneratorDescriptor:
        spec = self.spec
        return GeneratorDescriptor(
            name=self.name,
            provider=spec.provider,
            model=spec.model,
            revision=spec.revision,
            config_hash=spec.config_hash(),
            local=True,
            loaded=self._loaded is not None,
        )

    def info(self) -> GeneratorInfo:
        return self._load()[0].info

    def generate(self, inp: GenerationInput) -> GenerationOutput:
        m, load_ms = self._load()
        tok = self.tokenizer()
        render = CHAT_TEMPLATES[self.spec.chat_template][0]
        ids = tok.encode(render(inp.messages), add_special_tokens=False).ids
        params = inp.params
        limit = m.info.max_context_tokens or 0
        if len(ids) + params.max_new_tokens > limit:
            raise ContextBudgetError(
                f"prompt ({len(ids)} tokens) + max_new_tokens ({params.max_new_tokens}) exceeds "
                f"the model's context of {limit} tokens"
            )
        rng = np.random.default_rng(params.seed)
        past = {
            f"past_key_values.{i}.{kv}": np.zeros((1, m.kv_heads, 0, m.head_dim), m.kv_dtype)
            for i in range(m.layers)
            for kv in ("key", "value")
        }
        step = np.array([ids], dtype=np.int64)
        total = len(ids)
        out: list[int] = []
        finish = FinishReason.LENGTH
        with self._run_lock:
            started = time.perf_counter()
            for _ in range(params.max_new_tokens):
                feeds: dict[str, NDArray[Any]] = {
                    "input_ids": step,
                    "attention_mask": np.ones((1, total), dtype=np.int64),
                    **past,
                }
                if "position_ids" in m.inputs:
                    feeds["position_ids"] = np.arange(total - step.shape[1], total)[None]
                try:
                    results = m.session.run(None, feeds)
                except Exception as exc:
                    raise GeneratorUnavailableError(
                        f"{self.spec.model}: generation failed: {exc}"
                    ) from exc
                for name, value in zip(m.outputs[1:], results[1:], strict=True):
                    past[name.replace("present", "past_key_values", 1)] = value
                token = _sample(results[0][0, -1], params, rng)
                if token in m.eos:
                    finish = FinishReason.STOP
                    break
                out.append(token)
                step = np.array([[token]], dtype=np.int64)
                total += 1
            latency = round((time.perf_counter() - started) * 1000, 3)
        return GenerationOutput(
            text=tok.decode(out, skip_special_tokens=True).strip(),
            finish_reason=finish,
            prompt_tokens=len(ids),
            completion_tokens=len(out),
            load_ms=load_ms,
            latency_ms=latency,
        )


# --- Extractive baseline -------------------------------------------------------------------

_WORDISH = re.compile(r"\w+|[^\w\s]")


class ExtractiveGenerator:
    """Copies, from each passage in evidence order, its sentence sharing most query terms.

    Overlap = share of the question's content terms in the sentence; a passage contributes its
    best sentence (earliest on ties) if the overlap is above zero, up to `max_sentences`
    passages. Each sentence is cited with its passage. With no overlapping sentence it
    abstains. It never paraphrases, so every sentence it writes is verbatim evidence; it
    inherits retrieval's ordering, so it is a floor that shows what the evidence alone says.
    """

    name = "extractive-baseline"
    version = 1

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    @property
    def tokenizer_id(self) -> str:
        return "regex-word-punct@1"

    def count_tokens(self, text: str) -> int:
        return len(_WORDISH.findall(text))

    def describe(self) -> GeneratorDescriptor:
        i = self.info()
        return GeneratorDescriptor(
            name=i.name,
            provider=i.provider,
            model=i.model,
            revision=None,
            config_hash=i.config_hash,
            local=True,
            loaded=True,
        )

    def info(self) -> GeneratorInfo:
        config = {"name": self.name, "version": self.version, "max_sentences": self.max_sentences}
        return GeneratorInfo(
            name=self.name,
            provider="extractive",
            model=f"{self.name}@{self.version}",
            revision=None,
            config_hash=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            local=True,
            deterministic_at_zero_temperature=True,
            max_context_tokens=None,
            tokenizer=self.tokenizer_id,
            weights_sha256=None,
            details={"max_sentences": self.max_sentences, "scoring": "query content-term overlap"},
        )

    def generate(self, inp: GenerationInput) -> GenerationOutput:
        started = time.perf_counter()
        query = set(content_terms(inp.question))
        lines = []
        for block in inp.blocks:
            scored = (
                [
                    (len(query & terms(block.text[a:b])) / len(query), -a, block.text[a:b])
                    for a, b in sentence_spans(block.text)
                ]
                if query
                else []
            )
            best = max(scored, default=None)
            if best is not None and best[0] > 0:
                lines.append(f"{best[2]} [{block.citation}]")
            if len(lines) == self.max_sentences:
                break
        text = "\n".join(lines) if lines else INSUFFICIENT
        tokens = self.count_tokens(text)
        cut = tokens > inp.params.max_new_tokens
        if cut:  # honour the budget like a model would: stop mid-answer and say so
            text = " ".join(_WORDISH.findall(text)[: inp.params.max_new_tokens])
        return GenerationOutput(
            text=text,
            finish_reason=FinishReason.LENGTH if cut else FinishReason.STOP,
            prompt_tokens=None,
            completion_tokens=min(tokens, inp.params.max_new_tokens),
            load_ms=0.0,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )


# --- OpenAI-compatible HTTP endpoint -------------------------------------------------------


class OpenAICompatibleGenerator:
    """POSTs the rendered messages to `{base_url}/chat/completions`. Never deterministic by
    contract: the remote model and its revision are outside this process's control."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        name: str = "openai-compatible",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key  # never recorded in provenance
        self.name = name
        self.timeout_s = timeout_s

    @property
    def tokenizer_id(self) -> str:
        return "approx-chars-per-4@1"  # the endpoint's tokenizer is unknown here

    def count_tokens(self, text: str) -> int:
        return -(-len(text) // 4)

    def describe(self) -> GeneratorDescriptor:
        return GeneratorDescriptor(
            name=self.name,
            provider="openai-compatible",
            model=self.model,
            revision=None,
            config_hash=self.info().config_hash,
            local=False,
            loaded=True,
        )

    def info(self) -> GeneratorInfo:
        config = {"base_url": self.base_url, "model": self.model}
        return GeneratorInfo(
            name=self.name,
            provider="openai-compatible",
            model=self.model,
            revision=None,
            config_hash=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            local=False,
            deterministic_at_zero_temperature=False,
            max_context_tokens=None,
            tokenizer=self.tokenizer_id,
            weights_sha256=None,
            details={"base_url": self.base_url},
        )

    def generate(self, inp: GenerationInput) -> GenerationOutput:
        p = inp.params
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in inp.messages],
            "temperature": p.temperature,
            "top_p": p.top_p,
            "max_tokens": p.max_new_tokens,
            "seed": p.seed,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise GeneratorUnavailableError(f"{self.name} at {self.base_url}: {exc}") from exc
        latency = round((time.perf_counter() - started) * 1000, 3)
        try:
            choice = payload["choices"][0]
            text = str(choice["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise GeneratorUnavailableError(f"{self.name}: malformed response: {exc}") from exc
        usage = payload.get("usage") or {}
        return GenerationOutput(
            text=text.strip(),
            finish_reason=(
                FinishReason.LENGTH
                if choice.get("finish_reason") == "length"
                else FinishReason.STOP
            ),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            load_ms=0.0,
            latency_ms=latency,
        )
