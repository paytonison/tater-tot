from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from .data import Batch, Vocabulary, _as_bytes, make_batch
from .tensor import (
    Tensor,
    add,
    causal_self_attention,
    embedding_lookup,
    gelu,
    layer_norm,
    matmul,
    numel,
    softmax_cross_entropy,
)


@dataclass
class ModelConfig:
    vocab_size: int = 0
    context: int = 32
    layers: int = 2
    heads: int = 4
    embed: int = 64
    hidden: int = 128


@dataclass
class GenerationOptions:
    tokens: int = 300
    temperature: float = 1.0
    top_k: int = 0


def _random_tensor(shape: Sequence[int], stddev: float, rng: random.Random) -> Tensor:
    values = [rng.gauss(0.0, stddev) for _ in range(numel(tuple(shape)))]
    return Tensor.from_data(values, tuple(shape), True)


def _xavier_stddev(fan_in: int, fan_out: int) -> float:
    return math.sqrt(2.0 / float(fan_in + fan_out))


def _ones(shape: Sequence[int]) -> Tensor:
    return Tensor(tuple(shape), 1.0, True)


def _zeros(shape: Sequence[int]) -> Tensor:
    return Tensor(tuple(shape), 0.0, True)


def _validate_config(config: ModelConfig) -> None:
    if config.vocab_size == 0:
        raise ValueError("model config vocab_size must be non-zero")
    if config.context == 0:
        raise ValueError("model config context must be non-zero")
    if config.embed == 0:
        raise ValueError("model config embed must be non-zero")
    if config.hidden == 0:
        raise ValueError("model config hidden must be non-zero")
    if config.layers == 0:
        raise ValueError("model config layers must be non-zero")
    if config.heads == 0:
        raise ValueError("model config heads must be non-zero")
    if config.embed % config.heads != 0:
        raise ValueError("model config embed must be divisible by heads")


class TokenEmbedding:
    def __init__(
        self, vocab_size: int | None = None, d_model: int | None = None, rng: random.Random | None = None
    ) -> None:
        self.table = Tensor()
        if vocab_size is None and d_model is None:
            return
        if vocab_size is None or d_model is None or rng is None:
            raise ValueError("token embedding construction requires vocab_size, d_model, and rng")
        if vocab_size == 0:
            raise ValueError("token embedding vocab_size must be non-zero")
        if d_model == 0:
            raise ValueError("token embedding d_model must be non-zero")
        self.table = _random_tensor((vocab_size, d_model), 0.02, rng)

    def forward(self, token_ids: Sequence[int]) -> Tensor:
        if len(self.table.shape) != 2:
            raise ValueError("token embedding table must be rank 2")
        rows = self.table.shape[0]
        for token_id in token_ids:
            if token_id < 0 or token_id >= rows:
                raise ValueError("token_id out of range for token embedding")
        return embedding_lookup(self.table, token_ids)

    def vocab_size(self) -> int:
        if len(self.table.shape) != 2:
            raise ValueError("token embedding table must be rank 2")
        return self.table.shape[0]

    def d_model(self) -> int:
        if len(self.table.shape) != 2:
            raise ValueError("token embedding table must be rank 2")
        return self.table.shape[1]


class PositionalEmbedding:
    def __init__(
        self,
        max_seq_len: int | None = None,
        d_model: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.table = Tensor()
        if max_seq_len is None and d_model is None:
            return
        if max_seq_len is None or d_model is None or rng is None:
            raise ValueError(
                "positional embedding construction requires max_seq_len, d_model, and rng"
            )
        if max_seq_len == 0:
            raise ValueError("positional embedding max_seq_len must be non-zero")
        if d_model == 0:
            raise ValueError("positional embedding d_model must be non-zero")
        self.table = _random_tensor((max_seq_len, d_model), 0.02, rng)

    def forward(self, batch_size: int, sequence_length: int) -> Tensor:
        if batch_size == 0:
            raise ValueError("positional embedding batch_size must be non-zero")
        if sequence_length == 0:
            raise ValueError("positional embedding sequence_length must be non-zero")
        if sequence_length > self.max_seq_len():
            raise ValueError("position out of range for positional embedding")

        positions = [0] * (batch_size * sequence_length)
        for b in range(batch_size):
            for t in range(sequence_length):
                positions[b * sequence_length + t] = t
        return embedding_lookup(self.table, positions)

    def max_seq_len(self) -> int:
        if len(self.table.shape) != 2:
            raise ValueError("positional embedding table must be rank 2")
        return self.table.shape[0]

    def d_model(self) -> int:
        if len(self.table.shape) != 2:
            raise ValueError("positional embedding table must be rank 2")
        return self.table.shape[1]


@dataclass
class TransformerBlock:
    ln1_gamma: Tensor
    ln1_beta: Tensor
    q_w: Tensor
    q_b: Tensor
    k_w: Tensor
    k_b: Tensor
    v_w: Tensor
    v_b: Tensor
    out_w: Tensor
    out_b: Tensor
    ln2_gamma: Tensor
    ln2_beta: Tensor
    ff1_w: Tensor
    ff1_b: Tensor
    ff2_w: Tensor
    ff2_b: Tensor


class TinyCharModel:
    def __init__(self, config: ModelConfig, seed: int = 1337) -> None:
        self._config = ModelConfig(
            vocab_size=config.vocab_size,
            context=config.context,
            layers=config.layers,
            heads=config.heads,
            embed=config.embed,
            hidden=config.hidden,
        )
        _validate_config(self._config)

        rng = random.Random(seed)
        self.token_embedding = TokenEmbedding(config.vocab_size, config.embed, rng)
        self.positional_embedding = PositionalEmbedding(config.context, config.embed, rng)
        self.blocks: list[TransformerBlock] = []

        for _ in range(config.layers):
            block = TransformerBlock(
                ln1_gamma=_ones((config.embed,)),
                ln1_beta=_zeros((config.embed,)),
                q_w=_random_tensor(
                    (config.embed, config.embed),
                    _xavier_stddev(config.embed, config.embed),
                    rng,
                ),
                q_b=_zeros((config.embed,)),
                k_w=_random_tensor(
                    (config.embed, config.embed),
                    _xavier_stddev(config.embed, config.embed),
                    rng,
                ),
                k_b=_zeros((config.embed,)),
                v_w=_random_tensor(
                    (config.embed, config.embed),
                    _xavier_stddev(config.embed, config.embed),
                    rng,
                ),
                v_b=_zeros((config.embed,)),
                out_w=_random_tensor(
                    (config.embed, config.embed),
                    _xavier_stddev(config.embed, config.embed),
                    rng,
                ),
                out_b=_zeros((config.embed,)),
                ln2_gamma=_ones((config.embed,)),
                ln2_beta=_zeros((config.embed,)),
                ff1_w=_random_tensor(
                    (config.embed, config.hidden),
                    _xavier_stddev(config.embed, config.hidden),
                    rng,
                ),
                ff1_b=_zeros((config.hidden,)),
                ff2_w=_random_tensor(
                    (config.hidden, config.embed),
                    _xavier_stddev(config.hidden, config.embed),
                    rng,
                ),
                ff2_b=_zeros((config.embed,)),
            )
            self.blocks.append(block)

        self.ln_f_gamma = _ones((config.embed,))
        self.ln_f_beta = _zeros((config.embed,))
        self.lm_head_w = _random_tensor(
            (config.embed, config.vocab_size),
            _xavier_stddev(config.embed, config.vocab_size),
            rng,
        )
        self.lm_head_b = _zeros((config.vocab_size,))

    @property
    def config(self) -> ModelConfig:
        return self._config

    def forward(self, tokens: Sequence[int], batch_size: int) -> Tensor:
        if batch_size == 0:
            raise ValueError("model forward batch_size must be non-zero")
        if len(tokens) != batch_size * self._config.context:
            raise ValueError("model input token count does not match batch_size * context")

        token_vectors = self.token_embedding.forward(tokens)
        position_vectors = self.positional_embedding.forward(batch_size, self._config.context)
        x = add(token_vectors, position_vectors)

        for block in self.blocks:
            normalized = layer_norm(x, block.ln1_gamma, block.ln1_beta)
            attention = causal_self_attention(
                normalized,
                block.q_w,
                block.q_b,
                block.k_w,
                block.k_b,
                block.v_w,
                block.v_b,
                block.out_w,
                block.out_b,
                batch_size,
                self._config.context,
                self._config.heads,
            )
            x = add(x, attention)

            normalized = layer_norm(x, block.ln2_gamma, block.ln2_beta)
            ff = add(matmul(normalized, block.ff1_w), block.ff1_b)
            ff = gelu(ff)
            ff = add(matmul(ff, block.ff2_w), block.ff2_b)
            x = add(x, ff)

        x = layer_norm(x, self.ln_f_gamma, self.ln_f_beta)
        return add(matmul(x, self.lm_head_w), self.lm_head_b)

    def parameters(self) -> list[Tensor]:
        return [tensor for _, tensor in self.named_parameters()]

    def named_parameters(self) -> list[tuple[str, Tensor]]:
        params: list[tuple[str, Tensor]] = [
            ("token_embedding", self.token_embedding.table),
            ("positional_embedding", self.positional_embedding.table),
        ]
        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}"
            params.extend(
                [
                    (prefix + ".ln1_gamma", block.ln1_gamma),
                    (prefix + ".ln1_beta", block.ln1_beta),
                    (prefix + ".q_w", block.q_w),
                    (prefix + ".q_b", block.q_b),
                    (prefix + ".k_w", block.k_w),
                    (prefix + ".k_b", block.k_b),
                    (prefix + ".v_w", block.v_w),
                    (prefix + ".v_b", block.v_b),
                    (prefix + ".out_w", block.out_w),
                    (prefix + ".out_b", block.out_b),
                    (prefix + ".ln2_gamma", block.ln2_gamma),
                    (prefix + ".ln2_beta", block.ln2_beta),
                    (prefix + ".ff1_w", block.ff1_w),
                    (prefix + ".ff1_b", block.ff1_b),
                    (prefix + ".ff2_w", block.ff2_w),
                    (prefix + ".ff2_b", block.ff2_b),
                ]
            )
        params.extend(
            [
                ("ln_f_gamma", self.ln_f_gamma),
                ("ln_f_beta", self.ln_f_beta),
                ("lm_head_w", self.lm_head_w),
                ("lm_head_b", self.lm_head_b),
            ]
        )
        return params


def sample_from_logits(
    logits: Sequence[float], temperature: float, top_k: int, rng: random.Random
) -> int:
    if not logits:
        raise ValueError("cannot sample from empty logits")
    temperature = max(temperature, 1e-6)

    candidates = list(range(len(logits)))
    if top_k > 0 and top_k < len(candidates):
        candidates.sort(key=lambda idx: logits[idx], reverse=True)
        candidates = candidates[:top_k]

    max_scaled = max(logits[idx] / temperature for idx in candidates)
    weights = [math.exp(logits[idx] / temperature - max_scaled) for idx in candidates]
    total = sum(weights)
    threshold = rng.random() * total
    running = 0.0
    for candidate, weight in zip(candidates, weights):
        running += weight
        if threshold <= running:
            return candidate
    return candidates[-1]


def generate_text(
    model: TinyCharModel,
    vocab: Vocabulary,
    prompt: bytes | bytearray | str,
    options: GenerationOptions,
    rng: random.Random,
) -> bytes:
    ids = vocab.encode(_as_bytes(prompt), True)
    if not ids:
        ids.append(0)

    for _ in range(options.tokens):
        context = [0] * model.config.context
        copy_count = min(model.config.context, len(ids))
        src_start = len(ids) - copy_count
        dst_start = model.config.context - copy_count
        for i in range(copy_count):
            context[dst_start + i] = ids[src_start + i]

        logits = model.forward(context, 1)
        vocab_size = model.config.vocab_size
        final_row = model.config.context - 1
        start = final_row * vocab_size
        next_logits = logits.data[start : start + vocab_size]
        next_id = sample_from_logits(
            next_logits, options.temperature, options.top_k, rng
        )
        ids.append(next_id)

    return vocab.decode(ids)


def estimate_loss(
    model: TinyCharModel,
    data: Sequence[int],
    batch_size: int,
    batches: int,
    rng: random.Random,
) -> float:
    if batches == 0:
        return 0.0

    total = 0.0
    for _ in range(batches):
        batch: Batch = make_batch(data, model.config.context, batch_size, rng)
        logits = model.forward(batch.x, batch.batch_size)
        loss = softmax_cross_entropy(logits, batch.y)
        total += loss.data[0]
    return total / float(batches)
