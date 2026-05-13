from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .data import Vocabulary
from .model import ModelConfig, TinyCharModel
from .tensor import Tensor, shape_to_string

MAGIC = b"TATER_TOT_CHECKPOINT_V2_TRANSFORMER"
V1_MAGIC = b"TATER_TOT_CHECKPOINT_V1"


@dataclass
class LoadedCheckpoint:
    model: TinyCharModel
    vocab: Vocabulary


def _write_u64(out, value: int) -> None:
    out.write(struct.pack("<Q", int(value)))


def _read_u64(inp) -> int:
    data = inp.read(8)
    if len(data) != 8:
        raise ValueError("failed to read checkpoint")
    return struct.unpack("<Q", data)[0]


def _write_string(out, value: bytes | str) -> None:
    raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    _write_u64(out, len(raw))
    out.write(raw)


def _read_string(inp) -> bytes:
    size = _read_u64(inp)
    data = inp.read(size)
    if len(data) != size:
        raise ValueError("failed to read checkpoint string")
    return data


def _write_tensor(out, name: str, tensor: Tensor) -> None:
    _write_string(out, name)
    _write_u64(out, len(tensor.shape))
    for dim in tensor.shape:
        _write_u64(out, dim)
    _write_u64(out, tensor.size)
    out.write(struct.pack("<" + "d" * tensor.size, *tensor.data))


def _read_tensor_into(inp, expected_name: str, tensor: Tensor) -> None:
    name = _read_string(inp).decode("utf-8")
    if name != expected_name:
        raise ValueError(
            f"checkpoint tensor order mismatch: expected {expected_name}, got {name}"
        )

    rank = _read_u64(inp)
    shape = tuple(_read_u64(inp) for _ in range(rank))
    if shape != tensor.shape:
        raise ValueError(
            f"checkpoint tensor shape mismatch for {name}: expected "
            f"{shape_to_string(tensor.shape)}, got {shape_to_string(shape)}"
        )

    values = _read_u64(inp)
    if values != tensor.size:
        raise ValueError(f"checkpoint tensor size mismatch for {name}")
    raw = inp.read(values * 8)
    if len(raw) != values * 8:
        raise ValueError("failed to read tensor data")
    tensor.data[:] = list(struct.unpack("<" + "d" * values, raw))


def save_checkpoint(path: str | Path, model: TinyCharModel, vocab: Vocabulary) -> None:
    checkpoint_path = Path(path)
    if checkpoint_path.parent != Path(""):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with checkpoint_path.open("wb") as out:
        _write_string(out, MAGIC)
        _write_u64(out, model.config.vocab_size)
        _write_u64(out, model.config.context)
        _write_u64(out, model.config.embed)
        _write_u64(out, model.config.hidden)
        _write_u64(out, model.config.layers)
        _write_u64(out, model.config.heads)
        _write_string(out, bytes(vocab.chars))

        params = model.named_parameters()
        _write_u64(out, len(params))
        for name, tensor in params:
            _write_tensor(out, name, tensor)


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    checkpoint_path = Path(path)
    with checkpoint_path.open("rb") as inp:
        magic = _read_string(inp)
        if magic != MAGIC:
            if magic == V1_MAGIC:
                raise ValueError(
                    "checkpoint is an old V1 MLP checkpoint and cannot be loaded into the "
                    "Transformer model"
                )
            raise ValueError(f"not a Tater Tot checkpoint: {path}")

        config = ModelConfig()
        config.vocab_size = _read_u64(inp)
        config.context = _read_u64(inp)
        config.embed = _read_u64(inp)
        config.hidden = _read_u64(inp)
        config.layers = _read_u64(inp)
        config.heads = _read_u64(inp)

        chars = _read_string(inp)
        vocab = Vocabulary(list(chars))
        if vocab.size() != config.vocab_size:
            raise ValueError("checkpoint vocabulary size does not match model config")

        model = TinyCharModel(config, 1)
        param_count = _read_u64(inp)
        params = model.named_parameters()
        if param_count != len(params):
            raise ValueError("checkpoint parameter count does not match model")
        for name, tensor in params:
            _read_tensor_into(inp, name, tensor)

    return LoadedCheckpoint(model=model, vocab=vocab)
