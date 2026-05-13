from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def _as_bytes(text: bytes | bytearray | str) -> bytes:
    if isinstance(text, bytes):
        return text
    if isinstance(text, bytearray):
        return bytes(text)
    return text.encode("utf-8")


def _cpp_signed_char_sort_key(value: int) -> int:
    return value - 256 if value >= 128 else value


@dataclass
class Vocabulary:
    chars: list[int]

    def __post_init__(self) -> None:
        self.chars = [int(ch) & 0xFF for ch in self.chars]
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

    @staticmethod
    def from_text(text: bytes | bytearray | str) -> "Vocabulary":
        raw = _as_bytes(text)
        if not raw:
            raise ValueError("cannot build a vocabulary from empty text")
        chars = sorted(set(raw), key=_cpp_signed_char_sort_key)
        return Vocabulary(chars)

    def size(self) -> int:
        return len(self.chars)

    def id_for(self, ch: int | bytes | str, allow_unknown: bool = False) -> int:
        if isinstance(ch, int):
            value = ch & 0xFF
        else:
            raw = _as_bytes(ch)
            if len(raw) != 1:
                raise ValueError("id_for expects exactly one byte")
            value = raw[0]

        if value in self.stoi:
            return self.stoi[value]
        if allow_unknown and self.chars:
            return 0
        raise ValueError("character is not present in the vocabulary")

    def encode(self, text: bytes | bytearray | str, allow_unknown: bool = False) -> list[int]:
        raw = _as_bytes(text)
        ids: list[int] = []
        for ch in raw:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            elif allow_unknown and self.chars:
                ids.append(0)
            else:
                raise ValueError("text contains a character outside the vocabulary")
        return ids

    def decode(self, ids: Sequence[int]) -> bytes:
        out = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if token_id < 0 or token_id >= len(self.chars):
                raise ValueError("token id is outside the vocabulary")
            out.append(self.chars[token_id])
        return bytes(out)


@dataclass
class Batch:
    x: list[int]
    y: list[int]
    batch_size: int


def read_text_file(path: str | Path) -> bytes:
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"failed to open data file: {path}") from error
    if not data:
        raise ValueError(f"data file is empty: {path}")
    return data


def train_validation_split(
    data: Sequence[int], train_fraction: float, min_sequence: int
) -> tuple[list[int], list[int]]:
    if len(data) < min_sequence:
        raise ValueError("not enough encoded data for the requested context length")

    train_fraction = min(max(train_fraction, 0.5), 0.99)
    cut = int(float(len(data)) * train_fraction)
    cut = min(max(cut, min_sequence), len(data))

    train = list(data[:cut])
    valid = list(data[cut:])
    if len(valid) < min_sequence:
        valid = list(train)
    return train, valid


def make_batch(
    data: Sequence[int], context: int, batch_size: int, rng: random.Random
) -> Batch:
    if len(data) <= context + 1:
        raise ValueError("not enough data to create full-sequence targets")

    x = [0] * (batch_size * context)
    y = [0] * (batch_size * context)
    for b in range(batch_size):
        start = rng.randrange(0, len(data) - context)
        for t in range(context):
            x[b * context + t] = int(data[start + t])
            y[b * context + t] = int(data[start + t + 1])
    return Batch(x=x, y=y, batch_size=batch_size)
