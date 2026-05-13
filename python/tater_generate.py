#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys

from tater import GenerationOptions, generate_text, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tater_generate.py",
        description="Generate text from a Tater Tot Transformer checkpoint.",
    )
    parser.add_argument("--checkpoint", default="checkpoints/model.bin")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        loaded = load_checkpoint(args.checkpoint)
        options = GenerationOptions(
            tokens=args.tokens, temperature=args.temperature, top_k=args.top_k
        )
        rng = random.Random(args.seed)
        output = generate_text(loaded.model, loaded.vocab, args.prompt, options, rng)
        sys.stdout.buffer.write(output + b"\n")
        sys.stdout.buffer.flush()
        return 0
    except Exception as error:
        print(f"tater_generate error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
