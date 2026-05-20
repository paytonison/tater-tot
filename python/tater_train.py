#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys

from tater import (
    Adam,
    GenerationOptions,
    ModelConfig,
    TinyCharModel,
    Vocabulary,
    estimate_loss,
    generate_text,
    make_batch,
    read_text_file,
    save_checkpoint,
    softmax_cross_entropy,
    train_validation_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tater_train.py",
        description="Train the standard-library Python Tater Tot Transformer.",
    )
    parser.add_argument("--data", default="../data/input.txt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--embed", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--checkpoint", default="../checkpoints/model.bin")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        hidden = args.hidden if args.hidden is not None else args.embed * 4

        text = read_text_file(args.data)
        vocab = Vocabulary.from_text(text)
        encoded = vocab.encode(text)
        train_data, valid_data = train_validation_split(encoded, 0.9, args.context + 2)

        config = ModelConfig(
            vocab_size=vocab.size(),
            context=args.context,
            layers=args.layers,
            heads=args.heads,
            embed=args.embed,
            hidden=hidden,
        )
        rng = random.Random(args.seed)
        model = TinyCharModel(config, args.seed)
        optimizer = Adam(model.parameters(), args.lr)

        print(
            "Tater Tot training\n"
            f"  data: {args.data}\n"
            f"  chars: {len(text)}\n"
            f"  vocab: {vocab.size()}\n"
            "  context/layers/heads/embed/hidden: "
            f"{args.context}/{args.layers}/{args.heads}/{args.embed}/{hidden}\n"
            f"  batch/steps/lr: {args.batch}/{args.steps}/{args.lr}",
            flush=True,
        )

        prompt = text[: min(len(text), 32)]
        for step in range(1, args.steps + 1):
            batch = make_batch(train_data, args.context, args.batch, rng)
            logits = model.forward(batch.x, batch.batch_size)
            loss = softmax_cross_entropy(logits, batch.y)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = optimizer.step(args.clip)

            if args.print_every != 0 and (
                step == 1 or step % args.print_every == 0 or step == args.steps
            ):
                val_loss = estimate_loss(model, valid_data, args.batch, args.eval_batches, rng)
                print(
                    f"step {step} train_loss={loss.data[0]} "
                    f"val_loss={val_loss} grad_norm={grad_norm}",
                    flush=True,
                )

            if args.sample_every != 0 and (
                step % args.sample_every == 0 or step == args.steps
            ):
                options = GenerationOptions(tokens=160, temperature=0.8, top_k=10)
                sample = generate_text(model, vocab, prompt, options, rng)
                print("--- sample ---", flush=True)
                sys.stdout.buffer.write(sample + b"\n--------------\n")
                sys.stdout.buffer.flush()

        save_checkpoint(args.checkpoint, model, vocab)
        print(f"saved checkpoint: {args.checkpoint}")
        return 0
    except Exception as error:
        print(f"tater_train error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
