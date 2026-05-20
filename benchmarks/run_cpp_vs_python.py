#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shlex
import string
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = "1,2,3,4,5,1337"
LOSS_RE = re.compile(
    r"^step\s+(?P<step>\d+)\s+train_loss=(?P<train>[-+0-9.eE]+)\s+"
    r"val_loss=(?P<val>[-+0-9.eE]+)"
)
PUNCTUATION_BYTES = {ord(ch) for ch in string.punctuation}
SUMMARY_METRICS = [
    "training_runtime_seconds",
    "generation_runtime_seconds",
    "final_train_loss",
    "final_val_loss",
    "sample_length",
    "space_count",
    "punctuation_count",
    "average_span_length_between_spaces",
    "repeated_character_rate",
    "unique_character_count",
]


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def write_summary(rows: list[dict[str, str]], summary_path: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["implementation"], []).append(row)

    fieldnames = ["implementation", "runs", *[f"avg_{metric}" for metric in SUMMARY_METRICS]]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for implementation in sorted(grouped):
            group = grouped[implementation]
            summary = {
                "implementation": implementation,
                "runs": str(len(group)),
            }
            for metric in SUMMARY_METRICS:
                average = sum(float(row[metric]) for row in group) / len(group)
                summary[f"avg_{metric}"] = f"{average:.6f}"
            writer.writerow(summary)


def print_summary(rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["implementation"], []).append(row)

    print("\nAverages", flush=True)
    for implementation in sorted(grouped):
        group = grouped[implementation]
        train_time = sum(float(row["training_runtime_seconds"]) for row in group) / len(group)
        gen_time = sum(float(row["generation_runtime_seconds"]) for row in group) / len(group)
        train_loss = sum(float(row["final_train_loss"]) for row in group) / len(group)
        val_loss = sum(float(row["final_val_loss"]) for row in group) / len(group)
        repeat_rate = sum(float(row["repeated_character_rate"]) for row in group) / len(group)
        unique_chars = sum(float(row["unique_character_count"]) for row in group) / len(group)
        print(
            f"  {implementation}: runs={len(group)} "
            f"train_s={train_time:.6f} gen_s={gen_time:.6f} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"repeat_rate={repeat_rate:.6f} unique_chars={unique_chars:.6f}",
            flush=True,
        )


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_combined_corpus(parts: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_ended_with_newline = True
    with output.open("wb") as out:
        for part in parts:
            data = part.read_bytes()
            if not data:
                raise ValueError(f"corpus part is empty: {part}")
            if not previous_ended_with_newline:
                out.write(b"\n")
            out.write(data)
            previous_ended_with_newline = data.endswith((b"\n", b"\r"))


def run_logged(command: list[str], stdout_log: Path, stderr_log: Path, env: dict[str, str] | None) -> tuple[bytes, float]:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    runtime = time.perf_counter() - start
    stdout_log.write_bytes(completed.stdout)
    stderr_log.write_bytes(completed.stderr)
    if completed.returncode != 0:
        quoted = " ".join(shlex.quote(part) for part in command)
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {quoted}\n"
            f"stdout log: {display_path(stdout_log)}\n"
            f"stderr log: {display_path(stderr_log)}"
        )
    return completed.stdout, runtime


def parse_final_losses(stdout: bytes) -> tuple[str, str]:
    final_match: re.Match[str] | None = None
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        match = LOSS_RE.match(line)
        if match:
            final_match = match
    if final_match is None:
        raise ValueError("trainer output did not contain a parseable final loss line")
    return final_match.group("train"), final_match.group("val")


def sample_metrics(sample: bytes) -> dict[str, str]:
    spaces = sample.count(b" ")
    spans = sample.split(b" ") if sample else []
    repeated = 0
    if len(sample) > 1:
        repeated = sum(1 for left, right in zip(sample, sample[1:]) if left == right)
    average_span = (sum(len(span) for span in spans) / len(spans)) if spans else 0.0
    repeat_rate = repeated / (len(sample) - 1) if len(sample) > 1 else 0.0
    punctuation = sum(1 for byte in sample if byte in PUNCTUATION_BYTES)
    return {
        "sample_length": str(len(sample)),
        "space_count": str(spaces),
        "punctuation_count": str(punctuation),
        "average_span_length_between_spaces": f"{average_span:.6f}",
        "repeated_character_rate": f"{repeat_rate:.6f}",
        "unique_character_count": str(len(set(sample))),
    }


def train_command(args: argparse.Namespace, implementation: str, checkpoint: Path, seed: int, corpus: Path) -> list[str]:
    if implementation == "cpp":
        executable = repo_path(args.cpp_train)
        if not executable.exists():
            raise FileNotFoundError(
                f"missing C++ trainer: {executable}. "
                "Run cmake -S cpp -B cpp/build && cmake --build cpp/build first."
            )
        command = [str(executable)]
    else:
        command = [args.python, str(repo_path(args.python_train))]

    command.extend(
        [
            "--data",
            str(corpus),
            "--steps",
            str(args.steps),
            "--context",
            str(args.context),
            "--embed",
            str(args.embed),
            "--layers",
            str(args.layers),
            "--heads",
            str(args.heads),
            "--hidden",
            str(args.hidden),
            "--batch",
            str(args.batch),
            "--lr",
            str(args.lr),
            "--clip",
            str(args.clip),
            "--print-every",
            str(args.steps),
            "--sample-every",
            "0",
            "--eval-batches",
            str(args.eval_batches),
            "--checkpoint",
            str(checkpoint),
            "--seed",
            str(seed),
        ]
    )
    return command


def generate_command(args: argparse.Namespace, implementation: str, checkpoint: Path, seed: int) -> list[str]:
    if implementation == "cpp":
        executable = repo_path(args.cpp_generate)
        if not executable.exists():
            raise FileNotFoundError(
                f"missing C++ generator: {executable}. "
                "Run cmake -S cpp -B cpp/build && cmake --build cpp/build first."
            )
        command = [str(executable)]
    else:
        command = [args.python, str(repo_path(args.python_generate))]

    command.extend(
        [
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            args.prompt,
            "--tokens",
            str(args.sample_length),
            "--temperature",
            str(args.temperature),
            "--top-k",
            str(args.top_k),
            "--seed",
            str(seed),
        ]
    )
    return command


def add_common_row_fields(args: argparse.Namespace, row: dict[str, str], corpus: Path, corpus_sha: str) -> None:
    row.update(
        {
            "corpus_path": display_path(corpus),
            "corpus_sha256": corpus_sha,
            "steps": str(args.steps),
            "eval_batches": str(args.eval_batches),
            "context": str(args.context),
            "embed": str(args.embed),
            "layers": str(args.layers),
            "heads": str(args.heads),
            "hidden": str(args.hidden),
            "batch": str(args.batch),
            "learning_rate": str(args.lr),
            "clip": str(args.clip),
            "prompt": args.prompt,
            "sample_length_requested": str(args.sample_length),
            "temperature": str(args.temperature),
            "top_k": str(args.top_k),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible C++ vs. pure-Python Tater Tot benchmark."
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds(DEFAULT_SEEDS))
    parser.add_argument(
        "--trials",
        type=int,
        help="Use seeds 1..N for N benchmark trials per implementation.",
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--embed", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--sample-length", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--corpus-parts",
        nargs="+",
        default=["data/pg1998.txt", "data/pg52124.txt"],
        help="Text files to concatenate into the shared benchmark corpus.",
    )
    parser.add_argument("--corpus-output", default="benchmarks/cpp_vs_python_corpus.txt")
    parser.add_argument("--csv", default="benchmarks/tater_tot_cpp_vs_python.csv")
    parser.add_argument("--summary-csv", default="benchmarks/tater_tot_cpp_vs_python_summary.csv")
    parser.add_argument("--sample-dir", default="benchmarks/samples")
    parser.add_argument("--checkpoint-dir", default="benchmarks/checkpoints")
    parser.add_argument("--log-dir", default="benchmarks/logs")
    parser.add_argument("--cpp-train", default="cpp/build/tater_train")
    parser.add_argument("--cpp-generate", default="cpp/build/tater_generate")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--python-train", default="python/tater_train.py")
    parser.add_argument("--python-generate", default="python/tater_generate.py")
    args = parser.parse_args()

    if args.trials is not None:
        if args.trials < 1:
            parser.error("--trials must be at least 1")
        args.seeds = list(range(1, args.trials + 1))
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.eval_batches < 1:
        parser.error("--eval-batches must be at least 1")
    if args.embed % args.heads != 0:
        parser.error("--embed must be divisible by --heads")
    return args


def main() -> int:
    args = parse_args()
    corpus_parts = [repo_path(part) for part in args.corpus_parts]
    corpus = repo_path(args.corpus_output)
    csv_path = repo_path(args.csv)
    summary_path = repo_path(args.summary_csv)
    sample_dir = repo_path(args.sample_dir)
    checkpoint_dir = repo_path(args.checkpoint_dir)
    log_dir = repo_path(args.log_dir)

    write_combined_corpus(corpus_parts, corpus)
    corpus_sha = sha256_file(corpus)
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    python_env = os.environ.copy()
    python_path = str(ROOT / "python")
    python_env["PYTHONPATH"] = (
        python_path
        if not python_env.get("PYTHONPATH")
        else python_path + os.pathsep + python_env["PYTHONPATH"]
    )

    fieldnames = [
        "implementation",
        "seed",
        "corpus_path",
        "corpus_sha256",
        "steps",
        "eval_batches",
        "context",
        "embed",
        "layers",
        "heads",
        "hidden",
        "batch",
        "learning_rate",
        "clip",
        "prompt",
        "sample_length_requested",
        "temperature",
        "top_k",
        "training_runtime_seconds",
        "generation_runtime_seconds",
        "final_train_loss",
        "final_val_loss",
        "checkpoint_filename",
        "checkpoint_sha256",
        "sample_path",
        "generated_sample",
        "sample_length",
        "space_count",
        "punctuation_count",
        "average_span_length_between_spaces",
        "repeated_character_rate",
        "unique_character_count",
    ]

    rows: list[dict[str, str]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for seed in args.seeds:
            for implementation in ("cpp", "python"):
                checkpoint = checkpoint_dir / f"{implementation}_seed{seed}.bin"
                sample_path = sample_dir / f"{implementation}_seed{seed}.txt"

                print(f"[{implementation} seed={seed}] training", flush=True)
                train_stdout, train_runtime = run_logged(
                    train_command(args, implementation, checkpoint, seed, corpus),
                    log_dir / f"{implementation}_seed{seed}_train.stdout.log",
                    log_dir / f"{implementation}_seed{seed}_train.stderr.log",
                    python_env if implementation == "python" else None,
                )
                final_train_loss, final_val_loss = parse_final_losses(train_stdout)

                print(f"[{implementation} seed={seed}] generating", flush=True)
                generated, generation_runtime = run_logged(
                    generate_command(args, implementation, checkpoint, seed),
                    log_dir / f"{implementation}_seed{seed}_generate.stdout.log",
                    log_dir / f"{implementation}_seed{seed}_generate.stderr.log",
                    python_env if implementation == "python" else None,
                )
                if generated.endswith(b"\n"):
                    generated = generated[:-1]
                sample_path.write_bytes(generated)

                row: dict[str, str] = {
                    "implementation": implementation,
                    "seed": str(seed),
                    "training_runtime_seconds": f"{train_runtime:.6f}",
                    "generation_runtime_seconds": f"{generation_runtime:.6f}",
                    "final_train_loss": final_train_loss,
                    "final_val_loss": final_val_loss,
                    "checkpoint_filename": display_path(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "sample_path": display_path(sample_path),
                    "generated_sample": generated.decode("utf-8", errors="replace"),
                }
                add_common_row_fields(args, row, corpus, corpus_sha)
                row.update(sample_metrics(generated))
                writer.writerow(row)
                rows.append(row)
                out.flush()

    write_summary(rows, summary_path)
    print_summary(rows)
    print(f"wrote {display_path(csv_path)}", flush=True)
    print(f"wrote {display_path(summary_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
