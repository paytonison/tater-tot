# C++ vs. Pure-Python Benchmark

This benchmark compares the C++ Tater Tot Transformer against the standard-library
Python port using the same command-line settings and the same generated corpus.

## What Is Compared

- C++ trainer: `build/tater_train`
- C++ generator: `build/tater_generate`
- Python trainer: `python/tater_train.py`
- Python generator: `python/tater_generate.py`

Both implementations use the existing byte-level Transformer architecture. The
benchmark does not change model code or checkpoint format.

## Corpus

By default the runner concatenates:

- `data/pg1998.txt`
- `data/pg52124.txt`

The combined file is written to `benchmarks/cpp_vs_python_corpus.txt` and its
SHA-256 is recorded in the CSV. Use `--corpus-parts` to compare on a different
set of text files.

## Default Settings

The default run is intentionally small enough for the pure-Python implementation:

```text
seeds: 1,2,3,4,5,1337
prompt: "The "
steps: 25
eval-batches: 2
context: 16
embed: 16
layers: 1
heads: 2
hidden: 64
batch: 4
lr: 0.003
clip: 1.0
sample-length: 160
temperature: 0.8
top-k: 10
```

Increase `--steps`, `--context`, `--embed`, `--hidden`, or `--batch` for a more
substantial benchmark once the short pass works.

## Reproduce

Build and test first:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
python3 tests/test_python.py
```

Run the default benchmark:

```sh
python3 benchmarks/run_cpp_vs_python.py
```

Run a quick smoke pass:

```sh
python3 benchmarks/run_cpp_vs_python.py \
  --seeds 1 \
  --steps 1 \
  --eval-batches 1 \
  --context 8 \
  --embed 8 \
  --layers 1 \
  --heads 2 \
  --hidden 16 \
  --batch 2 \
  --sample-length 32
```

The runner writes:

- `benchmarks/tater_tot_cpp_vs_python.csv`
- `benchmarks/samples/*.txt`
- `benchmarks/checkpoints/*.bin`
- `benchmarks/logs/*.log`

The CSV includes runtime, final train/validation loss, checkpoint name and
SHA-256, generated sample text, and simple byte-level text-quality metrics.

## Caveats

- This is a fair protocol, not a bit-for-bit equivalence test. C++ `std::mt19937`
  and Python `random.Random` do not produce identical normal initialization,
  batch selection, or sampling streams from the same numeric seed.
- Samples and quality metrics are byte-level because Tater Tot is byte-oriented.
- The default settings are short and useful for reproducibility checks. Treat
  longer runs as the meaningful comparison.
- The Python implementation is intentionally dependency-free and much slower
  than the C++ version.
