# Tater Tot

A tiny character-level decoder-only Transformer language model implemented from
scratch in C++ with parallel standard-library Python and C ports.

Tater Tot is a compact GPT-style language model built to show the core mechanics
of modern autoregressive Transformers without relying on PyTorch, TensorFlow,
ONNX Runtime, CUDA, or another machine learning framework. It is intentionally
small, educational, and architecture-focused: the point is to make the pieces of
a Transformer visible in ordinary C++20, C11, and plain Python.

The project includes:

- a custom `Tensor` type and reverse-mode autodiff engine
- a character-level data pipeline
- learned token and positional embeddings
- causal multi-head self-attention
- pre-layernorm Transformer blocks
- GELU feed-forward layers
- residual connections
- an Adam optimizer
- binary checkpoint save/load
- autoregressive text generation

## Agent-Assisted One-Shot Development

Tater Tot began as a human-originated project idea: a small, from-scratch language-model implementation intended to make transformer internals concrete, inspectable, and experimentally comparable to a conventional Python baseline.

The initial version of the project was generated in a single Codex pass from high-level project instructions, and the later transformer upgrade was likewise produced as a one-shot architectural expansion.

The goal is not to present the code as hand-authored line by line. Instead, this repository documents a workflow: human concept formation, specification, agentic implementation, human review, debugging, training, evaluation, and iterative refinement.

This makes Tater Tot both a small language-model implementation and a case study in how coding agents can compress the distance between concept and working artifact. The human role shifts away from raw syntax production and toward architecture, constraint-setting, supervision, interpretation, and experimental design.

## Architecture

Tater Tot is a character-level autoregressive decoder-only Transformer. Raw text
is encoded as character IDs, embedded, passed through causal Transformer blocks,
and projected back to vocabulary logits.

```text
Raw text
  -> character vocabulary
  -> token IDs
  -> token embeddings
  -> positional embeddings
  -> Transformer blocks
       -> pre-LayerNorm
       -> causal multi-head self-attention
       -> residual add
       -> pre-LayerNorm
       -> MLP/feed-forward with GELU
       -> residual add
  -> final LayerNorm
  -> LM head
  -> logits over character vocabulary
  -> sampling
  -> generated text
```

The model keeps the sequence axis alive logically as
`[batch, context, embed]`. Internally, the C++ tensor code stores that activation
as `[batch * context, embed]` when convenient, so each row is still a specific
token position.

Attention mixes information across token positions. The causal mask prevents a
position from attending to future characters, so position `t` can only use
positions `<= t`. After attention, the feed-forward network transforms each
token vector independently. Residual connections preserve information across
sub-layers and make optimization easier, while LayerNorm stabilizes the scale of
activations. The final LM head maps each final hidden vector to next-character
scores over the vocabulary.

## What Changed

The original prototype was a character-level MLP language model. It embedded the
characters in a context window, flattened the whole context into one vector, and
predicted a single next character.

Old model:

```text
token embeddings + position embeddings
  -> flatten context
  -> MLP
  -> next-character logits
```

The current model keeps token positions separate, applies causal self-attention,
and predicts the next character at every position in the context window.

New model:

```text
token embeddings + position embeddings
  -> causal Transformer blocks
  -> LM head
  -> next-character logits at every position
```

This is now a real tiny decoder-only Transformer, not a flattened-context MLP.

## Training Objective

Training uses next-character prediction. For each sampled text window, the input
characters are shifted by one position to form the targets:

```text
Input:  N i e t z s c h
Target: i e t z s c h e
```

The batching code trains every position in the context window. For a batch with
`batch_size` rows and `context` characters per row, the model returns logits
shaped approximately:

```text
[batch_size * context, vocab_size]
```

`softmax_cross_entropy` consumes those logits together with
`batch_size * context` target IDs.

## Generation

Generation is autoregressive:

1. Encode the prompt into character IDs.
2. Keep the most recent `context` IDs, left-padding with ID `0` if needed.
3. Run the model over that context.
4. Take only the final position's logits.
5. Sample one character ID.
6. Append the sampled character.
7. Repeat until the requested number of characters has been generated.

Sampling currently supports temperature and top-k filtering. There is no top-p,
beam search, or repetition penalty.

## Project Structure

```text
cpp/
  C++20 implementation. Contains its own `CMakeLists.txt`, headers, sources,
  training CLI, generation CLI, and C++ tests.

python/
  Standard-library Python implementation. Contains the `tater` package,
  training CLI, generation CLI, and Python tests. It has no dependency on
  PyTorch, NumPy, or any other third-party package.

c/
  C11 implementation. Contains its own `Makefile`, header, source, training
  CLI, sampling CLI, and C tests.

data/
  Shared plain-text training corpora.

checkpoints/
  Shared local checkpoint output directory. This directory is ignored by git.

benchmarks/
  Cross-language benchmark harness and benchmark documentation.
```

## Build

Each implementation can be built or tested from its own directory.

C++ uses CMake and the C++20 standard library:

```sh
cd cpp
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The build creates:

- `cpp/build/tater_train`
- `cpp/build/tater_generate`
- `cpp/build/tater_tests`

The Python port does not need a build step:

```sh
cd python
python3 tests/test_python.py
```

The C port uses its own Makefile:

```sh
cd c
make
make test
```

The C build creates:

- `c/tater_train_c`
- `c/tater_sample_c`
- `c/tater_tests_c`

## Train

Put a plain text file at `data/input.txt`, or pass another text path with
`--data`.

```sh
cd cpp
./build/tater_train \
  --data ../data/input.txt \
  --steps 1000 \
  --context 64 \
  --embed 64 \
  --layers 2 \
  --heads 4 \
  --hidden 256 \
  --batch 16 \
  --lr 0.003 \
  --clip 1.0 \
  --print-every 50 \
  --sample-every 200 \
  --eval-batches 4 \
  --checkpoint ../checkpoints/model.bin \
  --seed 1337
```

The equivalent Python entry point accepts the same training options:

```sh
cd python
python3 tater_train.py \
  --data ../data/input.txt \
  --steps 1000 \
  --context 64 \
  --embed 64 \
  --layers 2 \
  --heads 4 \
  --hidden 256 \
  --batch 16 \
  --lr 0.003 \
  --clip 1.0 \
  --print-every 50 \
  --sample-every 200 \
  --eval-batches 4 \
  --checkpoint ../checkpoints/model.bin \
  --seed 1337
```

The C training entry point accepts the same core options:

```sh
cd c
./tater_train_c \
  --data ../data/input.txt \
  --steps 1000 \
  --context 64 \
  --embed 64 \
  --layers 2 \
  --heads 4 \
  --hidden 256 \
  --batch 16 \
  --lr 0.003 \
  --clip 1.0 \
  --print-every 50 \
  --sample-every 200 \
  --eval-batches 4 \
  --checkpoint ../checkpoints/model.bin \
  --seed 1337
```

Training prints mini-batch loss, an estimated validation loss, global gradient
norm, and periodic samples. The checkpoint is written at the end of training.

Important model constraints:

- `vocab_size`, `context`, `embed`, `hidden`, `layers`, and `heads` must be
  non-zero.
- `embed` must be divisible by `heads`.
- `--hidden` defaults to `embed * 4` in the training CLI.

## Generate

After training, generate text from a checkpoint:

```sh
cd cpp
./build/tater_generate \
  --checkpoint ../checkpoints/model.bin \
  --prompt "Once upon a time" \
  --tokens 300 \
  --temperature 0.9 \
  --top-k 20 \
  --seed 1337
```

The Python generator can load the same V2 checkpoint format:

```sh
cd python
python3 tater_generate.py \
  --checkpoint ../checkpoints/model.bin \
  --prompt "Once upon a time" \
  --tokens 300 \
  --temperature 0.9 \
  --top-k 20 \
  --seed 1337
```

The C sampler can also load the same V2 checkpoint format:

```sh
cd c
./tater_sample_c \
  --checkpoint ../checkpoints/model.bin \
  --prompt "Once upon a time" \
  --tokens 300 \
  --temperature 0.9 \
  --top-k 20 \
  --seed 1337
```

Generation prints the prompt followed by sampled characters. `temperature`
controls randomness. `top-k` limits sampling to the most likely `k` characters
when greater than zero.

## Checkpointing

Checkpoints store:

1. Magic string: `TATER_TOT_CHECKPOINT_V2_TRANSFORMER`
2. Transformer config: vocabulary size, context length, embedding size, hidden
   size, layer count, and attention head count
3. Vocabulary characters
4. Named parameter tensors in deterministic model order

Older MLP checkpoints are not compatible with the Transformer checkpoint format.
The loader rejects V1 MLP checkpoints with a clear error rather than silently
loading mismatched parameters.

## Tests

Run the full local test binary through CTest:

```sh
cd cpp
ctest --test-dir build --output-on-failure
```

Run the C sanity tests through Make:

```sh
cd c
make test
```

Run the Python smoke tests from the Python implementation directory:

```sh
cd python
python3 tests/test_python.py
```

The current test suite covers:

- tensor shape handling and basic math outputs
- finite-difference gradient checks for core ops, GELU, LayerNorm, and
  softmax cross-entropy
- Transformer logits shape for `[batch * context, vocab_size]`
- finite gradients for every named model parameter after backpropagation
- full-position batch target creation
- checkpoint save/load round trip
- tiny repeated-text overfit smoke test
- generation smoke test

## Benchmarks

The `benchmarks/` directory contains a reproducible C++ vs. pure-Python
comparison harness. It builds a shared corpus from `data/pg1998.txt` and
`data/pg52124.txt` by default, runs both implementations with the same prompt,
model settings, optimization settings, generation settings, and seed list, then
writes a machine-readable CSV plus generated samples.

```sh
cd cpp
cmake -S . -B build
cmake --build build
cd ..
python3 benchmarks/run_cpp_vs_python.py
```

For 50 trials per implementation:

```sh
python3 benchmarks/run_cpp_vs_python.py --trials 50
```

For a fast smoke check:

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

See `benchmarks/README.md` for the exact default settings, output files, and
caveats. Benchmark outputs, including the per-run CSV and summary CSV, are
generated artifacts and are ignored by git.

The system used to run the benchmark tests was an Apple M1 Max. All benchmark
runs were performed on the CPU only, without GPU, Neural Engine, or other
accelerator backends.

### Current Findings

The completed 50-trial benchmark used seeds `1` through `50` with the default
benchmark settings. In those runs, the C++ implementation is currently
outperforming the pure-Python baseline in wall-clock speed:

```text
implementation  avg train sec  avg generate sec  avg train loss  avg val loss
cpp                    1.340021          0.657506        3.465902      3.433631
python                 7.237891          3.289465        3.450698      3.422890
```

That is about `5.4x` faster for training and `5.0x` faster for generation in
this benchmark. The early C++ samples also appeared more sentence-shaped in the
observed runs: C++ produced non-zero punctuation in `3` of `50` samples, while
the Python baseline produced none in those same measured runs. This is a
qualitative signal, not a proof of better language modeling. The Python run had
slightly lower average train and validation loss, and model-quality conclusions
need repeated seed-controlled benchmarks at meaningful training lengths.

C++ is a strong fit for this project's AI goals because the core model is meant
to be transparent and inspectable rather than hidden behind a framework. The C++
path has low abstraction overhead, faster training and generation loops in the
measured benchmark, explicit control over the tensor/autodiff/model internals,
and a layout that makes memory, ownership, and performance costs easier to
reason about. That makes it useful as the transparent experimental model core.

The current C++ implementation is not more compact by raw source size. Counting
the comparable implementation and CLI source sets, `cpp/include/tater`,
`cpp/src`, and `cpp/examples` contain `12` files, `2,275` total lines, and
`87,978` bytes. The
Python baseline in `python/tater`, `python/tater_train.py`, and
`python/tater_generate.py` contains `8` files, `1,768` total lines, and `57,616`
bytes.

## Limitations

- The model is tiny and character-level.
- Character handling is byte-oriented, not Unicode code point tokenization.
- There is no BPE, WordPiece, or sentencepiece-style tokenizer.
- Training examples are plain text files loaded into memory.
- There is no GPU acceleration, CUDA backend, BLAS integration, SIMD kernel, or
  fused attention implementation.
- There is no dropout.
- There is no top-p sampling, beam search, repetition penalty, or advanced
  decoding strategy.
- Checkpoints do not store optimizer state.
- This is an educational and portfolio implementation, not production inference
  infrastructure and not comparable to industrial LLMs.

## Roadmap

Reasonable next improvements:

- cleaner CLI help and configuration files
- more focused gradchecks, especially for causal attention
- tokenizer experiments beyond character IDs
- optional weight tying between token embeddings and the LM head
- dropout in attention and feed-forward layers
- learning-rate scheduling
- perplexity and richer validation reporting
- faster kernels for matrix multiplication and attention
- optional CMake cleanup and install targets
- richer sampling controls such as top-p and repetition penalties
- include the C port in the benchmark harness
- richer benchmark summaries, memory measurements, and visualizations

Longer-term experimental roadmap:

- **C++ Tater Tot baseline:** continue developing the current from-scratch C++ implementation as the primary systems-level reference model. The purpose is to keep the Transformer machinery visible, inspectable, and close to the hardware.

- **Python Tater Tot baseline:** keep the dependency-free Python port aligned with the C++ implementation. It already uses the same byte-level data model, checkpoint format, training options, and generation options, which makes it a useful high-level comparison target.

- **C Tater Tot reference:** continue maturing the C11 port as a compact systems reference beside the C++ and Python implementations. The C port is useful for making allocation, ownership, and model mechanics explicit without C++ abstractions.

- **Cross-language performance benchmark:** extend the existing C++ vs. Python benchmark harness into a broader comparison across the C++, C, and Python implementations. Useful dimensions include:
  - training speed
  - inference speed
  - memory use
  - tokens or characters generated per second
  - validation loss / perplexity
  - qualitative generation samples from matched prompts

  The goal is not merely to ask which language is faster in the abstract, but to measure what each implementation exposes, costs, and makes easier to inspect under comparable constraints.

- **Handmade MoE experiment:** prototype a small mixture-of-experts system using existing Gemma-family models as external expert modules. Instead of training one monolithic model from scratch, the system would stitch together multiple specialized Gemma instances and route tasks between them.

- **Mediator model:** add a central mediator model that sits between the user/task and the expert models. The mediator would decide which expert or combination of experts should handle a request, merge their outputs, resolve disagreements, and produce the final response.

- **Lexical / definition expert:** explore a dedicated expert for words, definitions, morphology, etymology, and sense disambiguation. This connects to the broader idea of using dictionary-like knowledge not as a literal embedding matrix, but as a specialized semantic subsystem.

- **MoE evaluation harness:** design tests that compare the handmade MoE against single-model baselines on:
  - factual consistency
  - definition accuracy
  - reasoning quality
  - code generation
  - stylistic control
  - routing accuracy
  - disagreement resolution

This roadmap turns Tater Tot from a single tiny Transformer into a staged research artifact: a C++ implementation, matching Python and C comparison targets, a controlled benchmark suite, and finally a handmade MoE architecture where multiple existing models act as experts coordinated by a mediator.

## License

The original Tater Tot source code and project documentation are licensed under
the MIT License. See [LICENSE](LICENSE).

Training corpora under `data/` are input texts and retain their original source
terms. Verify those terms before redistributing the corpus files or derived
datasets.
