# Tater Tot

A tiny character-level decoder-only Transformer language model implemented from
scratch in C++.

Tater Tot is a compact GPT-style language model built to show the core mechanics
of modern autoregressive Transformers without relying on PyTorch, TensorFlow,
ONNX Runtime, CUDA, or another machine learning framework. It is intentionally
small, educational, and architecture-focused: the point is to make the pieces of
a Transformer visible in ordinary C++20.

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
CMakeLists.txt
  CMake build configuration for the library, training binary, generation binary,
  and test binary.

include/tater/tensor.hpp
src/tensor.cpp
  Tensor object, shapes, reverse-mode autodiff, broadcasting, matrix multiply,
  embeddings, GELU, LayerNorm, causal self-attention, softmax cross-entropy,
  gradient zeroing, and gradient clipping.

include/tater/model.hpp
src/model.cpp
  ModelConfig, TinyCharModel, Transformer block parameters, initialization,
  forward pass, named parameter access, sampling, text generation, and loss
  estimation.

include/tater/data.hpp
src/data.cpp
  Character vocabulary construction, encoding/decoding, text loading,
  train/validation splitting, and full-position batch creation.

include/tater/optimizer.hpp
src/optimizer.cpp
  Adam optimizer implementation.

include/tater/checkpoint.hpp
src/checkpoint.cpp
  Checkpoint save/load for Transformer config, vocabulary, and named parameter
  tensors.

examples/train.cpp
  `tater_train` CLI entry point and training loop.

examples/generate.cpp
  `tater_generate` CLI entry point for loading a checkpoint and sampling text.

tests/test_main.cpp
  Unit tests, gradient checks, Transformer shape checks, checkpoint round-trip,
  tiny overfit smoke test, and generation smoke test.
```

## Build

The project uses CMake and the C++20 standard library.

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The build creates:

- `build/tater_train`
- `build/tater_generate`
- `build/tater_tests`

## Train

Put a plain text file at `data/input.txt`, or pass another text path with
`--data`.

```sh
./build/tater_train \
  --data data/input.txt \
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
  --checkpoint checkpoints/model.bin \
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
./build/tater_generate \
  --checkpoint checkpoints/model.bin \
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
ctest --test-dir build --output-on-failure
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
- a small benchmark corpus and repeatable benchmark script
