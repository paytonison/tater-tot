# Tater Tot

Tater Tot is a small educational C++20 character-level language model. It trains
on a plain text file, learns to predict the next character from a short context
window, saves a checkpoint, and generates sample text autoregressively.

This is a portfolio-quality proof of concept for the core mechanics of a tiny
language model. It is not a competitive LLM, not GPU accelerated, and not built
on PyTorch, TensorFlow, ONNX Runtime, llama.cpp, or any other machine learning
framework.

## Build

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The project uses only the C++ standard library.

## Train

Put a text file at `data/input.txt`, then run:

```sh
./build/tater_train --data data/input.txt --steps 1000 --context 64 --embed 64
```

Useful options:

```text
--hidden 128
--batch 16
--lr 0.003
--clip 1.0
--print-every 50
--sample-every 200
--checkpoint checkpoints/model.bin
--seed 1337
```

Training prints the current mini-batch loss, a validation estimate, gradient
norm, and periodic samples. The checkpoint is written at the end.

## Generate

```sh
./build/tater_generate --checkpoint checkpoints/model.bin \
  --prompt "Once upon a time" \
  --tokens 300 \
  --temperature 0.9 \
  --top-k 20
```

`temperature` controls randomness. `top-k` limits sampling to the most likely
characters when it is greater than zero.

## Architecture

The first model is intentionally simple:

1. Token embedding table: `vocab_size x embed`
2. Positional embedding table: `context x embed`
3. Add token and positional embeddings
4. Flatten the context
5. One tanh hidden layer
6. Output projection to vocabulary logits
7. Stable softmax cross-entropy for next-character prediction

The model predicts one next character per training example from a fixed context
window. The code is organized so a small masked self-attention block can be added
later without changing the CLI or checkpoint boundary.

## Training Loop

Training does the following:

1. Load a plain text file as bytes/chars.
2. Build a sorted character vocabulary.
3. Encode the text to integer token IDs.
4. Split into train and validation sequences.
5. Draw random mini-batches of `context` characters and one target character.
6. Run the model forward.
7. Compute stable softmax cross-entropy with log-sum-exp.
8. Backpropagate through the computation graph.
9. Clip global gradient norm.
10. Update parameters with Adam.
11. Periodically generate sample text.

## Autodiff Engine

`include/tater/tensor.hpp` and `src/tensor.cpp` implement a small reverse-mode
autodiff engine. A `Tensor` owns:

- row-major `data`
- matching `grad`
- a `shape`
- parent tensors that produced it
- a backward function for its local operation

Each operation creates a result tensor and records parent links. Calling
`backward()` on a scalar loss builds a topological ordering from the loss back to
leaf tensors, seeds the loss gradient with `1`, then walks the graph in reverse
topological order. Each local backward function adds gradient contributions into
its parents.

Implemented ops include addition with simple broadcasting, multiplication,
matrix multiplication, sum, mean, tanh, ReLU, exp, log, transpose, reshape,
embedding lookup, and stable softmax cross-entropy.

## Checkpoint Format

Checkpoints are simple binary files containing:

1. Magic string: `TATER_TOT_CHECKPOINT_V1`
2. Model config: vocabulary size, context length, embedding size, hidden size
3. Vocabulary characters
4. Named parameter tensors with shape and double-precision values

This format is meant to be easy to inspect in code, not portable across every
machine architecture.

## Tests

The test executable covers:

- tensor shape handling
- basic operation outputs
- finite-difference gradient checks for addition, multiplication, matrix
  multiplication, mean, and softmax cross-entropy
- checkpoint save/load round trip
- a tiny repeated-text overfitting smoke test

Run it with:

```sh
./build/tater_tests
```

## Known Limitations

- Character handling is byte-oriented, not full Unicode code point tokenization.
- The model predicts one next character per context window rather than every
  position in a block.
- The implementation favors clarity over speed and does not use SIMD, BLAS, or
  multithreading.
- There is no CUDA, GPU backend, or external tensor library.
- The architecture is a tiny MLP, not a Transformer.
- Checkpoints do not store optimizer state.
