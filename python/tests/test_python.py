from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tater


def finite_difference(fn, values: list[float], index: int) -> float:
    eps = 1e-5
    shifted = list(values)
    shifted[index] += eps
    plus = fn(shifted)
    shifted[index] -= 2.0 * eps
    minus = fn(shifted)
    return (plus - minus) / (2.0 * eps)


def any_nonzero(values: list[float], tolerance: float = 1e-12) -> bool:
    return any(abs(value) > tolerance for value in values)


def any_difference(
    lhs: list[float], rhs: list[float], tolerance: float = 1e-12
) -> bool:
    return len(lhs) != len(rhs) or any(
        abs(left - right) > tolerance for left, right in zip(lhs, rhs)
    )


class PythonTaterTests(unittest.TestCase):
    def test_tensor_shapes_and_basic_ops(self) -> None:
        a = tater.Tensor.from_data([1, 2, 3, 4, 5, 6], (2, 3))
        b = tater.Tensor.from_data([10, 20, 30], (3,))
        c = tater.add(a, b)

        self.assertEqual(c.shape, (2, 3))
        self.assertAlmostEqual(c.data[0], 11.0)
        self.assertAlmostEqual(c.data[5], 36.0)

        loss = tater.mean(c)
        loss.backward()
        self.assertAlmostEqual(b.grad[0], 2.0 / 6.0)
        self.assertAlmostEqual(b.grad[2], 2.0 / 6.0)

        m1 = tater.Tensor.from_data([1, 2, 3, 4, 5, 6], (2, 3))
        m2 = tater.Tensor.from_data([7, 8, 9, 10, 11, 12], (3, 2))
        mm = tater.matmul(m1, m2)
        self.assertEqual(mm.shape, (2, 2))
        self.assertAlmostEqual(mm.data[0], 58.0)
        self.assertAlmostEqual(mm.data[3], 154.0)

        tr = tater.transpose(m1)
        self.assertEqual(tr.shape, (3, 2))
        self.assertAlmostEqual(tr.data[1], 4.0)

        reshaped = tater.reshape(m1, (3, 2))
        self.assertEqual(reshaped.shape, (3, 2))
        self.assertAlmostEqual(reshaped.data[4], 5.0)

    def test_matmul_gradient(self) -> None:
        aval = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
        bval = [-0.7, 0.8, 0.9, -1.0, 1.1, -1.2]
        a = tater.Tensor.from_data(aval, (2, 3))
        b = tater.Tensor.from_data(bval, (3, 2))
        loss = tater.mean(tater.matmul(a, b))
        loss.backward()

        def fa(values: list[float]) -> float:
            return tater.mean(
                tater.matmul(
                    tater.Tensor.from_data(values, (2, 3), False),
                    tater.Tensor.from_data(bval, (3, 2), False),
                )
            ).data[0]

        def fb(values: list[float]) -> float:
            return tater.mean(
                tater.matmul(
                    tater.Tensor.from_data(aval, (2, 3), False),
                    tater.Tensor.from_data(values, (3, 2), False),
                )
            ).data[0]

        for i in range(len(aval)):
            self.assertAlmostEqual(a.grad[i], finite_difference(fa, aval, i), places=6)
        for i in range(len(bval)):
            self.assertAlmostEqual(b.grad[i], finite_difference(fb, bval, i), places=6)

    def test_softmax_cross_entropy_gradient(self) -> None:
        values = [1.0, -2.0, 0.5, 0.2, 1.2, -0.7]
        targets = [2, 0]
        logits = tater.Tensor.from_data(values, (2, 3))
        loss = tater.softmax_cross_entropy(logits, targets)
        loss.backward()

        def fn(v: list[float]) -> float:
            return tater.softmax_cross_entropy(
                tater.Tensor.from_data(v, (2, 3), False), targets
            ).data[0]

        for i in range(len(values)):
            self.assertAlmostEqual(
                logits.grad[i], finite_difference(fn, values, i), places=6
            )

    def test_gelu_and_layer_norm_gradients(self) -> None:
        gelu_values = [-1.2, -0.1, 0.0, 0.7, 1.4]
        x = tater.Tensor.from_data(gelu_values, (5,))
        loss = tater.mean(tater.gelu(x))
        loss.backward()

        def gelu_fn(values: list[float]) -> float:
            return tater.mean(tater.gelu(tater.Tensor.from_data(values, (5,), False))).data[0]

        for i in range(len(gelu_values)):
            self.assertAlmostEqual(
                x.grad[i], finite_difference(gelu_fn, gelu_values, i), places=6
            )

        x_values = [0.2, -0.4, 1.1, 0.7, -0.8, 0.3]
        gamma_values = [1.0, 0.8, -0.5]
        beta_values = [0.1, -0.2, 0.3]
        x = tater.Tensor.from_data(x_values, (2, 3))
        gamma = tater.Tensor.from_data(gamma_values, (3,))
        beta = tater.Tensor.from_data(beta_values, (3,))
        loss = tater.mean(tater.layer_norm(x, gamma, beta))
        loss.backward()

        def fx(values: list[float]) -> float:
            return tater.mean(
                tater.layer_norm(
                    tater.Tensor.from_data(values, (2, 3), False),
                    tater.Tensor.from_data(gamma_values, (3,), False),
                    tater.Tensor.from_data(beta_values, (3,), False),
                )
            ).data[0]

        for i in range(len(x_values)):
            self.assertAlmostEqual(
                x.grad[i], finite_difference(fx, x_values, i), places=5
            )

    def test_transformer_shape_loss_and_gradients(self) -> None:
        config = tater.ModelConfig(
            vocab_size=5, context=4, layers=1, heads=2, embed=8, hidden=16
        )
        model = tater.TinyCharModel(config, 11)

        tokens = [0, 1, 2, 3, 1, 2, 3, 4]
        targets = [1, 2, 3, 4, 2, 3, 4, 0]
        logits = model.forward(tokens, 2)
        self.assertEqual(logits.shape, (8, config.vocab_size))

        loss = tater.softmax_cross_entropy(logits, targets)
        self.assertTrue(abs(loss.data[0]) < float("inf"))
        loss.backward()

        for name, tensor in model.named_parameters():
            for grad in tensor.grad:
                self.assertTrue(abs(grad) < float("inf"), name)

    def test_token_embedding_vectors_and_update(self) -> None:
        config = tater.ModelConfig(
            vocab_size=5, context=4, layers=1, heads=2, embed=8, hidden=16
        )
        model = tater.TinyCharModel(config, 23)

        self.assertEqual(model.token_embedding.table.shape, (config.vocab_size, config.embed))
        self.assertEqual(
            model.positional_embedding.table.shape, (config.context, config.embed)
        )

        token_vectors = model.token_embedding.forward([0, 1])
        self.assertEqual(token_vectors.shape, (2, config.embed))
        self.assertNotEqual(
            token_vectors.data[: config.embed],
            token_vectors.data[config.embed : 2 * config.embed],
        )

        before = list(model.token_embedding.table.data)
        optimizer = tater.Adam(model.parameters(), 0.05)
        tokens = [0, 1, 2, 3, 1, 2, 3, 4]
        targets = [1, 2, 3, 4, 2, 3, 4, 0]
        logits = model.forward(tokens, 2)
        loss = tater.softmax_cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()

        self.assertTrue(any_nonzero(model.token_embedding.table.grad))
        optimizer.step(1.0)
        self.assertTrue(any_difference(before, model.token_embedding.table.data))

    def test_embedding_bounds_checks(self) -> None:
        config = tater.ModelConfig(
            vocab_size=4, context=4, layers=1, heads=2, embed=8, hidden=16
        )
        model = tater.TinyCharModel(config, 31)

        with self.assertRaisesRegex(ValueError, "token_id out of range"):
            model.token_embedding.forward([-1])
        with self.assertRaisesRegex(ValueError, "token_id out of range"):
            model.forward([0, 1, 2, 4], 1)
        with self.assertRaisesRegex(ValueError, "position out of range"):
            model.positional_embedding.forward(1, config.context + 1)

    def test_data_batch_and_signed_char_vocab_order(self) -> None:
        data = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        batch = tater.make_batch(data, 4, 2, random.Random(5))
        self.assertEqual(len(batch.x), 8)
        self.assertEqual(len(batch.y), 8)

        vocab = tater.Vocabulary.from_text(bytes([0, 65, 128, 255]))
        self.assertEqual(vocab.chars, [128, 255, 0, 65])

    def test_checkpoint_round_trip(self) -> None:
        text = b"tatertot"
        vocab = tater.Vocabulary.from_text(text)
        config = tater.ModelConfig(
            vocab_size=vocab.size(), context=4, layers=1, heads=2, embed=8, hidden=16
        )
        model = tater.TinyCharModel(config, 123)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tater_tot_checkpoint_test.bin"
            tater.save_checkpoint(path, model, vocab)
            loaded = tater.load_checkpoint(path)

        self.assertEqual(loaded.model.config.context, config.context)
        self.assertEqual(loaded.model.config.layers, config.layers)
        self.assertEqual(loaded.model.config.heads, config.heads)
        self.assertEqual(loaded.vocab.chars, vocab.chars)
        self.assertAlmostEqual(
            loaded.model.blocks[0].q_w.data[3], model.blocks[0].q_w.data[3]
        )

    def test_generation_smoke(self) -> None:
        text = b"abababababababab"
        vocab = tater.Vocabulary.from_text(text)
        config = tater.ModelConfig(
            vocab_size=vocab.size(), context=4, layers=1, heads=2, embed=8, hidden=16
        )
        model = tater.TinyCharModel(config, 17)
        options = tater.GenerationOptions(tokens=8, temperature=1.0)
        generated = tater.generate_text(model, vocab, b"ab", options, random.Random(19))
        self.assertEqual(len(generated), 10)


if __name__ == "__main__":
    unittest.main()
