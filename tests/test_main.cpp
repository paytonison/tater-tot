#include "tater/checkpoint.hpp"
#include "tater/data.hpp"
#include "tater/model.hpp"
#include "tater/optimizer.hpp"
#include "tater/tensor.hpp"

#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect_true(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << "\n";
    }
}

void expect_near(double actual, double expected, double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        ++failures;
        std::cerr << "FAIL: " << message << " actual=" << actual << " expected=" << expected
                  << " tol=" << tolerance << "\n";
    }
}

void expect_throws_containing(const std::function<void()>& fn,
                              const std::string& expected,
                              const std::string& message) {
    try {
        fn();
        ++failures;
        std::cerr << "FAIL: " << message << " did not throw\n";
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(expected) == std::string::npos) {
            ++failures;
            std::cerr << "FAIL: " << message << " threw '" << error.what()
                      << "' without expected text '" << expected << "'\n";
        }
    }
}

bool any_nonzero(const std::vector<double>& values, double tolerance = 1e-12) {
    for (double value : values) {
        if (std::abs(value) > tolerance) {
            return true;
        }
    }
    return false;
}

bool any_difference(const std::vector<double>& lhs,
                    const std::vector<double>& rhs,
                    double tolerance = 1e-12) {
    if (lhs.size() != rhs.size()) {
        return true;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (std::abs(lhs[i] - rhs[i]) > tolerance) {
            return true;
        }
    }
    return false;
}

double finite_difference(const std::function<double(const std::vector<double>&)>& fn,
                         std::vector<double> values,
                         std::size_t index) {
    constexpr double eps = 1e-5;
    values[index] += eps;
    const double plus = fn(values);
    values[index] -= 2.0 * eps;
    const double minus = fn(values);
    return (plus - minus) / (2.0 * eps);
}

void test_tensor_shapes_and_basic_ops() {
    tater::Tensor a = tater::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    tater::Tensor b = tater::Tensor::from_data({10, 20, 30}, {3});
    tater::Tensor c = tater::add(a, b);

    expect_true(c.shape() == tater::Shape({2, 3}), "broadcast add shape");
    expect_near(c.data()[0], 11.0, 1e-12, "broadcast add data 0");
    expect_near(c.data()[5], 36.0, 1e-12, "broadcast add data 5");

    tater::Tensor loss = tater::mean(c);
    loss.backward();
    expect_near(b.grad()[0], 2.0 / 6.0, 1e-12, "broadcast bias grad 0");
    expect_near(b.grad()[2], 2.0 / 6.0, 1e-12, "broadcast bias grad 2");

    tater::Tensor m1 = tater::Tensor::from_data({1, 2, 3, 4, 5, 6}, {2, 3});
    tater::Tensor m2 = tater::Tensor::from_data({7, 8, 9, 10, 11, 12}, {3, 2});
    tater::Tensor mm = tater::matmul(m1, m2);
    expect_true(mm.shape() == tater::Shape({2, 2}), "matmul shape");
    expect_near(mm.data()[0], 58.0, 1e-12, "matmul data 0");
    expect_near(mm.data()[3], 154.0, 1e-12, "matmul data 3");

    tater::Tensor tr = tater::transpose(m1);
    expect_true(tr.shape() == tater::Shape({3, 2}), "transpose shape");
    expect_near(tr.data()[1], 4.0, 1e-12, "transpose data");

    tater::Tensor reshaped = tater::reshape(m1, {3, 2});
    expect_true(reshaped.shape() == tater::Shape({3, 2}), "reshape shape");
    expect_near(reshaped.data()[4], 5.0, 1e-12, "reshape data");
}

void test_add_gradient() {
    const std::vector<double> aval = {0.2, -0.4, 1.0};
    const std::vector<double> bval = {1.5, 2.0, -3.0};
    tater::Tensor a = tater::Tensor::from_data(aval, {3});
    tater::Tensor b = tater::Tensor::from_data(bval, {3});
    tater::Tensor loss = tater::mean(tater::add(a, b));
    loss.backward();

    auto fa = [&](const std::vector<double>& x) {
        return tater::mean(tater::add(tater::Tensor::from_data(x, {3}, false),
                                      tater::Tensor::from_data(bval, {3}, false)))
            .data()[0];
    };
    for (std::size_t i = 0; i < aval.size(); ++i) {
        expect_near(a.grad()[i], finite_difference(fa, aval, i), 1e-6, "add grad a");
    }
}

void test_multiply_gradient() {
    const std::vector<double> aval = {0.2, -0.4, 1.0};
    const std::vector<double> bval = {1.5, 2.0, -3.0};
    tater::Tensor a = tater::Tensor::from_data(aval, {3});
    tater::Tensor b = tater::Tensor::from_data(bval, {3});
    tater::Tensor loss = tater::mean(tater::multiply(a, b));
    loss.backward();

    auto fa = [&](const std::vector<double>& x) {
        return tater::mean(tater::multiply(tater::Tensor::from_data(x, {3}, false),
                                           tater::Tensor::from_data(bval, {3}, false)))
            .data()[0];
    };
    auto fb = [&](const std::vector<double>& x) {
        return tater::mean(tater::multiply(tater::Tensor::from_data(aval, {3}, false),
                                           tater::Tensor::from_data(x, {3}, false)))
            .data()[0];
    };
    for (std::size_t i = 0; i < aval.size(); ++i) {
        expect_near(a.grad()[i], finite_difference(fa, aval, i), 1e-6, "multiply grad a");
        expect_near(b.grad()[i], finite_difference(fb, bval, i), 1e-6, "multiply grad b");
    }
}

void test_matmul_gradient() {
    const std::vector<double> aval = {0.1, -0.2, 0.3, 0.4, -0.5, 0.6};
    const std::vector<double> bval = {-0.7, 0.8, 0.9, -1.0, 1.1, -1.2};
    tater::Tensor a = tater::Tensor::from_data(aval, {2, 3});
    tater::Tensor b = tater::Tensor::from_data(bval, {3, 2});
    tater::Tensor loss = tater::mean(tater::matmul(a, b));
    loss.backward();

    auto fa = [&](const std::vector<double>& x) {
        return tater::mean(tater::matmul(tater::Tensor::from_data(x, {2, 3}, false),
                                         tater::Tensor::from_data(bval, {3, 2}, false)))
            .data()[0];
    };
    auto fb = [&](const std::vector<double>& x) {
        return tater::mean(tater::matmul(tater::Tensor::from_data(aval, {2, 3}, false),
                                         tater::Tensor::from_data(x, {3, 2}, false)))
            .data()[0];
    };

    for (std::size_t i = 0; i < aval.size(); ++i) {
        expect_near(a.grad()[i], finite_difference(fa, aval, i), 1e-6, "matmul grad a");
    }
    for (std::size_t i = 0; i < bval.size(); ++i) {
        expect_near(b.grad()[i], finite_difference(fb, bval, i), 1e-6, "matmul grad b");
    }
}

void test_mean_gradient() {
    const std::vector<double> values = {1.0, 2.0, 4.0, 8.0};
    tater::Tensor x = tater::Tensor::from_data(values, {4});
    tater::Tensor loss = tater::mean(x);
    loss.backward();

    auto fn = [](const std::vector<double>& v) {
        return tater::mean(tater::Tensor::from_data(v, {4}, false)).data()[0];
    };
    for (std::size_t i = 0; i < values.size(); ++i) {
        expect_near(x.grad()[i], finite_difference(fn, values, i), 1e-6, "mean grad");
    }
}

void test_softmax_cross_entropy_gradient() {
    const std::vector<double> values = {1.0, -2.0, 0.5, 0.2, 1.2, -0.7};
    const std::vector<int> targets = {2, 0};
    tater::Tensor logits = tater::Tensor::from_data(values, {2, 3});
    tater::Tensor loss = tater::softmax_cross_entropy(logits, targets);
    loss.backward();

    auto fn = [&](const std::vector<double>& v) {
        return tater::softmax_cross_entropy(tater::Tensor::from_data(v, {2, 3}, false), targets)
            .data()[0];
    };
    for (std::size_t i = 0; i < values.size(); ++i) {
        expect_near(logits.grad()[i],
                    finite_difference(fn, values, i),
                    1e-6,
                    "softmax cross entropy grad");
    }
}

void test_gelu_gradient() {
    const std::vector<double> values = {-1.2, -0.1, 0.0, 0.7, 1.4};
    tater::Tensor x = tater::Tensor::from_data(values, {5});
    tater::Tensor loss = tater::mean(tater::gelu(x));
    loss.backward();

    auto fn = [](const std::vector<double>& v) {
        return tater::mean(tater::gelu(tater::Tensor::from_data(v, {5}, false))).data()[0];
    };
    for (std::size_t i = 0; i < values.size(); ++i) {
        expect_near(x.grad()[i], finite_difference(fn, values, i), 1e-6, "gelu grad");
    }
}

void test_layer_norm_gradient() {
    const std::vector<double> x_values = {0.2, -0.4, 1.1, 0.7, -0.8, 0.3};
    const std::vector<double> gamma_values = {1.0, 0.8, -0.5};
    const std::vector<double> beta_values = {0.1, -0.2, 0.3};
    tater::Tensor x = tater::Tensor::from_data(x_values, {2, 3});
    tater::Tensor gamma = tater::Tensor::from_data(gamma_values, {3});
    tater::Tensor beta = tater::Tensor::from_data(beta_values, {3});
    tater::Tensor loss = tater::mean(tater::layer_norm(x, gamma, beta));
    loss.backward();

    auto fx = [&](const std::vector<double>& v) {
        return tater::mean(tater::layer_norm(tater::Tensor::from_data(v, {2, 3}, false),
                                             tater::Tensor::from_data(gamma_values, {3}, false),
                                             tater::Tensor::from_data(beta_values, {3}, false)))
            .data()[0];
    };
    auto fgamma = [&](const std::vector<double>& v) {
        return tater::mean(tater::layer_norm(tater::Tensor::from_data(x_values, {2, 3}, false),
                                             tater::Tensor::from_data(v, {3}, false),
                                             tater::Tensor::from_data(beta_values, {3}, false)))
            .data()[0];
    };
    auto fbeta = [&](const std::vector<double>& v) {
        return tater::mean(tater::layer_norm(tater::Tensor::from_data(x_values, {2, 3}, false),
                                             tater::Tensor::from_data(gamma_values, {3}, false),
                                             tater::Tensor::from_data(v, {3}, false)))
            .data()[0];
    };

    for (std::size_t i = 0; i < x_values.size(); ++i) {
        expect_near(x.grad()[i], finite_difference(fx, x_values, i), 1e-5, "layer_norm grad x");
    }
    for (std::size_t i = 0; i < gamma_values.size(); ++i) {
        expect_near(gamma.grad()[i],
                    finite_difference(fgamma, gamma_values, i),
                    1e-6,
                    "layer_norm grad gamma");
        expect_near(beta.grad()[i],
                    finite_difference(fbeta, beta_values, i),
                    1e-6,
                    "layer_norm grad beta");
    }
}

void test_transformer_shape_loss_and_gradients() {
    tater::ModelConfig config;
    config.vocab_size = 5;
    config.context = 4;
    config.layers = 1;
    config.heads = 2;
    config.embed = 8;
    config.hidden = 16;
    tater::TinyCharModel model(config, 11);

    const std::vector<int> tokens = {0, 1, 2, 3, 1, 2, 3, 4};
    const std::vector<int> targets = {1, 2, 3, 4, 2, 3, 4, 0};
    tater::Tensor logits = model.forward(tokens, 2);
    expect_true(logits.shape() == tater::Shape({8, config.vocab_size}), "transformer logits shape");

    tater::Tensor loss = tater::softmax_cross_entropy(logits, targets);
    expect_true(std::isfinite(loss.data()[0]), "transformer loss is finite");
    loss.backward();

    for (const auto& [name, tensor] : model.named_parameters()) {
        for (double grad : tensor->grad()) {
            expect_true(std::isfinite(grad), "finite gradient for " + name);
        }
    }
}

void test_token_embedding_vectors_and_update() {
    tater::ModelConfig config;
    config.vocab_size = 5;
    config.context = 4;
    config.layers = 1;
    config.heads = 2;
    config.embed = 8;
    config.hidden = 16;
    tater::TinyCharModel model(config, 23);

    expect_true(model.token_embedding.table.shape() == tater::Shape({config.vocab_size, config.embed}),
                "token embedding table shape");
    expect_true(model.positional_embedding.table.shape() == tater::Shape({config.context, config.embed}),
                "positional embedding table shape");

    tater::Tensor token_vectors = model.token_embedding.forward({0, 1});
    expect_true(token_vectors.shape() == tater::Shape({2, config.embed}),
                "token IDs map to d_model vectors");

    bool distinct_rows = false;
    for (std::size_t d = 0; d < config.embed; ++d) {
        if (token_vectors.data()[d] != token_vectors.data()[config.embed + d]) {
            distinct_rows = true;
            break;
        }
    }
    expect_true(distinct_rows, "different token IDs have different initialized embeddings");

    const std::vector<double> before = model.token_embedding.table.data();
    tater::Adam optimizer(model.parameters(), 0.05);
    const std::vector<int> tokens = {0, 1, 2, 3, 1, 2, 3, 4};
    const std::vector<int> targets = {1, 2, 3, 4, 2, 3, 4, 0};
    tater::Tensor logits = model.forward(tokens, 2);
    tater::Tensor loss = tater::softmax_cross_entropy(logits, targets);
    optimizer.zero_grad();
    loss.backward();

    expect_true(any_nonzero(model.token_embedding.table.grad()),
                "token embedding receives gradients");
    optimizer.step(1.0);
    expect_true(any_difference(before, model.token_embedding.table.data()),
                "token embedding changes after optimizer step");
}

void test_embedding_bounds_checks() {
    tater::ModelConfig config;
    config.vocab_size = 4;
    config.context = 4;
    config.layers = 1;
    config.heads = 2;
    config.embed = 8;
    config.hidden = 16;
    tater::TinyCharModel model(config, 31);

    expect_throws_containing(
        [&model]() { (void)model.token_embedding.forward({-1}); },
        "token_id out of range",
        "negative token id is rejected");
    expect_throws_containing(
        [&model]() { (void)model.forward({0, 1, 2, 4}, 1); },
        "token_id out of range",
        "token id at vocab_size is rejected");
    expect_throws_containing(
        [&model, &config]() { (void)model.positional_embedding.forward(1, config.context + 1); },
        "position out of range",
        "position beyond max_seq_len is rejected");
}

void test_full_position_batch_targets() {
    const std::vector<int> data = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    std::mt19937 rng(5);
    tater::Batch batch = tater::make_batch(data, 4, 2, rng);
    expect_true(batch.x.size() == 8, "batch input count");
    expect_true(batch.y.size() == 8, "batch target count");
}

void test_checkpoint_round_trip() {
    const std::string text = "tatertot";
    tater::Vocabulary vocab = tater::Vocabulary::from_text(text);
    tater::ModelConfig config;
    config.vocab_size = vocab.size();
    config.context = 4;
    config.layers = 1;
    config.heads = 2;
    config.embed = 8;
    config.hidden = 16;
    tater::TinyCharModel model(config, 123);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "tater_tot_checkpoint_test.bin";
    tater::save_checkpoint(path.string(), model, vocab);
    tater::LoadedCheckpoint loaded = tater::load_checkpoint(path.string());
    std::filesystem::remove(path);

    expect_true(loaded.model.config().context == config.context, "checkpoint context");
    expect_true(loaded.model.config().layers == config.layers, "checkpoint layers");
    expect_true(loaded.model.config().heads == config.heads, "checkpoint heads");
    expect_true(loaded.vocab.chars == vocab.chars, "checkpoint vocab");
    expect_near(loaded.model.blocks[0].q_w.data()[3],
                model.blocks[0].q_w.data()[3],
                1e-12,
                "checkpoint parameter");
}

void test_tiny_overfit() {
    std::string text;
    for (int i = 0; i < 120; ++i) {
        text += "ab";
    }

    tater::Vocabulary vocab = tater::Vocabulary::from_text(text);
    std::vector<int> data = vocab.encode(text);

    tater::ModelConfig config;
    config.vocab_size = vocab.size();
    config.context = 4;
    config.layers = 1;
    config.heads = 2;
    config.embed = 8;
    config.hidden = 16;
    tater::TinyCharModel model(config, 7);
    tater::Adam optimizer(model.parameters(), 0.02);
    std::mt19937 rng(99);

    std::mt19937 eval_rng_initial(123);
    const double initial = tater::estimate_loss(model, data, 8, 4, eval_rng_initial);
    for (int step = 0; step < 160; ++step) {
        tater::Batch batch = tater::make_batch(data, config.context, 8, rng);
        tater::Tensor logits = model.forward(batch.x, batch.batch_size);
        tater::Tensor loss = tater::softmax_cross_entropy(logits, batch.y);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step(1.0);
    }
    std::mt19937 eval_rng_final(123);
    const double final = tater::estimate_loss(model, data, 8, 4, eval_rng_final);
    expect_true(final < initial, "tiny repeated text overfit loss decreases");
    expect_true(final < 0.45, "tiny repeated text overfit absolute loss");
}

void test_generation_smoke() {
    const std::string text = "abababababababab";
    tater::Vocabulary vocab = tater::Vocabulary::from_text(text);
    tater::ModelConfig config;
    config.vocab_size = vocab.size();
    config.context = 4;
    config.layers = 1;
    config.heads = 2;
    config.embed = 8;
    config.hidden = 16;
    tater::TinyCharModel model(config, 17);
    tater::GenerationOptions options;
    options.tokens = 8;
    options.temperature = 1.0;
    std::mt19937 rng(19);
    const std::string generated = tater::generate_text(model, vocab, "ab", options, rng);
    expect_true(generated.size() == 10, "generation returns prompt plus requested tokens");
}

} // namespace

int main() {
    try {
        test_tensor_shapes_and_basic_ops();
        test_add_gradient();
        test_multiply_gradient();
        test_matmul_gradient();
        test_mean_gradient();
        test_softmax_cross_entropy_gradient();
        test_gelu_gradient();
        test_layer_norm_gradient();
        test_transformer_shape_loss_and_gradients();
        test_token_embedding_vectors_and_update();
        test_embedding_bounds_checks();
        test_full_position_batch_targets();
        test_checkpoint_round_trip();
        test_tiny_overfit();
        test_generation_smoke();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "UNCAUGHT: " << error.what() << "\n";
    }

    if (failures != 0) {
        std::cerr << failures << " test failure(s)\n";
        return 1;
    }
    std::cout << "all tests passed\n";
    return 0;
}
