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

void test_checkpoint_round_trip() {
    const std::string text = "tatertot";
    tater::Vocabulary vocab = tater::Vocabulary::from_text(text);
    tater::ModelConfig config;
    config.vocab_size = vocab.size();
    config.context = 4;
    config.embed = 8;
    config.hidden = 12;
    tater::TinyCharModel model(config, 123);

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "tater_tot_checkpoint_test.bin";
    tater::save_checkpoint(path.string(), model, vocab);
    tater::LoadedCheckpoint loaded = tater::load_checkpoint(path.string());
    std::filesystem::remove(path);

    expect_true(loaded.model.config().context == config.context, "checkpoint context");
    expect_true(loaded.vocab.chars == vocab.chars, "checkpoint vocab");
    expect_near(loaded.model.w1.data()[3], model.w1.data()[3], 1e-12, "checkpoint parameter");
}

void test_tiny_overfit() {
    std::string text;
    for (int i = 0; i < 120; ++i) {
        text += "abc";
    }

    tater::Vocabulary vocab = tater::Vocabulary::from_text(text);
    std::vector<int> data = vocab.encode(text);

    tater::ModelConfig config;
    config.vocab_size = vocab.size();
    config.context = 3;
    config.embed = 8;
    config.hidden = 16;
    tater::TinyCharModel model(config, 7);
    tater::Adam optimizer(model.parameters(), 0.04);
    std::mt19937 rng(99);

    const double initial = tater::estimate_loss(model, data, 16, 4, rng);
    for (int step = 0; step < 240; ++step) {
        tater::Batch batch = tater::make_batch(data, config.context, 16, rng);
        tater::Tensor logits = model.forward(batch.x, batch.batch_size);
        tater::Tensor loss = tater::softmax_cross_entropy(logits, batch.y);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step(1.0);
    }
    const double final = tater::estimate_loss(model, data, 16, 4, rng);
    expect_true(final < 0.20, "tiny repeated text overfit absolute loss");
    expect_true(final < initial * 0.25, "tiny repeated text overfit relative loss");
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
        test_checkpoint_round_trip();
        test_tiny_overfit();
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

