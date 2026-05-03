#include "tater/model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace tater {
namespace {

Tensor random_tensor(const Shape& shape, double stddev, std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, stddev);
    std::vector<double> values(numel(shape));
    for (double& value : values) {
        value = dist(rng);
    }
    return Tensor::from_data(std::move(values), shape, true);
}

double xavier_stddev(std::size_t fan_in, std::size_t fan_out) {
    return std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
}

} // namespace

TinyCharModel::TinyCharModel(ModelConfig config, std::uint32_t seed) : config_(config) {
    if (config_.vocab_size == 0 || config_.context == 0 || config_.embed == 0 || config_.hidden == 0) {
        throw std::runtime_error("model config values must be non-zero");
    }

    std::mt19937 rng(seed);
    token_embedding = random_tensor({config_.vocab_size, config_.embed}, 0.02, rng);
    positional_embedding = random_tensor({config_.context, config_.embed}, 0.02, rng);
    w1 = random_tensor({config_.context * config_.embed, config_.hidden},
                       xavier_stddev(config_.context * config_.embed, config_.hidden),
                       rng);
    b1 = Tensor({config_.hidden}, 0.0, true);
    w2 = random_tensor({config_.hidden, config_.vocab_size},
                       xavier_stddev(config_.hidden, config_.vocab_size),
                       rng);
    b2 = Tensor({config_.vocab_size}, 0.0, true);
}

const ModelConfig& TinyCharModel::config() const {
    return config_;
}

Tensor TinyCharModel::forward(const std::vector<int>& tokens, std::size_t batch_size) const {
    if (tokens.size() != batch_size * config_.context) {
        throw std::runtime_error("model input token count does not match batch_size * context");
    }

    std::vector<int> positions(tokens.size());
    for (std::size_t b = 0; b < batch_size; ++b) {
        for (std::size_t t = 0; t < config_.context; ++t) {
            positions[b * config_.context + t] = static_cast<int>(t);
        }
    }

    Tensor token_vectors = embedding_lookup(token_embedding, tokens);
    Tensor position_vectors = embedding_lookup(positional_embedding, positions);
    Tensor x = add(token_vectors, position_vectors);
    Tensor flat = reshape(x, {batch_size, config_.context * config_.embed});
    Tensor hidden = tanh(add(matmul(flat, w1), b1));
    return add(matmul(hidden, w2), b2);
}

std::vector<Tensor*> TinyCharModel::parameters() {
    return {&token_embedding, &positional_embedding, &w1, &b1, &w2, &b2};
}

std::vector<const Tensor*> TinyCharModel::parameters() const {
    return {&token_embedding, &positional_embedding, &w1, &b1, &w2, &b2};
}

std::vector<std::pair<std::string, Tensor*>> TinyCharModel::named_parameters() {
    return {{"token_embedding", &token_embedding},
            {"positional_embedding", &positional_embedding},
            {"w1", &w1},
            {"b1", &b1},
            {"w2", &w2},
            {"b2", &b2}};
}

std::vector<std::pair<std::string, const Tensor*>> TinyCharModel::named_parameters() const {
    return {{"token_embedding", &token_embedding},
            {"positional_embedding", &positional_embedding},
            {"w1", &w1},
            {"b1", &b1},
            {"w2", &w2},
            {"b2", &b2}};
}

int sample_from_logits(const std::vector<double>& logits,
                       double temperature,
                       int top_k,
                       std::mt19937& rng) {
    if (logits.empty()) {
        throw std::runtime_error("cannot sample from empty logits");
    }
    temperature = std::max(temperature, 1e-6);

    std::vector<int> candidates(logits.size());
    std::iota(candidates.begin(), candidates.end(), 0);
    if (top_k > 0 && static_cast<std::size_t>(top_k) < candidates.size()) {
        std::partial_sort(candidates.begin(),
                          candidates.begin() + top_k,
                          candidates.end(),
                          [&logits](int lhs, int rhs) { return logits[lhs] > logits[rhs]; });
        candidates.resize(static_cast<std::size_t>(top_k));
    }

    double max_scaled = -std::numeric_limits<double>::infinity();
    for (int id : candidates) {
        max_scaled = std::max(max_scaled, logits[static_cast<std::size_t>(id)] / temperature);
    }

    std::vector<double> weights;
    weights.reserve(candidates.size());
    for (int id : candidates) {
        weights.push_back(std::exp(logits[static_cast<std::size_t>(id)] / temperature - max_scaled));
    }

    std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
    return candidates[dist(rng)];
}

std::string generate_text(const TinyCharModel& model,
                          const Vocabulary& vocab,
                          const std::string& prompt,
                          const GenerationOptions& options,
                          std::mt19937& rng) {
    std::vector<int> ids = vocab.encode(prompt, true);
    if (ids.empty()) {
        ids.push_back(0);
    }

    for (std::size_t step = 0; step < options.tokens; ++step) {
        std::vector<int> context(model.config().context, 0);
        const std::size_t copy_count = std::min(model.config().context, ids.size());
        const std::size_t src_start = ids.size() - copy_count;
        const std::size_t dst_start = model.config().context - copy_count;
        for (std::size_t i = 0; i < copy_count; ++i) {
            context[dst_start + i] = ids[src_start + i];
        }

        Tensor logits = model.forward(context, 1);
        const int next = sample_from_logits(logits.data(), options.temperature, options.top_k, rng);
        ids.push_back(next);
    }

    return vocab.decode(ids);
}

double estimate_loss(const TinyCharModel& model,
                     const std::vector<int>& data,
                     std::size_t batch_size,
                     std::size_t batches,
                     std::mt19937& rng) {
    if (batches == 0) {
        return 0.0;
    }

    double total = 0.0;
    for (std::size_t i = 0; i < batches; ++i) {
        Batch batch = make_batch(data, model.config().context, batch_size, rng);
        Tensor logits = model.forward(batch.x, batch.batch_size);
        Tensor loss = softmax_cross_entropy(logits, batch.y);
        total += loss.data()[0];
    }
    return total / static_cast<double>(batches);
}

} // namespace tater
