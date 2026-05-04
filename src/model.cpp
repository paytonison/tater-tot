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

Tensor ones(const Shape& shape) {
    return Tensor(shape, 1.0, true);
}

Tensor zeros(const Shape& shape) {
    return Tensor(shape, 0.0, true);
}

void validate_config(const ModelConfig& config) {
    if (config.vocab_size == 0) {
        throw std::runtime_error("model config vocab_size must be non-zero");
    }
    if (config.context == 0) {
        throw std::runtime_error("model config context must be non-zero");
    }
    if (config.embed == 0) {
        throw std::runtime_error("model config embed must be non-zero");
    }
    if (config.hidden == 0) {
        throw std::runtime_error("model config hidden must be non-zero");
    }
    if (config.layers == 0) {
        throw std::runtime_error("model config layers must be non-zero");
    }
    if (config.heads == 0) {
        throw std::runtime_error("model config heads must be non-zero");
    }
    if (config.embed % config.heads != 0) {
        throw std::runtime_error("model config embed must be divisible by heads");
    }
}

std::vector<std::pair<std::string, TinyCharModel::TransformerBlock*>>
mutable_blocks(std::vector<TinyCharModel::TransformerBlock>& blocks) {
    std::vector<std::pair<std::string, TinyCharModel::TransformerBlock*>> named;
    named.reserve(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        named.push_back({"blocks." + std::to_string(i), &blocks[i]});
    }
    return named;
}

std::vector<std::pair<std::string, const TinyCharModel::TransformerBlock*>>
const_blocks(const std::vector<TinyCharModel::TransformerBlock>& blocks) {
    std::vector<std::pair<std::string, const TinyCharModel::TransformerBlock*>> named;
    named.reserve(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        named.push_back({"blocks." + std::to_string(i), &blocks[i]});
    }
    return named;
}

} // namespace

TinyCharModel::TinyCharModel(ModelConfig config, std::uint32_t seed) : config_(config) {
    validate_config(config_);

    std::mt19937 rng(seed);
    token_embedding = random_tensor({config_.vocab_size, config_.embed}, 0.02, rng);
    positional_embedding = random_tensor({config_.context, config_.embed}, 0.02, rng);

    blocks.reserve(config_.layers);
    for (std::size_t layer = 0; layer < config_.layers; ++layer) {
        TransformerBlock block;
        block.ln1_gamma = ones({config_.embed});
        block.ln1_beta = zeros({config_.embed});
        block.q_w = random_tensor({config_.embed, config_.embed},
                                  xavier_stddev(config_.embed, config_.embed),
                                  rng);
        block.q_b = zeros({config_.embed});
        block.k_w = random_tensor({config_.embed, config_.embed},
                                  xavier_stddev(config_.embed, config_.embed),
                                  rng);
        block.k_b = zeros({config_.embed});
        block.v_w = random_tensor({config_.embed, config_.embed},
                                  xavier_stddev(config_.embed, config_.embed),
                                  rng);
        block.v_b = zeros({config_.embed});
        block.out_w = random_tensor({config_.embed, config_.embed},
                                    xavier_stddev(config_.embed, config_.embed),
                                    rng);
        block.out_b = zeros({config_.embed});
        block.ln2_gamma = ones({config_.embed});
        block.ln2_beta = zeros({config_.embed});
        block.ff1_w = random_tensor({config_.embed, config_.hidden},
                                    xavier_stddev(config_.embed, config_.hidden),
                                    rng);
        block.ff1_b = zeros({config_.hidden});
        block.ff2_w = random_tensor({config_.hidden, config_.embed},
                                    xavier_stddev(config_.hidden, config_.embed),
                                    rng);
        block.ff2_b = zeros({config_.embed});
        blocks.push_back(std::move(block));
    }

    ln_f_gamma = ones({config_.embed});
    ln_f_beta = zeros({config_.embed});
    lm_head_w = random_tensor({config_.embed, config_.vocab_size},
                              xavier_stddev(config_.embed, config_.vocab_size),
                              rng);
    lm_head_b = zeros({config_.vocab_size});
}

const ModelConfig& TinyCharModel::config() const {
    return config_;
}

Tensor TinyCharModel::forward(const std::vector<int>& tokens, std::size_t batch_size) const {
    if (batch_size == 0) {
        throw std::runtime_error("model forward batch_size must be non-zero");
    }
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

    for (const TransformerBlock& block : blocks) {
        Tensor normalized = layer_norm(x, block.ln1_gamma, block.ln1_beta);
        Tensor attention =
            causal_self_attention(normalized,
                                  block.q_w,
                                  block.q_b,
                                  block.k_w,
                                  block.k_b,
                                  block.v_w,
                                  block.v_b,
                                  block.out_w,
                                  block.out_b,
                                  batch_size,
                                  config_.context,
                                  config_.heads);
        x = add(x, attention);

        normalized = layer_norm(x, block.ln2_gamma, block.ln2_beta);
        Tensor ff = add(matmul(normalized, block.ff1_w), block.ff1_b);
        ff = gelu(ff);
        ff = add(matmul(ff, block.ff2_w), block.ff2_b);
        x = add(x, ff);
    }

    x = layer_norm(x, ln_f_gamma, ln_f_beta);
    return add(matmul(x, lm_head_w), lm_head_b);
}

std::vector<Tensor*> TinyCharModel::parameters() {
    std::vector<Tensor*> params;
    params.reserve(2 + blocks.size() * 16 + 4);
    params.push_back(&token_embedding);
    params.push_back(&positional_embedding);
    for (TransformerBlock& block : blocks) {
        params.push_back(&block.ln1_gamma);
        params.push_back(&block.ln1_beta);
        params.push_back(&block.q_w);
        params.push_back(&block.q_b);
        params.push_back(&block.k_w);
        params.push_back(&block.k_b);
        params.push_back(&block.v_w);
        params.push_back(&block.v_b);
        params.push_back(&block.out_w);
        params.push_back(&block.out_b);
        params.push_back(&block.ln2_gamma);
        params.push_back(&block.ln2_beta);
        params.push_back(&block.ff1_w);
        params.push_back(&block.ff1_b);
        params.push_back(&block.ff2_w);
        params.push_back(&block.ff2_b);
    }
    params.push_back(&ln_f_gamma);
    params.push_back(&ln_f_beta);
    params.push_back(&lm_head_w);
    params.push_back(&lm_head_b);
    return params;
}

std::vector<const Tensor*> TinyCharModel::parameters() const {
    std::vector<const Tensor*> params;
    params.reserve(2 + blocks.size() * 16 + 4);
    params.push_back(&token_embedding);
    params.push_back(&positional_embedding);
    for (const TransformerBlock& block : blocks) {
        params.push_back(&block.ln1_gamma);
        params.push_back(&block.ln1_beta);
        params.push_back(&block.q_w);
        params.push_back(&block.q_b);
        params.push_back(&block.k_w);
        params.push_back(&block.k_b);
        params.push_back(&block.v_w);
        params.push_back(&block.v_b);
        params.push_back(&block.out_w);
        params.push_back(&block.out_b);
        params.push_back(&block.ln2_gamma);
        params.push_back(&block.ln2_beta);
        params.push_back(&block.ff1_w);
        params.push_back(&block.ff1_b);
        params.push_back(&block.ff2_w);
        params.push_back(&block.ff2_b);
    }
    params.push_back(&ln_f_gamma);
    params.push_back(&ln_f_beta);
    params.push_back(&lm_head_w);
    params.push_back(&lm_head_b);
    return params;
}

std::vector<std::pair<std::string, Tensor*>> TinyCharModel::named_parameters() {
    std::vector<std::pair<std::string, Tensor*>> params;
    params.reserve(2 + blocks.size() * 16 + 4);
    params.push_back({"token_embedding", &token_embedding});
    params.push_back({"positional_embedding", &positional_embedding});
    for (auto [prefix, block] : mutable_blocks(blocks)) {
        params.push_back({prefix + ".ln1_gamma", &block->ln1_gamma});
        params.push_back({prefix + ".ln1_beta", &block->ln1_beta});
        params.push_back({prefix + ".q_w", &block->q_w});
        params.push_back({prefix + ".q_b", &block->q_b});
        params.push_back({prefix + ".k_w", &block->k_w});
        params.push_back({prefix + ".k_b", &block->k_b});
        params.push_back({prefix + ".v_w", &block->v_w});
        params.push_back({prefix + ".v_b", &block->v_b});
        params.push_back({prefix + ".out_w", &block->out_w});
        params.push_back({prefix + ".out_b", &block->out_b});
        params.push_back({prefix + ".ln2_gamma", &block->ln2_gamma});
        params.push_back({prefix + ".ln2_beta", &block->ln2_beta});
        params.push_back({prefix + ".ff1_w", &block->ff1_w});
        params.push_back({prefix + ".ff1_b", &block->ff1_b});
        params.push_back({prefix + ".ff2_w", &block->ff2_w});
        params.push_back({prefix + ".ff2_b", &block->ff2_b});
    }
    params.push_back({"ln_f_gamma", &ln_f_gamma});
    params.push_back({"ln_f_beta", &ln_f_beta});
    params.push_back({"lm_head_w", &lm_head_w});
    params.push_back({"lm_head_b", &lm_head_b});
    return params;
}

std::vector<std::pair<std::string, const Tensor*>> TinyCharModel::named_parameters() const {
    std::vector<std::pair<std::string, const Tensor*>> params;
    params.reserve(2 + blocks.size() * 16 + 4);
    params.push_back({"token_embedding", &token_embedding});
    params.push_back({"positional_embedding", &positional_embedding});
    for (auto [prefix, block] : const_blocks(blocks)) {
        params.push_back({prefix + ".ln1_gamma", &block->ln1_gamma});
        params.push_back({prefix + ".ln1_beta", &block->ln1_beta});
        params.push_back({prefix + ".q_w", &block->q_w});
        params.push_back({prefix + ".q_b", &block->q_b});
        params.push_back({prefix + ".k_w", &block->k_w});
        params.push_back({prefix + ".k_b", &block->k_b});
        params.push_back({prefix + ".v_w", &block->v_w});
        params.push_back({prefix + ".v_b", &block->v_b});
        params.push_back({prefix + ".out_w", &block->out_w});
        params.push_back({prefix + ".out_b", &block->out_b});
        params.push_back({prefix + ".ln2_gamma", &block->ln2_gamma});
        params.push_back({prefix + ".ln2_beta", &block->ln2_beta});
        params.push_back({prefix + ".ff1_w", &block->ff1_w});
        params.push_back({prefix + ".ff1_b", &block->ff1_b});
        params.push_back({prefix + ".ff2_w", &block->ff2_w});
        params.push_back({prefix + ".ff2_b", &block->ff2_b});
    }
    params.push_back({"ln_f_gamma", &ln_f_gamma});
    params.push_back({"ln_f_beta", &ln_f_beta});
    params.push_back({"lm_head_w", &lm_head_w});
    params.push_back({"lm_head_b", &lm_head_b});
    return params;
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
        const std::size_t vocab_size = model.config().vocab_size;
        const std::size_t final_row = model.config().context - 1;
        const auto first =
            logits.data().begin() + static_cast<std::ptrdiff_t>(final_row * vocab_size);
        std::vector<double> next_logits(first, first + static_cast<std::ptrdiff_t>(vocab_size));
        const int next = sample_from_logits(next_logits, options.temperature, options.top_k, rng);
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
