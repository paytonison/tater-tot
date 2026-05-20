#pragma once

#include "tater/data.hpp"
#include "tater/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace tater {

struct ModelConfig {
    std::size_t vocab_size = 0;
    std::size_t context = 32;
    std::size_t layers = 2;
    std::size_t heads = 4;
    std::size_t embed = 64;
    std::size_t hidden = 128;
};

struct GenerationOptions {
    std::size_t tokens = 300;
    double temperature = 1.0;
    int top_k = 0;
};

struct TokenEmbedding {
    Tensor table;

    TokenEmbedding() = default;
    TokenEmbedding(std::size_t vocab_size, std::size_t d_model, std::mt19937& rng);

    Tensor forward(const std::vector<int>& token_ids) const;
    std::size_t vocab_size() const;
    std::size_t d_model() const;
};

struct PositionalEmbedding {
    Tensor table;

    PositionalEmbedding() = default;
    PositionalEmbedding(std::size_t max_seq_len, std::size_t d_model, std::mt19937& rng);

    Tensor forward(std::size_t batch_size, std::size_t sequence_length) const;
    std::size_t max_seq_len() const;
    std::size_t d_model() const;
};

class TinyCharModel {
public:
    explicit TinyCharModel(ModelConfig config, std::uint32_t seed = 1337);

    const ModelConfig& config() const;
    Tensor forward(const std::vector<int>& tokens, std::size_t batch_size) const;

    std::vector<Tensor*> parameters();
    std::vector<const Tensor*> parameters() const;
    std::vector<std::pair<std::string, Tensor*>> named_parameters();
    std::vector<std::pair<std::string, const Tensor*>> named_parameters() const;

private:
    ModelConfig config_;

public:
    struct TransformerBlock {
        Tensor ln1_gamma;
        Tensor ln1_beta;
        Tensor q_w;
        Tensor q_b;
        Tensor k_w;
        Tensor k_b;
        Tensor v_w;
        Tensor v_b;
        Tensor out_w;
        Tensor out_b;
        Tensor ln2_gamma;
        Tensor ln2_beta;
        Tensor ff1_w;
        Tensor ff1_b;
        Tensor ff2_w;
        Tensor ff2_b;
    };

    TokenEmbedding token_embedding;
    PositionalEmbedding positional_embedding;
    std::vector<TransformerBlock> blocks;
    Tensor ln_f_gamma;
    Tensor ln_f_beta;
    Tensor lm_head_w;
    Tensor lm_head_b;
};

int sample_from_logits(const std::vector<double>& logits,
                       double temperature,
                       int top_k,
                       std::mt19937& rng);

std::string generate_text(const TinyCharModel& model,
                          const Vocabulary& vocab,
                          const std::string& prompt,
                          const GenerationOptions& options,
                          std::mt19937& rng);

double estimate_loss(const TinyCharModel& model,
                     const std::vector<int>& data,
                     std::size_t batch_size,
                     std::size_t batches,
                     std::mt19937& rng);

} // namespace tater
