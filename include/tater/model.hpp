#pragma once

#include "tater/data.hpp"
#include "tater/tensor.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace tater {

struct ModelConfig {
    std::size_t vocab_size = 0;
    std::size_t context = 32;
    std::size_t embed = 32;
    std::size_t hidden = 64;
};

struct GenerationOptions {
    std::size_t tokens = 300;
    double temperature = 1.0;
    int top_k = 0;
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
    Tensor token_embedding;
    Tensor positional_embedding;
    Tensor w1;
    Tensor b1;
    Tensor w2;
    Tensor b2;
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

