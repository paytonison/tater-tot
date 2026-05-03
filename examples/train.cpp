#include "tater/checkpoint.hpp"
#include "tater/data.hpp"
#include "tater/model.hpp"
#include "tater/optimizer.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

std::string option_value(const std::vector<std::string>& args,
                         const std::string& name,
                         const std::string& fallback) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == name) {
            return args[i + 1];
        }
    }
    return fallback;
}

std::size_t size_option(const std::vector<std::string>& args,
                        const std::string& name,
                        std::size_t fallback) {
    return static_cast<std::size_t>(std::stoull(option_value(args, name, std::to_string(fallback))));
}

double double_option(const std::vector<std::string>& args,
                     const std::string& name,
                     double fallback) {
    return std::stod(option_value(args, name, std::to_string(fallback)));
}

void print_usage() {
    std::cerr << "Usage: tater_train --data data/input.txt --steps 1000 "
                 "--context 64 --embed 64 [--hidden 128] [--batch 16] "
                 "[--lr 0.003] [--checkpoint checkpoints/model.bin]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> args(argv + 1, argv + argc);
        if (std::find(args.begin(), args.end(), "--help") != args.end()) {
            print_usage();
            return 0;
        }

        const std::string data_path = option_value(args, "--data", "data/input.txt");
        const std::size_t steps = size_option(args, "--steps", 1000);
        const std::size_t context = size_option(args, "--context", 64);
        const std::size_t embed = size_option(args, "--embed", 64);
        const std::size_t hidden = size_option(args, "--hidden", embed * 2);
        const std::size_t batch_size = size_option(args, "--batch", 16);
        const std::size_t print_every = size_option(args, "--print-every", 50);
        const std::size_t sample_every = size_option(args, "--sample-every", 200);
        const std::size_t eval_batches = size_option(args, "--eval-batches", 4);
        const double learning_rate = double_option(args, "--lr", 0.003);
        const double clip_norm = double_option(args, "--clip", 1.0);
        const std::uint32_t seed = static_cast<std::uint32_t>(size_option(args, "--seed", 1337));
        const std::string checkpoint_path =
            option_value(args, "--checkpoint", "checkpoints/model.bin");

        const std::string text = tater::read_text_file(data_path);
        tater::Vocabulary vocab = tater::Vocabulary::from_text(text);
        std::vector<int> encoded = vocab.encode(text);
        auto [train_data, valid_data] =
            tater::train_validation_split(encoded, 0.9, context + 1);

        tater::ModelConfig config;
        config.vocab_size = vocab.size();
        config.context = context;
        config.embed = embed;
        config.hidden = hidden;

        std::mt19937 rng(seed);
        tater::TinyCharModel model(config, seed);
        tater::Adam optimizer(model.parameters(), learning_rate);

        std::cout << "Tater Tot training\n"
                  << "  data: " << data_path << "\n"
                  << "  chars: " << text.size() << "\n"
                  << "  vocab: " << vocab.size() << "\n"
                  << "  context/embed/hidden: " << context << "/" << embed << "/" << hidden
                  << "\n"
                  << "  batch/steps/lr: " << batch_size << "/" << steps << "/" << learning_rate
                  << "\n";

        const std::string prompt = text.substr(0, std::min<std::size_t>(text.size(), 32));
        for (std::size_t step = 1; step <= steps; ++step) {
            tater::Batch batch = tater::make_batch(train_data, context, batch_size, rng);
            tater::Tensor logits = model.forward(batch.x, batch.batch_size);
            tater::Tensor loss = tater::softmax_cross_entropy(logits, batch.y);

            optimizer.zero_grad();
            loss.backward();
            const double grad_norm = optimizer.step(clip_norm);

            if (print_every != 0 && (step == 1 || step % print_every == 0 || step == steps)) {
                const double val_loss =
                    tater::estimate_loss(model, valid_data, batch_size, eval_batches, rng);
                std::cout << "step " << step << " train_loss=" << loss.data()[0]
                          << " val_loss=" << val_loss << " grad_norm=" << grad_norm << "\n";
            }

            if (sample_every != 0 && (step % sample_every == 0 || step == steps)) {
                tater::GenerationOptions options;
                options.tokens = 160;
                options.temperature = 0.8;
                options.top_k = 10;
                std::cout << "--- sample ---\n"
                          << tater::generate_text(model, vocab, prompt, options, rng)
                          << "\n--------------\n";
            }
        }

        tater::save_checkpoint(checkpoint_path, model, vocab);
        std::cout << "saved checkpoint: " << checkpoint_path << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tater_train error: " << error.what() << "\n";
        return 1;
    }
}

