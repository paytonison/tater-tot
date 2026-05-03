#include "tater/checkpoint.hpp"
#include "tater/model.hpp"

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
    std::cerr << "Usage: tater_generate --checkpoint checkpoints/model.bin "
                 "--prompt \"Once upon a time\" --tokens 300 "
                 "[--temperature 1.0] [--top-k 0]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> args(argv + 1, argv + argc);
        if (std::find(args.begin(), args.end(), "--help") != args.end()) {
            print_usage();
            return 0;
        }

        const std::string checkpoint_path =
            option_value(args, "--checkpoint", "checkpoints/model.bin");
        const std::string prompt = option_value(args, "--prompt", "");
        const std::size_t tokens = size_option(args, "--tokens", 300);
        const double temperature = double_option(args, "--temperature", 1.0);
        const int top_k = static_cast<int>(size_option(args, "--top-k", 0));
        const std::uint32_t seed = static_cast<std::uint32_t>(size_option(args, "--seed", 1337));

        tater::LoadedCheckpoint loaded = tater::load_checkpoint(checkpoint_path);
        tater::GenerationOptions options;
        options.tokens = tokens;
        options.temperature = temperature;
        options.top_k = top_k;

        std::mt19937 rng(seed);
        std::cout << tater::generate_text(loaded.model, loaded.vocab, prompt, options, rng)
                  << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tater_generate error: " << error.what() << "\n";
        return 1;
    }
}

