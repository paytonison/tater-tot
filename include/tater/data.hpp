#pragma once

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace tater {

struct Vocabulary {
    std::vector<char> chars;
    std::unordered_map<char, int> stoi;

    static Vocabulary from_text(const std::string& text);

    std::size_t size() const;
    int id_for(char ch, bool allow_unknown = false) const;
    std::vector<int> encode(const std::string& text, bool allow_unknown = false) const;
    std::string decode(const std::vector<int>& ids) const;
};

struct Batch {
    std::vector<int> x;
    std::vector<int> y;
    std::size_t batch_size = 0;
};

std::string read_text_file(const std::string& path);
std::pair<std::vector<int>, std::vector<int>> train_validation_split(const std::vector<int>& data,
                                                                      double train_fraction,
                                                                      std::size_t min_sequence);
Batch make_batch(const std::vector<int>& data,
                 std::size_t context,
                 std::size_t batch_size,
                 std::mt19937& rng);

} // namespace tater

