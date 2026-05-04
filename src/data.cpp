#include "tater/data.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <utility>

namespace tater {

Vocabulary Vocabulary::from_text(const std::string& text) {
    if (text.empty()) {
        throw std::runtime_error("cannot build a vocabulary from empty text");
    }

    std::set<char> unique(text.begin(), text.end());
    Vocabulary vocab;
    vocab.chars.assign(unique.begin(), unique.end());
    for (std::size_t i = 0; i < vocab.chars.size(); ++i) {
        vocab.stoi[vocab.chars[i]] = static_cast<int>(i);
    }
    return vocab;
}

std::size_t Vocabulary::size() const {
    return chars.size();
}

int Vocabulary::id_for(char ch, bool allow_unknown) const {
    const auto it = stoi.find(ch);
    if (it != stoi.end()) {
        return it->second;
    }
    if (allow_unknown && !chars.empty()) {
        return 0;
    }
    throw std::runtime_error("character is not present in the vocabulary");
}

std::vector<int> Vocabulary::encode(const std::string& text, bool allow_unknown) const {
    std::vector<int> ids;
    ids.reserve(text.size());
    for (char ch : text) {
        const auto it = stoi.find(ch);
        if (it != stoi.end()) {
            ids.push_back(it->second);
        } else if (allow_unknown && !chars.empty()) {
            ids.push_back(0);
        } else if (!allow_unknown) {
            throw std::runtime_error("text contains a character outside the vocabulary");
        }
    }
    return ids;
}

std::string Vocabulary::decode(const std::vector<int>& ids) const {
    std::string text;
    text.reserve(ids.size());
    for (int id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= chars.size()) {
            throw std::runtime_error("token id is outside the vocabulary");
        }
        text.push_back(chars[static_cast<std::size_t>(id)]);
    }
    return text;
}

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open data file: " + path);
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (text.empty()) {
        throw std::runtime_error("data file is empty: " + path);
    }
    return text;
}

std::pair<std::vector<int>, std::vector<int>> train_validation_split(const std::vector<int>& data,
                                                                      double train_fraction,
                                                                      std::size_t min_sequence) {
    if (data.size() < min_sequence) {
        throw std::runtime_error("not enough encoded data for the requested context length");
    }

    train_fraction = std::clamp(train_fraction, 0.5, 0.99);
    std::size_t cut = static_cast<std::size_t>(static_cast<double>(data.size()) * train_fraction);
    cut = std::clamp(cut, min_sequence, data.size());

    std::vector<int> train(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(cut));
    std::vector<int> valid(data.begin() + static_cast<std::ptrdiff_t>(cut), data.end());
    if (valid.size() < min_sequence) {
        valid = train;
    }
    return {std::move(train), std::move(valid)};
}

Batch make_batch(const std::vector<int>& data,
                 std::size_t context,
                 std::size_t batch_size,
                 std::mt19937& rng) {
    if (data.size() <= context + 1) {
        throw std::runtime_error("not enough data to create full-sequence targets");
    }

    std::uniform_int_distribution<std::size_t> start_dist(0, data.size() - context - 1);
    Batch batch;
    batch.batch_size = batch_size;
    batch.x.resize(batch_size * context);
    batch.y.resize(batch_size * context);

    for (std::size_t b = 0; b < batch_size; ++b) {
        const std::size_t start = start_dist(rng);
        for (std::size_t t = 0; t < context; ++t) {
            batch.x[b * context + t] = data[start + t];
            batch.y[b * context + t] = data[start + t + 1];
        }
    }

    return batch;
}

} // namespace tater
