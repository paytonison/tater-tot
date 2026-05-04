#include "tater/checkpoint.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace tater {
namespace {

constexpr char kMagic[] = "TATER_TOT_CHECKPOINT_V2_TRANSFORMER";
constexpr char kV1Magic[] = "TATER_TOT_CHECKPOINT_V1";

template <typename T>
void write_pod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) {
        throw std::runtime_error("failed to write checkpoint");
    }
}

template <typename T>
T read_pod(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("failed to read checkpoint");
    }
    return value;
}

void write_string(std::ostream& out, const std::string& value) {
    const std::uint64_t size = static_cast<std::uint64_t>(value.size());
    write_pod(out, size);
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!out) {
        throw std::runtime_error("failed to write checkpoint string");
    }
}

std::string read_string(std::istream& in) {
    const std::uint64_t size = read_pod<std::uint64_t>(in);
    std::string value(size, '\0');
    in.read(value.data(), static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error("failed to read checkpoint string");
    }
    return value;
}

void write_tensor(std::ostream& out, const std::string& name, const Tensor& tensor) {
    write_string(out, name);
    const std::uint64_t rank = static_cast<std::uint64_t>(tensor.shape().size());
    write_pod(out, rank);
    for (std::size_t dim : tensor.shape()) {
        write_pod(out, static_cast<std::uint64_t>(dim));
    }
    const std::uint64_t values = static_cast<std::uint64_t>(tensor.size());
    write_pod(out, values);
    out.write(reinterpret_cast<const char*>(tensor.data().data()),
              static_cast<std::streamsize>(values * sizeof(double)));
    if (!out) {
        throw std::runtime_error("failed to write tensor data");
    }
}

void read_tensor_into(std::istream& in, const std::string& expected_name, Tensor& tensor) {
    const std::string name = read_string(in);
    if (name != expected_name) {
        throw std::runtime_error("checkpoint tensor order mismatch: expected " + expected_name +
                                 ", got " + name);
    }

    const std::uint64_t rank = read_pod<std::uint64_t>(in);
    Shape shape(rank);
    for (std::uint64_t i = 0; i < rank; ++i) {
        shape[static_cast<std::size_t>(i)] = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    }
    if (shape != tensor.shape()) {
        throw std::runtime_error("checkpoint tensor shape mismatch for " + name);
    }

    const std::uint64_t values = read_pod<std::uint64_t>(in);
    if (values != tensor.size()) {
        throw std::runtime_error("checkpoint tensor size mismatch for " + name);
    }
    in.read(reinterpret_cast<char*>(tensor.data().data()),
            static_cast<std::streamsize>(values * sizeof(double)));
    if (!in) {
        throw std::runtime_error("failed to read tensor data");
    }
}

} // namespace

void save_checkpoint(const std::string& path, const TinyCharModel& model, const Vocabulary& vocab) {
    const std::filesystem::path checkpoint_path(path);
    if (checkpoint_path.has_parent_path()) {
        std::filesystem::create_directories(checkpoint_path.parent_path());
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open checkpoint for writing: " + path);
    }

    write_string(out, kMagic);
    write_pod(out, static_cast<std::uint64_t>(model.config().vocab_size));
    write_pod(out, static_cast<std::uint64_t>(model.config().context));
    write_pod(out, static_cast<std::uint64_t>(model.config().embed));
    write_pod(out, static_cast<std::uint64_t>(model.config().hidden));
    write_pod(out, static_cast<std::uint64_t>(model.config().layers));
    write_pod(out, static_cast<std::uint64_t>(model.config().heads));

    std::string chars(vocab.chars.begin(), vocab.chars.end());
    write_string(out, chars);

    const auto params = model.named_parameters();
    write_pod(out, static_cast<std::uint64_t>(params.size()));
    for (const auto& [name, tensor] : params) {
        write_tensor(out, name, *tensor);
    }
}

LoadedCheckpoint load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open checkpoint for reading: " + path);
    }

    const std::string magic = read_string(in);
    if (magic != kMagic) {
        if (magic == kV1Magic) {
            throw std::runtime_error(
                "checkpoint is an old V1 MLP checkpoint and cannot be loaded into the "
                "Transformer model");
        }
        throw std::runtime_error("not a Tater Tot checkpoint: " + path);
    }

    ModelConfig config;
    config.vocab_size = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    config.context = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    config.embed = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    config.hidden = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    config.layers = static_cast<std::size_t>(read_pod<std::uint64_t>(in));
    config.heads = static_cast<std::size_t>(read_pod<std::uint64_t>(in));

    const std::string chars = read_string(in);
    Vocabulary vocab;
    vocab.chars.assign(chars.begin(), chars.end());
    for (std::size_t i = 0; i < vocab.chars.size(); ++i) {
        vocab.stoi[vocab.chars[i]] = static_cast<int>(i);
    }
    if (vocab.size() != config.vocab_size) {
        throw std::runtime_error("checkpoint vocabulary size does not match model config");
    }

    TinyCharModel model(config, 1);
    const std::uint64_t param_count = read_pod<std::uint64_t>(in);
    const auto params = model.named_parameters();
    if (param_count != params.size()) {
        throw std::runtime_error("checkpoint parameter count does not match model");
    }
    for (const auto& [name, tensor] : params) {
        read_tensor_into(in, name, *tensor);
    }

    return LoadedCheckpoint{std::move(model), std::move(vocab)};
}

} // namespace tater
