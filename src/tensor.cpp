#include "tater/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace tater {
namespace {

void require_defined(const Tensor& tensor, const char* name) {
    if (!tensor.defined()) {
        throw std::runtime_error(std::string(name) + " is an undefined tensor");
    }
}

std::vector<std::size_t> strides_for(const Shape& shape) {
    std::vector<std::size_t> strides(shape.size(), 1);
    if (shape.empty()) {
        return strides;
    }
    for (std::size_t i = shape.size() - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}

Shape broadcast_shape(const Shape& a, const Shape& b) {
    const std::size_t rank = std::max(a.size(), b.size());
    Shape out(rank, 1);
    for (std::size_t i = 0; i < rank; ++i) {
        const std::size_t ai = i + a.size() < rank ? 1 : a[i - (rank - a.size())];
        const std::size_t bi = i + b.size() < rank ? 1 : b[i - (rank - b.size())];
        if (ai != bi && ai != 1 && bi != 1) {
            throw std::runtime_error("cannot broadcast shapes " + shape_to_string(a) +
                                     " and " + shape_to_string(b));
        }
        out[i] = std::max(ai, bi);
    }
    return out;
}

std::size_t broadcast_offset(std::size_t out_index, const Shape& out_shape, const Shape& in_shape) {
    if (in_shape.empty()) {
        return 0;
    }

    const auto out_strides = strides_for(out_shape);
    const auto in_strides = strides_for(in_shape);
    const std::size_t rank_delta = out_shape.size() - in_shape.size();
    std::size_t offset = 0;

    for (std::size_t i = 0; i < in_shape.size(); ++i) {
        const std::size_t out_dim = i + rank_delta;
        const std::size_t coord = (out_index / out_strides[out_dim]) % out_shape[out_dim];
        const std::size_t in_coord = in_shape[i] == 1 ? 0 : coord;
        offset += in_coord * in_strides[i];
    }
    return offset;
}

void build_topology(const Tensor& tensor,
                    std::unordered_set<const Tensor::Node*>& visited,
                    std::vector<Tensor>& topo) {
    if (!tensor.defined()) {
        return;
    }
    const Tensor::Node* ptr = tensor.node().get();
    if (!visited.insert(ptr).second) {
        return;
    }

    for (const Tensor& parent : tensor.node()->parents) {
        build_topology(parent, visited, topo);
    }
    topo.push_back(tensor);
}

void require_rank(const Tensor& tensor, std::size_t rank, const char* op) {
    if (tensor.shape().size() != rank) {
        throw std::runtime_error(std::string(op) + " expected rank " + std::to_string(rank) +
                                 ", got " + shape_to_string(tensor.shape()));
    }
}

} // namespace

std::size_t numel(const Shape& shape) {
    if (shape.empty()) {
        return 1;
    }
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
}

std::string shape_to_string(const Shape& shape) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << shape[i];
    }
    out << "]";
    return out.str();
}

Tensor::Tensor() = default;

Tensor::Tensor(const Shape& shape, double value, bool requires_grad)
    : node_(std::make_shared<Node>()) {
    node_->shape = shape;
    node_->data.assign(numel(shape), value);
    node_->grad.assign(numel(shape), 0.0);
    node_->requires_grad = requires_grad;
    node_->op = "leaf";
}

Tensor::Tensor(std::shared_ptr<Node> node) : node_(std::move(node)) {}

Tensor Tensor::from_data(std::vector<double> data, const Shape& shape, bool requires_grad) {
    if (data.size() != numel(shape)) {
        throw std::runtime_error("tensor data size does not match shape " + shape_to_string(shape));
    }
    auto node = std::make_shared<Node>();
    node->data = std::move(data);
    node->grad.assign(numel(shape), 0.0);
    node->shape = shape;
    node->requires_grad = requires_grad;
    node->op = "leaf";
    return Tensor(node);
}

Tensor Tensor::scalar(double value, bool requires_grad) {
    return Tensor::from_data({value}, {}, requires_grad);
}

Tensor Tensor::create(std::vector<double> data,
                      const Shape& shape,
                      bool requires_grad,
                      std::string op,
                      std::vector<Tensor> parents,
                      std::function<void(Node&)> backward) {
    if (data.size() != numel(shape)) {
        throw std::runtime_error("tensor data size does not match shape " + shape_to_string(shape));
    }
    auto node = std::make_shared<Node>();
    node->data = std::move(data);
    node->grad.assign(numel(shape), 0.0);
    node->shape = shape;
    node->requires_grad = requires_grad;
    node->op = std::move(op);
    node->parents = std::move(parents);
    node->backward = std::move(backward);
    return Tensor(node);
}

bool Tensor::defined() const {
    return static_cast<bool>(node_);
}

const Shape& Tensor::shape() const {
    require_defined(*this, "tensor");
    return node_->shape;
}

std::size_t Tensor::size() const {
    require_defined(*this, "tensor");
    return node_->data.size();
}

bool Tensor::requires_grad() const {
    return defined() && node_->requires_grad;
}

std::vector<double>& Tensor::data() {
    require_defined(*this, "tensor");
    return node_->data;
}

const std::vector<double>& Tensor::data() const {
    require_defined(*this, "tensor");
    return node_->data;
}

std::vector<double>& Tensor::grad() {
    require_defined(*this, "tensor");
    return node_->grad;
}

const std::vector<double>& Tensor::grad() const {
    require_defined(*this, "tensor");
    return node_->grad;
}

void Tensor::zero_grad() {
    require_defined(*this, "tensor");
    std::fill(node_->grad.begin(), node_->grad.end(), 0.0);
}

void Tensor::backward() {
    require_defined(*this, "tensor");
    if (size() != 1) {
        throw std::runtime_error("backward requires a scalar tensor output");
    }

    std::unordered_set<const Node*> visited;
    std::vector<Tensor> topo;
    build_topology(*this, visited, topo);

    // The graph stores parent links from each result tensor to the tensors that
    // produced it. A reverse topological walk therefore visits outputs first and
    // pushes gradients back into each parent exactly after all downstream uses
    // have contributed to the current node's gradient.
    for (Tensor& tensor : topo) {
        std::fill(tensor.grad().begin(), tensor.grad().end(), 0.0);
    }
    grad()[0] = 1.0;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if (it->node()->backward) {
            it->node()->backward(*it->node());
        }
    }
}

std::shared_ptr<Tensor::Node> Tensor::node() const {
    return node_;
}

Tensor add(const Tensor& a, const Tensor& b) {
    require_defined(a, "a");
    require_defined(b, "b");
    const Shape out_shape = broadcast_shape(a.shape(), b.shape());
    std::vector<double> out(numel(out_shape));

    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = a.data()[broadcast_offset(i, out_shape, a.shape())] +
                 b.data()[broadcast_offset(i, out_shape, b.shape())];
    }

    const bool requires_grad = a.requires_grad() || b.requires_grad();
    return Tensor::create(std::move(out), out_shape, requires_grad, "add", {a, b},
                          [a, b, out_shape](Tensor::Node& self) {
                              for (std::size_t i = 0; i < self.grad.size(); ++i) {
                                  if (a.requires_grad()) {
                                      a.node()->grad[broadcast_offset(i, out_shape, a.shape())] +=
                                          self.grad[i];
                                  }
                                  if (b.requires_grad()) {
                                      b.node()->grad[broadcast_offset(i, out_shape, b.shape())] +=
                                          self.grad[i];
                                  }
                              }
                          });
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    require_defined(a, "a");
    require_defined(b, "b");
    const Shape out_shape = broadcast_shape(a.shape(), b.shape());
    std::vector<double> out(numel(out_shape));

    for (std::size_t i = 0; i < out.size(); ++i) {
        const std::size_t ai = broadcast_offset(i, out_shape, a.shape());
        const std::size_t bi = broadcast_offset(i, out_shape, b.shape());
        out[i] = a.data()[ai] * b.data()[bi];
    }

    const bool requires_grad = a.requires_grad() || b.requires_grad();
    return Tensor::create(std::move(out), out_shape, requires_grad, "multiply", {a, b},
                          [a, b, out_shape](Tensor::Node& self) {
                              for (std::size_t i = 0; i < self.grad.size(); ++i) {
                                  const std::size_t ai = broadcast_offset(i, out_shape, a.shape());
                                  const std::size_t bi = broadcast_offset(i, out_shape, b.shape());
                                  if (a.requires_grad()) {
                                      a.node()->grad[ai] += self.grad[i] * b.data()[bi];
                                  }
                                  if (b.requires_grad()) {
                                      b.node()->grad[bi] += self.grad[i] * a.data()[ai];
                                  }
                              }
                          });
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    require_defined(a, "a");
    require_defined(b, "b");
    require_rank(a, 2, "matmul");
    require_rank(b, 2, "matmul");

    const std::size_t m = a.shape()[0];
    const std::size_t n = a.shape()[1];
    const std::size_t b_rows = b.shape()[0];
    const std::size_t p = b.shape()[1];
    if (n != b_rows) {
        throw std::runtime_error("matmul shape mismatch: " + shape_to_string(a.shape()) + " x " +
                                 shape_to_string(b.shape()));
    }

    std::vector<double> out(m * p, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            const double av = a.data()[i * n + k];
            for (std::size_t j = 0; j < p; ++j) {
                out[i * p + j] += av * b.data()[k * p + j];
            }
        }
    }

    const bool requires_grad = a.requires_grad() || b.requires_grad();
    return Tensor::create(std::move(out), {m, p}, requires_grad, "matmul", {a, b},
                          [a, b, m, n, p](Tensor::Node& self) {
                              for (std::size_t i = 0; i < m; ++i) {
                                  for (std::size_t j = 0; j < p; ++j) {
                                      const double go = self.grad[i * p + j];
                                      if (a.requires_grad()) {
                                          for (std::size_t k = 0; k < n; ++k) {
                                              a.node()->grad[i * n + k] +=
                                                  go * b.data()[k * p + j];
                                          }
                                      }
                                      if (b.requires_grad()) {
                                          for (std::size_t k = 0; k < n; ++k) {
                                              b.node()->grad[k * p + j] +=
                                                  a.data()[i * n + k] * go;
                                          }
                                      }
                                  }
                              }
                          });
}

Tensor sum(const Tensor& x) {
    require_defined(x, "x");
    const double total = std::accumulate(x.data().begin(), x.data().end(), 0.0);
    return Tensor::create({total}, {}, x.requires_grad(), "sum", {x},
                          [x](Tensor::Node& self) {
                              if (!x.requires_grad()) {
                                  return;
                              }
                              for (double& g : x.node()->grad) {
                                  g += self.grad[0];
                              }
                          });
}

Tensor mean(const Tensor& x) {
    require_defined(x, "x");
    if (x.size() == 0) {
        throw std::runtime_error("mean of empty tensor");
    }
    const double total = std::accumulate(x.data().begin(), x.data().end(), 0.0);
    const double inv = 1.0 / static_cast<double>(x.size());
    return Tensor::create({total * inv}, {}, x.requires_grad(), "mean", {x},
                          [x, inv](Tensor::Node& self) {
                              if (!x.requires_grad()) {
                                  return;
                              }
                              for (double& g : x.node()->grad) {
                                  g += self.grad[0] * inv;
                              }
                          });
}

Tensor tanh(const Tensor& x) {
    require_defined(x, "x");
    std::vector<double> out(x.size());
    std::transform(x.data().begin(), x.data().end(), out.begin(),
                   [](double v) { return std::tanh(v); });
    return Tensor::create(std::move(out), x.shape(), x.requires_grad(), "tanh", {x},
                          [x](Tensor::Node& self) {
                              if (!x.requires_grad()) {
                                  return;
                              }
                              for (std::size_t i = 0; i < self.grad.size(); ++i) {
                                  x.node()->grad[i] += self.grad[i] * (1.0 - self.data[i] * self.data[i]);
                              }
                          });
}

Tensor relu(const Tensor& x) {
    require_defined(x, "x");
    std::vector<double> out(x.size());
    std::transform(x.data().begin(), x.data().end(), out.begin(),
                   [](double v) { return std::max(0.0, v); });
    return Tensor::create(std::move(out), x.shape(), x.requires_grad(), "relu", {x},
                          [x](Tensor::Node& self) {
                              if (!x.requires_grad()) {
                                  return;
                              }
                              for (std::size_t i = 0; i < self.grad.size(); ++i) {
                                  x.node()->grad[i] += self.grad[i] * (x.data()[i] > 0.0 ? 1.0 : 0.0);
                              }
                          });
}

Tensor exp(const Tensor& x) {
    require_defined(x, "x");
    std::vector<double> out(x.size());
    std::transform(x.data().begin(), x.data().end(), out.begin(),
                   [](double v) { return std::exp(v); });
    return Tensor::create(std::move(out), x.shape(), x.requires_grad(), "exp", {x},
                          [x](Tensor::Node& self) {
                              if (!x.requires_grad()) {
                                  return;
                              }
                              for (std::size_t i = 0; i < self.grad.size(); ++i) {
                                  x.node()->grad[i] += self.grad[i] * self.data[i];
                              }
                          });
}

Tensor log(const Tensor& x) {
    require_defined(x, "x");
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x.data()[i] <= 0.0) {
            throw std::runtime_error("log requires positive inputs");
        }
        out[i] = std::log(x.data()[i]);
    }
    return Tensor::create(std::move(out), x.shape(), x.requires_grad(), "log", {x},
                          [x](Tensor::Node& self) {
                              if (!x.requires_grad()) {
                                  return;
                              }
                              for (std::size_t i = 0; i < self.grad.size(); ++i) {
                                  x.node()->grad[i] += self.grad[i] / x.data()[i];
                              }
                          });
}

Tensor transpose(const Tensor& x) {
    require_defined(x, "x");
    require_rank(x, 2, "transpose");
    const std::size_t rows = x.shape()[0];
    const std::size_t cols = x.shape()[1];
    std::vector<double> out(x.size());
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            out[c * rows + r] = x.data()[r * cols + c];
        }
    }

    return Tensor::create(std::move(out), {cols, rows}, x.requires_grad(), "transpose", {x},
                          [x, rows, cols](Tensor::Node& self) {
                              if (!x.requires_grad()) {
                                  return;
                              }
                              for (std::size_t r = 0; r < rows; ++r) {
                                  for (std::size_t c = 0; c < cols; ++c) {
                                      x.node()->grad[r * cols + c] += self.grad[c * rows + r];
                                  }
                              }
                          });
}

Tensor reshape(const Tensor& x, const Shape& shape) {
    require_defined(x, "x");
    if (x.size() != numel(shape)) {
        throw std::runtime_error("reshape from " + shape_to_string(x.shape()) + " to " +
                                 shape_to_string(shape) + " changes element count");
    }
    return Tensor::create(x.data(), shape, x.requires_grad(), "reshape", {x},
                          [x](Tensor::Node& self) {
                              if (!x.requires_grad()) {
                                  return;
                              }
                              for (std::size_t i = 0; i < self.grad.size(); ++i) {
                                  x.node()->grad[i] += self.grad[i];
                              }
                          });
}

Tensor embedding_lookup(const Tensor& table, const std::vector<int>& indices) {
    require_defined(table, "table");
    require_rank(table, 2, "embedding_lookup");
    const std::size_t rows = table.shape()[0];
    const std::size_t dim = table.shape()[1];
    std::vector<double> out(indices.size() * dim);

    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || static_cast<std::size_t>(indices[i]) >= rows) {
            throw std::runtime_error("embedding index out of range");
        }
        const std::size_t row = static_cast<std::size_t>(indices[i]);
        std::copy_n(table.data().begin() + static_cast<std::ptrdiff_t>(row * dim),
                    dim,
                    out.begin() + static_cast<std::ptrdiff_t>(i * dim));
    }

    return Tensor::create(std::move(out), {indices.size(), dim}, table.requires_grad(),
                          "embedding_lookup", {table},
                          [table, indices, dim](Tensor::Node& self) {
                              if (!table.requires_grad()) {
                                  return;
                              }
                              for (std::size_t i = 0; i < indices.size(); ++i) {
                                  const std::size_t row = static_cast<std::size_t>(indices[i]);
                                  for (std::size_t d = 0; d < dim; ++d) {
                                      table.node()->grad[row * dim + d] += self.grad[i * dim + d];
                                  }
                              }
                          });
}

Tensor softmax_cross_entropy(const Tensor& logits, const std::vector<int>& targets) {
    require_defined(logits, "logits");
    require_rank(logits, 2, "softmax_cross_entropy");
    const std::size_t batch = logits.shape()[0];
    const std::size_t classes = logits.shape()[1];
    if (targets.size() != batch) {
        throw std::runtime_error("target count does not match logits batch");
    }

    std::vector<double> probabilities(batch * classes, 0.0);
    double loss = 0.0;
    for (std::size_t row = 0; row < batch; ++row) {
        if (targets[row] < 0 || static_cast<std::size_t>(targets[row]) >= classes) {
            throw std::runtime_error("target class out of range");
        }
        const auto begin = logits.data().begin() + static_cast<std::ptrdiff_t>(row * classes);
        const auto end = begin + static_cast<std::ptrdiff_t>(classes);
        const double max_logit = *std::max_element(begin, end);

        double sum_exp = 0.0;
        for (std::size_t cls = 0; cls < classes; ++cls) {
            const double e = std::exp(logits.data()[row * classes + cls] - max_logit);
            probabilities[row * classes + cls] = e;
            sum_exp += e;
        }

        const double log_sum_exp = max_logit + std::log(sum_exp);
        loss += log_sum_exp - logits.data()[row * classes + static_cast<std::size_t>(targets[row])];

        for (std::size_t cls = 0; cls < classes; ++cls) {
            probabilities[row * classes + cls] /= sum_exp;
        }
    }
    loss /= static_cast<double>(batch);

    return Tensor::create({loss}, {}, logits.requires_grad(), "softmax_cross_entropy", {logits},
                          [logits, targets, probabilities, batch, classes](Tensor::Node& self) {
                              if (!logits.requires_grad()) {
                                  return;
                              }
                              const double scale = self.grad[0] / static_cast<double>(batch);
                              for (std::size_t row = 0; row < batch; ++row) {
                                  for (std::size_t cls = 0; cls < classes; ++cls) {
                                      double grad = probabilities[row * classes + cls];
                                      if (cls == static_cast<std::size_t>(targets[row])) {
                                          grad -= 1.0;
                                      }
                                      logits.node()->grad[row * classes + cls] += scale * grad;
                                  }
                              }
                          });
}

void zero_grad(const std::vector<Tensor*>& tensors) {
    for (Tensor* tensor : tensors) {
        if (tensor != nullptr && tensor->defined()) {
            tensor->zero_grad();
        }
    }
}

double clip_grad_norm(const std::vector<Tensor*>& tensors, double max_norm) {
    double squared = 0.0;
    for (const Tensor* tensor : tensors) {
        if (tensor == nullptr || !tensor->defined()) {
            continue;
        }
        for (double g : tensor->grad()) {
            squared += g * g;
        }
    }

    const double norm = std::sqrt(squared);
    if (max_norm > 0.0 && norm > max_norm) {
        const double scale = max_norm / (norm + 1e-12);
        for (Tensor* tensor : tensors) {
            if (tensor == nullptr || !tensor->defined()) {
                continue;
            }
            for (double& g : tensor->grad()) {
                g *= scale;
            }
        }
    }
    return norm;
}

} // namespace tater
