#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tater {

using Shape = std::vector<std::size_t>;

std::size_t numel(const Shape& shape);
std::string shape_to_string(const Shape& shape);

class Tensor {
public:
    struct Node {
        std::vector<double> data;
        std::vector<double> grad;
        Shape shape;
        std::vector<Tensor> parents;
        std::function<void(Node&)> backward;
        bool requires_grad = true;
        std::string op;
    };

    Tensor();
    Tensor(const Shape& shape, double value = 0.0, bool requires_grad = true);

    static Tensor from_data(std::vector<double> data, const Shape& shape, bool requires_grad = true);
    static Tensor scalar(double value, bool requires_grad = false);
    static Tensor create(std::vector<double> data,
                         const Shape& shape,
                         bool requires_grad,
                         std::string op,
                         std::vector<Tensor> parents,
                         std::function<void(Node&)> backward);

    bool defined() const;
    const Shape& shape() const;
    std::size_t size() const;
    bool requires_grad() const;

    std::vector<double>& data();
    const std::vector<double>& data() const;
    std::vector<double>& grad();
    const std::vector<double>& grad() const;

    void zero_grad();
    void backward();

    std::shared_ptr<Node> node() const;

private:
    explicit Tensor(std::shared_ptr<Node> node);

    std::shared_ptr<Node> node_;
};

Tensor add(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor sum(const Tensor& x);
Tensor mean(const Tensor& x);
Tensor tanh(const Tensor& x);
Tensor relu(const Tensor& x);
Tensor gelu(const Tensor& x);
Tensor exp(const Tensor& x);
Tensor log(const Tensor& x);
Tensor transpose(const Tensor& x);
Tensor reshape(const Tensor& x, const Shape& shape);
Tensor embedding_lookup(const Tensor& table, const std::vector<int>& indices);
Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, double eps = 1e-5);
Tensor causal_self_attention(const Tensor& x,
                             const Tensor& wq,
                             const Tensor& bq,
                             const Tensor& wk,
                             const Tensor& bk,
                             const Tensor& wv,
                             const Tensor& bv,
                             const Tensor& wo,
                             const Tensor& bo,
                             std::size_t batch_size,
                             std::size_t context,
                             std::size_t heads);
Tensor softmax_cross_entropy(const Tensor& logits, const std::vector<int>& targets);

void zero_grad(const std::vector<Tensor*>& tensors);
double clip_grad_norm(const std::vector<Tensor*>& tensors, double max_norm);

} // namespace tater
