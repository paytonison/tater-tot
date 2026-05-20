#include "tater/optimizer.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace tater {

Adam::Adam(std::vector<Tensor*> params,
           double learning_rate,
           double beta1,
           double beta2,
           double epsilon)
    : params_(std::move(params)),
      learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon) {
    if (learning_rate_ <= 0.0) {
        throw std::runtime_error("Adam learning rate must be positive");
    }
    for (Tensor* param : params_) {
        if (param == nullptr || !param->defined()) {
            throw std::runtime_error("Adam received an undefined parameter");
        }
        m_.emplace_back(param->size(), 0.0);
        v_.emplace_back(param->size(), 0.0);
    }
}

void Adam::zero_grad() {
    tater::zero_grad(params_);
}

double Adam::step(double clip_norm) {
    const double unclipped_norm = clip_grad_norm(params_, clip_norm);
    ++t_;

    const double bias1 = 1.0 - std::pow(beta1_, static_cast<double>(t_));
    const double bias2 = 1.0 - std::pow(beta2_, static_cast<double>(t_));

    for (std::size_t p = 0; p < params_.size(); ++p) {
        Tensor& param = *params_[p];
        for (std::size_t i = 0; i < param.size(); ++i) {
            const double g = param.grad()[i];
            m_[p][i] = beta1_ * m_[p][i] + (1.0 - beta1_) * g;
            v_[p][i] = beta2_ * v_[p][i] + (1.0 - beta2_) * g * g;
            const double m_hat = m_[p][i] / bias1;
            const double v_hat = v_[p][i] / bias2;
            param.data()[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }

    return unclipped_norm;
}

} // namespace tater
