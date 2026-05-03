#pragma once

#include "tater/tensor.hpp"

#include <vector>

namespace tater {

class Adam {
public:
    Adam(std::vector<Tensor*> params,
         double learning_rate = 0.001,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double epsilon = 1e-8);

    void zero_grad();
    double step(double clip_norm = 1.0);

private:
    std::vector<Tensor*> params_;
    std::vector<std::vector<double>> m_;
    std::vector<std::vector<double>> v_;
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    std::size_t t_ = 0;
};

} // namespace tater

