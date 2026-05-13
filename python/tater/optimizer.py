from __future__ import annotations

import math
from typing import Sequence

from .tensor import Tensor, clip_grad_norm, zero_grad


class Adam:
    def __init__(
        self,
        params: Sequence[Tensor],
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("Adam learning rate must be positive")
        self.params = list(params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m: list[list[float]] = []
        self.v: list[list[float]] = []
        for param in self.params:
            if param is None or not param.defined:
                raise ValueError("Adam received an undefined parameter")
            self.m.append([0.0] * param.size)
            self.v.append([0.0] * param.size)

    def zero_grad(self) -> None:
        zero_grad(self.params)

    def step(self, clip_norm: float = 1.0) -> float:
        unclipped_norm = clip_grad_norm(self.params, clip_norm)
        self.t += 1

        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t

        for p, param in enumerate(self.params):
            for i in range(param.size):
                grad = param.grad[i]
                self.m[p][i] = self.beta1 * self.m[p][i] + (1.0 - self.beta1) * grad
                self.v[p][i] = self.beta2 * self.v[p][i] + (1.0 - self.beta2) * grad * grad
                m_hat = self.m[p][i] / bias1
                v_hat = self.v[p][i] / bias2
                param.data[i] -= self.learning_rate * m_hat / (
                    math.sqrt(v_hat) + self.epsilon
                )

        return unclipped_norm
