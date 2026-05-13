from __future__ import annotations

import builtins
import math
from typing import Callable, Iterable, Optional, Sequence

Shape = tuple[int, ...]


def _shape(shape: Sequence[int] | int | None) -> Shape:
    if shape is None:
        return ()
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(dim) for dim in shape)


def numel(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def shape_to_string(shape: Sequence[int]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


class _Node:
    def __init__(
        self,
        data: Iterable[float],
        shape: Sequence[int],
        requires_grad: bool,
        op: str,
        parents: Sequence["Tensor"] | None = None,
        backward: Optional[Callable[["_Node"], None]] = None,
    ) -> None:
        self.data = [float(value) for value in data]
        self.shape = _shape(shape)
        if len(self.data) != numel(self.shape):
            raise ValueError(
                "tensor data size does not match shape " + shape_to_string(self.shape)
            )
        self.grad = [0.0 for _ in self.data]
        self.parents = list(parents or [])
        self.backward = backward
        self.requires_grad = bool(requires_grad)
        self.op = op


class Tensor:
    def __init__(
        self,
        shape: Sequence[int] | int | None = None,
        value: float = 0.0,
        requires_grad: bool = True,
        *,
        _node: _Node | None = None,
    ) -> None:
        if _node is not None:
            self._node = _node
            return
        if shape is None:
            self._node = None
            return
        normalized_shape = _shape(shape)
        self._node = _Node(
            [float(value)] * numel(normalized_shape),
            normalized_shape,
            requires_grad,
            "leaf",
        )

    @staticmethod
    def from_data(
        data: Iterable[float], shape: Sequence[int] | int | None, requires_grad: bool = True
    ) -> "Tensor":
        return Tensor(_node=_Node(data, _shape(shape), requires_grad, "leaf"))

    @staticmethod
    def scalar(value: float, requires_grad: bool = False) -> "Tensor":
        return Tensor.from_data([value], (), requires_grad)

    @staticmethod
    def create(
        data: Iterable[float],
        shape: Sequence[int] | int | None,
        requires_grad: bool,
        op: str,
        parents: Sequence["Tensor"],
        backward: Callable[["_Node"], None] | None,
    ) -> "Tensor":
        return Tensor(
            _node=_Node(data, _shape(shape), requires_grad, op, parents, backward)
        )

    @property
    def defined(self) -> bool:
        return self._node is not None

    @property
    def shape(self) -> Shape:
        _require_defined(self, "tensor")
        return self._node.shape

    @property
    def size(self) -> int:
        _require_defined(self, "tensor")
        return len(self._node.data)

    @property
    def requires_grad(self) -> bool:
        return self.defined and self._node.requires_grad

    @property
    def data(self) -> list[float]:
        _require_defined(self, "tensor")
        return self._node.data

    @property
    def grad(self) -> list[float]:
        _require_defined(self, "tensor")
        return self._node.grad

    @property
    def node(self) -> _Node:
        _require_defined(self, "tensor")
        return self._node

    def zero_grad(self) -> None:
        _require_defined(self, "tensor")
        self.grad[:] = [0.0 for _ in self.grad]

    def backward(self) -> None:
        _require_defined(self, "tensor")
        if self.size != 1:
            raise ValueError("backward requires a scalar tensor output")

        visited: set[int] = set()
        topo: list[Tensor] = []

        def build_topology(tensor: Tensor) -> None:
            if not tensor.defined:
                return
            ptr = id(tensor.node)
            if ptr in visited:
                return
            visited.add(ptr)
            for parent in tensor.node.parents:
                build_topology(parent)
            topo.append(tensor)

        build_topology(self)

        # Result tensors keep links to the tensors that produced them. Walking
        # the graph in reverse topological order pushes each accumulated
        # gradient back to its parents after all downstream paths have arrived.
        for tensor in topo:
            tensor.zero_grad()
        self.grad[0] = 1.0

        for tensor in reversed(topo):
            if tensor.node.backward is not None:
                tensor.node.backward(tensor.node)


def _require_defined(tensor: Tensor, name: str) -> None:
    if not tensor.defined:
        raise ValueError(f"{name} is an undefined tensor")


def _require_rank(tensor: Tensor, rank: int, op: str) -> None:
    if len(tensor.shape) != rank:
        raise ValueError(
            f"{op} expected rank {rank}, got {shape_to_string(tensor.shape)}"
        )


def _strides_for(shape: Sequence[int]) -> list[int]:
    strides = [1 for _ in shape]
    if not shape:
        return strides
    for i in range(len(shape) - 1, 0, -1):
        strides[i - 1] = strides[i] * shape[i]
    return strides


def _broadcast_shape(a: Shape, b: Shape) -> Shape:
    rank = builtins.max(len(a), len(b))
    out: list[int] = [1] * rank
    for i in range(rank):
        ai = 1 if i + len(a) < rank else a[i - (rank - len(a))]
        bi = 1 if i + len(b) < rank else b[i - (rank - len(b))]
        if ai != bi and ai != 1 and bi != 1:
            raise ValueError(
                "cannot broadcast shapes "
                + shape_to_string(a)
                + " and "
                + shape_to_string(b)
            )
        out[i] = builtins.max(ai, bi)
    return tuple(out)


def _broadcast_offset(out_index: int, out_shape: Shape, in_shape: Shape) -> int:
    if not in_shape:
        return 0

    out_strides = _strides_for(out_shape)
    in_strides = _strides_for(in_shape)
    rank_delta = len(out_shape) - len(in_shape)
    offset = 0
    for i, dim in enumerate(in_shape):
        out_dim = i + rank_delta
        coord = (out_index // out_strides[out_dim]) % out_shape[out_dim]
        in_coord = 0 if dim == 1 else coord
        offset += in_coord * in_strides[i]
    return offset


def add(a: Tensor, b: Tensor) -> Tensor:
    _require_defined(a, "a")
    _require_defined(b, "b")
    out_shape = _broadcast_shape(a.shape, b.shape)
    out = [0.0] * numel(out_shape)

    for i in range(len(out)):
        out[i] = a.data[_broadcast_offset(i, out_shape, a.shape)] + b.data[
            _broadcast_offset(i, out_shape, b.shape)
        ]

    requires_grad = a.requires_grad or b.requires_grad

    def backward(self: _Node) -> None:
        for i, grad in enumerate(self.grad):
            if a.requires_grad:
                a.grad[_broadcast_offset(i, out_shape, a.shape)] += grad
            if b.requires_grad:
                b.grad[_broadcast_offset(i, out_shape, b.shape)] += grad

    return Tensor.create(out, out_shape, requires_grad, "add", [a, b], backward)


def multiply(a: Tensor, b: Tensor) -> Tensor:
    _require_defined(a, "a")
    _require_defined(b, "b")
    out_shape = _broadcast_shape(a.shape, b.shape)
    out = [0.0] * numel(out_shape)

    for i in range(len(out)):
        ai = _broadcast_offset(i, out_shape, a.shape)
        bi = _broadcast_offset(i, out_shape, b.shape)
        out[i] = a.data[ai] * b.data[bi]

    requires_grad = a.requires_grad or b.requires_grad

    def backward(self: _Node) -> None:
        for i, grad in enumerate(self.grad):
            ai = _broadcast_offset(i, out_shape, a.shape)
            bi = _broadcast_offset(i, out_shape, b.shape)
            if a.requires_grad:
                a.grad[ai] += grad * b.data[bi]
            if b.requires_grad:
                b.grad[bi] += grad * a.data[ai]

    return Tensor.create(out, out_shape, requires_grad, "multiply", [a, b], backward)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    _require_defined(a, "a")
    _require_defined(b, "b")
    _require_rank(a, 2, "matmul")
    _require_rank(b, 2, "matmul")

    m, n = a.shape
    b_rows, p = b.shape
    if n != b_rows:
        raise ValueError(
            "matmul shape mismatch: "
            + shape_to_string(a.shape)
            + " x "
            + shape_to_string(b.shape)
        )

    out = [0.0] * (m * p)
    for i in range(m):
        for k in range(n):
            av = a.data[i * n + k]
            for j in range(p):
                out[i * p + j] += av * b.data[k * p + j]

    requires_grad = a.requires_grad or b.requires_grad

    def backward(self: _Node) -> None:
        for i in range(m):
            for j in range(p):
                go = self.grad[i * p + j]
                if a.requires_grad:
                    for k in range(n):
                        a.grad[i * n + k] += go * b.data[k * p + j]
                if b.requires_grad:
                    for k in range(n):
                        b.grad[k * p + j] += a.data[i * n + k] * go

    return Tensor.create(out, (m, p), requires_grad, "matmul", [a, b], backward)


def sum(x: Tensor) -> Tensor:  # noqa: A001 - keep the C++ API name.
    _require_defined(x, "x")
    total = builtins.sum(x.data)

    def backward(self: _Node) -> None:
        if not x.requires_grad:
            return
        for i in range(len(x.grad)):
            x.grad[i] += self.grad[0]

    return Tensor.create([total], (), x.requires_grad, "sum", [x], backward)


def mean(x: Tensor) -> Tensor:
    _require_defined(x, "x")
    if x.size == 0:
        raise ValueError("mean of empty tensor")
    inv = 1.0 / float(x.size)
    total = builtins.sum(x.data)

    def backward(self: _Node) -> None:
        if not x.requires_grad:
            return
        for i in range(len(x.grad)):
            x.grad[i] += self.grad[0] * inv

    return Tensor.create([total * inv], (), x.requires_grad, "mean", [x], backward)


def tanh(x: Tensor) -> Tensor:
    _require_defined(x, "x")
    out = [math.tanh(value) for value in x.data]

    def backward(self: _Node) -> None:
        if not x.requires_grad:
            return
        for i, grad in enumerate(self.grad):
            x.grad[i] += grad * (1.0 - self.data[i] * self.data[i])

    return Tensor.create(out, x.shape, x.requires_grad, "tanh", [x], backward)


def relu(x: Tensor) -> Tensor:
    _require_defined(x, "x")
    out = [builtins.max(0.0, value) for value in x.data]

    def backward(self: _Node) -> None:
        if not x.requires_grad:
            return
        for i, grad in enumerate(self.grad):
            x.grad[i] += grad * (1.0 if x.data[i] > 0.0 else 0.0)

    return Tensor.create(out, x.shape, x.requires_grad, "relu", [x], backward)


def gelu(x: Tensor) -> Tensor:
    _require_defined(x, "x")
    cubic = 0.044715
    scale = math.sqrt(2.0 / math.pi)
    out: list[float] = []
    for value in x.data:
        inner = scale * (value + cubic * value * value * value)
        out.append(0.5 * value * (1.0 + math.tanh(inner)))

    def backward(self: _Node) -> None:
        if not x.requires_grad:
            return
        for i, grad in enumerate(self.grad):
            value = x.data[i]
            inner = scale * (value + cubic * value * value * value)
            tanh_inner = math.tanh(inner)
            sech2 = 1.0 - tanh_inner * tanh_inner
            d_inner = scale * (1.0 + 3.0 * cubic * value * value)
            local_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * value * sech2 * d_inner
            x.grad[i] += grad * local_grad

    return Tensor.create(out, x.shape, x.requires_grad, "gelu", [x], backward)


def exp(x: Tensor) -> Tensor:
    _require_defined(x, "x")
    out = [math.exp(value) for value in x.data]

    def backward(self: _Node) -> None:
        if not x.requires_grad:
            return
        for i, grad in enumerate(self.grad):
            x.grad[i] += grad * self.data[i]

    return Tensor.create(out, x.shape, x.requires_grad, "exp", [x], backward)


def log(x: Tensor) -> Tensor:
    _require_defined(x, "x")
    out: list[float] = []
    for value in x.data:
        if value <= 0.0:
            raise ValueError("log requires positive inputs")
        out.append(math.log(value))

    def backward(self: _Node) -> None:
        if not x.requires_grad:
            return
        for i, grad in enumerate(self.grad):
            x.grad[i] += grad / x.data[i]

    return Tensor.create(out, x.shape, x.requires_grad, "log", [x], backward)


def transpose(x: Tensor) -> Tensor:
    _require_defined(x, "x")
    _require_rank(x, 2, "transpose")
    rows, cols = x.shape
    out = [0.0] * x.size
    for r in range(rows):
        for c in range(cols):
            out[c * rows + r] = x.data[r * cols + c]

    def backward(self: _Node) -> None:
        if not x.requires_grad:
            return
        for r in range(rows):
            for c in range(cols):
                x.grad[r * cols + c] += self.grad[c * rows + r]

    return Tensor.create(out, (cols, rows), x.requires_grad, "transpose", [x], backward)


def reshape(x: Tensor, shape: Sequence[int] | int | None) -> Tensor:
    _require_defined(x, "x")
    normalized_shape = _shape(shape)
    if x.size != numel(normalized_shape):
        raise ValueError(
            "reshape from "
            + shape_to_string(x.shape)
            + " to "
            + shape_to_string(normalized_shape)
            + " changes element count"
        )

    def backward(self: _Node) -> None:
        if not x.requires_grad:
            return
        for i, grad in enumerate(self.grad):
            x.grad[i] += grad

    return Tensor.create(list(x.data), normalized_shape, x.requires_grad, "reshape", [x], backward)


def embedding_lookup(table: Tensor, indices: Sequence[int]) -> Tensor:
    _require_defined(table, "table")
    _require_rank(table, 2, "embedding_lookup")
    rows, dim = table.shape
    out = [0.0] * (len(indices) * dim)

    for i, index in enumerate(indices):
        if index < 0 or index >= rows:
            raise ValueError("embedding index out of range")
        start = index * dim
        out[i * dim : (i + 1) * dim] = table.data[start : start + dim]

    def backward(self: _Node) -> None:
        if not table.requires_grad:
            return
        for i, index in enumerate(indices):
            start = int(index) * dim
            for d in range(dim):
                table.grad[start + d] += self.grad[i * dim + d]

    return Tensor.create(
        out,
        (len(indices), dim),
        table.requires_grad,
        "embedding_lookup",
        [table],
        backward,
    )


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    _require_defined(x, "x")
    _require_defined(gamma, "gamma")
    _require_defined(beta, "beta")
    _require_rank(x, 2, "layer_norm")
    _require_rank(gamma, 1, "layer_norm")
    _require_rank(beta, 1, "layer_norm")
    if eps <= 0.0:
        raise ValueError("layer_norm eps must be positive")

    rows, dim = x.shape
    if gamma.shape[0] != dim or beta.shape[0] != dim:
        raise ValueError(
            "layer_norm parameter shape mismatch: x "
            + shape_to_string(x.shape)
            + ", gamma "
            + shape_to_string(gamma.shape)
            + ", beta "
            + shape_to_string(beta.shape)
        )

    normalized = [0.0] * x.size
    inv_std = [0.0] * rows
    out = [0.0] * x.size

    for row in range(rows):
        base = row * dim
        mean_value = builtins.sum(x.data[base : base + dim]) / float(dim)
        variance = 0.0
        for d in range(dim):
            centered = x.data[base + d] - mean_value
            variance += centered * centered
        variance /= float(dim)
        inv_std[row] = 1.0 / math.sqrt(variance + eps)

        for d in range(dim):
            idx = base + d
            normalized[idx] = (x.data[idx] - mean_value) * inv_std[row]
            out[idx] = normalized[idx] * gamma.data[d] + beta.data[d]

    requires_grad = x.requires_grad or gamma.requires_grad or beta.requires_grad

    def backward(self: _Node) -> None:
        if gamma.requires_grad or beta.requires_grad:
            for row in range(rows):
                base = row * dim
                for d in range(dim):
                    idx = base + d
                    if gamma.requires_grad:
                        gamma.grad[d] += self.grad[idx] * normalized[idx]
                    if beta.requires_grad:
                        beta.grad[d] += self.grad[idx]

        if not x.requires_grad:
            return

        for row in range(rows):
            base = row * dim
            sum_dxhat = 0.0
            sum_dxhat_xhat = 0.0
            for d in range(dim):
                idx = base + d
                dxhat = self.grad[idx] * gamma.data[d]
                sum_dxhat += dxhat
                sum_dxhat_xhat += dxhat * normalized[idx]

            scale = inv_std[row] / float(dim)
            for d in range(dim):
                idx = base + d
                dxhat = self.grad[idx] * gamma.data[d]
                x.grad[idx] += scale * (
                    float(dim) * dxhat - sum_dxhat - normalized[idx] * sum_dxhat_xhat
                )

    return Tensor.create(out, x.shape, requires_grad, "layer_norm", [x, gamma, beta], backward)


def _attention_offset(
    batch: int,
    time: int,
    head: int,
    dim: int,
    context: int,
    heads: int,
    head_dim: int,
) -> int:
    return (batch * context + time) * heads * head_dim + head * head_dim + dim


def _causal_attention_core(
    q: Tensor, k: Tensor, v: Tensor, batch_size: int, context: int, heads: int
) -> Tensor:
    _require_defined(q, "q")
    _require_defined(k, "k")
    _require_defined(v, "v")
    _require_rank(q, 2, "causal_attention_core")
    _require_rank(k, 2, "causal_attention_core")
    _require_rank(v, 2, "causal_attention_core")
    if batch_size == 0 or context == 0 or heads == 0:
        raise ValueError("causal attention batch_size, context, and heads must be non-zero")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("causal attention q/k/v shape mismatch")
    if q.shape[0] != batch_size * context:
        raise ValueError("causal attention input rows must equal batch_size * context")
    embed = q.shape[1]
    if embed == 0 or embed % heads != 0:
        raise ValueError("causal attention requires embed divisible by heads")

    head_dim = embed // heads
    score_scale = 1.0 / math.sqrt(float(head_dim))
    out = [0.0] * q.size
    probabilities = [0.0] * (batch_size * heads * context * context)

    for b in range(batch_size):
        for h in range(heads):
            for t in range(context):
                # Causal masking is enforced by only scoring source positions
                # s <= t; future positions stay at exactly zero probability.
                max_score = -math.inf
                for s in range(t + 1):
                    dot = 0.0
                    for d in range(head_dim):
                        dot += q.data[_attention_offset(b, t, h, d, context, heads, head_dim)] * k.data[
                            _attention_offset(b, s, h, d, context, heads, head_dim)
                        ]
                    score = dot * score_scale
                    prob_idx = ((b * heads + h) * context + t) * context + s
                    probabilities[prob_idx] = score
                    max_score = builtins.max(max_score, score)

                sum_exp = 0.0
                for s in range(t + 1):
                    prob_idx = ((b * heads + h) * context + t) * context + s
                    exp_score = math.exp(probabilities[prob_idx] - max_score)
                    probabilities[prob_idx] = exp_score
                    sum_exp += exp_score

                for s in range(t + 1):
                    prob_idx = ((b * heads + h) * context + t) * context + s
                    prob = probabilities[prob_idx] / sum_exp
                    probabilities[prob_idx] = prob
                    for d in range(head_dim):
                        # The flat row-major activation represents
                        # [batch, time, head, head_dim].
                        out[_attention_offset(b, t, h, d, context, heads, head_dim)] += prob * v.data[
                            _attention_offset(b, s, h, d, context, heads, head_dim)
                        ]

    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    def backward(self: _Node) -> None:
        dprob = [0.0] * context

        for b in range(batch_size):
            for h in range(heads):
                for t in range(context):
                    for i in range(context):
                        dprob[i] = 0.0

                    for s in range(t + 1):
                        grad_prob = 0.0
                        for d in range(head_dim):
                            out_idx = _attention_offset(b, t, h, d, context, heads, head_dim)
                            v_idx = _attention_offset(b, s, h, d, context, heads, head_dim)
                            grad_prob += self.grad[out_idx] * v.data[v_idx]
                            if v.requires_grad:
                                prob = probabilities[((b * heads + h) * context + t) * context + s]
                                v.grad[v_idx] += prob * self.grad[out_idx]
                        dprob[s] = grad_prob

                    expected_grad_prob = 0.0
                    for s in range(t + 1):
                        prob = probabilities[((b * heads + h) * context + t) * context + s]
                        expected_grad_prob += dprob[s] * prob

                    for s in range(t + 1):
                        prob = probabilities[((b * heads + h) * context + t) * context + s]
                        # Softmax backward over the valid causal row:
                        # dscore_i = p_i * (dprob_i - sum_j dprob_j * p_j).
                        grad_score = prob * (dprob[s] - expected_grad_prob)
                        for d in range(head_dim):
                            q_idx = _attention_offset(b, t, h, d, context, heads, head_dim)
                            k_idx = _attention_offset(b, s, h, d, context, heads, head_dim)
                            if q.requires_grad:
                                q.grad[q_idx] += grad_score * k.data[k_idx] * score_scale
                            if k.requires_grad:
                                k.grad[k_idx] += grad_score * q.data[q_idx] * score_scale

    return Tensor.create(
        out,
        q.shape,
        requires_grad,
        "causal_attention_core",
        [q, k, v],
        backward,
    )


def causal_self_attention(
    x: Tensor,
    wq: Tensor,
    bq: Tensor,
    wk: Tensor,
    bk: Tensor,
    wv: Tensor,
    bv: Tensor,
    wo: Tensor,
    bo: Tensor,
    batch_size: int,
    context: int,
    heads: int,
) -> Tensor:
    for name, tensor in (
        ("x", x),
        ("wq", wq),
        ("bq", bq),
        ("wk", wk),
        ("bk", bk),
        ("wv", wv),
        ("bv", bv),
        ("wo", wo),
        ("bo", bo),
    ):
        _require_defined(tensor, name)
    _require_rank(x, 2, "causal_self_attention")
    for tensor in (wq, wk, wv, wo):
        _require_rank(tensor, 2, "causal_self_attention")
    for tensor in (bq, bk, bv, bo):
        _require_rank(tensor, 1, "causal_self_attention")
    if batch_size == 0 or context == 0 or heads == 0:
        raise ValueError("causal self-attention dimensions must be non-zero")
    if x.shape[0] != batch_size * context:
        raise ValueError("causal self-attention x rows must equal batch_size * context")

    embed = x.shape[1]
    weight_shape = (embed, embed)
    bias_shape = (embed,)
    if (
        wq.shape != weight_shape
        or wk.shape != weight_shape
        or wv.shape != weight_shape
        or wo.shape != weight_shape
        or bq.shape != bias_shape
        or bk.shape != bias_shape
        or bv.shape != bias_shape
        or bo.shape != bias_shape
    ):
        raise ValueError("causal self-attention projection shape mismatch")
    if embed % heads != 0:
        raise ValueError("causal self-attention requires embed divisible by heads")

    q = add(matmul(x, wq), bq)
    k = add(matmul(x, wk), bk)
    v = add(matmul(x, wv), bv)
    attended = _causal_attention_core(q, k, v, batch_size, context, heads)
    return add(matmul(attended, wo), bo)


def softmax_cross_entropy(logits: Tensor, targets: Sequence[int]) -> Tensor:
    _require_defined(logits, "logits")
    _require_rank(logits, 2, "softmax_cross_entropy")
    batch, classes = logits.shape
    if len(targets) != batch:
        raise ValueError("target count does not match logits batch")

    probabilities = [0.0] * (batch * classes)
    loss = 0.0
    for row in range(batch):
        target = int(targets[row])
        if target < 0 or target >= classes:
            raise ValueError("target class out of range")
        base = row * classes
        max_logit = builtins.max(logits.data[base : base + classes])
        sum_exp = 0.0
        for cls in range(classes):
            value = math.exp(logits.data[base + cls] - max_logit)
            probabilities[base + cls] = value
            sum_exp += value

        log_sum_exp = max_logit + math.log(sum_exp)
        loss += log_sum_exp - logits.data[base + target]

        for cls in range(classes):
            probabilities[base + cls] /= sum_exp

    loss /= float(batch)

    def backward(self: _Node) -> None:
        if not logits.requires_grad:
            return
        scale = self.grad[0] / float(batch)
        for row in range(batch):
            base = row * classes
            target = int(targets[row])
            for cls in range(classes):
                grad = probabilities[base + cls]
                if cls == target:
                    grad -= 1.0
                logits.grad[base + cls] += scale * grad

    return Tensor.create(
        [loss],
        (),
        logits.requires_grad,
        "softmax_cross_entropy",
        [logits],
        backward,
    )


def zero_grad(tensors: Sequence[Tensor]) -> None:
    for tensor in tensors:
        if tensor is not None and tensor.defined:
            tensor.zero_grad()


def clip_grad_norm(tensors: Sequence[Tensor], max_norm: float) -> float:
    squared = 0.0
    for tensor in tensors:
        if tensor is None or not tensor.defined:
            continue
        for grad in tensor.grad:
            squared += grad * grad

    norm = math.sqrt(squared)
    if max_norm > 0.0 and norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for tensor in tensors:
            if tensor is None or not tensor.defined:
                continue
            for i in range(len(tensor.grad)):
                tensor.grad[i] *= scale
    return norm
