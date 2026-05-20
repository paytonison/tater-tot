#include "tater_tot.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct TTEmbeddingLookupCtx {
    int *indices;
    size_t count;
    size_t dim;
} TTEmbeddingLookupCtx;

typedef struct TTLayerNormCtx {
    double *normalized;
    double *inv_std;
    size_t rows;
    size_t dim;
} TTLayerNormCtx;

typedef struct TTAttentionCtx {
    double *probabilities;
    double *dprob;
    size_t batch_size;
    size_t context;
    size_t heads;
    size_t head_dim;
    double score_scale;
} TTAttentionCtx;

typedef struct TTSoftmaxCtx {
    int *targets;
    double *probabilities;
    size_t rows;
    size_t classes;
} TTSoftmaxCtx;

static int tt_fail(TTError *error, const char *fmt, ...) {
    va_list args;

    if (error == NULL) {
        return 0;
    }
    va_start(args, fmt);
    vsnprintf(error->message, sizeof(error->message), fmt, args);
    va_end(args);
    return 0;
}

void tt_error_clear(TTError *error) {
    if (error != NULL) {
        error->message[0] = '\0';
    }
}

static int tt_checked_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > ((size_t)-1) / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int tt_shape_equal(const TTTensor *a, const TTTensor *b) {
    size_t i;

    if (a->rank != b->rank) {
        return 0;
    }
    for (i = 0; i < a->rank; ++i) {
        if (a->dims[i] != b->dims[i]) {
            return 0;
        }
    }
    return 1;
}

static int tt_require_rank(const TTTensor *tensor,
                           size_t rank,
                           const char *op,
                           TTError *error) {
    if (tensor == NULL || tensor->data == NULL) {
        return tt_fail(error, "%s received an undefined tensor", op);
    }
    if (tensor->rank != rank) {
        return tt_fail(error, "%s expected rank %zu, got rank %zu", op, rank, tensor->rank);
    }
    return 1;
}

int tt_tensor_init(TTTensor *tensor,
                   size_t rank,
                   const size_t *dims,
                   double value,
                   int requires_grad,
                   TTError *error) {
    size_t i;

    if (tensor == NULL) {
        return tt_fail(error, "cannot initialize a NULL tensor");
    }
    if (rank > 2) {
        return tt_fail(error, "C tensor rank %zu exceeds supported rank 2", rank);
    }

    memset(tensor, 0, sizeof(*tensor));
    tensor->rank = rank;
    tensor->size = 1;
    for (i = 0; i < rank; ++i) {
        if (dims[i] == 0) {
            return tt_fail(error, "tensor dimension %zu must be non-zero", i);
        }
        if (!tt_checked_mul(tensor->size, dims[i], &tensor->size)) {
            return tt_fail(error, "tensor shape is too large");
        }
        tensor->dims[i] = dims[i];
    }

    tensor->data = (double *)malloc(tensor->size * sizeof(double));
    tensor->grad = (double *)calloc(tensor->size, sizeof(double));
    if (tensor->data == NULL || tensor->grad == NULL) {
        tt_tensor_free(tensor);
        return tt_fail(error, "failed to allocate tensor buffers");
    }

    for (i = 0; i < tensor->size; ++i) {
        tensor->data[i] = value;
    }
    tensor->requires_grad = requires_grad ? 1 : 0;
    tensor->owns_buffers = 1;
    tensor->op = TT_OP_NONE;
    return 1;
}

void tt_tensor_zero_grad(TTTensor *tensor) {
    if (tensor != NULL && tensor->grad != NULL) {
        memset(tensor->grad, 0, tensor->size * sizeof(double));
    }
}

static void tt_free_ctx(TTTensor *tensor) {
    if (tensor == NULL || tensor->ctx == NULL) {
        return;
    }

    switch (tensor->op) {
    case TT_OP_EMBEDDING_LOOKUP: {
        TTEmbeddingLookupCtx *ctx = (TTEmbeddingLookupCtx *)tensor->ctx;
        free(ctx->indices);
        free(ctx);
        break;
    }
    case TT_OP_LAYER_NORM: {
        TTLayerNormCtx *ctx = (TTLayerNormCtx *)tensor->ctx;
        free(ctx->normalized);
        free(ctx->inv_std);
        free(ctx);
        break;
    }
    case TT_OP_CAUSAL_ATTENTION_CORE: {
        TTAttentionCtx *ctx = (TTAttentionCtx *)tensor->ctx;
        free(ctx->probabilities);
        free(ctx->dprob);
        free(ctx);
        break;
    }
    case TT_OP_SOFTMAX_CROSS_ENTROPY: {
        TTSoftmaxCtx *ctx = (TTSoftmaxCtx *)tensor->ctx;
        free(ctx->targets);
        free(ctx->probabilities);
        free(ctx);
        break;
    }
    default:
        free(tensor->ctx);
        break;
    }
    tensor->ctx = NULL;
}

void tt_tensor_free(TTTensor *tensor) {
    if (tensor == NULL) {
        return;
    }
    tt_free_ctx(tensor);
    if (tensor->owns_buffers) {
        free(tensor->data);
        free(tensor->grad);
    }
    memset(tensor, 0, sizeof(*tensor));
}

void tt_tape_init(TTTape *tape) {
    if (tape != NULL) {
        memset(tape, 0, sizeof(*tape));
    }
}

static int tt_tape_push(TTTape *tape, TTTensor *tensor, TTError *error) {
    TTTensor **next;
    size_t next_capacity;

    if (tape->count == tape->capacity) {
        next_capacity = tape->capacity == 0 ? 32 : tape->capacity * 2;
        next = (TTTensor **)realloc(tape->items, next_capacity * sizeof(*tape->items));
        if (next == NULL) {
            return tt_fail(error, "failed to grow autograd tape");
        }
        tape->items = next;
        tape->capacity = next_capacity;
    }
    tape->items[tape->count++] = tensor;
    return 1;
}

static TTTensor *tt_tape_tensor(TTTape *tape,
                                size_t rank,
                                const size_t *dims,
                                int requires_grad,
                                TTOp op,
                                TTTensor **parents,
                                size_t parent_count,
                                TTBackwardFn backward,
                                TTError *error) {
    TTTensor *tensor;
    size_t i;

    tensor = (TTTensor *)calloc(1, sizeof(*tensor));
    if (tensor == NULL) {
        tt_fail(error, "failed to allocate tape tensor");
        return NULL;
    }
    if (!tt_tensor_init(tensor, rank, dims, 0.0, requires_grad, error)) {
        free(tensor);
        return NULL;
    }
    tensor->op = op;
    tensor->backward = backward;
    tensor->parent_count = parent_count;
    for (i = 0; i < parent_count && i < 4; ++i) {
        tensor->parents[i] = parents[i];
    }
    if (!tt_tape_push(tape, tensor, error)) {
        tt_tensor_free(tensor);
        free(tensor);
        return NULL;
    }
    return tensor;
}

void tt_tape_free(TTTape *tape) {
    size_t i;

    if (tape == NULL) {
        return;
    }
    for (i = 0; i < tape->count; ++i) {
        tt_tensor_free(tape->items[i]);
        free(tape->items[i]);
    }
    free(tape->items);
    memset(tape, 0, sizeof(*tape));
}

int tt_tape_backward(TTTape *tape, TTTensor *loss, TTError *error) {
    size_t i;

    if (tape == NULL || loss == NULL) {
        return tt_fail(error, "backward requires a tape and a loss tensor");
    }
    if (loss->rank != 0 || loss->size != 1) {
        return tt_fail(error, "backward requires a scalar loss");
    }

    for (i = 0; i < tape->count; ++i) {
        tt_tensor_zero_grad(tape->items[i]);
    }
    loss->grad[0] = 1.0;

    for (i = tape->count; i > 0; --i) {
        TTTensor *tensor = tape->items[i - 1];
        if (tensor->backward != NULL) {
            tensor->backward(tensor);
        }
    }
    return 1;
}

/*
 * The C port uses a tiny deterministic generator instead of std::mt19937 or
 * Python's random.Random. This means same-seed samples are reproducible within
 * C, but exact initialization and sampling streams intentionally differ from
 * the C++/Python ports.
 */
void tt_rng_seed(TTRng *rng, uint64_t seed) {
    if (rng == NULL) {
        return;
    }
    rng->state = seed == 0 ? 0x9e3779b97f4a7c15ULL : seed;
    rng->has_spare_normal = 0;
    rng->spare_normal = 0.0;
    (void)tt_rng_u32(rng);
}

uint32_t tt_rng_u32(TTRng *rng) {
    uint64_t x;

    x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return (uint32_t)((x * 2685821657736338717ULL) >> 32);
}

double tt_rng_uniform(TTRng *rng) {
    return ((double)tt_rng_u32(rng) + 0.5) / 4294967296.0;
}

double tt_rng_normal(TTRng *rng) {
    double u1;
    double u2;
    double radius;
    double theta;

    if (rng->has_spare_normal) {
        rng->has_spare_normal = 0;
        return rng->spare_normal;
    }

    u1 = tt_rng_uniform(rng);
    u2 = tt_rng_uniform(rng);
    if (u1 < DBL_MIN) {
        u1 = DBL_MIN;
    }
    radius = sqrt(-2.0 * log(u1));
    theta = 2.0 * acos(-1.0) * u2;
    rng->spare_normal = radius * sin(theta);
    rng->has_spare_normal = 1;
    return radius * cos(theta);
}

static size_t tt_rng_bounded(TTRng *rng, size_t bound) {
    if (bound == 0) {
        return 0;
    }
    return (size_t)(tt_rng_uniform(rng) * (double)bound);
}

static void tt_strides(size_t rank, const size_t *dims, size_t *strides) {
    if (rank == 0) {
        return;
    }
    strides[rank - 1] = 1;
    while (rank > 1) {
        --rank;
        strides[rank - 1] = strides[rank] * dims[rank];
    }
}

static int tt_broadcast_shape(const TTTensor *a,
                              const TTTensor *b,
                              size_t *out_rank,
                              size_t *out_dims,
                              TTError *error) {
    size_t rank;
    size_t i;

    rank = a->rank > b->rank ? a->rank : b->rank;
    for (i = 0; i < rank; ++i) {
        size_t ai = 1;
        size_t bi = 1;
        size_t a_pos;
        size_t b_pos;

        if (i + a->rank >= rank) {
            a_pos = i - (rank - a->rank);
            ai = a->dims[a_pos];
        }
        if (i + b->rank >= rank) {
            b_pos = i - (rank - b->rank);
            bi = b->dims[b_pos];
        }
        if (ai != bi && ai != 1 && bi != 1) {
            return tt_fail(error, "cannot broadcast tensor shapes");
        }
        out_dims[i] = ai > bi ? ai : bi;
    }
    *out_rank = rank;
    return 1;
}

static size_t tt_broadcast_offset(size_t out_index,
                                  size_t out_rank,
                                  const size_t *out_dims,
                                  size_t in_rank,
                                  const size_t *in_dims) {
    size_t out_strides[2] = {1, 1};
    size_t in_strides[2] = {1, 1};
    size_t rank_delta;
    size_t offset = 0;
    size_t i;

    if (in_rank == 0) {
        return 0;
    }
    tt_strides(out_rank, out_dims, out_strides);
    tt_strides(in_rank, in_dims, in_strides);
    rank_delta = out_rank - in_rank;

    for (i = 0; i < in_rank; ++i) {
        size_t out_dim = i + rank_delta;
        size_t coord = (out_index / out_strides[out_dim]) % out_dims[out_dim];
        size_t in_coord = in_dims[i] == 1 ? 0 : coord;
        offset += in_coord * in_strides[i];
    }
    return offset;
}

static void tt_backward_add(TTTensor *self) {
    TTTensor *a = self->parents[0];
    TTTensor *b = self->parents[1];
    size_t i;

    for (i = 0; i < self->size; ++i) {
        if (a->requires_grad) {
            a->grad[tt_broadcast_offset(i, self->rank, self->dims, a->rank, a->dims)] +=
                self->grad[i];
        }
        if (b->requires_grad) {
            b->grad[tt_broadcast_offset(i, self->rank, self->dims, b->rank, b->dims)] +=
                self->grad[i];
        }
    }
}

static TTTensor *tt_add(TTTape *tape, TTTensor *a, TTTensor *b, TTError *error) {
    TTTensor *parents[2];
    TTTensor *out;
    size_t out_rank = 0;
    size_t out_dims[2] = {0, 0};
    size_t i;

    if (a == NULL || b == NULL || a->data == NULL || b->data == NULL) {
        tt_fail(error, "add received an undefined tensor");
        return NULL;
    }
    if (!tt_broadcast_shape(a, b, &out_rank, out_dims, error)) {
        return NULL;
    }
    parents[0] = a;
    parents[1] = b;
    out = tt_tape_tensor(tape,
                         out_rank,
                         out_dims,
                         a->requires_grad || b->requires_grad,
                         TT_OP_ADD,
                         parents,
                         2,
                         tt_backward_add,
                         error);
    if (out == NULL) {
        return NULL;
    }

    for (i = 0; i < out->size; ++i) {
        out->data[i] =
            a->data[tt_broadcast_offset(i, out_rank, out_dims, a->rank, a->dims)] +
            b->data[tt_broadcast_offset(i, out_rank, out_dims, b->rank, b->dims)];
    }
    return out;
}

static void tt_backward_matmul(TTTensor *self) {
    TTTensor *a = self->parents[0];
    TTTensor *b = self->parents[1];
    size_t m = a->dims[0];
    size_t n = a->dims[1];
    size_t p = b->dims[1];
    size_t i;
    size_t j;
    size_t k;

    for (i = 0; i < m; ++i) {
        for (j = 0; j < p; ++j) {
            double go = self->grad[i * p + j];
            if (a->requires_grad) {
                for (k = 0; k < n; ++k) {
                    a->grad[i * n + k] += go * b->data[k * p + j];
                }
            }
            if (b->requires_grad) {
                for (k = 0; k < n; ++k) {
                    b->grad[k * p + j] += a->data[i * n + k] * go;
                }
            }
        }
    }
}

static TTTensor *tt_matmul(TTTape *tape, TTTensor *a, TTTensor *b, TTError *error) {
    TTTensor *parents[2];
    TTTensor *out;
    size_t dims[2];
    size_t m;
    size_t n;
    size_t p;
    size_t i;
    size_t j;
    size_t k;

    if (!tt_require_rank(a, 2, "matmul", error) || !tt_require_rank(b, 2, "matmul", error)) {
        return NULL;
    }
    if (a->dims[1] != b->dims[0]) {
        tt_fail(error, "matmul shape mismatch");
        return NULL;
    }
    m = a->dims[0];
    n = a->dims[1];
    p = b->dims[1];
    dims[0] = m;
    dims[1] = p;
    parents[0] = a;
    parents[1] = b;
    out = tt_tape_tensor(tape,
                         2,
                         dims,
                         a->requires_grad || b->requires_grad,
                         TT_OP_MATMUL,
                         parents,
                         2,
                         tt_backward_matmul,
                         error);
    if (out == NULL) {
        return NULL;
    }

    for (i = 0; i < m; ++i) {
        for (k = 0; k < n; ++k) {
            double av = a->data[i * n + k];
            for (j = 0; j < p; ++j) {
                out->data[i * p + j] += av * b->data[k * p + j];
            }
        }
    }
    return out;
}

static void tt_backward_gelu(TTTensor *self) {
    TTTensor *x = self->parents[0];
    const double cubic = 0.044715;
    const double scale = sqrt(2.0 / acos(-1.0));
    size_t i;

    if (!x->requires_grad) {
        return;
    }
    for (i = 0; i < self->size; ++i) {
        double v = x->data[i];
        double inner = scale * (v + cubic * v * v * v);
        double tanh_inner = tanh(inner);
        double sech2 = 1.0 - tanh_inner * tanh_inner;
        double d_inner = scale * (1.0 + 3.0 * cubic * v * v);
        double local_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * v * sech2 * d_inner;
        x->grad[i] += self->grad[i] * local_grad;
    }
}

static TTTensor *tt_gelu(TTTape *tape, TTTensor *x, TTError *error) {
    TTTensor *parents[1];
    TTTensor *out;
    const double cubic = 0.044715;
    const double scale = sqrt(2.0 / acos(-1.0));
    size_t i;

    if (x == NULL || x->data == NULL) {
        tt_fail(error, "gelu received an undefined tensor");
        return NULL;
    }
    parents[0] = x;
    out = tt_tape_tensor(tape,
                         x->rank,
                         x->dims,
                         x->requires_grad,
                         TT_OP_GELU,
                         parents,
                         1,
                         tt_backward_gelu,
                         error);
    if (out == NULL) {
        return NULL;
    }

    for (i = 0; i < x->size; ++i) {
        double v = x->data[i];
        double inner = scale * (v + cubic * v * v * v);
        out->data[i] = 0.5 * v * (1.0 + tanh(inner));
    }
    return out;
}

static void tt_backward_embedding_lookup(TTTensor *self) {
    TTTensor *table = self->parents[0];
    TTEmbeddingLookupCtx *ctx = (TTEmbeddingLookupCtx *)self->ctx;
    size_t i;
    size_t d;

    if (!table->requires_grad) {
        return;
    }
    for (i = 0; i < ctx->count; ++i) {
        size_t row = (size_t)ctx->indices[i];
        for (d = 0; d < ctx->dim; ++d) {
            table->grad[row * ctx->dim + d] += self->grad[i * ctx->dim + d];
        }
    }
}

static TTTensor *tt_embedding_lookup(TTTape *tape,
                                     TTTensor *table,
                                     const int *indices,
                                     size_t count,
                                     TTError *error) {
    TTTensor *parents[1];
    TTTensor *out;
    TTEmbeddingLookupCtx *ctx;
    size_t dims[2];
    size_t rows;
    size_t dim;
    size_t i;

    if (!tt_require_rank(table, 2, "embedding_lookup", error)) {
        return NULL;
    }
    rows = table->dims[0];
    dim = table->dims[1];
    for (i = 0; i < count; ++i) {
        if (indices[i] < 0 || (size_t)indices[i] >= rows) {
            tt_fail(error, "embedding index out of range");
            return NULL;
        }
    }

    ctx = (TTEmbeddingLookupCtx *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        tt_fail(error, "failed to allocate embedding lookup context");
        return NULL;
    }
    if (count > 0) {
        ctx->indices = (int *)malloc(count * sizeof(int));
        if (ctx->indices == NULL) {
            free(ctx);
            tt_fail(error, "failed to copy embedding lookup indices");
            return NULL;
        }
        memcpy(ctx->indices, indices, count * sizeof(int));
    }
    ctx->count = count;
    ctx->dim = dim;

    dims[0] = count;
    dims[1] = dim;
    parents[0] = table;
    out = tt_tape_tensor(tape,
                         2,
                         dims,
                         table->requires_grad,
                         TT_OP_EMBEDDING_LOOKUP,
                         parents,
                         1,
                         tt_backward_embedding_lookup,
                         error);
    if (out == NULL) {
        free(ctx->indices);
        free(ctx);
        return NULL;
    }
    out->ctx = ctx;

    for (i = 0; i < count; ++i) {
        size_t row = (size_t)indices[i];
        memcpy(out->data + i * dim, table->data + row * dim, dim * sizeof(double));
    }
    return out;
}

static void tt_backward_layer_norm(TTTensor *self) {
    TTTensor *x = self->parents[0];
    TTTensor *gamma = self->parents[1];
    TTTensor *beta = self->parents[2];
    TTLayerNormCtx *ctx = (TTLayerNormCtx *)self->ctx;
    size_t row;
    size_t d;

    if (gamma->requires_grad || beta->requires_grad) {
        for (row = 0; row < ctx->rows; ++row) {
            size_t base = row * ctx->dim;
            for (d = 0; d < ctx->dim; ++d) {
                size_t idx = base + d;
                if (gamma->requires_grad) {
                    gamma->grad[d] += self->grad[idx] * ctx->normalized[idx];
                }
                if (beta->requires_grad) {
                    beta->grad[d] += self->grad[idx];
                }
            }
        }
    }

    if (!x->requires_grad) {
        return;
    }

    for (row = 0; row < ctx->rows; ++row) {
        size_t base = row * ctx->dim;
        double sum_dxhat = 0.0;
        double sum_dxhat_xhat = 0.0;
        double scale;

        for (d = 0; d < ctx->dim; ++d) {
            size_t idx = base + d;
            double dxhat = self->grad[idx] * gamma->data[d];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * ctx->normalized[idx];
        }

        scale = ctx->inv_std[row] / (double)ctx->dim;
        for (d = 0; d < ctx->dim; ++d) {
            size_t idx = base + d;
            double dxhat = self->grad[idx] * gamma->data[d];
            x->grad[idx] += scale *
                            ((double)ctx->dim * dxhat - sum_dxhat -
                             ctx->normalized[idx] * sum_dxhat_xhat);
        }
    }
}

static TTTensor *tt_layer_norm(TTTape *tape,
                               TTTensor *x,
                               TTTensor *gamma,
                               TTTensor *beta,
                               double eps,
                               TTError *error) {
    TTTensor *parents[3];
    TTTensor *out;
    TTLayerNormCtx *ctx;
    size_t rows;
    size_t dim;
    size_t row;
    size_t d;

    if (!tt_require_rank(x, 2, "layer_norm", error) ||
        !tt_require_rank(gamma, 1, "layer_norm", error) ||
        !tt_require_rank(beta, 1, "layer_norm", error)) {
        return NULL;
    }
    if (eps <= 0.0) {
        tt_fail(error, "layer_norm eps must be positive");
        return NULL;
    }
    rows = x->dims[0];
    dim = x->dims[1];
    if (gamma->dims[0] != dim || beta->dims[0] != dim) {
        tt_fail(error, "layer_norm parameter shape mismatch");
        return NULL;
    }

    ctx = (TTLayerNormCtx *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        tt_fail(error, "failed to allocate layer_norm context");
        return NULL;
    }
    ctx->normalized = (double *)calloc(x->size, sizeof(double));
    ctx->inv_std = (double *)calloc(rows, sizeof(double));
    if (ctx->normalized == NULL || ctx->inv_std == NULL) {
        free(ctx->normalized);
        free(ctx->inv_std);
        free(ctx);
        tt_fail(error, "failed to allocate layer_norm saved buffers");
        return NULL;
    }
    ctx->rows = rows;
    ctx->dim = dim;

    parents[0] = x;
    parents[1] = gamma;
    parents[2] = beta;
    out = tt_tape_tensor(tape,
                         x->rank,
                         x->dims,
                         x->requires_grad || gamma->requires_grad || beta->requires_grad,
                         TT_OP_LAYER_NORM,
                         parents,
                         3,
                         tt_backward_layer_norm,
                         error);
    if (out == NULL) {
        free(ctx->normalized);
        free(ctx->inv_std);
        free(ctx);
        return NULL;
    }
    out->ctx = ctx;

    for (row = 0; row < rows; ++row) {
        size_t base = row * dim;
        double mean = 0.0;
        double variance = 0.0;

        for (d = 0; d < dim; ++d) {
            mean += x->data[base + d];
        }
        mean /= (double)dim;
        for (d = 0; d < dim; ++d) {
            double centered = x->data[base + d] - mean;
            variance += centered * centered;
        }
        variance /= (double)dim;
        ctx->inv_std[row] = 1.0 / sqrt(variance + eps);
        for (d = 0; d < dim; ++d) {
            size_t idx = base + d;
            ctx->normalized[idx] = (x->data[idx] - mean) * ctx->inv_std[row];
            out->data[idx] = ctx->normalized[idx] * gamma->data[d] + beta->data[d];
        }
    }
    return out;
}

static size_t tt_attention_offset(size_t batch,
                                  size_t time,
                                  size_t head,
                                  size_t dim,
                                  size_t context,
                                  size_t heads,
                                  size_t head_dim) {
    return (batch * context + time) * heads * head_dim + head * head_dim + dim;
}

static void tt_backward_causal_attention_core(TTTensor *self) {
    TTTensor *q = self->parents[0];
    TTTensor *k = self->parents[1];
    TTTensor *v = self->parents[2];
    TTAttentionCtx *ctx = (TTAttentionCtx *)self->ctx;
    size_t b;
    size_t h;
    size_t t;
    size_t s;
    size_t d;

    for (b = 0; b < ctx->batch_size; ++b) {
        for (h = 0; h < ctx->heads; ++h) {
            for (t = 0; t < ctx->context; ++t) {
                double expected_grad_prob = 0.0;
                memset(ctx->dprob, 0, ctx->context * sizeof(double));

                for (s = 0; s <= t; ++s) {
                    double grad_prob = 0.0;
                    double prob = ctx->probabilities[((b * ctx->heads + h) * ctx->context + t) *
                                                         ctx->context +
                                                     s];
                    for (d = 0; d < ctx->head_dim; ++d) {
                        size_t out_idx = tt_attention_offset(
                            b, t, h, d, ctx->context, ctx->heads, ctx->head_dim);
                        size_t v_idx = tt_attention_offset(
                            b, s, h, d, ctx->context, ctx->heads, ctx->head_dim);
                        grad_prob += self->grad[out_idx] * v->data[v_idx];
                        if (v->requires_grad) {
                            v->grad[v_idx] += prob * self->grad[out_idx];
                        }
                    }
                    ctx->dprob[s] = grad_prob;
                    expected_grad_prob += grad_prob * prob;
                }

                for (s = 0; s <= t; ++s) {
                    double prob = ctx->probabilities[((b * ctx->heads + h) * ctx->context + t) *
                                                         ctx->context +
                                                     s];
                    double grad_score = prob * (ctx->dprob[s] - expected_grad_prob);
                    for (d = 0; d < ctx->head_dim; ++d) {
                        size_t q_idx = tt_attention_offset(
                            b, t, h, d, ctx->context, ctx->heads, ctx->head_dim);
                        size_t k_idx = tt_attention_offset(
                            b, s, h, d, ctx->context, ctx->heads, ctx->head_dim);
                        if (q->requires_grad) {
                            q->grad[q_idx] += grad_score * k->data[k_idx] * ctx->score_scale;
                        }
                        if (k->requires_grad) {
                            k->grad[k_idx] += grad_score * q->data[q_idx] * ctx->score_scale;
                        }
                    }
                }
            }
        }
    }
}

static TTTensor *tt_causal_attention_core(TTTape *tape,
                                          TTTensor *q,
                                          TTTensor *k,
                                          TTTensor *v,
                                          size_t batch_size,
                                          size_t context,
                                          size_t heads,
                                          TTError *error) {
    TTTensor *parents[3];
    TTTensor *out;
    TTAttentionCtx *ctx;
    size_t embed;
    size_t prob_count;
    size_t b;
    size_t h;
    size_t t;
    size_t s;
    size_t d;

    if (!tt_require_rank(q, 2, "causal_attention_core", error) ||
        !tt_require_rank(k, 2, "causal_attention_core", error) ||
        !tt_require_rank(v, 2, "causal_attention_core", error)) {
        return NULL;
    }
    if (!tt_shape_equal(q, k) || !tt_shape_equal(q, v)) {
        tt_fail(error, "causal attention q/k/v shape mismatch");
        return NULL;
    }
    if (batch_size == 0 || context == 0 || heads == 0 || q->dims[0] != batch_size * context) {
        tt_fail(error, "causal attention batch/context/head shape mismatch");
        return NULL;
    }
    embed = q->dims[1];
    if (embed == 0 || embed % heads != 0) {
        tt_fail(error, "causal attention requires embed divisible by heads");
        return NULL;
    }

    ctx = (TTAttentionCtx *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        tt_fail(error, "failed to allocate attention context");
        return NULL;
    }
    ctx->batch_size = batch_size;
    ctx->context = context;
    ctx->heads = heads;
    ctx->head_dim = embed / heads;
    ctx->score_scale = 1.0 / sqrt((double)ctx->head_dim);

    prob_count = batch_size * heads * context * context;
    ctx->probabilities = (double *)calloc(prob_count, sizeof(double));
    ctx->dprob = (double *)calloc(context, sizeof(double));
    if (ctx->probabilities == NULL || ctx->dprob == NULL) {
        free(ctx->probabilities);
        free(ctx->dprob);
        free(ctx);
        tt_fail(error, "failed to allocate attention saved buffers");
        return NULL;
    }

    parents[0] = q;
    parents[1] = k;
    parents[2] = v;
    out = tt_tape_tensor(tape,
                         q->rank,
                         q->dims,
                         q->requires_grad || k->requires_grad || v->requires_grad,
                         TT_OP_CAUSAL_ATTENTION_CORE,
                         parents,
                         3,
                         tt_backward_causal_attention_core,
                         error);
    if (out == NULL) {
        free(ctx->probabilities);
        free(ctx->dprob);
        free(ctx);
        return NULL;
    }
    out->ctx = ctx;

    for (b = 0; b < batch_size; ++b) {
        for (h = 0; h < heads; ++h) {
            for (t = 0; t < context; ++t) {
                double max_score = -DBL_MAX;
                double sum_exp = 0.0;

                /*
                 * Causal masking is explicit: source position s only visits
                 * values <= t. Future positions keep probability 0.
                 */
                for (s = 0; s <= t; ++s) {
                    double dot = 0.0;
                    double score;
                    size_t prob_idx = ((b * heads + h) * context + t) * context + s;
                    for (d = 0; d < ctx->head_dim; ++d) {
                        dot += q->data[tt_attention_offset(
                                   b, t, h, d, context, heads, ctx->head_dim)] *
                               k->data[tt_attention_offset(
                                   b, s, h, d, context, heads, ctx->head_dim)];
                    }
                    score = dot * ctx->score_scale;
                    ctx->probabilities[prob_idx] = score;
                    if (score > max_score) {
                        max_score = score;
                    }
                }

                for (s = 0; s <= t; ++s) {
                    size_t prob_idx = ((b * heads + h) * context + t) * context + s;
                    double exp_score = exp(ctx->probabilities[prob_idx] - max_score);
                    ctx->probabilities[prob_idx] = exp_score;
                    sum_exp += exp_score;
                }

                for (s = 0; s <= t; ++s) {
                    size_t prob_idx = ((b * heads + h) * context + t) * context + s;
                    double prob = ctx->probabilities[prob_idx] / sum_exp;
                    ctx->probabilities[prob_idx] = prob;
                    for (d = 0; d < ctx->head_dim; ++d) {
                        out->data[tt_attention_offset(
                            b, t, h, d, context, heads, ctx->head_dim)] +=
                            prob *
                            v->data[tt_attention_offset(
                                b, s, h, d, context, heads, ctx->head_dim)];
                    }
                }
            }
        }
    }
    return out;
}

static TTTensor *tt_causal_self_attention(TTTape *tape,
                                          TTTensor *x,
                                          TTTensor *wq,
                                          TTTensor *bq,
                                          TTTensor *wk,
                                          TTTensor *bk,
                                          TTTensor *wv,
                                          TTTensor *bv,
                                          TTTensor *wo,
                                          TTTensor *bo,
                                          size_t batch_size,
                                          size_t context,
                                          size_t heads,
                                          TTError *error) {
    TTTensor *q;
    TTTensor *k;
    TTTensor *v;
    TTTensor *attended;
    TTTensor *projected;
    size_t embed;

    if (!tt_require_rank(x, 2, "causal_self_attention", error)) {
        return NULL;
    }
    embed = x->dims[1];
    if (x->dims[0] != batch_size * context || embed % heads != 0) {
        tt_fail(error, "causal self-attention input shape mismatch");
        return NULL;
    }

    q = tt_add(tape, tt_matmul(tape, x, wq, error), bq, error);
    if (q == NULL) {
        return NULL;
    }
    k = tt_add(tape, tt_matmul(tape, x, wk, error), bk, error);
    if (k == NULL) {
        return NULL;
    }
    v = tt_add(tape, tt_matmul(tape, x, wv, error), bv, error);
    if (v == NULL) {
        return NULL;
    }
    attended = tt_causal_attention_core(tape, q, k, v, batch_size, context, heads, error);
    if (attended == NULL) {
        return NULL;
    }
    projected = tt_add(tape, tt_matmul(tape, attended, wo, error), bo, error);
    return projected;
}

TTTensor *tt_softmax_cross_entropy(TTTape *tape,
                                   TTTensor *logits,
                                   const int *targets,
                                   size_t target_count,
                                   TTError *error);

static void tt_backward_softmax_cross_entropy(TTTensor *self) {
    TTTensor *logits = self->parents[0];
    TTSoftmaxCtx *ctx = (TTSoftmaxCtx *)self->ctx;
    double scale = self->grad[0] / (double)ctx->rows;
    size_t row;
    size_t cls;

    if (!logits->requires_grad) {
        return;
    }
    for (row = 0; row < ctx->rows; ++row) {
        for (cls = 0; cls < ctx->classes; ++cls) {
            double grad = ctx->probabilities[row * ctx->classes + cls];
            if (cls == (size_t)ctx->targets[row]) {
                grad -= 1.0;
            }
            logits->grad[row * ctx->classes + cls] += scale * grad;
        }
    }
}

TTTensor *tt_softmax_cross_entropy(TTTape *tape,
                                   TTTensor *logits,
                                   const int *targets,
                                   size_t target_count,
                                   TTError *error) {
    TTTensor *parents[1];
    TTTensor *out;
    TTSoftmaxCtx *ctx;
    size_t rows;
    size_t classes;
    size_t scalar_rank = 0;
    double loss = 0.0;
    size_t row;
    size_t cls;

    if (!tt_require_rank(logits, 2, "softmax_cross_entropy", error)) {
        return NULL;
    }
    rows = logits->dims[0];
    classes = logits->dims[1];
    if (targets == NULL || target_count != rows) {
        tt_fail(error, "target count does not match logits batch");
        return NULL;
    }

    ctx = (TTSoftmaxCtx *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        tt_fail(error, "failed to allocate softmax context");
        return NULL;
    }
    ctx->targets = (int *)malloc(rows * sizeof(int));
    ctx->probabilities = (double *)calloc(rows * classes, sizeof(double));
    if (ctx->targets == NULL || ctx->probabilities == NULL) {
        free(ctx->targets);
        free(ctx->probabilities);
        free(ctx);
        tt_fail(error, "failed to allocate softmax saved buffers");
        return NULL;
    }
    memcpy(ctx->targets, targets, rows * sizeof(int));
    ctx->rows = rows;
    ctx->classes = classes;

    for (row = 0; row < rows; ++row) {
        double max_logit = -DBL_MAX;
        double sum_exp = 0.0;
        double log_sum_exp;
        if (targets[row] < 0 || (size_t)targets[row] >= classes) {
            free(ctx->targets);
            free(ctx->probabilities);
            free(ctx);
            tt_fail(error, "target class out of range");
            return NULL;
        }
        for (cls = 0; cls < classes; ++cls) {
            double v = logits->data[row * classes + cls];
            if (v > max_logit) {
                max_logit = v;
            }
        }
        for (cls = 0; cls < classes; ++cls) {
            double e = exp(logits->data[row * classes + cls] - max_logit);
            ctx->probabilities[row * classes + cls] = e;
            sum_exp += e;
        }
        log_sum_exp = max_logit + log(sum_exp);
        loss += log_sum_exp - logits->data[row * classes + (size_t)targets[row]];
        for (cls = 0; cls < classes; ++cls) {
            ctx->probabilities[row * classes + cls] /= sum_exp;
        }
    }
    loss /= (double)rows;

    parents[0] = logits;
    out = tt_tape_tensor(tape,
                         scalar_rank,
                         NULL,
                         logits->requires_grad,
                         TT_OP_SOFTMAX_CROSS_ENTROPY,
                         parents,
                         1,
                         tt_backward_softmax_cross_entropy,
                         error);
    if (out == NULL) {
        free(ctx->targets);
        free(ctx->probabilities);
        free(ctx);
        return NULL;
    }
    out->ctx = ctx;
    out->data[0] = loss;
    return out;
}

static double tt_xavier_stddev(size_t fan_in, size_t fan_out) {
    return sqrt(2.0 / (double)(fan_in + fan_out));
}

static int tt_tensor_init_random(TTTensor *tensor,
                                 size_t rank,
                                 const size_t *dims,
                                 double stddev,
                                 TTRng *rng,
                                 TTError *error) {
    size_t i;

    if (!tt_tensor_init(tensor, rank, dims, 0.0, 1, error)) {
        return 0;
    }
    for (i = 0; i < tensor->size; ++i) {
        tensor->data[i] = tt_rng_normal(rng) * stddev;
    }
    return 1;
}

static int tt_model_validate_config(TTModelConfig config, TTError *error) {
    if (config.vocab_size == 0) {
        return tt_fail(error, "model config vocab_size must be non-zero");
    }
    if (config.context == 0) {
        return tt_fail(error, "model config context must be non-zero");
    }
    if (config.layers == 0) {
        return tt_fail(error, "model config layers must be non-zero");
    }
    if (config.heads == 0) {
        return tt_fail(error, "model config heads must be non-zero");
    }
    if (config.embed == 0) {
        return tt_fail(error, "model config embed must be non-zero");
    }
    if (config.hidden == 0) {
        return tt_fail(error, "model config hidden must be non-zero");
    }
    if (config.embed % config.heads != 0) {
        return tt_fail(error, "model config embed must be divisible by heads");
    }
    return 1;
}

int tt_model_init(TTTinyCharModel *model,
                  TTModelConfig config,
                  uint32_t seed,
                  TTError *error) {
    TTRng rng;
    size_t dims2[2];
    size_t dims1[1];
    size_t layer;

    if (model == NULL) {
        return tt_fail(error, "cannot initialize a NULL model");
    }
    memset(model, 0, sizeof(*model));
    if (!tt_model_validate_config(config, error)) {
        return 0;
    }
    model->config = config;
    model->blocks = (TTTransformerBlock *)calloc(config.layers, sizeof(*model->blocks));
    if (model->blocks == NULL) {
        return tt_fail(error, "failed to allocate Transformer blocks");
    }

    tt_rng_seed(&rng, seed);

    dims2[0] = config.vocab_size;
    dims2[1] = config.embed;
    if (!tt_tensor_init_random(&model->token_embedding, 2, dims2, 0.02, &rng, error)) {
        goto fail;
    }
    dims2[0] = config.context;
    dims2[1] = config.embed;
    if (!tt_tensor_init_random(&model->positional_embedding, 2, dims2, 0.02, &rng, error)) {
        goto fail;
    }

    for (layer = 0; layer < config.layers; ++layer) {
        TTTransformerBlock *block = &model->blocks[layer];

        dims1[0] = config.embed;
        if (!tt_tensor_init(&block->ln1_gamma, 1, dims1, 1.0, 1, error) ||
            !tt_tensor_init(&block->ln1_beta, 1, dims1, 0.0, 1, error)) {
            goto fail;
        }

        dims2[0] = config.embed;
        dims2[1] = config.embed;
        if (!tt_tensor_init_random(
                &block->q_w, 2, dims2, tt_xavier_stddev(config.embed, config.embed), &rng, error) ||
            !tt_tensor_init(&block->q_b, 1, dims1, 0.0, 1, error) ||
            !tt_tensor_init_random(
                &block->k_w, 2, dims2, tt_xavier_stddev(config.embed, config.embed), &rng, error) ||
            !tt_tensor_init(&block->k_b, 1, dims1, 0.0, 1, error) ||
            !tt_tensor_init_random(
                &block->v_w, 2, dims2, tt_xavier_stddev(config.embed, config.embed), &rng, error) ||
            !tt_tensor_init(&block->v_b, 1, dims1, 0.0, 1, error) ||
            !tt_tensor_init_random(
                &block->out_w, 2, dims2, tt_xavier_stddev(config.embed, config.embed), &rng, error) ||
            !tt_tensor_init(&block->out_b, 1, dims1, 0.0, 1, error)) {
            goto fail;
        }

        if (!tt_tensor_init(&block->ln2_gamma, 1, dims1, 1.0, 1, error) ||
            !tt_tensor_init(&block->ln2_beta, 1, dims1, 0.0, 1, error)) {
            goto fail;
        }

        dims2[0] = config.embed;
        dims2[1] = config.hidden;
        if (!tt_tensor_init_random(
                &block->ff1_w, 2, dims2, tt_xavier_stddev(config.embed, config.hidden), &rng, error)) {
            goto fail;
        }
        dims1[0] = config.hidden;
        if (!tt_tensor_init(&block->ff1_b, 1, dims1, 0.0, 1, error)) {
            goto fail;
        }
        dims2[0] = config.hidden;
        dims2[1] = config.embed;
        if (!tt_tensor_init_random(
                &block->ff2_w, 2, dims2, tt_xavier_stddev(config.hidden, config.embed), &rng, error)) {
            goto fail;
        }
        dims1[0] = config.embed;
        if (!tt_tensor_init(&block->ff2_b, 1, dims1, 0.0, 1, error)) {
            goto fail;
        }
    }

    dims1[0] = config.embed;
    if (!tt_tensor_init(&model->ln_f_gamma, 1, dims1, 1.0, 1, error) ||
        !tt_tensor_init(&model->ln_f_beta, 1, dims1, 0.0, 1, error)) {
        goto fail;
    }
    dims2[0] = config.embed;
    dims2[1] = config.vocab_size;
    if (!tt_tensor_init_random(&model->lm_head_w,
                               2,
                               dims2,
                               tt_xavier_stddev(config.embed, config.vocab_size),
                               &rng,
                               error)) {
        goto fail;
    }
    dims1[0] = config.vocab_size;
    if (!tt_tensor_init(&model->lm_head_b, 1, dims1, 0.0, 1, error)) {
        goto fail;
    }
    return 1;

fail:
    tt_model_free(model);
    return 0;
}

void tt_model_free(TTTinyCharModel *model) {
    size_t layer;

    if (model == NULL) {
        return;
    }
    tt_tensor_free(&model->token_embedding);
    tt_tensor_free(&model->positional_embedding);
    if (model->blocks != NULL) {
        for (layer = 0; layer < model->config.layers; ++layer) {
            TTTransformerBlock *block = &model->blocks[layer];
            tt_tensor_free(&block->ln1_gamma);
            tt_tensor_free(&block->ln1_beta);
            tt_tensor_free(&block->q_w);
            tt_tensor_free(&block->q_b);
            tt_tensor_free(&block->k_w);
            tt_tensor_free(&block->k_b);
            tt_tensor_free(&block->v_w);
            tt_tensor_free(&block->v_b);
            tt_tensor_free(&block->out_w);
            tt_tensor_free(&block->out_b);
            tt_tensor_free(&block->ln2_gamma);
            tt_tensor_free(&block->ln2_beta);
            tt_tensor_free(&block->ff1_w);
            tt_tensor_free(&block->ff1_b);
            tt_tensor_free(&block->ff2_w);
            tt_tensor_free(&block->ff2_b);
        }
        free(model->blocks);
    }
    tt_tensor_free(&model->ln_f_gamma);
    tt_tensor_free(&model->ln_f_beta);
    tt_tensor_free(&model->lm_head_w);
    tt_tensor_free(&model->lm_head_b);
    memset(model, 0, sizeof(*model));
}

size_t tt_model_parameter_count(const TTTinyCharModel *model) {
    if (model == NULL) {
        return 0;
    }
    return 2 + model->config.layers * 16 + 4;
}

static TTTensor *tt_block_param(TTTransformerBlock *block, size_t index) {
    switch (index) {
    case 0:
        return &block->ln1_gamma;
    case 1:
        return &block->ln1_beta;
    case 2:
        return &block->q_w;
    case 3:
        return &block->q_b;
    case 4:
        return &block->k_w;
    case 5:
        return &block->k_b;
    case 6:
        return &block->v_w;
    case 7:
        return &block->v_b;
    case 8:
        return &block->out_w;
    case 9:
        return &block->out_b;
    case 10:
        return &block->ln2_gamma;
    case 11:
        return &block->ln2_beta;
    case 12:
        return &block->ff1_w;
    case 13:
        return &block->ff1_b;
    case 14:
        return &block->ff2_w;
    case 15:
        return &block->ff2_b;
    default:
        return NULL;
    }
}

static const char *tt_block_param_name(size_t index) {
    static const char *names[16] = {
        "ln1_gamma", "ln1_beta", "q_w",     "q_b",    "k_w",  "k_b",
        "v_w",       "v_b",      "out_w",   "out_b",  "ln2_gamma",
        "ln2_beta",  "ff1_w",    "ff1_b",   "ff2_w",  "ff2_b"};
    return index < 16 ? names[index] : "";
}

int tt_model_named_parameters(TTTinyCharModel *model,
                              TTNamedParam *params,
                              size_t capacity,
                              size_t *out_count,
                              TTError *error) {
    size_t need;
    size_t cursor = 0;
    size_t layer;
    size_t index;

    if (model == NULL || params == NULL) {
        return tt_fail(error, "named_parameters requires a model and output array");
    }
    need = tt_model_parameter_count(model);
    if (out_count != NULL) {
        *out_count = need;
    }
    if (capacity < need) {
        return tt_fail(error, "named parameter output array is too small");
    }

    snprintf(params[cursor].name, sizeof(params[cursor].name), "%s", "token_embedding");
    params[cursor++].tensor = &model->token_embedding;
    snprintf(params[cursor].name, sizeof(params[cursor].name), "%s", "positional_embedding");
    params[cursor++].tensor = &model->positional_embedding;

    for (layer = 0; layer < model->config.layers; ++layer) {
        for (index = 0; index < 16; ++index) {
            snprintf(params[cursor].name,
                     sizeof(params[cursor].name),
                     "blocks.%zu.%s",
                     layer,
                     tt_block_param_name(index));
            params[cursor].tensor = tt_block_param(&model->blocks[layer], index);
            ++cursor;
        }
    }

    snprintf(params[cursor].name, sizeof(params[cursor].name), "%s", "ln_f_gamma");
    params[cursor++].tensor = &model->ln_f_gamma;
    snprintf(params[cursor].name, sizeof(params[cursor].name), "%s", "ln_f_beta");
    params[cursor++].tensor = &model->ln_f_beta;
    snprintf(params[cursor].name, sizeof(params[cursor].name), "%s", "lm_head_w");
    params[cursor++].tensor = &model->lm_head_w;
    snprintf(params[cursor].name, sizeof(params[cursor].name), "%s", "lm_head_b");
    params[cursor++].tensor = &model->lm_head_b;
    return 1;
}

static TTTensor *tt_positional_forward(TTTape *tape,
                                       TTTinyCharModel *model,
                                       size_t batch_size,
                                       size_t sequence_length,
                                       TTError *error) {
    int *positions;
    TTTensor *out;
    size_t b;
    size_t t;

    if (batch_size == 0 || sequence_length == 0) {
        tt_fail(error, "positional embedding dimensions must be non-zero");
        return NULL;
    }
    if (sequence_length > model->config.context) {
        tt_fail(error, "position out of range for positional embedding");
        return NULL;
    }
    positions = (int *)malloc(batch_size * sequence_length * sizeof(int));
    if (positions == NULL) {
        tt_fail(error, "failed to allocate position indices");
        return NULL;
    }
    for (b = 0; b < batch_size; ++b) {
        for (t = 0; t < sequence_length; ++t) {
            positions[b * sequence_length + t] = (int)t;
        }
    }
    out = tt_embedding_lookup(tape,
                              &model->positional_embedding,
                              positions,
                              batch_size * sequence_length,
                              error);
    free(positions);
    return out;
}

TTTensor *tt_model_forward(TTTape *tape,
                           TTTinyCharModel *model,
                           const int *tokens,
                           size_t token_count,
                           size_t batch_size,
                           TTError *error) {
    TTTensor *token_vectors;
    TTTensor *position_vectors;
    TTTensor *x;
    size_t layer;

    if (tape == NULL || model == NULL || tokens == NULL) {
        tt_fail(error, "model forward requires tape, model, and tokens");
        return NULL;
    }
    if (batch_size == 0) {
        tt_fail(error, "model forward batch_size must be non-zero");
        return NULL;
    }
    if (token_count != batch_size * model->config.context) {
        tt_fail(error, "model input token count does not match batch_size * context");
        return NULL;
    }

    token_vectors = tt_embedding_lookup(tape, &model->token_embedding, tokens, token_count, error);
    if (token_vectors == NULL) {
        return NULL;
    }
    position_vectors =
        tt_positional_forward(tape, model, batch_size, model->config.context, error);
    if (position_vectors == NULL) {
        return NULL;
    }
    x = tt_add(tape, token_vectors, position_vectors, error);
    if (x == NULL) {
        return NULL;
    }

    for (layer = 0; layer < model->config.layers; ++layer) {
        TTTransformerBlock *block = &model->blocks[layer];
        TTTensor *normalized;
        TTTensor *attention;
        TTTensor *ff;

        normalized = tt_layer_norm(tape, x, &block->ln1_gamma, &block->ln1_beta, 1e-5, error);
        if (normalized == NULL) {
            return NULL;
        }
        attention = tt_causal_self_attention(tape,
                                             normalized,
                                             &block->q_w,
                                             &block->q_b,
                                             &block->k_w,
                                             &block->k_b,
                                             &block->v_w,
                                             &block->v_b,
                                             &block->out_w,
                                             &block->out_b,
                                             batch_size,
                                             model->config.context,
                                             model->config.heads,
                                             error);
        if (attention == NULL) {
            return NULL;
        }
        x = tt_add(tape, x, attention, error);
        if (x == NULL) {
            return NULL;
        }

        normalized = tt_layer_norm(tape, x, &block->ln2_gamma, &block->ln2_beta, 1e-5, error);
        if (normalized == NULL) {
            return NULL;
        }
        ff = tt_add(tape, tt_matmul(tape, normalized, &block->ff1_w, error), &block->ff1_b, error);
        if (ff == NULL) {
            return NULL;
        }
        ff = tt_gelu(tape, ff, error);
        if (ff == NULL) {
            return NULL;
        }
        ff = tt_add(tape, tt_matmul(tape, ff, &block->ff2_w, error), &block->ff2_b, error);
        if (ff == NULL) {
            return NULL;
        }
        x = tt_add(tape, x, ff, error);
        if (x == NULL) {
            return NULL;
        }
    }

    x = tt_layer_norm(tape, x, &model->ln_f_gamma, &model->ln_f_beta, 1e-5, error);
    if (x == NULL) {
        return NULL;
    }
    return tt_add(tape, tt_matmul(tape, x, &model->lm_head_w, error), &model->lm_head_b, error);
}

static int tt_signed_char_key(unsigned char value) {
    return value >= 128 ? (int)value - 256 : (int)value;
}

static int tt_compare_vocab_byte(const void *lhs, const void *rhs) {
    unsigned char a = *(const unsigned char *)lhs;
    unsigned char b = *(const unsigned char *)rhs;
    int ka = tt_signed_char_key(a);
    int kb = tt_signed_char_key(b);
    return (ka > kb) - (ka < kb);
}

static int tt_vocab_from_chars(TTVocabulary *vocab,
                               const unsigned char *chars,
                               size_t count,
                               TTError *error) {
    size_t i;

    if (vocab == NULL) {
        return tt_fail(error, "cannot initialize a NULL vocabulary");
    }
    memset(vocab, 0, sizeof(*vocab));
    for (i = 0; i < 256; ++i) {
        vocab->stoi[i] = -1;
    }
    if (count == 0) {
        return tt_fail(error, "cannot build a vocabulary from empty text");
    }
    vocab->chars = (unsigned char *)malloc(count);
    if (vocab->chars == NULL) {
        return tt_fail(error, "failed to allocate vocabulary");
    }
    memcpy(vocab->chars, chars, count);
    vocab->size = count;
    for (i = 0; i < count; ++i) {
        vocab->stoi[vocab->chars[i]] = (int)i;
    }
    return 1;
}

int tt_vocab_from_text(TTVocabulary *vocab,
                       const unsigned char *text,
                       size_t text_len,
                       TTError *error) {
    int seen[256] = {0};
    unsigned char chars[256];
    size_t count = 0;
    size_t i;

    if (text == NULL || text_len == 0) {
        return tt_fail(error, "cannot build a vocabulary from empty text");
    }
    for (i = 0; i < text_len; ++i) {
        unsigned char ch = text[i];
        if (!seen[ch]) {
            seen[ch] = 1;
            chars[count++] = ch;
        }
    }
    qsort(chars, count, sizeof(chars[0]), tt_compare_vocab_byte);
    return tt_vocab_from_chars(vocab, chars, count, error);
}

void tt_vocab_free(TTVocabulary *vocab) {
    if (vocab == NULL) {
        return;
    }
    free(vocab->chars);
    memset(vocab, 0, sizeof(*vocab));
}

int tt_vocab_encode(const TTVocabulary *vocab,
                    const unsigned char *text,
                    size_t text_len,
                    int allow_unknown,
                    TTIntBuffer *out,
                    TTError *error) {
    size_t i;

    if (vocab == NULL || out == NULL) {
        return tt_fail(error, "encode requires a vocabulary and output buffer");
    }
    memset(out, 0, sizeof(*out));
    if (text_len > 0) {
        out->data = (int *)malloc(text_len * sizeof(int));
        if (out->data == NULL) {
            return tt_fail(error, "failed to allocate encoded text");
        }
    }
    out->size = text_len;
    for (i = 0; i < text_len; ++i) {
        int id = vocab->stoi[text[i]];
        if (id >= 0) {
            out->data[i] = id;
        } else if (allow_unknown && vocab->size > 0) {
            out->data[i] = 0;
        } else {
            tt_int_buffer_free(out);
            return tt_fail(error, "text contains a character outside the vocabulary");
        }
    }
    return 1;
}

int tt_vocab_decode(const TTVocabulary *vocab,
                    const int *ids,
                    size_t id_count,
                    unsigned char **out_text,
                    size_t *out_len,
                    TTError *error) {
    unsigned char *text;
    size_t i;

    if (vocab == NULL || out_text == NULL || out_len == NULL) {
        return tt_fail(error, "decode requires vocabulary and output pointers");
    }
    text = (unsigned char *)malloc(id_count + 1);
    if (text == NULL) {
        return tt_fail(error, "failed to allocate decoded text");
    }
    for (i = 0; i < id_count; ++i) {
        if (ids[i] < 0 || (size_t)ids[i] >= vocab->size) {
            free(text);
            return tt_fail(error, "token id is outside the vocabulary");
        }
        text[i] = vocab->chars[ids[i]];
    }
    text[id_count] = '\0';
    *out_text = text;
    *out_len = id_count;
    return 1;
}

void tt_int_buffer_free(TTIntBuffer *buffer) {
    if (buffer == NULL) {
        return;
    }
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}

int tt_read_text_file(const char *path,
                      unsigned char **out_text,
                      size_t *out_len,
                      TTError *error) {
    FILE *file;
    long length;
    unsigned char *data;
    size_t read_count;

    if (path == NULL || out_text == NULL || out_len == NULL) {
        return tt_fail(error, "read_text_file requires path and output pointers");
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return tt_fail(error, "failed to open data file: %s", path);
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return tt_fail(error, "failed to seek data file: %s", path);
    }
    length = ftell(file);
    if (length < 0) {
        fclose(file);
        return tt_fail(error, "failed to measure data file: %s", path);
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return tt_fail(error, "failed to rewind data file: %s", path);
    }
    if (length == 0) {
        fclose(file);
        return tt_fail(error, "data file is empty: %s", path);
    }
    data = (unsigned char *)malloc((size_t)length + 1);
    if (data == NULL) {
        fclose(file);
        return tt_fail(error, "failed to allocate data file buffer");
    }
    read_count = fread(data, 1, (size_t)length, file);
    fclose(file);
    if (read_count != (size_t)length) {
        free(data);
        return tt_fail(error, "failed to read data file: %s", path);
    }
    data[length] = '\0';
    *out_text = data;
    *out_len = (size_t)length;
    return 1;
}

int tt_train_validation_split(const int *data,
                              size_t data_len,
                              double train_fraction,
                              size_t min_sequence,
                              TTTrainValidation *out,
                              TTError *error) {
    size_t cut;
    size_t valid_len;

    if (data == NULL || out == NULL) {
        return tt_fail(error, "train_validation_split requires data and output");
    }
    memset(out, 0, sizeof(*out));
    if (data_len < min_sequence) {
        return tt_fail(error, "not enough encoded data for the requested context length");
    }
    if (train_fraction < 0.5) {
        train_fraction = 0.5;
    }
    if (train_fraction > 0.99) {
        train_fraction = 0.99;
    }
    cut = (size_t)((double)data_len * train_fraction);
    if (cut < min_sequence) {
        cut = min_sequence;
    }
    if (cut > data_len) {
        cut = data_len;
    }

    out->train.data = (int *)malloc(cut * sizeof(int));
    if (out->train.data == NULL) {
        return tt_fail(error, "failed to allocate training split");
    }
    memcpy(out->train.data, data, cut * sizeof(int));
    out->train.size = cut;

    valid_len = data_len - cut;
    if (valid_len < min_sequence) {
        valid_len = out->train.size;
        out->valid.data = (int *)malloc(valid_len * sizeof(int));
        if (out->valid.data == NULL) {
            tt_train_validation_free(out);
            return tt_fail(error, "failed to allocate validation split");
        }
        memcpy(out->valid.data, out->train.data, valid_len * sizeof(int));
    } else {
        out->valid.data = (int *)malloc(valid_len * sizeof(int));
        if (out->valid.data == NULL) {
            tt_train_validation_free(out);
            return tt_fail(error, "failed to allocate validation split");
        }
        memcpy(out->valid.data, data + cut, valid_len * sizeof(int));
    }
    out->valid.size = valid_len;
    return 1;
}

void tt_train_validation_free(TTTrainValidation *split) {
    if (split == NULL) {
        return;
    }
    tt_int_buffer_free(&split->train);
    tt_int_buffer_free(&split->valid);
}

int tt_make_batch(const int *data,
                  size_t data_len,
                  size_t context,
                  size_t batch_size,
                  TTRng *rng,
                  TTBatch *batch,
                  TTError *error) {
    size_t b;
    size_t t;

    if (data == NULL || rng == NULL || batch == NULL) {
        return tt_fail(error, "make_batch requires data, rng, and batch output");
    }
    memset(batch, 0, sizeof(*batch));
    if (data_len <= context + 1) {
        return tt_fail(error, "not enough data to create full-sequence targets");
    }
    batch->x = (int *)malloc(batch_size * context * sizeof(int));
    batch->y = (int *)malloc(batch_size * context * sizeof(int));
    if (batch->x == NULL || batch->y == NULL) {
        tt_batch_free(batch);
        return tt_fail(error, "failed to allocate batch");
    }
    batch->batch_size = batch_size;
    batch->context = context;

    for (b = 0; b < batch_size; ++b) {
        size_t start = tt_rng_bounded(rng, data_len - context);
        for (t = 0; t < context; ++t) {
            batch->x[b * context + t] = data[start + t];
            batch->y[b * context + t] = data[start + t + 1];
        }
    }
    return 1;
}

void tt_batch_free(TTBatch *batch) {
    if (batch == NULL) {
        return;
    }
    free(batch->x);
    free(batch->y);
    memset(batch, 0, sizeof(*batch));
}

int tt_adam_init(TTAdam *adam,
                 TTTinyCharModel *model,
                 double learning_rate,
                 TTError *error) {
    TTNamedParam *named = NULL;
    size_t count;
    size_t i;

    if (adam == NULL || model == NULL) {
        return tt_fail(error, "Adam requires optimizer and model");
    }
    memset(adam, 0, sizeof(*adam));
    if (learning_rate <= 0.0) {
        return tt_fail(error, "Adam learning rate must be positive");
    }

    count = tt_model_parameter_count(model);
    named = (TTNamedParam *)calloc(count, sizeof(*named));
    adam->params = (TTTensor **)calloc(count, sizeof(*adam->params));
    adam->m = (double **)calloc(count, sizeof(*adam->m));
    adam->v = (double **)calloc(count, sizeof(*adam->v));
    if (named == NULL || adam->params == NULL || adam->m == NULL || adam->v == NULL) {
        free(named);
        tt_adam_free(adam);
        return tt_fail(error, "failed to allocate Adam state");
    }
    if (!tt_model_named_parameters(model, named, count, NULL, error)) {
        free(named);
        tt_adam_free(adam);
        return 0;
    }
    adam->count = count;
    adam->learning_rate = learning_rate;
    adam->beta1 = 0.9;
    adam->beta2 = 0.999;
    adam->epsilon = 1e-8;

    for (i = 0; i < count; ++i) {
        adam->params[i] = named[i].tensor;
        adam->m[i] = (double *)calloc(named[i].tensor->size, sizeof(double));
        adam->v[i] = (double *)calloc(named[i].tensor->size, sizeof(double));
        if (adam->m[i] == NULL || adam->v[i] == NULL) {
            free(named);
            tt_adam_free(adam);
            return tt_fail(error, "failed to allocate Adam moment buffers");
        }
    }
    free(named);
    return 1;
}

void tt_adam_free(TTAdam *adam) {
    size_t i;

    if (adam == NULL) {
        return;
    }
    if (adam->m != NULL) {
        for (i = 0; i < adam->count; ++i) {
            free(adam->m[i]);
        }
    }
    if (adam->v != NULL) {
        for (i = 0; i < adam->count; ++i) {
            free(adam->v[i]);
        }
    }
    free(adam->params);
    free(adam->m);
    free(adam->v);
    memset(adam, 0, sizeof(*adam));
}

void tt_adam_zero_grad(TTAdam *adam) {
    size_t i;

    if (adam == NULL) {
        return;
    }
    for (i = 0; i < adam->count; ++i) {
        tt_tensor_zero_grad(adam->params[i]);
    }
}

static double tt_clip_grad_norm(TTAdam *adam, double max_norm) {
    double squared = 0.0;
    double norm;
    size_t p;
    size_t i;

    for (p = 0; p < adam->count; ++p) {
        TTTensor *param = adam->params[p];
        for (i = 0; i < param->size; ++i) {
            squared += param->grad[i] * param->grad[i];
        }
    }
    norm = sqrt(squared);
    if (max_norm > 0.0 && norm > max_norm) {
        double scale = max_norm / (norm + 1e-12);
        for (p = 0; p < adam->count; ++p) {
            TTTensor *param = adam->params[p];
            for (i = 0; i < param->size; ++i) {
                param->grad[i] *= scale;
            }
        }
    }
    return norm;
}

double tt_adam_step(TTAdam *adam, double clip_norm) {
    double unclipped_norm;
    double bias1;
    double bias2;
    size_t p;
    size_t i;

    if (adam == NULL) {
        return 0.0;
    }
    unclipped_norm = tt_clip_grad_norm(adam, clip_norm);
    ++adam->step;
    bias1 = 1.0 - pow(adam->beta1, (double)adam->step);
    bias2 = 1.0 - pow(adam->beta2, (double)adam->step);

    for (p = 0; p < adam->count; ++p) {
        TTTensor *param = adam->params[p];
        for (i = 0; i < param->size; ++i) {
            double g = param->grad[i];
            double m_hat;
            double v_hat;
            adam->m[p][i] = adam->beta1 * adam->m[p][i] + (1.0 - adam->beta1) * g;
            adam->v[p][i] = adam->beta2 * adam->v[p][i] + (1.0 - adam->beta2) * g * g;
            m_hat = adam->m[p][i] / bias1;
            v_hat = adam->v[p][i] / bias2;
            param->data[i] -= adam->learning_rate * m_hat / (sqrt(v_hat) + adam->epsilon);
        }
    }
    return unclipped_norm;
}

int tt_estimate_loss(TTTinyCharModel *model,
                     const int *data,
                     size_t data_len,
                     size_t batch_size,
                     size_t batches,
                     TTRng *rng,
                     double *out_loss,
                     TTError *error) {
    double total = 0.0;
    size_t i;

    if (out_loss == NULL) {
        return tt_fail(error, "estimate_loss requires output pointer");
    }
    if (batches == 0) {
        *out_loss = 0.0;
        return 1;
    }
    for (i = 0; i < batches; ++i) {
        TTBatch batch;
        TTTape tape;
        TTTensor *logits;
        TTTensor *loss;

        memset(&batch, 0, sizeof(batch));
        tt_tape_init(&tape);
        if (!tt_make_batch(data, data_len, model->config.context, batch_size, rng, &batch, error)) {
            return 0;
        }
        logits = tt_model_forward(
            &tape, model, batch.x, batch.batch_size * batch.context, batch.batch_size, error);
        if (logits == NULL) {
            tt_batch_free(&batch);
            tt_tape_free(&tape);
            return 0;
        }
        loss = tt_softmax_cross_entropy(
            &tape, logits, batch.y, batch.batch_size * batch.context, error);
        if (loss == NULL) {
            tt_batch_free(&batch);
            tt_tape_free(&tape);
            return 0;
        }
        total += loss->data[0];
        tt_batch_free(&batch);
        tt_tape_free(&tape);
    }
    *out_loss = total / (double)batches;
    return 1;
}

typedef struct TTLogitCandidate {
    int id;
    double logit;
} TTLogitCandidate;

static int tt_compare_logit_desc(const void *lhs, const void *rhs) {
    const TTLogitCandidate *a = (const TTLogitCandidate *)lhs;
    const TTLogitCandidate *b = (const TTLogitCandidate *)rhs;
    return (b->logit > a->logit) - (b->logit < a->logit);
}

int tt_sample_from_logits(const double *logits,
                          size_t count,
                          double temperature,
                          int top_k,
                          TTRng *rng,
                          int *out_id,
                          TTError *error) {
    TTLogitCandidate *candidates;
    size_t candidate_count;
    size_t i;
    double max_scaled = -DBL_MAX;
    double total = 0.0;
    double threshold;
    double running = 0.0;

    if (logits == NULL || count == 0 || rng == NULL || out_id == NULL) {
        return tt_fail(error, "cannot sample from empty logits");
    }
    if (temperature < 1e-6) {
        temperature = 1e-6;
    }

    candidates = (TTLogitCandidate *)malloc(count * sizeof(*candidates));
    if (candidates == NULL) {
        return tt_fail(error, "failed to allocate sampling candidates");
    }
    for (i = 0; i < count; ++i) {
        candidates[i].id = (int)i;
        candidates[i].logit = logits[i];
    }
    candidate_count = count;
    if (top_k > 0 && (size_t)top_k < candidate_count) {
        qsort(candidates, count, sizeof(*candidates), tt_compare_logit_desc);
        candidate_count = (size_t)top_k;
    }
    for (i = 0; i < candidate_count; ++i) {
        double scaled = candidates[i].logit / temperature;
        if (scaled > max_scaled) {
            max_scaled = scaled;
        }
    }
    for (i = 0; i < candidate_count; ++i) {
        total += exp(candidates[i].logit / temperature - max_scaled);
    }
    threshold = tt_rng_uniform(rng) * total;
    for (i = 0; i < candidate_count; ++i) {
        running += exp(candidates[i].logit / temperature - max_scaled);
        if (threshold <= running) {
            *out_id = candidates[i].id;
            free(candidates);
            return 1;
        }
    }
    *out_id = candidates[candidate_count - 1].id;
    free(candidates);
    return 1;
}

int tt_generate_text(TTTinyCharModel *model,
                     const TTVocabulary *vocab,
                     const unsigned char *prompt,
                     size_t prompt_len,
                     TTGenerationOptions options,
                     TTRng *rng,
                     unsigned char **out_text,
                     size_t *out_len,
                     TTError *error) {
    TTIntBuffer ids;
    int *context = NULL;
    size_t capacity;
    size_t step;

    if (model == NULL || vocab == NULL || rng == NULL || out_text == NULL || out_len == NULL) {
        return tt_fail(error, "generate_text requires model, vocab, rng, and outputs");
    }
    memset(&ids, 0, sizeof(ids));
    if (!tt_vocab_encode(vocab, prompt, prompt_len, 1, &ids, error)) {
        return 0;
    }
    capacity = ids.size + options.tokens + 1;
    if (capacity == 0) {
        capacity = 1;
    }
    ids.data = (int *)realloc(ids.data, capacity * sizeof(int));
    if (ids.data == NULL) {
        ids.size = 0;
        return tt_fail(error, "failed to grow generation id buffer");
    }
    if (ids.size == 0) {
        ids.data[ids.size++] = 0;
    }

    context = (int *)calloc(model->config.context, sizeof(int));
    if (context == NULL) {
        tt_int_buffer_free(&ids);
        return tt_fail(error, "failed to allocate generation context");
    }

    for (step = 0; step < options.tokens; ++step) {
        TTTape tape;
        TTTensor *logits;
        size_t copy_count = ids.size < model->config.context ? ids.size : model->config.context;
        size_t src_start = ids.size - copy_count;
        size_t dst_start = model->config.context - copy_count;
        size_t i;
        int next_id = 0;

        memset(context, 0, model->config.context * sizeof(int));
        for (i = 0; i < copy_count; ++i) {
            context[dst_start + i] = ids.data[src_start + i];
        }

        tt_tape_init(&tape);
        logits = tt_model_forward(&tape, model, context, model->config.context, 1, error);
        if (logits == NULL) {
            tt_tape_free(&tape);
            free(context);
            tt_int_buffer_free(&ids);
            return 0;
        }
        if (!tt_sample_from_logits(logits->data + (model->config.context - 1) * model->config.vocab_size,
                                   model->config.vocab_size,
                                   options.temperature,
                                   options.top_k,
                                   rng,
                                   &next_id,
                                   error)) {
            tt_tape_free(&tape);
            free(context);
            tt_int_buffer_free(&ids);
            return 0;
        }
        tt_tape_free(&tape);
        ids.data[ids.size++] = next_id;
    }

    free(context);
    if (!tt_vocab_decode(vocab, ids.data, ids.size, out_text, out_len, error)) {
        tt_int_buffer_free(&ids);
        return 0;
    }
    tt_int_buffer_free(&ids);
    return 1;
}

static int tt_write_u64(FILE *file, uint64_t value, TTError *error) {
    unsigned char bytes[8];
    size_t i;

    for (i = 0; i < 8; ++i) {
        bytes[i] = (unsigned char)((value >> (8 * i)) & 0xffU);
    }
    if (fwrite(bytes, 1, 8, file) != 8) {
        return tt_fail(error, "failed to write checkpoint");
    }
    return 1;
}

static int tt_read_u64(FILE *file, uint64_t *out, TTError *error) {
    unsigned char bytes[8];
    size_t i;
    uint64_t value = 0;

    if (fread(bytes, 1, 8, file) != 8) {
        return tt_fail(error, "failed to read checkpoint");
    }
    for (i = 0; i < 8; ++i) {
        value |= ((uint64_t)bytes[i]) << (8 * i);
    }
    *out = value;
    return 1;
}

static int tt_write_double(FILE *file, double value, TTError *error) {
    uint64_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return tt_write_u64(file, bits, error);
}

static int tt_read_double(FILE *file, double *out, TTError *error) {
    uint64_t bits;

    if (!tt_read_u64(file, &bits, error)) {
        return 0;
    }
    memcpy(out, &bits, sizeof(*out));
    return 1;
}

static int tt_write_string(FILE *file,
                           const unsigned char *data,
                           size_t len,
                           TTError *error) {
    if (!tt_write_u64(file, (uint64_t)len, error)) {
        return 0;
    }
    if (len > 0 && fwrite(data, 1, len, file) != len) {
        return tt_fail(error, "failed to write checkpoint string");
    }
    return 1;
}

static int tt_read_string(FILE *file,
                          unsigned char **out,
                          size_t *out_len,
                          TTError *error) {
    uint64_t len64;
    unsigned char *data;

    if (!tt_read_u64(file, &len64, error)) {
        return 0;
    }
    if (len64 > (uint64_t)((size_t)-1) - 1) {
        return tt_fail(error, "checkpoint string is too large");
    }
    data = (unsigned char *)malloc((size_t)len64 + 1);
    if (data == NULL) {
        return tt_fail(error, "failed to allocate checkpoint string");
    }
    if (len64 > 0 && fread(data, 1, (size_t)len64, file) != (size_t)len64) {
        free(data);
        return tt_fail(error, "failed to read checkpoint string");
    }
    data[len64] = '\0';
    *out = data;
    *out_len = (size_t)len64;
    return 1;
}

static int tt_write_tensor(FILE *file,
                           const TTNamedParam *param,
                           TTError *error) {
    size_t i;

    if (!tt_write_string(file,
                         (const unsigned char *)param->name,
                         strlen(param->name),
                         error) ||
        !tt_write_u64(file, (uint64_t)param->tensor->rank, error)) {
        return 0;
    }
    for (i = 0; i < param->tensor->rank; ++i) {
        if (!tt_write_u64(file, (uint64_t)param->tensor->dims[i], error)) {
            return 0;
        }
    }
    if (!tt_write_u64(file, (uint64_t)param->tensor->size, error)) {
        return 0;
    }
    for (i = 0; i < param->tensor->size; ++i) {
        if (!tt_write_double(file, param->tensor->data[i], error)) {
            return 0;
        }
    }
    return 1;
}

static int tt_read_tensor_into(FILE *file,
                               const TTNamedParam *param,
                               TTError *error) {
    unsigned char *name = NULL;
    size_t name_len = 0;
    uint64_t rank64;
    uint64_t size64;
    size_t i;

    if (!tt_read_string(file, &name, &name_len, error)) {
        return 0;
    }
    (void)name_len;
    if (strcmp((const char *)name, param->name) != 0) {
        tt_fail(error,
                "checkpoint tensor order mismatch: expected %s, got %s",
                param->name,
                (const char *)name);
        free(name);
        return 0;
    }
    free(name);

    if (!tt_read_u64(file, &rank64, error)) {
        return 0;
    }
    if ((size_t)rank64 != param->tensor->rank) {
        return tt_fail(error, "checkpoint tensor rank mismatch for %s", param->name);
    }
    for (i = 0; i < param->tensor->rank; ++i) {
        uint64_t dim64;
        if (!tt_read_u64(file, &dim64, error)) {
            return 0;
        }
        if ((size_t)dim64 != param->tensor->dims[i]) {
            return tt_fail(error, "checkpoint tensor shape mismatch for %s", param->name);
        }
    }
    if (!tt_read_u64(file, &size64, error)) {
        return 0;
    }
    if ((size_t)size64 != param->tensor->size) {
        return tt_fail(error, "checkpoint tensor size mismatch for %s", param->name);
    }
    for (i = 0; i < param->tensor->size; ++i) {
        if (!tt_read_double(file, &param->tensor->data[i], error)) {
            return 0;
        }
    }
    return 1;
}

int tt_save_checkpoint(const char *path,
                       TTTinyCharModel *model,
                       const TTVocabulary *vocab,
                       TTError *error) {
    FILE *file;
    TTNamedParam *params = NULL;
    size_t count;
    size_t i;
    int ok = 0;

    if (path == NULL || model == NULL || vocab == NULL) {
        return tt_fail(error, "save_checkpoint requires path, model, and vocabulary");
    }
    file = fopen(path, "wb");
    if (file == NULL) {
        return tt_fail(error, "failed to open checkpoint for writing: %s", path);
    }
    count = tt_model_parameter_count(model);
    params = (TTNamedParam *)calloc(count, sizeof(*params));
    if (params == NULL) {
        tt_fail(error, "failed to allocate checkpoint parameter list");
        goto done;
    }
    if (!tt_model_named_parameters(model, params, count, NULL, error)) {
        goto done;
    }

    if (!tt_write_string(file,
                         (const unsigned char *)TT_CHECKPOINT_MAGIC,
                         strlen(TT_CHECKPOINT_MAGIC),
                         error) ||
        !tt_write_u64(file, (uint64_t)model->config.vocab_size, error) ||
        !tt_write_u64(file, (uint64_t)model->config.context, error) ||
        !tt_write_u64(file, (uint64_t)model->config.embed, error) ||
        !tt_write_u64(file, (uint64_t)model->config.hidden, error) ||
        !tt_write_u64(file, (uint64_t)model->config.layers, error) ||
        !tt_write_u64(file, (uint64_t)model->config.heads, error) ||
        !tt_write_string(file, vocab->chars, vocab->size, error) ||
        !tt_write_u64(file, (uint64_t)count, error)) {
        goto done;
    }

    for (i = 0; i < count; ++i) {
        if (!tt_write_tensor(file, &params[i], error)) {
            goto done;
        }
    }
    ok = 1;

done:
    free(params);
    if (fclose(file) != 0 && ok) {
        return tt_fail(error, "failed to close checkpoint: %s", path);
    }
    return ok;
}

int tt_load_checkpoint(const char *path,
                       TTTinyCharModel *model,
                       TTVocabulary *vocab,
                       TTError *error) {
    FILE *file;
    unsigned char *magic = NULL;
    unsigned char *chars = NULL;
    size_t magic_len = 0;
    size_t chars_len = 0;
    uint64_t value;
    uint64_t param_count64;
    TTModelConfig config;
    TTNamedParam *params = NULL;
    size_t count;
    size_t i;
    int ok = 0;

    if (path == NULL || model == NULL || vocab == NULL) {
        return tt_fail(error, "load_checkpoint requires path, model, and vocabulary");
    }
    memset(model, 0, sizeof(*model));
    memset(vocab, 0, sizeof(*vocab));
    file = fopen(path, "rb");
    if (file == NULL) {
        return tt_fail(error, "failed to open checkpoint: %s", path);
    }

    if (!tt_read_string(file, &magic, &magic_len, error)) {
        goto done;
    }
    (void)magic_len;
    if (strcmp((const char *)magic, TT_CHECKPOINT_MAGIC) != 0) {
        tt_fail(error, "not a Tater Tot Transformer checkpoint: %s", path);
        goto done;
    }

    memset(&config, 0, sizeof(config));
    if (!tt_read_u64(file, &value, error)) {
        goto done;
    }
    config.vocab_size = (size_t)value;
    if (!tt_read_u64(file, &value, error)) {
        goto done;
    }
    config.context = (size_t)value;
    if (!tt_read_u64(file, &value, error)) {
        goto done;
    }
    config.embed = (size_t)value;
    if (!tt_read_u64(file, &value, error)) {
        goto done;
    }
    config.hidden = (size_t)value;
    if (!tt_read_u64(file, &value, error)) {
        goto done;
    }
    config.layers = (size_t)value;
    if (!tt_read_u64(file, &value, error)) {
        goto done;
    }
    config.heads = (size_t)value;

    if (!tt_read_string(file, &chars, &chars_len, error)) {
        goto done;
    }
    if (chars_len != config.vocab_size) {
        tt_fail(error, "checkpoint vocabulary size does not match model config");
        goto done;
    }
    if (!tt_vocab_from_chars(vocab, chars, chars_len, error)) {
        goto done;
    }
    if (!tt_model_init(model, config, 1, error)) {
        goto done;
    }

    count = tt_model_parameter_count(model);
    params = (TTNamedParam *)calloc(count, sizeof(*params));
    if (params == NULL) {
        tt_fail(error, "failed to allocate checkpoint parameter list");
        goto done;
    }
    if (!tt_model_named_parameters(model, params, count, NULL, error)) {
        goto done;
    }
    if (!tt_read_u64(file, &param_count64, error)) {
        goto done;
    }
    if ((size_t)param_count64 != count) {
        tt_fail(error, "checkpoint parameter count does not match model");
        goto done;
    }
    for (i = 0; i < count; ++i) {
        if (!tt_read_tensor_into(file, &params[i], error)) {
            goto done;
        }
    }
    ok = 1;

done:
    free(params);
    free(magic);
    free(chars);
    fclose(file);
    if (!ok) {
        tt_model_free(model);
        tt_vocab_free(vocab);
    }
    return ok;
}
