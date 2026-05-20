#ifndef TATER_TOT_H
#define TATER_TOT_H

#include <stddef.h>
#include <stdint.h>

/*
 * Tater Tot's C port keeps the model machinery deliberately visible.
 *
 * Ownership rule summary:
 * - TTTinyCharModel owns parameter tensors and frees them with tt_model_free.
 * - TTTape owns temporary tensors created during one forward pass and frees
 *   them with tt_tape_free.
 * - TTVocabulary owns chars and frees them with tt_vocab_free.
 * - TTAdam owns optimizer state buffers and frees them with tt_adam_free.
 * - TTBatch and TTIntBuffer own integer arrays and have matching free helpers.
 */

#define TT_ERROR_SIZE 512
#define TT_NAME_SIZE 64
#define TT_CHECKPOINT_MAGIC "TATER_TOT_CHECKPOINT_V2_TRANSFORMER"

typedef struct TTError {
    char message[TT_ERROR_SIZE];
} TTError;

typedef struct TTRng {
    uint64_t state;
    int has_spare_normal;
    double spare_normal;
} TTRng;

typedef enum TTOp {
    TT_OP_NONE = 0,
    TT_OP_ADD,
    TT_OP_MATMUL,
    TT_OP_GELU,
    TT_OP_EMBEDDING_LOOKUP,
    TT_OP_LAYER_NORM,
    TT_OP_CAUSAL_ATTENTION_CORE,
    TT_OP_SOFTMAX_CROSS_ENTROPY
} TTOp;

typedef struct TTTensor TTTensor;
typedef void (*TTBackwardFn)(TTTensor *self);

struct TTTensor {
    /*
     * rank/dims make every tensor shape visible. The current Transformer uses:
     * - rank 0 scalar losses
     * - rank 1 biases, LayerNorm gamma/beta, and target-independent vectors
     * - rank 2 matrices for activations and parameters
     */
    size_t rank;
    size_t dims[2];
    size_t size;

    /* This tensor owns data/grad when owns_buffers is non-zero. */
    double *data;
    double *grad;
    int requires_grad;
    int owns_buffers;

    /*
     * Autograd metadata for temporary tensors. parents are non-owning pointers
     * to either model parameters or earlier tape tensors. ctx is op-specific
     * memory owned by this tensor and released by tt_tape_free.
     */
    TTOp op;
    TTTensor *parents[4];
    size_t parent_count;
    void *ctx;
    TTBackwardFn backward;
};

typedef struct TTTape {
    TTTensor **items;
    size_t count;
    size_t capacity;
} TTTape;

typedef struct TTModelConfig {
    size_t vocab_size;
    size_t context;
    size_t layers;
    size_t heads;
    size_t embed;
    size_t hidden;
} TTModelConfig;

typedef struct TTGenerationOptions {
    size_t tokens;
    double temperature;
    int top_k;
} TTGenerationOptions;

typedef struct TTTransformerBlock {
    TTTensor ln1_gamma;
    TTTensor ln1_beta;
    TTTensor q_w;
    TTTensor q_b;
    TTTensor k_w;
    TTTensor k_b;
    TTTensor v_w;
    TTTensor v_b;
    TTTensor out_w;
    TTTensor out_b;
    TTTensor ln2_gamma;
    TTTensor ln2_beta;
    TTTensor ff1_w;
    TTTensor ff1_b;
    TTTensor ff2_w;
    TTTensor ff2_b;
} TTTransformerBlock;

typedef struct TTTinyCharModel {
    TTModelConfig config;

    /* token_embedding: [vocab_size, embed], learned character vectors. */
    TTTensor token_embedding;

    /* positional_embedding: [context, embed], learned position vectors. */
    TTTensor positional_embedding;

    /* blocks owns config.layers TransformerBlock values. */
    TTTransformerBlock *blocks;

    /* final LayerNorm and language-model projection parameters. */
    TTTensor ln_f_gamma;
    TTTensor ln_f_beta;
    TTTensor lm_head_w;
    TTTensor lm_head_b;
} TTTinyCharModel;

typedef struct TTNamedParam {
    char name[TT_NAME_SIZE];
    TTTensor *tensor;
} TTNamedParam;

typedef struct TTVocabulary {
    unsigned char *chars;
    size_t size;
    int stoi[256];
} TTVocabulary;

typedef struct TTIntBuffer {
    int *data;
    size_t size;
} TTIntBuffer;

typedef struct TTTrainValidation {
    TTIntBuffer train;
    TTIntBuffer valid;
} TTTrainValidation;

typedef struct TTBatch {
    int *x;
    int *y;
    size_t batch_size;
    size_t context;
} TTBatch;

typedef struct TTAdam {
    TTTensor **params;
    double **m;
    double **v;
    size_t count;
    size_t step;
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
} TTAdam;

void tt_error_clear(TTError *error);

void tt_rng_seed(TTRng *rng, uint64_t seed);
uint32_t tt_rng_u32(TTRng *rng);
double tt_rng_uniform(TTRng *rng);
double tt_rng_normal(TTRng *rng);

void tt_tape_init(TTTape *tape);
void tt_tape_free(TTTape *tape);
int tt_tape_backward(TTTape *tape, TTTensor *loss, TTError *error);

void tt_tensor_free(TTTensor *tensor);
void tt_tensor_zero_grad(TTTensor *tensor);
int tt_tensor_init(TTTensor *tensor,
                   size_t rank,
                   const size_t *dims,
                   double value,
                   int requires_grad,
                   TTError *error);

int tt_model_init(TTTinyCharModel *model,
                  TTModelConfig config,
                  uint32_t seed,
                  TTError *error);
void tt_model_free(TTTinyCharModel *model);
size_t tt_model_parameter_count(const TTTinyCharModel *model);
int tt_model_named_parameters(TTTinyCharModel *model,
                              TTNamedParam *params,
                              size_t capacity,
                              size_t *out_count,
                              TTError *error);

TTTensor *tt_model_forward(TTTape *tape,
                           TTTinyCharModel *model,
                           const int *tokens,
                           size_t token_count,
                           size_t batch_size,
                           TTError *error);
TTTensor *tt_softmax_cross_entropy(TTTape *tape,
                                   TTTensor *logits,
                                   const int *targets,
                                   size_t target_count,
                                   TTError *error);

int tt_vocab_from_text(TTVocabulary *vocab,
                       const unsigned char *text,
                       size_t text_len,
                       TTError *error);
void tt_vocab_free(TTVocabulary *vocab);
int tt_vocab_encode(const TTVocabulary *vocab,
                    const unsigned char *text,
                    size_t text_len,
                    int allow_unknown,
                    TTIntBuffer *out,
                    TTError *error);
int tt_vocab_decode(const TTVocabulary *vocab,
                    const int *ids,
                    size_t id_count,
                    unsigned char **out_text,
                    size_t *out_len,
                    TTError *error);

int tt_read_text_file(const char *path,
                      unsigned char **out_text,
                      size_t *out_len,
                      TTError *error);
int tt_train_validation_split(const int *data,
                              size_t data_len,
                              double train_fraction,
                              size_t min_sequence,
                              TTTrainValidation *out,
                              TTError *error);
void tt_int_buffer_free(TTIntBuffer *buffer);
void tt_train_validation_free(TTTrainValidation *split);
int tt_make_batch(const int *data,
                  size_t data_len,
                  size_t context,
                  size_t batch_size,
                  TTRng *rng,
                  TTBatch *batch,
                  TTError *error);
void tt_batch_free(TTBatch *batch);

int tt_adam_init(TTAdam *adam,
                 TTTinyCharModel *model,
                 double learning_rate,
                 TTError *error);
void tt_adam_free(TTAdam *adam);
void tt_adam_zero_grad(TTAdam *adam);
double tt_adam_step(TTAdam *adam, double clip_norm);

int tt_estimate_loss(TTTinyCharModel *model,
                     const int *data,
                     size_t data_len,
                     size_t batch_size,
                     size_t batches,
                     TTRng *rng,
                     double *out_loss,
                     TTError *error);

int tt_sample_from_logits(const double *logits,
                          size_t count,
                          double temperature,
                          int top_k,
                          TTRng *rng,
                          int *out_id,
                          TTError *error);
int tt_generate_text(TTTinyCharModel *model,
                     const TTVocabulary *vocab,
                     const unsigned char *prompt,
                     size_t prompt_len,
                     TTGenerationOptions options,
                     TTRng *rng,
                     unsigned char **out_text,
                     size_t *out_len,
                     TTError *error);

int tt_save_checkpoint(const char *path,
                       TTTinyCharModel *model,
                       const TTVocabulary *vocab,
                       TTError *error);
int tt_load_checkpoint(const char *path,
                       TTTinyCharModel *model,
                       TTVocabulary *vocab,
                       TTError *error);

#endif
