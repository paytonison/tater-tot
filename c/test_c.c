#include "tater_tot.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void expect_true(int condition, const char *message) {
    if (!condition) {
        ++failures;
        fprintf(stderr, "FAIL: %s\n", message);
    }
}

static int any_nonzero(const double *values, size_t count) {
    size_t i;

    for (i = 0; i < count; ++i) {
        if (fabs(values[i]) > 1e-12) {
            return 1;
        }
    }
    return 0;
}

static void test_transformer_shape_loss_and_gradients(void) {
    TTError error;
    TTModelConfig config;
    TTTinyCharModel model;
    TTTape tape;
    TTTensor *logits;
    TTTensor *loss;
    int tokens[8] = {0, 1, 2, 3, 1, 2, 3, 4};
    int targets[8] = {1, 2, 3, 4, 2, 3, 4, 0};

    tt_error_clear(&error);
    memset(&config, 0, sizeof(config));
    memset(&model, 0, sizeof(model));
    tt_tape_init(&tape);

    config.vocab_size = 5;
    config.context = 4;
    config.layers = 1;
    config.heads = 2;
    config.embed = 8;
    config.hidden = 16;

    expect_true(tt_model_init(&model, config, 11, &error), error.message);
    logits = tt_model_forward(&tape, &model, tokens, 8, 2, &error);
    expect_true(logits != NULL, error.message);
    if (logits != NULL) {
        expect_true(logits->rank == 2, "logits rank");
        expect_true(logits->dims[0] == 8, "logits rows");
        expect_true(logits->dims[1] == config.vocab_size, "logits classes");
        loss = tt_softmax_cross_entropy(&tape, logits, targets, 8, &error);
        expect_true(loss != NULL, error.message);
        if (loss != NULL) {
            expect_true(isfinite(loss->data[0]), "finite loss");
            expect_true(tt_tape_backward(&tape, loss, &error), error.message);
            expect_true(any_nonzero(model.token_embedding.grad, model.token_embedding.size),
                        "token embedding receives gradient");
        }
    }

    tt_tape_free(&tape);
    tt_model_free(&model);
}

static void test_checkpoint_and_generation(void) {
    TTError error;
    const unsigned char text[] = "abababababababab";
    TTVocabulary vocab;
    TTTinyCharModel model;
    TTTinyCharModel loaded_model;
    TTVocabulary loaded_vocab;
    TTModelConfig config;
    TTRng rng;
    TTGenerationOptions options;
    unsigned char *generated = NULL;
    size_t generated_len = 0;
    const char *path = "/tmp/tater_tot_c_checkpoint_test.bin";

    tt_error_clear(&error);
    memset(&vocab, 0, sizeof(vocab));
    memset(&model, 0, sizeof(model));
    memset(&loaded_model, 0, sizeof(loaded_model));
    memset(&loaded_vocab, 0, sizeof(loaded_vocab));
    memset(&config, 0, sizeof(config));

    expect_true(tt_vocab_from_text(&vocab, text, strlen((const char *)text), &error),
                error.message);
    config.vocab_size = vocab.size;
    config.context = 4;
    config.layers = 1;
    config.heads = 2;
    config.embed = 8;
    config.hidden = 16;
    expect_true(tt_model_init(&model, config, 17, &error), error.message);
    expect_true(tt_save_checkpoint(path, &model, &vocab, &error), error.message);
    expect_true(tt_load_checkpoint(path, &loaded_model, &loaded_vocab, &error), error.message);
    remove(path);

    expect_true(loaded_model.config.context == config.context, "checkpoint context");
    expect_true(loaded_vocab.size == vocab.size, "checkpoint vocab size");
    expect_true(fabs(loaded_model.blocks[0].q_w.data[3] - model.blocks[0].q_w.data[3]) < 1e-12,
                "checkpoint parameter round trip");

    options.tokens = 8;
    options.temperature = 1.0;
    options.top_k = 0;
    tt_rng_seed(&rng, 19);
    expect_true(tt_generate_text(&loaded_model,
                                 &loaded_vocab,
                                 (const unsigned char *)"ab",
                                 2,
                                 options,
                                 &rng,
                                 &generated,
                                 &generated_len,
                                 &error),
                error.message);
    expect_true(generated_len == 10, "generation returns prompt plus requested tokens");

    free(generated);
    tt_vocab_free(&loaded_vocab);
    tt_model_free(&loaded_model);
    tt_model_free(&model);
    tt_vocab_free(&vocab);
}

static void test_tiny_loss_decreases(void) {
    TTError error;
    unsigned char text[240];
    TTVocabulary vocab;
    TTIntBuffer encoded;
    TTTinyCharModel model;
    TTAdam adam;
    TTModelConfig config;
    TTRng rng;
    TTRng eval_rng;
    double initial = 0.0;
    double final = 0.0;
    size_t i;

    tt_error_clear(&error);
    memset(&vocab, 0, sizeof(vocab));
    memset(&encoded, 0, sizeof(encoded));
    memset(&model, 0, sizeof(model));
    memset(&adam, 0, sizeof(adam));
    memset(&config, 0, sizeof(config));

    for (i = 0; i < sizeof(text); i += 2) {
        text[i] = 'a';
        text[i + 1] = 'b';
    }

    expect_true(tt_vocab_from_text(&vocab, text, sizeof(text), &error), error.message);
    expect_true(tt_vocab_encode(&vocab, text, sizeof(text), 0, &encoded, &error), error.message);

    config.vocab_size = vocab.size;
    config.context = 4;
    config.layers = 1;
    config.heads = 2;
    config.embed = 8;
    config.hidden = 16;
    expect_true(tt_model_init(&model, config, 7, &error), error.message);
    expect_true(tt_adam_init(&adam, &model, 0.02, &error), error.message);

    tt_rng_seed(&eval_rng, 123);
    expect_true(tt_estimate_loss(
                    &model, encoded.data, encoded.size, 8, 4, &eval_rng, &initial, &error),
                error.message);
    tt_rng_seed(&rng, 99);
    for (i = 0; i < 100; ++i) {
        TTBatch batch;
        TTTape tape;
        TTTensor *logits;
        TTTensor *loss;

        memset(&batch, 0, sizeof(batch));
        tt_tape_init(&tape);
        expect_true(tt_make_batch(
                        encoded.data, encoded.size, config.context, 8, &rng, &batch, &error),
                    error.message);
        logits = tt_model_forward(&tape, &model, batch.x, batch.batch_size * batch.context, 8, &error);
        expect_true(logits != NULL, error.message);
        loss = logits == NULL
                   ? NULL
                   : tt_softmax_cross_entropy(
                         &tape, logits, batch.y, batch.batch_size * batch.context, &error);
        expect_true(loss != NULL, error.message);
        if (loss != NULL) {
            tt_adam_zero_grad(&adam);
            expect_true(tt_tape_backward(&tape, loss, &error), error.message);
            (void)tt_adam_step(&adam, 1.0);
        }
        tt_batch_free(&batch);
        tt_tape_free(&tape);
        if (failures != 0) {
            break;
        }
    }
    tt_rng_seed(&eval_rng, 123);
    expect_true(tt_estimate_loss(
                    &model, encoded.data, encoded.size, 8, 4, &eval_rng, &final, &error),
                error.message);
    expect_true(final < initial, "tiny repeated text training loss decreases");

    tt_adam_free(&adam);
    tt_model_free(&model);
    tt_int_buffer_free(&encoded);
    tt_vocab_free(&vocab);
}

int main(void) {
    test_transformer_shape_loss_and_gradients();
    test_checkpoint_and_generation();
    test_tiny_loss_decreases();

    if (failures != 0) {
        fprintf(stderr, "%d test failure(s)\n", failures);
        return 1;
    }
    printf("all C tests passed\n");
    return 0;
}
