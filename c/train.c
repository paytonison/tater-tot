#include "tater_tot.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *option_value(int argc, char **argv, const char *name, const char *fallback) {
    int i;

    for (i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return fallback;
}

static int has_flag(int argc, char **argv, const char *name) {
    int i;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return 1;
        }
    }
    return 0;
}

static size_t size_option(int argc, char **argv, const char *name, size_t fallback) {
    const char *value = option_value(argc, argv, name, NULL);

    return value == NULL ? fallback : (size_t)strtoull(value, NULL, 10);
}

static double double_option(int argc, char **argv, const char *name, double fallback) {
    const char *value = option_value(argc, argv, name, NULL);

    return value == NULL ? fallback : strtod(value, NULL);
}

static void print_usage(void) {
    fprintf(stderr,
            "Usage: tater_train_c --data ../data/input.txt --steps 1000 "
            "--context 64 --embed 64 [--layers 2] [--heads 4] [--hidden 256] "
            "[--batch 16] [--lr 0.003] [--checkpoint ../checkpoints/model.bin]\n");
}

int main(int argc, char **argv) {
    TTError error;
    unsigned char *text = NULL;
    size_t text_len = 0;
    TTVocabulary vocab;
    TTIntBuffer encoded;
    TTTrainValidation split;
    TTTinyCharModel model;
    TTAdam optimizer;
    TTRng rng;
    int ok = 0;

    const char *data_path = option_value(argc, argv, "--data", "../data/input.txt");
    size_t steps = size_option(argc, argv, "--steps", 1000);
    size_t context = size_option(argc, argv, "--context", 64);
    size_t embed = size_option(argc, argv, "--embed", 64);
    size_t layers = size_option(argc, argv, "--layers", 2);
    size_t heads = size_option(argc, argv, "--heads", 4);
    size_t hidden = size_option(argc, argv, "--hidden", embed * 4);
    size_t batch_size = size_option(argc, argv, "--batch", 16);
    size_t print_every = size_option(argc, argv, "--print-every", 50);
    size_t sample_every = size_option(argc, argv, "--sample-every", 200);
    size_t eval_batches = size_option(argc, argv, "--eval-batches", 4);
    double learning_rate = double_option(argc, argv, "--lr", 0.003);
    double clip_norm = double_option(argc, argv, "--clip", 1.0);
    uint32_t seed = (uint32_t)size_option(argc, argv, "--seed", 1337);
    const char *checkpoint_path =
        option_value(argc, argv, "--checkpoint", "../checkpoints/model.bin");

    tt_error_clear(&error);
    memset(&vocab, 0, sizeof(vocab));
    memset(&encoded, 0, sizeof(encoded));
    memset(&split, 0, sizeof(split));
    memset(&model, 0, sizeof(model));
    memset(&optimizer, 0, sizeof(optimizer));

    if (has_flag(argc, argv, "--help")) {
        print_usage();
        return 0;
    }

    if (!tt_read_text_file(data_path, &text, &text_len, &error) ||
        !tt_vocab_from_text(&vocab, text, text_len, &error) ||
        !tt_vocab_encode(&vocab, text, text_len, 0, &encoded, &error) ||
        !tt_train_validation_split(
            encoded.data, encoded.size, 0.9, context + 2, &split, &error)) {
        goto done;
    }

    {
        TTModelConfig config;
        memset(&config, 0, sizeof(config));
        config.vocab_size = vocab.size;
        config.context = context;
        config.layers = layers;
        config.heads = heads;
        config.embed = embed;
        config.hidden = hidden;

        if (!tt_model_init(&model, config, seed, &error) ||
            !tt_adam_init(&optimizer, &model, learning_rate, &error)) {
            goto done;
        }
    }

    tt_rng_seed(&rng, seed);

    printf("Tater Tot training\n");
    printf("  data: %s\n", data_path);
    printf("  chars: %zu\n", text_len);
    printf("  vocab: %zu\n", vocab.size);
    printf("  context/layers/heads/embed/hidden: %zu/%zu/%zu/%zu/%zu\n",
           context,
           layers,
           heads,
           embed,
           hidden);
    printf("  batch/steps/lr: %zu/%zu/%g\n", batch_size, steps, learning_rate);

    for (size_t step = 1; step <= steps; ++step) {
        TTBatch batch;
        TTTape tape;
        TTTensor *logits;
        TTTensor *loss;
        double grad_norm;

        memset(&batch, 0, sizeof(batch));
        tt_tape_init(&tape);

        if (!tt_make_batch(
                split.train.data, split.train.size, context, batch_size, &rng, &batch, &error)) {
            tt_tape_free(&tape);
            goto done;
        }
        logits = tt_model_forward(
            &tape, &model, batch.x, batch.batch_size * batch.context, batch.batch_size, &error);
        if (logits == NULL) {
            tt_batch_free(&batch);
            tt_tape_free(&tape);
            goto done;
        }
        loss =
            tt_softmax_cross_entropy(&tape, logits, batch.y, batch.batch_size * batch.context, &error);
        if (loss == NULL) {
            tt_batch_free(&batch);
            tt_tape_free(&tape);
            goto done;
        }

        tt_adam_zero_grad(&optimizer);
        if (!tt_tape_backward(&tape, loss, &error)) {
            tt_batch_free(&batch);
            tt_tape_free(&tape);
            goto done;
        }
        grad_norm = tt_adam_step(&optimizer, clip_norm);

        if (print_every != 0 && (step == 1 || step % print_every == 0 || step == steps)) {
            double val_loss = 0.0;
            if (!tt_estimate_loss(
                    &model, split.valid.data, split.valid.size, batch_size, eval_batches, &rng, &val_loss, &error)) {
                tt_batch_free(&batch);
                tt_tape_free(&tape);
                goto done;
            }
            printf("step %zu train_loss=%.12g val_loss=%.12g grad_norm=%.12g\n",
                   step,
                   loss->data[0],
                   val_loss,
                   grad_norm);
        }

        if (sample_every != 0 && (step % sample_every == 0 || step == steps)) {
            TTGenerationOptions options;
            unsigned char *sample = NULL;
            size_t sample_len = 0;
            size_t prompt_len = text_len < 32 ? text_len : 32;

            options.tokens = 160;
            options.temperature = 0.8;
            options.top_k = 10;
            if (!tt_generate_text(
                    &model, &vocab, text, prompt_len, options, &rng, &sample, &sample_len, &error)) {
                tt_batch_free(&batch);
                tt_tape_free(&tape);
                goto done;
            }
            printf("--- sample ---\n");
            fwrite(sample, 1, sample_len, stdout);
            printf("\n--------------\n");
            free(sample);
        }

        tt_batch_free(&batch);
        tt_tape_free(&tape);
    }

    if (!tt_save_checkpoint(checkpoint_path, &model, &vocab, &error)) {
        goto done;
    }
    printf("saved checkpoint: %s\n", checkpoint_path);
    ok = 1;

done:
    if (!ok && error.message[0] != '\0') {
        fprintf(stderr, "tater_train_c error: %s\n", error.message);
    }
    tt_adam_free(&optimizer);
    tt_model_free(&model);
    tt_train_validation_free(&split);
    tt_int_buffer_free(&encoded);
    tt_vocab_free(&vocab);
    free(text);
    return ok ? 0 : 1;
}
