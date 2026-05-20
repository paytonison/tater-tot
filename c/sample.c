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
            "Usage: tater_sample_c --checkpoint ../checkpoints/model.bin "
            "--prompt \"Once upon a time\" --tokens 300 "
            "[--temperature 1.0] [--top-k 0]\n");
}

int main(int argc, char **argv) {
    TTError error;
    TTTinyCharModel model;
    TTVocabulary vocab;
    TTRng rng;
    TTGenerationOptions options;
    unsigned char *generated = NULL;
    size_t generated_len = 0;
    int ok = 0;

    const char *checkpoint_path =
        option_value(argc, argv, "--checkpoint", "../checkpoints/model.bin");
    const char *prompt = option_value(argc, argv, "--prompt", "");
    uint32_t seed = (uint32_t)size_option(argc, argv, "--seed", 1337);

    tt_error_clear(&error);
    memset(&model, 0, sizeof(model));
    memset(&vocab, 0, sizeof(vocab));

    if (has_flag(argc, argv, "--help")) {
        print_usage();
        return 0;
    }

    options.tokens = size_option(argc, argv, "--tokens", 300);
    options.temperature = double_option(argc, argv, "--temperature", 1.0);
    options.top_k = (int)size_option(argc, argv, "--top-k", 0);

    if (!tt_load_checkpoint(checkpoint_path, &model, &vocab, &error)) {
        goto done;
    }
    tt_rng_seed(&rng, seed);
    if (!tt_generate_text(&model,
                          &vocab,
                          (const unsigned char *)prompt,
                          strlen(prompt),
                          options,
                          &rng,
                          &generated,
                          &generated_len,
                          &error)) {
        goto done;
    }
    fwrite(generated, 1, generated_len, stdout);
    fputc('\n', stdout);
    ok = 1;

done:
    if (!ok && error.message[0] != '\0') {
        fprintf(stderr, "tater_sample_c error: %s\n", error.message);
    }
    free(generated);
    tt_vocab_free(&vocab);
    tt_model_free(&model);
    return ok ? 0 : 1;
}
