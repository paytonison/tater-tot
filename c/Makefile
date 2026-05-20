CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -O2
LDFLAGS ?= -lm

.PHONY: all clean test

all: tater_train_c tater_sample_c tater_tests_c

tater_train_c: train.c tater_tot.c tater_tot.h
	$(CC) $(CFLAGS) train.c tater_tot.c $(LDFLAGS) -o $@

tater_sample_c: sample.c tater_tot.c tater_tot.h
	$(CC) $(CFLAGS) sample.c tater_tot.c $(LDFLAGS) -o $@

tater_tests_c: test_c.c tater_tot.c tater_tot.h
	$(CC) $(CFLAGS) test_c.c tater_tot.c $(LDFLAGS) -o $@

test: tater_tests_c
	./tater_tests_c

clean:
	rm -f tater_train_c tater_sample_c tater_tests_c
