# ears — whisper on notorch. Pure C, no Python anywhere, not even in the tests.
#
#   make                build ./ears
#   make test           the four parity gates, in order (mel, encoder, transcript, speed)
#   make test-mel       one gate at a time
#   make oracle         build harness/oracle_dump (needs the whisper.cpp checkout)
#   make clean
#
# Weights are not in this repo; see README.md.

CC       ?= cc
CFLAGS   ?= -O2 -Wall -Werror -std=c11 -D_POSIX_C_SOURCE=200809L
LDFLAGS  ?= -lnotorch -lm -lpthread

# OpenBLAS through pkg-config: on this phone the headers sit under
# /usr/include/aarch64-linux-gnu/openblas-pthread, which no default include path
# reaches. libnotorch.a itself is built against cblas and pulls the symbols in.
BLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
BLAS_LIBS   := $(shell pkg-config --libs   openblas 2>/dev/null)
ifeq ($(strip $(BLAS_LIBS)),)
  BLAS_LIBS := -lopenblas
endif

CFLAGS  += $(BLAS_CFLAGS)
LDFLAGS += $(BLAS_LIBS)

SRC = ears.c encoder.c decoder.c tokenizer.c mel.c ggml_bin.c nnops.c
HDR = ears.h nnops.h mel.h ggml_bin.h

# where the oracle lives; override on the command line if the checkout moved
WHISPER ?= $(HOME)/arianna/whisper.cpp
REF     ?= $(HOME)/arianna/ears-reference

all: ears

ears: $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

tests/cmp_f32: tests/cmp_f32.c
	$(CC) $(CFLAGS) -o $@ $< -lm

tests/dump_mel: tests/dump_mel.c mel.c ggml_bin.c $(HDR)
	$(CC) $(CFLAGS) -o $@ tests/dump_mel.c mel.c ggml_bin.c $(LDFLAGS)

tests/dump_enc: tests/dump_enc.c encoder.c decoder.c tokenizer.c mel.c ggml_bin.c nnops.c $(HDR)
	$(CC) $(CFLAGS) -o $@ tests/dump_enc.c encoder.c decoder.c tokenizer.c mel.c ggml_bin.c nnops.c $(LDFLAGS)

oracle: harness/oracle_dump

harness/oracle_dump: harness/oracle_dump.cpp
	g++ -O1 -std=c++17 -DWHISPER_VERSION='"1.9.4"' \
	    -I$(WHISPER)/src -I$(WHISPER)/include -I$(WHISPER)/ggml/include -I$(WHISPER)/ggml/src \
	    -o $@ $< -L$(WHISPER)/build-blas/bin -lggml -lggml-base -lggml-cpu -lpthread \
	    -Wl,-rpath,$(WHISPER)/build-blas/bin

test: test-mel test-encoder test-transcript test-speed

test-mel: tests/dump_mel tests/cmp_f32 harness/oracle_dump
	WHISPER=$(WHISPER) REF=$(REF) sh tests/gate_mel.sh

test-encoder: tests/dump_enc tests/cmp_f32 harness/oracle_dump
	WHISPER=$(WHISPER) REF=$(REF) sh tests/gate_encoder.sh

test-transcript: ears
	WHISPER=$(WHISPER) REF=$(REF) sh tests/gate_transcript.sh

test-speed: ears
	WHISPER=$(WHISPER) REF=$(REF) sh tests/gate_speed.sh

clean:
	rm -f ears tests/cmp_f32 tests/dump_mel tests/dump_enc harness/oracle_dump

.PHONY: all test test-mel test-encoder test-transcript test-speed oracle clean
