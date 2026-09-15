# ears — whisper on notorch. Pure C, no Python anywhere, not even in the tests.
#
#   make                build ./ears and ./soundscape
#   make test           the gates, in order (soundscape, mel, encoder, transcript, speed)
#   make test-mel       one gate at a time
#   make soundscape     the sound describer alone — needs no weights and no oracle
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

# The describer is the other half of hearing and shares only the wav reader with
# the recognizer: no weights, no tokenizer, no encoder. It is its own binary for
# that reason — `senses.sh` runs both over the same wav and writes two different
# fragments.
SND_SRC = soundscape.c mel.c
SND_HDR = soundscape.h mel.h

# Where the oracle and the reference wavs live. Neither is inside this repo and
# neither is needed to build: `make` wants only notorch and OpenBLAS; `make test`
# wants both of these.
#
# The default is derived from this Makefile's own path, not from $(HOME): inside
# the phone's chroot $(HOME) is /root while the tree lives under
# /data/data/com.termux/files/home/arianna, and a $(HOME)-relative default
# pointed the gates at a directory that does not exist. Three levels up from
# senses/ears is the repository's parent — the `arianna/` directory that holds
# molequla, whisper.cpp and ears-reference side by side. Override either on the
# command line when a checkout sits somewhere else:
#
#   make test WHISPER=/elsewhere/whisper.cpp REF=/elsewhere/ears-reference
ARIANNA ?= $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../../..)
WHISPER ?= $(ARIANNA)/whisper.cpp
REF     ?= $(ARIANNA)/ears-reference

all: ears soundscape

ears: $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

soundscape: soundscape_main.c $(SND_SRC) $(SND_HDR)
	$(CC) $(CFLAGS) -o $@ soundscape_main.c $(SND_SRC) $(LDFLAGS)

tests/test_soundscape: tests/test_soundscape.c $(SND_SRC) $(SND_HDR)
	$(CC) $(CFLAGS) -o $@ tests/test_soundscape.c $(SND_SRC) $(LDFLAGS)

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

test: test-soundscape test-mel test-encoder test-transcript test-speed

# The only gate here that needs neither weights nor the whisper.cpp checkout:
# the fixtures are synthesised by the test itself into tests/out/soundscape.
test-soundscape: tests/test_soundscape
	mkdir -p tests/out/soundscape
	./tests/test_soundscape

test-mel: tests/dump_mel tests/cmp_f32 harness/oracle_dump
	WHISPER=$(WHISPER) REF=$(REF) sh tests/gate_mel.sh

test-encoder: tests/dump_enc tests/cmp_f32 harness/oracle_dump
	WHISPER=$(WHISPER) REF=$(REF) sh tests/gate_encoder.sh

test-transcript: ears
	WHISPER=$(WHISPER) REF=$(REF) sh tests/gate_transcript.sh

test-speed: ears
	WHISPER=$(WHISPER) REF=$(REF) sh tests/gate_speed.sh

clean:
	rm -f ears soundscape tests/cmp_f32 tests/dump_mel tests/dump_enc tests/test_soundscape harness/oracle_dump
	rm -rf tests/out/soundscape

.PHONY: all test test-soundscape test-mel test-encoder test-transcript test-speed oracle clean
