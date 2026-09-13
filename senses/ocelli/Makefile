# ocelli — the eyes of Yent. Pure C on notorch, no cloud, no Python.
#
#   make            build ./ocelli (BLAS on by default where we know it exists)
#   make BLAS=0     portable build, no BLAS (slow but dependency-free)
#   make asan       ASan/UBSan build for probes
#   make clean
#
# Weights are not in this repo. See README.

CC      ?= cc
CFLAGS  ?= -O2 -Wall
LDFLAGS ?= -lm
BLAS    ?= 1

UNAME_S := $(shell uname -s)

ifeq ($(BLAS),1)
  ifeq ($(UNAME_S),Darwin)
    CFLAGS  += -DUSE_BLAS -DACCELERATE
    LDFLAGS += -framework Accelerate
  else
    # Linux / Termux: expects an OpenBLAS with a cblas header on the include path.
    CFLAGS  += -DUSE_BLAS
    LDFLAGS += -lopenblas
  endif
endif

SRC = ocelli.c notorch.c gguf.c bpe.c vision.c
HDR = notorch.h gguf.h bpe.h vision.h notorch_vision.h stb_image.h

ocelli: $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

asan: $(SRC) $(HDR)
	$(CC) -O0 -g -fsanitize=address,undefined $(filter-out -O2,$(CFLAGS)) \
		-o ocelli_asan $(SRC) $(LDFLAGS)

clean:
	rm -rf ocelli ocelli_asan *.o *.dSYM

.PHONY: asan clean
