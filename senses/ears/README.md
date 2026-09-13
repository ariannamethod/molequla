# ears

Whisper on notorch. One wav in, one transcript out, in C, on a phone.

`ears` reads OpenAI Whisper weights in the flat ggml `.bin` format, computes the
log-mel spectrogram, runs the audio encoder and the text decoder, and prints what
was said. There is no Python anywhere in the project — not in the build, not in
the tests, not in the conversion path, because there is no conversion path: the
model file is read as it ships.

The reference implementation, [whisper.cpp](https://github.com/ggml-org/whisper.cpp),
is the oracle. Everything below is measured against it on the same machine, with
the command line that produced the number.

    ears <model.bin> <wav> [-l en|auto] [-t threads] [--no-speech-thold x] [--tokens] [--timestamps]

    $ ./ears ~/arianna/whisper.cpp/models/ggml-tiny.bin jfk.wav -l en -t 4
     And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.

## What it is made of

| file | what it holds |
|---|---|
| `ggml_bin.c/h` | the flat ggml `.bin` reader: header, hparams, mel filter bank, vocabulary, f16/f32 tensors |
| `mel.c/h` | the log-mel front end, and the wav reader |
| `encoder.c` | model loading, the two convolutions, and the audio encoder blocks |
| `decoder.c` | the text decoder: KV cache, cross-attention, tied logits |
| `tokenizer.c` | token ids to bytes, and the language table |
| `nnops.c/h` | LayerNorm, GELU, linear and multi-head attention over notorch |
| `ears.c` | the CLI, greedy decoding, the sliding window |
| `harness/oracle_dump.cpp` | whisper.cpp's own mel and encoder output, for the gates |
| `tests/` | the four gates and the tools they use |

Everything with arithmetic in it rides notorch primitives:

- `nt_qmatmul` — every encoder linear, f16 weights kept packed, f32 activations.
  One call per projection: `[1500, in] @ [out, in]^T`.
- `nt_qmatvec` — every decoder linear, including the 51865 × 512 tied output
  projection, which is why base decodes without materialising its 53 MB embedding
  as f32.
- `nt_attention` — self-attention in the encoder and both self- and
  cross-attention in the decoder. It takes separate query and key lengths, so
  cross-attention over 1500 audio frames with one query row is the same call.
- `nt_blas_mmT` — the f32 product behind each convolution.

Two things notorch does not have, and what was done about them:

- **No FFT, STFT, Hann or mel anywhere** in `notorch.h` or `notorch_vision.h`.
  The front end in `mel.c` is written from scratch, and it is the block that
  decides parity, so it follows whisper.cpp's arithmetic exactly: the same
  400-entry sin/cos table, the same real-input Cooley-Tukey recursion falling
  back to a naive DFT at the odd length 25, the same double-accumulated filter
  sums, the same global clamp. It reproduces whisper.cpp's mel bit for bit.
- **No `nt_conv1d`.** `nt_im2col` and `nt_conv2d` exist but assume a 2-D image,
  and the convolution here needs ggml's f16 column rounding to stay in step with
  the oracle, so `conv1d()` in `encoder.c` builds the columns itself and hands
  the product to `nt_blas_mmT`. No matvec and no attention is hand-rolled next to
  a notorch kernel that already does it.

## Build

    make                    # ./ears
    make oracle             # harness/oracle_dump, needs the whisper.cpp checkout
    make test               # all four gates, in order
    make test-mel           # one at a time
    WHISPER=... REF=...     # where the checkout and the reference wavs live

`-O2 -Wall -Werror`, C11, linking the system `libnotorch.a` with OpenBLAS found
through `pkg-config openblas`. notorch is not vendored.

**The two halves have different appetites.** `make` builds `ears` and needs
nothing but notorch, OpenBLAS and a C compiler — 1.7 s from cold on the A56, no
checkout of anything else, no weights. `make test` is the parity suite and needs
both of the things that are not here: a built whisper.cpp to be the oracle
(`harness/oracle_dump` `#include`s `src/whisper.cpp`, and `gate_speed` runs
`build-blas/bin/whisper-cli`), and the reference wavs with their stored oracle
transcripts. `WHISPER` and `REF` say where those are:

    make test WHISPER=/elsewhere/whisper.cpp REF=/elsewhere/ears-reference

The defaults are derived from this Makefile's own path rather than from `$HOME`,
which is what they used before this organ moved into molequla — inside the
phone's chroot `$HOME` is `/root` while the tree lives under
`/data/data/com.termux/files/home/arianna`, so a `$HOME`-relative default named a
directory that does not exist. Three levels up from `senses/ears` is the
directory holding molequla, `whisper.cpp` and `ears-reference` side by side,
which on the phone is exactly where they are.

Weights are not in this repo. They are whisper.cpp's own, unconverted — `ears`
reads the ggml format as it ships:

    bash models/download-ggml-model.sh tiny
    bash models/download-ggml-model.sh base

Multilingual, not the `.en` variants — `ears` v1 requires a multilingual vocab
(51865 entries) because the special-token layout is derived from it. On phone-1
the same two files are kept at `~/models/ears/`, outside the tree, where
`phone1/senses.sh` reads them; they are mirrored on Hugging Face under
[`ataeff/molequla`](https://huggingface.co/ataeff/molequla) in `ears/`.

## Parity

Four gates, in order, on a Galaxy A56 (Exynos 1580, 8 GB) under Ubuntu 24.04 in a
chroot, aarch64. Every run below is pinned with `taskset -c 0-3`: the big cores
4-7 belong to a scheduled molequla colony session, and a measurement taken while
that runs is not a measurement. The cores matter for the speed table and for
nothing else.

### 1. mel — bit-identical

    $ make test-mel
    cmp_f32: 328000 values  max|d| = 0.000e+00 at -1  mean|d| = 0.000e+00
    cmp_f32: PASS (tolerance 1.000e-04)

All 80 × 4100 values of the log-mel spectrogram of `jfk.wav`, against
`whisper_pcm_to_mel` called from `harness/oracle_dump`. Not close — equal.

Red, with one column of the filter bank zeroed
(`EARS_MEL_BREAK=100 make test-mel`):

    dump_mel: EARS_MEL_BREAK=100 — filter bin zeroed on purpose
    cmp_f32: 328000 values  max|d| = 1.286e-01 at 254273 (0.077497 vs 0.206103)  mean|d| = 1.124e-04
    cmp_f32: FAIL (tolerance 1.000e-04)

### 2. encoder — within the oracle's own half precision

    $ make test-encoder
    gate_encoder: against the flash-attention oracle (reported, not gated)
      cmp_f32: 576000 values  max|d| = 1.931e+00 at 2705  mean|d| = 1.979e-02
    gate_encoder: against the no-flash oracle (gated at 1e-1)
      cmp_f32: 576000 values  max|d| = 4.861e-02 at 2705 (5.258580 vs 5.307192)  mean|d| = 2.248e-04
      cmp_f32: PASS (tolerance 1.000e-01)

The asked-for gate was 1e-3 max abs. It is not reachable, and the reason is
arithmetic rather than effort. whisper.cpp holds K and V at f16 through the
encoder attention (`ggml_cast(..., wctx.itype)` on the plain path, an f16 KV pad
on the flash path). One f16 ULP at |x| = 5 is 3.9e-3, and encoder outputs here
reach |x| = 11. A max-abs tolerance below one ULP of the oracle's own
representation can only be met by a bit-exact reimplementation of ggml's kernels,
which is a different project from a port.

The two oracle columns say where the rest of the difference lives. With flash
attention — whisper-cli's default, and what `ears-reference/*.json` was recorded
with — ggml's CPU kernel accumulates the attention output in an **f16 register**
(`VKQ16` / `ggml_vec_mad_f16`, `ggml/src/ggml-cpu/ops.cpp:8710-8778`), and this
build reports `HAVE_FP16_VECTOR_ARITHMETIC` failed, so all 1500 accumulation
steps round back to half precision. That random walk is the 1.93 column. Turn
flash attention off and the same comparison drops to 4.9e-2 max and 2.2e-4 mean
over 576000 values: the port and the oracle agree to within half-precision noise,
and the noise is the oracle's.

Rounding Q, K and V to f16 on the way into `nt_attention` — which is what
whisper.cpp feeds its kernels — moved the no-flash figure from 1.345e-01 to
4.861e-02. The accumulation stays f32 here; that part of ggml's behaviour is a
backend artifact, not the model.

Red, with the stored positional embedding zeroed (the exact mistake `PORT_NOTES`
warns about — `encoder.positional_embedding` is a weight, not a sinusoid to
synthesise), `EARS_ENC_BREAK=1 make test-encoder`:

    cmp_f32: 576000 values  max|d| = 2.551e+01 at 553361 (-8.202149 vs 17.305824)  mean|d| = 1.290e+00
    cmp_f32: FAIL (tolerance 1.000e-01)

That red run found a second thing. The gate printed FAIL and exited 0, because
`cmp_f32 | sed` returns sed's status; `make test` would have gone green on a
broken encoder. Fixed in `tests/gate_encoder.sh`, and the only reason it was
found is that the gate was run red on purpose.

### 3. transcript — token for token, all six rows

    $ make test-transcript
    gate_transcript: PASS jfk/tiny  26 tokens
    gate_transcript: PASS speech_air_14s/tiny  27 tokens
    gate_transcript: PASS ambient_8s/tiny  0 tokens
    gate_transcript: PASS jfk/base  27 tokens
    gate_transcript: PASS speech_air_14s/base  28 tokens
    gate_transcript: PASS ambient_8s/base  0 tokens
    gate_transcript: all 6 rows equal token for token

The oracle is not `ears-reference/*.json`: those runs used whisper-cli's default
strategy, greedy with best-of 5 and a temperature-fallback ladder, which is a
different search and would have measured the search rather than the port.
`tests/make_greedy_oracle.sh` reruns whisper-cli with `-bo 1 -bs 1 -tp 0 -nf` —
one candidate, no beam, temperature zero, no fallback — into
`ears-reference/*.greedy.{json,txt}`, and the gate diffs the decode-order token id
stream against those.

The two microphone wavs are the interesting rows. `speech_air_14s.wav` is
`jfk.wav` played through the phone's speaker and re-recorded through its
microphone: real acoustic audio, room and speaker colouration included, with a
transcript known independently. Both models put the same tokens on it as
whisper.cpp does.

`ambient_8s.wav` is eight seconds of room noise, and both sides decode the same
tokens there too — tiny produces ` [Music]` (50364 542 8710 60 50764) and base
produces ` [Motor]` — but the no-speech gate suppresses them, on both sides, so
the transcript is empty. Verified by forcing the gate open on the oracle:

    whisper-cli ... -nth 1.1   ->  [00:00:00.000 --> 00:00:08.000]   [Music]

**One divergence, deliberate.** whisper.cpp's no-speech probability is read from
the wrong slot. `whisper_decode_internal` writes a prompt's logits into slot
`prompt.size()-1` of `state->logits`, and `whisper_full` then reads slot 0
(`src/whisper.cpp:7295-7299`). Under `-l auto` that slot still holds the
single-SOT decode that language detection ran, which happens to be the quantity
Whisper defines; under `-l en` nothing ever wrote it, and the gate scores a zeroed
buffer as a uniform distribution, so it never fires. On the same audio and the
same tokens:

    whisper-cli ... ambient_8s.wav -l auto   ->  (nothing)
    whisper-cli ... ambient_8s.wav -l en     ->  [00:00:00.000 --> 00:00:08.000]   [Music]

`ears` decodes the prompt in two parts and reads the probability at the SOT
position in both cases, which is the defined quantity and agrees with the oracle
wherever the oracle is defined. Measured on `ambient_8s.wav` with tiny:
`no_speech 0.6124`, `avg_logprob -1.3026`, both past their thresholds, suppressed.

Red, by patch rather than by a switch, because the thing to break lives in the
organ and not in the harness. Scaling Q by `head_dim^-0.25` in `ears_mha` on top
of the scale `nt_attention` already applies — the classic double-scale port bug,
since whisper.cpp's source literally writes that factor onto Q and onto K:

    -- jfk/tiny
      ears   : 13 50257
      oracle : 50364 400 370 452 7177 6280 1029 406 437 428 1941 393 360 337 291 11 1029 437 291 393 360 337 428 1941 13 50889
      gate_transcript: FAIL jfk/tiny
    -- ambient_8s/tiny
      ears   : 13 50257
      oracle :
      gate_transcript: FAIL ambient_8s/tiny

Three earlier candidate breaks did **not** go red, and that is worth writing down
because it says what this gate does and does not cover. Deleting the suppress-blank
filter, deleting the timestamp-pair rule, and giving `attn.key` the query's bias
each left all of `jfk/tiny` unchanged, token for token. The first two are logit
filters that simply do not bind on confident speech — the model's own distribution
already puts a timestamp first and never proposes a bare space. The third is a
no-op by construction: adding the same vector to every key shifts each score by a
per-query constant, which softmax cancels, which is presumably why Whisper has no
key bias to begin with. A gate is only evidence about the failures it can see.

### 4. speed and memory

Both sides pure greedy, 4 threads, `taskset -c 0-3`, `/usr/bin/time -v`:

| wav | model | lang | ears wall | ears peak RSS | whisper-cli wall | whisper-cli peak RSS |
|---|---|---|---|---|---|---|
| jfk (11.0 s) | tiny | `-l en` | 0:10.00 | 130 480 kB | 0:02.90 | 152 712 kB |
| speech_air_14s (14.1 s) | tiny | `-l auto` | 0:18.28 | 133 296 kB | 0:05.72 | 153 176 kB |
| jfk (11.0 s) | base | `-l en` | 0:22.72 | 211 104 kB | 0:05.39 | 246 096 kB |
| speech_air_14s (14.1 s) | base | `-l auto` | 0:41.77 | 218 224 kB | 0:09.68 | 246 492 kB |

`speech_air_14s.wav` is the phone's own microphone recording, so the second and
fourth rows are the ones that say what this costs on audio the device captured
itself.

An earlier run of the same gate on the same binaries, with the machine busier,
read 0:10.26 / 0:20.45 / 0:24.19 / 0:44.02 on the ears side and 0:02.92 / 0:05.37
/ 0:05.81 / 0:11.03 on whisper's — about 10 % spread run to run, in the same
direction on both sides. Single runs, not medians; treat the ratio, not the digit.

Honestly: whisper.cpp is 3.4 to 4.3 times faster here, and `ears` uses 11 to 15 %
less peak memory. The gap is the encoder. whisper.cpp runs it through a threaded
ggml graph with a REPACK kernel path and f16 operands end to end; `nt_qmatmul`
fans out over pthreads once per call and dequantises f16 to f32 inside the inner
loop, so the same GEMM moves twice the bytes. Nothing here is tuned — the project
was parity first, and the parity is what was measured.

The `-l auto` rows cost roughly double because language detection encodes the
first window and the transcription loop then encodes it again. That is whisper's
own shape and it is not a porting artifact; it is simply where half of those two
rows goes.

The `ears-reference/REFERENCE.md` table (tiny/jfk 10.46 s, base/jfk 19.90 s on
cores 4-7) is *not* comparable and was not used: those runs were the default
5-candidate decoder with temperature fallback. Comparing against them would have
credited `ears` for running a cheaper search. The table above runs both sides on
the same algorithm, the same cores and the same clock.

## What v1 does not do

- **Beam search and best-of.** One greedy decoder, always.
- **Temperature fallback.** No retry ladder. A window that fails under greedy is
  reported on stderr and dropped, rather than re-decoded hotter. The
  `ambient_8s`/base row in `REFERENCE.md` spent 169 seconds in exactly that path
  to arrive at `[Motor]`.
- **Prompt conditioning.** `no_context` is always on: no initial prompt, no
  carry-over of previous text into the next window. Nothing encodes text to ids,
  so there is no tokenizer direction for a prompt to use.
- **VAD, tinydiarize, DTW word timestamps, token suppression by regex, non-speech
  token suppression, grammar, translation.**
- **Quantised models.** f16 and f32 tensors only; a `whisper-quantize` output
  will fail at load with the tensor whose size it could not explain.
- **Anything but 16 kHz 16-bit mono wav.** No resampling, no ffmpeg.
- `.en` models — v1 needs the multilingual vocabulary.

Timestamps are decoded and used for the window advance, but printed only with
`--timestamps`.

## Language detection

`-l auto` encodes the first window, runs one decoder step on SOT alone, and takes
the argmax over the language token logits, before any filtering — the same
quantity `whisper_lang_auto_detect` takes. It agrees with the oracle on every row,
including the one where the oracle is wrong: `en` on `ambient_8s`/tiny, where
whisper-cli reports `en (p = 0.579721)`, and `nn` on `ambient_8s`/base, where it
reports `nn (p = 0.340178)` — eight seconds of room noise called Norwegian Nynorsk
by both sides, and then suppressed by both sides (`no_speech 0.6918`,
`avg_logprob -1.6276`).

One quirk is reproduced rather than corrected: whisper.cpp's language table has
100 entries while a model reporting `n_langs = 99` has room for ids 0-98, so the
last entry, `yue`, lands on the translate token. The scan here covers the same 100
ids for the same reason — a port that quietly disagreed with the oracle on one
input would be harder to trust on the rest.

## Licence

GPL-3.0-or-later. See `LICENSE`.
