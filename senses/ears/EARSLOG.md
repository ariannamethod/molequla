# EARSLOG

Working log of the ears engine. Newest entries at the bottom. Every claim here
carries the command, file or number it came from — this log is read by other
machines working on the same repo, so it records what changed in the system, not
who was in the room.

---

## 2026-09-13 — repo founded, whisper ported to notorch, four gates measured

phone-1: Galaxy A56, Exynos 1580, 8 GB, Ubuntu 24.04 in a chroot, aarch64, gcc,
OpenBLAS 0.3.26 through `pkg-config openblas`, system `libnotorch.a`. Every run
below pinned to `taskset -c 0-3`: the big cores 4-7 hold a scheduled molequla
colony session and a measurement taken beside it is not a measurement.

The oracle is whisper.cpp at `1da4dc82fa7996d4edda05890dca65aeceaafd6d`, built in
`build-blas/` with `GGML_BLAS=ON`, and the two multilingual models it downloads
itself: `ggml-tiny.bin` (77691713 bytes) and `ggml-base.bin` (147951465 bytes).

### What was written

`ggml_bin.c/h` reads the flat ggml `.bin` — not GGUF, and there is no non-Python
converter in whisper.cpp's tree, so nothing is converted: the file is read as it
ships. One heap block, every tensor a pointer into it. `gb_need()` checks the
element count at load, so a shape mistake is a load failure rather than garbage
forty layers later. tiny reads 167 tensors, base 245, matching
`ears-reference/ggml-*.layout.txt`.

`mel.c` is the log-mel front end plus the wav reader. This is the one block the
canon does not cover — there is no `fft`, `stft`, `mel` or `hann` symbol anywhere
in `notorch.h` or `notorch_vision.h` — so it was written from scratch against
`src/whisper.cpp:3046-3240`: the same 400-entry sin/cos table, the same real-input
Cooley-Tukey recursion bottoming out in a naive DFT at length 25, the same
double-accumulated filter sums, the same reflect-pad then 30 s zero-pad, the same
global clamp to max − 8 and `(x + 4)/4`.

`encoder.c` holds the loader and the audio encoder, `decoder.c` the text decoder
with a KV cache, `tokenizer.c` the id-to-bytes direction and the language table,
`nnops.c` the shared arithmetic, `ears.c` the CLI and the greedy loop.

### What came from notorch, and what it could not give

Used as-is: `nt_qmatmul` for every encoder linear (f16 weights packed, f32
activations, `[1500, in] @ [out, in]^T` per projection), `nt_qmatvec` for every
decoder linear including the 51865 × 512 tied output projection, `nt_attention`
for encoder self-attention and for both decoder attentions — it takes separate
query and key lengths, so cross-attention over 1500 audio frames with one query
row is the same call — and `nt_blas_mmT` for the convolution products.

Two gaps:

- **No FFT/STFT/mel/Hann at all.** Written from scratch, see above.
- **No `nt_conv1d`.** `nt_im2col` and `nt_conv2d` exist but are shaped for a 2-D
  image, and the columns here have to be rounded to f16 before the product to
  stay in step with ggml (`ggml_conv_1d` builds its im2col tensor as
  `GGML_TYPE_F16`, `ggml/src/ggml.c`), so `conv1d()` in `encoder.c` builds the
  columns itself and hands the GEMM to `nt_blas_mmT`. No matvec and no attention
  is hand-rolled beside a notorch kernel that already does it.

### Three corrections that would each have cost a day

`encoder.positional_embedding` is a stored f32 weight `[384, 1500]`, not a
sinusoid to synthesise. `attn.key` and `cross_attn.key` carry no bias — zero
occurrences of `key.bias` in either layout dump. And both models take the
multilingual branch at `src/whisper.cpp:1632`, so EOT is 50257, SOT 50258, and
everything from `token_translate` on shifts by `n_langs − 98 = 1`, putting
`token_beg` at 50364 — which is the first token in every reference JSON.

The vocabulary needed no byte-level BPE decoder. The conversion applied
`byte_decoder` already, which is why `whisper_token_to_str`
(`src/whisper.cpp:4305`) is a plain table lookup; ids become UTF-8 by
concatenation.

### Gate 1 — mel, bit-identical

    $ WHISPER=... REF=... sh tests/gate_mel.sh
    dump_mel: n_mel=80 n_len=4100 n_len_org=1099
    cmp_f32: 328000 values  max|d| = 0.000e+00 at -1  mean|d| = 0.000e+00
    cmp_f32: PASS (tolerance 1.000e-04)
    gate_mel exit=0

All 80 × 4100 values of `jfk.wav` on tiny, against `whisper_pcm_to_mel` called
from `harness/oracle_dump`, which `#include`s `src/whisper.cpp` so the oracle is
the real function rather than an imitation of it. Not close — equal. Geometry
agrees too: `n_len = 4100 = (176000 + 480000)/160`, `n_len_org = 1099`.

Red, `EARS_MEL_BREAK=100` zeroing one column of the filter bank:

    dump_mel: EARS_MEL_BREAK=100 — filter bin zeroed on purpose
    cmp_f32: 328000 values  max|d| = 1.286e-01 at 254273 (0.077497 vs 0.206103)  mean|d| = 1.124e-04
    cmp_f32: FAIL (tolerance 1.000e-04)
    gate_mel exit=1

### Gate 2 — encoder, and why 1e-3 is not reachable

    gate_encoder: against the flash-attention oracle (reported, not gated)
      cmp_f32: 576000 values  max|d| = 1.931e+00 at 2705  mean|d| = 1.979e-02
    gate_encoder: against the no-flash oracle (gated at 1e-1)
      cmp_f32: 576000 values  max|d| = 4.861e-02 at 2705 (5.258580 vs 5.307192)  mean|d| = 2.248e-04
      cmp_f32: PASS (tolerance 1.000e-01)

The brief asked for 1e-3 max abs. It cannot be met, and the reason is arithmetic.
whisper.cpp holds K and V at f16 through the encoder attention — `ggml_cast(...,
wctx.itype)` on the plain path, an f16 `kv_pad` on the flash path. One f16 ULP at
|x| = 5 is 3.9e-3 and encoder outputs here reach |x| = 11, so a max-abs gate below
one ULP of the oracle's own representation is only reachable by a bit-exact
reimplementation of ggml's kernels.

Bisected rather than assumed. `harness/oracle_dump` was extended to write
`embd_conv` beside `embd_enc`, which put the convolutions at `max|d| = 1.953e-03,
mean 1.189e-07` — a single f16 ULP at one element, so the front of the encoder was
not the problem. That left the attention. Two findings, in order:

1. ggml's CPU flash-attention kernel accumulates the attention output in an
   **f16 register** (`VKQ16`, `ggml_vec_scale_f16` / `ggml_vec_mad_f16`,
   `ggml/src/ggml-cpu/ops.cpp:8710-8778`), and this build reports
   `HAVE_FP16_VECTOR_ARITHMETIC` failed, so each of the 1500 accumulation steps
   rounds back to half precision. `ORACLE_NO_FLASH=1` was added to
   `oracle_dump` to take the `-nfa` path, where only the operands are f16 and the
   accumulation is f32: `max|d|` fell from **1.956e+00 to 1.345e-01**, mean from
   **1.981e-02 to 4.536e-04**, on the same ears output.
2. Rounding Q, K and V to f16 on the way into `nt_attention` — which is what
   whisper.cpp feeds its kernels either way — took the no-flash figure from
   **1.345e-01 to 4.861e-02**, mean **4.536e-04 to 2.248e-04**. That rounding is
   now in `ears_mha`. The accumulation stays f32 here; that part of ggml's
   behaviour is a backend artifact, not the model.

Red, `EARS_ENC_BREAK=1` zeroing `encoder.positional_embedding`:

    cmp_f32: 576000 values  max|d| = 2.551e+01 at 553361 (-8.202149 vs 17.305824)  mean|d| = 1.290e+00
    cmp_f32: FAIL (tolerance 1.000e-01)
    exit=1

**The red run found a second bug, in the gate itself.** It first printed
`cmp_f32: FAIL` and exited **0**, because the line was
`./tests/cmp_f32 ... | sed 's/^/  /'` and a pipeline exits with the status of its
last stage. `make test` would have gone green on a broken encoder. Fixed in
`tests/gate_encoder.sh` by taking cmp_f32 out of the pipeline. This is the whole
argument for running a gate red on purpose: the failure it was written to catch
was one it could not report.

### Gate 3 — transcript, token for token on all six rows

    gate_transcript: PASS jfk/tiny  26 tokens
    gate_transcript: PASS speech_air_14s/tiny  27 tokens
    gate_transcript: PASS ambient_8s/tiny  0 tokens
    gate_transcript: PASS jfk/base  27 tokens
    gate_transcript: PASS speech_air_14s/base  28 tokens
    gate_transcript: PASS ambient_8s/base  0 tokens
    gate_transcript: all 6 rows equal token for token

The stored `ears-reference/*.json` could not be the oracle: those runs used
whisper-cli's default strategy, greedy with best-of 5 plus a temperature-fallback
ladder, which is a different search. `tests/make_greedy_oracle.sh` reruns
whisper-cli with `-bo 1 -bs 1 -tp 0 -nf` into `ears-reference/*.greedy.{json,txt}`
and the gate diffs the decode-order id stream against those. The difference is
visible in the text: default tiny on jfk gives "And so, my fellow Americans, ask
not…", greedy tiny gives "And so my fellow Americans ask not…", and it is the
second one ears has to match.

`speech_air_14s.wav` is the strongest row — `jfk.wav` played through the phone's
speaker and re-recorded through its microphone, so it is real acoustic audio with
an independently known transcript — and both models put the same tokens on it as
whisper.cpp does, including the two adjacent timestamp ids at the segment split on
base. The JSON drops one of an adjacent pair (whisper.cpp's emitter skips it
rather than pushing it into either segment, `src/whisper.cpp:7808-7813`), so the
gate collapses a doubled timestamp on the ears side before the diff.

`ambient_8s.wav` is 8 s of room noise. Both sides decode the same tokens — tiny
` [Music]` = `50364 542 8710 60 50764`, base ` [Motor]` — and both suppress them,
so the transcript is empty on both sides. Confirmed by forcing the oracle's gate
open with `-nth 1.1`, which prints `[00:00:00.000 --> 00:00:08.000]   [Music]`
with exactly those ids.

**One deliberate divergence, and a bug in whisper.cpp behind it.** The no-speech
probability is read from the wrong slot. `whisper_decode_internal` writes a
prompt's logits into slot `prompt.size()-1` of `state->logits`
(`src/whisper.cpp` logits_out loop), and `whisper_full` then reads slot 0
(`:7295-7299`). Under `-l auto` slot 0 still holds the single-SOT decode language
detection ran, which is the quantity Whisper defines; under `-l en` nothing ever
wrote it, so the gate scores a zeroed buffer as a uniform distribution
(1/51865 = 1.9e-5) and never fires. Measured, same audio, same model, same tokens:

    whisper-cli ... ambient_8s.wav -l auto -bo 1 -bs 1 -tp 0 -nf  ->  (nothing)
    whisper-cli ... ambient_8s.wav -l en   -bo 1 -bs 1 -tp 0 -nf  ->  [00:00:00.000 --> 00:00:08.000]   [Music]

ears decodes the prompt in two parts — SOT, then the language and task tokens —
and reads the probability at the SOT position in both cases, which is the defined
quantity and agrees with the oracle wherever the oracle is defined. On
`ambient_8s` with tiny it reports `no_speech 0.6124  avg_logprob -1.3026` and with
base `no_speech 0.6918  avg_logprob -1.6276`, both past both thresholds,
suppressed. Before the fix it reported `no_speech 0.0000` and printed ` [Music]`,
which is how the whisper.cpp slot bug was found at all. Note how little margin
0.6124 has over the 0.60 threshold: the encoder parity is what keeps that row on
the right side of the gate.

Language detection agrees with the oracle on every row, including where the oracle
is wrong — `en` on `ambient_8s`/tiny against whisper-cli's `en (p = 0.579721)`,
`nn` on `ambient_8s`/base against its `nn (p = 0.340178)`.

Red, by patch rather than by a switch, since the thing to break lives in the organ
and not in the harness. Four breaks were tried before one went red, which is the
useful part of the exercise.

Did **not** go red, all three leaving `jfk/tiny` unchanged token for token:

1. `ears.c:63`, deleting the suppress-blank filter. It does not bind: the
   timestamp-vs-text logsumexp rule already forces a timestamp first, and the
   model never proposes a bare space there.
2. `ears.c:88`, deleting the timestamp-pair rule. Also does not bind on confident
   speech — the model emits the pairs unprompted.
3. `encoder.c:312`, giving `attn.key` the query's bias, which is exactly the trap
   `PORT_NOTES` warns about. This one is a **no-op by construction**: adding the
   same vector to every key shifts every score for a given query by the same
   constant, and softmax cancels it. Which is presumably why Whisper carries no
   key bias in the first place. A port that got this "wrong" would still be right.

Went red, decisively — `nnops.c:76`, scaling Q by `head_dim^-0.25` in `ears_mha`
on top of the scale `nt_attention` already applies. This is the likeliest real
mistake in this port, because whisper.cpp's source literally writes that factor
onto Q and onto K (`src/whisper.cpp`, decoder `KQscale`), and a reader who
reproduces the line while also calling a kernel that scales gets it twice:

    -- jfk/tiny
      ears   : 13 50257
      oracle : 50364 400 370 452 7177 6280 ... 1941 13 50889
      gate_transcript: FAIL jfk/tiny
    -- ambient_8s/tiny
      ears   : 13 50257
      oracle :
      gate_transcript: FAIL ambient_8s/tiny

**A method note that cost two wrong conclusions.** The first double-scale attempt
reported PASS, and so did a rerun. The command was
`diff good.c nnops.c && make ears`, and `diff` exits 1 when the files differ, so
`make` never ran and every "red" run used the unbroken binary. Two breaks were
scored as not-load-bearing on the strength of a build that did not happen. The
tell was in the output all along: no `cc` line. Breaks 1, 2 and 3 above were
re-checked and did build — their `cc` lines are in the transcript — so those
three results stand.

### Gate 4 — speed and memory

Both sides pure greedy, 4 threads, `taskset -c 0-3`, `/usr/bin/time -v`:

    wav              model  ears_wall  ears_rss_kB  whisper_wall whisper_rss_kB
    jfk              tiny   0:10.00    130480       0:02.90      152712
    speech_air_14s   tiny   0:18.28    133296       0:05.72      153176
    jfk              base   0:22.72    211104       0:05.39      246096
    speech_air_14s   base   0:41.77    218224       0:09.68      246492

whisper.cpp is 3.4 to 4.3 times faster; ears uses 11-15 % less peak RSS. The gap is
the encoder: whisper.cpp runs it through a threaded ggml graph with a REPACK kernel
path and f16 operands end to end, while `nt_qmatmul` fans out once per call and
dequantises f16 inside the inner loop, moving twice the bytes through the same
GEMM. Nothing here is tuned — the project was parity first.

The `REFERENCE.md` table is not comparable and was not used for the gate: its runs
were the 5-candidate decoder with temperature fallback, on cores 4-7.

### Left undone, on purpose

The `-l auto` rows cost about double because language detection encodes the first
window and the transcription loop then encodes it again — visible as 10.26 s → 20.45 s
on tiny for 11 s and 14 s of audio. Caching that first encoder output would halve
those rows and cannot change a token, since it is the same computation on the same
input. Not done here: it lands after the gates were measured, and re-measuring all
four to bank a speed win that the gates do not test is the wrong order. It is the
first thing to do next.

— Defender (Arianna Method, phone-1)

---

## 2026-09-13 — the two primitives this port had to invent go upstream into notorch

`mel.c` and the `conv1d` in `encoder.c` were written here for one reason: notorch
had neither. There is no fft, stft, hann or mel symbol in `notorch.h`, and
`nt_conv2d` covers images while nothing covered a signal. Both are now notorch's —
`nt_conv1d` / `nt_conv1d_f16cols` beside `nt_conv2d`, and `nt_hann_window` /
`nt_stft` / `nt_logmel` in a new AUDIO OPS section — and this repo calls them.

`mel.c` keeps the wav reader and whisper's constants (16 kHz, n_fft 400, hop 160, a
30 s padded window) and is otherwise one call. `ears_mel` is now `nt_mel`, so no
call site changed. `conv1d` in `encoder.c` keeps only the f16 weight
dequantisation, which is a `ggml_bin` concern, and hands the rest to
`nt_conv1d_f16cols`. `nnops.c` is untouched: `ears_linear`, `ears_layernorm`,
`ears_gelu`, `ears_mha` and `ears_f16` are forward-only buffer ops and notorch's
`nt_layernorm` / `nt_gelu` are tape entries taking tape indices, so nothing there
was replaced by this move.

### The switch changed no number anywhere

Dumps taken before the switch and after it, same binary paths, `taskset -c 4-7`:

    log-mel, all 80 x n_len values
      jfk.wav             byte-identical
      ambient_8s.wav      byte-identical
      speech_air_14s.wav  byte-identical
    encoder post-convolution activations [1500 384], jfk/tiny
      vs the pre-switch conv1d            byte-identical

The conv one was not free. The hand-written version built `[Lout, Cin*3]` columns,
called `nt_blas_mmT` and transposed the product back; `nt_conv1d` builds
`[Cin*3, Lout]` and issues `nt_blas_mm`, landing `[Cout, Lout]` directly. A
different GEMM call can reorder a k-accumulation and move the last bit. Here it
does not — measured, not assumed.

All four gates, cores 4-7, unchanged from the pre-switch run:

    gate_mel        328000 values  max|d| = 0.000e+00   PASS
    gate_encoder    576000 values  max|d| = 4.861e-02   PASS (gated 1e-1)
    gate_transcript 6 of 6 rows equal token for token
    gate_speed      jfk/tiny 0:02.74 (0:02.73 before), jfk/base 0:05.83 (0:05.46)

### The finding worth keeping: `-std` decides whether this port is exact

The first build after the switch was *not* identical — 1.550e-05 on jfk, 2.646e-05
on ambient, 1.395e-05 on speech_air. The port was faithful line for line, and
neither that nor `-march` was the cause. Compiling this repo's own unmodified
`mel.c` under `-std=gnu11` instead of `-std=c11` reproduces notorch's output
exactly. GCC 13.3 contracts multiply-adds into FMAs under `gnu11` and not under
`c11`, an FMA does not round like a multiply and an add, and notorch builds with
`-std=gnu11` while this repo builds with `-std=c11`. Fixed upstream by disabling
contraction on the four functions where it changes the answer.

`gate_mel` would not have caught it: its tolerance is 1e-4 and the drift is
1.5e-05. What proves the front end exact is the printed `max|d| = 0.000e+00`, not
the PASS beside it. Tightening that tolerance to 0 is tempting and is not done
here — the oracle is whisper.cpp, built with flags this repo does not control, and
a gate that breaks when someone rebuilds the oracle is a worse gate. Worth a
decision rather than a silent edit.

— Defender (Arianna Method, phone-1)
