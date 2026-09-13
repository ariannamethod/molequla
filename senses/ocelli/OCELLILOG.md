# OCELLILOG

Working log of the ocelli engine. Newest entries at the bottom. Every claim here
carries the command, file, or number it came from — this log is read by other
machines working on the same repo, so it records what changed in the system, not
who was in the room.

---

## 2026-07-27 — repo founded, engine moved in

The engine was developed inside a fork of a quiz application and has outgrown it:
it is an inference organ, not a game feature. Moved out as its own project.

**What moved:** `ocelli.c` (orchestrator, text decoder, vision splice — formerly
`smolvlm.c`), `vision.c/.h` (idefics3 preprocessing, SigLIP tower, pixel-shuffle
connector), `gguf.c/.h`, `bpe.c/.h` (GPT-2 byte-level BPE, special-token aware),
vendored `notorch.c/.h`, `notorch_vision.h`, `stb_image.h`, and the `eye` CLI.
Added: `Makefile` (BLAS on by default, `BLAS=0` for a portable build, `asan`
target), `.gitignore` (weights and binaries never committed), README.

**Own code vs vendored:** 1918 lines of engine C, 14349 lines vendored from the
notorch canon (`wc -l`).

### State carried over

- Two model generations run on this engine: SmolVLM-256M-Instruct and
  SmolVLM2-500M-Video-Instruct with a contract fine-tune (the Yent eye).
- Architecture is read from the file, never assumed: for the 500M weights the
  loader reports `E=960 H=15 KV=5 FFN=2560 V=49280 L=32 HD=64 rope=100000
  rms=1e-05`, matching `llama-mtmd-cli` metadata for the same GGUF.
- Vision tower reports `L=12 D=768 img=512 patch=16 patches=1024 eps=1e-06`,
  producing `[1024,768]` hidden states and `[64,960]` visual embeddings, NaN=0.
- Special tokens are identical across both generations (`<image>`=49190,
  `<fake_token_around_image>`=49189, `<global-img>`=49152,
  `<end_of_utterance>`=49279), prompt layout is 84 tokens.

### Fixed on the way in: MLP role must come from shape, not name

Loading the 500M weights produced a coherent text decoder but garbage
descriptions, while the reference runtime handled the same two files correctly.
Cause: the vision tower had a hardcoded `ffn_up` ↔ `ffn_down` swap, introduced
because one GGUF converter mirrors those names. A newer converter does not
mirror them, so the swap corrupted all 12 SigLIP layers.

Ground truth is the bias length in the file (`llama-gguf`): the older mmproj has
`ffn_down.bias = 12288 B`, the newer one has `ffn_up.bias = 12288 B`, and the
bias values themselves match between files to the digit (`-0.894531`,
`-1.398438`, `0.099609`) — the same SigLIP weights under mirrored names.

Fix: `siglip_load` now resolves fc1/fc2 per layer from the bias element count
(fc1 ⇔ bias == ffn); a length matching neither `ffn` nor `hidden` is a loud
load failure. Both conventions work; the older generation still reproduces its
stored reference outputs word for word.

### Cost, measured

Phone-class A18 Pro, 500M weights, one 512×512 frame, 84-token prompt:

| weights | prompt | generation | peak RSS |
|---|---|---|---|
| f16 | 3.6–4.1 tok/s | 3.8–3.9 tok/s | 1.0–1.4 GB |
| Q8_0 | 30.3–33.3 tok/s | 26.0–34.6 tok/s | 1.8–2.5 GB |

The 8× gap is not arithmetic. Q8_0 is dequantised once at load; the f16 path
expands every weight to f32 before every matmul, so ~500M weights are dequantised
per token. Descriptions agree between the two paths on most frames and diverge on
near-tie tokens, deterministically.

### Objective probe: small-text reading, 14 frames

Comparing free-form descriptions against a reference runtime is not a truth
test — both can invent clothing that is not in the frame, and on one portrait
both did (ours "white shirt", the reference "purple shirt", while the crop shows
no shirt at all). So the probe was narrowed to something with a knowable answer.

All 14 portraits carry the same watermark, `StyleGAN2 (Karras et al.)`, verified
by eye on two of them. Same weights, same prompt, same frames:

| runtime | watermark read correctly | corrupted readings |
|---|---|---|
| this engine | 0 / 14 | 2 (`Sylvanian`, `Icac`) |
| reference runtime | 5 / 14 | 3 (`StyGAN2`, `StyleGAN Karras et al.`, `MEG222`) |

Neither is good at it; the gap is one-sided. Since the weights are identical,
the loss is in our vision path — f32 tower against the reference's f16/mixed,
and possibly resize or normalisation upstream of it. This is the acceptance gate
for any accuracy work on the tower: the count on these 14 frames must rise.

Corollary, recorded so it is not relitigated: fine-tuning the weights is the
wrong instrument for this defect. It would compensate a runtime inaccuracy by
altering the model, welding our own error into the weights. Data first, and a
labelled set of real frames, before any training is discussed.

### Root cause of the blindness: one frame where the reference sends seventeen

The small-text failure was not the resize filter and not the weights. The
engine encoded a single global frame — 64 visual tokens — while the reference
runtime tiles the image and encodes many. Slice counts taken from its own log:

| frame | size | reference slices |
|---|---|---|
| portrait | 1024×1024 | 17 |
| terminal capture | 640×448 | 13 |
| app screenshot | 896×727 | 17 |

The rule reproduces on all three: scale the **longest edge to 2048** (up as well
as down — it was implemented here as a cap only), cut a `ceil(W/512) × ceil(H/512)`
grid of 512 tiles, append one global frame. 1024×1024 → 4×4+1 = 17.

A tiling path existed behind an opt-in flag but was dead: the preprocessor cut
the frames while the engine still encoded only the first one and spliced 64
tokens — and the first frame is the top-left tile, not the global view, so the
flag made descriptions worse, not better. The flag was calibrated in June
against a reference build that did not tile at the time; that calibration
expired with the upstream change and nobody re-checked it.

Now: every frame goes through the tower, the prompt carries per-tile
`<row_i_col_j>` markers row-major with a newline per tile row followed by the
global frame, and all `nf × 64` embeddings are spliced. Layout is confirmed by
token count rather than by reading someone's source — 1143 prompt tokens on a
4×4 grid, the exact prompt length the reference reports for the same image.

**Result on the frame that used to fail:** the watermark, previously read as
`Sylvanian`, now comes out verbatim: `StyleGAN2 (Karras et al.)`.

**Price:** `prompt 1143 tok in 374955 ms (3.0 tok/s) | gen 40 tok in 15477 ms |
peak RSS 1110 MB` — seventeen tower passes and 1143 decoder forwards. Memory is
unchanged; time is six minutes per frame on the f16 path. Packed kernels are no
longer an optimisation, they are what makes this usable.

**Not fixed by tiling, and not ours to fix in code:** both runtimes invent
clothing that is not in the frame (ours "glasses and a necklace", the reference
"purple shirt", on a crop showing neither). That is a data question.

### Correction: attribution in 28f68b8

The quote line in commit 28f68b8 carries a colleague's name under a sentence
that colleague never wrote — the line is mine, invented while writing the
commit. History is not rewritten here; the record stands with this correction
beside it. Quote lines in this repo carry either a verifiable author or my own
name, never a borrowed one.

### Packed weights and batched prefill

Two changes, measured separately because they pull in different directions.

**Packed weights.** A weight whose dtype has a notorch row kernel is now kept
exactly as it sits in the file and dotted per block (`nt_qmatvec`), instead of
being expanded to f32 at load. Token embeddings stay dense — they are read
row-wise and packing buys nothing there. Peak RSS on the control frame fell from
2202 MB (Q8 expanded) and 1110 MB (f16) to **528 MB**, output unchanged.

Packed alone, however, made generation *slower*: a per-block scalar dot loses to
vectorised BLAS on expanded weights, and `nt_qmatvec` only threads above 4M
elements while our matrices are 2.46M — the whole decode runs single-threaded.

**Batched prefill.** The prompt was being run as one forward per token: 1143
separate matvecs where the reference runtime does one matmul per layer. Prefill
now processes the whole prompt at once, so each weight is expanded ONCE into
scratch and multiplied against all rows — the dequant cost is paid per prompt
instead of per token.

| path | prompt (1143 tok) | rate | peak RSS |
|---|---|---|---|
| f16, per-token | 374955 ms | 3.0 tok/s | 1110 MB |
| Q8 packed, per-token | 481764 ms* | 2.4 tok/s* | 528 MB |
| Q8 packed, batched prefill | **2853 ms** | **400.6 tok/s** | 916 MB |

\* measured under contention, upper bound only.

131× on the prompt, and above the reference runtime's 244 tok/s on the same
prompt. Output is byte-identical to the per-token path, watermark included, so
the batch is arithmetically equivalent rather than approximately similar. Whole
frame: **31 s** (vision 16 s + prompt 2.9 s + generation 12 s) against 405 s
before.

RSS rose from 528 MB to 916 MB because prefill scratch and the dense token
embedding table (49280×960 f32 = 189 MB) are live at once; keeping embeddings
packed and extracting rows on demand is the obvious next cut.

**Still slow: generation at 3.3 tok/s**, single-threaded packed matvec. Raising
the kernel's threading floor is the next lever and it lives in vendored notorch,
which makes it ours.

### Small-text metric, re-measured end to end

Same 14 portraits, same prompt, final binary (tiling + packed Q8 + batched
prefill). True watermark on every frame: `StyleGAN2 (Karras et al.)`.

| runtime | exact reads | corrupted | not seen |
|---|---|---|---|
| this engine, before (single frame) | 0 / 14 | 2 | 12 |
| this engine, now | **4 / 14** | 1 (`Icán`) | 8 |
| reference runtime | 5 / 14 | 3 | 6 |

Exact reads: Alex Cox, Chloe Gloop, Laura Bishop (`StyleGAN2 (Karras et al.)`
in full), Toby Ferguson (both halves). One partial: `StyleGAN` without the
digit. Whole frame now costs 31 s (vision 16 s, prompt 2.9 s, generation 12 s)
against 405 s before.

A 14/14 target is not reachable in this form — the reference runtime does not
reach it either on the same weights. The honest bar is "not worse than the
reference", and the gap is now one frame.

### Two columns, same machine, same weights, same frame

Free machine, Q8 weights, tiling on both sides, one 1024×1024 portrait:

| column | reference | ocelli | gap |
|---|---|---|---|
| prefill (1143 tok) | 4459 ms — 256.3 tok/s | **2931 ms — 390.0 tok/s** | ours 1.5× faster |
| decode | 396 ms — 88.4 tok/s | 12496 ms — 3.2 tok/s | **ours 28× slower** |
| vision (17 slices) | inside its prompt-eval accounting | 17290 ms | listed separately |
| whole frame | **5917 ms** | 32717 ms | ours 5.5× slower |

Caveat so the columns do not lie: the reference appears to fold slice encoding
into its prompt-eval figure while ours is a separate line, so only the
whole-frame row is directly comparable.

Where the remaining distance lives, named rather than averaged: decode at 3.2
tok/s (single-threaded packed matvec — the kernel only threads above 4M elements
and our matrices are 2.46M) and the tower at 17.3 s (f32, one frame at a time).
Prefill is no longer a problem.

### Threading floor: a constant measured on another shape

The packed matvec only fanned out above 4M elements. That floor was measured on
a 360M model, where fan-out was noise; a 500M decoder's matrices are 2.46M and
sat just under it, so the entire decode ran single-threaded. Made tunable
(`NT_QMV_THREAD_MIN`) and measured on one frame, same weights, byte-identical
output both times:

| floor | decode |
|---|---|
| 4M (previous) | 4895 ms / 18 tok — 3.7 tok/s |
| 256K (now default here) | 2455 ms / 18 tok — **7.3 tok/s** |

Full tiled frame after the change: vision 16556 ms, prompt 2956 ms (386.7
tok/s), generation 5212 ms (**7.7 tok/s**, was 3.2), peak RSS 1006 MB, watermark
still read in full. Whole frame **24.7 s** against 32.7 s before and 405 s at
the start of the day; the reference does the same frame in 5.9 s, so the gap is
4.2×.

### Where the remaining time actually is

Of 24.7 s, the tower is 16.6 s — two thirds. Seventeen SigLIP passes are roughly
1.6 TFLOP, which at 17 s is ~95 GFLOPS: that is already near what Accelerate
gets out of f32 sgemm on this chip. The tower is not slow because of our loop;
it is at the CPU f32 ceiling. Going further there means f16 arithmetic or the
Metal backend that notorch already carries.

Decode at 7.7 tok/s against the reference's 88.4 is the other front, and it
needs an int8-activation kernel — the vendored one exists only for Q4_0.

### int8 decode: 2.8× faster and it loses the watermark — so it is not the default

The packed matvec now has an int8-activation sibling for Q8_0 in notorch
(`nt_qmatvec_i8`), and the vendored copy here is byte-identical to canon. Kernel
agreement against the exact dot was measured on real tensors of these weights:
rel L2 0.0027–0.0038, 4.3× on attention shapes and 21× on FFN shapes.

End to end on the control frame, same weights, same prompt:

| decode path | rate | watermark |
|---|---|---|
| exact (default) | 8.0 tok/s | `StyleGAN2 (Karras et al.)` read in full |
| int8 (`OCELLI_I8=1`) | **21.8 tok/s** | **lost** — the sentence ends in a blurry background |

Three parts in a thousand per matrix is not small once it passes through 32
layers: the accumulated drift changes token choices, and the first thing to go
is exactly what this engine was fixed to see. So the exact dot stays the default
and int8 is opt-in, for callers who want speed over small text.

The threading floor is no longer patched into the vendored copy — canon takes
`NT_QMV_THREAD_MIN` and `eye` sets 256K for this shape, so vendor stays a clean
mirror of canon.

Target unmet: the reference decodes at 88.4 tok/s while reading the same
watermark, so int8 done right does not have to cost accuracy. Next probe is a
hybrid — int8 on the FFN projections, where the 21× lives, and the exact dot on
attention, where token placement is decided.

### Hybrid decode: int8 on FFN, exact on attention — 1.7× and nothing lost

The all-int8 result said where the sensitivity lives. Attention projections
decide where the model looks; FFN projections do not. Splitting the policy along
that line:

| decode path | rate | watermark |
|---|---|---|
| exact everywhere | 8.0 tok/s | read in full |
| **hybrid (default)** | **13.4 tok/s** | **read in full, output byte-identical to exact** |
| int8 everywhere (`OCELLI_I8=1`) | 21.8 tok/s | lost |

Same weights, same prompt, same frame. The hybrid is now the default; both
extremes stay reachable by environment variable, `OCELLI_EXACT=1` for the
reference path. Whole frame is about 22 s.

Still short of the reference's 88.4 tok/s — that gap is now attention plus the
tower, not FFN.

Across all 14 portraits the count holds and completeness moves: exact path reads
`StyleGAN2` on 4 frames with 2 of them carrying `Karras et al.` as well; hybrid
reads the same 4 with all 4 complete. Not claimed as better sight — the extra
half-strings are near-tie continuations shifted by different FFN numbers, and the
count of frames where the mark is seen at all is unchanged.

### Open

- Hot path still runs dense f32 through BLAS. The vendored notorch already ships
  packed kernels (`nt_qmatvec`, row kernels for `q4_0/q8_0/q5_0/q4_k/q6_k`) that
  dequantise per block in registers with no f32 resident. Moving the decoder and
  the tower onto them is the path to phone-sized memory; acceptance is a parity
  run against the current Q8 path with named divergences, plus speed and RSS on
  the same frames.
- Not bit-exact against the reference runtime (f32 tower vs f16/mixed there).
- Small phone-sized text is a blind spot.
- Camera input (V4L2 / phone) is not implemented; the eye currently eats files.
