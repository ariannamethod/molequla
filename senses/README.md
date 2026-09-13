# senses

The organs. An organism that only ever eats its siblings' speech is a closed
loop; these are where the world gets in. Both are C engines on notorch, both
were their own repositories before 2026-09-13, and both are in this tree now by
Oleg's decision — one folder, no satellite repository for either, history
carried across with `git subtree add` rather than flattened into a copy.

Nothing here is linked into the Go build. `go list ./...` does not see these
directories, because there is no `.go` file in them and cgo compiles only the
`.c` beside a Go package; each engine is built by its own Makefile and run as a
separate process by a shell script. That is deliberate. The eye carries its own
copy of notorch (`ocelli/notorch.c`) and the ears link the system `libnotorch.a`;
putting either inside the organism's address space would put a gigabyte of vision
weights next to four growing organisms on an 8 GB phone.

| | what it is | build | gate |
|---|---|---|---|
| `ocelli/` | the eye — SmolVLM2-500M end to end in C, vision tower, projector and text decoder, with its own notorch vendored | `cd senses/ocelli && make` → `./ocelli`, driven by `./eye <image> [prompt]` | the 14-frame watermark probe in `OCELLILOG.md`: same portraits, same prompt, the count of frames whose `StyleGAN2 (Karras et al.)` mark is read must not fall. Hand-run, not `make test`. |
| `ears/` | the ear — OpenAI Whisper in C on notorch: ggml `.bin` reader, a log-mel front end written from scratch because notorch has no FFT, encoder, decoder, tokenizer | `cd senses/ears && make` → `./ears <model.bin> <wav>` | `make test` — four gates against whisper.cpp as oracle: mel, encoder, transcript (token for token under pure greedy, 6 rows) and speed. Runs in `EARSLOG.md`. |

`place` has no folder here. It is thirty lines of `bash` inside
`../phone1/senses.sh` — `termux-location`, then open-meteo and nominatim over
`curl` and `jq` — and an organ that is a shell function does not get a directory
to look like the other two.

## What drives them

`../phone1/senses.sh` is the only caller: one pass of eye, ears and place, then
exit. It runs in its own short slots between the colony's windows, writes what it
found into `$MOLEQULA_RUN/dna/output/{world,sound,place}/` as timestamped
fragments, and the organisms eat those with `--dna-extra-sources
world,sound,place`. Every engine path and every model path in it is one
environment variable — `SENSES_EYE`, `SENSES_EYE_MODEL`, `SENSES_EYE_MMPROJ`,
`SENSES_ASR`, `SENSES_ASR_MODEL` — which is how `ears` replaced whisper.cpp's
`whisper-cli` without the rest of the script changing, and how `whisper-cli`
goes back in as the fallback. `phone1/README.md` has the measured behaviour:
what the eye wrote on the first night, what it costs, and when it refuses to
open.

## Weights

Not in this tree, and not committed: `senses/ocelli/models` and
`senses/ears/models` are ignored in the repository root, and the phone keeps the
files under `~/models/ocelli` and `~/models/ears`. Both engines take their paths
from the environment, so neither needs a symlink in the tree — an absolute
symlink is what used to sit in `senses/ocelli/models` and it is not what the
repository should carry.

They live on Hugging Face at
[`ataeff/molequla`](https://huggingface.co/ataeff/molequla), in the two folders
this one mirrors:

    ocelli/  yent_eye_ours_q6_k.gguf              the eye's decoder, and q4_0 / q8_0
             yent_eye_smolvlm2_lora_v2_mmproj_q8_0.gguf   its projector, and the f16
    ears/    ggml-tiny.bin  ggml-base.bin         whisper's own multilingual ggml files

`SHA256SUMS.txt` sits beside each. The ears' two files are whisper.cpp's own
downloads, unconverted — `ears` reads the ggml format as it ships, which is why
there is no conversion step and therefore no Python anywhere in the project. The
eye's are the Yent eye: SmolVLM2-500M with a LoRA merged in, quantised for the
phone.

## Licence

GPL-3.0, like the rest of the tree; each folder keeps the LICENSE it arrived
with.
