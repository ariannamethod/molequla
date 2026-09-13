# ocelli

Eyes of Yent. A vision-language engine in plain C: one image in, one honest
sentence out — no cloud, no API, no Python at runtime.

Ocelli are the simple side-eyes of arthropods. There are several of them, they
sit off-centre, and they report field and motion rather than sharp detail. That
is exactly what this engine is: a peripheral organ, meant to run on many nodes
at once, each looking its own way.

## What it is

A SmolVLM-class model executed end-to-end on [notorch](https://github.com/ariannamethod/notorch),
our C runtime: SigLIP vision tower → pixel-shuffle connector → llama-family text
decoder, with the visual embeddings spliced into the token stream. GGUF weights
are read directly; f16 tensors stay f16 in RAM and are dequantised lazily.

Two model generations are known to run:

- `SmolVLM-256M-Instruct` — the original target, generic captioning.
- `SmolVLM2-500M-Video-Instruct` with a contract fine-tune — the Yent eye:
  reads on-screen labels and names what is actually on the frame.

The MLP role inside the vision tower is resolved from the tensor shapes in the
file, not from tensor names, because GGUF converters disagree about which of
`ffn_up` / `ffn_down` holds the first projection. Both conventions load.

## Build

```sh
make                # macOS: Accelerate; Linux: OpenBLAS
make BLAS=0         # portable, no BLAS
make asan           # sanitiser build for probes
```

## Run

Weights are not in this repo. Put a model GGUF and its `mmproj` GGUF under
`models/`, then:

```sh
./eye photo.jpg
./eye photo.jpg "What text is visible?"
EYE_MODEL=models/other.gguf EYE_MMPROJ=models/other-mmproj.gguf ./eye photo.jpg
```

`eye` prints the description and the cost of producing it. Prompt evaluation and
generation are timed separately on purpose: a single number would report 84
prompt forwards as a generation rate.

```
OURS: " A terminal window shows a python script running, with a status line and a checkpoint list."
-- prompt 84 tok in 22616 ms (3.7 tok/s) | gen 18 tok in 4783 ms (3.8 tok/s) | peak RSS 1332 MB --
```

## What it sees, and what it does not

Measured on an A18 Pro phone-class chip, 500M weights, f16:

- Scenes and objects are described correctly across faces, screenshots, indoor
  and outdoor frames.
- On-screen labels are read when they are large enough (`node: neo`,
  `yard-camera 1`, `kairos`).
- Small phone-sized text is a blind spot: a dim terminal log photographed on a
  phone screen was described as a car dashboard.
- f16 runs at 3.6–4.1 tok/s with ~1.0–1.4 GB peak RSS; Q8_0 runs at 30–34 tok/s
  but currently expands to f32 in RAM (~1.8–2.5 GB). Descriptions agree between
  the two on most frames and diverge on near-tie tokens.

The engine is not bit-exact against `llama-mtmd-cli` on the same weights: the
vision tower is f32 here and f16/mixed there, so wording diverges on near-ties
while the content agrees.

## Credits

This engine grew out of work on a fork of
[anniebelkin/image_name_quiz](https://github.com/anniebelkin/image_name_quiz) —
the quiz concept and its first implementation are hers. None of that code is
here; the lineage is, and it is worth naming.

Runtime: [notorch](https://github.com/ariannamethod/notorch). Image decoding via
vendored `stb_image`. Reference output for verification comes from
`llama-mtmd-cli`.

## License

GPL-3.0. See LICENSE.
