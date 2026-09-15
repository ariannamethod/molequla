# Naming sounds — what to port next, and what it costs

`soundscape` says what kind of sound a recording was; it cannot say what made it.
"A car passed nearby", "a door closed", "two people speaking" (molequla_new_logic.md
§3) need a model with a vocabulary, and the vocabulary that exists is AudioSet's
527 labels. This file is the survey behind that choice. Nothing is ported yet and
nothing was downloaded; the sizes below are read off the Hugging Face file
listings on 2026-09-15 and the ones that are not are marked as unverified rather
than guessed.

## The candidates

| | what it is | weights, as published | ops it needs | fit here |
|---|---|---|---|---|
| **YAMNet** | MobileNetV1, depthwise-separable 2-D convolutions over a 96×64 log-mel patch (0.96 s), 521 AudioSet classes | `lite-model_yamnet_classification_tflite_1.tflite` **4 126 810 B**, SavedModel tar 18 288 640 B (`hf://models/thelou1s/yamnet`) | conv2d 3×3 stride 1-2, **depthwise** conv2d, batch norm, ReLU, global average pool, one dense layer, sigmoid | the smallest by a wide margin and the only one that fits a phone slot without quantising anything first |
| **PANNs CNN14** | 14-layer VGG-style CNN, 64-mel front end, mAP ≈ 0.43 on AudioSet (paper figure, unverified here) | `model.safetensors` **327 421 748 B** f32 (`hf://models/nicofarr/panns_Cnn14`); fp16 TFLite **161 533 712 B** (`hf://models/litert-community/PANNs-CNN14-AudioSet-LiteRT`) | conv2d 3×3, batch norm, ReLU, average pool, two dense layers | over the 200 MB line as published; the fp16 TFLite is under it but is a second format to read |
| **PANNs CNN10 / CNN6** | the same family, 10 and 6 layers | no Hugging Face repository found in this search; sizes **unverified** | same as CNN14 | would be the natural middle if the weights can be found without a conversion step |
| **AST** | ViT-base over a spectrogram, patch 16×16, 12 layers | `model.safetensors` **346 404 948 B** (`hf://models/MIT/ast-finetuned-audioset-10-10-0.4593`) | patch embedding conv2d, 12 transformer blocks, layer norm, softmax | transformer ops are the ones the ears already run, but 330 MB beside a 1 GB eye on an 8 GB phone is the wrong shape |
| **BEATs** | self-supervised audio transformer, iter3 fine-tuned on AudioSet | no first-party Hugging Face checkpoint found in this search; size **unverified** | as AST, plus a tokenizer for the SSL objective that inference does not need | best published accuracy, worst provenance for a port that must not download 300 MB |

## What notorch already has, and what a port would have to add

Present in the system library (`/usr/local/include/ariannamethod/notorch.h`, the
`claude/conv1d-and-logmel` work): `nt_conv2d` (dense, im2col + GEMM),
`nt_conv1d`, `nt_im2col` / `nt_im2col_1d`, `nt_group_norm`, `nt_attention`,
`nt_hann_window`, `nt_stft`, `nt_logmel`, and the quantized matmul family the
ears decode through.

Missing for YAMNet, smallest first:

1. **Depthwise conv2d.** `nt_conv2d` is dense: one call per channel with `Cin=1`
   is correct and is how a first port should do it — 512 channels means 512 small
   GEMMs, which is slow but measurable. A grouped variant belongs upstream in
   notorch afterwards, not in this folder.
2. **Average / max pooling.** Not in notorch and not worth putting there for one
   caller; twenty lines in the port.
3. **Batch norm.** Folds into the preceding convolution's weight and bias before
   inference — no runtime op at all, and the fold is the same arithmetic the
   ears' own loader does for its scales.
4. **The mel filter bank.** `nt_logmel` deliberately takes the bank from the
   caller, because generating a nominally equivalent one is how a port stops
   matching the model it is porting. Whisper ships its bank inside the ggml file;
   YAMNet and PANNs do not. Two ways out, in order of preference: take the bank
   as a plain binary — the LiteRT PANNs repository publishes exactly that,
   `mel_basis.bin`, 131 328 B, which is 64 × 513 f32 for an n_fft of 1024 — or
   compute the bank in C from the published formula and gate it against a dumped
   reference. No Python either way.
5. **A weight reader.** The ears read ggml `.bin`. YAMNet publishes TFLite (a
   flatbuffer) and a SavedModel (protobuf); PANNs publishes safetensors, whose
   header is JSON followed by raw tensors and is the least work of the three.
   This is the real cost of the port, not the arithmetic.

## The order this suggests

YAMNet first: 4 MB of weights, one architecture family the phone already runs the
pieces of, and a label set that answers the question §3 asks. The reader is the
work; safetensors is the format to read if a safetensors copy of YAMNet's weights
can be had without a conversion step, and the TFLite flatbuffer otherwise.
PANNs CNN10 second, if weights turn up. AST and BEATs are not phone-1 work while
the eye holds a gigabyte in the same slot.

Whatever lands, it does not replace `soundscape`: a tagger that says "Speech,
Vehicle, Music" still says nothing about whether the room was quiet, and the two
answers are different evidence about the same twelve seconds.
