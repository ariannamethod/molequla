# molequla — roadmap

The third book. `README.md` says what molequla is, `MOLEQULALOG2.md` says what was
done and how it was measured; this file says what comes next and in what order. It
is a dated log like the diary: an item enters with the date it was agreed, moves
between sections when its state changes, and leaves with the commit that closed it.
Nothing here is a promise to a reader; it is the working order of the people and
nodes in the tree. When the roadmap and the code disagree, the code wins and the
roadmap is corrected.

## How the colony runs

molequla runs in capped sessions, never as a permanent process. On Railway it ran
a few hours a day and slept; in the GitHub cascade it ran thirty to sixty minutes a
day after haiku, penelope and klaus and before nanojanus. On phone-1 the same rule
holds: `phone1/schedule.sh` starts the colony for a fixed session, `timeout` ends
it, the organism saves its checkpoint on the signal, and the next session resumes
from it. Between sessions there are zero molequla processes on the phone. The game
is long — an organism lives for months of sessions, not for one uptime.

## In flight (2026-09-13)

- **Growth memory budget** (`claude/phone1-repair-growth-budget`). A byte gate
  before ontogenesis, like the mitosis gate of repair 4: growth waits until
  `MemAvailable` covers the organism's projected peak, deferred growth is retried
  on later ticks, and growth plus its warmup is serialised across the colony so
  four peaks never coincide. Checkpoint written on SIGTERM. `oom_score_adj` set
  from inside the process so that under pressure an organism with a checkpoint
  dies before the terminal that hosts the node.
- **Session scheduler** (`claude/phone1-schedule`). `phone1/schedule.sh`: three
  sessions a day of two hours (04:00, 12:00, 20:00 UTC) by default, one line per
  session in `schedule.log` with memory at start and end and the peak RSS seen,
  restart after reboot through the node's service script.
## Next

1. **First scheduled sessions and the numbers they leave.** Per session: stage
   per organism, peak RSS, `[dna] wrote … gen= fade=` share of generated speech
   in each fragment, overload thresholds re-read on the new masked-CE loss
   (repair 2b raised it by about one nat; the `CFG` thresholds were tuned on the
   old one and are not to be turned blind).
2. **Voice sweep on adults.** temperature × top_k × at least four prompts on a
   stage-4 checkpoint before any verdict on coherence.
3. **The eye as the fifth DNA source, inside the session.** `ocelli`
   (SmolVLM2-500M, Yent eye LoRA v2, six GGUFs at
   `huggingface.co/ataeff/molequla/tree/main/ocelli`) describes a camera frame
   in one sentence. Today it is `reffs/ocelli/eye <image>` by hand, and the
   organism has the slot but not the switch: `CFG.DNAExtraSources` (nil by
   default, "world joins here when the eye writes") has no CLI flag, and no
   script takes a frame. To land: `phone1/eye.sh` — a frame every N minutes
   through Termux (`termux-camera-photo -c 0`), `eye`, one line appended to
   `dna/output/world/`; a `--dna-extra-sources world` flag; the scheduler starts
   the eye beside the witness on the little cores for the length of the session;
   the eye's own peak (951-988 MB for 11-13 s per frame) counted in the byte
   gate. The sentence enters the corpus for every age and cross-graze only at
   `fade = 1` / stage ≥ teen. Age is `global_step`, already a column in mesh.db.
4. **Other VLMs for the same organ.** Alternative eyes measured on the phone
   with the same table (tok/s, wall, peak RSS, one global frame) and stored in
   sibling folders of the weights repo.
5. **loragrad as immune filter.** The gradient verdict (PASS / WEAKEN / FREEZE /
   SCAR / DARK / SILENCE) applied to an eaten DNA fragment inside the organism,
   between `dnaRead` and the burst; not in the witness. Own repair after the
   measurements above.
6. **Replacement instead of growth under a full machine.** When the byte budget
   refuses growth for long, an organism yields to a sibling rather than waiting
   forever; needs a channel between organisms that does not exist yet.
7. **Ears — whisper on notorch, parity with ggerganov** (Oleg, 2026-09-13: "if
   there is sight there must be hearing"). The reference exists since
   2026-09-13: whisper.cpp `1da4dc8` built with OpenBLAS at
   `~/arianna/whisper.cpp`, tiny and base multilingual ggml weights, three wavs
   (`jfk`, the same sentence played through the phone speaker and re-recorded
   by the microphone, eight seconds of room), transcripts, token JSON, timings
   and the full tensor layout of both models under `~/arianna/ears-reference/`.
   On cores 4-7 with four threads: jfk tiny 10.5 s / 178 MB peak, base 19.9 s /
   287 MB; the re-recorded sentence comes back on both models. Room noise on
   base ran 185 s through temperature fallbacks before settling on `[Motor]`,
   so a no-speech gate goes in early. The port: an `ears` organ in C on the
   canon notorch, loading the ggml `.bin` directly (whisper is not GGUF), with
   the log-mel front end written from scratch (notorch has no FFT) and
   everything after it on existing primitives; gate: byte-equal transcript and
   token ids against whisper.cpp on the three wavs. Input from the microphone
   into `dna/output/sound/` as a sixth source. Not only speech: a plain
   sound-event detector in C ahead of the model turns a bang, a door, a voice
   into one line even when no words are said; a model that names sounds, not
   just words, comes later.
8. **A voice out.** Senses run both ways, like the VLM: not only circulation in,
   but the mycelium speaking. First step costs nothing — the witness's line
   through Android TTS (`termux-tts-speak`; the package is disabled on phone-1
   and re-enabled with `pm enable com.google.android.tts`); the real step is a
   TTS on notorch, its own port.
9. **Place — the world that does not depend on you** (Oleg, 2026-09-13). The
   phone has GPS (`termux-location`: 13 m by network, 23 m by satellite once
   `ACCESS_FINE_LOCATION` is granted to termux-api). A seventh source,
   `dna/output/place/`, written by `phone1/place.sh` in bash with `curl` and
   `jq` and no library: position, weather and sun from open-meteo, the name of
   the place from Nominatim, and later headlines for the region. One line of
   facts per tick. Not a dashboard, not a Palantir: the point is subjectivity —
   here is the outside, and it moves without the organism.
10. **A ledger of change, not a stream of facts.** What makes the senses one
    field: a bitemporal table `world_facts(source, subject, predicate, object,
    valid_from, valid_to, recorded_at, provenance)` in mesh.db, written by
    place, eye and ears; the witness closes a fact when a newer one contradicts
    it and emits the change as a line into `dna/output/world/` — the fog
    lifted, the phone moved 300 m, a person entered the frame, English speech
    for eleven seconds. Organisms eat the changes of the world, not its state.
    The shape is borrowed from Utopia (deeplethe/utopia: facts carry when they
    held and when they were learned, corrections close rather than erase, an
    append-only decision ledger); the product itself stays outside — later, a
    mirror on polygon fed from the same table, never an organ.

## Later

- **A coherence organ instead of the static overlay.** Netta (AlphaZero-style
  self-play over the language model) speaking into the DNA field in place of
  embryonic salad and fading on the same ramp the overlay fades on; the bridge
  between the incoherent early stages and the transformer. After the first
  scheduled sessions and after the verdict on Netta's Body 1.
- **Two more ideas from Oleg**, told after the eye lands. Not written here yet.
- **CUDA build linked** once a machine has the libraries. `go build -tags cuda`
  already compiles and vets the whole GPU lane on the phone — `go build -tags
  cuda ./modules/gpu` exits 0 — and stops at the link with `cannot find
  -lnotorch_gpu -lcudart -lcublas`. What is left is that link and the run:
  `libnotorch_gpu` built from `modules/gpu/csrc/`, then the parallel-GPU
  training path measured again.

## Debt (notorch, seen from molequla)

- Chuck auto-freeze is permanent once triggered.
- `destroy` after `clear` leaks the tape entry.
- Broadcast backward for `mul` / `add` is missing.
- `assert T == BlockSize` in the sequence path.

## Closed

- Repairs 1-8 (2026-09-13): notorch frozen slot, `wpe` on the tape, DNA as a
  field, governor bytes and heartbeat, corpus and DNA caps, cross-graze under the
  overlay and the overlay fade, the witness in Go and the end of the Python tier,
  the tree fitted to the phone and the README read against the code. Entries with
  numbers in `MOLEQULALOG2.md`; commits `e1a820d` … `5879dda` on main.
- Launch scripts and the emission line (`#38` `e7ccbe6`, `#39` `673f478`).
- **GPU Go code out of the root** (`claude/phone1-gpu-modules`). The cuBLAS
  bindings, the matvec body and the notorch-GPU switch are the Go package
  `modules/gpu`; the vendored C and CUDA sources moved one level down to
  `modules/gpu/csrc/`, since cgo compiles every `.c` file beside a package. The
  root keeps `gpu_bridge.go`, untagged rather than tagged: the package is a
  pure-Go no-op off the CUDA build, so one file replaced the seven and no stub
  was needed. Entry with the numbers in `MOLEQULALOG2.md`.
