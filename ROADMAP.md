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
- **The senses on their own schedule** (`claude/phone1-senses`). `senses.sh`
  is one pass of eye, ears and place, and the scheduler gained a second kind of
  slot: nine senses slots a day in the gaps between the colony windows, capped
  at 600 s, skipped and logged if they meet a live colony. The organisms are
  told to eat the result with `--dna-extra-sources world,sound,place`, and since
  the routing repairs of 2026-09-15 so is the witness. Waiting
  on the merge: the daemon still runs the old conf and the binary in
  `$MOLEQULA_RUN` predates the flag.
- **A ledger of change, not a stream of facts** (`claude/phone1-world-ledger`,
  2026-09-15). The bitemporal `world_facts(source, subject, predicate, object,
  valid_from, valid_to, recorded_at, provenance)` is in mesh.db; `senses.sh`
  writes one JSON line per observation beside every fragment; `molequla
  --world-ingest` files them, closes what a newer observation contradicts and
  leaves only the changes in `dna/output/world/` — the phone moved from A to B,
  the sky changed from fog to overcast, a person entered the front camera's
  frame, speech was heard for 12 s. Organisms eat the changes of the world and
  not its state. The writer is its own process because the witness's mesh
  handle is `query_only` and stays that way; the witness reads the table and
  prints `world <n> facts/<m> open`. The shape is borrowed from Utopia
  (deeplethe/utopia); the product itself stays outside. Numbers and the
  red→green runs in `MOLEQULALOG2.md`. Still open: the eye and the ears write
  one observation per pass rather than the short trajectory of §2, the
  `sees_person` reading is lexical and will miss what a sentence does not name,
  and routing — which organism eats which change, §12's cafeteria — is a wave
  of its own.

## Next

1. **First scheduled sessions and the numbers they leave.** Per session: stage
   per organism, peak RSS, `[dna] wrote … gen= fade=` share of generated speech
   in each fragment, overload thresholds re-read on the new masked-CE loss
   (repair 2b raised it by about one nat; the `CFG` thresholds were tuned on the
   old one and are not to be turned blind).
2. **Voice sweep on adults.** temperature × top_k × at least four prompts on a
   stage-4 checkpoint before any verdict on coherence.
3. **The eye inside the session, and the byte gate that counts it.** The organ
   itself landed on `claude/phone1-senses`, but on the opposite schedule to the
   one written here (Oleg, 2026-09-13: the senses are gatherers, they run on
   their own): `phone1/senses.sh` takes both cameras through
   `senses/ocelli/eye` in senses slots *between* the colony windows, not beside
   the witness inside one, and `--dna-extra-sources world,sound,place` is the
   switch `CFG.DNAExtraSources` was waiting for. What is still open is the
   session case: if the eye is ever to run while organisms are alive, its
   measured peak (1020 MB for 14-16 s per frame with the q8_0 projector, 951 MB
   with f16) has to enter the growth byte gate, which today knows nothing about
   it — hence the crude fence, a MemAvailable floor plus the colony-window
   skip. Still open too: the sentence entering the corpus at every age but
   cross-graze only at `fade = 1` / stage ≥ teen. Age is `global_step` and the
   voice is `gen_mag` / `overlay_fade`, all three columns in mesh.db since the
   routing repairs of 2026-09-15, so the gate has its inputs and what is left
   is the gate itself — audit §4 item 7.
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
7. **A model that names sounds rather than classifying their shape.** The
   detector half of this item is closed below: `senses/ears/soundscape` says what
   kind of sound twelve seconds were, without weights. What it cannot say is what
   made the sound — "a car passed nearby", "a door closed" — and that needs an
   AudioSet-vocabulary tagger on notorch. The survey with the measured weight
   sizes and the ops notorch is missing is `senses/ears/PORT_NOTES_SOUND.md`:
   YAMNet first (4 126 810 B of TFLite, depthwise conv2d and pooling are what
   notorch lacks, and the mel bank has to arrive as a binary rather than be
   generated), PANNs CNN10 second if weights can be found, AST and BEATs not
   while the eye holds a gigabyte in the same slot. Also still open: the encoder
   gap, `senses/ears/EARSLOG.md` measures whisper.cpp 3.4-4.3× faster on the same
   wav and names the reason (its threaded REPACK kernel against a single
   `nt_qmatmul` per projection), which is a notorch debt below as much as an ears
   one.
8. **A voice out.** Senses run both ways, like the VLM: not only circulation in,
   but the mycelium speaking. First step costs nothing — the witness's line
   through Android TTS (`termux-tts-speak`; the package is disabled on phone-1
   and re-enabled with `pm enable com.google.android.tts`); the real step is a
   TTS on notorch, its own port.
9. **Place — the world that does not depend on you** (Oleg, 2026-09-13). Landed
   on `claude/phone1-senses` as the third organ of `senses.sh` rather than a
   `place.sh` of its own: `termux-location` (13 m by network here), open-meteo
   for temperature, humidity, wind, the WMO code as a word and today's sun,
   Nominatim reverse for the name, `curl` and `jq` and no library, one fragment
   per pass into `dna/output/place/`, and a haversine against
   `senses/place.last` that says whether the phone moved more than 50 m. Still
   open: headlines for the region, and the point of the whole item — that this
   is subjectivity, not a dashboard, which only shows once the ledger above
   turns these facts into changes.

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

- **Routing repairs 1, 2, 3 and 6** (`claude/phone1-routing-food`, 2026-09-15). The
  first four items of the audit of `molequla_new_logic.md`
  (`reports/2026-09-15_new_logic_audit/README.md` §4) that are routing rather than
  infrastructure: the witness is told about the senses, the extra sources read
  under their own per-tick budget, an eaten fragment enters the corpus cut into
  sentences instead of truncated at 240 bytes, and the organism's voice — the raw
  transformer magnitude and the overlay fade — rides the heartbeat into `mesh.db`
  beside `global_step`. Items 4, 5, 7, 8, 9 and 10 of that list stay open; the
  measurements are in `MOLEQULALOG2.md`.

- **The sensing window, and hearing that is not only speech**
  (`claude/phone1-sensing-window`, 2026-09-15; molequla_new_logic.md §2, §3 and
  §18 steps 1-5). An eye pass is a window of `SENSES_EYE_WINDOW` frames taken
  `SENSES_EYE_SPACING` apart with the cameras cycled from `SENSES_EYE_PATTERN`,
  and one summary line per window carries its novelty — the share of descriptions
  that did not repeat an earlier frame, by token overlap. The cadence is not
  frozen by decree: n ∈ {1, 2, 4} was measured twice each on cores 4-7 (15-18 s,
  46-49 s, 104-107 s, peak RSS 1020 MB throughout, novelty 1.000 / 1.000 / 0.750)
  and the default n=4, spacing 30 s comes out of that table. Hearing gained
  `senses/ears/soundscape`: no weights, `nt_stft`, seven labels from measured
  features, an `[ears env …]` fragment on every pass whether or not anybody spoke,
  and the recognizer's own non-speech tags kept instead of stripped. Gates:
  `phone1/senses_test.sh` 20 cases (red first: 0 pass, 20 fail),
  `make test-soundscape` 7 fixtures (red twice by moving a threshold). Numbers in
  `MOLEQULALOG2.md` and `senses/ears/EARSLOG.md`; the tagger port stays open above
  as item 7.

- **The senses gathered into one folder** (`claude/phone1-senses-tree`,
  2026-09-13). Oleg's decision: the organs live inside molequla, in `senses/`,
  and neither gets a repository of its own. The eye came across from the
  gitignored `reffs/ocelli` and the ears from the local `arianna/ears` checkout,
  both by `git subtree add`, so both arrive with their own commits behind them —
  eleven for ocelli up to `c4fe095`, two for ears up to `047a141`. `senses.sh`
  now runs `senses/ocelli/eye` and, for the first time, molequla's own
  recognizer instead of whisper.cpp's binary. Numbers in `MOLEQULALOG2.md`.
- **Ears — whisper on notorch, parity with ggerganov** (2026-09-13). The organ is
  `senses/ears`: the ggml `.bin` reader, a log-mel front end written from scratch
  because notorch has no FFT, and encoder, decoder and tokenizer on notorch
  primitives. No Python in the build, the tests or the conversion path — there is
  no conversion path. Gated token for token against whisper.cpp under pure greedy
  on six rows (tiny and base × `jfk`, `speech_air_14s`, `ambient_8s`), plus mel,
  encoder and speed gates; `senses/ears/EARSLOG.md` holds the runs, including the
  slot bug found in whisper.cpp's own no-speech probability. Wired into the field
  as the default of `SENSES_ASR` / `SENSES_ASR_MODEL`, with whisper-cli kept
  behind the same two variables.
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
