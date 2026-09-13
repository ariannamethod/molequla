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
- **GPU Go code out of the root** (`claude/phone1-gpu-modules`). The cuBLAS
  bindings, the matvec body and the notorch-GPU switch move to `modules/gpu/`
  beside the C and CUDA sources they wrap; the root keeps one tagged hook file
  and its stub. The `-tags cuda` side is a mechanical move until a machine with
  nvcc can build it.

## Next

1. **First scheduled sessions and the numbers they leave.** Per session: stage
   per organism, peak RSS, `[dna] wrote … gen= fade=` share of generated speech
   in each fragment, overload thresholds re-read on the new masked-CE loss
   (repair 2b raised it by about one nat; the `CFG` thresholds were tuned on the
   old one and are not to be turned blind).
2. **Voice sweep on adults.** temperature × top_k × at least four prompts on a
   stage-4 checkpoint before any verdict on coherence.
3. **The eye as the fifth DNA source.** `ocelli` (SmolVLM2-500M, Yent eye LoRA
   v2, weights at `huggingface.co/ataeff/molequla`) describes a camera frame;
   the sentence enters the corpus for every age and cross-graze only at
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

## Later

- **A coherence organ instead of the static overlay.** Netta (AlphaZero-style
  self-play over the language model) speaking into the DNA field in place of
  embryonic salad and fading on the same ramp the overlay fades on; the bridge
  between the incoherent early stages and the transformer. After the first
  scheduled sessions and after the verdict on Netta's Body 1.
- **Two more ideas from Oleg**, told after the eye lands. Not written here yet.
- **CUDA build verified** once polygon has a GPU: `go build -tags cuda` against
  `modules/gpu/`, then the parallel-GPU training path measured again.

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
