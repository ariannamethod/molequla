# phone1 — running the colony on the A56

**The colony never runs non-stop.** It runs in capped sessions and is down in
between, the way it has always been deployed — the old cascade ran four elements
for 30 minutes, the pod runs carried a 35-minute cap. The one time that rule was
skipped, on 2026-09-13, all four organisms reached stage 4 within 22 minutes,
each holding 758-928 MB, and Android's lmkd killed Termux to get the memory
back. The default since: three sessions a day of two hours each, at 04:00, 12:00
and 20:00 UTC, started by `phone1/schedule.sh`, ended by the daemon's own
wall-clock deadline, with the `timeout` inside `launch.sh` behind it as a
backstop for the case where the scheduler dies first.

The senses are the exception to that rule, and the reason it survives: camera,
microphone and place run in their own short slots between the colony windows —
01:00, 03:00, 07:00, 09:00, 11:00, 15:00, 17:00, 19:00 and 23:00 UTC, capped at
600 s — so that fragments keep arriving in the DNA field while the organisms are
down, and an organism waking at 04:00 finds eight hours of the world already
waiting as food.

Six scripts around one environment variable, `MOLEQULA_RUN`, which defaults to
`/data/data/com.termux/files/home/arianna/molequla-run` and holds the whole run:
the binary `molequla_cgo`, one directory per organism (`earth air water fire`)
with its own corpus copy, `witness/`, `dna/output/`, `pids/`, `daily/`,
`senses/` and the one-line `BUILD` record.

`bash phone1/build.sh` builds the checkout the script lives in with the recipe
measured on this phone (`CGO_ENABLED=1`, `-O3 -march=native -mtune=native
-DUSE_BLAS`, `-a -trimpath -buildvcs=false`) into `$MOLEQULA_RUN/molequla_cgo`
and writes the commit and the UTC date into `$MOLEQULA_RUN/BUILD`. The C compile
prints a harmless `calloc` warning from `ariannamethod.c`.

`bash phone1/launch.sh` creates the layout if it is missing, copies each
`nonames_<element>.txt` only when the organism has none — a live corpus is never
overwritten — takes a Termux wake lock over ssh (a warning and a start without it
if that fails), and starts five detached processes, each `setsid nohup … </dev/null`
with its own session, so they outlive the shell that launched them: the four
organisms on the big cores as `taskset -c 4-7 molequla_cgo --organism-id <e>
--element <e> --evolution --cross-graze --corpus-overlay`, and five seconds later
the witness on the little ones as `taskset -c 0-3 molequla_cgo --witness
--witness-interval 5`. It also creates `dna/output/{world,sound,place}` and
passes `--dna-extra-sources world,sound,place`, so the organisms eat what the
senses left there. Each pid lands in `pids/<name>.pid`; a name whose pid file
points at a live process is refused, not started twice. An optional first argument
is a duration in seconds — `bash phone1/launch.sh 120` wraps every start in
`timeout 120` for dry runs.

`bash phone1/stop.sh` sends SIGTERM to every process group named in `pids/`,
waits up to 20 s, reports who exited, SIGKILLs the survivors, removes the pid
files of the dead and releases the wake lock.

`bash phone1/schedule.sh start|stop|status|next` is the daemon that keeps the
rule. It reads `phone1/schedule.conf` — UTC start times, session length, grace,
sampling interval, catch-up window, `oom_score_adj`, and the senses slots with
their own command and cap — where every name can be overridden from the
environment and the file itself with `SCHEDULE_CONF`. Two kinds of slot share
one clock: `SCHEDULE_SLOTS` runs the colony for `SCHEDULE_DUR`, `SENSES_SLOTS`
runs `SENSES_CMD` (`senses.sh all`) under a `SENSES_TIMEOUT` cap counted in
wall-clock seconds. `next`
names the nearest slot of either kind and which kind it is; `next --kind` and
`in-window` exist for the gate. A senses slot that falls inside a colony
window, or finds the colony alive from a manual launch, is skipped and logged
as `reason=skipped-colony-window` or `skipped-colony-alive` — the eye alone
holds a gigabyte, and four organisms holding 758-928 MB each is how this phone
lost Termux to lmkd once already. A pass that could still be running when the
organisms are due does not start either: a colony window nearer than the cap
itself gives `reason=skipped-colony-soon`.
`SENSES_SLOTS` empty turns all of this off and leaves the colony scheduler that
was here before. It
sleeps to the next slot in naps of at most a minute, each decided against the
wall clock because a long `sleep` does not count the time a phone spends
suspended; runs `launch.sh $SCHEDULE_DUR`; writes `SCHEDULE_OOM_ADJ` (500) into
the organisms' `oom_score_adj`, so that a memory squeeze costs a checkpointed
organism instead of the whole terminal; samples the colony's memory every
30 s; and when the cap has passed calls `stop.sh` on every path, which clears
the pid files and releases the wake lock. One line per session goes into
`$MOLEQULA_RUN/schedule.log` with `kind=colony`, the slot, the start and end,
the reason (`capped`, `early-exit`, `overran`, `skipped-running`,
`launch-failed`, `missed`), MemAvailable before and after, the
per-organism VmHWM peak, and the colony's simultaneous footprint; a senses
slot writes the same line with
`kind=senses`, its reason (`ok`, `timeout`, `failed-rc<N>`,
`skipped-colony-window`, `skipped-colony-alive`, `skipped-colony-soon`), the
fragments the pass reported, `suspend_s=` — the seconds of the pass that
CLOCK_MONOTONIC did not count, which is the difference between an eye that was
slow and a phone that was asleep — and `strays=`, how many processes the pass
still had running when it was over; the
daemon's own console is `$MOLEQULA_RUN/schedule.out`, and the raw output of the
last capped pass is `$MOLEQULA_RUN/schedule.cap.out`. It is detached
(`setsid nohup`, pid in `pids/schedule.pid`, which `stop.sh` skips), refuses a
second instance, and a slot whose colony is already up — a manual `launch.sh` —
is logged as `skipped-running` and left alone. A slot reached more than
`SCHEDULE_CATCHUP` (1800 s) late is not run at all: after a suspend or a
reboot, a session that starts hours late is not the session that was scheduled.
`/usr/local/bin/defender-services.sh` calls `schedule.sh start` so the daemon
comes back after a reboot.

**A cap on this phone is counted against the wall clock, never by `timeout`.**
`timeout` arms its cap with `alarm(2)` — `readelf --dyn-syms /usr/bin/timeout`
on coreutils 9.4 imports `alarm@GLIBC_2.17` and no `timer_create` — and that is
`ITIMER_REAL`, an hrtimer on `CLOCK_MONOTONIC`, a clock that stops while the
phone is suspended: this boot carries 289 341 s of `CLOCK_BOOTTIME` against
225 488 s of `CLOCK_MONOTONIC`. On 2026-09-17 the 11:00 senses slot ran 8176 s
under `timeout 600` and came back `reason=ok` — the 600 s of that clock never
elapsed — and the 12:00 colony window was gone. So the senses cap is a watchdog
that reads `date`, takes one-second naps, and when the wall clock says the cap
is spent sends SIGTERM and then SIGKILL to the pass's whole process group: the
pass runs under `setsid`, so its ssh and its eye are in that group and go with
it. Its output goes to a file and never through `out="$(…)"`, because a command
substitution does not return until the last grandchild closes the pipe — that,
not the work, is what kept the daemon inside `run_senses` for all 8176 s.
The group is swept on the clean path too: nothing outlives a pass. The
`timeout` inside `launch.sh` has the same monotonic weakness and is kept for
what it is, a backstop for the case where the daemon itself dies; the cap that
actually ends a colony session is the daemon's own wall-clock deadline and
`stop.sh`, which is what ends every session in `schedule.log`.

A pass that overruns anyway no longer swallows the next slot in silence. When a
slot returns, the daemon asks which slots of either kind opened while it ran: a
senses slot inside a colony window is not one of them, the newest of the rest is
run late if it is still inside `SCHEDULE_CATCHUP`, and everything else is
written into `schedule.log` as `reason=overran-into` with `by=<kind>@<slot>` —
the slot that took it.

The session line carries two different memory readings and they answer two
different questions. `hwm_mb=earth:1691,air:1185,…` is each organism's `VmHWM`,
a per-process lifetime high-water mark: the largest that organism ever was,
whenever that was. `rss_sum_max_mb=N at HH:MM:SSZ mem_min_mb=M` is the colony
as one thing: `N` is the largest sum of *current* resident sets the sampler
ever saw across whatever was alive at that instant, `HH:MM:SSZ` is when it saw
it, and `M` is the least MemAvailable of the session. Four high-water marks
reached at four different moments cannot say whether four peaks ever stood
together, which is exactly what `SerialBursts` was landed to change — after it,
per-organism peaks rose while the colony never fell below 1 GB free
(`MOLEQULALOG2.md`, 2026-09-15, "the third session"). `rss_sum_max_mb` is the
number to compare session against session to see whether serialising the bursts
bought anything, and it is the left-hand side of the §4.2 sleep arithmetic in
`docs/resonator_design.md`. A session that never saw a live organism prints
`rss_sum_max_mb=-` rather than a colony of 0 MB.

`SCHEDULE_PREKILL` (default `android am kill-all`) is run in the real Android
environment immediately before a colony session, and its MemAvailable before
and after go into the session line as `prekill_mb=A->B`. Android's cached bin
looks free in its own accounting and is not: dropping it on 2026-09-15 took 595
MB of cached app PSS, moved MemAvailable 3 812 → 4 130 MB at once and left it at
4 025 MB eleven minutes later — **+213 MB sustained** — with fifteen of the
nineteen killed processes never coming back
(`reports/2026-09-15_phone1_android_memory/README.md`, §4). Only the `cch` bin
is touched, so the red line is safe by construction: at the moment of that
measurement the keyboard was Perceptible, Termux Foreground, Tailscale Visible,
telephony and Bluetooth Persistent. The senses slots deliberately do not get
this. Their burst is 955 MB of anonymous memory held for forty seconds (§3.1),
it fit in free memory without pushing one page into swap, the eye reads
MemAvailable itself before it opens, and everything killed for a two-hundred-
second pass would only be paged back in before the colony window an hour later;
the cost — cold app starts and the media indexers rescanning — buys nothing
there and buys the whole difference before four organisms that hold 758-1091 MB
each for two hours. Empty the value and nothing runs. A prekill that is missing
or exits non-zero never costs the slot: it is logged, `prekill_mb` carries the
return code, and `launch.sh` starts anyway.

`bash phone1/senses.sh [eye|ears|place|all]` is the other thing the schedule
runs, and the reason the phone has organs at all. The colony is down sixteen
hours a day; the senses are not. One pass takes a short window of camera frames,
twelve seconds from the microphone and one fix of where the phone is, and leaves
what it found as fragments in `$MOLEQULA_RUN/dna/output/world/`, `sound/` and
`place/` — the same `gen_<unix>_<seq>.txt` names the organisms order their
reading by, each fragment one sentence behind a bracketed header
(`[eye cam0 2026-09-13T22:12:01Z] A blurry kitchen table shows a green bowl…`)
that reads as plain text to whatever eats it. The sequence number only ever
grows, so two fragments written in the same second still have an order.

What the first night's passes looked like, for scale: the back camera on a
dark balcony wrote «A close-up view shows a keyboard with white keys and a dark
background», and there was a keyboard. On a lit table it wrote «a green bowl,
a spoon, and a plate», and there were. The front camera, facing tiled walls
and a towel hung up to dry in what the owner of the balcony calls terrible
light, wrote «A bathroom with a shower curtain hanging» — tile and hanging
cloth, named with the nearest word a 500M model has. Nothing is filtered:
the fragment is what the eye believed, and the organisms eat beliefs. The
correction, when it comes, comes from the other senses and from time
(ROADMAP item 10), not from a filter in this script.

A pass of the eye is a window, not a sample: `SENSES_EYE_WINDOW` frames (4) taken
`SENSES_EYE_SPACING` seconds apart (30), the cameras cycled from
`SENSES_EYE_PATTERN` (`0 1 0 0` — rear, front, rear, rear), one fragment per
sentence as before and one summary line per window. That line carries the
pattern, the spacing, how many descriptions came back, how many of them repeated
an earlier frame of the same window and the novelty that leaves, the wall time,
the peak RSS and the battery before and after. A repeat is a token overlap of
`SENSES_EYE_SAME` (0.8) or more against an earlier frame — lowercased,
non-alphanumerics as separators, intersection over union of the distinct tokens;
`senses.sh overlap "<a>" "<b>"` prints the number and the gate drives that same
code. Measured 2026-09-15 on cores 4-7, two runs each: n=1 15-18 s, n=2 46-49 s,
n=4 104-107 s, peak RSS 1020 MB whatever n is, novelty 1.000 at n=1 and n=2 and
0.750 at n=4 both times. n=4 is the default because at n=2 the two frames come
from different cameras and can only be new; n=4 puts two rear frames a minute
apart, and in both runs one of them repeated — the window seeing that the scene
held still. The memory floor is re-read before every frame, so a window stops
where MemAvailable falls short instead of failing the slot.

The eye is `senses/ocelli/eye`, the pure-C SmolVLM2-500M engine, on the q6_k Yent
decoder with one global frame (`SMOLVLM_NOSPLIT=1`): 14-16 s and a peak of
1020 MB per frame on this phone, measured with `/usr/bin/time -v`. It lives in
the tree now (`senses/README.md`) but nothing links against it — it is a separate
process with its own notorch, and `SENSES_EYE` is the one variable that names the
wrapper. Its weights are not in the tree either: `SENSES_EYE_MODEL` and
`SENSES_EYE_MMPROJ` are passed to the wrapper as `EYE_MODEL` / `EYE_MMPROJ` and
point into `~/models/ocelli`. The camera's 4080×3060 jpeg is scaled to a 1024 px longest edge before
the engine sees it, because the engine's first act is to resize to 2048 and a
12 MP frame would be decoded into 150 MB of float on the way. Below 1300 MB of
MemAvailable the eye does not open at all and the pass says so
(`eye=…,skip-mem:1204`).

The ears are `senses/ears/ears`, molequla's own recognizer — whisper on notorch,
in C, gated token for token against whisper.cpp on six rows
(`senses/ears/EARSLOG.md`) — on the `tiny` weights at `~/models/ears/`, run as
`-l auto -t 4 --no-speech-thold 0.6`. `SENSES_ASR` and `SENSES_ASR_MODEL` are
still the two variables, and pointing `SENSES_ASR` at whisper.cpp's `whisper-cli`
puts the old path back: the command line switches with the binary's name, since
`ears` takes its model and wav positionally and `whisper-cli` through `-m` and
`-f` (`SENSES_ASR_KIND` forces the choice when the binary is named something
else). Room noise does not become a sentence: `ears` drops a whole window on its
own no-speech probability and prints nothing, and on top of that bracketed tags
are stripped — `base` under whisper.cpp once spent 185 s on eight seconds of
ambience and emitted `[Motor]` (`~/arianna/ears-reference/REFERENCE.md`) — and
what is left must still be eight characters with a letter in it before a `mic`
fragment is written. A twelve-second pass that does hear something costs about
22 s end to end on cores 4-7 (measured 2026-09-13:
`ears=rc0,22s,speechyes,frags1`).

A quiet twelve seconds no longer produces nothing. `senses/ears/soundscape` —
the same folder, no weights, 0.057 s on a 12 s wav — reads that wav after the
recognizer and writes one English line about what kind of sound it was as an
`[ears env …]` fragment, every pass, speech or not: quiet room, a single loud
transient, music is audible, speech-like modulation words unclear, repeated
mechanical noise, steady broadband noise, or an unsteady sound without clear
structure. The recognizer's bracketed tags go into that fragment instead of the
bin (`[ears env …] Quiet room. The recognizer also marked [BLANK_AUDIO].`).
`SENSES_SOUNDSCAPE` names the binary and emptying it puts the speech-only pass
back; every threshold inside it is an `SND_*` environment variable with a
measured default (`senses/ears/EARSLOG.md`, 2026-09-15). The two fragments are
separate evidence about the same twelve seconds and are not expected to agree.
Beside the fragment the pass also writes one fact into `senses/facts.jsonl`,
`ears microphone soundscape "<line>"`, so the world ledger can tell a room that
went quiet from a room that was always quiet. The predicate is `soundscape` and
not `hearing`: what the recording sounded like is a different claim from whether
anybody spoke in it, and the ledger holds both about one window.

Place is `termux-location` (network first, satellites if that fails), then two
keyless APIs over `curl` and `jq`: open-meteo for temperature, humidity, wind,
the WMO weather code as a word, and today's sunrise and sunset; nominatim,
reverse, zoom 14, with a User-Agent and English names. The last fix is kept in
`senses/place.last` and the fragment says whether the phone has moved more than
50 m since the previous pass.

Every pass appends one line to `$MOLEQULA_RUN/senses/senses.log` with the cores
it used, MemAvailable before and after, and per organ the exit code, the wall
time, the fragments written, plus the eye's peak RSS and novelty and the
describer's label; a pass that opened the eye appends a second, `eyewin` line for
the window itself. Frames and wavs are kept
under `senses/frames/` and `senses/audio/`, the newest 48 of each; the fragment
directories keep the newest 64. Nothing is pruned by age — an organism's cursor
only advances while it runs, and it may have slept through eight hours of
passes. One pass at a time: `senses/.lock` holds the pid and a stale lock is
cleared.

The organisms are told about those three directories with
`--dna-extra-sources world,sound,place`, which `launch.sh` passes and which
sets `CFG.DNAExtraSources`. One flag feeds two paths: `dnaRead` appends the
fragments to the organism's own corpus, and `NewCrossField` takes its sibling
list from the same `dnaSources()`, so the senses also reach the logit overlay.
Without the flag the directories exist and are simply never read.

`bash phone1/schedule_test.sh` is the gate: 46 cases through the real
`schedule.sh`. Thirty-six are the slot arithmetic, with a fake now. Twenty-one
drive
`next --epoch` on colony slots — before, at and after a boundary, across
midnight and across a month, unsorted lists, single slots, base-ten hours, a
non-UTC host time zone, and five malformed configurations that must be refused.
The other fifteen are the two kinds together: eight drive `next --epoch` and
`next --kind` on interleaved colony and senses lists, including the tie and the
empty senses list, and seven drive `schedule.sh in-window`, the predicate that
keeps the senses out of a colony session — its opening moment, its closing
moment, and a session long enough to reach past midnight into the next day.
The last ten are the prekill, and they run a whole slot: a stub `android` on
`PATH` records its argv while stub `launch.sh` and `stop.sh` sit beside a
symlink to the real `schedule.sh`, and `schedule.sh __slot colony|senses`
drives one session with no daemon around it. A colony slot must call the stub
exactly once, with `am kill-all`, before `launch.sh` and not after, and write
`prekill_mb=A->B`; a senses slot must not call it at all; an empty
`SCHEDULE_PREKILL` in the conf must not call it and must still run the session;
and a stub that exits 3, or a command that does not exist, must leave the
session running and the failure in the console.

`bash phone1/senses_test.sh` is the gate for the pass itself: 20 cases through
the real `senses.sh` with the hardware faked and nothing else — an `ssh` that
copies fixture files where `termux-camera-photo` and `termux-microphone-record`
would have written them, a scripted `eye`, a scripted recognizer and a scripted
describer; `ffmpeg` is real, because the frame path runs through it. Ten cases
are the window: a window of four leaving four fragments, the cameras in the
pattern's order both at capture and in the headers, a pattern shorter than the
window cycling, the summary line's size, pattern, repeat count and novelty, and
the spacing holding a capture back. Five are the overlap metric — identical,
disjoint, punctuation and case, empty against text, and one noun changed in a
long sentence, which must stay above the 0.8 that counts as a repeat. Five are
hearing: silence still leaving an `[ears env …]` fragment, speech leaving both
`env` and `mic`, a bare `[Motor]` not counting as a transcript, and the tag
surviving into the environmental line.

`bash phone1/status.sh` prints one screen: per organism the pid and whether it is
alive, VmRSS and VmHWM in MB, the stage and ingested count from the last
`[debug-onto]` line, the NaN line count, the corpus size in bytes and lines, the
number of DNA fragments it has written; then the witness's last stdout line and a
fresh `molequla_cgo --witness --once` reduced to organism count, field entropy,
action and alerts.

`bash phone1/daily.sh` appends a dated, timed section to
`$MOLEQULA_RUN/daily/<UTC date>.md`: the status screen, per-organism DNA traffic
(`wrote` lines and bytes, `consumed` bytes) over the whole stdout history — launches append, so a restart keeps it —
the cafeteria table (passes, admitted by reason, declined, measured, decline
share) summed from the `[cafeteria]` lines since each organism's last
`[ecology] Element:` banner, the newest three `[dna] … wrote` lines verbatim, the witness's last five lines,
and `df -h` of the run root. Run it as often as you like; every run adds a
section.
