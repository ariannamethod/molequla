# phone1 — running the colony on the A56

**The colony never runs non-stop.** It runs in capped sessions and is down in
between, the way it has always been deployed — the old cascade ran four elements
for 30 minutes, the pod runs carried a 35-minute cap. The one time that rule was
skipped, on 2026-09-13, all four organisms reached stage 4 within 22 minutes,
each holding 758-928 MB, and Android's lmkd killed Termux to get the memory
back. The default since: three sessions a day of two hours each, at 04:00, 12:00
and 20:00 UTC, started by `phone1/schedule.sh` and capped by `timeout` inside
`launch.sh` so the cap survives the scheduler's own death.

Five scripts around one environment variable, `MOLEQULA_RUN`, which defaults to
`/data/data/com.termux/files/home/arianna/molequla-run` and holds the whole run:
the binary `molequla_cgo`, one directory per organism (`earth air water fire`)
with its own corpus copy, `witness/`, `dna/output/`, `pids/`, `daily/` and the
one-line `BUILD` record.

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
--witness-interval 5`. Each pid lands in `pids/<name>.pid`; a name whose pid file
points at a live process is refused, not started twice. An optional first argument
is a duration in seconds — `bash phone1/launch.sh 120` wraps every start in
`timeout 120` for dry runs.

`bash phone1/stop.sh` sends SIGTERM to every process group named in `pids/`,
waits up to 20 s, reports who exited, SIGKILLs the survivors, removes the pid
files of the dead and releases the wake lock.

`bash phone1/schedule.sh start|stop|status|next` is the daemon that keeps the
rule. It reads `phone1/schedule.conf` — UTC start times, session length, grace,
sampling interval, catch-up window, `oom_score_adj` — where every name can be
overridden from the environment and the file itself with `SCHEDULE_CONF`. It
sleeps to the next slot in naps of at most a minute, each decided against the
wall clock because a long `sleep` does not count the time a phone spends
suspended; runs `launch.sh $SCHEDULE_DUR`; writes `SCHEDULE_OOM_ADJ` (500) into
the organisms' `oom_score_adj`, so that a memory squeeze costs a checkpointed
organism instead of the whole terminal; samples every organism's `VmHWM` every
30 s; and when the cap has passed calls `stop.sh` on every path, which clears
the pid files and releases the wake lock. One line per session goes into
`$MOLEQULA_RUN/schedule.log` with the slot, the start and end, the reason
(`capped`, `early-exit`, `overran`, `skipped-running`, `launch-failed`,
`missed`), MemAvailable before and after, and the per-organism VmHWM peak; the
daemon's own console is `$MOLEQULA_RUN/schedule.out`. It is detached
(`setsid nohup`, pid in `pids/schedule.pid`, which `stop.sh` skips), refuses a
second instance, and a slot whose colony is already up — a manual `launch.sh` —
is logged as `skipped-running` and left alone. A slot reached more than
`SCHEDULE_CATCHUP` (1800 s) late is not run at all: after a suspend or a
reboot, a session that starts hours late is not the session that was scheduled.
`/usr/local/bin/defender-services.sh` calls `schedule.sh start` so the daemon
comes back after a reboot.

`bash phone1/schedule_test.sh` is the gate for the slot arithmetic: 21 cases
through the real `schedule.sh next --epoch` with a fake now — before, at and
after a boundary, across midnight and across a month, unsorted lists, single
slots, base-ten hours, a non-UTC host time zone, and five malformed
configurations that must be refused.

`bash phone1/status.sh` prints one screen: per organism the pid and whether it is
alive, VmRSS and VmHWM in MB, the stage and ingested count from the last
`[debug-onto]` line, the NaN line count, the corpus size in bytes and lines, the
number of DNA fragments it has written; then the witness's last stdout line and a
fresh `molequla_cgo --witness --once` reduced to organism count, field entropy,
action and alerts.

`bash phone1/daily.sh` appends a dated, timed section to
`$MOLEQULA_RUN/daily/<UTC date>.md`: the status screen, per-organism DNA traffic
(`wrote` lines and bytes, `consumed` bytes) over the whole stdout history — launches append, so a restart keeps it —
the newest three `[dna] … wrote` lines verbatim, the witness's last five lines,
and `df -h` of the run root. Run it as often as you like; every run adds a
section.
