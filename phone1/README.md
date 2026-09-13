# phone1 — running the colony on the A56

Four scripts around one environment variable, `MOLEQULA_RUN`, which defaults to
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

`bash phone1/status.sh` prints one screen: per organism the pid and whether it is
alive, VmRSS and VmHWM in MB, the stage and ingested count from the last
`[debug-onto]` line, the NaN line count, the corpus size in bytes and lines, the
number of DNA fragments it has written; then the witness's last stdout line and a
fresh `molequla_cgo --witness --once` reduced to organism count, field entropy,
action and alerts.

`bash phone1/daily.sh` appends a dated, timed section to
`$MOLEQULA_RUN/daily/<UTC date>.md`: the status screen, per-organism DNA traffic
(`wrote` lines and bytes, `consumed` bytes) since the stdout files were opened,
the newest three `[dna] … wrote` lines verbatim, the witness's last five lines,
and `df -h` of the run root. Run it as often as you like; every run adds a
section.
