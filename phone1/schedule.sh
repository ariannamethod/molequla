#!/bin/bash
# The colony never runs non-stop. This daemon keeps it that way: it sleeps
# until the next UTC slot, runs launch.sh with the session cap, confirms
# everything is down afterwards, writes one line about the session into
# $MOLEQULA_RUN/schedule.log, and sleeps to the next slot.
#
# Two kinds of slot share one clock. A colony slot runs the organisms for
# SCHEDULE_DUR; a senses slot runs phone1/senses.sh — camera, microphone,
# place — for at most SENSES_TIMEOUT, so that fragments keep arriving in the
# DNA field while the organisms are down. The senses never run inside a colony
# window: memory is the phone's scarcest thing and the eye alone holds a
# gigabyte. Senses slots are configuration, not a built-in default: with
# SENSES_SLOTS empty this is the colony scheduler it always was.
#
# Usage: schedule.sh start|stop|status|next
#        schedule.sh next --epoch          — the next slot as an epoch, for tests
#        schedule.sh next --kind           — colony | senses, for tests
#        schedule.sh in-window <HH:MM|epoch> — is that moment inside a colony
#                                              window (exit 0) or not (exit 1)
#        MOLEQULA_SCHED_NOW=<epoch>        — pretend it is that moment (tests)
#        MOLEQULA_SCHED_PROC=<dir>         — read memory from this tree instead
#                                              of /proc (tests)
#        schedule.sh __slot colony|senses [epoch] — run one slot now, without
#                                              the daemon, for the gate
# Configuration: phone1/schedule.conf, or SCHEDULE_CONF=<file>, or the same
# names in the environment, which win over the file.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
RUN="${MOLEQULA_RUN:-/data/data/com.termux/files/home/arianna/molequla-run}"
PIDDIR="$RUN/pids"
PIDF="$PIDDIR/schedule.pid"
LOGF="$RUN/schedule.log"
OUTF="$RUN/schedule.out"
NAMES="earth air water fire witness"
# Every memory reading in this file goes through one path, so the gate can put a
# tree of its own under it and drive the real sampler against a colony it
# controls. /proc everywhere else.
PROC="${MOLEQULA_SCHED_PROC:-/proc}"

# --- configuration: defaults, then the file, then the environment -----------
CONF="${SCHEDULE_CONF:-$HERE/schedule.conf}"
env_slots="${SCHEDULE_SLOTS:-}"
env_dur="${SCHEDULE_DUR:-}"
env_grace="${SCHEDULE_GRACE:-}"
env_sample="${SCHEDULE_SAMPLE:-}"
env_catchup="${SCHEDULE_CATCHUP:-}"
env_oom="${SCHEDULE_OOM_ADJ:-}"
env_prekill="${SCHEDULE_PREKILL:-}"
env_senses_slots="${SENSES_SLOTS:-}"
env_senses_cmd="${SENSES_CMD:-}"
env_senses_timeout="${SENSES_TIMEOUT:-}"

SCHEDULE_SLOTS="04:00 12:00 20:00"
SCHEDULE_DUR=7200
SCHEDULE_GRACE=90
SCHEDULE_SAMPLE=30
SCHEDULE_CATCHUP=1800
SCHEDULE_OOM_ADJ=500
SCHEDULE_PREKILL="android am kill-all"
SENSES_SLOTS=""
SENSES_CMD=""
SENSES_TIMEOUT=600
# shellcheck source=/dev/null
[ -f "$CONF" ] && . "$CONF"
[ -n "$env_slots" ] && SCHEDULE_SLOTS="$env_slots"
[ -n "$env_dur" ] && SCHEDULE_DUR="$env_dur"
[ -n "$env_grace" ] && SCHEDULE_GRACE="$env_grace"
[ -n "$env_sample" ] && SCHEDULE_SAMPLE="$env_sample"
[ -n "$env_catchup" ] && SCHEDULE_CATCHUP="$env_catchup"
[ -n "$env_oom" ] && SCHEDULE_OOM_ADJ="$env_oom"
[ -n "$env_prekill" ] && SCHEDULE_PREKILL="$env_prekill"
[ -n "$env_senses_slots" ] && SENSES_SLOTS="$env_senses_slots"
[ -n "$env_senses_cmd" ] && SENSES_CMD="$env_senses_cmd"
[ -n "$env_senses_timeout" ] && SENSES_TIMEOUT="$env_senses_timeout"
[ -n "$SENSES_CMD" ] || SENSES_CMD="bash '$HERE/senses.sh' all"

die() { echo "[schedule] $*" >&2; exit 1; }

case "$SCHEDULE_DUR" in ''|*[!0-9]*) die "SCHEDULE_DUR must be a whole number of seconds, got '$SCHEDULE_DUR'";; esac
[ "$SCHEDULE_DUR" -gt 0 ] || die "SCHEDULE_DUR must be positive"
case "$SCHEDULE_SAMPLE" in ''|*[!0-9]*) die "SCHEDULE_SAMPLE must be a whole number of seconds";; esac
[ "$SCHEDULE_SAMPLE" -gt 0 ] || die "SCHEDULE_SAMPLE must be positive"
case "$SENSES_TIMEOUT" in ''|*[!0-9]*) die "SENSES_TIMEOUT must be a whole number of seconds, got '$SENSES_TIMEOUT'";; esac
[ "$SENSES_TIMEOUT" -gt 0 ] || die "SENSES_TIMEOUT must be positive"

# --- slot arithmetic --------------------------------------------------------
# HH:MM -> seconds since UTC midnight. Rejects everything else; 08 and 09 are
# read base 10, not as broken octal.
slot_secs() {
    local s="$1" h m
    case "$s" in [0-2][0-9]:[0-5][0-9]) ;; *) return 1 ;; esac
    h=$((10#${s%%:*})); m=$((10#${s##*:}))
    [ "$h" -le 23 ] || return 1
    printf '%d' $((h * 3600 + m * 60))
}

now_epoch() { printf '%d' "${MOLEQULA_SCHED_NOW:-$(date -u +%s)}"; }

# next_slot <now-epoch> [slot-list] -> epoch of the next slot at or after now.
# The epoch is UTC by definition, so the day starts at now - now % 86400 and a
# slot already past today is the same slot tomorrow — that is the whole
# midnight case.
next_slot() {
    local now="$1" slots="${2:-$SCHEDULE_SLOTS}" day best="" s sec c
    day=$((now - now % 86400))
    for s in $slots; do
        sec="$(slot_secs "$s")" || { echo "[schedule] bad slot '$s' (want HH:MM, 00:00-23:59)" >&2; return 1; }
        c=$((day + sec))
        [ "$c" -lt "$now" ] && c=$((c + 86400))
        if [ -z "$best" ] || [ "$c" -lt "$best" ]; then best="$c"; fi
    done
    [ -n "$best" ] || { echo "[schedule] no slots configured" >&2; return 1; }
    printf '%d' "$best"
}

# in_colony_window <epoch> -> 0 if that moment falls inside a colony session,
# 1 if it does not. A window opens at its slot and lasts SCHEDULE_DUR, so a
# session started late yesterday can still cover a moment early today: both
# the current day's occurrence and the previous day's are tested.
in_colony_window() {
    local t="$1" day s sec c
    day=$((t - t % 86400))
    for s in $SCHEDULE_SLOTS; do
        sec="$(slot_secs "$s")" || { echo "[schedule] bad slot '$s' (want HH:MM, 00:00-23:59)" >&2; return 2; }
        for c in $((day + sec - 86400)) $((day + sec)); do
            if [ "$t" -ge "$c" ] && [ "$t" -lt $((c + SCHEDULE_DUR)) ]; then return 0; fi
        done
    done
    return 1
}

# next_any <now-epoch> -> "<epoch> <kind>" for the nearest slot of either kind.
# A tie goes to the colony: the senses would be skipped inside its window
# anyway, and the organisms are the point of the phone.
next_any() {
    local now="$1" c s=""
    c="$(next_slot "$now" "$SCHEDULE_SLOTS")" || return 1
    if [ -n "$SENSES_SLOTS" ]; then
        s="$(next_slot "$now" "$SENSES_SLOTS")" || return 1
    fi
    if [ -n "$s" ] && [ "$s" -lt "$c" ]; then
        printf '%d senses' "$s"
    else
        printf '%d colony' "$c"
    fi
}

iso() { date -u -d "@$1" +%FT%TZ; }
hhmm() { date -u -d "@$1" +%H:%M; }
hhmmss() { date -u -d "@$1" +%H:%M:%SZ; }

human_gap() {
    local d="$1"
    printf '%dh%02dm%02ds' $((d / 3600)) $(((d % 3600) / 60)) $((d % 60))
}

# --- process helpers --------------------------------------------------------
alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

# A pid file may name the `timeout` wrapper; the memory belongs to the organism
# under it (same walk as status.sh).
real_pid() {
    local p="$1" c
    while [ -r "$PROC/$p/comm" ] && [ "$(cat "$PROC/$p/comm" 2>/dev/null)" = "timeout" ]; do
        c="$(pgrep -P "$p" 2>/dev/null | head -1)"
        [ -n "$c" ] || break
        p="$c"
    done
    printf '%s' "$p"
}

# Names of the colony processes whose pid file points at something living.
live_names() {
    local n p out=""
    for n in $NAMES; do
        p="$(cat "$PIDDIR/$n.pid" 2>/dev/null)"
        alive "$p" && out="$out $n"
    done
    printf '%s' "${out# }"
}

mem_avail_mb() { awk '/^MemAvailable:/{printf "%.0f", $2/1024}' "$PROC/meminfo"; }

# --- the daemon -------------------------------------------------------------
log_session() { printf '%s\n' "$*" >> "$LOGF"; }

# One pass over the live colony, two different readings out of the same
# /proc/<pid>/status.
#
# VmHWM is per-process and lifetime — the largest each organism ever was — and
# it is what hwm_mb has always reported. It cannot answer the question the
# burst lock was landed to change. After SerialBursts the per-organism peaks
# rose (earth 1416->1691, air 861->1185) while the colony never fell below 1 GB
# free, and four high-water marks that happened at four different moments say
# nothing about whether four peaks ever stood together (MOLEQULALOG2.md,
# 2026-09-15, "the third session": *the simultaneity claim itself is not
# measured by these numbers and wants a sampler that records the sum of
# resident sets*). This is that sampler.
#
# VmRSS is what a process is holding at this instant. The sum of VmRSS across
# whatever is alive right now, maximised over the session, is the colony's
# simultaneous footprint, and it is the number to compare session against
# session to see whether serialising the bursts bought anything. A process that
# died between `alive` and the read contributes nothing to it, which is the
# point: the sum is of the live set at that instant, not of the last figure
# each name reported.
#
# MemAvailable is read at the same tick and its minimum kept, because the sum
# says what the colony held and the minimum says how close the machine came to
# the wall — the two halves of the §4.2 sleep arithmetic, which reads a floor
# against MemAvailable and a cost against the organisms.
sample_mem() {
    local n p rp kb rss sum=0 free
    for n in $NAMES; do
        p="$(cat "$PIDDIR/$n.pid" 2>/dev/null)"
        alive "$p" || continue
        rp="$(real_pid "$p")"
        kb=""; rss=""
        read -r kb rss < <(awk '/^VmHWM:/{h=$2} /^VmRSS:/{r=$2} END{if (h != "") print h, r+0}' "$PROC/$rp/status" 2>/dev/null)
        [ -n "$kb" ] || continue
        if [ -z "${HWM[$n]:-}" ] || [ "$kb" -gt "${HWM[$n]}" ]; then HWM[$n]="$kb"; fi
        sum=$((sum + rss))
    done
    if [ "$sum" -gt "${RSS_SUM_MAX:-0}" ]; then
        RSS_SUM_MAX="$sum"
        RSS_SUM_AT="$(date -u +%s)"
    fi
    free="$(mem_avail_mb)"
    if [ -n "$free" ] && { [ -z "$MEM_MIN" ] || [ "$free" -lt "$MEM_MIN" ]; }; then MEM_MIN="$free"; fi
}

hwm_field() {
    local n out=""
    for n in $NAMES; do
        [ -n "${HWM[$n]:-}" ] || continue
        out="$out,$n:$(( (HWM[$n] + 512) / 1024 ))"
    done
    printf '%s' "${out#,}"
}

# The simultaneous footprint for the session line: the largest sum of resident
# sets seen, the moment it was seen, and the least MemAvailable of the session.
# A session with no sample that found a live organism has no sum to report and
# says so rather than reporting 0 MB.
rss_field() {
    if [ "${RSS_SUM_MAX:-0}" -gt 0 ] && [ -n "${RSS_SUM_AT:-}" ]; then
        printf 'rss_sum_max_mb=%d at %s mem_min_mb=%s' \
            $(( (RSS_SUM_MAX + 512) / 1024 )) "$(hhmmss "$RSS_SUM_AT")" "${MEM_MIN:--}"
    else
        printf 'rss_sum_max_mb=- mem_min_mb=%s' "${MEM_MIN:--}"
    fi
}

set_oom_adj() {
    local n p rp done_=0
    [ -n "$SCHEDULE_OOM_ADJ" ] || return 0
    for n in earth air water fire; do
        p="$(cat "$PIDDIR/$n.pid" 2>/dev/null)"
        alive "$p" || continue
        rp="$(real_pid "$p")"
        echo "$SCHEDULE_OOM_ADJ" > "$PROC/$rp/oom_score_adj" 2>/dev/null && done_=$((done_ + 1))
    done
    echo "[schedule] oom_score_adj=$SCHEDULE_OOM_ADJ on $done_ organism(s)"
}

# prekill: hand the organisms back the memory Android is sitting on, immediately
# before a colony session. Measured on this phone 2026-09-15: `am kill-all` drops
# the cached bin — 595 MB of cached app PSS — and MemAvailable rose 3 812 -> 4 130
# MB at once and was still 4 025 MB eleven minutes later, +213 MB sustained, with
# fifteen of the nineteen killed processes never coming back
# (reports/2026-09-15_phone1_android_memory/README.md §4).
#
# Colony slots only. The senses burst is 955 MB of anonymous memory held for
# forty seconds (§3.1); it fit in free memory without moving a single page to
# swap, the eye reads MemAvailable itself before it opens, and a pass is short
# enough that everything killed for it would be paged back in by the colony
# window an hour later. The cost of the kill — cold app starts, the media
# indexers rescanning — buys nothing there. Before four organisms that hold
# 758-1091 MB each for two hours it buys the whole difference.
#
# A prekill never costs a slot: a missing command, a non-zero exit, anything —
# it is logged and the session launches anyway. PREKILL_MB carries "A->B" for
# the session line, or "-" when nothing ran.
PREKILL_MB="-"
prekill() {
    local before after out rc
    PREKILL_MB="-"
    [ -n "$SCHEDULE_PREKILL" ] || return 0
    before="$(mem_avail_mb)"
    out="$($SCHEDULE_PREKILL 2>&1)"; rc=$?
    after="$(mem_avail_mb)"
    PREKILL_MB="${before}->${after}"
    if [ "$rc" -ne 0 ]; then
        PREKILL_MB="${PREKILL_MB}!rc$rc"
        echo "[schedule] prekill '$SCHEDULE_PREKILL' failed rc=$rc (${out:-no output}) — the session runs anyway"
    else
        echo "[schedule] prekill '$SCHEDULE_PREKILL': MemAvailable ${before}->${after} MB"
    fi
    return 0
}

# run_session <slot-epoch>: launch, watch, cap, confirm down, log one line.
run_session() {
    local slot="$1" t0 t1 mem0 mem1 out rc reason elapsed left samples=0 RSS_SUM_MAX=0 RSS_SUM_AT="" MEM_MIN=""
    declare -A HWM=()

    left="$(live_names)"
    if [ -n "$left" ]; then
        t0="$(date -u +%s)"
        log_session "$(iso "$t0") kind=colony slot=$(hhmm "$slot") start=$(iso "$t0") end=$(iso "$t0") dur=$SCHEDULE_DUR elapsed=0 reason=skipped-running alive=${left// /,} mem_mb=$(mem_avail_mb) hwm_mb=- rss_sum_max_mb=- mem_min_mb=- samples=0"
        echo "[schedule] slot $(hhmm "$slot"): colony already up ($left) — slot skipped"
        return 0
    fi

    t0="$(date -u +%s)"
    # Before the reading, not after it: mem0 is what the session actually starts
    # with, and the prekill's own before/after is logged separately.
    prekill
    mem0="$(mem_avail_mb)"
    echo "[schedule] slot $(hhmm "$slot") at $(iso "$t0"): launching for ${SCHEDULE_DUR}s, MemAvailable ${mem0} MB"
    out="$(bash "$HERE/launch.sh" "$SCHEDULE_DUR" 2>&1)"; rc=$?
    printf '%s\n' "$out"

    if [ "$rc" -ne 0 ]; then
        if printf '%s' "$out" | grep -q 'already running'; then
            # A manual launch.sh won the race. It carries its own cap; leave it.
            reason=skipped-running
        else
            bash "$HERE/stop.sh" 2>&1 | sed 's/^/[schedule] /'
            reason=launch-failed
        fi
        t1="$(date -u +%s)"
        left="$(live_names)"
        log_session "$(iso "$t1") kind=colony slot=$(hhmm "$slot") start=$(iso "$t0") end=$(iso "$t1") dur=$SCHEDULE_DUR elapsed=$((t1 - t0)) reason=$reason alive=${left:--} mem_mb=${mem0}->$(mem_avail_mb) prekill_mb=$PREKILL_MB hwm_mb=- rss_sum_max_mb=- mem_min_mb=- samples=0"
        return 0
    fi

    set_oom_adj

    local deadline=$((t0 + SCHEDULE_DUR + SCHEDULE_GRACE))
    reason=""
    while :; do
        sample_mem
        samples=$((samples + 1))
        t1="$(date -u +%s)"
        if [ -z "$(live_names)" ]; then
            if [ $((t1 - t0)) -ge $((SCHEDULE_DUR - SCHEDULE_SAMPLE - 5)) ]; then reason=capped; else reason=early-exit; fi
            break
        fi
        if [ "$t1" -ge "$deadline" ]; then reason=overran; break; fi
        local nap=$SCHEDULE_SAMPLE rem=$((deadline - t1))
        [ "$rem" -lt "$nap" ] && nap=$rem
        [ "$nap" -lt 1 ] && nap=1
        sleep "$nap"
    done

    # Down is not assumed, it is made: stop.sh is called on every path — it
    # clears the pid files of the already dead and releases the wake lock.
    bash "$HERE/stop.sh" 2>&1 | sed 's/^/[schedule] /'
    left="$(live_names)"
    [ -n "$left" ] && reason="$reason+survivors:${left// /,}"

    t1="$(date -u +%s)"
    mem1="$(mem_avail_mb)"
    elapsed=$((t1 - t0))
    log_session "$(iso "$t1") kind=colony slot=$(hhmm "$slot") start=$(iso "$t0") end=$(iso "$t1") dur=$SCHEDULE_DUR elapsed=$elapsed reason=$reason alive=${left:--} mem_mb=${mem0}->${mem1} prekill_mb=$PREKILL_MB hwm_mb=$(hwm_field) $(rss_field) samples=$samples"
    echo "[schedule] slot $(hhmm "$slot") done: reason=$reason elapsed=${elapsed}s MemAvailable ${mem0}->${mem1} MB"
    return 0
}

# run_senses <slot-epoch>: one pass of the senses under a timeout, or a logged
# refusal. The colony is checked twice — against the configured windows and
# against what is actually alive — because a manual launch.sh obeys neither.
run_senses() {
    local slot="$1" t0 t1 out rc reason frags left
    t0="$(date -u +%s)"

    if in_colony_window "$slot"; then
        log_session "$(iso "$t0") kind=senses slot=$(hhmm "$slot") start=- end=- timeout=${SENSES_TIMEOUT} elapsed=0 reason=skipped-colony-window frags=0 mem_mb=$(mem_avail_mb)"
        echo "[schedule] senses slot $(hhmm "$slot"): inside a colony window — skipped"
        return 0
    fi
    left="$(live_names)"
    if [ -n "$left" ]; then
        log_session "$(iso "$t0") kind=senses slot=$(hhmm "$slot") start=- end=- timeout=${SENSES_TIMEOUT} elapsed=0 reason=skipped-colony-alive alive=${left// /,} frags=0 mem_mb=$(mem_avail_mb)"
        echo "[schedule] senses slot $(hhmm "$slot"): colony up ($left) — skipped"
        return 0
    fi

    echo "[schedule] senses slot $(hhmm "$slot") at $(iso "$t0"): $SENSES_CMD (cap ${SENSES_TIMEOUT}s)"
    out="$(timeout "$SENSES_TIMEOUT" bash -c "$SENSES_CMD" 2>&1)"; rc=$?
    printf '%s\n' "$out" | sed 's/^/[schedule] /'
    t1="$(date -u +%s)"
    case "$rc" in
        0) reason=ok ;;
        124) reason=timeout ;;
        *) reason="failed-rc$rc" ;;
    esac
    # The pass reports its own count on its last line; absent that, nothing.
    frags="$(printf '%s' "$out" | sed -n 's/.* frags=\([0-9][0-9]*\) .*/\1/p' | tail -1)"
    [ -n "$frags" ] || frags=0
    log_session "$(iso "$t1") kind=senses slot=$(hhmm "$slot") start=$(iso "$t0") end=$(iso "$t1") timeout=${SENSES_TIMEOUT} elapsed=$((t1 - t0)) reason=$reason frags=$frags mem_mb=$(mem_avail_mb)"
    echo "[schedule] senses slot $(hhmm "$slot") done: reason=$reason elapsed=$((t1 - t0))s frags=$frags"
    return 0
}

loop() {
    local now target kind line late
    mkdir -p "$PIDDIR" || exit 1
    trap 'echo "[schedule] signal — daemon exits, any live session keeps its own timeout cap"; rm -f "$PIDF"; exit 0' TERM INT
    echo "[schedule] daemon pid $$ up at $(iso "$(date -u +%s)"): colony [$SCHEDULE_SLOTS] UTC session ${SCHEDULE_DUR}s, senses [${SENSES_SLOTS:-none}] cap ${SENSES_TIMEOUT}s, grace ${SCHEDULE_GRACE}s, sample ${SCHEDULE_SAMPLE}s, catchup ${SCHEDULE_CATCHUP}s, conf $CONF"
    while :; do
        now="$(date -u +%s)"
        line="$(next_any "$now")" || exit 1
        target="${line%% *}"; kind="${line##* }"
        echo "[schedule] next slot $(iso "$target") kind=$kind (in $(human_gap $((target - now))))"
        # Short naps, decided against the wall clock: a long sleep does not
        # count the time the phone spends suspended.
        while :; do
            now="$(date -u +%s)"
            [ "$now" -ge "$target" ] && break
            local nap=$((target - now))
            [ "$nap" -gt 60 ] && nap=60
            sleep "$nap"
        done
        late=$(($(date -u +%s) - target))
        if [ "$late" -gt "$SCHEDULE_CATCHUP" ]; then
            echo "[schedule] slot $(iso "$target") kind=$kind reached ${late}s late — skipped"
            log_session "$(iso "$(date -u +%s)") kind=$kind slot=$(hhmm "$target") start=- end=- dur=$SCHEDULE_DUR elapsed=0 reason=missed late=${late}s mem_mb=$(mem_avail_mb) hwm_mb=- rss_sum_max_mb=- mem_min_mb=- samples=0"
        elif [ "$kind" = senses ]; then
            run_senses "$target"
        else
            run_session "$target"
        fi
        # Never look at the same slot twice.
        while [ "$(date -u +%s)" -le "$target" ]; do sleep 1; done
    done
}

# --- commands ---------------------------------------------------------------
cmd_start() {
    local old; old="$(cat "$PIDF" 2>/dev/null)"
    if alive "$old"; then
        echo "[schedule] already running (pid $old) — refusing a second daemon"
        return 1
    fi
    mkdir -p "$PIDDIR" || return 1
    rm -f "$PIDF"
    setsid nohup bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$PIDF" \
        bash "$SELF" __loop >> "$OUTF" 2>&1 < /dev/null &
    local i=0 pid=""
    while [ $i -lt 50 ]; do
        pid="$(cat "$PIDF" 2>/dev/null)"
        alive "$pid" && break
        i=$((i + 1)); sleep 0.1
    done
    if alive "$pid"; then
        echo "[schedule] daemon pid $pid, log $LOGF, console $OUTF"
        cmd_next
        return 0
    fi
    echo "[schedule] FAILED to start — see $OUTF"
    return 1
}

cmd_stop() {
    local p; p="$(cat "$PIDF" 2>/dev/null)"
    if ! alive "$p"; then
        echo "[schedule] not running (no live pid in $PIDF)"
        rm -f "$PIDF"
        return 0
    fi
    kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null
    local i=0
    while [ $i -lt 15 ] && alive "$p"; do i=$((i + 1)); sleep 1; done
    if alive "$p"; then
        kill -KILL -- "-$p" 2>/dev/null || kill -KILL "$p" 2>/dev/null
        sleep 1
        echo "[schedule] daemon pid $p killed"
    else
        echo "[schedule] daemon pid $p stopped"
    fi
    rm -f "$PIDF"
    local left; left="$(live_names)"
    if [ -n "$left" ]; then
        echo "[schedule] note: colony still up ($left) under its own timeout cap — phone1/stop.sh ends it now"
    fi
    return 0
}

cmd_status() {
    local p; p="$(cat "$PIDF" 2>/dev/null)"
    if alive "$p"; then echo "daemon: pid $p alive"; else echo "daemon: down${p:+ (stale pid $p)}"; fi
    echo "conf:   $CONF"
    echo "colony: $SCHEDULE_SLOTS UTC, session ${SCHEDULE_DUR}s, grace ${SCHEDULE_GRACE}s, sample ${SCHEDULE_SAMPLE}s, catchup ${SCHEDULE_CATCHUP}s"
    echo "senses: ${SENSES_SLOTS:-none} UTC, cap ${SENSES_TIMEOUT}s, $SENSES_CMD"
    cmd_next
    local left; left="$(live_names)"
    echo "alive:  ${left:-nothing}"
    echo "memory: MemAvailable $(mem_avail_mb) MB"
    if [ -s "$LOGF" ]; then
        echo "last sessions ($LOGF):"
        tail -3 "$LOGF" | sed 's/^/  /'
    else
        echo "last sessions: none yet ($LOGF)"
    fi
}

cmd_next() {
    local now line target kind note=""
    now="$(now_epoch)"
    line="$(next_any "$now")" || return 1
    target="${line%% *}"; kind="${line##* }"
    case "${1:-}" in
        --epoch) printf '%s\n' "$target"; return 0 ;;
        --kind)  printf '%s\n' "$kind"; return 0 ;;
    esac
    if [ "$kind" = senses ]; then
        in_colony_window "$target" && note=" — inside a colony window, it will be skipped"
        echo "next:   $(iso "$target") (in $(human_gap $((target - now)))) kind=senses, cap ${SENSES_TIMEOUT}s$note"
    else
        echo "next:   $(iso "$target") (in $(human_gap $((target - now)))) kind=colony, session ${SCHEDULE_DUR}s, ends $(iso $((target + SCHEDULE_DUR)))"
    fi
}

# in-window <HH:MM|epoch>: the colony-window predicate the senses slots are
# filtered by, callable on its own so a gate can drive it. HH:MM is read
# against the current UTC day (or MOLEQULA_SCHED_NOW's day).
cmd_in_window() {
    local t="${1:-}" sec now
    [ -n "$t" ] || die "in-window needs a time: HH:MM or an epoch"
    case "$t" in
        *[!0-9]*)
            sec="$(slot_secs "$t")" || die "in-window: '$t' is neither HH:MM nor an epoch"
            now="$(now_epoch)"
            t=$((now - now % 86400 + sec))
            ;;
    esac
    if in_colony_window "$t"; then echo "inside"; return 0; fi
    echo "outside"; return 1
}

case "${1:-}" in
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    status)    cmd_status ;;
    next)      cmd_next "${2:-}" ;;
    in-window) cmd_in_window "${2:-}" ;;
    __loop)    loop ;;
    # One slot, run now, with no daemon around it: what the gate drives so that
    # run_session and run_senses are the code under test and not a copy.
    __slot)
        mkdir -p "$PIDDIR" || exit 1
        case "${2:-colony}" in
            colony) run_session "${3:-$(date -u +%s)}" ;;
            senses) run_senses  "${3:-$(date -u +%s)}" ;;
            *) die "__slot: want colony or senses, got '${2:-}'" ;;
        esac
        ;;
    *) echo "usage: schedule.sh start|stop|status|next [--epoch|--kind]|in-window <HH:MM|epoch>" >&2; exit 2 ;;
esac
