#!/bin/bash
# The colony never runs non-stop. This daemon keeps it that way: it sleeps
# until the next UTC slot, runs launch.sh with the session cap, confirms
# everything is down afterwards, writes one line about the session into
# $MOLEQULA_RUN/schedule.log, and sleeps to the next slot.
#
# Usage: schedule.sh start|stop|status|next
#        schedule.sh next --epoch          — the next slot as an epoch, for tests
#        MOLEQULA_SCHED_NOW=<epoch>        — pretend it is that moment (tests)
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

# --- configuration: defaults, then the file, then the environment -----------
CONF="${SCHEDULE_CONF:-$HERE/schedule.conf}"
env_slots="${SCHEDULE_SLOTS:-}"
env_dur="${SCHEDULE_DUR:-}"
env_grace="${SCHEDULE_GRACE:-}"
env_sample="${SCHEDULE_SAMPLE:-}"
env_catchup="${SCHEDULE_CATCHUP:-}"
env_oom="${SCHEDULE_OOM_ADJ:-}"

SCHEDULE_SLOTS="04:00 12:00 20:00"
SCHEDULE_DUR=7200
SCHEDULE_GRACE=90
SCHEDULE_SAMPLE=30
SCHEDULE_CATCHUP=1800
SCHEDULE_OOM_ADJ=500
# shellcheck source=/dev/null
[ -f "$CONF" ] && . "$CONF"
[ -n "$env_slots" ] && SCHEDULE_SLOTS="$env_slots"
[ -n "$env_dur" ] && SCHEDULE_DUR="$env_dur"
[ -n "$env_grace" ] && SCHEDULE_GRACE="$env_grace"
[ -n "$env_sample" ] && SCHEDULE_SAMPLE="$env_sample"
[ -n "$env_catchup" ] && SCHEDULE_CATCHUP="$env_catchup"
[ -n "$env_oom" ] && SCHEDULE_OOM_ADJ="$env_oom"

die() { echo "[schedule] $*" >&2; exit 1; }

case "$SCHEDULE_DUR" in ''|*[!0-9]*) die "SCHEDULE_DUR must be a whole number of seconds, got '$SCHEDULE_DUR'";; esac
[ "$SCHEDULE_DUR" -gt 0 ] || die "SCHEDULE_DUR must be positive"
case "$SCHEDULE_SAMPLE" in ''|*[!0-9]*) die "SCHEDULE_SAMPLE must be a whole number of seconds";; esac
[ "$SCHEDULE_SAMPLE" -gt 0 ] || die "SCHEDULE_SAMPLE must be positive"

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

# next_slot <now-epoch> -> epoch of the next slot at or after now. The epoch is
# UTC by definition, so the day starts at now - now % 86400 and a slot already
# past today is the same slot tomorrow — that is the whole midnight case.
next_slot() {
    local now="$1" day best="" s sec c
    day=$((now - now % 86400))
    for s in $SCHEDULE_SLOTS; do
        sec="$(slot_secs "$s")" || { echo "[schedule] bad slot '$s' (want HH:MM, 00:00-23:59)" >&2; return 1; }
        c=$((day + sec))
        [ "$c" -lt "$now" ] && c=$((c + 86400))
        if [ -z "$best" ] || [ "$c" -lt "$best" ]; then best="$c"; fi
    done
    [ -n "$best" ] || { echo "[schedule] no slots configured" >&2; return 1; }
    printf '%d' "$best"
}

iso() { date -u -d "@$1" +%FT%TZ; }
hhmm() { date -u -d "@$1" +%H:%M; }

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
    while [ -r "/proc/$p/comm" ] && [ "$(cat "/proc/$p/comm" 2>/dev/null)" = "timeout" ]; do
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

mem_avail_mb() { awk '/^MemAvailable:/{printf "%.0f", $2/1024}' /proc/meminfo; }

# --- the daemon -------------------------------------------------------------
log_session() { printf '%s\n' "$*" >> "$LOGF"; }

sample_hwm() {
    local n p rp kb
    for n in $NAMES; do
        p="$(cat "$PIDDIR/$n.pid" 2>/dev/null)"
        alive "$p" || continue
        rp="$(real_pid "$p")"
        kb="$(awk '/^VmHWM:/{print $2}' "/proc/$rp/status" 2>/dev/null)"
        [ -n "$kb" ] || continue
        if [ -z "${HWM[$n]:-}" ] || [ "$kb" -gt "${HWM[$n]}" ]; then HWM[$n]="$kb"; fi
    done
}

hwm_field() {
    local n out=""
    for n in $NAMES; do
        [ -n "${HWM[$n]:-}" ] || continue
        out="$out,$n:$(( (HWM[$n] + 512) / 1024 ))"
    done
    printf '%s' "${out#,}"
}

set_oom_adj() {
    local n p rp done_=0
    [ -n "$SCHEDULE_OOM_ADJ" ] || return 0
    for n in earth air water fire; do
        p="$(cat "$PIDDIR/$n.pid" 2>/dev/null)"
        alive "$p" || continue
        rp="$(real_pid "$p")"
        echo "$SCHEDULE_OOM_ADJ" > "/proc/$rp/oom_score_adj" 2>/dev/null && done_=$((done_ + 1))
    done
    echo "[schedule] oom_score_adj=$SCHEDULE_OOM_ADJ on $done_ organism(s)"
}

# run_session <slot-epoch>: launch, watch, cap, confirm down, log one line.
run_session() {
    local slot="$1" t0 t1 mem0 mem1 out rc reason elapsed left samples=0
    declare -A HWM=()

    left="$(live_names)"
    if [ -n "$left" ]; then
        t0="$(date -u +%s)"
        log_session "$(iso "$t0") slot=$(hhmm "$slot") start=$(iso "$t0") end=$(iso "$t0") dur=$SCHEDULE_DUR elapsed=0 reason=skipped-running alive=${left// /,} mem_mb=$(mem_avail_mb) hwm_mb=- samples=0"
        echo "[schedule] slot $(hhmm "$slot"): colony already up ($left) — slot skipped"
        return 0
    fi

    t0="$(date -u +%s)"
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
        log_session "$(iso "$t1") slot=$(hhmm "$slot") start=$(iso "$t0") end=$(iso "$t1") dur=$SCHEDULE_DUR elapsed=$((t1 - t0)) reason=$reason alive=${left:--} mem_mb=${mem0}->$(mem_avail_mb) hwm_mb=- samples=0"
        return 0
    fi

    set_oom_adj

    local deadline=$((t0 + SCHEDULE_DUR + SCHEDULE_GRACE))
    reason=""
    while :; do
        sample_hwm
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
    log_session "$(iso "$t1") slot=$(hhmm "$slot") start=$(iso "$t0") end=$(iso "$t1") dur=$SCHEDULE_DUR elapsed=$elapsed reason=$reason alive=${left:--} mem_mb=${mem0}->${mem1} hwm_mb=$(hwm_field) samples=$samples"
    echo "[schedule] slot $(hhmm "$slot") done: reason=$reason elapsed=${elapsed}s MemAvailable ${mem0}->${mem1} MB"
    return 0
}

loop() {
    local now target late
    mkdir -p "$PIDDIR" || exit 1
    trap 'echo "[schedule] signal — daemon exits, any live session keeps its own timeout cap"; rm -f "$PIDF"; exit 0' TERM INT
    echo "[schedule] daemon pid $$ up at $(iso "$(date -u +%s)"): slots [$SCHEDULE_SLOTS] UTC, session ${SCHEDULE_DUR}s, grace ${SCHEDULE_GRACE}s, sample ${SCHEDULE_SAMPLE}s, catchup ${SCHEDULE_CATCHUP}s, conf $CONF"
    while :; do
        now="$(date -u +%s)"
        target="$(next_slot "$now")" || exit 1
        echo "[schedule] next slot $(iso "$target") (in $(human_gap $((target - now))))"
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
            echo "[schedule] slot $(iso "$target") reached ${late}s late — skipped"
            log_session "$(iso "$(date -u +%s)") slot=$(hhmm "$target") start=- end=- dur=$SCHEDULE_DUR elapsed=0 reason=missed late=${late}s mem_mb=$(mem_avail_mb) hwm_mb=- samples=0"
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
    echo "slots:  $SCHEDULE_SLOTS UTC, session ${SCHEDULE_DUR}s, grace ${SCHEDULE_GRACE}s, sample ${SCHEDULE_SAMPLE}s, catchup ${SCHEDULE_CATCHUP}s"
    cmd_next
    local left; left="$(live_names)"
    echo "colony: ${left:-down}"
    echo "memory: MemAvailable $(mem_avail_mb) MB"
    if [ -s "$LOGF" ]; then
        echo "last sessions ($LOGF):"
        tail -3 "$LOGF" | sed 's/^/  /'
    else
        echo "last sessions: none yet ($LOGF)"
    fi
}

cmd_next() {
    local now target
    now="$(now_epoch)"
    target="$(next_slot "$now")" || return 1
    if [ "${1:-}" = "--epoch" ]; then printf '%s\n' "$target"; return 0; fi
    echo "next:   $(iso "$target") (in $(human_gap $((target - now)))), session ${SCHEDULE_DUR}s, ends $(iso $((target + SCHEDULE_DUR)))"
}

case "${1:-}" in
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    next)   cmd_next "${2:-}" ;;
    __loop) loop ;;
    *) echo "usage: schedule.sh start|stop|status|next [--epoch]" >&2; exit 2 ;;
esac
