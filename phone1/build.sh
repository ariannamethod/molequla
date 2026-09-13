#!/bin/bash
# Build the checkout this script lives in, with the recipe measured on the A56,
# into $MOLEQULA_RUN/molequla_cgo. Records the commit and the date in BUILD.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
RUN="${MOLEQULA_RUN:-/data/data/com.termux/files/home/arianna/molequla-run}"
OUT="$RUN/molequla_cgo"

mkdir -p "$RUN" || exit 1

echo "[build] repo=$REPO out=$OUT"
cd "$REPO" || exit 1

# -a is mandatory (CGO cache trap: without it Go reuses stale compiled C).
# ariannamethod.c:1500 prints a harmless calloc warning.
env CGO_ENABLED=1 \
    CGO_CFLAGS="-O3 -march=native -mtune=native -DUSE_BLAS" \
    CGO_LDFLAGS="-lopenblas -lm -lpthread" \
    go build -a -trimpath -buildvcs=false -o "$OUT" .
rc=$?
if [ $rc -ne 0 ]; then
    echo "[build] FAILED (go build exit $rc)"
    exit $rc
fi

REV="$(git -C "$REPO" -c safe.directory="$REPO" rev-parse --short HEAD 2>/dev/null)"
[ -n "$REV" ] || REV="unknown"
WHEN="$(date -u +%FT%TZ)"
printf '%s %s %s %s bytes\n' "$REV" "$WHEN" "$OUT" "$(wc -c < "$OUT")" > "$RUN/BUILD"

echo "[build] commit=$REV date=$WHEN"
echo "[build] $(cat "$RUN/BUILD")"
