# molequla ecology — Railway continuous deploy.
# Two-stage build: Go + CGO compile against libopenblas, runtime is debian-slim
# with libopenblas0 and the four element corpora. Launcher script in the runtime
# spawns one organism per element under /data/<element>/ on the persistent volume.

# === Build stage ===
FROM golang:1.21-bookworm AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# CGO links against AML/notorch C inline in molequla.go. cgo_aml.go declares
# Darwin-only BLAS directives; on Linux we wire OpenBLAS via env vars so AML's
# cblas_dgemv path activates without editing source.
#
# Railway CPU-fix recipe (Henry session 2026-04-29 measured 7.4× end-to-end
# speedup on similar workload):
# 1. -O3 + -march=native + -mtune=native — aggressive auto-vectorisation
#    on the host vCPU. Inner-loop matmuls outrun OpenBLAS once vectorised
#    for our small-organism dim (16-320 across ontogenesis stages).
# 2. OPENBLAS_NUM_THREADS=1 + OMP / GOTO (set in runtime stage below) —
#    disables OpenBLAS's pthread spawn per sgemv. With four organisms
#    running in parallel through launcher.sh, the spawn overhead is the
#    thread-storm class that recipe is designed for.
ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-O3 -march=native -mtune=native -DUSE_BLAS"
ENV CGO_LDFLAGS="-lopenblas -lm -lpthread"

RUN go build -trimpath -ldflags="-s -w" -o /out/molequla .

# === Runtime stage ===
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /out/molequla /app/molequla
COPY nonames_earth.txt nonames_air.txt nonames_water.txt nonames_fire.txt /app/
COPY launcher.sh /app/launcher.sh
RUN chmod +x /app/launcher.sh /app/molequla

# Persistent state on the mounted Railway volume.
ENV MOLEQULA_DATA_DIR=/data
RUN mkdir -p /data

# Railway CPU-fix recipe (runtime half) — single-thread OpenBLAS.
# Four parallel organisms calling sgemv concurrently otherwise spawn a
# thread storm that beats their actual matmul on small dim. Pair with
# the build-stage CGO_CFLAGS above (-O3 -march=native -mtune=native).
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV GOTO_NUM_THREADS=1

# Launch all four organisms; on any exit the container restarts via launcher's
# wait -n + exit 1.
CMD ["/app/launcher.sh"]
