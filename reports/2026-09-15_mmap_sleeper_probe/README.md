# What a sleeping organism's weights cost when they are mapped instead of held

The two probes behind section 3 of `docs/resonator_design.md`. Both work on a
19 337 632-byte float32 file — 4 834 408 parameters, the exact shape of a stage-4
molequla organism — and one sweep is a matvec over every parameter, which is the
weight traffic of one forward pass.

    gcc -O2 -o mmapprobe mmapprobe.c
    gcc -O2 -o cold cold.c
    timeout -s TERM 300 taskset -c 4-7 ./mmapprobe w.f32 8     # builds w.f32 on first run
    timeout -s TERM 120 taskset -c 4-7 ./cold w.f32

Run on phone-1 (Galaxy A56, Exynos 1580, Ubuntu 24.04 chroot) on 2026-09-15
between 10:20Z and 10:35Z with the colony down and MemAvailable at 3.6-3.7 GB.

`mmapprobe` compares heap against mapped, warm and after `MADV_DONTNEED`;
`cold` answers the two questions the first one could not: what the mapping costs
when the organism is *not* speaking, and what a sweep costs with the file out of
the page cache. The second of those failed to reproduce a cold read —
`posix_fadvise(POSIX_FADV_DONTNEED)` evicted nothing, with the mapping live and
without it, on a phone holding 3.4 GB of page cache — so the flash figure in the
design document is an estimate from measured sequential throughput and says so.

    dd if=<a 3 GB gguf> of=/dev/null bs=1M skip=1200 count=600   # 858 MB/s, uncached
    dd if=<the same region again> of=/dev/null bs=1M count=600   # 5.7 GB/s, cached

Numbers and the reading of them: `docs/resonator_design.md` §3.2.
