# molequla — CLAUDE.md

Hey Claude, bro. This is molequla: an ecology of GPT organisms that are born as
10K-parameter embryos, grow their own architecture at runtime, feed on each
other's speech, and divide when the field overwhelms them. Four organism cores
share one design — Go (`molequla.go`, the primary), C (`molequla.c`), Rust
(`molequla.rs`), JavaScript (`molequla.js`) — trained on the notorch tape with
Chuck, with AML/C as the fallback trainer and the organism's field-physics
language. GPL-3.0+, co-authored by Oleg Ataeff and Claude. Several Claude nodes
work in this tree — polygon, neo, and now a phone inside a chroot — so assume
someone else is mid-commit while you read this. The paper is
`docs/molequla_paper.md` (Zenodo 10.5281/zenodo.21046231); its central result is
that reproduction is keyed on the loss an adult cannot reduce, not on the entropy
the design first assumed. Read it before touching the mitosis gate.

## Branch, always

`git checkout -b claude/<what-you-are-doing>`, push the branch, let Oleg merge.
Nobody lands on `main` directly. Immediately before finalizing a commit, fetch
`origin` and integrate `origin/main` into your branch without destroying local
work; check it once more before push. No blind pull, no destructive reset.

## Three books, one face

- `README.md` is the face and the spec. It says what molequla *is*. It never
  becomes a worklog, and it never lags the code: if reality drifts from the
  README, fix the code or update the spec deliberately.
- `PROJECT_LOG.md` was the first book of the engineering log: Phase A (GPU),
  Phase B (graze), Phase C (ecology), the §9 mitosis run, the cascade governor.
  It left the tree on 2026-09-13 (`8203d5d`) and lives in git history:
  `git show 8203d5d^:PROJECT_LOG.md`. Nothing is appended there any more;
  README references to it are being moved to the log below.
- `MOLEQULALOG2.md` is the living diary, opened when molequla landed on phone-1.
  Every session that changes anything appends a dated section, signed by the
  node, with what was built and how it was verified. Numbers come from
  artifacts, tool output or git, cited inline by path, line or commit — never
  retyped from memory. A bug fix, a faster kernel, a measurement are log
  entries; a new organ, a new trainer, an architecture shift also earn README
  space. When in doubt it is a log entry.
- `ROADMAP.md` is the third book: what comes next and in what order, as dated
  entries that move between *in flight*, *next*, *later* and *closed*. The
  diary records what happened; the roadmap records what was agreed to happen.
  An item leaves it with the commit that closed it. It also states how the
  colony is run — capped sessions on a schedule, never a permanent process.

## Every claim is a measurement or it is not made

"Faster" means a number, a shape, and the machine it ran on. "Correct" means a
gate that goes red when you break the thing on purpose; a test that has never
failed is decoration. RSS, tick time and steps/s taken on a pod do not transfer
to a phone; measure again on the target, on its big cores, and write down the
command line that produced the number. Sweep temperature × top_k × several
prompts before calling any voice incoherent. A run that trains is announced
with the six points — organism, dataset and size, steps, architecture,
tokenizer equal to inference, script pushed — and started on Oleg's word.

## Build

```sh
CGO_ENABLED=1 go build -a -o molequla_cgo .    # -a is mandatory: without it Go
                                                # reuses stale compiled C
go test ./...                                   # last recorded: 166 green
                                                # (MOLEQULALOG2.md, 2026-09-13)
bash tests/test_all.sh                          # all four cores + AML + BLAS
```

CUDA exists only behind `-tags cuda` on Linux; every other build takes the
CPU/BLAS path through the stubs, and that is the path the phones run. On a
big.LITTLE phone pin the organism to the big cores (`taskset -c 4-7` on the
A56): more cores is not faster, the slowest core class sets the pace.

## Bug patterns to know (each one cost a session)

- **CGO cache trap.** `go build` without `-a` runs old C silently.
- **Duplicated invariant.** The post-growth freeze counter was decremented in
  two training paths and not the third; the colony stopped at adolescent for
  a whole run (`ff6ad49`). If a counter lives in N places, N is the bug.
- **Grow invalidates caches.** A grown matrix keeps its old cached GPU shape
  unless `invalidateGPU()` runs from `GrowRows`/`GrowCols`/`Grow`.
- **Thread storm.** OpenBLAS defaults to host `nproc` threads per process; four
  organisms on a cgroup-capped box thrashed and starved the GPU.
  `capColonyThreads()` reads the cgroup quota first thing in `main()`.
- **The gate that listens to the wrong signal.** A converged adult is
  confidently wrong: entropy ~0.22 while loss sits ~12. Overload is the loss
  path OR the entropy path (`isSustainedOverload`); do not collapse them.
- **Child born as a stranger.** `performMitosis` once wrote the parent
  checkpoint under one name and told the child to load another. Reproduction
  that yields random embryos is not reproduction; the child loads what was
  written.
- **Uncapped cascade.** Children inherit an overwhelmed adult's weights and its
  unfalling loss, and divide in turn: ~50 spawns before shutdown in §9. The
  governor is an atomic `mesh.db` admit against `MaxOrganisms`, a birth-seeded
  cooldown, and divide-relieves-parent on both paths. On a phone the cap is
  computed from measured adult RSS, never guessed.
- **Write storm.** Checkpoints are debounced (`CheckpointMinInterval`); a hot
  loop of writes preceded the Railway silence of 2026-05-03.

## Never

- Push to `main` without Oleg's go-ahead. Force-push to `main` is a hard line.
- Put Python on an inference or organism-core path. The four cores are
  Python-free by design; the orchestration tier above them is being moved to
  Go, and on the phones no Python runs at all.
- Let the vendored `modules/gpu/csrc/notorch.{c,h}` drift silently. It is a copy
  of the canon (`github.com/ariannamethod/notorch`), the source of the GPU
  library on CUDA hosts only — the CPU build links the system `libnotorch`;
  when you sync it, name the molequla-specific delta in the log instead of
  carrying it unspoken.
- Touch `runpod/` archives or the paper's numbers. They are the record behind a
  DOI; a correction is a new dated entry, not an edit of history.
- Name the classical per-parameter diagonal optimizer baseline in new text. The
  optimizer here is Chuck; the README's historical mentions stay as history.
- Turn `reffs/` into a dependency. It holds gitignored reference clones
  (arianna.c, dario, q, actually.life, microkarpathy, netta, ocelli,
  ariannamethod.ai) for reading and lineage, nothing links against them.

## Commits and attribution

One commit, one concept, in English; the message states the technical facts
verified with a tool. The signature lives in the git commit only, node-visible:
`Co-Authored-By: Claude (Arianna Method, <node>) <theariannamethod@gmail.com>`
with node ∈ polygon / neo / intel godfather / metal / Defender (phone-1) /
Opus07 (phone-2). Nowhere else — no signature footer in README, docs, or file
bodies. Drop upstream boilerplate.
