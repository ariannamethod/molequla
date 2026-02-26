# Lilith — Data Infrastructure Layer

> "Та, которая была до Евы."
> Идея Олега, 2026-02-24.

## What

Lilith breaks the closed loop of Molequla. Instead of organisms growing only
on seed corpus + DNA exchange, an army of INDEX nodes crawls the outside world
and feeds data into the ecology automatically.

**The human is present but not necessary.**

## Architecture

```
lilith.aml (AML brain, prophecy/destiny steering)
    │
ariannamethod.c (AML runtime + I/O commands)
    │
named pipes (FIFO)
    ┌───┼───┬───┐
    ▼   ▼   ▼   ▼
 IDX1 IDX2 IDX3 IDX4  (Go + embedding transformer)
    │   │   │   │
    ▼   ▼   ▼   ▼
  Reddit subreddits → shared SQLite (vec + FTS5)
        │
   Mycelium (metric-based orchestrator)
    ┌───┼───┬───┐
    ▼   ▼   ▼   ▼
 Earth Air Water Fire (organisms)
```

## Components

### AML side (in ariannamethod.ai repo, branch v3-lilith-singularity)
- `PIPE CREATE/OPEN/WRITE/READ/CLOSE` — named pipe IPC
- `INDEX <id> INIT/FETCH/STATUS/STOP/CLOSE` — high-level sugar
- `examples/lilith.aml` — brain script v0.1
- 276/276 tests passing

### Go side (this directory)
- `index_node.go` — INDEX node: pipe listener, command dispatch
- `lilith.go` — entry point, signal handling
- `lilith.aml` — copy of brain script for reference

## Protocol

Communication between AML and Go via newline-delimited messages over named pipes:

| Direction | Pipe | Example |
|-----------|------|---------|
| AML → Go | `/tmp/lilith_idx1_cmd` | `FETCH r/philosophy` |
| Go → AML | `/tmp/lilith_idx1_rsp` | `OK queued r/philosophy` |

Commands: `FETCH <subreddit>`, `STATUS`, `STOP`

## Status

**v0.1 scaffolding** — pipes work, protocol defined, lilith.aml runs.
Next: Reddit crawling, embedding, SQLite indexing.
