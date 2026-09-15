# 2026-09-15 — what already exists under `molequla_new_logic.md`

This is the inspection §17 of the brief asks for, done before anything is built. Every file and
line cited below was read at `5d46fc0`, the tip of `origin/main` on the day of the audit, and
re-grepped immediately before this document was written; live numbers come from the run directory
`/data/data/com.termux/files/home/arianna/molequla-run`, whose last colony session ended
`2026-09-13T20:29:30Z` and whose last senses pass ran `2026-09-14T01:02:16Z` (`schedule.log`). The
question the audit was commissioned to answer — how an eaten fragment actually enters an organism —
is section 2; the section-by-section table is section 1; the lineage reading is section 3 and the
proposed order is section 4.

The short version is that the brief's early-regime path already exists end to end and nobody has to
build it: a fragment written by the eye reaches the organism's corpus, the n-gram and co-occurrence
field, the Q-style overlay and the training batches, through code that has been running since repair
3. What does not exist is any asymmetry in who receives it, any ledger of what changed, any inquiry,
and any glyph. Three things were found along the way that were not in anybody's plan, and they are
flagged in place: the 240-character ceiling that cuts an eaten fragment down before it reaches the
field, the read queue in which the senses stand last behind the siblings, and a witness that has
never been told the senses exist.

## 1. Section by section

| § | What exists today (verified) | Partial | Missing | Smallest extension point |
|---|---|---|---|---|
| 1 — two rhythms | Two slot kinds on one clock: `SCHEDULE_SLOTS="04:00 12:00 20:00"` and `SENSES_SLOTS="01:00 03:00 07:00 09:00 11:00 15:00 17:00 19:00 23:00"` (`phone1/schedule.conf:7,38`), `next_any` and `in_colony_window` (`phone1/schedule.sh:111,126`), a senses pass refused inside a colony window or beside a live colony (`phone1/schedule.sh:283-290`), and a second refusal in the pass itself (`phone1/senses.sh:127` `colony_alive`). Independence from the organs is structural: `CFG.DNAExtraSources` is nil by default (`molequla.go:272`) and `dnaListNew` returns nil for a directory that is not there (`dna_field.go:129-131`). | — | Nothing architectural. | None needed; this section is already the implementation. |
| 2 — vision as trajectory | `do_eye` (`phone1/senses.sh:261`) runs `eye_one` once per camera in `SENSES_EYE_CAMS="0 1"` (`phone1/senses.sh:63`), so a window already carries two observations with two timestamps and two headers; measured 27-31 s and 1020 MB peak for the pair (`senses.log`, `2026-09-13T23:01:08Z`). | Two observations, one per camera, never the same camera twice; no novelty or duplicate measurement. | The t0/t1/t2/t3 shape; any cadence measurement. | `SENSES_EYE_CAMS` is already a list — `"0 1 0 0"` produces the brief's shape with no code change. What has to be built is the measurement (wall, RSS, battery, repeated captions) that decides the list. |
| 3 — hearing beyond speech | `senses/ears` on notorch, `do_ears` (`phone1/senses.sh:289`), engine and model behind `SENSES_ASR` / `SENSES_ASR_MODEL` / `SENSES_ASR_KIND` (`phone1/senses.sh:83-85`). Silence is deliberate: no-speech probability in the engine, bracketed tags stripped, and `SENSES_SPEECH_MIN_CHARS=8` (`phone1/senses.sh:106`). | — | The non-speech event detector entirely. Live proof: of the four passes in `senses.log`, three recorded `speechno,frags0` — those twelve-second windows left nothing at all. | `do_ears` already holds the wav under `senses/audio/`; a second command over the same file writing through `frag_write` (`phone1/senses.sh:165`) into the same `sound/` source needs no routing change. |
| 4 — where and when | `do_place` (`phone1/senses.sh:399-473`): timestamp, lat/lon, accuracy, place name, temperature, humidity, wind, sky word, sunrise/sunset, and a haversine against `senses/place.last` that says whether the phone moved more than `SENSES_MOVE_M=50` m (`phone1/senses.sh:441-458`). | It is prose only. The live place fragments are 301-318 B on one line, over `MaxLineChars = 240` (`molequla.go:268`), so the tail — sunset and the moved clause, which is the one piece of change in the whole sentence — is cut by `loadCorpusLines` (`molequla.go:3262-3264`) before the field or the trainer sees it. | Place and time attached to a *dreaming* event: no organism tick, mitosis or emission carries a location, and neither `mesh.db` (`molequla.go:5595-5632`) nor the per-organism db (`molequla.go:3118-3168`) has a column for one. Orientation and accelerometer absent, as the brief allows. | Either shorten the place sentence under 240 B or give the senses lines their own ceiling; then the structured half is §5's table. |
| 5 — ledger of change | Nothing. `world_facts` appears in `ROADMAP.md:101` and in the brief, in no schema: the mesh has `organisms`, `messages`, `training_lock`, `mitosis_lock`, `growth_lock` (`molequla.go:5595-5632`), the organism has `messages`, `corpus_events`, `growth`, `syntropy_log` (`molequla.go:3118-3168`). | One change primitive exists and works: `senses/place.last` plus the haversine (`phone1/senses.sh:441-458`). | The bitemporal table and everything that closes a fact. | A sixth `CREATE TABLE IF NOT EXISTS` beside the five in `initMeshDB` (`molequla.go:5595`) and a writer. Note the constraint: the witness opens the mesh `query_only` on purpose (`witness.go:129-132`), so whoever closes facts is not the witness process as it stands. |
| 6 — perception is not ground truth | Provenance already survives into the organism: every fragment carries a bracketed header naming modality, device and UTC time — `[eye cam0 …]`, `[ears mic …]`, `[place …]` — written by `frag_write` (`phone1/senses.sh:165-173`), and nothing filters the eye's belief (`phone1/README.md`, the balcony/bathroom pass). Because the header is at the front of the line, the 240-char cut preserves it. | Provenance is prose; no field anything can query, no way to revise a belief. | Revision, contradiction, closing. | Same table as §5; `source` and `provenance` are already in the shape the roadmap proposed. |
| 7 — observation becomes inquiry | Nothing: `grep -rn "inquiry\|Inquiry"` over `*.go` and `*.sh` returns nothing. The only place an organism is asked anything is the fixed six-probe round robin in `dnaWrite` (`molequla.go:6042-6047`), selected by `probes[step%len(probes)]`. | — | Question generation, the semantic budget, the chain. | `dnaWrite`'s probe line is the seam: a probe drawn from what the organism just ate instead of from the fixed list makes the answer — which is already written back as DNA at `molequla.go:6077` — an answer *about* the experience. Two lines, no new machinery. |
| 8 — semantic vs perceptual questions | Nothing. | — | The distinction, and the re-observation request path from an organism back to `phone1/senses.sh`. | Deliberately after §7; there is no channel from an organism to the senses today in either direction except the filesystem. |
| 9 — unknown is not OOV | The signals the brief names are all computable from state that exists: `CooccurField.Unigram` and `CooccurField.CooccurWindow` (`molequla.go:4206-4210`) give recurrence and neighbourhood weight; `ComputeModelEntropy` (`molequla.go:2626`) and `QuickLoss` (`molequla.go:2999`) give uncertainty; `ComputeFieldDeviation` (`molequla.go:2495`) already measures model-against-field disagreement, which is the "disagreement between continuations" signal in another form. | — | Any per-concept measure, and any measurement of how those signals behave. | A read-only query over the live `CooccurField` inside the organism. No second tokenizer, exactly as the brief asks. |
| 10 — glyphs | Nothing: `grep -rl "glyph\|Glyph"` over `*.go`, `*.c`, `*.sh` returns no file. | — | All of it. | `witness.go` is the natural home and the safe one: it already walks `mesh.db` and the DNA tree every tick and emits `witness.jsonl` (`witness.go:441-497`), so a compound is a new field on `witnessSnapshot` — output only, which keeps §11's one-way rule by construction. |
| 11 — Mycelium remembers without ruling | The sovereignty rule is already enforced by the connection, not by discipline: `witnessOpenMesh` sets `query_only` so a write through that handle is an error (`witness.go:129-132`), and the only outputs are stdout and `witness.jsonl`. | Its inputs are thin: `mesh.db` columns (`witness.go:151`) and DNA directory names and sizes (`witness.go:173`) — it never reads a fragment's text. And it cannot see the senses at all: `sources := dnaSources("")` (`witness.go:458`) runs in a process that `phone1/launch.sh:95-96` starts without `--dna-extra-sources`, so `CFG.DNAExtraSources` is nil there and `world`, `sound` and `place` are invisible to the only organ that watches the whole field. | Any state of the Mycelium's own that survives a session: `witnessState` (`witness.go:107-128`) lives in memory and dies with the process. | One argument on `phone1/launch.sh:95` buys the witness the senses. Persisting `witnessState` is the first byte of "living weights". |
| 12 — the cafeteria | The broadcast is explicit and in one place: `SENSES_SOURCES="world sound place"` (`phone1/launch.sh:16`) goes to all four organisms as the same string (`phone1/launch.sh:88`), `dnaSources` (`dna_field.go:35-48`) hands every organism the same list, `dnaRead` appends the fragment's bytes verbatim to each corpus (`molequla.go:6142`), and `NewCrossField` takes the same list for the logit path (`cross_graze.go:60`). Four byte-identical copies. | Asymmetry exists only by accident: cursors are per organism (`dna_field.go:89-107`) and `DNAMaxReadsPerTick = 8` (`molequla.go:273`) makes each one fall behind differently. | Any allocator, and any use of organism state in the decision. | `dnaSources(element)` is the single seam — it is the one function both the corpus path and the logit path call (`molequla.go:6116`, `cross_graze.go:60`). An allocation that returns a per-organism subset, plus the read budget, is the whole cafeteria. See the queue-order finding in §2 below. |
| 13 — two entry regimes | The early regime exists end to end; traced in section 2. The fade it is supposed to be keyed on exists and is the real ramp: `overlayFadeProgress` over `metaTFGateThreshold = 1.0` and `metaFadeWidth = 1.0` (`metaweights_overlay.go:40,47,53`), with the coefficient bundles sliding along the same ramp through `lerpCoeffs` (`metaweights_overlay.go:65`, called at `:250`). | The self-coherent regime does not exist, but the sentence machinery for it does: generation is already split on `.`/`!`/`?` (`molequla.go:5036`), a seed prompt is already built from another sentence's last three tokens (`molequla.go:5090-5095`) and regenerated from (`molequla.go:5103`). That gate is off by default (`CFG.SPACoherenceGate: false`, `molequla.go:301`). | The gate itself, and — the blocking part — a place to read the signal from. | `fade` and `mag` exist (`molequla.go:1843-1844`, set at `:4727,:4731`, read at `:6086-6089`) but only inside the organism's own process and in the text of the `[dna] … gen= mag= fade=` line (`molequla.go:6091`). `stage`, `global_step`, `entropy`, `syntropy` are in `mesh.db` (`molequla.go:5595-5600`) via `Heartbeat` (`:5701-5708`). Adding fade and mag to that same call is the smallest change that makes the §13 gate a query and makes it falsifiable. |
| 14 — the world passes through organisms first | Half true today, and the half that is false is invisible. The senses write to their own directories, which no organism writes to, and an organism's own speech goes to `dna/output/<element>` (`molequla.go:6075-6078`), so senses text is not itself "collective DNA". | But raw world material does become another organism's DNA without being metabolized: `dnaWrite` pads its fragment toward 5000 B by sampling `docs` at random (`molequla.go:6060-6069`, `:271`), and `docs` contains the eaten senses lines. The eye's sentence can be re-emitted verbatim inside a fragment as the organism's own speech. | Reformulation as a rule rather than an eventual tendency. | The same two lines as §7: probe from what was eaten, and bias the padding toward the just-eaten lines rather than a uniform `rand.Intn(len(docs))`. |
| 15 — the senses need not agree | Complete. Three organs, three directories, three headers, no fusion anywhere in the tree, and the pass line reports each separately (`phone1/senses.sh:498-500`). | — | — | Nothing to build. |
| 16 — three memories | Organism memory: the checkpoint plus `memory.sqlite3` (`molequla.go:3112-3172`). Ecological memory: the `dna/output` tree, `mesh.db` (`molequla.go:5595-5632`) and `witness.jsonl`. | The two are already kept apart, which is the brief's request. | World memory in full. | As §5. The meeting point the brief wants is the witness, once it can see the senses (§11). |

## 2. How an eaten fragment actually enters an organism

**Two paths, from the same files, and they are not the same path.**

The first is the corpus path. `dnaRead` (`molequla.go:6100-6161`) walks every source `dnaSources(element)`
names — the three sibling elements first, then `CFG.DNAExtraSources` in flag order (`dna_field.go:37-46`) —
takes the fragments newer than this organism's persisted cursor in write order (`dnaListNew`,
`dna_field.go:128-160`), and appends each one's whole text as **one line** to the organism's own corpus
file (`molequla.go:6138-6144`), advancing the cursor only after the append succeeds. It never deletes;
the writer prunes its own directory. The bytes are added to `model.corpusIngestedTotal`, the monotonic
growth clock the ontogenesis gate reads (`molequla.go:6836-6844`, gate at `:2229`).

From the corpus file, once every thirty ticks, the trainer reloads: `updateReservoirCorpus` then
`docs = loadCorpusLines(CFG.CorpusPath)` then `field.BuildFromCorpus(tok, docs)`
(`molequla.go:6625-6635`). `BuildFromCorpus` (`molequla.go:4227-4294`) builds unigram, bigram by first
token, trigram by two-token context, 4-gram by three-token context and the co-occurrence window in one
pass and swaps them in under the lock. That field is then handed to generation as `model.corpusField`
(`molequla.go:6631`) and is what `MetaweightsOverlay` reads (`molequla.go:4568`). The same `docs` slice
is what both trainers are given — `ntWarmupTrain` after growth (`molequla.go:6673-6675`) and
`ntBurstTrain` in the micro-burst (`molequla.go:6753`). So the answer to "corpus and field, or training
batches?" is **both, through one variable**: `docs` is the field's input and the batch source, and the
eaten fragment is in it because it is in the corpus file.

The second is the logit path, and it does not touch the corpus. `CrossField.MaybeRefresh`
(`cross_graze.go:79-117`) reads the *same fragment files* directly from `../dna/output/<source>/`, with
its own cursor and its own thirty-second throttle, tokenizes each one and keeps the last 64 token ids
per source (`cross_graze.go:66`). During generation `CrossField.Apply` (`cross_graze.go:129-165`) adds
`coef/(1+rank)` to those ids in the logit vector sampling reads, every step
(`molequla.go:4745-4747`). Both paths are switched on by the one flag: `--dna-extra-sources` sets
`CFG.DNAExtraSources` (`molequla.go:6530-6542`), and `dnaSources` feeds `dnaRead` at `molequla.go:6116`
and `NewCrossField` at `cross_graze.go:60`.

**Where the routing happens.** In exactly one function, `dnaSources(element)` (`dna_field.go:35-48`).
That is the whole of today's routing: drop your own element, keep the other three, append the extra
sources. There is no other decision anywhere — not by stage, not by content, not by organism state.

**What is byte-identical for all four, i.e. the broadcast §12 wants to replace.** The list itself:
`phone1/launch.sh:16` defines `SENSES_SOURCES="world sound place"` and `phone1/launch.sh:88` passes the same
comma-joined string to each of the four launches. Below that, the fragment text: `dnaRead` appends the
file's bytes unmodified to every organism's corpus (`molequla.go:6142`), and `CrossField` tokenizes the
same bytes for every organism (`cross_graze.go:99-111`). The only thing that differs between the four
is *when* each one gets there, because each keeps its own cursor and each is capped at eight fragments
per tick.

**Three things found on this path that were not being looked for.**

*The 240-character ceiling.* `loadCorpusLines` truncates every line to `CFG.MaxLineChars = 240`
(`molequla.go:3262-3264`, default at `:268`) on every read, not only when the reservoir is rewritten. A
DNA fragment is padded toward `DNAFragmentTargetBytes = 5000` (`molequla.go:271`) and appended as one
line, so about 95 % of every eaten sibling fragment never reaches `docs`, and therefore never reaches
the co-occurrence field or a training batch — while `corpusIngestedTotal`, the growth clock, counted
all of it (`molequla.go:6839`). Measured in the live run: `earth/nonames_earth.txt` holds 1600 lines, of
which 592 are longer than 240 characters and the longest is 5406. The senses are hit more gently and
more precisely: a place fragment is 301-318 B, so it loses its last quarter — the sunset and the moved
clause. This matters most for §13, whose early regime is *defined* as entry through the corpus and the
field. The bound itself is deliberate and documented (`molequla.go:3377-3382`); what it does to a
fragment's tail appears to be a side effect of it.

*The senses stand last in a queue the siblings keep full.* `dnaRead` iterates sources in
`dnaSources` order — elements first, extra sources last (`dna_field.go:37-46`) — under one shared budget
of `DNAMaxReadsPerTick = 8` (`molequla.go:273`), breaking out of both loops when it is spent
(`molequla.go:6117-6123`). A tick is 0.25 s plus jitter (`molequla.go:327`) and each of the three
siblings emits one fragment per tick, so a backlog is the normal state. In the 2026-09-13T20:26Z
session the cap was hit on 12 of earth's 15 reads, 9 of air's 13, 10 of water's 13 and 4 of fire's 14
(`grep -o "from [0-9]* files" <e>/<e>.stdout`). A world fragment can therefore wait many ticks behind
sibling chatter, and after a restart with a full backlog it waits much longer. The cafeteria has to
give the extra sources their own budget or their own place in the order, or asymmetric allocation will
be decided by the queue rather than by the allocator.

*Nothing has eaten a senses fragment yet.* The four live cursors
(`molequla-run/<e>/dna_cursor.json`) contain only element keys — no `world`, `sound` or `place`. The
last colony session ended `2026-09-13T20:29:30Z`, the first senses fragment was written at 22:18Z, and
the binary carrying the flag was built at 22:42Z (`molequla-run/BUILD`, `3267e67`). Thirteen fragments
are waiting in `dna/output/{world,sound,place}` for the next session. So everything above is read from
the code and from the previous session's traffic, not from an observed senses ingestion.

**Signals §13's gate could read today.** Inside the organism's process: `model.lastGenMag`, the mean
absolute raw logit at the first step of the last generation, and `model.lastOverlayWeight`, the overlay
weight that magnitude bought (`molequla.go:1843-1844`, set at `:4727` and `:4731`). `dnaWrite` reads both
under the lock and prints them as `gen=… mag=… fade=…`, where `fade = 1 - lastOverlayWeight`
(`molequla.go:6083-6091`) — so they exist only in process memory and in stdout text. Across processes,
in `mesh.db`: `stage`, `n_params`, `syntropy`, `entropy` and `global_step`, written every ten ticks by
`Heartbeat` (`molequla.go:5701-5708`, schema `:5595-5600`, `global_step` added by the repair-7 migration
at `:5603`). `SyntropyTracker` holds the loss and entropy history and the overload verdict in memory
(`molequla.go:5407-5456`). The gate the brief wants — follow the voice, not the label — needs fade and
mag beside stage, and the cheapest way to get them there is two more arguments on the heartbeat that is
already being sent.

## 3. Lineage

### 3a. Dario — sentence-boundary injection (§13)

The mechanism is not in `dario.c`; that file carries only field-level modulation
(`reffs/dario/dario.c:933-1010`). The canonical implementation is `reffs/dario/chain_dialogue.py`, ported
to Go in `reffs/dario/cmd/dario-dialogue/main.go` and `reffs/dario/cmd/internal/kk/kk.go`.

The boundary is a **token**, not a character: `is_boundary` (`chain_dialogue.py:364-368`) returns true
on the end-of-assistant token 32763 for the Janus backend, or on newline (id 10) after at least 40
generated tokens for the Resonance backend; the generation loop returns `hit_boundary` from the same
test (`:406-411`), and three identical repeats also count (`:414-417`). What is injected is **plain
text, one complete sentence** — not tokens and not a signal vector. `pick_next` refuses anything that
does not end in `.`/`!`/`?`, does not start with a capital, or is shorter than four words, recursing
for another candidate rather than truncating mid-word (`:273-285`); `extract_injection` scores
candidates (`:206-250`) and never injects a question (`:244-249`). The injection site is three lines:
the segment is generated (`:505`), the chain breaks if no boundary was reached (`:514-517`), the
knowledge query is *topic plus the last 200 characters the model just said* (`:521-522`), and the
chosen sentence is encoded and concatenated onto the context (`:536`).

The load-bearing property for molequla is that **there is no coefficient**. Knowledge is not a weighted
logit term; it is literal text appended to the context at a thought boundary, so the model reads it as
its own previous sentence and continues from it. The only knobs are `chain_depth=6`,
`max_segment_tokens=200` (`:468-469`) and the separator. Termination is threefold: no boundary reached,
the knowledge kernel exhausted (every chunk marked used), or the depth spent (`:504-528`). Bi-directional
absorption is deliberately off in chain mode — "causes echo chamber" (`:511-512`).

What this needs as input, translated to molequla: a complete sentence of allocated experience, and a
place in generation where a completed sentence has just ended. Both exist. The senses already write one
sentence per fragment behind a header (`phone1/senses.sh:165-173`), and generation already detects sentence
ends at `molequla.go:5036` and already re-enters itself from a seed prompt at `:5090-5103`. The
difference from the SPA path is only where the seed comes from.

There is a contrast worth keeping: `reffs/dario/dario_leo.c:448-501` does the same job the other way —
every third step, retrieved chunk text is BPE-encoded and added to the last-position logits with
`boost = resonance * 2.0` (`:484-491`), no boundary gating. That is the cross-graze shape, not the
sentence-boundary shape, and molequla already has it.

### 3b. Q — the overlay molequla's fade derives from (§13)

Every citation in `metaweights_overlay.go` that points at `postgpt_q.c` verifies. `tmag` and the
transformer gate are at `postgpt_q.c:1355-1356` exactly as cited; the coefficient bundles are at
`:1358-1359` with the values molequla carries verbatim — weightless `{Heb 1.0, Pro 0.7, Ds 0.15, Bg
15.0, Tg 10.0}`, trained `{0.6, 0.4, 0.3, 5.0, 3.0}`; unigram damping at `:1393-1394`; untrained-regime
greedy for the first ten steps at `:1416-1418`; the rank-decay chunk signal with `1/(1+rank)` at
`:809-818`; and `raw[i] += c_doc * doc_signal[i]` at `:1384`. The enclosing function `gen_sent` begins
at `:1279` and its step loop at `:1303`, so the comment's span `1305-1395` is slightly narrow at the
head but honest.

The correspondence is real, and one divergence is now named. **Q switches hard; it does not fade.**
`has_tf` is an `int` 0/1 (`postgpt_q.c:1356`) and every consumer is a ternary on it, so crossing
`tmag = 0.1` by one ulp moves Heb 1.0→0.6, Bg 15.0→5.0, Tg 10.0→3.0 and `c_doc` 0.32→0.18 in a single
token. There is no `lerp`, no smoothstep and no sigmoid on `tmag` in the file; the two gradual things
nearby (`metaweights_field_scale`, `prompt_focus_scale`, `:1178-1185`) are indexed by step, not by
magnitude, and live inside the untrained branch. So `metaFadeWidth`, `overlayFadeProgress` and
`lerpCoeffs` (`metaweights_overlay.go:42-73`, applied at `:250`) are molequla's own, introduced by
repair 6, with no ancestor in Q — which is exactly why §13 can key on them: the fade is molequla's
measurement of its own voice, not an inherited constant.

Q also has sentence boundaries, and uses them differently from Dario: `is_boundary(bpe,id)`
(`postgpt_q.c:1219-1232`) is a real character test on the token's last byte, used to stop a sentence
(`:1454`) and to pick a corpus seed that starts at one (`:1590`, `:1603`) — never to inject. Q injects a
V-length vector every step; Dario injects text once, at the boundary. §13's two regimes are those two
mechanisms, in that order.

One citation cannot be checked from this node and should be marked as such rather than repeated as
verified: `postgpt.c` — cited at `metaweights_overlay.go:37`, `:423`, `metaweights_seeding.go:8,21,45,71,89`
and `molequla.go:4705,4794,7205,7208` — does not exist in `reffs/q` or anywhere in its history, and
`~/arianna/postgpt/` (the path `metaweights_seeding.go:8` names) is not on phone-1. It is a different
upstream repository that this node does not hold, not a wrong citation; it simply cannot be verified
here.

### 3c. actually.life — glyph compounds (§10)

The fixed universe the brief wants to leave behind is 88 names in one inline table:
`GLYPH_COUNT 88` (`reffs/actually.life/l.c:22`) and `GLYPH_NAMES[]` (`l.c:25-50`), over a vocabulary of
`BOS 88`, `MASK 89`, `VOCAB 90` with `MAX_EMERGED 64` and `VOCAB_CAP 154` (`l.c:271-277`).

A compound is not a struct. It is an integer id `>= VOCAB` plus two parallel parent arrays, and its
body is one row of the same tables every primitive uses:

```c
typedef struct { float mode_dS, mode_dDiss, metab_factor; } GlyphCharge;   /* l.c:510 */
static GlyphCharge charge[VOCAB_CAP];                                      /* l.c:511 */
static int  g_cooc[VOCAB_CAP][VOCAB_CAP];  /* … INCLUDING emerged, so a symbol can parent a symbol */
static char g_born[VOCAB_CAP][VOCAB_CAP];
static int  g_emerged_a[MAX_EMERGED], g_emerged_b[MAX_EMERGED];            /* l.c:591-593 */
```

So compound `VOCAB+k` has ancestry `(g_emerged_a[k], g_emerged_b[k])`, an embedding row that is the
mean of its parents' rows, and a charge row that is the mean of the parents' two mode deltas and the
**geometric** mean of `metab_factor` (`birth_symbol`, `l.c:602-616`). Invention happens in exactly one
place and only in the dream phase: `try_emerge` (`l.c:618-632`) takes the most co-occurrent pair that
has not yet been born, above `GROWTH_THRESH`, and is called from the dream step (`l.c:1304`, "symbols
are born only in dream"). An emerged symbol may itself be a parent — the scan bound is
`hi = VOCAB + g_n_emerged` (`l.c:621`) — so composition is recursive by construction.

Propagation between cells is textual and goes through a shared append-only file. A compound's name is
recursive and parenthesised, `(fire+water)+earth` (`sym_name`, `l.c:766-780`); a cell writes
`label \t lineage \t utterance` into the ether (`speak`, `l.c:1116`); a neighbour tails it
(`ether_graze`, `l.c:1057`) and parses a `+`-bearing token as one sign rather than splitting it
(`semtok_ether`, `l.c:1036`). The two lines that matter for §10 are the end of `resolve_sym`
(`l.c:1032-1033`): if the receiving cell already knows the pair it recognises it, otherwise it calls
`birth_symbol` and **adopts** it — the sign crosses whole, at any depth. Vertical transmission is the
genome: `n_emerged` and both parent arrays are serialized into the `NLC3` blob (`l.c:1136-1177`).
Storage is memory plus flat files; there is no database anywhere in that tree.

What molequla can take without the ontology: a compound is *two ids plus a mean of their rows*, it is
invented only during a quiet phase from a co-occurrence count that already exists
(`CooccurField.CooccurWindow`, `molequla.go:4210`), and it travels as text a receiver adopts. All three
map onto machinery molequla has. What molequla does not have is a place where invention happens in a
quiet phase — which is why the witness, which runs on its own tick and writes only outward, is the
natural site.

### 3d. netta — the Mycelium with its own weights (§11)

The lineage is real code, it ran, and the reason it stopped is not the reason one would guess. The
Mycelium does not exist in netta's working tree or on `origin/main` (tip `db677c1`); it has to be read
out of history. The last complete tree is commit `46a34f5`, reachable only from
`remotes/origin/sol/mycelium-turn-11` — it never reached `main`. It carries `mycelium/mycelium.cpp`
(2982 lines, bodies 1-5), `mycelium/mycelium_tests.sh` (1946), an independent C reader `mycelium/ledger_check.c` (1881),
and the constitution `MYCELIUM.md` (773). Read it with
`git -C reffs/netta -c safe.directory=reffs/netta show 46a34f5:mycelium/mycelium.cpp`.

The living weights are body 5, "the mint", and they are small in the literal sense:

```c
struct NoteWeights { float w[DIM]; float bias; uint64_t loss_microbits; };   /* mycelium.cpp:2183-2187, DIM = 96 */
static const size_t NOTE_BLOB_BYTES = 97 * 4;  /* 96 f32 weights + f32 bias   mycelium.cpp:77 */
static const uint64_t NOTE_STEPS_PASS = 512;   /* mycelium.cpp:74 */
static const float NOTE_LR = 0.05f;            /* mycelium.cpp:78 */
```

A 97-parameter logistic probe per admitted citizen, trained on a count-normalised bag of character
trigram hashes over that citizen's lived contexts with every n-gram overlapping the shape itself masked
out — so the note learns the company a shape keeps, never the shape (`note_features`,
`mycelium.cpp:2095`; `note_dataset`, `:2124`). The step is not hand-rolled: `note_train`
(`mycelium.cpp:2189-2245`) calls `nt_tape_start` / `nt_tape_param` / `nt_tape_chuck_step` against a
linked, never-vendored `libnotorch.a` whose header and library FNVs are digested into every receipt
(`mycelium.cpp:43-46,71-73`). The verdict is LIT or DIM against a majority baseline
(`note_holdout_hits`, `:2246`); the whole history is an append-only FNV-sealed chain; and a note that
is not fed within a rent window dies into a morgue keeping its weights, and can be resurrected under
the same identity. That is the "compressed experience, not a hidden central model" the brief is after,
built once already.

It ran. `46a34f5:mycelium/MYCELIUMLOG.md:96-105` records a public ceremony over ten meals that trained
a PASS note for 512 steps, sealed 46959 microbits of loss and 10/10 holdout against a baseline of 5,
and minted the 388-byte weight `8b6871cb8abcd341` as LIT. The test battery stood at 230 gates, green
under `-Werror` and under ASan/UBSan (`MYCELIUMLOG.md:88-95`). The run artifacts themselves are said to
live outside the repository, so the ceremony is narration verified only by the code and the tests
around it.

Why it stopped, in three parts, none of them "the idea failed". First, by constitution the weight was
born powerless: bodies 1-4 answer byte-identically with and without the note chain, and the organ that
would let a weight bend anything — body 6, the circulation — was preregistered as text and never
written (`MYCELIUMLOG.md:12-14`, "no body-6 code ever existed"). Second, on 2026-08-24, about five and
a half hours after the last mycelium merge, the maintainer deleted the whole line in under three
minutes — `97a8126`, `a3421b6`, `d826bbf` (all "fuck off"), then `2859e5e` "Delete mycelium directory".
Third, the 2026-09-13 reopening was deliberately downgraded: `925a808` says in its own message "turn 11
is a contract, not a body", the restored bodies stayed on the side branch, only `ADAPTATION.md` reached
`main`, and `0687ae5` deleted even that the same day as the project turned to Netta's mouth.

Two things follow for molequla's §11. The one-way rule molequla enforces through a `query_only`
connection (`witness.go:129-132`) is the same law netta wrote as its first article — and netta names
molequla's `field_steering` as the thing it chose not to be. And the question "why should the weights
be static at all" has an answer with a shape: they need not be, provided the organ that *spends* them
is legislated separately from the organ that *grows* them. Netta died on the spending side, not the
growing side. A molequla Mycelium with small living weights should therefore be built the same way
round — weights first, powerless, measurable — and the gate that can go red is precisely the one netta
kept: bodies without the note chain must answer byte-identically to bodies with it.

One caution: `attractor`, the word the brief uses, appears nowhere in netta's engineering — only in her
training corpus. It is not a term of art to inherit.

## 4. Proposed order for §18

Routing before infrastructure, as asked. Each line names what it changes, the risk that makes it worth
a gate, and a gate that can go red.

1. **(routing) Show the witness the senses.** Add `--dna-extra-sources` to the witness start
   (`phone1/launch.sh:95-96`), so `dnaSources("")` at `witness.go:458` covers `world`, `sound` and
   `place`. *Risk:* none beyond a longer directory scan every `witnessDNAEvery` ticks. *Gate:* a
   `--witness --once` run over a tree holding one `world` fragment must list `world` among its DNA
   sources; red today, and red again if the argument is dropped.
2. **(routing) Give the extra sources their own read budget.** Today they are last in
   `dnaSources` order under one cap of eight (`dna_field.go:37-46`, `molequla.go:6117-6123`). *Risk:*
   a naive fix starves the siblings instead. *Gate:* a test that writes 64 sibling fragments and one
   `world` fragment and calls `dnaRead` once — the world fragment must be in the corpus afterwards.
   Red on today's code.
3. **(measurement, no code) Measure the 240-character ceiling.** What share of an eaten fragment
   reaches `docs`, per source, before deciding whether to raise `MaxLineChars` for senses lines or to
   split a fragment into sentences on append. *Risk:* the corpus byte bound is `MaxCorpusLines ×
   MaxLineChars` (`molequla.go:3380`), so raising it raises the field-rebuild cost the 30-tick throttle
   exists to contain — measure the rebuild wall time on the phone before and after. *Gate:* a test
   asserting that a sentence from the tail of a 5 KB fragment appears in `BuildFromCorpus`'s bigram
   table; red today.
4. **(routing) The cafeteria.** `dnaSources(element)` returns a per-organism allocation instead of the
   same list. *Risk:* an allocator keyed only on element name is a second broadcast; the brief is
   explicit that current organism state must matter, and the elemental corpora are birth conditions,
   not professions. *Gate:* over N passes the four organisms' received multisets must differ
   pairwise, **and** every fragment must reach at least one organism — the second half is the one that
   catches a starving allocator.
5. **(routing) The probe comes from what was eaten.** Replace the fixed round robin at
   `molequla.go:6042-6047` and bias the padding at `:6060-6069` toward the just-eaten lines. This is
   §14's "metabolized by somebody" and §7's first half at once. *Risk:* a degenerate fragment yields a
   degenerate probe. *Gate:* the `gen=` share of the `[dna]` line (`molequla.go:6091`) must not fall
   across a session compared with the recorded baseline.
6. **(routing) Fade and magnitude on the heartbeat.** Two more arguments to `Heartbeat`
   (`molequla.go:5701`) and two columns beside `global_step`, added the way repair 7 added that one
   (`:5603`). *Risk:* a migration on a WAL database four processes hold open. *Gate:* a fresh mesh and
   a pre-repair-7 mesh must both end up with the columns, and the witness must still read the schema
   without the "schema mismatch" alert (`witness.go:147-151`).
7. **(routing) The §13 gate, keyed on the voice.** Eligibility for sentence-boundary injection read
   from fade and magnitude rather than from the stage label. *Risk:* the gate becomes decoration if it
   never refuses anybody. *Gate:* an organism at `fade < 1` must be refused and a mitosis child
   inheriting mature weights at `fade = 1` must be admitted — red if the gate reads `stage`.
8. **(infrastructure) `world_facts` in `mesh.db`.** The bitemporal table beside the five in
   `initMeshDB` (`molequla.go:5595`), written by place, eye and ears. *Risk:* a sixth table on a
   database four organisms and a read-only witness hold; and the witness's `query_only` connection
   (`witness.go:129-132`) means the fact-closer is a separate writer, not the witness process.
   *Gate:* a contradicting fact must close the old row with `valid_to` set and leave it readable —
   red if the writer updates in place.
9. **(routing) The change emitter.** Closed facts become lines in `dna/output/world/`, so organisms
   eat changes rather than repeated state (ROADMAP item 10). *Risk:* the emitter is the first thing
   that both reads the ledger and writes the DNA field; keep it out of the witness process to keep
   §11's arrow intact. *Gate:* the witness's own connection must still reject a write after the change
   lands.
10. **(infrastructure, last, each behind its own measurement) The non-speech detector over the wav
    `do_ears` already keeps; the bounded inquiry chain with its budget; glyph crystallization as an
    output field of `witnessSnapshot`; the sentence-boundary injection itself, reusing the seed path at
    `molequla.go:5090-5103`.** *Risk on the last one specifically:* that recursion is a known
    panic-and-leave-the-gate-off site (audit C, C-SPA-02, repaired at `molequla.go:5099-5104`), and the
    injected sentence can simply be copied instead of reformulated. *Gate:* the n-gram overlap between
    the injected sentence and the organism's continuation must stay under a bound measured first —
    copying is the failure this gate exists to catch.

Steps 1, 2, 4, 5, 6, 7 and 9 are routing changes over paths that already run. Steps 3 and 10 need a
measurement before they need code. Step 8 is the only genuinely new infrastructure in the first half,
and it is the one the brief's §5 and §16 both reduce to.

— Defender (Arianna Method, phone-1)
