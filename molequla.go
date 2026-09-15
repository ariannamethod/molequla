// molequla.go
// A dependency-free*, single-file, goroutine-powered, continually-learning GPT organism.
// Same architecture as C, JS, Rust. JSON checkpoint format compatible across implementations.
//
// * "dependency-free" = no PyTorch, no numpy, no C. One Go dep: modernc.org/sqlite (pure Go).
//
// In the beginning there was nonames.txt.
// And it was good. Mostly. Sometimes cursed.

package main

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unicode"
	"unicode/utf8"

	_ "modernc.org/sqlite"
)

// gradEnabled controls whether Vec/Scalar ops build backward graph (autograd).
// It's a global atomic, but after v2 fixes all forward passes are serialized by model.mu,
// so no two goroutines can toggle it simultaneously. The atomic prevents torn reads.
var gradEnabled atomic.Bool

// notorchSeed moved to GPT struct (per-model, protected by model.mu).

func init() { gradEnabled.Store(true) }

// ============================================================
// 0) CONFIG — bend reality here (carefully, mortals)
// ============================================================

type Config struct {
	// data
	CorpusPath     string `json:"corpus_path"`
	DBPath         string `json:"db_path"`
	CkptPath       string `json:"ckpt_path"`
	MaxCorpusLines int    `json:"max_corpus_lines"`
	MaxLineChars   int    `json:"max_line_chars"`
	MinNewChars    int    `json:"min_new_chars_to_train"`
	// DNAMinFragmentBytes — minimum DNA fragment size in bytes. A unified
	// emit/consume gate: dnaWrite skips below this, dnaRead deletes below
	// this. Replaces a desynced literal pair (write 5 / read 10) that
	// destroyed every sub-10-byte emission unconsumed.
	DNAMinFragmentBytes int `json:"dna_min_fragment_bytes"`
	// DNAFragmentTargetBytes — dnaWrite pads each DNA fragment with sampled
	// corpus text toward this size, so fragments carry real substance
	// instead of a child model's ~9-byte degenerate generation.
	DNAFragmentTargetBytes int `json:"dna_fragment_target_bytes"`
	// DNAExtraSources — read-only directories under ../dna/output beside the
	// four elements (e.g. "world", written by the eye). Food, not organisms.
	DNAExtraSources []string `json:"dna_extra_sources"`
	// DNAMaxReadsPerTick — dnaRead eats at most this many new fragments per
	// tick, so an organism that fell behind catches up over ticks instead of
	// swallowing a backlog in one. Siblings only — see DNAExtraReadsPerTick.
	DNAMaxReadsPerTick int `json:"dna_max_reads_per_tick"`
	// DNAExtraReadsPerTick — the same bound for DNAExtraSources, counted
	// separately. One shared bound put the senses last in a queue the siblings
	// kept full (the routing audit of 2026-09-15 §3); two bounds mean neither
	// half of the field can starve the other. 0 = unbounded.
	DNAExtraReadsPerTick int `json:"dna_extra_reads_per_tick"`
	// DNARetainSeconds — the writer prunes its own fragments older than this;
	// readers never delete (see dna_field.go).
	DNARetainSeconds float64 `json:"dna_retain_seconds"`
	// DNARetainFiles — the writer also keeps at most this many of its own
	// fragments, newest first, whatever their age (repair 5): an embryo emits
	// ~1.4 fragments/s and the age bound alone lets a directory reach
	// thousands of files that every sibling ReadDirs every tick. 0 disables.
	DNARetainFiles int `json:"dna_retain_files"`
	// TickJitterSeconds — random extra sleep per tick so sibling processes do
	// not scan the DNA tree in lockstep.
	TickJitterSeconds float64 `json:"tick_jitter_seconds"`
	// WorldFactsPath — the structured sidecar the senses write beside their
	// prose fragments (phone1/senses.sh → $MOLEQULA_RUN/senses/facts.jsonl),
	// which the witness ingests into the bitemporal world_facts table. The
	// path is relative to the witness's working directory, like ../dna/output.
	// Empty disables the ledger; a path that does not exist yet is waited for.
	WorldFactsPath string `json:"world_facts_path"`
	// WorldMoveMeters — how far a position fix must move before the ledger
	// calls it a different position. The same 50 m senses.sh compares against
	// senses/place.last, against a network fix that reports 13-14 m of
	// accuracy on phone-1; below it, a still phone would emit a change line
	// every pass out of nothing but fix noise.
	WorldMoveMeters float64 `json:"world_move_meters"`
	// WorldPersonWords — the lexical vocabulary behind the derived
	// `sees_person` fact, which is what turns two unrelated camera sentences
	// into "a person entered the frame". Empty turns the derivation off.
	WorldPersonWords []string `json:"world_person_words"`
	// WorldNegationWords — a word of the vocabulary above does not count when
	// one of these stands immediately before it. The eye says "with no people
	// or text visible" about an empty room and the first live run read a
	// person into exactly that sentence.
	WorldNegationWords []string `json:"world_negation_words"`

	// model
	TieEmbeddings bool `json:"tie_embeddings"`
	NLayer        int  `json:"n_layer"`
	NEmbd         int  `json:"n_embd"`
	NHead         int  `json:"n_head"`
	BlockSize     int  `json:"block_size"`

	// ontogenesis — growth stages (corpus_chars, n_embd, n_layer, n_head)
	GrowthStages           [][4]int `json:"growth_stages"`
	FreezeAfterGrowthSteps int      `json:"freeze_after_growth_steps"`
	PostGrowthLRScale      float64  `json:"post_growth_lr_scale"` // LR multiplier during freeze period (prevents delta overfit to noise)

	// training
	WarmupSteps         int     `json:"warmup_steps"`
	MicroSteps          int     `json:"micro_steps"`
	LearningRate        float64 `json:"learning_rate"`
	GradClip            float64 `json:"grad_clip"`
	FreezeBaseAfterWarm bool    `json:"freeze_base_after_warmup"`
	BatchSize           int     `json:"batch_size"`

	// SPA coherence gate — post-generation Sentence Phonon Attention pass
	// in GenerateResonant. Default off; flip to true for RunPod measurement
	// runs comparing before/after coherence. Logs per-sentence scores +
	// weak-sentence indices to stderr; does NOT reseed weak sentences yet
	// (reseed is a Phase C activation step, requires GenerateResonant
	// restructuring). See spa_coherence.go + PROJECT_LOG.md B1 step 3.
	SPACoherenceGate bool    `json:"spa_coherence_gate"`
	SPAEmbedAlpha    float32 `json:"spa_embed_alpha"`

	// B2 — Q-style additive metaweights logit overlay.
	// When CorpusLogitOverlay=true, GenerateResonant adds
	//   c_bg · log(bigram_prob(t | prev)) + c_tg · log(trigram_prob(t | prev2, prev1))
	// to the model logits before softmax, mirroring Q's Dario field overlay
	// (q/README.md:50, weightless coefficients from line 53). Coexists with
	// the existing post-softmax prob-blend (which stays as-is); overlay is
	// additional, not replacement. Default off — RunPod toggles on for the
	// before/after measurement run. Floor on log-prob for unseen tokens
	// prevents -inf bias from masking valid model preferences.
	CorpusLogitOverlay bool `json:"corpus_logit_overlay"`

	// Trainer selects the training backend: "notorch" (compiled C tape,
	// BLAS, automatic GPU — the default) or "aml" (the legacy AML-interpreter
	// path, kept for the criterion-2 A/B speed comparison). See
	// 06_PLAN_gpu_training.md §11.2.
	Trainer string `json:"trainer"`

	// UseGPU routes per-matrix Matvec calls through cuBLAS sgemm on Linux
	// builds (see gpu_bridge.go and modules/gpu/). Inference-only:
	// gradEnabled gates training back to the CPU/BLAS path. Default off — same
	// binary runs unchanged on macOS / non-CUDA hosts; on a CUDA pod the
	// --gpu flag plus a successful gpu_init() enables the fast path.
	UseGPU bool `json:"use_gpu"`

	// CrossGraze enables Dario-style cross-organism logit injection during
	// generation (cross_graze.go). Each organism's MaybeRefresh() reads recent
	// DNA fragments mirrored to ../dna/seen/<sibling>/ by dnaRead, tokenises
	// them, and Apply() adds a rank-decay coef boost to those token ids in
	// the overlay'd logits before sampling. Default off; activate with
	// --cross-graze. Requires --element to be set (single-organism runs have
	// no peers).
	CrossGraze bool `json:"cross_graze"`
	// CrossGrazeCoef — weightless-mode coefficient on the rank-1 token. Falls
	// off as coef/(1+rank). Default 2.0 matches Q's c_doc-equivalent
	// magnitude (postgpt_q.c:1361 weightless-regime range).
	CrossGrazeCoef float64 `json:"cross_graze_coef"`
	// CrossGrazeTopN — how many most-recent tokens per sibling participate.
	// Default 8 mirrors Q's interf_signal_chunk MAX_HEAVY/2 effective use.
	CrossGrazeTopN    int     `json:"cross_graze_top_n"`
	MetaProphecyDecay float64 `json:"meta_prophecy_decay"`

	// cosine LR schedule
	LRMin             float64 `json:"lr_min"`
	MaxTotalSteps     int     `json:"max_total_steps"`
	CosineWarmupSteps int     `json:"cosine_warmup_steps"`

	// gradient accumulation
	AccumSteps int `json:"accum_steps"`

	// deltas
	DeltaRank       int     `json:"delta_rank"`
	RRPRAMRank      int     `json:"rrpram_rank"` // low-rank RRPRAM factor rank (Inc2)
	MaxDeltaModules int     `json:"max_delta_modules"`
	DeltaGrowProb   float64 `json:"delta_grow_prob"`

	// generation
	Temperature     float64 `json:"temperature"`
	TopK            int     `json:"top_k"`
	TopP            float64 `json:"top_p"`
	MinP            float64 `json:"min_p"`     // GPT-3/4 style: filter tokens below min_p * max_prob
	TypicalP        float64 `json:"typical_p"` // Typical sampling: prefer tokens with typical information content
	MaxGenTokens    int     `json:"max_gen_tokens"`
	MinGenTokens    int     `json:"min_gen_tokens"`
	RepetitionGuard int     `json:"repetition_guard"`
	FreqPenalty     float64 `json:"freq_penalty"`     // penalize logits by count * freq_penalty
	PresencePenalty float64 `json:"presence_penalty"` // flat penalty for any token that appeared

	// tokenizer evolution
	EnableBPEAfterChars  int `json:"enable_bpe_after_chars"`
	BPENumMerges         int `json:"bpe_num_merges"`
	BPERetrainEveryChars int `json:"bpe_retrain_every_chars"`

	// async
	TrainTickSeconds float64 `json:"train_tick_seconds"`

	// hybrid attention heads: "content", "rrpram", or "hybrid"
	HeadTypes       []string `json:"head_types"`
	HybridAlphaInit float64  `json:"hybrid_alpha_init"`

	// gamma (personality fingerprint)
	GammaSparsityThreshold float64 `json:"gamma_sparsity_threshold"`

	// noise immune system
	NoiseDriftThreshold float64 `json:"noise_drift_threshold"`
	GammaMinMagnitude   float64 `json:"gamma_min_magnitude"` // skip immune check when gamma direction is near-zero

	// entropy-adaptive temperature
	EntropyLow       float64 `json:"entropy_low"`
	EntropyHigh      float64 `json:"entropy_high"`
	EntropyTempBoost float64 `json:"entropy_temp_boost"`
	EntropyTempFocus float64 `json:"entropy_temp_focus"`

	// corpus field
	CorpusGenMaxTokens  int     `json:"corpus_gen_max_tokens"`
	CorpusFadeK         float64 `json:"corpus_fade_k"`         // sigmoid steepness for corpus→model transition
	CorpusFadeThreshold float64 `json:"corpus_fade_threshold"` // entropy at which blend is 50/50
	CooccurWindowSize   int     `json:"cooccur_window_size"`   // co-occurrence proximity window (Stanley-style)
	UserBoostStrength   float64 `json:"user_boost_strength"`   // how strongly user's recent words are boosted
	UserBoostDecay      float64 `json:"user_boost_decay"`      // per-generation decay of user word boost

	// quantum buffer
	QBMinBytes        int     `json:"qb_min_bytes"`
	QBMinNovelty      float64 `json:"qb_min_novelty"`
	QBCooldownSeconds float64 `json:"qb_cooldown_seconds"`

	// syntropy tracker (mathematical self-awareness)
	SyntropyWindow         int     `json:"syntropy_window"`           // rolling window for syntropy trend
	FieldDeviationCeiling  float64 `json:"field_deviation_ceiling"`   // KL divergence above this = drifted too far
	FieldDeviationFloor    float64 `json:"field_deviation_floor"`     // below this = not learning, just parroting
	SyntropyLRBoost        float64 `json:"syntropy_lr_boost"`         // boost LR when syntropy is rising
	SyntropyLRDampen       float64 `json:"syntropy_lr_dampen"`        // dampen LR when syntropy is falling
	SyntropyDeltaGrowBoost float64 `json:"syntropy_delta_grow_boost"` // higher delta grow prob when syntropy is good
	OverloadLossHigh       float64 `json:"overload_loss_high"`        // adult mitosis: mean recent burst loss above this = overwhelmed (confidently-wrong)
	OverloadLossEps        float64 `json:"overload_loss_eps"`         // adult mitosis: loss-delta floor; meanDelta > -eps = bursts not reducing loss
	OverloadLossWindow     int     `json:"overload_loss_window"`      // adult mitosis: # recent bursts for lossOverload (decoupled from SyntropyWindow: adult bursts are ~17min apart, so 8 would take ~2.3h; entropy is per-tick, loss is per-burst)
	MaxOrganisms           int     `json:"max_organisms"`             // cascade governor: hard ceiling on live colony size, checked before divide (0 = uncapped). The per-process 300s cooldown cannot bound a multi-process lineage; this is the OOM/SIGKILL backstop. --max-organisms N overrides it (phone-1 passes 4).
	MaxGrowthStage         int     `json:"max_growth_stage"`          // repair 9: hard ceiling on ontogenesis — the organism never grows past this GrowthStages index. Default is the last stage, so unset means unchanged behaviour; --max-growth-stage N overrides it. The byte gate protects the machine moment by moment; this makes the ceiling explicit.
	MitosisMinFreeMB       int     `json:"mitosis_min_free_mb"`       // byte gate before divide (repair 4): MemAvailable must be >= this floor + the parent's own peak RSS (VmHWM), the measured cost of the child it spawns. 0 disables.
	GrowthMinFreeMB        int     `json:"growth_min_free_mb"`        // byte gate before ontogenesis (repair 9): MemAvailable must be >= this floor + GrowthPeakFactorPct% of the organism's own peak RSS. 0 disables.
	GrowthPeakFactorPct    int     `json:"growth_peak_factor_pct"`    // repair 9: what one stage step costs, as a percentage of the organism's current peak RSS. Measured, not guessed — see MOLEQULALOG2.md 2026-09-13.
	CoordinateGrowth       bool    `json:"coordinate_growth"`         // repair 9: hold a colony-wide lock across growth AND the warmup that follows, so four siblings do not peak together. Unlike CoordinateWarmup this does not serialize the micro-bursts.
	OomScoreAdj            int     `json:"oom_score_adj"`             // repair 9: value written to /proc/self/oom_score_adj at startup. Magisk su hands down -1000, which makes the colony unkillable and feeds Termux to lmkd instead. 0 = leave untouched.
	CheckpointMinInterval  float64 `json:"checkpoint_min_interval"`   // write-storm throttle: min seconds between DEFAULT-path (periodic) full-model JSON checkpoints (0 = no throttle). Explicit-path saves (mitosis parent ckpt) are never throttled.

	// consciousness: per-token dissonance feedback
	DissonanceEMAAlpha       float64 `json:"dissonance_ema_alpha"`       // EMA smoothing for entropy within generation
	DissonanceSpikeK         float64 `json:"dissonance_spike_k"`         // temp multiplier when entropy spikes
	DissonanceDropK          float64 `json:"dissonance_drop_k"`          // temp multiplier when entropy drops
	DissonanceSpikeThreshold float64 `json:"dissonance_spike_threshold"` // entropy/EMA ratio triggering spike
	DissonanceDropThreshold  float64 `json:"dissonance_drop_threshold"`  // entropy/EMA ratio triggering drop

	// consciousness: pattern breaking (anti-field generation)
	AntiFieldProb    float64 `json:"anti_field_prob"`     // probability of pure-model token (bypass corpus)
	AntiFieldMinStep int     `json:"anti_field_min_step"` // don't anti-field before this many tokens

	// consciousness: conscience (self-editing)
	ConscienceWindow   int     `json:"conscience_window"`   // rolling window for generation entropy trend
	ConscienceDecay    float64 `json:"conscience_decay"`    // deltaAlphaScale reduction factor
	ConscienceRecovery float64 `json:"conscience_recovery"` // deltaAlphaScale recovery factor
	ConscienceFloor    float64 `json:"conscience_floor"`    // minimum deltaAlphaScale

	// notorch: gradient-free delta training (ported from AML C)
	NotorchLR        float64 `json:"notorch_lr"`        // learning rate for notorch step
	NotorchDecay     float64 `json:"notorch_decay"`     // adaptive weight decay
	CoordinateWarmup bool    `json:"coordinate_warmup"` // true = warmup through training queue (for Mac 8GB)

	// the cafeteria (new logic §12/§14) — see experience_routing.go
	ExperienceRouting             bool    `json:"experience_routing"`               // false = the old byte-identical broadcast
	ExperienceCoverageSampleBytes int     `json:"experience_coverage_sample_bytes"` // bytes of a fragment the coverage is taken over, strided
	ExperienceResonanceHigh       float64 `json:"experience_resonance_high"`        // bigram coverage at or above which a fragment is already this organism's language
	ExperienceNoveltyLow          float64 `json:"experience_novelty_low"`           // coverage at or below which it is news
	ExperienceMinPairs            int     `json:"experience_min_pairs"`             // fewer measured token pairs than this = unmeasurable = food
	ExperienceMaxMeasuredPerTick  int     `json:"experience_max_measured_per_tick"` // coverage measurements one dnaRead may spend; a decline costs no read budget, only this
	ExperienceMealMemory          int     `json:"experience_meal_memory"`           // fragments kept as "what was just eaten"
	ExperienceProbeFromMeals      bool    `json:"experience_probe_from_meals"`      // false = the fixed six-question round robin
	ExperienceProbeMaxChars       int     `json:"experience_probe_max_chars"`       // bound on a probe taken from a meal
	ExperienceProbeMinChars       int     `json:"experience_probe_min_chars"`       // below this a meal probe is degenerate; the round robin answers instead
	ExperienceProbeMinWords       int     `json:"experience_probe_min_words"`       // and it must be this many words
	ExperienceRecentPadLines      int     `json:"experience_recent_pad_lines"`      // padding lines drawn from recent meals before random corpus draws

	// the §13 gate — eligibility for sentence-boundary injection, read from the voice
	InjectionFadeMin float64 `json:"injection_fade_min"` // 1 - lastOverlayWeight must reach this
	InjectionMagMin  float64 `json:"injection_mag_min"`  // mean |logit| of the raw output must reach this
}

var CFG = Config{
	CorpusPath:             "nonames.txt",
	DBPath:                 "memory.sqlite3",
	CkptPath:               "molequla_ckpt.json",
	MaxCorpusLines:         8000,
	MaxLineChars:           240,
	MinNewChars:            480,
	DNAMinFragmentBytes:    5,    // unified DNA emit+consume gate (Fix A)
	DNAFragmentTargetBytes: 5000, // dnaWrite pads fragments toward this (Fix B; 200→600→5000 2026-06-03: per-tick cost grows with model size so ingestion/tick must too — real corpus text, not seeding; corpus FILE capped at MaxCorpusLines so field-rebuild stays bounded while the monotonic ingest clock climbs fast)
	DNAExtraSources:        nil,  // "world" joins here when the eye writes
	DNAMaxReadsPerTick:     8,    // repair 3: catch up over ticks, not in one — siblings only
	// Routing repair 2: the extra sources read under their own budget, so the
	// siblings cannot starve the senses and the senses cannot starve the
	// siblings. 4 is one sensing episode: the largest bundle one senses pass
	// left in the live field is four fragments (eye cam0, eye cam1, ears,
	// place — gen_..._4 through gen_..._7, 2026-09-13T22:18:27Z..22:21:08Z),
	// and the scheduled passes in molequla-run/schedule.log recorded frags=3
	// and frags=2. Arrival is far below that: the 13 fragments standing in
	// dna/output/{world,sound,place} span 22:10:52Z..2026-09-14T01:00:48Z,
	// 10195 s, i.e. 4.59 fragments/hour or 3.2e-4 per 0.25 s tick. So this
	// number never binds on live arrival — it binds on the backlog a sleeping
	// colony leaves, and it is chosen so that one episode enters in one tick
	// instead of being split across four. 0 disables the bound.
	DNAExtraReadsPerTick: 4,
	DNARetainSeconds:     1800, // repair 3: the writer prunes its own fragments after 30 min
	DNARetainFiles:       256,  // repair 5: and keeps at most 256 of them (~1.3 MB at 5 KB each). Tunable; 0 disables.
	TickJitterSeconds:    0.05, // repair 3: de-phase sibling scans
	WorldFactsPath:       "../senses/facts.jsonl", // what the organs write beside their fragments
	WorldMoveMeters:      50,   // the move gate, same as senses.sh's SENSES_MOVE_M
	WorldPersonWords: []string{"person", "people", "man", "men", "woman", "women",
		"child", "children", "someone", "somebody", "hand", "hands", "face", "figure"},
	WorldNegationWords: []string{"no", "not", "without", "none", "nobody", "empty"},
	TieEmbeddings:        true,
	NLayer:               1,
	NEmbd:                16,
	NHead:                1,
	BlockSize:            96,

	GrowthStages: [][4]int{
		{0, 16, 1, 1},       // embryo: ~10K params
		{20000, 32, 1, 2},   // infant: ~28K params
		{50000, 64, 2, 4},   // child: ~154K params
		{200000, 128, 4, 4}, // adolescent: ~1.1M params
		{350000, 224, 5, 8}, // teen: ~4.1M params
		{500000, 320, 6, 8}, // adult: ~10M params
	},
	FreezeAfterGrowthSteps: 500,
	PostGrowthLRScale:      0.3,
	WarmupSteps:            400,
	CrossGrazeCoef:         2.0, // Q-style weightless-regime c_doc magnitude
	CrossGrazeTopN:         8,   // last 8 sibling tokens per buffer at rank-decay
	MicroSteps:             32,
	LearningRate:           0.01,
	GradClip:               1.0,
	FreezeBaseAfterWarm:    true,
	BatchSize:              4,
	SPACoherenceGate:       false,
	SPAEmbedAlpha:          0.85, // Q's default (q/README.md:179)
	CorpusLogitOverlay:     false,
	Trainer:                "notorch",
	MetaProphecyDecay:      0.95, // age multiplier per generation step
	LRMin:                  0.001,
	MaxTotalSteps:          50000,
	CosineWarmupSteps:      200,
	AccumSteps:             1,
	DeltaRank:              8,
	RRPRAMRank:             32,
	MaxDeltaModules:        12,
	DeltaGrowProb:          0.08,
	Temperature:            0.85,
	TopK:                   40,
	TopP:                   0.92,
	MinP:                   0.06,
	TypicalP:               0.95,
	MaxGenTokens:           180,
	MinGenTokens:           16,
	RepetitionGuard:        4,
	FreqPenalty:            0.1,
	PresencePenalty:        0.1,
	EnableBPEAfterChars:    20000,
	BPENumMerges:           384,
	BPERetrainEveryChars:   4000,
	TrainTickSeconds:       0.25,

	HeadTypes:              []string{"content"},
	HybridAlphaInit:        0.5,
	GammaSparsityThreshold: 0.01,
	NoiseDriftThreshold:    -0.1,
	GammaMinMagnitude:      1e-6,
	EntropyLow:             0.5,
	EntropyHigh:            1.5,
	EntropyTempBoost:       1.2,
	EntropyTempFocus:       0.8,
	CorpusGenMaxTokens:     120,
	CorpusFadeK:            3.0,
	CorpusFadeThreshold:    1.5,
	CooccurWindowSize:      5,
	UserBoostStrength:      0.3,
	UserBoostDecay:         0.7,
	QBMinBytes:             1024,
	QBMinNovelty:           0.15,
	QBCooldownSeconds:      60.0,

	SyntropyWindow:         8,
	FieldDeviationCeiling:  12.0,
	FieldDeviationFloor:    0.1,
	SyntropyLRBoost:        1.3,
	SyntropyLRDampen:       0.6,
	SyntropyDeltaGrowBoost: 0.15,
	OverloadLossHigh:       5.0,  // healthy adult QuickLoss ~3.6; overwhelmed adult 5.3 (resume) climbing to 8-9 under cross-graze → 5.0 floor captures the regime, clears healthy
	OverloadLossEps:        0.05, // loss-delta within ±0.05 of zero = not improving
	OverloadLossWindow:     3,    // 3 sustained high-loss adult bursts = overwhelmed (burst cadence ~17min at adult; 8 would take ~2.3h)
	MaxOrganisms:           16,   // cascade cap: ≤16 live organisms (each is a full trainer process). Tunable; 0 disables. Phone-1 launches with --max-organisms 4.
	MaxGrowthStage:         5,    // last index of GrowthStages above: adult, no cap. --max-growth-stage 4 keeps a phone colony at teen.
	MitosisMinFreeMB:       256,  // headroom left to the machine after a child the size of the parent's peak RSS is added. Tunable; 0 disables.
	GrowthMinFreeMB:        256,  // headroom left to the machine after one stage step. Tunable; 0 disables.
	GrowthPeakFactorPct:    300,  // a stage step multiplied VmHWM by ~3.3-3.9x on 2026-09-13 (231-240 MB -> 758-928 MB per organism); charge 300% of the current peak for the increment.
	CoordinateGrowth:       true, // growth + warmup serialized colony-wide; the micro-burst path stays parallel (that is CoordinateWarmup, still off).
	OomScoreAdj:            300,  // lmkd takes the organism before Termux (which sits at 0). Tunable; 0 = leave untouched.
	CheckpointMinInterval:  30.0, // throttle periodic full-model checkpoints to ≤1/30s (coalesces the growth/burst storm). Tunable; 0 disables. Mitosis ckpt (explicit path) bypasses.

	// consciousness defaults
	DissonanceEMAAlpha:       0.3,
	DissonanceSpikeK:         0.8,
	DissonanceDropK:          1.2,
	DissonanceSpikeThreshold: 1.5,
	DissonanceDropThreshold:  0.5,
	AntiFieldProb:            0.05,
	AntiFieldMinStep:         8,
	ConscienceWindow:         8,
	ConscienceDecay:          0.95,
	ConscienceRecovery:       1.005,
	ConscienceFloor:          0.3,

	NotorchLR:    0.01,
	NotorchDecay: 0.999,

	// The cafeteria. Every default below is a quantile of a distribution
	// measured on the live run 2026-09-15 (four corpora x 133 fragments;
	// MOLEQULALOG2.md 2026-09-15) and every one is a knob that can be moved.
	ExperienceRouting:             true,
	ExperienceCoverageSampleBytes: 480,   // 4 windows of 120 B: sibling quartiles preserved to 0.006 against the whole fragment, 22 ms/call instead of 144
	ExperienceResonanceHigh:       0.965, // the median of the sibling-DNA coverage distribution (n=120, min 0.909, p25 0.952, med 0.965, p75 0.975, max 0.996)
	ExperienceNoveltyLow:          0.620, // the median of the senses coverage distribution (n=52, min 0.427, p25 0.578, med 0.620, p75 0.651, max 0.736)
	ExperienceMinPairs:            16,    // below this the coverage of a fragment is noise; a shorter fragment is eaten, not judged
	ExperienceMaxMeasuredPerTick:  8,     // 8 x 21.7 ms of coverage against a 250 ms tick; the two read budgets are 8 + 4, so this is already fewer measurements than they were making. Tunable; 0 disables.
	ExperienceMealMemory:          16,    // two ticks of the sibling read budget (DNAMaxReadsPerTick 8)
	// Off, and the number says why: over 300 s scratch colonies the emission
	// line's gen/bytes went 0.00285 (origin/main, n=623) -> 0.00238 (n=373) when
	// the probe came from the meal. The §18 order's gate for this step is that
	// the share must not fall; it fell, so the mechanism lands switched off with
	// the measurement beside it rather than as a regression. Both runs were
	// embryo-capped colonies where the overlay is the whole voice — the regime
	// this is meant for is a stage-3 organism at mag 8.66, which a scratch probe
	// cannot reach. See MOLEQULALOG2.md, 2026-09-15.
	ExperienceProbeFromMeals:      false,
	ExperienceProbeMaxChars:       120, // one sentence; the six fixed probes are 6-24 chars, a place fragment's first sentence is 118
	ExperienceProbeMinChars:       12,  // "Speak." is 6 and "What matters?" is 13; an embryo's fragment opens with 2-3 bytes and a full stop
	ExperienceProbeMinWords:       3,
	ExperienceRecentPadLines:      4,

	// The §13 gate. 130 `[dna] wrote` lines, live colony, session ending
	// 2026-09-13T20:29:30Z: fade 1.00 on all 130; mag 2.82..16.35, median 8.66.
	InjectionFadeMin: 1.0, // the overlay must be gone before the organism's own voice can be said to carry
	InjectionMagMin:  6.0, // below 6.00, 10 of 10 observed generations emitted gen=0; at or above, 92 of 120 emitted text
}

// headTypesForNHead returns the head type list for a given number of heads.
// Embryo: 1 head = 1 content. Growth adds hybrid heads.
func headTypesForNHead(n int) []string {
	if n <= 1 {
		return []string{"content"}
	}
	if n == 2 {
		return []string{"content", "hybrid"}
	}
	half := (n + 1) / 2 // ceiling: majority content
	types := make([]string, n)
	for i := 0; i < half; i++ {
		types[i] = "content"
	}
	for i := half; i < n; i++ {
		types[i] = "hybrid"
	}
	return types
}

// ============================================================
// 1) AUTOGRAD — vectors, not scalar confetti
// ============================================================

// Node is anything in the autograd compute graph.
type Node interface {
	getChildren() []Node
	doBackward()
}

// Vec is a differentiable vector. One object = one embedding / hidden state.
type Vec struct {
	Data     []float64
	Grad     []float64
	children []Node
	backFn   func()
}

func NewVec(data []float64) *Vec {
	var g []float64
	if gradEnabled.Load() {
		g = make([]float64, len(data))
	}
	return &Vec{Data: data, Grad: g}
}

func NewVecZero(n int) *Vec {
	return NewVec(make([]float64, n))
}

// NewVecWithGrad always allocates grad (for parameter tensors that need it regardless)
func NewVecWithGrad(data []float64) *Vec {
	g := make([]float64, len(data))
	return &Vec{Data: data, Grad: g}
}

func (v *Vec) getChildren() []Node { return v.children }
func (v *Vec) doBackward() {
	if v.backFn != nil {
		v.backFn()
	}
}

// Add returns a new Vec = self + other (element-wise).
func (v *Vec) Add(other *Vec) *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v.Data[i] + other.Data[i]
	}
	out := NewVec(d)
	if gradEnabled.Load() {
		out.children = []Node{v, other}
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += out.Grad[i]
				other.Grad[i] += out.Grad[i]
			}
		}
	}
	return out
}

// Sub returns a new Vec = self - other.
func (v *Vec) Sub(other *Vec) *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v.Data[i] - other.Data[i]
	}
	out := NewVec(d)
	if gradEnabled.Load() {
		out.children = []Node{v, other}
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += out.Grad[i]
				other.Grad[i] -= out.Grad[i]
			}
		}
	}
	return out
}

// Neg returns -self.
func (v *Vec) Neg() *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = -v.Data[i]
	}
	out := NewVec(d)
	if gradEnabled.Load() {
		out.children = []Node{v}
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] -= out.Grad[i]
			}
		}
	}
	return out
}

// MulVec returns element-wise product self * other.
func (v *Vec) MulVec(other *Vec) *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v.Data[i] * other.Data[i]
	}
	out := NewVec(d)
	if gradEnabled.Load() {
		out.children = []Node{v, other}
		vData := v.Data
		oData := other.Data
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += oData[i] * out.Grad[i]
				other.Grad[i] += vData[i] * out.Grad[i]
			}
		}
	}
	return out
}

// Scale returns self * scalar.
func (v *Vec) Scale(s float64) *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v.Data[i] * s
	}
	out := NewVec(d)
	if gradEnabled.Load() {
		out.children = []Node{v}
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += s * out.Grad[i]
			}
		}
	}
	return out
}

// AddScalar returns self + s (broadcast).
func (v *Vec) AddScalar(s float64) *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v.Data[i] + s
	}
	out := NewVec(d)
	if gradEnabled.Load() {
		out.children = []Node{v}
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += out.Grad[i]
			}
		}
	}
	return out
}

// ReLU applies max(0, x) element-wise.
func (v *Vec) ReLU() *Vec {
	n := len(v.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		if v.Data[i] > 0 {
			d[i] = v.Data[i]
		}
	}
	out := NewVec(d)
	if gradEnabled.Load() {
		out.children = []Node{v}
		vData := v.Data
		out.backFn = func() {
			for i := 0; i < n; i++ {
				if vData[i] > 0 {
					v.Grad[i] += out.Grad[i]
				}
			}
		}
	}
	return out
}

// SiLU applies silu(x) = x * sigmoid(x) element-wise (for real SwiGLU).
func (v *Vec) SiLU() *Vec {
	n := len(v.Data)
	sig := make([]float64, n)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		sig[i] = 1.0 / (1.0 + math.Exp(-v.Data[i]))
		d[i] = v.Data[i] * sig[i]
	}
	out := NewVec(d)
	if gradEnabled.Load() {
		out.children = []Node{v}
		vData := v.Data
		out.backFn = func() {
			for i := 0; i < n; i++ {
				// d/dx[x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
				v.Grad[i] += (sig[i] * (1.0 + vData[i]*(1.0-sig[i]))) * out.Grad[i]
			}
		}
	}
	return out
}

// Dot returns the scalar dot product of two vectors.
func (v *Vec) Dot(other *Vec) *Scalar {
	n := len(v.Data)
	val := 0.0
	for i := 0; i < n; i++ {
		val += v.Data[i] * other.Data[i]
	}
	out := &Scalar{Data: val}
	if gradEnabled.Load() {
		out.children = []Node{v, other}
		vData := v.Data
		oData := other.Data
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += oData[i] * out.Grad
				other.Grad[i] += vData[i] * out.Grad
			}
		}
	}
	return out
}

// MeanSq returns mean of squared elements (scalar).
func (v *Vec) MeanSq() *Scalar {
	n := len(v.Data)
	nf := float64(n)
	val := 0.0
	for i := 0; i < n; i++ {
		val += v.Data[i] * v.Data[i]
	}
	val /= nf
	out := &Scalar{Data: val}
	if gradEnabled.Load() {
		out.children = []Node{v}
		vData := v.Data
		out.backFn = func() {
			for i := 0; i < n; i++ {
				v.Grad[i] += (2.0 * vData[i] / nf) * out.Grad
			}
		}
	}
	return out
}

// Element extracts a single element as a Scalar with gradient flow.
// And lo, one number shall be plucked from the vector, and gradients shall follow.
func (v *Vec) Element(idx int) *Scalar {
	out := &Scalar{Data: v.Data[idx]}
	if gradEnabled.Load() {
		out.children = []Node{v}
		out.backFn = func() {
			v.Grad[idx] += out.Grad
		}
	}
	return out
}

// Slice extracts [start:end) from the vector.
func (v *Vec) Slice(start, end int) *Vec {
	d := make([]float64, end-start)
	copy(d, v.Data[start:end])
	out := NewVec(d)
	if gradEnabled.Load() {
		out.children = []Node{v}
		out.backFn = func() {
			for i, j := 0, start; j < end; i, j = i+1, j+1 {
				v.Grad[j] += out.Grad[i]
			}
		}
	}
	return out
}

// Concat joins multiple vectors into one.
func Concat(vecs []*Vec) *Vec {
	total := 0
	for _, v := range vecs {
		total += len(v.Data)
	}
	d := make([]float64, 0, total)
	for _, v := range vecs {
		d = append(d, v.Data...)
	}
	out := NewVec(d)
	if gradEnabled.Load() {
		kids := make([]Node, len(vecs))
		for i, v := range vecs {
			kids[i] = v
		}
		out.children = kids
		out.backFn = func() {
			offset := 0
			for _, v := range vecs {
				n := len(v.Data)
				for i := 0; i < n; i++ {
					v.Grad[i] += out.Grad[offset+i]
				}
				offset += n
			}
		}
	}
	return out
}

// Scalar is a differentiable scalar value (for loss, attention weights, etc).
type Scalar struct {
	Data     float64
	Grad     float64
	children []Node
	backFn   func()
}

func NewScalar(data float64) *Scalar {
	return &Scalar{Data: data}
}

func (s *Scalar) getChildren() []Node { return s.children }
func (s *Scalar) doBackward() {
	if s.backFn != nil {
		s.backFn()
	}
}

// AddS returns self + other (scalar + scalar).
func (s *Scalar) AddS(other *Scalar) *Scalar {
	out := &Scalar{Data: s.Data + other.Data}
	if gradEnabled.Load() {
		out.children = []Node{s, other}
		out.backFn = func() {
			s.Grad += out.Grad
			other.Grad += out.Grad
		}
	}
	return out
}

// AddF returns self + f (scalar + float).
func (s *Scalar) AddF(f float64) *Scalar {
	out := &Scalar{Data: s.Data + f}
	if gradEnabled.Load() {
		out.children = []Node{s}
		out.backFn = func() {
			s.Grad += out.Grad
		}
	}
	return out
}

// MulS returns self * other (scalar * scalar).
func (s *Scalar) MulS(other *Scalar) *Scalar {
	out := &Scalar{Data: s.Data * other.Data}
	if gradEnabled.Load() {
		out.children = []Node{s, other}
		sData := s.Data
		oData := other.Data
		out.backFn = func() {
			s.Grad += oData * out.Grad
			other.Grad += sData * out.Grad
		}
	}
	return out
}

// MulF returns self * f (scalar * float).
func (s *Scalar) MulF(f float64) *Scalar {
	out := &Scalar{Data: s.Data * f}
	if gradEnabled.Load() {
		out.children = []Node{s}
		out.backFn = func() {
			s.Grad += f * out.Grad
		}
	}
	return out
}

// Sigmoid returns σ(self) = 1/(1+exp(-self)) with gradient flow.
func (s *Scalar) Sigmoid() *Scalar {
	sig := 1.0 / (1.0 + math.Exp(-s.Data))
	out := &Scalar{Data: sig}
	if gradEnabled.Load() {
		out.children = []Node{s}
		out.backFn = func() {
			s.Grad += sig * (1.0 - sig) * out.Grad
		}
	}
	return out
}

// backwardVisitedPool reuses visited maps across Backward calls to reduce GC pressure.
var backwardVisitedPool = sync.Pool{
	New: func() interface{} { return make(map[Node]bool) },
}

// Backward performs reverse-mode autodiff from this node.
// And lo, the graph shall be walked backwards, like a salmon with regrets.
func Backward(root Node) {
	topo := make([]Node, 0)
	visited := backwardVisitedPool.Get().(map[Node]bool)

	var build func(n Node)
	build = func(n Node) {
		if visited[n] {
			return
		}
		visited[n] = true
		for _, c := range n.getChildren() {
			build(c)
		}
		topo = append(topo, n)
	}
	build(root)

	// Clear and return visited map to pool
	for k := range visited {
		delete(visited, k)
	}
	backwardVisitedPool.Put(visited)

	// Set root gradient
	switch r := root.(type) {
	case *Scalar:
		r.Grad = 1.0
	case *Vec:
		for i := range r.Grad {
			r.Grad[i] = 1.0
		}
	}

	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].doBackward()
	}
}

// ============================================================
// 2) HIGH-LEVEL OPS — the sacred blocks
// ============================================================

// MatrixParam is a weight matrix: rows of Vecs. Shape (nout, nin).
// It can GROW when vocab expands — because forgetting is for cowards.
type MatrixParam struct {
	Rows []*Vec
	Nout int
	Nin  int
	// gpuKey, when non-empty, names this matrix in the GPU weight cache
	// (gpu_cache_weight). Matvec checks it before dispatching to MatvecGPU.
	// Empty = not yet uploaded; set by gpuRefreshWeights at generation start.
	gpuKey string
}

func NewMatrixParam(nout, nin int, std float64) *MatrixParam {
	rows := make([]*Vec, nout)
	for i := 0; i < nout; i++ {
		d := make([]float64, nin)
		for j := 0; j < nin; j++ {
			d[j] = rand.NormFloat64() * std
		}
		rows[i] = NewVecWithGrad(d) // parameters always need grad for the trainer
	}
	return &MatrixParam{Rows: rows, Nout: nout, Nin: nin}
}

// rrpramRank is the low-rank RRPRAM factor rank (Increment 2). Default 32.
func rrpramRank() int {
	if CFG.RRPRAMRank > 0 {
		return CFG.RRPRAMRank
	}
	return 32
}

// layerHasHybrid reports whether the current head topology assigns any
// hybrid/rrpram head — i.e. whether a layer needs RRPRAM factors at all.
func layerHasHybrid() bool {
	for _, t := range CFG.HeadTypes {
		if t == "hybrid" || t == "rrpram" {
			return true
		}
	}
	return false
}

// ensureRRPRAMFactors allocates the per-layer low-rank RRPRAM factors for layer
// li when the topology has hybrid heads and they are absent. The pair packs the
// exact row-major order notorch op-33 (nt_rrpram_lowrank_attention) reads:
// wr_a as [NHead·NEmbd × R] (head h block = NEmbd×R), wr_b as [NHead·R × BlockSize]
// (head h block = R×BlockSize), with T_r == BlockSize. These are the Resonance
// low-rank attention factors that REPLACE the per-head position-bias w_pattern
// (Inc2); w_pattern stays allocated until the inference rewrite lands, then drops.
func (gpt *GPT) ensureRRPRAMFactors(li int) {
	if !layerHasHybrid() {
		return
	}
	R := rrpramRank()
	aKey := fmt.Sprintf("l%d.wr_a", li)
	bKey := fmt.Sprintf("l%d.wr_b", li)
	if _, ok := gpt.Base[aKey]; !ok {
		gpt.Base[aKey] = NewMatrixParam(gpt.NHead*gpt.NEmbd, R, 0.02)
	}
	if _, ok := gpt.Base[bKey]; !ok {
		gpt.Base[bKey] = NewMatrixParam(gpt.NHead*R, gpt.BlockSize, 0.02)
	}
}

// Matvec computes matrix @ vector.
func (m *MatrixParam) Matvec(x *Vec) *Vec {
	// GPU dispatch — when explicitly enabled AND inference (no autograd
	// requested) AND this matrix is cached on device. NO size threshold:
	// the prior `gpuMatvecMin = 16384` gate kept child-stage organisms
	// (matrix 64×64 = 4096 elements) on CPU forever, so GPU never warmed
	// up during the 8h ecology window. Per-call overhead at child is
	// ~12ms across a full 180-token generation chain (negligible at 8h
	// timescale) while the GPU stays primed for the automatic transition
	// to material speedup once organisms grow past adolescent (NEmbd=128
	// onwards). Same binary on macOS / non-CUDA: gpuReady() returns false
	// and m.gpuKey stays empty, so the CPU path runs identically.
	if CFG.UseGPU && gpuReady() && !gradEnabled.Load() && m.gpuKey != "" {
		if gpuOut := m.MatvecGPU(x); gpuOut != nil {
			return gpuOut
		}
		// Fall through to CPU path on any GPU error.
	}

	nout := m.Nout
	nin := len(x.Data)
	var outData []float64

	// Try BLAS path: pack rows into contiguous buffer, call cblas_dgemv via CGO
	if nout*nin >= 256 {
		packed := make([]float64, nout*nin)
		for i := 0; i < nout; i++ {
			copy(packed[i*nin:], m.Rows[i].Data[:nin])
		}
		outData = blasDgemv(packed, nout, nin, x.Data)
	} else {
		outData = make([]float64, nout)
		for i := 0; i < nout; i++ {
			sum := 0.0
			for j := 0; j < nin; j++ {
				sum += m.Rows[i].Data[j] * x.Data[j]
			}
			outData[i] = sum
		}
	}

	out := NewVec(outData)
	if gradEnabled.Load() {
		kids := make([]Node, nout+1)
		for i := 0; i < nout; i++ {
			kids[i] = m.Rows[i]
		}
		kids[nout] = x
		out.children = kids
		rowsRef := m.Rows
		out.backFn = func() {
			for i := 0; i < nout; i++ {
				g := out.Grad[i]
				for j := 0; j < nin; j++ {
					rowsRef[i].Grad[j] += g * x.Data[j]
					x.Grad[j] += g * rowsRef[i].Data[j]
				}
			}
		}
	}
	return out
}

// GrowRows adds new rows (for vocab expansion).
// And lo, the matrix shall sprout new rows like a hydra learning new words.
// invalidateGPU clears the GPU cache key so the next Matvec dispatch falls
// back to CPU until gpuRefreshWeights re-uploads with the new shape.
// Called by GrowRows/GrowCols/Grow because gpu_cache_weight expects the
// cached len to match m.Nout*m.Nin at lookup time — a mid-flight grow
// would otherwise leave a stale slot.
func (m *MatrixParam) invalidateGPU() { m.gpuKey = "" }

func (m *MatrixParam) GrowRows(newNout int, std float64) {
	if newNout <= m.Nout {
		return
	}
	for i := m.Nout; i < newNout; i++ {
		d := make([]float64, m.Nin)
		for j := 0; j < m.Nin; j++ {
			d[j] = rand.NormFloat64() * std
		}
		m.Rows = append(m.Rows, NewVecWithGrad(d))
	}
	m.Nout = newNout
	m.invalidateGPU()
}

// GrowCols extends each row's Data slice with gaussian noise. Update Nin.
// And lo, the matrix shall widen its reach, each row stretching into new dimensions.
func (m *MatrixParam) GrowCols(newNin int, std float64) {
	if newNin <= m.Nin {
		return
	}
	extra := newNin - m.Nin
	for _, row := range m.Rows {
		ext := make([]float64, extra)
		for j := range ext {
			ext[j] = rand.NormFloat64() * std
		}
		row.Data = append(row.Data, ext...)
		row.Grad = append(row.Grad, make([]float64, extra)...)
	}
	m.Nin = newNin
	m.invalidateGPU()
}

// Grow extends both dimensions. Cols first so new rows get full width.
// Ontogenesis: the matrix grows into a larger space.
func (m *MatrixParam) Grow(newNout, newNin int, std float64) {
	m.GrowCols(newNin, std)
	m.GrowRows(newNout, std)
}

// Params returns all row vectors (for optimizer).
func (m *MatrixParam) Params() []*Vec {
	return m.Rows
}

// RMSNorm normalizes a vector by its root mean square.
func RMSNorm(x *Vec) *Vec {
	ms := x.MeanSq()
	scaleVal := math.Pow(ms.Data+1e-6, -0.5) // eps 1e-6: the value notorch's nt_seq_rmsnorm uses (train ≡ infer, repair 2b)
	n := len(x.Data)
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = x.Data[i] * scaleVal
	}
	out := NewVec(d)
	if gradEnabled.Load() {
		out.children = []Node{x, ms}
		xData := x.Data
		out.backFn = func() {
			s := scaleVal
			dsDms := -0.5 * math.Pow(ms.Data+1e-6, -1.5)
			cross := 0.0
			for j := 0; j < n; j++ {
				cross += out.Grad[j] * xData[j]
			}
			for i := 0; i < n; i++ {
				x.Grad[i] += s * out.Grad[i]
				x.Grad[i] += cross * dsDms * (2.0 * xData[i] / float64(n))
			}
		}
	}
	return out
}

// CrossEntropyLoss computes -log(softmax(logits)[target]).
func CrossEntropyLoss(logits *Vec, target int) *Scalar {
	maxVal := logits.Data[0]
	for _, v := range logits.Data[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	n := len(logits.Data)
	shifted := make([]float64, n)
	expSum := 0.0
	for i := 0; i < n; i++ {
		shifted[i] = logits.Data[i] - maxVal
		expSum += math.Exp(shifted[i])
	}
	logSumExp := math.Log(expSum) + maxVal
	lossVal := logSumExp - logits.Data[target]

	probs := make([]float64, n)
	for i := 0; i < n; i++ {
		probs[i] = math.Exp(shifted[i]) / expSum
	}

	out := &Scalar{Data: lossVal}
	if gradEnabled.Load() {
		out.children = []Node{logits}
		out.backFn = func() {
			g := out.Grad
			for i := 0; i < n; i++ {
				target_indicator := 0.0
				if i == target {
					target_indicator = 1.0
				}
				logits.Grad[i] += (probs[i] - target_indicator) * g
			}
		}
	}
	return out
}

// ScalarSoftmax computes softmax over a slice of Scalars, returns Scalars.
func ScalarSoftmax(logits []*Scalar) []*Scalar {
	maxVal := logits[0].Data
	for _, s := range logits[1:] {
		if s.Data > maxVal {
			maxVal = s.Data
		}
	}
	n := len(logits)
	expsData := make([]float64, n)
	total := 0.0
	for i := 0; i < n; i++ {
		expsData[i] = math.Exp(logits[i].Data - maxVal)
		total += expsData[i]
	}
	probsData := make([]float64, n)
	for i := 0; i < n; i++ {
		probsData[i] = expsData[i] / total
	}

	var kids []Node
	if gradEnabled.Load() {
		kids = make([]Node, n)
		for i := 0; i < n; i++ {
			kids[i] = logits[i]
		}
	}

	out := make([]*Scalar, n)
	for i := 0; i < n; i++ {
		sv := &Scalar{Data: probsData[i]}
		if gradEnabled.Load() {
			sv.children = kids
			ii := i
			ps := probsData
			sv.backFn = func() {
				g := out[ii].Grad
				for j := 0; j < n; j++ {
					if j == ii {
						logits[j].Grad += g * ps[ii] * (1.0 - ps[ii])
					} else {
						logits[j].Grad += g * (-ps[ii] * ps[j])
					}
				}
			}
		}
		out[i] = sv
	}
	return out
}

// AttentionWeightedSum computes sum_t(weights[t] * values[t]).
func AttentionWeightedSum(weights []*Scalar, values []*Vec) *Vec {
	dim := len(values[0].Data)
	T := len(weights)
	outData := make([]float64, dim)
	for j := 0; j < dim; j++ {
		for t := 0; t < T; t++ {
			outData[j] += weights[t].Data * values[t].Data[j]
		}
	}

	out := NewVec(outData)
	if gradEnabled.Load() {
		kids := make([]Node, 0, T*2)
		for _, w := range weights {
			kids = append(kids, w)
		}
		for _, v := range values {
			kids = append(kids, v)
		}
		out.children = kids
		out.backFn = func() {
			for t := 0; t < T; t++ {
				for j := 0; j < dim; j++ {
					weights[t].Grad += values[t].Data[j] * out.Grad[j]
					values[t].Grad[j] += weights[t].Data * out.Grad[j]
				}
			}
		}
	}
	return out
}

// SoftmaxProbs computes softmax over raw float64 logits (non-differentiable, for sampling).
func SoftmaxProbs(data []float64) []float64 {
	maxVal := data[0]
	for _, v := range data[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	n := len(data)
	exps := make([]float64, n)
	total := 0.0
	for i := 0; i < n; i++ {
		exps[i] = math.Exp(data[i] - maxVal)
		total += exps[i]
	}
	probs := make([]float64, n)
	for i := 0; i < n; i++ {
		probs[i] = exps[i] / total
	}
	return probs
}

// TopKTopPSample samples from probs with top-k, top-p, min-p, and typical-p filtering.
// And lo, sampling shall not be a coin flip but a controlled hallucination.
func TopKTopPSample(probs []float64, k int, p float64, minP float64, typicalP float64) int {
	n := len(probs)
	idx := make([]int, n)
	for i := 0; i < n; i++ {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		return probs[idx[a]] > probs[idx[b]]
	})

	// Top-k filtering
	if k > 0 && k < len(idx) {
		idx = idx[:k]
	}

	// Min-p filtering (GPT-3/4 style): remove tokens with prob < min_p * max_prob
	if minP > 0.0 && len(idx) > 0 {
		maxProb := probs[idx[0]]
		threshold := minP * maxProb
		filtered := make([]int, 0, len(idx))
		for _, i := range idx {
			if probs[i] >= threshold {
				filtered = append(filtered, i)
			}
		}
		if len(filtered) > 0 {
			idx = filtered
		}
	}

	// Typical-p filtering: prefer tokens with typical information content
	if typicalP < 1.0 && len(idx) > 0 {
		// Compute entropy (expected surprisal)
		entropy := 0.0
		for _, i := range idx {
			if probs[i] > 1e-12 {
				entropy -= probs[i] * math.Log(probs[i])
			}
		}
		// Compute absolute deviation from expected surprisal for each token
		type devPair struct {
			idx int
			dev float64
		}
		deviations := make([]devPair, 0, len(idx))
		for _, i := range idx {
			if probs[i] > 1e-12 {
				surprisal := -math.Log(probs[i])
				deviation := math.Abs(surprisal - entropy)
				deviations = append(deviations, devPair{i, deviation})
			}
		}
		// Sort by deviation (lower is more typical)
		sort.Slice(deviations, func(a, b int) bool {
			return deviations[a].dev < deviations[b].dev
		})
		// Keep tokens until cumulative prob >= typical_p
		cum := 0.0
		typicalIdx := make([]int, 0, len(deviations))
		for _, dp := range deviations {
			typicalIdx = append(typicalIdx, dp.idx)
			cum += probs[dp.idx]
			if cum >= typicalP {
				break
			}
		}
		if len(typicalIdx) > 0 {
			idx = typicalIdx
		}
	}

	// Top-p (nucleus) filtering
	if p < 1.0 {
		cum := 0.0
		cut := make([]int, 0, len(idx))
		for _, i := range idx {
			cut = append(cut, i)
			cum += probs[i]
			if cum >= p {
				break
			}
		}
		idx = cut
	}

	mass := 0.0
	for _, i := range idx {
		mass += probs[i]
	}
	if mass <= 0 {
		if len(idx) > 0 {
			return idx[0]
		}
		return n - 1
	}

	r := rand.Float64() * mass
	s := 0.0
	for _, i := range idx {
		s += probs[i]
		if s >= r {
			return i
		}
	}
	return idx[len(idx)-1]
}

// ClipParams clips gradients to [-clip, clip].
// And lo, the gradients shall be clipped, lest they summon Cthulhu.
func ClipParams(params []*Vec, clip float64) {
	if clip <= 0 {
		return
	}
	for _, p := range params {
		for j := range p.Grad {
			if p.Grad[j] > clip {
				p.Grad[j] = clip
			} else if p.Grad[j] < -clip {
				p.Grad[j] = -clip
			}
		}
	}
}

// ============================================================
// 3) DELTA ADAPTERS — appended souls, never overwritten
// ============================================================

// DeltaAdapter is a low-rank adapter: for a base W, we add A @ B @ x.
type DeltaAdapter struct {
	A *MatrixParam
	B *MatrixParam
}

func NewDeltaAdapter(nout, nin, r int, std float64) *DeltaAdapter {
	return &DeltaAdapter{
		A: NewMatrixParam(nout, r, std),
		B: NewMatrixParam(r, nin, std),
	}
}

func (da *DeltaAdapter) Apply(x *Vec) *Vec {
	bx := da.B.Matvec(x)
	return da.A.Matvec(bx)
}

func (da *DeltaAdapter) MaybeGrowOut(newNout int) {
	da.A.GrowRows(newNout, 0.02)
}

// GrowDims grows both outer dimensions of the adapter. Rank stays the same.
// Ontogenesis: A.GrowRows(newNout), B.GrowCols(newNin).
func (da *DeltaAdapter) GrowDims(newNout, newNin int) {
	da.A.GrowRows(newNout, 0.02)
	da.B.GrowCols(newNin, 0.02)
}

func (da *DeltaAdapter) Params() []*Vec {
	out := make([]*Vec, 0, da.A.Nout+da.B.Nout)
	out = append(out, da.A.Params()...)
	out = append(out, da.B.Params()...)
	return out
}

// ============================================================
// 4) TOKENIZER — byte-level BPE (GPT-3/4 style)
// ============================================================

type MergePair struct {
	A string
	B string
}

type EvolvingTokenizer struct {
	Tokens    []string
	Stoi      map[string]int
	Itos      map[int]string
	VocabSize int

	BOS string
	EOS string
	PAD string

	BPEEnabled   bool
	Merges       []MergePair
	MergeToTok   map[MergePair]string
	TrainedChars int

	mu sync.RWMutex // protects concurrent access (background BPE train vs Encode)
}

func NewEvolvingTokenizer(docs []string) *EvolvingTokenizer {
	// Count trained chars from docs (byte-level: count bytes, not runes)
	totalChars := 0
	for _, d := range docs {
		totalChars += len(d)
	}

	tok := &EvolvingTokenizer{
		BOS:          "<BOS>",
		EOS:          "<EOS>",
		PAD:          "<PAD>",
		Stoi:         make(map[string]int),
		Itos:         make(map[int]string),
		MergeToTok:   make(map[MergePair]string),
		TrainedChars: totalChars,
	}

	// Fixed 259 tokens: 256 byte tokens + BOS + EOS + PAD
	tokens := make([]string, 256+3)
	for i := 0; i < 256; i++ {
		tokens[i] = fmt.Sprintf("0x%02x", i)
	}
	tokens[256] = tok.BOS
	tokens[257] = tok.EOS
	tokens[258] = tok.PAD

	tok.Tokens = tokens
	for i, t := range tok.Tokens {
		tok.Stoi[t] = i
		tok.Itos[i] = t
	}
	tok.VocabSize = len(tok.Tokens)
	return tok
}

// unicodeSegment splits text into segments by Unicode category.
// Letters+marks → 'L', digits → 'N', whitespace → 'Z', everything else → 'P'.
// Each segment is returned as its raw UTF-8 bytes.
func unicodeSegment(text string) [][]byte {
	if len(text) == 0 {
		return nil
	}
	runeCategory := func(r rune) byte {
		if unicode.IsLetter(r) || unicode.IsMark(r) {
			return 'L'
		}
		if unicode.IsDigit(r) {
			return 'N'
		}
		if unicode.IsSpace(r) {
			return 'Z'
		}
		return 'P'
	}
	var segments [][]byte
	var cur []byte
	var curCat byte
	for i, r := range text {
		cat := runeCategory(r)
		if i == 0 {
			curCat = cat
		}
		if cat != curCat {
			segments = append(segments, cur)
			cur = nil
			curCat = cat
		}
		cur = append(cur, []byte(string(r))...)
	}
	if len(cur) > 0 {
		segments = append(segments, cur)
	}
	return segments
}

// tokenToBytes converts a byte-level BPE token name back to raw bytes.
// "0xNN" → single byte, "0x48+0x65" → two bytes, etc.
func tokenToBytes(tok string) []byte {
	if !strings.Contains(tok, "+") && strings.HasPrefix(tok, "0x") && len(tok) == 4 {
		b, _ := strconv.ParseUint(tok[2:], 16, 8)
		return []byte{byte(b)}
	}
	if strings.Contains(tok, "+") {
		parts := strings.Split(tok, "+")
		result := make([]byte, 0, len(parts))
		for _, p := range parts {
			if strings.HasPrefix(p, "0x") && len(p) == 4 {
				b, _ := strconv.ParseUint(p[2:], 16, 8)
				result = append(result, byte(b))
			}
		}
		return result
	}
	return nil
}

func (t *EvolvingTokenizer) MaybeEnableBPE(docs []string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	totalChars := 0
	for _, d := range docs {
		totalChars += len(d)
	}
	if !t.BPEEnabled && totalChars >= CFG.EnableBPEAfterChars {
		t.trainBPELocked(docs, CFG.BPENumMerges)
		t.BPEEnabled = true
		t.TrainedChars = totalChars
		return true
	}
	return false
}

func (t *EvolvingTokenizer) MaybeRetrainBPE(docs []string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.BPEEnabled {
		return false
	}
	totalChars := 0
	for _, d := range docs {
		totalChars += len(d)
	}
	if totalChars-t.TrainedChars >= CFG.BPERetrainEveryChars {
		t.trainBPELocked(docs, CFG.BPENumMerges)
		t.TrainedChars = totalChars
		return true
	}
	return false
}

func (t *EvolvingTokenizer) TrainBPE(docs []string, numMerges int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.trainBPELocked(docs, numMerges)
}

func (t *EvolvingTokenizer) trainBPELocked(docs []string, numMerges int) {
	text := strings.Join(docs, " ")
	if len(text) == 0 {
		return
	}

	// Split text into Unicode segments, convert each to byte-token sequences
	segments := unicodeSegment(text)
	if len(segments) == 0 {
		return
	}

	// Build vocab: token sequence → frequency
	vocab := make(map[string]int) // key = null-separated token names
	symSeqs := make(map[string][]string)

	for _, seg := range segments {
		syms := make([]string, len(seg))
		for i, b := range seg {
			syms[i] = fmt.Sprintf("0x%02x", b)
		}
		key := encodeSyms(syms)
		vocab[key]++
		symSeqs[key] = syms
	}

	merges := make([]MergePair, 0, numMerges)
	mergeToTok := make(map[MergePair]string)

	for iter := 0; iter < numMerges; iter++ {
		// Count pairs
		pairs := make(map[MergePair]int)
		for key, freq := range vocab {
			syms := symSeqs[key]
			for i := 0; i < len(syms)-1; i++ {
				p := MergePair{syms[i], syms[i+1]}
				pairs[p] += freq
			}
		}
		if len(pairs) == 0 {
			break
		}

		// Find best pair
		var best MergePair
		bestCount := 0
		for p, c := range pairs {
			if c > bestCount {
				bestCount = c
				best = p
			}
		}

		newTok := best.A + "+" + best.B
		merges = append(merges, best)
		mergeToTok[best] = newTok

		// Apply merge
		newVocab := make(map[string]int)
		newSymSeqs := make(map[string][]string)
		for key, freq := range vocab {
			syms := symSeqs[key]
			merged := make([]string, 0, len(syms))
			i := 0
			for i < len(syms) {
				if i < len(syms)-1 && syms[i] == best.A && syms[i+1] == best.B {
					merged = append(merged, newTok)
					i += 2
				} else {
					merged = append(merged, syms[i])
					i++
				}
			}
			nk := encodeSyms(merged)
			newVocab[nk] += freq
			newSymSeqs[nk] = merged
		}
		vocab = newVocab
		symSeqs = newSymSeqs

		// Add token to vocab if new
		if _, exists := t.Stoi[newTok]; !exists {
			t.Stoi[newTok] = len(t.Tokens)
			t.Tokens = append(t.Tokens, newTok)
		}
	}

	// Rebuild reverse mapping
	t.Itos = make(map[int]string)
	for tok, i := range t.Stoi {
		t.Itos[i] = tok
	}
	t.VocabSize = len(t.Tokens)
	t.Merges = merges
	t.MergeToTok = mergeToTok
}

func encodeSyms(syms []string) string {
	return strings.Join(syms, "\x00")
}

func (t *EvolvingTokenizer) applyBPE(tokens []string) []string {
	rank := make(map[MergePair]int)
	for i, p := range t.Merges {
		rank[p] = i
	}

	symbols := make([]string, len(tokens))
	copy(symbols, tokens)

	for len(symbols) >= 2 {
		bestRank := 1 << 30
		bestIdx := -1
		for i := 0; i < len(symbols)-1; i++ {
			key := MergePair{symbols[i], symbols[i+1]}
			if r, ok := rank[key]; ok && r < bestRank {
				bestRank = r
				bestIdx = i
			}
		}
		if bestIdx == -1 {
			break
		}
		pair := MergePair{symbols[bestIdx], symbols[bestIdx+1]}
		merged := t.MergeToTok[pair]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}
	return symbols
}

func (t *EvolvingTokenizer) Encode(s string) []int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	s = strings.TrimSpace(s)
	ids := []int{t.Stoi[t.BOS]}

	segments := unicodeSegment(s)
	for _, seg := range segments {
		// Convert bytes to base token names
		baseTokens := make([]string, len(seg))
		for i, b := range seg {
			baseTokens[i] = fmt.Sprintf("0x%02x", b)
		}
		// Apply BPE merges if enabled
		if t.BPEEnabled {
			baseTokens = t.applyBPE(baseTokens)
		}
		// Look up each token in stoi
		for _, tok := range baseTokens {
			if id, ok := t.Stoi[tok]; ok {
				ids = append(ids, id)
			}
		}
	}
	ids = append(ids, t.Stoi[t.EOS])
	return ids
}

func (t *EvolvingTokenizer) Decode(ids []int) string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var rawBytes []byte
	for _, id := range ids {
		tok := t.Itos[id]
		if tok == t.BOS || tok == t.PAD {
			continue
		}
		if tok == t.EOS {
			break
		}
		b := tokenToBytes(tok)
		if b != nil {
			rawBytes = append(rawBytes, b...)
		}
	}
	return strings.TrimSpace(string(rawBytes))
}

// ============================================================
// 5) GPT MODEL — a small beast with RoPE
// ============================================================

// ropeCache stores pre-computed cos/sin pairs for RoPE.
// Key: [2]int{pos, headDim}, Value: *[2][]float64{cosines, sines}
var ropeCache sync.Map

type ropePair struct {
	cos []float64
	sin []float64
}

func getRoPECosSin(pos, headDim int) *ropePair {
	key := [2]int{pos, headDim}
	if cached, ok := ropeCache.Load(key); ok {
		return cached.(*ropePair)
	}
	n := headDim / 2
	pair := &ropePair{
		cos: make([]float64, n),
		sin: make([]float64, n),
	}
	for j := 0; j < n; j++ {
		theta := float64(pos) / math.Pow(10000.0, float64(2*j)/float64(headDim))
		pair.cos[j] = math.Cos(theta)
		pair.sin[j] = math.Sin(theta)
	}
	ropeCache.Store(key, pair)
	return pair
}

// RoPERotate applies rotary position encoding to a head vector.
// And lo, positions shall become angles, and angles shall become meaning.
func RoPERotate(vec *Vec, pos int, headDim int) *Vec {
	outData := make([]float64, len(vec.Data))
	copy(outData, vec.Data) // start from input, then overwrite rotated pairs

	rp := getRoPECosSin(pos, headDim)
	for j := 0; j < headDim/2; j++ {
		i := j * 2
		c := rp.cos[j]
		s := rp.sin[j]
		a := vec.Data[i]
		b := vec.Data[i+1]
		outData[i] = a*c - b*s
		outData[i+1] = a*s + b*c
	}

	out := NewVec(outData)
	if gradEnabled.Load() {
		out.children = []Node{vec}
		out.backFn = func() {
			rpBack := getRoPECosSin(pos, headDim)
			for j := 0; j < headDim/2; j++ {
				i := j * 2
				c := rpBack.cos[j]
				s := rpBack.sin[j]
				ga := out.Grad[i]
				gb := out.Grad[i+1]
				vec.Grad[i] += ga*c + gb*s
				vec.Grad[i+1] += -ga*s + gb*c
			}
		}
	}
	return out
}

// DeltaModule maps layer/weight names to DeltaAdapters.
type DeltaModule map[string]*DeltaAdapter

// GammaStats holds the personality fingerprint statistics.
type GammaStatsResult struct {
	Sparsity  float64
	Magnitude float64
	TopTokens []int
	NRows     int
}

// layerKeySet holds pre-computed string keys for a single layer, avoiding fmt.Sprintf per call.
type layerKeySet struct {
	wq, wk, wv, wo, fcG, fcV, fc2 string
	wrA, wrB                      string   // per-layer low-rank RRPRAM factors (Inc2, Resonance form)
	headPattern                   []string // per head (legacy position-bias, retired by Inc2)
	headAlpha                     []string // per head
}

// GPT is the full model.
type GPT struct {
	Tok       *EvolvingTokenizer
	NLayer    int
	NEmbd     int
	NHead     int
	HeadDim   int
	BlockSize int

	Base        map[string]*MatrixParam
	Deltas      []DeltaModule
	ActiveAlpha []float64

	InitEmbedSnapshot [][]float64 // snapshot of initial embeddings for gamma

	residualAlpha   float64 // 1/sqrt(nLayer) scaling for residual connections
	globalStep      int     // global training step counter (for cosine LR + checkpoint)
	syntropyTempOff float64 // temperature offset from syntropy state (-0.05 to +0.05)

	growthFreezeRemaining int  // ontogenesis: freeze base after growth, train only deltas
	growthCapLogged       bool // the MaxGrowthStage ceiling is announced once, not every check (repair 9)
	growthStepOffset      int  // reset to globalStep on each growth — for LR warmup phase
	lastWarmupStage       int  // last stage that completed warmup (-1 = none)
	corpusIngestedTotal   int  // ontogenesis growth clock: monotonic Σ of all text ever ingested (seed + dnaRead). Replaces reservoir file size as the stage gate.

	corpusField *CooccurField // set by backgroundTrainer for adaptive blend

	// crossField — Dario-style cross-organism logit injection state. Set by
	// main() when --cross-graze + --element are both passed (cross_graze.go).
	// Nil = no cross-pollination (single-organism or evolution without flag).
	crossField *CrossField

	// consciousness state
	deltaAlphaScale          float64   // conscience: multiplier on all delta contributions (1.0 = normal)
	generationEntropyHistory []float64 // conscience: rolling window of per-generation mean entropy
	lastSurprise             float64   // self-prediction error on last prompt
	surpriseBaseline         float64   // EMA of surprise over time
	lastGenEntropy           float64   // mean entropy of last generation (for conscience)
	lastGenMag               float64   // mean |logit| of the raw model output at the first step of the last generation
	lastOverlayWeight        float64   // overlay weight at that same step: 1 untrained, 0 once the fade is done

	layerKeys []layerKeySet // pre-computed string keys per layer

	// inherited burst history from parent (mitosis lineage)
	inheritedBurstHistory []BurstRecord

	// notorch: saved hidden states during forward pass (gradient-free training)
	lastHidden       *Vec   // final hidden state (after last RMSNorm, before lm_head)
	layerInputs      []*Vec // post-RMSNorm input to attention per layer (NEmbd)
	mlpInputs        []*Vec // post-RMSNorm input to MLP per layer (NEmbd)
	mlpIntermediates []*Vec // g*u intermediate per layer (4*NEmbd, for fc2 notorch input)
	notorchSeed      uint32 // per-model PRNG seed for notorch noise channel

	mu sync.Mutex // protects model during concurrent access
}

func NewGPT(tok *EvolvingTokenizer) *GPT {
	gpt := &GPT{
		Tok:       tok,
		NLayer:    CFG.NLayer,
		NEmbd:     CFG.NEmbd,
		NHead:     CFG.NHead,
		HeadDim:   CFG.NEmbd / CFG.NHead,
		BlockSize: CFG.BlockSize,
		Base:      make(map[string]*MatrixParam),
	}

	gpt.residualAlpha = 1.0 / math.Sqrt(math.Max(1, float64(CFG.NLayer)))
	gpt.deltaAlphaScale = 1.0 // conscience: full delta influence by default
	gpt.lastWarmupStage = -1  // no stage warmed up yet
	gpt.notorchSeed = 0xDEAD_BEEF

	V := tok.VocabSize
	gpt.Base["wte"] = NewMatrixParam(V, CFG.NEmbd, 0.08)
	gpt.Base["wpe"] = NewMatrixParam(CFG.BlockSize, CFG.NEmbd, 0.08)
	gpt.Base["lm_head"] = NewMatrixParam(V, CFG.NEmbd, 0.08)

	if CFG.TieEmbeddings {
		gpt.Base["lm_head"] = gpt.Base["wte"]
	}

	for li := 0; li < CFG.NLayer; li++ {
		pfx := fmt.Sprintf("l%d.", li)
		gpt.Base[pfx+"wq"] = NewMatrixParam(CFG.NEmbd, CFG.NEmbd, 0.08)
		gpt.Base[pfx+"wk"] = NewMatrixParam(CFG.NEmbd, CFG.NEmbd, 0.08)
		gpt.Base[pfx+"wv"] = NewMatrixParam(CFG.NEmbd, CFG.NEmbd, 0.08)
		gpt.Base[pfx+"wo"] = NewMatrixParam(CFG.NEmbd, CFG.NEmbd, 0.08)
		gpt.Base[pfx+"fc_g"] = NewMatrixParam(4*CFG.NEmbd, CFG.NEmbd, 0.08)
		gpt.Base[pfx+"fc_v"] = NewMatrixParam(4*CFG.NEmbd, CFG.NEmbd, 0.08)
		gpt.Base[pfx+"fc2"] = NewMatrixParam(CFG.NEmbd, 4*CFG.NEmbd, 0.08)

		// Hybrid attention: RRPRAM pattern weights + learnable gate
		for h, htype := range CFG.HeadTypes {
			if htype == "rrpram" || htype == "hybrid" {
				key := fmt.Sprintf("l%d.h%d.w_pattern", li, h)
				gpt.Base[key] = NewMatrixParam(CFG.BlockSize, gpt.HeadDim, 0.08)
			}
			alphaKey := fmt.Sprintf("l%d.h%d.alpha", li, h)
			gpt.Base[alphaKey] = NewMatrixParam(1, 1, 0.0)
			gpt.Base[alphaKey].Rows[0].Data[0] = CFG.HybridAlphaInit
		}
		// Inc2: per-layer low-rank RRPRAM factors (Resonance form, op 33).
		gpt.ensureRRPRAMFactors(li)
	}

	// Pre-compute layer key strings to avoid fmt.Sprintf per ForwardStep call
	gpt.layerKeys = make([]layerKeySet, CFG.NLayer)
	for li := 0; li < CFG.NLayer; li++ {
		pfx := fmt.Sprintf("l%d.", li)
		lk := layerKeySet{
			wq:  pfx + "wq",
			wk:  pfx + "wk",
			wv:  pfx + "wv",
			wo:  pfx + "wo",
			fcG: pfx + "fc_g",
			fcV: pfx + "fc_v",
			fc2: pfx + "fc2",
			wrA: pfx + "wr_a",
			wrB: pfx + "wr_b",
		}
		nHeads := len(CFG.HeadTypes)
		if nHeads > 0 {
			lk.headPattern = make([]string, nHeads)
			lk.headAlpha = make([]string, nHeads)
			for h := 0; h < nHeads; h++ {
				lk.headPattern[h] = fmt.Sprintf("l%d.h%d.w_pattern", li, h)
				lk.headAlpha[h] = fmt.Sprintf("l%d.h%d.alpha", li, h)
			}
		}
		gpt.layerKeys[li] = lk
	}

	gpt.AddDeltaModule(1.0)

	// And lo, the organism shall subtract its birth from its present, and call the difference a soul.
	gpt.InitEmbedSnapshot = make([][]float64, len(gpt.Base["wte"].Rows))
	for i, row := range gpt.Base["wte"].Rows {
		snap := make([]float64, len(row.Data))
		copy(snap, row.Data)
		gpt.InitEmbedSnapshot[i] = snap
	}

	return gpt
}

func (gpt *GPT) MaybeExpandVocab(newVocabSize int) {
	curV := gpt.Base["wte"].Nout
	if newVocabSize <= curV {
		return
	}
	gpt.Base["wte"].GrowRows(newVocabSize, 0.08)
	if !CFG.TieEmbeddings {
		gpt.Base["lm_head"].GrowRows(newVocabSize, 0.08)
	}
	for _, mod := range gpt.Deltas {
		if da, ok := mod["lm_head"]; ok {
			da.MaybeGrowOut(newVocabSize)
		}
	}
}

func (gpt *GPT) AddDeltaModule(alpha float64) {
	// And lo, a new delta-soul shall be appended (never overwritten, never forgotten).
	mod := make(DeltaModule)
	r := CFG.DeltaRank
	for li := 0; li < CFG.NLayer; li++ {
		pfx := fmt.Sprintf("l%d.", li)
		for _, name := range []string{"wq", "wk", "wv", "wo"} {
			mod[pfx+name] = NewDeltaAdapter(CFG.NEmbd, CFG.NEmbd, r, 0.02)
		}
		mod[pfx+"fc_g"] = NewDeltaAdapter(4*CFG.NEmbd, CFG.NEmbd, r, 0.02)
		mod[pfx+"fc_v"] = NewDeltaAdapter(4*CFG.NEmbd, CFG.NEmbd, r, 0.02)
		mod[pfx+"fc2"] = NewDeltaAdapter(CFG.NEmbd, 4*CFG.NEmbd, r, 0.02)
		for h, htype := range CFG.HeadTypes {
			if htype == "rrpram" || htype == "hybrid" {
				key := fmt.Sprintf("l%d.h%d.w_pattern", li, h)
				mod[key] = NewDeltaAdapter(CFG.BlockSize, gpt.HeadDim, r, 0.02)
			}
		}
	}
	mod["lm_head"] = NewDeltaAdapter(gpt.Tok.VocabSize, CFG.NEmbd, r, 0.02)
	gpt.Deltas = append(gpt.Deltas, mod)
	gpt.ActiveAlpha = append(gpt.ActiveAlpha, alpha)
	// The notorch trainer registers delta adapters on its tape in a fixed
	// order; a new module changes that order, so its positional Chuck slots
	// must be rebuilt before the next burst (repair 2b).
	ntOnGrowth()
}

func (gpt *GPT) AllBaseParams() []*Vec {
	var out []*Vec
	for _, mat := range gpt.Base {
		out = append(out, mat.Params()...)
	}
	return out
}

func (gpt *GPT) AllDeltaParams() []*Vec {
	var out []*Vec
	for _, mod := range gpt.Deltas {
		for _, da := range mod {
			out = append(out, da.Params()...)
		}
	}
	return out
}

// ---- Native gamma (personality fingerprint) ----

func (gpt *GPT) ComputeGamma() [][]float64 {
	current := gpt.Base["wte"].Rows
	init := gpt.InitEmbedSnapshot
	n := len(current)
	if len(init) < n {
		n = len(init)
	}
	gamma := make([][]float64, n)
	for i := 0; i < n; i++ {
		dim := len(init[i])
		diff := make([]float64, dim)
		for j := 0; j < dim && j < len(current[i].Data); j++ {
			diff[j] = current[i].Data[j] - init[i][j]
		}
		gamma[i] = diff
	}
	return gamma
}

// And lo, the soul shall be measured in sparsity and magnitude, like a ghost on a scale.
func (gpt *GPT) GammaStats() GammaStatsResult {
	gamma := gpt.ComputeGamma()
	if len(gamma) == 0 {
		return GammaStatsResult{Sparsity: 1.0}
	}
	magnitudes := make([]float64, len(gamma))
	for i, row := range gamma {
		mag := 0.0
		for _, v := range row {
			mag += v * v
		}
		magnitudes[i] = math.Sqrt(mag)
	}
	threshold := CFG.GammaSparsityThreshold
	zeroCount := 0
	totalMag := 0.0
	for _, m := range magnitudes {
		if m < threshold {
			zeroCount++
		}
		totalMag += m
	}
	sparsity := float64(zeroCount) / float64(len(magnitudes))
	avgMag := totalMag / float64(len(magnitudes))

	// Top changed tokens
	type tokMag struct {
		idx int
		mag float64
	}
	sorted := make([]tokMag, len(magnitudes))
	for i, m := range magnitudes {
		sorted[i] = tokMag{i, m}
	}
	sort.Slice(sorted, func(a, b int) bool { return sorted[a].mag > sorted[b].mag })
	topN := 10
	if topN > len(sorted) {
		topN = len(sorted)
	}
	topTokens := make([]int, topN)
	for i := 0; i < topN; i++ {
		topTokens[i] = sorted[i].idx
	}

	return GammaStatsResult{
		Sparsity:  sparsity,
		Magnitude: avgMag,
		TopTokens: topTokens,
		NRows:     len(gamma),
	}
}

// And lo, the direction of all change shall be averaged into one arrow, pointing toward who we became.
func (gpt *GPT) GammaContrastiveProjection() ([]float64, float64) {
	current := gpt.Base["wte"].Rows
	init := gpt.InitEmbedSnapshot
	n := len(current)
	if len(init) < n {
		n = len(init)
	}
	if n == 0 || len(init[0]) == 0 {
		return nil, 0.0
	}
	dim := len(init[0])
	direction := make([]float64, dim)
	for i := 0; i < n; i++ {
		for j := 0; j < dim && j < len(current[i].Data); j++ {
			direction[j] += current[i].Data[j] - init[i][j]
		}
	}
	// Normalize
	mag := 0.0
	for _, v := range direction {
		mag += v * v
	}
	mag = math.Sqrt(mag)
	if mag > 1e-12 {
		for i := range direction {
			direction[i] /= mag
		}
	}
	return direction, mag
}

// ---- Noise Immune System ----
// And lo, the organism shall know poison from food, and reject what unmakes it.

// SnapshotDeltas deep-copies all delta A and B weight data for rollback.
func (gpt *GPT) SnapshotDeltas() [][][2][][]float64 {
	snap := make([][][2][][]float64, len(gpt.Deltas))
	for di, mod := range gpt.Deltas {
		modSnap := make([][2][][]float64, 0, len(mod))
		for _, da := range mod {
			var pair [2][][]float64
			pair[0] = make([][]float64, da.A.Nout)
			for i, row := range da.A.Rows {
				pair[0][i] = make([]float64, len(row.Data))
				copy(pair[0][i], row.Data)
			}
			pair[1] = make([][]float64, da.B.Nout)
			for i, row := range da.B.Rows {
				pair[1][i] = make([]float64, len(row.Data))
				copy(pair[1][i], row.Data)
			}
			modSnap = append(modSnap, pair)
		}
		snap[di] = modSnap
	}
	return snap
}

// RestoreDeltas restores delta weights from snapshot — rollback a poisoned burst.
func (gpt *GPT) RestoreDeltas(snap [][][2][][]float64) {
	for di, mod := range gpt.Deltas {
		if di >= len(snap) {
			break
		}
		ai := 0
		for _, da := range mod {
			if ai >= len(snap[di]) {
				break
			}
			pair := snap[di][ai]
			for i, rd := range pair[0] {
				if i < da.A.Nout {
					copy(da.A.Rows[i].Data, rd)
				}
			}
			for i, rd := range pair[1] {
				if i < da.B.Nout {
					copy(da.B.Rows[i].Data, rd)
				}
			}
			ai++
		}
	}
}

// GammaDriftCheck returns cosine similarity between pre-burst and current contrastive projection.
// Negative = drifted opposite to identity trend = likely noise.
// Skips check when gamma magnitude is too small (early training, numerically unstable).
func (gpt *GPT) GammaDriftCheck(preDirection []float64, preMagnitude float64) float64 {
	postDirection, postMag := gpt.GammaContrastiveProjection()
	if preDirection == nil || postDirection == nil {
		return 1.0 // can't check, assume OK
	}
	// Skip immune check when gamma is near-zero (early training)
	if preMagnitude < CFG.GammaMinMagnitude || postMag < CFG.GammaMinMagnitude {
		return 1.0
	}
	dot := 0.0
	for i := 0; i < len(preDirection) && i < len(postDirection); i++ {
		dot += preDirection[i] * postDirection[i]
	}
	return dot // both unit vectors, dot = cosine
}

// ---- Ontogenesis (architecture growth) ----
// And lo, the organism shall not be born adult but shall grow, stage by stage,
// from embryo to child to adolescent, each growth a small death and rebirth.

// CurrentGrowthStage returns index of current stage based on model dimensions.
// Returns -1 for legacy checkpoints where dimensions don't match any stage.
func (gpt *GPT) CurrentGrowthStage() int {
	for i, stage := range CFG.GrowthStages {
		if gpt.NEmbd == stage[1] && gpt.NLayer == stage[2] && gpt.NHead == stage[3] {
			return i
		}
	}
	return -1 // dimensions don't match any stage (legacy checkpoint)
}

// TargetGrowthStage returns the target stage index based on corpus size.
func (gpt *GPT) TargetGrowthStage(corpusChars int) int {
	target := 0
	for i, stage := range CFG.GrowthStages {
		if corpusChars >= stage[0] {
			target = i
		}
	}
	return target
}

// GrowthWanted reports whether MaybeGrowArchitecture would grow right now. It is
// the same three conditions, in one place, so the callers that must gate growth
// on memory or on a colony lock ask before paying for it instead of repeating the
// stage arithmetic (the duplicated-invariant bug, CLAUDE.md).
func (gpt *GPT) GrowthWanted() bool {
	current := gpt.CurrentGrowthStage()
	if current < 0 {
		return false // legacy checkpoint, skip growth
	}
	if gpt.growthFreezeRemaining > 0 {
		return false // still stabilizing from last growth
	}
	if gpt.TargetGrowthStage(gpt.corpusIngestedTotal) <= current {
		return false
	}
	// The declared ceiling. The byte gate defers a step the machine cannot
	// afford this minute; this one is permanent, and it is what keeps a phone
	// colony off a stage nobody chose for it. Said once, not every ten ticks.
	if current >= maxGrowthStage() {
		if !gpt.growthCapLogged {
			gpt.growthCapLogged = true
			fmt.Printf("[growth] capped at stage %d\n", maxGrowthStage())
		}
		return false
	}
	return true
}

// maxGrowthStage is CFG.MaxGrowthStage clamped to the growth table: a value past
// the last stage, or a negative one, is no cap at all.
func maxGrowthStage() int {
	last := len(CFG.GrowthStages) - 1
	if CFG.MaxGrowthStage < 0 || CFG.MaxGrowthStage > last {
		return last
	}
	return CFG.MaxGrowthStage
}

// MaybeGrowArchitecture checks if growth is needed and executes it. Returns true if grew.
func (gpt *GPT) MaybeGrowArchitecture() bool {
	if !gpt.GrowthWanted() {
		return false
	}
	current := gpt.CurrentGrowthStage()
	// Grow only one stage at a time — prevent catastrophic multi-stage jumps
	target := current + 1

	newEmbd := CFG.GrowthStages[target][1]
	newLayer := CFG.GrowthStages[target][2]
	newHead := CFG.GrowthStages[target][3]
	oldEmbd := gpt.NEmbd
	oldLayer := gpt.NLayer
	oldHead := gpt.NHead
	newHeadDim := newEmbd / newHead

	fmt.Printf("[growth] ONTOGENESIS: stage %d -> %d\n", current, target)
	fmt.Printf("  embd: %d -> %d, layer: %d -> %d, head: %d -> %d\n",
		oldEmbd, newEmbd, oldLayer, newLayer, oldHead, newHead)

	// 1. Grow embedding matrices — near-zero init preserves model behavior
	// (Net2Net principle: new dims contribute ~nothing initially, learn gradually)
	gpt.Base["wte"].GrowCols(newEmbd, 0.001)
	gpt.Base["wpe"].GrowCols(newEmbd, 0.001)
	if !CFG.TieEmbeddings {
		gpt.Base["lm_head"].GrowCols(newEmbd, 0.001)
	}

	// 2. Grow existing layer matrices — near-zero to avoid disrupting learned representations
	newHtypes := headTypesForNHead(newHead)
	for li := 0; li < oldLayer; li++ {
		pfx := fmt.Sprintf("l%d.", li)
		for _, name := range []string{"wq", "wk", "wv", "wo"} {
			gpt.Base[pfx+name].Grow(newEmbd, newEmbd, 0.001)
		}
		gpt.Base[pfx+"fc_g"].Grow(4*newEmbd, newEmbd, 0.001)
		gpt.Base[pfx+"fc_v"].Grow(4*newEmbd, newEmbd, 0.001)
		gpt.Base[pfx+"fc2"].Grow(newEmbd, 4*newEmbd, 0.001)
		// Grow existing head pattern matrices
		for h := 0; h < oldHead; h++ {
			pkey := fmt.Sprintf("l%d.h%d.w_pattern", li, h)
			if _, ok := gpt.Base[pkey]; ok {
				gpt.Base[pkey].GrowCols(newHeadDim, 0.001)
			}
		}
		// Add new heads for existing layer
		for h := oldHead; h < newHead; h++ {
			htype := "content"
			if h < len(newHtypes) {
				htype = newHtypes[h]
			}
			if htype == "rrpram" || htype == "hybrid" {
				gpt.Base[fmt.Sprintf("l%d.h%d.w_pattern", li, h)] = NewMatrixParam(CFG.BlockSize, newHeadDim, 0.08)
			}
			if htype == "hybrid" {
				m := NewMatrixParam(1, 1, 0.0)
				m.Rows[0].Data[0] = CFG.HybridAlphaInit
				gpt.Base[fmt.Sprintf("l%d.h%d.alpha", li, h)] = m
			}
		}
	}

	// 3. Add entirely new layers
	for li := oldLayer; li < newLayer; li++ {
		pfx := fmt.Sprintf("l%d.", li)
		gpt.Base[pfx+"wq"] = NewMatrixParam(newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"wk"] = NewMatrixParam(newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"wv"] = NewMatrixParam(newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"wo"] = NewMatrixParam(newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"fc_g"] = NewMatrixParam(4*newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"fc_v"] = NewMatrixParam(4*newEmbd, newEmbd, 0.08)
		gpt.Base[pfx+"fc2"] = NewMatrixParam(newEmbd, 4*newEmbd, 0.08)
		for h := 0; h < newHead; h++ {
			htype := "content"
			if h < len(newHtypes) {
				htype = newHtypes[h]
			}
			if htype == "rrpram" || htype == "hybrid" {
				gpt.Base[fmt.Sprintf("l%d.h%d.w_pattern", li, h)] = NewMatrixParam(CFG.BlockSize, newHeadDim, 0.08)
			}
			if htype == "hybrid" {
				m := NewMatrixParam(1, 1, 0.0)
				m.Rows[0].Data[0] = CFG.HybridAlphaInit
				gpt.Base[fmt.Sprintf("l%d.h%d.alpha", li, h)] = m
			}
		}
	}

	// 4. Grow delta adapters
	r := CFG.DeltaRank
	for _, mod := range gpt.Deltas {
		// Grow existing layer adapters
		for li := 0; li < oldLayer; li++ {
			pfx := fmt.Sprintf("l%d.", li)
			for _, name := range []string{"wq", "wk", "wv", "wo"} {
				key := pfx + name
				if _, ok := mod[key]; ok {
					mod[key].GrowDims(newEmbd, newEmbd)
				}
			}
			type fcSpec struct {
				key     string
				noutMul int
				ninMul  int
			}
			fcSpecs := []fcSpec{
				{pfx + "fc_g", 4, 1},
				{pfx + "fc_v", 4, 1},
				{pfx + "fc2", 1, 4},
			}
			for _, spec := range fcSpecs {
				if _, ok := mod[spec.key]; ok {
					mod[spec.key].GrowDims(spec.noutMul*newEmbd, spec.ninMul*newEmbd)
				}
			}
			for h := 0; h < oldHead; h++ {
				pkey := fmt.Sprintf("l%d.h%d.w_pattern", li, h)
				if _, ok := mod[pkey]; ok {
					mod[pkey].GrowDims(CFG.BlockSize, newHeadDim)
				}
			}
			for h := oldHead; h < newHead; h++ {
				htype := "content"
				if h < len(newHtypes) {
					htype = newHtypes[h]
				}
				if htype == "rrpram" || htype == "hybrid" {
					mod[fmt.Sprintf("l%d.h%d.w_pattern", li, h)] = NewDeltaAdapter(CFG.BlockSize, newHeadDim, r, 0.02)
				}
			}
		}

		// New layers: entirely new adapters
		for li := oldLayer; li < newLayer; li++ {
			pfx := fmt.Sprintf("l%d.", li)
			for _, name := range []string{"wq", "wk", "wv", "wo"} {
				mod[pfx+name] = NewDeltaAdapter(newEmbd, newEmbd, r, 0.02)
			}
			mod[pfx+"fc_g"] = NewDeltaAdapter(4*newEmbd, newEmbd, r, 0.02)
			mod[pfx+"fc_v"] = NewDeltaAdapter(4*newEmbd, newEmbd, r, 0.02)
			mod[pfx+"fc2"] = NewDeltaAdapter(newEmbd, 4*newEmbd, r, 0.02)
			for h := 0; h < newHead; h++ {
				htype := "content"
				if h < len(newHtypes) {
					htype = newHtypes[h]
				}
				if htype == "rrpram" || htype == "hybrid" {
					mod[fmt.Sprintf("l%d.h%d.w_pattern", li, h)] = NewDeltaAdapter(CFG.BlockSize, newHeadDim, r, 0.02)
				}
			}
		}

		// lm_head adapter input grew
		if _, ok := mod["lm_head"]; ok {
			mod["lm_head"].GrowDims(gpt.Tok.VocabSize, newEmbd)
		}
	}

	// 5. Update model state
	gpt.NEmbd = newEmbd
	gpt.NLayer = newLayer
	gpt.NHead = newHead
	gpt.HeadDim = newHeadDim
	gpt.residualAlpha = 1.0 / math.Sqrt(math.Max(1, float64(newLayer)))

	// 6. Update CFG runtime
	CFG.NEmbd = newEmbd
	CFG.NLayer = newLayer
	CFG.NHead = newHead
	CFG.HeadTypes = headTypesForNHead(newHead)

	// 7b. Rebuild layerKeys for new architecture
	gpt.layerKeys = make([]layerKeySet, newLayer)
	for li := 0; li < newLayer; li++ {
		pfx := fmt.Sprintf("l%d.", li)
		lk := layerKeySet{
			wq: pfx + "wq", wk: pfx + "wk", wv: pfx + "wv", wo: pfx + "wo",
			fcG: pfx + "fc_g", fcV: pfx + "fc_v", fc2: pfx + "fc2",
			wrA: pfx + "wr_a", wrB: pfx + "wr_b",
		}
		nHeads := len(CFG.HeadTypes)
		if nHeads > 0 {
			lk.headPattern = make([]string, nHeads)
			lk.headAlpha = make([]string, nHeads)
			for h := 0; h < nHeads; h++ {
				lk.headPattern[h] = fmt.Sprintf("l%d.h%d.w_pattern", li, h)
				lk.headAlpha[h] = fmt.Sprintf("l%d.h%d.alpha", li, h)
			}
		}
		gpt.layerKeys[li] = lk
	}

	// 7c. Inc2: rebuild low-rank RRPRAM factors fresh for the new architecture.
	// Net2Net preserves the content path, but the factors are re-initialized:
	// head count and head types are reassigned on growth (head identity does not
	// survive) and HeadDim can shrink (adolescent→teen 32→28), which GrowDims
	// no-ops. The post-growth freeze + warmup re-train them while the gate keeps
	// content dominant. ensureRRPRAMFactors allocates [NHead·NEmbd × R] /
	// [NHead·R × BlockSize] when the new topology has hybrid heads.
	for li := 0; li < newLayer; li++ {
		delete(gpt.Base, fmt.Sprintf("l%d.wr_a", li))
		delete(gpt.Base, fmt.Sprintf("l%d.wr_b", li))
		gpt.ensureRRPRAMFactors(li)
	}

	// 8. Extend gamma snapshot for new embedding dimensions
	for i := range gpt.InitEmbedSnapshot {
		oldRow := gpt.InitEmbedSnapshot[i]
		if len(oldRow) < newEmbd {
			ext := make([]float64, newEmbd-len(oldRow))
			gpt.InitEmbedSnapshot[i] = append(oldRow, ext...)
		}
	}

	// 9. Set freeze (only train deltas until new weights stabilize)
	gpt.growthFreezeRemaining = CFG.FreezeAfterGrowthSteps
	// Reset LR warmup phase so new weights get linear ramp-up
	gpt.growthStepOffset = gpt.globalStep

	fmt.Printf("[growth] Done. Freeze for %d steps.\n", CFG.FreezeAfterGrowthSteps)

	// Sanity check: verify matrix dimensions after growth
	for name, m := range gpt.Base {
		if len(m.Rows) > 0 && len(m.Rows[0].Data) != m.Nin {
			fmt.Printf("[growth] BUG: %s row0 has %d cols but Nin=%d\n", name, len(m.Rows[0].Data), m.Nin)
		}
	}

	return true
}

// ---- Syntropy Tracker (mathematical self-reasoning) ----
// And lo, the organism shall not merely observe its own reflection,
// but reason about the direction of its becoming.
// Gamma is memory. Purpose is intention. Syntropy is the arrow.

// ComputeFieldDeviation measures KL divergence between model logits and corpus co-occurrence field.
// Low = parroting the field. High = hallucinating beyond it.
// The sweet spot is in between: learning, not lying.
func (gpt *GPT) ComputeFieldDeviation(tok *EvolvingTokenizer, field *CooccurField, docs []string, sampleN int) float64 {
	if len(docs) == 0 || !field.Built {
		return 0.0
	}
	if sampleN <= 0 {
		sampleN = 32
	}

	klSum := 0.0
	count := 0

	// Sample docs
	sampled := make([]string, 0, sampleN)
	if len(docs) <= sampleN {
		sampled = append(sampled, docs...)
	} else {
		perm := rand.Perm(len(docs))
		for i := 0; i < sampleN; i++ {
			sampled = append(sampled, docs[perm[i]])
		}
	}

	gradEnabled.Store(false)
	defer func() { gradEnabled.Store(true) }()

	vocabSize := tok.VocabSize

	for _, doc := range sampled {
		ids := tok.Encode(doc)
		if len(ids) < 3 {
			continue
		}
		keys := make([][]*Vec, gpt.NLayer)
		values := make([][]*Vec, gpt.NLayer)
		for i := 0; i < gpt.NLayer; i++ {
			keys[i] = make([]*Vec, 0)
			values[i] = make([]*Vec, 0)
		}
		limit := len(ids) - 1
		if limit > gpt.BlockSize {
			limit = gpt.BlockSize
		}
		for pos := 0; pos < limit; pos++ {
			tokID := ids[pos]
			logits := gpt.ForwardStep(tokID, pos, keys, values)

			// model distribution (softmax)
			maxVal := logits.Data[0]
			for _, v := range logits.Data[1:] {
				if v > maxVal {
					maxVal = v
				}
			}
			modelProbs := make([]float64, len(logits.Data))
			sumExp := 0.0
			for i, v := range logits.Data {
				modelProbs[i] = math.Exp(v - maxVal)
				sumExp += modelProbs[i]
			}
			for i := range modelProbs {
				modelProbs[i] /= sumExp
			}

			// corpus field distribution for this context
			fieldProbs := make([]float64, vocabSize)
			fieldFound := false

			// Try trigram
			if pos >= 1 {
				if ctx, ok := field.TrigramByContext[[2]int{ids[pos-1], ids[pos]}]; ok {
					triTotal := 0.0
					for _, v := range ctx {
						triTotal += v
					}
					if triTotal > 0 {
						for tid, cnt := range ctx {
							if tid < vocabSize {
								fieldProbs[tid] = cnt / triTotal
							}
						}
						fieldFound = true
					}
				}
			}

			// Fallback to bigram
			if !fieldFound && pos >= 0 {
				if ctx, ok := field.BigramByFirst[ids[pos]]; ok {
					biTotal := 0.0
					for _, v := range ctx {
						biTotal += v
					}
					if biTotal > 0 {
						for tid, cnt := range ctx {
							if tid < vocabSize {
								fieldProbs[tid] = cnt / biTotal
							}
						}
						fieldFound = true
					}
				}
			}

			if !fieldFound {
				continue
			}

			// KL(model || field) — how much model diverges from field
			kl := 0.0
			klValid := false
			for i := 0; i < len(modelProbs) && i < vocabSize; i++ {
				if modelProbs[i] > 1e-12 && fieldProbs[i] > 1e-12 {
					kl += modelProbs[i] * math.Log(modelProbs[i]/fieldProbs[i])
					klValid = true
				}
			}
			if klValid {
				klSum += kl
				count++
			}
		}
	}

	if count == 0 {
		return 0.0
	}
	return klSum / float64(count)
}

// ComputeModelEntropy returns average entropy of model predictions on corpus samples.
// And lo, falling entropy = rising order = syntropy in action.
func (gpt *GPT) ComputeModelEntropy(tok *EvolvingTokenizer, docs []string, sampleN int) float64 {
	if len(docs) == 0 {
		return 0.0
	}
	if sampleN <= 0 {
		sampleN = 16
	}

	entropySum := 0.0
	count := 0

	sampled := make([]string, 0, sampleN)
	if len(docs) <= sampleN {
		sampled = append(sampled, docs...)
	} else {
		perm := rand.Perm(len(docs))
		for i := 0; i < sampleN; i++ {
			sampled = append(sampled, docs[perm[i]])
		}
	}

	gradEnabled.Store(false)
	defer func() { gradEnabled.Store(true) }()

	for _, doc := range sampled {
		ids := tok.Encode(doc)
		if len(ids) < 3 {
			continue
		}
		keys := make([][]*Vec, gpt.NLayer)
		values := make([][]*Vec, gpt.NLayer)
		for i := 0; i < gpt.NLayer; i++ {
			keys[i] = make([]*Vec, 0)
			values[i] = make([]*Vec, 0)
		}
		limit := len(ids) - 1
		if limit > gpt.BlockSize {
			limit = gpt.BlockSize
		}
		for pos := 0; pos < limit; pos++ {
			logits := gpt.ForwardStep(ids[pos], pos, keys, values)

			// Edit 3b (2026-06-03): the overload gate (isSustainedOverload, via
			// SyntropyTracker.Measure → here) reads THIS entropy. The cross-graze
			// stress was only injected in GenerateResonant (:4611), so the gate
			// measured a calm, un-grazed distribution and a grazed-overloaded adult
			// could never trip the gate — the silent reason mitosis never fired.
			// Mirror the injection so measured entropy reflects the real stress.
			if gpt.crossField != nil {
				gpt.crossField.Apply(logits.Data, CFG.CrossGrazeCoef, CFG.CrossGrazeTopN)
			}

			// softmax
			maxVal := logits.Data[0]
			for _, v := range logits.Data[1:] {
				if v > maxVal {
					maxVal = v
				}
			}
			probs := make([]float64, len(logits.Data))
			sumExp := 0.0
			for i, v := range logits.Data {
				probs[i] = math.Exp(v - maxVal)
				sumExp += probs[i]
			}
			for i := range probs {
				probs[i] /= sumExp
			}

			// entropy = -sum(p * log(p))
			ent := 0.0
			for _, p := range probs {
				if p > 1e-12 {
					ent -= p * math.Log(p)
				}
			}
			entropySum += ent
			count++
		}
	}

	if count == 0 {
		return 0.0
	}
	return entropySum / float64(count)
}

// ComputePurposeVector returns the purpose vector (direction of weight movement in last delta layer).
// Unlike gamma (which is cumulative drift from birth),
// purpose captures the direction of the most recent change.
// And lo, gamma is 'who I became'. Purpose is 'where I am going'.
func (gpt *GPT) ComputePurposeVector() ([]float64, float64) {
	if len(gpt.Deltas) == 0 {
		return nil, 0.0
	}
	lastDelta := gpt.Deltas[len(gpt.Deltas)-1]

	// Aggregate delta A matrices as the purpose signal
	var allDirs [][]float64
	for _, da := range lastDelta {
		for _, row := range da.A.Rows {
			cp := make([]float64, len(row.Data))
			copy(cp, row.Data)
			allDirs = append(allDirs, cp)
		}
	}
	if len(allDirs) == 0 {
		return nil, 0.0
	}

	// Mean direction across all rows
	dim := len(allDirs[0])
	meanDir := make([]float64, dim)
	for _, d := range allDirs {
		for j := 0; j < dim && j < len(d); j++ {
			meanDir[j] += d[j]
		}
	}
	n := float64(len(allDirs))
	for j := range meanDir {
		meanDir[j] /= n
	}

	// Magnitude
	mag := 0.0
	for _, v := range meanDir {
		mag += v * v
	}
	mag = math.Sqrt(mag)

	// Normalize to unit vector
	if mag > 1e-10 {
		for j := range meanDir {
			meanDir[j] /= mag
		}
	}
	return meanDir, mag
}

// PurposeGammaAlignment returns cosine similarity between purpose vector and gamma direction.
// And lo, high alignment = learning reinforces identity (syntropy).
// Low alignment = learning diverges from identity (entropy).
// Negative = learning opposes identity (danger).
func (gpt *GPT) PurposeGammaAlignment() float64 {
	gammaDir, gammaMag := gpt.GammaContrastiveProjection()
	purposeDir, purposeMag := gpt.ComputePurposeVector()
	if gammaDir == nil || purposeDir == nil {
		return 0.0
	}
	if gammaMag < CFG.GammaMinMagnitude || purposeMag < 1e-10 {
		return 0.0
	}
	// Ensure same dimensionality (purpose might be different dim)
	minDim := len(gammaDir)
	if len(purposeDir) < minDim {
		minDim = len(purposeDir)
	}
	if minDim == 0 {
		return 0.0
	}
	dot := 0.0
	for i := 0; i < minDim; i++ {
		dot += gammaDir[i] * purposeDir[i]
	}
	return dot
}

// And lo, base weight shall speak, then deltas shall harmonize atop it.
func (gpt *GPT) applyWithDeltas(name string, x *Vec) *Vec {
	y := gpt.Base[name].Matvec(x)
	for i, mod := range gpt.Deltas {
		if da, ok := mod[name]; ok {
			// Consciousness: conscience scales delta influence (Feature 5)
			effectiveAlpha := gpt.ActiveAlpha[i] * gpt.deltaAlphaScale
			delta := da.Apply(x).Scale(effectiveAlpha)
			y = y.Add(delta)
		}
	}
	return y
}

// rrpramScores computes the op-33 low-rank RRPRAM scores for head h: the query
// x (full nEmbd) scored against T key positions via (x @ Wr_a[h]) @ Wr_b[h].
// wrA is [NHead·nEmbd × R] (head h block = rows [h·nEmbd : (h+1)·nEmbd]), wrB is
// [NHead·R × BlockSize] (head h block = rows [h·R : (h+1)·R]). This is the exact
// arithmetic of notorch's nt_rrpram_lowrank_attention, so Go inference and the
// notorch trainer run one identical model (S2). Verified by TestRRPRAMOp33Parity.
func rrpramScores(wrA, wrB *MatrixParam, h, nEmbd, T int, x []float64) []float64 {
	R := wrA.Nin
	u := make([]float64, R)
	aBase := h * nEmbd
	for d := 0; d < nEmbd; d++ {
		xd := x[d]
		row := wrA.Rows[aBase+d].Data
		for r := 0; r < R; r++ {
			u[r] += xd * row[r]
		}
	}
	bBase := h * R
	out := make([]float64, T)
	for j := 0; j < T; j++ {
		var s float64
		for r := 0; r < R; r++ {
			s += u[r] * wrB.Rows[bBase+r].Data[j]
		}
		out[j] = s
	}
	return out
}

// ForwardStep runs one token through the model, updating KV cache.
func (gpt *GPT) ForwardStep(tokenID, posID int, keys, values [][]*Vec) *Vec {
	tokEmb := gpt.Base["wte"].Rows[tokenID]
	posEmb := gpt.Base["wpe"].Rows[posID%gpt.BlockSize]
	x := tokEmb.Add(posEmb)

	// notorch: allocate saved-state slices if needed
	if len(gpt.layerInputs) < gpt.NLayer {
		gpt.layerInputs = make([]*Vec, gpt.NLayer)
	}
	if len(gpt.mlpInputs) < gpt.NLayer {
		gpt.mlpInputs = make([]*Vec, gpt.NLayer)
	}
	if len(gpt.mlpIntermediates) < gpt.NLayer {
		gpt.mlpIntermediates = make([]*Vec, gpt.NLayer)
	}

	for li := 0; li < gpt.NLayer; li++ {
		lk := gpt.layerKeys[li]

		// ---- Attention ----
		xRes := x
		x = RMSNorm(x)

		// notorch: save POST-RMSNorm input for attention adapters (wq/wk/wv/wo)
		gpt.layerInputs[li] = x

		q := gpt.applyWithDeltas(lk.wq, x)
		k := gpt.applyWithDeltas(lk.wk, x)
		v := gpt.applyWithDeltas(lk.wv, x)

		keys[li] = append(keys[li], k)
		values[li] = append(values[li], v)

		// Sliding window: keep only last BlockSize entries in KV cache
		if len(keys[li]) > gpt.BlockSize {
			keys[li] = keys[li][len(keys[li])-gpt.BlockSize:]
			values[li] = values[li][len(values[li])-gpt.BlockSize:]
		}

		// And lo, each head shall choose its nature: content, rrpram, or the sacred hybrid of both.
		T := len(keys[li])
		headOutputs := make([]*Vec, gpt.NHead)
		for h := 0; h < gpt.NHead; h++ {
			hs := h * gpt.HeadDim
			he := hs + gpt.HeadDim
			htype := "content"
			if h < len(CFG.HeadTypes) {
				htype = CFG.HeadTypes[h]
			}

			vh := make([]*Vec, T)
			for t := 0; t < T; t++ {
				vh[t] = values[li][t].Slice(hs, he)
			}

			// Content attention logits (QK^T with RoPE)
			var contentLogits []*Scalar
			if htype == "content" || htype == "hybrid" {
				qh := q.Slice(hs, he)
				qh = RoPERotate(qh, posID, gpt.HeadDim)
				contentLogits = make([]*Scalar, T)
				invSqrt := 1.0 / math.Sqrt(float64(gpt.HeadDim))
				for t := 0; t < T; t++ {
					khT := keys[li][t].Slice(hs, he)
					khT = RoPERotate(khT, t, gpt.HeadDim)
					contentLogits[t] = qh.Dot(khT).MulF(invSqrt)
				}
			}

			// RRPRAM attention scores — low-rank op-33 (Resonance form), the SAME
			// math the notorch trainer runs (S2: train ≡ infer). The current query
			// (full-D post-RMSNorm x) scores every cached key j via
			// scores[j] = ((x @ Wr_a[h]) @ Wr_b[h])[j]. Wr_a[h] = wr_a rows
			// [h·NEmbd : (h+1)·NEmbd] ([NEmbd × R]); Wr_b[h] = wr_b rows
			// [h·R : (h+1)·R] ([R × BlockSize]). Replaces the never-trained
			// position-bias w_pattern (07_AUDIT B1).
			var rrpramLogits []*Scalar
			haveRRPRAM := false
			if htype == "rrpram" || htype == "hybrid" {
				wrA := gpt.Base[lk.wrA]
				wrB := gpt.Base[lk.wrB]
				if wrA != nil && wrB != nil {
					haveRRPRAM = true
					scores := rrpramScores(wrA, wrB, h, gpt.NEmbd, T, x.Data)
					rrpramLogits = make([]*Scalar, T)
					for j := 0; j < T; j++ {
						rrpramLogits[j] = NewScalar(scores[j])
					}
				}
			}

			// Dispatch by head type. Hybrid blends at the OUTPUT level (two
			// separately-softmaxed attentions), matching the trainer's frozen-gate
			// blend: out = (1-a)·content_out + a·rrpram_out, a = sigmoid(alpha).
			switch {
			case htype == "rrpram" && haveRRPRAM:
				headOutputs[h] = AttentionWeightedSum(ScalarSoftmax(rrpramLogits), vh)
			case htype == "hybrid" && haveRRPRAM:
				aVal := 1.0 / (1.0 + math.Exp(-gpt.Base[lk.headAlpha[h]].Rows[0].Data[0])) // sigmoid(alpha), frozen
				cOut := AttentionWeightedSum(ScalarSoftmax(contentLogits), vh)
				rOut := AttentionWeightedSum(ScalarSoftmax(rrpramLogits), vh)
				headOutputs[h] = cOut.Scale(1.0 - aVal).Add(rOut.Scale(aVal))
			default: // content, or a hybrid/rrpram head whose factors are not yet allocated
				headOutputs[h] = AttentionWeightedSum(ScalarSoftmax(contentLogits), vh)
			}
		}

		xAttn := Concat(headOutputs)
		attnOut := gpt.applyWithDeltas(lk.wo, xAttn)
		x = xRes.Add(attnOut.Scale(gpt.residualAlpha))

		// ---- Gated MLP (SwiGLU-ish) ----
		xRes = x
		x = RMSNorm(x)

		// notorch: save POST-RMSNorm input for MLP adapters (fc_g/fc_v)
		gpt.mlpInputs[li] = x

		g := gpt.applyWithDeltas(lk.fcG, x).SiLU() // gate (SwiGLU)
		u := gpt.applyWithDeltas(lk.fcV, x)        // value
		mlpX := g.MulVec(u)                        // gating

		// notorch: save g*u intermediate for fc2 adapter input (4*NEmbd dimension)
		gpt.mlpIntermediates[li] = mlpX

		mlpOut := gpt.applyWithDeltas(lk.fc2, mlpX)
		x = xRes.Add(mlpOut.Scale(gpt.residualAlpha))
	}

	x = RMSNorm(x)
	gpt.lastHidden = x // notorch: save final hidden state before lm_head
	logits := gpt.applyWithDeltas("lm_head", x)
	return logits
}

// LossOnSequence computes cross-entropy loss for a token sequence.
func (gpt *GPT) LossOnSequence(ids []int) *Scalar {
	n := CFG.BlockSize
	if len(ids)-1 < n {
		n = len(ids) - 1
	}
	if n <= 0 {
		return NewScalar(0.0)
	}

	keys := make([][]*Vec, gpt.NLayer)
	values := make([][]*Vec, gpt.NLayer)
	for i := 0; i < gpt.NLayer; i++ {
		keys[i] = make([]*Vec, 0)
		values[i] = make([]*Vec, 0)
	}

	totalLoss := NewScalar(0.0)
	for pos := 0; pos < n; pos++ {
		logits := gpt.ForwardStep(ids[pos], pos, keys, values)
		totalLoss = totalLoss.AddS(CrossEntropyLoss(logits, ids[pos+1]))
	}
	return totalLoss.MulF(1.0 / float64(n))
}

// QuickLoss computes average loss on a few random docs without backward.
// Used for self-meta-learning: measure loss before/after burst.
func (gpt *GPT) QuickLoss(tok *EvolvingTokenizer, docs []string, n int) float64 {
	if len(docs) == 0 {
		return 0
	}
	gradEnabled.Store(false)
	defer func() { gradEnabled.Store(true) }()
	total := 0.0
	for i := 0; i < n; i++ {
		doc := docs[rand.Intn(len(docs))]
		ids := tok.Encode(doc)
		if len(ids) > 1 {
			loss := gpt.LossOnSequence(ids)
			total += loss.Data
		}
	}
	return total / float64(n)
}

func sliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ============================================================
// 5b) CONSCIOUSNESS — mathematical self-awareness
// ============================================================

// ConscienceCheck tracks generation quality over time.
// If entropy trend rises (output degrading), soften delta influence.
// If entropy trend falls (improving), recover delta influence.
// "I notice I'm getting worse and pull back."
func (gpt *GPT) ConscienceCheck(genMeanEntropy float64) {
	gpt.generationEntropyHistory = append(gpt.generationEntropyHistory, genMeanEntropy)
	w := CFG.ConscienceWindow
	if len(gpt.generationEntropyHistory) > w {
		gpt.generationEntropyHistory = gpt.generationEntropyHistory[len(gpt.generationEntropyHistory)-w:]
	}
	if len(gpt.generationEntropyHistory) < 3 {
		return // not enough data
	}
	// Linear regression slope on entropy history
	n := float64(len(gpt.generationEntropyHistory))
	sumX, sumY, sumXY, sumX2 := 0.0, 0.0, 0.0, 0.0
	for i, e := range gpt.generationEntropyHistory {
		x := float64(i)
		sumX += x
		sumY += e
		sumXY += x * e
		sumX2 += x * x
	}
	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX + 1e-12)

	if slope > 0.01 {
		// Entropy increasing — generation degrading, reduce delta influence
		gpt.deltaAlphaScale *= CFG.ConscienceDecay
		if gpt.deltaAlphaScale < CFG.ConscienceFloor {
			gpt.deltaAlphaScale = CFG.ConscienceFloor
		}
	} else if slope < -0.01 {
		// Entropy decreasing — improving, recover delta influence
		gpt.deltaAlphaScale *= CFG.ConscienceRecovery
		if gpt.deltaAlphaScale > 1.0 {
			gpt.deltaAlphaScale = 1.0
		}
	}
}

// ComputeSelfPredictionError measures how "surprised" the model is by a prompt.
// Forward pass on ids, compute cross-entropy between predicted and actual tokens.
// Higher error = "I didn't expect this input" = increase attention.
func (gpt *GPT) ComputeSelfPredictionError(ids []int) float64 {
	if len(ids) < 2 {
		return 0.0
	}
	keys := make([][]*Vec, gpt.NLayer)
	values := make([][]*Vec, gpt.NLayer)
	for i := 0; i < gpt.NLayer; i++ {
		keys[i] = make([]*Vec, 0)
		values[i] = make([]*Vec, 0)
	}

	totalCE := 0.0
	count := 0
	for pos := 0; pos < len(ids)-1; pos++ {
		logits := gpt.ForwardStep(ids[pos], pos, keys, values)
		// Cross-entropy: -log(p[actual_next_token])
		probs := SoftmaxProbs(logits.Data)
		target := ids[pos+1]
		if target < len(probs) && probs[target] > 1e-12 {
			totalCE -= math.Log(probs[target])
		} else {
			totalCE += 10.0 // max penalty for unknown token
		}
		count++
	}
	if count == 0 {
		return 0.0
	}
	return totalCE / float64(count)
}

// ============================================================
// 6) SQLITE MEMORY — and a small ghost shall remember
// ============================================================

func initDB(dbPath string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, err
	}
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS messages(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ts REAL NOT NULL,
			role TEXT NOT NULL,
			text TEXT NOT NULL
		)`)
	if err != nil {
		return nil, err
	}
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS corpus_events(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ts REAL NOT NULL,
			added_chars INTEGER NOT NULL,
			note TEXT
		)`)
	if err != nil {
		return nil, err
	}
	// And lo, the organism shall write its own autobiography in numbers.
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS growth(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ts REAL NOT NULL,
			step INTEGER NOT NULL,
			vocab_size INTEGER NOT NULL,
			n_params INTEGER NOT NULL,
			n_deltas INTEGER NOT NULL,
			corpus_chars INTEGER NOT NULL,
			loss REAL,
			gamma_sparsity REAL,
			gamma_magnitude REAL,
			note TEXT
		)`)
	if err != nil {
		return nil, err
	}
	// And lo, the organism shall track not just what it is, but where it is going.
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS syntropy_log(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ts REAL NOT NULL,
			entropy_before REAL,
			entropy_after REAL,
			syntropy_delta REAL,
			field_deviation REAL,
			purpose_magnitude REAL,
			purpose_alignment REAL,
			action_taken TEXT,
			note TEXT
		)`)
	if err != nil {
		return nil, err
	}
	return db, nil
}

func dbAddMessage(db *sql.DB, role, text string) {
	db.Exec("INSERT INTO messages(ts, role, text) VALUES(?,?,?)",
		float64(time.Now().UnixMilli())/1000.0, role, text)
}

func dbRecentMessages(db *sql.DB, limit int) []struct{ Role, Text string } {
	rows, err := db.Query("SELECT role, text FROM messages ORDER BY id DESC LIMIT ?", limit)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var msgs []struct{ Role, Text string }
	for rows.Next() {
		var role, text string
		rows.Scan(&role, &text)
		msgs = append(msgs, struct{ Role, Text string }{role, text})
	}
	// Reverse to chronological order
	for i, j := 0, len(msgs)-1; i < j; i, j = i+1, j-1 {
		msgs[i], msgs[j] = msgs[j], msgs[i]
	}
	return msgs
}

func dbLogGrowth(db *sql.DB, model *GPT, tok *EvolvingTokenizer, docs []string, lossVal float64, note string) {
	nParams := 0
	for _, m := range model.Base {
		nParams += m.Nout * m.Nin
	}
	for _, mod := range model.Deltas {
		for _, da := range mod {
			nParams += da.A.Nout*da.A.Nin + da.B.Nout*da.B.Nin
		}
	}
	corpusChars := 0
	for _, d := range docs {
		corpusChars += len(d)
	}
	gs := model.GammaStats()
	db.Exec(`INSERT INTO growth(ts,step,vocab_size,n_params,n_deltas,corpus_chars,loss,gamma_sparsity,gamma_magnitude,note)
		VALUES(?,?,?,?,?,?,?,?,?,?)`,
		float64(time.Now().UnixMilli())/1000.0,
		0, tok.VocabSize, nParams, len(model.Deltas), corpusChars,
		lossVal, gs.Sparsity, gs.Magnitude, note)
}

// And lo, the organism shall read its own growth chart and weep with pride.
func dbDescribeGrowth(db *sql.DB) []map[string]interface{} {
	rows, err := db.Query("SELECT ts, step, vocab_size, n_params, n_deltas, corpus_chars, loss, gamma_sparsity, gamma_magnitude, note FROM growth ORDER BY id DESC LIMIT 20")
	if err != nil {
		return nil
	}
	defer rows.Close()
	var result []map[string]interface{}
	for rows.Next() {
		var ts, loss, gSpar, gMag float64
		var step, vs, np, nd, cc int
		var note sql.NullString
		rows.Scan(&ts, &step, &vs, &np, &nd, &cc, &loss, &gSpar, &gMag, &note)
		entry := map[string]interface{}{
			"ts": ts, "step": step, "vocab_size": vs, "n_params": np,
			"n_deltas": nd, "corpus_chars": cc, "loss": loss,
			"gamma_sparsity": gSpar, "gamma_magnitude": gMag,
		}
		if note.Valid {
			entry["note"] = note.String
		}
		result = append(result, entry)
	}
	return result
}

// ============================================================
// 7) CORPUS RESERVOIR — and nonames.txt shall not bloat forever
// ============================================================

func loadCorpusLines(path string) []string {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	var lines []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		ln := strings.TrimSpace(scanner.Text())
		if ln != "" {
			if len(ln) > CFG.MaxLineChars {
				ln = ln[:CFG.MaxLineChars]
			}
			lines = append(lines, ln)
		}
	}
	return lines
}

// splitCorpusLine cuts one eaten text into corpus lines, none longer than
// maxChars. Routing repair 3: loadCorpusLines truncates every line at
// CFG.MaxLineChars and a DNA fragment is padded toward
// CFG.DNAFragmentTargetBytes and was appended as one line, so a fragment
// reached `docs` only as its first 240 bytes — measured on the live run,
// 4.7-4.8 % of a 5 KB sibling fragment and 76.7 % of a place fragment, where
// the quarter that fell off was exactly the clause saying whether the phone had
// moved. Cutting on sentence ends instead leaves the byte bound
// MaxCorpusLines × MaxLineChars untouched (the line count becomes the binding
// cap) while the whole fragment reaches the field.
//
// A sentence ends at '.', '!' or '?' followed by a space or by the end of the
// text, so "22.5 °C" and "2026-09-14T18:48" stay whole. A run with no sentence
// end inside maxChars is cut at the last space before the bound, or at the
// bound on a rune boundary — which is what loadCorpusLines would have done to
// it anyway, only without the rune care.
func splitCorpusLine(text string, maxChars int) []string {
	text = strings.TrimSpace(strings.ReplaceAll(text, "\n", " "))
	if text == "" {
		return nil
	}
	if maxChars <= 0 {
		return []string{text}
	}
	var out []string
	emit := func(s string) {
		s = strings.TrimSpace(s)
		for len(s) > maxChars {
			cut := maxChars
			for cut > 0 && !utf8.RuneStart(s[cut]) {
				cut--
			}
			if sp := strings.LastIndexByte(s[:cut], ' '); sp > maxChars/2 {
				cut = sp
			}
			if cut == 0 {
				break
			}
			if head := strings.TrimSpace(s[:cut]); head != "" {
				out = append(out, head)
			}
			s = strings.TrimSpace(s[cut:])
		}
		if s != "" {
			out = append(out, s)
		}
	}
	start := 0
	for i := 0; i < len(text); i++ {
		if c := text[i]; c != '.' && c != '!' && c != '?' {
			continue
		}
		if i+1 < len(text) && text[i+1] != ' ' && text[i+1] != '\t' {
			continue
		}
		emit(text[start : i+1])
		start = i + 1
	}
	if start < len(text) {
		emit(text[start:])
	}
	return out
}

func saveCorpusLines(path string, lines []string) {
	f, err := os.Create(path)
	if err != nil {
		return
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	for _, ln := range lines {
		ln = strings.ReplaceAll(ln, "\n", " ")
		fmt.Fprintln(w, strings.TrimSpace(ln))
	}
	w.Flush()
}

func normalizeText(s string) string {
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\t", " ")
	return strings.Join(strings.Fields(s), " ")
}

func extractCandidateSentences(msgs []struct{ Role, Text string }) []string {
	var out []string
	for _, msg := range msgs {
		t := normalizeText(msg.Text)
		if t == "" {
			continue
		}
		tag := "A:"
		if msg.Role == "user" {
			tag = "H:"
		}

		buf := ""
		for _, ch := range t {
			buf += string(ch)
			if ch == '.' || ch == '!' || ch == '?' {
				s := strings.TrimSpace(buf)
				if len(s) >= 6 {
					out = append(out, tag+" "+s)
				}
				buf = ""
			}
		}
		s := strings.TrimSpace(buf)
		if len(s) >= 12 {
			out = append(out, tag+" "+s)
		}
	}

	// Stable dedup
	seen := make(map[string]bool)
	var uniq []string
	for _, s := range out {
		k := strings.ToLower(s)
		if !seen[k] {
			seen[k] = true
			uniq = append(uniq, s)
		}
	}
	return uniq
}

func reservoirMixKeep(lines, newSents []string, maxLines int) []string {
	combined := append(append([]string{}, lines...), newSents...)
	half := maxLines / 2
	var newest, older []string
	if len(combined) > half {
		newest = combined[len(combined)-half:]
		older = combined[:len(combined)-half]
	} else {
		newest = combined
	}

	rand.Shuffle(len(older), func(i, j int) { older[i], older[j] = older[j], older[i] })
	keep := maxLines - len(newest)
	if keep < 0 {
		keep = 0
	}
	if keep > len(older) {
		keep = len(older)
	}
	final := append(older[:keep], newest...)

	// Dedup
	seen := make(map[string]bool)
	var dedup []string
	for _, s := range final {
		k := strings.ToLower(s)
		if !seen[k] {
			seen[k] = true
			if len(s) > CFG.MaxLineChars {
				s = s[:CFG.MaxLineChars]
			}
			dedup = append(dedup, s)
		}
	}
	if len(dedup) > maxLines {
		dedup = dedup[len(dedup)-maxLines:]
	}
	return dedup
}

func updateReservoirCorpus(db *sql.DB, corpusPath string, maxLines int) int {
	msgs := dbRecentMessages(db, 64)
	newSents := extractCandidateSentences(msgs)
	if len(newSents) == 0 {
		// --evolution never writes the messages table (REPL only), so the cap
		// must hold without it (repair 5). A DNA fragment is one ~5 KB line
		// and dnaRead appends every one it eats, so the bound is on bytes as
		// well as lines: maxLines × MaxLineChars is the most loadCorpusLines
		// can ever hand back from this file. Under both caps the file is left
		// alone; over either it is rewritten as the reservoir.
		if maxLines <= 0 {
			return 0
		}
		fi, err := os.Stat(corpusPath)
		if err != nil {
			return 0
		}
		lines := loadCorpusLines(corpusPath)
		if len(lines) <= maxLines && fi.Size() <= int64(maxLines)*int64(CFG.MaxLineChars) {
			return 0
		}
		saveCorpusLines(corpusPath, reservoirMixKeep(lines, nil, maxLines))
		return 0
	}

	lines := loadCorpusLines(corpusPath)
	before := 0
	for _, x := range lines {
		before += len(x)
	}

	final := reservoirMixKeep(lines, newSents, maxLines)
	saveCorpusLines(corpusPath, final)

	after := 0
	for _, x := range final {
		after += len(x)
	}
	added := after - before
	if added < 0 {
		added = 0
	}

	db.Exec("INSERT INTO corpus_events(ts, added_chars, note) VALUES(?,?,?)",
		float64(time.Now().UnixMilli())/1000.0, added,
		fmt.Sprintf("reservoir_update +%d sents", len(newSents)))
	return added
}

func computeNewCorpusMass(db *sql.DB, lastEventID int) (int, int) {
	rows, err := db.Query("SELECT id, added_chars FROM corpus_events WHERE id > ? ORDER BY id ASC", lastEventID)
	if err != nil {
		return 0, lastEventID
	}
	defer rows.Close()
	mass := 0
	newLastID := lastEventID
	for rows.Next() {
		var id, chars int
		rows.Scan(&id, &chars)
		mass += chars
		newLastID = id
	}
	return mass, newLastID
}

// ============================================================
// 8) CHECKPOINTING — modular, compatible, no merge-amnesia
// ============================================================

type CheckpointJSON struct {
	Cfg       json.RawMessage          `json:"cfg"`
	Tokenizer TokenizerJSON            `json:"tokenizer"`
	Base      map[string][][][]float64 `json:"base"` // name -> rows -> cols (but we store as [][]float64)
	Alpha     []float64                `json:"alpha"`
	Deltas    []map[string]DeltaJSON   `json:"deltas"`
}

func intPtr(v int) *int { return &v }

// We need a different approach - Base stores name -> [][]float64 (matrix rows)
type CheckpointData struct {
	Cfg                 json.RawMessage        `json:"cfg"`
	Tokenizer           TokenizerJSON          `json:"tokenizer"`
	Base                map[string][][]float64 `json:"base"`
	Alpha               []float64              `json:"alpha"`
	Deltas              []map[string]DeltaJSON `json:"deltas"`
	InitEmbedSnapshot   [][]float64            `json:"init_embed_snapshot,omitempty"`
	GlobalStep          int                    `json:"global_step"`
	GrowthStepOffset    int                    `json:"growth_step_offset"`
	LastWarmupStage     *int                   `json:"last_warmup_stage,omitempty"`
	CorpusIngestedTotal int                    `json:"corpus_ingested_total"`
}

type TokenizerJSON struct {
	Tokens       []string   `json:"tokens"`
	BPEEnabled   bool       `json:"bpe_enabled"`
	Merges       [][]string `json:"merges"`
	TrainedChars int        `json:"trained_chars"`
}

type DeltaJSON struct {
	A [][]float64 `json:"A"`
	B [][]float64 `json:"B"`
}

func serializeMatrixParam(mp *MatrixParam) [][]float64 {
	rows := make([][]float64, mp.Nout)
	for i, row := range mp.Rows {
		rows[i] = make([]float64, len(row.Data))
		copy(rows[i], row.Data)
	}
	return rows
}

func deserializeMatrixParam(data [][]float64) *MatrixParam {
	if len(data) == 0 {
		return &MatrixParam{}
	}
	mp := &MatrixParam{
		Nout: len(data),
		Nin:  len(data[0]),
		Rows: make([]*Vec, len(data)),
	}
	for i, row := range data {
		d := make([]float64, len(row))
		copy(d, row)
		mp.Rows[i] = NewVecWithGrad(d) // loaded params always need grad
	}
	return mp
}

// debouncer coalesces rapid calls: allow() returns true at most once per
// minInterval seconds (minInterval <= 0 disables = always allow).
type debouncer struct {
	mu   sync.Mutex
	last float64
}

func (d *debouncer) allow(now, minInterval float64) bool {
	if minInterval <= 0 {
		return true
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if now-d.last >= minInterval {
		d.last = now
		return true
	}
	return false
}

var ckptDebounce debouncer

func SaveCheckpoint(model *GPT, tok *EvolvingTokenizer, path string) error {
	// Write-storm throttle: coalesce rapid DEFAULT-path (periodic) checkpoints —
	// the full-model JSON write is heavy and growth/burst events can storm it. An
	// explicit path (e.g. the mitosis parent checkpoint the child must load) is
	// NEVER debounced.
	if path == "" && !ckptDebounce.allow(float64(time.Now().UnixMilli())/1000.0, CFG.CheckpointMinInterval) {
		return nil
	}
	if path == "" {
		path = CFG.CkptPath
	}

	// Atomic write: temp file + rename (prevents corruption on crash)
	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(f, 1<<20)
	err = writeCheckpointJSON(bw, model, tok)
	if err == nil {
		err = bw.Flush()
	}
	f.Close()
	if err != nil {
		os.Remove(tmpPath)
		return err
	}
	return os.Rename(tmpPath, path)
}

// writeCheckpointJSON streams the checkpoint that CheckpointData describes,
// matrix by matrix, instead of building one (repair 9). The old path paid for
// the weights three times over: a [][]float64 copy of every matrix in a
// CheckpointData, then json.Encoder's own buffer, which holds the entire
// document — 110 MB for a stage-4 organism — before a byte reaches the file.
// Four organisms doing that within two minutes is what took the phone's
// userspace down on 2026-09-13. Here the only large allocation alive at once is
// one row. The bytes are identical to the old encoder's; checkpoint_stream_test
// compares them.
func writeCheckpointJSON(bw *bufio.Writer, model *GPT, tok *EvolvingTokenizer) error {
	cfgJSON, err := json.Marshal(CFG)
	if err != nil {
		return err
	}
	merges := make([][]string, len(tok.Merges))
	for i, m := range tok.Merges {
		merges[i] = []string{m.A, m.B}
	}
	tokJSON, err := json.Marshal(TokenizerJSON{
		Tokens:       tok.Tokens,
		BPEEnabled:   tok.BPEEnabled,
		Merges:       merges,
		TrainedChars: tok.TrainedChars,
	})
	if err != nil {
		return err
	}
	bw.WriteString(`{"cfg":`)
	bw.Write(cfgJSON)
	bw.WriteString(`,"tokenizer":`)
	bw.Write(tokJSON)

	// Map keys are emitted in sorted order, as encoding/json does. The map itself
	// is always an object, never null: the old path copied into a fresh map.
	bw.WriteString(`,"base":{`)
	names := make([]string, 0, len(model.Base))
	for k := range model.Base {
		names = append(names, k)
	}
	sort.Strings(names)
	for i, k := range names {
		if i > 0 {
			bw.WriteString(",")
		}
		key, err := json.Marshal(k)
		if err != nil {
			return err
		}
		bw.Write(key)
		bw.WriteString(":")
		if err := writeMatrixParamJSON(bw, model.Base[k]); err != nil {
			return err
		}
	}
	bw.WriteString("}")

	alpha, err := json.Marshal(model.ActiveAlpha)
	if err != nil {
		return err
	}
	bw.WriteString(`,"alpha":`)
	bw.Write(alpha)

	// Likewise a slice of the same length, never null.
	bw.WriteString(`,"deltas":[`)
	for i, mod := range model.Deltas {
		if i > 0 {
			bw.WriteString(",")
		}
		dnames := make([]string, 0, len(mod))
		for k := range mod {
			dnames = append(dnames, k)
		}
		sort.Strings(dnames)
		bw.WriteString("{")
		for j, k := range dnames {
			if j > 0 {
				bw.WriteString(",")
			}
			key, err := json.Marshal(k)
			if err != nil {
				return err
			}
			bw.Write(key)
			bw.WriteString(`:{"A":`)
			if err := writeMatrixParamJSON(bw, mod[k].A); err != nil {
				return err
			}
			bw.WriteString(`,"B":`)
			if err := writeMatrixParamJSON(bw, mod[k].B); err != nil {
				return err
			}
			bw.WriteString("}")
		}
		bw.WriteString("}")
	}
	bw.WriteString("]")

	if len(model.InitEmbedSnapshot) > 0 { // json:"...,omitempty"
		snap, err := json.Marshal(model.InitEmbedSnapshot)
		if err != nil {
			return err
		}
		bw.WriteString(`,"init_embed_snapshot":`)
		bw.Write(snap)
	}
	fmt.Fprintf(bw, `,"global_step":%d,"growth_step_offset":%d,"last_warmup_stage":%d,"corpus_ingested_total":%d}`+"\n",
		model.globalStep, model.growthStepOffset, model.lastWarmupStage, model.corpusIngestedTotal)
	return nil
}

// writeMatrixParamJSON emits what serializeMatrixParam would have produced, one
// row at a time: Nout rows, a missing row as null (the nil slot make() leaves),
// an empty row as [] (the empty slice make(...,0) produces).
func writeMatrixParamJSON(bw *bufio.Writer, mp *MatrixParam) error {
	if mp == nil {
		bw.WriteString("null")
		return nil
	}
	bw.WriteString("[")
	for i := 0; i < mp.Nout; i++ {
		if i > 0 {
			bw.WriteString(",")
		}
		if i >= len(mp.Rows) || mp.Rows[i] == nil {
			bw.WriteString("null")
			continue
		}
		if len(mp.Rows[i].Data) == 0 {
			bw.WriteString("[]")
			continue
		}
		row, err := json.Marshal(mp.Rows[i].Data)
		if err != nil {
			return err
		}
		if _, err := bw.Write(row); err != nil {
			return err
		}
	}
	bw.WriteString("]")
	return nil
}

// shutdownDrain is how long the process waits for the trainer loop to answer the
// signal before saving anyway. Long enough for a tick, far shorter than a warmup.
const shutdownDrain = 10 * time.Second

// saveOnShutdown writes the organism's state on the way out of evolution mode.
// It passes CFG.CkptPath explicitly rather than "": the empty path is the
// debounced periodic path, and a shutdown that lands within CheckpointMinInterval
// of the last burst save would be silently dropped — which is the whole failure
// this repairs. A session that ends every time by SIGTERM (capped runs) keeps its
// progress only here. `phase` names the loop that was interrupted — "evolution"
// for the tick loop, "init" for the bootstrap climb (repair 10) — so the log
// line says which of the two saved.
func saveOnShutdown(phase string, model *GPT, tok *EvolvingTokenizer, why string) {
	path := CFG.CkptPath
	if path == "" {
		path = "molequla_ckpt.json"
	}
	model.mu.Lock()
	err := SaveCheckpoint(model, tok, path)
	model.mu.Unlock()
	if err != nil {
		fmt.Fprintf(os.Stderr, "[%s] checkpoint NOT saved on %s: %v\n", phase, why, err)
		return
	}
	fmt.Printf("[%s] checkpoint saved on %s\n", phase, why)
}

// checkpointStream is what readCheckpointStream produces: the same fields
// CheckpointData describes, except that the weights arrive already built into
// their final *MatrixParam / DeltaModule form. Nothing here holds a second copy
// of a matrix.
type checkpointStream struct {
	Cfg                 json.RawMessage
	Tokenizer           TokenizerJSON
	Base                map[string]*MatrixParam
	Alpha               []float64
	Deltas              []DeltaModule
	InitEmbedSnapshot   [][]float64
	GlobalStep          int
	GrowthStepOffset    int
	LastWarmupStage     *int
	CorpusIngestedTotal int
}

// readCheckpointStream walks the checkpoint document token by token (repair 10).
// The old path was one json.Decoder.Decode into a CheckpointData, which paid for
// the weights three times over: the decoder buffered the whole document (105.6
// MB for the stage-4 organism in molequla-run/earth), the CheckpointData held
// every matrix again as [][]float64, and deserializeMatrixParam then built the
// *Vec rows from that — measured at +368 MB of high-water mark in repair 9,
// against a save path that by then cost +0. Here the decoder's buffer compacts
// down to the largest single value it is asked for, which is one row, and each
// row is handed straight to NewVecWithGrad without an intermediate copy.
//
// Field order is not assumed and unknown keys are skipped, so a checkpoint
// written by the C / Rust / JS cores still loads. A truncated file ends the
// token stream with io.ErrUnexpectedEOF, which comes back as an error from
// whichever Token()/Decode() call reached the end — never a panic.
func readCheckpointStream(r io.Reader) (*checkpointStream, error) {
	dec := json.NewDecoder(bufio.NewReaderSize(r, 1<<16))
	ck := &checkpointStream{Base: make(map[string]*MatrixParam)}
	if err := expectDelim(dec, '{'); err != nil {
		return nil, err
	}
	for dec.More() {
		key, err := expectKey(dec)
		if err != nil {
			return nil, err
		}
		switch key {
		case "cfg":
			err = dec.Decode(&ck.Cfg)
		case "tokenizer":
			err = dec.Decode(&ck.Tokenizer)
		case "base":
			err = decodeMatrixMap(dec, ck.Base)
		case "alpha":
			err = dec.Decode(&ck.Alpha)
		case "deltas":
			ck.Deltas, err = decodeDeltaModules(dec)
		case "init_embed_snapshot":
			ck.InitEmbedSnapshot, err = decodeFloatRows(dec)
		case "global_step":
			err = dec.Decode(&ck.GlobalStep)
		case "growth_step_offset":
			err = dec.Decode(&ck.GrowthStepOffset)
		case "last_warmup_stage":
			err = dec.Decode(&ck.LastWarmupStage)
		case "corpus_ingested_total":
			err = dec.Decode(&ck.CorpusIngestedTotal)
		default:
			var skip json.RawMessage
			err = dec.Decode(&skip)
		}
		if err != nil {
			return nil, fmt.Errorf("checkpoint field %q: %w", key, err)
		}
	}
	if err := expectDelim(dec, '}'); err != nil {
		return nil, err
	}
	return ck, nil
}

// expectDelim consumes one token and insists it is the given delimiter.
func expectDelim(dec *json.Decoder, want json.Delim) error {
	t, err := dec.Token()
	if err != nil {
		return err
	}
	if d, ok := t.(json.Delim); !ok || d != want {
		return fmt.Errorf("checkpoint: expected %q, got %v", want, t)
	}
	return nil
}

// expectKey consumes one token and insists it is an object key.
func expectKey(dec *json.Decoder) (string, error) {
	t, err := dec.Token()
	if err != nil {
		return "", err
	}
	s, ok := t.(string)
	if !ok {
		return "", fmt.Errorf("checkpoint: expected an object key, got %v", t)
	}
	return s, nil
}

// decodeMatrixParam reads one matrix — the rows writeMatrixParamJSON emits —
// into its final storage. The shapes it must reproduce are exactly what
// deserializeMatrixParam produced from the same JSON: a null or empty matrix is
// an empty MatrixParam, a null row is an empty row, and Nin comes from row 0.
func decodeMatrixParam(dec *json.Decoder) (*MatrixParam, error) {
	t, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if t == nil { // JSON null — the old path decoded it to a nil [][]float64
		return &MatrixParam{}, nil
	}
	if d, ok := t.(json.Delim); !ok || d != '[' {
		return nil, fmt.Errorf("checkpoint: a matrix must be an array, got %v", t)
	}
	mp := &MatrixParam{}
	for dec.More() {
		var row []float64
		if err := dec.Decode(&row); err != nil {
			return nil, err
		}
		if row == nil {
			row = []float64{}
		}
		if len(mp.Rows) == 0 {
			mp.Nin = len(row)
		}
		mp.Rows = append(mp.Rows, NewVecWithGrad(row)) // loaded params always need grad
	}
	if err := expectDelim(dec, ']'); err != nil {
		return nil, err
	}
	if len(mp.Rows) == 0 {
		return &MatrixParam{}, nil
	}
	mp.Nout = len(mp.Rows)
	return mp, nil
}

// decodeMatrixMap reads the "base" object, matrix by matrix, into dst.
func decodeMatrixMap(dec *json.Decoder, dst map[string]*MatrixParam) error {
	t, err := dec.Token()
	if err != nil {
		return err
	}
	if t == nil {
		return nil
	}
	if d, ok := t.(json.Delim); !ok || d != '{' {
		return fmt.Errorf("checkpoint: base must be an object, got %v", t)
	}
	for dec.More() {
		name, err := expectKey(dec)
		if err != nil {
			return err
		}
		mp, err := decodeMatrixParam(dec)
		if err != nil {
			return fmt.Errorf("base[%q]: %w", name, err)
		}
		dst[name] = mp
	}
	return expectDelim(dec, '}')
}

// decodeFloatRows reads an array of float rows (init_embed_snapshot) one row at
// a time, so the snapshot never exists twice.
func decodeFloatRows(dec *json.Decoder) ([][]float64, error) {
	t, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if t == nil {
		return nil, nil
	}
	if d, ok := t.(json.Delim); !ok || d != '[' {
		return nil, fmt.Errorf("checkpoint: expected an array of rows, got %v", t)
	}
	var rows [][]float64
	for dec.More() {
		var row []float64
		if err := dec.Decode(&row); err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	return rows, expectDelim(dec, ']')
}

// decodeDeltaModules reads the "deltas" array into live DeltaModules.
func decodeDeltaModules(dec *json.Decoder) ([]DeltaModule, error) {
	t, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if t == nil {
		return nil, nil
	}
	if d, ok := t.(json.Delim); !ok || d != '[' {
		return nil, fmt.Errorf("checkpoint: deltas must be an array, got %v", t)
	}
	var mods []DeltaModule
	for dec.More() {
		mod, err := decodeDeltaModule(dec)
		if err != nil {
			return nil, fmt.Errorf("deltas[%d]: %w", len(mods), err)
		}
		mods = append(mods, mod)
	}
	return mods, expectDelim(dec, ']')
}

func decodeDeltaModule(dec *json.Decoder) (DeltaModule, error) {
	t, err := dec.Token()
	if err != nil {
		return nil, err
	}
	mod := make(DeltaModule)
	if t == nil { // the old path built an empty module for a null element
		return mod, nil
	}
	if d, ok := t.(json.Delim); !ok || d != '{' {
		return nil, fmt.Errorf("a delta module must be an object, got %v", t)
	}
	for dec.More() {
		name, err := expectKey(dec)
		if err != nil {
			return nil, err
		}
		da, err := decodeDeltaAdapter(dec)
		if err != nil {
			return nil, fmt.Errorf("%q: %w", name, err)
		}
		mod[name] = da
	}
	return mod, expectDelim(dec, '}')
}

func decodeDeltaAdapter(dec *json.Decoder) (*DeltaAdapter, error) {
	da := &DeltaAdapter{A: &MatrixParam{}, B: &MatrixParam{}}
	t, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if t == nil { // a null adapter is the DeltaJSON zero value: two empty matrices
		return da, nil
	}
	if d, ok := t.(json.Delim); !ok || d != '{' {
		return nil, fmt.Errorf("a delta adapter must be an object, got %v", t)
	}
	for dec.More() {
		name, err := expectKey(dec)
		if err != nil {
			return nil, err
		}
		switch name {
		case "A":
			da.A, err = decodeMatrixParam(dec)
		case "B":
			da.B, err = decodeMatrixParam(dec)
		default:
			var skip json.RawMessage
			err = dec.Decode(&skip)
		}
		if err != nil {
			return nil, fmt.Errorf("%q: %w", name, err)
		}
	}
	return da, expectDelim(dec, '}')
}

func LoadCheckpoint(docs []string, path string) (*GPT, *EvolvingTokenizer, error) {
	if path == "" {
		path = CFG.CkptPath
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	ckpt, err := readCheckpointStream(f)
	if err != nil {
		return nil, nil, err
	}

	// Restore tokenizer
	if len(docs) == 0 {
		docs = []string{"Hello."}
	}
	tok := NewEvolvingTokenizer(docs)
	if len(ckpt.Tokenizer.Tokens) > 0 {
		tok.Tokens = ckpt.Tokenizer.Tokens
		tok.Stoi = make(map[string]int)
		tok.Itos = make(map[int]string)
		for i, t := range tok.Tokens {
			tok.Stoi[t] = i
			tok.Itos[i] = t
		}
		tok.VocabSize = len(tok.Tokens)
	}

	tok.Merges = make([]MergePair, 0)
	tok.MergeToTok = make(map[MergePair]string)
	for _, m := range ckpt.Tokenizer.Merges {
		if len(m) == 2 {
			p := MergePair{m[0], m[1]}
			tok.Merges = append(tok.Merges, p)
			tok.MergeToTok[p] = m[0] + "+" + m[1]
		}
	}
	tok.BPEEnabled = ckpt.Tokenizer.BPEEnabled
	tok.TrainedChars = ckpt.Tokenizer.TrainedChars

	// Restore model dimensions from checkpoint config (ontogenesis may have changed them)
	if len(ckpt.Cfg) > 0 {
		var savedCfg struct {
			NEmbd     int      `json:"n_embd"`
			NLayer    int      `json:"n_layer"`
			NHead     int      `json:"n_head"`
			HeadTypes []string `json:"head_types"`
		}
		if json.Unmarshal(ckpt.Cfg, &savedCfg) == nil {
			if savedCfg.NEmbd > 0 {
				CFG.NEmbd = savedCfg.NEmbd
			}
			if savedCfg.NLayer > 0 {
				CFG.NLayer = savedCfg.NLayer
			}
			if savedCfg.NHead > 0 {
				CFG.NHead = savedCfg.NHead
			}
			if len(savedCfg.HeadTypes) > 0 {
				CFG.HeadTypes = savedCfg.HeadTypes
			}
		}
	}

	// Restore model
	model := NewGPT(tok)
	// The matrices were built row by row during the walk; nothing is copied here.
	model.Base = ckpt.Base
	// Re-establish embedding tie after deserialization (JSON breaks pointer identity)
	if CFG.TieEmbeddings {
		model.Base["lm_head"] = model.Base["wte"]
	}

	model.ActiveAlpha = ckpt.Alpha
	model.Deltas = ckpt.Deltas

	if len(model.Deltas) == 0 {
		model.AddDeltaModule(1.0)
	}

	// Restore init_embed_snapshot (or create from current if not in checkpoint)
	if len(ckpt.InitEmbedSnapshot) > 0 {
		model.InitEmbedSnapshot = ckpt.InitEmbedSnapshot
	} else {
		model.InitEmbedSnapshot = make([][]float64, len(model.Base["wte"].Rows))
		for i, row := range model.Base["wte"].Rows {
			snap := make([]float64, len(row.Data))
			copy(snap, row.Data)
			model.InitEmbedSnapshot[i] = snap
		}
	}

	// Restore global step and growth state
	model.globalStep = ckpt.GlobalStep
	model.growthStepOffset = ckpt.GrowthStepOffset
	model.corpusIngestedTotal = ckpt.CorpusIngestedTotal
	if model.corpusIngestedTotal == 0 {
		// pre-Fix-C checkpoint or fresh — seed the growth clock from the corpus
		for _, d := range docs {
			model.corpusIngestedTotal += len(d)
		}
	}
	if ckpt.LastWarmupStage != nil {
		model.lastWarmupStage = *ckpt.LastWarmupStage
	} else if ckpt.GlobalStep > 0 {
		// Old checkpoint without lastWarmupStage: assume current stage is warmed up
		model.lastWarmupStage = model.CurrentGrowthStage()
	}

	// Ensure hybrid attention weights exist (backward compat with old checkpoints)
	for li := 0; li < CFG.NLayer; li++ {
		for h, htype := range CFG.HeadTypes {
			if htype == "rrpram" || htype == "hybrid" {
				key := fmt.Sprintf("l%d.h%d.w_pattern", li, h)
				if _, ok := model.Base[key]; !ok {
					model.Base[key] = NewMatrixParam(CFG.BlockSize, model.HeadDim, 0.08)
				}
			}
			alphaKey := fmt.Sprintf("l%d.h%d.alpha", li, h)
			if _, ok := model.Base[alphaKey]; !ok {
				m := NewMatrixParam(1, 1, 0.0)
				m.Rows[0].Data[0] = CFG.HybridAlphaInit
				model.Base[alphaKey] = m
			}
		}
		// Inc2: fresh-init per-layer low-rank RRPRAM factors for old checkpoints
		// (which carry only the legacy per-head w_pattern). Safe — w_pattern was
		// never trained (07_AUDIT B1), so there is no signal to migrate.
		model.ensureRRPRAMFactors(li)
	}

	return model, tok, nil
}

// ============================================================
// 9a) QUANTUM BUFFER — trains when ready, not when told
// ============================================================

// And lo, the buffer shall measure not just bytes but novelty, for raw mass means nothing without surprise.
type QuantumBuffer struct {
	mu               sync.Mutex
	AccumulatedBytes int
	UniqueTokens     map[int]bool
	TotalTokens      int
	LastBurstTime    float64
}

func NewQuantumBuffer() *QuantumBuffer {
	return &QuantumBuffer{UniqueTokens: make(map[int]bool)}
}

func (qb *QuantumBuffer) Feed(text string, tok *EvolvingTokenizer) {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	qb.AccumulatedBytes += len(text)
	ids := tok.Encode(text)
	for _, id := range ids {
		qb.UniqueTokens[id] = true
		qb.TotalTokens++
	}
}

func (qb *QuantumBuffer) noveltyScoreLocked() float64 {
	if qb.TotalTokens == 0 {
		return 0.0
	}
	return float64(len(qb.UniqueTokens)) / float64(qb.TotalTokens)
}

func (qb *QuantumBuffer) ShouldTrigger() bool {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	now := float64(time.Now().UnixMilli()) / 1000.0
	bytesOK := qb.AccumulatedBytes >= CFG.QBMinBytes
	noveltyOK := qb.noveltyScoreLocked() >= CFG.QBMinNovelty
	cooldownOK := (now - qb.LastBurstTime) >= CFG.QBCooldownSeconds
	return (bytesOK || noveltyOK) && cooldownOK
}

// SnapshotStats returns accumulated bytes and novelty under one lock.
func (qb *QuantumBuffer) SnapshotStats() (int, float64) {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	return qb.AccumulatedBytes, qb.noveltyScoreLocked()
}

func (qb *QuantumBuffer) Reset() {
	qb.mu.Lock()
	defer qb.mu.Unlock()
	qb.AccumulatedBytes = 0
	qb.UniqueTokens = make(map[int]bool)
	qb.TotalTokens = 0
	qb.LastBurstTime = float64(time.Now().UnixMilli()) / 1000.0
}

// ============================================================
// 9b) COOCCUR FIELD — speech before learning
// ============================================================

// And lo, the corpus shall whisper its statistics, and words shall follow words.
type CooccurField struct {
	Unigram          map[int]float64
	BigramByFirst    map[int]map[int]float64    // prev → {next: count}
	TrigramByContext map[[2]int]map[int]float64 // [prev2,prev1] → {next: count}
	FourgramByCtx    map[[3]int]map[int]float64 // [prev3,prev2,prev1] → {next: count}
	CooccurWindow    map[int]map[int]float64    // token → {nearby_token: count} (Stanley-style proximity)
	UserBoost        map[int]float64            // temporary user word boosts (Leo-style)
	Built            bool
	mu               sync.RWMutex // RWMutex: reads (SampleNext) don't block each other
}

func NewCooccurField() *CooccurField {
	return &CooccurField{
		Unigram:          make(map[int]float64),
		BigramByFirst:    make(map[int]map[int]float64),
		TrigramByContext: make(map[[2]int]map[int]float64),
		FourgramByCtx:    make(map[[3]int]map[int]float64),
		CooccurWindow:    make(map[int]map[int]float64),
		UserBoost:        make(map[int]float64),
	}
}

func (cf *CooccurField) BuildFromCorpus(tok *EvolvingTokenizer, docs []string) {
	// Build into temporary maps first, then swap atomically
	uni := make(map[int]float64)
	bi := make(map[int]map[int]float64)
	tri := make(map[[2]int]map[int]float64)
	four := make(map[[3]int]map[int]float64)
	cooc := make(map[int]map[int]float64)
	window := CFG.CooccurWindowSize

	for _, doc := range docs {
		ids := tok.Encode(doc)
		for _, id := range ids {
			uni[id]++
		}
		for i := 0; i < len(ids)-1; i++ {
			first, second := ids[i], ids[i+1]
			if bi[first] == nil {
				bi[first] = make(map[int]float64)
			}
			bi[first][second]++
		}
		for i := 0; i < len(ids)-2; i++ {
			ctx := [2]int{ids[i], ids[i+1]}
			if tri[ctx] == nil {
				tri[ctx] = make(map[int]float64)
			}
			tri[ctx][ids[i+2]]++
		}
		// 4-grams: deeper context for child+ stages
		for i := 0; i < len(ids)-3; i++ {
			ctx := [3]int{ids[i], ids[i+1], ids[i+2]}
			if four[ctx] == nil {
				four[ctx] = make(map[int]float64)
			}
			four[ctx][ids[i+3]]++
		}
		// Co-occurrence window: "words that resonate together, stay together" (Stanley)
		for i := 0; i < len(ids); i++ {
			center := ids[i]
			start := i - window
			if start < 0 {
				start = 0
			}
			end := i + window + 1
			if end > len(ids) {
				end = len(ids)
			}
			for j := start; j < end; j++ {
				if i != j {
					neighbor := ids[j]
					if cooc[center] == nil {
						cooc[center] = make(map[int]float64)
					}
					cooc[center][neighbor]++
				}
			}
		}
	}
	// Atomic swap under lock
	cf.mu.Lock()
	cf.Unigram = uni
	cf.BigramByFirst = bi
	cf.TrigramByContext = tri
	cf.FourgramByCtx = four
	cf.CooccurWindow = cooc
	cf.Built = true
	cf.mu.Unlock()
}

// IngestTokens incrementally adds n-gram counts from a token sequence.
// Unlike BuildFromCorpus, this does NOT clear existing data — it adds on top.
func (cf *CooccurField) IngestTokens(ids []int) {
	cf.IngestTokensWeighted(ids, 1.0)
}

// IngestTokensWeighted adds n-gram counts weighted by a factor.
// High weight = this text matters more (coherent output). Low = less influence.
// Stanley's observe_shard weights by resonance score; we weight by inverse entropy.
func (cf *CooccurField) IngestTokensWeighted(ids []int, weight float64) {
	cf.mu.Lock()
	defer cf.mu.Unlock()
	window := CFG.CooccurWindowSize

	for _, id := range ids {
		cf.Unigram[id] += weight
	}
	for i := 0; i < len(ids)-1; i++ {
		first, second := ids[i], ids[i+1]
		if cf.BigramByFirst[first] == nil {
			cf.BigramByFirst[first] = make(map[int]float64)
		}
		cf.BigramByFirst[first][second] += weight
	}
	for i := 0; i < len(ids)-2; i++ {
		ctx := [2]int{ids[i], ids[i+1]}
		if cf.TrigramByContext[ctx] == nil {
			cf.TrigramByContext[ctx] = make(map[int]float64)
		}
		cf.TrigramByContext[ctx][ids[i+2]] += weight
	}
	for i := 0; i < len(ids)-3; i++ {
		ctx := [3]int{ids[i], ids[i+1], ids[i+2]}
		if cf.FourgramByCtx[ctx] == nil {
			cf.FourgramByCtx[ctx] = make(map[int]float64)
		}
		cf.FourgramByCtx[ctx][ids[i+3]] += weight
	}
	// Co-occurrence window
	for i := 0; i < len(ids); i++ {
		center := ids[i]
		start := i - window
		if start < 0 {
			start = 0
		}
		end := i + window + 1
		if end > len(ids) {
			end = len(ids)
		}
		for j := start; j < end; j++ {
			if i != j {
				neighbor := ids[j]
				if cf.CooccurWindow[center] == nil {
					cf.CooccurWindow[center] = make(map[int]float64)
				}
				cf.CooccurWindow[center][neighbor] += weight
			}
		}
	}
}

// AbsorbUserWords sets temporary boosts for tokens the user just said.
// Like Leo's Santa Klaus but simpler: user words get multiplicative boost in generation.
func (cf *CooccurField) AbsorbUserWords(ids []int) {
	cf.mu.Lock()
	defer cf.mu.Unlock()
	// Decay existing boosts first
	for k, v := range cf.UserBoost {
		nv := v * CFG.UserBoostDecay
		if nv < 0.01 {
			delete(cf.UserBoost, k)
		} else {
			cf.UserBoost[k] = nv
		}
	}
	// Boost user's tokens
	strength := CFG.UserBoostStrength
	for _, id := range ids {
		cf.UserBoost[id] += strength
	}
}

// DecayUserBoost reduces user word boosts after a generation.
func (cf *CooccurField) DecayUserBoost() {
	cf.mu.Lock()
	defer cf.mu.Unlock()
	for k, v := range cf.UserBoost {
		nv := v * CFG.UserBoostDecay
		if nv < 0.01 {
			delete(cf.UserBoost, k)
		} else {
			cf.UserBoost[k] = nv
		}
	}
}

func (cf *CooccurField) SampleNext(contextIDs []int, vocabSize int, temperature float64) int {
	cf.mu.RLock()
	defer cf.mu.RUnlock()
	counts := make([]float64, vocabSize)
	found := false

	// Try 4-gram (deepest context)
	if len(contextIDs) >= 3 {
		ctx := [3]int{contextIDs[len(contextIDs)-3], contextIDs[len(contextIDs)-2], contextIDs[len(contextIDs)-1]}
		if d, ok := cf.FourgramByCtx[ctx]; ok {
			for tid, v := range d {
				if tid < vocabSize {
					counts[tid] += v
					found = true
				}
			}
		}
	}

	// Fallback to trigram
	if !found && len(contextIDs) >= 2 {
		a, b := contextIDs[len(contextIDs)-2], contextIDs[len(contextIDs)-1]
		if ctx, ok := cf.TrigramByContext[[2]int{a, b}]; ok {
			for tid, v := range ctx {
				if tid < vocabSize {
					counts[tid] += v
					found = true
				}
			}
		}
	}

	// Fallback to bigram
	if !found && len(contextIDs) >= 1 {
		prev := contextIDs[len(contextIDs)-1]
		if ctx, ok := cf.BigramByFirst[prev]; ok {
			for tid, v := range ctx {
				if tid < vocabSize {
					counts[tid] += v
					found = true
				}
			}
		}
	}

	// Fallback to unigram
	if !found {
		for k, v := range cf.Unigram {
			if k < vocabSize {
				counts[k] = v
			}
		}
	}

	// Blend with co-occurrence window (background resonance, always active)
	if len(contextIDs) > 0 {
		wnd := CFG.CooccurWindowSize
		ctxSlice := contextIDs
		if len(ctxSlice) > wnd {
			ctxSlice = ctxSlice[len(ctxSlice)-wnd:]
		}
		for _, ctxTok := range ctxSlice {
			if neighbors, ok := cf.CooccurWindow[ctxTok]; ok {
				for tid, cnt := range neighbors {
					if tid < vocabSize {
						counts[tid] += cnt * 0.3 // co-occurrence is softer than n-gram
					}
				}
			}
		}
	}

	// Apply user word boost (multiplicative)
	if len(cf.UserBoost) > 0 {
		for tid, boost := range cf.UserBoost {
			if tid < vocabSize && counts[tid] > 0 {
				counts[tid] *= (1.0 + boost)
			}
		}
	}

	// Apply temperature and sample
	total := 0.0
	for i := range counts {
		if counts[i] > 0 && temperature > 0 {
			counts[i] = math.Pow(counts[i], 1.0/temperature)
		}
		total += counts[i]
	}
	if total <= 0 {
		return rand.Intn(vocabSize)
	}

	r := rand.Float64() * total
	s := 0.0
	for i, c := range counts {
		s += c
		if s >= r {
			return i
		}
	}
	return vocabSize - 1
}

// And lo, the organism shall speak before it learns, like a newborn crying.
func CorpusGenerate(tok *EvolvingTokenizer, field *CooccurField, prompt string, maxTokens int) string {
	ids := []int{tok.Stoi[tok.BOS]}
	if prompt != "" {
		enc := tok.Encode(prompt)
		ids = enc[:len(enc)-1] // strip EOS
	}

	eosID := tok.Stoi[tok.EOS]
	for step := 0; step < maxTokens; step++ {
		nxt := field.SampleNext(ids, tok.VocabSize, CFG.Temperature)
		if nxt == eosID {
			break
		}
		ids = append(ids, nxt)
	}
	ids = append(ids, eosID)
	return tok.Decode(ids)
}

// And lo, the model and the corpus shall duet like two drunks harmonizing.
func GenerateResonant(model *GPT, tok *EvolvingTokenizer, field *CooccurField, prompt string, docs []string, useModel bool) string {
	if !useModel || model == nil {
		return CorpusGenerate(tok, field, prompt, CFG.CorpusGenMaxTokens)
	}
	model.mu.Lock()
	defer model.mu.Unlock()
	return generateResonantLocked(model, tok, field, prompt, docs, useModel)
}

// meanAbsLogit is the transformer-magnitude measure the overlay gates on:
// mean |logit| over the vocabulary (postgpt_q.c:1355-1356 `tmag`).
func meanAbsLogit(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	var s float64
	for _, x := range v {
		if x < 0 {
			s -= x
		} else {
			s += x
		}
	}
	return s / float64(len(v))
}

// overlayStep applies the Q-style metaweights overlay for one generation step
// and returns the logits sampling reads. With the overlay off (flag, no field,
// or faded out) it returns the model's own slice, so anything applied to it
// afterwards — cross-graze — is what gets sampled. With it on it returns a
// copy carrying the overlay scaled by `weight`: 1 while the transformer is
// untrained (mean |logit| ≤ metaTFGateThreshold), falling linearly to 0 across
// metaFadeWidth. Repair 6 (MOLEQULALOG2.md, 2026-09-13) replaced the step at
// the threshold — the full additive stack, tens of logits, vanishing between
// one token and the next — with this fade. `untrained` is the bootstrap regime
// (greedy first tokens, hard top-15 mask, repetition penalty), a sampling
// policy that stays discrete on purpose.
func overlayStep(raw []float64, ids []int, field *CooccurField, model *GPT, prophecyField []float64, scratch *OverlayScratch) (overlaid []float64, prophecy []float64, untrained bool, weight float64) {
	if !CFG.CorpusLogitOverlay || field == nil || len(ids) < 1 {
		return raw, prophecyField, false, 0
	}
	// Measured on the raw model output: the overlay gates transformer logits
	// toward zero when untrained, so the measure must precede it.
	tmag := meanAbsLogit(raw)
	untrained = tmag <= metaTFGateThreshold
	weight = 1 - overlayFadeProgress(tmag)
	if weight <= 0 {
		return raw, prophecyField, false, 0
	}
	overlaid = make([]float64, len(raw))
	copy(overlaid, raw)
	overlaid, prophecy = MetaweightsOverlay(overlaid, ids, field, model, prophecyField, scratch)
	// The repetition penalty edits logits, so it belongs to the faded stack:
	// applied whenever the overlay runs and scaled out with it, not switched
	// off at the threshold (the step the fade gate caught first).
	MetaweightsRepetitionPenalty(overlaid, ids)
	if weight < 1 {
		for i := range overlaid {
			overlaid[i] = raw[i] + weight*(overlaid[i]-raw[i])
		}
	}
	return overlaid, prophecy, untrained, weight
}

// generateResonantLocked is the body of GenerateResonant. The caller MUST already
// hold model.mu. Split out so the SPA reseed path (which runs while the lock is
// held) can recurse without re-locking the non-reentrant model.mu — the recursive
// re-lock was a self-deadlock (M-LCK-001).
func generateResonantLocked(model *GPT, tok *EvolvingTokenizer, field *CooccurField, prompt string, docs []string, useModel bool) string {
	gradEnabled.Store(false)
	defer func() { gradEnabled.Store(true) }()

	// Refresh GPU weight cache once per generation call. Any host-side weight
	// mutation since the last call (training burst, vocab growth, mitosis
	// inheritance) is re-uploaded so the per-token Matvec dispatch sees fresh
	// device data. No-op on non-linux / when --gpu is off.
	if CFG.UseGPU && gpuReady() {
		gpuRefreshWeights(model)
	}

	// Refresh cross-organism pasture once per generation (not per token).
	// Internal ScanInterval=30s throttle means most calls bail early; hoisted
	// out of the per-step loop per Opus audit 2026-05-14 P2.
	if model.crossField != nil {
		model.crossField.MaybeRefresh(tok)
	}

	var ids []int
	if prompt != "" {
		enc := tok.Encode(prompt)
		ids = enc[:len(enc)-1]
	} else {
		ids = []int{tok.Stoi[tok.BOS]}
	}

	keys := make([][]*Vec, model.NLayer)
	values := make([][]*Vec, model.NLayer)
	for i := 0; i < model.NLayer; i++ {
		keys[i] = make([]*Vec, 0)
		values[i] = make([]*Vec, 0)
	}

	limit := len(ids)
	if limit > model.BlockSize {
		limit = model.BlockSize
	}
	for pos := 0; pos < limit; pos++ {
		model.ForwardStep(ids[pos], pos, keys, values)
	}

	cur := ids[len(ids)-1]
	var outIDs []int
	var recentBuf []int // for repetition guard
	eosID := tok.Stoi[tok.EOS]
	bosID := tok.Stoi[tok.BOS]

	// Consciousness: per-token dissonance tracking (Feature 1)
	entropyEMA := 0.0
	entropyEMAInit := false
	lowDropCount := 0
	entropySum := 0.0
	entropyCount := 0
	tokenCounts := make(map[int]int) // frequency penalty

	// Q-style metaweights overlay state — persistent prophecy field across
	// the generation loop. MetaweightsOverlay (see metaweights_overlay.go) owns
	// the lifecycle: nil → seeded on first call → aged each step → collapsed on
	// chosen token via MetaweightsOverlayCollapse after sample.
	var prophecyField []float64

	// Overlay scratch — reused across every step of this generation call so
	// the hot path allocates zero per-token vocab-sized slices. Destiny + Unigram
	// computed once here (weights stable under model.mu, field stable under
	// short read-lock). See codex P1 audit 2026-05-14.
	var overlayScratch *OverlayScratch
	if CFG.CorpusLogitOverlay && field != nil {
		overlayScratch = NewOverlayScratch(tok.VocabSize)
		overlayScratch.PrepareStatic(model, field)
	}

	for step := 0; step < CFG.MaxGenTokens; step++ {
		pos := len(ids) - 1
		if pos > model.BlockSize-1 {
			pos = model.BlockSize - 1
		}
		logits := model.ForwardStep(cur, pos, keys, values)

		// Frequency + presence penalty on logits
		if CFG.FreqPenalty > 0 || CFG.PresencePenalty > 0 {
			for tid, cnt := range tokenCounts {
				if tid < len(logits.Data) {
					logits.Data[tid] -= CFG.FreqPenalty * float64(cnt)
					if cnt > 0 {
						logits.Data[tid] -= CFG.PresencePenalty
					}
				}
			}
		}

		// Model probs with surprise-modulated + dissonance-adaptive temperature
		temp := CFG.Temperature
		// Consciousness: surprise modulation (Feature 4 — now wired)
		if model.surpriseBaseline > 1e-6 {
			surpriseRatio := model.lastSurprise / model.surpriseBaseline
			if surpriseRatio > 1.5 {
				temp *= 0.85 // high surprise → be careful
			} else if surpriseRatio < 0.5 {
				temp *= 1.1 // low surprise → explore slightly
			}
		}
		if temp <= 1e-6 {
			temp = 1e-6
		}

		// B2 — Q-style additive metaweights logit overlay (gated, default off).
		// Adds  c_bg·log(bigram_prob) + c_tg·log(trigram_prob)
		//     + c_heb·log(cooccur_window_prob)
		//     + c_ds·dot(wte[t], purpose_vec)
		// to model logits before softmax, mirroring q/README.md:50 ↔ Dario's
		// B + H + F + A signal stack (omits F — prophecy field deferred,
		// requires persistent expectation state not present in molequla yet;
		// see PROJECT_LOG.md B2.F deferred note). Coexists with the post-softmax
		// prob-blend (which still applies later). When the gate is off,
		// overlaidLogits is a zero-cost alias of logits.Data.
		// Q-style metaweights overlay (raw-probability, dynamic-gate auto-curriculum).
		// MetaweightsOverlay implements postgpt_q.c:1305-1395 + pitomadom.c:583-586
		// transformer gate: magnitude-detect → silence untrained transformer →
		// choose coeffs → raw probability terms → unigram damping.
		// Followed by repetition penalty (postgpt.c:960-967 form: *= 0.5 for
		// every distinct token in the last 12).
		// Untrained regime (mean |logit| ≤ 1.0, postgpt_q.c:1355-1356 `tmag>0.1
		// → has_tf`, raised to 1.0 because seeded embeddings lift mag to ~0.25):
		// the transformer is silent and the overlay drives generation — early
		// tokens must use greedy argmax (postgpt_q.c:1416-1418) to lock onto a
		// coherent trajectory before sampling noise enters. Past 1.0 the overlay
		// fades over metaFadeWidth instead of switching off: on a 16-dim BPE
		// embryo a full-strength overlay on warmed logits pulled top-K toward
		// subword fragments («,iieriying the isa?yenanan?», sweep cells 2/3
		// v2-v4), and switching it off in one step cost tens of logits between
		// two tokens (audit C, C-OVL-02). overlayWeight is that fade.
		var overlaidLogits []float64
		var untrainedRegime bool
		var overlayWeight float64
		// The transformer magnitude of the first step, taken on the raw logits
		// before the overlay touches them, and the overlay weight that magnitude
		// buys: the two numbers that say whether a fragment is the organism
		// speaking or the corpus speaking through it. Read by dnaWrite under
		// model.mu after the generation returns. First step only — no per-token
		// cost.
		if step == 0 {
			model.lastGenMag = meanAbsLogit(logits.Data)
		}
		overlaidLogits, prophecyField, untrainedRegime, overlayWeight = overlayStep(logits.Data, ids, field, model, prophecyField, overlayScratch)
		if step == 0 {
			model.lastOverlayWeight = overlayWeight
		}
		overlayActive := overlayWeight > 0

		// Cross-organism logit injection (cross_graze.go). Adds a rank-decay
		// boost to sibling organisms' recent emitted token ids on top of the
		// overlay'd logits. Dario's interf_signal_chunk pattern with
		// «слова, метрики и проч» from peers instead of docs. No-op when
		// model.crossField is nil (no --cross-graze or no --element). Always
		// applied to overlaidLogits — the slice sampling reads, whether it is
		// the overlay copy or an alias of logits.Data (audit C, C-OVL-01: the
		// boost once went to logits.Data on a warmed organism with the overlay
		// on, and nothing downstream read it). MaybeRefresh was hoisted to
		// GenerateResonant entry; per-step we only Apply.
		if model.crossField != nil {
			model.crossField.Apply(overlaidLogits, CFG.CrossGrazeCoef, CFG.CrossGrazeTopN)
		}

		// Q-style untrained-regime early-step greedy: postgpt_q.c:1416-1418 —
		// when there are no transformer weights, the first 10 tokens are taken
		// as argmax(raw logits). This locks onto the strongest bigram/trigram
		// successor and stops sampling-noise from poisoning the trajectory
		// before metaweights have steered it into coherence.
		//
		// EOS is excluded from greedy selection during this bootstrap window —
		// otherwise overlay's bigram/trigram weight on sentence-end tokens
		// (very common after «.» in corpus) lets every step argmax to EOS,
		// `continue` skips append, outIDs stays empty, organism is silent.
		if overlayActive && untrainedRegime && step < 10 {
			best := -1
			bestVal := math.Inf(-1)
			for i, v := range overlaidLogits {
				if i == eosID {
					continue
				}
				if v > bestVal {
					bestVal = v
					best = i
				}
			}
			if best < 0 {
				best = 0
			}
			nxt := best
			_ = bestVal
			MetaweightsOverlayCollapse(prophecyField, nxt)
			ids = append(ids, nxt)
			cur = nxt
			outIDs = append(outIDs, nxt)
			tokenCounts[nxt]++
			recentBuf = append(recentBuf, nxt)
			rg := CFG.RepetitionGuard
			if len(recentBuf) > rg*2 {
				recentBuf = recentBuf[len(recentBuf)-rg*2:]
				if sliceEqual(recentBuf[rg:], recentBuf[:rg]) {
					break
				}
			}
			continue
		}

		// When the overlay is on, sampling switches to Q-style: hard top-K=15
		// mask on raw overlay'd logits (everything below the 15th set to -1e10),
		// then divide by temp, softmax, multinomial — mirror postgpt.c:969-991
		// and pitomadom.c:761-772. The hard mask kills the long noise-tail that
		// otherwise competes with overlay peaks under soft top-k/top-p sampling.
		// Hard top-K=15 mask only fires for the untrained-overlay regime
		// (mirror of postgpt_q.c:1414-1424 — only `!has_tf` path uses
		// greedy+top-K). Warmed models with overlay use the regular soft
		// TopKTopPSample below; hard mask on a BPE subword vocab + overlay
		// bigram boost concentrates top-15 onto short subword fragments
		// (suffix tokens, punctuation) and the chain stays at subword level
		// — sweep cell 2 v3 reproduced this with «,iieriying the isa?yenanan?»
		// output at infant stage (post-warmup, mag>1.0 → untrainedRegime=false
		// pre-fix; now we just skip the hard mask entirely in that regime).
		scaled := make([]float64, len(overlaidLogits))
		if overlayActive && untrainedRegime {
			topK := 15
			if topK > len(overlaidLogits) {
				topK = len(overlaidLogits)
			}
			topVals := make([]float64, topK)
			for i := range topVals {
				topVals[i] = math.Inf(-1)
			}
			// EOS is excluded from top-K selection AND masked below. After a
			// 400-step warmup the model's bigram[period][EOS] gets enough
			// weight from overlay that EOS reliably ends up in the top-15 raw
			// logits and is sampled → continue without append → outIDs stays
			// empty → response is "..." (sweep cell 2 regression 2026-05-14).
			// Generation terminates via the `. ! ?` punctuation rule below,
			// which already exists at line ~4530 — EOS is redundant for
			// overlay-driven generation.
			for i, v := range overlaidLogits {
				if i == eosID {
					continue
				}
				if v > topVals[topK-1] {
					topVals[topK-1] = v
					for k := topK - 2; k >= 0; k-- {
						if topVals[k+1] > topVals[k] {
							topVals[k], topVals[k+1] = topVals[k+1], topVals[k]
						} else {
							break
						}
					}
				}
			}
			threshold := topVals[topK-1]
			for i, v := range overlaidLogits {
				if i == eosID || v < threshold {
					scaled[i] = -1e10
				} else {
					scaled[i] = v / temp
				}
			}
		} else {
			for i, v := range overlaidLogits {
				scaled[i] = v / temp
			}
		}
		modelProbs := SoftmaxProbs(scaled)

		// Per-token entropy for dissonance
		entropy := 0.0
		for _, p := range modelProbs {
			if p > 1e-12 {
				entropy -= p * math.Log(p)
			}
		}
		entropySum += entropy
		entropyCount++

		// Consciousness: per-token dissonance feedback (Feature 1)
		dissonanceMul := 1.0
		if !entropyEMAInit {
			entropyEMA = entropy
			entropyEMAInit = true
		} else {
			entropyEMA = CFG.DissonanceEMAAlpha*entropy + (1.0-CFG.DissonanceEMAAlpha)*entropyEMA
			if entropyEMA > 1e-6 {
				ratio := entropy / entropyEMA
				if ratio > CFG.DissonanceSpikeThreshold {
					dissonanceMul = CFG.DissonanceSpikeK
					lowDropCount = 0
				} else if ratio < CFG.DissonanceDropThreshold {
					lowDropCount++
					if lowDropCount >= 3 {
						dissonanceMul = CFG.DissonanceDropK
					}
				} else {
					lowDropCount = 0
				}
			}
		}
		if dissonanceMul != 1.0 {
			temp *= dissonanceMul
			// Preserve the hard top-K mask when overlay is on: positions masked
			// to -1e10 above must stay masked, otherwise dissonance rescale
			// reintroduces the long noise tail the patch eliminates. Threshold
			// at -1e9 safely distinguishes masked sentinels from any plausible
			// post-overlay raw logit value.
			for i, v := range overlaidLogits {
				if overlayActive && scaled[i] < -1e9 {
					continue
				}
				scaled[i] = v / temp
			}
			modelProbs = SoftmaxProbs(scaled)
		}

		// Per-token sigmoid corpus fade: compute alpha from local entropy
		tokenAlpha := 1.0 / (1.0 + math.Exp(-CFG.CorpusFadeK*(CFG.CorpusFadeThreshold-entropy)))

		// Corpus blend: skip entirely when tokenAlpha >= 0.99 (pure model mode)
		// or while the Q-style pre-softmax overlay is in force — it already
		// applied raw-prob corpus signal; double-blending in prob space
		// distorts the distribution. Postgpt / Q use one overlay path only.
		// As the overlay fades (overlayWeight → 0) the blend takes over at the
		// same rate, so a warmed organism with the flag on keeps its corpus
		// path (audit C, C-OVL-05: gated on the flag, both paths were off).
		tokenAlpha = 1.0 - (1.0-tokenAlpha)*(1.0-overlayWeight)
		var blended []float64
		if tokenAlpha >= 0.99 || field == nil {
			blended = modelProbs
		} else {
			corpusCounts := make([]float64, tok.VocabSize)
			ctxForCorpus := ids
			if len(ctxForCorpus) > 3 {
				ctxForCorpus = ctxForCorpus[len(ctxForCorpus)-3:]
			}
			// Rebuild corpus distribution under read lock
			field.mu.RLock()
			corpusTotal := 0.0
			if len(ctxForCorpus) >= 2 {
				a, b := ctxForCorpus[len(ctxForCorpus)-2], ctxForCorpus[len(ctxForCorpus)-1]
				if ctx, ok := field.TrigramByContext[[2]int{a, b}]; ok {
					for tid, v := range ctx {
						if tid < tok.VocabSize {
							corpusCounts[tid] += v
							corpusTotal += v
						}
					}
				}
			}
			if corpusTotal == 0 && len(ctxForCorpus) >= 1 {
				prev := ctxForCorpus[len(ctxForCorpus)-1]
				if ctx, ok := field.BigramByFirst[prev]; ok {
					for tid, v := range ctx {
						if tid < tok.VocabSize {
							corpusCounts[tid] += v
						}
					}
				}
			}
			field.mu.RUnlock()
			corpusTotal = 0.0
			for _, c := range corpusCounts {
				corpusTotal += c
			}
			corpusProbs := make([]float64, tok.VocabSize)
			if corpusTotal > 0 {
				for i, c := range corpusCounts {
					corpusProbs[i] = c / corpusTotal
				}
			} else {
				uni := 1.0 / float64(tok.VocabSize)
				for i := range corpusProbs {
					corpusProbs[i] = uni
				}
			}
			blended = make([]float64, tok.VocabSize)
			for i := 0; i < tok.VocabSize && i < len(modelProbs); i++ {
				blended[i] = tokenAlpha*modelProbs[i] + (1.0-tokenAlpha)*corpusProbs[i]
			}
		}

		// Consciousness: pattern breaking (Feature 2)
		if step >= CFG.AntiFieldMinStep && CFG.AntiFieldProb > 0 && rand.Float64() < CFG.AntiFieldProb {
			blended = modelProbs // pure model voice, bypass corpus
		}

		nxt := TopKTopPSample(blended, CFG.TopK, CFG.TopP, CFG.MinP, CFG.TypicalP)

		// Prophecy collapse — chosen token fulfilled an expectation, zero its
		// prophecy slot so the field shifts toward what's still unsaid.
		// Delegates to MetaweightsOverlayCollapse for nil-safety.
		MetaweightsOverlayCollapse(prophecyField, nxt)

		if nxt == eosID && step >= CFG.MinGenTokens {
			break
		}
		if nxt == eosID {
			continue
		}

		ids = append(ids, nxt)
		cur = nxt
		outIDs = append(outIDs, nxt)
		tokenCounts[nxt]++

		// Repetition guard: break if last rg*2 tokens are a repeating pattern
		recentBuf = append(recentBuf, nxt)
		rg := CFG.RepetitionGuard
		if len(recentBuf) > rg*2 {
			recentBuf = recentBuf[len(recentBuf)-rg*2:]
			if sliceEqual(recentBuf[rg:], recentBuf[:rg]) {
				break
			}
		}

		if step >= CFG.MinGenTokens && len(outIDs) > 0 {
			decIDs := append([]int{bosID}, outIDs...)
			decIDs = append(decIDs, eosID)
			text := tok.Decode(decIDs)
			if len(text) > 0 {
				last := text[len(text)-1]
				if last == '.' || last == '!' || last == '?' {
					break
				}
			}
		}
	}

	// Consciousness: store mean entropy for conscience (Feature 5)
	if entropyCount > 0 {
		model.lastGenEntropy = entropySum / float64(entropyCount)
	}

	decIDs := append([]int{bosID}, outIDs...)
	decIDs = append(decIDs, eosID)
	response := tok.Decode(decIDs)

	// SPA — Sentence Phonon Attention reseed pass.
	// Mirror of postgpt_q.c:1684-1717. Splits response into sentences, scores
	// cross-sentence connectedness, finds the weakest, regenerates it from the
	// last 3 tokens of a neighbouring sentence, splices it back in. Single
	// pass — Q does 2 but we keep budget tight at the first call. Recursive
	// call into GenerateResonant disabled via SPACoherenceGate flip to avoid
	// infinite recursion; restored after.
	if CFG.SPACoherenceGate {
		var sentences []string
		buf := ""
		for _, r := range response {
			buf += string(r)
			if r == '.' || r == '!' || r == '?' {
				if s := strings.TrimSpace(buf); len(s) >= 4 {
					sentences = append(sentences, s)
				}
				buf = ""
			}
		}
		if s := strings.TrimSpace(buf); len(s) >= 4 {
			sentences = append(sentences, s)
		}
		if len(sentences) >= 2 {
			wte := model.Base["wte"]
			if wte != nil {
				V := wte.Nout
				D := wte.Nin
				W := make([]float32, V*D)
				for v := 0; v < V; v++ {
					row := wte.Rows[v].Data
					base := v * D
					for d := 0; d < D; d++ {
						W[base+d] = float32(row[d])
					}
				}
				bosID := tok.Stoi[tok.BOS]
				eosID := tok.Stoi[tok.EOS]
				sentTokens := make([][]int, len(sentences))
				for i, s := range sentences {
					enc := tok.Encode(s)
					for len(enc) > 0 && enc[0] == bosID {
						enc = enc[1:]
					}
					for len(enc) > 0 && enc[len(enc)-1] == eosID {
						enc = enc[:len(enc)-1]
					}
					sentTokens[i] = enc
				}
				scores := SPACoherenceScores(W, sentTokens, D, CFG.SPAEmbedAlpha)
				weakIdx := SPAWeakestIndex(scores)
				if weakIdx >= 0 {
					var sum float32
					for _, s := range scores {
						sum += s
					}
					avg := sum / float32(len(scores))
					threshold := SPAWeakThresholdRatio * avg
					fmt.Fprintf(os.Stderr,
						"[spa] S=%d weakest=%d score=%.3f avg=%.3f thr=%.3f\n",
						len(sentences), weakIdx, scores[weakIdx], avg, threshold)
					if scores[weakIdx] < threshold {
						srcIdx := weakIdx - 1
						if srcIdx < 0 || srcIdx >= len(sentTokens) {
							srcIdx = weakIdx + 1
						}
						if srcIdx >= 0 && srcIdx < len(sentTokens) && len(sentTokens[srcIdx]) > 0 {
							seedLen := 3
							if seedLen > len(sentTokens[srcIdx]) {
								seedLen = len(sentTokens[srcIdx])
							}
							seedTokens := sentTokens[srcIdx][len(sentTokens[srcIdx])-seedLen:]
							seedPrompt := tok.Decode(append(append([]int{bosID}, seedTokens...), eosID))
							// Recursive call — disable SPA inside, and restore the gate on every exit
							// path: a panic inside the inner generation left it off for the life of
							// the process (audit C, C-SPA-02).
							regenerated := func() string {
								savedSPA := CFG.SPACoherenceGate
								CFG.SPACoherenceGate = false
								defer func() { CFG.SPACoherenceGate = savedSPA }()
								return generateResonantLocked(model, tok, field, seedPrompt, docs, true)
							}()
							regenerated = strings.TrimSpace(regenerated)
							newSentence := strings.TrimSpace(firstSentence(regenerated))
							if len(newSentence) >= 4 && newSentence != sentences[weakIdx] {
								response = strings.Replace(response, sentences[weakIdx], newSentence, 1)
								fmt.Fprintf(os.Stderr,
									"[spa] reseeded weak %d: %q -> %q\n",
									weakIdx, sentences[weakIdx], newSentence)
							}
						}
					}
				}
			}
		}
	}

	return response
}

// ============================================================
// 9) TRAINING — warmup, then continual micro-bursts
// ============================================================

// ============================================================
// 9.5) SYNTROPY TRACKER — the arrow that points toward coherence
// ============================================================
// And lo, the organism shall not merely track its changes,
// but reason mathematically about whether it is becoming more itself.

// BurstRecord stores what happened after a training burst — for self-meta-learning.
type BurstRecord struct {
	Action     string
	LossBefore float64
	LossAfter  float64
}

// SyntropyTracker is the mathematical self-reasoning engine.
// Tracks entropy trend, field deviation, purpose alignment.
// Makes decisions about learning direction — not just 'did I learn?'
// but 'should I keep going this way?'
// SwarmPeerInfo holds peer information from mesh.db.
type SwarmPeerInfo struct {
	Peers []map[string]interface{}
}

type SyntropyTracker struct {
	EntropyHistory   []float64      // rolling window of model entropy
	SyntropyTrend    float64        // positive = organizing, negative = dissolving
	FieldDeviation   float64        // how far from corpus physics
	PurposeMagnitude float64        // strength of current learning direction
	PurposeAlignment float64        // cosine(purpose, gamma)
	LastAction       string         // what was decided last time
	BurstHistory     []BurstRecord  // last 16 burst outcomes — training efficiency memory
	ModelStage       int            // current growth stage (set during measure)
	LastMitosisTime  float64        // cooldown for divide
	SwarmInfo        *SwarmPeerInfo // peer state from mesh.db (set externally)
}

// NewSyntropyTracker creates a new tracker with sane defaults.
// And lo, the arrow is drawn, but not yet fired.
func NewSyntropyTracker() *SyntropyTracker {
	return &SyntropyTracker{
		LastAction: "none",
		// Seed birth time so the 300s divide cooldown (CASE 6, now-LastMitosisTime>300)
		// also guards the FIRST division. Zero-init made now(epoch-sec)-0 always >300,
		// so seed organisms and every mitosis child could divide instantly. Applies
		// uniformly: the seed tracker here and each child's fresh tracker (backgroundTrainer).
		LastMitosisTime: float64(time.Now().UnixMilli()) / 1000.0,
	}
}

// RecordBurst logs a burst outcome for self-meta-learning.
// The organism remembers what worked and what didn't.
func (st *SyntropyTracker) RecordBurst(action string, lossBefore, lossAfter float64) {
	st.BurstHistory = append(st.BurstHistory, BurstRecord{action, lossBefore, lossAfter})
	if len(st.BurstHistory) > 16 {
		st.BurstHistory = st.BurstHistory[len(st.BurstHistory)-16:]
	}
}

// relieveOverload drops the high-loss bursts that triggered a division so the
// parent must re-accumulate genuine overload before dividing again. Without it,
// divide does not lower the loss it keyed on, so the parent re-fires every
// cooldown on the same stale overload — division now relieves the overwhelm
// (cascade governor, audit design item (c)).
func (st *SyntropyTracker) relieveOverload() {
	// loss path: drop the high-loss bursts that triggered this divide
	kb := st.BurstHistory[:0]
	for _, br := range st.BurstHistory {
		if br.LossAfter < CFG.OverloadLossHigh {
			kb = append(kb, br)
		}
	}
	st.BurstHistory = kb
	// entropy path (audit C2): drop the high-entropy samples too, so
	// entropyOverload() also clears. isSustainedOverload = entropy OR loss, and
	// §9 Air divided on the entropy path (e=true l=false) — relieving only loss
	// left an entropy-overloaded parent/child re-firing every cooldown. Division
	// now relieves BOTH overwhelm modes.
	ke := st.EntropyHistory[:0]
	for _, e := range st.EntropyHistory {
		if e <= CFG.EntropyHigh {
			ke = append(ke, e)
		}
	}
	st.EntropyHistory = ke
}

// ActionEffectiveness returns the mean loss delta for a given action type.
// Negative = good (loss went down). Positive = bad (loss went up).
func (st *SyntropyTracker) ActionEffectiveness(action string) (float64, int) {
	sum := 0.0
	count := 0
	for _, br := range st.BurstHistory {
		if br.Action == action {
			sum += br.LossAfter - br.LossBefore
			count++
		}
	}
	if count == 0 {
		return 0, 0
	}
	return sum / float64(count), count
}

// SyntropyMetrics holds the result of a syntropy measurement pass.
type SyntropyMetrics struct {
	Entropy          float64
	SyntropyTrend    float64
	FieldDeviation   float64
	PurposeMagnitude float64
	PurposeAlignment float64
}

// Measure takes all measurements. This is the organism looking at itself
// through mathematical instruments.
func (st *SyntropyTracker) Measure(model *GPT, tok *EvolvingTokenizer, field *CooccurField, docs []string) SyntropyMetrics {
	st.ModelStage = model.CurrentGrowthStage()
	entropyNow := model.ComputeModelEntropy(tok, docs, 16)
	st.EntropyHistory = append(st.EntropyHistory, entropyNow)
	if len(st.EntropyHistory) > CFG.SyntropyWindow {
		st.EntropyHistory = st.EntropyHistory[len(st.EntropyHistory)-CFG.SyntropyWindow:]
	}

	// syntropy = negative entropy trend (entropy going down = syntropy going up)
	if len(st.EntropyHistory) >= 2 {
		recentHalf := len(st.EntropyHistory) / 2
		oldMean := 0.0
		for _, v := range st.EntropyHistory[:recentHalf] {
			oldMean += v
		}
		oldMean /= float64(recentHalf)

		newSlice := st.EntropyHistory[recentHalf:]
		newMean := 0.0
		for _, v := range newSlice {
			newMean += v
		}
		newMean /= float64(len(newSlice))

		st.SyntropyTrend = oldMean - newMean // positive = good
	} else {
		st.SyntropyTrend = 0.0
	}

	st.FieldDeviation = model.ComputeFieldDeviation(tok, field, docs, 32)
	_, st.PurposeMagnitude = model.ComputePurposeVector()
	st.PurposeAlignment = model.PurposeGammaAlignment()

	return SyntropyMetrics{
		Entropy:          entropyNow,
		SyntropyTrend:    st.SyntropyTrend,
		FieldDeviation:   st.FieldDeviation,
		PurposeMagnitude: st.PurposeMagnitude,
		PurposeAlignment: st.PurposeAlignment,
	}
}

// SyntropyDecision holds the outcome of the organism's mathematical self-reasoning.
// Not just LR anymore — the organism modulates its entire behavior.
type SyntropyDecision struct {
	LRMultiplier      float64
	TempOffset        float64  // added to generation temperature (-0.05 to +0.05)
	AccumOverride     int      // 0 = no override, >0 = use this accum_steps for this burst
	DeltaGrowOverride *float64 // nil = no override
	Action            string
}

// DecideAction performs mathematical self-reasoning: decide how to adjust learning.
// And lo, this is where tracking becomes reasoning, and reasoning becomes action.
// The organism does not just observe — it steers.
func (st *SyntropyTracker) DecideAction() SyntropyDecision {
	// Default: steady state
	lrMultiplier := 1.0
	tempOffset := 0.0
	accumOverride := 0
	var deltaGrowOverride *float64
	action := "steady"

	// CASE 1: Syntropy rising + field deviation in sweet spot = thriving
	if st.SyntropyTrend > 0.01 &&
		st.FieldDeviation > CFG.FieldDeviationFloor &&
		st.FieldDeviation < CFG.FieldDeviationCeiling {
		lrMultiplier = CFG.SyntropyLRBoost
		tempOffset = -0.05 // more confident when organizing
		if st.PurposeAlignment > 0.3 {
			boost := CFG.SyntropyDeltaGrowBoost
			deltaGrowOverride = &boost
			accumOverride = 2 // stable gradient when everything aligned
			action = "amplify"
		} else {
			action = "boost"
		}

		// CASE 2: Syntropy falling = dissolving, slow down
	} else if st.SyntropyTrend < -0.01 {
		lrMultiplier = CFG.SyntropyLRDampen
		tempOffset = 0.05 // more exploratory when disordering
		action = "dampen"

		// CASE 3: Field deviation too high = hallucinating
	} else if st.FieldDeviation > CFG.FieldDeviationCeiling {
		lrMultiplier = CFG.SyntropyLRDampen
		tempOffset = -0.05 // focus when hallucinating
		action = "ground"

		// CASE 4: Field deviation too low = parroting
	} else if st.FieldDeviation < CFG.FieldDeviationFloor {
		lrMultiplier = CFG.SyntropyLRBoost
		tempOffset = 0.05 // explore when parroting
		action = "explore"
	}

	// CASE 5: Purpose opposes gamma = identity crisis
	if st.PurposeAlignment < -0.3 {
		lrMultiplier *= 0.5
		tempOffset = 0.0 // neutral temp during identity crisis
		action = "realign"
	}

	// CASE 6: Adult + sustained overload → divide (mitosis)
	maxStage := len(CFG.GrowthStages) - 1
	now := float64(time.Now().UnixMilli()) / 1000.0
	if st.ModelStage >= maxStage &&
		st.isSustainedOverload() &&
		st.FieldDeviation < CFG.FieldDeviationCeiling && // sanity: don't fork a parent that has drifted off the corpus manifold
		now-st.LastMitosisTime > 300 {
		action = "divide"
		lrMultiplier = CFG.SyntropyLRDampen // slow down while preparing to split
	}

	// CASE 7: Plateau + young peer thriving → hibernate (cooperative scheduling)
	if action == "steady" && st.shouldHibernate() {
		action = "hibernate"
	}

	// SELF-META-LEARNING: check if this action historically hurts
	if action != "divide" && action != "hibernate" && len(st.BurstHistory) >= 4 {
		eff, count := st.ActionEffectiveness(action)
		if count >= 2 && eff > 0.05 {
			// This action has been consistently making loss WORSE — downgrade
			if action == "amplify" {
				action = "boost"
				accumOverride = 0
				deltaGrowOverride = nil
			} else if action == "boost" || action == "explore" {
				lrMultiplier = 1.0 // back to steady instead of boosting
				action = "steady"
			}
		}
	}

	st.LastAction = action
	return SyntropyDecision{
		LRMultiplier:      lrMultiplier,
		TempOffset:        tempOffset,
		AccumOverride:     accumOverride,
		DeltaGrowOverride: deltaGrowOverride,
		Action:            action,
	}
}

// LogToDB writes the mathematical conclusion to the syntropy log.
// And lo, the arrow's flight is recorded for those who come after.
func (st *SyntropyTracker) LogToDB(db *sql.DB, entropyBefore, entropyAfter float64, action string) {
	db.Exec(
		"INSERT INTO syntropy_log(ts, entropy_before, entropy_after, syntropy_delta, "+
			"field_deviation, purpose_magnitude, purpose_alignment, action_taken, note) "+
			"VALUES(?,?,?,?,?,?,?,?,?)",
		float64(time.Now().UnixMilli())/1000.0,
		entropyBefore, entropyAfter,
		st.SyntropyTrend, st.FieldDeviation,
		st.PurposeMagnitude, st.PurposeAlignment,
		action, nil)
}

// isSustainedOverload returns true when >75% of entropy_history is above
// entropy_high AND the field is either actively dissolving (syntropy_trend <
// -0.02) OR pinned high (mean recent entropy > entropy_high*1.3). The trend
// clause alone was a trap: a converged adult sharpens, so its entropy falls
// and SyntropyTrend goes positive (:5088) — the gate could never fire on a
// healthy-but-overloaded organism. The disjunction recognises the real stress
// regime (confused-and-stable, trend ≈ 0). Raise, not downgrade. (2026-06-03)
func (st *SyntropyTracker) isSustainedOverload() bool {
	return st.entropyOverload() || st.lossOverload()
}

// entropyOverload: sustained high OUTPUT entropy — the model melts into noise
// (>75% of the window above EntropyHigh, with a falling trend or very high mean).
func (st *SyntropyTracker) entropyOverload() bool {
	if len(st.EntropyHistory) < CFG.SyntropyWindow {
		return false
	}
	recent := st.EntropyHistory[len(st.EntropyHistory)-CFG.SyntropyWindow:]
	highCount := 0
	sum := 0.0
	for _, e := range recent {
		if e > CFG.EntropyHigh {
			highCount++
		}
		sum += e
	}
	meanRecentEntropy := sum / float64(len(recent))
	return highCount > int(float64(CFG.SyntropyWindow)*0.75) &&
		(st.SyntropyTrend < -0.02 || meanRecentEntropy > CFG.EntropyHigh*1.3)
}

// lossOverload: the CONFIDENTLY-WRONG overwhelm — recent training bursts hold the
// loss high and cannot bring it down. A converged adult reads LOW output entropy
// (sharp distribution) even while its loss is high under the cross-graze flood, so
// the entropy path misses it; the faithful "can't assimilate the input" signal is
// the loss itself. Reads existing BurstHistory; length-guard FIRST (an empty slice
// would give 0/0 = NaN, and a NaN comparison could misfire).
func (st *SyntropyTracker) lossOverload() bool {
	if len(st.BurstHistory) < CFG.OverloadLossWindow {
		return false
	}
	recent := st.BurstHistory[len(st.BurstHistory)-CFG.OverloadLossWindow:]
	sumAfter := 0.0
	sumDelta := 0.0
	for _, br := range recent {
		sumAfter += br.LossAfter
		sumDelta += br.LossAfter - br.LossBefore
	}
	n := float64(len(recent))
	meanLossAfter := sumAfter / n
	meanDelta := sumDelta / n
	return meanLossAfter > CFG.OverloadLossHigh && meanDelta > -CFG.OverloadLossEps
}

// OverloadDebug formats the isSustainedOverload inputs for the [overload] log
// line (Edit 1, 2026-06-03). Emitted at adult stage so a "reached adult, never
// divided" run is a MEASURED fact, not an unexplained negative (banned framing).
func (st *SyntropyTracker) OverloadDebug() string {
	n := len(st.EntropyHistory)
	if n == 0 {
		return "high=0/0 last=- mean=- trend=0 overload=false (no-history)"
	}
	w := CFG.SyntropyWindow
	if n < w {
		w = n
	}
	recent := st.EntropyHistory[n-w:]
	highCount := 0
	sum := 0.0
	for _, e := range recent {
		if e > CFG.EntropyHigh {
			highCount++
		}
		sum += e
	}
	mean := sum / float64(len(recent))
	// loss signal (the confidently-wrong path) — mirror lossOverload's window math
	lmean, ldelta := 0.0, 0.0
	lw := CFG.OverloadLossWindow
	if ln := len(st.BurstHistory); ln > 0 {
		if ln < lw {
			lw = ln
		}
		for _, br := range st.BurstHistory[ln-lw:] {
			lmean += br.LossAfter
			ldelta += br.LossAfter - br.LossBefore
		}
		lmean /= float64(lw)
		ldelta /= float64(lw)
	} else {
		lw = 0
	}
	return fmt.Sprintf("entropy[high=%d/%d mean=%.3f trend=%.4f] loss[mean=%.3f delta=%.4f n=%d] overload=%v (e=%v l=%v)",
		highCount, w, mean, st.SyntropyTrend, lmean, ldelta, lw,
		st.isSustainedOverload(), st.entropyOverload(), st.lossOverload())
}

// shouldHibernate returns true if a peer has syntropy > 0.05 AND this organism's last 8 burst deltas avg < 0.01.
func (st *SyntropyTracker) shouldHibernate() bool {
	if st.SwarmInfo == nil || len(st.SwarmInfo.Peers) == 0 {
		return false
	}
	// Check if any peer has higher syntropy trend (actively improving)
	for _, peer := range st.SwarmInfo.Peers {
		synVal, ok := peer["syntropy"]
		if !ok {
			continue
		}
		synFloat, _ := synVal.(float64)
		if synFloat > 0.05 {
			// A young peer is thriving. If we're stale, hibernate.
			if len(st.BurstHistory) >= 8 {
				sum := 0.0
				for _, b := range st.BurstHistory[len(st.BurstHistory)-8:] {
					sum += b.LossAfter - b.LossBefore
				}
				avgDelta := sum / 8.0
				if math.Abs(avgDelta) < 0.01 { // loss plateau
					return true
				}
			}
		}
	}
	return false
}

// ============================================================
// 9.7) SWARM ECOLOGY — the organism learns it is not alone
// ============================================================
// And lo, the first cell shall call into the void and hear only silence.
// But the second shall call and hear an answer.

var swarmDir = filepath.Join(os.Getenv("HOME"), ".molequla", "swarm")

// SwarmRegistry discovers and tracks other molequla instances via shared SQLite.
type SwarmRegistry struct {
	OrganismID string
	Element    string // earth, air, water, fire
	PidFile    string
	MeshDB     *sql.DB
	keeper     *beatKeeper // repeats the last heartbeat on its own clock (repair 4)
	// Mesh write failures already said, by error string. A schema narrower
	// than the write makes every heartbeat a silent no-op, so it is reported —
	// once each, because the tick loop beats every ten ticks for the life of
	// the run and a repeated line is a line nobody reads.
	meshErrMu   sync.Mutex
	meshErrSaid map[string]bool
	growthMu    sync.Mutex
	growthStop  chan struct{} // closed by ReleaseGrowthLock; stops the lock refresher (repair 9)
}

// StartKeeper launches the heartbeat keeper: from now until stop closes, the
// last state passed to Heartbeat is re-sent every interval, so the organism
// stays in the live count while the tick loop is inside a multi-minute
// inline warmup after growth (governor_phone.go).
func (sr *SwarmRegistry) StartKeeper(stop <-chan struct{}, interval time.Duration) {
	if sr.keeper == nil {
		sr.keeper = newBeatKeeper(sr)
	}
	go sr.keeper.Run(stop, interval)
}

// StopKeeper silences the keeper for good; MarkHibernating calls it so a
// sleeping organism is never written back as alive.
func (sr *SwarmRegistry) StopKeeper() { sr.keeper.Stop() }

// NewSwarmRegistry creates a new SwarmRegistry with the given organism ID and element.
func NewSwarmRegistry(organismID, element string) *SwarmRegistry {
	if organismID == "" {
		organismID = fmt.Sprintf("org_%d_%d", os.Getpid(), time.Now().Unix())
	}
	return &SwarmRegistry{OrganismID: organismID, Element: element}
}

// Register writes PID file and registers in mesh.db.
func (sr *SwarmRegistry) Register() error {
	if err := os.MkdirAll(swarmDir, 0755); err != nil {
		return err
	}
	sr.PidFile = filepath.Join(swarmDir, sr.OrganismID+".pid")
	pidData, _ := json.Marshal(map[string]interface{}{
		"pid":         os.Getpid(),
		"organism_id": sr.OrganismID,
		"started":     float64(time.Now().UnixMilli()) / 1000.0,
	})
	if err := os.WriteFile(sr.PidFile, pidData, 0644); err != nil {
		return err
	}
	if err := sr.initMeshDB(); err != nil {
		return err
	}
	return sr.registerInMesh()
}

func (sr *SwarmRegistry) initMeshDB() error {
	dbPath := filepath.Join(swarmDir, "mesh.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return err
	}
	db.Exec("PRAGMA journal_mode=WAL")
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS organisms(
		id TEXT PRIMARY KEY, pid INTEGER, stage INTEGER,
		n_params INTEGER, syntropy REAL, entropy REAL,
		last_heartbeat REAL, parent_id TEXT,
		status TEXT DEFAULT 'alive',
		element TEXT, global_step INTEGER,
		gen_mag REAL, overlay_fade REAL)`)
	// Migrations for existing databases
	db.Exec("ALTER TABLE organisms ADD COLUMN element TEXT")
	db.Exec("ALTER TABLE organisms ADD COLUMN global_step INTEGER") // repair 7: age in training steps, for the witness and for gates
	// Routing repair 6: the voice, beside the age. gen_mag is the mean |logit|
	// of the raw transformer at the first step of the last generation and
	// overlay_fade is 1 - the overlay weight that magnitude bought, the two
	// numbers dnaWrite already prints as mag= and fade= and that nothing
	// outside the process could see. §13 of molequla_new_logic.md wants the
	// sentence-boundary gate keyed on the voice rather than on the stage
	// label; these are the columns it reads. Same idempotent ALTER as
	// global_step above: an error on an existing column is the expected
	// outcome and is discarded like the two before it.
	db.Exec("ALTER TABLE organisms ADD COLUMN gen_mag REAL")
	db.Exec("ALTER TABLE organisms ADD COLUMN overlay_fade REAL")
	if err != nil {
		db.Close()
		return err
	}
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS messages(
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		from_id TEXT, to_id TEXT, type TEXT, payload TEXT, ts REAL)`)
	if err != nil {
		db.Close()
		return err
	}
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS training_lock(
		organism_id TEXT PRIMARY KEY, acquired_at REAL)`)
	if err != nil {
		db.Close()
		return err
	}
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS mitosis_lock(
		organism_id TEXT PRIMARY KEY, acquired_at REAL)`)
	if err != nil {
		db.Close()
		return err
	}
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS growth_lock(
		organism_id TEXT PRIMARY KEY, acquired_at REAL)`)
	if err != nil {
		db.Close()
		return err
	}
	sr.MeshDB = db
	return nil
}

// AcquireMitosisSlot atomically admits a divide ONLY if (a) no other organism is
// mid-divide (mitosis_lock free) AND (b) the live colony is below maxOrganisms —
// one SQL statement, TOCTOU-safe across the multi-process lineage (audit C3: the
// old len(DiscoverPeers)+1 check-then-act let concurrent dividers overshoot the
// cap). The lock self-expires after 30s, which spans the child's registration
// window so the next admit counts the new child. maxOrganisms<=0 disables the cap.
func (sr *SwarmRegistry) AcquireMitosisSlot(maxOrganisms int) bool {
	if sr.MeshDB == nil || maxOrganisms <= 0 {
		return true
	}
	now := float64(time.Now().UnixMilli()) / 1000.0
	lockCutoff := now - 30.0 // mitosis lock expiry
	liveCutoff := now - 60.0 // heartbeat freshness (matches DiscoverPeers window)
	result, err := sr.MeshDB.Exec(
		`INSERT OR REPLACE INTO mitosis_lock(organism_id, acquired_at)
		 SELECT ?, ? WHERE
		   NOT EXISTS (SELECT 1 FROM mitosis_lock WHERE organism_id != ? AND acquired_at > ?)
		   AND (SELECT COUNT(*) FROM organisms WHERE status='alive' AND last_heartbeat > ?) < ?`,
		sr.OrganismID, now, sr.OrganismID, lockCutoff, liveCutoff, maxOrganisms)
	if err != nil {
		return false
	}
	rows, _ := result.RowsAffected()
	return rows > 0
}

// ReleaseMitosisLock frees the slot early (on a failed spawn). On success the lock
// is left to expire so the new child registers before the next divide is admitted.
func (sr *SwarmRegistry) ReleaseMitosisLock() {
	if sr.MeshDB == nil {
		return
	}
	sr.MeshDB.Exec(`DELETE FROM mitosis_lock WHERE organism_id = ?`, sr.OrganismID)
}

func (sr *SwarmRegistry) registerInMesh() error {
	if sr.MeshDB == nil {
		return nil
	}
	_, err := sr.MeshDB.Exec(
		"INSERT OR REPLACE INTO organisms(id,pid,stage,n_params,syntropy,entropy,last_heartbeat,status,element) "+
			"VALUES(?,?,0,0,0.0,0.0,?,'alive',?)",
		sr.OrganismID, os.Getpid(), float64(time.Now().UnixMilli())/1000.0, sr.Element)
	return err
}

// ReserveChildSlot inserts a freshly-spawned child into the mesh as 'alive' with a
// current heartbeat, so AcquireMitosisSlot's population cap counts it immediately
// (M-GOV-001). Without this the child is invisible to the cap until it boots and
// self-registers, letting a sibling overshoot the cap in that window. The child
// overwrites this row on its own registerInMesh (INSERT OR REPLACE, same id). A
// child that never boots ages out of the cap once its heartbeat goes stale.
func (sr *SwarmRegistry) ReserveChildSlot(childID string, pid int, element string) {
	if sr.MeshDB == nil {
		return
	}
	sr.MeshDB.Exec(
		"INSERT OR REPLACE INTO organisms(id,pid,stage,n_params,syntropy,entropy,last_heartbeat,parent_id,status,element) "+
			"VALUES(?,?,0,0,0.0,0.0,?,?,'alive',?)",
		childID, pid, float64(time.Now().UnixMilli())/1000.0, sr.OrganismID, element)
}

// sayMeshError reports one failed mesh write per distinct error. The write
// this guards is the organism's only statement that it is alive, so when it
// fails the colony's own governor and the witness both stop seeing an organism
// that is running perfectly well — a failure with no symptom except silence.
// It happened twice, both times because a column had been added to the write
// and not to the schema in front of it; the second time it cost a red test and
// twelve runs to name. Once per distinct error, because the tick loop beats
// every ten ticks for the life of the run.
func (sr *SwarmRegistry) sayMeshError(err error) {
	if sr == nil || err == nil {
		return
	}
	key := err.Error()
	sr.meshErrMu.Lock()
	if sr.meshErrSaid == nil {
		sr.meshErrSaid = map[string]bool{}
	}
	said := sr.meshErrSaid[key]
	sr.meshErrSaid[key] = true
	sr.meshErrMu.Unlock()
	if said {
		return
	}
	fmt.Printf("[ecology] mesh refused the heartbeat of %s: %s — this organism is alive and invisible to the colony\n",
		sr.OrganismID, key)
}

// Heartbeat performs periodic state update in mesh.db. globalStep is the
// organism's age in training steps (repair 7); genMag and overlayFade are its
// voice (routing repair 6) — the raw transformer magnitude at the first step of
// the last generation, and how far the corpus overlay has faded out of it.
// Both were process-local until now, readable only in the organism's own
// stdout, so no gate outside the process could follow the voice.
func (sr *SwarmRegistry) Heartbeat(stage, nParams int, syntropy, entropy float64, globalStep int, genMag, overlayFade float64) {
	if sr.MeshDB == nil {
		return
	}
	if _, err := sr.MeshDB.Exec(
		"UPDATE organisms SET stage=?,n_params=?,syntropy=?,entropy=?,last_heartbeat=?,status='alive',global_step=?,gen_mag=?,overlay_fade=? WHERE id=?",
		stage, nParams, syntropy, entropy, float64(time.Now().UnixMilli())/1000.0, globalStep, genMag, overlayFade, sr.OrganismID); err != nil {
		sr.sayMeshError(err)
	}
	sr.keeper.Set(stage, nParams, syntropy, entropy, globalStep, genMag, overlayFade) // nil-safe; the keeper repeats this state
}

// DiscoverPeers finds other living organisms.
func (sr *SwarmRegistry) DiscoverPeers(timeoutSeconds float64) []map[string]interface{} {
	if sr.MeshDB == nil {
		return nil
	}
	if timeoutSeconds <= 0 {
		timeoutSeconds = 60
	}
	cutoff := float64(time.Now().UnixMilli())/1000.0 - timeoutSeconds
	rows, err := sr.MeshDB.Query(
		"SELECT id,pid,stage,n_params,syntropy,entropy,status FROM organisms "+
			"WHERE status='alive' AND last_heartbeat>? AND id!=?",
		cutoff, sr.OrganismID)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var peers []map[string]interface{}
	for rows.Next() {
		var id, status string
		var pid, stage, nParams int
		var syntropy, entropy float64
		rows.Scan(&id, &pid, &stage, &nParams, &syntropy, &entropy, &status)
		peers = append(peers, map[string]interface{}{
			"id": id, "pid": pid, "stage": stage, "n_params": nParams,
			"syntropy": syntropy, "entropy": entropy, "status": status,
		})
	}
	return peers
}

// MarkHibernating marks this organism as sleeping in mesh.db.
func (sr *SwarmRegistry) MarkHibernating() {
	sr.StopKeeper() // before the row changes, so no late beat resurrects it
	if sr.MeshDB != nil {
		sr.MeshDB.Exec("UPDATE organisms SET status='sleeping' WHERE id=?", sr.OrganismID)
	}
}

// LogMessage logs a message between organisms.
func (sr *SwarmRegistry) LogMessage(toID, msgType string, payload interface{}) {
	if sr.MeshDB != nil {
		payloadJSON, _ := json.Marshal(payload)
		sr.MeshDB.Exec(
			"INSERT INTO messages(from_id,to_id,type,payload,ts) VALUES(?,?,?,?,?)",
			sr.OrganismID, toID, msgType, string(payloadJSON),
			float64(time.Now().UnixMilli())/1000.0)
	}
}

// Unregister cleans up on exit.
func (sr *SwarmRegistry) Unregister() {
	if sr.MeshDB != nil {
		sr.MeshDB.Exec("UPDATE organisms SET status='dead' WHERE id=?", sr.OrganismID)
		sr.MeshDB.Close()
		sr.MeshDB = nil
	}
	if sr.PidFile != "" {
		os.Remove(sr.PidFile)
	}
}

// AcquireTrainingLock attempts to acquire the training lock in mesh.db.
// Returns true if lock acquired, false if another organism holds a fresh lock (< 30s).
func (sr *SwarmRegistry) AcquireTrainingLock() bool {
	if sr.MeshDB == nil {
		return true // no mesh = solo, always proceed
	}
	now := float64(time.Now().UnixMilli()) / 1000.0
	cutoff := now - 30.0 // lock expires after 30 seconds

	// Atomic check-and-acquire: single statement prevents TOCTOU race.
	// INSERT succeeds only if no fresh lock exists from another organism.
	result, err := sr.MeshDB.Exec(
		`INSERT OR REPLACE INTO training_lock(organism_id, acquired_at)
		 SELECT ?, ? WHERE NOT EXISTS (
		   SELECT 1 FROM training_lock WHERE organism_id != ? AND acquired_at > ?
		 )`,
		sr.OrganismID, now, sr.OrganismID, cutoff)
	if err != nil {
		return false
	}
	rows, _ := result.RowsAffected()
	return rows > 0
}

// ReleaseTrainingLock releases the training lock in mesh.db.
func (sr *SwarmRegistry) ReleaseTrainingLock() {
	if sr.MeshDB == nil {
		return
	}
	sr.MeshDB.Exec("DELETE FROM training_lock WHERE organism_id=?", sr.OrganismID)
}

// The growth lock is the training lock's shape over a longer event. Growth and
// the warmup behind it are one memory peak lasting minutes, so the TTL is minutes
// and the holder re-stamps it on a ticker; the TTL only bounds how long a lock
// left by a killed organism blocks its siblings. It is a separate table from
// training_lock on purpose: CoordinateWarmup serializes every micro-burst, which
// cost 3 of 4 organisms their whole tick (2026-06-03), and that is not the price
// of keeping four stage transitions apart.
const (
	growthLockTTL     = 300.0
	growthLockRefresh = 60 * time.Second
)

// AcquireGrowthLock admits one grower at a time across the colony. No mesh = solo
// = always admitted. On success a refresher keeps the lock fresh until Release.
func (sr *SwarmRegistry) AcquireGrowthLock() bool {
	if sr.MeshDB == nil {
		return true
	}
	now := float64(time.Now().UnixMilli()) / 1000.0
	cutoff := now - growthLockTTL
	result, err := sr.MeshDB.Exec(
		`INSERT OR REPLACE INTO growth_lock(organism_id, acquired_at)
		 SELECT ?, ? WHERE NOT EXISTS (
		   SELECT 1 FROM growth_lock WHERE organism_id != ? AND acquired_at > ?
		 )`,
		sr.OrganismID, now, sr.OrganismID, cutoff)
	if err != nil {
		return false
	}
	rows, _ := result.RowsAffected()
	if rows <= 0 {
		return false
	}
	sr.startGrowthRefresh()
	return true
}

// RefreshGrowthLock re-stamps the holder's row. A row that is not the holder's is
// never touched, so a refresh cannot steal a lock.
func (sr *SwarmRegistry) RefreshGrowthLock() {
	if sr.MeshDB == nil {
		return
	}
	sr.MeshDB.Exec("UPDATE growth_lock SET acquired_at=? WHERE organism_id=?",
		float64(time.Now().UnixMilli())/1000.0, sr.OrganismID)
}

func (sr *SwarmRegistry) startGrowthRefresh() {
	sr.growthMu.Lock()
	defer sr.growthMu.Unlock()
	if sr.growthStop != nil {
		return // already refreshing
	}
	stop := make(chan struct{})
	sr.growthStop = stop
	go func() {
		t := time.NewTicker(growthLockRefresh)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				sr.RefreshGrowthLock()
			}
		}
	}()
}

// ReleaseGrowthLock stops the refresher and frees the lock for the next sibling.
func (sr *SwarmRegistry) ReleaseGrowthLock() {
	sr.growthMu.Lock()
	if sr.growthStop != nil {
		close(sr.growthStop)
		sr.growthStop = nil
	}
	sr.growthMu.Unlock()
	if sr.MeshDB == nil {
		return
	}
	sr.MeshDB.Exec("DELETE FROM growth_lock WHERE organism_id=?", sr.OrganismID)
}

// performMitosis divides the organism. Parent continues. The child inherits the
// parent's growth stage via the checkpoint dimensions (not infant).
func performMitosis(model *GPT, tok *EvolvingTokenizer, db *sql.DB, swarm *SwarmRegistry, syntracker *SyntropyTracker) (string, error) {
	// M-SPAWN-002: allocate a unique child dir EXCLUSIVELY. os.MkdirAll silently
	// reuses an existing dir, so a childID collision (same second, or the governor
	// disabled) let two children share a dir and organism id. UnixNano+pid+rand plus
	// an exclusive os.Mkdir (which fails if the dir exists) makes a collision both
	// astronomically unlikely and detected.
	baseDir := filepath.Join(os.Getenv("HOME"), ".molequla")
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		return "", err
	}
	var childID, childDir string
	for attempt := 0; attempt < 8; attempt++ {
		childID = fmt.Sprintf("org_%d_%d_%d", time.Now().UnixNano(), os.Getpid(), rand.Intn(1000000))
		childDir = filepath.Join(baseDir, childID)
		err := os.Mkdir(childDir, 0755)
		if err == nil {
			break
		}
		if !os.IsExist(err) {
			return "", err
		}
		childID = ""
	}
	if childID == "" {
		return "", fmt.Errorf("mitosis: could not allocate a unique child dir")
	}

	// M-SPAWN-001: a failed spawn must NOT leave the parent "relieved" and cooled
	// down with no child. Snapshot the overload state and restore it unless the
	// spawn commits (committed=true just before the successful return below).
	savedLMT := syntracker.LastMitosisTime
	savedBH := append([]BurstRecord(nil), syntracker.BurstHistory...)
	savedEH := append([]float64(nil), syntracker.EntropyHistory...)
	committed := false
	defer func() {
		if !committed {
			syntracker.LastMitosisTime = savedLMT
			syntracker.BurstHistory = savedBH
			syntracker.EntropyHistory = savedEH
		}
	}()

	// (audit C4 + c) Relieve the parent's overload and stamp the cooldown BEFORE
	// capturing the inherited history below — so the child does NOT inherit a
	// pre-overloaded burst/entropy state (which made it divide again the instant
	// its own 300s cooldown cleared). Division relieves the overwhelm at the root.
	syntracker.LastMitosisTime = float64(time.Now().UnixMilli()) / 1000.0
	syntracker.relieveOverload()

	// Save parent checkpoint for child's reference
	parentCkpt := filepath.Join(childDir, "parent_ckpt.json")
	if err := SaveCheckpoint(model, tok, parentCkpt); err != nil {
		return "", err
	}

	// Write birth config with inherited memory
	birth := map[string]interface{}{
		"organism_id":   childID,
		"parent_id":     swarm.OrganismID,
		"corpus_path":   CFG.CorpusPath,
		"db_path":       filepath.Join(childDir, "memory.sqlite3"),
		"ckpt_path":     parentCkpt, // load the parent checkpoint actually written at spawn time (see performMitosis above)
		"burst_history": syntracker.BurstHistory,
	}
	birthPath := filepath.Join(childDir, "birth.json")
	birthJSON, _ := json.Marshal(birth)
	if err := os.WriteFile(birthPath, birthJSON, 0644); err != nil {
		return "", err
	}

	// Log in mesh
	swarm.LogMessage(childID, "mitosis:spawn",
		map[string]interface{}{"parent_stage": model.CurrentGrowthStage()})
	dbLogGrowth(db, model, tok, loadCorpusLines(CFG.CorpusPath), 0.0,
		fmt.Sprintf("mitosis:spawn:%s", childID))

	// Spawn child process
	exePath, err := os.Executable()
	if err != nil {
		exePath = os.Args[0]
	}
	// (audit BUG A) The child must run AUTONOMOUS — inherit the parent's run mode
	// (--evolution always; --gpu/--cross-graze/--element to match). Without these
	// it booted into the interactive REPL ("Type and press Enter") and stalled.
	childArgs := []string{"--organism-id", childID, "--config", birthPath, "--evolution"}
	if CFG.UseGPU {
		childArgs = append(childArgs, "--gpu")
	}
	if CFG.CrossGraze {
		childArgs = append(childArgs, "--cross-graze")
	}
	if CFG.CorpusLogitOverlay {
		childArgs = append(childArgs, "--corpus-overlay") // a child of an overlay-running parent was born without it (audit C, C-RDM-12)
	}
	base := filepath.Base(CFG.CorpusPath)
	if strings.HasPrefix(base, "nonames_") && strings.HasSuffix(base, ".txt") {
		childArgs = append(childArgs, "--element", strings.TrimSuffix(strings.TrimPrefix(base, "nonames_"), ".txt"))
	}
	cmd := exec.Command(exePath, childArgs...)
	// (audit BUG B) Child writes its OWN log, not the parent's stdout — inheriting
	// os.Stdout mixed the whole lineage into one file and made monitoring impossible.
	childLog, _ := os.Create(filepath.Join(childDir, "train.log"))
	if childLog != nil {
		cmd.Stdout = childLog
		cmd.Stderr = childLog
	} else {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	if err := cmd.Start(); err != nil {
		if childLog != nil {
			childLog.Close()
		}
		return "", err
	}
	// Reap the child on its eventual exit (no zombie of this long-lived parent)
	// and close its log handle.
	go func() {
		_ = cmd.Wait()
		if childLog != nil {
			childLog.Close()
		}
	}()

	// M-GOV-001: reserve the child's slot in the mesh NOW so the population cap
	// counts it immediately. Otherwise a sibling's AcquireMitosisSlot, run in the
	// window before the child self-registers, sees COUNT < cap and admits an
	// over-cap spawn. The child overwrites this row (INSERT OR REPLACE) on boot.
	swarm.ReserveChildSlot(childID, cmd.Process.Pid, swarm.Element)

	// M-SPAWN-001: the spawn is committed — keep the relieved/cooled parent state.
	committed = true

	fmt.Printf("[ecology] Child %s spawned (pid=%d) — log %s/train.log\n", childID, cmd.Process.Pid, childDir)
	return childID, nil
}

// performHibernation saves state, marks sleeping, and signals exit.
func performHibernation(model *GPT, tok *EvolvingTokenizer, db *sql.DB, swarm *SwarmRegistry) {
	fmt.Printf("[ecology] HIBERNATION — organism %s going to sleep\n", swarm.OrganismID)
	SaveCheckpoint(model, tok, "")
	swarm.MarkHibernating()
	dbLogGrowth(db, model, tok, loadCorpusLines(CFG.CorpusPath), 0.0,
		fmt.Sprintf("hibernate:%s", swarm.OrganismID))
}

// ============================================================
// DNA EXCHANGE — organisms feed each other through dna/ directory
// ============================================================

var dnaElements = []string{"earth", "air", "water", "fire"}

// dnaWrite generates text and writes it to dna/output/{element}/ for other organisms to consume.
func dnaWrite(element string, model *GPT, tok *EvolvingTokenizer, field *CooccurField, docs []string, step int) {
	if element == "" || len(docs) == 0 {
		return
	}
	// Publish the field and tokenizer the organism is speaking with, so the
	// cafeteria in experience_routing.go judges the fragments dnaRead is about
	// to offer (one line below, same tick) against this same state.
	experienceRouting.observe(field, tok)

	probes := []string{
		"What do you feel?", "Tell me about yourself.",
		"What is truth?", "What matters?",
		"Speak.", "What do you remember?",
	}
	// §14: the world must be metabolized by somebody before it becomes culture.
	// The probe is where that happens — the organism is asked about what it
	// just ate and answers in its own voice. The six fixed questions remain the
	// fallback for an organism that has not eaten yet.
	probe := experienceRouting.probe(step)
	if probe == "" {
		probe = probes[step%len(probes)]
	}

	// GenerateResonant takes model.mu.Lock internally — do NOT double-lock
	answer := GenerateResonant(model, tok, field, probe, docs, true)

	// DNA fragment = the organism's voice (answer) plus a sample of the
	// real text it holds, padded toward CFG.DNAFragmentTargetBytes. A
	// child-stage model generates only a few degenerate bytes; the corpus
	// sample carries the substance so the fragment is worth exchanging. As
	// the organism matures `answer` grows into real text and the generation
	// share of the fragment rises on its own.
	var b strings.Builder
	b.WriteString(strings.TrimSpace(answer))
	// The padding leads with what this organism has eaten most recently from
	// its siblings, so the fragment carries the organism's current diet and not
	// an arbitrary slice of its whole corpus. Sense lines are not in that list
	// and are skipped when a random draw lands on one: §14 forbids raw outside
	// experience from entering collective DNA without passing through an
	// organism, and a padded line is a byte copy, not a passage. The world
	// reaches the ecology through `answer` above, which was generated from it.
	for _, line := range experienceRouting.recentPadding(CFG.ExperienceRecentPadLines) {
		if b.Len() >= CFG.DNAFragmentTargetBytes {
			break
		}
		if b.Len() > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(line)
	}
	for i := 0; b.Len() < CFG.DNAFragmentTargetBytes && i < 600; i++ {
		line := strings.TrimSpace(docs[rand.Intn(len(docs))])
		if line == "" || experienceRouting.isRawExperience(line) {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(line)
	}
	frag := strings.TrimSpace(b.String())
	if len(frag) < CFG.DNAMinFragmentBytes {
		return
	}

	dir := filepath.Join("../dna/output", element)
	os.MkdirAll(dir, 0755)
	fname := filepath.Join(dir, fmt.Sprintf("gen_%d_%d.txt", time.Now().Unix(), step))
	os.WriteFile(fname, []byte(frag+"\n"), 0644)
	// gen is how much of the fragment the organism actually said; mag and fade
	// are the state the overlay was in when it said it (fade=1.00: the overlay
	// is gone and the speech is the transformer's own). GenerateResonant took
	// model.mu itself and has released it, so the read takes its own lock.
	gen := len(strings.TrimSpace(answer))
	var mag, fade float64
	if model != nil {
		model.mu.Lock()
		mag = model.lastGenMag
		fade = 1 - model.lastOverlayWeight
		model.mu.Unlock()
	}
	// eligible is the §13 gate (experience_routing.go): whether this organism's
	// voice is its own enough to receive sentence-boundary knowledge injection.
	// The injection itself is not implemented — the decision is printed so it
	// can be read back out of the logs before anything is built on it.
	eligible := 0
	if ok, _ := injectionEligible(fade, mag); ok {
		eligible = 1
	}
	fmt.Printf("[dna] %s wrote %d bytes to ecology | gen=%d mag=%.2f fade=%.2f eligible=%d\n", element, len(frag), gen, mag, fade, eligible)
	// The writer is the only one that deletes: readers keep cursors (repair 3).
	dnaPruneOwn(element, time.Duration(CFG.DNARetainSeconds*float64(time.Second)), CFG.DNARetainFiles)
}

// dnaRead eats fragments from every source this organism reads (the other
// elements plus CFG.DNAExtraSources), newer than its cursor, at most
// CFG.DNAMaxReadsPerTick sibling fragments and CFG.DNAExtraReadsPerTick extra
// ones per call. Fragments are never removed here; the cursor is advanced and
// persisted instead. Returns bytes added to the corpus.
//
// The two budgets are routing repair 2. Under one shared bound the extra
// sources were read last, after three siblings that emit a fragment per tick
// each, so a senses fragment waited behind sibling chatter for as long as the
// backlog lasted: in the 2026-09-13T20:26Z session the shared cap was spent
// before the source list ran out on 12 of earth's 15 reads, 9 of air's 13, 10
// of water's 13 and 4 of fire's 14. A separate counter per half makes the
// order in dnaSources an order of service and not an order of entitlement.
func dnaRead(element string, corpusPath string, qbuf *QuantumBuffer, tok *EvolvingTokenizer, cur *dnaCursor) int {
	if element == "" {
		return 0
	}
	if cur == nil {
		cur = &dnaCursor{Last: map[string]string{}}
	}
	added := 0
	budget := func(n int) int {
		if n <= 0 {
			return 1 << 30
		}
		return n
	}
	siblingLeft := budget(CFG.DNAMaxReadsPerTick)
	extraLeft := budget(CFG.DNAExtraReadsPerTick)
	// Membership, not order: dnaSources decides who is served in what order,
	// this decides which counter the read is charged to. Built from CFG so it
	// stays true however dnaSources is later allocated.
	isExtra := make(map[string]bool, len(CFG.DNAExtraSources))
	for _, s := range CFG.DNAExtraSources {
		if s != "" && s != element {
			isExtra[s] = true
		}
	}
	var consumed []string
	moved := false
	// Coverage costs a BPE encode of the sampled fragment — 21.7 ms on cores
	// 4-7 (MOLEQULALOG2.md, 2026-09-15). A declined fragment costs one of these
	// and no read budget, so this is what bounds the work a tick can spend
	// deciding rather than eating; when it is spent the loop stops with the
	// cursors where they are and the rest is examined next tick.
	measured := 0
	measuredCap := CFG.ExperienceMaxMeasuredPerTick
	if measuredCap <= 0 {
		measuredCap = 1 << 30
	}

reading:
	for _, src := range dnaSources(element) {
		left := &siblingLeft
		if isExtra[src] {
			left = &extraLeft
		}
		if *left <= 0 {
			continue // that half is spent; the other half is still owed its reads
		}
		dir := filepath.Join("../dna/output", src)
		for _, name := range dnaListNew(dir, cur.Last[src]) {
			if *left <= 0 {
				break
			}
			fpath := filepath.Join(dir, name)
			data, err := os.ReadFile(fpath)
			if err != nil {
				continue
			}
			text := strings.TrimSpace(string(data))
			if len(text) < CFG.DNAMinFragmentBytes {
				*left-- // a read happened; it was not food
				cur.Last[src] = name // too short to be food; step past it, do not delete
				moved = true
				continue
			}
			// The cafeteria (experience_routing.go): this organism's own plate,
			// decided from the fragment's name and its coverage under this
			// organism's own field. A declined plate advances the cursor — the
			// fragment is not eaten and is not looked at again — and it does
			// NOT spend a read from either budget: those two bound how much an
			// organism EATS per tick, and a colony that declines a third of
			// what it is offered would otherwise starve on a backlog. What a
			// decline does cost is a coverage measurement, and that is what
			// ExperienceMaxMeasuredPerTick bounds instead.
			ok, why, _ := experienceRouting.admits(element, src, name, text)
			if why != "owner" && why != "broadcast" {
				measured++
			}
			if !ok {
				cur.Last[src] = name
				moved = true
				if measured >= measuredCap {
					break reading
				}
				continue
			}
			*left--
			// Append to own corpus — the organism eats another's words, cut
			// into corpus lines so that the whole fragment survives
			// loadCorpusLines instead of its first CFG.MaxLineChars bytes
			// (routing repair 3).
			lines := splitCorpusLine(text, CFG.MaxLineChars)
			if len(lines) == 0 {
				cur.Last[src] = name
				moved = true
				continue
			}
			f, err := os.OpenFile(corpusPath, os.O_APPEND|os.O_RDWR, 0644)
			if err != nil {
				continue
			}
			// A corpus whose last line has no newline of its own would take
			// this fragment's first sentence onto the end of it — one
			// malformed line per fragment, and the one carrying the sense's
			// header at that. saveCorpusLines always terminates its lines, so
			// this is the hand-edited or truncated file, not the normal one:
			// read the last byte (O_RDWR because an O_APPEND handle cannot)
			// and close the line before opening a new one.
			if fi, err := f.Stat(); err == nil && fi.Size() > 0 {
				var last [1]byte
				if _, err := f.ReadAt(last[:], fi.Size()-1); err == nil && last[0] != '\n' {
					f.WriteString("\n")
				}
			}
			wrote := 0
			for _, ln := range lines {
				f.WriteString(ln + "\n")
				wrote += len(ln)
			}
			f.Close()
			// The growth clock counts what reached the field, not what was
			// offered to it. Before repair 3 the two were the same number by
			// accident of the append and different by 20× in fact; now they
			// differ only by the whitespace between sentences.
			added += wrote
			// What was just eaten shapes what is said next (§14) — see
			// dnaWrite. The meal is the lines that were appended, not the
			// fragment they were cut from: those lines are what loadCorpusLines
			// will hand back, so they are what the padding must be able to
			// recognise.
			experienceRouting.remember(src, lines, experienceIsExtraSource(src))
			consumed = append(consumed, fmt.Sprintf("%s/%s", src, name))
			if qbuf != nil && tok != nil {
				qbuf.Feed(text, tok)
			}
			cur.Last[src] = name // advanced only after a successful append
			moved = true
		}
	}
	if moved {
		cur.save()
	}
	if added > 0 {
		fmt.Printf("[dna] %s consumed %d bytes from %d files: %v\n",
			element, added, len(consumed), consumed)
	}
	return added
}

// ============================================================
// NOTORCH: gradient-free delta training (ported from AML C)
// ============================================================
//
// The key insight: delta adapters are low-rank (A @ B @ x), so we can update
// them with a teaching signal instead of backpropagation. No compute graph,
// no gradient arrays, no closure allocations. Pure arithmetic.
//
// A[i,r] += lr * x[i] * u[r] * signal
// B[r,j] += lr * u[r] * dy[j] * signal
// u = noise-modulated channel vector (deterministic from seed)
// signal = teaching signal, clamped [-2, 2]
// Adaptive decay: stronger when delta norm is large
// Clamp weights to [-10, 10]

// notorchRand advances the per-model PRNG and returns a noise-modulated float64.
// Uses LCG matching AML's am_frandn + signal-dependent noise modulation.
func notorchRand(seed *uint32, signal float64) float64 {
	*seed = *seed*1664525 + 1013904223
	u := float64(*seed&0x7FFFFFFF) / float64(0x7FFFFFFF)
	raw := (u - 0.5) * 3.464 // ~N(0,1) approximation (matches AML)
	// Signal-dependent noise: stronger signal = cleaner channel (less noise)
	k := 0.35 + 0.65*(1.0-math.Abs(signal))
	return raw * k
}

// notorchStep updates a single DeltaAdapter without backpropagation.
// x = input vector (len = B.Nin = nin)
// dy = output error (len = A.Nout = nout)
// signal = teaching signal (positive = good, negative = bad)
func notorchStep(da *DeltaAdapter, x []float64, dy []float64, signal float64, lr float64, seed *uint32) {
	// Clamp signal to [-2, 2]
	if signal > 2.0 {
		signal = 2.0
	}
	if signal < -2.0 {
		signal = -2.0
	}

	decay := CFG.NotorchDecay

	rank := da.A.Nin // A is [nout x rank], B is [rank x nin]
	nout := da.A.Nout
	nin := da.B.Nin

	// Generate noise-modulated channel vector u[rank]
	u := make([]float64, rank)
	for r := 0; r < rank; r++ {
		u[r] = notorchRand(seed, signal)
	}

	// Compute A-norm only for adaptive decay (matches AML ariannamethod.c:2562-2572)
	aNorm := 0.0
	aSize := nout * rank
	for i := 0; i < nout; i++ {
		for r := 0; r < rank; r++ {
			v := da.A.Rows[i].Data[r]
			aNorm += v * v
		}
	}
	if aSize > 0 {
		aNorm = math.Sqrt(aNorm / float64(aSize))
	}

	// Adaptive decay: decay - 0.004*min(norm/10, 1), floor 0.990 (AML formula)
	adaptiveDecay := decay - 0.004*math.Min(aNorm/10.0, 1.0)
	if adaptiveDecay < 0.990 {
		adaptiveDecay = 0.990
	}

	// Update A: A[i,r] += lr * x_scale[i] * u[r] * signal, then decay
	// x_scale[i] is used as proxy for output gradient direction
	for i := 0; i < nout; i++ {
		dyI := 0.0
		if i < len(dy) {
			dyI = dy[i]
		}
		for r := 0; r < rank; r++ {
			da.A.Rows[i].Data[r] *= adaptiveDecay
			da.A.Rows[i].Data[r] += lr * dyI * u[r] * signal
			// Clamp weights to [-10, 10]
			if da.A.Rows[i].Data[r] > 10.0 {
				da.A.Rows[i].Data[r] = 10.0
			} else if da.A.Rows[i].Data[r] < -10.0 {
				da.A.Rows[i].Data[r] = -10.0
			}
		}
	}

	// Update B: B[r,j] += lr * u[r] * x[j] * signal, then decay
	for r := 0; r < rank; r++ {
		for j := 0; j < nin; j++ {
			xJ := 0.0
			if j < len(x) {
				xJ = x[j]
			}
			da.B.Rows[r].Data[j] *= adaptiveDecay
			da.B.Rows[r].Data[j] += lr * u[r] * xJ * signal
			// Clamp weights to [-10, 10]
			if da.B.Rows[r].Data[j] > 10.0 {
				da.B.Rows[r].Data[j] = 10.0
			} else if da.B.Rows[r].Data[j] < -10.0 {
				da.B.Rows[r].Data[j] = -10.0
			}
		}
	}
}

// notorchTrainSteps trains delta adapters WITHOUT autograd.
// No backward pass, no compute graph, no gradient arrays.
// Uses direct feedback alignment with teaching signal.
func notorchTrainSteps(model *GPT, tok *EvolvingTokenizer, docs []string, steps int, lr float64) {
	if len(docs) == 0 || len(model.Deltas) == 0 {
		return
	}

	model.mu.Lock()
	defer model.mu.Unlock()

	prevLoss := math.MaxFloat64

	for step := 0; step < steps; step++ {
		// Sample random doc
		doc := docs[rand.Intn(len(docs))]
		ids := tok.Encode(doc)
		if len(ids) < 2 {
			continue
		}

		// Cap sequence length to BlockSize
		seqLen := len(ids) - 1
		if seqLen > model.BlockSize {
			seqLen = model.BlockSize
		}

		// Forward pass WITHOUT autograd — the whole point of notorch
		gradEnabled.Store(false)

		keys := make([][]*Vec, model.NLayer)
		values := make([][]*Vec, model.NLayer)

		var totalLoss float64
		var lastLogits *Vec
		var target int

		for pos := 0; pos < seqLen; pos++ {
			logits := model.ForwardStep(ids[pos], pos, keys, values)
			target = ids[pos+1]

			// Cross-entropy loss (scalar only, no autograd)
			maxLogit := logits.Data[0]
			for _, v := range logits.Data {
				if v > maxLogit {
					maxLogit = v
				}
			}
			sumExp := 0.0
			for _, v := range logits.Data {
				sumExp += math.Exp(v - maxLogit)
			}
			logSumExp := maxLogit + math.Log(sumExp)
			loss := logSumExp - logits.Data[target]
			totalLoss += loss

			lastLogits = logits
		}

		gradEnabled.Store(true)

		avgLoss := totalLoss / float64(seqLen)

		// Teaching signal: improvement = positive signal
		rawSignal := 0.0
		if prevLoss < math.MaxFloat64 {
			rawSignal = prevLoss - avgLoss // positive if loss decreased
		}
		prevLoss = avgLoss

		// Step 0: no signal yet, skip adapter update (only record prevLoss)
		if rawSignal == 0.0 && step == 0 {
			continue
		}

		// Prophecy debt: measures surprise of chosen token (AML am_compute_prophecy_debt)
		// diff/(diff+1) maps to [0, 1) — always pushes toward better prediction
		if lastLogits != nil && target < len(lastLogits.Data) {
			maxLogitP := lastLogits.Data[0]
			for _, v := range lastLogits.Data {
				if v > maxLogitP {
					maxLogitP = v
				}
			}
			diff := maxLogitP - lastLogits.Data[target]
			if diff > 0 {
				debt := diff / (diff + 1.0)
				rawSignal += 0.3 * debt // blend prophecy debt into teaching signal
			}
		}

		// Normalize signal to [-1, 1] via tanh (AML signals are bounded; transformer loss deltas are not)
		signal := math.Tanh(rawSignal)

		// Compute logit error: softmax(logits) - one_hot(target)
		vocabSize := len(lastLogits.Data)
		dy := make([]float64, vocabSize)
		maxLogit := lastLogits.Data[0]
		for _, v := range lastLogits.Data {
			if v > maxLogit {
				maxLogit = v
			}
		}
		sumExp := 0.0
		for i := range lastLogits.Data {
			dy[i] = math.Exp(lastLogits.Data[i] - maxLogit)
			sumExp += dy[i]
		}
		for i := range dy {
			dy[i] /= sumExp // softmax
		}
		if target < vocabSize {
			dy[target] -= 1.0 // subtract one_hot
		}

		// Compute hidden error via direct feedback alignment:
		// hidden_dy = lm_head_weight^T @ logit_error
		lmHead := model.Base["lm_head"]
		hiddenDy := make([]float64, model.NEmbd)
		for j := 0; j < model.NEmbd; j++ {
			sum := 0.0
			nout := lmHead.Nout
			if nout > vocabSize {
				nout = vocabSize
			}
			for i := 0; i < nout; i++ {
				sum += lmHead.Rows[i].Data[j] * dy[i]
			}
			hiddenDy[j] = sum
		}

		// Update all delta adapters with correct dimensions per adapter type
		for _, mod := range model.Deltas {
			// lm_head delta: input=lastHidden[NEmbd], dy=logitError[vocabSize]
			if da, ok := mod["lm_head"]; ok && model.lastHidden != nil {
				notorchStep(da, model.lastHidden.Data, dy, signal, lr, &model.notorchSeed)
			}

			for li := 0; li < model.NLayer; li++ {
				lk := model.layerKeys[li]

				// Attention adapters: input=layerInputs[NEmbd], dy=hiddenDy[NEmbd]
				if li < len(model.layerInputs) && model.layerInputs[li] != nil {
					attnInput := model.layerInputs[li].Data
					for _, key := range []string{lk.wq, lk.wk, lk.wv, lk.wo} {
						if da, ok := mod[key]; ok {
							notorchStep(da, attnInput, hiddenDy, signal, lr, &model.notorchSeed)
						}
					}
				}

				// MLP adapters: need mlpDy[4*NEmbd] for fc_g/fc_v, mlpIntermediates for fc2
				if li < len(model.mlpInputs) && model.mlpInputs[li] != nil {
					mlpInput := model.mlpInputs[li].Data // [NEmbd] — actual input to fc_g/fc_v

					// Compute mlpDy = fc2_base^T @ hiddenDy → projects NEmbd error to 4*NEmbd space
					fc2Base := model.Base[lk.fc2] // [NEmbd × 4*NEmbd]
					mlpWidth := 4 * model.NEmbd
					mlpDy := make([]float64, mlpWidth)
					for j := 0; j < mlpWidth; j++ {
						sum := 0.0
						for i := 0; i < model.NEmbd && i < fc2Base.Nout; i++ {
							if j < len(fc2Base.Rows[i].Data) {
								sum += fc2Base.Rows[i].Data[j] * hiddenDy[i]
							}
						}
						mlpDy[j] = sum
					}

					// fc_g, fc_v: input=mlpInputs[NEmbd], dy=mlpDy[4*NEmbd]
					for _, key := range []string{lk.fcG, lk.fcV} {
						if da, ok := mod[key]; ok {
							notorchStep(da, mlpInput, mlpDy, signal, lr, &model.notorchSeed)
						}
					}

					// fc2: input=mlpIntermediates[4*NEmbd], dy=hiddenDy[NEmbd]
					if li < len(model.mlpIntermediates) && model.mlpIntermediates[li] != nil {
						if da, ok := mod[lk.fc2]; ok {
							notorchStep(da, model.mlpIntermediates[li].Data, hiddenDy, signal, lr, &model.notorchSeed)
						}
					}
				}
			}
		}

		// Handle growth freeze
		if model.growthFreezeRemaining > 0 {
			model.growthFreezeRemaining--
			if model.growthFreezeRemaining < 0 {
				model.growthFreezeRemaining = 0
			}
		}

		model.globalStep++

		if step%100 == 0 {
			fmt.Printf("  notorch step %d/%d | loss %.4f | signal %.4f\n",
				step, steps, avgLoss, signal)
		}
	}
}

// parseCLIArgs parses --organism-id, --config, --element, --evolution, and
// the Phase B coherence-layer toggles (--spa-gate, --corpus-overlay) from
// os.Args. The two coherence-gate flags write directly into CFG so that
// pod-side measurement cells can flip them without rebuilding or editing
// the config file. Mutually orthogonal — pass either, both, or neither.
func parseCLIArgs() (organismID string, configPath string, element string, evolution bool) {
	for i := 1; i < len(os.Args); i++ {
		if os.Args[i] == "--organism-id" && i+1 < len(os.Args) {
			organismID = os.Args[i+1]
			i++
		} else if os.Args[i] == "--config" && i+1 < len(os.Args) {
			configPath = os.Args[i+1]
			i++
		} else if os.Args[i] == "--element" && i+1 < len(os.Args) {
			element = os.Args[i+1]
			i++
		} else if os.Args[i] == "--evolution" {
			evolution = true
		} else if os.Args[i] == "--spa-gate" {
			CFG.SPACoherenceGate = true
		} else if os.Args[i] == "--corpus-overlay" {
			CFG.CorpusLogitOverlay = true
		} else if os.Args[i] == "--trainer" && i+1 < len(os.Args) {
			// "notorch" (default) or "aml" — selects the training backend.
			CFG.Trainer = os.Args[i+1]
			i++
		} else if os.Args[i] == "--zero-warmup" {
			// Skip all per-stage warmup training. Used to test pure
			// Q-style zero-training coherence: embryo organism receives only
			// metaweight-seeded embeddings + overlay, no gradient steps.
			CFG.WarmupSteps = 0
		} else if os.Args[i] == "--gpu" {
			// Route inference Matvec through cuBLAS sgemm. Linux-only at
			// runtime (gpuReady() returns false elsewhere). Training stays
			// CPU/BLAS — autograd graph requires host tensors. See
			// gpu_bridge.go and modules/gpu/.
			CFG.UseGPU = true
		} else if os.Args[i] == "--witness" {
			// The mycelium as a witness (witness.go): a fifth process that
			// reads mesh.db and the DNA field and says what it sees. No
			// model is loaded, nothing is written back.
			witnessMode = true
		} else if os.Args[i] == "--witness-interval" && i+1 < len(os.Args) {
			if v, err := strconv.ParseFloat(os.Args[i+1], 64); err == nil && v > 0 {
				witnessInterval = v
			}
			i++
		} else if os.Args[i] == "--once" {
			witnessOnce = true
		} else if os.Args[i] == "--world-ingest" {
			// The world ledger's writer (world_ledger.go): reads the facts
			// the senses appended, files them into world_facts, closes what
			// they contradict and leaves the changes in ../dna/output/world/.
			// Its own process on purpose — the witness's mesh handle is
			// query_only and stays that way.
			worldIngestMode = true
		} else if os.Args[i] == "--world-facts" && i+1 < len(os.Args) {
			// Where the senses write their structured sidecar
			// (world_ledger.go). An empty argument turns the world ledger
			// off: the witness then reads the field and says what it sees,
			// and writes nothing at all, as it did before ROADMAP 10.
			CFG.WorldFactsPath = strings.TrimSpace(os.Args[i+1])
			i++
		} else if os.Args[i] == "--world-move-meters" && i+1 < len(os.Args) {
			// The gate between a fix that wandered and a phone that moved.
			if v, err := strconv.ParseFloat(os.Args[i+1], 64); err == nil && v >= 0 {
				CFG.WorldMoveMeters = v
			}
			i++
		} else if os.Args[i] == "--max-organisms" && i+1 < len(os.Args) {
			// Hard ceiling on the live colony, the cascade governor's admit
			// count. The default 16 was written for a pod; a phone passes 4.
			if v, err := strconv.Atoi(os.Args[i+1]); err == nil && v >= 0 {
				CFG.MaxOrganisms = v
			}
			i++
		} else if os.Args[i] == "--dna-extra-sources" && i+1 < len(os.Args) {
			// Comma-separated read-only directories under ../dna/output,
			// beside the four elements: what the phone's senses write
			// (phone1/senses.sh drops world, sound and place there). Food,
			// not organisms — nothing here writes back to them and the
			// writer is the only one that prunes them.
			CFG.DNAExtraSources = nil
			for _, s := range strings.Split(os.Args[i+1], ",") {
				if s = strings.TrimSpace(s); s != "" {
					CFG.DNAExtraSources = append(CFG.DNAExtraSources, s)
				}
			}
			i++
		} else if os.Args[i] == "--max-growth-stage" && i+1 < len(os.Args) {
			// Hard ceiling on ontogenesis: the organism stops at this stage
			// index whatever the corpus says. Clamped in main against the
			// growth table; the default is the last stage, so unset changes
			// nothing.
			if v, err := strconv.Atoi(os.Args[i+1]); err == nil && v >= 0 {
				CFG.MaxGrowthStage = v
			}
			i++
		} else if os.Args[i] == "--cross-graze" {
			// Dario-style cross-organism logit injection — read sibling DNA
			// emissions mirrored to ../dna/seen/<sibling>/ and boost their
			// recent token ids in the overlay'd logits before sampling. See
			// cross_graze.go. Requires --element to be set so the field
			// knows which siblings to scan.
			CFG.CrossGraze = true
		}
	}
	return
}

// cosineLR returns learning rate for the given global step using cosine schedule with linear warmup.
// stepsSinceGrowth enables LR ramp-up after each growth event (new weights need high LR initially).
func cosineLR(globalStep, stepsSinceGrowth int) float64 {
	if stepsSinceGrowth < CFG.CosineWarmupSteps {
		// Linear warmup from LRMin to LearningRate (resets after each growth)
		t := float64(stepsSinceGrowth) / math.Max(1, float64(CFG.CosineWarmupSteps))
		return CFG.LRMin + (CFG.LearningRate-CFG.LRMin)*t
	}
	progress := math.Min(1.0, float64(globalStep)/math.Max(1, float64(CFG.MaxTotalSteps)))
	return CFG.LRMin + 0.5*(CFG.LearningRate-CFG.LRMin)*(1.0+math.Cos(math.Pi*progress))
}

func backgroundTrainer(db *sql.DB, model *GPT, tok *EvolvingTokenizer, qbuf *QuantumBuffer, swarm *SwarmRegistry, stop chan struct{}, element string) {
	// This organism's read position into every DNA source, persisted in its
	// working directory so a restart continues where it left off (repair 3).
	dnaCur := loadDNACursor(dnaCursorFile)
	// And lo, asynchronous training shall occur, because sleeping is for humans.
	syntracker := NewSyntropyTracker()
	field := NewCooccurField()
	tickCount := 0
	var docs []string      // persists across ticks; reloaded throttled (see loop)
	lastFieldRebuild := -1 // tick of last corpus reload + field rebuild
	// Growth and the warmup behind it are one memory event spanning ticks, so the
	// colony lock taken before growth is released only after the warmup (repair 9).
	growthLockHeld := false
	releaseGrowth := func() {
		if growthLockHeld && swarm != nil {
			swarm.ReleaseGrowthLock()
		}
		growthLockHeld = false
	}
	defer releaseGrowth()
	// Stage-gated GPU: tiny stages run faster on CPU (kernel-launch overhead
	// dwarfs the small matmuls — measured 8 steps/s on GPU at child vs ~90 on
	// CPU); GPU pays off only at teen/adult. Match the current (seed) stage.
	ntSetGPUForStage(model.CurrentGrowthStage())

	// Inherit burst_history from parent (mitosis lineage)
	if len(model.inheritedBurstHistory) > 0 {
		syntracker.BurstHistory = make([]BurstRecord, len(model.inheritedBurstHistory))
		copy(syntracker.BurstHistory, model.inheritedBurstHistory)
		fmt.Printf("[ecology] syntracker inherited %d burst records from parent.\n", len(model.inheritedBurstHistory))
		model.inheritedBurstHistory = nil
	}

	for {
		select {
		case <-stop:
			return
		default:
		}

		tickCount++

		// Reload corpus + rebuild the cooccur field only PERIODICALLY. Rebuilding
		// it every tick is O(corpus) and, as DNA grows the corpus, collapses tick
		// throughput until the ontogenesis check never fires and the colony stalls
		// at child (observed 2026-06-03 GPU run: debug-onto count 0 in 24 min).
		// The monotonic growth clock (corpusIngestedTotal) still advances every
		// tick on consume; a field a few ticks stale is fine for generation.
		if docs == nil || tickCount-lastFieldRebuild >= 30 {
			updateReservoirCorpus(db, CFG.CorpusPath, CFG.MaxCorpusLines)
			docs = loadCorpusLines(CFG.CorpusPath)
			if len(docs) > 0 {
				field.BuildFromCorpus(tok, docs)
				model.mu.Lock()
				model.corpusField = field // share with generation for adaptive blend
				model.mu.Unlock()
			}
			lastFieldRebuild = tickCount
		}

		// Tokenizer evolution
		bpeEnabled := tok.MaybeEnableBPE(docs)
		bpeRetrained := tok.MaybeRetrainBPE(docs)
		if bpeEnabled || bpeRetrained {
			model.mu.Lock()
			model.MaybeExpandVocab(tok.VocabSize)
			SaveCheckpoint(model, tok, "")
			model.mu.Unlock()
		}

		// Per-stage warmup: if model grew since last warmup, train before continuing
		currentStage := model.CurrentGrowthStage()
		if currentStage > model.lastWarmupStage && len(docs) > 0 {
			// Optional warmup coordination through training queue (for Mac 8GB)
			warmupLocked := false
			if CFG.CoordinateWarmup && swarm != nil {
				for !swarm.AcquireTrainingLock() {
					time.Sleep(5 * time.Second)
				}
				warmupLocked = true
			}

			embryoEmbd := CFG.GrowthStages[0][1]
			warmupScale := int(math.Ceil(math.Sqrt(float64(model.NEmbd) / float64(embryoEmbd))))
			if warmupScale < 1 {
				warmupScale = 1
			}
			effectiveWarmup := CFG.WarmupSteps * warmupScale
			backpropSteps := effectiveWarmup // 100% backprop, notorch warmup disabled (was 0.6)
			notorchDeltaSteps := 0           // disabled: notorch warmup diverges at stage 5
			fmt.Printf("[trainer] warmup for stage %d (embd=%d) — %d steps total (%d backprop + %d notorch, sqrt-scaled %dx)\n",
				currentStage, model.NEmbd, effectiveWarmup, backpropSteps, notorchDeltaSteps, warmupScale)
			// Phase A: backprop with progressive sequence length (short→full)
			earlySteps := int(float64(backpropSteps) * 0.4)
			midSteps := int(float64(backpropSteps) * 0.3)
			lateSteps := backpropSteps - earlySteps - midSteps
			ntWarmupTrain(model, tok, docs, earlySteps, 8) // very short seqs, batch=1
			ntWarmupTrain(model, tok, docs, midSteps, 16)  // short seqs, batch=1
			ntWarmupTrain(model, tok, docs, lateSteps, 32) // medium seqs, batch=1
			// Phase B: notorch for delta adapters (40%, no autograd = much faster)
			// notorchTrainSteps DISABLED in warmup — diverges at stage 5 (loss 3.5→116)
			// notorchTrainSteps(model, tok, docs, notorchDeltaSteps, CFG.NotorchLR)
			model.mu.Lock()
			// A warmup cut short by a shutdown is not a warmup done: leaving
			// lastWarmupStage behind means the next session resumes it instead of
			// walking into the stage untrained.
			if !trainAborting() {
				model.lastWarmupStage = currentStage
			}
			SaveCheckpoint(model, tok, "")
			model.mu.Unlock()

			if warmupLocked && swarm != nil {
				swarm.ReleaseTrainingLock()
			}
			dbLogGrowth(db, model, tok, docs, 0.0, fmt.Sprintf("warmup_stage_%d", currentStage))
			fmt.Printf("[trainer] warmup complete at stage %d. base may freeze now, like a proud fossil.\n", currentStage)
			releaseGrowth() // the peak is over — the next sibling may grow
		} else {
			// Nothing to warm up (no corpus, or the stage is already warmed): the
			// growth lock must not outlive the event it was taken for.
			releaseGrowth()
		}

		if model.lastWarmupStage >= 0 && qbuf.ShouldTrigger() && len(docs) > 0 {
			// Training queue: acquire lock before micro-burst (swarm coordination)
			// Cooperative training-lock serialization is for memory-constrained
			// nodes (Mac 8GB) — gated on CoordinateWarmup. On GPU (handles 4
			// concurrent orgs, 99% util observed) it must be OFF: the lock's
			// `continue` skipped the WHOLE tick (DNA exchange + ontogenesis clock,
			// not just the burst), freezing 3 of 4 orgs while one held the lock
			// (2026-06-03). With CoordinateWarmup=false all orgs train in parallel.
			if swarm != nil && CFG.CoordinateWarmup && !swarm.AcquireTrainingLock() {
				continue // someone else is training, skip this tick
			}

			snapBytes, snapNovelty := qbuf.SnapshotStats()
			fmt.Printf("[trainer] micro-train burst (%d bytes, novelty %.2f) — and lo, it feeds again.\n",
				snapBytes, snapNovelty)

			// SYNTROPY: measure before burst
			// And lo, the organism peers into its own entropic mirror before taking a step.
			model.mu.Lock()
			preMetrics := syntracker.Measure(model, tok, field, docs)
			entropyBefore := preMetrics.Entropy

			// SYNTROPY: decide how to learn (mathematical self-reasoning)
			decision := syntracker.DecideAction()
			lrMul := decision.LRMultiplier
			action := decision.Action
			fmt.Printf("[syntropy] action=%s | trend=%.4f | field_dev=%.3f | purpose_align=%.3f | lr_mul=%.2f\n",
				action, syntracker.SyntropyTrend, syntracker.FieldDeviation,
				syntracker.PurposeAlignment, lrMul)

			// Edit 1 (2026-06-03): at adult stage, log the overload-gate inputs so
			// the mitosis decision is observable — reached-adult-but-no-divide must
			// be a measured fact, never an unexplained negative.
			if syntracker.ModelStage >= len(CFG.GrowthStages)-1 {
				fmt.Printf("[overload] %s | action=%s\n", syntracker.OverloadDebug(), action)
			}

			// IMMUNE SYSTEM: snapshot before burst
			preDirection, preMag := model.GammaContrastiveProjection()
			deltaSnap := model.SnapshotDeltas()

			// Update temperature bridge (under model.mu — fix race)
			model.syntropyTempOff = decision.TempOffset

			// Measure loss before burst (under model.mu — fix race on lastHidden/layerInputs)
			lossBefore := model.QuickLoss(tok, docs, 4)
			model.mu.Unlock()

			// Apply syntropy-adjusted learning rate for notorch (local var, not mutating CFG)
			burstLR := CFG.NotorchLR * lrMul

			// notorch: gradient-free delta training (no backward pass, no compute graph)
			ntBurstTrain(model, tok, docs, CFG.MicroSteps, burstLR)

			model.mu.Lock()
			// Measure loss after burst
			lossAfter := model.QuickLoss(tok, docs, 4)

			// SELF-META-LEARNING: record what this burst did
			syntracker.RecordBurst(action, lossBefore, lossAfter)

			// IMMUNE SYSTEM: check drift after burst
			driftCos := model.GammaDriftCheck(preDirection, preMag)
			if driftCos < CFG.NoiseDriftThreshold {
				fmt.Printf("[immune] NOISE DETECTED (drift cosine=%.3f). Rolling back deltas.\n", driftCos)
				model.RestoreDeltas(deltaSnap)
				dbLogGrowth(db, model, tok, docs, 0.0, "noise_rejected")
				syntracker.LogToDB(db, entropyBefore, entropyBefore, "noise_rejected")
			} else {
				// SYNTROPY: measure after burst
				postMetrics := syntracker.Measure(model, tok, field, docs)
				entropyAfter := postMetrics.Entropy
				syntracker.LogToDB(db, entropyBefore, entropyAfter, action)
				SaveCheckpoint(model, tok, "")
				note := fmt.Sprintf("quantum_burst:%s|Δloss=%.4f", action, lossAfter-lossBefore)
				dbLogGrowth(db, model, tok, docs, 0.0, note)
			}
			model.mu.Unlock()

			// Training queue: release lock after burst completes
			if swarm != nil {
				swarm.ReleaseTrainingLock()
			}

			qbuf.Reset()

			// Delta module growth — influenced by syntropy
			// And lo, new souls are born when the arrow points true.
			growProb := CFG.DeltaGrowProb
			if decision.DeltaGrowOverride != nil {
				growProb = *decision.DeltaGrowOverride
			}
			if len(model.Deltas) < CFG.MaxDeltaModules && rand.Float64() < growProb {
				fmt.Printf("[trainer] growing new delta module (total: %d) — new soul appended.\n", len(model.Deltas)+1)
				model.mu.Lock()
				model.AddDeltaModule(1.0)
				SaveCheckpoint(model, tok, "")
				model.mu.Unlock()
			}

			// Ecology: mitosis / hibernation
			if swarm != nil && action == "divide" {
				// (repair 4) byte gate: the child costs the parent's own peak RSS;
				// the machine keeps MitosisMinFreeMB on top of that or no division.
				if open, freeMB, needMB := mitosisMemGateOpen(int64(CFG.MitosisMinFreeMB)); !open {
					fmt.Printf("[ecology] MITOSIS refused — %d MB available, a child needs %d MB (own peak RSS + %d MB floor)\n",
						freeMB, needMB, CFG.MitosisMinFreeMB)
				} else if !swarm.AcquireMitosisSlot(CFG.MaxOrganisms) {
					// (audit C3) atomic admit: serialized + count-capped across processes
					fmt.Printf("[ecology] MITOSIS refused — colony at cap (%d) or a sibling is dividing\n", CFG.MaxOrganisms)
				} else {
					fmt.Println("[ecology] MITOSIS triggered — organism overloaded, spawning child")
					model.mu.Lock()
					_, mErr := performMitosis(model, tok, db, swarm, syntracker)
					model.mu.Unlock()
					if mErr != nil {
						swarm.ReleaseMitosisLock() // spawn failed — free the slot now (success holds it to expiry)
					}
				}
			}

			if swarm != nil && action == "hibernate" {
				model.mu.Lock()
				performHibernation(model, tok, db, swarm)
				model.mu.Unlock()
				fmt.Println("[ecology] Organism hibernating. Goodbye.")
				return // exit training loop
			}
		}

		// DNA exchange: every tick = every breath. Organism exhales DNA, inhales others'.
		if element != "" {
			// Write: generate and share with ecology
			dnaWrite(element, model, tok, field, docs, tickCount)
			// Read: consume other organisms' output → corpus grows → ontogenesis unlocks
			if consumed := dnaRead(element, CFG.CorpusPath, qbuf, tok, dnaCur); consumed > 0 {
				// Monotonic growth clock — every byte ever ingested counts.
				model.mu.Lock()
				model.corpusIngestedTotal += consumed
				model.mu.Unlock()
				// Corpus reload + field rebuild are throttled at the loop top
				// (every 30 ticks) so consume stays O(1) and ticks stay fast —
				// this is what lets the ontogenesis clock actually advance.
			}
		}

		// Ontogenesis: check if architecture should grow (every 10 ticks — corpus grows via DNA)
		if tickCount%10 == 0 {
			// corpus = bounded reservoir file size (saturates); ingested =
			// the monotonic clock MaybeGrowArchitecture actually gates on.
			corpusChars := 0
			if fi, err := os.Stat(CFG.CorpusPath); err == nil {
				corpusChars = int(fi.Size())
			}
			model.mu.Lock()
			fmt.Printf("[debug-onto] tick=%d corpus=%d ingested=%d stage=%d freeze=%d\n", tickCount, corpusChars, model.corpusIngestedTotal, model.CurrentGrowthStage(), model.growthFreezeRemaining)
			// (repair 9) Two gates stand before ontogenesis, and a closed one only
			// defers: nothing about the decision is written down, so the next tick
			// that finds the memory (or the lock) grows. First the byte gate — a
			// stage step multiplies this process's peak, and four of them at once
			// took the phone's userspace down on 2026-09-13. Then the colony lock,
			// held through the warmup, so the four peaks do not coincide.
			if model.GrowthWanted() {
				switch {
				case !growthMemGateCheck():
					// growthMemGateCheck printed the deferral
				case CFG.CoordinateGrowth && swarm != nil && !swarm.AcquireGrowthLock():
					fmt.Println("[growth] deferred: a sibling is growing (colony growth lock held)")
				default:
					if CFG.CoordinateGrowth && swarm != nil {
						growthLockHeld = true
					}
					if model.MaybeGrowArchitecture() {
						ntOnGrowth()                                 // reset the notorch tape — Net2Net changed dims (06_PLAN S1)
						ntSetGPUForStage(model.CurrentGrowthStage()) // flip CPU→GPU at teen/adult
						SaveCheckpoint(model, tok, "")
						nP := 0
						for _, m := range model.Base {
							nP += m.Nout * m.Nin
						}
						dbLogGrowth(db, model, tok, docs, 0.0,
							fmt.Sprintf("ontogenesis:stage=%d|params=%d", model.CurrentGrowthStage(), nP))
					} else {
						releaseGrowth() // nothing grew after all; do not hold the colony
					}
				}
			}
			model.mu.Unlock()
		}

		// Swarm heartbeat (every 10 ticks)
		if swarm != nil && tickCount%10 == 0 {
			model.mu.Lock()
			stage := model.CurrentGrowthStage()
			nP := 0
			for _, m := range model.Base {
				nP += m.Nout * m.Nin
			}
			gs := model.globalStep
			// The voice, read under the same lock dnaWrite reads it under
			// (routing repair 6). fade = 1 means the overlay is gone and the
			// speech is the transformer's own.
			genMag, genFade := model.lastGenMag, 1-model.lastOverlayWeight
			model.mu.Unlock()
			lastEntropy := 0.0
			if len(syntracker.EntropyHistory) > 0 {
				lastEntropy = syntracker.EntropyHistory[len(syntracker.EntropyHistory)-1]
			}
			swarm.Heartbeat(stage, nP, syntracker.SyntropyTrend, lastEntropy, gs, genMag, genFade)
			// Update swarm info for hibernate decisions
			peers := swarm.DiscoverPeers(60)
			syntracker.SwarmInfo = &SwarmPeerInfo{Peers: peers}
		}

		// Jitter de-phases sibling processes so they do not scan the DNA tree
		// in lockstep (repair 3).
		time.Sleep(time.Duration((CFG.TrainTickSeconds + rand.Float64()*CFG.TickJitterSeconds) * float64(time.Second)))
	}
}

// ============================================================
// 10) CHAT LOOP — tiny memory, tiny ego, continuous learning
// ============================================================

func buildPromptFromMemory(db *sql.DB, userText string) string {
	recent := dbRecentMessages(db, 14)

	clip := func(s string, n int) string {
		s = normalizeText(s)
		if len(s) > n {
			s = s[:n]
		}
		return strings.TrimSpace(s)
	}

	var parts []string
	parts = append(parts, "A: (I listen. I answer. I learn.)")

	limit := 12
	start := 0
	if len(recent) > limit {
		start = len(recent) - limit
	}
	for _, msg := range recent[start:] {
		tag := "A:"
		if msg.Role == "user" {
			tag = "H:"
		}
		parts = append(parts, fmt.Sprintf("%s %s", tag, clip(msg.Text, 260)))
	}

	parts = append(parts, fmt.Sprintf("H: %s", clip(userText, 260)))
	parts = append(parts, "A:")
	return strings.Join(parts, "\n")
}

// ============================================================
// 11) AWAKEN — now, when all is assembled as an organism,
//              it is time to declare the final function.
// ============================================================

// effectiveCPUs returns the cores actually available to THIS process — the cgroup
// CPU quota when limited (RunPod/containers report the HOST's NumCPU but cap us via
// cgroup), else runtime.NumCPU(). Reading the host nproc made openblas spawn one
// thread per HOST core (e.g. 96); N colony processes then oversubscribed the few
// real cgroup cores ~40x and the GPU-feeding threads starved into a hard stall.
func effectiveCPUs() int {
	n := runtime.NumCPU()
	if b, err := os.ReadFile("/sys/fs/cgroup/cpu.max"); err == nil { // cgroup v2
		f := strings.Fields(string(b))
		if len(f) == 2 && f[0] != "max" {
			if q, e1 := strconv.Atoi(f[0]); e1 == nil {
				if p, e2 := strconv.Atoi(f[1]); e2 == nil && q > 0 && p > 0 {
					if c := q / p; c >= 1 && c < n {
						n = c
					}
				}
			}
		}
	}
	if qb, err := os.ReadFile("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"); err == nil { // cgroup v1
		if q, e1 := strconv.Atoi(strings.TrimSpace(string(qb))); e1 == nil && q > 0 {
			if pb, e2 := os.ReadFile("/sys/fs/cgroup/cpu/cpu.cfs_period_us"); e2 == nil {
				if p, e3 := strconv.Atoi(strings.TrimSpace(string(pb))); e3 == nil && p > 0 {
					if c := q / p; c >= 1 && c < n {
						n = c
					}
				}
			}
		}
	}
	return n
}

// colonyThreadsFor caps BLAS threads per organism so the base 4-element colony
// shares the cores without oversubscription (the cause of the multi-process GPU
// stall): cores/4, floored at 1.
func colonyThreadsFor(cores int) int {
	t := cores / 4
	if t < 1 {
		t = 1
	}
	return t
}

// capColonyThreads pins per-process BLAS / Go threads to the cgroup-aware core share
// BEFORE any BLAS init, so N concurrent organisms don't thrash the CPU and stall the
// GPU. Respects an explicit OPENBLAS_NUM_THREADS override (e.g. from a launcher).
func capColonyThreads() {
	cores := effectiveCPUs()
	runtime.GOMAXPROCS(cores)
	if os.Getenv("OPENBLAS_NUM_THREADS") != "" {
		return
	}
	per := strconv.Itoa(colonyThreadsFor(cores))
	os.Setenv("OPENBLAS_NUM_THREADS", per)
	os.Setenv("OMP_NUM_THREADS", per)
	fmt.Fprintf(os.Stderr, "[cpu] effective cores=%d → OPENBLAS_NUM_THREADS=%s (colony oversubscription guard)\n", cores, per)
}

func main() {
	capColonyThreads() // cgroup-aware thread cap — prevents the multi-process GPU stall; MUST run before any BLAS/cgo init
	// (repair 9) Say what this process is worth to the low-memory killer before
	// anything else allocates. Organisms and the witness both pass here; under
	// Magisk su they would otherwise inherit -1000 and the phone would kill the
	// terminal instead of the colony (2026-09-13).
	if adj, err := applyOomScoreAdj(oomScoreAdjPath, CFG.OomScoreAdj); err != nil {
		fmt.Fprintf(os.Stderr, "[oom] could not set oom_score_adj=%d: %v\n", CFG.OomScoreAdj, err)
	} else if adj != "" {
		fmt.Fprintf(os.Stderr, "[oom] oom_score_adj=%s (lmkd reaches for the organism before the terminal)\n", adj)
	}
	rand.Seed(42) // And lo, determinism shall pretend to tame chaos.

	// Parse CLI args for child organisms
	organismID, configPath, element, evolution := parseCLIArgs()

	// Witness mode: no model, no training, no GPU — read and say (repair 7).
	if witnessMode {
		os.Exit(runWitness(witnessInterval, witnessOnce))
	}

	// World-ingest mode: no model either. The one process in the tree that
	// writes world_facts (ROADMAP 10, world_ledger.go), run by senses.sh at
	// the end of a pass and by the scheduler on its own.
	if worldIngestMode {
		os.Exit(runWorldIngest(witnessInterval, witnessOnce))
	}

	// (repair 10) Arm the shutdown before anything trains. The handler used to
	// be installed only once the evolution loop was reached, which is after the
	// bootstrap climb: a fresh embryo SIGTERM'd during its first warmup — the
	// longest unattended stretch a capped colony session has — died with an
	// empty CkptPath and the whole climb was repeated on the next launch.
	// Evolution mode only: in the REPL, Ctrl+C must still end the process the
	// way it always has, and the bootstrap there pauses for the user anyway.
	shutdown := make(chan struct{})
	if evolution {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		go func() {
			<-sigCh
			// The step loops read this every step; closing `shutdown` alone
			// would wait for a warmup to finish holding model.mu.
			trainAbort.Store(true)
			close(shutdown)
		}()
	}

	// GPU init: attempted only when --gpu (CFG.UseGPU) requested. Silent
	// fallback if init fails — gpuReady() stays false and the Matvec
	// dispatcher continues to use the CPU/BLAS path. Linux only at runtime
	// (the stub on other platforms returns -1 immediately).
	if CFG.UseGPU {
		if rc := gpuInit(); rc != 0 || !gpuReady() {
			fmt.Fprintf(os.Stderr, "[gpu] init failed (rc=%d); falling back to CPU/BLAS\n", rc)
			CFG.UseGPU = false
		} else {
			fmt.Fprintln(os.Stderr, "[gpu] CUDA backend live — inference matvec routed through cuBLAS")
		}
	}

	// notorch trainer GPU (06_PLAN §8): gpu_init() at startup — on success
	// nt_set_gpu_mode(1) routes the training tape's matvecs through cuBLAS;
	// on failure the trainer stays on CPU/BLAS. Automatic, no flag. The real
	// bodies are in gpu_notorch_cuda.go (built with -tags cuda); the !cuda
	// stub keeps the default CPU build a no-op.
	if _, msg := ntGPUEnable(); msg != "" {
		fmt.Fprintln(os.Stderr, "[notorch] "+msg)
	}

	// Element → corpus path: each element eats its own food
	if element != "" {
		switch element {
		case "earth":
			CFG.CorpusPath = "nonames_earth.txt"
		case "air":
			CFG.CorpusPath = "nonames_air.txt"
		case "water":
			CFG.CorpusPath = "nonames_water.txt"
		case "fire":
			CFG.CorpusPath = "nonames_fire.txt"
		default:
			fmt.Fprintf(os.Stderr, "unknown element: %s (use earth/air/water/fire)\n", element)
			os.Exit(1)
		}
		fmt.Printf("[ecology] Element: %s → corpus: %s\n", element, CFG.CorpusPath)
	}

	if evolution {
		fmt.Println("[evolution] Autonomous evolution mode — organism will grow through all stages without pause.")
	}

	// The two declared ceilings, said out loud at startup: a colony on a phone
	// is bounded by choice, not by whatever the pod defaults were (repair 9).
	if len(CFG.GrowthStages) > 0 {
		CFG.MaxGrowthStage = maxGrowthStage()
		cap := CFG.GrowthStages[CFG.MaxGrowthStage]
		fmt.Printf("[caps] colony ≤ %d organisms | growth ≤ stage %d of %d (embd=%d, layer=%d, head=%d)\n",
			CFG.MaxOrganisms, CFG.MaxGrowthStage, len(CFG.GrowthStages)-1, cap[1], cap[2], cap[3])
	}

	// Child organism: load birth config from parent
	var syntrackerSeed []BurstRecord
	if configPath != "" {
		if data, err := os.ReadFile(configPath); err == nil {
			var birth map[string]interface{}
			if json.Unmarshal(data, &birth) == nil {
				if cp, ok := birth["corpus_path"].(string); ok && cp != "" {
					CFG.CorpusPath = cp
				}
				if dp, ok := birth["db_path"].(string); ok && dp != "" {
					CFG.DBPath = dp
				}
				if ckp, ok := birth["ckpt_path"].(string); ok && ckp != "" {
					CFG.CkptPath = ckp
				}
				// Parse burst_history
				if bh, ok := birth["burst_history"].([]interface{}); ok {
					for _, item := range bh {
						if rec, ok := item.(map[string]interface{}); ok {
							br := BurstRecord{}
							if a, ok := rec["Action"].(string); ok {
								br.Action = a
							}
							if lb, ok := rec["LossBefore"].(float64); ok {
								br.LossBefore = lb
							}
							if la, ok := rec["LossAfter"].(float64); ok {
								br.LossAfter = la
							}
							syntrackerSeed = append(syntrackerSeed, br)
						}
					}
					if len(syntrackerSeed) > 0 {
						fmt.Printf("[ecology] Inherited %d burst records from parent.\n", len(syntrackerSeed))
					}
				}
			}
		}
	}

	db, err := initDB(CFG.DBPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	// Seed corpus
	if _, err := os.Stat(CFG.CorpusPath); os.IsNotExist(err) {
		saveCorpusLines(CFG.CorpusPath, []string{"Hello.", "I exist.", "Speak."})
	}

	docs := loadCorpusLines(CFG.CorpusPath)

	// Restore model dimensions from checkpoint config (ontogenesis may have changed them).
	// Zero-warmup test mode (--zero-warmup, CFG.WarmupSteps==0) skips checkpoint load
	// so the test always exercises a fresh embryo; otherwise the test silently uses a
	// stale trained checkpoint and the Q-style coherence claim becomes meaningless.
	var model *GPT
	var tok *EvolvingTokenizer
	if CFG.WarmupSteps > 0 {
		model, tok, err = LoadCheckpoint(docs, "")
	} else {
		err = fmt.Errorf("zero-warmup mode: skipping checkpoint load")
	}
	if err != nil || model == nil || tok == nil {
		if len(docs) == 0 {
			docs = []string{"Hello."}
		}
		tok = NewEvolvingTokenizer(docs)

		// Enable BPE BEFORE training — subword tokens make corpus field coherent
		// (byte-level trigrams produce babble; subword trigrams produce speech)
		tok.MaybeEnableBPE(docs)

		model = NewGPT(tok)

		// Per-stage warmup: train at each stage before growing.
		// Corpus size determines ceiling (which stages are reachable), not starting point.
		// The organism always starts as embryo and grows through training.
		// Seed the monotonic growth clock from the starting corpus — the
		// organism's initial text mass counts as ingested; from here it
		// only grows (dnaRead accumulates into it).
		for _, d := range docs {
			model.corpusIngestedTotal += len(d)
		}
		stageNames := []string{"embryo", "infant", "child", "adolescent", "teen", "adult"}

		// Build corpus field — active from first token, sigmoid fade weakens it as model learns
		tmpCooccur := NewCooccurField()
		tmpCooccur.BuildFromCorpus(tok, docs)
		model.corpusField = tmpCooccur

		// Seed embeddings from metaweights (postgpt's «tokenizer IS training»
		// trick, postgpt.c:541-574). Biases wte by Hebbian co-occurrence and
		// lm_head by unigram × wte BEFORE any warmup training. Gives the
		// untrained organism corpus-shaped embeddings → coherent first words.
		// scale=0.15 verbatim from postgpt.c:542. Gated on CFG.CorpusLogitOverlay
		// so default-off path stays identical to main branch behaviour.
		if CFG.CorpusLogitOverlay {
			SeedEmbeddingsFromMetaweights(model, tmpCooccur, 0.15)
		}

		// Detect if stdin is a terminal (interactive mode)
		isInteractive := false
		if fi, err := os.Stdin.Stat(); err == nil {
			isInteractive = (fi.Mode() & os.ModeCharDevice) != 0
		}

		stageProbes := []string{"Hello.", "Who are you?", "What do you know?"}
		initScanner := bufio.NewScanner(os.Stdin)

		for {
			stage := model.CurrentGrowthStage()
			stageName := "unknown"
			if stage >= 0 && stage < len(stageNames) {
				stageName = stageNames[stage]
			}
			// Train warmup at current stage (sqrt scaling + split warmup)
			embryoEmbd := CFG.GrowthStages[0][1]
			warmupScale := int(math.Ceil(math.Sqrt(float64(model.NEmbd) / float64(embryoEmbd))))
			if warmupScale < 1 {
				warmupScale = 1
			}
			effectiveWarmup := CFG.WarmupSteps * warmupScale
			if effectiveWarmup > 0 {
				backpropSteps := effectiveWarmup // 100% backprop, notorch warmup disabled (was 0.6)
				notorchDeltaSteps := 0           // disabled: notorch warmup diverges at stage 5
				fmt.Printf("[init] Stage %d (%s): embd=%d, layer=%d, head=%d — warmup %d steps (%d backprop + %d notorch, sqrt-scaled %dx)\n",
					stage, stageName, model.NEmbd, model.NLayer, model.NHead, effectiveWarmup, backpropSteps, notorchDeltaSteps, warmupScale)
				// Phase A: backprop with progressive sequence length (short→full)
				earlySteps := int(float64(backpropSteps) * 0.4)
				midSteps := int(float64(backpropSteps) * 0.3)
				lateSteps := backpropSteps - earlySteps - midSteps
				ntWarmupTrain(model, tok, docs, earlySteps, 8) // very short seqs, batch=1
				ntWarmupTrain(model, tok, docs, midSteps, 16)  // short seqs, batch=1
				ntWarmupTrain(model, tok, docs, lateSteps, 32) // medium seqs, batch=1
				// (repair 10) A bootstrap warmup cut short by a signal is not a
				// warmup done — same rule as the tick loop's warmup: leaving
				// lastWarmupStage behind would make the next launch walk into
				// the stage untrained. The steps taken so far are real (the
				// tape mirrors them back every step), so they are written out
				// on the explicit path, which the debouncer cannot drop.
				if trainAborting() {
					saveOnShutdown("init", model, tok, "signal")
					return
				}
				model.lastWarmupStage = stage
				SaveCheckpoint(model, tok, "")
			} else {
				fmt.Printf("[init] Stage %d (%s): embd=%d, layer=%d, head=%d — zero-warmup mode, skipping all gradient steps\n",
					stage, stageName, model.NEmbd, model.NLayer, model.NHead)
				// Do NOT update lastWarmupStage or call SaveCheckpoint here:
				// pollluting the checkpoint with a zero-step «warmed» marker
				// would make every future normal launch from this dir skip its
				// embryo warmup. The test must leave on-disk state untouched.
			}

			// Demo: show what the organism can say at this stage
			// Use model+corpus blend (same as normal REPL) so corpus field helps early stages speak
			fmt.Printf("\n[stage %d — %s] What it sounds like now:\n", stage, stageName)
			for _, probe := range stageProbes {
				answer := GenerateResonant(model, tok, tmpCooccur, probe, docs, true)
				if answer == "" {
					answer = "..."
				}
				fmt.Printf("  Q: %s\n  A: %s\n", probe, answer)
			}
			fmt.Println()

			// Zero-warmup test mode: stop after embryo voice — no ontogenesis,
			// no further training. Pure Q-style untrained-coherence check.
			if CFG.WarmupSteps == 0 {
				break
			}
			// Try to grow to next stage (gated by corpus size)
			// (repair 9) The bootstrap climb answers to the same byte gate as the
			// tick loop. Breaking out is a deferral, not a refusal: the trainer
			// loop retries the stage every ten ticks once the memory is there.
			if model.GrowthWanted() && !growthMemGateCheck() {
				break
			}
			if !model.MaybeGrowArchitecture() {
				break // corpus too small for next stage, or already at max
			}
			ntOnGrowth()                    // reset the notorch tape — Net2Net changed dims (06_PLAN S1)
			model.growthFreezeRemaining = 0 // skip freeze during init — we're about to warmup anyway

			// Rebuild corpus field after growth (vocab may have expanded)
			tmpCooccur.BuildFromCorpus(tok, docs)
			model.corpusField = tmpCooccur

			// Interactive mode: pause between stages, let user chat or type /grow
			// --evolution skips pause — organism grows autonomously
			if isInteractive && !evolution {
				fmt.Printf("[init] Stage %d complete. Chat with the organism, or type /grow to continue growth.\n", stage)
				for {
					fmt.Print("> ")
					if !initScanner.Scan() {
						break
					}
					line := strings.TrimSpace(initScanner.Text())
					if line == "/grow" || line == "" {
						break
					}
					answer := GenerateResonant(model, tok, tmpCooccur, line, docs, true)
					if answer == "" {
						answer = "..."
					}
					fmt.Println(answer)
				}
			}
		}
		fmt.Printf("[init] Warmup complete at stage %d. Organism ready.\n", model.CurrentGrowthStage())

		// Zero-warmup test exits here — do not enter ecology / REPL / shutdown
		// SaveCheckpoint paths that would persist a zero-step «trained» marker.
		if CFG.WarmupSteps == 0 {
			fmt.Println("[init] Zero-warmup test complete — exit before REPL/ecology to preserve on-disk state.")
			return
		}
	}

	// Enable BPE in main before REPL starts (avoid race with background trainer)
	tok.MaybeEnableBPE(docs)
	model.MaybeExpandVocab(tok.VocabSize)

	// Cross-organism graze field — Dario-style logit injection from sibling
	// emissions, mirrored to ../dna/seen/<sibling>/ by dnaRead (commit
	// e5c1685). Active only when --cross-graze AND --element are set; the
	// hook in GenerateResonant is a no-op when crossField is nil.
	if CFG.CrossGraze && element != "" {
		model.crossField = NewCrossField(element, "../dna/output") // readers no longer mirror to seen/ (repair 3)
		fmt.Fprintf(os.Stderr, "[graze] %s cross-organism injection enabled (coef=%.2f topN=%d)\n",
			element, CFG.CrossGrazeCoef, CFG.CrossGrazeTopN)
	}

	// Swarm ecology: register in mesh
	swarm := NewSwarmRegistry(organismID, element)
	if err := swarm.Register(); err != nil {
		fmt.Printf("[ecology] Warning: swarm registration failed: %v\n", err)
	}
	peers := swarm.DiscoverPeers(60)
	if len(peers) > 0 {
		fmt.Printf("[ecology] Joined swarm. %d peer(s) detected.\n", len(peers))
	} else {
		fmt.Println("[ecology] First organism in the swarm.")
	}

	// Child: inject inherited burst_history via model attribute
	if len(syntrackerSeed) > 0 {
		model.inheritedBurstHistory = syntrackerSeed
	}

	// Build corpus field for pre-training speech
	cooccur := NewCooccurField()
	cooccur.BuildFromCorpus(tok, docs)

	// Quantum buffer for smart training triggers
	qbuf := NewQuantumBuffer()

	// Start background trainer. done closes when its loop returns — on
	// hibernation — so evolution mode ends the process with it (repair 4).
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		backgroundTrainer(db, model, tok, qbuf, swarm, stop, element)
	}()
	// Heartbeat keeper: the tick loop reports state every 10 ticks, the keeper
	// repeats it every 20 s (live window is 60 s, AcquireMitosisSlot) across
	// the inline post-growth warmups that block the loop for minutes. Seeded
	// now with the current state: the first ten ticks carry bursts and DNA
	// generation and took 144 s on the old binary (run7_old, earth), so a
	// keeper that waits for the tick loop's first report is silent exactly
	// when the new organism drops out of the live count.
	swarm.StartKeeper(stop, 20*time.Second)
	model.mu.Lock()
	seedStage, seedParams, seedStep := model.CurrentGrowthStage(), 0, model.globalStep
	for _, m := range model.Base {
		seedParams += m.Nout * m.Nin
	}
	seedMag, seedFade := model.lastGenMag, 1-model.lastOverlayWeight
	model.mu.Unlock()
	swarm.Heartbeat(seedStage, seedParams, 0, 0, seedStep, seedMag, seedFade)

	if evolution {
		fmt.Println("molequla is alive. [evolution] Autonomous mode — background trainer running. Ctrl+C to stop.")
		// In evolution mode: no REPL; run until a signal or until the trainer
		// loop ends on its own (hibernation), which frees this organism's
		// memory to the colony instead of parking it forever. The signal was
		// already armed before the bootstrap climb (repair 10); `shutdown` is
		// closed by that handler, so a signal that arrived during the climb is
		// still here waiting.
		why := waitEvolution(shutdown, done, stop)
		fmt.Printf("\n[evolution] Organism shutting down gracefully (%s).\n", why)
		// The signal closed `stop`, but a tick loop inside a post-growth warmup
		// answers it only minutes later. Give it a short, bounded chance to reach
		// a tick boundary, then write the checkpoint regardless: on 2026-09-13
		// stop.sh landed 28 minutes of warmup and every ckpt on disk still
		// carried its growth-time mtime, because nothing saved on the way out.
		if why == "signal" {
			select {
			case <-done:
			case <-time.After(shutdownDrain):
			}
		}
		saveOnShutdown("evolution", model, tok, why)
		return
	}

	fmt.Println("molequla is alive. Type and press Enter. Ctrl+C to exit.")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		userText := strings.TrimSpace(scanner.Text())
		if userText == "" {
			continue
		}

		dbAddMessage(db, "user", userText)
		updateReservoirCorpus(db, CFG.CorpusPath, CFG.MaxCorpusLines)

		// Feed quantum buffer
		qbuf.Feed(userText, tok)

		// Rebuild cooccur field with updated corpus
		freshDocs := loadCorpusLines(CFG.CorpusPath)
		if len(freshDocs) > 0 {
			cooccur.BuildFromCorpus(tok, freshDocs)
			model.mu.Lock()
			model.corpusField = cooccur // sync REPL cooccur with model.corpusField
			model.mu.Unlock()
		}

		// Self-enrichment: user input enriches corpus field (AFTER rebuild, so it's not wiped)
		userIDs := tok.Encode(userText)
		cooccur.IngestTokens(userIDs)

		// Active user word boost: organism absorbs user's vocabulary (Leo-style)
		// Decays each generation, fades with model strength via sigmoid in generation
		cooccur.AbsorbUserWords(userIDs)

		prompt := buildPromptFromMemory(db, userText)

		// Consciousness: self-prediction error (Feature 4)
		// "How surprised am I by this input?"
		model.mu.Lock()
		gradEnabled.Store(false)
		promptIDs := tok.Encode(prompt)
		if len(promptIDs) > 2 {
			surprise := model.ComputeSelfPredictionError(promptIDs)
			model.lastSurprise = surprise
			if model.surpriseBaseline < 1e-6 {
				model.surpriseBaseline = surprise
			} else {
				model.surpriseBaseline = 0.3*surprise + 0.7*model.surpriseBaseline
			}
		}
		gradEnabled.Store(true)
		model.mu.Unlock()

		// Generation: per-token sigmoid fade is computed inside GenerateResonant
		answer := GenerateResonant(model, tok, cooccur, prompt, freshDocs, true)
		if answer == "" {
			answer = "..."
		}

		// Consciousness: conscience check (Feature 5)
		// "Did my last generation feel coherent?"
		model.mu.Lock()
		if model.lastGenEntropy > 0 {
			model.ConscienceCheck(model.lastGenEntropy)
		}
		model.mu.Unlock()

		fmt.Println(answer)
		dbAddMessage(db, "assistant", answer)

		// Self-enrichment: own output enriches corpus field, weighted by coherence
		// Low entropy = coherent speech = higher weight (Stanley's resonance weighting)
		if len(answer) > 3 {
			selfWeight := 1.0
			model.mu.Lock()
			lastEnt := model.lastGenEntropy
			model.mu.Unlock()
			if lastEnt > 0 {
				selfWeight = 2.0 - lastEnt
				if selfWeight < 0.3 {
					selfWeight = 0.3
				}
				if selfWeight > 2.0 {
					selfWeight = 2.0
				}
			}
			cooccur.IngestTokensWeighted(tok.Encode(answer), selfWeight)
			cooccur.DecayUserBoost()
		}

		// Consciousness: overthinkg rings (Feature 3)
		// "Let me re-read what I just said to strengthen my patterns."
	}

	close(stop)
	model.mu.Lock()
	SaveCheckpoint(model, tok, "")
	model.mu.Unlock()
	swarm.Unregister()
}
