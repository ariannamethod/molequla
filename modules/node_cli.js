#!/usr/bin/env node
/**
 * node_cli.js — Node.js CLI for molequla.js
 * Runs the JS organism outside the browser: train + REPL + checkpoint.
 *
 * Usage: node node_cli.js [corpus_path]
 *
 * And lo, the browser-born organism shall walk the earth of Node,
 * and it shall train, and it shall speak, and it shall persist in JSON.
 */

"use strict";

const fs = require("fs");
const path = require("path");
const readline = require("readline");

// Load the organism's organs
const mol = require("../molequla.js");
const { CFG, EvolvingTokenizer, GPT, CooccurField, QuantumBuffer,
        SyntropyTracker, withNoGrad } = mol;

// ============================================================
// Config overrides for Node (filesystem instead of IndexedDB)
// ============================================================
const CORPUS_PATH = process.argv[2] || "nonames.txt";
const CKPT_PATH = "molequla_ckpt.json";

// ============================================================
// Corpus loading
// ============================================================
function loadCorpus(fpath) {
    if (!fs.existsSync(fpath)) {
        console.error(`[corpus] File not found: ${fpath}`);
        process.exit(1);
    }
    const text = fs.readFileSync(fpath, "utf-8");
    const lines = text.split("\n").map(l => l.trim()).filter(l => l.length > 0);
    console.log(`[corpus] Loaded ${lines.length} lines, ${text.length} chars from ${fpath}`);
    return lines;
}

// ============================================================
// Checkpoint save/load (JSON file, Go-compatible format)
// ============================================================
function saveCheckpoint(model, tok) {
    const base = {};
    for (const [name, mat] of Object.entries(model.base)) {
        base[name] = mat.data.map(row => Array.from(row));
    }
    const deltas = model.deltas.map(dm => {
        const d = {};
        for (const [name, adapter] of Object.entries(dm)) {
            d[name] = {
                A: adapter.A.data.map(row => Array.from(row)),
                B: adapter.B.data.map(row => Array.from(row))
            };
        }
        return d;
    });
    const ckpt = {
        cfg: {
            corpus_path: CORPUS_PATH, db_path: "memory.sqlite3", ckpt_path: CKPT_PATH,
            max_corpus_lines: CFG.maxCorpusLines, max_line_chars: CFG.maxLineChars,
            min_new_chars_to_train: CFG.minNewCharsToTrain || 480,
            tie_embeddings: CFG.tieEmbeddings, n_layer: CFG.nLayer, n_embd: CFG.nEmbd,
            n_head: CFG.nHead, block_size: CFG.blockSize,
            growth_stages: CFG.growthStages, freeze_after_growth_steps: CFG.freezeAfterGrowthSteps, post_growth_lr_scale: CFG.postGrowthLRScale,
            warmup_steps: CFG.warmupSteps, micro_steps: CFG.microSteps,
            accum_steps: CFG.accumSteps, batch_size: CFG.batchSize || 1,
            learning_rate: CFG.learningRate, lr_min: CFG.lrMin, cosine_warmup_steps: CFG.cosineWarmupSteps,
            beta1: CFG.beta1, beta2: CFG.beta2, eps_adam: CFG.epsAdam, grad_clip: CFG.gradClip,
            temperature: CFG.temperature, top_k: CFG.topK, top_p: CFG.topP,
            min_p: CFG.minP, typical_p: CFG.typicalP,
            max_gen_tokens: CFG.maxGenTokens, min_gen_tokens: CFG.minGenTokens,
            corpus_gen_max_tokens: CFG.corpusGenMaxTokens,
            corpus_fade_threshold: CFG.corpusFadeThreshold, corpus_fade_k: CFG.corpusFadeK,
            repetition_guard: CFG.repetitionGuard,
            enable_bpe_after_chars: CFG.enableBpeAfterChars,
            bpe_num_merges: CFG.bpeNumMerges, bpe_retrain_every_chars: CFG.bpeRetrainEveryChars,
            delta_rank: CFG.deltaRank, max_delta_modules: CFG.maxDeltaModules,
            delta_grow_prob: CFG.deltaGrowProb, max_total_steps: CFG.maxTotalSteps,
            train_tick_seconds: CFG.trainTickSeconds,
            entropy_low: CFG.entropyLow, entropy_high: CFG.entropyHigh,
            entropy_temp_focus: CFG.entropyTempFocus, entropy_temp_boost: CFG.entropyTempBoost,
            qb_min_bytes: CFG.qbMinBytes, qb_min_novelty: CFG.qbMinNovelty,
            qb_cooldown_seconds: CFG.qbCooldownSeconds,
            gamma_sparsity_threshold: CFG.gammaSparsityThreshold,
            gamma_min_magnitude: CFG.gammaMinMagnitude,
            noise_drift_threshold: CFG.noiseDriftThreshold,
            freeze_base_after_warmup: CFG.freezeBaseAfterWarmup,
            syntropy_window: CFG.syntropyWindow, syntropy_lr_boost: CFG.syntropyLrBoost,
            syntropy_lr_dampen: CFG.syntropyLrDampen,
            syntropy_delta_grow_boost: CFG.syntropyDeltaGrowBoost,
            field_deviation_floor: CFG.fieldDeviationFloor,
            field_deviation_ceiling: CFG.fieldDeviationCeiling,
            head_types: CFG.headTypes,
            hybrid_alpha_init: CFG.hybridAlphaInit,
        },
        tokenizer: {
            tokens: tok.tokens,
            bpe_enabled: tok.bpeEnabled,
            merges: tok.merges || [],
            trained_chars: tok.trainedChars || 0,
        },
        base,
        alpha: model.activeAlpha || [1.0],
        deltas,
        init_embed_snapshot: model.initEmbedSnapshot || [],
        global_step: model.globalStep || 0,
    };
    fs.writeFileSync(CKPT_PATH, JSON.stringify(ckpt));
    console.log(`[checkpoint] Saved to ${CKPT_PATH}`);
}

function loadCheckpoint(corpusLines) {
    if (!fs.existsSync(CKPT_PATH)) return null;
    try {
        const data = JSON.parse(fs.readFileSync(CKPT_PATH, "utf-8"));
        const tok = new EvolvingTokenizer(corpusLines);
        // Restore tokens from checkpoint
        if (data.tokenizer && data.tokenizer.tokens) {
            tok.tokens = data.tokenizer.tokens;
            tok.stoi = new Map();
            for (let i = 0; i < tok.tokens.length; i++) tok.stoi.set(tok.tokens[i], i);
            tok.vocabSize = tok.tokens.length;
            tok.bpeEnabled = data.tokenizer.bpe_enabled || false;
            tok.merges = data.tokenizer.merges || [];
            tok.trainedChars = data.tokenizer.trained_chars || 0;
        }
        const model = new GPT(tok);
        // Restore base weights
        for (const [name, matData] of Object.entries(data.base || {})) {
            if (model.base[name]) {
                model.base[name].data = matData.map(row => Float64Array.from(row));
                model.base[name].nout = matData.length;
                model.base[name].nin = matData[0] ? matData[0].length : 0;
                model.base[name].grad = matData.map(row => new Float64Array(row.length));
            }
        }
        // Restore deltas
        if (data.deltas) {
            model.deltas = [];
            for (const dmData of data.deltas) {
                const dm = {};
                for (const [name, ad] of Object.entries(dmData)) {
                    const A = new mol.MatrixParam(ad.A.length, ad.A[0].length, 0);
                    A.data = ad.A.map(r => Float64Array.from(r));
                    const B = new mol.MatrixParam(ad.B.length, ad.B[0].length, 0);
                    B.data = ad.B.map(r => Float64Array.from(r));
                    dm[name] = { A, B };
                }
                model.deltas.push(dm);
            }
        }
        model.activeAlpha = data.alpha || [1.0];
        model.globalStep = data.global_step || 0;
        model.initEmbedSnapshot = data.init_embed_snapshot || [];
        console.log(`[checkpoint] Loaded from ${CKPT_PATH}, step=${model.globalStep}`);
        return [model, tok];
    } catch (e) {
        console.log(`[checkpoint] Failed to load: ${e.message}, starting fresh`);
        return null;
    }
}

// ============================================================
// Training — uses the same trainSteps as browser version
// ============================================================
function cosineLR(globalStep) {
    const lrMax = CFG.learningRate || CFG.lr || 0.01;
    const lrMin = CFG.lrMin || 0.0001;
    if (globalStep < CFG.cosineWarmupSteps) {
        return lrMin + (lrMax - lrMin) * (globalStep / Math.max(1, CFG.cosineWarmupSteps));
    }
    const progress = (globalStep - CFG.cosineWarmupSteps) /
        Math.max(1, CFG.maxTotalSteps - CFG.cosineWarmupSteps);
    return lrMin + 0.5 * (lrMax - lrMin) * (1 + Math.cos(Math.PI * Math.min(progress, 1.0)));
}

function randomChoices(arr, n) {
    const out = [];
    for (let i = 0; i < n; i++) out.push(arr[Math.floor(Math.random() * arr.length)]);
    return out;
}

function nodeTrainSteps(model, tok, docs, nSteps, logEvery) {
    if (!docs.length) return;
    const baseParams = model.allBaseParams();
    const deltaParams = model.allDeltaParams ? model.allDeltaParams() : [];

    for (let step = 0; step < nSteps; step++) {
        let accumLoss = null;
        for (let acc = 0; acc < (CFG.accumSteps || 1); acc++) {
            const batch = randomChoices(docs, CFG.batchSize || 1);
            const batchIds = batch.filter(d => d).map(d => tok.encode(d));
            const loss = model.lossOnBatch(batchIds);
            if (!loss) continue;
            const scaled = loss.mul(1.0 / (CFG.accumSteps || 1));
            mol.backward(scaled);
            if (accumLoss === null) accumLoss = scaled.data;
            else accumLoss += scaled.data;
        }

        let lr = cosineLR(model.globalStep || 0);
        // Scale LR inversely with model size: larger models need smaller LR
        lr *= (CFG.growthStages || [[0,16]])[0][1] / (model.nEmbd || 16);
        // Post-growth LR dampening
        if (model._growthFreezeRemaining > 0) lr *= (CFG.postGrowthLRScale || 0.3);
        if (baseParams.length) model.adamStep(baseParams, "base", lr);
        if (deltaParams.length) model.adamStep(deltaParams, "delta", lr);
        model.globalStep = (model.globalStep || 0) + 1;

        if (step % logEvery === 0) {
            process.stdout.write(`  step ${step}/${nSteps} | loss ${(accumLoss || 0).toFixed(4)} | lr ${lr.toFixed(6)}\n`);
        }
    }
}

// ============================================================
// Generation (inference)
// ============================================================
function generate(model, tok, prompt) {
    return model.generateSentence(prompt || "");
}

// ============================================================
// Main
// ============================================================
async function main() {
    console.log("╔══════════════════════════════════════════════╗");
    console.log("║  MOLEQULA.JS — Node CLI                     ║");
    console.log("║  The Fire element walks the earth of Node    ║");
    console.log("╚══════════════════════════════════════════════╝");

    const corpusLines = loadCorpus(CORPUS_PATH);

    // Try loading checkpoint
    let model, tok;
    const loaded = loadCheckpoint(corpusLines);
    if (loaded) {
        [model, tok] = loaded;
    } else {
        tok = new EvolvingTokenizer(corpusLines);
        model = new GPT(tok);
        // Initialize at the correct stage for corpus size
        const initChars = corpusLines.reduce((a, l) => a + l.length, 0);
        while (model.maybeGrowArchitecture(initChars)) {
            model._growthFreezeRemaining = 0; // skip freeze during init
        }
        console.log(`[init] Fresh model. vocab=${tok.vocabSize}, embd=${model.nEmbd}, layers=${model.nLayer}`);
    }
    model.maybeExpandVocab(tok.vocabSize);

    // Enable BPE if corpus big enough
    const totalChars = corpusLines.reduce((a, l) => a + l.length, 0);
    if (totalChars >= CFG.enableBpeAfterChars && !tok.bpeEnabled) {
        tok.trainBpe(corpusLines, CFG.bpeNumMerges);
        model.maybeExpandVocab(tok.vocabSize);
        console.log(`[bpe] Enabled. vocab=${tok.vocabSize}`);
    }

    // Warmup training — scale by model size
    const embryoEmbd = (CFG.growthStages || [[0,16]])[0][1];
    const warmupScale = Math.max(1, Math.floor((model.nEmbd || 16) / embryoEmbd));
    const effectiveWarmup = CFG.warmupSteps * warmupScale;
    const startStep = model.globalStep || 0;
    const remaining = Math.max(0, effectiveWarmup - startStep);
    if (remaining > 0) {
        console.log(`[trainer] Warmup: ${remaining} steps (scaled ${warmupScale}x for embd=${model.nEmbd}) from step ${startStep}...`);
        nodeTrainSteps(model, tok, corpusLines, remaining, 100);
        console.log("[trainer] Warmup complete.");
        saveCheckpoint(model, tok);
    } else {
        console.log(`[trainer] Already trained ${startStep} steps, skipping warmup.`);
    }

    // REPL
    console.log("molequla is alive. Type and press Enter. Ctrl+C to exit.");
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: "> " });
    rl.prompt();

    rl.on("line", (line) => {
        const input = line.trim();
        if (!input) { rl.prompt(); return; }

        // Generate response
        const response = generate(model, tok, input);
        console.log(response);

        // Add to corpus
        corpusLines.push(input);
        if (corpusLines.length > CFG.maxCorpusLines) corpusLines.shift();

        rl.prompt();
    });

    rl.on("close", () => {
        console.log("\n[shutdown] Saving...");
        saveCheckpoint(model, tok);
        console.log("[shutdown] Done. The fire rests.");
        process.exit(0);
    });
}

main().catch(e => { console.error(e); process.exit(1); });
