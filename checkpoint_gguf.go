package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
)

// ═══════════════════════════════════════════════════════════════════════════════
// The checkpoint as a GGUF, beside the JSON one.
//
// The JSON checkpoint is the interchange format between the four cores and stays
// exactly as it is: 22.9 bytes of decimal text per float32 parameter, written and
// read by molequla.c, molequla.rs and molequla.js as well as by this file's
// neighbour. The GGUF is the same organism in the shape the machine trains it in
// — F32, tensor by tensor, with a directory in front — so it can be mapped
// instead of parsed. Step 2 of docs/resonator_design.md; the sleeper (step 3) and
// the resonator (step 4) both need a file that can be mapped.
//
// What is NOT here: Chuck's moment slots. §2.2 of the design puts them in a
// sibling moments.gguf that only the resonator maps, and they are not in this
// file for a reason older than the design — they live in the notorch tape
// (nt_tape_chuck_step keeps m and v per registered parameter), the tape is
// process-local, and nothing has ever carried them across a restart. Writing them
// here would mean writing state no loader reads. The file says so in one key,
// molequla.chuck_moments, so a reader can tell "no optimizer state" from "this
// writer did not know about optimizer state".
// ═══════════════════════════════════════════════════════════════════════════════

// ggufPathFor names the binary sibling of a JSON checkpoint path.
func ggufPathFor(jsonPath string) string {
	return strings.TrimSuffix(jsonPath, ".json") + ".gguf"
}

// gptConstructions counts NewGPT calls. The GGUF load path exists to avoid one:
// NewGPT fills every matrix with rand.NormFloat64 before LoadCheckpoint throws
// the lot away, which is 69 MB of a stage-4 organism's 152 MB load. The counter
// is what makes that assertion able to go red — checkpoint_gguf_test watches it
// across both paths.
var gptConstructions atomic.Int64

// checkpointIdentity is what the GGUF must agree with the JSON about before it is
// preferred: the tokenizer, whole, and the three dimensions that are the growth
// stage (CurrentGrowthStage matches on exactly this triple). Lengths are written
// before the strings because a token may contain any byte, newlines included.
func checkpointIdentity(nEmbd, nLayer, nHead int, tk TokenizerJSON) string {
	h := sha256.New()
	fmt.Fprintf(h, "molequla-ckpt/1 %d %d %d %t %d %d %d\n",
		nEmbd, nLayer, nHead, tk.BPEEnabled, tk.TrainedChars, len(tk.Tokens), len(tk.Merges))
	for _, s := range tk.Tokens {
		fmt.Fprintf(h, "%d:", len(s))
		h.Write([]byte(s))
	}
	for _, m := range tk.Merges {
		for _, s := range m {
			fmt.Fprintf(h, "%d:", len(s))
			h.Write([]byte(s))
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// checkpointIdentityFromJSON reads the identity off the front of the JSON
// checkpoint. "cfg" and "tokenizer" are the first two fields the writer emits, so
// this reads some tens of kilobytes and stops; it gives up rather than skip a
// matrix, because skipping one through json.RawMessage would buffer it.
func checkpointIdentityFromJSON(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	dec := json.NewDecoder(bufio.NewReaderSize(f, 1<<16))
	if err := expectDelim(dec, '{'); err != nil {
		return "", err
	}
	var cfg struct {
		NEmbd  int `json:"n_embd"`
		NLayer int `json:"n_layer"`
		NHead  int `json:"n_head"`
	}
	var tk TokenizerJSON
	haveCfg, haveTok := false, false
	for dec.More() && !(haveCfg && haveTok) {
		key, err := expectKey(dec)
		if err != nil {
			return "", err
		}
		switch key {
		case "cfg":
			if err := dec.Decode(&cfg); err != nil {
				return "", err
			}
			haveCfg = true
		case "tokenizer":
			if err := dec.Decode(&tk); err != nil {
				return "", err
			}
			haveTok = true
		case "base", "deltas", "init_embed_snapshot":
			return "", fmt.Errorf("checkpoint %s reaches %q before its identity", path, key)
		default:
			var skip json.RawMessage
			if err := dec.Decode(&skip); err != nil {
				return "", err
			}
		}
	}
	if !haveCfg || !haveTok {
		return "", fmt.Errorf("checkpoint %s carries no cfg/tokenizer pair", path)
	}
	return checkpointIdentity(cfg.NEmbd, cfg.NLayer, cfg.NHead, tk), nil
}

// ── writing ───────────────────────────────────────────────────────────────────

// ggufTensor is one declared tensor and the matrix it comes from.
type ggufTensor struct {
	name string
	mp   *MatrixParam
}

// ggufRectangular refuses a matrix whose rows are not all there and all the same
// length. The JSON writer emits a missing row as null and a loader rebuilds it as
// empty; a GGUF has one shape or it has nothing, so the ragged case ends the
// write instead of being papered over.
func ggufRectangular(name string, mp *MatrixParam) error {
	if mp == nil || mp.Nout <= 0 || mp.Nin <= 0 {
		return fmt.Errorf("%s: an empty matrix has no GGUF shape", name)
	}
	if len(mp.Rows) != mp.Nout {
		return fmt.Errorf("%s: %d rows for a declared %d", name, len(mp.Rows), mp.Nout)
	}
	for i, r := range mp.Rows {
		if r == nil || len(r.Data) != mp.Nin {
			return fmt.Errorf("%s: row %d is not %d wide", name, i, mp.Nin)
		}
	}
	return nil
}

// ggufModelTensors lists every tensor of the checkpoint in a fixed order: the
// base matrices by sorted key (the order the JSON writer uses), then each delta
// module's adapters by sorted key, then the initial-embedding snapshot. Order is
// not load-bearing for the reader, which resolves by name, but it is what makes
// two saves of the same organism the same file.
func ggufModelTensors(model *GPT) ([]ggufTensor, error) {
	out := make([]ggufTensor, 0, 128)
	names := make([]string, 0, len(model.Base))
	for k := range model.Base {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		if err := ggufRectangular("base."+k, model.Base[k]); err != nil {
			return nil, err
		}
		out = append(out, ggufTensor{"base." + k, model.Base[k]})
	}
	for i, mod := range model.Deltas {
		dnames := make([]string, 0, len(mod))
		for k := range mod {
			dnames = append(dnames, k)
		}
		sort.Strings(dnames)
		for _, k := range dnames {
			da := mod[k]
			if da == nil {
				return nil, fmt.Errorf("delta.%d.%s: nil adapter", i, k)
			}
			if err := ggufRectangular(fmt.Sprintf("delta.%d.%s.A", i, k), da.A); err != nil {
				return nil, err
			}
			if err := ggufRectangular(fmt.Sprintf("delta.%d.%s.B", i, k), da.B); err != nil {
				return nil, err
			}
			out = append(out,
				ggufTensor{fmt.Sprintf("delta.%d.%s.A", i, k), da.A},
				ggufTensor{fmt.Sprintf("delta.%d.%s.B", i, k), da.B})
		}
	}
	if len(model.InitEmbedSnapshot) > 0 {
		w := len(model.InitEmbedSnapshot[0])
		for i, r := range model.InitEmbedSnapshot {
			if len(r) != w {
				return nil, fmt.Errorf("init_embed_snapshot: row %d is %d wide, not %d", i, len(r), w)
			}
		}
		if w == 0 {
			return nil, errors.New("init_embed_snapshot: rows of zero width")
		}
	}
	return out, nil
}

// writeCheckpointGGUF writes the organism as F32 tensors with the metadata a
// loader needs. Nothing larger than one row is allocated: every tensor is
// declared first (the format puts the whole directory before any bytes) and then
// delivered a row at a time through the chunked entry points.
func writeCheckpointGGUF(path string, model *GPT, tok *EvolvingTokenizer) error {
	tensors, err := ggufModelTensors(model)
	if err != nil {
		return err
	}
	alpha, err := json.Marshal(model.ActiveAlpha)
	if err != nil {
		return err
	}
	merges := make([]string, 0, 2*len(tok.Merges))
	mergePairs := make([][]string, 0, len(tok.Merges))
	for _, m := range tok.Merges {
		merges = append(merges, m.A, m.B)
		mergePairs = append(mergePairs, []string{m.A, m.B})
	}
	identity := checkpointIdentity(model.NEmbd, model.NLayer, model.NHead, TokenizerJSON{
		Tokens: tok.Tokens, BPEEnabled: tok.BPEEnabled, Merges: mergePairs, TrainedChars: tok.TrainedChars,
	})

	w, err := ggufWriteOpen(path)
	if err != nil {
		return err
	}
	fail := func(err error) error {
		w.abort()
		return err
	}
	for _, e := range []error{
		w.kvStr("general.architecture", "molequla"),
		w.kvU32("molequla.block_count", uint32(model.NLayer)),
		w.kvU32("molequla.embedding_length", uint32(model.NEmbd)),
		w.kvU32("molequla.attention.head_count", uint32(model.NHead)),
		w.kvU32("molequla.context_length", uint32(model.BlockSize)),
		w.kvU32("molequla.vocab_size", uint32(len(tok.Tokens))),
		w.kvStrArray("molequla.head_types", CFG.HeadTypes),
		w.kvU64("molequla.global_step", uint64(model.globalStep)),
		w.kvI32("molequla.growth_step_offset", int32(model.growthStepOffset)),
		w.kvI32("molequla.last_warmup_stage", int32(model.lastWarmupStage)),
		w.kvU64("molequla.corpus_ingested_total", uint64(model.corpusIngestedTotal)),
		w.kvU32("molequla.delta_modules", uint32(len(model.Deltas))),
		w.kvStr("molequla.active_alpha", string(alpha)),
		w.kvBool("molequla.chuck_moments", false),
		w.kvStr("molequla.identity", identity),
		w.kvStrArray("tokenizer.ggml.tokens", tok.Tokens),
		w.kvStrArray("molequla.tokenizer.merges", merges),
		w.kvBool("molequla.tokenizer.bpe_enabled", tok.BPEEnabled),
		w.kvU32("molequla.tokenizer.trained_chars", uint32(tok.TrainedChars)),
	} {
		if e != nil {
			return fail(e)
		}
	}

	widest := 0
	for _, t := range tensors {
		if err := w.declF32(t.name, []uint64{uint64(t.mp.Nin), uint64(t.mp.Nout)}); err != nil {
			return fail(err)
		}
		if t.mp.Nin > widest {
			widest = t.mp.Nin
		}
	}
	if len(model.InitEmbedSnapshot) > 0 {
		sw := len(model.InitEmbedSnapshot[0])
		if err := w.declF32("init_embed_snapshot",
			[]uint64{uint64(sw), uint64(len(model.InitEmbedSnapshot))}); err != nil {
			return fail(err)
		}
		if sw > widest {
			widest = sw
		}
	}

	row := make([]float32, widest)
	for _, t := range tensors {
		if err := w.tensorBegin(t.name); err != nil {
			return fail(err)
		}
		for i := 0; i < t.mp.Nout; i++ {
			src := t.mp.Rows[i].Data
			for j := range src {
				row[j] = float32(src[j])
			}
			if err := w.chunkF32(row[:len(src)]); err != nil {
				return fail(err)
			}
		}
		if err := w.tensorEnd(); err != nil {
			return fail(err)
		}
	}
	if len(model.InitEmbedSnapshot) > 0 {
		if err := w.tensorBegin("init_embed_snapshot"); err != nil {
			return fail(err)
		}
		for _, src := range model.InitEmbedSnapshot {
			for j := range src {
				row[j] = float32(src[j])
			}
			if err := w.chunkF32(row[:len(src)]); err != nil {
				return fail(err)
			}
		}
		if err := w.tensorEnd(); err != nil {
			return fail(err)
		}
	}
	return w.close()
}

// ── loading ───────────────────────────────────────────────────────────────────

// ggufCheckpointFresh answers whether the binary sibling of jsonPath may be
// preferred. Two conditions, both cheap. It has to exist, and it has to be at
// least as new as the JSON: SaveCheckpoint renames the JSON first and the GGUF
// second, so a process killed between the two renames leaves a GGUF older than
// the JSON — which is exactly the stale pair this refuses. The identity match is
// the second gate and belongs to the load itself, which has the file open.
func ggufCheckpointFresh(jsonPath string) (string, bool) {
	gp := ggufPathFor(jsonPath)
	gs, err := os.Stat(gp)
	if err != nil {
		return "", false
	}
	js, err := os.Stat(jsonPath)
	if err != nil {
		return "", false
	}
	if gs.ModTime().Before(js.ModTime()) {
		fmt.Printf("[ckpt] %s is older than the JSON checkpoint — loading the JSON\n", gp)
		return "", false
	}
	return gp, true
}

// loadCheckpointGGUF builds the organism from the mapped file. It does not call
// NewGPT: the shell carries the fields and the pre-computed layer keys, and every
// matrix is allocated at the shape the file declares and filled straight from the
// mapping. wantIdentity is the JSON's; an empty string skips the comparison and
// is only for a file with no JSON beside it.
func loadCheckpointGGUF(docs []string, path, wantIdentity string) (*GPT, *EvolvingTokenizer, error) {
	gf, err := ggufOpen(path)
	if err != nil {
		return nil, nil, err
	}
	defer gf.close()

	if arch, ok := gf.kvStr("general.architecture"); !ok || arch != "molequla" {
		return nil, nil, fmt.Errorf("%s: general.architecture is %q", path, arch)
	}
	nEmbd, ok1 := gf.kvU64("molequla.embedding_length")
	nLayer, ok2 := gf.kvU64("molequla.block_count")
	nHead, ok3 := gf.kvU64("molequla.attention.head_count")
	if !ok1 || !ok2 || !ok3 || nEmbd == 0 || nLayer == 0 || nHead == 0 {
		return nil, nil, fmt.Errorf("%s: the shape metadata is missing", path)
	}
	tokens, err := ggufReadStrArray(path, "tokenizer.ggml.tokens")
	if err != nil {
		return nil, nil, err
	}
	flat, err := ggufReadStrArray(path, "molequla.tokenizer.merges")
	if err != nil {
		return nil, nil, err
	}
	if len(flat)%2 != 0 {
		return nil, nil, fmt.Errorf("%s: %d merge halves is not a whole number of pairs", path, len(flat))
	}
	bpe, _ := gf.kvU64("molequla.tokenizer.bpe_enabled")
	trained, _ := gf.kvU64("molequla.tokenizer.trained_chars")

	pairs := make([][]string, 0, len(flat)/2)
	for i := 0; i+1 < len(flat); i += 2 {
		pairs = append(pairs, []string{flat[i], flat[i+1]})
	}
	tkJSON := TokenizerJSON{
		Tokens: tokens, BPEEnabled: bpe != 0, Merges: pairs, TrainedChars: int(trained),
	}
	if have := checkpointIdentity(int(nEmbd), int(nLayer), int(nHead), tkJSON); wantIdentity != "" && have != wantIdentity {
		return nil, nil, fmt.Errorf("identity %s does not match the JSON checkpoint's %s", have[:12], wantIdentity[:12])
	}

	// Tokenizer, restored exactly as the JSON path restores it.
	if len(docs) == 0 {
		docs = []string{"Hello."}
	}
	tok := NewEvolvingTokenizer(docs)
	if len(tkJSON.Tokens) > 0 {
		tok.Tokens = tkJSON.Tokens
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
	for _, m := range tkJSON.Merges {
		p := MergePair{m[0], m[1]}
		tok.Merges = append(tok.Merges, p)
		tok.MergeToTok[p] = m[0] + "+" + m[1]
	}
	tok.BPEEnabled = tkJSON.BPEEnabled
	tok.TrainedChars = tkJSON.TrainedChars

	// The saved shape wins over the configured one, as it does on the JSON path.
	CFG.NEmbd, CFG.NLayer, CFG.NHead = int(nEmbd), int(nLayer), int(nHead)
	if ht, err := ggufReadStrArray(path, "molequla.head_types"); err == nil && len(ht) > 0 {
		CFG.HeadTypes = ht
	}

	model := newGPTShell(tok)
	nModules, _ := gf.kvU64("molequla.delta_modules")
	model.Deltas = make([]DeltaModule, nModules)
	for i := range model.Deltas {
		model.Deltas[i] = make(DeltaModule)
	}

	for i := 0; i < gf.nTensors(); i++ {
		name := gf.tensorName(i)
		shape := gf.tensorShape(i)
		if len(shape) != 2 || shape[0] == 0 || shape[1] == 0 {
			return nil, nil, fmt.Errorf("%s: tensor %q is not a matrix", path, name)
		}
		nin, nout := int(shape[0]), int(shape[1])
		switch {
		case name == "init_embed_snapshot":
			snap := make([][]float64, nout)
			for r := 0; r < nout; r++ {
				snap[r] = make([]float64, nin)
				if err := gf.readRow(i, r, snap[r]); err != nil {
					return nil, nil, err
				}
			}
			model.InitEmbedSnapshot = snap
		case strings.HasPrefix(name, "base."):
			mp, err := ggufReadMatrix(gf, i, nout, nin)
			if err != nil {
				return nil, nil, err
			}
			model.Base[strings.TrimPrefix(name, "base.")] = mp
		case strings.HasPrefix(name, "delta."):
			mod, key, half, err := ggufSplitDeltaName(name)
			if err != nil {
				return nil, nil, err
			}
			if mod >= len(model.Deltas) {
				return nil, nil, fmt.Errorf("%s: %q names module %d of %d", path, name, mod, len(model.Deltas))
			}
			mp, err := ggufReadMatrix(gf, i, nout, nin)
			if err != nil {
				return nil, nil, err
			}
			da := model.Deltas[mod][key]
			if da == nil {
				da = &DeltaAdapter{A: &MatrixParam{}, B: &MatrixParam{}}
				model.Deltas[mod][key] = da
			}
			if half == "A" {
				da.A = mp
			} else {
				da.B = mp
			}
		default:
			return nil, nil, fmt.Errorf("%s: unknown tensor %q", path, name)
		}
	}

	if CFG.TieEmbeddings {
		if wte, ok := model.Base["wte"]; ok {
			model.Base["lm_head"] = wte
		}
	}
	if s, ok := gf.kvStr("molequla.active_alpha"); ok {
		if err := json.Unmarshal([]byte(s), &model.ActiveAlpha); err != nil {
			return nil, nil, fmt.Errorf("%s: active_alpha %q: %w", path, s, err)
		}
	}
	if len(model.Deltas) == 0 {
		model.AddDeltaModule(1.0)
	}
	if len(model.InitEmbedSnapshot) == 0 {
		wte, ok := model.Base["wte"]
		if !ok {
			return nil, nil, fmt.Errorf("%s: no wte tensor", path)
		}
		model.InitEmbedSnapshot = make([][]float64, len(wte.Rows))
		for i, row := range wte.Rows {
			snap := make([]float64, len(row.Data))
			copy(snap, row.Data)
			model.InitEmbedSnapshot[i] = snap
		}
	}

	step, _ := gf.kvU64("molequla.global_step")
	model.globalStep = int(step)
	if off, ok := gf.kvI32("molequla.growth_step_offset"); ok {
		model.growthStepOffset = int(off)
	}
	ingested, _ := gf.kvU64("molequla.corpus_ingested_total")
	model.corpusIngestedTotal = int(ingested)
	if model.corpusIngestedTotal == 0 {
		for _, d := range docs {
			model.corpusIngestedTotal += len(d)
		}
	}
	if warm, ok := gf.kvI32("molequla.last_warmup_stage"); ok {
		model.lastWarmupStage = int(warm)
	}

	// Backward compatibility, the same three ensures the JSON path runs: a
	// checkpoint from before a weight existed gets it fresh-initialised here.
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
		model.ensureRRPRAMFactors(li)
	}
	return model, tok, nil
}

// ggufReadMatrix allocates a matrix at the declared shape and fills it from the
// mapping, one row into one freshly made []float64. There is no intermediate
// buffer: the file's bytes are widened into the organism's own array.
func ggufReadMatrix(gf *ggufFile, idx, nout, nin int) (*MatrixParam, error) {
	mp := &MatrixParam{Nout: nout, Nin: nin, Rows: make([]*Vec, nout)}
	for r := 0; r < nout; r++ {
		d := make([]float64, nin)
		if err := gf.readRow(idx, r, d); err != nil {
			return nil, err
		}
		mp.Rows[r] = NewVecWithGrad(d) // loaded params always need grad
	}
	return mp, nil
}

// ggufSplitDeltaName takes "delta.<module>.<weight key>.<A|B>" apart. The weight
// key carries dots of its own (l4.h7.w_pattern), so the split is at the first dot
// after the prefix and at the last dot before the half.
func ggufSplitDeltaName(name string) (int, string, string, error) {
	rest := strings.TrimPrefix(name, "delta.")
	dot := strings.Index(rest, ".")
	if dot <= 0 {
		return 0, "", "", fmt.Errorf("malformed delta tensor name %q", name)
	}
	mod, err := strconv.Atoi(rest[:dot])
	if err != nil || mod < 0 {
		return 0, "", "", fmt.Errorf("malformed delta module index in %q", name)
	}
	rest = rest[dot+1:]
	last := strings.LastIndex(rest, ".")
	if last <= 0 {
		return 0, "", "", fmt.Errorf("malformed delta tensor name %q", name)
	}
	half := rest[last+1:]
	if half != "A" && half != "B" {
		return 0, "", "", fmt.Errorf("delta tensor %q is neither A nor B", name)
	}
	return mod, rest[:last], half, nil
}
