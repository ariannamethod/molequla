package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// ═══════════════════════════════════════════════════════════════════════════════
// AML Trainer — trains the GPT model using AML/C autograd via CGO
// "Each organism trains itself by calling what it needs from the language."
// ═══════════════════════════════════════════════════════════════════════════════

func amlModelScript(nLayers, nEmbd, nHeads, seqLen, vocabSize int) string {
	var b strings.Builder
	b.Grow(nLayers*768 + 1024)

	b.WriteString("TAPE START\nTAPE PARAM wte\nTAPE PARAM wpe\n")
	for l := 0; l < nLayers; l++ {
		fmt.Fprintf(&b, "TAPE PARAM wq%d\nTAPE PARAM wk%d\nTAPE PARAM wv%d\nTAPE PARAM wo%d\n",
			l, l, l, l)
		fmt.Fprintf(&b, "TAPE PARAM fc_g%d\nTAPE PARAM fc_v%d\nTAPE PARAM fc2_%d\n",
			l, l, l)
	}
	b.WriteString("TAPE PARAM lm_head\n")

	b.WriteString("h = seq_embed(wte, wpe, tokens, seq_len)\n")

	for l := 0; l < nLayers; l++ {
		fmt.Fprintf(&b,
			"h_norm = seq_rmsnorm(h, seq_len, n_embd)\n"+
				"q = seq_matvec(wq%d, h_norm, seq_len)\n"+
				"k = seq_matvec(wk%d, h_norm, seq_len)\n"+
				"v = seq_matvec(wv%d, h_norm, seq_len)\n"+
				"attn_out = multi_head_attention(q, k, v, seq_len, n_embd, n_heads)\n"+
				"attn_proj = seq_matvec(wo%d, attn_out, seq_len)\n"+
				"h = add(h, attn_proj)\n"+
				"h_norm = seq_rmsnorm(h, seq_len, n_embd)\n"+
				"gate_pre = seq_matvec(fc_g%d, h_norm, seq_len)\n"+
				"gate = silu(gate_pre)\n"+
				"up = seq_matvec(fc_v%d, h_norm, seq_len)\n"+
				"mlp_out = mul(gate, up)\n"+
				"mlp_proj = seq_matvec(fc2_%d, mlp_out, seq_len)\n"+
				"h = add(h, mlp_proj)\n",
			l, l, l, l, l, l, l)
	}

	b.WriteString(
		"h_norm = seq_rmsnorm(h, seq_len, n_embd)\n" +
			"logits = seq_matvec(lm_head, h_norm, seq_len)\n" +
			"loss = seq_cross_entropy(logits, targets, seq_len, vocab_size)\n" +
			"TAPE BACKWARD loss\n" +
			"TAPE ADAM_STEP lr\n" +
			"TAPE CLEAR\n")

	return b.String()
}

func pushMatrixToAML(name string, mp *MatrixParam) {
	if mp == nil || mp.Nout == 0 || mp.Nin == 0 {
		return
	}
	rows := mp.Nout
	cols := mp.Nin
	flat := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		if i >= len(mp.Rows) || len(mp.Rows[i].Data) == 0 {
			continue
		}
		for j := 0; j < cols && j < len(mp.Rows[i].Data); j++ {
			flat[i*cols+j] = float32(mp.Rows[i].Data[j])
		}
	}
	amlSetMatrix(name, flat, rows, cols)
}

func pullMatrixFromAML(name string, mp *MatrixParam) {
	if mp == nil || mp.Nout == 0 || mp.Nin == 0 {
		return
	}
	flat := amlGetArray(name)
	if flat == nil {
		return
	}
	rows := mp.Nout
	cols := mp.Nin
	expected := rows * cols
	if len(flat) < expected {
		fmt.Printf("[aml] WARNING: %s has %d values, expected %d\n", name, len(flat), expected)
		return
	}
	for i := 0; i < rows; i++ {
		if i >= len(mp.Rows) {
			break
		}
		for j := 0; j < cols && j < len(mp.Rows[i].Data); j++ {
			mp.Rows[i].Data[j] = float64(flat[i*cols+j])
		}
	}
}

func amlPushWeights(model *GPT) {
	pushMatrixToAML("wte", model.Base["wte"])
	pushMatrixToAML("wpe", model.Base["wpe"])
	pushMatrixToAML("lm_head", model.Base["lm_head"])

	for l := 0; l < model.NLayer; l++ {
		pfx := fmt.Sprintf("l%d.", l)
		pushMatrixToAML(fmt.Sprintf("wq%d", l), model.Base[pfx+"wq"])
		pushMatrixToAML(fmt.Sprintf("wk%d", l), model.Base[pfx+"wk"])
		pushMatrixToAML(fmt.Sprintf("wv%d", l), model.Base[pfx+"wv"])
		pushMatrixToAML(fmt.Sprintf("wo%d", l), model.Base[pfx+"wo"])
		pushMatrixToAML(fmt.Sprintf("fc_g%d", l), model.Base[pfx+"fc_g"])
		pushMatrixToAML(fmt.Sprintf("fc_v%d", l), model.Base[pfx+"fc_v"])
		pushMatrixToAML(fmt.Sprintf("fc2_%d", l), model.Base[pfx+"fc2"])
	}
}

func amlPullWeights(model *GPT) {
	pullMatrixFromAML("wte", model.Base["wte"])
	pullMatrixFromAML("wpe", model.Base["wpe"])
	pullMatrixFromAML("lm_head", model.Base["lm_head"])

	for l := 0; l < model.NLayer; l++ {
		pfx := fmt.Sprintf("l%d.", l)
		pullMatrixFromAML(fmt.Sprintf("wq%d", l), model.Base[pfx+"wq"])
		pullMatrixFromAML(fmt.Sprintf("wk%d", l), model.Base[pfx+"wk"])
		pullMatrixFromAML(fmt.Sprintf("wv%d", l), model.Base[pfx+"wv"])
		pullMatrixFromAML(fmt.Sprintf("wo%d", l), model.Base[pfx+"wo"])
		pullMatrixFromAML(fmt.Sprintf("fc_g%d", l), model.Base[pfx+"fc_g"])
		pullMatrixFromAML(fmt.Sprintf("fc_v%d", l), model.Base[pfx+"fc_v"])
		pullMatrixFromAML(fmt.Sprintf("fc2_%d", l), model.Base[pfx+"fc2"])
	}
}

// amlTrainSteps runs N training steps using AML/C autograd.
// Acquires model.mu internally. Clears AML state after training to free memory.
func amlTrainSteps(model *GPT, tok *EvolvingTokenizer, docs []string, steps int, overrides ...int) {
	if len(docs) == 0 || steps <= 0 {
		return
	}

	model.mu.Lock()
	defer model.mu.Unlock()

	seqLen := model.BlockSize
	if len(overrides) > 0 && overrides[0] > 0 && overrides[0] < seqLen {
		seqLen = overrides[0]
	}

	vocabSize := tok.VocabSize
	embryoEmbd := CFG.GrowthStages[0][1]
	stepsSinceGrowth := model.globalStep - model.growthStepOffset
	lr := cosineLR(model.globalStep, stepsSinceGrowth)
	lr *= float64(embryoEmbd) / float64(model.NEmbd)
	if model.growthFreezeRemaining > 0 {
		lr *= CFG.PostGrowthLRScale
	}

	amlInit()
	hyper := fmt.Sprintf("n_embd = %d\nn_heads = %d\nn_layer = %d\nvocab_size = %d\nseq_len = %d\nlr = %.8f\n",
		model.NEmbd, model.NHead, model.NLayer, vocabSize, seqLen, lr)
	if err := amlExec(hyper); err != nil {
		fmt.Printf("[aml] ERROR setting hyperparams: %v\n", err)
		return
	}

	amlPushWeights(model)

	script := amlModelScript(model.NLayer, model.NEmbd, model.NHead, seqLen, vocabSize)
	tokens := make([]float32, seqLen)
	targets := make([]float32, seqLen)

	var lossSum float64
	var lossCount int

	for step := 0; step < steps; step++ {
		doc := docs[rand.Intn(len(docs))]
		ids := tok.Encode(doc)
		if len(ids) < 2 {
			continue
		}

		startIdx := 0
		if len(ids) > seqLen+1 {
			startIdx = rand.Intn(len(ids) - seqLen - 1)
		}
		for i := 0; i < seqLen; i++ {
			idx := startIdx + i
			if idx < len(ids) {
				tokens[i] = float32(ids[idx])
			} else {
				tokens[i] = 0
			}
			if idx+1 < len(ids) {
				targets[i] = float32(ids[idx+1])
			} else {
				targets[i] = 0
			}
		}

		amlSetArray("tokens", tokens)
		amlSetArray("targets", targets)

		if err := amlExec(script); err != nil {
			fmt.Printf("[aml] ERROR at step %d: %v\n", step, err)
			break
		}

		loss := float64(amlGetFloat("loss"))
		if !math.IsNaN(loss) && !math.IsInf(loss, 0) {
			lossSum += loss
			lossCount++
		}

		if step%10 == 0 && lossCount > 0 {
			avgLoss := lossSum / float64(lossCount)
			fmt.Printf("  [aml] step %d/%d | loss %.4f (avg %.4f) | lr %.6f | seq %d\n",
				step, steps, loss, avgLoss, lr, seqLen)
		}

		model.globalStep++

		stepsSinceGrowth = model.globalStep - model.growthStepOffset
		lr = cosineLR(model.globalStep, stepsSinceGrowth)
		lr *= float64(embryoEmbd) / float64(model.NEmbd)
		if model.growthFreezeRemaining > 0 {
			lr *= CFG.PostGrowthLRScale
		}
		lrScript := fmt.Sprintf("lr = %.8f\n", lr)
		amlExec(lrScript)
	}

	// Pull trained weights back to Go model
	amlPullWeights(model)

	// FREE AML STATE to release ~75GB of memory
	amlClear()

	if lossCount > 0 {
		fmt.Printf("[aml] training complete: %d steps, avg loss %.4f (memory freed)\n", steps, lossSum/float64(lossCount))
	}
}

// amlBurstTrain runs micro-burst training with syntropy-adjusted LR.
func amlBurstTrain(model *GPT, tok *EvolvingTokenizer, docs []string, steps int, burstLR float64) {
	if len(docs) == 0 || steps <= 0 {
		return
	}

	model.mu.Lock()
	defer model.mu.Unlock()

	seqLen := model.BlockSize
	vocabSize := tok.VocabSize
	embryoEmbd := CFG.GrowthStages[0][1]
	lr := burstLR * float64(embryoEmbd) / float64(model.NEmbd)

	amlInit()
	hyper := fmt.Sprintf("n_embd = %d\nn_heads = %d\nn_layer = %d\nvocab_size = %d\nseq_len = %d\nlr = %.8f\n",
		model.NEmbd, model.NHead, model.NLayer, vocabSize, seqLen, lr)
	if err := amlExec(hyper); err != nil {
		fmt.Printf("[aml] ERROR setting burst hyperparams: %v\n", err)
		return
	}

	amlPushWeights(model)

	script := amlModelScript(model.NLayer, model.NEmbd, model.NHead, seqLen, vocabSize)
	tokens := make([]float32, seqLen)
	targets := make([]float32, seqLen)

	var lossSum float64
	var lossCount int

	for step := 0; step < steps; step++ {
		doc := docs[rand.Intn(len(docs))]
		ids := tok.Encode(doc)
		if len(ids) < 2 {
			continue
		}

		startIdx := 0
		if len(ids) > seqLen+1 {
			startIdx = rand.Intn(len(ids) - seqLen - 1)
		}
		for i := 0; i < seqLen; i++ {
			idx := startIdx + i
			if idx < len(ids) {
				tokens[i] = float32(ids[idx])
			} else {
				tokens[i] = 0
			}
			if idx+1 < len(ids) {
				targets[i] = float32(ids[idx+1])
			} else {
				targets[i] = 0
			}
		}

		amlSetArray("tokens", tokens)
		amlSetArray("targets", targets)

		if err := amlExec(script); err != nil {
			fmt.Printf("[aml] burst ERROR at step %d: %v\n", step, err)
			break
		}

		loss := float64(amlGetFloat("loss"))
		if !math.IsNaN(loss) && !math.IsInf(loss, 0) {
			lossSum += loss
			lossCount++
		}
		model.globalStep++
	}

	amlPullWeights(model)

	// FREE AML STATE to release memory after micro-burst
	amlClear()

	if lossCount > 0 {
		fmt.Printf("[aml] burst complete: %d steps, avg loss %.4f (memory freed)\n", steps, lossSum/float64(lossCount))
	}
}
