/*
 * vision.h — SmolVLM / idefics3 image preprocessing (PHASE 2) and, later,
 * the SigLIP vision tower + pixel-shuffle connector.
 *
 * smolvlm_preprocess() turns an image file into N normalized 512x512 RGB frames
 * the way the idefics3 image processor does:
 *   - scale the longest edge TO 2048 (up or down — not a cap)
 *   - split into an n_rows x n_cols grid of 512x512 tiles (if >1 tile)
 *   - append one global image (full frame resized to 512x512), always last
 *   - rescale 1/255 (done on load) + normalize mean/std = 0.5/0.5/0.5  -> [-1,1]
 * Each frame yields 64 visual tokens via the pixel-shuffle connector, so a
 * 1024x1024 photo becomes 17 frames = 1088 visual tokens, matching the slice
 * count of the reference runtime. SMOLVLM_NOSPLIT=1 forces a single global
 * frame: cheap, and blind to small text.
 */
#ifndef VISION_H
#define VISION_H

/* Preprocess image -> out_n_frames frames, each [3, S, S] (S = out_S = 512),
 * channel-first, normalized to [-1,1]. Returns malloc'd float* of
 * n_frames*3*S*S (caller frees), or NULL on failure.
 * out_rows/out_cols receive the tile grid (1x1 when there is only a global
 * frame); the caller needs it to emit the <row_i_col_j> markers. Either may
 * be NULL. Frame order is row-major tiles first, global frame last. */
float* smolvlm_preprocess_grid(const char* path, int* out_n_frames, int* out_S,
                               int* out_rows, int* out_cols);
/* grid-less shorthand, kept for the probe modes */
float* smolvlm_preprocess(const char* path, int* out_n_frames, int* out_S);

/* ── SigLIP vision tower (PHASE 3) ──────────────────────────────────────────
 * Forward verified against llama.cpp clip.cpp (our oracle) + HF SigLIP:
 * conv patch-embed -> +pos -> 12x [LN1->attn->res, LN2->mlp->res] -> post_ln.
 * Pre-norm, LayerNorm eps 1e-6, attn scale 1/sqrt(head_dim), non-causal,
 * gelu_pytorch_tanh MLP, NO cls token, NO pre_ln, NO pooling. */
typedef struct siglip_model siglip_model;

/* Load the SigLIP tower weights (v.*) from a SmolVLM mmproj GGUF. NULL on fail. */
siglip_model* siglip_load(const char* mmproj_path);
void siglip_free(siglip_model* m);

/* tower dims (after load): n_patches (1024) and hidden (768). */
int siglip_n_patches(const siglip_model* m);
int siglip_hidden(const siglip_model* m);

/* Encode one preprocessed frame [3,S,S] (S=512) -> out[n_patches*hidden]
 * post-layernorm hidden states. Returns 0 on success. */
int siglip_encode(const siglip_model* m, const float* frame, float* out);

/* ── pixel-shuffle connector (PHASE 4) ──────────────────────────────────────
 * idefics3 pixel-shuffle (scale 4): [n_patches=1024, hidden=768] -> [64, 12288]
 * (4x4 spatial-neighborhood merge) then Linear mm.model.fc [text_dim, 12288]
 * (no bias, no activation) -> [n_vis_tokens=64, text_dim=576] visual embeddings
 * already in the text decoder's hidden dim, ready to splice at <image> tokens. */
int siglip_n_vis_tokens(const siglip_model* m);   /* 64 */
int siglip_text_dim(const siglip_model* m);        /* 576 */

/* hidden[n_patches*hidden] (siglip_encode output) -> out[n_vis_tokens*text_dim]. 0 on ok. */
int siglip_connect(const siglip_model* m, const float* hidden, float* out);

#endif /* VISION_H */
