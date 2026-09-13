/* bpe.h — byte-level BPE (GPT-2 / Tekken style) over a GGUF tokenizer.
 * Reads tokenizer.ggml.tokens + tokenizer.ggml.merges from the GGUF and
 * implements encode (UTF-8 text -> token ids) and per-token decode.
 * Generic: works for any GGUF whose tokenizer is byte-level BPE
 * (LFM2.5 dev target, Qwen3 qyent, Mistral/Tekken oyent).
 */
#ifndef BPE_H
#define BPE_H

typedef struct bpe_tokenizer bpe_tokenizer;

/* Load tokens+merges from a GGUF file. NULL on failure. */
bpe_tokenizer *bpe_load(const char *gguf_path);
void bpe_free(bpe_tokenizer *t);

int bpe_n_vocab(const bpe_tokenizer *t);

/* Encode UTF-8 text -> token ids. Writes up to cap ids into out_ids,
 * returns the number written (may be < needed if cap is hit). */
int bpe_encode(const bpe_tokenizer *t, const char *text, int *out_ids, int cap);

/* Decode one token id -> its UTF-8 bytes, appended to buf (cap incl. NUL).
 * Returns bytes appended. */
int bpe_decode_token(const bpe_tokenizer *t, int id, char *buf, int cap);

/* id of an exact token string (e.g. "<|im_start|>"), or -1 if absent. */
int bpe_token_id(const bpe_tokenizer *t, const char *token);

#endif /* BPE_H */
