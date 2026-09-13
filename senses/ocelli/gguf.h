// gguf.h — GGUF file parser for notorch
// Reads llama.cpp-compatible GGUF files
// Supports: F32, F16, Q8_0, Q4_0 tensor types

#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>

#define GGUF_MAGIC 0x46554747  // "GGUF"

// Tensor data types
#define GGUF_TYPE_F32   0
#define GGUF_TYPE_F16   1
#define GGUF_TYPE_Q4_0  2
#define GGUF_TYPE_Q4_1  3
#define GGUF_TYPE_Q5_0  6
#define GGUF_TYPE_Q8_0  8
#define GGUF_TYPE_Q4_K 12
#define GGUF_TYPE_Q6_K 14

#define GGUF_MAX_TENSORS 2048   // covers Llama-70B-class (~723 tensors); loader fails loud beyond this
#define GGUF_MAX_NAME    128
#define GGUF_MAX_KV      128

typedef struct {
    char     name[GGUF_MAX_NAME];
    uint32_t ndim;
    uint64_t shape[4];
    uint32_t dtype;
    uint64_t offset;     // offset from data section start
    uint64_t n_elements;
} gguf_tensor_info;

typedef struct {
    char     key[GGUF_MAX_NAME];
    uint32_t type;
    union {
        uint32_t u32;
        int32_t  i32;
        float    f32;
        uint8_t  b;
        char     str[256];
        uint64_t u64;
    } val;
} gguf_kv;

typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;

    // Parsed metadata
    gguf_kv        kv[GGUF_MAX_KV];
    int            n_kv_parsed;

    // Tensor directory
    gguf_tensor_info tensors[GGUF_MAX_TENSORS];

    // Data section
    uint8_t*       data;          // page-aligned (posix_memalign) raw tensor bytes
    uint64_t       data_offset;   // file offset where tensor data starts
    uint64_t       data_size;     // page-rounded byte size of `data` (for Metal NoCopy)

    // Architecture params (extracted from metadata)
    int  n_layers;
    int  n_heads;
    int  n_kv_heads;
    int  embed_dim;
    int  ffn_dim;
    int  vocab_size;
    int  ctx_len;
    float rope_freq_base;
    float rms_eps;
    char arch[64];
} gguf_file;

// Open and parse a GGUF file. Returns NULL on failure.
gguf_file* gguf_open(const char* path);

// Free GGUF file
void gguf_close(gguf_file* gf);

// Find tensor by name. Returns index or -1.
int gguf_find_tensor(const gguf_file* gf, const char* name);

// Dequantize tensor to float32 array. Caller must free returned pointer.
// Handles: F32 (copy), F16 (convert), Q8_0 (dequant), Q4_0 (dequant).
float* gguf_dequant(const gguf_file* gf, int tensor_idx);

// On-disk byte size of tensor `idx` as stored (quantised blocks included), or 0
// if the dtype is unknown or the element count is malformed. Lets a caller keep
// a tensor PACKED instead of dequantising it into f32.
uint64_t gguf_tensor_nbytes(const gguf_file* gf, int tensor_idx);

// Dequantise a packed buffer held by the caller (kept via gguf_tensor_nbytes)
// into `dst`, n_elements floats. Same block layouts as gguf_dequant, without
// needing the gguf_file to still be open. Returns 0 on success, -1 on an
// unsupported dtype. Used to expand a weight ONCE per prompt instead of once
// per token.
int gguf_dequant_raw(const uint8_t* src, uint32_t dtype, uint64_t n_elements, float* dst);

// Load tensor as f16 (uint16_t) — half the RAM of gguf_dequant. Raw copy if the
// tensor is already F16; otherwise dequant to f32 then round to f16. Caller frees.
// Used for matmul weights (lazy-dequant to a scratch f32 buffer at matmul time).
uint16_t* gguf_load_f16(const gguf_file* gf, int tensor_idx);

// Batch f16 -> f32 (for dequant-to-scratch in the matmul). dst must hold n floats.
void gguf_f16_to_f32_n(const uint16_t* src, float* dst, long n);

// Get metadata value by key. Returns NULL if not found.
const gguf_kv* gguf_get_kv(const gguf_file* gf, const char* key);

// Read a GGUF type-9 string array (e.g. "tokenizer.ggml.tokens" / ".merges") by key.
// Re-scans the file (arrays are skipped during gguf_open). Returns malloc'd char**
// of *out_n strdup'd strings, or NULL if absent. Caller frees each string + the array.
char** gguf_read_str_array(const char* path, const char* key, int* out_n);

// Read a GGUF int32/uint32 array (e.g. "tokenizer.ggml.token_type") by key.
// Re-scans the file (arrays are skipped during gguf_open). Returns malloc'd int32_t*
// of *out_n values, or NULL if absent. Caller frees the array.
int32_t* gguf_read_i32_array(const char* path, const char* key, int* out_n);

// Print GGUF summary
void gguf_print_info(const gguf_file* gf);

#endif // GGUF_H
