// oracle_dump — dumps whisper.cpp's own log-mel and encoder output for a wav.
//
// The mel lives in `whisper_state::mel` and the encoder output in
// `whisper_state::embd_enc`; neither is reachable through <whisper.h>. Rather than
// patch the checked-out tree, this translation unit #includes src/whisper.cpp
// verbatim, so the internal types are complete here and the arithmetic is byte for
// byte the one whisper-cli runs. Link against the tree's libggml*.so only — linking
// libwhisper.so as well would duplicate every symbol.
//
//   oracle_dump <model.bin> <wav> <mel.f32> <enc.f32>
//
// mel.f32:       i32 n_mel, i32 n_len, i32 n_len_org, then n_mel*n_len f32 (mel-major).
// enc.f32:       i32 n_ctx, i32 n_state, then n_ctx*n_state f32 (row per audio frame).
// enc.f32.conv:  i32 n_len, i32 n_state, then the post-convolution activations —
//                written unasked, so a parity failure can be pinned to the front
//                of the encoder or to the blocks without a second run.
//
// ORACLE_NO_FLASH=1 turns flash attention off (whisper-cli's -nfa). It matters:
// the flash path sums the attention output in an FP16 register, so the two
// settings are two different oracles. See tests/gate_encoder.sh.
//
// Arianna Method — Defender (phone-1).

#include "whisper.cpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// 16 kHz mono s16 RIFF reader — the same shape as ears' own, kept separate so a bug
// in one does not hide itself in the other.
static bool read_wav(const char * path, std::vector<float> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "oracle_dump: cannot open %s\n", path); return false; }
    unsigned char hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
        fprintf(stderr, "oracle_dump: %s is not RIFF/WAVE\n", path); fclose(f); return false;
    }
    int channels = 0, rate = 0, bits = 0;
    while (true) {
        unsigned char ch[8];
        if (fread(ch, 1, 8, f) != 8) { fclose(f); return false; }
        uint32_t sz = (uint32_t)ch[4] | ((uint32_t)ch[5] << 8) | ((uint32_t)ch[6] << 16) | ((uint32_t)ch[7] << 24);
        if (!memcmp(ch, "fmt ", 4)) {
            std::vector<unsigned char> b(sz);
            if (fread(b.data(), 1, sz, f) != sz) { fclose(f); return false; }
            channels = b[2] | (b[3] << 8);
            rate     = b[4] | (b[5] << 8) | (b[6] << 16) | (b[7] << 24);
            bits     = b[14] | (b[15] << 8);
        } else if (!memcmp(ch, "data", 4)) {
            size_t n = sz / 2;
            std::vector<int16_t> pcm(n);
            if (fread(pcm.data(), 2, n, f) != n) { fclose(f); return false; }
            out.resize(n / (channels ? channels : 1));
            for (size_t i = 0; i < out.size(); i++) out[i] = pcm[i * channels] / 32768.0f;
            fclose(f);
            fprintf(stderr, "oracle_dump: %s  %d Hz %d ch %d bit  %zu samples\n", path, rate, channels, bits, out.size());
            return rate == 16000 && bits == 16;
        } else {
            fseek(f, sz + (sz & 1), SEEK_CUR);
        }
    }
}

static void wr_i32(FILE * f, int v) { fwrite(&v, sizeof(int), 1, f); }

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: oracle_dump <model.bin> <wav> <mel.f32> <enc.f32>\n");
        return 2;
    }
    std::vector<float> pcm;
    if (!read_wav(argv[2], pcm)) return 1;

    // Flash attention is whisper-cli's default and is what the reference runs used,
    // but on the CPU backend it accumulates the attention output in an FP16
    // register (ops.cpp: VKQ16 with ggml_vec_mad_f16), so 1500 encoder keys are
    // summed at half precision. ORACLE_NO_FLASH=1 takes the -nfa path instead,
    // where only the operands are FP16 and the accumulation is FP32 — the fair
    // reference for an f32 port's arithmetic. Both are measured; see README.
    whisper_context_params cp = whisper_context_default_params();
    if (getenv("ORACLE_NO_FLASH")) {
        cp.flash_attn = false;
        fprintf(stderr, "oracle_dump: flash attention off\n");
    }
    whisper_context * ctx = whisper_init_from_file_with_params(argv[1], cp);
    if (!ctx) { fprintf(stderr, "oracle_dump: model load failed\n"); return 1; }

    const int n_threads = 4;
    if (whisper_pcm_to_mel(ctx, pcm.data(), (int) pcm.size(), n_threads) != 0) {
        fprintf(stderr, "oracle_dump: pcm_to_mel failed\n"); return 1;
    }

    whisper_state * st = ctx->state;
    {
        FILE * f = fopen(argv[3], "wb");
        if (!f) { fprintf(stderr, "oracle_dump: cannot write %s\n", argv[3]); return 1; }
        wr_i32(f, st->mel.n_mel);
        wr_i32(f, st->mel.n_len);
        wr_i32(f, st->mel.n_len_org);
        fwrite(st->mel.data.data(), sizeof(float), st->mel.data.size(), f);
        fclose(f);
        fprintf(stderr, "oracle_dump: mel n_mel=%d n_len=%d n_len_org=%d -> %s\n",
                st->mel.n_mel, st->mel.n_len, st->mel.n_len_org, argv[3]);
    }

    if (whisper_encode(ctx, 0, n_threads) != 0) {
        fprintf(stderr, "oracle_dump: encode failed\n"); return 1;
    }
    /* embd_conv — the two convolutions and their GELUs, before the positional
     * embedding and the blocks. Written beside the encoder dump as <enc>.conv so a
     * disagreement can be pinned to the front of the encoder or to the back of it
     * without a second run. Layout is [n_state][n_len], ggml ne = {n_len, n_state}. */
    if (st->embd_conv) {
        ggml_tensor * c = st->embd_conv;
        const int L = (int) c->ne[0], C = (int) c->ne[1];
        std::vector<float> buf((size_t) L * C);
        ggml_backend_tensor_get(c, buf.data(), 0, buf.size() * sizeof(float));
        std::string p = std::string(argv[4]) + ".conv";
        FILE * f = fopen(p.c_str(), "wb");
        if (f) {
            wr_i32(f, L); wr_i32(f, C);
            fwrite(buf.data(), sizeof(float), buf.size(), f);
            fclose(f);
            fprintf(stderr, "oracle_dump: conv n_len=%d n_state=%d -> %s\n", L, C, p.c_str());
        }
    }
    {
        ggml_tensor * e = st->embd_enc;
        const int n_state = (int) e->ne[0];
        const int n_ctx   = (int) e->ne[1];
        std::vector<float> buf((size_t) n_state * n_ctx);
        ggml_backend_tensor_get(e, buf.data(), 0, buf.size() * sizeof(float));
        FILE * f = fopen(argv[4], "wb");
        if (!f) { fprintf(stderr, "oracle_dump: cannot write %s\n", argv[4]); return 1; }
        wr_i32(f, n_ctx);
        wr_i32(f, n_state);
        fwrite(buf.data(), sizeof(float), buf.size(), f);
        fclose(f);
        fprintf(stderr, "oracle_dump: enc n_ctx=%d n_state=%d -> %s\n", n_ctx, n_state, argv[4]);
    }

    whisper_free(ctx);
    return 0;
}
