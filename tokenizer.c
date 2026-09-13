/* tokenizer.c — token ids back to text, and the language table.
 *
 * The byte-level BPE table was applied when the model was converted: each vocab
 * entry in the `.bin` is already the byte string the token stands for, which is
 * why `whisper_token_to_str` (`src/whisper.cpp:4305`) is a plain table lookup and
 * why concatenating entries yields valid UTF-8 with no byte_decoder step. Ids past
 * the stored table are the special tokens whisper.cpp synthesises at load; none of
 * them is ever printed, so they render as their bracketed names for --tokens and
 * contribute nothing to the transcript.
 *
 * Nothing here encodes text: v1 has no initial prompt and no prompt conditioning,
 * so no path needs a text-to-id direction.
 */

#include "ears.h"

#include <stdio.h>
#include <string.h>

/* g_lang in whisper.cpp declares 100 entries, ids 0..99, while a model that says
 * n_langs = 99 only has room for ids 0..98 before the task tokens begin. The list
 * is reproduced in full and in id order because whisper_lang_auto_detect iterates
 * all of it; the last entry aliases onto the translate token, and detection here
 * keeps that quirk rather than quietly disagreeing with the oracle. */
static const char *const g_lang[] = {
    "en","zh","de","es","ru","ko","fr","ja","pt","tr","pl","ca","nl","ar","sv","it",
    "id","hi","fi","vi","he","uk","el","ms","cs","ro","da","hu","ta","no","th","ur",
    "hr","bg","lt","la","mi","ml","cy","sk","te","fa","lv","bn","sr","az","sl","kn",
    "et","mk","br","eu","is","hy","ne","mn","bs","kk","sq","sw","gl","mr","pa","si",
    "km","sn","yo","so","af","oc","ka","be","tg","sd","gu","am","yi","lo","uz","fo",
    "ht","ps","tk","nn","mt","sa","lb","my","bo","tl","mg","as","tt","haw","ln","ha",
    "ba","jw","su","yue",
};

int ears_n_langs(void) { return (int) (sizeof(g_lang) / sizeof(g_lang[0])); }

int ears_lang_id(const char *code) {
    if (!code) return -1;
    for (int i = 0; i < ears_n_langs(); i++)
        if (!strcmp(g_lang[i], code)) return i;
    return -1;
}

const char *ears_lang_str(int id) {
    return (id >= 0 && id < ears_n_langs()) ? g_lang[id] : "??";
}

int ears_token_append(const ears_vocab *v, int id, char *dst, size_t cap) {
    if (id < 0 || id >= v->n_vocab || cap == 0) return 0;
    if (id < v->n_tok) {
        size_t n = (size_t) v->tok_len[id];
        if (n > cap) n = cap;
        memcpy(dst, v->tok[id], n);
        return (int) n;
    }
    char buf[48];
    int n;
    if (id > v->beg)            n = snprintf(buf, sizeof(buf), "[_TT_%d]", id - v->beg);
    else if (id == v->eot)      n = snprintf(buf, sizeof(buf), "[_EOT_]");
    else if (id == v->sot)      n = snprintf(buf, sizeof(buf), "[_SOT_]");
    else if (id == v->translate)  n = snprintf(buf, sizeof(buf), "[_TRANSLATE_]");
    else if (id == v->transcribe) n = snprintf(buf, sizeof(buf), "[_TRANSCRIBE_]");
    else if (id == v->solm)     n = snprintf(buf, sizeof(buf), "[_SOLM_]");
    else if (id == v->prev)     n = snprintf(buf, sizeof(buf), "[_PREV_]");
    else if (id == v->nosp)     n = snprintf(buf, sizeof(buf), "[_NOSP_]");
    else if (id == v->not_ts)   n = snprintf(buf, sizeof(buf), "[_NOT_]");
    else if (id == v->beg)      n = snprintf(buf, sizeof(buf), "[_BEG_]");
    else if (id > v->sot && id <= v->sot + v->n_langs)
        n = snprintf(buf, sizeof(buf), "[_LANG_%s]", ears_lang_str(id - v->sot - 1));
    else n = snprintf(buf, sizeof(buf), "[_extra_token_%d]", id);
    if (n < 0) return 0;
    size_t w = (size_t) n < cap ? (size_t) n : cap;
    memcpy(dst, buf, w);
    return (int) w;
}
