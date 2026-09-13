// notorch_vision.h — image loading, transforms, patch extraction for VLMs
// Zero dependencies beyond stb_image.h (included automatically)
//
// Part of notorch — neural networks in pure C
// Copyright (C) 2026 Oleg Ataeff & Arianna Method contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef NOTORCH_VISION_H
#define NOTORCH_VISION_H

#include "notorch.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

// ═══════════════════════════════════════════════════════════════════════════════
// IMAGE LOADING — JPEG, PNG, BMP, GIF via stb_image
// ═══════════════════════════════════════════════════════════════════════════════

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_NO_FAILURE_STRINGS
#include "stb_image.h"

typedef struct {
    float* data;    // [C, H, W] — channel-first layout
    int width;
    int height;
    int channels;   // 1 (grayscale) or 3 (RGB)
} nt_image;

// Load image from file. Returns NULL on failure.
// Channels: 0=auto, 1=grayscale, 3=RGB
static nt_image* nt_image_load(const char* path, int desired_channels) {
    int w, h, c;
    unsigned char* raw = stbi_load(path, &w, &h, &c, desired_channels);
    if (!raw) return NULL;
    if (desired_channels > 0) c = desired_channels;

    nt_image* img = (nt_image*)calloc(1, sizeof(nt_image));
    img->width = w;
    img->height = h;
    img->channels = c;
    img->data = (float*)malloc(c * h * w * sizeof(float));

    // Convert uint8 HWC → float CHW, normalized to [0, 1]
    for (int ch = 0; ch < c; ch++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                img->data[ch * h * w + y * w + x] = raw[(y * w + x) * c + ch] / 255.0f;

    stbi_image_free(raw);
    return img;
}

// Load image from memory buffer
static nt_image* nt_image_load_mem(const unsigned char* buf, int len, int desired_channels) {
    int w, h, c;
    unsigned char* raw = stbi_load_from_memory(buf, len, &w, &h, &c, desired_channels);
    if (!raw) return NULL;
    if (desired_channels > 0) c = desired_channels;

    nt_image* img = (nt_image*)calloc(1, sizeof(nt_image));
    img->width = w;
    img->height = h;
    img->channels = c;
    img->data = (float*)malloc(c * h * w * sizeof(float));

    for (int ch = 0; ch < c; ch++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                img->data[ch * h * w + y * w + x] = raw[(y * w + x) * c + ch] / 255.0f;

    stbi_image_free(raw);
    return img;
}

// Free image
static void nt_image_free(nt_image* img) {
    if (!img) return;
    free(img->data);
    free(img);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRANSFORMS — resize, crop, normalize, flip
// ═══════════════════════════════════════════════════════════════════════════════

// Bilinear resize to target width/height. Returns new image.
static nt_image* nt_image_resize(const nt_image* src, int tw, int th) {
    nt_image* dst = (nt_image*)calloc(1, sizeof(nt_image));
    dst->width = tw;
    dst->height = th;
    dst->channels = src->channels;
    dst->data = (float*)malloc(src->channels * th * tw * sizeof(float));

    float sx = (float)src->width / tw;
    float sy = (float)src->height / th;

    for (int ch = 0; ch < src->channels; ch++) {
        const float* sc = src->data + ch * src->height * src->width;
        float* dc = dst->data + ch * th * tw;
        for (int y = 0; y < th; y++) {
            float fy = (y + 0.5f) * sy - 0.5f;
            int y0 = (int)floorf(fy);
            int y1 = y0 + 1;
            float wy = fy - y0;
            if (y0 < 0) y0 = 0;
            if (y1 >= src->height) y1 = src->height - 1;
            for (int x = 0; x < tw; x++) {
                float fx = (x + 0.5f) * sx - 0.5f;
                int x0 = (int)floorf(fx);
                int x1 = x0 + 1;
                float wx = fx - x0;
                if (x0 < 0) x0 = 0;
                if (x1 >= src->width) x1 = src->width - 1;
                float v = sc[y0 * src->width + x0] * (1-wx) * (1-wy)
                        + sc[y0 * src->width + x1] * wx * (1-wy)
                        + sc[y1 * src->width + x0] * (1-wx) * wy
                        + sc[y1 * src->width + x1] * wx * wy;
                dc[y * tw + x] = v;
            }
        }
    }
    return dst;
}

// Center crop to target size (must be <= src size)
static nt_image* nt_image_center_crop(const nt_image* src, int tw, int th) {
    if (tw > src->width) tw = src->width;
    if (th > src->height) th = src->height;
    int ox = (src->width - tw) / 2;
    int oy = (src->height - th) / 2;

    nt_image* dst = (nt_image*)calloc(1, sizeof(nt_image));
    dst->width = tw;
    dst->height = th;
    dst->channels = src->channels;
    dst->data = (float*)malloc(src->channels * th * tw * sizeof(float));

    for (int ch = 0; ch < src->channels; ch++) {
        const float* sc = src->data + ch * src->height * src->width;
        float* dc = dst->data + ch * th * tw;
        for (int y = 0; y < th; y++)
            memcpy(dc + y * tw, sc + (oy + y) * src->width + ox, tw * sizeof(float));
    }
    return dst;
}

// Normalize: (pixel - mean) / std, per channel
// mean/std arrays of length channels (e.g. ImageNet: {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})
static void nt_image_normalize(nt_image* img, const float* mean, const float* std) {
    for (int ch = 0; ch < img->channels; ch++) {
        float* c = img->data + ch * img->height * img->width;
        int n = img->height * img->width;
        float m = mean[ch], s = std[ch];
        for (int i = 0; i < n; i++) c[i] = (c[i] - m) / s;
    }
}

// Horizontal flip (in-place)
static void nt_image_hflip(nt_image* img) {
    for (int ch = 0; ch < img->channels; ch++) {
        float* c = img->data + ch * img->height * img->width;
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width / 2; x++) {
                int x2 = img->width - 1 - x;
                float tmp = c[y * img->width + x];
                c[y * img->width + x] = c[y * img->width + x2];
                c[y * img->width + x2] = tmp;
            }
        }
    }
}

// Convert to grayscale (in-place, must be 3-channel)
static void nt_image_to_gray(nt_image* img) {
    if (img->channels != 3) return;
    int n = img->height * img->width;
    float* r = img->data;
    float* g = img->data + n;
    float* b = img->data + 2 * n;
    for (int i = 0; i < n; i++)
        r[i] = 0.2989f * r[i] + 0.5870f * g[i] + 0.1140f * b[i];
    img->channels = 1;
    img->data = (float*)realloc(img->data, n * sizeof(float));
}

// ═══════════════════════════════════════════════════════════════════════════════
// PATCH EXTRACTION — for ViT / VLM models
// ═══════════════════════════════════════════════════════════════════════════════

// Extract non-overlapping patches from image.
// Returns nt_tensor of shape [n_patches, patch_dim] where patch_dim = C * pH * pW
// Image must be divisible by patch size (resize first if needed).
static nt_tensor* nt_image_to_patches(const nt_image* img, int patch_h, int patch_w) {
    int gh = img->height / patch_h;  // grid height
    int gw = img->width / patch_w;   // grid width
    int n_patches = gh * gw;
    int patch_dim = img->channels * patch_h * patch_w;

    nt_tensor* patches = nt_tensor_new2d(n_patches, patch_dim);

    for (int gy = 0; gy < gh; gy++) {
        for (int gx = 0; gx < gw; gx++) {
            int pi = gy * gw + gx;
            float* p = patches->data + pi * patch_dim;
            int di = 0;
            for (int ch = 0; ch < img->channels; ch++) {
                const float* c = img->data + ch * img->height * img->width;
                for (int y = 0; y < patch_h; y++) {
                    for (int x = 0; x < patch_w; x++) {
                        p[di++] = c[(gy * patch_h + y) * img->width + (gx * patch_w + x)];
                    }
                }
            }
        }
    }
    return patches;
}

// Convert image to flat nt_tensor [C * H * W] for direct embedding
static nt_tensor* nt_image_to_tensor(const nt_image* img) {
    int n = img->channels * img->height * img->width;
    nt_tensor* t = nt_tensor_new(n);
    memcpy(t->data, img->data, n * sizeof(float));
    return t;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONVENIENCE — common VLM preprocessing pipelines
// ═══════════════════════════════════════════════════════════════════════════════

// Standard ViT preprocessing: load → resize → center crop → normalize → patches
// Returns [n_patches, patch_dim] tensor. Caller frees.
static nt_tensor* nt_vit_preprocess(const char* path, int img_size, int patch_size) {
    nt_image* img = nt_image_load(path, 3);
    if (!img) return NULL;

    // Resize shortest side to img_size, then center crop
    float scale = (float)img_size / (img->width < img->height ? img->width : img->height);
    int rw = (int)(img->width * scale + 0.5f);
    int rh = (int)(img->height * scale + 0.5f);
    nt_image* resized = nt_image_resize(img, rw, rh);
    nt_image_free(img);

    nt_image* cropped = nt_image_center_crop(resized, img_size, img_size);
    nt_image_free(resized);

    // ImageNet normalization
    float mean[] = {0.485f, 0.456f, 0.406f};
    float std[]  = {0.229f, 0.224f, 0.225f};
    nt_image_normalize(cropped, mean, std);

    nt_tensor* patches = nt_image_to_patches(cropped, patch_size, patch_size);
    nt_image_free(cropped);

    return patches;
}

// Simple grayscale preprocessing for ASCII-style VLMs (like neovlm)
// Returns [H * W] tensor with values in [0, 1]
static nt_tensor* nt_gray_preprocess(const char* path, int size) {
    nt_image* img = nt_image_load(path, 1);
    if (!img) return NULL;
    nt_image* resized = nt_image_resize(img, size, size);
    nt_image_free(img);
    nt_tensor* t = nt_image_to_tensor(resized);
    nt_image_free(resized);
    return t;
}

#endif // NOTORCH_VISION_H
