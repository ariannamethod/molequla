/* cmp_f32 — compare two f32 dumps and fail on a tolerance.
 *
 *   cmp_f32 <a.f32> <b.f32> <n_header_i32> <tol> [max_values]
 *
 * Both files start with `n_header_i32` int32 fields, which must be equal, followed
 * by floats. With `max_values` given, only the first that many floats are read —
 * the mel gate compares the first 3000 frames, not the 30 s of padding behind them.
 *
 * Prints max abs diff, where it was, and the two values there, then exits 0 under
 * the tolerance and 1 over it. A gate that cannot print the number it compared is
 * not a gate.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float *slurp(const char *path, int nh, int *hdr, long *n_float, long cap) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cmp_f32: cannot open %s\n", path); return NULL; }
    for (int i = 0; i < nh; i++)
        if (fread(&hdr[i], sizeof(int), 1, f) != 1) {
            fprintf(stderr, "cmp_f32: %s is shorter than its header\n", path);
            fclose(f); return NULL;
        }
    long start = ftell(f);
    fseek(f, 0, SEEK_END);
    long n = (ftell(f) - start) / (long) sizeof(float);
    fseek(f, start, SEEK_SET);
    if (cap > 0 && n > cap) n = cap;
    float *d = (float *) malloc((size_t) n * sizeof(float));
    if (!d || fread(d, sizeof(float), (size_t) n, f) != (size_t) n) {
        fprintf(stderr, "cmp_f32: short read on %s\n", path);
        free(d); fclose(f); return NULL;
    }
    fclose(f);
    *n_float = n;
    return d;
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 6) {
        fprintf(stderr, "usage: cmp_f32 <a.f32> <b.f32> <n_header_i32> <tol> [max_values]\n");
        return 2;
    }
    const int nh = atoi(argv[3]);
    const double tol = atof(argv[4]);
    const long cap = (argc == 6) ? atol(argv[5]) : 0;
    if (nh < 0 || nh > 8) { fprintf(stderr, "cmp_f32: silly header count\n"); return 2; }

    int ha[8] = {0}, hb[8] = {0};
    long na = 0, nb = 0;
    float *a = slurp(argv[1], nh, ha, &na, cap);
    float *b = slurp(argv[2], nh, hb, &nb, cap);
    if (!a || !b) { free(a); free(b); return 2; }

    for (int i = 0; i < nh; i++)
        if (ha[i] != hb[i]) {
            fprintf(stderr, "cmp_f32: FAIL header field %d differs: %d vs %d\n", i, ha[i], hb[i]);
            free(a); free(b); return 1;
        }
    if (na != nb) {
        fprintf(stderr, "cmp_f32: FAIL element count differs: %ld vs %ld\n", na, nb);
        free(a); free(b); return 1;
    }

    double worst = 0.0, sum = 0.0;
    long at = -1, n_nan = 0;
    for (long i = 0; i < na; i++) {
        if (isnan(a[i]) || isnan(b[i])) { n_nan++; continue; }
        double d = fabs((double) a[i] - (double) b[i]);
        sum += d;
        if (d > worst) { worst = d; at = i; }
    }

    printf("cmp_f32: %ld values  max|d| = %.3e at %ld (%.6f vs %.6f)  mean|d| = %.3e\n",
           na, worst, at, at >= 0 ? a[at] : 0.0f, at >= 0 ? b[at] : 0.0f, na ? sum / na : 0.0);
    if (n_nan) printf("cmp_f32: %ld NaN pairs skipped\n", n_nan);

    const int ok = (worst <= tol) && n_nan == 0;
    printf("cmp_f32: %s (tolerance %.3e)\n", ok ? "PASS" : "FAIL", tol);
    free(a); free(b);
    return ok ? 0 : 1;
}
