// src/kernels/gdn_core_parity.cpp - qwen35 / qwen4exp's delta-net middle: q35::gdn_core (one launch) against the launches
// it replaces - token path: gdn_prep_gates + gdn_scan + gdn_out_norm + quantize_q8_1 (prep in the kernel); window path:
// gdn_scan + gdn_out_norm + quantize_q8_1 after gdn_prep_gates (T rows).  GPU, synthetic, no model: Qwen3.8's shapes
// (S 128, 16 k heads, 48 v heads; 16 / 32 too), random rows, gates, states and norms.  The state, y and y's q8_1 must be the same
// bytes; then both are timed over a token's 36 delta-net layers.
// Usage: gdn_core_parity [reps=200]
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/qwen35_fast.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace q35 = strata::kernels::q35;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
float* up(const std::vector<float>& h) {
    float* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(float) + 256), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), "upload");
    return d;
}
std::vector<uint8_t> down(const void* d, size_t bytes) {
    std::vector<uint8_t> h(bytes);
    ck(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost), "download");
    return h;
}
}  // namespace

int run(int HK, int HV, int reps) {
    std::printf("h_k %d, h_v %d\n", HK, HV);
    const int S = 128, C = 2 * HK * S + HV * S, V = HV * S, TM = 8;
    const float eps = 1e-6f;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const auto rnd = [&](size_t n, float sc, float off = 0.0f) {
        std::vector<float> v(n);
        for (auto& x : v) x = nd(rng) * sc + off;
        return v;
    };
    const std::vector<float> h0 = rnd((size_t) TM * C, 1.0f), al0 = rnd((size_t) TM * HV, 1.0f),
                             be0 = rnd((size_t) TM * HV, 1.0f), st0 = rnd((size_t) S * HV * S, 0.05f),
                             z0 = rnd((size_t) TM * V, 1.0f), dt0 = rnd(HV, 0.5f), A0 = rnd(HV, 0.3f, -1.0f),
                             nrm0 = rnd(S, 0.2f, 1.0f);
    float *h = up(h0), *al = up(al0), *be = up(be0), *st = up(st0), *st0d = st, *z = up(z0), *dt = up(dt0), *A = up(A0), *nrm = up(nrm0);
    float *o, *y;
    void* yq;
    ck(cudaMalloc(&o, (size_t) TM * V * 4), "m");
    ck(cudaMalloc(&y, (size_t) TM * V * 4), "m");
    ck(cudaMalloc(&yq, (size_t) TM * V / 32 * 36), "m");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    const auto reset = [&] {
        ck(cudaMemcpy(h, h0.data(), h0.size() * 4, cudaMemcpyHostToDevice), "h");
        ck(cudaMemcpy(al, al0.data(), al0.size() * 4, cudaMemcpyHostToDevice), "a");
        ck(cudaMemcpy(be, be0.data(), be0.size() * 4, cudaMemcpyHostToDevice), "b");
        ck(cudaMemcpy(st, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "s");
        ck(cudaMemset(y, 0, (size_t) TM * V * 4), "y");
        ck(cudaMemset(yq, 0, (size_t) TM * V / 32 * 36), "yq");
    };
    // token path: prep in the kernel (T = 1), or (window) prep first then the kernel, T rows
    // (the out norm's eps differs from the L2 norm's on purpose: the kernel must not mix them up)
    const float out_eps = 1e-5f;
    const auto old_way = [&](int T, bool token) {
        q35::gdn_prep_gates(h, T, C, HK, S, eps, al, be, dt, A, HV, s);
        q35::gdn_scan(st, h, T, C, al, be, S, HK, HV, o, s);
        strata::kernels::gdn_out_norm(o, z, nrm, y, (int64_t) T * HV, S, out_eps, (void*) s);
        gf::quantize_q8_1(y, yq, T * V, s);
        (void) token;
    };
    const auto new_way = [&](int T, bool token) {
        if (!token) q35::gdn_prep_gates(h, T, C, HK, S, eps, al, be, dt, A, HV, s);
        q35::gdn_core(st, h, T, C, al, be, S, HK, HV, token, eps, dt, A, z, nrm, out_eps, y, yq, s);
    };
    int bad = 0;
    for (int token = 1; token >= 0; --token)
        for (int T = 1; T <= TM; ++T) {   // (prep in the kernel: the token path's T 1, more rows checked too)
            std::vector<uint8_t> r[2][3];
            for (int v = 0; v < 2; ++v) {
                reset();
                v == 0 ? old_way(T, token) : new_way(T, token);
                ck(cudaStreamSynchronize(s), "run");
                r[v][0] = down(st, st0.size() * 4);
                r[v][1] = down(y, (size_t) T * V * 4);
                r[v][2] = down(yq, (size_t) T * V / 32 * 36);
            }
            const char* names[3] = {"state", "y", "y q8_1"};
            for (int k = 0; k < 3; ++k)
                if (r[0][k] != r[1][k]) {
                    std::printf("%s T=%d: %s differs\n", token ? "token" : "window", T, names[k]);
                    ++bad;
                }
        }
    // a token's 36 delta-net layers, each its own state (36 x 8 MB: from memory, as in the engine)
    std::vector<float*> sts(36);
    for (auto& p : sts) {
        ck(cudaMalloc(&p, st0.size() * 4), "m");
        ck(cudaMemcpy(p, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "c");
    }
    for (int v = 0; v < 2; ++v) {
        ck(cudaStreamSynchronize(s), "sync");
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            for (int l = 0; l < 36; ++l) {
                st = sts[(size_t) l];
                v == 0 ? old_way(1, true) : new_way(1, true);
            }
        ck(cudaStreamSynchronize(s), "sync");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("%s: %.3f ms a token's 36 layers (%.1f us a layer)\n", v == 0 ? "four launches" : "gdn_core", ms,
                    ms * 1000.0 / 36.0);
    }
    for (auto p : sts) cudaFree(p);
    for (float* p : {h, al, be, st0d, z, dt, A, nrm, o, y}) cudaFree(p);
    cudaFree(yq);
    cudaStreamDestroy(s);
    return bad;
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
    int bad = 0;
    bad += run(16, 48, reps);   // Qwen3.8-Flash-Next (qwen4exp)
    bad += run(16, 32, reps);
    std::printf(bad == 0 ? "gdn_core parity: OK (every output the same bytes)\n" : "gdn_core parity: %d outputs differ\n", bad);
    return bad == 0 ? 0 : 1;
}
