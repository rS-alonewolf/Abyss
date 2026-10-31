// src/kernels/hc4_fused_parity.cpp - qwen4exp's hyper-connection mix: q35::hc4_fused (two launches) against the five
// launches it replaces (hc4_combine_norm, gf::mv down + inject, silu_scale, gf::mv up, hc4_mix) - GPU, synthetic, no
// model.  Qwen3.8-Flash-Next's shapes (4 streams x 2560, low rank 320, BF16 down / up, F32 or BF16 inject), random
// residuals, weights and norms, with and without the combine's `out` and the inject.  Every output must be the same
// bytes (R, inject, xn, gate, the mix and its q8_1); then both are timed over a token's 97 mixes.
// Usage: hc4_fused_parity [reps=200]
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
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 256), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<uint8_t> down(const T* d, size_t n) {
    std::vector<uint8_t> h(n * sizeof(T));
    ck(cudaMemcpy(h.data(), d, h.size(), cudaMemcpyDeviceToHost), "download");
    return h;
}
uint16_t bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) (u >> 16); }
}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
    const int E = 2560, HC = 4, HE = HC * E, LR = 320;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const auto rnd = [&](size_t n, float sc) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * sc; return v; };
    const auto rbf = [&](size_t n, float sc) { std::vector<uint16_t> v(n); for (auto& x : v) x = bf16(nd(rng) * sc); return v; };
    const std::vector<float> R0 = rnd(HE, 1.0f), outv = rnd(E, 0.5f), injp = rnd(HC, 1.0f), gam = rnd(HE, 0.3f);
    uint16_t* dW = up(rbf((size_t) LR * HE, 0.02f));
    uint16_t* uW = up(rbf((size_t) HE * LR, 0.05f));
    float* iWf = up(rnd((size_t) HC * HE, 0.01f));
    uint16_t* iWb = up(rbf((size_t) HC * HE, 0.01f));
    float *R = up(R0), *out = up(outv), *inj_prev = up(injp), *gamma = up(gam);
    float *xn, *lo, *gate, *x, *inj, *sc;
    void *xnq, *loq, *xq;
    ck(cudaMalloc(&xn, HE * 4), "m"); ck(cudaMalloc(&lo, 4096), "m"); ck(cudaMalloc(&gate, HE * 4), "m");
    ck(cudaMalloc(&x, E * 4), "m"); ck(cudaMalloc(&inj, 64), "m"); ck(cudaMalloc(&sc, 64), "m");
    ck(cudaMalloc(&xnq, HE / 32 * 36), "m"); ck(cudaMalloc(&loq, LR / 32 * 36), "m"); ck(cudaMalloc(&xq, E / 32 * 36), "m");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    const float eps = 1e-6f;
    const auto five = [&](bool with_out, int itype) {
        const void* iw = itype == 0 ? (const void*) iWf : (const void*) iWb;
        q35::hc4_combine_norm(R, with_out ? out : nullptr, inj_prev, HC, gamma, eps, E, HC, 1, xn, s, xnq);
        gf::MvJob j[2];
        j[0] = {dW, xnq, xn, lo, nullptr, 1.0f, gf::kTypeBF16, HE, LR};
        j[1] = {iw, xnq, xn, inj, nullptr, 1.0f, itype, HE, HC};
        gf::mv(j, itype >= 0 ? 2 : 1, s);
        q35::silu_scale(lo, LR, 1.0f / (float) HC, s, loq);
        gf::MvJob u = {uW, loq, lo, gate, nullptr, 1.0f, gf::kTypeBF16, LR, HE};
        gf::mv(&u, 1, s);
        q35::hc4_mix(xn, gate, E, HC, 1, x, s, xq);
    };
    const auto two = [&](bool with_out, int itype) {
        const void* iw = itype < 0 ? nullptr : itype == 0 ? (const void*) iWf : (const void*) iWb;
        q35::hc4_fused(R, with_out ? out : nullptr, inj_prev, gamma, eps, E, HC, LR, dW, iw, uW, lo, inj, sc, xn, gate, x,
                       xq, s, itype == 0);
    };
    if (!q35::hc4_fused_ok(E, HC, LR, gf::kTypeBF16, gf::kTypeBF16, gf::kTypeF32)) { std::fprintf(stderr, "shapes\n"); return 2; }
    int bad = 0;
    for (int with_out = 0; with_out < 2; ++with_out)
        for (int itype : {gf::kTypeF32, gf::kTypeBF16, -1}) {
            std::vector<uint8_t> res[2][6];
            for (int v = 0; v < 2; ++v) {
                ck(cudaMemcpy(R, R0.data(), HE * 4, cudaMemcpyHostToDevice), "R");
                ck(cudaMemset(inj, 0, 64), "inj");
                v == 0 ? five(with_out, itype) : two(with_out, itype);
                ck(cudaStreamSynchronize(s), "run");
                res[v][0] = down(R, HE); res[v][1] = down(inj, HC); res[v][2] = down(xn, HE);
                res[v][3] = down(gate, HE); res[v][4] = down(x, E); res[v][5] = down((uint8_t*) xq, E / 32 * 36);
            }
            const char* names[6] = {"R", "inject", "xn", "gate", "mix", "mix q8_1"};
            for (int k = 0; k < 6; ++k)
                if (res[0][k] != res[1][k]) {
                    std::printf("out=%d inject type %d: %s differs\n", with_out, itype, names[k]);
                    ++bad;
                }
        }
    // a token's 97 mixes, each way, each mix its own weights (1.3 GB: read from memory, not L2, as in the engine)
    std::vector<uint16_t*> dWs(97), uWs(97);
    for (int m = 0; m < 97; ++m) {
        ck(cudaMalloc(&dWs[m], (size_t) LR * HE * 2), "m");
        ck(cudaMalloc(&uWs[m], (size_t) LR * HE * 2), "m");
        ck(cudaMemcpy(dWs[m], dW, (size_t) LR * HE * 2, cudaMemcpyDeviceToDevice), "c");
        ck(cudaMemcpy(uWs[m], uW, (size_t) LR * HE * 2, cudaMemcpyDeviceToDevice), "c");
    }
    for (int v = 0; v < 2; ++v) {
        ck(cudaStreamSynchronize(s), "sync");
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            for (int m = 0; m < 97; ++m) {
                dW = dWs[m];
                uW = uWs[m];
                v == 0 ? five(true, gf::kTypeF32) : two(true, gf::kTypeF32);
            }
        ck(cudaStreamSynchronize(s), "sync");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("%s: %.3f ms a token's 97 mixes (%.1f us a mix)\n", v == 0 ? "five launches" : "hc4_fused", ms,
                    ms * 1000.0 / 97.0);
    }
    std::printf(bad == 0 ? "hc4_fused parity: OK (every output the same bytes)\n" : "hc4_fused parity: %d outputs differ\n", bad);
    return bad == 0 ? 0 : 1;
}
