// qwen35moe's decode and verify-window attention (q35::attn_decode / attn_window: the keys in chunks, then merged)
// against the one-pass attn_prompt, on random FP16 caches: a token and windows of 2-8 rows at short and long
// positions, and their times per layer up to 115K positions (32 heads over 2 KV heads, head_dim 256).
// STRATA_Q35_ATTN=1 times the older kernels.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "strata/kernels/qwen35_fast.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#define CHECK(call) do { const auto e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); return 2; } } while (0)

namespace q35 = strata::kernels::q35;

int main() {
    constexpr int kH = 32, kHkv = 2, kD = 256, kCtx = 131072, kRows = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<__half> kc((size_t) kCtx * kHkv * kD), vc(kc.size());
    for (auto& x : kc) x = __float2half(nd(rng));
    for (auto& x : vc) x = __float2half(nd(rng));
    std::vector<float> q((size_t) kRows * kH * kD);
    for (auto& x : q) x = nd(rng);

    __half *dk = nullptr, *dv = nullptr;
    float *dq = nullptr, *da = nullptr, *db = nullptr, *dpart = nullptr;
    const size_t part_n = std::max(q35::attn_window_floats(kH, kD, kCtx, kRows), q35::attn_part_floats(kH, kD, kCtx));
    CHECK(hipMalloc(&dk, kc.size() * 2));
    CHECK(hipMalloc(&dv, vc.size() * 2));
    CHECK(hipMalloc(&dq, q.size() * 4));
    CHECK(hipMalloc(&da, q.size() * 4));
    CHECK(hipMalloc(&db, q.size() * 4));
    CHECK(hipMalloc(&dpart, part_n * 4));
    CHECK(hipMemcpy(dk, kc.data(), kc.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dv, vc.data(), vc.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice));
    const auto* k16 = reinterpret_cast<const uint16_t*>(dk);
    const auto* v16 = reinterpret_cast<const uint16_t*>(dv);
    const float scale = 1.0f / std::sqrt((float) kD);

    int bad = 0;
    float worst_all = 0.0f;
    for (const int64_t p0 : {0LL, 1LL, 31LL, 127LL, 128LL, 255LL, 256LL, 1000LL, 5000LL, 20000LL, 40000LL}) {
        for (int T = 1; T <= kRows; ++T) {
            if (p0 + T > kCtx) continue;
            q35::attn_prompt(dq, k16, v16, kH, kHkv, kD, p0, T, scale, da, nullptr);
            if (T == 1) q35::attn_decode(dq, k16, v16, kH, kHkv, kD, (int) p0 + 1, scale, dpart, db, nullptr);
            else q35::attn_window(dq, k16, v16, kH, kHkv, kD, p0, T, scale, dpart, db, nullptr);
            CHECK(hipDeviceSynchronize());
            std::vector<float> a((size_t) T * kH * kD), b(a.size());
            CHECK(hipMemcpy(a.data(), da, a.size() * 4, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(b.data(), db, b.size() * 4, hipMemcpyDeviceToHost));
            float worst = 0.0f;
            for (size_t i = 0; i < a.size(); ++i)
                worst = std::max(worst, std::fabs(a[i] - b[i]) / std::max(1e-3f, std::fabs(a[i])));
            worst_all = std::max(worst_all, worst);
            if (!(worst < 1e-3f)) {
                std::printf("p0 %lld T %d: relative difference %.3g\n", (long long) p0, T, worst);
                ++bad;
            }
        }
    }
    std::printf("parity: %s, the largest relative difference %.3g\n",
                bad ? "FAIL" : "ok (a token and windows of 2-8 rows, positions 0-40000)", worst_all);

    // times at the positions a long answer reaches: a single token (attn_decode) and windows of 2-4 rows
    for (const int64_t p0 : {300LL, 2000LL, 20000LL, 57600LL, 115000LL}) {
        std::printf("p0 %6lld:", (long long) p0);
        for (int T = 1; T <= 4; ++T) {
            auto run = [&] {
                if (T == 1) q35::attn_decode(dq, k16, v16, kH, kHkv, kD, (int) p0 + 1, scale, dpart, db, nullptr);
                else q35::attn_window(dq, k16, v16, kH, kHkv, kD, p0, T, scale, dpart, db, nullptr);
            };
            run();
            CHECK(hipDeviceSynchronize());
            const int reps = 20;
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) run();
            CHECK(hipDeviceSynchronize());
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
            const double gb = (double) (p0 + T) * kHkv * kD * 2 * 2 / 1e9;   // K + V read once
            std::printf("  %s %.3f ms (%.0f GB/s)", T == 1 ? "decode" : (T == 2 ? "win2" : (T == 3 ? "win3" : "win4")), ms,
                        gb / ms * 1e3);
        }
        std::printf("  (a layer)\n");
    }
    hipFree(dk);
    hipFree(dv);
    hipFree(dq);
    hipFree(da);
    hipFree(db);
    hipFree(dpart);
    return bad ? 1 : 0;
}
