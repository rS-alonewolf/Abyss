// src/kernels/qsa_rows_parity.cpp - qwen4exp's QSA prompt attention: q35::attn_decode_rows (a group of rows in one
// launch, each over its own selected cells read in place) against the per-row gather_rows + attn_decode it replaces.
// GPU, synthetic, no model: Qwen3.8's shapes (k::qsa_real_shapes(): 24 heads, 2 KV heads, head_dim 256, top-k 2,048 +
// 3), random K / V caches, queries and ascending selections; rows past the dense window (width 2,051) and short rows
// (width < 2,051, the empty-chunk case).  And indexer_key_append_rows against indexer_key_append a cell.  The outputs must be the same bytes; then both are timed for a group of 64.
// Usage: qsa_rows_parity [reps=50]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qwen35_fast.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace k = strata::kernels;
namespace q35 = strata::kernels::q35;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <class T>
T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 256), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
uint16_t f2h(float f) {   // round to nearest even FP16 (values here are small and normal)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int e = (int) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}
}  // namespace

// rows at positions pos0.., a cache of ctx cells; true: the same bytes
bool run(int ctx, int64_t pos0, int rows, int reps, std::mt19937& rng) {
    const k::QsaShapes sh = k::qsa_real_shapes();
    const int NH = (int) sh.n_head, NKV = (int) sh.n_head_kv, D = (int) sh.head_dim, KV = NKV * D, HD = NH * D;
    const int cap = (int) k::qsa_selection_width(k::kTopkMaxCells, sh);
    const float scale = 1.0f / std::sqrt((float) D);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> kc((size_t) ctx * KV), vc((size_t) ctx * KV);
    for (auto& x : kc) x = f2h(nd(rng) * 0.5f);
    for (auto& x : vc) x = f2h(nd(rng));
    std::vector<float> q((size_t) rows * HD);
    for (auto& x : q) x = nd(rng) * 1.5f;
    std::vector<int32_t> ids((size_t) rows * cap, 0), steps((size_t) rows * 4);
    std::vector<int> width(rows);
    std::vector<int32_t> all;
    for (int r = 0; r < rows; ++r) {
        const int64_t pos = pos0 + r;
        k::qsa_step_fill(steps.data() + (size_t) r * 4, pos, sh);
        const int w = steps[(size_t) r * 4 + k::kStepWidth];
        width[r] = w;
        all.resize((size_t) pos + 1);
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        std::sort(all.begin(), all.begin() + w);   // a random ascending selection of w cells of pos + 1
        std::copy(all.begin(), all.begin() + w, ids.begin() + (size_t) r * cap);
    }
    uint16_t *dkc = up(kc), *dvc = up(vc), *ks, *vs;
    float* dq = up(q);
    int32_t *dids = up(ids), *dst = up(steps);
    float *oa, *ob, *part_a, *part_b;
    ck(cudaMalloc(&ks, (size_t) cap * KV * 2), "m");
    ck(cudaMalloc(&vs, (size_t) cap * KV * 2), "m");
    ck(cudaMalloc(&oa, (size_t) rows * HD * 4), "m");
    ck(cudaMalloc(&ob, (size_t) rows * HD * 4), "m");
    ck(cudaMalloc(&part_a, q35::attn_part_floats(NH, D, ctx) * 4), "m");
    ck(cudaMalloc(&part_b, q35::attn_rows_part_floats(rows, NH, D, cap) * 4), "m");
    ck(cudaMemset(oa, 0xff, (size_t) rows * HD * 4), "set");
    ck(cudaMemset(ob, 0x7f, (size_t) rows * HD * 4), "set");
    cudaStream_t s = nullptr;
    const auto old_path = [&] {
        for (int r = 0; r < rows; ++r) {
            q35::gather_rows(dkc, dvc, dids + (size_t) r * cap, width[r], KV, ks, vs, s);
            q35::attn_decode(dq + (size_t) r * HD, ks, vs, NH, NKV, D, width[r], scale, part_a, oa + (size_t) r * HD, s);
        }
    };
    bool launched = true;
    const auto new_path = [&] {
        launched = q35::attn_decode_rows(dq, dkc, dvc, dids, cap, dst, rows, cap, NH, NKV, D, scale, part_b, ob, s);
    };
    old_path();
    new_path();
    ck(cudaDeviceSynchronize(), "sync");
    if (!launched) { std::printf("attn_decode_rows did not take the shape\n"); return false; }
    std::vector<float> ha((size_t) rows * HD), hb((size_t) rows * HD);
    ck(cudaMemcpy(ha.data(), oa, ha.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(hb.data(), ob, hb.size() * 4, cudaMemcpyDeviceToHost), "down");
    size_t diff = 0;
    double maxd = 0.0;
    for (size_t i = 0; i < ha.size(); ++i) {
        if (std::memcmp(&ha[i], &hb[i], 4) != 0) ++diff;
        maxd = std::max(maxd, (double) std::fabs(ha[i] - hb[i]));
    }
    std::printf("ctx %d, rows %d at %lld.. (widths %d..%d): %zu of %zu floats differ (max |d| %.3g) %s\n", ctx, rows,
                (long long) pos0, width.front(), width.back(), diff, ha.size(), maxd, diff == 0 ? "SAME" : "DIFFERENT");
    if (reps > 0) {
        const auto time = [&](auto&& f) {
            f();
            ck(cudaDeviceSynchronize(), "sync");
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) f();
            ck(cudaDeviceSynchronize(), "sync");
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        const double ta = time(old_path), tb = time(new_path);
        std::printf("  a group: per-row %.1f us, batched %.1f us (%.1fx)\n", ta, tb, ta / tb);
    }
    cudaFree(dkc); cudaFree(dvc); cudaFree(ks); cudaFree(vs); cudaFree(dq); cudaFree(dids); cudaFree(dst);
    cudaFree(oa); cudaFree(ob); cudaFree(part_a); cudaFree(part_b);
    return diff == 0;
}

// indexer_key_append_rows (two launches a run) against indexer_key_append a cell, over runs of cells that start and end
// on and off block boundaries (from cell 0, so the spare key's first write too); after every run the tail, the spare
// key, the pooled rows and block_pos must be the same bytes
bool run_append(std::mt19937& rng) {
    const k::QsaShapes sh = k::qsa_real_shapes();
    const int ID = (int) sh.idx_dim, half = (int) sh.n_rot / 2, R = (int) sh.idx_block;
    const int runs[] = {1, 2, 1, 3, 5, 7, 4, 4, 1024, 13, 1, 2, 999, 64, 3, 1021, 6, 8};
    int N = 0;
    for (int n : runs) N += n;
    const int max_blocks = N / R + 2;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const auto rnd = [&](size_t n, float sc, float off = 0.0f) {
        std::vector<float> v(n);
        for (auto& x : v) x = nd(rng) * sc + off;
        return v;
    };
    const std::vector<float> raw = rnd((size_t) N * ID, 1.0f), wkn = rnd(ID, 0.2f, 1.0f), cs = rnd((size_t) N * half, 0.5f),
                             sn = rnd((size_t) N * half, 0.5f);
    std::vector<int32_t> pos(N);
    std::iota(pos.begin(), pos.end(), 0);
    float *draw = up(raw), *dw = up(wkn), *dcos = up(cs), *dsin = up(sn);
    int32_t* dpos = up(pos);
    const size_t nt = (size_t) (R - 1) * ID, np = (size_t) max_blocks * ID;
    const std::vector<float> init_t = rnd(nt, 3.0f), init_d = rnd(ID, 3.0f), init_p = rnd(np, 3.0f);
    const std::vector<int32_t> init_b = {12345};
    k::QsaIndexerBuffers A{up(init_t), up(init_d), up(init_p), up(init_b)}, B{up(init_t), up(init_d), up(init_p), up(init_b)};
    const auto same = [&](const void* a, const void* b, size_t bytes) {
        std::vector<uint8_t> ha(bytes), hb(bytes);
        ck(cudaMemcpy(ha.data(), a, bytes, cudaMemcpyDeviceToHost), "down");
        ck(cudaMemcpy(hb.data(), b, bytes, cudaMemcpyDeviceToHost), "down");
        return std::memcmp(ha.data(), hb.data(), bytes) == 0;
    };
    bool ok = true;
    int p0 = 0;
    for (int n : runs) {
        for (int t = 0; t < n; ++t)
            k::indexer_key_append(draw + (size_t) (p0 + t) * ID, dpos + p0 + t, 0, dw, 1e-6f, A, sh, dcos, dsin, nullptr);
        k::indexer_key_append_rows(draw + (size_t) p0 * ID, ID, p0, n, 0, dw, 1e-6f, B, sh, dcos, dsin, nullptr);
        ck(cudaDeviceSynchronize(), "sync");
        const bool st = same(A.tail, B.tail, nt * 4), sd = same(A.dead, B.dead, ID * 4),
                   sp = same(A.pooled, B.pooled, np * 4), sb = same(A.block_pos, B.block_pos, 4);
        if (!(st && sd && sp && sb)) {
            std::printf("append run %d..%d: tail %s, dead %s, pooled %s, block_pos %s\n", p0, p0 + n - 1,
                        st ? "same" : "DIFF", sd ? "same" : "DIFF", sp ? "same" : "DIFF", sb ? "same" : "DIFF");
            ok = false;
        }
        p0 += n;
    }
    std::printf("indexer key append over %zu runs, %d cells: %s\n", sizeof(runs) / sizeof(runs[0]), N,
                ok ? "SAME" : "DIFFERENT");
    for (auto* b : {&A, &B}) { cudaFree(b->tail); cudaFree(b->dead); cudaFree(b->pooled); cudaFree(b->block_pos); }
    cudaFree(draw); cudaFree(dw); cudaFree(dcos); cudaFree(dsin); cudaFree(dpos);
    return ok;
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 50;
    std::mt19937 rng(7);
    bool ok = run_append(rng);
    ok = run(8192, 2051, 64, reps, rng) && ok;     // the first rows past the dense window
    ok = run(8192, 6000, 64, reps, rng) && ok;
    ok = run(32768, 30000, 64, reps, rng) && ok;
    ok = run(32768, 12345, 37, 0, rng) && ok;      // a short group
    ok = run(4096, 1990, 64, 0, rng) && ok;        // widths below 2,051: rows end in empty chunks
    ok = run(4096, 3, 20, 0, rng) && ok;           // tiny widths
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
