// See shard_io.hpp.  Moved out of glm_fast_path.cu (read_slice) in v1.3.78 so the file layer has a host-only test.
#include "shard_io.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace strata::core::shard_io {

std::atomic<uint64_t> mirror_bytes[2];
std::atomic<uint64_t> fallback_buffered{0}, fallback_mapped{0};

namespace {
std::atomic<bool> mirror_off{false};
std::atomic<int> mirror_inflight[2];
std::atomic<float> mirror_speed[2] = {1.0f, 1.0f};   // bytes / us, a running average
std::atomic<uint32_t> mirror_reads{0};

// The 2nd SSD, best effort: every read goes to the drive that should finish it first - the one with the least in
// flight for its measured speed (a running average of bytes per microsecond a read, both drives start equal) - so two
// drives of any speeds add up and a slow or busy one simply takes less.
int mirror_pick() {
    const float c0 = (float) (mirror_inflight[0].load(std::memory_order_relaxed) + 1) / mirror_speed[0].load(),
                c1 = (float) (mirror_inflight[1].load(std::memory_order_relaxed) + 1) / mirror_speed[1].load();
    const int best = c1 < c0 ? 1 : 0;
    // one read in 32 goes to the other drive: a drive that looked slow once (a busy spell, the warm-up) gets measured
    // again - without it, its stale speed kept it unpicked for the rest of the run (0.0% from the mirror, 2026-10-29)
    return (mirror_reads.fetch_add(1, std::memory_order_relaxed) & 31) == 31 ? 1 - best : best;
}
}  // namespace

bool mirror_on() { return !mirror_off.load(std::memory_order_relaxed); }

void reset_stats() {
    mirror_bytes[0] = mirror_bytes[1] = 0;
    fallback_buffered = fallback_mapped = 0;
    mirror_off = false;
    mirror_inflight[0] = mirror_inflight[1] = 0;
    mirror_speed[0] = mirror_speed[1] = 1.0f;
    mirror_reads = 0;
}

#ifndef _WIN32

static ssize_t pread_sys(int fd, void* buf, size_t n, off_t off) { return ::pread(fd, buf, n, off); }
ssize_t (*pread_fn)(int, void*, size_t, off_t) = pread_sys;

ssize_t pread_full(int fd, void* buf, size_t n, uint64_t off) {
    size_t got = 0;
    while (got < n) {
        const ssize_t r = pread_fn(fd, (uint8_t*) buf + got, n - got, (off_t) (off + got));
        if (r < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;   // a signal (a profiler's, a debugger's): the same read again
            return got > 0 ? (ssize_t) got : -1;
        }
        if (r == 0) break;   // the end of the file
        got += (size_t) r;
    }
    return (ssize_t) got;
}

bool read_direct(int fd_direct, int fd_mirror, uint64_t off, size_t len, uint8_t* dst) {
    // the bounce buffer: freed when its thread ends
    struct Bounce {
        uint8_t* p = nullptr;
        size_t cap = 0;
        ~Bounce() { std::free(p); }
    };
    static thread_local Bounce bb;
    const uint64_t a0 = off & ~(uint64_t) 4095, a1 = (off + len + 4095) & ~(uint64_t) 4095;
    const size_t need = (size_t) (a1 - a0);
    if (bb.cap < need) {
        std::free(bb.p);
        bb.p = (uint8_t*) std::aligned_alloc(4096, need);
        bb.cap = bb.p ? need : 0;
    }
    uint8_t* bounce = bb.p;
    if (bounce == nullptr) return false;
    const bool two = fd_mirror >= 0 && mirror_on();
    for (int pass = two && mirror_pick() == 1 ? 0 : 1; pass < 2; ++pass) {
        const int fdr = pass == 0 ? fd_mirror : fd_direct, d = pass == 0 ? 1 : 0;
        if (two) mirror_inflight[d].fetch_add(1, std::memory_order_relaxed);
        const auto t0 = std::chrono::steady_clock::now();
        const ssize_t r = pread_full(fdr, bounce, need, a0);   // short where the file ends (the last aligned read)
        const size_t got = r > 0 ? (size_t) r : 0;
        if (two) {
            mirror_inflight[d].fetch_sub(1, std::memory_order_relaxed);
            if (got > 0) {
                const float us = (float) std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - t0).count() + 1.0f;
                // per read in flight: the drive's speed while it was shared with the others in flight
                const float sp = (float) got / us * (float) (mirror_inflight[d].load(std::memory_order_relaxed) + 1);
                mirror_speed[d].store(0.9f * mirror_speed[d].load() + 0.1f * sp, std::memory_order_relaxed);
            }
        }
        if (got >= (size_t) (off - a0) + len) {
            std::memcpy(dst, bounce + (off - a0), len);
            mirror_bytes[d].fetch_add(need, std::memory_order_relaxed);
            return true;
        }
        if (pass == 0 && !mirror_off.exchange(true))
            std::fprintf(stderr, "[mirror] read failed at %llu (%s): the mirror is off, reading from the shards' drive\n",
                         (unsigned long long) off, r < 0 ? std::strerror(errno) : "short read");
    }
    return false;
}

bool read_buffered(int fd, const uint8_t* base, uint64_t off, size_t len, uint8_t* dst) {
    fallback_buffered.fetch_add(1, std::memory_order_relaxed);
    const ssize_t r = fd >= 0 ? pread_full(fd, dst, len, off) : -1;
    if (r == (ssize_t) len) return true;
    if (base == nullptr) return false;   // the views went after the load (pack_release_views): nothing to fall back to
    fallback_mapped.fetch_add(1, std::memory_order_relaxed);
    std::memcpy(dst, base + off, len);
    return true;
}

bool same_content(int fd, int mf, uint64_t size, std::string& why) {
    constexpr uint64_t kMiB = 1u << 20, kBlock = 256u << 10;
    constexpr int kSamples = 32;
    std::vector<uint8_t> x(kMiB), y(kMiB);
    const auto same_at = [&](uint64_t at, size_t n) {
        n = (size_t) std::min<uint64_t>(n, size - at);
        if (n == 0) return true;
        if (pread_full(fd, x.data(), n, at) != (ssize_t) n) { why = "the shard could not be read at " + std::to_string(at); return false; }
        if (pread_full(mf, y.data(), n, at) != (ssize_t) n) { why = "the copy could not be read at " + std::to_string(at); return false; }
        if (std::memcmp(x.data(), y.data(), n) != 0) { why = "the bytes differ at " + std::to_string(at); return false; }
        return true;
    };
    if (!same_at(0, kMiB)) return false;
    if (size > kMiB && !same_at(size - kMiB, kMiB)) return false;
    if (size > 2 * kMiB + kBlock) {
        const uint64_t lo = kMiB, span = size - 2 * kMiB - kBlock;   // (between the head and the tail already compared)
        for (int i = 0; i < kSamples; ++i)
            if (!same_at(lo + span * (uint64_t) i / (kSamples - 1), kBlock)) return false;
    }
    return true;
}

#endif  // !_WIN32

}  // namespace strata::core::shard_io
