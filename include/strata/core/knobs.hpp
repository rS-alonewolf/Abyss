// Small pure helpers behind the engine's STRATA_* knobs and bounds, kept free of CUDA so a host-only test can run them
// (tools/test_knobs.py compiles this header with the system C++ compiler).
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

// A numeric STRATA_* setting: the default when it is unset or not a number, else its value clamped to [lo, hi]; a note
// on stderr the first time a value had to be changed, so a typo (`STRATA_GLM_RAM_FRAC=1,0`) or an out-of-range value
// (a negative pool, 200 Engram threads) is seen instead of silently becoming 0 or unbounded.
inline double env_num(const char* name, double def, double lo, double hi, const char* raw = nullptr, bool* changed = nullptr) {
    if (raw == nullptr) raw = std::getenv(name);
    if (changed != nullptr) *changed = false;
    if (raw == nullptr || *raw == 0) return def;
    char* end = nullptr;
    const double v = std::strtod(raw, &end);
    if (end == raw) {
        if (changed != nullptr) *changed = true;
        std::fprintf(stderr, "%s=%s is not a number: using %g\n", name, raw, def);
        return def;
    }
    const double c = std::max(lo, std::min(hi, v));
    if (c != v) {
        if (changed != nullptr) *changed = true;
        std::fprintf(stderr, "%s=%s is outside %g..%g: using %g\n", name, raw, lo, hi, c);
    }
    return c;
}

// A drafter's chain length at pending position p: its steps write KV rows p .. p + k - 1, which must stay inside the
// context of max_ctx rows (the caller's `room` bounds how many drafts may be ACCEPTED, not how many are tried; the
// server's CTX_SLACK of 8 hid the overrun, the CLI had none).  0 when p itself is at or past the end.
inline int draft_kmax(int kmax, long long p, long long max_ctx) {
    if (p < 0 || p >= max_ctx) return 0;
    const long long left = max_ctx - p;
    return (int) std::max(0LL, std::min<long long>(kmax, left));
}

}  // namespace strata::core
