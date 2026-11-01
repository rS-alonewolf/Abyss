// CPU-only log-softmax/top-N writer for the opt-in LOGPOS wire command.
#pragma once
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace strata::program {
// Return false for control lines handled immediately by the input thread.
inline bool queue_input(const std::string& line, std::atomic<bool>& stop, std::atomic<bool>& quit) {
    if (line == "STOP") { stop.store(true); return false; }
    if (line == "QUIT") { quit.store(true); stop.store(true); }
    return true;
}
inline void begin_logpos(std::atomic<bool>& stop) { stop.store(false); }
inline bool logpos_cancelled(const std::atomic<bool>& stop, const std::atomic<bool>& quit) {
    return stop.load() || quit.load();
}
class LogposWriter {
    std::FILE* file_ = nullptr;
    size_t topk_ = 64;
public:
    ~LogposWriter() { if (file_) std::fclose(file_); }
    LogposWriter() = default;
    LogposWriter(const LogposWriter&) = delete;
    LogposWriter& operator=(const LogposWriter&) = delete;
    bool open(const char* path, const char* topk, size_t vocab, std::string& err) {
        if (file_) return true;
        if (!path || !*path) { err = "LOGPOS requires STRATA_GLM_LOGPOS"; return false; }
        if (topk) {
            char* end = nullptr;
            errno = 0;
            long k = std::strtol(topk, &end, 10);
            if (errno || end == topk || *end || k < 1 || (size_t) k > vocab) {
                err = "invalid STRATA_GLM_LOGPOS_TOPK"; return false;
            }
            topk_ = (size_t) k;
        } else topk_ = std::min<size_t>(64, vocab);
        // Never truncate a previous output. One file per engine process.
        file_ = std::fopen(path, "wx");
        if (!file_) { err = std::string("LOGPOS open: ") + std::strerror(errno); return false; }
        return true;
    }
    bool write(size_t pos, int target, const std::vector<float>& logits, std::string& err) {
        if (!file_ || target < 0 || (size_t) target >= logits.size() || topk_ > logits.size()) {
            err = "LOGPOS invalid logits/target"; return false;
        }
        for (float x : logits) if (!std::isfinite(x)) {
            err = "LOGPOS nonfinite logits"; return false;
        }
        std::vector<int> ids(logits.size());
        std::iota(ids.begin(), ids.end(), 0);
        auto better = [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
        std::partial_sort(ids.begin(), ids.begin() + topk_, ids.end(), better);
        const double mx = logits[ids[0]];
        double sum = 0;
        for (float x : logits) sum += std::exp((double) x - mx);
        const double lse = mx + std::log(sum);
        // Position, target, target log probability, top token, its log probability, hit.
        std::fprintf(file_, "%zu\t%d\t%.12g\t%d\t%.12g\t%d",
                     pos, target, logits[target] - lse, ids[0], logits[ids[0]] - lse, ids[0] == target);
        for (size_t j = 0; j < topk_; ++j)
            std::fprintf(file_, "\t%d:%.12g", ids[j], logits[ids[j]] - lse);
        std::fputc('\n', file_);
        if (std::ferror(file_)) { err = "LOGPOS write failed"; return false; }
        return true;
    }
    bool flush(std::string& err) {
        if (!file_ || std::fflush(file_) != 0) { err = "LOGPOS flush failed"; return false; }
        return true;
    }
};
} // namespace strata::program
