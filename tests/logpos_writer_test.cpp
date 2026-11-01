#include "strata/program/logpos.hpp"
#include <cassert>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <deque>
#include <sstream>
#include <limits>
#include <string>
#include <vector>

int main() {
    std::atomic<bool> stop{false}, quit{false};
    std::deque<std::string> lines;
    auto receive = [&](const std::string& line) {
        if (strata::program::queue_input(line, stop, quit)) lines.push_back(line);
    };
    receive("STOP"); receive("LOGPOS 1,2");
    assert(lines.front() == "LOGPOS 1,2"); lines.pop_front();
    strata::program::begin_logpos(stop);
    assert(!strata::program::logpos_cancelled(stop, quit));
    receive("STOP");
    assert(strata::program::logpos_cancelled(stop, quit));
    strata::program::begin_logpos(stop);
    receive("LOGPOS 3,4"); receive("QUIT");
    assert(strata::program::logpos_cancelled(stop, quit));
    assert(lines.front() == "LOGPOS 3,4"); lines.pop_front();
    assert(lines.front() == "QUIT");
    strata::program::begin_logpos(stop);
    assert(strata::program::logpos_cancelled(stop, quit));
    const char* tmp = std::getenv("TMPDIR");
    if (!tmp || !*tmp) tmp = ".";
    const std::string path = std::string(tmp) + "/logpos-writer-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".tsv";
    std::string err;
    strata::program::LogposWriter writer;
    assert(writer.open(path.c_str(), "2", 3, err));
    assert(writer.write(0, 2, {0.f, 1.f, 1.f}, err));
    assert(writer.flush(err));
    std::ifstream input(path);
    size_t pos; int target, top, hit; double lp, top_lp;
    input >> pos >> target >> lp >> top >> top_lp >> hit;
    std::string first, second, extra;
    input >> first >> second;
    assert(first.find(':') != std::string::npos && second.find(':') != std::string::npos);
    assert(!(input >> extra));
    assert(pos == 0 && target == 2 && top == 1 && hit == 0);
    assert(std::abs(lp + std::log(2.0 + std::exp(-1.0))) < 1e-9);
    strata::program::LogposWriter duplicate;
    assert(!duplicate.open(path.c_str(), "2", 3, err));
    assert(!writer.write(1, 0, {std::numeric_limits<float>::infinity(), 0.f, 0.f}, err));
    return 0;
}
