"""The expert reads' file layer (src/core/shard_io.cpp) on small files, without CUDA or a model: unaligned offsets,
the end of a file, a mirror that is not a copy, EINTR on a read, a mirror read error, the buffered fallbacks and
their counters.  Needs a C++ compiler (skipped without one); O_DIRECT cases are skipped on a filesystem that refuses
it (tmpfs), so the temporary files are made next to the repository."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")

DRIVER = r"""
#include "shard_io.hpp"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <algorithm>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
namespace sio = strata::core::shard_io;
static int eintr_left = 0, eio_fd = -1, calls = 0;
static ssize_t hooked(int fd, void* buf, size_t n, off_t off) {
    ++calls;
    if (eintr_left > 0) { --eintr_left; errno = EINTR; return -1; }
    if (fd == eio_fd) { errno = EIO; return -1; }
    return ::pread(fd, buf, n, off);
}
int main(int argc, char** argv) {
    // same <a> <b>                      -> "same" | "differ: <why>"
    // read <a> <mirror|-> <off> <len> [eintr=N] [eio=mirror] [nodirect] [nofd]
    //    -> the bytes (hex) and "direct=0/1 mirror_on=0/1 buffered=N mapped=N calls=N"
    const std::string cmd = argv[1];
    if (cmd == "same") {
        const int a = open(argv[2], O_RDONLY), b = open(argv[3], O_RDONLY);
        struct stat st{}; fstat(a, &st);
        std::string why;
        const bool ok = sio::same_content(a, b, (uint64_t) st.st_size, why);
        std::printf(ok ? "same\n" : "differ: %s\n", why.c_str());
        return 0;
    }
    const char* path = argv[2];
    const bool has_mirror = std::strcmp(argv[3], "-") != 0;
    const uint64_t off = std::strtoull(argv[4], nullptr, 10);
    const size_t len = (size_t) std::strtoull(argv[5], nullptr, 10);
    bool nodirect = false, nofd = false;
    int reps = 1;
    for (int i = 6; i < argc; ++i) {
        if (std::strncmp(argv[i], "eintr=", 6) == 0) eintr_left = std::atoi(argv[i] + 6);
        if (std::strncmp(argv[i], "reps=", 5) == 0) reps = std::atoi(argv[i] + 5);
        if (std::strcmp(argv[i], "eio=mirror") == 0) eio_fd = -2;
        if (std::strcmp(argv[i], "nodirect") == 0) nodirect = true;
        if (std::strcmp(argv[i], "nofd") == 0) nofd = true;
    }
    sio::pread_fn = hooked;
    const int fd = open(path, O_RDONLY);
    const int fdd = nodirect ? -1 : open(path, O_RDONLY | O_DIRECT);
    if (!nodirect && fdd < 0) { std::printf("nodirect\n"); return 0; }
    int fdm = -1;
    if (has_mirror) {
        fdm = open(argv[3], O_RDONLY | O_DIRECT);
        if (fdm < 0) { std::printf("nodirect\n"); return 0; }
        if (eio_fd == -2) eio_fd = fdm;
    }
    std::vector<uint8_t> dst(len, 0xEE), first;
    bool direct = false;
    for (int r = 0; r < reps; ++r) {   // (reps: the same read again and again - the pick's one-in-32 turn to the mirror)
        std::fill(dst.begin(), dst.end(), 0xEE);
        direct = false;
        if (fdd >= 0) direct = sio::read_direct(fdd, fdm, off, len, dst.data());
        if (!direct && !sio::read_buffered(nofd ? -1 : fd, nullptr, off, len, dst.data())) { std::printf("failed\n"); return 0; }
        if (r == 0) first = dst;
        else if (first != dst) { std::printf("differs on repeat %d\n", r); return 0; }
    }
    for (uint8_t b : dst) std::printf("%02x", b);
    std::printf("\ndirect=%d mirror_on=%d buffered=%llu mapped=%llu calls=%d\n", direct ? 1 : 0, sio::mirror_on() ? 1 : 0,
                (unsigned long long) sio::fallback_buffered.load(), (unsigned long long) sio::fallback_mapped.load(), calls);
    return 0;
}
"""


@unittest.skipUnless(CXX and os.name != "nt", "needs a C++ compiler on POSIX")
class ShardIoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="shard-io-", dir=ROOT))
        (cls.tmp / "driver.cpp").write_text(DRIVER)
        cls.exe = cls.tmp / "driver"
        r = subprocess.run([CXX, "-std=c++20", "-O1", "-I", str(ROOT / "src" / "core"), str(cls.tmp / "driver.cpp"),
                            str(ROOT / "src" / "core" / "shard_io.cpp"), "-o", str(cls.exe)], capture_output=True, text=True)
        if r.returncode != 0:
            shutil.rmtree(cls.tmp, ignore_errors=True)
            raise AssertionError("shard_io.cpp did not compile:\n" + r.stderr)
        # a 3 MiB + 777 byte shard of a pattern that is not periodic over 4 KiB, and copies of it
        cls.n = (3 << 20) + 777
        cls.data = bytes((i * 7 + (i >> 12) * 13) & 0xFF for i in range(cls.n))
        cls.shard = cls.tmp / "shard.gguf"
        cls.shard.write_bytes(cls.data)
        cls.copy = cls.tmp / "copy.gguf"
        cls.copy.write_bytes(cls.data)
        bad = bytearray(cls.data)
        bad[(1 << 20) + 5000] ^= 0xFF                          # one byte wrong in the middle
        cls.corrupt = cls.tmp / "corrupt.gguf"
        cls.corrupt.write_bytes(bytes(bad))
        part = bytearray(cls.data)
        part[(2 << 20):(cls.n - (1 << 20))] = bytes(len(part[(2 << 20):(cls.n - (1 << 20))]))   # a copy that stopped
        cls.partial = cls.tmp / "partial.gguf"
        cls.partial.write_bytes(bytes(part))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_driver(self, *args):
        r = subprocess.run([str(self.exe), *[str(a) for a in args]], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip().splitlines(), r.stderr

    def read(self, mirror, off, length, *flags):
        out, err = self.run_driver("read", self.shard, mirror or "-", off, length, *flags)
        if out == ["nodirect"]:
            self.skipTest("O_DIRECT is refused on this filesystem")
        self.assertEqual(len(out), 2, out)
        got = bytes.fromhex(out[0])
        stats = dict(kv.split("=") for kv in out[1].split())
        return got, stats, err

    def test_same_content(self):
        self.assertEqual(self.run_driver("same", self.shard, self.copy)[0], ["same"])
        self.assertTrue(self.run_driver("same", self.shard, self.corrupt)[0][0].startswith("differ: the bytes differ at"))
        self.assertTrue(self.run_driver("same", self.shard, self.partial)[0][0].startswith("differ: the bytes differ at"))

    def test_direct_reads_unaligned_and_at_the_end(self):
        for off, length in ((0, 100), (4095, 2), (12345, 70000), (self.n - 300, 300), (self.n - 1, 1)):
            got, stats, _ = self.read(None, off, length)
            self.assertEqual(got, self.data[off:off + length], (off, length))
            self.assertEqual(stats["direct"], "1")
            self.assertEqual(stats["buffered"], "0")

    def test_mirror_reads_the_same_bytes(self):
        got, stats, _ = self.read(self.copy, 8000, 50000)
        self.assertEqual(got, self.data[8000:58000])
        self.assertEqual(stats, {**stats, "direct": "1", "mirror_on": "1"})

    def test_eintr_is_retried_and_keeps_the_mirror(self):
        got, stats, err = self.read(self.copy, 4096 * 3 + 17, 9000, "eintr=3")
        self.assertEqual(got, self.data[4096 * 3 + 17:4096 * 3 + 17 + 9000])
        self.assertEqual(stats["mirror_on"], "1")
        self.assertEqual(stats["direct"], "1")
        self.assertNotIn("[mirror]", err)
        self.assertGreaterEqual(int(stats["calls"]), 4)

    def test_a_mirror_read_error_turns_it_off_and_reads_the_shard(self):
        # with equal speeds the pick is the shards' drive, and one read in 32 goes to the mirror: 40 reads reach it
        got, stats, err = self.read(self.copy, 70000, 5000, "eio=mirror", "reps=40")
        self.assertEqual(got, self.data[70000:75000])
        self.assertEqual(stats["direct"], "1")             # (the failed mirror read was done again on the shard)
        self.assertEqual(stats["mirror_on"], "0")
        self.assertEqual(stats["buffered"], "0")
        self.assertIn("the mirror is off", err)
        self.assertEqual(err.count("[mirror]"), 1)         # said once, not on every read after

    def test_reads_spread_over_both_drives(self):
        got, stats, _ = self.read(self.copy, 100, 3000, "reps=200")
        self.assertEqual(got, self.data[100:3100])
        self.assertEqual(stats["mirror_on"], "1")

    def test_buffered_fallback_is_counted(self):
        got, stats, _ = self.read(None, 777, 4321, "nodirect")
        self.assertEqual(got, self.data[777:777 + 4321])
        self.assertEqual(stats["direct"], "0")
        self.assertEqual(stats["buffered"], "1")
        self.assertEqual(stats["mapped"], "0")

    def test_nothing_to_read_from_fails_cleanly(self):
        out, _ = self.run_driver("read", self.shard, "-", 0, 10, "nodirect", "nofd")
        self.assertEqual(out, ["failed"])


if __name__ == "__main__":
    unittest.main()
