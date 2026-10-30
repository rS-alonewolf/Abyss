// The expert reads' file layer (POSIX): O_DIRECT reads through an aligned bounce buffer, the second SSD
// (STRATA_GLM_MIRROR) and its pick, the buffered / mapped fallbacks and their counters, and the check that a mirror
// file is a copy of the shard.  No CUDA in here, so tools/test_shard_io.py compiles it with the system compiler and
// runs it against small files (unaligned offsets, the end of a file, a corrupt mirror, EINTR, a mirror read error).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

#ifndef _WIN32
#include <sys/types.h>
#endif

namespace strata::core::shard_io {

#ifndef _WIN32
// the read call every path goes through: a test points it at a wrapper that fails with EINTR or EIO
extern ssize_t (*pread_fn)(int fd, void* buf, size_t n, off_t off);
#endif

// bytes read from the shards' own drive [0] and from the mirror [1] (the stats line)
extern std::atomic<uint64_t> mirror_bytes[2];
// expert reads that left the O_DIRECT path: done with a buffered pread (the page cache), or copied from the mapping
extern std::atomic<uint64_t> fallback_buffered, fallback_mapped;

bool mirror_on();      // the mirror is still used (a read error on it turns it off for the run)
void reset_stats();    // counters and the mirror back to their start (tests)

#ifndef _WIN32
// pread of up to n bytes at off, EINTR / EAGAIN retried: bytes read (short only at the end of the file), -1 on error
ssize_t pread_full(int fd, void* buf, size_t n, uint64_t off);

// [off, off + len) through O_DIRECT into dst (an aligned per-thread bounce buffer in between).  With a mirror fd and
// the mirror on, the read goes to the drive that should finish it first; a mirror read that fails is done again on
// fd_direct and turns the mirror off.  False when the direct path could not fill dst (the caller falls back).
bool read_direct(int fd_direct, int fd_mirror, uint64_t off, size_t len, uint8_t* dst);

// [off, off + len) with buffered preads, the mapping (base, when there is one) as the last resort; false when neither
// could (counted in fallback_buffered / fallback_mapped)
bool read_buffered(int fd, const uint8_t* base, uint64_t off, size_t len, uint8_t* dst);

// Is the file behind mf a copy of the one behind fd (both `size` bytes)?  Their first and last MiB and 32 blocks of
// 256 KiB spread over the file are compared (~10 MiB a shard: a copy that stopped part way, or a file only
// fallocated to the size, is caught; the old head/tail check let both through).  `why` says where they differ.
bool same_content(int fd, int mf, uint64_t size, std::string& why);
#endif

}  // namespace strata::core::shard_io
