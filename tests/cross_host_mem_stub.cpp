// Minimal stand-in for forge::host_mem used by standalone kernel tests that
// are cross-compiled for QEMU (ARM64 NEON / PPC64 VSX) and cannot link the
// full host_mem_pool.cpp from the main library.
#include <cstdlib>

namespace forge {
namespace host_mem {

void* allocate(size_t bytes) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
    return p;
}

void deallocate(void* ptr) { free(ptr); }

}  // namespace host_mem
}  // namespace forge
