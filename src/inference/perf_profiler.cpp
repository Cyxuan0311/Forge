#include "forge/perf_profiler.h"

namespace forge {

PerfProfiler& PerfProfiler::instance() {
    static PerfProfiler profiler;
    return profiler;
}

} // namespace forge
