#include "nanoinfer/perf_profiler.h"

namespace nanoinfer {

PerfProfiler& PerfProfiler::instance() {
    static PerfProfiler profiler;
    return profiler;
}

} // namespace nanoinfer
