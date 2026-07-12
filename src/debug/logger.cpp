#include "nanoinfer/logger.h"
#include <cstdio>
#include <mutex>

namespace nanoinfer {

static const char* level_name(LogLevel level) {
    switch (level) {
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::TRACE: return "TRACE";
        default:              return "NONE";
    }
}

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

void Logger::set_level(LogLevel level) {
    level_ = level;
}

LogLevel Logger::level() const {
    return level_;
}

bool Logger::should_log(LogLevel level) const {
    return static_cast<int>(level) <= static_cast<int>(level_);
}

void Logger::log(LogLevel level, const std::string& msg) const {
    if (sink_) {
        sink_(level, msg);
    } else {
        std::fprintf(stderr, "[NanoInfer %s] %s\n", level_name(level), msg.c_str());
    }
}

void Logger::set_sink(Sink sink) {
    sink_ = std::move(sink);
}

void Logger::reset_sink() {
    sink_ = nullptr;
}

} // namespace nanoinfer
