#pragma once

#include <string>
#include <functional>
#include <cstddef>

namespace nanoinfer {

enum class LogLevel : int {
    NONE = 0,
    ERROR = 1,
    WARN = 2,
    INFO = 3,
    DEBUG = 4,
    TRACE = 5,
};

class Logger {
public:
    static Logger& instance();

    void set_level(LogLevel level);
    LogLevel level() const;

    void log(LogLevel level, const std::string& msg) const;

    using Sink = std::function<void(LogLevel, const std::string&)>;
    void set_sink(Sink sink);
    void reset_sink();

    bool should_log(LogLevel level) const;

private:
    Logger() = default;
    LogLevel level_ = LogLevel::WARN;
    Sink sink_;
};

} // namespace nanoinfer

#define NI_LOG(level, msg)                                          \
    do {                                                            \
        if (::nanoinfer::Logger::instance().should_log(level)) {    \
            ::nanoinfer::Logger::instance().log(level, msg);        \
        }                                                           \
    } while (0)

#define LOG_ERROR(msg) NI_LOG(::nanoinfer::LogLevel::ERROR, msg)
#define LOG_WARN(msg)  NI_LOG(::nanoinfer::LogLevel::WARN, msg)
#define LOG_INFO(msg)  NI_LOG(::nanoinfer::LogLevel::INFO, msg)
#define LOG_DEBUG(msg) NI_LOG(::nanoinfer::LogLevel::DEBUG, msg)
#define LOG_TRACE(msg) NI_LOG(::nanoinfer::LogLevel::TRACE, msg)
