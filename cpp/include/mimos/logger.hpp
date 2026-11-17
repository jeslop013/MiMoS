// MiMoS logger.
//
// Based on geeksforgeeks' "logging system in C++".
//   * MIMOS_LOG(...) checks the level BEFORE evaluating the message arguments, so expensive
//     arguments cost nothing when the message is not written.
//   * No flush after every line (INFO logging inside the planning loops would be slow). The
//     file is flushed on close and on WARNING or higher, so problems are still on disk if
//     the process dies.
//   * open() can be called after construction (paths come from the configuration).
//   * Thread-safe: routing runs agents in parallel, so writes take a mutex.
#pragma once

#include <cstdarg>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

namespace mimos {

enum LogLevel { TRACE = 0, DEBUG = 1, INFO = 2, WARNING = 3, ERROR = 4, CRITICAL = 5 };

class Logger {
public:
    Logger() = default;

    void open(const std::string& filename, LogLevel min_level) {
        min_level_ = min_level;
        file_.open(filename, std::ios::trunc);
        if (!file_.is_open()) std::cerr << "MiMoS: could not open log file " << filename << std::endl;
    }

    bool enabled(LogLevel level) const { return level >= min_level_ && file_.is_open(); }

    void log(LogLevel level, const char* fmt, ...) {
        if (!enabled(level)) return;
        char buffer[512];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);

        std::lock_guard<std::mutex> lock(mutex_);
        std::time_t now = std::time(nullptr);
        char stamp[20];
        std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        file_ << '[' << stamp << "] [" << name(level) << "] " << buffer << '\n';
        if (level >= WARNING) file_.flush();
    }

private:
    static const char* name(LogLevel level) {
        switch (level) {
            case TRACE: return "TRACE";
            case DEBUG: return "DEBUG";
            case INFO: return "INFO";
            case WARNING: return "WARNING";
            case ERROR: return "ERROR";
            case CRITICAL: return "CRITICAL";
        }
        return "UNKNOWN";
    }

    std::ofstream file_;
    std::mutex mutex_;
    LogLevel min_level_ = INFO;
};

}  // namespace mimos

// Arguments are only evaluated when the level is enabled.
#define MIMOS_LOG(logger, level, ...)                                   \
    do {                                                                \
        if ((logger).enabled(::mimos::level)) (logger).log(::mimos::level, __VA_ARGS__); \
    } while (0)
