#include "src/cli/serve/logging.hpp"

#if defined(_WIN32)
#include <io.h>
#include <psapi.h>  // after windows.h: needs BOOL/DWORD definitions
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <array>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace gufo::server {
namespace {

std::mutex& LogMutex() {
  static std::mutex mutex;
  return mutex;
}

std::string SafeLine(std::string_view input, std::size_t limit = 8192) {
  std::string output;
  for (const unsigned char c : input.substr(0, limit)) {
    if (c < 0x20 || c == 0x7f) {
      constexpr char hex[] = "0123456789abcdef";
      output += "\\x";
      output += hex[c >> 4];
      output += hex[c & 15];
    } else {
      output += static_cast<char>(c);
    }
  }
  if (input.size() > limit)
    output += "...";
  return output;
}

std::string CurrentTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::system_clock::to_time_t(now);
  std::tm calendar{};
#if defined(_WIN32)
  (void)localtime_s(&calendar, &seconds);
#else
  ::localtime_r(&seconds, &calendar);
#endif
  std::array<char, 32> buffer{};
  const auto written = std::strftime(buffer.data(), buffer.size(),
                                     "%Y-%m-%d %H:%M:%S", &calendar);
  return {buffer.data(), written};
}

#if !defined(_WIN32)
std::size_t ReadMemoryKiB(const char* path, std::string_view field) {
  std::ifstream input(path);
  for (std::string line; std::getline(input, line);) {
    if (line.starts_with(field)) {
      std::istringstream value(line.substr(field.size()));
      std::size_t kib = 0;
      if (value >> kib)
        return kib;
    }
  }
  return 0;
}
#endif

}  // namespace

void Logger::Log(LogLevel level, std::string_view component,
                 std::string_view message) {
  static const bool color = std::getenv("NO_COLOR") == nullptr &&
#if defined(_WIN32)
                            _isatty(_fileno(stderr)) != 0;
#else
                            ::isatty(STDERR_FILENO) != 0;
#endif
  const char* tag = level == LogLevel::kError  ? "ERROR"
                    : level == LogLevel::kWarn ? "WARN"
                                               : "INFO";
  const char* tint = level == LogLevel::kError  ? "\033[31m"
                     : level == LogLevel::kWarn ? "\033[33m"
                                                : "\033[32m";
  std::ostringstream output;
  output << CurrentTimestamp() << ' ';
  if (color)
    output << tint;
  output << '[' << tag << ']';
  if (color)
    output << "\033[0m";
  output << " [" << SafeLine(component, 64) << "] " << SafeLine(message)
         << '\n';
  const std::lock_guard lock(LogMutex());
  std::clog << output.str() << std::flush;
}

std::string Logger::MemoryStatus() {
  std::ostringstream output;
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  PROCESS_MEMORY_COUNTERS_EX counters{};
  if (GlobalMemoryStatusEx(&status))
    output << " host_available_mib=" << status.ullAvailPhys / (1024 * 1024);
  if (GetProcessMemoryInfo(
          GetCurrentProcess(),
          reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
          sizeof(counters)))
    output << " rss_mib=" << counters.WorkingSetSize / (1024 * 1024);
#else
  output << "rss_mib=" << ReadMemoryKiB("/proc/self/status", "VmRSS:") / 1024
         << " host_available_mib="
         << ReadMemoryKiB("/proc/meminfo", "MemAvailable:") / 1024;
#endif
  return output.str();
}

void Logger::LogRequest(std::string_view id, std::string_view method,
                        std::string_view path, int status_code,
                        double duration_ms, std::string_view details,
                        std::string_view outcome) {
  std::ostringstream message;
  message << "request=" << id << " event=completed method=" << method
          << " path=" << path << " status=" << status_code
          << " duration_ms=" << std::fixed << std::setprecision(1)
          << duration_ms << " outcome=" << outcome;
  if (!details.empty())
    message << ' ' << details;
  message << ' ' << MemoryStatus();
  Log(status_code >= 500 || outcome == "stream_error"   ? LogLevel::kError
      : status_code >= 400 || outcome == "disconnected" ? LogLevel::kWarn
                                                        : LogLevel::kInfo,
      "http", message.str());
}

}  // namespace gufo::server
