#include "src/core/platform/os_info.hpp"

#if defined(_WIN32)
// clang-format off: psapi.h needs the windows.h types above it, and the
// console and CRT isatty calls need the io and stdio headers.
#include <windows.h>
#include <psapi.h>
#include <io.h>
#include <stdio.h>
// clang-format on
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace gufo::platform {

std::size_t PageSizeBytes() {
#if defined(_WIN32)
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  return info.dwPageSize != 0 ? static_cast<std::size_t>(info.dwPageSize)
                              : static_cast<std::size_t>(4096);
#else
  const long page_size = ::sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<std::size_t>(page_size)
                       : static_cast<std::size_t>(4096);
#endif
}

std::uint64_t AvailableMemoryBytes() {
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (!GlobalMemoryStatusEx(&status))
    return 0;
  return status.ullAvailPhys;
#else
  const long pages = ::sysconf(_SC_AVPHYS_PAGES);
  const long page_size = ::sysconf(_SC_PAGESIZE);
  return (pages > 0 && page_size > 0)
             ? static_cast<std::uint64_t>(pages) * page_size
             : 0;
#endif
}

int ProcessId() noexcept {
#if defined(_WIN32)
  return static_cast<int>(GetCurrentProcessId());
#else
  return ::getpid();
#endif
}

bool LocalTime(std::time_t seconds, std::tm* out) noexcept {
#if defined(_WIN32)
  return ::localtime_s(out, &seconds) == 0;
#else
  return ::localtime_r(&seconds, out) != nullptr;
#endif
}

bool UtcTime(std::time_t seconds, std::tm* out) noexcept {
#if defined(_WIN32)
  return ::gmtime_s(out, &seconds) == 0;
#else
  return ::gmtime_r(&seconds, out) != nullptr;
#endif
}

bool StderrIsTerminal() noexcept {
#if defined(_WIN32)
  return ::_isatty(::_fileno(stderr)) != 0;
#else
  return ::isatty(STDERR_FILENO) != 0;
#endif
}

PageFaults ReadPageFaults() noexcept {
#if defined(_WIN32)
  // The Windows counter totals soft and hard faults; the split is not
  // reported, so the count lands in the minor field.
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
    return {};
  return {counters.PageFaultCount, 0};
#else
  struct rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0)
    return {};
  return {static_cast<std::uint64_t>(usage.ru_minflt),
          static_cast<std::uint64_t>(usage.ru_majflt)};
#endif
}

}  // namespace gufo::platform
