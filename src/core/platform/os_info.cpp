#include "src/core/platform/os_info.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>

#if defined(_WIN32)
#include <windows.h>
#else
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

#if !defined(_WIN32)
namespace {

// A cgroup limit can be much smaller than the host's available memory.
// Walk parents too: a child may say "max" beneath a limited ancestor.
std::uint64_t ClampToCgroupLimit(std::uint64_t available) {
  std::ifstream membership("/proc/self/cgroup");
  for (std::string line; std::getline(membership, line);) {
    if (!line.starts_with("0::/"))
      continue;
    const auto relative = std::filesystem::path(line.substr(4));
    if (std::ranges::any_of(relative,
                            [](const auto& part) { return part == ".."; }))
      continue;
    const std::filesystem::path root("/sys/fs/cgroup");
    for (auto path = root / relative;; path = path.parent_path()) {
      std::ifstream limit_file(path / "memory.max");
      std::ifstream used_file(path / "memory.current");
      std::uint64_t limit = 0, used = 0;
      if ((limit_file >> limit) && (used_file >> used))
        available = std::min(available, limit > used ? limit - used : 0);
      if (path == root)
        break;
    }
    break;
  }
  return available;
}

}  // namespace
#endif

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
  std::uint64_t available = (pages > 0 && page_size > 0)
                                ? static_cast<std::uint64_t>(pages) * page_size
                                : 0;
  std::ifstream meminfo("/proc/meminfo");
  for (std::string line; std::getline(meminfo, line);) {
    if (line.starts_with("MemAvailable:")) {
      std::istringstream fields(line.substr(13));
      std::uint64_t kib = 0;
      if (fields >> kib)
        available = kib * 1024;
      break;
    }
  }
  return ClampToCgroupLimit(available);
#endif
}

}  // namespace gufo::platform
